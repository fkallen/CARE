#ifndef CARE_GPUCORRECTOR_CUH
#define CARE_GPUCORRECTOR_CUH


#include <hpc_helpers.cuh>
#include <hpc_helpers/include/nvtx_markers.cuh>

#include <gpu/distributedreadstorage.hpp>
#include <gpu/gpuminhasher.cuh>
#include <gpu/kernels.hpp>
#include <gpu/kernellaunch.hpp>

#include <threadpool.hpp>
#include <minhasher.hpp>
#include <options.hpp>

namespace care{
namespace gpu{

namespace gpucorrectorkernels{

    template<int gridsize, int blocksize>
    __global__
    void setAnchorIndicesOfCandidateskernel(
        int* __restrict__ d_anchorIndicesOfCandidates,
        const int* __restrict__ numAnchorsPtr,
        const int* __restrict__ d_candidates_per_subject,
        const int* __restrict__ d_candidates_per_subject_prefixsum
    ){
        for(int anchorIndex = blockIdx.x; anchorIndex < *numAnchorsPtr; anchorIndex += gridsize){
            const int offset = d_candidates_per_subject_prefixsum[anchorIndex];
            const int numCandidatesOfAnchor = d_candidates_per_subject[anchorIndex];
            int* const beginptr = &d_anchorIndicesOfCandidates[offset];

            for(int localindex = threadIdx.x; localindex < numCandidatesOfAnchor; localindex += blocksize){
                beginptr[localindex] = anchorIndex;
            }
        }
    }


    template<int blocksize, class Flags>
    __global__
    void selectIndicesOfFlagsOnlyOneBlock(
        int* __restrict__ selectedIndices,
        int* __restrict__ numSelectedIndices,
        const Flags flags,
        const int* __restrict__ numFlags
    ){
        constexpr int ITEMS_PER_THREAD = 4;

        using BlockScan = cub::BlockScan<int, blocksize>;

        __shared__ typename BlockScan::TempStorage temp_storage;

        int aggregate = 0;
        const int iters = SDIV(*numFlags, blocksize * ITEMS_PER_THREAD);
        const int threadoffset = ITEMS_PER_THREAD * threadIdx.x;

        for(int iter = 0; iter < iters; iter++){

            int data[ITEMS_PER_THREAD];
            int prefixsum[ITEMS_PER_THREAD];

            const int iteroffset = blocksize * ITEMS_PER_THREAD * iter;

            #pragma unroll
            for(int k = 0; k < ITEMS_PER_THREAD; k++){
                if(iteroffset + threadoffset + k < *numFlags){
                    data[k] = flags[iteroffset + threadoffset + k];
                }else{
                    data[k] = 0;
                }
            }

            int block_aggregate = 0;
            BlockScan(temp_storage).ExclusiveSum(data, prefixsum, block_aggregate);

            #pragma unroll
            for(int k = 0; k < ITEMS_PER_THREAD; k++){
                if(data[k] == 1){
                    selectedIndices[prefixsum[k]] = iteroffset + threadoffset + k;
                }
            }

            aggregate += block_aggregate;

            __syncthreads();
        }

        if(threadIdx.x == 0){
            *numSelectedIndices = aggregate;
        }

    }
}
    

class GpuErrorCorrector{

public:

    template<class T>
    using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T>;

    template<class T>
    using DeviceBuffer = helpers::SimpleAllocationDevice<T>;

    struct ReadCorrectionFlags{
        friend class CpuErrorCorrector;
    public:
        ReadCorrectionFlags() = default;

        ReadCorrectionFlags(std::size_t numReads)
            : size(numReads), flags(std::make_unique<std::uint8_t[]>(numReads)){
            std::fill(flags.get(), flags.get() + size, 0);
        }

        std::size_t sizeInBytes() const noexcept{
            return size * sizeof(std::uint8_t);
        }

    private:
        static constexpr std::uint8_t readCorrectedAsHQAnchor() noexcept{ return 1; };
        static constexpr std::uint8_t readCouldNotBeCorrectedAsAnchor() noexcept{ return 2; };

        void setCorrectedAsHqAnchor(std::int64_t position) const noexcept{
            flags[position] = readCorrectedAsHQAnchor();
        }

        void setCouldNotBeCorrectedAsAnchor(std::int64_t position) const noexcept{
            flags[position] = readCouldNotBeCorrectedAsAnchor();
        }

        bool isCorrectedAsHQAnchor(std::int64_t position) const noexcept{
            return (flags[position] & readCorrectedAsHQAnchor()) > 0;
        }

        bool isNotCorrectedAsAnchor(std::int64_t position) const noexcept{
            return (flags[position] & readCouldNotBeCorrectedAsAnchor()) > 0;
        }

        std::size_t size;
        std::unique_ptr<std::uint8_t[]> flags{};
    };

    struct CorrectionInput{
        std::vector<read_number> readIds;
    };

    struct CorrectionOutput{

    };

    CorrectionOutput correct(const CorrectionInput& input, cudaStream_t stream){
        const std::size_t inputSize = input.readIds.size();
        if(inputSize == 0){
            return CorrectionOutput{};
        }

        int curId = 0;
        cudaGetDevice(&curId); CUERR;
        cudaSetDevice(deviceId); CUERR;

        resizeBuffers(inputSize);
   
        //copy input to pinned memory
        h_numAnchors[0] = inputSize;
        std::copy(input.readIds.begin(), input.readIds.end(), h_subject_read_ids.get());

        nvtx::push_range("copyInputDataToDevice", 0);
        copyInputDataToDevice(stream);
        nvtx::pop_range();

        nvtx::push_range("getAnchorReads", 0);
        getAnchorReads(stream);
        nvtx::pop_range();

        nvtx::push_range("getCandidateReadIdsWithMinhashing", 0);
        getCandidateReadIdsWithMinhashing(stream);
        nvtx::pop_range();

        cudaStreamSynchronize(stream); CUERR; // wait for h_numCandidates, required to transpose candidate sequence data

        nvtx::push_range("getCandidateSequenceData", 0);
        getCandidateSequenceData(stream); 
        nvtx::pop_range();

        if(correctionOptions->useQualityScores) {
            nvtx::push_range("getQualities", 0);

            getQualities(stream);

            nvtx::pop_range();
        }

        nvtx::push_range("getCandidateAlignments", 0);
        getCandidateAlignments(stream); 
        nvtx::pop_range();


        return CorrectionOutput{};

    }


private:

    void init(int devId){
        int curId = 0;
        cudaGetDevice(&curId); CUERR;
        cudaSetDevice(devId); CUERR;

        deviceId = devId;

        GpuMinhasher::QueryHandle minhashHandle = GpuMinhasher::makeQueryHandle();

        const int resultsPerMap = calculateResultsPerMapThreshold(correctionOptions->estimatedCoverage);
        maxCandidatesPerRead = resultsPerMap * gpuMinhasher->getNumberOfMaps();

        kernelLaunchHandle = make_kernel_launch_handle(deviceId);

        cudaSetDevice(curId); CUERR;
    }
    void init(){
        cudaGetDevice(&deviceId); CUERR;

        init(deviceId);
    }

    void initBuffers(){

    }

    void resizeBuffers(std::size_t numReads){

    }

    void copyInputDataToDevice(cudaStream_t stream){
        cudaMemcpyAsync(
            d_numAnchors.get(),
            h_numAnchors.get(),
            sizeof(int),
            H2D,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            d_subject_read_ids.get(),
            h_subject_read_ids.get(),
            sizeof(read_number) * (*h_numAnchors.get()),
            H2D,
            stream
        ); CUERR;
    }

    void getAnchorReads(cudaStream_t stream){
        gpuReadStorage->gatherSequenceDataToGpuBufferAsync(
            threadPool,
            subjectSequenceGatherHandle,
            d_subject_sequences_data.get(),
            encodedSequencePitchInInts,
            d_subject_read_ids.get(),
            h_subject_read_ids.get(),
            (*h_numAnchors.get()),
            deviceId,
            stream
        );

        gpuReadStorage->gatherSequenceLengthsToGpuBufferAsync(
            d_subject_sequences_lengths.get(),
            deviceId,
            d_subject_read_ids.get(),
            (*h_numAnchors.get()),
            stream
        );
    }

    void getCandidateReadIdsWithMinhashing(cudaStream_t stream){
        ForLoopExecutor forLoopExecutor(threadPool, &pforHandle);

        gpuMinhasher->getIdsOfSimilarReadsExcludingSelf(
            minhashHandle,
            d_subject_read_ids.get(),
            h_subject_read_ids.get(),
            d_subject_sequences_data.get(),
            encodedSequencePitchInInts,
            d_subject_sequences_lengths.get(),
            (*h_numAnchors.get()),
            deviceId, 
            stream,
            forLoopExecutor,
            d_candidate_read_ids.get(),
            d_candidates_per_subject.get(),
            d_candidates_per_subject_prefixsum.get()
        );

        cudaMemcpyAsync(
            d_numCandidates.get(),
            d_candidates_per_subject_prefixsum.get() + (*h_numAnchors.get()),
            sizeof(int),
            D2D,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            h_numCandidates.get(),
            d_candidates_per_subject_prefixsum.get() + (*h_numAnchors.get()),
            sizeof(int),
            H2D,
            stream
        ); CUERR;
    }

    void getCandidateSequenceData(cudaStream_t stream){

        const auto maxCandidates = maxCandidatesPerRead * (*h_numAnchors.get());

        //TODO
        // cudaMemcpyAsync(
        //     batch.h_subject_sequences_lengths,
        //     batch.d_subject_sequences_lengths, //filled by nextiteration data
        //     sizeof(int) * (*h_numAnchors.get()),
        //     D2H,
        //     streams[secondary_stream_index]
        // ); CUERR;

        gpuReadStorage->readsContainN_async(
            deviceId,
            d_anchorContainsN.get(), 
            d_subject_read_ids.get(), 
            d_numAnchors,
            (*h_numAnchors.get()), 
            stream
        );

        gpuReadStorage->readsContainN_async(
            deviceId,
            d_candidateContainsN.get(), 
            d_candidate_read_ids.get(), 
            d_numCandidates,
            maxCandidates, 
            stream
        );  

        gpuReadStorage->gatherSequenceLengthsToGpuBufferAsync(
            d_candidate_sequences_lengths.get(),
            deviceId,
            d_candidate_read_ids.get(),
            (*h_numAnchors.get()),            
            stream
        );

        gpuReadStorage->gatherSequenceDataToGpuBufferAsync(
            threadPool,
            candidateSequenceGatherHandle,
            d_candidate_sequences_data.get(),
            encodedSequencePitchInInts,
            h_candidate_read_ids,
            d_candidate_read_ids,
            (*h_numAnchors.get()),
            deviceId,
            stream
        );

        helpers::call_transpose_kernel(
            d_transposedCandidateSequencesData.get(), 
            d_candidate_sequences_data.get(), 
            h_numCandidates[0], 
            encodedSequencePitchInInts, 
            encodedSequencePitchInInts, 
            stream
        );

        //TODO
        // cudaEventRecord(events[alignment_data_transfer_h2d_finished_event_index], streams[primary_stream_index]); CUERR;

        // cudaStreamWaitEvent(streams[secondary_stream_index], events[alignment_data_transfer_h2d_finished_event_index], 0) ;


        // cudaMemcpyAsync(batch.h_candidate_sequences_lengths,
        //                 batch.d_candidate_sequences_lengths,
        //                 sizeof(int) * maxCandidates,
        //                 D2H,
        //                 streams[secondary_stream_index]); CUERR;
    }

    void getQualities(cudaStream_t stream){

		if(correctionOptions->useQualityScores) {
            // std::cerr << "gather anchor qual\n";
            gpuReadStorage->gatherQualitiesToGpuBufferAsync(
                threadPool,
                subjectQualitiesGatherHandle,
                d_subject_qualities,
                qualityPitchInBytes,
                h_subject_read_ids,
                d_subject_read_ids,
                (*h_numAnchors.get()),
                deviceId,
                stream
            );

            // std::cerr << "gather candidate qual\n";
            gpuReadStorage->gatherQualitiesToGpuBufferAsync(
                threadPool,
                candidateQualitiesGatherHandle,
                d_candidate_qualities,
                qualityPitchInBytes,
                h_candidate_read_ids.get(),
                d_candidate_read_ids.get(),
                (*h_numAnchors.get()),
                deviceId,
                stream
            );
        }

        //cudaStreamSynchronize(streams[primary_stream_index]); CUERR;
	}

    void getCandidateAlignments(cudaStream_t stream){

        const auto maxCandidates = maxCandidatesPerRead * (*h_numAnchors.get());

        {
            
            gpucorrectorkernels::setAnchorIndicesOfCandidateskernel<1024, 128>
                    <<<1024, 128, 0, stream>>>(
                d_anchorIndicesOfCandidates.get(),
                d_numAnchors.get(),
                d_candidates_per_subject.get(),
                d_candidates_per_subject_prefixsum.get()
            ); CUERR;
        }

        std::size_t tempBytes = d_tempstorage.sizeInBytes();

        const bool removeAmbiguousAnchors = correctionOptions->excludeAmbiguousReads;
        const bool removeAmbiguousCandidates = correctionOptions->excludeAmbiguousReads;

        call_popcount_shifted_hamming_distance_kernel_async(
            d_tempstorage.get(),
            tempBytes,
            d_alignment_overlaps.get(),
            d_alignment_shifts.get(),
            d_alignment_nOps.get(),
            d_alignment_isValid.get(),
            d_alignment_best_alignment_flags.get(),
            d_subject_sequences_data.get(),
            d_candidate_sequences_data.get(),
            d_subject_sequences_lengths.get(),
            d_candidate_sequences_lengths.get(),
            d_candidates_per_subject_prefixsum.get(),
            d_candidates_per_subject.get(),
            d_anchorIndicesOfCandidates.get(),
            d_numAnchors.get(),
            d_numCandidates.get(),
            d_anchorContainsN.get(),
            removeAmbiguousAnchors,
            d_candidateContainsN.get(),
            removeAmbiguousCandidates,
            (*h_numAnchors.get()),
            maxCandidates,
            sequenceFileProperties->maxSequenceLength,
            encodedSequencePitchInInts,
            goodAlignmentProperties->min_overlap,
            goodAlignmentProperties->maxErrorRate,
            goodAlignmentProperties->min_overlap_ratio,
            correctionOptions->estimatedErrorrate,
            stream,
            kernelLaunchHandle
        );

        call_cuda_filter_alignments_by_mismatchratio_kernel_async(
            d_alignment_best_alignment_flags.get(),
            d_alignment_nOps.get(),
            d_alignment_overlaps.get(),
            d_candidates_per_subject_prefixsum.get(),
            d_numAnchors.get(),
            d_numCandidates.get(),
            (*h_numAnchors.get()),
            maxCandidates,
            correctionOptions->estimatedErrorrate,
            correctionOptions->estimatedCoverage * correctionOptions->m_coverage,
            stream,
            kernelLaunchHandle
        );

        callSelectIndicesOfGoodCandidatesKernelAsync(
            d_indices.get(),
            d_indices_per_subject.get(),
            d_num_indices.get(),
            d_alignment_best_alignment_flags.get(),
            d_candidates_per_subject.get(),
            d_candidates_per_subject_prefixsum.get(),
            d_anchorIndicesOfCandidates.get(),
            d_numAnchors.get(),
            d_numCandidates.get(),
            (*h_numAnchors.get()),
            maxCandidates,
            stream,
            kernelLaunchHandle
        );
	}

private:

    int deviceId;

    std::size_t encodedSequencePitchInInts;
    std::size_t decodedSequencePitchInBytes;
    std::size_t qualityPitchInBytes;

    int maxCandidatesPerRead;

    const DistributedReadStorage* gpuReadStorage;
    const GpuMinhasher* gpuMinhasher;
    ReadCorrectionFlags* correctionFlags;

    const CorrectionOptions* correctionOptions;
    const GoodAlignmentProperties* goodAlignmentProperties;
    const SequenceFileProperties* sequenceFileProperties;


    ThreadPool* threadPool;
    ThreadPool::ParallelForHandle pforHandle;
    DistributedReadStorage::GatherHandleSequences subjectSequenceGatherHandle;
    DistributedReadStorage::GatherHandleSequences candidateSequenceGatherHandle;
    DistributedReadStorage::GatherHandleQualities subjectQualitiesGatherHandle;
    DistributedReadStorage::GatherHandleQualities candidateQualitiesGatherHandle;
    GpuMinhasher::QueryHandle minhashHandle;
    KernelLaunchHandle kernelLaunchHandle;  

    PinnedBuffer<unsigned int> h_subject_sequences_data;
    PinnedBuffer<int> h_subject_sequences_lengths;
    PinnedBuffer<read_number> h_subject_read_ids;
    PinnedBuffer<read_number> h_candidate_read_ids;
    PinnedBuffer<int> h_numAnchors;
    PinnedBuffer<int> h_numCandidates;
    DeviceBuffer<unsigned int> d_subject_sequences_data;
    DeviceBuffer<int> d_subject_sequences_lengths;
    DeviceBuffer<read_number> d_subject_read_ids;
    DeviceBuffer<read_number> d_candidate_read_ids;
    DeviceBuffer<int> d_candidates_per_subject;
    DeviceBuffer<int> d_candidates_per_subject_tmp;
    DeviceBuffer<int> d_candidates_per_subject_prefixsum;
    DeviceBuffer<int> d_numAnchors;
    DeviceBuffer<int> d_numCandidates;
    DeviceBuffer<bool> d_anchorContainsN;
    DeviceBuffer<bool> d_candidateContainsN;
    DeviceBuffer<int> d_candidate_sequences_lengths;
    DeviceBuffer<unsigned int> d_candidate_sequences_data;
    DeviceBuffer<unsigned int> d_transposedCandidateSequencesData;
    DeviceBuffer<char> d_subject_qualities;
    DeviceBuffer<char> d_candidate_qualities;
    DeviceBuffer<int> d_anchorIndicesOfCandidates;
    DeviceBuffer<char> d_tempstorage;
    DeviceBuffer<int> d_alignment_overlaps;
    DeviceBuffer<int> d_alignment_shifts;
    DeviceBuffer<int> d_alignment_nOps;
    DeviceBuffer<bool> d_alignment_isValid;
    DeviceBuffer<BestAlignment_t> d_alignment_best_alignment_flags; 
    DeviceBuffer<int> d_indices;
    DeviceBuffer<int> d_indices_per_subject;
    DeviceBuffer<int> d_num_indices;
    DeviceBuffer<int> d_indices_tmp;
    DeviceBuffer<int> d_indices_per_subject_tmp;
    DeviceBuffer<int> d_num_indices_tmp;

};



    }
}






#endif