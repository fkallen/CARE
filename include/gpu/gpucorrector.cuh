#ifndef CARE_GPUCORRECTOR_CUH
#define CARE_GPUCORRECTOR_CUH


#include <hpc_helpers.cuh>
#include <hpc_helpers/include/nvtx_markers.cuh>

#include <gpu/distributedreadstorage.hpp>
#include <gpu/gpuminhasher.cuh>
#include <gpu/kernels.hpp>
#include <gpu/kernellaunch.hpp>
#include <gpu/cudagraphhelpers.cuh>

#include <threadpool.hpp>
#include <minhasher.hpp>
#include <options.hpp>
#include <correctionresultprocessing.hpp>
#include <memorymanagement.hpp>

#include <algorithm>
#include <array>
#include <map>

#include <cub/cub.cuh>

namespace care{
namespace gpu{

namespace gpucorrectorkernels{

    __global__
    void copyCorrectionInputDeviceData(
        int* __restrict__ output_numAnchors,
        int* __restrict__ output_numCandidates,
        read_number* __restrict__ output_anchor_read_ids,
        unsigned int* __restrict__ output_anchor_sequences_data,
        int* __restrict__ output_anchor_sequences_lengths,
        read_number* __restrict__ output_candidate_read_ids,
        int* __restrict__ output_candidates_per_anchor,
        int* __restrict__ output_candidates_per_anchor_prefixsum,
        const int encodedSequencePitchInInts,
        const int* __restrict__ input_numAnchors,
        const int* __restrict__ input_numCandidates,
        const read_number* __restrict__ input_anchor_read_ids,
        const unsigned int* __restrict__ input_anchor_sequences_data,
        const int* __restrict__ input_anchor_sequences_lengths,
        const read_number* __restrict__ input_candidate_read_ids,
        const int* __restrict__ input_candidates_per_anchor,
        const int* __restrict__ input_candidates_per_anchor_prefixsum
    ){
        const int numAnchors = *input_numAnchors;
        const int numCandidates = *input_numCandidates;

        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        if(tid == 0){
            *output_numAnchors = numAnchors;
            *output_numCandidates = numCandidates;
        }

        for(int i = tid; i < numAnchors; i += stride){
            output_anchor_read_ids[i] = input_anchor_read_ids[i];
        }

        for(int i = tid; i < numAnchors * encodedSequencePitchInInts; i += stride){
            output_anchor_sequences_data[i] = input_anchor_sequences_data[i];
        }

        for(int i = tid; i < numAnchors; i += stride){
            output_anchor_sequences_lengths[i] = input_anchor_sequences_lengths[i];
        }

        for(int i = tid; i < numCandidates; i += stride){
            output_candidate_read_ids[i] = input_candidate_read_ids[i];
        }

        for(int i = tid; i < numAnchors; i += stride){
            output_candidates_per_anchor[i] = input_candidates_per_anchor[i];
        }

        for(int i = tid; i < numAnchors + 1; i += stride){
            output_candidates_per_anchor_prefixsum[i] = input_candidates_per_anchor_prefixsum[i];
        }

    }

    __global__ 
    void copyMinhashResultsKernel(
        int* __restrict__ d_numCandidates,
        int* __restrict__ h_numCandidates,
        read_number* __restrict__ h_candidate_read_ids,
        const int* __restrict__ d_candidates_per_anchor_prefixsum,
        const read_number* __restrict__ d_candidate_read_ids,
        const int numAnchors
    ){
        const int numCandidates = d_candidates_per_anchor_prefixsum[numAnchors];

        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        if(tid == 0){
            *d_numCandidates = numCandidates;
            *h_numCandidates = numCandidates;
        }

        for(int i = tid; i < numCandidates; i += stride){
            h_candidate_read_ids[i] = d_candidate_read_ids[i];
        }
    }

    template<int gridsize, int blocksize>
    __global__
    void setAnchorIndicesOfCandidateskernel(
        int* __restrict__ d_anchorIndicesOfCandidates,
        const int* __restrict__ numAnchorsPtr,
        const int* __restrict__ d_candidates_per_anchor,
        const int* __restrict__ d_candidates_per_anchor_prefixsum
    ){
        for(int anchorIndex = blockIdx.x; anchorIndex < *numAnchorsPtr; anchorIndex += gridsize){
            const int offset = d_candidates_per_anchor_prefixsum[anchorIndex];
            const int numCandidatesOfAnchor = d_candidates_per_anchor[anchorIndex];
            int* const beginptr = &d_anchorIndicesOfCandidates[offset];

            for(int localindex = threadIdx.x; localindex < numCandidatesOfAnchor; localindex += blocksize){
                beginptr[localindex] = anchorIndex;
            }
        }
    }


    template<int blocksize, class Flags>
    __global__
    void selectIndicesOfFlagsOneBlock(
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

    __global__ 
    void initArraysBeforeCandidateCorrectionKernel(
        const int* __restrict__ d_numCandidates,
        const int* __restrict__ d_numAnchors,
        int* __restrict__ d_num_corrected_candidates_per_anchor,
        bool* __restrict__ d_candidateCanBeCorrected
    ){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        const int numAnchors = *d_numAnchors;
        const int numCandidates = *d_numCandidates;

        for(int i = tid; i < numAnchors; i += stride){
            d_num_corrected_candidates_per_anchor[i] = 0;
        }

        for(int i = tid; i < numCandidates; i += stride){
            d_candidateCanBeCorrected[i] = 0;
        }
    }

    __global__
    void copyShiftsAndCorrectedCandidateIndices(
        int* __restrict__ output_alignment_shifts,
        int* __restrict__ output_indices_of_corrected_candidates,
        const int* __restrict__ d_numCandidates,
        const int* __restrict__ input_alignment_shifts,
        const int* __restrict__ input_indices_of_corrected_candidates
    ){
        using CopyType = int;

        const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t stride = blockDim.x * gridDim.x;

        const int numElements = *d_numCandidates;

        for(int index = tid; index < numElements; index += stride){
            output_alignment_shifts[index] = input_alignment_shifts[index];
            output_indices_of_corrected_candidates[index] = input_indices_of_corrected_candidates[index];
        } 
    }

} //namespace gpucorrectorkernels   

    class GpuErrorCorrectorInput{
    public:
        template<class T>
        using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T>;

        template<class T>
        using DeviceBuffer = helpers::SimpleAllocationDevice<T>;


        CudaEvent event{cudaEventDisableTiming};

        PinnedBuffer<int> h_numAnchors;
        PinnedBuffer<int> h_numCandidates;
        PinnedBuffer<read_number> h_anchorReadIds;
        PinnedBuffer<read_number> h_candidate_read_ids;

        DeviceBuffer<int> d_numAnchors;
        DeviceBuffer<int> d_numCandidates;
        DeviceBuffer<read_number> d_anchorReadIds;
        DeviceBuffer<unsigned int> d_anchor_sequences_data;
        DeviceBuffer<int> d_anchor_sequences_lengths;
        DeviceBuffer<read_number> d_candidate_read_ids;
        DeviceBuffer<int> d_candidates_per_anchor;
        DeviceBuffer<int> d_candidates_per_anchor_prefixsum;    

        MemoryUsage getMemoryInfo() const{
            MemoryUsage info{};
            auto handleHost = [&](const auto& h){
                info.host += h.sizeInBytes();
            };
            auto handleDevice = [&](const auto& d){
                info.device[event.getDeviceId()] += d.sizeInBytes();
            };

            handleHost(h_numAnchors);
            handleHost(h_numCandidates);
            handleHost(h_anchorReadIds);
            handleHost(h_candidate_read_ids);

            handleDevice(d_numAnchors);
            handleDevice(d_numCandidates);
            handleDevice(d_anchorReadIds);
            handleDevice(d_anchor_sequences_data);
            handleDevice(d_anchor_sequences_lengths);
            handleDevice(d_candidate_read_ids);
            handleDevice(d_candidates_per_anchor);
            handleDevice(d_candidates_per_anchor_prefixsum);

            return info;
        }  
    };

    class GpuErrorCorrectorRawOutput{
    public:
        template<class T>
        using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T>;

        template<class T>
        using DeviceBuffer = helpers::SimpleAllocationDevice<T>;

        bool nothingToDo;
        int numAnchors;
        int numCandidates;
        int doNotUseEditsValue;
        std::size_t editsPitchInBytes;
        std::size_t decodedSequencePitchInBytes;
        CudaEvent event{cudaEventDisableTiming};
        PinnedBuffer<read_number> h_anchorReadIds;
        PinnedBuffer<read_number> h_candidate_read_ids;
        PinnedBuffer<bool> h_anchor_is_corrected;
        PinnedBuffer<AnchorHighQualityFlag> h_is_high_quality_anchor;
        PinnedBuffer<int> h_num_corrected_candidates_per_anchor;
        PinnedBuffer<int> h_num_corrected_candidates_per_anchor_prefixsum;
        PinnedBuffer<int> h_indices_of_corrected_candidates;

        PinnedBuffer<int> h_candidate_sequences_lengths;
        PinnedBuffer<int> h_numEditsPerCorrectedanchor;
        PinnedBuffer<TempCorrectedSequence::EncodedEdit> h_editsPerCorrectedanchor;
        PinnedBuffer<char> h_corrected_anchors;
        PinnedBuffer<int> h_anchor_sequences_lengths;
        PinnedBuffer<char> h_corrected_candidates;
        PinnedBuffer<int> h_alignment_shifts;
        PinnedBuffer<int> h_numEditsPerCorrectedCandidate;
        PinnedBuffer<TempCorrectedSequence::EncodedEdit> h_editsPerCorrectedCandidate;

        MemoryUsage getMemoryInfo() const{
            MemoryUsage info{};
            auto handleHost = [&](const auto& h){
                info.host += h.sizeInBytes();
            };

            handleHost(h_anchorReadIds);
            handleHost(h_candidate_read_ids);
            handleHost(h_anchor_is_corrected);
            handleHost(h_is_high_quality_anchor);
            handleHost(h_num_corrected_candidates_per_anchor);
            handleHost(h_num_corrected_candidates_per_anchor_prefixsum);
            handleHost(h_indices_of_corrected_candidates);
            handleHost(h_candidate_sequences_lengths);
            handleHost(h_numEditsPerCorrectedanchor);
            handleHost(h_editsPerCorrectedanchor);
            handleHost(h_corrected_anchors);
            handleHost(h_anchor_sequences_lengths);
            handleHost(h_corrected_candidates);
            handleHost(h_alignment_shifts);
            handleHost(h_numEditsPerCorrectedCandidate);
            handleHost(h_editsPerCorrectedCandidate);

            return info;
        }  
    };


    class GpuAnchorHasher{
    public:

        GpuAnchorHasher() = default;

        GpuAnchorHasher(
            const DistributedReadStorage& gpuReadStorage_,
            const GpuMinhasher& gpuMinhasher_,
            const SequenceFileProperties& sequenceFileProperties_,
            ThreadPool* threadPool_
        ) : 
            gpuReadStorage{&gpuReadStorage_},
            gpuMinhasher{&gpuMinhasher_},
            sequenceFileProperties{&sequenceFileProperties_},
            threadPool{threadPool_}
        {
            cudaGetDevice(&deviceId); CUERR;

            minhashHandle = GpuMinhasher::makeQueryHandle();
            maxCandidatesPerRead = gpuMinhasher->getNumResultsPerMapThreshold() * gpuMinhasher->getNumberOfMaps();

            backgroundStream = CudaStream{};

            encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(sequenceFileProperties->maxSequenceLength);
            anchorSequenceGatherHandle = gpuReadStorage->makeGatherHandleSequences();
        }

        void makeErrorCorrectorInput(
            const read_number* anchorIds,
            int numIds,
            GpuErrorCorrectorInput& ecinput,
            cudaStream_t stream
        ){
            int curId = 0;
            cudaGetDevice(&curId); CUERR;
            cudaSetDevice(deviceId); CUERR;

            assert(cudaSuccess == ecinput.event.query());

            resizeBuffers(ecinput, numIds);
    
            //copy input to pinned memory
            *ecinput.h_numAnchors.get() = numIds;
            std::copy_n(anchorIds, numIds, ecinput.h_anchorReadIds.get());

            cudaMemcpyAsync(
                ecinput.d_numAnchors.get(),
                ecinput.h_numAnchors.get(),
                sizeof(int),
                H2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                ecinput.d_anchorReadIds.get(),
                ecinput.h_anchorReadIds.get(),
                sizeof(read_number) * (*ecinput.h_numAnchors.get()),
                H2D,
                stream
            ); CUERR;

            nvtx::push_range("getAnchorReads", 0);
            getAnchorReads(ecinput, stream);
            nvtx::pop_range();

            nvtx::push_range("getCandidateReadIdsWithMinhashing", 1);
            getCandidateReadIdsWithMinhashing(ecinput, stream);
            nvtx::pop_range();

            ecinput.event.record(stream);

            cudaSetDevice(curId); CUERR;
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage info{};
            info += minhashHandle.getMemoryInfo();
            info += gpuReadStorage->getMemoryInfoOfGatherHandleSequences(anchorSequenceGatherHandle);
            return info;
        } 

    private:
        void resizeBuffers(GpuErrorCorrectorInput& ecinput, int numAnchors){
            const std::size_t maxCandidates = maxCandidatesPerRead * numAnchors;
            // large enough to store all minhash results
            ecinput.h_candidate_read_ids.resize(maxCandidates);
            ecinput.d_candidate_read_ids.resize(maxCandidates); 

            ecinput.h_numAnchors.resize(1);
            ecinput.h_numCandidates.resize(1);
            ecinput.h_anchorReadIds.resize(numAnchors);

            ecinput.d_numAnchors.resize(1);
            ecinput.d_numCandidates.resize(1);
            ecinput.d_anchorReadIds.resize(numAnchors);
            ecinput.d_anchor_sequences_data.resize(encodedSequencePitchInInts * numAnchors);
            ecinput.d_anchor_sequences_lengths.resize(numAnchors);
            ecinput.d_candidates_per_anchor.resize(numAnchors);
            ecinput.d_candidates_per_anchor_prefixsum.resize(numAnchors + 1);
        }
        
        void getAnchorReads(GpuErrorCorrectorInput& ecinput, cudaStream_t stream){
            gpuReadStorage->gatherSequenceDataToGpuBufferAsync(
                threadPool,
                anchorSequenceGatherHandle,
                ecinput.d_anchor_sequences_data.get(),
                encodedSequencePitchInInts,
                ecinput.d_anchorReadIds.get(),
                ecinput.h_anchorReadIds.get(),
                (*ecinput.h_numAnchors.get()),
                deviceId,
                stream
            );

            gpuReadStorage->gatherSequenceLengthsToGpuBufferAsync(
                ecinput.d_anchor_sequences_lengths.get(),
                deviceId,
                ecinput.d_anchorReadIds.get(),
                (*ecinput.h_numAnchors.get()),
                stream
            );
        }

        void getCandidateReadIdsWithMinhashing(GpuErrorCorrectorInput& ecinput, cudaStream_t stream){
            ForLoopExecutor forLoopExecutor(threadPool, &pforHandle);

            // helpers::SimpleAllocationPinnedHost<read_number> d_candidate_read_idsAAAA(ecinput.d_candidate_read_ids.size());
            // helpers::SimpleAllocationPinnedHost<int> d_candidates_per_anchorAAAA(ecinput.d_candidates_per_anchor.size());
            // helpers::SimpleAllocationPinnedHost<int> d_candidates_per_anchor_prefixsumAAAA(ecinput.d_candidates_per_anchor_prefixsum.size());

            //gpuMinhasher->getIdsOfSimilarReadsNormalExcludingSelfNew(
            gpuMinhasher->getIdsOfSimilarReadsExcludingSelf(
                minhashHandle,
                ecinput.d_anchorReadIds.get(),
                ecinput.h_anchorReadIds.get(),
                ecinput.d_anchor_sequences_data.get(),
                encodedSequencePitchInInts,
                ecinput.d_anchor_sequences_lengths.get(),
                (*ecinput.h_numAnchors.get()),
                deviceId, 
                stream,
                forLoopExecutor,
                ecinput.d_candidate_read_ids.get(),
                ecinput.d_candidates_per_anchor.get(),
                ecinput.d_candidates_per_anchor_prefixsum.get()
            );

            // cudaMemset(d_candidate_read_idsAAAA.get(), 0, d_candidate_read_idsAAAA.sizeInBytes());
            // cudaMemset(d_candidates_per_anchorAAAA.get(), 0, d_candidates_per_anchorAAAA.sizeInBytes());
            // cudaMemset(d_candidates_per_anchor_prefixsumAAAA.get(), 0, d_candidates_per_anchor_prefixsumAAAA.sizeInBytes());

            // gpuMinhasher->getIdsOfSimilarReadsNormalExcludingSelfNew(
            //     minhashHandle,
            //     ecinput.d_anchorReadIds.get(),
            //     ecinput.h_anchorReadIds.get(),
            //     ecinput.d_anchor_sequences_data.get(),
            //     encodedSequencePitchInInts,
            //     ecinput.d_anchor_sequences_lengths.get(),
            //     (*ecinput.h_numAnchors.get()),
            //     deviceId, 
            //     stream,
            //     forLoopExecutor,
            //     d_candidate_read_idsAAAA.get(),
            //     d_candidates_per_anchorAAAA.get(),
            //     d_candidates_per_anchor_prefixsumAAAA.get()
            // );

            gpucorrectorkernels::copyMinhashResultsKernel<<<640, 256, 0, stream>>>(
                ecinput.d_numCandidates.get(),
                ecinput.h_numCandidates.get(),
                ecinput.h_candidate_read_ids.get(),
                ecinput.d_candidates_per_anchor_prefixsum.get(),
                ecinput.d_candidate_read_ids.get(),
                *ecinput.h_numAnchors.get()
            ); CUERR;

            // cudaStreamSynchronize(stream); CUERR;

            // bool error = false;

            // for(int i = 0; i < *ecinput.h_numAnchors.get() && !error; i++){
            //     if(ecinput.d_candidates_per_anchor[i] != d_candidates_per_anchorAAAA[i]){
            //         error = true;
            //         std::cerr << "error A " << i << "\n";
            //         break;
            //     }
            // }

            // for(int i = 0; i < (*ecinput.h_numAnchors.get()) + 1 && !error; i++){
            //     if(ecinput.d_candidates_per_anchor_prefixsum[i] != d_candidates_per_anchor_prefixsumAAAA[i]){
            //         error = true;
            //         std::cerr << "error B " << i << "\n";
            //         break;
            //     }
            // }

            // for(int i = 0; i < ecinput.d_candidates_per_anchor_prefixsum[(*ecinput.h_numAnchors.get())] && !error; i++){
            //     if(ecinput.h_candidate_read_ids[i] != d_candidate_read_idsAAAA[i]){
            //         error = true;
            //         std::cerr << "error C " << i << "\n";
            //         break;
            //     }
            // }

            // if(error){

            //     std::cerr << "d_candidates_per_anchor orig\n";
            //     for(int i = 0; i < *ecinput.h_numAnchors.get(); i++){
            //         std::cerr << ecinput.d_candidates_per_anchor[i] << ",";
            //     }
            //     std::cerr << "\n";

            //     std::cerr << "d_candidates_per_anchor new\n";
            //     for(int i = 0; i < *ecinput.h_numAnchors.get(); i++){
            //         std::cerr << d_candidates_per_anchorAAAA[i] << ",";
            //     }
            //     std::cerr << "\n";

            //     std::cerr << "d_candidates_per_anchor_prefixsum orig\n";
            //     for(int i = 0; i < (*ecinput.h_numAnchors.get())+1; i++){
            //         std::cerr << ecinput.d_candidates_per_anchor_prefixsum[i] << ",";
            //     }
            //     std::cerr << "\n";

            //     std::cerr << "d_candidates_per_anchor_prefixsum new\n";
            //     for(int i = 0; i < (*ecinput.h_numAnchors.get())+1; i++){
            //         std::cerr << d_candidates_per_anchor_prefixsumAAAA[i] << ",";
            //     }
            //     std::cerr << "\n";

            //     std::cerr << "d_candidates orig\n";
            //     for(int i = 0; i < ecinput.d_candidates_per_anchor_prefixsum[(*ecinput.h_numAnchors.get())]; i++){
            //         std::cerr << ecinput.h_candidate_read_ids[i] << ",";
            //     }
            //     std::cerr << "\n";

            //     std::cerr << "d_candidates new\n";
            //     for(int i = 0; i < d_candidates_per_anchor_prefixsumAAAA[(*ecinput.h_numAnchors.get())]; i++){
            //         std::cerr << d_candidate_read_idsAAAA[i] << ",";
            //     }
            //     std::cerr << "\n";

            //     assert(false);
            // }

            
        }
    
        int deviceId;
        int maxCandidatesPerRead;
        std::size_t encodedSequencePitchInInts;
        CudaStream backgroundStream;
        const DistributedReadStorage* gpuReadStorage;
        const GpuMinhasher* gpuMinhasher;
        const SequenceFileProperties* sequenceFileProperties;
        ThreadPool* threadPool;
        ThreadPool::ParallelForHandle pforHandle;
        DistributedReadStorage::GatherHandleSequences anchorSequenceGatherHandle;
        GpuMinhasher::QueryHandle minhashHandle;
    };


    class OutputConstructor{
    public:
        struct CorrectionOutput{
            std::vector<TempCorrectedSequence> anchorCorrections;
            std::vector<TempCorrectedSequence> candidateCorrections;
        };

        struct ReadCorrectionFlags{
            friend class OutputConstructor;
        public:
            ReadCorrectionFlags() = default;

            ReadCorrectionFlags(std::size_t numReads)
                : size(numReads), flags(std::make_unique<std::uint8_t[]>(numReads)){
                std::fill(flags.get(), flags.get() + size, 0);
            }

            std::size_t sizeInBytes() const noexcept{
                return size * sizeof(std::uint8_t);
            }

            bool isCorrectedAsHQAnchor(std::int64_t position) const noexcept{
                return (flags[position] & readCorrectedAsHQAnchor()) > 0;
            }

            bool isNotCorrectedAsAnchor(std::int64_t position) const noexcept{
                return (flags[position] & readCouldNotBeCorrectedAsAnchor()) > 0;
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

            std::size_t size;
            std::unique_ptr<std::uint8_t[]> flags{};
        };

        OutputConstructor() = default;

        OutputConstructor(
            ReadCorrectionFlags& correctionFlags_,
            const CorrectionOptions& correctionOptions_
        ) :
            correctionFlags{&correctionFlags_},
            correctionOptions{&correctionOptions_}
        {

        }

        template<class ForLoop>
        CorrectionOutput constructResults(const GpuErrorCorrectorRawOutput& currentOutput, ForLoop loopExecutor){
            assert(cudaSuccess == currentOutput.event.query());

            if(currentOutput.nothingToDo){
                return CorrectionOutput{};
            }

            std::vector<int> anchorIndicesToProcess;
            std::vector<std::pair<int,int>> candidateIndicesToProcess;

            anchorIndicesToProcess.reserve(currentOutput.numAnchors);
            if(correctionOptions->correctCandidates){
                candidateIndicesToProcess.reserve(16 * currentOutput.numAnchors);
            }

            nvtx::push_range("preprocess anchor results",0);

            for(int anchor_index = 0; anchor_index < currentOutput.numAnchors; anchor_index++){
                const read_number readId = currentOutput.h_anchorReadIds[anchor_index];
                const bool isCorrected = currentOutput.h_anchor_is_corrected[anchor_index];
                const bool isHQ = currentOutput.h_is_high_quality_anchor[anchor_index].hq();

                if(isHQ){
                    correctionFlags->setCorrectedAsHqAnchor(readId);
                }

                if(isCorrected){
                    anchorIndicesToProcess.emplace_back(anchor_index);
                }else{
                    correctionFlags->setCouldNotBeCorrectedAsAnchor(readId);
                }

                assert(!(isHQ && !isCorrected));
            }

            nvtx::pop_range();

            if(correctionOptions->correctCandidates){

                nvtx::push_range("preprocess candidate results",0);

                for(int anchor_index = 0; anchor_index < currentOutput.numAnchors; anchor_index++){

                    const int globalOffset = currentOutput.h_num_corrected_candidates_per_anchor_prefixsum[anchor_index];
                    const int n_corrected_candidates = currentOutput.h_num_corrected_candidates_per_anchor[anchor_index];

                    const int* const my_indices_of_corrected_candidates = currentOutput.h_indices_of_corrected_candidates
                                                        + globalOffset;

                    for(int i = 0; i < n_corrected_candidates; ++i) {
                        const int global_candidate_index = my_indices_of_corrected_candidates[i];
                        const read_number candidate_read_id = currentOutput.h_candidate_read_ids[global_candidate_index];

                        if (!correctionFlags->isCorrectedAsHQAnchor(candidate_read_id)) {
                            candidateIndicesToProcess.emplace_back(std::make_pair(anchor_index, i));
                        }
                    }
                }

                nvtx::pop_range();

            }

            const int numCorrectedAnchors = anchorIndicesToProcess.size();
            const int numCorrectedCandidates = candidateIndicesToProcess.size();

            // std::cerr << "numCorrectedAnchors: " << numCorrectedAnchors << 
            //     ", numCorrectedCandidates: " << numCorrectedCandidates << "\n";

            CorrectionOutput correctionOutput;
            correctionOutput.anchorCorrections.resize(numCorrectedAnchors);

            if(correctionOptions->correctCandidates){
                correctionOutput.candidateCorrections.resize(numCorrectedCandidates);
            }

            auto unpackAnchors = [&](int begin, int end){
                nvtx::push_range("Anchor unpacking", 3);
                            
                for(int positionInVector = begin; positionInVector < end; ++positionInVector) {
                    const int anchor_index = anchorIndicesToProcess[positionInVector];

                    auto& tmp = correctionOutput.anchorCorrections[positionInVector];
                    
                    const read_number readId = currentOutput.h_anchorReadIds[anchor_index];

                    tmp.hq = currentOutput.h_is_high_quality_anchor[anchor_index].hq();                    
                    tmp.type = TempCorrectedSequence::Type::Anchor;
                    tmp.readId = readId;
                    
                    const int numEdits = currentOutput.h_numEditsPerCorrectedanchor[positionInVector];
                    if(numEdits != currentOutput.doNotUseEditsValue){
                        tmp.edits.resize(numEdits);
                        const TempCorrectedSequence::EncodedEdit* const gpuedits 
                            = (const TempCorrectedSequence::EncodedEdit*)(((const char*)currentOutput.h_editsPerCorrectedanchor.get()) 
                                + positionInVector * currentOutput.editsPitchInBytes);
                        std::copy_n(gpuedits, numEdits, tmp.edits.begin());
                        tmp.useEdits = true;
                    }else{
                        tmp.edits.clear();
                        tmp.useEdits = false;

                        const char* const my_corrected_anchor_data = currentOutput.h_corrected_anchors + anchor_index * currentOutput.decodedSequencePitchInBytes;
                        const int anchor_length = currentOutput.h_anchor_sequences_lengths[anchor_index];   
                        tmp.sequence.assign(my_corrected_anchor_data, anchor_length);
                    }
                }

                nvtx::pop_range();
            };

            auto unpackcandidates = [&](int begin, int end){
                nvtx::push_range("candidate unpacking", 3);

                for(int positionInVector = begin; positionInVector < end; ++positionInVector) {
                    //TIMERSTARTCPU(setup);
                    const int anchor_index = candidateIndicesToProcess[positionInVector].first;
                    const int candidateIndex = candidateIndicesToProcess[positionInVector].second;
                    const read_number anchorReadId = currentOutput.h_anchorReadIds[anchor_index];

                    auto& tmp = correctionOutput.candidateCorrections[positionInVector];

                    const size_t offsetForCorrectedCandidateData = currentOutput.h_num_corrected_candidates_per_anchor_prefixsum[anchor_index];

                    const char* const my_corrected_candidates_data = currentOutput.h_corrected_candidates
                                                    + offsetForCorrectedCandidateData * currentOutput.decodedSequencePitchInBytes;
                    const int* const my_indices_of_corrected_candidates = currentOutput.h_indices_of_corrected_candidates
                                                    + offsetForCorrectedCandidateData;

                    const TempCorrectedSequence::EncodedEdit* const my_editsPerCorrectedCandidate 
                        = (const TempCorrectedSequence::EncodedEdit*)(((const char*)currentOutput.h_editsPerCorrectedCandidate.get()) 
                            + offsetForCorrectedCandidateData * currentOutput.editsPitchInBytes);


                    const int global_candidate_index = my_indices_of_corrected_candidates[candidateIndex];
                    //std::cerr << global_candidate_index << "\n";
                    const read_number candidate_read_id = currentOutput.h_candidate_read_ids[global_candidate_index];

                    const int candidate_shift = currentOutput.h_alignment_shifts[global_candidate_index];

                    if(correctionOptions->new_columns_to_correct < candidate_shift){
                        std::cerr << "readid " << anchorReadId << " candidate readid " << candidate_read_id << " : "
                        << candidate_shift << " " << correctionOptions->new_columns_to_correct <<"\n";

                        assert(correctionOptions->new_columns_to_correct >= candidate_shift);
                    }                
                    
                    tmp.type = TempCorrectedSequence::Type::Candidate;
                    tmp.shift = candidate_shift;
                    tmp.readId = candidate_read_id;
                    
                    const int numEdits = currentOutput.h_numEditsPerCorrectedCandidate[offsetForCorrectedCandidateData + candidateIndex];

                    if(numEdits != currentOutput.doNotUseEditsValue){
                        tmp.edits.resize(numEdits);
                        const TempCorrectedSequence::EncodedEdit* gpuedits 
                            = (const TempCorrectedSequence::EncodedEdit*)(((const char*)my_editsPerCorrectedCandidate) 
                                + candidateIndex * currentOutput.editsPitchInBytes);
                        std::copy_n(gpuedits, numEdits, tmp.edits.begin());
                        tmp.useEdits = true;
                    }else{
                        const int candidate_length = currentOutput.h_candidate_sequences_lengths[global_candidate_index];
                        const char* const candidate_data = my_corrected_candidates_data + candidateIndex * currentOutput.decodedSequencePitchInBytes;
                        //tmp.sequence = std::string{candidate_data, candidate_data + candidate_length};
                        tmp.sequence.assign(candidate_data, candidate_length);
                        tmp.edits.clear();
                        tmp.useEdits = false;
                    }
                }

                nvtx::pop_range();
            };


            if(!correctionOptions->correctCandidates){
                loopExecutor(0, numCorrectedAnchors, [=](auto begin, auto end, auto /*threadId*/){
                    unpackAnchors(begin, end);
                });
            }else{
        
  
                loopExecutor(0, numCorrectedAnchors, [=](auto begin, auto end, auto /*threadId*/){
                    unpackAnchors(begin, end);
                });
           

                loopExecutor(0, numCorrectedCandidates, [=](auto begin, auto end, auto /*threadId*/){
                        unpackcandidates(begin, end);
                    } //,  threadPool->getConcurrency() * 4
                );         
            }

            return correctionOutput;
        }
    
    private:
        ReadCorrectionFlags* correctionFlags;
        const CorrectionOptions* correctionOptions;
    };

    class GpuErrorCorrector{

    public:

        template<class T>
        using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T>;

        template<class T>
        using DeviceBuffer = helpers::SimpleAllocationDevice<T>;
        //using DeviceBuffer = helpers::SimpleAllocationPinnedHost<T>;


        static constexpr int getNumRefinementIterations() noexcept{
            return 5;
        }

        static constexpr bool useMsaRefinement() noexcept{
            return getNumRefinementIterations() > 0;
        }

        GpuErrorCorrector() = default;

        GpuErrorCorrector(
            const DistributedReadStorage& gpuReadStorage_,
            const GpuMinhasher& gpuMinhasher_,
            const CorrectionOptions& correctionOptions_,
            const GoodAlignmentProperties& goodAlignmentProperties_,
            const SequenceFileProperties& sequenceFileProperties_,
            int maxAnchorsPerCall,
            ThreadPool* threadPool_
        ) : 
            maxAnchors{maxAnchorsPerCall},
            maxCandidates{0},
            gpuReadStorage{&gpuReadStorage_},
            gpuMinhasher{&gpuMinhasher_},
            correctionOptions{&correctionOptions_},
            goodAlignmentProperties{&goodAlignmentProperties_},
            sequenceFileProperties{&sequenceFileProperties_},
            threadPool{threadPool_}
        {
            cudaGetDevice(&deviceId); CUERR;

            GpuMinhasher::QueryHandle minhashHandle = GpuMinhasher::makeQueryHandle();

            kernelLaunchHandle = make_kernel_launch_handle(deviceId);

            for(auto& event: events){
                event = std::move(CudaEvent{cudaEventDisableTiming});
            }
            backgroundStream = CudaStream{};

            encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(sequenceFileProperties->maxSequenceLength);
            decodedSequencePitchInBytes = SDIV(sequenceFileProperties->maxSequenceLength, 4) * 4;
            qualityPitchInBytes = SDIV(sequenceFileProperties->maxSequenceLength, 32) * 32;
            maxNumEditsPerSequence = std::max(1,sequenceFileProperties->maxSequenceLength / 7);
            //pad to multiple of 128 bytes
            editsPitchInBytes = SDIV(maxNumEditsPerSequence * sizeof(TempCorrectedSequence::EncodedEdit), 128) * 128;

            const std::size_t min_overlap = std::max(
                1, 
                std::max(
                    goodAlignmentProperties->min_overlap, 
                    int(sequenceFileProperties->maxSequenceLength * goodAlignmentProperties->min_overlap_ratio)
                )
            );
            const std::size_t msa_max_column_count = (3*sequenceFileProperties->maxSequenceLength - 2*min_overlap);
            //round up to 32 elements
            msaColumnPitchInElements = SDIV(msa_max_column_count, 32) * 32;

            anchorSequenceGatherHandle = gpuReadStorage->makeGatherHandleSequences();
            candidateSequenceGatherHandle = gpuReadStorage->makeGatherHandleSequences();
            anchorQualitiesGatherHandle = gpuReadStorage->makeGatherHandleQualities();
            candidateQualitiesGatherHandle = gpuReadStorage->makeGatherHandleQualities();

            initFixedSizeBuffers();
        }

        void correct(GpuErrorCorrectorInput& input, GpuErrorCorrectorRawOutput& output, cudaStream_t stream){
            assert(cudaSuccess == input.event.query());
            assert(cudaSuccess == output.event.query());

            currentInput = &input;
            currentOutput = &output;

            assert(*currentInput->h_numAnchors.get() <= maxAnchors);

            currentNumAnchors = *currentInput->h_numAnchors.get();
            currentNumCandidates = *currentInput->h_numCandidates.get();

            currentOutput->nothingToDo = false;
            currentOutput->numAnchors = currentNumAnchors;
            currentOutput->numCandidates = currentNumCandidates;
            currentOutput->doNotUseEditsValue = getDoNotUseEditsValue();
            currentOutput->editsPitchInBytes = editsPitchInBytes;
            currentOutput->decodedSequencePitchInBytes = decodedSequencePitchInBytes;

            if(currentNumCandidates == 0){
                currentOutput->nothingToDo = true;
                return;
            }
            

            int curId = 0;
            cudaGetDevice(&curId); CUERR;
            cudaSetDevice(deviceId); CUERR;

            resizeBuffers(currentNumAnchors, currentNumCandidates);

            //gpuMemsetZero(stream);

#if 0
            cudaMemcpyAsync(
                d_numAnchors.get(),
                currentInput->d_numAnchors.get(),
                sizeof(int),
                D2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                d_numCandidates.get(),
                currentInput->d_numCandidates.get(),
                sizeof(int),
                D2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                d_anchorReadIds.get(),
                currentInput->d_anchorReadIds.get(),
                sizeof(read_number) * (*currentInput->h_numAnchors.get()),
                D2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                d_candidate_read_ids.get(),
                currentInput->d_candidate_read_ids.get(),
                sizeof(read_number) * (*currentInput->h_numCandidates.get()),
                D2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                d_anchor_sequences_data.get(),
                currentInput->d_anchor_sequences_data.get(),
                encodedSequencePitchInInts* (*currentInput->h_numAnchors.get()),
                D2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                d_anchor_sequences_lengths.get(),
                currentInput->d_anchor_sequences_lengths.get(),
                sizeof(int) * (*currentInput->h_numAnchors.get()),
                D2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                d_candidates_per_anchor.get(),
                currentInput->d_candidates_per_anchor.get(),
                sizeof(int) * (*currentInput->h_numAnchors.get()),
                D2D,
                stream
            ); CUERR;

            cudaMemcpyAsync(
                d_candidates_per_anchor_prefixsum.get(),
                currentInput->d_candidates_per_anchor_prefixsum.get(),
                sizeof(int) * ((*currentInput->h_numAnchors.get()) + 1),
                D2D,
                stream
            ); CUERR;
#else 

            gpucorrectorkernels::copyCorrectionInputDeviceData<<<32768,256, 0, stream>>>(
                d_numAnchors,
                d_numCandidates,
                d_anchorReadIds,
                d_anchor_sequences_data,
                d_anchor_sequences_lengths,
                d_candidate_read_ids,
                d_candidates_per_anchor,
                d_candidates_per_anchor_prefixsum,
                encodedSequencePitchInInts,
                currentInput->d_numAnchors,
                currentInput->d_numCandidates,
                currentInput->d_anchorReadIds,
                currentInput->d_anchor_sequences_data,
                currentInput->d_anchor_sequences_lengths,
                currentInput->d_candidate_read_ids,
                currentInput->d_candidates_per_anchor,
                currentInput->d_candidates_per_anchor_prefixsum
            ); CUERR;

#endif

            //after gpu data has been copied to local working set, the gpu data of currentInput can be reused
            currentInput->event.record(stream);

            nvtx::push_range("getCandidateSequenceData", 3);
            getCandidateSequenceData(stream); 
            nvtx::pop_range();

            if(correctionOptions->useQualityScores) {
                nvtx::push_range("getQualities", 4);

                getQualities(stream);

                nvtx::pop_range();
            }

            //execute(stream);
            graphMap[currentOutput].execute(stream);

            //fill missing output arrays. This may overlap with gpu execution
            std::copy_n(currentInput->h_anchorReadIds.get(), currentNumAnchors, currentOutput->h_anchorReadIds.get());
            std::copy_n(currentInput->h_candidate_read_ids.get(), currentNumCandidates, currentOutput->h_candidate_read_ids.get());

            //after the current work in stream is completed, all results in currentOutput are ready to use.
            cudaEventRecord(currentOutput->event, stream); CUERR;
        }

        


    private:

        void gpuMemsetZero(cudaStream_t stream){
            auto zero = [&](auto& devicebuffer){
                cudaMemsetAsync(devicebuffer.get(), 0, devicebuffer.sizeInBytes(), stream);
            };

            zero(d_candidates_per_anchor_tmp);
            zero(d_anchorContainsN);
            zero(d_anchor_qualities);
 
            zero(d_indices_per_anchor);
            zero(d_num_indices);
            zero(d_indices_per_anchor_tmp);
            zero(d_num_indices_tmp);
            zero(d_consensus);
            zero(d_support);
            zero(d_coverage);
            zero(d_origWeights);
            zero(d_origCoverages);
            zero(d_msa_column_properties);
            zero(d_counts);
            zero(d_weights);
            zero(d_corrected_anchors);
            zero(d_num_corrected_candidates_per_anchor);
            zero(d_num_corrected_candidates_per_anchor_prefixsum);
            zero(d_num_total_corrected_candidates);
            zero(d_anchor_is_corrected);
            zero(d_is_high_quality_anchor);
            zero(d_high_quality_anchor_indices);
            zero(d_num_high_quality_anchor_indices);
            zero(d_editsPerCorrectedanchor);
            zero(d_numEditsPerCorrectedanchor);
            zero(d_indices_of_corrected_anchors);
            zero(d_num_indices_of_corrected_anchors);
            zero(d_numAnchors);
            zero(d_numCandidates);
            zero(d_anchorReadIds);
            zero(d_anchor_sequences_data);
            zero(d_anchor_sequences_lengths);
            zero(d_candidates_per_anchor);
            zero(d_candidates_per_anchor_prefixsum);
            zero(d_anchorIndicesOfCandidates);
            zero(d_candidateContainsN);
            zero(d_candidate_read_ids);
            zero(d_candidate_sequences_lengths);
            zero(d_candidate_sequences_data);
            zero(d_transposedCandidateSequencesData);            
            zero(d_candidate_qualities);
            zero(d_alignment_overlaps);
            zero(d_alignment_shifts);
            zero(d_alignment_nOps);
            zero(d_alignment_isValid);
            zero(d_alignment_best_alignment_flags);
            zero(d_indices);
            zero(d_indices_tmp);
            zero(d_corrected_candidates);
            zero(d_editsPerCorrectedCandidate);
            zero(d_numEditsPerCorrectedCandidate);
            zero(d_indices_of_corrected_candidates);
            
        }

        void initFixedSizeBuffers(){
            const std::size_t numEditsAnchors = SDIV(editsPitchInBytes * maxAnchors, sizeof(TempCorrectedSequence::EncodedEdit));          

            //does not depend on number of candidates
            h_indices_per_anchor.resize(maxAnchors);          
            h_high_quality_anchor_indices.resize(maxAnchors);
            h_num_high_quality_anchor_indices.resize(1);
            h_num_total_corrected_candidates.resize(1);        

            //does not depend on number of candidates
            d_candidates_per_anchor_tmp.resize(maxAnchors);
            d_anchorContainsN.resize(maxAnchors);

            if(correctionOptions->useQualityScores){
                d_anchor_qualities.resize(maxAnchors * qualityPitchInBytes);
            }

            d_indices_per_anchor.resize(maxAnchors);
            d_num_indices.resize(1);
            d_indices_per_anchor_tmp.resize(maxAnchors);
            d_num_indices_tmp.resize(1);
            d_consensus.resize(maxAnchors * msaColumnPitchInElements);
            d_support.resize(maxAnchors * msaColumnPitchInElements);
            d_coverage.resize(maxAnchors * msaColumnPitchInElements);
            d_origWeights.resize(maxAnchors * msaColumnPitchInElements);
            d_origCoverages.resize(maxAnchors * msaColumnPitchInElements);
            d_msa_column_properties.resize(maxAnchors);
            d_counts.resize(maxAnchors * 4 * msaColumnPitchInElements);
            d_weights.resize(maxAnchors * 4 * msaColumnPitchInElements);
            d_corrected_anchors.resize(maxAnchors * decodedSequencePitchInBytes);
            d_num_corrected_candidates_per_anchor.resize(maxAnchors);
            d_num_corrected_candidates_per_anchor_prefixsum.resize(maxAnchors);
            d_num_total_corrected_candidates.resize(1);
            d_anchor_is_corrected.resize(maxAnchors);
            d_is_high_quality_anchor.resize(maxAnchors);
            d_high_quality_anchor_indices.resize(maxAnchors);
            d_num_high_quality_anchor_indices.resize(1); 
            d_editsPerCorrectedanchor.resize(numEditsAnchors);
            d_numEditsPerCorrectedanchor.resize(maxAnchors);
            d_indices_of_corrected_anchors.resize(maxAnchors);
            d_num_indices_of_corrected_anchors.resize(1);

            d_numAnchors.resize(1);
            d_numCandidates.resize(1);
            d_anchorReadIds.resize(maxAnchors);
            d_anchor_sequences_data.resize(encodedSequencePitchInInts * maxAnchors);
            d_anchor_sequences_lengths.resize(maxAnchors);
            d_candidates_per_anchor.resize(maxAnchors);
            d_candidates_per_anchor_prefixsum.resize(maxAnchors + 1);
        }
 
        void resizeBuffers(int numReads, int numCandidates){  
            assert(numReads <= maxAnchors);

            bool maxCandidatesDidChange = false;
            constexpr int stepsizeForMaxCandidates = 10000;
            if(numCandidates > maxCandidates){
                //round up numCandidates to next multiple of stepsize
                maxCandidates = SDIV(numCandidates, stepsizeForMaxCandidates) * stepsizeForMaxCandidates;
                maxCandidatesDidChange = true;
            }

            std::size_t numEditsCandidates = SDIV(editsPitchInBytes * maxCandidates, sizeof(TempCorrectedSequence::EncodedEdit));

            const std::size_t numEditsAnchors = SDIV(editsPitchInBytes * maxAnchors, sizeof(TempCorrectedSequence::EncodedEdit));          

            //does not depend on number of candidates
            bool outputBuffersReallocated = false;
            outputBuffersReallocated |= currentOutput->h_anchor_sequences_lengths.resize(maxAnchors);
            outputBuffersReallocated |= currentOutput->h_corrected_anchors.resize(maxAnchors * decodedSequencePitchInBytes);            
            outputBuffersReallocated |= currentOutput->h_anchor_is_corrected.resize(maxAnchors);
            outputBuffersReallocated |= currentOutput->h_is_high_quality_anchor.resize(maxAnchors);
            outputBuffersReallocated |= currentOutput->h_editsPerCorrectedanchor.resize(numEditsAnchors);
            outputBuffersReallocated |= currentOutput->h_numEditsPerCorrectedanchor.resize(maxAnchors);            
            outputBuffersReallocated |= currentOutput->h_num_corrected_candidates_per_anchor.resize(maxAnchors);
            outputBuffersReallocated |= currentOutput->h_num_corrected_candidates_per_anchor_prefixsum.resize(maxAnchors);
            outputBuffersReallocated |= currentOutput->h_candidate_sequences_lengths.resize(maxCandidates);
            outputBuffersReallocated |= currentOutput->h_corrected_candidates.resize(maxCandidates * decodedSequencePitchInBytes);
            outputBuffersReallocated |= currentOutput->h_editsPerCorrectedCandidate.resize(numEditsCandidates);
            outputBuffersReallocated |= currentOutput->h_numEditsPerCorrectedCandidate.resize(maxCandidates);
            outputBuffersReallocated |= currentOutput->h_indices_of_corrected_candidates.resize(maxCandidates);
            outputBuffersReallocated |= currentOutput->h_alignment_shifts.resize(maxCandidates);
            outputBuffersReallocated |= currentOutput->h_candidate_read_ids.resize(maxCandidates);
            outputBuffersReallocated |= currentOutput->h_anchorReadIds.resize(maxAnchors);
            
            d_anchorIndicesOfCandidates.resize(maxCandidates);
            d_candidateContainsN.resize(maxCandidates);
            d_candidate_read_ids.resize(maxCandidates);
            d_candidate_sequences_lengths.resize(maxCandidates);
            d_candidate_sequences_data.resize(maxCandidates * encodedSequencePitchInInts);
            d_transposedCandidateSequencesData.resize(maxCandidates * encodedSequencePitchInInts);
            
            if(correctionOptions->useQualityScores){
                d_candidate_qualities.resize(maxCandidates * qualityPitchInBytes);
            }
            
            d_alignment_overlaps.resize(maxCandidates);
            d_alignment_shifts.resize(maxCandidates);
            d_alignment_nOps.resize(maxCandidates);
            d_alignment_isValid.resize(maxCandidates);
            d_alignment_best_alignment_flags.resize(maxCandidates);
            d_indices.resize(maxCandidates);
            d_indices_tmp.resize(maxCandidates);
            d_corrected_candidates.resize(maxCandidates * decodedSequencePitchInBytes);
            d_editsPerCorrectedCandidate.resize(numEditsCandidates);
            d_numEditsPerCorrectedCandidate.resize(maxCandidates);
            d_indices_of_corrected_candidates.resize(maxCandidates);

            std::size_t flagTemp = sizeof(bool) * maxCandidates;
            std::size_t popcountShdTempBytes = 0; 
            
            const bool removeAmbiguousAnchors = correctionOptions->excludeAmbiguousReads;
            const bool removeAmbiguousCandidates = correctionOptions->excludeAmbiguousReads;
    
            call_popcount_shifted_hamming_distance_kernel_async(
                nullptr,
                popcountShdTempBytes,
                d_alignment_overlaps.get(),
                d_alignment_shifts.get(),
                d_alignment_nOps.get(),
                d_alignment_isValid.get(),
                d_alignment_best_alignment_flags.get(),
                d_anchor_sequences_data.get(),
                d_candidate_sequences_data.get(),
                d_anchor_sequences_lengths.get(),
                d_candidate_sequences_lengths.get(),
                d_candidates_per_anchor_prefixsum.get(),
                d_candidates_per_anchor.get(),
                d_anchorIndicesOfCandidates.get(),
                d_numAnchors.get(),
                d_numCandidates.get(),
                d_anchorContainsN.get(),
                removeAmbiguousAnchors,
                d_candidateContainsN.get(),
                removeAmbiguousCandidates,
                maxAnchors,
                maxCandidates,
                sequenceFileProperties->maxSequenceLength,
                encodedSequencePitchInInts,
                goodAlignmentProperties->min_overlap,
                goodAlignmentProperties->maxErrorRate,
                goodAlignmentProperties->min_overlap_ratio,
                correctionOptions->estimatedErrorrate,                
                (cudaStream_t)0,
                kernelLaunchHandle
            );
            
            // this buffer will also serve as temp storage for cub. The required memory for cub 
            // is less than popcountShdTempBytes.
            popcountShdTempBytes = std::max(flagTemp, popcountShdTempBytes);
            d_tempstorage.resize(popcountShdTempBytes);

            if(maxCandidatesDidChange){
                std::cerr << "maxCandidates changed to " << maxCandidates << "\n";
            }

            if(outputBuffersReallocated || !graphMap[currentOutput].valid){
                if(outputBuffersReallocated){
                    std::cerr << "outputBuffersReallocated " << currentOutput << "\n";
                }

                graphMap[currentOutput].capture(
                    [&](cudaStream_t capstream){
                        execute(capstream);
                    }
                );
            }
        }

        void execute(cudaStream_t stream){
            cudaEventRecord(events[0], stream); CUERR;
            cudaStreamWaitEvent(backgroundStream, events[0], 0); CUERR;

            cudaMemcpyAsync(
                currentOutput->h_anchor_sequences_lengths.get(),
                d_anchor_sequences_lengths.get(),
                sizeof(int) * maxAnchors,
                D2H,
                backgroundStream
            ); CUERR;

            nvtx::push_range("getCandidateAlignments", 5);
            getCandidateAlignments(stream); 
            nvtx::pop_range();

            nvtx::push_range("buildMultipleSequenceAlignment", 6);
            buildMultipleSequenceAlignment(stream);
            nvtx::pop_range();

            if(useMsaRefinement()){

                nvtx::push_range("refineMultipleSequenceAlignment", 7);
                refineMultipleSequenceAlignment(stream);
                nvtx::pop_range();

            }


            nvtx::push_range("correctanchors", 8);
            correctanchors(stream);
            nvtx::pop_range();

            if(correctionOptions->correctCandidates) {                        

                nvtx::push_range("correctCandidates", 9);
                correctCandidates(stream);
                nvtx::pop_range();
                
            }

            cudaEventRecord(events[0], backgroundStream); CUERR;
            cudaStreamWaitEvent(stream, events[0], 0); CUERR;
        };


        void getCandidateSequenceData(cudaStream_t stream){

            gpuReadStorage->readsContainN_async(
                deviceId,
                d_anchorContainsN.get(), 
                d_anchorReadIds.get(), 
                d_numAnchors,
                maxAnchors, 
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
                currentNumCandidates,            
                stream
            );

            gpuReadStorage->gatherSequenceDataToGpuBufferAsync(
                threadPool,
                candidateSequenceGatherHandle,
                d_candidate_sequences_data.get(),
                encodedSequencePitchInInts,
                currentInput->h_candidate_read_ids,
                d_candidate_read_ids,
                currentNumCandidates,
                deviceId,
                stream
            );

            helpers::call_transpose_kernel(
                d_transposedCandidateSequencesData.get(), 
                d_candidate_sequences_data.get(), 
                currentNumCandidates, 
                encodedSequencePitchInInts, 
                encodedSequencePitchInInts, 
                stream
            );

            cudaEventRecord(events[0], stream); CUERR;
            cudaStreamWaitEvent(backgroundStream, events[0], 0); CUERR;

            cudaMemcpyAsync(
                currentOutput->h_candidate_sequences_lengths,
                d_candidate_sequences_lengths,
                sizeof(int) * currentNumCandidates,
                D2H,
                backgroundStream
            ); CUERR;
        }

        void getQualities(cudaStream_t stream){

            if(correctionOptions->useQualityScores) {
                // std::cerr << "gather anchor qual\n";
                gpuReadStorage->gatherQualitiesToGpuBufferAsync(
                    threadPool,
                    anchorQualitiesGatherHandle,
                    d_anchor_qualities,
                    qualityPitchInBytes,
                    currentInput->h_anchorReadIds,
                    d_anchorReadIds,
                    maxAnchors,
                    deviceId,
                    stream
                );

                // std::cerr << "gather candidate qual\n";
                gpuReadStorage->gatherQualitiesToGpuBufferAsync(
                    threadPool,
                    candidateQualitiesGatherHandle,
                    d_candidate_qualities,
                    qualityPitchInBytes,
                    currentInput->h_candidate_read_ids.get(),
                    d_candidate_read_ids.get(),
                    currentNumCandidates,
                    deviceId,
                    stream
                );
            }

            //cudaStreamSynchronize(stream); CUERR;
        }

        void getCandidateAlignments(cudaStream_t stream){

            {
                
                gpucorrectorkernels::setAnchorIndicesOfCandidateskernel<1024, 128>
                        <<<1024, 128, 0, stream>>>(
                    d_anchorIndicesOfCandidates.get(),
                    d_numAnchors.get(),
                    d_candidates_per_anchor.get(),
                    d_candidates_per_anchor_prefixsum.get()
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
                d_anchor_sequences_data.get(),
                d_candidate_sequences_data.get(),
                d_anchor_sequences_lengths.get(),
                d_candidate_sequences_lengths.get(),
                d_candidates_per_anchor_prefixsum.get(),
                d_candidates_per_anchor.get(),
                d_anchorIndicesOfCandidates.get(),
                d_numAnchors.get(),
                d_numCandidates.get(),
                d_anchorContainsN.get(),
                removeAmbiguousAnchors,
                d_candidateContainsN.get(),
                removeAmbiguousCandidates,
                maxAnchors,
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
                d_candidates_per_anchor_prefixsum.get(),
                d_numAnchors.get(),
                d_numCandidates.get(),
                maxAnchors,
                maxCandidates,
                correctionOptions->estimatedErrorrate,
                correctionOptions->estimatedCoverage * correctionOptions->m_coverage,
                stream,
                kernelLaunchHandle
            );

            callSelectIndicesOfGoodCandidatesKernelAsync(
                d_indices.get(),
                d_indices_per_anchor.get(),
                d_num_indices.get(),
                d_alignment_best_alignment_flags.get(),
                d_candidates_per_anchor.get(),
                d_candidates_per_anchor_prefixsum.get(),
                d_anchorIndicesOfCandidates.get(),
                d_numAnchors.get(),
                d_numCandidates.get(),
                maxAnchors,
                maxCandidates,
                stream,
                kernelLaunchHandle
            );
        }

        void buildMultipleSequenceAlignment(cudaStream_t stream){

            GPUMultiMSA multiMSA;

            multiMSA.numMSAs = maxAnchors;
            multiMSA.columnPitchInElements = msaColumnPitchInElements;
            multiMSA.counts = d_counts.get();
            multiMSA.weights = d_weights.get();
            multiMSA.coverages = d_coverage.get();
            multiMSA.consensus = d_consensus.get();
            multiMSA.support = d_support.get();
            multiMSA.origWeights = d_origWeights.get();
            multiMSA.origCoverages = d_origCoverages.get();
            multiMSA.columnProperties = d_msa_column_properties.get();

            callConstructMultipleSequenceAlignmentsKernel_async(
                multiMSA,
                d_alignment_overlaps.get(),
                d_alignment_shifts.get(),
                d_alignment_nOps.get(),
                d_alignment_best_alignment_flags.get(),
                d_anchor_sequences_lengths.get(),
                d_candidate_sequences_lengths.get(),
                d_indices,
                d_indices_per_anchor,
                d_candidates_per_anchor_prefixsum,
                d_anchor_sequences_data.get(),
                d_candidate_sequences_data.get(),
                d_anchor_qualities.get(),
                d_candidate_qualities.get(),
                d_numAnchors.get(),
                goodAlignmentProperties->maxErrorRate,
                maxAnchors,
                maxCandidates,
                correctionOptions->useQualityScores,
                encodedSequencePitchInInts,
                qualityPitchInBytes,
                stream,
                kernelLaunchHandle
            );
        }

        void refineMultipleSequenceAlignment(cudaStream_t stream){

            std::array<int*,2> d_indices_dblbuf{
                d_indices.get(), 
                d_indices_tmp.get()
            };
            std::array<int*,2> d_indices_per_anchor_dblbuf{
                d_indices_per_anchor.get(), 
                d_indices_per_anchor_tmp.get()
            };
            std::array<int*,2> d_num_indices_dblbuf{
                d_num_indices.get(), 
                d_num_indices_tmp.get()
            };

            GPUMultiMSA multiMSA;

            multiMSA.numMSAs = maxAnchors;
            multiMSA.columnPitchInElements = msaColumnPitchInElements;
            multiMSA.counts = d_counts.get();
            multiMSA.weights = d_weights.get();
            multiMSA.coverages = d_coverage.get();
            multiMSA.consensus = d_consensus.get();
            multiMSA.support = d_support.get();
            multiMSA.origWeights = d_origWeights.get();
            multiMSA.origCoverages = d_origCoverages.get();
            multiMSA.columnProperties = d_msa_column_properties.get();

    #if 0        

            static_assert(getNumRefinementIterations() % 2 == 1, "");

            const std::size_t requiredTempStorageBytes = 
                SDIV(sizeof(bool) * maxCandidates, 128) * 128 // d_shouldBeKept + padding to align the next pointer
                + (sizeof(bool) * maxAnchors); // d_anchorIsFinished
            assert(d_tempstorage.sizeInBytes() >= requiredTempStorageBytes);
            
            bool* d_shouldBeKept = (bool*)d_tempstorage.get();
            bool* d_anchorIsFinished = d_shouldBeKept + (SDIV(sizeof(bool) * (*h_numCandidates.get()), 128) * 128);

            cudaMemsetAsync(d_anchorIsFinished, 0, sizeof(bool) * maxAnchors, stream);

            for(int iteration = 0; iteration < getNumRefinementIterations(); iteration++){

                callMsaCandidateRefinementKernel_singleiter_async(
                    d_indices_dblbuf[(1 + iteration) % 2],
                    d_indices_per_anchor_dblbuf[(1 + iteration) % 2],
                    d_num_indices_dblbuf[(1 + iteration) % 2],
                    multiMSA,
                    d_alignment_best_alignment_flags.get(),
                    d_alignment_shifts.get(),
                    d_alignment_nOps.get(),
                    d_alignment_overlaps.get(),
                    d_anchor_sequences_data.get(),
                    d_candidate_sequences_data.get(),
                    d_anchor_sequences_lengths.get(),
                    d_candidate_sequences_lengths.get(),
                    d_anchor_qualities.get(),
                    d_candidate_qualities.get(),
                    d_shouldBeKept,
                    d_candidates_per_anchor_prefixsum,
                    d_numAnchors.get(),
                    goodAlignmentProperties->maxErrorRate,
                    maxAnchors,
                    maxCandidates,
                    correctionOptions->useQualityScores,
                    encodedSequencePitchInInts,
                    qualityPitchInBytes,
                    d_indices_dblbuf[(0 + iteration) % 2],
                    d_indices_per_anchor_dblbuf[(0 + iteration) % 2],
                    correctionOptions->estimatedCoverage,
                    iteration,
                    d_anchorIsFinished,
                    stream,
                    kernelLaunchHandle
                );


            }

    #else 

            const std::size_t requiredTempStorageBytes = sizeof(bool) * maxCandidates; // d_shouldBeKept
                
            assert(d_tempstorage.sizeInBytes() >= requiredTempStorageBytes);

            bool* d_shouldBeKept = (bool*)d_tempstorage.get();

            callMsaCandidateRefinementKernel_multiiter_async(
                d_indices_dblbuf[1],
                d_indices_per_anchor_dblbuf[1],
                d_num_indices_dblbuf[1],
                multiMSA,
                d_alignment_best_alignment_flags.get(),
                d_alignment_shifts.get(),
                d_alignment_nOps.get(),
                d_alignment_overlaps.get(),
                d_anchor_sequences_data.get(),
                d_candidate_sequences_data.get(),
                d_anchor_sequences_lengths.get(),
                d_candidate_sequences_lengths.get(),
                d_anchor_qualities.get(),
                d_candidate_qualities.get(),
                d_shouldBeKept,
                d_candidates_per_anchor_prefixsum,
                d_numAnchors.get(),
                goodAlignmentProperties->maxErrorRate,
                maxAnchors,
                maxCandidates,
                correctionOptions->useQualityScores,
                encodedSequencePitchInInts,
                qualityPitchInBytes,
                d_indices.get(),
                d_indices_per_anchor.get(),
                correctionOptions->estimatedCoverage,
                getNumRefinementIterations(),
                stream,
                kernelLaunchHandle
            );
    #endif 

        }

        void correctanchors(cudaStream_t stream){

            const float avg_support_threshold = 1.0f - 1.0f * correctionOptions->estimatedErrorrate;
            const float min_support_threshold = 1.0f - 3.0f * correctionOptions->estimatedErrorrate;
            // coverage is always >= 1
            const float min_coverage_threshold = std::max(1.0f,
                correctionOptions->m_coverage / 6.0f * correctionOptions->estimatedCoverage);
            const float max_coverage_threshold = 0.5 * correctionOptions->estimatedCoverage;

            // correct anchors

            std::array<int*,2> d_indices_per_anchor_dblbuf{
                d_indices_per_anchor.get(), 
                d_indices_per_anchor_tmp.get()
            };
            std::array<int*,2> d_num_indices_dblbuf{
                d_num_indices.get(), 
                d_num_indices_tmp.get()
            };

            const int* d_indices_per_anchor = d_indices_per_anchor_dblbuf[/*getNumRefinementIterations() % 2*/ 1];
            const int* d_num_indices = d_num_indices_dblbuf[/*getNumRefinementIterations() % 2*/ 1];

            GPUMultiMSA multiMSA;

            multiMSA.numMSAs = maxAnchors;
            multiMSA.columnPitchInElements = msaColumnPitchInElements;
            multiMSA.counts = d_counts.get();
            multiMSA.weights = d_weights.get();
            multiMSA.coverages = d_coverage.get();
            multiMSA.consensus = d_consensus.get();
            multiMSA.support = d_support.get();
            multiMSA.origWeights = d_origWeights.get();
            multiMSA.origCoverages = d_origCoverages.get();
            multiMSA.columnProperties = d_msa_column_properties.get();


            call_msaCorrectAnchorsKernel_async(
                d_corrected_anchors.get(),
                d_anchor_is_corrected.get(),
                d_is_high_quality_anchor.get(),
                multiMSA,
                d_anchor_sequences_data.get(),
                d_candidate_sequences_data.get(),
                d_candidate_sequences_lengths.get(),
                d_indices_per_anchor,
                d_numAnchors.get(),
                maxAnchors,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                sequenceFileProperties->maxSequenceLength,
                correctionOptions->estimatedErrorrate,
                goodAlignmentProperties->maxErrorRate,
                avg_support_threshold,
                min_support_threshold,
                min_coverage_threshold,
                max_coverage_threshold,
                correctionOptions->kmerlength,
                sequenceFileProperties->maxSequenceLength,
                stream,
                kernelLaunchHandle
            );

            cudaEventRecord(events[0], stream); CUERR;
            cudaStreamWaitEvent(backgroundStream, events[0], 0); CUERR;

            //std::cerr << "cudaMemcpyAsync " << (void*)h_indices_per_anchor.get() << (void*) d_indices_per_anchor << "\n";
            cudaMemcpyAsync(
                h_indices_per_anchor,
                d_indices_per_anchor,
                sizeof(int) * maxAnchors,
                D2H,
                backgroundStream
            ); CUERR;

            cudaMemcpyAsync(
                currentOutput->h_corrected_anchors,
                d_corrected_anchors,
                decodedSequencePitchInBytes * maxAnchors,
                D2H,
                backgroundStream
            ); CUERR;
            cudaMemcpyAsync(
                currentOutput->h_anchor_is_corrected,
                d_anchor_is_corrected,
                sizeof(bool) * maxAnchors,
                D2H,
                backgroundStream
            ); CUERR;
            cudaMemcpyAsync(
                currentOutput->h_is_high_quality_anchor,
                d_is_high_quality_anchor,
                sizeof(AnchorHighQualityFlag) * maxAnchors,
                D2H,
                backgroundStream
            ); CUERR;

            // cudaMemcpyAsync(h_num_uncorrected_positions_per_anchor,
            //                 d_num_uncorrected_positions_per_anchor,
            //                 sizeof(int) * maxAnchors,
            //                 D2H,
            //                 backgroundStream); CUERR;

            // cudaMemcpyAsync(h_uncorrected_positions_per_anchor,
            //                 d_uncorrected_positions_per_anchor,
            //                 sizeof(int) * sequenceFileProperties.maxSequenceLength * maxAnchors,
            //                 D2H,
            //                 backgroundStream); CUERR;

            gpucorrectorkernels::selectIndicesOfFlagsOneBlock<256><<<1,256,0, stream>>>(
                d_indices_of_corrected_anchors.get(),
                d_num_indices_of_corrected_anchors.get(),
                d_anchor_is_corrected.get(),
                d_numAnchors.get()
            ); CUERR;

            callConstructAnchorResultsKernelAsync(
                d_editsPerCorrectedanchor.get(),
                d_numEditsPerCorrectedanchor.get(),
                getDoNotUseEditsValue(),
                d_indices_of_corrected_anchors.get(),
                d_num_indices_of_corrected_anchors.get(),
                d_anchorContainsN.get(),
                d_anchor_sequences_data.get(),
                d_anchor_sequences_lengths.get(),
                d_corrected_anchors.get(),
                maxNumEditsPerSequence,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                editsPitchInBytes,
                d_numAnchors.get(),
                maxAnchors,
                stream,
                kernelLaunchHandle
            );

            cudaEventRecord(events[0], stream); CUERR;
            cudaStreamWaitEvent(backgroundStream, events[0], 0); CUERR;

            cudaMemcpyAsync(
                currentOutput->h_editsPerCorrectedanchor,
                d_editsPerCorrectedanchor,
                editsPitchInBytes * maxAnchors,
                D2H,
                backgroundStream
            ); CUERR;

            cudaMemcpyAsync(
                currentOutput->h_numEditsPerCorrectedanchor,
                d_numEditsPerCorrectedanchor,
                sizeof(int) * maxAnchors,
                D2H,
                backgroundStream
            ); CUERR;
            
        }

        void correctCandidates(cudaStream_t stream){

            const float min_support_threshold = 1.0f-3.0f*correctionOptions->estimatedErrorrate;
            // coverage is always >= 1
            const float min_coverage_threshold = std::max(1.0f,
                correctionOptions->m_coverage / 6.0f * correctionOptions->estimatedCoverage);
            const int new_columns_to_correct = correctionOptions->new_columns_to_correct;

            bool* const d_candidateCanBeCorrected = d_alignment_isValid.get(); //repurpose

            cub::TransformInputIterator<bool, IsHqAnchor, AnchorHighQualityFlag*>
                d_isHqanchor(d_is_high_quality_anchor, IsHqAnchor{});

            gpucorrectorkernels::selectIndicesOfFlagsOneBlock<256><<<1,256,0, stream>>>(
                d_high_quality_anchor_indices.get(),
                d_num_high_quality_anchor_indices.get(),
                d_isHqanchor,
                d_numAnchors.get()
            ); CUERR;

            cudaEventRecord(events[0], stream); CUERR;
            cudaStreamWaitEvent(backgroundStream, events[0], 0); CUERR;

            cudaMemcpyAsync(
                h_high_quality_anchor_indices.get(),
                d_high_quality_anchor_indices.get(),
                sizeof(int) * maxAnchors,
                D2H,
                backgroundStream
            ); CUERR;

            cudaMemcpyAsync(
                h_num_high_quality_anchor_indices.get(),
                d_num_high_quality_anchor_indices.get(),
                sizeof(int),
                D2H,
                backgroundStream
            ); CUERR;

            gpucorrectorkernels::initArraysBeforeCandidateCorrectionKernel<<<640, 128, 0, stream>>>(
                d_numCandidates.get(),
                d_numAnchors.get(),
                d_num_corrected_candidates_per_anchor.get(),
                d_candidateCanBeCorrected
            ); CUERR;

            std::array<int*,2> d_indices_dblbuf{
                d_indices.get(), 
                d_indices_tmp.get()
            };
            std::array<int*,2> d_indices_per_anchor_dblbuf{
                d_indices_per_anchor.get(), 
                d_indices_per_anchor_tmp.get()
            };
            std::array<int*,2> d_num_indices_dblbuf{
                d_num_indices.get(), 
                d_num_indices_tmp.get()
            };

            GPUMultiMSA multiMSA;

            multiMSA.numMSAs = maxAnchors;
            multiMSA.columnPitchInElements = msaColumnPitchInElements;
            multiMSA.counts = d_counts.get();
            multiMSA.weights = d_weights.get();
            multiMSA.coverages = d_coverage.get();
            multiMSA.consensus = d_consensus.get();
            multiMSA.support = d_support.get();
            multiMSA.origWeights = d_origWeights.get();
            multiMSA.origCoverages = d_origCoverages.get();
            multiMSA.columnProperties = d_msa_column_properties.get();

            callFlagCandidatesToBeCorrectedKernel_async(
                d_candidateCanBeCorrected,
                d_num_corrected_candidates_per_anchor.get(),
                multiMSA,
                d_alignment_shifts.get(),
                d_candidate_sequences_lengths.get(),
                d_anchorIndicesOfCandidates.get(),
                d_is_high_quality_anchor.get(),
                d_candidates_per_anchor_prefixsum,
                d_indices_dblbuf[/*getNumRefinementIterations() % 2*/1],
                d_indices_per_anchor_dblbuf[/*getNumRefinementIterations() % 2*/1],
                d_numAnchors,
                d_numCandidates,
                min_support_threshold,
                min_coverage_threshold,
                new_columns_to_correct,
                stream,
                kernelLaunchHandle
            );

            size_t cubTempSize = d_tempstorage.sizeInBytes();

            cub::DeviceSelect::Flagged(
                d_tempstorage.get(),
                cubTempSize,
                cub::CountingInputIterator<int>(0),
                d_candidateCanBeCorrected,
                d_indices_of_corrected_candidates.get(),
                d_num_total_corrected_candidates.get(),
                maxCandidates,
                stream
            ); CUERR;

            cudaEventRecord(events[0], stream); CUERR;
            cudaStreamWaitEvent(backgroundStream, events[0], 0); CUERR;


            //start result transfer of already calculated data in second stream

            cudaMemcpyAsync(
                h_num_total_corrected_candidates.get(),
                d_num_total_corrected_candidates.get(),
                sizeof(int),
                D2H,
                backgroundStream
            ); CUERR;

            //cudaEventRecord(events[numTotalCorrectedCandidates_event_index], backgroundStream); CUERR;

            cub::DeviceScan::ExclusiveSum(
                d_tempstorage.get(), 
                cubTempSize, 
                d_num_corrected_candidates_per_anchor.get(), 
                d_num_corrected_candidates_per_anchor_prefixsum.get(), 
                maxAnchors, 
                backgroundStream
            );

            cudaMemcpyAsync(
                currentOutput->h_num_corrected_candidates_per_anchor,
                d_num_corrected_candidates_per_anchor,
                sizeof(int) * maxAnchors,
                D2H,
                backgroundStream
            ); CUERR;

            cudaMemcpyAsync(
                currentOutput->h_num_corrected_candidates_per_anchor_prefixsum.get(),
                d_num_corrected_candidates_per_anchor_prefixsum.get(),
                sizeof(int) * maxAnchors,
                D2H,
                backgroundStream
            ); CUERR;


            //copy alignment shifts and indices of corrected candidates from device to host

            gpucorrectorkernels::copyShiftsAndCorrectedCandidateIndices<<<320, 256, 0, backgroundStream>>>(
                currentOutput->h_alignment_shifts.get(),
                currentOutput->h_indices_of_corrected_candidates.get(),
                d_numCandidates.get(),
                d_alignment_shifts.get(),
                d_indices_of_corrected_candidates.get()
            ); CUERR;

            callCorrectCandidatesKernel_async(
                currentOutput->h_corrected_candidates.get(),
                currentOutput->h_editsPerCorrectedCandidate.get(),
                currentOutput->h_numEditsPerCorrectedCandidate.get(),
                multiMSA,
                d_alignment_shifts.get(),
                d_alignment_best_alignment_flags.get(),
                d_candidate_sequences_data.get(),
                d_candidate_sequences_lengths.get(),
                d_candidateContainsN.get(),
                d_indices_of_corrected_candidates.get(),
                d_num_total_corrected_candidates.get(),
                d_anchorIndicesOfCandidates.get(),
                d_numAnchors,
                d_numCandidates,
                getDoNotUseEditsValue(),
                maxNumEditsPerSequence,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                editsPitchInBytes,
                sequenceFileProperties->maxSequenceLength,
                stream,
                kernelLaunchHandle
            );
  
        }



        

        static constexpr int getDoNotUseEditsValue() noexcept{
            return -1;
        }

    private:

        int deviceId;
        std::array<CudaEvent, 1> events;
        CudaStream backgroundStream;

        std::size_t msaColumnPitchInElements;
        std::size_t encodedSequencePitchInInts;
        std::size_t decodedSequencePitchInBytes;
        std::size_t qualityPitchInBytes;
        std::size_t editsPitchInBytes;

        int maxAnchors;
        int maxCandidates;
        int maxNumEditsPerSequence;
        int currentNumAnchors;
        int currentNumCandidates;

        const DistributedReadStorage* gpuReadStorage;
        const GpuMinhasher* gpuMinhasher;

        const CorrectionOptions* correctionOptions;
        const GoodAlignmentProperties* goodAlignmentProperties;
        const SequenceFileProperties* sequenceFileProperties;

        GpuErrorCorrectorInput* currentInput;
        GpuErrorCorrectorRawOutput* currentOutput;
        GpuErrorCorrectorRawOutput currentOutputData;

        ThreadPool* threadPool;
        ThreadPool::ParallelForHandle pforHandle;
        DistributedReadStorage::GatherHandleSequences anchorSequenceGatherHandle;
        DistributedReadStorage::GatherHandleSequences candidateSequenceGatherHandle;
        DistributedReadStorage::GatherHandleQualities anchorQualitiesGatherHandle;
        DistributedReadStorage::GatherHandleQualities candidateQualitiesGatherHandle;
        GpuMinhasher::QueryHandle minhashHandle;
        KernelLaunchHandle kernelLaunchHandle;  


        PinnedBuffer<int> h_indices_per_anchor;
        PinnedBuffer<int> h_high_quality_anchor_indices;
        PinnedBuffer<int> h_num_high_quality_anchor_indices; 
        PinnedBuffer<int> h_num_total_corrected_candidates;

        DeviceBuffer<int> d_candidates_per_anchor_tmp;
        DeviceBuffer<bool> d_anchorContainsN;
        DeviceBuffer<bool> d_candidateContainsN;
        DeviceBuffer<int> d_candidate_sequences_lengths;
        DeviceBuffer<unsigned int> d_candidate_sequences_data;
        DeviceBuffer<unsigned int> d_transposedCandidateSequencesData;
        DeviceBuffer<char> d_anchor_qualities;
        DeviceBuffer<char> d_candidate_qualities;
        DeviceBuffer<int> d_anchorIndicesOfCandidates;
        DeviceBuffer<char> d_tempstorage;
        DeviceBuffer<int> d_alignment_overlaps;
        DeviceBuffer<int> d_alignment_shifts;
        DeviceBuffer<int> d_alignment_nOps;
        DeviceBuffer<bool> d_alignment_isValid;
        DeviceBuffer<BestAlignment_t> d_alignment_best_alignment_flags; 
        DeviceBuffer<int> d_indices;
        DeviceBuffer<int> d_indices_per_anchor;
        DeviceBuffer<int> d_num_indices;
        DeviceBuffer<int> d_indices_tmp;
        DeviceBuffer<int> d_indices_per_anchor_tmp;
        DeviceBuffer<int> d_num_indices_tmp;
        DeviceBuffer<std::uint8_t> d_consensus;
        DeviceBuffer<float> d_support;
        DeviceBuffer<int> d_coverage;
        DeviceBuffer<float> d_origWeights;
        DeviceBuffer<int> d_origCoverages;
        DeviceBuffer<MSAColumnProperties> d_msa_column_properties;
        DeviceBuffer<int> d_counts;
        DeviceBuffer<float> d_weights;
        DeviceBuffer<char> d_corrected_anchors;
        DeviceBuffer<char> d_corrected_candidates;
        DeviceBuffer<int> d_num_corrected_candidates_per_anchor;
        DeviceBuffer<int> d_num_corrected_candidates_per_anchor_prefixsum;
        DeviceBuffer<int> d_num_total_corrected_candidates;
        DeviceBuffer<bool> d_anchor_is_corrected;
        DeviceBuffer<AnchorHighQualityFlag> d_is_high_quality_anchor;
        DeviceBuffer<int> d_high_quality_anchor_indices;
        DeviceBuffer<int> d_num_high_quality_anchor_indices; 
        DeviceBuffer<TempCorrectedSequence::EncodedEdit> d_editsPerCorrectedanchor;
        DeviceBuffer<int> d_numEditsPerCorrectedanchor;
        DeviceBuffer<TempCorrectedSequence::EncodedEdit> d_editsPerCorrectedCandidate;
        DeviceBuffer<int> d_numEditsPerCorrectedCandidate;
        DeviceBuffer<int> d_indices_of_corrected_anchors;
        DeviceBuffer<int> d_num_indices_of_corrected_anchors;
        DeviceBuffer<int> d_indices_of_corrected_candidates;

        DeviceBuffer<int> d_numAnchors;
        DeviceBuffer<int> d_numCandidates;
        DeviceBuffer<read_number> d_anchorReadIds;
        DeviceBuffer<unsigned int> d_anchor_sequences_data;
        DeviceBuffer<int> d_anchor_sequences_lengths;
        DeviceBuffer<read_number> d_candidate_read_ids;
        DeviceBuffer<int> d_candidates_per_anchor;
        DeviceBuffer<int> d_candidates_per_anchor_prefixsum; 
        
        std::map<GpuErrorCorrectorRawOutput*, CudaGraph> graphMap;
    };



}
}






#endif