#ifndef READ_EXTENDER_GPU_HPP
#define READ_EXTENDER_GPU_HPP

#include <readextenderbase.hpp>
#include <config.hpp>
#include <hpc_helpers.cuh>

#include <gpu/gpumsa.cuh>
#include <gpu/kernels.hpp>
#include <gpu/kernellaunch.hpp>
#include <gpu/gpuminhasher.cuh>
#include <gpu/segmented_set_operations.cuh>
#include <gpu/cachingallocator.cuh>
#include <sequencehelpers.hpp>
#include <hostdevicefunctions.cuh>

#include <algorithm>
#include <vector>
#include <numeric>

#include <cub/cub.cuh>

#include <thrust/device_new_allocator.h>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>




namespace care{




namespace readextendergpukernels{

    __global__
    void flagCandidateIdsWhichAreEqualToAnchorOrMateKernel(
        const read_number* __restrict__ candidateReadIds,
        const read_number* __restrict__ anchorReadIds,
        const read_number* __restrict__ mateReadIds,
        const int* __restrict__ numCandidatesPerAnchorPrefixSum,
        const int* __restrict__ numCandidatesPerAnchor,
        bool* __restrict__ keepflags, // size numCandidates
        bool* __restrict__ mateRemovedFlags, //size numTasks
        int* __restrict__ numCandidatesPerAnchorOut,
        int numTasks,
        bool isPairedEnd
    );

    template<int dummy=0>
    __global__
    void setAnchorIndicesOfCandidateskernel(
        int* __restrict__ d_anchorIndicesOfCandidates,
        const int* __restrict__ numAnchorsPtr,
        const int* __restrict__ d_candidates_per_anchor,
        const int* __restrict__ d_candidates_per_anchor_prefixsum
    ){
        for(int anchorIndex = blockIdx.x; anchorIndex < *numAnchorsPtr; anchorIndex += gridDim.x){
            const int offset = d_candidates_per_anchor_prefixsum[anchorIndex];
            const int numCandidatesOfAnchor = d_candidates_per_anchor[anchorIndex];
            int* const beginptr = &d_anchorIndicesOfCandidates[offset];

            for(int localindex = threadIdx.x; localindex < numCandidatesOfAnchor; localindex += blockDim.x){
                beginptr[localindex] = anchorIndex;
            }
        }
    }
    

    template<int blocksize>
    __global__
    void reverseComplement2bitKernel(
        const int* __restrict__ lengths,
        const unsigned int* __restrict__ forward,
        unsigned int* __restrict__ reverse,
        int num,
        int encodedSequencePitchInInts
    ){

        for(int s = threadIdx.x + blockIdx.x * blockDim.x; s < num; s += blockDim.x * gridDim.x){
            const unsigned int* input = forward + encodedSequencePitchInInts * s;
            unsigned int* output = reverse + encodedSequencePitchInInts * s;
            const int length = lengths[s];

            SequenceHelpers::reverseComplementSequence2Bit(
                output,
                input,
                length,
                [](auto i){return i;},
                [](auto i){return i;}
            );
        }

        // constexpr int smemsizeints = blocksize * 16;
        // __shared__ unsigned int sharedsequences[smemsizeints]; //sequences will be stored transposed

        // const int sequencesPerSmem = std::min(blocksize, smemsizeints / encodedSequencePitchInInts);
        // assert(sequencesPerSmem > 0);

        // const int smemiterations = SDIV(num, sequencesPerSmem);

        // for(int smemiteration = blockIdx.x; smemiteration < smemiterations; smemiteration += gridDim.x){

        //     const int idBegin = smemiteration * sequencesPerSmem;
        //     const int idEnd = std::min((smemiteration+1) * sequencesPerSmem, num);

        //     __syncthreads();

        //     for(int s = idBegin + threadIdx.x; s < idEnd; s += blockDim.x){
        //         for(int intindex = 0; intindex < encodedSequencePitchInInts; intindex++){ //load intindex-th element of sequence s
        //             sharedsequences[intindex * sequencesPerSmem + s] = forward[encodedSequencePitchInInts * s + intindex];
        //         }
        //     }

        //     __syncthreads();

        //     for(int s = idBegin + threadIdx.x; s < idEnd; s += blockDim.x){
        //         SequenceHelpers::reverseComplementSequenceInplace2Bit(&sharedsequences[s], lengths[s], [&](auto i){return i * sequencesPerSmem;});
        //     }

        //     __syncthreads();

        //     for(int s = idBegin + threadIdx.x; s < idEnd; s += blockDim.x){
        //         for(int intindex = 0; intindex < encodedSequencePitchInInts; intindex++){ //load intindex-th element of sequence s
        //             reverse[encodedSequencePitchInInts * s + intindex] = sharedsequences[intindex * sequencesPerSmem + s];
        //         }
        //     }
        // }
    }


    template<int blocksize, int groupsize>
    __global__
    void filtermatekernel(
        const unsigned int* __restrict__ anchormatedata,
        const unsigned int* __restrict__ candidatefwddata,
        //const unsigned int* __restrict__ candidatefwddata2,
        int encodedSequencePitchInInts,
        const int* __restrict__ numCandidatesPerAnchor,
        const int* __restrict__ numCandidatesPerAnchorPrefixSum,
        const int* __restrict__ activeTaskIndices,
        int numTasksWithRemovedMate,
        bool* __restrict__ output_keepflags
    ){

        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int groupindex = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int numgroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupindexinblock = threadIdx.x / groupsize;

        static_assert(blocksize % groupsize == 0);
        //constexpr int groupsperblock = blocksize / groupsize;

        extern __shared__ unsigned int smem[]; //sizeof(unsigned int) * groupsperblock * encodedSequencePitchInInts

        unsigned int* sharedMate = smem + groupindexinblock * encodedSequencePitchInInts;

        for(int task = groupindex; task < numTasksWithRemovedMate; task += numgroups){

            const int globalTaskIndex = activeTaskIndices[task];
            const int numCandidates = numCandidatesPerAnchor[globalTaskIndex];
            const int candidatesOffset = numCandidatesPerAnchorPrefixSum[globalTaskIndex];

            for(int p = group.thread_rank(); p < encodedSequencePitchInInts; p++){
                sharedMate[p] = anchormatedata[encodedSequencePitchInInts * task + p];
            }
            group.sync();

            //compare mate to candidates. 1 thread per candidate
            for(int c = group.thread_rank(); c < numCandidates; c += group.size()){
                bool doKeep = false;
                const unsigned int* const candidateptr = candidatefwddata + encodedSequencePitchInInts * (candidatesOffset + c);

                for(int p = 0; p < encodedSequencePitchInInts; p++){
                    const unsigned int aaa = sharedMate[p];
                    const unsigned int bbb = candidateptr[p];

                    if(aaa != bbb){
                        doKeep = true;
                        break;
                    }
                }

                output_keepflags[(candidatesOffset + c)] = doKeep;
            }
        }
    }
}


struct SequenceFlagMultiplier{
    int pitch{};
    const bool* flags{};

    __host__ __device__
    SequenceFlagMultiplier(const bool* flags_, int pitch_)
        :pitch(pitch_), flags(flags_){

    }

    __host__ __device__
    bool operator()(int i) const{
        return flags[i / pitch];
    }
};

struct ReadExtenderGpu final : public ReadExtenderBase{
public:

    static constexpr int primary_stream_index = 0;

    template<class T>
    using DeviceBuffer = helpers::SimpleAllocationDevice<T>;
    //using DeviceBuffer = helpers::SimpleAllocationPinnedHost<T>;

    template<class T>
    using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T>;


    ReadExtenderGpu(
        int insertSize,
        int insertSizeStddev,
        int maxextensionPerStep,
        int maximumSequenceLength,
        int kmerLength_,
        const gpu::GpuReadStorage& rs, 
        const gpu::GpuMinhasher& gmh,
        const CorrectionOptions& coropts,
        const GoodAlignmentProperties& gap,
        cub::CachingDeviceAllocator& cubAllocator_
    ) 
    : ReadExtenderBase(insertSize, insertSizeStddev, maxextensionPerStep, maximumSequenceLength, coropts, gap),
        kmerLength(kmerLength_),
        gpuReadStorage(&rs),
        gpuMinhasher(&gmh),
        readStorageHandle(gpuReadStorage->makeHandle()),
        minhashHandle(gpuMinhasher->makeQueryHandle()),
        cubAllocator(&cubAllocator_){

        cudaGetDevice(&deviceId); CUERR;

        kernelLaunchHandle = gpu::make_kernel_launch_handle(deviceId);

        const std::size_t min_overlap = std::max(
            1, 
            std::max(
                goodAlignmentProperties.min_overlap, 
                int(gpuReadStorage->getSequenceLengthUpperBound() * goodAlignmentProperties.min_overlap_ratio)
            )
        );
        const std::size_t msa_max_column_count = (3*gpuReadStorage->getSequenceLengthUpperBound() - 2*min_overlap);
        //round up to 32 elements
        msaColumnPitchInElements = SDIV(msa_max_column_count, 32) * 32;
    }

    ~ReadExtenderGpu(){
        gpuReadStorage->destroyHandle(readStorageHandle);
        gpuMinhasher->destroyHandle(minhashHandle);
    }

    struct BatchData{
        bool pairedEnd = false;
        int numTasks = 0;
        int numTasksWithMateRemoved = 0;

        int totalNumCandidates = 0;
        int totalNumberOfUsedIds = 0;

        PinnedBuffer<read_number> h_anchorReadIds{};
        DeviceBuffer<read_number> d_anchorReadIds{};
        PinnedBuffer<read_number> h_mateReadIds{};
        DeviceBuffer<read_number> d_mateReadIds{};
        PinnedBuffer<read_number> h_candidateReadIds{};
        DeviceBuffer<read_number> d_candidateReadIds{};

        PinnedBuffer<int> h_anchorIndicesOfCandidates{};
        PinnedBuffer<int> h_anchorIndicesOfCandidates2{};

        PinnedBuffer<int> h_segmentIds2{};
        PinnedBuffer<int> h_segmentIds4{};

        DeviceBuffer<int> d_anchorIndicesOfCandidates{};
        DeviceBuffer<int> d_anchorIndicesOfCandidates2{};
        DeviceBuffer<int> d_segmentIdsOfUsedReadIds{};
        DeviceBuffer<int> d_segmentIds4{};

        PinnedBuffer<int> h_segmentIdsOfUsedReadIds{};
        PinnedBuffer<int> h_segmentIdsOfReadIds{};

        PinnedBuffer<unsigned int> h_anchormatedata{};
        DeviceBuffer<unsigned int> d_anchormatedata{};


        PinnedBuffer<unsigned int> h_inputanchormatedata{};
        DeviceBuffer<unsigned int> d_inputanchormatedata{};

        DeviceBuffer<int> d_anchorIndicesWithRemovedMates{};

        DeviceBuffer<int> d_indexlist1{};

        PinnedBuffer<int> h_numCandidatesPerAnchor{};
        PinnedBuffer<int> h_numCandidatesPerAnchor2{};
        DeviceBuffer<int> d_numCandidatesPerAnchor{};
        DeviceBuffer<int> d_numCandidatesPerAnchor2{};
        PinnedBuffer<int> h_numCandidatesPerAnchorPrefixSum{};
        DeviceBuffer<int> d_numCandidatesPerAnchorPrefixSum{};
        DeviceBuffer<int> d_numCandidatesPerAnchorPrefixSum2{};
        PinnedBuffer<int> h_alignment_overlaps{};
        PinnedBuffer<int> h_alignment_shifts{};
        PinnedBuffer<int> h_alignment_nOps{};

        PinnedBuffer<BestAlignment_t> h_alignment_best_alignment_flags{};

        DeviceBuffer<int> d_alignment_overlaps{};
        DeviceBuffer<int> d_alignment_shifts{};
        DeviceBuffer<int> d_alignment_nOps{};
        DeviceBuffer<bool> d_alignment_isValid{};
        DeviceBuffer<BestAlignment_t> d_alignment_best_alignment_flags{};

        DeviceBuffer<int> d_alignment_overlaps2{};
        DeviceBuffer<int> d_alignment_shifts2{};
        DeviceBuffer<int> d_alignment_nOps2{};
        DeviceBuffer<BestAlignment_t> d_alignment_best_alignment_flags2{};

        PinnedBuffer<int> h_numAnchors{};
        PinnedBuffer<int> h_numCandidates{};
        DeviceBuffer<int> d_numAnchors{};
        DeviceBuffer<int> d_numCandidates{};
        DeviceBuffer<int> d_numCandidates2{};
        PinnedBuffer<int> h_numAnchorsWithRemovedMates{};

        PinnedBuffer<int> h_anchorSequencesLength{};
        DeviceBuffer<int> d_anchorSequencesLength{};
        PinnedBuffer<int> h_candidateSequencesLength{};
        DeviceBuffer<int> d_candidateSequencesLength{};
        PinnedBuffer<unsigned int> h_subjectSequencesData{};
        PinnedBuffer<unsigned int> h_candidateSequencesData{};
        PinnedBuffer<unsigned int> h_candidateSequencesRevcData{};

        DeviceBuffer<int> d_candidateSequencesLength2{};
        DeviceBuffer<unsigned int> d_candidateSequencesData2{};
        DeviceBuffer<unsigned int> d_candidateSequencesRevcData2{};
        DeviceBuffer<read_number> d_candidateReadIds2{};
        

        DeviceBuffer<unsigned int> d_subjectSequencesData{};
        DeviceBuffer<unsigned int> d_candidateSequencesData{};
        DeviceBuffer<unsigned int> d_candidateSequencesRevcData{};

        DeviceBuffer<int> d_activeTaskIndices{};
        PinnedBuffer<int> h_activeTaskIndices{};

        DeviceBuffer<int> d_intbuffercandidates{};

        DeviceBuffer<char> d_tempstorage{};

        DeviceBuffer<bool> d_flagsanchors{};

        DeviceBuffer<bool> d_flagscandidates{};

        PinnedBuffer<read_number> h_usedReadIds{};
        PinnedBuffer<int> h_numUsedReadIdsPerAnchor{};
        PinnedBuffer<int> h_numUsedReadIdsPerAnchorPrefixSum{};

        DeviceBuffer<read_number> d_usedReadIds{};
        DeviceBuffer<int> d_numUsedReadIdsPerAnchor{};
        DeviceBuffer<int> d_numUsedReadIdsPerAnchorPrefixSum{};

        PinnedBuffer<char> h_consensus;
        PinnedBuffer<gpu::MSAColumnProperties> h_msa_column_properties;

        DeviceBuffer<std::uint8_t> d_consensus; //encoded , 0-4
        DeviceBuffer<float> d_support;
        DeviceBuffer<int> d_coverage;
        DeviceBuffer<float> d_origWeights;
        DeviceBuffer<int> d_origCoverages;
        DeviceBuffer<gpu::MSAColumnProperties> d_msa_column_properties;
        DeviceBuffer<int> d_counts;
        DeviceBuffer<float> d_weights;

        PinnedBuffer<MultipleSequenceAlignment::PossibleSplitColumn> h_possibleSplitColumns;
        PinnedBuffer<int> h_numPossibleSplitColumnsPerAnchor;
        DeviceBuffer<MultipleSequenceAlignment::PossibleSplitColumn> d_possibleSplitColumns;
        DeviceBuffer<int> d_numPossibleSplitColumnsPerAnchor;

        
    };

    static constexpr int getNumRefinementIterations() noexcept{
        return 5;
    }
     
public: //private:

    std::vector<ExtendResult> processPairedEndTasks(std::vector<Task>& tasks) override;

    std::vector<ExtendResult> processSingleEndTasks(std::vector<Task>& tasks) override;


public:
    void getCandidateReadIds(BatchData& batchData, cudaStream_t stream) const{

        int totalNumValues = 0;

        gpuMinhasher->determineNumValues(
            minhashHandle,
            batchData.d_subjectSequencesData.get(),
            encodedSequencePitchInInts,
            batchData.d_anchorSequencesLength.get(),
            batchData.numTasks,
            batchData.d_numCandidatesPerAnchor.get(),
            totalNumValues,
            stream
        );

        cudaStreamSynchronize(stream); CUERR;

        batchData.d_candidateReadIds.resize(totalNumValues);        

        if(totalNumValues == 0){
            cudaMemsetAsync(batchData.d_numCandidatesPerAnchor.get(), 0, sizeof(int) * batchData.numTasks , stream); CUERR;
            cudaMemsetAsync(batchData.d_numCandidatesPerAnchorPrefixSum.get(), 0, sizeof(int) * (1 + batchData.numTasks), stream); CUERR;
            batchData.totalNumCandidates = 0;
            return;
        }

        gpuMinhasher->retrieveValues(
            minhashHandle,
            nullptr,
            batchData.numTasks,              
            totalNumValues,
            batchData.d_candidateReadIds.get(),
            batchData.d_numCandidatesPerAnchor.get(),
            batchData.d_numCandidatesPerAnchorPrefixSum.get(),
            stream
        );

        cudaMemcpyAsync(
            &batchData.totalNumCandidates,
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + batchData.numTasks,
            sizeof(int),
            D2H,
            stream
        ); CUERR;

        cudaStreamSynchronize(stream); CUERR;
    }

    void removeUsedIdsAndMateIds(BatchData& batchData, cudaStream_t firstStream, cudaStream_t secondStream) const{
        batchData.d_anchorIndicesOfCandidates.resize(batchData.totalNumCandidates);
        batchData.d_anchorIndicesOfCandidates2.resize(batchData.totalNumCandidates);
        batchData.d_flagscandidates.resize(batchData.totalNumCandidates);
        batchData.d_flagsanchors.resize(batchData.numTasks);
        batchData.d_candidateReadIds2.resize(batchData.totalNumCandidates);

        batchData.h_segmentIdsOfReadIds.resize(batchData.totalNumCandidates);

        batchData.h_numCandidatesPerAnchor.resize(batchData.numTasks);

        //determine required temp bytes for following cub calls, and allocate temp storage

        cudaError_t cubstatus = cudaSuccess;
        std::size_t cubBytes = 0;
        std::size_t cubBytes2 = 0;

        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            cubBytes2,
            batchData.d_candidateReadIds.data(),
            batchData.d_flagscandidates.data(),
            batchData.d_candidateReadIds2.data(),
            batchData.h_numCandidates.data(),
            batchData.totalNumCandidates,
            firstStream
        );
        assert(cubstatus == cudaSuccess);
        
        cubBytes = std::max(cubBytes, cubBytes2);
        
        cubstatus = cub::DeviceScan::InclusiveSum(
            nullptr, 
            cubBytes2, 
            (int*)nullptr,
            (int*)nullptr,
            batchData.numTasks,
            firstStream
        );
        assert(cubstatus == cudaSuccess);
        
        cubBytes = std::max(cubBytes, cubBytes2);
        
        cubstatus = cub::DeviceScan::InclusiveScan(
            nullptr, 
            cubBytes2, 
            batchData.d_anchorIndicesOfCandidates.data(), 
            batchData.d_anchorIndicesOfCandidates.data(), 
            cub::Max{},
            batchData.totalNumCandidates,
            firstStream
        );
        assert(cubstatus == cudaSuccess);
        
        cubBytes = std::max(cubBytes, cubBytes2);
        
        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            cubBytes2,
            thrust::make_counting_iterator(0),
            batchData.d_flagsanchors.data(),
            batchData.d_anchorIndicesWithRemovedMates.data(),
            batchData.h_numAnchorsWithRemovedMates.data(),
            batchData.numTasks,
            firstStream
        );
        assert(cubstatus == cudaSuccess);
        
        cubBytes = std::max(cubBytes, cubBytes2);
        
        void* cubtempstorage; cubAllocator->DeviceAllocate((void**)&cubtempstorage, cubBytes, firstStream);

        //cub storage for second stream
        //std::size_t cubtempstream2bytes = 0;

        // cubstatus = cub::DeviceScan::InclusiveScan(
        //     nullptr, 
        //     cubBytes2, 
        //     batchData.d_segmentIdsOfUsedReadIds.data(), 
        //     batchData.d_segmentIdsOfUsedReadIds.data(), 
        //     cub::Max{},
        //     batchData.totalNumberOfUsedIds,
        //     secondStream
        // );
        // assert(cubstatus == cudaSuccess);
        // cubtempstream2bytes = std::max(cubtempstream2bytes, cubBytes2);

        // cubstatus = cub::DeviceSelect::Flagged(
        //     nullptr,
        //     cubBytes2,
        //     batchData.d_inputanchormatedata.data(),
        //     thrust::make_transform_iterator(
        //         thrust::make_counting_iterator(0),
        //         SequenceFlagMultiplier{batchData.d_flagsanchors.data(), int(encodedSequencePitchInInts)}
        //     ),
        //     batchData.d_anchormatedata.data(),
        //     thrust::make_discard_iterator(),
        //     batchData.numTasks * encodedSequencePitchInInts,
        //     secondStream
        // );
        // assert(cubstatus == cudaSuccess);
        // cubtempstream2bytes = std::max(cubtempstream2bytes, cubBytes2);

        // void* cubtempstream2 = nullptr; cubAllocator->DeviceAllocate((void**)&cubtempstream2, cubtempstream2bytes, secondStream);

        // helpers::call_fill_kernel_async(batchData.d_segmentIdsOfUsedReadIds.data(), batchData.totalNumberOfUsedIds, 0, secondStream);

        // setFirstSegmentIdsKernel<<<SDIV(batchData.numTasks, 256), 256, 0, secondStream>>>(
        //     batchData.d_numUsedReadIdsPerAnchor.data(),
        //     batchData.d_segmentIdsOfUsedReadIds.data(),
        //     batchData.d_numUsedReadIdsPerAnchorPrefixSum.data(),
        //     batchData.numTasks
        // );          

        // cubstatus = cub::DeviceScan::InclusiveScan(
        //     cubtempstream2, 
        //     cubBytes, 
        //     batchData.d_segmentIdsOfUsedReadIds.data(), 
        //     batchData.d_segmentIdsOfUsedReadIds.data(), 
        //     cub::Max{},
        //     batchData.totalNumberOfUsedIds,
        //     secondStream
        // );
        // assert(cubstatus == cudaSuccess);

        //cudaEventRecord(events[0], secondStream); CUERR;
        
    
        
        helpers::call_fill_kernel_async(batchData.d_flagscandidates.data(), batchData.d_flagscandidates.size(), false, firstStream);
        
        //flag candidates ids to remove because they are equal to anchor id or equal to mate id
        readextendergpukernels::flagCandidateIdsWhichAreEqualToAnchorOrMateKernel<<<4096, 128, 0, firstStream>>>(
            batchData.d_candidateReadIds.data(),
            batchData.d_anchorReadIds.data(),
            batchData.d_mateReadIds.data(),
            batchData.d_numCandidatesPerAnchorPrefixSum.data(),
            batchData.d_numCandidatesPerAnchor.data(),
            batchData.d_flagscandidates.data(),
            batchData.d_flagsanchors.data(),
            batchData.d_numCandidatesPerAnchor2.data(),
            batchData.numTasks,
            batchData.pairedEnd
        );
        CUERR;

        cudaEventRecord(events[0], firstStream);
        cudaStreamWaitEvent(secondStream, events[0], 0); CUERR;

        cudaMemcpyAsync(
            batchData.h_numCandidatesPerAnchor.data(),
            batchData.d_numCandidatesPerAnchor2.data(),
            sizeof(int) * batchData.numTasks,
            D2H,
            secondStream
        ); CUERR;

        cudaEventRecord(events[0], secondStream);

        //determine task ids with removed mates

        cubstatus = cub::DeviceSelect::Flagged(
            cubtempstorage,
            cubBytes,
            thrust::make_counting_iterator(0),
            batchData.d_flagsanchors.data(),
            batchData.d_anchorIndicesWithRemovedMates.data(),
            batchData.h_numAnchorsWithRemovedMates.data(),
            batchData.numTasks,
            firstStream
        );
        assert(cubstatus == cudaSuccess);

        //copy selected candidate ids

        cubstatus = cub::DeviceSelect::Flagged(
            cubtempstorage,
            cubBytes,
            batchData.d_candidateReadIds.data(),
            batchData.d_flagscandidates.data(),
            batchData.d_candidateReadIds2.data(),
            batchData.h_numCandidates.data(),
            batchData.totalNumCandidates,
            firstStream
        );
        assert(cubstatus == cudaSuccess);

        cudaStreamSynchronize(firstStream); CUERR; //wait for h_numCandidates   and h_numAnchorsWithRemovedMates
        batchData.numTasksWithMateRemoved = *batchData.h_numAnchorsWithRemovedMates;
        batchData.totalNumCandidates = *batchData.h_numCandidates;

        if(batchData.numTasksWithMateRemoved > 0){

            //copy mate sequence data of removed mates

            std::size_t cubtempstream2bytes = 0;
            cubstatus = cub::DeviceSelect::Flagged(
                nullptr,
                cubtempstream2bytes,
                batchData.d_inputanchormatedata.data(),
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator(0),
                    SequenceFlagMultiplier{batchData.d_flagsanchors.data(), int(encodedSequencePitchInInts)}
                ),
                batchData.d_anchormatedata.data(),
                thrust::make_discard_iterator(),
                batchData.numTasks * encodedSequencePitchInInts,
                secondStream
            );
            assert(cubstatus == cudaSuccess);
    
            void* cubtempstream2 = nullptr; cubAllocator->DeviceAllocate((void**)&cubtempstream2, cubtempstream2bytes, secondStream);
                
            cubstatus = cub::DeviceSelect::Flagged(
                cubtempstream2,
                cubtempstream2bytes,
                batchData.d_inputanchormatedata.data(),
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator(0),
                    SequenceFlagMultiplier{batchData.d_flagsanchors.data(), int(encodedSequencePitchInInts)}
                ),
                batchData.d_anchormatedata.data(),
                thrust::make_discard_iterator(),
                batchData.numTasks * encodedSequencePitchInInts,
                secondStream
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceFree(cubtempstream2);
        }

        cudaEventSynchronize(events[0]); CUERR;

        for(int i = 0, sum = 0; i < batchData.numTasks; i++){
            std::fill(
                batchData.h_segmentIdsOfReadIds.data() + sum,
                batchData.h_segmentIdsOfReadIds.data() + sum + batchData.h_numCandidatesPerAnchor[i],
                i
            );
            sum += batchData.h_numCandidatesPerAnchor[i];
        }

        cudaMemcpyAsync(
            batchData.d_anchorIndicesOfCandidates.data(),
            batchData.h_segmentIdsOfReadIds.data(),
            sizeof(int) * batchData.totalNumCandidates,
            H2D,
            firstStream
        ); CUERR;

        

        // //compute prefix sum of number of candidates per anchor
        // helpers::call_set_kernel_async(batchData.d_numCandidatesPerAnchorPrefixSum2.data(), 0, 0, firstStream);

        // cubstatus = cub::DeviceScan::InclusiveSum(
        //     cubtempstorage, 
        //     cubBytes, 
        //     batchData.d_numCandidatesPerAnchor2.data(), 
        //     batchData.d_numCandidatesPerAnchorPrefixSum2.data() + 1, 
        //     batchData.numTasks,
        //     firstStream
        // );
        // assert(cubstatus == cudaSuccess);

        // //compute segment ids for candidate read ids

        // helpers::call_fill_kernel_async(batchData.d_anchorIndicesOfCandidates.data(), batchData.totalNumCandidates, 0, firstStream);

        // setFirstSegmentIdsKernel<<<SDIV(batchData.numTasks, 256), 256, 0, firstStream>>>(
        //     batchData.d_numCandidatesPerAnchor2.data(),
        //     batchData.d_anchorIndicesOfCandidates.data(),
        //     batchData.d_numCandidatesPerAnchorPrefixSum2.data(),
        //     batchData.numTasks
        // );

        // cubstatus = cub::DeviceScan::InclusiveScan(
        //     cubtempstorage, 
        //     cubBytes, 
        //     batchData.d_anchorIndicesOfCandidates.data(), 
        //     batchData.d_anchorIndicesOfCandidates.data(), 
        //     cub::Max{},
        //     batchData.totalNumCandidates,
        //     firstStream
        // );
        // assert(cubstatus == cudaSuccess);
   

        ThrustCachingAllocator<char> thrustCachingAllocator1(deviceId, cubAllocator, firstStream);

        //cudaStreamWaitEvent(firstStream, events[0], 0); CUERR;

        
        //compute segmented set difference between candidate read ids and used candidate read ids
        auto d_candidateReadIds_end = GpuSegmentedSetOperation{}.difference(
            thrustCachingAllocator1,
            batchData.d_candidateReadIds2.data(),
            batchData.d_numCandidatesPerAnchor2.data(),
            batchData.d_numCandidatesPerAnchorPrefixSum2.data(),
            batchData.d_anchorIndicesOfCandidates.data(),
            batchData.totalNumCandidates,
            batchData.d_usedReadIds.data(),
            batchData.d_numUsedReadIdsPerAnchor.data(),
            batchData.d_numUsedReadIdsPerAnchorPrefixSum.data(),
            batchData.d_segmentIdsOfUsedReadIds.data(),
            batchData.totalNumberOfUsedIds,
            batchData.numTasks,        
            batchData.d_candidateReadIds.data(),
            batchData.d_numCandidatesPerAnchor.data(),
            batchData.d_anchorIndicesOfCandidates2.data(),
            firstStream
        );

        std::swap(batchData.d_anchorIndicesOfCandidates, batchData.d_anchorIndicesOfCandidates2);

        batchData.totalNumCandidates = std::distance(batchData.d_candidateReadIds.data(), d_candidateReadIds_end);

        //compute prefix sum of new segment sizes
    
        cubstatus = cub::DeviceScan::InclusiveSum(
            cubtempstorage, 
            cubBytes, 
            batchData.d_numCandidatesPerAnchor.data(), 
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            batchData.numTasks,
            firstStream
        );
        assert(cubstatus == cudaSuccess);

        if(batchData.numTasksWithMateRemoved > 0){
            cudaEventRecord(events[0], secondStream);
            cudaStreamWaitEvent(firstStream, events[0], 0); CUERR;
        }

        cubAllocator->DeviceFree(cubtempstorage);
    }


    void loadCandidateSequenceData(BatchData& batchData, cudaStream_t stream) const{

        const int totalNumCandidates = batchData.totalNumCandidates;

        batchData.d_candidateSequencesLength.resize(batchData.totalNumCandidates);
        batchData.d_candidateSequencesData.resize(encodedSequencePitchInInts * batchData.totalNumCandidates);

        gpuReadStorage->gatherSequences(
            readStorageHandle,
            batchData.d_candidateSequencesData.get(),
            encodedSequencePitchInInts,
            batchData.h_candidateReadIds.get(),
            batchData.d_candidateReadIds.get(), //device accessible
            totalNumCandidates,
            stream
        );

        gpuReadStorage->gatherSequenceLengths(
            readStorageHandle,
            batchData.d_candidateSequencesLength.get(),
            batchData.d_candidateReadIds.get(),
            totalNumCandidates,
            stream
        );
    }


    void eraseDataOfRemovedMates(BatchData& batchData, cudaStream_t stream) const{


        auto vecAccess = [](auto& vec, auto index) -> decltype(vec.at(index)){
            return vec.at(index);
        };

        const int totalNumCandidates = batchData.totalNumCandidates;

        batchData.d_intbuffercandidates.resize(batchData.totalNumCandidates);
        batchData.d_flagscandidates.resize(batchData.totalNumCandidates);

        batchData.d_candidateSequencesLength2.resize(batchData.totalNumCandidates);
        batchData.d_candidateSequencesData2.resize(encodedSequencePitchInInts * batchData.totalNumCandidates);
        batchData.d_candidateReadIds2.resize(batchData.totalNumCandidates);

        constexpr int groupsize = 32;
        constexpr int blocksize = 128;
        constexpr int groupsperblock = blocksize / groupsize;
        dim3 block(blocksize,1,1);
        dim3 grid(SDIV(batchData.numTasksWithMateRemoved * groupsize, blocksize), 1, 1);
        const std::size_t smembytes = sizeof(unsigned int) * groupsperblock * encodedSequencePitchInInts;

        bool* const d_keepflags = batchData.d_flagscandidates.data();

        helpers::call_fill_kernel_async(d_keepflags, totalNumCandidates, true, stream);

        readextendergpukernels::filtermatekernel<blocksize,groupsize><<<grid, block, smembytes, stream>>>(
            batchData.d_anchormatedata.data(),
            batchData.d_candidateSequencesData.data(),
            encodedSequencePitchInInts,
            batchData.d_numCandidatesPerAnchor.data(),
            batchData.d_numCandidatesPerAnchorPrefixSum.data(),
            batchData.d_anchorIndicesWithRemovedMates.data(),
            batchData.numTasksWithMateRemoved,
            d_keepflags
        ); CUERR;

        // auto negate = [] __device__ (bool b){
        //     return !b;
        // };

        // cub::TransformInputIterator<bool, decltype(negate), bool*> d_keepflags(batchData.d_flagscandidates.data(), negate);

        std::size_t requiredCubSize = 0;
        std::size_t requiredCubSize1 = 0;
        std::size_t requiredCubSize2 = 0;
        cudaError_t cubstatus = cub::DeviceScan::ExclusiveSum(
            nullptr,
            requiredCubSize1,
            d_keepflags, 
            batchData.d_intbuffercandidates.data(), 
            totalNumCandidates, 
            stream
        );
        assert(cudaSuccess == cubstatus);

        cubstatus = cub::DeviceScan::InclusiveSum(
            nullptr,
            requiredCubSize2,
            batchData.d_numCandidatesPerAnchor.data(), 
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            batchData.numTasks, 
            stream
        );
        assert(cudaSuccess == cubstatus);

        requiredCubSize = std::max(requiredCubSize1, requiredCubSize2);

        void* cubtemp; cubAllocator->DeviceAllocate((void**)&cubtemp, requiredCubSize, stream);

        cubstatus = cub::DeviceScan::ExclusiveSum(
            cubtemp,
            requiredCubSize,
            d_keepflags, 
            batchData.d_intbuffercandidates.data(), 
            totalNumCandidates, 
            stream
        );
        assert(cudaSuccess == cubstatus);

        helpers::lambda_kernel<<<4096, 128, 0, stream>>>(
            [
                numTasks = batchData.numTasks,
                encodedSequencePitchInInts = encodedSequencePitchInInts,
                d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchorPrefixSum = batchData.d_numCandidatesPerAnchorPrefixSum.data(),
                d_keepflags,
                d_outputpositions = batchData.d_intbuffercandidates.data(),
                d_candidateReadIds = batchData.d_candidateReadIds.data(),
                d_candidateSequencesLength = batchData.d_candidateSequencesLength.data(),
                d_candidateSequencesData = batchData.d_candidateSequencesData.data(),
                d_anchorIndicesOfCandidates = batchData.d_anchorIndicesOfCandidates.data(),
                d_candidateReadIdsOut = batchData.d_candidateReadIds2.data(),
                d_candidateSequencesLengthOut = batchData.d_candidateSequencesLength2.data(),
                d_candidateSequencesDataOut = batchData.d_candidateSequencesData2.data(),
                d_anchorIndicesOfCandidatesOut = batchData.d_anchorIndicesOfCandidates2.data()
            ] __device__ (){

                constexpr int elementsPerIteration = 128;

                using BlockReduce = cub::BlockReduce<int, elementsPerIteration>;
                __shared__ typename BlockReduce::TempStorage temp_storage;

                for(int t = blockIdx.x; t < numTasks; t += gridDim.x){
                    const int numCandidates = d_numCandidatesPerAnchor[t];
                    const int inputOffset = d_numCandidatesPerAnchorPrefixSum[t];

                    int numSelected = 0;

                    for(int i = threadIdx.x; i < numCandidates; i += blockDim.x){
                        if(d_keepflags[inputOffset + i]){
                            const int outputLocation = d_outputpositions[inputOffset + i];

                            d_candidateReadIdsOut[outputLocation] = d_candidateReadIds[inputOffset + i];
                            d_candidateSequencesLengthOut[outputLocation] = d_candidateSequencesLength[inputOffset + i];
                            d_anchorIndicesOfCandidatesOut[outputLocation] = d_anchorIndicesOfCandidates[inputOffset + i];

                            numSelected++;
                        }
                    }

                    for(int i = threadIdx.x; i < numCandidates * encodedSequencePitchInInts; i += blockDim.x){
                        const int which = i / encodedSequencePitchInInts;
                        const int what = i % encodedSequencePitchInInts;

                        if(d_keepflags[inputOffset + which]){
                            d_candidateSequencesDataOut[d_outputpositions[inputOffset + which] * encodedSequencePitchInInts + what] = d_candidateSequencesData[(inputOffset + which) * encodedSequencePitchInInts + what];
                        }
                    }

                    numSelected = BlockReduce(temp_storage).Sum(numSelected);
                    __syncthreads();
                    
                    if(threadIdx.x == 0){
                        if(numSelected != numCandidates){
                            assert(numSelected < numCandidates);
                            d_numCandidatesPerAnchor[t] = numSelected;
                            //printf("task %d, removed %d\n", t, numCandidates - numSelected);
                        }
                    }

                }
            }
        ); CUERR;

        //update prefix sum
        
        cubstatus = cub::DeviceScan::InclusiveSum(
            cubtemp,
            requiredCubSize,
            batchData.d_numCandidatesPerAnchor.data(), 
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            batchData.numTasks, 
            stream
        );
        assert(cudaSuccess == cubstatus);       

        cubAllocator->DeviceFree(cubtemp);

        std::swap(batchData.d_candidateReadIds2, batchData.d_candidateReadIds);
        std::swap(batchData.d_candidateSequencesLength2, batchData.d_candidateSequencesLength);
        std::swap(batchData.d_candidateSequencesData2, batchData.d_candidateSequencesData);
        std::swap(batchData.d_anchorIndicesOfCandidates2, batchData.d_anchorIndicesOfCandidates);

        cudaMemcpyAsync(
            &batchData.totalNumCandidates,
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + batchData.numTasks,
            sizeof(int),
            D2H,
            stream
        ); CUERR;

        cudaStreamSynchronize(stream); CUERR;
       
    }


    void calculateAlignments(BatchData& batchData, cudaStream_t stream) const{

        batchData.d_alignment_overlaps.resize(batchData.totalNumCandidates);
        batchData.d_alignment_shifts.resize(batchData.totalNumCandidates);
        batchData.d_alignment_nOps.resize(batchData.totalNumCandidates);
        batchData.d_alignment_isValid.resize(batchData.totalNumCandidates);
        batchData.d_alignment_best_alignment_flags.resize(batchData.totalNumCandidates);
        
        batchData.h_numAnchors[0] = batchData.numTasks;

        const bool* const d_anchorContainsN = nullptr;
        const bool* const d_candidateContainsN = nullptr;
        const bool removeAmbiguousAnchors = false;
        const bool removeAmbiguousCandidates = false;
        const int maxNumAnchors = batchData.numTasks;
        const int maxNumCandidates = batchData.totalNumCandidates; //this does not need to be exact, but it must be >= batchData.d_numCandidatesPerAnchorPrefixSum[batchData.numTasks]
        const int maximumSequenceLength = 100; //encodedSequencePitchInInts * 16;
        const int encodedSequencePitchInInts2Bit = encodedSequencePitchInInts;
        const int min_overlap = goodAlignmentProperties.min_overlap;
        const float maxErrorRate = goodAlignmentProperties.maxErrorRate;
        const float min_overlap_ratio = goodAlignmentProperties.min_overlap_ratio;
        const float estimatedNucleotideErrorRate = correctionOptions.estimatedErrorrate;

        auto callAlignmentKernel = [&](void* d_tempstorage, size_t& tempstoragebytes){

            call_popcount_rightshifted_hamming_distance_kernel_async(
                d_tempstorage,
                tempstoragebytes,
                batchData.d_alignment_overlaps.get(),
                batchData.d_alignment_shifts.get(),
                batchData.d_alignment_nOps.get(),
                batchData.d_alignment_isValid.get(),
                batchData.d_alignment_best_alignment_flags.get(),
                batchData.d_subjectSequencesData.get(),
                batchData.d_candidateSequencesData.get(),
                batchData.d_anchorSequencesLength.get(),
                batchData.d_candidateSequencesLength.get(),
                batchData.d_numCandidatesPerAnchorPrefixSum.get(),
                batchData.d_numCandidatesPerAnchor.get(),
                batchData.d_anchorIndicesOfCandidates.get(),
                batchData.h_numAnchors.get(),
                &batchData.d_numCandidatesPerAnchorPrefixSum[batchData.numTasks],
                d_anchorContainsN,
                removeAmbiguousAnchors,
                d_candidateContainsN,
                removeAmbiguousCandidates,
                maxNumAnchors,
                maxNumCandidates,
                maximumSequenceLength,
                encodedSequencePitchInInts2Bit,
                min_overlap,
                maxErrorRate,
                min_overlap_ratio,
                estimatedNucleotideErrorRate,
                stream,
                kernelLaunchHandle
            );
        };

        size_t tempstoragebytes = 0;
        callAlignmentKernel(nullptr, tempstoragebytes);

        batchData.d_tempstorage.resize(tempstoragebytes);

        callAlignmentKernel(batchData.d_tempstorage.get(), tempstoragebytes);
    }



    void filterAlignments(BatchData& batchData, cudaStream_t stream) const{

        const int totalNumCandidates = batchData.totalNumCandidates;
        const int numAnchors = batchData.numTasks;

        batchData.d_alignment_overlaps2.resize(batchData.totalNumCandidates);
        batchData.d_alignment_shifts2.resize(batchData.totalNumCandidates);
        batchData.d_alignment_nOps2.resize(batchData.totalNumCandidates);
        batchData.d_alignment_best_alignment_flags2.resize(batchData.totalNumCandidates);

        helpers::call_fill_kernel_async(batchData.d_flagscandidates.data(), batchData.d_flagscandidates.size(), true, stream);

        bool* const d_keepflags = batchData.d_flagscandidates.data();

        dim3 block(128,1,1);
        dim3 grid(numAnchors, 1, 1);

        //filter alignments of candidates. d_keepflags[i] will be set to false if candidate[i] should be removed
        //batchData.d_numCandidatesPerAnchor2[i] contains new number of candidates for anchor i
        helpers::lambda_kernel<<<grid, block, 0, stream>>>(
            [
                d_alignment_best_alignment_flags = batchData.d_alignment_best_alignment_flags.data(),
                d_alignment_shifts = batchData.d_alignment_shifts.data(),
                d_alignment_overlaps = batchData.d_alignment_overlaps.data(),
                d_anchorSequencesLength = batchData.d_anchorSequencesLength.data(),
                d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchor2 = batchData.d_numCandidatesPerAnchor2.data(),
                d_numCandidatesPerAnchorPrefixSum = batchData.d_numCandidatesPerAnchorPrefixSum.data(),
                d_keepflags,
                min_overlap_ratio = goodAlignmentProperties.min_overlap_ratio,
                numAnchors
            ] __device__ (){

                using BlockReduceFloat = cub::BlockReduce<float, 128>;
                using BlockReduceInt = cub::BlockReduce<int, 128>;

                __shared__ union {
                    typename BlockReduceFloat::TempStorage floatreduce;
                    typename BlockReduceInt::TempStorage intreduce;
                } cubtemp;

                __shared__ int intbroadcast;
                __shared__ float floatbroadcast;

                for(int a = blockIdx.x; a < numAnchors; a += gridDim.x){
                    const int num = d_numCandidatesPerAnchor[a];
                    const int offset = d_numCandidatesPerAnchorPrefixSum[a];
                    const float anchorLength = d_anchorSequencesLength[a];
                    int removed = 0;

                    int threadReducedGoodAlignmentExists = 0;
                    float threadReducedRelativeOverlapThreshold = 0.0f;

                    //loop over candidates to compute relative overlap threshold

                    for(int c = threadIdx.x; c < num; c += blockDim.x){
                        const auto alignmentflag = d_alignment_best_alignment_flags[offset + c];
                        const int shift = d_alignment_shifts[offset + c];

                        if(alignmentflag != BestAlignment_t::None && shift >= 0){
                            const float overlap = d_alignment_overlaps[offset + c];                            
                            const float relativeOverlap = overlap / anchorLength;
                            
                            if(relativeOverlap < 1.0f && fgeq(relativeOverlap, min_overlap_ratio)){
                                threadReducedGoodAlignmentExists = 1;
                                const float tmp = floorf(relativeOverlap * 10.0f) / 10.0f;
                                threadReducedRelativeOverlapThreshold = fmaxf(threadReducedRelativeOverlapThreshold, tmp);
                            }
                        }else{
                            //remove alignment with negative shift or bad alignments
                            d_keepflags[offset + c] = false;
                            removed++;
                        }                       
                    }

                    int blockreducedGoodAlignmentExists = BlockReduceInt(cubtemp.intreduce)
                        .Sum(threadReducedGoodAlignmentExists);
                    if(threadIdx.x == 0){
                        intbroadcast = blockreducedGoodAlignmentExists;
                        //printf("task %d good: %d\n", a, blockreducedGoodAlignmentExists);
                    }
                    __syncthreads();

                    blockreducedGoodAlignmentExists = intbroadcast;

                    if(blockreducedGoodAlignmentExists > 0){
                        float blockreducedRelativeOverlapThreshold = BlockReduceFloat(cubtemp.floatreduce)
                            .Reduce(threadReducedRelativeOverlapThreshold, cub::Max());
                        if(threadIdx.x == 0){
                            floatbroadcast = blockreducedRelativeOverlapThreshold;
                            //printf("task %d thresh: %f\n", a, blockreducedRelativeOverlapThreshold);
                        }
                        __syncthreads();

                        blockreducedRelativeOverlapThreshold = floatbroadcast;

                        // loop over candidates and remove those with an alignment overlap threshold smaller than the computed threshold
                        for(int c = threadIdx.x; c < num; c += blockDim.x){
    
                            if(d_keepflags[offset + c]){
                                const float overlap = d_alignment_overlaps[offset + c];                            
                                const float relativeOverlap = overlap / anchorLength;                 
    
                                if(!fgeq(relativeOverlap, blockreducedRelativeOverlapThreshold)){
                                    d_keepflags[offset + c] = false;
                                    removed++;
                                }
                            }
                        }
                    }else{
                        //NOOP.
                        //if no good alignment exists, no candidate is removed. we will try to work with the not-so-good alignments
                    }

                    removed = BlockReduceInt(cubtemp.intreduce).Sum(removed);

                    if(threadIdx.x == 0){
                        d_numCandidatesPerAnchor2[a] = num - removed;
                    }
                    __syncthreads();
                }
            }
        ); CUERR;

        //setup cub 
        auto d_zip_input = thrust::make_zip_iterator(
            thrust::make_tuple(
                batchData.d_alignment_nOps.data(),
                batchData.d_alignment_overlaps.data(),
                batchData.d_alignment_shifts.data(),
                batchData.d_alignment_best_alignment_flags.data(),
                batchData.d_candidateReadIds.data(),
                batchData.d_candidateSequencesLength.data()
            )
        );

        auto d_zip_output = thrust::make_zip_iterator(
            thrust::make_tuple(
                batchData.d_alignment_nOps2.data(),
                batchData.d_alignment_overlaps2.data(),
                batchData.d_alignment_shifts2.data(),
                batchData.d_alignment_best_alignment_flags2.data(),
                batchData.d_candidateReadIds2.data(),
                batchData.d_candidateSequencesLength2.data()
            )
        );

        std::size_t requiredCubSize1 = 0;
        cudaError_t cubstatus = cub::DeviceSelect::Flagged(
            nullptr, 
            requiredCubSize1, 
            d_zip_input, 
            d_keepflags, 
            d_zip_output, 
            batchData.d_numCandidates.data(), 
            totalNumCandidates, 
            stream
        );
        assert(cubstatus == cudaSuccess);

        std::size_t requiredCubSize2 = 0;
        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            requiredCubSize2,
            batchData.d_candidateSequencesData.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                SequenceFlagMultiplier{d_keepflags, int(encodedSequencePitchInInts)}
            ),
            batchData.d_candidateSequencesData2.data(),
            thrust::make_discard_iterator(),
            totalNumCandidates * encodedSequencePitchInInts,
            stream
        );
        assert(cubstatus == cudaSuccess);

        std::size_t requiredCubSize3 = 0;
        cubstatus = cub::DeviceScan::InclusiveSum(
            nullptr,
            requiredCubSize3,
            batchData.d_numCandidatesPerAnchor2.data(), 
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            batchData.numTasks, 
            stream
        );
        assert(cudaSuccess == cubstatus);

        std::size_t requiredCubSize = std::max(std::max(requiredCubSize1, requiredCubSize2), requiredCubSize3);
        void* cubtemp; cubAllocator->DeviceAllocate((void**)&cubtemp, requiredCubSize, stream);

        //compact zip data
        cubstatus = cub::DeviceSelect::Flagged(
            cubtemp, 
            requiredCubSize, 
            d_zip_input, 
            d_keepflags, 
            d_zip_output, 
            batchData.d_numCandidates.data(), 
            totalNumCandidates, 
            stream
        );
        assert(cubstatus == cudaSuccess);

        //compact sequence data.

        cubstatus = cub::DeviceSelect::Flagged(
            cubtemp,
            requiredCubSize,
            batchData.d_candidateSequencesData.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                SequenceFlagMultiplier{d_keepflags, int(encodedSequencePitchInInts)}
            ),
            batchData.d_candidateSequencesData2.data(),
            thrust::make_discard_iterator(),
            totalNumCandidates * encodedSequencePitchInInts,
            stream
        );
        assert(cubstatus == cudaSuccess);

        //update prefix sum
        cubstatus = cub::DeviceScan::InclusiveSum(
            cubtemp, 
            requiredCubSize, 
            batchData.d_numCandidatesPerAnchor2.data(), 
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            batchData.numTasks, 
            stream
        );
        assert(cudaSuccess == cubstatus);


        cubAllocator->DeviceFree(cubtemp);

        std::swap(batchData.d_alignment_nOps2, batchData.d_alignment_nOps);
        std::swap(batchData.d_alignment_overlaps2, batchData.d_alignment_overlaps);
        std::swap(batchData.d_alignment_shifts2, batchData.d_alignment_shifts);
        std::swap(batchData.d_alignment_best_alignment_flags2, batchData.d_alignment_best_alignment_flags);
        std::swap(batchData.d_candidateReadIds2, batchData.d_candidateReadIds);
        std::swap(batchData.d_candidateSequencesLength2, batchData.d_candidateSequencesLength);
        std::swap(batchData.d_numCandidatesPerAnchor2, batchData.d_numCandidatesPerAnchor);
        std::swap(batchData.d_candidateSequencesData2, batchData.d_candidateSequencesData);

        cudaMemcpyAsync(
            &batchData.totalNumCandidates,
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + batchData.numTasks,
            sizeof(int),
            D2H,
            stream
        ); CUERR;

        cudaStreamSynchronize(stream); CUERR;
    }


    void computeMSAs(BatchData& batchData, cudaStream_t firstStream, cudaStream_t secondStream) const{
        batchData.d_consensus.resize(batchData.numTasks * msaColumnPitchInElements);
        batchData.d_support.resize(batchData.numTasks * msaColumnPitchInElements);
        batchData.d_coverage.resize(batchData.numTasks * msaColumnPitchInElements);
        batchData.d_origWeights.resize(batchData.numTasks * msaColumnPitchInElements);
        batchData.d_origCoverages.resize(batchData.numTasks * msaColumnPitchInElements);
        batchData.d_msa_column_properties.resize(batchData.numTasks);
        batchData.d_counts.resize(batchData.numTasks * 4 * msaColumnPitchInElements);
        batchData.d_weights.resize(batchData.numTasks * 4 * msaColumnPitchInElements);

        batchData.d_intbuffercandidates.resize(batchData.totalNumCandidates);
        batchData.d_indexlist1.resize(batchData.totalNumCandidates);
        batchData.d_numCandidatesPerAnchor2.resize(batchData.numTasks);
        batchData.d_flagscandidates.resize(batchData.totalNumCandidates);

        int* const indices1 = batchData.d_intbuffercandidates.data();
        int* const indices2 = batchData.d_indexlist1.data();

        helpers::lambda_kernel<<<batchData.numTasks, 128, 0, firstStream>>>(
            [
                indices1,
                d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.get(),
                d_numCandidatesPerAnchorPrefixSum = batchData.d_numCandidatesPerAnchorPrefixSum.get()
            ] __device__ (){
                const int num = d_numCandidatesPerAnchor[blockIdx.x];
                const int offset = d_numCandidatesPerAnchorPrefixSum[blockIdx.x];
                
                for(int i = threadIdx.x; i < num; i += blockDim.x){
                    indices1[offset + i] = i;
                }
            }
        );

        gpu::GPUMultiMSA multiMSA;

        *batchData.h_numAnchors = batchData.numTasks;

        multiMSA.numMSAs = batchData.numTasks;
        multiMSA.columnPitchInElements = msaColumnPitchInElements;
        multiMSA.counts = batchData.d_counts.get();
        multiMSA.weights = batchData.d_weights.get();
        multiMSA.coverages = batchData.d_coverage.get();
        multiMSA.consensus = batchData.d_consensus.get();
        multiMSA.support = batchData.d_support.get();
        multiMSA.origWeights = batchData.d_origWeights.get();
        multiMSA.origCoverages = batchData.d_origCoverages.get();
        multiMSA.columnProperties = batchData.d_msa_column_properties.get();

        callConstructMultipleSequenceAlignmentsKernel_async(
            multiMSA,
            batchData.d_alignment_overlaps.get(),
            batchData.d_alignment_shifts.get(),
            batchData.d_alignment_nOps.get(),
            batchData.d_alignment_best_alignment_flags.get(),
            batchData.d_anchorSequencesLength.get(),
            batchData.d_candidateSequencesLength.get(),
            indices1, //d_indices,
            batchData.d_numCandidatesPerAnchor.get(),
            batchData.d_numCandidatesPerAnchorPrefixSum.get(),
            batchData.d_subjectSequencesData.get(),
            batchData.d_candidateSequencesData.get(),
            nullptr, //d_anchor_qualities.get(),
            nullptr, //d_candidate_qualities.get(),
            batchData.h_numAnchors.get(), //d_numAnchors
            goodAlignmentProperties.maxErrorRate,
            batchData.numTasks,
            batchData.totalNumCandidates,
            false, //correctionOptions->useQualityScores,
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            firstStream,
            kernelLaunchHandle
        );

        //refine msa
        bool* const d_shouldBeKept = (bool*)batchData.d_flagscandidates.get();

        callMsaCandidateRefinementKernel_multiiter_async(
            indices2,
            batchData.d_numCandidatesPerAnchor2.data(),
            batchData.d_numCandidates2.get(),
            multiMSA,
            batchData.d_alignment_best_alignment_flags.get(),
            batchData.d_alignment_shifts.get(),
            batchData.d_alignment_nOps.get(),
            batchData.d_alignment_overlaps.get(),
            batchData.d_subjectSequencesData.get(),
            batchData.d_candidateSequencesData.get(),
            batchData.d_anchorSequencesLength.get(),
            batchData.d_candidateSequencesLength.get(),
            nullptr, //d_anchor_qualities.get(),
            nullptr, //d_candidate_qualities.get(),
            d_shouldBeKept,
            batchData.d_numCandidatesPerAnchorPrefixSum.get(),
            batchData.h_numAnchors.get(),
            goodAlignmentProperties.maxErrorRate,
            batchData.numTasks,
            batchData.totalNumCandidates,
            false, //correctionOptions->useQualityScores,
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            indices1, //d_indices,
            batchData.d_numCandidatesPerAnchor.get(),
            correctionOptions.estimatedCoverage,
            getNumRefinementIterations(),
            firstStream,
            kernelLaunchHandle
        );

        cudaEventRecord(events[0], firstStream); CUERR;
        cudaStreamWaitEvent(secondStream, events[0], 0); CUERR;

        cudaMemcpyAsync(
            batchData.h_numCandidates.data(),
            batchData.d_numCandidates.data(),
            sizeof(int),
            D2H,
            secondStream
        ); CUERR;

        cudaEventRecord(events[0], secondStream); CUERR;

        bool cubdebugsync = false;
        cudaError_t cubstatus = cudaSuccess;

        //allocate cub storage
        std::size_t cubBytes = 0;
        std::size_t cubBytes2 = 0;
        cubstatus = cub::DeviceScan::InclusiveSum(
            nullptr,
            cubBytes,
            batchData.d_numCandidatesPerAnchor2.get(), 
            batchData.d_numCandidatesPerAnchorPrefixSum2.get() + 1, 
            batchData.numTasks, 
            firstStream,
            cubdebugsync
        );
        assert(cubstatus == cudaSuccess);

        auto in_zipped_begin = thrust::make_zip_iterator(
            thrust::make_tuple(
                batchData.d_candidateReadIds.data(),
                batchData.d_candidateSequencesLength.data(),
                batchData.d_alignment_overlaps.data(),
                batchData.d_alignment_shifts.data(),
                batchData.d_alignment_nOps.data(),
                batchData.d_alignment_best_alignment_flags.data()
            )
        );

        auto out_zipped_begin = thrust::make_zip_iterator(
            thrust::make_tuple(
                batchData.d_candidateReadIds2.data(),
                batchData.d_candidateSequencesLength2.data(),
                batchData.d_alignment_overlaps2.data(),
                batchData.d_alignment_shifts2.data(),
                batchData.d_alignment_nOps2.data(),
                batchData.d_alignment_best_alignment_flags2.data()
            )
        );

        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            cubBytes2,
            in_zipped_begin,
            batchData.d_flagscandidates.data(),
            out_zipped_begin,
            thrust::make_discard_iterator(),
            batchData.totalNumCandidates,
            firstStream,
            cubdebugsync
        );

        assert(cubstatus == cudaSuccess);
        cubBytes = std::max(cubBytes, cubBytes2);

        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            cubBytes2,
            batchData.d_candidateSequencesData.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                SequenceFlagMultiplier{batchData.d_flagscandidates.data(), int(encodedSequencePitchInInts)}
            ),
            batchData.d_candidateSequencesData2.data(),
            thrust::make_discard_iterator(),
            batchData.totalNumCandidates * encodedSequencePitchInInts,
            firstStream,
            cubdebugsync
        );

        assert(cubstatus == cudaSuccess);
        cubBytes = std::max(cubBytes, cubBytes2);

        void* cubtemp; cubAllocator->DeviceAllocate((void**)&cubtemp, cubBytes, firstStream);

        cubstatus = cub::DeviceScan::InclusiveSum(
            cubtemp,
            cubBytes,
            batchData.d_numCandidatesPerAnchor2.get(), 
            batchData.d_numCandidatesPerAnchorPrefixSum2.get() + 1, 
            batchData.numTasks, 
            firstStream,
            cubdebugsync
        );
        assert(cubstatus == cudaSuccess);
      

        helpers::call_fill_kernel_async(batchData.d_flagscandidates.data(), batchData.totalNumCandidates, false, firstStream); CUERR;

        //convert output indices from task-local indices to global flags
        helpers::lambda_kernel<<<batchData.numTasks, 128, 0, firstStream>>>(
            [
                d_flagscandidates = batchData.d_flagscandidates.data(),
                indices2,
                d_numCandidatesPerAnchor2 = batchData.d_numCandidatesPerAnchor2.get(),
                d_numCandidatesPerAnchorPrefixSum = batchData.d_numCandidatesPerAnchorPrefixSum.get()
            ] __device__ (){
                /*
                    Input:
                    indices2: 0,1,2,0,0,0,0,3,5,0
                    d_numCandidatesPerAnchorPrefixSum: 0,6,10

                    Output:
                    d_flagscandidates: 1,1,1,0,0,0,1,0,0,1,0,1
                */
                const int num = d_numCandidatesPerAnchor2[blockIdx.x];
                const int offset = d_numCandidatesPerAnchorPrefixSum[blockIdx.x];
                
                for(int i = threadIdx.x; i < num; i += blockDim.x){
                    const int globalIndex = indices2[offset + i] + offset;
                    d_flagscandidates[globalIndex] = true;
                }
            }
        ); CUERR;

        //compact candidate sequences according to flags                

        cubstatus = cub::DeviceSelect::Flagged(
            cubtemp,
            cubBytes,
            batchData.d_candidateSequencesData.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                SequenceFlagMultiplier{batchData.d_flagscandidates.data(), int(encodedSequencePitchInInts)}
            ),
            batchData.d_candidateSequencesData2.data(),
            thrust::make_discard_iterator(),
            batchData.totalNumCandidates * encodedSequencePitchInInts,
            firstStream,
            cubdebugsync
        );

        assert(cubstatus == cudaSuccess);

        //compact other candidate buffers according to flags

        cubstatus = cub::DeviceSelect::Flagged(
            cubtemp,
            cubBytes,
            in_zipped_begin,
            batchData.d_flagscandidates.data(),
            out_zipped_begin,
            thrust::make_discard_iterator(),
            batchData.totalNumCandidates,
            firstStream,
            cubdebugsync
        );

        assert(cubstatus == cudaSuccess);

        std::swap(batchData.d_numCandidatesPerAnchor, batchData.d_numCandidatesPerAnchor2);
        std::swap(batchData.d_numCandidatesPerAnchorPrefixSum, batchData.d_numCandidatesPerAnchorPrefixSum2);                
        std::swap(batchData.d_candidateSequencesData, batchData.d_candidateSequencesData2);
        std::swap(batchData.d_candidateReadIds, batchData.d_candidateReadIds2);
        std::swap(batchData.d_candidateSequencesLength, batchData.d_candidateSequencesLength2);
        std::swap(batchData.d_alignment_overlaps, batchData.d_alignment_overlaps2);
        std::swap(batchData.d_alignment_shifts, batchData.d_alignment_shifts2);
        std::swap(batchData.d_alignment_nOps, batchData.d_alignment_nOps2);
        std::swap(batchData.d_alignment_best_alignment_flags, batchData.d_alignment_best_alignment_flags2);

        //compute possible msa splits
        batchData.d_possibleSplitColumns.resize(32 * batchData.numTasks);
        batchData.d_numPossibleSplitColumnsPerAnchor.resize(batchData.numTasks);

        helpers::lambda_kernel<<<batchData.numTasks, 128, 0, firstStream>>>(
            [
                d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.data(),
                d_counts = batchData.d_counts.data(),
                d_coverage = batchData.d_coverage.data(),
                msaColumnPitchInElements = this->msaColumnPitchInElements,
                d_msa_column_properties = batchData.d_msa_column_properties.data(),
                numTasks = batchData.numTasks,
                splitcolumnsPitchElements = 32,
                d_possibleSplitColumns = batchData.d_possibleSplitColumns.data(),
                d_numPossibleSplitColumnsPerTask = batchData.d_numPossibleSplitColumnsPerAnchor.data()
            ] __device__ (){

                using PSC = MultipleSequenceAlignment::PossibleSplitColumn;
                constexpr int maxColumnsPerTask = 32;

                __shared__ PSC sharedPSC[maxColumnsPerTask];
                __shared__ int broadcastint;

                using BlockReduce = cub::BlockReduce<int, 128>;
                using BlockScan = cub::BlockScan<int, 128>;
                __shared__ typename BlockReduce::TempStorage blockreducetemp;
                __shared__ typename BlockScan::TempStorage blockscantemp;

                for(int task = blockIdx.x; task < numTasks; task += gridDim.x){

                    int* const numSplitColumnsPtr = d_numPossibleSplitColumnsPerTask + task;
                    PSC* const splitColumnsPtr = d_possibleSplitColumns + splitcolumnsPitchElements * task;

                    if(d_numCandidatesPerAnchor[task] > 0){

                        //only check columns to the right of anchor
                        const int firstColumn = d_msa_column_properties[task].subjectColumnsEnd_excl;
                        const int lastColumnExcl = d_msa_column_properties[task].lastColumn_excl;                                

                        int* myCountsPtr[4];
                        myCountsPtr[0] = d_counts + 4 * msaColumnPitchInElements * task + 0 * msaColumnPitchInElements;
                        myCountsPtr[1] = d_counts + 4 * msaColumnPitchInElements * task + 1 * msaColumnPitchInElements;
                        myCountsPtr[2] = d_counts + 4 * msaColumnPitchInElements * task + 2 * msaColumnPitchInElements;
                        myCountsPtr[3] = d_counts + 4 * msaColumnPitchInElements * task + 3 * msaColumnPitchInElements;   
                        
                        int* myCoveragesPtr = d_coverage + msaColumnPitchInElements * task;

                        int totalNumResults = 0;

                        const int numIterations = SDIV(lastColumnExcl - firstColumn, blockDim.x);

                        for(int iteration = 0; iteration < numIterations; iteration++){
                            const int col = firstColumn + iteration * blockDim.x + threadIdx.x;

                            PSC myresults[3];
                            int myNumResults = 0;

                            if(col < lastColumnExcl){                                   

                                auto checkNuc = [&](const auto& counts, const char nuc){
                                    if(myNumResults < 3){

                                        const float ratio = float(counts[col]) / float(myCoveragesPtr[col]);
                                        //if((counts[col] == 2 && fgeq(ratio, 0.4f) && fleq(ratio, 0.6f)) || counts[col] > 2){
                                        if(counts[col] >= 2 && fgeq(ratio, 0.4f) && fleq(ratio, 0.6f)){

                                            #pragma unroll
                                            for(int k = 0; k < 3; k++){
                                                if(myNumResults == k){
                                                    myresults[k] = {nuc, col, ratio};
                                                    myNumResults++;
                                                    break;
                                                }
                                            }
                                            
                                        }
                                    }
                                };

                                checkNuc(myCountsPtr[0], 'A');
                                checkNuc(myCountsPtr[1], 'C');
                                checkNuc(myCountsPtr[2], 'G');
                                checkNuc(myCountsPtr[3], 'T');

                                if(myNumResults != 2){
                                    myNumResults = 0;
                                }    
                            }

                            int totalNumResultsIteration = BlockReduce(blockreducetemp).Sum(myNumResults);
                            if(threadIdx.x == 0){
                                broadcastint = totalNumResultsIteration;
                            }
                            __syncthreads();
                            totalNumResultsIteration = broadcastint;

                            if(totalNumResultsIteration + totalNumResults > maxColumnsPerTask){
                                totalNumResults = 0;
                                break;
                            }else{
                                int outputoffset = 0;

                                BlockScan(blockscantemp).ExclusiveSum(myNumResults, outputoffset);

                                if(myNumResults == 2){
                                    sharedPSC[totalNumResults + outputoffset + 0] = myresults[0];
                                    sharedPSC[totalNumResults + outputoffset + 1] = myresults[1];
                                }

                                totalNumResults += totalNumResultsIteration;
                            }

                            __syncthreads();
                        }

                        for(int i = threadIdx.x; i < totalNumResults; i += blockDim.x){
                            splitColumnsPtr[i] = sharedPSC[i];
                        }

                        if(threadIdx.x == 0){
                            *numSplitColumnsPtr = totalNumResults;
                        }
                    }else{
                        if(threadIdx.x == 0){
                            *numSplitColumnsPtr = 0;
                        }
                    }
                }
            }
        ); CUERR;

        cudaEventSynchronize(events[0]); CUERR; //wait for h_numCandidates

        batchData.totalNumCandidates = *batchData.h_numCandidates; 

        cubAllocator->DeviceFree(cubtemp);
    }

    void copyBuffersToHost(BatchData& batchData, cudaStream_t firstStream, cudaStream_t secondStream) const{
        batchData.h_candidateReadIds.resize(batchData.totalNumCandidates);
        batchData.h_candidateSequencesLength.resize(batchData.totalNumCandidates);
        batchData.h_candidateSequencesData.resize(encodedSequencePitchInInts * batchData.totalNumCandidates);
        batchData.h_candidateSequencesRevcData.resize(encodedSequencePitchInInts * batchData.totalNumCandidates);

        batchData.h_alignment_overlaps.resize(batchData.totalNumCandidates);
        batchData.h_alignment_shifts.resize(batchData.totalNumCandidates);
        batchData.h_alignment_nOps.resize(batchData.totalNumCandidates);
        batchData.h_alignment_best_alignment_flags.resize(batchData.totalNumCandidates);

        batchData.h_possibleSplitColumns.resize(32 * batchData.numTasks);
        batchData.h_numPossibleSplitColumnsPerAnchor.resize(batchData.numTasks);
        batchData.h_consensus.resize(batchData.numTasks * msaColumnPitchInElements);
        batchData.h_msa_column_properties.resize(batchData.numTasks);

        //convert encoded consensus to characters and copy to host
        //copy column properties to host
        helpers::lambda_kernel<<<batchData.numTasks, 128, 0, secondStream>>>(
            [
                msaColumnPitchInElements = this->msaColumnPitchInElements,
                d_consensus = batchData.d_consensus.data(),
                h_consensus = batchData.h_consensus.data(),
                h_msa_column_properties = batchData.h_msa_column_properties.data(),
                d_msa_column_properties = batchData.d_msa_column_properties.data(),
                numTasks = batchData.numTasks
            ] __device__ (){

                auto decode = [](const std::uint8_t encoded){
                    char decoded = 'F';
                    if(encoded == std::uint8_t{0}){
                        decoded = 'A';
                    }else if(encoded == std::uint8_t{1}){
                        decoded = 'C';
                    }else if(encoded == std::uint8_t{2}){
                        decoded = 'G';
                    }else if(encoded == std::uint8_t{3}){
                        decoded = 'T';
                    }
                    return decoded;
                };

                const std::uint8_t* inputConsensus = &d_consensus[blockIdx.x * msaColumnPitchInElements];
                char* outputConsensus = &h_consensus[blockIdx.x * msaColumnPitchInElements];

                const int numVecIters = msaColumnPitchInElements / sizeof(char4);

                for(int i = threadIdx.x; i < numVecIters; i += blockDim.x){
                    char4 data = ((const char4*)inputConsensus)[i];

                    char4 result;
                    result.x = decode(data.x);
                    result.y = decode(data.y);
                    result.z = decode(data.z);
                    result.w = decode(data.w);

                    ((char4*)outputConsensus)[i] = result;
                }

                const int remaining = msaColumnPitchInElements - (numVecIters * sizeof(char4));
                
                for(int i = threadIdx.x; i < remaining; i += blockDim.x){
                    std::uint8_t encoded = d_consensus[blockIdx.x * msaColumnPitchInElements + numVecIters * sizeof(char4) + i];

                    char decoded = decode(encoded);

                    h_consensus[blockIdx.x * msaColumnPitchInElements + numVecIters * sizeof(char4) + i] = decoded;
                }

                //copy column properties
                for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < numTasks; i += blockDim.x * gridDim.x){
                    h_msa_column_properties[i] = d_msa_column_properties[i];
                }
            }
        );

        cudaMemcpyAsync(
            batchData.h_possibleSplitColumns.get(),
            batchData.d_possibleSplitColumns.get(),
            sizeof(MultipleSequenceAlignment::PossibleSplitColumn) * 32 * batchData.numTasks,
            D2H,
            firstStream
        ); CUERR;

        helpers::call_copy_n_kernel(
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1,
                batchData.d_numCandidatesPerAnchor.data(),
                batchData.d_numPossibleSplitColumnsPerAnchor.get()
            )),
            batchData.numTasks,
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.h_numCandidatesPerAnchorPrefixSum.data() + 1,
                batchData.h_numCandidatesPerAnchor.data(),
                batchData.h_numPossibleSplitColumnsPerAnchor.get()
            )),
            firstStream
        );          

        cudaMemcpyAsync(
            batchData.h_candidateSequencesData.get(),
            batchData.d_candidateSequencesData.get(),
            sizeof(unsigned int) * batchData.totalNumCandidates * encodedSequencePitchInInts,
            D2H,
            firstStream
        ); CUERR;

        auto d_zipped_begin = thrust::make_zip_iterator(
            thrust::make_tuple(
                batchData.d_candidateReadIds.data(),
                batchData.d_candidateSequencesLength.data(),
                batchData.d_alignment_overlaps.data(),
                batchData.d_alignment_shifts.data(),
                batchData.d_alignment_nOps.data(),
                batchData.d_alignment_best_alignment_flags.data()
            )
        );

        auto h_zipped_begin = thrust::make_zip_iterator(
            thrust::make_tuple(
                batchData.h_candidateReadIds.data(),
                batchData.h_candidateSequencesLength.data(),
                batchData.h_alignment_overlaps.data(),
                batchData.h_alignment_shifts.data(),
                batchData.h_alignment_nOps.data(),
                batchData.h_alignment_best_alignment_flags.data()
            )
        );

        helpers::call_copy_n_kernel(
            d_zipped_begin,
            batchData.totalNumCandidates,
            h_zipped_begin,
            firstStream
        );
    }

    void copyBatchDataIntoTask(Task& task, int taskindex, const BatchData& data) const{
        const int numCandidates = data.h_numCandidatesPerAnchor[taskindex];
        const int offset = data.h_numCandidatesPerAnchorPrefixSum[taskindex];

        task.candidateReadIds.resize(numCandidates);
        std::copy_n(data.h_candidateReadIds.data() + offset, numCandidates, task.candidateReadIds.begin());

        task.candidateSequenceLengths.resize(numCandidates);
        std::copy_n(data.h_candidateSequencesLength.data() + offset, numCandidates, task.candidateSequenceLengths.begin());

        task.candidateSequenceData.resize(numCandidates * encodedSequencePitchInInts);
        std::copy_n(
            data.h_candidateSequencesData.data() + offset * encodedSequencePitchInInts, 
            numCandidates * encodedSequencePitchInInts, 
            task.candidateSequenceData.begin()
        );

        task.alignmentFlags.resize(numCandidates);
        task.alignments.resize(numCandidates);

        for(int c = 0; c < numCandidates; c++){
            task.alignments[c].shift = data.h_alignment_shifts[offset + c];
            task.alignments[c].overlap = data.h_alignment_overlaps[offset + c];
            task.alignments[c].nOps = data.h_alignment_nOps[offset + c];
            task.alignmentFlags[c] = data.h_alignment_best_alignment_flags[offset + c];
        }

        task.numRemainingCandidates = numCandidates;

        if(task.numRemainingCandidates == 0){
            task.abort = true;
            task.abortReason = AbortReason::NoPairedCandidatesAfterAlignment;
        }
    }

    MultipleSequenceAlignment constructMsaWithDataFromTask(Task& task) const{
        const std::string& decodedAnchor = task.totalDecodedAnchors.back();

        MultipleSequenceAlignment msa;

        auto build = [&](){

            task.candidateShifts.resize(task.numRemainingCandidates);
            task.candidateOverlapWeights.resize(task.numRemainingCandidates);

            //gather data required for msa
            for(int c = 0; c < task.numRemainingCandidates; c++){
                task.candidateShifts[c] = task.alignments[c].shift;

                task.candidateOverlapWeights[c] = calculateOverlapWeight(
                    task.currentAnchorLength, 
                    task.alignments[c].nOps,
                    task.alignments[c].overlap,
                    goodAlignmentProperties.maxErrorRate
                );
            }

            task.candidateStrings.resize(decodedSequencePitchInBytes * task.numRemainingCandidates, '\0');

            //decode the candidates for msa
            for(int c = 0; c < task.numRemainingCandidates; c++){
                SequenceHelpers::decode2BitSequence(
                    task.candidateStrings.data() + c * decodedSequencePitchInBytes,
                    task.candidateSequenceData.data() + c * encodedSequencePitchInInts,
                    task.candidateSequenceLengths[c]
                );

                if(task.alignmentFlags[c] == BestAlignment_t::ReverseComplement){
                    SequenceHelpers::reverseComplementSequenceDecodedInplace(
                        task.candidateStrings.data() + c * decodedSequencePitchInBytes, 
                        task.candidateSequenceLengths[c]
                    );
                }
            }

            MultipleSequenceAlignment::InputData msaInput;
            msaInput.useQualityScores = false;
            msaInput.subjectLength = task.currentAnchorLength;
            msaInput.nCandidates = task.numRemainingCandidates;
            msaInput.candidatesPitch = decodedSequencePitchInBytes;
            msaInput.candidateQualitiesPitch = 0;
            msaInput.subject = decodedAnchor.c_str();
            msaInput.candidates = task.candidateStrings.data();
            msaInput.subjectQualities = nullptr;
            msaInput.candidateQualities = nullptr;
            msaInput.candidateLengths = task.candidateSequenceLengths.data();
            msaInput.candidateShifts = task.candidateShifts.data();
            msaInput.candidateDefaultWeightFactors = task.candidateOverlapWeights.data();                    

            msa.build(msaInput);
        };

        build();

        #if 1

        auto removeCandidatesOfDifferentRegion = [&](const auto& minimizationResult){
            const int numCandidates = task.candidateReadIds.size();

            int insertpos = 0;
            for(int i = 0; i < numCandidates; i++){
                if(!minimizationResult.differentRegionCandidate[i]){               
                    //keep candidate

                    task.candidateReadIds[insertpos] = task.candidateReadIds[i];

                    std::copy_n(
                        task.candidateSequenceData.data() + i * size_t(encodedSequencePitchInInts),
                        encodedSequencePitchInInts,
                        task.candidateSequenceData.data() + insertpos * size_t(encodedSequencePitchInInts)
                    );

                    task.candidateSequenceLengths[insertpos] = task.candidateSequenceLengths[i];
                    task.alignmentFlags[insertpos] = task.alignmentFlags[i];
                    task.alignments[insertpos] = task.alignments[i];
                    task.candidateOverlapWeights[insertpos] = task.candidateOverlapWeights[i];
                    task.candidateShifts[insertpos] = task.candidateShifts[i];

                    std::copy_n(
                        task.candidateStrings.data() + i * size_t(decodedSequencePitchInBytes),
                        decodedSequencePitchInBytes,
                        task.candidateStrings.data() + insertpos * size_t(decodedSequencePitchInBytes)
                    );

                    insertpos++;
                }
            }

            task.numRemainingCandidates = insertpos;

            task.candidateReadIds.erase(
                task.candidateReadIds.begin() + insertpos, 
                task.candidateReadIds.end()
            );
            task.candidateSequenceData.erase(
                task.candidateSequenceData.begin() + encodedSequencePitchInInts * insertpos, 
                task.candidateSequenceData.end()
            );
            task.candidateSequenceLengths.erase(
                task.candidateSequenceLengths.begin() + insertpos, 
                task.candidateSequenceLengths.end()
            );
            task.alignmentFlags.erase(
                task.alignmentFlags.begin() + insertpos, 
                task.alignmentFlags.end()
            );
            task.alignments.erase(
                task.alignments.begin() + insertpos, 
                task.alignments.end()
            );

            task.candidateStrings.erase(
                task.candidateStrings.begin() + decodedSequencePitchInBytes * insertpos, 
                task.candidateStrings.end()
            );
            task.candidateOverlapWeights.erase(
                task.candidateOverlapWeights.begin() + insertpos, 
                task.candidateOverlapWeights.end()
            );
            task.candidateShifts.erase(
                task.candidateShifts.begin() + insertpos, 
                task.candidateShifts.end()
            );
            
        };

        if(getNumRefinementIterations() > 0){                

            for(int numIterations = 0; numIterations < getNumRefinementIterations(); numIterations++){
                const auto minimizationResult = msa.findCandidatesOfDifferentRegion(
                    correctionOptions.estimatedCoverage
                );

                if(minimizationResult.performedMinimization){
                    removeCandidatesOfDifferentRegion(minimizationResult);

                    //build minimized multiple sequence alignment
                    build();
                }else{
                    break;
                }               
                
            }
        }   

        #endif

        return msa;
    }

private:
    int deviceId;
    int kmerLength;
    std::size_t msaColumnPitchInElements = 0;

    thrust::device_new_allocator<char> thrustallocator{};
    cub::CachingDeviceAllocator* cubAllocator{};

    std::array<CudaStream, 4> streams{};
    std::array<CudaEvent, 1> events{};

    mutable gpu::KernelLaunchHandle kernelLaunchHandle;

    const gpu::GpuReadStorage* gpuReadStorage;
    const gpu::GpuMinhasher* gpuMinhasher;

    mutable ReadStorageHandle readStorageHandle;
    mutable gpu::GpuMinhasher::QueryHandle minhashHandle;


    BatchData batchData{};

};


}


#endif