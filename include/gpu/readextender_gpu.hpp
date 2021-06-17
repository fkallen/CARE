#ifndef READ_EXTENDER_GPU_HPP
#define READ_EXTENDER_GPU_HPP

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
#include <util.hpp>
#include <gpu/gpucpureadstorageadapter.cuh>
#include <gpu/gpucpuminhasheradapter.cuh>
#include <readextender_cpu.hpp>

#include <readextender_common.hpp>

#include <algorithm>
#include <vector>
#include <numeric>

#include <cub/cub.cuh>

#include <thrust/device_new_allocator.h>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/logical.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>


//#define DO_REMOVE_USED_IDS_AND_MATE_IDS_ON_GPU

#define DO_ONLY_REMOVE_MATE_IDS



#if 0
    #define DEBUGDEVICESYNC { \
        cudaDeviceSynchronize(); CUERR; \
    }

#else 
    #define DEBUGDEVICESYNC {}

#endif

namespace care{




namespace readextendergpukernels{

    template<int blocksize, int itemsPerThread, bool inclusive, class T>
    __global__
    void prefixSumSingleBlockKernel(
        const T* input,
        T* output,
        int N
    ){
        struct BlockPrefixCallbackOp{
            // Running prefix
            int running_total;

            __device__
            BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
            // Callback operator to be entered by the first warp of threads in the block.
            // Thread-0 is responsible for returning a value for seeding the block-wide scan.
            __device__
            int operator()(int block_aggregate){
                int old_prefix = running_total;
                running_total += block_aggregate;
                return old_prefix;
            }
        };

        assert(blocksize == blockDim.x);

        using BlockScan = cub::BlockScan<T, blocksize>;
        using BlockLoad = cub::BlockLoad<T, blocksize, itemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
        using BlockStore = cub::BlockStore<T, blocksize, itemsPerThread, cub::BLOCK_STORE_WARP_TRANSPOSE>;

        __shared__ typename BlockScan::TempStorage blockscantemp;
        __shared__ union{
            typename BlockLoad::TempStorage load;
            typename BlockStore::TempStorage store;
        } temp;

        T items[itemsPerThread];

        BlockPrefixCallbackOp prefix_op(0);

        const int iterations = SDIV(N, blocksize);

        int remaining = N;

        const T* currentInput = input;
        T* currentOutput = output;

        for(int iteration = 0; iteration < iterations; iteration++){
            const int valid_items = min(itemsPerThread * blocksize, remaining);

            BlockLoad(temp.load).Load(currentInput, items, valid_items, 0);

            if(inclusive){
                BlockScan(blockscantemp).InclusiveSum(
                    items, items, prefix_op
                );
            }else{
                BlockScan(blockscantemp).ExclusiveSum(
                    items, items, prefix_op
                );
            }
            __syncthreads();

            BlockStore(temp.store).Store(currentOutput, items, valid_items);
            __syncthreads();

            remaining -= valid_items;
            currentInput += valid_items;
            currentOutput += valid_items;
        }
    }

    //output[map[i]] = input[i];
    template<class T, class U>
    __global__ 
    void setFirstSegmentIdsKernel(
        const T* __restrict__ segmentSizes,
        int* __restrict__ segmentIds,
        const U* __restrict__ segmentOffsets,
        int N
    ){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        for(int i = tid; i < N; i += stride){
            if(segmentSizes[i] > 0){
                segmentIds[segmentOffsets[i]] = i;
            }
        }
    }

    __global__
    void setSegmentIndicesKernel(
        int* __restrict__ d_indices,
        int N,
        const int* __restrict__ d_segment_sizes,
        const int* __restrict__ d_segment_sizes_prefixsum
    ){
        for(int segmentIndex = blockIdx.x; segmentIndex < N; segmentIndex += gridDim.x){
            const int offset = d_segment_sizes_prefixsum[segmentIndex];
            const int size = d_segment_sizes[segmentIndex];
            int* const beginptr = &d_indices[offset];

            for(int localindex = threadIdx.x; localindex < size; localindex += blockDim.x){
                beginptr[localindex] = segmentIndex;
            }
        }
    }

    //flag candidates to remove because they are equal to anchor id or equal to mate id
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
    ){

        using BlockReduceInt = cub::BlockReduce<int, 128>;

        __shared__ typename BlockReduceInt::TempStorage intreduce1;
        __shared__ typename BlockReduceInt::TempStorage intreduce2;

        for(int a = blockIdx.x; a < numTasks; a += gridDim.x){
            const int size = numCandidatesPerAnchor[a];
            const int offset = numCandidatesPerAnchorPrefixSum[a];
            const read_number anchorId = anchorReadIds[a];
            read_number mateId = 0;
            if(isPairedEnd){
                mateId = mateReadIds[a];
            }

            int mateIsRemoved = 0;
            int numRemoved = 0;

            // if(threadIdx.x == 0){
            //     printf("looking for anchor %u, mate %u\n", anchorId, mateId);
            // }
            __syncthreads();

            for(int i = threadIdx.x; i < size; i+= blockDim.x){
                bool keep = true;

                const read_number candidateId = candidateReadIds[offset + i];
                //printf("tid %d, comp %u at position %d\n", threadIdx.x, candidateId, offset + i);

                if(candidateId == anchorId){
                    keep = false;
                    numRemoved++;
                }

                if(isPairedEnd && candidateId == mateId){
                    if(keep){
                        keep = false;
                        numRemoved++;
                    }
                    mateIsRemoved++;
                    //printf("mate removed. i = %d\n", i);
                }

                keepflags[offset + i] = keep;
            }
            //printf("tid = %d, mateIsRemoved = %d\n", threadIdx.x, mateIsRemoved);
            int numRemovedBlock = BlockReduceInt(intreduce1).Sum(numRemoved);
            int numMateRemovedBlock = BlockReduceInt(intreduce2).Sum(mateIsRemoved);
            if(threadIdx.x == 0){
                numCandidatesPerAnchorOut[a] = size - numRemovedBlock;
                //printf("numMateRemovedBlock %d\n", numMateRemovedBlock);
                if(numMateRemovedBlock > 0){
                    mateRemovedFlags[a] = true;
                }else{
                    mateRemovedFlags[a] = false;
                }
            }
        }
    }

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

struct BatchData{
    template<class T>
    using DeviceBuffer = helpers::SimpleAllocationDevice<T>;
    //using DeviceBuffer = helpers::SimpleAllocationPinnedHost<T>;

    template<class T>
    using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T>;

    enum class State{
        BeforePrepare,
        BeforeHash,
        BeforeRemoveIds,
        BeforeComputePairFlags,
        BeforeLoadCandidates,
        BeforeEraseData,
        BeforeAlignment,
        BeforeAlignmentFilter,
        BeforeMSA,
        BeforeExtend,
        BeforeCopyToHost,
        BeforeUnpack,
        Finished,
        None
    };

    static std::string to_string(State s){
        switch(s){
            case State::BeforePrepare: return "BeforePrepare";
            case State::BeforeHash: return "BeforeHash";
            case State::BeforeRemoveIds: return "BeforeRemoveIds";
            case State::BeforeComputePairFlags: return "BeforeComputePairFlags";
            case State::BeforeLoadCandidates: return "BeforeLoadCandidates";
            case State::BeforeEraseData: return "BeforeEraseData";
            case State::BeforeAlignment: return "BeforeAlignment";
            case State::BeforeAlignmentFilter: return "BeforeAlignmentFilter";
            case State::BeforeMSA: return "BeforeMSA";
            case State::BeforeExtend: return "BeforeExtend";
            case State::BeforeCopyToHost: return "BeforeCopyToHost";
            case State::BeforeUnpack: return "BeforeUnpack";
            case State::Finished: return "Finished";
            case State::None: return "None";
            default: return "Missing case BatchData::to_string(State)\n";
        };
    }

    bool isEmpty() const noexcept{
        return indicesOfActiveTasks.empty();
    }

    void setState(State newstate){      
        #if 0
        std::cerr << "batchdata " << someId << " statechange " << to_string(state) << " -> " << to_string(newstate) << "\n";
        #endif

        state = newstate;
    }




    bool needPrepareStep = false;
    bool pairedEnd = false;
    State state = State::None;
    int numTasks = 0;
    int numTasksWithMateRemoved = 0;
    int someId = 0;
    int numReadPairs = 0;

    int totalNumCandidates = 0;
    int totalNumberOfUsedIds = 0;
    int totalNumberOfFullyUsedIds = 0;

    std::size_t encodedSequencePitchInInts = 0;
    std::size_t decodedSequencePitchInBytes = 0;
    std::size_t msaColumnPitchInElements = 0;
    std::size_t qualityPitchInBytes = 0;

    std::size_t outputAnchorPitchInBytes = 0;
    std::size_t outputAnchorQualityPitchInBytes = 0;
    std::size_t decodedMatesRevCPitchInBytes = 0;

    PinnedBuffer<read_number> h_anchorReadIds{};
    DeviceBuffer<read_number> d_anchorReadIds{};
    PinnedBuffer<read_number> h_mateReadIds{};
    DeviceBuffer<read_number> d_mateReadIds{};
    PinnedBuffer<read_number> h_candidateReadIds{};
    DeviceBuffer<read_number> d_candidateReadIds{};

    PinnedBuffer<int> h_anchorIndicesOfCandidates{};

    PinnedBuffer<bool> h_isPairedCandidate{};
    DeviceBuffer<bool> d_isPairedCandidate{};

    DeviceBuffer<int> d_anchorIndicesOfCandidates{};
    DeviceBuffer<int> d_segmentIdsOfFullyUsedReadIds{};

    PinnedBuffer<int> h_segmentIdsOfFullyUsedReadIds{};
    PinnedBuffer<int> h_segmentIdsOfReadIds{};

    DeviceBuffer<unsigned int> d_anchormatedata{};

    PinnedBuffer<unsigned int> h_inputanchormatedata{};
    DeviceBuffer<unsigned int> d_inputanchormatedata{};

    DeviceBuffer<int> d_anchorIndicesWithRemovedMates{};

    PinnedBuffer<int> h_numCandidatesPerAnchor{};
    DeviceBuffer<int> d_numCandidatesPerAnchor{};
    DeviceBuffer<int> d_numCandidatesPerAnchor2{};
    PinnedBuffer<int> h_numCandidatesPerAnchorPrefixSum{};
    DeviceBuffer<int> d_numCandidatesPerAnchorPrefixSum{};
    DeviceBuffer<int> d_numCandidatesPerAnchorPrefixSum2{};

    DeviceBuffer<int> d_alignment_overlaps{};
    DeviceBuffer<int> d_alignment_shifts{};
    DeviceBuffer<int> d_alignment_nOps{};
    DeviceBuffer<BestAlignment_t> d_alignment_best_alignment_flags{};

    PinnedBuffer<int> h_numAnchors{};
    PinnedBuffer<int> h_numCandidates{};
    DeviceBuffer<int> d_numAnchors{};
    DeviceBuffer<int> d_numCandidates{};
    DeviceBuffer<int> d_numCandidates2{};
    PinnedBuffer<int> h_numAnchorsWithRemovedMates{};

    PinnedBuffer<int> h_anchorSequencesLength{};
    DeviceBuffer<int> d_anchorSequencesLength{};
    DeviceBuffer<int> d_candidateSequencesLength{};
    PinnedBuffer<unsigned int> h_subjectSequencesData{};
    PinnedBuffer<char> h_subjectSequencesDataDecoded{};

    DeviceBuffer<unsigned int> d_candidateSequencesData2{};

    DeviceBuffer<unsigned int> d_subjectSequencesData{};
    DeviceBuffer<unsigned int> d_candidateSequencesData{};

    PinnedBuffer<char> h_anchorQualityScores{};
    DeviceBuffer<char> d_anchorQualityScores{};

    PinnedBuffer<read_number> h_usedReadIds{};
    PinnedBuffer<int> h_numUsedReadIdsPerAnchor{};
    PinnedBuffer<int> h_numUsedReadIdsPerAnchorPrefixSum{};

    DeviceBuffer<read_number> d_usedReadIds{};
    DeviceBuffer<int> d_numUsedReadIdsPerAnchor{};
    DeviceBuffer<int> d_numUsedReadIdsPerAnchorPrefixSum{};

    DeviceBuffer<read_number> d_fullyUsedReadIds{};
    DeviceBuffer<int> d_numFullyUsedReadIdsPerAnchor{};
    DeviceBuffer<int> d_numFullyUsedReadIdsPerAnchorPrefixSum{};

    PinnedBuffer<read_number> h_fullyUsedReadIds{};
    PinnedBuffer<int> h_numFullyUsedReadIdsPerAnchor{};
    PinnedBuffer<int> h_numFullyUsedReadIdsPerAnchorPrefixSum{};

    DeviceBuffer<std::uint8_t> d_consensusEncoded; //encoded , 0-4
    DeviceBuffer<int> d_coverage;
    DeviceBuffer<gpu::MSAColumnProperties> d_msa_column_properties;

    DeviceBuffer<char> d_consensusQuality;

    PinnedBuffer<int> h_firstTasksOfPairsToCheck;

    PinnedBuffer<int> h_inputMateLengths;
    PinnedBuffer<bool> h_isPairedTask;
    PinnedBuffer<char> h_decodedMatesRevC;
    PinnedBuffer<int> h_scatterMap;

    PinnedBuffer<int> h_accumExtensionsLengths;
    PinnedBuffer<extension::AbortReason> h_abortReasons;
    PinnedBuffer<char> h_outputAnchors;
    PinnedBuffer<char> h_outputAnchorQualities;
    PinnedBuffer<int> h_outputAnchorLengths;
    PinnedBuffer<bool> h_outputMateHasBeenFound;
    PinnedBuffer<int> h_sizeOfGapToMate;
    PinnedBuffer<bool> h_isFullyUsedCandidate{};

    
    std::array<CudaEvent, 1> events{};
    std::array<CudaStream, 4> streams{};
    std::vector<int> indicesOfActiveTasks{};
    std::vector<extension::Task> tasks;

};




struct GpuReadHasher{
public:

    GpuReadHasher() = default;

    GpuReadHasher(
        const gpu::GpuMinhasher& mh, MinhasherHandle minhashHandle_
    ) : gpuMinhasher(&mh),
        minhashHandle(minhashHandle_) {
    }
   
    void getCandidateReadIds(BatchData& batchData, cudaStream_t stream) const{

        int totalNumValues = 0;

        gpuMinhasher->determineNumValues(
            minhashHandle,
            batchData.d_subjectSequencesData.get(),
            batchData.encodedSequencePitchInInts,
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
            batchData.h_numCandidates.data(),
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + batchData.numTasks,
            sizeof(int),
            D2H,
            stream
        ); CUERR;

        cudaStreamSynchronize(stream); CUERR;

        batchData.totalNumCandidates = *batchData.h_numCandidates;

        // {

        //     std::vector<int> offsets(batchData.numTasks + 1);
        //     cudaMemcpyAsync(
        //         offsets.data(),
        //         batchData.d_numCandidatesPerAnchorPrefixSum.data(),
        //         sizeof(int) * (batchData.numTasks + 1),
        //         D2H,
        //         stream
        //     );

        //     cudaDeviceSynchronize(); CUERR;
        //     std::cerr << "Offsets after retrieveValues:\n";
        //     for(int i = 0; i < batchData.numTasks+1; i++){
        //         std::cerr << offsets[i] << " ";
        //     }
        //     std::cerr << "\n";

        // }
    }

    const gpu::GpuMinhasher* gpuMinhasher{};
    mutable MinhasherHandle minhashHandle;
};


struct GpuExtensionStepper{
public:

    enum class ComputeType {CPU, GPU};

    int deviceId{};
    int insertSize{};
    int insertSizeStddev{};
    int maxextensionPerStep{1};
    int minCoverageForExtension{1};
    cub::CachingDeviceAllocator* cubAllocator{};
    const gpu::GpuReadStorage* gpuReadStorage{};
    const gpu::GpuMinhasher* gpuMinhasher{};
    mutable MinhasherHandle minhashHandle{};
    GpuReadHasher gpuReadHasher{};
    const CorrectionOptions* correctionOptions{};
    const GoodAlignmentProperties* goodAlignmentProperties{};
    const cpu::QualityScoreConversion* qualityConversion{};
    mutable ReadStorageHandle readStorageHandle{};
    mutable gpu::KernelLaunchHandle kernelLaunchHandle{};

    gpu::GPUCPUReadStorageAdapter cpuReadStorage;
    gpu::GPUCPUMinhasherAdapter cpuMinhasher;
    ReadExtenderCpu cpuExtender{};
    
    GpuExtensionStepper() = default;

    GpuExtensionStepper(
        const gpu::GpuReadStorage& rs, 
        const gpu::GpuMinhasher& gpuMinhasher_,
        const CorrectionOptions& coropts,
        const GoodAlignmentProperties& gap,
        const cpu::QualityScoreConversion& qualityConversion_,
        int insertSize_,
        int insertSizeStddev_,
        int maxextensionPerStep_,
        cub::CachingDeviceAllocator& cubAllocator_
    ) : insertSize(insertSize_),
        insertSizeStddev(insertSizeStddev_),
        maxextensionPerStep(maxextensionPerStep_),
        cubAllocator(&cubAllocator_),
        gpuReadStorage(&rs),
        gpuMinhasher(&gpuMinhasher_),
        minhashHandle(gpuMinhasher->makeMinhasherHandle()),
        gpuReadHasher{*gpuMinhasher, minhashHandle},
        correctionOptions(&coropts),
        goodAlignmentProperties(&gap),
        qualityConversion(&qualityConversion_),
        readStorageHandle(gpuReadStorage->makeHandle()),
        cpuReadStorage(*gpuReadStorage, readStorageHandle, cudaStreamPerThread, *cubAllocator),
        cpuMinhasher(*gpuMinhasher, minhashHandle, cudaStreamPerThread, *cubAllocator),
        cpuExtender{
            insertSize,
            insertSizeStddev,
            maxextensionPerStep,
            gpuReadStorage->getSequenceLengthUpperBound(),
            cpuReadStorage,
            cpuMinhasher,
            *correctionOptions,
            *goodAlignmentProperties,
            qualityConversion
        }
        {

        cudaGetDevice(&deviceId); CUERR;

        kernelLaunchHandle = gpu::make_kernel_launch_handle(deviceId);
    }

    ~GpuExtensionStepper(){
        gpuMinhasher->destroyHandle(minhashHandle);
    }

    static constexpr int getNumRefinementIterations() noexcept{
        return 5;
    }

    //batches with less than getGpuBatchThreshold() tasks are handled on the cpu
    static constexpr int getGpuBatchThreshold() noexcept{
        return 1;
    }

    void setMaxExtensionPerStep(int e) noexcept{
        maxextensionPerStep = e;

        cpuExtender.setMaxExtensionPerStep(e);
    }

    void setMinCoverageForExtension(int c) noexcept{
        minCoverageForExtension = c;

        cpuExtender.setMinCoverageForExtension(c);
    }

    static ComputeType typeOfNextStep(BatchData& batchData){
        if(int(batchData.indicesOfActiveTasks.size()) < getGpuBatchThreshold()){
            return ComputeType::CPU;
        }

        switch(batchData.state){
            case BatchData::State::BeforePrepare: return ComputeType::CPU;
            case BatchData::State::BeforeHash: return ComputeType::CPU;
            case BatchData::State::BeforeRemoveIds: return ComputeType::CPU;
            case BatchData::State::BeforeComputePairFlags: return ComputeType::GPU;
            case BatchData::State::BeforeLoadCandidates: return ComputeType::GPU;
            case BatchData::State::BeforeEraseData: return ComputeType::GPU;
            case BatchData::State::BeforeAlignment: return ComputeType::GPU;
            case BatchData::State::BeforeAlignmentFilter: return ComputeType::GPU;
            case BatchData::State::BeforeMSA: return ComputeType::GPU;
            case BatchData::State::BeforeExtend: return ComputeType::GPU;
            case BatchData::State::BeforeCopyToHost: return ComputeType::CPU;
            case BatchData::State::BeforeUnpack: return ComputeType::CPU;
            case BatchData::State::Finished: return ComputeType::CPU;
            case BatchData::State::None: return ComputeType::CPU;
            default: return ComputeType::CPU;
        };
    }

    void performNextStep(BatchData& batchData) const{
        if(batchData.state == BatchData::State::BeforePrepare && int(batchData.indicesOfActiveTasks.size()) < getGpuBatchThreshold()){
            processOnCpu(batchData);
        }else{

            // if((batchData.state == BatchData::State::BeforeRemoveIds) || (batchData.state == BatchData::State::BeforeComputePairFlags)
            //     || (batchData.state == BatchData::State::BeforeMSA)){

            //     std::string what = "";
            //     if((batchData.state == BatchData::State::BeforeRemoveIds)){
            //         what = "raw";
            //     }
            //     if((batchData.state == BatchData::State::BeforeComputePairFlags)){
            //         what = "after remove";
            //     }
            //     if((batchData.state == BatchData::State::BeforeMSA)){
            //         what = "after filter";
            //     }
            //     cudaDeviceSynchronize(); CUERR;

            //     std::vector<read_number> readids(batchData.totalNumCandidates);
            //     cudaMemcpyAsync(
            //         readids.data(),
            //         batchData.d_candidateReadIds.data(),
            //         sizeof(read_number) * batchData.totalNumCandidates,
            //         D2H,
            //         cudaStreamPerThread
            //     );

            //     std::vector<int> nums(batchData.numTasks);
            //     cudaMemcpyAsync(
            //         nums.data(),
            //         batchData.d_numCandidatesPerAnchor.data(),
            //         sizeof(int) * batchData.numTasks,
            //         D2H,
            //         cudaStreamPerThread
            //     );

            //     std::vector<int> offsets(batchData.numTasks);
            //     cudaMemcpyAsync(
            //         offsets.data(),
            //         batchData.d_numCandidatesPerAnchorPrefixSum.data(),
            //         sizeof(int) * batchData.numTasks,
            //         D2H,
            //         cudaStreamPerThread
            //     );

            //     cudaDeviceSynchronize(); CUERR;

            //     for(int i = 0; i < batchData.numTasks; i++){
            //         const int index = batchData.indicesOfActiveTasks[i];
            //         const auto& task = batchData.tasks[index];

            //         if(task.myReadId == 0 && task.id == 3 && maxextensionPerStep == 6){
            //             if(batchData.state == BatchData::State::BeforeRemoveIds){
            //                 std::cerr << "Anchor: " << task.totalDecodedAnchors.back() << "\n";
            //             }
            //             std::cerr << "iteration " << task.iteration << ", candidates " << what << "\n";
            //             for(int k = 0; k < nums[i]; k++){
            //                 std::cerr << readids[offsets[i] + k] << " ";
            //             }
            //             std::cerr << "\n";
            //         }

            //     }
            //     cudaDeviceSynchronize(); CUERR;
            // }

            switch(batchData.state){
                case BatchData::State::BeforePrepare: prepareStep(batchData); break;
                case BatchData::State::BeforeHash: getCandidateReadIds(batchData); break;
                case BatchData::State::BeforeRemoveIds: removeUsedIdsAndMateIds(batchData); break;
                case BatchData::State::BeforeComputePairFlags: computePairFlags(batchData); break;
                case BatchData::State::BeforeLoadCandidates: loadCandidateSequenceData(batchData); break;
                case BatchData::State::BeforeEraseData: eraseDataOfRemovedMates(batchData); break;
                case BatchData::State::BeforeAlignment: calculateAlignments(batchData); break;
                case BatchData::State::BeforeAlignmentFilter: filterAlignments(batchData); break;
                case BatchData::State::BeforeMSA: computeMSAs(batchData); break;
                case BatchData::State::BeforeExtend: computeExtendedSequencesFromMSAs(batchData); break;
                case BatchData::State::BeforeCopyToHost: copyBuffersToHost(batchData); break;
                case BatchData::State::BeforeUnpack: unpackResults(batchData); break;
                case BatchData::State::Finished: break;
                case BatchData::State::None: break;
                default: break;
            };

        }
    }

    void prepareStep(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforePrepare);
        
        nvtx::push_range("prepareStep", 1);

        const int numActiveTasks = batchData.indicesOfActiveTasks.size();
        batchData.numTasks = numActiveTasks;

        batchData.h_numAnchors.resize(1);
        batchData.d_numAnchors.resize(1);
        batchData.h_numCandidates.resize(1);
        batchData.d_numCandidates.resize(1);
        batchData.d_numCandidates2.resize(1);
        batchData.h_numAnchorsWithRemovedMates.resize(1);

        batchData.h_anchorReadIds.resize(numActiveTasks);
        batchData.d_anchorReadIds.resize(numActiveTasks);
        batchData.h_mateReadIds.resize(numActiveTasks);
        batchData.d_mateReadIds.resize(numActiveTasks);
        
        batchData.h_subjectSequencesData.resize(batchData.encodedSequencePitchInInts * numActiveTasks);
        batchData.d_subjectSequencesData.resize(batchData.encodedSequencePitchInInts * numActiveTasks);
        batchData.h_subjectSequencesDataDecoded.resize(batchData.decodedSequencePitchInBytes * numActiveTasks);   

        batchData.h_anchorSequencesLength.resize(numActiveTasks);
        batchData.d_anchorSequencesLength.resize(numActiveTasks);

        batchData.h_anchorQualityScores.resize(batchData.qualityPitchInBytes * numActiveTasks);
        batchData.d_anchorQualityScores.resize(batchData.qualityPitchInBytes * numActiveTasks);

        batchData.d_anchormatedata.resize(numActiveTasks * batchData.encodedSequencePitchInInts);

        batchData.h_inputanchormatedata.resize(numActiveTasks * batchData.encodedSequencePitchInInts);
        batchData.d_inputanchormatedata.resize(numActiveTasks * batchData.encodedSequencePitchInInts);

        batchData.h_numCandidatesPerAnchor.resize(numActiveTasks);
        batchData.d_numCandidatesPerAnchor.resize(numActiveTasks);
        batchData.d_numCandidatesPerAnchor2.resize(numActiveTasks);
        batchData.h_numCandidatesPerAnchorPrefixSum.resize(numActiveTasks+1);
        batchData.d_numCandidatesPerAnchorPrefixSum.resize(numActiveTasks+1);
        batchData.d_numCandidatesPerAnchorPrefixSum2.resize(numActiveTasks+1);

        batchData.d_anchorIndicesWithRemovedMates.resize(numActiveTasks);

        batchData.h_numCandidatesPerAnchorPrefixSum[0] = 0;

        batchData.totalNumberOfUsedIds = 0;
        batchData.totalNumberOfFullyUsedIds = 0;

        for(int t = 0; t < numActiveTasks; t++){
            auto& task = batchData.tasks[batchData.indicesOfActiveTasks[t]];
            task.dataIsAvailable = false;

            batchData.h_anchorReadIds[t] = task.myReadId;
            batchData.h_mateReadIds[t] = task.mateReadId;
            batchData.totalNumberOfUsedIds += task.allUsedCandidateReadIdPairs.size();
            batchData.totalNumberOfFullyUsedIds += task.allFullyUsedCandidateReadIdPairs.size();

            #ifdef DO_REMOVE_USED_IDS_AND_MATE_IDS_ON_GPU

            std::copy(
                task.encodedMate.begin(),
                task.encodedMate.end(),
                batchData.h_inputanchormatedata.begin() + t * batchData.encodedSequencePitchInInts
            );

            #endif

            batchData.h_anchorSequencesLength[t] = task.currentAnchorLength;

            // std::copy(
            //     task.currentAnchor.begin(),
            //     task.currentAnchor.end(),
            //     batchData.h_subjectSequencesData.begin() + t * batchData.encodedSequencePitchInInts
            // );

            std::copy(
                task.totalDecodedAnchors.back().begin(),
                task.totalDecodedAnchors.back().end(),
                batchData.h_subjectSequencesDataDecoded.begin() + t * batchData.decodedSequencePitchInBytes
            );

            assert(batchData.h_anchorQualityScores.size() >= (t+1) * batchData.qualityPitchInBytes);

            std::copy(
                task.currentQualityScores.begin(),
                task.currentQualityScores.end(),
                batchData.h_anchorQualityScores.begin() + t * batchData.qualityPitchInBytes
            );
        }

        // helpers::call_copy_n_kernel(
        //     thrust::make_zip_iterator(thrust::make_tuple(
        //         //batchData.h_inputanchormatedata.data(),
        //         batchData.h_subjectSequencesData.data()
        //     )),
        //     batchData.numTasks * batchData.encodedSequencePitchInInts,
        //     thrust::make_zip_iterator(thrust::make_tuple(
        //         //batchData.d_inputanchormatedata.data(),
        //         batchData.d_subjectSequencesData.data()
        //     )),
        //     batchData.streams[0]
        // );

        #ifdef DO_REMOVE_USED_IDS_AND_MATE_IDS_ON_GPU
        cudaMemcpyAsync(
            batchData.d_inputanchormatedata.data(),
            batchData.h_inputanchormatedata.data(),
            sizeof(unsigned int) * batchData.encodedSequencePitchInInts * batchData.numTasks,
            H2D,
            batchData.streams[0]
        );
        #endif

        char* d_subjectSequencesDataDecoded = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_subjectSequencesDataDecoded, sizeof(char) * batchData.decodedSequencePitchInBytes * numActiveTasks, batchData.streams[0]); CUERR;

        cudaMemcpyAsync(
            d_subjectSequencesDataDecoded,
            batchData.h_subjectSequencesDataDecoded.data(),
            sizeof(char) * batchData.decodedSequencePitchInBytes * batchData.numTasks,
            H2D,
            batchData.streams[0]
        );

        helpers::call_copy_n_kernel(
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.h_anchorQualityScores.data()
            )),
            batchData.numTasks * batchData.qualityPitchInBytes,
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.d_anchorQualityScores.data()
            )),
            batchData.streams[0]
        );

        helpers::call_copy_n_kernel(
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.h_anchorSequencesLength.data(),
                batchData.h_anchorReadIds.data(),
                batchData.h_mateReadIds.data()
            )),
            batchData.numTasks,
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.d_anchorSequencesLength.data(),
                batchData.d_anchorReadIds.data(),
                batchData.d_mateReadIds.data()
            )),
            batchData.streams[0]
        );

        //2-bit encode anchorsequences
        helpers::lambda_kernel<<<SDIV(batchData.numTasks, (128 / 8)), 128, 0, batchData.streams[0]>>>(
            [
                decodedSequencePitchInBytes = batchData.decodedSequencePitchInBytes,
                encodedSequencePitchInInts = batchData.encodedSequencePitchInInts,
                numTasks = batchData.numTasks,
                encodedSequences = batchData.d_subjectSequencesData.data(),
                decodedSequences = d_subjectSequencesDataDecoded,
                sequenceLengths = batchData.d_anchorSequencesLength.data()
            ] __device__ (){

                auto group = cg::tiled_partition<8>(cg::this_thread_block());
                const int numGroups = (blockDim.x * gridDim.x) / group.size();
                const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / group.size();

                for(int a = groupId; a < numTasks; a += numGroups){
                    unsigned int* const out = encodedSequences + a * encodedSequencePitchInInts;
                    const char* const in = decodedSequences + a * decodedSequencePitchInBytes;
                    const int length = sequenceLengths[a];

                    const int nInts = SequenceHelpers::getEncodedNumInts2Bit(length);
                    constexpr int basesPerInt = SequenceHelpers::basesPerInt2Bit();

                    for(int i = group.thread_rank(); i < nInts; i += group.size()){
                        unsigned int data = 0;

                        const int loopend = min((i+1) * basesPerInt, length);
                        
                        for(int nucIndex = i * basesPerInt; nucIndex < loopend; nucIndex++){
                            switch(in[nucIndex]) {
                            case 'A':
                                data = (data << 2) | SequenceHelpers::encodedbaseA();
                                break;
                            case 'C':
                                data = (data << 2) | SequenceHelpers::encodedbaseC();
                                break;
                            case 'G':
                                data = (data << 2) | SequenceHelpers::encodedbaseG();
                                break;
                            case 'T':
                                data = (data << 2) | SequenceHelpers::encodedbaseT();
                                break;
                            default:
                                data = (data << 2) | SequenceHelpers::encodedbaseA();
                                break;
                            }
                        }

                        if(i == nInts-1){
                            //pack bits of last integer into higher order bits
                            int leftoverbits = 2 * (nInts * basesPerInt - length);
                            if(leftoverbits > 0){
                                data <<= leftoverbits;
                            }
                        }

                        out[i] = data;
                    }
                }
            }
        );

        cubAllocator->DeviceFree(d_subjectSequencesDataDecoded); CUERR;

        //#ifdef DO_REMOVE_USED_IDS_AND_MATE_IDS_ON_GPU
        #if 1
        if(1){
            batchData.h_usedReadIds.resize(batchData.totalNumberOfUsedIds);
            batchData.h_numUsedReadIdsPerAnchor.resize(batchData.numTasks);
            batchData.h_numUsedReadIdsPerAnchorPrefixSum.resize(batchData.numTasks);
            batchData.d_usedReadIds.resize(batchData.totalNumberOfUsedIds);
            batchData.d_numUsedReadIdsPerAnchor.resize(batchData.numTasks);
            batchData.d_numUsedReadIdsPerAnchorPrefixSum.resize(batchData.numTasks);

            batchData.h_fullyUsedReadIds.resize(batchData.totalNumberOfFullyUsedIds);
            batchData.h_numFullyUsedReadIdsPerAnchor.resize(batchData.numTasks);
            batchData.h_numFullyUsedReadIdsPerAnchorPrefixSum.resize(batchData.numTasks);
            batchData.d_fullyUsedReadIds.resize(batchData.totalNumberOfFullyUsedIds);
            batchData.d_numFullyUsedReadIdsPerAnchor.resize(batchData.numTasks);
            batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum.resize(batchData.numTasks);

            batchData.d_segmentIdsOfFullyUsedReadIds.resize(batchData.totalNumberOfFullyUsedIds);
            batchData.h_segmentIdsOfFullyUsedReadIds.resize(batchData.totalNumberOfFullyUsedIds);
            
            batchData.h_numUsedReadIdsPerAnchorPrefixSum[0] = 0;
            batchData.h_numFullyUsedReadIdsPerAnchorPrefixSum[0] = 0;

            auto segmentIdsIter = batchData.h_segmentIdsOfFullyUsedReadIds.begin();
            auto h_usedReadIdsIter = batchData.h_usedReadIds.begin();
            auto h_fullyUsedreadIdsIter = batchData.h_fullyUsedReadIds.begin();

            for(int i = 0; i < batchData.numTasks; i++){
                auto& task = batchData.tasks[batchData.indicesOfActiveTasks[i]];

                //handle fully used candidates
                const int numFullyUsedIds = task.allFullyUsedCandidateReadIdPairs.size();

                std::fill(segmentIdsIter, segmentIdsIter + numFullyUsedIds, i);
                segmentIdsIter += numFullyUsedIds;

                h_fullyUsedreadIdsIter = std::copy(
                    task.allFullyUsedCandidateReadIdPairs.begin(),
                    task.allFullyUsedCandidateReadIdPairs.end(),
                    h_fullyUsedreadIdsIter
                );
                batchData.h_numFullyUsedReadIdsPerAnchor[i] = numFullyUsedIds;

                if(i < batchData.numTasks - 1){
                    batchData.h_numFullyUsedReadIdsPerAnchorPrefixSum[i+1] 
                        = batchData.h_numFullyUsedReadIdsPerAnchorPrefixSum[i] + batchData.h_numFullyUsedReadIdsPerAnchor[i];
                }

                //handle used candidates
                const int numUsedIds = task.allUsedCandidateReadIdPairs.size();

                h_usedReadIdsIter = std::copy(
                    task.allUsedCandidateReadIdPairs.begin(),
                    task.allUsedCandidateReadIdPairs.end(),
                    h_usedReadIdsIter
                );
                batchData.h_numUsedReadIdsPerAnchor[i] = numUsedIds;

                if(i < batchData.numTasks - 1){
                    batchData.h_numUsedReadIdsPerAnchorPrefixSum[i+1] 
                        = batchData.h_numUsedReadIdsPerAnchorPrefixSum[i] + batchData.h_numUsedReadIdsPerAnchor[i];
                }
            }

            assert(std::distance(batchData.h_usedReadIds.data(), h_usedReadIdsIter) == batchData.totalNumberOfUsedIds);

            //task.allFullyUsedCandidateReadIdPairs

            // cudaMemcpyAsync(
            //     batchData.d_segmentIdsOfFullyUsedReadIds.data(),
            //     batchData.h_segmentIdsOfFullyUsedReadIds.data(),
            //     sizeof(int) * batchData.totalNumberOfUsedIds,
            //     H2D,
            //     batchData.streams[1]
            // ); CUERR;

            // cudaMemcpyAsync(
            //     batchData.d_usedReadIds.data(),
            //     batchData.h_usedReadIds.data(),
            //     sizeof(read_number) * batchData.totalNumberOfUsedIds,
            //     H2D,
            //     batchData.streams[1]
            // ); CUERR;

            helpers::call_copy_n_kernel(
                thrust::make_zip_iterator(thrust::make_tuple(
                    batchData.h_segmentIdsOfFullyUsedReadIds.data(),
                    batchData.h_fullyUsedReadIds.data()
                )),
                batchData.totalNumberOfFullyUsedIds,
                thrust::make_zip_iterator(thrust::make_tuple(
                    batchData.d_segmentIdsOfFullyUsedReadIds.data(),
                    batchData.d_fullyUsedReadIds.data()
                )),
                batchData.streams[1]
            );

            helpers::call_copy_n_kernel(
                thrust::make_zip_iterator(thrust::make_tuple(
                    batchData.h_usedReadIds.data()
                )),
                batchData.totalNumberOfUsedIds,
                thrust::make_zip_iterator(thrust::make_tuple(
                    batchData.d_usedReadIds.data()
                )),
                batchData.streams[1]
            );

            helpers::call_copy_n_kernel(
                thrust::make_zip_iterator(thrust::make_tuple(
                    batchData.h_numUsedReadIdsPerAnchorPrefixSum.data(),
                    batchData.h_numUsedReadIdsPerAnchor.data(),
                    batchData.h_numFullyUsedReadIdsPerAnchorPrefixSum.data(),
                    batchData.h_numFullyUsedReadIdsPerAnchor.data()
                )),
                batchData.numTasks,
                thrust::make_zip_iterator(thrust::make_tuple(
                    batchData.d_numUsedReadIdsPerAnchorPrefixSum.data(),
                    batchData.d_numUsedReadIdsPerAnchor.data(),
                    batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum.data(),
                    batchData.d_numFullyUsedReadIdsPerAnchor.data()
                )),
                batchData.streams[1]
            );
        }
        #endif

        batchData.setState(BatchData::State::BeforeHash);

        nvtx::pop_range();
    }

    void getCandidateReadIds(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforeHash);

        nvtx::push_range("getCandidateReadIds", 0);

        gpuReadHasher.getCandidateReadIds(batchData, batchData.streams[0]);
        batchData.setState(BatchData::State::BeforeRemoveIds);

        nvtx::pop_range();
    }

    void removeUsedIdsAndMateIds(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforeRemoveIds);
        
        nvtx::push_range("removeUsedIdsAndMateIds", 1);

        #ifdef DO_REMOVE_USED_IDS_AND_MATE_IDS_ON_GPU

        removeUsedIdsAndMateIds(batchData, batchData.streams[0], batchData.streams[1]);  

        #else 

        removeUsedIdsAndMateIdsCPU(batchData, batchData.streams[0], batchData.streams[1]);    
        
        #endif

        nvtx::pop_range();

        //removeUsedIdsAndMateIds is a compaction step. check early exit.
        if(batchData.totalNumCandidates == 0){
            batchData.setState(BatchData::State::Finished);
        }else{
            batchData.setState(BatchData::State::BeforeComputePairFlags);
        }
    }

    void computePairFlags(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforeComputePairFlags);

        nvtx::push_range("flagpairs", 7);

        //computePairFlagsCpu(batchData, batchData.streams[0]);
        computePairFlagsGpu(batchData, batchData.streams[0]);

        nvtx::pop_range();

        batchData.setState(BatchData::State::BeforeLoadCandidates);
    }

    void loadCandidateSequenceData(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforeLoadCandidates);

        nvtx::push_range("loadCandidateSequenceData", 2);

        loadCandidateSequenceData(batchData, batchData.streams[0]);

        nvtx::pop_range();

        batchData.setState(BatchData::State::BeforeEraseData);
    }

    void eraseDataOfRemovedMates(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforeEraseData);

        if(batchData.numTasksWithMateRemoved > 0){

            //for those tasks where a mate id has been removed, remove candidates whose sequence is equal to the mate sequence.
            //Sets batchData.totalNumCandidates to the sum of number of candidates for all tasks.

            nvtx::push_range("eraseDataOfRemovedMates", 3);

            eraseDataOfRemovedMates(batchData, batchData.streams[0]);

            nvtx::pop_range();

        }

        batchData.setState(BatchData::State::BeforeAlignment);
    }

    void calculateAlignments(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforeAlignment);

        nvtx::push_range("calculateAlignments", 4);

        calculateAlignments(batchData, batchData.streams[0]);

        nvtx::pop_range();

        batchData.setState(BatchData::State::BeforeAlignmentFilter);
    }

    void filterAlignments(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforeAlignmentFilter);

        nvtx::push_range("filterAlignments", 5);
    
        //Sets batchData.totalNumCandidates to the sum of number of candidates for all tasks.
        filterAlignments(batchData, batchData.streams[0]);

        nvtx::pop_range();

        //filterAlignments is a compaction step. check early exit.
        if(batchData.totalNumCandidates == 0){
            batchData.setState(BatchData::State::Finished);
        }else{
            batchData.setState(BatchData::State::BeforeMSA);
        }
    }

    void computeMSAs(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforeMSA);

        nvtx::push_range("computeMSAs", 6);

        //Sets batchData.totalNumCandidates to the sum of number of candidates for all tasks. (msa refinement can remove candidates)
        computeMSAs(batchData, batchData.streams[0], batchData.streams[1]);

        nvtx::pop_range();

        batchData.setState(BatchData::State::BeforeExtend);
    }

    void computeExtendedSequencesFromMSAs(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforeExtend);

        nvtx::push_range("computeExtendedSequences", 7);

        computeExtendedSequencesFromMSAs(batchData, batchData.streams[0]);

        nvtx::pop_range();

        batchData.setState(BatchData::State::BeforeCopyToHost);
    }

    void copyBuffersToHost(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforeCopyToHost);

        nvtx::push_range("copyBuffersToHost", 8);

        copyBuffersToHost(batchData, batchData.streams[0], batchData.streams[1]);

        cudaStreamSynchronize(batchData.streams[0]); CUERR;
        cudaStreamSynchronize(batchData.streams[1]); CUERR;

        nvtx::pop_range();

        batchData.setState(BatchData::State::BeforeUnpack);
    }

    void unpackResults(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforeUnpack);

        nvtx::push_range("unpackResults", 9);

        unpackResultsIntoTasks(batchData);

        nvtx::pop_range();

        if(!batchData.isEmpty()){
            batchData.setState(BatchData::State::BeforePrepare);
        }else{
            batchData.setState(BatchData::State::Finished);
        }
    }


    void process(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforePrepare);

        while(batchData.state != BatchData::State::Finished){
            performNextStep(batchData);
        }
    }

    void processOnCpu(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforePrepare);

        while(batchData.indicesOfActiveTasks.size() > 0){
            //perform one extension iteration for active tasks

            nvtx::push_range("doOneExtensionIteration", 5);

            cpuExtender.doOneExtensionIteration(batchData.tasks, batchData.indicesOfActiveTasks);

            nvtx::pop_range();

            //update list of active task indices

            batchData.indicesOfActiveTasks.erase(
                std::remove_if(
                    batchData.indicesOfActiveTasks.begin(), 
                    batchData.indicesOfActiveTasks.end(),
                    [&](int index){
                        return !batchData.tasks[index].isActive(insertSize, insertSizeStddev);
                    }
                ),
                batchData.indicesOfActiveTasks.end()
            );
        }

        batchData.state = BatchData::State::Finished;
    }

    void unpackResultsIntoTasks(BatchData& batchData) const{

        const int numActiveTasks = batchData.indicesOfActiveTasks.size();

        // for(int i = 0; i < numActiveTasks; i++){
        //     auto& task = batchData.tasks[batchData.indicesOfActiveTasks[i]];

        //     task.numRemainingCandidates = batchData.h_numCandidatesPerAnchor[i];

        //     if(task.numRemainingCandidates == 0){
        //         task.abort = true;
        //         task.abortReason = extension::AbortReason::NoPairedCandidatesAfterAlignment;
        //     }
        // }

        nvtx::push_range("Unpack gpu results", 6);

        for(int i = 0; i < numActiveTasks; i++){ 
            const int indexOfActiveTask = batchData.indicesOfActiveTasks[i];
            auto& task = batchData.tasks[indexOfActiveTask];

            // if(task.numRemainingCandidates == 0){
            //     continue;
            // }
            // assert(task.numRemainingCandidates > 0);

            task.numRemainingCandidates = batchData.h_numCandidatesPerAnchor[i];

            task.abortReason = batchData.h_abortReasons[i];
            if(task.abortReason == extension::AbortReason::None){
                task.mateHasBeenFound = batchData.h_outputMateHasBeenFound[i];

                if(!task.mateHasBeenFound){
                    const int newlength = batchData.h_outputAnchorLengths[i];

                    std::string newseq(batchData.h_outputAnchors.data() + i * batchData.outputAnchorPitchInBytes, newlength);
                    std::string newq(batchData.h_outputAnchorQualities.data() + i * batchData.outputAnchorQualityPitchInBytes, newlength);

                    task.currentAnchorLength = newlength;
                    task.accumExtensionLengths = batchData.h_accumExtensionsLengths[i];
                    task.totalDecodedAnchors.emplace_back(std::move(newseq));
                    task.totalAnchorQualityScores.emplace_back(std::move(newq));
                    task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);

                    task.currentQualityScores = task.totalAnchorQualityScores.back(); 
                    
                }else{
                    const int sizeofGap = batchData.h_sizeOfGapToMate[i];
                    if(sizeofGap == 0){
                        task.accumExtensionLengths = batchData.h_accumExtensionsLengths[i];
                        task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);
                        task.totalDecodedAnchors.emplace_back(task.decodedMateRevC);
                        task.totalAnchorQualityScores.emplace_back(task.mateQualityScoresReversed);
                    }else{
                        const int newlength = batchData.h_outputAnchorLengths[i];

                        std::string newseq(batchData.h_outputAnchors.data() + i * batchData.outputAnchorPitchInBytes, newlength);
                        std::string newq(batchData.h_outputAnchorQualities.data() + i * batchData.outputAnchorQualityPitchInBytes, newlength);

                        task.accumExtensionLengths = batchData.h_accumExtensionsLengths[i];
                        task.totalDecodedAnchors.emplace_back(std::move(newseq));
                        task.totalAnchorQualityScores.emplace_back(std::move(newq));
                        task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);

                        task.accumExtensionLengths += newlength;
                        task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);
                        task.totalDecodedAnchors.emplace_back(task.decodedMateRevC);
                        task.totalAnchorQualityScores.emplace_back(task.mateQualityScoresReversed);
                    }
                }
            }

            task.abort = task.abortReason != extension::AbortReason::None;
        }

        nvtx::pop_range();

        // nvtx::push_range("Encode remaining anchors", 6);

        // for(int i = 0; i < numActiveTasks; i++){ 
        //     const int indexOfActiveTask = batchData.indicesOfActiveTasks[i];
        //     auto& task = batchData.tasks[indexOfActiveTask];

        //     if(task.numRemainingCandidates == 0){
        //         continue;
        //     }
        //     assert(task.numRemainingCandidates > 0);

        //     if(task.abortReason == extension::AbortReason::None){
        //         if(!task.mateHasBeenFound){
        //             const int numInts = SequenceHelpers::getEncodedNumInts2Bit(task.currentAnchorLength);
        //             task.currentAnchor.resize(numInts);

        //             SequenceHelpers::encodeSequence2Bit(
        //                 task.currentAnchor.data(), 
        //                 task.totalDecodedAnchors.back().data(), 
        //                 task.currentAnchorLength
        //             );
        //         }
        //     }
        // }

        // nvtx::pop_range();

        assert(batchData.tasks.size() / 4 == batchData.numReadPairs);

        handleEarlyExitOfTasks4(batchData.tasks, batchData.indicesOfActiveTasks);

        // for(int i = 0; i < int(batchData.tasks.size()); i++){
        //     const auto& task = batchData.tasks[i];

        //     std::cerr << "i = " <<i << "\n";
        //     std::cerr << "id " << task.id << "\n";
        //     std::cerr << "numRemainingCandidates " << task.numRemainingCandidates << "\n";
        //     std::cerr << "iteration " << task.iteration << "\n";
        //     std::cerr << "mateHasBeenFound " << task.mateHasBeenFound << "\n";
        //     std::cerr << "abort " << task.abort << "\n";
        //     std::cerr << "abortReason " << to_string(task.abortReason) << "\n";
        // }


        /*
            update book-keeping of used candidates
        */  
        nvtx::push_range("usedcandidates", 6);
        for(int i = 0; i < numActiveTasks; i++){
            auto& task = batchData.tasks[batchData.indicesOfActiveTasks[i]];

            const int numCandidates = batchData.h_numCandidatesPerAnchor[i];
            const int offset = batchData.h_numCandidatesPerAnchorPrefixSum[i];
            const read_number* ids = &batchData.h_candidateReadIds[offset];
            const bool* isFullyUsed = &batchData.h_isFullyUsedCandidate[offset];

            std::vector<read_number> tmp(task.allUsedCandidateReadIdPairs.size() + numCandidates);
            auto tmp_end = std::set_union(
                task.allUsedCandidateReadIdPairs.begin(),
                task.allUsedCandidateReadIdPairs.end(),
                ids,
                ids + numCandidates,
                tmp.begin()
            );

            tmp.erase(tmp_end, tmp.end());
            std::swap(task.allUsedCandidateReadIdPairs, tmp);

            std::vector<read_number> fullyUsedIds(numCandidates);
            int numFullyUsed = 0;
            for(int i = 0; i < numCandidates; i++){
                if(isFullyUsed[i]){
                    fullyUsedIds[numFullyUsed++] = ids[i];
                }
            }
            fullyUsedIds.erase(fullyUsedIds.begin() + numFullyUsed, fullyUsedIds.end());


            std::vector<read_number> tmp2(task.allFullyUsedCandidateReadIdPairs.size() + numFullyUsed);
            auto tmp2_end = std::set_union(
                task.allFullyUsedCandidateReadIdPairs.begin(),
                task.allFullyUsedCandidateReadIdPairs.end(),
                fullyUsedIds.begin(),
                fullyUsedIds.end(),
                tmp2.begin()
            );

            tmp2.erase(tmp2_end, tmp2.end());
            std::swap(task.allFullyUsedCandidateReadIdPairs, tmp2);

            assert(task.allFullyUsedCandidateReadIdPairs.size() <= task.allUsedCandidateReadIdPairs.size());

            task.iteration++;
        }

        nvtx::pop_range();
        
        //update list of active task indices

       
        batchData.indicesOfActiveTasks.erase(
            std::remove_if(
                batchData.indicesOfActiveTasks.begin(), 
                batchData.indicesOfActiveTasks.end(),
                [&](int index){
                    return !batchData.tasks[index].isActive(insertSize, insertSizeStddev);
                }
            ),
            batchData.indicesOfActiveTasks.end()
        );
    }


    std::vector<extension::ExtendResult> constructResults(BatchData& batchData) const{
        std::vector<extension::ExtendResult> extendResults;
        extendResults.reserve(batchData.tasks.size());

        for(const auto& task : batchData.tasks){

            //std::cerr << task.allFullyUsedCandidateReadIdPairs.size() << " / " << task.allUsedCandidateReadIdPairs.size() << "\n";

            extension::ExtendResult extendResult;
            extendResult.direction = task.direction;
            extendResult.numIterations = task.iteration;
            extendResult.aborted = task.abort;
            extendResult.abortReason = task.abortReason;
            extendResult.readId1 = task.myReadId;
            extendResult.readId2 = task.mateReadId;
            extendResult.originalLength = task.myLength;
            extendResult.originalMateLength = task.mateLength;
            extendResult.read1begin = 0;

            //construct extended read
            //build msa of all saved totalDecodedAnchors[0]

            const int numsteps = task.totalDecodedAnchors.size();

            int maxlen = 0;
            for(const auto& s: task.totalDecodedAnchors){
                const int len = s.length();
                if(len > maxlen){
                    maxlen = len;
                }
            }

            const std::string& decodedAnchor = task.totalDecodedAnchors[0];
            const std::string& anchorQuality = task.totalAnchorQualityScores[0];

            const std::vector<int> shifts(task.totalAnchorBeginInExtendedRead.begin() + 1, task.totalAnchorBeginInExtendedRead.end());
            std::vector<float> initialWeights(numsteps-1, 1.0f);


            std::vector<char> stepstrings(maxlen * (numsteps-1), '\0');
            std::vector<char> stepqualities(maxlen * (numsteps-1), '\0');
            std::vector<int> stepstringlengths(numsteps-1);
            for(int c = 1; c < numsteps; c++){
                std::copy(
                    task.totalDecodedAnchors[c].begin(),
                    task.totalDecodedAnchors[c].end(),
                    stepstrings.begin() + (c-1) * maxlen
                );
                assert(task.totalAnchorQualityScores[c].size() <= maxlen);
                std::copy(
                    task.totalAnchorQualityScores[c].begin(),
                    task.totalAnchorQualityScores[c].end(),
                    stepqualities.begin() + (c-1) * maxlen
                );
                stepstringlengths[c-1] = task.totalDecodedAnchors[c].size();
            }

            MultipleSequenceAlignment::InputData msaInput;
            msaInput.useQualityScores = false;
            msaInput.subjectLength = decodedAnchor.length();
            msaInput.nCandidates = numsteps-1;
            msaInput.candidatesPitch = maxlen;
            msaInput.candidateQualitiesPitch = maxlen;
            msaInput.subject = decodedAnchor.c_str();
            msaInput.candidates = stepstrings.data();
            msaInput.subjectQualities = anchorQuality.data();
            msaInput.candidateQualities = stepqualities.data();
            msaInput.candidateLengths = stepstringlengths.data();
            msaInput.candidateShifts = shifts.data();
            msaInput.candidateDefaultWeightFactors = initialWeights.data();

            MultipleSequenceAlignment msa(qualityConversion);

            msa.build(msaInput);

            //msa.print(std::cerr);

            std::string extendedRead(msa.consensus.begin(), msa.consensus.end());
            std::string extendedReadQuality(msa.consensus.size(), '\0');
            std::transform(msa.support.begin(), msa.support.end(), extendedReadQuality.begin(),
                [](const float f){
                    return getQualityChar(f);
                }
            );

            std::copy(decodedAnchor.begin(), decodedAnchor.end(), extendedRead.begin());
            std::copy(anchorQuality.begin(), anchorQuality.end(), extendedReadQuality.begin());


            //alternative extendedRead. no msa + consensus, just concat

            // std::string extendedReadTmp;

            // if(numsteps > 1){
            //     extendedReadTmp.resize(shifts.back() + stepstringlengths.back(), '\0');

            //     auto b = std::copy(decodedAnchor.begin(), decodedAnchor.end(), extendedReadTmp.begin());
            //     for(int i = 0; i < numsteps - 1; i++){
            //         const int currentEnd = std::distance(extendedReadTmp.begin(), b);

            //         const int nextLength = stepstringlengths[i];
            //         const int nextBegin = shifts[i];

            //         if(nextBegin + nextLength > currentEnd){
            //             const int copybegin = currentEnd - nextBegin;
            //             b = std::copy(
            //                 task.totalDecodedAnchors[i+1].begin() + copybegin,
            //                 task.totalDecodedAnchors[i+1].end(),
            //                 b
            //             );
            //         }
            //     }

            //     assert(b == extendedReadTmp.end());

            //     // if(extendedReadTmp != extendedRead){
            //     //     std::cerr << "old: " << extendedRead << "\n";
            //     //     std::cerr << "new: " << extendedReadTmp << "\n";
            //     // }
            // }else{
            //     extendedReadTmp = decodedAnchor;
            // }

            
            //std::swap(extendedReadTmp, extendedRead);




            

            if(task.mateHasBeenFound){
                //std::cerr << "copy " << task.decodedMateRevC << " to end of consensus " << task.myReadId << "\n";
                std::copy(
                    task.decodedMateRevC.begin(),
                    task.decodedMateRevC.end(),
                    extendedRead.begin() + extendedRead.length() - task.decodedMateRevC.length()
                );

                std::copy(
                    task.mateQualityScoresReversed.begin(),
                    task.mateQualityScoresReversed.end(),
                    extendedReadQuality.begin() + extendedReadQuality.length() - task.decodedMateRevC.length()
                );

                extendResult.read2begin = extendedRead.length() - task.decodedMateRevC.length();
            }else{
                extendResult.read2begin = -1;
            }

            extendResult.extendedRead = std::move(extendedRead);
            extendResult.qualityScores = std::move(extendedReadQuality);

            extendResult.mateHasBeenFound = task.mateHasBeenFound;

            extendResults.emplace_back(std::move(extendResult));
        }

        #if 0

        std::vector<extension::ExtendResult> extendResultsCombined = ReadExtenderBase::combinePairedEndDirectionResults(
            extendResults,
            insertSize,
            insertSizeStddev
        );

        #else

        std::vector<extension::ExtendResult> extendResultsCombined = extension::combinePairedEndDirectionResults4(
            extendResults,
            insertSize,
            insertSizeStddev
        );

        #endif

        return extendResultsCombined;
    }

    void computePairFlagsGpu(BatchData& batchData, cudaStream_t stream) const{
        DEBUGDEVICESYNC

        batchData.d_isPairedCandidate.resize(batchData.totalNumCandidates);

        helpers::call_fill_kernel_async(batchData.d_isPairedCandidate.data(), batchData.totalNumCandidates, false, stream);

        DEBUGDEVICESYNC

        batchData.h_firstTasksOfPairsToCheck.resize(batchData.numTasks);
        int numChecks = 0;

        for(int first = 0, second = 1; second < batchData.numTasks; ){
            const int taskindex1 = batchData.indicesOfActiveTasks[first];
            const int taskindex2 = batchData.indicesOfActiveTasks[second];

            const bool areConsecutiveTasks = batchData.tasks[taskindex1].id + 1 == batchData.tasks[taskindex2].id;
            const bool arePairedTasks = (batchData.tasks[taskindex1].id % 2) + 1 == (batchData.tasks[taskindex2].id % 2);

            if(areConsecutiveTasks && arePairedTasks){
                batchData.h_firstTasksOfPairsToCheck[numChecks] = first;
                numChecks++;
                
                first += 2; second += 2;
            }else{
                first += 1; second += 1;
            }
        }

        if(numChecks > 0){

            int* d_firstTasksOfPairsToCheck = nullptr;
            cubAllocator->DeviceAllocate((void**)&d_firstTasksOfPairsToCheck, sizeof(int) * numChecks); CUERR;

            DEBUGDEVICESYNC

            // int* d_status = nullptr;
            // //cubAllocator->DeviceAllocate((void**)&d_status, sizeof(int) * numChecks); CUERR;
            // cudaMallocHost(&d_status, sizeof(int) * numChecks); CUERR;

            // std::fill(d_status, d_status + numChecks, 0);

            cudaMemcpyAsync(
                d_firstTasksOfPairsToCheck,
                batchData.h_firstTasksOfPairsToCheck.data(),
                sizeof(int) * numChecks,
                H2D,
                stream
            ); CUERR;

            DEBUGDEVICESYNC

            // {
            //     cudaDeviceSynchronize(); CUERR;

            //     std::cerr << "numChecks = " << numChecks << "\n";
            //     for(int i = 0; i < numChecks; i++){
            //         std::cerr << batchData.h_firstTasksOfPairsToCheck[i] << " ";
            //     }
            //     std::cerr << "\n";

            //     // std::vector<read_number> readids(batchData.totalNumCandidates);
            //     // cudaMemcpyAsync(
            //     //     readids.data(),
            //     //     batchData.d_candidateReadIds.data(),
            //     //     sizeof(read_number) * batchData.totalNumCandidates,
            //     //     D2H,
            //     //     stream
            //     // );

            //     // std::vector<std::uint8_t> consensusEncoded(batchData.msaColumnPitchInElements * batchData.numTasks);
            //     // cudaMemcpyAsync(
            //     //     consensusEncoded.data(),
            //     //     batchData.d_consensusEncoded.data(),
            //     //     sizeof(std::uint8_t) * batchData.msaColumnPitchInElements * batchData.numTasks,
            //     //     D2H,
            //     //     stream
            //     // );
            //     std::cerr << "batchData.numTasks = " << batchData.numTasks << "\n";
            //     std::vector<int> nums(batchData.numTasks);
            //     cudaMemcpyAsync(
            //         nums.data(),
            //         batchData.d_numCandidatesPerAnchor.data(),
            //         sizeof(int) * batchData.numTasks,
            //         D2H,
            //         stream
            //     );

            //     std::vector<int> offsets(batchData.numTasks);
            //     cudaMemcpyAsync(
            //         offsets.data(),
            //         batchData.d_numCandidatesPerAnchorPrefixSum.data(),
            //         sizeof(int) * batchData.numTasks,
            //         D2H,
            //         stream
            //     );

            //     std::vector<int> usednums(batchData.numTasks);
            //     cudaMemcpyAsync(
            //         usednums.data(),
            //         batchData.d_numUsedReadIdsPerAnchor.data(),
            //         sizeof(int) * batchData.numTasks,
            //         D2H,
            //         stream
            //     );

            //     std::vector<int> usedoffsets(batchData.numTasks);
            //     cudaMemcpyAsync(
            //         usedoffsets.data(),
            //         batchData.d_numUsedReadIdsPerAnchorPrefixSum.data(),
            //         sizeof(int) * batchData.numTasks,
            //         D2H,
            //         stream
            //     );

            //     cudaDeviceSynchronize(); CUERR;

            //     std::cerr << "nums:\n";
            //     for(int i = 0; i < batchData.numTasks; i++){
            //         std::cerr << nums[i] << " ";
            //     }
            //     std::cerr << "\n";

            //     std::cerr << "offsets:\n";
            //     for(int i = 0; i < batchData.numTasks; i++){
            //         std::cerr << offsets[i] << " ";
            //     }
            //     std::cerr << "\n";

            //     std::cerr << "usednums:\n";
            //     for(int i = 0; i < batchData.numTasks; i++){
            //         std::cerr << usednums[i] << " ";
            //     }
            //     std::cerr << "\n";

            //     std::cerr << "usedoffsets:\n";
            //     for(int i = 0; i < batchData.numTasks; i++){
            //         std::cerr << usedoffsets[i] << " ";
            //     }
            //     std::cerr << "\n";

            //     // std::vector<char> consensusDecoded(consensusEncoded.size());
            //     // std::transform(consensusEncoded.begin(), consensusEncoded.end(), consensusDecoded.begin(),
            //     //     [](const std::uint8_t encoded){
            //     //         char decoded = 'F';
            //     //         if(encoded == std::uint8_t{0}){
            //     //             decoded = 'A';
            //     //         }else if(encoded == std::uint8_t{1}){
            //     //             decoded = 'C';
            //     //         }else if(encoded == std::uint8_t{2}){
            //     //             decoded = 'G';
            //     //         }else if(encoded == std::uint8_t{3}){
            //     //             decoded = 'T';
            //     //         }
            //     //         return decoded;
            //     //     }
            //     // );

            //     // for(int i = 0; i < batchData.numTasks; i++){
            //     //     const int index = batchData.indicesOfActiveTasks[i];
            //     //     const auto& task = batchData.tasks[index];

            //     //     if(task.myReadId == 0 && task.id == 3 && maxextensionPerStep == 6){
            //     //         std::cerr << "task id " << task.id << " myReadId " << task.myReadId << "\n";
            //     //         std::cerr << "candidates\n";
            //     //         for(int k = 0; k < nums[i]; k++){
            //     //             std::cerr << readids[offsets[i] + k] << " ";
            //     //         }
            //     //         std::cerr << "\n";

            //     //         std::cerr << "consensus\n";
            //     //         for(int k = 0; k < batchData.msaColumnPitchInElements; k++){
            //     //             std::cerr << consensusDecoded[i * batchData.msaColumnPitchInElements + k];
            //     //         }
            //     //         std::cerr << "\n";
            //     //     }

            //     // }

            //     std::cerr << "batchData.d_numCandidatesPerAnchor.size(): " << batchData.d_numCandidatesPerAnchor.size() << "\n";
            //     std::cerr << "batchData.d_numCandidatesPerAnchorPrefixSum.size(): " << batchData.d_numCandidatesPerAnchorPrefixSum.size() << "\n";
            //     std::cerr << "batchData.d_candidateReadIds.size(): " << batchData.d_candidateReadIds.size() << "\n";
            //     std::cerr << "batchData.d_numUsedReadIdsPerAnchor.size(): " << batchData.d_numUsedReadIdsPerAnchor.size() << "\n";
            //     std::cerr << "batchData.d_numUsedReadIdsPerAnchorPrefixSum.size(): " << batchData.d_numUsedReadIdsPerAnchorPrefixSum.size() << "\n";
            //     std::cerr << "batchData.d_usedReadIds.size(): " << batchData.d_usedReadIds.size() << "\n";
            //     std::cerr << "batchData.d_isPairedCandidate.size(): " << batchData.d_isPairedCandidate.size() << "\n";

            //     std::cerr << "batchData.d_numCandidatesPerAnchor.data(): " << batchData.d_numCandidatesPerAnchor.data() << "\n";
            //     std::cerr << "batchData.d_numCandidatesPerAnchorPrefixSum.data(): " << batchData.d_numCandidatesPerAnchorPrefixSum.data() << "\n";
            //     std::cerr << "batchData.d_candidateReadIds.data(): " << batchData.d_candidateReadIds.data() << "\n";
            //     std::cerr << "batchData.d_numUsedReadIdsPerAnchor.data(): " << batchData.d_numUsedReadIdsPerAnchor.data() << "\n";
            //     std::cerr << "batchData.d_numUsedReadIdsPerAnchorPrefixSum.data(): " << batchData.d_numUsedReadIdsPerAnchorPrefixSum.data() << "\n";
            //     std::cerr << "batchData.d_usedReadIds.data(): " << batchData.d_usedReadIds.data() << "\n";
            //     std::cerr << "batchData.d_isPairedCandidate.data(): " << batchData.d_isPairedCandidate.data() << "\n";

                

            //     cudaDeviceSynchronize(); CUERR;
            // }
            

            dim3 block = 128;
            dim3 grid = numChecks;

            helpers::lambda_kernel<<<grid, block, 0, stream>>>(
                [
                    numChecks,
                    d_firstTasksOfPairsToCheck,
                    d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.data(),
                    d_numCandidatesPerAnchorPrefixSum = batchData.d_numCandidatesPerAnchorPrefixSum.data(), // numTasks + 1
                    d_candidateReadIds = batchData.d_candidateReadIds.data(),
                    d_numUsedReadIdsPerAnchor = batchData.d_numUsedReadIdsPerAnchor.data(),
                    d_numUsedReadIdsPerAnchorPrefixSum = batchData.d_numUsedReadIdsPerAnchorPrefixSum.data(), // numTasks
                    d_usedReadIds = batchData.d_usedReadIds.data(),
                    d_isPairedCandidate = batchData.d_isPairedCandidate.data()
                ] __device__ (){

                    constexpr int numSharedElements = 1024;

                    __shared__ read_number sharedElements[numSharedElements];

                    //search elements of array1 in array2. if found, set output element to true
                    //array1 and array2 must be sorted
                    auto process = [&](
                        const read_number* array1,
                        int numElements1,
                        const read_number* array2,
                        int numElements2,
                        bool* output
                    ){
                        const int numIterations = SDIV(numElements2, numSharedElements);

                        for(int iteration = 0; iteration < numIterations; iteration++){

                            const int begin = iteration * numSharedElements;
                            const int end = min((iteration+1) * numSharedElements, numElements2);
                            const int num = end - begin;

                            for(int i = threadIdx.x; i < num; i += blockDim.x){
                                sharedElements[i] = array2[begin + i];
                            }

                            __syncthreads();

                            //TODO in iteration > 0, we may skip elements at the beginning of first range

                            for(int i = threadIdx.x; i < numElements1; i += blockDim.x){
                                if(!output[i]){
                                    const read_number readId = array1[i];
                                    const read_number readIdToFind = readId % 2 == 0 ? readId + 1 : readId - 1;

                                    const bool found = thrust::binary_search(thrust::seq, sharedElements, sharedElements + num, readIdToFind);
                                    if(found){
                                        output[i] = true;
                                    }
                                }
                            }

                            __syncthreads();
                        }
                    };

                    for(int a = blockIdx.x; a < numChecks; a += gridDim.x){
                        const int firstTask = d_firstTasksOfPairsToCheck[a];
                        //const int secondTask = firstTask + 1;

                        //check for pairs in current candidates

                        const int rangeBegin = d_numCandidatesPerAnchorPrefixSum[firstTask];                        
                        const int rangeMid = d_numCandidatesPerAnchorPrefixSum[firstTask + 1];
                        const int rangeEnd = d_numCandidatesPerAnchorPrefixSum[firstTask + 2];

                        process(
                            d_candidateReadIds + rangeBegin,
                            rangeMid - rangeBegin,
                            d_candidateReadIds + rangeMid,
                            rangeEnd - rangeMid,
                            d_isPairedCandidate + rangeBegin
                        );

                        process(
                            d_candidateReadIds + rangeMid,
                            rangeEnd - rangeMid,
                            d_candidateReadIds + rangeBegin,
                            rangeMid - rangeBegin,
                            d_isPairedCandidate + rangeMid
                        );

                        //check for pairs in candidates of previous extension iterations

                        const int usedRangeBegin = d_numUsedReadIdsPerAnchorPrefixSum[firstTask];                        
                        const int usedRangeMid = d_numUsedReadIdsPerAnchorPrefixSum[firstTask + 1];
                        const int usedRangeEnd = usedRangeMid + d_numUsedReadIdsPerAnchor[firstTask + 1];

                        process(
                            d_candidateReadIds + rangeBegin,
                            rangeMid - rangeBegin,
                            d_usedReadIds + usedRangeMid,
                            usedRangeEnd - usedRangeMid,
                            d_isPairedCandidate + rangeBegin
                        );

                        process(
                            d_candidateReadIds + rangeMid,
                            rangeEnd - rangeMid,
                            d_usedReadIds + usedRangeBegin,
                            usedRangeMid - usedRangeBegin,
                            d_isPairedCandidate + rangeMid
                        );
                    }
                }
            ); CUERR;

            DEBUGDEVICESYNC

            cubAllocator->DeviceFree(d_firstTasksOfPairsToCheck); CUERR;

            DEBUGDEVICESYNC

        }

    }


    void computePairFlagsCpu(BatchData& batchData, cudaStream_t stream) const{
        //computed in removeUsedIdsAndMateIdsCPU

        #ifdef DO_REMOVE_USED_IDS_AND_MATE_IDS_ON_GPU

        batchData.h_candidateReadIds.resize(batchData.totalNumCandidates);
        
        cudaMemcpyAsync(
            batchData.h_numCandidatesPerAnchorPrefixSum.data(),
            batchData.d_numCandidatesPerAnchorPrefixSum.data(),
            sizeof(int) * (batchData.numTasks + 1),
            D2H,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            batchData.h_candidateReadIds.data(),
            batchData.d_candidateReadIds.data(),
            sizeof(read_number) * batchData.totalNumCandidates,
            D2H,
            stream
        ); CUERR;

        #endif

        batchData.h_isPairedCandidate.resize(batchData.totalNumCandidates);
        batchData.d_isPairedCandidate.resize(batchData.totalNumCandidates);

        std::fill(batchData.h_isPairedCandidate.begin(), batchData.h_isPairedCandidate.end(), false);

        #ifdef DO_REMOVE_USED_IDS_AND_MATE_IDS_ON_GPU
        cudaStreamSynchronize(stream); CUERR;
        #endif


        for(int first = 0, second = 1; second < batchData.numTasks; ){
            const int taskindex1 = batchData.indicesOfActiveTasks[first];
            const int taskindex2 = batchData.indicesOfActiveTasks[second];

            const bool areConsecutiveTasks = batchData.tasks[taskindex1].id + 1 == batchData.tasks[taskindex2].id;
            const bool arePairedTasks = (batchData.tasks[taskindex1].id % 2) + 1 == (batchData.tasks[taskindex2].id % 2);

            if(areConsecutiveTasks && arePairedTasks){
                const int begin1 = batchData.h_numCandidatesPerAnchorPrefixSum[first];
                const int end1 = batchData.h_numCandidatesPerAnchorPrefixSum[second];
                const int begin2 = batchData.h_numCandidatesPerAnchorPrefixSum[second];
                const int end2 = batchData.h_numCandidatesPerAnchorPrefixSum[second + 1];

                // assert(std::is_sorted(pairIds + begin1, pairIds + end1));
                // assert(std::is_sorted(pairIds + begin2, pairIds + end2));

                std::vector<int> pairedPositions(std::min(end1-begin1, end2-begin2));
                std::vector<int> pairedPositions2(std::min(end1-begin1, end2-begin2));

                auto endIters = findPositionsOfPairedReadIds(
                    batchData.h_candidateReadIds.data() + begin1,
                    batchData.h_candidateReadIds.data() + end1,
                    batchData.h_candidateReadIds.data() + begin2,
                    batchData.h_candidateReadIds.data() + end2,
                    pairedPositions.begin(),
                    pairedPositions2.begin()
                );

                pairedPositions.erase(endIters.first, pairedPositions.end());
                pairedPositions2.erase(endIters.second, pairedPositions2.end());
                for(auto i : pairedPositions){
                    batchData.h_isPairedCandidate[begin1 + i] = true;
                }
                for(auto i : pairedPositions2){
                    batchData.h_isPairedCandidate[begin2 + i] = true;
                }
                
                first += 2; second += 2;
            }else{
                first += 1; second += 1;
            }
        }
        
        cudaMemcpyAsync(
            batchData.d_isPairedCandidate.data(),
            batchData.h_isPairedCandidate.data(),
            sizeof(bool) * batchData.totalNumCandidates,
            H2D,
            stream
        ); CUERR;
    }


    void removeUsedIdsAndMateIdsCPU(BatchData& batchData, cudaStream_t firstStream, cudaStream_t secondStream) const{
        batchData.h_candidateReadIds.resize(batchData.totalNumCandidates);
        batchData.h_numCandidatesPerAnchor.resize(batchData.numTasks);
        batchData.h_numCandidatesPerAnchorPrefixSum.resize(batchData.numTasks + 1);
        
        cudaMemcpyAsync(
            batchData.h_candidateReadIds.data(),
            batchData.d_candidateReadIds.data(),
            sizeof(read_number) * batchData.totalNumCandidates,
            D2H,
            firstStream
        );
        
        // cudaMemcpyAsync(
        //     batchData.h_numCandidatesPerAnchor.data(),
        //     batchData.d_numCandidatesPerAnchor.data(),
        //     sizeof(int) * (batchData.numTasks),
        //     D2H,
        //     firstStream
        // );
        // cudaMemcpyAsync(
        //     batchData.h_numCandidatesPerAnchorPrefixSum.data(),
        //     batchData.d_numCandidatesPerAnchorPrefixSum.data(),
        //     sizeof(int) * (batchData.numTasks + 1),
        //     D2H,
        //     firstStream
        // );
        helpers::call_copy_n_kernel(
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1,
                batchData.d_numCandidatesPerAnchor.data()
            )),
            batchData.numTasks,
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.h_numCandidatesPerAnchorPrefixSum.data() + 1,
                batchData.h_numCandidatesPerAnchor.data()
            )),
            firstStream
        ); 
        cudaStreamSynchronize(firstStream); CUERR;

        std::vector<read_number> candidateReadIdsTmp(batchData.totalNumCandidates);
        std::vector<int> numCandidatesPerAnchorTmp(batchData.numTasks);
        std::vector<int> numCandidatesPerAnchorPrefixSumTmp(batchData.numTasks + 1);

        read_number* destids = candidateReadIdsTmp.data();
        batchData.h_numCandidatesPerAnchorPrefixSum[0] = 0;
        numCandidatesPerAnchorPrefixSumTmp[0] = 0;
        batchData.numTasksWithMateRemoved = 0;

        /*
            Remove anchor ids and mate ids from candidates
        */

        for(int k = 0; k < batchData.numTasks; k++){
            auto& task = batchData.tasks[batchData.indicesOfActiveTasks[k]];

            int num = batchData.h_numCandidatesPerAnchor[k];
            const int offset = batchData.h_numCandidatesPerAnchorPrefixSum[k];
            read_number* myIds = batchData.h_candidateReadIds.data() + offset;

            auto readIdPos = std::lower_bound(
                myIds,                                            
                myIds + num,
                task.myReadId
            );

            if(readIdPos != myIds + num && *readIdPos == task.myReadId){
                std::copy(readIdPos + 1, myIds + num, readIdPos);
                num--;
            }

            if(task.pairedEnd){                

                //remove mate of input from candidate list
                auto mateReadIdPos = std::lower_bound(
                    myIds,                                            
                    myIds + num,
                    task.mateReadId
                );

                if(mateReadIdPos != myIds + num && *mateReadIdPos == task.mateReadId){
                    numCandidatesPerAnchorTmp[k] = num - 1;
                    destids = std::copy(myIds, mateReadIdPos, destids);
                    destids = std::copy(mateReadIdPos + 1, myIds + num, destids);
                    task.mateRemovedFromCandidates = true;
                    batchData.numTasksWithMateRemoved++;
                }else{
                    numCandidatesPerAnchorTmp[k] = num;
                    destids = std::copy(myIds, myIds + num, destids);
                    task.mateRemovedFromCandidates = false;
                }

                numCandidatesPerAnchorPrefixSumTmp[k+1] =
                    numCandidatesPerAnchorPrefixSumTmp[k] + numCandidatesPerAnchorTmp[k];
            }else{
                numCandidatesPerAnchorTmp[k] = num;
                destids = std::copy(myIds, myIds + num, destids);
                task.mateRemovedFromCandidates = false;
                numCandidatesPerAnchorPrefixSumTmp[k+1] =
                    numCandidatesPerAnchorPrefixSumTmp[k] + numCandidatesPerAnchorTmp[k];
            }
        }

        /*
            Remove candidate pairs which have already been used for extension
        */

        destids = batchData.h_candidateReadIds.data();

        for(int k = 0; k < batchData.numTasks; k++){
            auto& task = batchData.tasks[batchData.indicesOfActiveTasks[k]];

            const int num = numCandidatesPerAnchorTmp[k];
            const int offset = numCandidatesPerAnchorPrefixSumTmp[k];
            read_number* myIds = candidateReadIdsTmp.data() + offset;

            std::vector<read_number> tmp(task.candidateReadIds.size());

            #ifdef DO_ONLY_REMOVE_MATE_IDS
                //remove none
                auto end = std::copy(myIds, myIds + num, destids);
            #else
            auto end = std::set_difference(
                myIds,
                myIds + num,
                task.allFullyUsedCandidateReadIdPairs.begin(),
                task.allFullyUsedCandidateReadIdPairs.end(),
                destids
            );            
            #endif

            batchData.h_numCandidatesPerAnchor[k] = std::distance(destids, end);
            destids = end;

            batchData.h_numCandidatesPerAnchorPrefixSum[k+1] =
                    batchData.h_numCandidatesPerAnchorPrefixSum[k] + batchData.h_numCandidatesPerAnchor[k];
        }

        batchData.totalNumCandidates = batchData.h_numCandidatesPerAnchorPrefixSum[batchData.numTasks];

        //std::cerr << "old numTasksWithMateRemoved = " << batchData.numTasksWithMateRemoved << ", totalNumCandidates = " << batchData.totalNumCandidates << "\n";

        cudaMemcpyAsync(
            batchData.d_candidateReadIds.data(),
            batchData.h_candidateReadIds.data(),
            sizeof(read_number) * batchData.totalNumCandidates,
            H2D,
            firstStream
        );
        // cudaMemcpyAsync(
        //     batchData.d_numCandidatesPerAnchor.data(),
        //     batchData.h_numCandidatesPerAnchor.data(),
        //     sizeof(int) * (batchData.numTasks),
        //     H2D,
        //     firstStream
        // );
        // cudaMemcpyAsync(
        //     batchData.d_numCandidatesPerAnchorPrefixSum.data(),
        //     batchData.h_numCandidatesPerAnchorPrefixSum.data(),
        //     sizeof(int) * (batchData.numTasks + 1),
        //     H2D,
        //     firstStream
        // );

        helpers::call_copy_n_kernel(
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.h_numCandidatesPerAnchorPrefixSum.data() + 1,
                batchData.h_numCandidatesPerAnchor.data()
            )),
            batchData.numTasks,
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1,
                batchData.d_numCandidatesPerAnchor.data()
            )),
            firstStream
        ); 

        batchData.d_anchorIndicesOfCandidates.resize(batchData.totalNumCandidates);
        batchData.h_anchorIndicesOfCandidates.resize(batchData.totalNumCandidates);

        for(int i = 0, sum = 0; i < batchData.numTasks; i++){
            std::fill(
                batchData.h_anchorIndicesOfCandidates.data() + sum,
                batchData.h_anchorIndicesOfCandidates.data() + sum + batchData.h_numCandidatesPerAnchor[i],
                i
            );
            sum += batchData.h_numCandidatesPerAnchor[i];
        }

        cudaMemcpyAsync(
            batchData.d_anchorIndicesOfCandidates.data(),
            batchData.h_anchorIndicesOfCandidates.data(),
            sizeof(int) * batchData.totalNumCandidates,
            H2D,
            firstStream
        ); CUERR;

        if(batchData.numTasksWithMateRemoved > 0){

            batchData.h_inputanchormatedata.resize(batchData.encodedSequencePitchInInts * batchData.numTasksWithMateRemoved);
            batchData.h_segmentIdsOfReadIds.resize(batchData.numTasksWithMateRemoved);

            unsigned int* destmatedata = batchData.h_inputanchormatedata.data();
            int* destids = batchData.h_segmentIdsOfReadIds.data();

            for(int k = 0; k < batchData.numTasks; k++){
                auto& task = batchData.tasks[batchData.indicesOfActiveTasks[k]];
                if(task.mateRemovedFromCandidates){
                    destmatedata = std::copy(
                        task.encodedMate.begin(),
                        task.encodedMate.end(),
                        destmatedata
                    );
                    *destids = k;
                    destids++;
                }
            }

            cudaMemcpyAsync(
                batchData.d_anchormatedata.data(),
                batchData.h_inputanchormatedata.data(),
                sizeof(unsigned int) * (batchData.encodedSequencePitchInInts * batchData.numTasksWithMateRemoved),
                H2D,
                firstStream
            );

            cudaMemcpyAsync(
                batchData.d_anchorIndicesWithRemovedMates.data(),
                batchData.h_segmentIdsOfReadIds.data(),
                sizeof(int) * (batchData.numTasksWithMateRemoved),
                H2D,
                firstStream
            );
        }

    }


    void removeUsedIdsAndMateIds(BatchData& batchData, cudaStream_t firstStream, cudaStream_t secondStream) const{
        //std::cerr << "\n" << batchData.totalNumCandidates << "\n";
        
        batchData.d_anchorIndicesOfCandidates.resize(batchData.totalNumCandidates);

        read_number* d_candidateReadIds2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_candidateReadIds2, sizeof(read_number) * batchData.totalNumCandidates, firstStream); CUERR;

        batchData.h_segmentIdsOfReadIds.resize(batchData.totalNumCandidates);

        batchData.h_numCandidatesPerAnchor.resize(batchData.numTasks);

        bool* d_shouldBeKept = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_shouldBeKept, sizeof(bool) * batchData.totalNumCandidates, firstStream);   

        bool* d_anchorFlags = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_anchorFlags, sizeof(bool) * batchData.numTasks, firstStream);   

        //determine required temp bytes for following cub calls, and allocate temp storage

        cudaError_t cubstatus = cudaSuccess;
        std::size_t cubBytes = 0;
        std::size_t cubBytes2 = 0;

        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            cubBytes2,
            batchData.d_candidateReadIds.data(),
            d_shouldBeKept,
            d_candidateReadIds2,
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
            d_anchorFlags,
            batchData.d_anchorIndicesWithRemovedMates.data(),
            batchData.h_numAnchorsWithRemovedMates.data(),
            batchData.numTasks,
            firstStream
        );
        assert(cubstatus == cudaSuccess);
        
        cubBytes = std::max(cubBytes, cubBytes2);
        
        void* cubtempstorage; cubAllocator->DeviceAllocate((void**)&cubtempstorage, cubBytes, firstStream);   
        
        helpers::call_fill_kernel_async(d_shouldBeKept, batchData.totalNumCandidates, false, firstStream);
        
        //flag candidates ids to remove because they are equal to anchor id or equal to mate id
        readextendergpukernels::flagCandidateIdsWhichAreEqualToAnchorOrMateKernel<<<batchData.numTasks, 128, 0, firstStream>>>(
            batchData.d_candidateReadIds.data(),
            batchData.d_anchorReadIds.data(),
            batchData.d_mateReadIds.data(),
            batchData.d_numCandidatesPerAnchorPrefixSum.data(),
            batchData.d_numCandidatesPerAnchor.data(),
            d_shouldBeKept,
            d_anchorFlags,
            batchData.d_numCandidatesPerAnchor2.data(),
            batchData.numTasks,
            batchData.pairedEnd
        );
        CUERR;

        cudaMemcpyAsync(
            batchData.h_numCandidatesPerAnchor.data(),
            batchData.d_numCandidatesPerAnchor2.data(),
            sizeof(int) * batchData.numTasks,
            D2H,
            firstStream
        ); CUERR;

        cudaEventRecord(batchData.events[0], firstStream);

        //determine task ids with removed mates

        assert(batchData.d_anchorIndicesWithRemovedMates.data() != nullptr);
        assert(batchData.h_numAnchorsWithRemovedMates.data() != nullptr);

        cubstatus = cub::DeviceSelect::Flagged(
            cubtempstorage,
            cubBytes,
            thrust::make_counting_iterator(0),
            d_anchorFlags,
            batchData.d_anchorIndicesWithRemovedMates.data(),
            batchData.h_numAnchorsWithRemovedMates.data(),
            batchData.numTasks,
            firstStream
        );
        assert(cubstatus == cudaSuccess);

        //copy selected candidate ids

        assert(d_candidateReadIds2 != nullptr);
        assert(batchData.h_numCandidates.data() != nullptr);

        assert(batchData.d_candidateReadIds.size() >= batchData.totalNumCandidates);

        cubstatus = cub::DeviceSelect::Flagged(
            cubtempstorage,
            cubBytes,
            batchData.d_candidateReadIds.data(),
            d_shouldBeKept,
            d_candidateReadIds2,
            batchData.h_numCandidates.data(),
            batchData.totalNumCandidates,
            firstStream
        );
        assert(cubstatus == cudaSuccess);

        cudaStreamSynchronize(firstStream); CUERR; //wait for h_numCandidates   and h_numAnchorsWithRemovedMates
        batchData.numTasksWithMateRemoved = *batchData.h_numAnchorsWithRemovedMates;
        batchData.totalNumCandidates = *batchData.h_numCandidates;

        cubAllocator->DeviceFree(d_shouldBeKept); CUERR;

        //std::cerr << "new numTasksWithMateRemoved = " << batchData.numTasksWithMateRemoved << ", totalNumCandidates = " << batchData.totalNumCandidates << "\n";

        if(batchData.numTasksWithMateRemoved > 0){

            //copy mate sequence data of removed mates

            std::size_t cubtempstream2bytes = 0;
            cubstatus = cub::DeviceSelect::Flagged(
                nullptr,
                cubtempstream2bytes,
                batchData.d_inputanchormatedata.data(),
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator(0),
                    SequenceFlagMultiplier{d_anchorFlags, int(batchData.encodedSequencePitchInInts)}
                ),
                batchData.d_anchormatedata.data(),
                thrust::make_discard_iterator(),
                batchData.numTasks * batchData.encodedSequencePitchInInts,
                secondStream
            );
            assert(cubstatus == cudaSuccess);
    
            void* cubtempstream2 = nullptr; cubAllocator->DeviceAllocate((void**)&cubtempstream2, cubtempstream2bytes, secondStream);
                
            assert(batchData.d_anchormatedata.data() != nullptr);

            cubstatus = cub::DeviceSelect::Flagged(
                cubtempstream2,
                cubtempstream2bytes,
                batchData.d_inputanchormatedata.data(),
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator(0),
                    SequenceFlagMultiplier{d_anchorFlags, int(batchData.encodedSequencePitchInInts)}
                ),
                batchData.d_anchormatedata.data(),
                thrust::make_discard_iterator(),
                batchData.numTasks * batchData.encodedSequencePitchInInts,
                secondStream
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceFree(cubtempstream2);
        }

        cubAllocator->DeviceFree(d_anchorFlags); CUERR;

        // //compute prefix sum of number of candidates per anchor
        helpers::call_set_kernel_async(batchData.d_numCandidatesPerAnchorPrefixSum2.data(), 0, 0, firstStream);

        cubstatus = cub::DeviceScan::InclusiveSum(
            cubtempstorage, 
            cubBytes, 
            batchData.d_numCandidatesPerAnchor2.data(), 
            batchData.d_numCandidatesPerAnchorPrefixSum2.data() + 1, 
            batchData.numTasks,
            firstStream
        );
        assert(cubstatus == cudaSuccess);

        #if 0

        //compute segment ids for candidate read ids
        cudaEventSynchronize(batchData.events[0]); CUERR; //wait for h_numCandidatesPerAnchor

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

        #else

        helpers::call_fill_kernel_async(batchData.d_anchorIndicesOfCandidates.data(), batchData.totalNumCandidates, 0, firstStream);

        readextendergpukernels::setFirstSegmentIdsKernel<<<SDIV(batchData.numTasks, 256), 256, 0, firstStream>>>(
            batchData.d_numCandidatesPerAnchor2.data(),
            batchData.d_anchorIndicesOfCandidates.data(),
            batchData.d_numCandidatesPerAnchorPrefixSum2.data(),
            batchData.numTasks
        );

        cubstatus = cub::DeviceScan::InclusiveScan(
            cubtempstorage, 
            cubBytes, 
            batchData.d_anchorIndicesOfCandidates.data(), 
            batchData.d_anchorIndicesOfCandidates.data(), 
            cub::Max{},
            batchData.totalNumCandidates,
            firstStream
        );
        assert(cubstatus == cudaSuccess);

        #endif
   

        ThrustCachingAllocator<char> thrustCachingAllocator1(deviceId, cubAllocator, firstStream);

        //cudaStreamWaitEvent(firstStream, batchData.events[0], 0); CUERR;

        int* d_anchorIndicesOfCandidates2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_anchorIndicesOfCandidates2, sizeof(int) * batchData.totalNumCandidates * 2, firstStream);

        assert(batchData.d_candidateReadIds.size() >= batchData.totalNumCandidates);

        #ifdef DO_ONLY_REMOVE_MATE_IDS
            cudaMemcpyAsync(
                batchData.d_candidateReadIds.data(),
                d_candidateReadIds2,
                sizeof(read_number) * batchData.totalNumCandidates,
                D2D,
                firstStream
            ); CUERR;
            cudaMemcpyAsync(
                d_anchorIndicesOfCandidates2,
                batchData.d_anchorIndicesOfCandidates.data(),
                sizeof(int) * batchData.totalNumCandidates,
                D2D,
                firstStream
            ); CUERR;
            cudaMemcpyAsync(
                batchData.d_numCandidatesPerAnchor.data(),
                batchData.d_numCandidatesPerAnchor2.data(),
                sizeof(int) * batchData.numTasks,
                D2D,
                firstStream
            ); CUERR;

            auto d_candidateReadIds_end = batchData.d_candidateReadIds.data() + batchData.totalNumCandidates;
        #else
        
        //compute segmented set difference.  batchData.d_candidateReadIds = d_candidateReadIds2 \ batchData.d_usedReadIds
        auto d_candidateReadIds_end = GpuSegmentedSetOperation::set_difference(
            thrustCachingAllocator1,
            d_candidateReadIds2,
            batchData.d_numCandidatesPerAnchor2.data(),
            batchData.d_numCandidatesPerAnchorPrefixSum2.data(),
            batchData.d_anchorIndicesOfCandidates.data(),
            batchData.totalNumCandidates,
            batchData.d_fullyUsedReadIds.data(),
            batchData.d_numFullyUsedReadIdsPerAnchor.data(),
            batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum.data(),
            batchData.d_segmentIdsOfFullyUsedReadIds.data(),
            batchData.totalNumberOfFullyUsedIds,
            batchData.numTasks,        
            batchData.d_candidateReadIds.data(),
            batchData.d_numCandidatesPerAnchor.data(),
            d_anchorIndicesOfCandidates2,
            firstStream
        );

        #endif


        cubAllocator->DeviceFree(d_candidateReadIds2); CUERR;

        batchData.totalNumCandidates = std::distance(batchData.d_candidateReadIds.data(), d_candidateReadIds_end);

        assert(batchData.d_candidateReadIds.size() >= batchData.totalNumCandidates);

        cudaMemcpyAsync(
            batchData.d_anchorIndicesOfCandidates.data(),
            d_anchorIndicesOfCandidates2,
            sizeof(int) * batchData.totalNumCandidates,
            D2D,
            firstStream
        ); CUERR;

        cubAllocator->DeviceFree(d_anchorIndicesOfCandidates2); CUERR;

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

        // {
        //     cudaDeviceSynchronize(); CUERR; 

        //     std::vector<int> offsets(batchData.numTasks + 1);
        //     cudaMemcpyAsync(
        //         offsets.data(),
        //         batchData.d_numCandidatesPerAnchorPrefixSum.data(),
        //         sizeof(int) * (batchData.numTasks + 1),
        //         D2H,
        //         firstStream
        //     );

        //     cudaDeviceSynchronize(); CUERR;
        //     std::cerr << "Offsets after removeusedidsandmateids:\n";
        //     for(int i = 0; i < batchData.numTasks+1; i++){
        //         std::cerr << offsets[i] << " ";
        //     }
        //     std::cerr << "\n";

        // }

        if(batchData.numTasksWithMateRemoved > 0){
            cudaEventRecord(batchData.events[0], secondStream);
            cudaStreamWaitEvent(firstStream, batchData.events[0], 0); CUERR;
        }

        cubAllocator->DeviceFree(cubtempstorage); CUERR;
    }

    void loadCandidateSequenceData(BatchData& batchData, cudaStream_t stream) const{

        const int totalNumCandidates = batchData.totalNumCandidates;

        batchData.d_candidateSequencesLength.resize(batchData.totalNumCandidates);
        batchData.d_candidateSequencesData.resize(batchData.encodedSequencePitchInInts * batchData.totalNumCandidates);

        assert(batchData.d_candidateReadIds.size() >= batchData.totalNumCandidates);

        gpuReadStorage->gatherSequences(
            readStorageHandle,
            batchData.d_candidateSequencesData.get(),
            batchData.encodedSequencePitchInInts,
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

        const int totalNumCandidates = batchData.totalNumCandidates;

        batchData.d_candidateSequencesData2.resize(batchData.encodedSequencePitchInInts * batchData.totalNumCandidates);

        read_number* d_candidateReadIds2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_candidateReadIds2, sizeof(read_number) * batchData.totalNumCandidates, stream); CUERR;

        int* d_candidateSequencesLength2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_candidateSequencesLength2, sizeof(int) * batchData.totalNumCandidates, stream); CUERR;

        bool* d_isPairedCandidate2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_isPairedCandidate2, sizeof(bool) * batchData.totalNumCandidates, stream); CUERR;

        int* d_anchorIndicesOfCandidates2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_anchorIndicesOfCandidates2, sizeof(int) * batchData.totalNumCandidates, stream);


        constexpr int groupsize = 32;
        constexpr int blocksize = 128;
        constexpr int groupsperblock = blocksize / groupsize;
        dim3 block(blocksize,1,1);
        dim3 grid(SDIV(batchData.numTasksWithMateRemoved * groupsize, blocksize), 1, 1);
        const std::size_t smembytes = sizeof(unsigned int) * groupsperblock * batchData.encodedSequencePitchInInts;

        bool* d_keepflags = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_keepflags, sizeof(bool) * batchData.totalNumCandidates, stream); CUERR;

        helpers::call_fill_kernel_async(d_keepflags, totalNumCandidates, true, stream);

        readextendergpukernels::filtermatekernel<blocksize,groupsize><<<grid, block, smembytes, stream>>>(
            batchData.d_anchormatedata.data(),
            batchData.d_candidateSequencesData.data(),
            batchData.encodedSequencePitchInInts,
            batchData.d_numCandidatesPerAnchor.data(),
            batchData.d_numCandidatesPerAnchorPrefixSum.data(),
            batchData.d_anchorIndicesWithRemovedMates.data(),
            batchData.numTasksWithMateRemoved,
            d_keepflags
        ); CUERR;

        int* d_outputpositions = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_outputpositions, sizeof(int) * batchData.totalNumCandidates, stream); CUERR;

        void* cubTemp = nullptr;
        std::size_t cubTempSize = 0;
        cudaError_t cubstatus = cudaSuccess;

        cubstatus = cub::DeviceScan::ExclusiveSum(
            nullptr,
            cubTempSize,
            d_keepflags, 
            d_outputpositions, 
            totalNumCandidates, 
            stream
        );
        assert(cudaSuccess == cubstatus);

        cubAllocator->DeviceAllocate((void**)&cubTemp, cubTempSize, stream);  CUERR;

        cubstatus = cub::DeviceScan::ExclusiveSum(
            cubTemp,
            cubTempSize,
            d_keepflags, 
            d_outputpositions, 
            totalNumCandidates, 
            stream
        );
        assert(cudaSuccess == cubstatus);

        cubAllocator->DeviceFree(cubTemp); CUERR;

        helpers::lambda_kernel<<<4096, 128, 0, stream>>>(
            [
                numTasks = batchData.numTasks,
                encodedSequencePitchInInts = batchData.encodedSequencePitchInInts,
                d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchorPrefixSum = batchData.d_numCandidatesPerAnchorPrefixSum.data(),
                d_keepflags,
                d_outputpositions = d_outputpositions,
                d_candidateReadIds = batchData.d_candidateReadIds.data(),
                d_candidateSequencesLength = batchData.d_candidateSequencesLength.data(),
                d_candidateSequencesData = batchData.d_candidateSequencesData.data(),
                d_anchorIndicesOfCandidates = batchData.d_anchorIndicesOfCandidates.data(),
                d_isPairedCandidate = batchData.d_isPairedCandidate.data(),
                d_candidateReadIdsOut = d_candidateReadIds2,
                d_candidateSequencesLengthOut = d_candidateSequencesLength2,
                d_candidateSequencesDataOut = batchData.d_candidateSequencesData2.data(),
                d_anchorIndicesOfCandidatesOut = d_anchorIndicesOfCandidates2,
                d_isPairedCandidateOut = d_isPairedCandidate2
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
                            d_isPairedCandidateOut[outputLocation] = d_isPairedCandidate[inputOffset + i];

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
                        }
                    }

                }
            }
        ); CUERR;

        cubAllocator->DeviceFree(d_outputpositions); CUERR;
        cubAllocator->DeviceFree(d_keepflags); CUERR;

        //update prefix sum        
        cubstatus = cub::DeviceScan::InclusiveSum(
            nullptr,
            cubTempSize,
            batchData.d_numCandidatesPerAnchor.data(), 
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            batchData.numTasks, 
            stream
        );
        assert(cudaSuccess == cubstatus);

        cubAllocator->DeviceAllocate((void**)&cubTemp, cubTempSize, stream); CUERR;

        cubstatus = cub::DeviceScan::InclusiveSum(
            cubTemp,
            cubTempSize,
            batchData.d_numCandidatesPerAnchor.data(), 
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            batchData.numTasks, 
            stream
        );
        assert(cudaSuccess == cubstatus);

        cubAllocator->DeviceFree(cubTemp); CUERR;

        cudaMemcpyAsync(
            batchData.h_numCandidates.data(),
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + batchData.numTasks,
            sizeof(int),
            D2H,
            stream
        ); CUERR;

        cudaStreamSynchronize(stream); CUERR;

        // {
        //     cudaDeviceSynchronize(); CUERR; 

        //     std::vector<int> offsets(batchData.numTasks + 1);
        //     cudaMemcpyAsync(
        //         offsets.data(),
        //         batchData.d_numCandidatesPerAnchorPrefixSum.data(),
        //         sizeof(int) * (batchData.numTasks + 1),
        //         D2H,
        //         stream
        //     );

        //     cudaDeviceSynchronize(); CUERR;
        //     std::cerr << "Offsets after erasedataofremovedmates:\n";
        //     for(int i = 0; i < batchData.numTasks+1; i++){
        //         std::cerr << offsets[i] << " ";
        //     }
        //     std::cerr << "\n";

        // }

        batchData.totalNumCandidates = *batchData.h_numCandidates;

        assert(batchData.d_candidateReadIds.size() >= batchData.totalNumCandidates);

        helpers::call_copy_n_kernel(
            thrust::make_zip_iterator(thrust::make_tuple(
                d_candidateReadIds2,
                d_candidateSequencesLength2,
                d_isPairedCandidate2,
                d_anchorIndicesOfCandidates2
            )),
            batchData.totalNumCandidates,
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.d_candidateReadIds.data(),
                batchData.d_candidateSequencesLength.data(),
                batchData.d_isPairedCandidate.data(),
                batchData.d_anchorIndicesOfCandidates.data()
            )),
            stream
        );

        cubAllocator->DeviceFree(d_candidateReadIds2); CUERR;
        cubAllocator->DeviceFree(d_candidateSequencesLength2); CUERR;
        cubAllocator->DeviceFree(d_isPairedCandidate2); CUERR;
        cubAllocator->DeviceFree(d_anchorIndicesOfCandidates2); CUERR;

        std::swap(batchData.d_candidateSequencesData2, batchData.d_candidateSequencesData); 
       
    }

    void calculateAlignments(BatchData& batchData, cudaStream_t stream) const{

        batchData.d_alignment_overlaps.resize(batchData.totalNumCandidates);
        batchData.d_alignment_shifts.resize(batchData.totalNumCandidates);
        batchData.d_alignment_nOps.resize(batchData.totalNumCandidates);
        batchData.d_alignment_best_alignment_flags.resize(batchData.totalNumCandidates);

        bool* d_alignment_isValid = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_alignment_isValid, sizeof(bool) * batchData.totalNumCandidates, stream); CUERR;
        
        batchData.h_numAnchors[0] = batchData.numTasks;

        const bool* const d_anchorContainsN = nullptr;
        const bool* const d_candidateContainsN = nullptr;
        const bool removeAmbiguousAnchors = false;
        const bool removeAmbiguousCandidates = false;
        const int maxNumAnchors = batchData.numTasks;
        const int maxNumCandidates = batchData.totalNumCandidates; //this does not need to be exact, but it must be >= batchData.d_numCandidatesPerAnchorPrefixSum[batchData.numTasks]
        const int maximumSequenceLength = batchData.encodedSequencePitchInInts * 16;
        const int encodedSequencePitchInInts2Bit = batchData.encodedSequencePitchInInts;
        const int min_overlap = goodAlignmentProperties->min_overlap;
        const float maxErrorRate = goodAlignmentProperties->maxErrorRate;
        const float min_overlap_ratio = goodAlignmentProperties->min_overlap_ratio;
        const float estimatedNucleotideErrorRate = correctionOptions->estimatedErrorrate;

        auto callAlignmentKernel = [&](void* d_tempstorage, size_t& tempstoragebytes){

            call_popcount_rightshifted_hamming_distance_kernel_async(
                d_tempstorage,
                tempstoragebytes,
                batchData.d_alignment_overlaps.get(),
                batchData.d_alignment_shifts.get(),
                batchData.d_alignment_nOps.get(),
                d_alignment_isValid,
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

        void* d_tempstorage = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_tempstorage, tempstoragebytes, stream); CUERR;

        callAlignmentKernel(d_tempstorage, tempstoragebytes);

        cubAllocator->DeviceFree(d_tempstorage); CUERR;

        cubAllocator->DeviceFree(d_alignment_isValid); CUERR;
    }

    void filterAlignments(BatchData& batchData, cudaStream_t stream) const{

        DEBUGDEVICESYNC



        // {
        //     DEBUGDEVICESYNC
        //     std::vector<read_number> ids(batchData.d_candidateReadIds.size());
        //     std::vector<int> nums(batchData.d_numCandidatesPerAnchor.size());
        //     std::vector<int> offsets(batchData.d_numCandidatesPerAnchorPrefixSum.size());

        //     cudaMemcpyAsync(ids.data(), batchData.d_candidateReadIds.data(), batchData.d_candidateReadIds.sizeInBytes(), D2H, stream); CUERR;
        //     cudaMemcpyAsync(nums.data(), batchData.d_numCandidatesPerAnchor.data(), batchData.d_numCandidatesPerAnchor.sizeInBytes(), D2H, stream); CUERR;
        //     cudaMemcpyAsync(offsets.data(), batchData.d_numCandidatesPerAnchorPrefixSum.data(), batchData.d_numCandidatesPerAnchorPrefixSum.sizeInBytes(), D2H, stream); CUERR;

        //     DEBUGDEVICESYNC

        //     std::cerr << "candidates before:\n";
        //     for(int i = 0; i < batchData.numTasks; i++){
        //         std::cerr << "i = " << i << ", taskIndex " << batchData.indicesOfActiveTasks[i] << "\n";
        //         const int num = nums[i];
        //         const int offset = offsets[i];
        //         const auto* myIds = ids.data() + offset;

        //         for(int k = 0; k < num; k++){
        //             std::cerr << myIds[k] << " ";
        //         }

        //         std::cerr << "\n";
        //     }
        // }


        const int totalNumCandidates = batchData.totalNumCandidates;
        const int numAnchors = batchData.numTasks;

        batchData.d_numCandidatesPerAnchor2.resize(batchData.numTasks);
        batchData.h_numCandidates.resize(1);

        batchData.d_candidateSequencesData2.resize(batchData.encodedSequencePitchInInts * batchData.totalNumCandidates);

        bool* d_keepflags = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_keepflags, sizeof(bool) * batchData.totalNumCandidates, stream); CUERR; 

        helpers::call_fill_kernel_async(d_keepflags, batchData.totalNumCandidates, true, stream);

        DEBUGDEVICESYNC

        dim3 block(128,1,1);
        dim3 grid(numAnchors, 1, 1);

        #if 0
            //filter alignments of candidates. d_keepflags[i] will be set to false if candidate[i] should be removed
        //batchData.d_numCandidatesPerAnchor2[i] contains new number of candidates for anchor i
        helpers::lambda_kernel<<<grid, block, 0, stream>>>(
            [
                d_alignment_best_alignment_flags = batchData.d_alignment_best_alignment_flags.data(),
                d_alignment_shifts = batchData.d_alignment_shifts.data(),
                d_alignment_overlaps = batchData.d_alignment_overlaps.data(),
                d_alignment_nOps = batchData.d_alignment_nOps.data(),
                d_anchorSequencesLength = batchData.d_anchorSequencesLength.data(),
                d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchor2 = batchData.d_numCandidatesPerAnchor2.data(),
                d_numCandidatesPerAnchorPrefixSum = batchData.d_numCandidatesPerAnchorPrefixSum.data(),
                d_isPairedCandidate = batchData.d_isPairedCandidate.data(),
                d_keepflags,
                min_overlap_ratio = goodAlignmentProperties->min_overlap_ratio,
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

                    //loop over candidates to compute relative overlap threshold

                    for(int c = threadIdx.x; c < num; c += blockDim.x){
                        
                        const auto alignmentflag = d_alignment_best_alignment_flags[offset + c];
                        const int shift = d_alignment_shifts[offset + c];

                        if(alignmentflag != BestAlignment_t::None && shift >= 0){
                            if(d_isPairedCandidate[offset+c]){
                                d_keepflags[offset + c] = true; //paired candidates always pass
                            }else{
                                const float overlap = d_alignment_overlaps[offset + c];
                                const float numMismatches = d_alignment_nOps[offset + c];                          
                                const float relativeOverlap = overlap / anchorLength;
                                const float errorrate = numMismatches / overlap;

                                if(fleq(errorrate, 0.03f)){
                                    d_keepflags[offset + c] = true;
                                }else{
                                    d_keepflags[offset + c] = false;
                                    removed++;
                                }
                            }
                        }else{
                            //remove alignment with negative shift or bad alignments
                            d_keepflags[offset + c] = false;
                            removed++;
                        }                  
                    }

                    removed = BlockReduceInt(cubtemp.intreduce).Sum(removed);

                    if(threadIdx.x == 0){
                        d_numCandidatesPerAnchor2[a] = num - removed;
                    }
                    __syncthreads();
                }
            }
        ); CUERR;
        #else

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
                d_isPairedCandidate = batchData.d_isPairedCandidate.data(),
                d_keepflags,
                min_overlap_ratio = goodAlignmentProperties->min_overlap_ratio,
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
                            if(!d_isPairedCandidate[offset+c]){
                                const float overlap = d_alignment_overlaps[offset + c];                            
                                const float relativeOverlap = overlap / anchorLength;
                                
                                if(relativeOverlap < 1.0f && fgeq(relativeOverlap, min_overlap_ratio)){
                                    threadReducedGoodAlignmentExists = 1;
                                    const float tmp = floorf(relativeOverlap * 10.0f) / 10.0f;
                                    threadReducedRelativeOverlapThreshold = fmaxf(threadReducedRelativeOverlapThreshold, tmp);
                                }
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
                            if(!d_isPairedCandidate[offset+c]){
                                if(d_keepflags[offset + c]){
                                    const float overlap = d_alignment_overlaps[offset + c];                            
                                    const float relativeOverlap = overlap / anchorLength;                 
        
                                    if(!fgeq(relativeOverlap, blockreducedRelativeOverlapThreshold)){
                                        d_keepflags[offset + c] = false;
                                        removed++;
                                    }
                                }
                            }
                        }
                    }else{
                        //NOOP.
                        //if no good alignment exists, no candidate is removed. we will try to work with the not-so-good alignments
                        // if(threadIdx.x == 0){
                        //     printf("no good alignment,nc %d\n", num);
                        // }
                    }

                    removed = BlockReduceInt(cubtemp.intreduce).Sum(removed);

                    if(threadIdx.x == 0){
                        d_numCandidatesPerAnchor2[a] = num - removed;
                    }
                    __syncthreads();
                }
            }
        ); CUERR;
        #endif

        DEBUGDEVICESYNC

        auto d_zip_data = thrust::make_zip_iterator(
            thrust::make_tuple(
                batchData.d_alignment_nOps.data(),
                batchData.d_alignment_overlaps.data(),
                batchData.d_alignment_shifts.data(),
                batchData.d_alignment_best_alignment_flags.data(),
                batchData.d_candidateReadIds.data(),
                batchData.d_candidateSequencesLength.data(),
                batchData.d_isPairedCandidate.data()
            )
        );

        int* d_alignment_overlaps2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_alignment_overlaps2, sizeof(int) * batchData.totalNumCandidates, stream); CUERR;

        int* d_alignment_shifts2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_alignment_shifts2, sizeof(int) * batchData.totalNumCandidates, stream); CUERR;

        int* d_alignment_nOps2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_alignment_nOps2, sizeof(int) * batchData.totalNumCandidates, stream); CUERR;

        BestAlignment_t* d_alignment_best_alignment_flags2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_alignment_best_alignment_flags2, sizeof(BestAlignment_t) * batchData.totalNumCandidates, stream); CUERR;

        int* d_candidateSequencesLength2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_candidateSequencesLength2, sizeof(int) * batchData.totalNumCandidates, stream); CUERR;

        read_number* d_candidateReadIds2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_candidateReadIds2, sizeof(read_number) * batchData.totalNumCandidates, stream); CUERR;

        bool* d_isPairedCandidate2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_isPairedCandidate2, sizeof(bool) * batchData.totalNumCandidates, stream); CUERR;

        DEBUGDEVICESYNC

        auto d_zip_data_tmp = thrust::make_zip_iterator(
            thrust::make_tuple(
                d_alignment_nOps2,
                d_alignment_overlaps2,
                d_alignment_shifts2,
                d_alignment_best_alignment_flags2,
                d_candidateReadIds2,
                d_candidateSequencesLength2,
                d_isPairedCandidate2
            )
        );

        assert(batchData.d_alignment_nOps.size() >= batchData.totalNumCandidates);
        assert(batchData.d_alignment_overlaps.size() >= batchData.totalNumCandidates);
        assert(batchData.d_alignment_shifts.size() >= batchData.totalNumCandidates);
        assert(batchData.d_alignment_best_alignment_flags.size() >= batchData.totalNumCandidates);
        assert(batchData.d_candidateReadIds.size() >= batchData.totalNumCandidates);
        assert(batchData.d_candidateSequencesLength.size() >= batchData.totalNumCandidates);
        assert(batchData.d_isPairedCandidate.size() >= batchData.totalNumCandidates);


        assert(d_alignment_nOps2 != nullptr);
        assert(d_alignment_overlaps2 != nullptr);
        assert(d_alignment_shifts2 != nullptr);
        assert(d_alignment_best_alignment_flags2 != nullptr);
        assert(d_candidateReadIds2 != nullptr);
        assert(d_candidateSequencesLength2 != nullptr);
        assert(d_isPairedCandidate2 != nullptr);

        //compact 1d arrays

        std::size_t cubTempSize = 0;
        void* cubTemp = nullptr;

        cudaError_t cubstatus = cub::DeviceSelect::Flagged(
            nullptr, 
            cubTempSize, 
            d_zip_data, 
            d_keepflags, 
            d_zip_data_tmp, 
            batchData.h_numCandidates.data(), 
            totalNumCandidates, 
            stream
        );
        assert(cubstatus == cudaSuccess);

        DEBUGDEVICESYNC

        cubAllocator->DeviceAllocate((void**)&cubTemp, cubTempSize, stream); CUERR;

        cubstatus = cub::DeviceSelect::Flagged(
            cubTemp, 
            cubTempSize, 
            d_zip_data, 
            d_keepflags, 
            d_zip_data_tmp, 
            batchData.h_numCandidates.data(), 
            totalNumCandidates, 
            stream
        );
        assert(cubstatus == cudaSuccess);

        DEBUGDEVICESYNC

        cubAllocator->DeviceFree(cubTemp); CUERR;

        cudaEventRecord(batchData.events[0], stream); CUERR;

        //compact 2d candidate sequences
        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            cubTempSize,
            batchData.d_candidateSequencesData.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                SequenceFlagMultiplier{d_keepflags, int(batchData.encodedSequencePitchInInts)}
            ),
            batchData.d_candidateSequencesData2.data(),
            thrust::make_discard_iterator(),
            totalNumCandidates * batchData.encodedSequencePitchInInts,
            stream
        );
        assert(cubstatus == cudaSuccess);

        cubAllocator->DeviceAllocate((void**)&cubTemp, cubTempSize, stream); CUERR;

        DEBUGDEVICESYNC

        cubstatus = cub::DeviceSelect::Flagged(
            cubTemp,
            cubTempSize,
            batchData.d_candidateSequencesData.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                SequenceFlagMultiplier{d_keepflags, int(batchData.encodedSequencePitchInInts)}
            ),
            batchData.d_candidateSequencesData2.data(),
            thrust::make_discard_iterator(), //number of remaining candidates already known from previous compaction call
            totalNumCandidates * batchData.encodedSequencePitchInInts,
            stream
        );

        cubAllocator->DeviceFree(cubTemp); CUERR;

        std::swap(batchData.d_candidateSequencesData2, batchData.d_candidateSequencesData);

        cubAllocator->DeviceFree(d_keepflags); CUERR;

        DEBUGDEVICESYNC


        //compute prefix sum of new number of candidates per anchor
        cubstatus = cub::DeviceScan::InclusiveSum(
            nullptr,
            cubTempSize,
            batchData.d_numCandidatesPerAnchor2.data(), 
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            batchData.numTasks, 
            stream
        );
        assert(cudaSuccess == cubstatus);

        cubAllocator->DeviceAllocate((void**)&cubTemp, cubTempSize, stream); CUERR;

        cubstatus = cub::DeviceScan::InclusiveSum(
            cubTemp,
            cubTempSize,
            batchData.d_numCandidatesPerAnchor2.data(), 
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            batchData.numTasks, 
            stream
        );
        assert(cudaSuccess == cubstatus);

        cubAllocator->DeviceFree(cubTemp); CUERR;

        std::swap(batchData.d_numCandidatesPerAnchor2, batchData.d_numCandidatesPerAnchor);

        DEBUGDEVICESYNC

        // {
        //     cudaDeviceSynchronize(); CUERR; 

        //     std::vector<int> offsets(batchData.numTasks + 1);
        //     cudaMemcpyAsync(
        //         offsets.data(),
        //         batchData.d_numCandidatesPerAnchorPrefixSum.data(),
        //         sizeof(int) * (batchData.numTasks + 1),
        //         D2H,
        //         stream
        //     );

        //     cudaDeviceSynchronize(); CUERR;
        //     std::cerr << "Offsets after filteralignment:\n";
        //     for(int i = 0; i < batchData.numTasks+1; i++){
        //         std::cerr << offsets[i] << " ";
        //     }
        //     std::cerr << "\n";

        // }
        


        cudaEventSynchronize(batchData.events[0]); CUERR;
        batchData.totalNumCandidates = *batchData.h_numCandidates;

        helpers::call_copy_n_kernel(
            d_zip_data_tmp,
            batchData.totalNumCandidates,
            d_zip_data,
            stream
        ); CUERR;

        DEBUGDEVICESYNC

        cubAllocator->DeviceFree(d_alignment_overlaps2); CUERR;
        cubAllocator->DeviceFree(d_alignment_shifts2); CUERR;
        cubAllocator->DeviceFree(d_alignment_nOps2); CUERR;
        cubAllocator->DeviceFree(d_alignment_best_alignment_flags2); CUERR;
        cubAllocator->DeviceFree(d_candidateSequencesLength2); CUERR;
        cubAllocator->DeviceFree(d_candidateReadIds2); CUERR;
        cubAllocator->DeviceFree(d_isPairedCandidate2); CUERR;


        // {
        //     DEBUGDEVICESYNC
        //     std::vector<read_number> ids(batchData.d_candidateReadIds.size());
        //     std::vector<int> nums(batchData.d_numCandidatesPerAnchor.size());
        //     std::vector<int> offsets(batchData.d_numCandidatesPerAnchorPrefixSum.size());

        //     cudaMemcpyAsync(ids.data(), batchData.d_candidateReadIds.data(), batchData.d_candidateReadIds.sizeInBytes(), D2H, stream); CUERR;
        //     cudaMemcpyAsync(nums.data(), batchData.d_numCandidatesPerAnchor.data(), batchData.d_numCandidatesPerAnchor.sizeInBytes(), D2H, stream); CUERR;
        //     cudaMemcpyAsync(offsets.data(), batchData.d_numCandidatesPerAnchorPrefixSum.data(), batchData.d_numCandidatesPerAnchorPrefixSum.sizeInBytes(), D2H, stream); CUERR;

        //     DEBUGDEVICESYNC

        //     std::cerr << "candidates after:\n";
        //     for(int i = 0; i < batchData.numTasks; i++){
        //         std::cerr << "i = " << i << ", taskIndex " << batchData.indicesOfActiveTasks[i] << "\n";
        //         const int num = nums[i];
        //         const int offset = offsets[i];
        //         const auto* myIds = ids.data() + offset;

        //         for(int k = 0; k < num; k++){
        //             std::cerr << myIds[k] << " ";
        //         }

        //         std::cerr << "\n";
        //     }
        // }
    }

    void loadCandidateQualityScores(BatchData& batchData, cudaStream_t stream, char* d_qualityscores) const{
        char* outputQualityScores = d_qualityscores;

        if(correctionOptions->useQualityScores){
            batchData.h_candidateReadIds.resize(batchData.totalNumCandidates);
            

            cudaMemcpyAsync(
                batchData.h_candidateReadIds.data(),
                batchData.d_candidateReadIds.data(),
                sizeof(read_number) * batchData.totalNumCandidates,
                D2H,
                stream
            ); CUERR;

            cudaStreamSynchronize(stream); CUERR;


            gpuReadStorage->gatherQualities(
                readStorageHandle,
                outputQualityScores,
                batchData.qualityPitchInBytes,
                batchData.h_candidateReadIds.data(),
                batchData.d_candidateReadIds.data(),
                batchData.totalNumCandidates,
                stream
            );

        }else{
            helpers::call_fill_kernel_async(
                outputQualityScores,
                batchData.qualityPitchInBytes * batchData.totalNumCandidates,
                'I',
                stream
            ); CUERR;
        }
        
    }


    void computeMSAs(BatchData& batchData, cudaStream_t firstStream, cudaStream_t secondStream) const{
        char* d_candidateQualityScores = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_candidateQualityScores, sizeof(char) * batchData.qualityPitchInBytes * batchData.totalNumCandidates, firstStream); CUERR;

        loadCandidateQualityScores(batchData, firstStream, d_candidateQualityScores);


        batchData.d_consensusEncoded.resize(batchData.numTasks * batchData.msaColumnPitchInElements);
        batchData.d_coverage.resize(batchData.numTasks * batchData.msaColumnPitchInElements);
        batchData.d_msa_column_properties.resize(batchData.numTasks);

        int* d_counts = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_counts, sizeof(int) * batchData.numTasks * 4 * batchData.msaColumnPitchInElements, firstStream); CUERR;

        float* d_weights = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_weights, sizeof(float) * batchData.numTasks * 4 * batchData.msaColumnPitchInElements, firstStream); CUERR;

        int* d_origCoverages = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_origCoverages, sizeof(int) * batchData.numTasks * batchData.msaColumnPitchInElements, firstStream); CUERR;

        float* d_origWeights = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_origWeights, sizeof(float) * batchData.numTasks * batchData.msaColumnPitchInElements, firstStream); CUERR;

        float* d_support = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_support, sizeof(float) * batchData.numTasks * batchData.msaColumnPitchInElements, firstStream); CUERR;

        batchData.d_consensusQuality.resize(batchData.numTasks * batchData.msaColumnPitchInElements);

        batchData.d_numCandidatesPerAnchor2.resize(batchData.numTasks);

        int* indices1 = nullptr; 
        int* indices2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&indices1, sizeof(int) * batchData.totalNumCandidates, firstStream); CUERR;
        cubAllocator->DeviceAllocate((void**)&indices2, sizeof(int) * batchData.totalNumCandidates, firstStream); CUERR;

        

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
        multiMSA.columnPitchInElements = batchData.msaColumnPitchInElements;
        multiMSA.counts = d_counts;
        multiMSA.weights = d_weights;
        multiMSA.coverages = batchData.d_coverage.get();
        multiMSA.consensus = batchData.d_consensusEncoded.get();
        multiMSA.support = d_support;
        multiMSA.origWeights = d_origWeights;
        multiMSA.origCoverages = d_origCoverages;
        multiMSA.columnProperties = batchData.d_msa_column_properties.get();

        const bool useQualityScoresForMSA = true;

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
            batchData.d_isPairedCandidate.get(),
            batchData.d_anchorQualityScores.get(), //d_anchor_qualities.get(),
            d_candidateQualityScores,
            batchData.h_numAnchors.get(), //d_numAnchors
            goodAlignmentProperties->maxErrorRate,
            batchData.numTasks,
            batchData.totalNumCandidates,
            useQualityScoresForMSA, //correctionOptions->useQualityScores,
            batchData.encodedSequencePitchInInts,
            batchData.qualityPitchInBytes,
            firstStream,
            kernelLaunchHandle
        );

        //refine msa
        bool* d_shouldBeKept = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_shouldBeKept, sizeof(bool) * batchData.totalNumCandidates, firstStream); CUERR;

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
            batchData.d_isPairedCandidate.get(),
            batchData.d_anchorSequencesLength.get(),
            batchData.d_candidateSequencesLength.get(),
            batchData.d_anchorQualityScores.get(), //d_anchor_qualities.get(),
            d_candidateQualityScores,
            d_shouldBeKept,
            batchData.d_numCandidatesPerAnchorPrefixSum.get(),
            batchData.h_numAnchors.get(),
            goodAlignmentProperties->maxErrorRate,
            batchData.numTasks,
            batchData.totalNumCandidates,
            useQualityScoresForMSA, //correctionOptions->useQualityScores,
            batchData.encodedSequencePitchInInts,
            batchData.qualityPitchInBytes,
            indices1, //d_indices,
            batchData.d_numCandidatesPerAnchor.get(),
            correctionOptions->estimatedCoverage,
            getNumRefinementIterations(),
            firstStream,
            kernelLaunchHandle
        );

        cubAllocator->DeviceFree(d_candidateQualityScores); CUERR;

        helpers::call_fill_kernel_async(d_shouldBeKept, batchData.totalNumCandidates, false, firstStream); CUERR;

        //convert output indices from task-local indices to global flags
        helpers::lambda_kernel<<<batchData.numTasks, 128, 0, firstStream>>>(
            [
                d_flagscandidates = d_shouldBeKept,
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


        std::size_t cubBytes = 0;
        void* cubtemp = nullptr;
        cudaError_t cubstatus = cudaSuccess;

        read_number* d_candidateReadIds2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_candidateReadIds2, sizeof(read_number) * batchData.totalNumCandidates, firstStream); CUERR;

        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            cubBytes,
            batchData.d_candidateReadIds.data(),
            d_shouldBeKept,
            d_candidateReadIds2,
            batchData.h_numCandidates.data(),
            batchData.totalNumCandidates,
            firstStream
        );

        cubAllocator->DeviceAllocate((void**)&cubtemp, cubBytes, firstStream);

        cubstatus = cub::DeviceSelect::Flagged(
            cubtemp,
            cubBytes,
            batchData.d_candidateReadIds.data(),
            d_shouldBeKept,
            d_candidateReadIds2,
            batchData.h_numCandidates.data(),
            batchData.totalNumCandidates,
            firstStream
        );

        assert(cubstatus == cudaSuccess);
        cubAllocator->DeviceFree(cubtemp);

        cudaEventRecord(batchData.events[0], firstStream); CUERR;



        //compute quality of consensus
        helpers::lambda_kernel<<<batchData.numTasks, 256, 0, firstStream>>>(
            [
                consensusQuality = batchData.d_consensusQuality.data(),
                support = d_support,
                coverages = batchData.d_coverage.data(),
                msa_column_properties = batchData.d_msa_column_properties.data(),
                d_numCandidatesInMsa = batchData.d_numCandidatesPerAnchor2.data(),
                columnPitchInElements = batchData.msaColumnPitchInElements,
                numTasks = batchData.numTasks
            ] __device__ (){

                for(int t = blockIdx.x; t < numTasks; t += gridDim.x){
                    if(d_numCandidatesInMsa[t] > 0){
                        const float* const taskSupport = support + t * columnPitchInElements;
                        const int* const taskCoverage = coverages + t * columnPitchInElements;
                        char* const taskConsensusQuality = consensusQuality + t * columnPitchInElements;
                        const int begin = msa_column_properties[t].firstColumn_incl;
                        const int end = msa_column_properties[t].lastColumn_excl;

                        // if(threadIdx.x == 0){
                        //     printf("t %d, begin %d end %d\n", t, begin, end);
                        // }
                        // __syncthreads();

                        assert(begin >= 0);
                        assert(end < columnPitchInElements);

                        for(int i = begin + threadIdx.x; i < end; i += blockDim.x){
                            const float support = taskSupport[i];
                            const float cov = taskCoverage[i];

                            char q = getQualityChar(taskSupport[i]);

                            //scale down quality depending on coverage
                            q = char(float(q) * min(1.0f, cov * 1.0f / 5.0f));

                            taskConsensusQuality[i] = getQualityChar(support);
                        }
                    }
                }
            }
        ); CUERR;

        cubAllocator->DeviceFree(d_counts); CUERR;
        cubAllocator->DeviceFree(d_weights); CUERR;
        cubAllocator->DeviceFree(d_origCoverages); CUERR;
        cubAllocator->DeviceFree(d_origWeights); CUERR;
        cubAllocator->DeviceFree(d_support); CUERR;        
        cubAllocator->DeviceFree(indices1); CUERR;
        cubAllocator->DeviceFree(indices2); CUERR;
        
        cubAllocator->DeviceFree(d_shouldBeKept); CUERR;        

        cubstatus = cub::DeviceScan::InclusiveSum(
            nullptr,
            cubBytes,
            batchData.d_numCandidatesPerAnchor2.get(), 
            batchData.d_numCandidatesPerAnchorPrefixSum.get() + 1, 
            batchData.numTasks, 
            firstStream
        );
        assert(cubstatus == cudaSuccess);

        cubAllocator->DeviceAllocate((void**)&cubtemp, cubBytes, firstStream); CUERR;

        cubstatus = cub::DeviceScan::InclusiveSum(
            cubtemp,
            cubBytes,
            batchData.d_numCandidatesPerAnchor2.get(), 
            batchData.d_numCandidatesPerAnchorPrefixSum.get() + 1, 
            batchData.numTasks, 
            firstStream
        );
        assert(cubstatus == cudaSuccess);
        cubAllocator->DeviceFree(cubtemp); CUERR;

        cudaEventSynchronize(batchData.events[0]); CUERR; //wait for h_numCandidates


        // {

        //     cudaDeviceSynchronize(); CUERR;
        //     std::cerr << "BEFORE\n";
        //     for(int i = 0; i < batchData.totalNumCandidates; i++){
        //         std::cerr << batchData.d_candidateReadIds[i] << " ";
        //     }
        //     std::cerr << "\n";

        //     for(int i = 0; i < batchData.numTasks; i++){
            
        //         const int numCandidates = batchData.d_numCandidatesPerAnchor[i];
        //         const int offset = batchData.d_numCandidatesPerAnchorPrefixSum[i];
        //         const read_number* ids = &batchData.d_candidateReadIds[offset];
        //         std::cerr << i << " " << numCandidates << " " << offset << "\n";

        //         assert(std::is_sorted(ids, ids + numCandidates));
        //     }
        // }

        //only information about number of candidates and readids are kept. all other information about candidates is discarded
        //auto oldnum = batchData.totalNumCandidates;
        batchData.totalNumCandidates = *batchData.h_numCandidates; 

        std::swap(batchData.d_numCandidatesPerAnchor, batchData.d_numCandidatesPerAnchor2);        

        cudaMemcpyAsync(
            batchData.d_candidateReadIds.data(),
            d_candidateReadIds2,
            sizeof(read_number) * batchData.totalNumCandidates,
            D2D,
            firstStream
        ); CUERR;

        cubAllocator->DeviceFree(d_candidateReadIds2); CUERR;

        // {
        //     std::vector<std::uint8_t> h_flags(oldnum); 
        //     std::vector<int> h_ints(oldnum);
        //     cudaMemcpyAsync(h_flags.data(), d_shouldBeKept, sizeof(bool) * oldnum, D2H, firstStream); CUERR;
        //     cudaMemcpyAsync(h_ints.data(), indices2, sizeof(int) * oldnum, D2H, firstStream); CUERR;

        //     cudaDeviceSynchronize(); CUERR;
        //     std::cerr << "AFTER\n";
        //     std::cerr << "d_numCandidates2 " << batchData.d_numCandidates2[0] << "\n";
        //     std::cerr << "h_numCandidates " << batchData.h_numCandidates[0] << "\n";

        //     std::cerr << "d_shouldBeKept\n";
        //     std::copy_n(h_flags.data(), oldnum, std::ostream_iterator<int>(std::cerr, " "));
        //     std::cerr << "\n";

        //     std::cerr << "indices2\n";
        //     std::copy_n(h_ints.data(), oldnum, std::ostream_iterator<int>(std::cerr, " "));
        //     std::cerr << "\n";
            
        //     for(int i = 0; i < batchData.totalNumCandidates; i++){
        //         std::cerr << batchData.d_candidateReadIds[i] << " ";
        //     }
        //     std::cerr << "\n";

        //     for(int i = 0; i < batchData.numTasks; i++){
            
        //         const int numCandidates = batchData.d_numCandidatesPerAnchor[i];
        //         const int offset = batchData.d_numCandidatesPerAnchorPrefixSum[i];
        //         const read_number* ids = &batchData.d_candidateReadIds[offset];

        //         std::cerr << i << " " << numCandidates << " " << offset << "\n";

        //         assert(std::is_sorted(ids, ids + numCandidates));
        //     }
        // }
        

    }

    void computeExtendedSequencesFromMSAs(BatchData& batchData, cudaStream_t stream) const{
        batchData.outputAnchorPitchInBytes = SDIV(batchData.decodedSequencePitchInBytes, 128) * 128;
        batchData.outputAnchorQualityPitchInBytes = SDIV(batchData.qualityPitchInBytes, 128) * 128;
        batchData.decodedMatesRevCPitchInBytes = SDIV(batchData.decodedSequencePitchInBytes, 128) * 128;

        batchData.h_accumExtensionsLengths.resize(batchData.numTasks);
        batchData.h_inputMateLengths.resize(batchData.numTasks);
        batchData.h_abortReasons.resize(batchData.numTasks);
        batchData.h_outputAnchors.resize(batchData.numTasks * batchData.outputAnchorPitchInBytes);
        batchData.h_outputAnchorQualities.resize(batchData.numTasks * batchData.outputAnchorQualityPitchInBytes);
        batchData.h_outputAnchorLengths.resize(batchData.numTasks);
        batchData.h_isPairedTask.resize(batchData.numTasks);
        batchData.h_decodedMatesRevC.resize(batchData.numTasks * batchData.decodedMatesRevCPitchInBytes);
        batchData.h_outputMateHasBeenFound.resize(batchData.numTasks);
        batchData.h_sizeOfGapToMate.resize(batchData.numTasks);
        batchData.h_isFullyUsedCandidate.resize(batchData.totalNumCandidates);

        batchData.h_scatterMap.resize(batchData.numTasks);

        int* d_accumExtensionsLengths = nullptr;
        int* d_inputMateLengths = nullptr;
        extension::AbortReason* d_abortReasons = nullptr;
        int* d_accumExtensionsLengthsOUT = nullptr;
        char* d_outputAnchors = nullptr;
        char* d_outputAnchorQualities = nullptr;
        int* d_outputAnchorLengths = nullptr;
        bool* d_isPairedTask = nullptr;
        char* d_decodedMatesRevC = nullptr;
        bool* d_outputMateHasBeenFound = nullptr;
        int* d_sizeOfGapToMate = nullptr;
        bool* d_isFullyUsedCandidate = nullptr;

        cubAllocator->DeviceAllocate((void**)&d_accumExtensionsLengths, sizeof(int) * batchData.numTasks, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_inputMateLengths, sizeof(int) * batchData.numTasks, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_abortReasons, sizeof(extension::AbortReason) * batchData.numTasks, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_accumExtensionsLengthsOUT, sizeof(int) * batchData.numTasks, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_outputAnchors, sizeof(char) * batchData.numTasks * batchData.outputAnchorPitchInBytes, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_outputAnchorQualities, sizeof(char) * batchData.numTasks * batchData.outputAnchorQualityPitchInBytes, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_outputAnchorLengths, sizeof(int) * batchData.numTasks, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_isPairedTask, sizeof(bool) * batchData.numTasks, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_decodedMatesRevC, sizeof(char) * batchData.numTasks * batchData.decodedMatesRevCPitchInBytes, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_outputMateHasBeenFound, sizeof(bool) * batchData.numTasks, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_sizeOfGapToMate, sizeof(int) * batchData.numTasks, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_isFullyUsedCandidate, sizeof(bool) * batchData.totalNumCandidates, stream); CUERR;



        helpers::call_fill_kernel_async(d_outputMateHasBeenFound, batchData.numTasks, false, stream); CUERR;
        helpers::call_fill_kernel_async(d_abortReasons, batchData.numTasks, extension::AbortReason::None, stream); CUERR;
        helpers::call_fill_kernel_async(d_isFullyUsedCandidate, batchData.totalNumCandidates, false, stream); CUERR;

        // helpers::call_fill_kernel_async(
        //     thrust::make_zip_iterator(thrust::make_tuple(d_outputMateHasBeenFound, d_abortReasons)),
        //     batchData.numTasks,
        //     thrust::make_tuple(false, extension::AbortReason::None),
        //     stream
        // );

        for(int i = 0; i < batchData.numTasks; i++){
            const int index = batchData.indicesOfActiveTasks[i];
            const auto& task = batchData.tasks[index];

            batchData.h_accumExtensionsLengths[i] = task.accumExtensionLengths;
            batchData.h_inputMateLengths[i] = task.mateLength;
            batchData.h_isPairedTask[i] = task.pairedEnd;
        }

        helpers::call_copy_n_kernel(
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.h_accumExtensionsLengths.data(),
                batchData.h_inputMateLengths.data(),
                batchData.h_isPairedTask.data()
            )),
            batchData.numTasks,
            thrust::make_zip_iterator(thrust::make_tuple(
                d_accumExtensionsLengths,
                d_inputMateLengths,
                d_isPairedTask
            )),
            stream
        );

        int numPairedEndTasks = 0;
        for(int i = 0; i < batchData.numTasks; i++){
            const int index = batchData.indicesOfActiveTasks[i];
            const auto& task = batchData.tasks[index];

            if(task.pairedEnd){

                // assert(task.decodedMateRevC.size() <= decodedMatesRevCPitchInBytes);
                // std::copy(task.decodedMateRevC.begin(), task.decodedMateRevC.end(), &h_decodedMatesRevC[i * decodedMatesRevCPitchInBytes]);
                std::copy(task.decodedMateRevC.begin(), task.decodedMateRevC.end(), &batchData.h_decodedMatesRevC[numPairedEndTasks * batchData.decodedMatesRevCPitchInBytes]);
                batchData.h_scatterMap[numPairedEndTasks] = i;
                numPairedEndTasks++;
            }
        }

        char* d_decodedMatesRevCDense = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_decodedMatesRevCDense, sizeof(char) * batchData.numTasks * batchData.decodedMatesRevCPitchInBytes, stream); CUERR;
        int* d_scatterMap = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_scatterMap, sizeof(int) * batchData.numTasks, stream); CUERR;

        cudaMemcpyAsync(
            d_decodedMatesRevCDense,
            batchData.h_decodedMatesRevC.data(),
            sizeof(char) * batchData.decodedMatesRevCPitchInBytes * numPairedEndTasks,
            H2D,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            d_scatterMap,
            batchData.h_scatterMap.data(),
            sizeof(int) * numPairedEndTasks,
            H2D,
            stream
        ); CUERR;

        helpers::lambda_kernel<<<batchData.numTasks, 128, 0, stream>>>(
            [
                numPairedEndTasks = numPairedEndTasks,
                decodedMatesRevCPitchInBytes = batchData.decodedMatesRevCPitchInBytes,
                d_scatterMap = d_scatterMap,
                d_decodedMatesRevCDense = d_decodedMatesRevCDense,
                d_decodedMatesRevC = d_decodedMatesRevC
            ] __device__ (){

                for(int t = blockIdx.x; t < numPairedEndTasks; t += gridDim.x){
                    const int destinationtask = d_scatterMap[t];

                    for(int i = threadIdx.x; i < decodedMatesRevCPitchInBytes; i += blockDim.x){
                        d_decodedMatesRevC[destinationtask * decodedMatesRevCPitchInBytes + i] = d_decodedMatesRevCDense[t * decodedMatesRevCPitchInBytes + i];
                    }
                }
            }
        ); CUERR;

        cubAllocator->DeviceFree(d_decodedMatesRevCDense); CUERR;
        cubAllocator->DeviceFree(d_scatterMap); CUERR;

        // {
        //     cudaDeviceSynchronize(); CUERR;

        //     // std::vector<read_number> readids(batchData.totalNumCandidates);
        //     // cudaMemcpyAsync(
        //     //     readids.data(),
        //     //     batchData.d_candidateReadIds.data(),
        //     //     sizeof(read_number) * batchData.totalNumCandidates,
        //     //     D2H,
        //     //     stream
        //     // );

        //     // std::vector<std::uint8_t> consensusEncoded(batchData.msaColumnPitchInElements * batchData.numTasks);
        //     // cudaMemcpyAsync(
        //     //     consensusEncoded.data(),
        //     //     batchData.d_consensusEncoded.data(),
        //     //     sizeof(std::uint8_t) * batchData.msaColumnPitchInElements * batchData.numTasks,
        //     //     D2H,
        //     //     stream
        //     // );
        //     std::cerr << "batchData.numTasks = " << batchData.numTasks << "\n";
        //     std::vector<int> nums(batchData.numTasks);
        //     cudaMemcpyAsync(
        //         nums.data(),
        //         batchData.d_numCandidatesPerAnchor.data(),
        //         sizeof(int) * batchData.numTasks,
        //         D2H,
        //         stream
        //     );

        //     std::vector<int> offsets(batchData.numTasks);
        //     cudaMemcpyAsync(
        //         offsets.data(),
        //         batchData.d_numCandidatesPerAnchorPrefixSum.data(),
        //         sizeof(int) * batchData.numTasks,
        //         D2H,
        //         stream
        //     );

        //     cudaDeviceSynchronize(); CUERR;

        //     std::cerr << "nums:\n";
        //     for(int i = 0; i < batchData.numTasks; i++){
        //         std::cerr << nums[i] << " ";
        //     }
        //     std::cerr << "\n";

        //     std::cerr << "offsets:\n";
        //     for(int i = 0; i < batchData.numTasks; i++){
        //         std::cerr << offsets[i] << " ";
        //     }
        //     std::cerr << "\n";

        //     // std::vector<char> consensusDecoded(consensusEncoded.size());
        //     // std::transform(consensusEncoded.begin(), consensusEncoded.end(), consensusDecoded.begin(),
        //     //     [](const std::uint8_t encoded){
        //     //         char decoded = 'F';
        //     //         if(encoded == std::uint8_t{0}){
        //     //             decoded = 'A';
        //     //         }else if(encoded == std::uint8_t{1}){
        //     //             decoded = 'C';
        //     //         }else if(encoded == std::uint8_t{2}){
        //     //             decoded = 'G';
        //     //         }else if(encoded == std::uint8_t{3}){
        //     //             decoded = 'T';
        //     //         }
        //     //         return decoded;
        //     //     }
        //     // );

        //     // for(int i = 0; i < batchData.numTasks; i++){
        //     //     const int index = batchData.indicesOfActiveTasks[i];
        //     //     const auto& task = batchData.tasks[index];

        //     //     if(task.myReadId == 0 && task.id == 3 && maxextensionPerStep == 6){
        //     //         std::cerr << "task id " << task.id << " myReadId " << task.myReadId << "\n";
        //     //         std::cerr << "candidates\n";
        //     //         for(int k = 0; k < nums[i]; k++){
        //     //             std::cerr << readids[offsets[i] + k] << " ";
        //     //         }
        //     //         std::cerr << "\n";

        //     //         std::cerr << "consensus\n";
        //     //         for(int k = 0; k < batchData.msaColumnPitchInElements; k++){
        //     //             std::cerr << consensusDecoded[i * batchData.msaColumnPitchInElements + k];
        //     //         }
        //     //         std::cerr << "\n";
        //     //     }

        //     // }

            

        //     cudaDeviceSynchronize(); CUERR;
        // }

        //compute extensions
           
        helpers::lambda_kernel<<<batchData.numTasks, 128, 0, stream>>>(
            [
                numTasks = batchData.numTasks,
                insertSize = insertSize,
                insertSizeStddev = insertSizeStddev,
                msaColumnPitchInElements = batchData.msaColumnPitchInElements,
                d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.data(),
                d_msa_column_properties = batchData.d_msa_column_properties.data(),
                d_consensusEncoded = batchData.d_consensusEncoded.data(),
                d_consensusQuality = batchData.d_consensusQuality.data(),
                d_coverage = batchData.d_coverage.data(),
                d_anchorSequencesLength = batchData.d_anchorSequencesLength.data(),
                d_accumExtensionsLengths = (int*)d_accumExtensionsLengths,
                d_inputMateLengths = (int*)d_inputMateLengths,
                d_abortReasons = (extension::AbortReason*)d_abortReasons,
                d_accumExtensionsLengthsOUT = (int*)d_accumExtensionsLengthsOUT,
                d_outputAnchors = (char*)d_outputAnchors,
                outputAnchorPitchInBytes = batchData.outputAnchorPitchInBytes,
                d_outputAnchorQualities = (char*)d_outputAnchorQualities,
                outputAnchorQualityPitchInBytes = batchData.outputAnchorQualityPitchInBytes,
                d_outputAnchorLengths = (int*)d_outputAnchorLengths,
                d_isPairedTask = (bool*)d_isPairedTask,
                d_decodedMatesRevC = (char*)d_decodedMatesRevC,
                decodedMatesRevCPitchInBytes = batchData.decodedMatesRevCPitchInBytes,
                d_outputMateHasBeenFound = (bool*)d_outputMateHasBeenFound,
                d_sizeOfGapToMate = (int*)d_sizeOfGapToMate,
                minCoverageForExtension = this->minCoverageForExtension,
                fixedStepsize = this->maxextensionPerStep
            ] __device__ (){

                auto decodeConsensus = [](const std::uint8_t encoded){
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

                using BlockReduce = cub::BlockReduce<int, 128>;

                __shared__ union{
                    typename BlockReduce::TempStorage reduce;
                } temp;

                __shared__ int broadcastsmem_int;

                for(int t = blockIdx.x; t < numTasks; t += gridDim.x){
                    const int numCandidates = d_numCandidatesPerAnchor[t];

                    if(numCandidates > 0){
                        const auto msaProps = d_msa_column_properties[t];

                        assert(msaProps.firstColumn_incl == 0);
                        assert(msaProps.lastColumn_excl <= msaColumnPitchInElements);

                        const int anchorLength = d_anchorSequencesLength[t];
                        int accumExtensionsLength = d_accumExtensionsLengths[t];
                        const int mateLength = d_inputMateLengths[t];
                        const bool isPaired = d_isPairedTask[t];

                        const int consensusLength = msaProps.lastColumn_excl - msaProps.firstColumn_incl;

                        const std::uint8_t* const consensusEncoded = d_consensusEncoded + t * msaColumnPitchInElements;
                        auto consensusDecoded = thrust::transform_iterator(consensusEncoded, decodeConsensus);
                        const char* const consensusQuality = d_consensusQuality + t * msaColumnPitchInElements;
                        const int* const msacoverage = d_coverage + t * msaColumnPitchInElements;
                        const char* const decodedMateRevC = d_decodedMatesRevC + t * decodedMatesRevCPitchInBytes;

                        extension::AbortReason* const abortReasonPtr = d_abortReasons + t;
                        char* const outputAnchor = d_outputAnchors + t * outputAnchorPitchInBytes;
                        char* const outputAnchorQuality = d_outputAnchorQualities + t * outputAnchorQualityPitchInBytes;
                        int* const outputAnchorLengthPtr = d_outputAnchorLengths + t;
                        bool* const mateHasBeenFoundPtr = d_outputMateHasBeenFound + t;

                        int extendBy = std::min(
                            consensusLength - anchorLength, 
                            std::max(0, fixedStepsize)
                        );
                        //cannot extend over fragment 
                        extendBy = std::min(extendBy, (insertSize + insertSizeStddev - mateLength) - accumExtensionsLength);

                        //auto firstLowCoverageIter = std::find_if(coverage + anchorLength, coverage + consensusLength, [&](int cov){ return cov < minCoverageForExtension; });
                        //coverage is monotonically decreasing. convert coverages to 1 if >= minCoverageForExtension, else 0. Find position of first 0
                        int myPos = consensusLength;
                        for(int i = anchorLength + threadIdx.x; i < consensusLength; i += blockDim.x){
                            int flag = msacoverage[i] < minCoverageForExtension ? 0 : 1;
                            if(flag == 0 && i < myPos){
                                myPos = i;
                            }
                        }

                        myPos = BlockReduce(temp.reduce).Reduce(myPos, cub::Min{});

                        if(threadIdx.x == 0){
                            broadcastsmem_int = myPos;
                        }
                        __syncthreads();
                        myPos = broadcastsmem_int;
                        __syncthreads();

                        if(fixedStepsize <= 0){
                            extendBy = myPos - anchorLength;
                            extendBy = std::min(extendBy, (insertSize + insertSizeStddev - mateLength) - accumExtensionsLength);
                        }

                        auto makeAnchorForNextIteration = [&](){
                            if(extendBy == 0){
                                if(threadIdx.x == 0){
                                    *abortReasonPtr = extension::AbortReason::MsaNotExtended;
                                }
                            }else{
                                if(threadIdx.x == 0){
                                    d_accumExtensionsLengthsOUT[t] = accumExtensionsLength + extendBy;
                                    *outputAnchorLengthPtr = anchorLength;
                                }

                                for(int i = threadIdx.x; i < anchorLength; i += blockDim.x){
                                    outputAnchor[i] = consensusDecoded[extendBy + i];
                                    outputAnchorQuality[i] = consensusQuality[extendBy + i];
                                }
                            }
                        };

                        constexpr int requiredOverlapMate = 70; //TODO relative overlap 
                        constexpr float maxRelativeMismatchesInOverlap = 0.06f;
                        constexpr int maxAbsoluteMismatchesInOverlap = 10;

                        const int maxNumMismatches = std::min(int(mateLength * maxRelativeMismatchesInOverlap), maxAbsoluteMismatchesInOverlap);

                        

                        if(isPaired && accumExtensionsLength + consensusLength - requiredOverlapMate + mateLength >= insertSize - insertSizeStddev){
                            //for each possibility to overlap the mate and consensus such that the merged sequence would end in the desired range [insertSize - insertSizeStddev, insertSize + insertSizeStddev]

                            const int firstStartpos = std::max(0, insertSize - insertSizeStddev - accumExtensionsLength - mateLength);
                            const int lastStartposExcl = std::min(
                                std::max(0, insertSize + insertSizeStddev - accumExtensionsLength - mateLength) + 1,
                                consensusLength - requiredOverlapMate
                            );

                            int bestOverlapMismatches = std::numeric_limits<int>::max();
                            int bestOverlapStartpos = -1;

                            for(int startpos = firstStartpos; startpos < lastStartposExcl; startpos++){
                                //compute metrics of overlap

                                //Hamming distance. positions which do not overlap are not accounted for
                                int ham = 0;
                                for(int i = threadIdx.x; i < min(consensusLength - startpos, mateLength); i += blockDim.x){
                                    ham += (consensusDecoded[startpos + i] != decodedMateRevC[i]) ? 1 : 0;
                                }

                                ham = BlockReduce(temp.reduce).Sum(ham);

                                if(threadIdx.x == 0){
                                    broadcastsmem_int = ham;
                                }
                                __syncthreads();
                                ham = broadcastsmem_int;
                                __syncthreads();

                                if(bestOverlapMismatches > ham){
                                    bestOverlapMismatches = ham;
                                    bestOverlapStartpos = startpos;
                                }

                                if(bestOverlapMismatches == 0){
                                    break;
                                }
                            }

                            // if(threadIdx.x == 0){
                            //     printf("gpu: bestOverlapMismatches %d,bestOverlapStartpos %d\n", bestOverlapMismatches, bestOverlapStartpos);
                            // }

                            if(bestOverlapMismatches <= maxNumMismatches){
                                const int mateStartposInConsensus = bestOverlapStartpos;
                                const int missingPositionsBetweenAnchorEndAndMateBegin = std::max(0, mateStartposInConsensus - anchorLength);
                                // if(threadIdx.x == 0){
                                //     printf("missingPositionsBetweenAnchorEndAndMateBegin %d\n", missingPositionsBetweenAnchorEndAndMateBegin);
                                // }

                                if(missingPositionsBetweenAnchorEndAndMateBegin > 0){
                                    //bridge the gap between current anchor and mate

                                    for(int i = threadIdx.x; i < missingPositionsBetweenAnchorEndAndMateBegin; i += blockDim.x){
                                        outputAnchor[i] = consensusDecoded[anchorLength + i];
                                        outputAnchorQuality[i] = consensusQuality[anchorLength + i];
                                    }

                                    if(threadIdx.x == 0){
                                        d_accumExtensionsLengthsOUT[t] = accumExtensionsLength + anchorLength;
                                        *outputAnchorLengthPtr = missingPositionsBetweenAnchorEndAndMateBegin;
                                        *mateHasBeenFoundPtr = true;
                                        d_sizeOfGapToMate[t] = missingPositionsBetweenAnchorEndAndMateBegin;
                                    }
                                }else{

                                    if(threadIdx.x == 0){
                                        d_accumExtensionsLengthsOUT[t] = accumExtensionsLength + mateStartposInConsensus;
                                        *outputAnchorLengthPtr = 0;
                                        *mateHasBeenFoundPtr = true;
                                        d_sizeOfGapToMate[t] = 0;
                                    }
                                }

                                
                            }else{
                                makeAnchorForNextIteration();
                            }
                        }else{
                            makeAnchorForNextIteration();
                        }

                    }else{ //numCandidates == 0
                        if(threadIdx.x == 0){
                            d_abortReasons[t] = extension::AbortReason::NoPairedCandidatesAfterAlignment;
                        }
                    }
                }
            }
        );

        //check which candidates are fully used in the extension
        helpers::lambda_kernel<<<batchData.numTasks, 128, 0, stream>>>(
            [
                numTasks = batchData.numTasks,
                d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchorPrefixSum = batchData.d_numCandidatesPerAnchorPrefixSum.data(),
                d_candidateSequencesLengths = batchData.d_candidateSequencesLength.data(),
                d_alignment_shifts = batchData.d_alignment_shifts.data(),
                d_anchorSequencesLength = batchData.d_anchorSequencesLength.data(),
                d_oldaccumExtensionsLengths = d_accumExtensionsLengths,
                d_newaccumExtensionsLengths = d_accumExtensionsLengthsOUT,
                d_abortReasons,
                d_outputMateHasBeenFound,
                d_sizeOfGapToMate,
                d_isFullyUsedCandidate
            ] __device__ (){


                for(int task = blockIdx.x; task < numTasks; task += gridDim.x){
                    const int numCandidates = d_numCandidatesPerAnchor[task];
                    const auto abortReason = d_abortReasons[task];

                    if(numCandidates > 0 && abortReason == extension::AbortReason::None){
                        const int anchorLength = d_anchorSequencesLength[task];
                        const int offset = d_numCandidatesPerAnchorPrefixSum[task];
                        const int oldAccumExtensionsLength = d_oldaccumExtensionsLengths[task];
                        const int newAccumExtensionsLength = d_newaccumExtensionsLengths[task];
                        const int lengthOfExtension = newAccumExtensionsLength - oldAccumExtensionsLength;

                        for(int c = threadIdx.x; c < numCandidates; c += blockDim.x){
                            const int candidateLength = d_candidateSequencesLengths[c];
                            const int shift = d_alignment_shifts[c];

                            if(candidateLength + shift <= anchorLength + lengthOfExtension){
                                d_isFullyUsedCandidate[offset + c] = true;
                            }
                        }
                    }
                }
            }
        );

        helpers::call_copy_n_kernel(
            thrust::make_zip_iterator(thrust::make_tuple(
                d_accumExtensionsLengthsOUT,
                d_abortReasons,
                d_outputMateHasBeenFound,
                d_sizeOfGapToMate,
                d_outputAnchorLengths
            )),
            batchData.numTasks,
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.h_accumExtensionsLengths.data(),
                batchData.h_abortReasons.data(),
                batchData.h_outputMateHasBeenFound.data(),
                batchData.h_sizeOfGapToMate.data(),
                batchData.h_outputAnchorLengths.data()
            )),
            stream
        );

        cudaMemcpyAsync(
            batchData.h_outputAnchors.data(),
            d_outputAnchors,
            sizeof(char) * batchData.outputAnchorPitchInBytes * batchData.numTasks,
            D2H,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            batchData.h_outputAnchorQualities.data(),
            d_outputAnchorQualities,
            sizeof(char) * batchData.outputAnchorQualityPitchInBytes * batchData.numTasks,
            D2H,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            batchData.h_isFullyUsedCandidate.data(),
            d_isFullyUsedCandidate,
            sizeof(bool) * batchData.totalNumCandidates,
            D2H,
            stream
        ); CUERR;

        cubAllocator->DeviceFree(d_accumExtensionsLengths); CUERR;
        cubAllocator->DeviceFree(d_inputMateLengths); CUERR;
        cubAllocator->DeviceFree(d_abortReasons); CUERR;
        cubAllocator->DeviceFree(d_accumExtensionsLengthsOUT); CUERR;
        cubAllocator->DeviceFree(d_outputAnchors); CUERR;
        cubAllocator->DeviceFree(d_outputAnchorQualities); CUERR;
        cubAllocator->DeviceFree(d_outputAnchorLengths); CUERR;
        cubAllocator->DeviceFree(d_isPairedTask); CUERR;
        cubAllocator->DeviceFree(d_decodedMatesRevC); CUERR;
        cubAllocator->DeviceFree(d_outputMateHasBeenFound); CUERR;
        cubAllocator->DeviceFree(d_sizeOfGapToMate); CUERR;
        cubAllocator->DeviceFree(d_isFullyUsedCandidate); CUERR;
    }
    
    
    
    void copyBuffersToHost(BatchData& batchData, cudaStream_t firstStream, cudaStream_t secondStream) const{
        batchData.h_candidateReadIds.resize(batchData.totalNumCandidates);
        batchData.h_numCandidatesPerAnchor.resize(batchData.numTasks);
        batchData.h_numCandidatesPerAnchorPrefixSum.resize(batchData.numTasks + 1);

        batchData.h_numCandidatesPerAnchorPrefixSum[0] = 0;
       
        helpers::call_copy_n_kernel(
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1,
                batchData.d_numCandidatesPerAnchor.data()
            )),
            batchData.numTasks,
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.h_numCandidatesPerAnchorPrefixSum.data() + 1,
                batchData.h_numCandidatesPerAnchor.data()
            )),
            firstStream
        );   

        cudaMemcpyAsync(
            batchData.h_candidateReadIds.data(),
            batchData.d_candidateReadIds.data(),
            sizeof(read_number) * batchData.totalNumCandidates,
            D2H,
            firstStream
        );
    }

};




}


#endif