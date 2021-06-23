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
#include <util_iterator.hpp>
#include <readextender_common.hpp>
#include <gpu/cubvector.cuh>



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


#define DO_REMOVE_USED_IDS_AND_MATE_IDS_ON_GPU

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

    template<int groupsize>
    __global__
    void encodeSequencesTo2BitKernel(
        unsigned int* __restrict__ encodedSequences,
        const char* __restrict__ decodedSequences,
        const int* __restrict__ sequenceLengths,
        int decodedSequencePitchInBytes,
        int encodedSequencePitchInInts,
        int numSequences
    ){
        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());

        const int numGroups = (blockDim.x * gridDim.x) / group.size();
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / group.size();

        for(int a = groupId; a < numSequences; a += numGroups){
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

    template<int groupsize>
    __global__
    void compactUsedIdsOfSelectedTasks(
        const int* __restrict__ indices,
        int numIndices,
        const read_number* __restrict__ d_usedReadIdsIn,
        read_number* __restrict__ d_usedReadIdsOut,
        int* __restrict__ segmentIdsOut,
        const int* __restrict__ d_numUsedReadIdsPerAnchor,
        const int* __restrict__ inputSegmentOffsets,
        const int* __restrict__ outputSegmentOffsets
    ){
        const int warpid = (threadIdx.x + blockDim.x * blockIdx.x) / groupsize;
        const int numwarps = (blockDim.x * gridDim.x) / groupsize;
        const int lane = threadIdx.x % groupsize;

        for(int t = warpid; t < numIndices; t += numwarps){
            const int activeIndex = indices[t];
            const int num = d_numUsedReadIdsPerAnchor[t];
            const int inputOffset = inputSegmentOffsets[activeIndex];
            const int outputOffset = outputSegmentOffsets[t];

            for(int i = lane; i < num; i += groupsize){
                //copy read id
                d_usedReadIdsOut[outputOffset + i] = d_usedReadIdsIn[inputOffset + i];
                //set new segment id
                segmentIdsOut[outputOffset + i] = t;
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
        BeforeHash,
        BeforeRemoveIds,
        BeforeComputePairFlags,
        BeforeLoadCandidates,
        BeforeEraseData,
        BeforeAlignment,
        BeforeAlignmentFilter,
        BeforeMSA,
        BeforeExtend,
        BeforeUpdateUsedCandidateIds,
        BeforeCopyToHost,
        BeforeUnpack,
        BeforePrepareNextIteration,
        Finished,
        None
    };

    static std::string to_string(State s){
        switch(s){
            case State::BeforeHash: return "BeforeHash";
            case State::BeforeRemoveIds: return "BeforeRemoveIds";
            case State::BeforeComputePairFlags: return "BeforeComputePairFlags";
            case State::BeforeLoadCandidates: return "BeforeLoadCandidates";
            case State::BeforeEraseData: return "BeforeEraseData";
            case State::BeforeAlignment: return "BeforeAlignment";
            case State::BeforeAlignmentFilter: return "BeforeAlignmentFilter";
            case State::BeforeMSA: return "BeforeMSA";
            case State::BeforeExtend: return "BeforeExtend";
            case State::BeforeUpdateUsedCandidateIds: return "BeforeUpdateUsedCandidateIds";            
            case State::BeforeCopyToHost: return "BeforeCopyToHost";
            case State::BeforeUnpack: return "BeforeUnpack";
            case State::BeforePrepareNextIteration: return "BeforePrepareNextIteration";
            case State::Finished: return "Finished";
            case State::None: return "None";
            default: return "Missing case BatchData::to_string(State)\n";
        };
    }

    bool isEmpty() const noexcept{
        return tasks.empty();
    }

    bool printTransitions = false;

    void setState(State newstate){      
        if(printTransitions){
            std::cerr << "batchdata " << someId << " statechange " << to_string(state) << " -> " << to_string(newstate) << "\n";
        }

        state = newstate;
    }



    BatchData(
        bool isPairedEnd_,
        const gpu::GpuReadStorage& rs, 
        const gpu::GpuMinhasher& gpuMinhasher_,
        const CorrectionOptions& coropts,
        const GoodAlignmentProperties& gap,
        const cpu::QualityScoreConversion& qualityConversion_,
        int insertSize_,
        int insertSizeStddev_,
        int maxextensionPerStep_,
        std::array<cudaStream_t, 4> streams_,
        cub::CachingDeviceAllocator& cubAllocator_
    ) : 
        pairedEnd(isPairedEnd_),
        insertSize(insertSize_),
        insertSizeStddev(insertSizeStddev_),
        maxextensionPerStep(maxextensionPerStep_),
        cubAllocator(&cubAllocator_),
        gpuReadStorage(&rs),
        gpuMinhasher(&gpuMinhasher_),
        minhashHandle(gpuMinhasher->makeMinhasherHandle()),
        correctionOptions(&coropts),
        goodAlignmentProperties(&gap),
        qualityConversion(&qualityConversion_),
        readStorageHandle(gpuReadStorage->makeHandle()),
        d_candidateSequencesData(cubAllocator_),
        d_candidateSequencesLength(cubAllocator_),    
        d_candidateReadIds(cubAllocator_),
        d_isPairedCandidate(cubAllocator_),
        d_anchorIndicesOfCandidates(cubAllocator_),
        d_alignment_overlaps(cubAllocator_),
        d_alignment_shifts(cubAllocator_),
        d_alignment_nOps(cubAllocator_),
        d_alignment_best_alignment_flags(cubAllocator_),
        d_numCandidatesPerAnchor(cubAllocator_),
        d_numCandidatesPerAnchorPrefixSum(cubAllocator_),
        d_inputanchormatedata(cubAllocator_),
        d_subjectSequencesDataDecoded(cubAllocator_),
        d_anchorQualityScores(cubAllocator_),
        d_anchorSequencesLength(cubAllocator_),
        d_anchorReadIds(cubAllocator_),
        d_mateReadIds(cubAllocator_),
        d_inputMateLengths(cubAllocator_),
        d_isPairedTask(cubAllocator_),
        d_subjectSequencesData(cubAllocator_),
        streams(streams_)
    {

        cudaGetDevice(&deviceId); CUERR;

        kernelLaunchHandle = gpu::make_kernel_launch_handle(deviceId);

        h_numUsedReadIds.resize(1);
        h_numFullyUsedReadIds.resize(1);
        h_numAnchors.resize(1);
        h_numCandidates.resize(1);
        h_numAnchorsWithRemovedMates.resize(1);

        d_numAnchors.resize(1);
        d_numCandidates.resize(1);
        d_numCandidates2.resize(1);

        *h_numUsedReadIds = 0;
        *h_numFullyUsedReadIds = 0;
        *h_numAnchors = 0;
        *h_numCandidates = 0;
        *h_numAnchorsWithRemovedMates = 0;

        numTasks = 0;   
    }

    ~BatchData(){
        gpuMinhasher->destroyHandle(minhashHandle);
    }

    static constexpr int getNumRefinementIterations() noexcept{
        return 5;
    }

    template<class TaskIter>
    void addTasks(TaskIter extraTasksBegin, TaskIter extraTasksEnd){
        const int numAdditionalTasks = std::distance(extraTasksBegin, extraTasksEnd);
        assert(numAdditionalTasks % 4 == 0);
        if(numAdditionalTasks == 0) return;

        const int currentNumTasks = tasks.size();
        const int newNumTasks = currentNumTasks + numAdditionalTasks;

        CachedDeviceUVector<char> cubTemp(*cubAllocator);
        std::size_t cubTempSize = 0;
        cudaError_t cubstatus = cudaSuccess;

        h_anchorReadIds.resize(newNumTasks);
        h_mateReadIds.resize(newNumTasks);
        h_subjectSequencesDataDecoded.resize(newNumTasks * decodedSequencePitchInBytes);
        h_anchorSequencesLength.resize(newNumTasks);
        h_anchorQualityScores.resize(newNumTasks * qualityPitchInBytes);
        h_inputanchormatedata.resize(newNumTasks * encodedSequencePitchInInts);
        h_inputMateLengths.resize(newNumTasks);
        h_isPairedTask.resize(newNumTasks);


        //d_anchorIndicesWithRemovedMates.resize(newNumTasks); // ???

        d_numUsedReadIdsPerAnchor2.resize(newNumTasks);
        d_numUsedReadIdsPerAnchorPrefixSum.resize(newNumTasks);
        d_numFullyUsedReadIdsPerAnchor2.resize(newNumTasks);
        d_numFullyUsedReadIdsPerAnchorPrefixSum.resize(newNumTasks);

        d_subjectSequencesData.resize(newNumTasks * encodedSequencePitchInInts, streams[0]);
        d_anchorSequencesLength.resize(newNumTasks, streams[0]);
        d_anchorQualityScores.resize(newNumTasks * qualityPitchInBytes, streams[0]);
        d_inputanchormatedata.resize(newNumTasks * encodedSequencePitchInInts, streams[0]);
        d_inputMateLengths.resize(newNumTasks, streams[0]);
        d_isPairedTask.resize(newNumTasks, streams[0]);


        cudaMemcpyAsync(
            d_numUsedReadIdsPerAnchor2.data(), 
            d_numUsedReadIdsPerAnchor.data(),
            sizeof(int) * currentNumTasks,
            D2D,
            streams[0]
        ); CUERR;
        std::swap(d_numUsedReadIdsPerAnchor, d_numUsedReadIdsPerAnchor2);

        cudaMemcpyAsync(
            d_numFullyUsedReadIdsPerAnchor2.data(), 
            d_numFullyUsedReadIdsPerAnchor.data(),
            sizeof(int) * currentNumTasks,
            D2D,
            streams[0]
        ); CUERR;
        std::swap(d_numFullyUsedReadIdsPerAnchor, d_numFullyUsedReadIdsPerAnchor2);

        cudaMemsetAsync(d_numUsedReadIdsPerAnchor.data() + currentNumTasks, 0, numAdditionalTasks * sizeof(int), streams[0]); CUERR;
        cudaMemsetAsync(d_numFullyUsedReadIdsPerAnchor.data() + currentNumTasks, 0, numAdditionalTasks * sizeof(int), streams[0]); CUERR;

        cubstatus = cub::DeviceScan::ExclusiveSum(
            nullptr,
            cubTempSize,
            d_numUsedReadIdsPerAnchor.data(), 
            d_numUsedReadIdsPerAnchorPrefixSum.data(), 
            newNumTasks, 
            streams[0]
        );
        assert(cudaSuccess == cubstatus);

        cubTemp.resizeWithoutCopy(cubTempSize, streams[0]);

        cubstatus = cub::DeviceScan::ExclusiveSum(
            cubTemp.data(),
            cubTempSize,
            d_numUsedReadIdsPerAnchor.data(), 
            d_numUsedReadIdsPerAnchorPrefixSum.data(), 
            newNumTasks, 
            streams[0]
        );
        assert(cudaSuccess == cubstatus);


        cubstatus = cub::DeviceScan::ExclusiveSum(
            nullptr,
            cubTempSize,
            d_numFullyUsedReadIdsPerAnchor.data(), 
            d_numFullyUsedReadIdsPerAnchorPrefixSum.data(), 
            newNumTasks, 
            streams[0]
        );
        assert(cudaSuccess == cubstatus);

        cubTemp.resizeWithoutCopy(cubTempSize, streams[0]);

        cubstatus = cub::DeviceScan::ExclusiveSum(
            cubTemp.data(),
            cubTempSize,
            d_numFullyUsedReadIdsPerAnchor.data(), 
            d_numFullyUsedReadIdsPerAnchorPrefixSum.data(), 
            newNumTasks, 
            streams[0]
        );
        assert(cudaSuccess == cubstatus);

        for(int t = 0; t < numAdditionalTasks; t++){
            const auto& task = *(extraTasksBegin + t);

            h_anchorReadIds[t] = task.myReadId;
            h_mateReadIds[t] = task.mateReadId;

            std::copy(
                task.encodedMate.begin(),
                task.encodedMate.end(),
                h_inputanchormatedata.begin() + t * encodedSequencePitchInInts
            );

            h_anchorSequencesLength[t] = task.currentAnchorLength;

            std::copy(
                task.totalDecodedAnchors.back().begin(),
                task.totalDecodedAnchors.back().end(),
                h_subjectSequencesDataDecoded.begin() + t * decodedSequencePitchInBytes
            );

            assert(h_anchorQualityScores.size() >= (t+1) * qualityPitchInBytes);

            std::copy(
                task.totalAnchorQualityScores.back().begin(),
                task.totalAnchorQualityScores.back().end(),
                h_anchorQualityScores.begin() + t * qualityPitchInBytes
            );

            h_inputMateLengths[t] = task.mateLength;
            h_isPairedTask[t] = task.pairedEnd;
        }

        cudaMemcpyAsync(
            d_inputanchormatedata.data() + currentNumTasks * encodedSequencePitchInInts,
            h_inputanchormatedata.data(),
            sizeof(unsigned int) * encodedSequencePitchInInts * numAdditionalTasks,
            H2D,
            streams[0]
        ); CUERR;

        cudaMemcpyAsync(
            d_anchorSequencesLength.data() + currentNumTasks,
            h_anchorSequencesLength.data(),
            sizeof(int) * numAdditionalTasks,
            H2D,
            streams[0]
        ); CUERR;

        d_anchorReadIds.resize(newNumTasks, streams[0]);
        cudaMemcpyAsync(
            d_anchorReadIds.data() + currentNumTasks,
            h_anchorReadIds.data(),
            sizeof(read_number) * numAdditionalTasks,
            H2D,
            streams[0]
        ); CUERR;

        d_mateReadIds.resize(newNumTasks, streams[0]);
        cudaMemcpyAsync(
            d_mateReadIds.data() + currentNumTasks,
            h_mateReadIds.data(),
            sizeof(read_number) * numAdditionalTasks,
            H2D,
            streams[0]
        ); CUERR;

        cudaMemcpyAsync(
            d_inputMateLengths.data() + currentNumTasks,
            h_inputMateLengths.data(),
            sizeof(int) * numAdditionalTasks,
            H2D,
            streams[0]
        ); CUERR;

        cudaMemcpyAsync(
            d_isPairedTask.data() + currentNumTasks,
            h_isPairedTask.data(),
            sizeof(bool) * numAdditionalTasks,
            H2D,
            streams[0]
        ); CUERR;


        d_subjectSequencesDataDecoded.resize(newNumTasks * decodedSequencePitchInBytes, streams[0]);
        cudaMemcpyAsync(
            d_subjectSequencesDataDecoded.data() + currentNumTasks * decodedSequencePitchInBytes,
            h_subjectSequencesDataDecoded.data(),
            sizeof(char) * numAdditionalTasks * decodedSequencePitchInBytes,
            H2D,
            streams[0]
        ); CUERR;

        cudaMemcpyAsync(
            d_anchorQualityScores.data() + currentNumTasks * qualityPitchInBytes,
            h_anchorQualityScores.data(),
            sizeof(char) * numAdditionalTasks * qualityPitchInBytes,
            H2D,
            streams[0]
        ); CUERR;

        readextendergpukernels::encodeSequencesTo2BitKernel<8>
        <<<SDIV(numAdditionalTasks, (128 / 8)), 128, 0, streams[0]>>>(
            d_subjectSequencesData.data() + currentNumTasks * encodedSequencePitchInInts,
            d_subjectSequencesDataDecoded.data() + currentNumTasks * decodedSequencePitchInBytes,
            d_anchorSequencesLength.data() + currentNumTasks,
            decodedSequencePitchInBytes,
            encodedSequencePitchInInts,
            numAdditionalTasks
        ); CUERR;


        //init flat arrays
        for(auto it = extraTasksBegin; it != extraTasksEnd; ++it){

            const int expectedNumIterations = 2 + (insertSize / maxextensionPerStep);
            
            it->totalDecodedAnchorsFlat.reserve(expectedNumIterations * decodedSequencePitchInBytes);
            it->totalDecodedAnchorsFlat.resize(decodedSequencePitchInBytes);
            std::copy(
                it->totalDecodedAnchors.back().begin(),
                it->totalDecodedAnchors.back().end(),
                it->totalDecodedAnchorsFlat.begin()
            );

            it->totalDecodedAnchorsLengths.reserve(expectedNumIterations);
            it->totalDecodedAnchorsLengths.emplace_back(it->totalDecodedAnchors.back().size());

            it->totalAnchorQualityScoresFlat.reserve(expectedNumIterations * qualityPitchInBytes);
            it->totalAnchorQualityScoresFlat.resize(qualityPitchInBytes);
            std::copy(
                it->totalAnchorQualityScores.back().begin(),
                it->totalAnchorQualityScores.back().end(),
                it->totalAnchorQualityScoresFlat.begin()
            );
        }

        //save tasks and update indices of active tasks

        tasks.insert(tasks.end(), std::make_move_iterator(extraTasksBegin), std::make_move_iterator(extraTasksEnd));
        assert(tasks.size() % 4 == 0);

        

        for(int i = currentNumTasks; i < int(tasks.size()); i++){
            tasks[i].id = i - currentNumTasks;
        }

        numTasks = tasks.size();
        numReadPairs = tasks.size() / 4;

        state = State::BeforeHash;
    }

    void resetTasks(){
        state = State::BeforeHash;
        numTasks = 0;
        numReadPairs = 0;
        tasks.clear();
        finishedTasks.clear();
    }

    void setMaxExtensionPerStep(int e) noexcept{
        maxextensionPerStep = e;
    }

    void setMinCoverageForExtension(int c) noexcept{
        minCoverageForExtension = c;
    }

    void process(){
        assert(state == BatchData::State::BeforeHash);

        while(state != BatchData::State::Finished){
            performNextStep();
        }
    }

    void performNextStep(){
        const auto name = BatchData::to_string(state);

        nvtx::push_range(name, static_cast<int>(state));

        switch(state){
            case BatchData::State::BeforeHash: getCandidateReadIds(); break;
            case BatchData::State::BeforeRemoveIds: removeUsedIdsAndMateIds(); break;
            case BatchData::State::BeforeComputePairFlags: computePairFlagsGpu(); break;
            case BatchData::State::BeforeLoadCandidates: loadCandidateSequenceData(); break;
            case BatchData::State::BeforeEraseData: eraseDataOfRemovedMates(); break;
            case BatchData::State::BeforeAlignment: calculateAlignments(); break;
            case BatchData::State::BeforeAlignmentFilter: filterAlignments(); break;
            case BatchData::State::BeforeMSA: computeMSAs(); break;
            case BatchData::State::BeforeExtend: computeExtendedSequencesFromMSAs(); break;
            case BatchData::State::BeforeUpdateUsedCandidateIds: updateUsedCandidateIds(); break;            
            case BatchData::State::BeforeCopyToHost: copyBuffersToHost(); break;
            case BatchData::State::BeforeUnpack: unpackResultsIntoTasks(); break;
            case BatchData::State::BeforePrepareNextIteration: prepareNextIteration(); break;
            case BatchData::State::Finished: break;
            case BatchData::State::None: break;
            default: break;
        };

        if(state == BatchData::State::Finished){
            assert(tasks.size() == 0);
            assert(finishedTasks.size() % 4 == 0);
        }

        nvtx::pop_range();
    }

    void getCandidateReadIds(){
        assert(state == BatchData::State::BeforeHash);

        cudaStream_t stream = streams[0];

        d_numCandidatesPerAnchor.resizeWithoutCopy(numTasks, stream);
        d_numCandidatesPerAnchorPrefixSum.resizeWithoutCopy(numTasks + 1, stream);

        int totalNumValues = 0;

        gpuMinhasher->determineNumValues(
            minhashHandle,
            d_subjectSequencesData.data(),
            encodedSequencePitchInInts,
            d_anchorSequencesLength.data(),
            numTasks,
            d_numCandidatesPerAnchor.data(),
            totalNumValues,
            stream
        );

        cudaStreamSynchronize(stream); CUERR;

        d_candidateReadIds.resizeWithoutCopy(totalNumValues, stream);    

        if(totalNumValues == 0){
            cudaMemsetAsync(d_numCandidatesPerAnchor.data(), 0, sizeof(int) * numTasks , stream); CUERR;
            cudaMemsetAsync(d_numCandidatesPerAnchorPrefixSum.data(), 0, sizeof(int) * (1 + numTasks), stream); CUERR;
            totalNumCandidates = 0;

            setStateToFinished();
            return;
        }

        gpuMinhasher->retrieveValues(
            minhashHandle,
            nullptr,
            numTasks,              
            totalNumValues,
            d_candidateReadIds.data(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            stream
        );

        cudaMemcpyAsync(
            h_numCandidates.data(),
            d_numCandidatesPerAnchorPrefixSum.data() + numTasks,
            sizeof(int),
            D2H,
            stream
        ); CUERR;

        cudaStreamSynchronize(stream); CUERR;

        totalNumCandidates = *h_numCandidates;

        setState(BatchData::State::BeforeRemoveIds);
    }

    void removeUsedIdsAndMateIds(){
        assert(state == BatchData::State::BeforeRemoveIds);

        cudaStream_t firstStream = streams[0];
        cudaStream_t secondStream = firstStream;

        //std::cerr << "\n" << totalNumCandidates << "\n";
        
        d_anchorIndicesOfCandidates.resizeWithoutCopy(totalNumCandidates, firstStream);

        read_number* d_candidateReadIds2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_candidateReadIds2, sizeof(read_number) * totalNumCandidates, firstStream); CUERR;

        h_segmentIdsOfReadIds.resize(totalNumCandidates);

        h_numCandidatesPerAnchor.resize(numTasks);

        d_anchorIndicesWithRemovedMates.resize(numTasks);

        bool* d_shouldBeKept = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_shouldBeKept, sizeof(bool) * totalNumCandidates, firstStream);   

        bool* d_anchorFlags = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_anchorFlags, sizeof(bool) * numTasks, firstStream);   

        //determine required temp bytes for following cub calls, and allocate temp storage

        cudaError_t cubstatus = cudaSuccess;
        std::size_t cubBytes = 0;
        std::size_t cubBytes2 = 0;

        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            cubBytes2,
            d_candidateReadIds.data(),
            d_shouldBeKept,
            d_candidateReadIds2,
            h_numCandidates.data(),
            totalNumCandidates,
            firstStream
        );
        assert(cubstatus == cudaSuccess);
        
        cubBytes = std::max(cubBytes, cubBytes2);
        
        cubstatus = cub::DeviceScan::InclusiveSum(
            nullptr, 
            cubBytes2, 
            (int*)nullptr,
            (int*)nullptr,
            numTasks,
            firstStream
        );
        assert(cubstatus == cudaSuccess);
        
        cubBytes = std::max(cubBytes, cubBytes2);
        
        cubstatus = cub::DeviceScan::InclusiveScan(
            nullptr, 
            cubBytes2, 
            d_anchorIndicesOfCandidates.data(), 
            d_anchorIndicesOfCandidates.data(), 
            cub::Max{},
            totalNumCandidates,
            firstStream
        );
        assert(cubstatus == cudaSuccess);
        
        cubBytes = std::max(cubBytes, cubBytes2);
        
        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            cubBytes2,
            thrust::make_counting_iterator(0),
            d_anchorFlags,
            d_anchorIndicesWithRemovedMates.data(),
            h_numAnchorsWithRemovedMates.data(),
            numTasks,
            firstStream
        );
        assert(cubstatus == cudaSuccess);
        
        cubBytes = std::max(cubBytes, cubBytes2);

        CachedDeviceUVector<char> cubTemp(cubBytes, firstStream, *cubAllocator);
               
        helpers::call_fill_kernel_async(d_shouldBeKept, totalNumCandidates, false, firstStream);

        CachedDeviceUVector<int> d_numCandidatesPerAnchor2(numTasks, firstStream, *cubAllocator);
        
        //flag candidates ids to remove because they are equal to anchor id or equal to mate id
        readextendergpukernels::flagCandidateIdsWhichAreEqualToAnchorOrMateKernel<<<numTasks, 128, 0, firstStream>>>(
            d_candidateReadIds.data(),
            d_anchorReadIds.data(),
            d_mateReadIds.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_numCandidatesPerAnchor.data(),
            d_shouldBeKept,
            d_anchorFlags,
            d_numCandidatesPerAnchor2.data(),
            numTasks,
            pairedEnd
        );
        CUERR;

        cudaMemcpyAsync(
            h_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchor2.data(),
            sizeof(int) * numTasks,
            D2H,
            firstStream
        ); CUERR;

        cudaEventRecord(events[0], firstStream);

        //determine task ids with removed mates

        assert(d_anchorIndicesWithRemovedMates.data() != nullptr);
        assert(h_numAnchorsWithRemovedMates.data() != nullptr);

        cubstatus = cub::DeviceSelect::Flagged(
            cubTemp.data(),
            cubBytes,
            thrust::make_counting_iterator(0),
            d_anchorFlags,
            d_anchorIndicesWithRemovedMates.data(),
            h_numAnchorsWithRemovedMates.data(),
            numTasks,
            firstStream
        );
        assert(cubstatus == cudaSuccess);

        //copy selected candidate ids

        assert(d_candidateReadIds2 != nullptr);
        assert(h_numCandidates.data() != nullptr);

        assert(d_candidateReadIds.size() >= totalNumCandidates);

        cubstatus = cub::DeviceSelect::Flagged(
            cubTemp.data(),
            cubBytes,
            d_candidateReadIds.data(),
            d_shouldBeKept,
            d_candidateReadIds2,
            h_numCandidates.data(),
            totalNumCandidates,
            firstStream
        );
        assert(cubstatus == cudaSuccess);

        cudaStreamSynchronize(firstStream); CUERR; //wait for h_numCandidates   and h_numAnchorsWithRemovedMates
        numTasksWithMateRemoved = *h_numAnchorsWithRemovedMates;
        totalNumCandidates = *h_numCandidates;

        d_anchorIndicesWithRemovedMates.resize(numTasksWithMateRemoved);

        cubAllocator->DeviceFree(d_shouldBeKept); CUERR;

        //std::cerr << "new numTasksWithMateRemoved = " << numTasksWithMateRemoved << ", totalNumCandidates = " << totalNumCandidates << "\n";

        if(numTasksWithMateRemoved > 0){

            d_anchormatedata.resize(numTasks * encodedSequencePitchInInts);

            //copy mate sequence data of removed mates

            std::size_t cubtempstream2bytes = 0;
            cubstatus = cub::DeviceSelect::Flagged(
                nullptr,
                cubtempstream2bytes,
                d_inputanchormatedata.data(),
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator(0),
                    SequenceFlagMultiplier{d_anchorFlags, int(encodedSequencePitchInInts)}
                ),
                d_anchormatedata.data(),
                thrust::make_discard_iterator(),
                numTasks * encodedSequencePitchInInts,
                secondStream
            );
            assert(cubstatus == cudaSuccess);
    
            CachedDeviceUVector<char> cubTemp2(cubtempstream2bytes, secondStream, *cubAllocator);

            assert(d_anchormatedata.data() != nullptr);

            cubstatus = cub::DeviceSelect::Flagged(
                cubTemp2.data(),
                cubtempstream2bytes,
                d_inputanchormatedata.data(),
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator(0),
                    SequenceFlagMultiplier{d_anchorFlags, int(encodedSequencePitchInInts)}
                ),
                d_anchormatedata.data(),
                thrust::make_discard_iterator(),
                numTasks * encodedSequencePitchInInts,
                secondStream
            );
            assert(cubstatus == cudaSuccess);
        }

        cubAllocator->DeviceFree(d_anchorFlags); CUERR;

        CachedDeviceUVector<int> d_numCandidatesPerAnchorPrefixSum2(numTasks + 1, firstStream, *cubAllocator);

        // //compute prefix sum of number of candidates per anchor
        helpers::call_set_kernel_async(d_numCandidatesPerAnchorPrefixSum2.data(), 0, 0, firstStream);

        cubstatus = cub::DeviceScan::InclusiveSum(
            cubTemp.data(), 
            cubBytes, 
            d_numCandidatesPerAnchor2.data(), 
            d_numCandidatesPerAnchorPrefixSum2.data() + 1, 
            numTasks,
            firstStream
        );
        assert(cubstatus == cudaSuccess);

        helpers::call_fill_kernel_async(d_anchorIndicesOfCandidates.data(), totalNumCandidates, 0, firstStream);

        readextendergpukernels::setFirstSegmentIdsKernel<<<SDIV(numTasks, 256), 256, 0, firstStream>>>(
            d_numCandidatesPerAnchor2.data(),
            d_anchorIndicesOfCandidates.data(),
            d_numCandidatesPerAnchorPrefixSum2.data(),
            numTasks
        );

        cubstatus = cub::DeviceScan::InclusiveScan(
            cubTemp.data(), 
            cubBytes, 
            d_anchorIndicesOfCandidates.data(), 
            d_anchorIndicesOfCandidates.data(), 
            cub::Max{},
            totalNumCandidates,
            firstStream
        );
        assert(cubstatus == cudaSuccess);


   

        ThrustCachingAllocator<char> thrustCachingAllocator1(deviceId, cubAllocator, firstStream);

        //cudaStreamWaitEvent(firstStream, events[0], 0); CUERR;

        int* d_anchorIndicesOfCandidates2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_anchorIndicesOfCandidates2, sizeof(int) * totalNumCandidates * 2, firstStream);

        assert(d_candidateReadIds.size() >= totalNumCandidates);

        #ifdef DO_ONLY_REMOVE_MATE_IDS
            cudaMemcpyAsync(
                d_candidateReadIds.data(),
                d_candidateReadIds2,
                sizeof(read_number) * totalNumCandidates,
                D2D,
                firstStream
            ); CUERR;
            cudaMemcpyAsync(
                d_anchorIndicesOfCandidates2,
                d_anchorIndicesOfCandidates.data(),
                sizeof(int) * totalNumCandidates,
                D2D,
                firstStream
            ); CUERR;
            cudaMemcpyAsync(
                d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchor2.data(),
                sizeof(int) * numTasks,
                D2D,
                firstStream
            ); CUERR;

            auto d_candidateReadIds_end = d_candidateReadIds.data() + totalNumCandidates;
        #else
        
        //compute segmented set difference.  d_candidateReadIds = d_candidateReadIds2 \ d_usedReadIds
        auto d_candidateReadIds_end = GpuSegmentedSetOperation::set_difference(
            thrustCachingAllocator1,
            d_candidateReadIds2,
            d_numCandidatesPerAnchor2.data(),
            d_numCandidatesPerAnchorPrefixSum2.data(),
            d_anchorIndicesOfCandidates.data(),
            totalNumCandidates,
            numTasks,
            d_fullyUsedReadIds.data(),
            d_numFullyUsedReadIdsPerAnchor.data(),
            d_numFullyUsedReadIdsPerAnchorPrefixSum.data(),
            d_segmentIdsOfFullyUsedReadIds.data(),
            *h_numFullyUsedReadIds,
            numTasks,        
            d_candidateReadIds.data(),
            d_numCandidatesPerAnchor.data(),
            d_anchorIndicesOfCandidates2,
            numTasks,
            firstStream
        );

        #endif


        cubAllocator->DeviceFree(d_candidateReadIds2); CUERR;

        totalNumCandidates = std::distance(d_candidateReadIds.data(), d_candidateReadIds_end);

        assert(d_candidateReadIds.size() >= totalNumCandidates);

        cudaMemcpyAsync(
            d_anchorIndicesOfCandidates.data(),
            d_anchorIndicesOfCandidates2,
            sizeof(int) * totalNumCandidates,
            D2D,
            firstStream
        ); CUERR;

        cubAllocator->DeviceFree(d_anchorIndicesOfCandidates2); CUERR;

        //compute prefix sum of new segment sizes
    
        cubstatus = cub::DeviceScan::InclusiveSum(
            cubTemp.data(), 
            cubBytes, 
            d_numCandidatesPerAnchor.data(), 
            d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            numTasks,
            firstStream
        );
        assert(cubstatus == cudaSuccess);


        if(numTasksWithMateRemoved > 0){
            cudaEventRecord(events[0], secondStream);
            cudaStreamWaitEvent(firstStream, events[0], 0); CUERR;
        }

        //removeUsedIdsAndMateIds is a compaction step. check early exit.
        if(totalNumCandidates == 0){
            setStateToFinished();
        }else{
            setState(BatchData::State::BeforeComputePairFlags);
        }
    }

    void computePairFlagsGpu() {
        assert(state == BatchData::State::BeforeComputePairFlags);

        cudaStream_t stream = streams[0];
        DEBUGDEVICESYNC

        d_isPairedCandidate.resizeWithoutCopy(totalNumCandidates, stream);

        helpers::call_fill_kernel_async(d_isPairedCandidate.data(), totalNumCandidates, false, stream);

        DEBUGDEVICESYNC

        h_firstTasksOfPairsToCheck.resize(numTasks);
        int numChecks = 0;

        for(int first = 0, second = 1; second < numTasks; ){
            const int taskindex1 = first;
            const int taskindex2 = second;

            const bool areConsecutiveTasks = tasks[taskindex1].id + 1 == tasks[taskindex2].id;
            const bool arePairedTasks = (tasks[taskindex1].id % 2) + 1 == (tasks[taskindex2].id % 2);

            if(areConsecutiveTasks && arePairedTasks){
                h_firstTasksOfPairsToCheck[numChecks] = first;
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

            cudaMemcpyAsync(
                d_firstTasksOfPairsToCheck,
                h_firstTasksOfPairsToCheck.data(),
                sizeof(int) * numChecks,
                H2D,
                stream
            ); CUERR;

            DEBUGDEVICESYNC

            

            dim3 block = 128;
            dim3 grid = numChecks;

            helpers::lambda_kernel<<<grid, block, 0, stream>>>(
                [
                    numChecks,
                    d_firstTasksOfPairsToCheck,
                    d_numCandidatesPerAnchor = d_numCandidatesPerAnchor.data(),
                    d_numCandidatesPerAnchorPrefixSum = d_numCandidatesPerAnchorPrefixSum.data(), // numTasks + 1
                    d_numCandidatesPerAnchorPrefixSumsize = d_numCandidatesPerAnchorPrefixSum.size(),
                    d_candidateReadIds = d_candidateReadIds.data(),
                    d_candidateReadIdssize = d_candidateReadIds.size(),
                    d_numUsedReadIdsPerAnchor = d_numUsedReadIdsPerAnchor.data(),
                    d_numUsedReadIdsPerAnchorsize = d_numUsedReadIdsPerAnchor.size(),
                    d_numUsedReadIdsPerAnchorPrefixSum = d_numUsedReadIdsPerAnchorPrefixSum.data(), // numTasks
                    d_numUsedReadIdsPerAnchorPrefixSumsize = d_numUsedReadIdsPerAnchorPrefixSum.size(), // numTasks
                    d_usedReadIds = d_usedReadIds.data(),
                    d_usedReadIdssize = d_usedReadIds.size(),

                    d_isPairedCandidate = d_isPairedCandidate.data()
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
                        bool* output,
                        const read_number* boundary1,
                        const read_number* boundary2,
                        const bool* boundaryoutput
                    ){
                        const int numIterations = SDIV(numElements2, numSharedElements);

                        for(int iteration = 0; iteration < numIterations; iteration++){

                            const int begin = iteration * numSharedElements;
                            const int end = min((iteration+1) * numSharedElements, numElements2);
                            const int num = end - begin;

                            for(int i = threadIdx.x; i < num; i += blockDim.x){
                                assert(array2 + begin + i < boundary2);
                                sharedElements[i] = array2[begin + i];
                            }

                            __syncthreads();

                            //TODO in iteration > 0, we may skip elements at the beginning of first range

                            for(int i = threadIdx.x; i < numElements1; i += blockDim.x){
                                assert(output + i < boundaryoutput);
                                if(!output[i]){
                                    assert(array1 + i < boundary1);
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

                    auto process2 = [&](
                        const read_number* array1,
                        int numElements1,
                        const read_number* array2,
                        int numElements2,
                        bool* output,
                        const read_number* boundary1,
                        const read_number* boundary2,
                        const bool* boundaryoutput
                    ){
                        const int numIterations = SDIV(numElements2, numSharedElements);

                        for(int iteration = 0; iteration < numIterations; iteration++){

                            const int begin = iteration * numSharedElements;
                            const int end = min((iteration+1) * numSharedElements, numElements2);
                            const int num = end - begin;

                            for(int i = threadIdx.x; i < num; i += blockDim.x){
                                assert(array2 + begin + i < boundary2);
                                sharedElements[i] = array2[begin + i];
                            }

                            __syncthreads();

                            //TODO in iteration > 0, we may skip elements at the beginning of first range

                            for(int i = threadIdx.x; i < numElements1; i += blockDim.x){
                                assert(output + i < boundaryoutput);
                                if(!output[i]){
                                    assert(array1 + i < boundary1);
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
                        assert(firstTask < d_numCandidatesPerAnchorPrefixSumsize);
                        assert(firstTask+2 < d_numCandidatesPerAnchorPrefixSumsize);
                        assert(firstTask+2 < d_numCandidatesPerAnchorPrefixSumsize);
                        const int rangeBegin = d_numCandidatesPerAnchorPrefixSum[firstTask];                        
                        const int rangeMid = d_numCandidatesPerAnchorPrefixSum[firstTask + 1];
                        const int rangeEnd = d_numCandidatesPerAnchorPrefixSum[firstTask + 2];

                        process(
                            d_candidateReadIds + rangeBegin,
                            rangeMid - rangeBegin,
                            d_candidateReadIds + rangeMid,
                            rangeEnd - rangeMid,
                            d_isPairedCandidate + rangeBegin,
                            d_candidateReadIds + d_candidateReadIdssize,
                            d_candidateReadIds + d_candidateReadIdssize,
                            d_isPairedCandidate + d_candidateReadIdssize
                        );

                        process(
                            d_candidateReadIds + rangeMid,
                            rangeEnd - rangeMid,
                            d_candidateReadIds + rangeBegin,
                            rangeMid - rangeBegin,
                            d_isPairedCandidate + rangeMid,
                            d_candidateReadIds + d_candidateReadIdssize,
                            d_candidateReadIds + d_candidateReadIdssize,
                            d_isPairedCandidate + d_candidateReadIdssize
                        );

                        //check for pairs in candidates of previous extension iterations

                        assert(firstTask < d_numUsedReadIdsPerAnchorPrefixSumsize);
                        assert(firstTask+1 < d_numUsedReadIdsPerAnchorPrefixSumsize);
                        assert(firstTask+1 < d_numUsedReadIdsPerAnchorsize);

                        const int usedRangeBegin = d_numUsedReadIdsPerAnchorPrefixSum[firstTask];                        
                        const int usedRangeMid = d_numUsedReadIdsPerAnchorPrefixSum[firstTask + 1];
                        const int usedRangeEnd = usedRangeMid + d_numUsedReadIdsPerAnchor[firstTask + 1];

                        process2(
                            d_candidateReadIds + rangeBegin,
                            rangeMid - rangeBegin,
                            d_usedReadIds + usedRangeMid,
                            usedRangeEnd - usedRangeMid,
                            d_isPairedCandidate + rangeBegin,
                            d_candidateReadIds + d_candidateReadIdssize,
                            d_usedReadIds + d_usedReadIdssize,
                            d_isPairedCandidate + d_candidateReadIdssize
                        );

                        process2(
                            d_candidateReadIds + rangeMid,
                            rangeEnd - rangeMid,
                            d_usedReadIds + usedRangeBegin,
                            usedRangeMid - usedRangeBegin,
                            d_isPairedCandidate + rangeMid,
                            d_candidateReadIds + d_candidateReadIdssize,
                            d_usedReadIds + d_usedReadIdssize,
                            d_isPairedCandidate + d_candidateReadIdssize
                        );
                    }
                }
            ); CUERR;

            DEBUGDEVICESYNC

            cubAllocator->DeviceFree(d_firstTasksOfPairsToCheck); CUERR;

            DEBUGDEVICESYNC

        }

        setState(BatchData::State::BeforeLoadCandidates);

    }

    void loadCandidateSequenceData() {
        assert(state == BatchData::State::BeforeLoadCandidates);

        cudaStream_t stream = streams[0];

        d_candidateSequencesLength.resizeWithoutCopy(totalNumCandidates, stream);
        d_candidateSequencesData.resizeWithoutCopy(encodedSequencePitchInInts * totalNumCandidates, stream);

        assert(d_candidateReadIds.size() >= totalNumCandidates);

        gpuReadStorage->gatherSequences(
            readStorageHandle,
            d_candidateSequencesData.data(),
            encodedSequencePitchInInts,
            h_candidateReadIds.data(),
            d_candidateReadIds.data(), //device accessible
            totalNumCandidates,
            stream
        );

        gpuReadStorage->gatherSequenceLengths(
            readStorageHandle,
            d_candidateSequencesLength.data(),
            d_candidateReadIds.data(),
            totalNumCandidates,
            stream
        );

        setState(BatchData::State::BeforeEraseData);
    }

    void eraseDataOfRemovedMates(){
        assert(state == BatchData::State::BeforeEraseData);

        if(numTasksWithMateRemoved > 0){

            cudaStream_t stream = streams[0];

            CachedDeviceUVector<unsigned int> d_candidateSequencesData2(encodedSequencePitchInInts * totalNumCandidates, stream, *cubAllocator);

            read_number* d_candidateReadIds2 = nullptr;
            cubAllocator->DeviceAllocate((void**)&d_candidateReadIds2, sizeof(read_number) * totalNumCandidates, stream); CUERR;

            int* d_candidateSequencesLength2 = nullptr;
            cubAllocator->DeviceAllocate((void**)&d_candidateSequencesLength2, sizeof(int) * totalNumCandidates, stream); CUERR;

            bool* d_isPairedCandidate2 = nullptr;
            cubAllocator->DeviceAllocate((void**)&d_isPairedCandidate2, sizeof(bool) * totalNumCandidates, stream); CUERR;

            int* d_anchorIndicesOfCandidates2 = nullptr;
            cubAllocator->DeviceAllocate((void**)&d_anchorIndicesOfCandidates2, sizeof(int) * totalNumCandidates, stream);


            constexpr int groupsize = 32;
            constexpr int blocksize = 128;
            constexpr int groupsperblock = blocksize / groupsize;
            dim3 block(blocksize,1,1);
            dim3 grid(SDIV(numTasksWithMateRemoved * groupsize, blocksize), 1, 1);
            const std::size_t smembytes = sizeof(unsigned int) * groupsperblock * encodedSequencePitchInInts;

            bool* d_keepflags = nullptr;
            cubAllocator->DeviceAllocate((void**)&d_keepflags, sizeof(bool) * totalNumCandidates, stream); CUERR;

            helpers::call_fill_kernel_async(d_keepflags, totalNumCandidates, true, stream);

            readextendergpukernels::filtermatekernel<blocksize,groupsize><<<grid, block, smembytes, stream>>>(
                d_anchormatedata.data(),
                d_candidateSequencesData.data(),
                encodedSequencePitchInInts,
                d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchorPrefixSum.data(),
                d_anchorIndicesWithRemovedMates.data(),
                numTasksWithMateRemoved,
                d_keepflags
            ); CUERR;

            int* d_outputpositions = nullptr;
            cubAllocator->DeviceAllocate((void**)&d_outputpositions, sizeof(int) * totalNumCandidates, stream); CUERR;

            CachedDeviceUVector<char> cubTemp(*cubAllocator);
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

            cubTemp.resizeWithoutCopy(cubTempSize, stream);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                cubTemp.data(),
                cubTempSize,
                d_keepflags, 
                d_outputpositions, 
                totalNumCandidates, 
                stream
            );
            assert(cudaSuccess == cubstatus);

            helpers::lambda_kernel<<<4096, 128, 0, stream>>>(
                [
                    numTasks = numTasks,
                    encodedSequencePitchInInts = encodedSequencePitchInInts,
                    d_numCandidatesPerAnchor = d_numCandidatesPerAnchor.data(),
                    d_numCandidatesPerAnchorPrefixSum = d_numCandidatesPerAnchorPrefixSum.data(),
                    d_keepflags,
                    d_outputpositions = d_outputpositions,
                    d_candidateReadIds = d_candidateReadIds.data(),
                    d_candidateSequencesLength = d_candidateSequencesLength.data(),
                    d_candidateSequencesData = d_candidateSequencesData.data(),
                    d_anchorIndicesOfCandidates = d_anchorIndicesOfCandidates.data(),
                    d_isPairedCandidate = d_isPairedCandidate.data(),
                    d_candidateReadIdsOut = d_candidateReadIds2,
                    d_candidateSequencesLengthOut = d_candidateSequencesLength2,
                    d_candidateSequencesDataOut = d_candidateSequencesData2.data(),
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
                d_numCandidatesPerAnchor.data(), 
                d_numCandidatesPerAnchorPrefixSum.data() + 1, 
                numTasks, 
                stream
            );
            assert(cudaSuccess == cubstatus);

            cubTemp.resizeWithoutCopy(cubTempSize, stream);

            cubstatus = cub::DeviceScan::InclusiveSum(
                cubTemp.data(),
                cubTempSize,
                d_numCandidatesPerAnchor.data(), 
                d_numCandidatesPerAnchorPrefixSum.data() + 1, 
                numTasks, 
                stream
            );
            assert(cudaSuccess == cubstatus);

            cudaMemcpyAsync(
                h_numCandidates.data(),
                d_numCandidatesPerAnchorPrefixSum.data() + numTasks,
                sizeof(int),
                D2H,
                stream
            ); CUERR;

            cudaStreamSynchronize(stream); CUERR;

            totalNumCandidates = *h_numCandidates;

            assert(d_candidateReadIds.size() >= totalNumCandidates);

            helpers::call_copy_n_kernel(
                thrust::make_zip_iterator(thrust::make_tuple(
                    d_candidateReadIds2,
                    d_candidateSequencesLength2,
                    d_isPairedCandidate2,
                    d_anchorIndicesOfCandidates2
                )),
                totalNumCandidates,
                thrust::make_zip_iterator(thrust::make_tuple(
                    d_candidateReadIds.data(),
                    d_candidateSequencesLength.data(),
                    d_isPairedCandidate.data(),
                    d_anchorIndicesOfCandidates.data()
                )),
                stream
            );

            cubAllocator->DeviceFree(d_candidateReadIds2); CUERR;
            cubAllocator->DeviceFree(d_candidateSequencesLength2); CUERR;
            cubAllocator->DeviceFree(d_isPairedCandidate2); CUERR;
            cubAllocator->DeviceFree(d_anchorIndicesOfCandidates2); CUERR;

            std::swap(d_candidateSequencesData2, d_candidateSequencesData); 
        }

        setState(BatchData::State::BeforeAlignment);
    }

    void calculateAlignments(){
        assert(state == BatchData::State::BeforeAlignment);

        cudaStream_t stream = streams[0];


        d_alignment_overlaps.resizeWithoutCopy(totalNumCandidates, stream);
        d_alignment_shifts.resizeWithoutCopy(totalNumCandidates, stream);
        d_alignment_nOps.resizeWithoutCopy(totalNumCandidates, stream);
        d_alignment_best_alignment_flags.resizeWithoutCopy(totalNumCandidates, stream);

        bool* d_alignment_isValid = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_alignment_isValid, sizeof(bool) * totalNumCandidates, stream); CUERR;
        
        h_numAnchors[0] = numTasks;

        const bool* const d_anchorContainsN = nullptr;
        const bool* const d_candidateContainsN = nullptr;
        const bool removeAmbiguousAnchors = false;
        const bool removeAmbiguousCandidates = false;
        const int maxNumAnchors = numTasks;
        const int maxNumCandidates = totalNumCandidates; //this does not need to be exact, but it must be >= d_numCandidatesPerAnchorPrefixSum[numTasks]
        const int maximumSequenceLength = encodedSequencePitchInInts * 16;
        const int encodedSequencePitchInInts2Bit = encodedSequencePitchInInts;
        const int min_overlap = goodAlignmentProperties->min_overlap;
        const float maxErrorRate = goodAlignmentProperties->maxErrorRate;
        const float min_overlap_ratio = goodAlignmentProperties->min_overlap_ratio;
        const float estimatedNucleotideErrorRate = correctionOptions->estimatedErrorrate;

        auto callAlignmentKernel = [&](void* d_tempstorage, size_t& tempstoragebytes){

            call_popcount_rightshifted_hamming_distance_kernel_async(
                d_tempstorage,
                tempstoragebytes,
                d_alignment_overlaps.data(),
                d_alignment_shifts.data(),
                d_alignment_nOps.data(),
                d_alignment_isValid,
                d_alignment_best_alignment_flags.data(),
                d_subjectSequencesData.data(),
                d_candidateSequencesData.data(),
                d_anchorSequencesLength.data(),
                d_candidateSequencesLength.data(),
                d_numCandidatesPerAnchorPrefixSum.data(),
                d_numCandidatesPerAnchor.data(),
                d_anchorIndicesOfCandidates.data(),
                h_numAnchors.data(),
                &d_numCandidatesPerAnchorPrefixSum[numTasks],
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

        CachedDeviceUVector<char> d_tempstorage(tempstoragebytes, stream, *cubAllocator);

        callAlignmentKernel(d_tempstorage.data(), tempstoragebytes);

        cubAllocator->DeviceFree(d_alignment_isValid); CUERR;

        setState(BatchData::State::BeforeAlignmentFilter);
    }

    void filterAlignments(){
        assert(state == BatchData::State::BeforeAlignmentFilter);

        cudaStream_t stream = streams[0];


        DEBUGDEVICESYNC

        const int numAnchors = numTasks;

        CachedDeviceUVector<int> d_numCandidatesPerAnchor2(numTasks, stream, *cubAllocator);

        h_numCandidates.resize(1);

        CachedDeviceUVector<unsigned int> d_candidateSequencesData2(encodedSequencePitchInInts * totalNumCandidates, stream, *cubAllocator);

        bool* d_keepflags = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_keepflags, sizeof(bool) * totalNumCandidates, stream); CUERR; 

        helpers::call_fill_kernel_async(d_keepflags, totalNumCandidates, true, stream);

        DEBUGDEVICESYNC

        dim3 block(128,1,1);
        dim3 grid(numAnchors, 1, 1);

        #if 0
            //filter alignments of candidates. d_keepflags[i] will be set to false if candidate[i] should be removed
        //d_numCandidatesPerAnchor2[i] contains new number of candidates for anchor i
        helpers::lambda_kernel<<<grid, block, 0, stream>>>(
            [
                d_alignment_best_alignment_flags = d_alignment_best_alignment_flags.data(),
                d_alignment_shifts = d_alignment_shifts.data(),
                d_alignment_overlaps = d_alignment_overlaps.data(),
                d_alignment_nOps = d_alignment_nOps.data(),
                d_anchorSequencesLength = d_anchorSequencesLength.data(),
                d_numCandidatesPerAnchor = d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchor2 = d_numCandidatesPerAnchor2.data(),
                d_numCandidatesPerAnchorPrefixSum = d_numCandidatesPerAnchorPrefixSum.data(),
                d_isPairedCandidate = d_isPairedCandidate.data(),
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
        //d_numCandidatesPerAnchor2[i] contains new number of candidates for anchor i
        helpers::lambda_kernel<<<grid, block, 0, stream>>>(
            [
                d_alignment_best_alignment_flags = d_alignment_best_alignment_flags.data(),
                d_alignment_shifts = d_alignment_shifts.data(),
                d_alignment_overlaps = d_alignment_overlaps.data(),
                d_anchorSequencesLength = d_anchorSequencesLength.data(),
                d_numCandidatesPerAnchor = d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchor2 = d_numCandidatesPerAnchor2.data(),
                d_numCandidatesPerAnchorPrefixSum = d_numCandidatesPerAnchorPrefixSum.data(),
                d_isPairedCandidate = d_isPairedCandidate.data(),
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
                d_alignment_nOps.data(),
                d_alignment_overlaps.data(),
                d_alignment_shifts.data(),
                d_alignment_best_alignment_flags.data(),
                d_candidateReadIds.data(),
                d_candidateSequencesLength.data(),
                d_isPairedCandidate.data()
            )
        );

        int* d_alignment_overlaps2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_alignment_overlaps2, sizeof(int) * totalNumCandidates, stream); CUERR;

        int* d_alignment_shifts2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_alignment_shifts2, sizeof(int) * totalNumCandidates, stream); CUERR;

        int* d_alignment_nOps2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_alignment_nOps2, sizeof(int) * totalNumCandidates, stream); CUERR;

        BestAlignment_t* d_alignment_best_alignment_flags2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_alignment_best_alignment_flags2, sizeof(BestAlignment_t) * totalNumCandidates, stream); CUERR;

        int* d_candidateSequencesLength2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_candidateSequencesLength2, sizeof(int) * totalNumCandidates, stream); CUERR;

        read_number* d_candidateReadIds2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_candidateReadIds2, sizeof(read_number) * totalNumCandidates, stream); CUERR;

        bool* d_isPairedCandidate2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_isPairedCandidate2, sizeof(bool) * totalNumCandidates, stream); CUERR;

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

        assert(d_alignment_nOps.size() >= totalNumCandidates);
        assert(d_alignment_overlaps.size() >= totalNumCandidates);
        assert(d_alignment_shifts.size() >= totalNumCandidates);
        assert(d_alignment_best_alignment_flags.size() >= totalNumCandidates);
        assert(d_candidateReadIds.size() >= totalNumCandidates);
        assert(d_candidateSequencesLength.size() >= totalNumCandidates);
        assert(d_isPairedCandidate.size() >= totalNumCandidates);


        assert(d_alignment_nOps2 != nullptr);
        assert(d_alignment_overlaps2 != nullptr);
        assert(d_alignment_shifts2 != nullptr);
        assert(d_alignment_best_alignment_flags2 != nullptr);
        assert(d_candidateReadIds2 != nullptr);
        assert(d_candidateSequencesLength2 != nullptr);
        assert(d_isPairedCandidate2 != nullptr);

        //compact 1d arrays

        CachedDeviceUVector<char> cubTemp(*cubAllocator);
        std::size_t cubTempSize = 0;

        cudaError_t cubstatus = cub::DeviceSelect::Flagged(
            nullptr, 
            cubTempSize, 
            d_zip_data, 
            d_keepflags, 
            d_zip_data_tmp, 
            h_numCandidates.data(), 
            totalNumCandidates, 
            stream
        );
        assert(cubstatus == cudaSuccess);

        cubTemp.resizeWithoutCopy(cubTempSize, stream);

        cubstatus = cub::DeviceSelect::Flagged(
            cubTemp.data(), 
            cubTempSize, 
            d_zip_data, 
            d_keepflags, 
            d_zip_data_tmp, 
            h_numCandidates.data(), 
            totalNumCandidates, 
            stream
        );
        assert(cubstatus == cudaSuccess);

        cudaEventRecord(events[0], stream); CUERR;

        //compact 2d candidate sequences
        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            cubTempSize,
            d_candidateSequencesData.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                SequenceFlagMultiplier{d_keepflags, int(encodedSequencePitchInInts)}
            ),
            d_candidateSequencesData2.data(),
            thrust::make_discard_iterator(),
            totalNumCandidates * encodedSequencePitchInInts,
            stream
        );
        assert(cubstatus == cudaSuccess);

        cubTemp.resizeWithoutCopy(cubTempSize, stream);

        cubstatus = cub::DeviceSelect::Flagged(
            cubTemp.data(),
            cubTempSize,
            d_candidateSequencesData.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                SequenceFlagMultiplier{d_keepflags, int(encodedSequencePitchInInts)}
            ),
            d_candidateSequencesData2.data(),
            thrust::make_discard_iterator(), //number of remaining candidates already known from previous compaction call
            totalNumCandidates * encodedSequencePitchInInts,
            stream
        );


        std::swap(d_candidateSequencesData2, d_candidateSequencesData);

        cubAllocator->DeviceFree(d_keepflags); CUERR;

        DEBUGDEVICESYNC


        //compute prefix sum of new number of candidates per anchor
        cubstatus = cub::DeviceScan::InclusiveSum(
            nullptr,
            cubTempSize,
            d_numCandidatesPerAnchor2.data(), 
            d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            numTasks, 
            stream
        );
        assert(cudaSuccess == cubstatus);

        cubTemp.resizeWithoutCopy(cubTempSize, stream);

        cubstatus = cub::DeviceScan::InclusiveSum(
            cubTemp.data(),
            cubTempSize,
            d_numCandidatesPerAnchor2.data(), 
            d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            numTasks, 
            stream
        );
        assert(cudaSuccess == cubstatus);

        std::swap(d_numCandidatesPerAnchor2, d_numCandidatesPerAnchor);

        DEBUGDEVICESYNC

        // {
        //     cudaDeviceSynchronize(); CUERR; 

        //     std::vector<int> offsets(numTasks + 1);
        //     cudaMemcpyAsync(
        //         offsets.data(),
        //         d_numCandidatesPerAnchorPrefixSum.data(),
        //         sizeof(int) * (numTasks + 1),
        //         D2H,
        //         stream
        //     );

        //     cudaDeviceSynchronize(); CUERR;
        //     std::cerr << "Offsets after filteralignment:\n";
        //     for(int i = 0; i < numTasks+1; i++){
        //         std::cerr << offsets[i] << " ";
        //     }
        //     std::cerr << "\n";

        // }
        


        cudaEventSynchronize(events[0]); CUERR;
        totalNumCandidates = *h_numCandidates;

        helpers::call_copy_n_kernel(
            d_zip_data_tmp,
            totalNumCandidates,
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

        //filterAlignments is a compaction step. check early exit.
        if(totalNumCandidates == 0){
            setStateToFinished();
        }else{
            setState(BatchData::State::BeforeMSA);
        }
    }

    void computeMSAs(){
        assert(state == BatchData::State::BeforeMSA);

        cudaStream_t firstStream = streams[0];
        //cudaStream_t secondStream = firstStream;

        char* d_candidateQualityScores = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_candidateQualityScores, sizeof(char) * qualityPitchInBytes * totalNumCandidates, firstStream); CUERR;

        loadCandidateQualityScores(firstStream, d_candidateQualityScores);


        d_consensusEncoded.resize(numTasks * msaColumnPitchInElements);
        d_coverage.resize(numTasks * msaColumnPitchInElements);
        d_msa_column_properties.resize(numTasks);

        int* d_counts = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_counts, sizeof(int) * numTasks * 4 * msaColumnPitchInElements, firstStream); CUERR;

        float* d_weights = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_weights, sizeof(float) * numTasks * 4 * msaColumnPitchInElements, firstStream); CUERR;

        int* d_origCoverages = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_origCoverages, sizeof(int) * numTasks * msaColumnPitchInElements, firstStream); CUERR;

        float* d_origWeights = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_origWeights, sizeof(float) * numTasks * msaColumnPitchInElements, firstStream); CUERR;

        float* d_support = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_support, sizeof(float) * numTasks * msaColumnPitchInElements, firstStream); CUERR;

        d_consensusQuality.resize(numTasks * msaColumnPitchInElements);

        CachedDeviceUVector<int> d_numCandidatesPerAnchor2(numTasks, firstStream, *cubAllocator);

        int* indices1 = nullptr; 
        int* indices2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&indices1, sizeof(int) * totalNumCandidates, firstStream); CUERR;
        cubAllocator->DeviceAllocate((void**)&indices2, sizeof(int) * totalNumCandidates, firstStream); CUERR;

        

        helpers::lambda_kernel<<<numTasks, 128, 0, firstStream>>>(
            [
                indices1,
                d_numCandidatesPerAnchor = d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchorPrefixSum = d_numCandidatesPerAnchorPrefixSum.data()
            ] __device__ (){
                const int num = d_numCandidatesPerAnchor[blockIdx.x];
                const int offset = d_numCandidatesPerAnchorPrefixSum[blockIdx.x];
                
                for(int i = threadIdx.x; i < num; i += blockDim.x){
                    indices1[offset + i] = i;
                }
            }
        );

        gpu::GPUMultiMSA multiMSA;

        *h_numAnchors = numTasks;

        multiMSA.numMSAs = numTasks;
        multiMSA.columnPitchInElements = msaColumnPitchInElements;
        multiMSA.counts = d_counts;
        multiMSA.weights = d_weights;
        multiMSA.coverages = d_coverage.data();
        multiMSA.consensus = d_consensusEncoded.data();
        multiMSA.support = d_support;
        multiMSA.origWeights = d_origWeights;
        multiMSA.origCoverages = d_origCoverages;
        multiMSA.columnProperties = d_msa_column_properties.data();

        const bool useQualityScoresForMSA = true;

        callConstructMultipleSequenceAlignmentsKernel_async(
            multiMSA,
            d_alignment_overlaps.data(),
            d_alignment_shifts.data(),
            d_alignment_nOps.data(),
            d_alignment_best_alignment_flags.data(),
            d_anchorSequencesLength.data(),
            d_candidateSequencesLength.data(),
            indices1, //d_indices,
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_subjectSequencesData.data(),
            d_candidateSequencesData.data(),
            d_isPairedCandidate.data(),
            d_anchorQualityScores.data(), //d_anchor_qualities.data(),
            d_candidateQualityScores,
            h_numAnchors.data(), //d_numAnchors
            goodAlignmentProperties->maxErrorRate,
            numTasks,
            totalNumCandidates,
            useQualityScoresForMSA, //correctionOptions->useQualityScores,
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            firstStream,
            kernelLaunchHandle
        );

        //refine msa
        bool* d_shouldBeKept = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_shouldBeKept, sizeof(bool) * totalNumCandidates, firstStream); CUERR;

        callMsaCandidateRefinementKernel_multiiter_async(
            indices2,
            d_numCandidatesPerAnchor2.data(),
            d_numCandidates2.data(),
            multiMSA,
            d_alignment_best_alignment_flags.data(),
            d_alignment_shifts.data(),
            d_alignment_nOps.data(),
            d_alignment_overlaps.data(),
            d_subjectSequencesData.data(),
            d_candidateSequencesData.data(),
            d_isPairedCandidate.data(),
            d_anchorSequencesLength.data(),
            d_candidateSequencesLength.data(),
            d_anchorQualityScores.data(), //d_anchor_qualities.data(),
            d_candidateQualityScores,
            d_shouldBeKept,
            d_numCandidatesPerAnchorPrefixSum.data(),
            h_numAnchors.data(),
            goodAlignmentProperties->maxErrorRate,
            numTasks,
            totalNumCandidates,
            useQualityScoresForMSA, //correctionOptions->useQualityScores,
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            indices1, //d_indices,
            d_numCandidatesPerAnchor.data(),
            correctionOptions->estimatedCoverage,
            getNumRefinementIterations(),
            firstStream,
            kernelLaunchHandle
        );

        cubAllocator->DeviceFree(d_candidateQualityScores); CUERR;

        helpers::call_fill_kernel_async(d_shouldBeKept, totalNumCandidates, false, firstStream); CUERR;

        //convert output indices from task-local indices to global flags
        helpers::lambda_kernel<<<numTasks, 128, 0, firstStream>>>(
            [
                d_flagscandidates = d_shouldBeKept,
                indices2,
                d_numCandidatesPerAnchor2 = d_numCandidatesPerAnchor2.data(),
                d_numCandidatesPerAnchorPrefixSum = d_numCandidatesPerAnchorPrefixSum.data()
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


        CachedDeviceUVector<char> cubTemp(*cubAllocator);
        std::size_t cubBytes = 0;
        cudaError_t cubstatus = cudaSuccess;

        read_number* d_candidateReadIds2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_candidateReadIds2, sizeof(read_number) * totalNumCandidates, firstStream); CUERR;

        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            cubBytes,
            d_candidateReadIds.data(),
            d_shouldBeKept,
            d_candidateReadIds2,
            h_numCandidates.data(),
            totalNumCandidates,
            firstStream
        );

        cubTemp.resizeWithoutCopy(cubBytes, firstStream);

        cubstatus = cub::DeviceSelect::Flagged(
            cubTemp.data(),
            cubBytes,
            d_candidateReadIds.data(),
            d_shouldBeKept,
            d_candidateReadIds2,
            h_numCandidates.data(),
            totalNumCandidates,
            firstStream
        );

        assert(cubstatus == cudaSuccess);

        cudaEventRecord(events[0], firstStream); CUERR;



        //compute quality of consensus
        helpers::lambda_kernel<<<numTasks, 256, 0, firstStream>>>(
            [
                consensusQuality = d_consensusQuality.data(),
                support = d_support,
                coverages = d_coverage.data(),
                msa_column_properties = d_msa_column_properties.data(),
                d_numCandidatesInMsa = d_numCandidatesPerAnchor2.data(),
                columnPitchInElements = msaColumnPitchInElements,
                numTasks = numTasks
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
            d_numCandidatesPerAnchor2.data(), 
            d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            numTasks, 
            firstStream
        );
        assert(cubstatus == cudaSuccess);

        cubTemp.resizeWithoutCopy(cubBytes, firstStream);

        cubstatus = cub::DeviceScan::InclusiveSum(
            cubTemp.data(),
            cubBytes,
            d_numCandidatesPerAnchor2.data(), 
            d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            numTasks, 
            firstStream
        );
        assert(cubstatus == cudaSuccess);

        cudaEventSynchronize(events[0]); CUERR; //wait for h_numCandidates



        //only information about number of candidates and readids are kept. all other information about candidates is discarded
        //auto oldnum = totalNumCandidates;
        totalNumCandidates = *h_numCandidates; 

        std::swap(d_numCandidatesPerAnchor, d_numCandidatesPerAnchor2);        

        cudaMemcpyAsync(
            d_candidateReadIds.data(),
            d_candidateReadIds2,
            sizeof(read_number) * totalNumCandidates,
            D2D,
            firstStream
        ); CUERR;

        cubAllocator->DeviceFree(d_candidateReadIds2); CUERR;

        setState(BatchData::State::BeforeExtend);
    }


    void computeExtendedSequencesFromMSAs(){
        assert(state == BatchData::State::BeforeExtend);

        cudaStream_t stream = streams[0];

        outputAnchorPitchInBytes = SDIV(decodedSequencePitchInBytes, 128) * 128;
        outputAnchorQualityPitchInBytes = SDIV(qualityPitchInBytes, 128) * 128;
        decodedMatesRevCPitchInBytes = SDIV(decodedSequencePitchInBytes, 128) * 128;

        h_accumExtensionsLengths.resize(numTasks);
        h_abortReasons.resize(numTasks);
        h_outputAnchors.resize(numTasks * outputAnchorPitchInBytes);
        h_outputAnchorQualities.resize(numTasks * outputAnchorQualityPitchInBytes);
        h_outputAnchorLengths.resize(numTasks);
        h_outputMateHasBeenFound.resize(numTasks);
        h_sizeOfGapToMate.resize(numTasks);
        h_isFullyUsedCandidate.resize(totalNumCandidates);

        int* d_accumExtensionsLengths = nullptr;
        int* d_accumExtensionsLengthsOUT = nullptr;
        int* d_sizeOfGapToMate = nullptr;

        cubAllocator->DeviceAllocate((void**)&d_accumExtensionsLengths, sizeof(int) * numTasks, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_accumExtensionsLengthsOUT, sizeof(int) * numTasks, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_sizeOfGapToMate, sizeof(int) * numTasks, stream); CUERR;
        
        d_isFullyUsedCandidate.resize(totalNumCandidates);
        d_outputAnchors.resize(numTasks * outputAnchorPitchInBytes);
        d_outputAnchorQualities.resize(numTasks * outputAnchorQualityPitchInBytes);
        d_outputMateHasBeenFound.resize(numTasks);
        d_abortReasons.resize(numTasks);
        d_outputAnchorLengths.resize(numTasks);      

        helpers::call_fill_kernel_async(d_outputMateHasBeenFound.data(), numTasks, false, stream); CUERR;
        helpers::call_fill_kernel_async(d_abortReasons.data(), numTasks, extension::AbortReason::None, stream); CUERR;
        helpers::call_fill_kernel_async(d_isFullyUsedCandidate.data(), totalNumCandidates, false, stream); CUERR;


        for(int i = 0; i < numTasks; i++){
            const int index = i;
            const auto& task = tasks[index];

            h_accumExtensionsLengths[i] = task.accumExtensionLengths;
        }

        cudaMemcpyAsync(
            d_accumExtensionsLengths,
            h_accumExtensionsLengths.data(),
            sizeof(int) * numTasks,
            H2D,
            stream
        ); CUERR;
       
        //compute extensions
           
        helpers::lambda_kernel<<<numTasks, 128, 0, stream>>>(
            [
                numTasks = numTasks,
                insertSize = insertSize,
                insertSizeStddev = insertSizeStddev,
                msaColumnPitchInElements = msaColumnPitchInElements,
                d_numCandidatesPerAnchor = d_numCandidatesPerAnchor.data(),
                d_msa_column_properties = d_msa_column_properties.data(),
                d_consensusEncoded = d_consensusEncoded.data(),
                d_consensusQuality = d_consensusQuality.data(),
                d_coverage = d_coverage.data(),
                d_anchorSequencesLength = d_anchorSequencesLength.data(),
                d_accumExtensionsLengths = (int*)d_accumExtensionsLengths,
                d_inputMateLengths = (int*)d_inputMateLengths.data(),
                d_abortReasons = d_abortReasons.data(),
                d_accumExtensionsLengthsOUT = (int*)d_accumExtensionsLengthsOUT,
                d_outputAnchors = (char*)d_outputAnchors,
                outputAnchorPitchInBytes = outputAnchorPitchInBytes,
                d_outputAnchorQualities = (char*)d_outputAnchorQualities,
                outputAnchorQualityPitchInBytes = outputAnchorQualityPitchInBytes,
                d_outputAnchorLengths = d_outputAnchorLengths.data(),
                d_isPairedTask = (bool*)d_isPairedTask.data(),
                d_inputanchormatedata = d_inputanchormatedata.data(),
                encodedSequencePitchInInts = encodedSequencePitchInInts,
                decodedMatesRevCPitchInBytes = decodedMatesRevCPitchInBytes,
                d_outputMateHasBeenFound = d_outputMateHasBeenFound.data(),
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

                constexpr int smemEncodedMateInts = 32;
                __shared__ unsigned int smemEncodedMate[smemEncodedMateInts];

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

                            const unsigned int* encodedMate = nullptr;
                            {
                                const unsigned int* const gmemEncodedMate = d_inputanchormatedata + t * encodedSequencePitchInInts;
                                const int requirednumints = SequenceHelpers::getEncodedNumInts2Bit(mateLength);
                                if(smemEncodedMateInts >= requirednumints){
                                    for(int i = threadIdx.x; i < requirednumints; i += blockDim.x){
                                        smemEncodedMate[i] = gmemEncodedMate[i];
                                    }
                                    encodedMate = &smemEncodedMate[0];
                                    __syncthreads();
                                }else{
                                    encodedMate = &gmemEncodedMate[0];
                                }
                            }

                            for(int startpos = firstStartpos; startpos < lastStartposExcl; startpos++){
                                //compute metrics of overlap

                                //Hamming distance. positions which do not overlap are not accounted for
                                int ham = 0;
                                for(int i = threadIdx.x; i < min(consensusLength - startpos, mateLength); i += blockDim.x){
                                    std::uint8_t encbasemate = SequenceHelpers::getEncodedNuc2Bit(encodedMate, mateLength, mateLength - 1 - i);
                                    std::uint8_t encbasematecomp = SequenceHelpers::complementBase2Bit(encbasemate);
                                    char decbasematecomp = SequenceHelpers::decodeBase(encbasematecomp);

                                    //TODO store consensusDecoded in smem ?
                                    ham += (consensusDecoded[startpos + i] != decbasematecomp) ? 1 : 0;
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
        helpers::lambda_kernel<<<numTasks, 128, 0, stream>>>(
            [
                numTasks = numTasks,
                d_numCandidatesPerAnchor = d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchorPrefixSum = d_numCandidatesPerAnchorPrefixSum.data(),
                d_candidateSequencesLengths = d_candidateSequencesLength.data(),
                d_alignment_shifts = d_alignment_shifts.data(),
                d_anchorSequencesLength = d_anchorSequencesLength.data(),
                d_oldaccumExtensionsLengths = d_accumExtensionsLengths,
                d_newaccumExtensionsLengths = d_accumExtensionsLengthsOUT,
                d_abortReasons = d_abortReasons.data(),
                d_outputMateHasBeenFound = d_outputMateHasBeenFound.data(),
                d_sizeOfGapToMate,
                d_isFullyUsedCandidate = d_isFullyUsedCandidate.data()
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
                d_abortReasons.data(),
                d_outputMateHasBeenFound.data(),
                d_sizeOfGapToMate,
                d_outputAnchorLengths.data()
            )),
            numTasks,
            thrust::make_zip_iterator(thrust::make_tuple(
                h_accumExtensionsLengths.data(),
                h_abortReasons.data(),
                h_outputMateHasBeenFound.data(),
                h_sizeOfGapToMate.data(),
                h_outputAnchorLengths.data()
            )),
            stream
        );

        cudaMemcpyAsync(
            h_outputAnchors.data(),
            d_outputAnchors,
            sizeof(char) * outputAnchorPitchInBytes * numTasks,
            D2H,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            h_outputAnchorQualities.data(),
            d_outputAnchorQualities,
            sizeof(char) * outputAnchorQualityPitchInBytes * numTasks,
            D2H,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            h_isFullyUsedCandidate.data(),
            d_isFullyUsedCandidate,
            sizeof(bool) * totalNumCandidates,
            D2H,
            stream
        ); CUERR;

        cubAllocator->DeviceFree(d_accumExtensionsLengths); CUERR;
        cubAllocator->DeviceFree(d_accumExtensionsLengthsOUT); CUERR;
        cubAllocator->DeviceFree(d_sizeOfGapToMate); CUERR;

        setState(BatchData::State::BeforeUpdateUsedCandidateIds);
    }


    void updateUsedCandidateIds(){
        assert(state == BatchData::State::BeforeUpdateUsedCandidateIds);

        cudaStream_t stream = streams[0];

        setGpuSegmentIds(
            d_anchorIndicesOfCandidates.data(),
            numTasks,
            totalNumCandidates,
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            stream
        );

        {

            read_number* d_newUsedReadIds = nullptr; 
            int* d_newNumUsedreadIdsPerAnchor = nullptr;
            int* d_newSegmentIdsOfUsedReadIds = nullptr;

            const int maxoutputsize = totalNumCandidates + *h_numUsedReadIds;

            cubAllocator->DeviceAllocate((void**)&d_newUsedReadIds, sizeof(read_number) * maxoutputsize);
            cubAllocator->DeviceAllocate((void**)&d_newNumUsedreadIdsPerAnchor, sizeof(int) * numTasks);
            cubAllocator->DeviceAllocate((void**)&d_newSegmentIdsOfUsedReadIds, sizeof(int) * maxoutputsize);

            ThrustCachingAllocator<char> thrustCachingAllocator1(deviceId, cubAllocator, stream);

            auto d_newUsedReadIds_end = GpuSegmentedSetOperation::set_union(
                thrustCachingAllocator1,
                d_candidateReadIds.data(),
                d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchorPrefixSum.data(),
                d_anchorIndicesOfCandidates.data(),
                totalNumCandidates,
                numTasks,
                d_usedReadIds.data(),
                d_numUsedReadIdsPerAnchor.data(),
                d_numUsedReadIdsPerAnchorPrefixSum.data(),
                d_segmentIdsOfUsedReadIds.data(),
                *h_numUsedReadIds,
                numTasks,        
                d_newUsedReadIds,
                d_newNumUsedreadIdsPerAnchor,
                d_newSegmentIdsOfUsedReadIds,
                std::max(numTasks, numTasks),
                stream
            );

            int newsize = std::distance(d_newUsedReadIds, d_newUsedReadIds_end);


            d_usedReadIds.resize(newsize);
            d_segmentIdsOfUsedReadIds.resize(newsize);

            cudaMemcpyAsync(
                d_usedReadIds.data(),
                d_newUsedReadIds,
                sizeof(read_number) * newsize,
                D2D,
                stream
            );
            cubAllocator->DeviceFree(d_newUsedReadIds);

            cudaMemcpyAsync(
                d_segmentIdsOfUsedReadIds.data(),
                d_newSegmentIdsOfUsedReadIds,
                sizeof(int) * newsize,
                D2D,
                stream
            );
            cubAllocator->DeviceFree(d_newSegmentIdsOfUsedReadIds);

            cudaMemcpyAsync(
                d_numUsedReadIdsPerAnchor.data(),
                d_newNumUsedreadIdsPerAnchor,
                sizeof(int) * numTasks,
                D2D,
                stream
            );
            cubAllocator->DeviceFree(d_newNumUsedreadIdsPerAnchor);

            std::size_t bytes = 0;

            cudaError_t cubstatus = cub::DeviceScan::ExclusiveSum(
                nullptr, 
                bytes, 
                d_numUsedReadIdsPerAnchor.data(), 
                d_numUsedReadIdsPerAnchorPrefixSum.data(), 
                numTasks,
                stream
            );
            assert(cubstatus == cudaSuccess);

            CachedDeviceUVector<char> cubTemp(bytes, stream, *cubAllocator);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                cubTemp.data(), 
                bytes, 
                d_numUsedReadIdsPerAnchor.data(), 
                d_numUsedReadIdsPerAnchorPrefixSum.data(), 
                numTasks,
                stream
            );
            assert(cubstatus == cudaSuccess);

            *h_numUsedReadIds = newsize;

        }

        {
            CachedDeviceUVector<char> cubTemp(*cubAllocator);
            std::size_t bytes = 0;
            cudaError_t cubstatus = cudaSuccess;

            read_number* d_currentFullyUsedReadIds = nullptr; 
            int* d_currentNumFullyUsedreadIdsPerAnchor = nullptr;
            int* d_currentNumFullyUsedreadIdsPerAnchorPS = nullptr;
            int* d_currentSegmentIdsOfFullyUsedReadIds = nullptr;
            int* h_currentNumFullyUsed = h_firstTasksOfPairsToCheck.data();
            

            cubAllocator->DeviceAllocate((void**)&d_currentFullyUsedReadIds, sizeof(read_number) * totalNumCandidates);
            cubAllocator->DeviceAllocate((void**)&d_currentNumFullyUsedreadIdsPerAnchor, sizeof(int) * numTasks);
            cubAllocator->DeviceAllocate((void**)&d_currentNumFullyUsedreadIdsPerAnchorPS, sizeof(int) * numTasks);
            cubAllocator->DeviceAllocate((void**)&d_currentSegmentIdsOfFullyUsedReadIds, sizeof(int) * totalNumCandidates);

            auto candidatesAndSegmentIdsIn = thrust::make_zip_iterator(
                thrust::make_tuple(
                    d_candidateReadIds.data(),
                    d_anchorIndicesOfCandidates.data()
                )
            );

            auto candidatesAndSegmentIdsOut = thrust::make_zip_iterator(
                thrust::make_tuple(
                    d_currentFullyUsedReadIds,
                    d_currentSegmentIdsOfFullyUsedReadIds
                )
            );

            //make compact list of current fully used candidates
            cubstatus = cub::DeviceSelect::Flagged(
                nullptr,
                bytes,
                candidatesAndSegmentIdsIn,
                d_isFullyUsedCandidate.data(),
                candidatesAndSegmentIdsOut,
                h_currentNumFullyUsed,
                totalNumCandidates,
                stream
            );
            assert(cubstatus == cudaSuccess);

            cubTemp.resizeWithoutCopy(bytes, stream);

            cubstatus = cub::DeviceSelect::Flagged(
                cubTemp.data(),
                bytes,
                candidatesAndSegmentIdsIn,
                d_isFullyUsedCandidate.data(),
                candidatesAndSegmentIdsOut,
                h_currentNumFullyUsed,
                totalNumCandidates,
                stream
            );
            assert(cubstatus == cudaSuccess);
            cudaEventRecord(events[0], stream); CUERR;

            //compute current number of fully used candidates per segment
            cubstatus = cub::DeviceSegmentedReduce::Sum(
                nullptr,
                bytes,
                d_isFullyUsedCandidate.data(),
                d_currentNumFullyUsedreadIdsPerAnchor,
                numTasks,
                d_numCandidatesPerAnchorPrefixSum.data(),
                d_numCandidatesPerAnchorPrefixSum.data() + 1,
                stream
            );
            assert(cubstatus == cudaSuccess);

            cubTemp.resizeWithoutCopy(bytes, stream);

            cubstatus = cub::DeviceSegmentedReduce::Sum(
                cubTemp.data(),
                bytes,
                d_isFullyUsedCandidate.data(),
                d_currentNumFullyUsedreadIdsPerAnchor,
                numTasks,
                d_numCandidatesPerAnchorPrefixSum.data(),
                d_numCandidatesPerAnchorPrefixSum.data() + 1,
                stream
            );
            assert(cubstatus == cudaSuccess);

            //compute prefix sum of current number of fully used candidates per segment

            cubstatus = cub::DeviceScan::ExclusiveSum(
                nullptr, 
                bytes, 
                d_currentNumFullyUsedreadIdsPerAnchor, 
                d_currentNumFullyUsedreadIdsPerAnchorPS, 
                numTasks,
                stream
            );
            assert(cubstatus == cudaSuccess);

            cubTemp.resizeWithoutCopy(bytes, stream);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                cubTemp.data(), 
                bytes, 
                d_currentNumFullyUsedreadIdsPerAnchor, 
                d_currentNumFullyUsedreadIdsPerAnchorPS, 
                numTasks,
                stream
            );
            assert(cubstatus == cudaSuccess);

            cudaEventSynchronize(events[0]); CUERR;



            read_number* d_newFullyUsedReadIds = nullptr; 
            int* d_newNumFullyUsedreadIdsPerAnchor = nullptr;
            int* d_newSegmentIdsOfFullyUsedReadIds = nullptr;

            const int maxoutputsize = totalNumCandidates + *h_numFullyUsedReadIds;

            cubAllocator->DeviceAllocate((void**)&d_newFullyUsedReadIds, sizeof(read_number) * maxoutputsize);
            cubAllocator->DeviceAllocate((void**)&d_newNumFullyUsedreadIdsPerAnchor, sizeof(int) * numTasks);
            cubAllocator->DeviceAllocate((void**)&d_newSegmentIdsOfFullyUsedReadIds, sizeof(int) * maxoutputsize);

            ThrustCachingAllocator<char> thrustCachingAllocator1(deviceId, cubAllocator, stream);

            auto d_newFullyUsedReadIds_end = GpuSegmentedSetOperation::set_union(
                thrustCachingAllocator1,
                d_currentFullyUsedReadIds,
                d_currentNumFullyUsedreadIdsPerAnchor,
                d_currentNumFullyUsedreadIdsPerAnchorPS,
                d_currentSegmentIdsOfFullyUsedReadIds,
                *h_currentNumFullyUsed,
                numTasks,
                d_fullyUsedReadIds.data(),
                d_numFullyUsedReadIdsPerAnchor.data(),
                d_numFullyUsedReadIdsPerAnchorPrefixSum.data(),
                d_segmentIdsOfFullyUsedReadIds.data(),
                *h_numFullyUsedReadIds,
                numTasks,        
                d_newFullyUsedReadIds,
                d_newNumFullyUsedreadIdsPerAnchor,
                d_newSegmentIdsOfFullyUsedReadIds,
                std::max(numTasks, numTasks),
                stream
            );

            cubAllocator->DeviceFree(d_currentFullyUsedReadIds);
            cubAllocator->DeviceFree(d_currentNumFullyUsedreadIdsPerAnchor);
            cubAllocator->DeviceFree(d_currentNumFullyUsedreadIdsPerAnchorPS);
            cubAllocator->DeviceFree(d_currentSegmentIdsOfFullyUsedReadIds);

            int newsize = std::distance(d_newFullyUsedReadIds, d_newFullyUsedReadIds_end);

            d_fullyUsedReadIds.resize(newsize);
            d_segmentIdsOfFullyUsedReadIds.resize(newsize);

            cudaMemcpyAsync(
                d_fullyUsedReadIds.data(),
                d_newFullyUsedReadIds,
                sizeof(read_number) * newsize,
                D2D,
                stream
            );
            cubAllocator->DeviceFree(d_newFullyUsedReadIds);

            cudaMemcpyAsync(
                d_segmentIdsOfFullyUsedReadIds.data(),
                d_newSegmentIdsOfFullyUsedReadIds,
                sizeof(int) * newsize,
                D2D,
                stream
            );
            cubAllocator->DeviceFree(d_newSegmentIdsOfFullyUsedReadIds);

            cudaMemcpyAsync(
                d_numFullyUsedReadIdsPerAnchor.data(),
                d_newNumFullyUsedreadIdsPerAnchor,
                sizeof(int) * numTasks,
                D2D,
                stream
            );
            cubAllocator->DeviceFree(d_newNumFullyUsedreadIdsPerAnchor);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                nullptr, 
                bytes, 
                d_numFullyUsedReadIdsPerAnchor.data(), 
                d_numFullyUsedReadIdsPerAnchorPrefixSum.data(), 
                numTasks,
                stream
            );
            assert(cubstatus == cudaSuccess);

            cubTemp.resizeWithoutCopy(bytes, stream);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                cubTemp.data(), 
                bytes, 
                d_numFullyUsedReadIdsPerAnchor.data(), 
                d_numFullyUsedReadIdsPerAnchorPrefixSum.data(), 
                numTasks,
                stream
            );
            assert(cubstatus == cudaSuccess);

            *h_numFullyUsedReadIds = newsize;

        }

        setState(BatchData::State::BeforeCopyToHost);
    }
    
    void copyBuffersToHost(){
        assert(state == BatchData::State::BeforeCopyToHost);

        nvtx::push_range("copyBuffersToHost", 8);

        // h_candidateReadIds.resize(totalNumCandidates);
        // h_numCandidatesPerAnchor.resize(numTasks);
        // h_numCandidatesPerAnchorPrefixSum.resize(numTasks + 1);

        // h_numCandidatesPerAnchorPrefixSum[0] = 0;
       
        // helpers::call_copy_n_kernel(
        //     thrust::make_zip_iterator(thrust::make_tuple(
        //         d_numCandidatesPerAnchorPrefixSum.data() + 1,
        //         d_numCandidatesPerAnchor.data()
        //     )),
        //     numTasks,
        //     thrust::make_zip_iterator(thrust::make_tuple(
        //         h_numCandidatesPerAnchorPrefixSum.data() + 1,
        //         h_numCandidatesPerAnchor.data()
        //     )),
        //     streams[0]
        // );   

        // cudaMemcpyAsync(
        //     h_candidateReadIds.data(),
        //     d_candidateReadIds.data(),
        //     sizeof(read_number) * totalNumCandidates,
        //     D2H,
        //     streams[0]
        // );

        cudaStreamSynchronize(streams[0]); CUERR;
        cudaStreamSynchronize(streams[1]); CUERR;

        nvtx::pop_range();

        setState(BatchData::State::BeforeUnpack);
    }

    void unpackResultsIntoTasks(){
        assert(state == BatchData::State::BeforeUnpack);

        for(int i = 0; i < numTasks; i++){ 
            const int indexOfActiveTask = i;
            auto& task = tasks[indexOfActiveTask];

            task.abortReason = h_abortReasons[i];
            if(task.abortReason == extension::AbortReason::None){
                task.mateHasBeenFound = h_outputMateHasBeenFound[i];

                const int myNumDecodedAnchors = task.totalDecodedAnchorsLengths.size();

                if(!task.mateHasBeenFound){
                    const int newlength = h_outputAnchorLengths[i];
                    
                    task.currentAnchorLength = newlength;
                    task.accumExtensionLengths = h_accumExtensionsLengths[i];

                    task.totalDecodedAnchorsFlat.resize((myNumDecodedAnchors+1) * decodedSequencePitchInBytes);
                    assert(newlength <= decodedSequencePitchInBytes);
                    std::copy_n(
                        h_outputAnchors.data() + i * outputAnchorPitchInBytes,
                        newlength,
                        task.totalDecodedAnchorsFlat.begin()
                            + myNumDecodedAnchors * decodedSequencePitchInBytes
                    );
                    task.totalDecodedAnchorsLengths.emplace_back(newlength);

                    task.totalAnchorQualityScoresFlat.resize((myNumDecodedAnchors+1) * qualityPitchInBytes);
                    assert(newlength <= qualityPitchInBytes);
                    std::copy_n(
                        h_outputAnchorQualities.data() + i * outputAnchorQualityPitchInBytes,
                        newlength,
                        task.totalAnchorQualityScoresFlat.begin()
                            + myNumDecodedAnchors * qualityPitchInBytes
                    );


                    task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);
                    
                }else{
                    const int sizeofGap = h_sizeOfGapToMate[i];
                    if(sizeofGap == 0){
                        task.accumExtensionLengths = h_accumExtensionsLengths[i];
                        task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);
  
                        task.totalDecodedAnchorsFlat.resize((myNumDecodedAnchors+1) * decodedSequencePitchInBytes);
                        assert(task.mateLength <= decodedSequencePitchInBytes);
                        std::copy(
                            task.decodedMateRevC.begin(),
                            task.decodedMateRevC.end(),
                            task.totalDecodedAnchorsFlat.begin()
                                + myNumDecodedAnchors * decodedSequencePitchInBytes
                        );
                        task.totalDecodedAnchorsLengths.emplace_back(task.mateLength);

                        task.totalAnchorQualityScoresFlat.resize((myNumDecodedAnchors + 1) * qualityPitchInBytes);
                        assert(task.mateLength <= qualityPitchInBytes);
                        std::copy(
                            task.mateQualityScoresReversed.begin(),
                            task.mateQualityScoresReversed.end(),
                            task.totalAnchorQualityScoresFlat.begin()
                                + myNumDecodedAnchors * qualityPitchInBytes
                        );

                    }else{
                        const int newlength = h_outputAnchorLengths[i];

                        task.totalDecodedAnchorsFlat.resize((myNumDecodedAnchors+2) * decodedSequencePitchInBytes);
                        task.totalAnchorQualityScoresFlat.resize((myNumDecodedAnchors + 2) * qualityPitchInBytes);

                        std::string newq(h_outputAnchorQualities.data() + i * outputAnchorQualityPitchInBytes, newlength);

                        task.accumExtensionLengths = h_accumExtensionsLengths[i];
                        assert(newlength <= decodedSequencePitchInBytes);
                        std::copy_n(
                            h_outputAnchors.data() + i * outputAnchorPitchInBytes,
                            newlength,
                            task.totalDecodedAnchorsFlat.begin()
                                + myNumDecodedAnchors * decodedSequencePitchInBytes
                        );
                        task.totalDecodedAnchorsLengths.emplace_back(newlength);

                        assert(newlength <= qualityPitchInBytes);
                        std::copy_n(
                            h_outputAnchorQualities.data() + i * outputAnchorQualityPitchInBytes,
                            newlength,
                            task.totalAnchorQualityScoresFlat.begin()
                                + myNumDecodedAnchors * qualityPitchInBytes
                        );

                        task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);

                        task.accumExtensionLengths += newlength;
                        task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);
                        //task.totalDecodedAnchors.emplace_back(task.decodedMateRevC);
                        assert(task.mateLength <= decodedSequencePitchInBytes);
                        std::copy(
                            task.decodedMateRevC.begin(),
                            task.decodedMateRevC.end(),
                            task.totalDecodedAnchorsFlat.begin()
                                + (myNumDecodedAnchors + 1) * decodedSequencePitchInBytes
                        );
                        task.totalDecodedAnchorsLengths.emplace_back(task.mateLength);
                        
                        assert(task.mateLength <= qualityPitchInBytes);
                        std::copy(
                            task.mateQualityScoresReversed.begin(),
                            task.mateQualityScoresReversed.end(),
                            task.totalAnchorQualityScoresFlat.begin()
                                + (myNumDecodedAnchors + 1) * qualityPitchInBytes
                        );
                    }
                }
            }

            task.abort = task.abortReason != extension::AbortReason::None;
        }

        handleEarlyExitOfTasks4();

        for(int i = 0; i < numTasks; i++){
            auto& task = tasks[i];

            task.iteration++;
        }

        setState(BatchData::State::BeforePrepareNextIteration);
    }

    void prepareNextIteration(){
        assert(state == BatchData::State::BeforePrepareNextIteration);

        //update list of active task indices
        h_newPositionsOfActiveTasks.resize(numTasks);
        int newPosSize = 0;

        assert(numTasks == int(tasks.size()));
        const int totalTasksBefore = tasks.size() + finishedTasks.size();

        std::vector<extension::Task> newActiveTasks;
        std::vector<extension::Task> newlyFinishedTasks;
        newActiveTasks.reserve(numTasks);
        newlyFinishedTasks.reserve(numTasks);

        for(int i = 0; i < numTasks; i++){
            if(tasks[i].isActive(insertSize, insertSizeStddev)){
                h_newPositionsOfActiveTasks[newPosSize] = i;

                newActiveTasks.emplace_back(std::move(tasks[i]));

                newPosSize++;
            }else{
                newlyFinishedTasks.emplace_back(std::move(tasks[i]));
            }
        }
        h_newPositionsOfActiveTasks.resize(newPosSize);
        std::swap(tasks, newActiveTasks);
        nvtx::push_range("addSortedFinishedTasks", 5);
        addSortedFinishedTasks(newlyFinishedTasks);
        nvtx::pop_range();

        const int totalTasksAfter = tasks.size() + finishedTasks.size();
        assert(totalTasksAfter == totalTasksBefore);
        // );

        if(!isEmpty()){

            d_newPositionsOfActiveTasks.resize(h_newPositionsOfActiveTasks.size());

            cudaMemcpyAsync(
                d_newPositionsOfActiveTasks.data(),
                h_newPositionsOfActiveTasks.data(),
                sizeof(int) * h_newPositionsOfActiveTasks.size(),
                H2D,
                streams[0]
            ); CUERR;

            // {
            //     std::size_t free, total;
            //     cudaMemGetInfo(&free, &total);
            //     std::cerr << "before updateBuffersForNextIteration " << free << "\n";
            // }

            nvtx::push_range("updateBuffersForNextIteration", 6);

            updateBuffersForNextIteration();

            nvtx::pop_range();

            // {
            //     std::size_t free, total;
            //     cudaMemGetInfo(&free, &total);
            //     std::cerr << "after updateBuffersForNextIteration " << free << "\n";
            // }

        }

        numTasks = tasks.size();

        if(!isEmpty()){
            setState(BatchData::State::BeforeHash);
        }else{
            setStateToFinished();
        }
        
    }


    void updateBuffersForNextIteration(){
        nvtx::push_range("removeUsedIdsOfFinishedTasks", 6);

        removeUsedIdsOfFinishedTasks();

        nvtx::pop_range();

        //compute selection flags of remaining tasks
        const int newNumTasks = d_newPositionsOfActiveTasks.size();
        bool* d_isActive = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_isActive, numTasks, streams[0]);
        cudaMemsetAsync(d_isActive, 0, numTasks, streams[0]); CUERR;

        helpers::lambda_kernel<<<SDIV(newNumTasks, 128), 128, 0, streams[0]>>>(
            [
                d_isActive,
                d_newPositionsOfActiveTasks = d_newPositionsOfActiveTasks.data(),
                newNumTasks
            ] __device__ (){
                const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                const int stride = blockDim.x * gridDim.x;

                for(int i = tid; i < newNumTasks; i += stride){
                    d_isActive[d_newPositionsOfActiveTasks[i]] = true;
                }
            }
        ); CUERR;

        CachedDeviceUVector<char> cubTemp(*cubAllocator);
        std::size_t bytes = 0;
        cudaError_t cubstatus = cudaSuccess;

        //set new decoded anchors
        d_subjectSequencesDataDecoded.resizeWithoutCopy(newNumTasks * decodedSequencePitchInBytes, streams[0]);

        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            bytes,
            d_outputAnchors.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_isActive, outputAnchorPitchInBytes)
            ),
            d_subjectSequencesDataDecoded.data(),
            thrust::make_discard_iterator(),
            numTasks * outputAnchorPitchInBytes,
            streams[0]
        );
        assert(cubstatus == cudaSuccess);

        cubTemp.resizeWithoutCopy(bytes, streams[0]);

        cubstatus = cub::DeviceSelect::Flagged(
            cubTemp.data(),
            bytes,
            d_outputAnchors.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_isActive, outputAnchorPitchInBytes)
            ),
            d_subjectSequencesDataDecoded.data(),
            thrust::make_discard_iterator(),
            numTasks * outputAnchorPitchInBytes,
            streams[0]
        );
        assert(cubstatus == cudaSuccess);
        
        // set new anchor quality scores
        d_anchorQualityScores.resizeWithoutCopy(newNumTasks * qualityPitchInBytes, streams[0]);

        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            bytes,
            d_outputAnchorQualities.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_isActive, outputAnchorQualityPitchInBytes)
            ),
            d_anchorQualityScores.data(),
            thrust::make_discard_iterator(),
            numTasks * outputAnchorQualityPitchInBytes,
            streams[0]
        );
        assert(cubstatus == cudaSuccess);

        cubTemp.resizeWithoutCopy(bytes, streams[0]);

        cubstatus = cub::DeviceSelect::Flagged(
            cubTemp.data(),
            bytes,
            d_outputAnchorQualities.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_isActive, outputAnchorQualityPitchInBytes)
            ),
            d_anchorQualityScores.data(),
            thrust::make_discard_iterator(),
            numTasks * outputAnchorQualityPitchInBytes,
            streams[0]
        );
        assert(cubstatus == cudaSuccess);

        //set new anchorReadIds, mateReadIds, and anchor lengths

        CachedDeviceUVector<read_number> d_anchorReadIds2(newNumTasks, streams[0], *cubAllocator);
        CachedDeviceUVector<read_number> d_mateReadIds2(newNumTasks, streams[0], *cubAllocator);
        CachedDeviceUVector<int> d_inputMateLengths2(newNumTasks, streams[0], *cubAllocator);
        CachedDeviceUVector<bool> d_isPairedTask2(newNumTasks, streams[0], *cubAllocator);;

        d_anchorSequencesLength.resizeWithoutCopy(newNumTasks, streams[0]);

        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            bytes,
            thrust::make_zip_iterator(thrust::make_tuple(
                d_anchorReadIds.data(),
                d_mateReadIds.data(),
                d_outputAnchorLengths.data(),
                d_inputMateLengths.data(),
                d_isPairedTask.data()
            )),
            d_isActive,
            thrust::make_zip_iterator(thrust::make_tuple(
                d_anchorReadIds2.data(),
                d_mateReadIds2.data(),
                d_anchorSequencesLength.data(),
                d_inputMateLengths2.data(),
                d_isPairedTask2.data()
            )),
            thrust::make_discard_iterator(),
            numTasks,
            streams[0]
        );
        assert(cubstatus == cudaSuccess);

        cubTemp.resizeWithoutCopy(bytes, streams[0]);

        cubstatus = cub::DeviceSelect::Flagged(
            cubTemp.data(),
            bytes,
            thrust::make_zip_iterator(thrust::make_tuple(
                d_anchorReadIds.data(),
                d_mateReadIds.data(),
                d_outputAnchorLengths.data(),
                d_inputMateLengths.data(),
                d_isPairedTask.data()
            )),
            d_isActive,
            thrust::make_zip_iterator(thrust::make_tuple(
                d_anchorReadIds2.data(),
                d_mateReadIds2.data(),
                d_anchorSequencesLength.data(),
                d_inputMateLengths2.data(),
                d_isPairedTask2.data()
            )),
            thrust::make_discard_iterator(),
            numTasks,
            streams[0]
        );
        assert(cubstatus == cudaSuccess);

        std::swap(d_anchorReadIds, d_anchorReadIds2);
        std::swap(d_mateReadIds, d_mateReadIds2);
        std::swap(d_inputMateLengths, d_inputMateLengths2);
        std::swap(d_isPairedTask, d_isPairedTask2);


        //set new encoded mate data

        CachedDeviceUVector<unsigned int> d_inputanchormatedata2(newNumTasks * encodedSequencePitchInInts, streams[0], *cubAllocator);

        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            bytes,
            d_inputanchormatedata.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_isActive, encodedSequencePitchInInts)
            ),
            d_inputanchormatedata2.data(),
            thrust::make_discard_iterator(),
            numTasks * encodedSequencePitchInInts,
            streams[0]
        );
        assert(cubstatus == cudaSuccess);

        cubTemp.resizeWithoutCopy(bytes, streams[0]);

        cubstatus = cub::DeviceSelect::Flagged(
            cubTemp.data(),
            bytes,
            d_inputanchormatedata.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_isActive, encodedSequencePitchInInts)
            ),
            d_inputanchormatedata2.data(),
            thrust::make_discard_iterator(),
            numTasks * encodedSequencePitchInInts,
            streams[0]
        );
        assert(cubstatus == cudaSuccess);

        std::swap(d_inputanchormatedata, d_inputanchormatedata2);
        
        cubAllocator->DeviceFree(d_isActive);

        //convert new anchors to 2bit representation

        d_subjectSequencesData.resize(newNumTasks * encodedSequencePitchInInts, streams[0]);

        readextendergpukernels::encodeSequencesTo2BitKernel<8>
        <<<SDIV(newNumTasks, (128 / 8)), 128, 0, streams[0]>>>(
            d_subjectSequencesData.data(),
            d_subjectSequencesDataDecoded.data(),
            d_anchorSequencesLength.data(),
            decodedSequencePitchInBytes,
            encodedSequencePitchInInts,
            newNumTasks
        ); CUERR;


        //shrink remaining buffers
        d_anchormatedata.resize(newNumTasks * encodedSequencePitchInInts);
        d_anchorIndicesWithRemovedMates.resize(newNumTasks);

        d_numCandidatesPerAnchor.resize(newNumTasks, streams[0]);
        d_numCandidatesPerAnchorPrefixSum.resize(newNumTasks+1, streams[0]);


    }

    void removeUsedIdsOfFinishedTasks(){
        const int newNumTasks = d_newPositionsOfActiveTasks.size();

        if(newNumTasks == 0) return;

        assert(newNumTasks <= numTasks);

        // {
        //     std::size_t free, total;
        //     cudaMemGetInfo(&free, &total);
        //     std::cerr << "before removeUsedIdsOfFinishedTasks " << free << "\n";
        // }


        //update used ids

        {

            d_numUsedReadIdsPerAnchor2.resize(newNumTasks);
            d_numUsedReadIdsPerAnchorPrefixSum2.resize(newNumTasks);           

            helpers::lambda_kernel<<<SDIV(newNumTasks,256), 256, 0, streams[0]>>>(
                [
                    indicesOfActiveTasks = d_newPositionsOfActiveTasks.data(),
                    newNumTasks,
                    d_numUsedReadIdsPerAnchorOut = d_numUsedReadIdsPerAnchor2.data(),
                    d_numUsedReadIdsPerAnchorIn = d_numUsedReadIdsPerAnchor.data(),
                    d_numUsedReadIdsPerAnchorOutsize = d_numUsedReadIdsPerAnchor2.size(),
                    d_numUsedReadIdsPerAnchorInsize = d_numUsedReadIdsPerAnchor.size()
                ] __device__ (){
                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    const int stride = blockDim.x * gridDim.x;

                    for(int t = tid; t < newNumTasks; t += stride){
                        // assert(t < d_numUsedReadIdsPerAnchorOutsize);
                        // assert(indicesOfActiveTasks[t] < d_numUsedReadIdsPerAnchorInsize);
                        d_numUsedReadIdsPerAnchorOut[t] = d_numUsedReadIdsPerAnchorIn[indicesOfActiveTasks[t]];
                    }
                }
            ); CUERR;

            CachedDeviceUVector<char> cubTemp(*cubAllocator);
            std::size_t bytes = 0;
            cudaError_t cubstatus = cudaSuccess;
            
            cubstatus = cub::DeviceReduce::Sum(
                nullptr, 
                bytes, 
                d_numUsedReadIdsPerAnchor2.data(), 
                h_numUsedReadIds.data(),
                newNumTasks,
                streams[0]
            );
            assert(cubstatus == cudaSuccess);

            cubTemp.resizeWithoutCopy(bytes, streams[0]);

            cubstatus = cub::DeviceReduce::Sum(
                cubTemp.data(), 
                bytes, 
                d_numUsedReadIdsPerAnchor2.data(), 
                h_numUsedReadIds.data(),
                newNumTasks,
                streams[0]
            );
            assert(cubstatus == cudaSuccess);

            cudaEventRecord(events[0], streams[0]);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                nullptr, 
                bytes, 
                d_numUsedReadIdsPerAnchor2.data(), 
                d_numUsedReadIdsPerAnchorPrefixSum2.data(),  
                newNumTasks,
                streams[0]
            );
            assert(cubstatus == cudaSuccess);

            cubTemp.resizeWithoutCopy(bytes, streams[0]);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                cubTemp.data(), 
                bytes, 
                d_numUsedReadIdsPerAnchor2.data(), 
                d_numUsedReadIdsPerAnchorPrefixSum2.data(), 
                newNumTasks,
                streams[0]
            );
            assert(cubstatus == cudaSuccess);

            cudaEventSynchronize(events[0]); CUERR; //wait until h_numUsedReadIds is ready

            d_usedReadIds2.resize(*h_numUsedReadIds);
            d_segmentIdsOfUsedReadIds2.resize(*h_numUsedReadIds);            

            const int possibleNumWarps = newNumTasks;
            const int possibleNumBlocks = SDIV(possibleNumWarps, 128 / 32);
            const int numBlocks = std::min(256, possibleNumBlocks);

            readextendergpukernels::compactUsedIdsOfSelectedTasks<32><<<numBlocks, 128, 0, streams[0]>>>(
                d_newPositionsOfActiveTasks.data(),
                newNumTasks,
                d_usedReadIds.data(),
                d_usedReadIds2.data(),
                d_segmentIdsOfUsedReadIds2.data(),
                d_numUsedReadIdsPerAnchor2.data(),
                d_numUsedReadIdsPerAnchorPrefixSum.data(), 
                d_numUsedReadIdsPerAnchorPrefixSum2.data()
            ); CUERR;

            std::swap(d_usedReadIds, d_usedReadIds2);
            std::swap(d_numUsedReadIdsPerAnchor, d_numUsedReadIdsPerAnchor2);
            std::swap(d_numUsedReadIdsPerAnchorPrefixSum, d_numUsedReadIdsPerAnchorPrefixSum2);
            std::swap(d_segmentIdsOfUsedReadIds, d_segmentIdsOfUsedReadIds2);
        }

        //update fully used ids
        
        {

            d_numFullyUsedReadIdsPerAnchor2.resize(newNumTasks);
            d_numFullyUsedReadIdsPerAnchorPrefixSum2.resize(newNumTasks);

            helpers::lambda_kernel<<<SDIV(newNumTasks,256), 256, 0, streams[0]>>>(
                [
                    indicesOfActiveTasks = d_newPositionsOfActiveTasks.data(),
                    newNumTasks,
                    d_numFullyUsedReadIdsPerAnchorOut = d_numFullyUsedReadIdsPerAnchor2.data(),
                    d_numFullyUsedReadIdsPerAnchorIn = d_numFullyUsedReadIdsPerAnchor.data(),
                    d_numFullyUsedReadIdsPerAnchorOutsize = d_numFullyUsedReadIdsPerAnchor2.size(),
                    d_numFullyUsedReadIdsPerAnchorInsize = d_numFullyUsedReadIdsPerAnchor.size()
                ] __device__ (){
                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    const int stride = blockDim.x * gridDim.x;

                    for(int t = tid; t < newNumTasks; t += stride){
                        // assert(t < d_numFullyUsedReadIdsPerAnchorOutsize);
                        // if(!(indicesOfActiveTasks[t] < d_numFullyUsedReadIdsPerAnchorInsize)){
                        //     printf("t %d indicesOfActiveTasks[t] %d d_numFullyUsedReadIdsPerAnchorInsize %lu\n", t, indicesOfActiveTasks[t], d_numFullyUsedReadIdsPerAnchorInsize);
                        //     assert(indicesOfActiveTasks[t] < d_numFullyUsedReadIdsPerAnchorInsize);
                        // }
                        d_numFullyUsedReadIdsPerAnchorOut[t] = d_numFullyUsedReadIdsPerAnchorIn[indicesOfActiveTasks[t]];
                    }
                }
            ); CUERR;

            CachedDeviceUVector<char> cubTemp(*cubAllocator);
            std::size_t bytes = 0;
            cudaError_t cubstatus = cudaSuccess;
            
            cubstatus = cub::DeviceReduce::Sum(
                nullptr, 
                bytes, 
                d_numFullyUsedReadIdsPerAnchor2.data(), 
                h_numFullyUsedReadIds.data(),
                newNumTasks,
                streams[0]
            );
            assert(cubstatus == cudaSuccess);

            cubTemp.resizeWithoutCopy(bytes, streams[0]);

            cubstatus = cub::DeviceReduce::Sum(
                cubTemp.data(), 
                bytes, 
                d_numFullyUsedReadIdsPerAnchor2.data(), 
                h_numFullyUsedReadIds.data(),
                newNumTasks,
                streams[0]
            );
            assert(cubstatus == cudaSuccess);

            cudaEventRecord(events[0], streams[0]);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                nullptr, 
                bytes, 
                d_numFullyUsedReadIdsPerAnchor2.data(), 
                d_numFullyUsedReadIdsPerAnchorPrefixSum2.data(),  
                newNumTasks,
                streams[0]
            );
            assert(cubstatus == cudaSuccess);

            cubTemp.resizeWithoutCopy(bytes, streams[0]);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                cubTemp.data(), 
                bytes, 
                d_numFullyUsedReadIdsPerAnchor2.data(), 
                d_numFullyUsedReadIdsPerAnchorPrefixSum2.data(), 
                newNumTasks,
                streams[0]
            );
            assert(cubstatus == cudaSuccess);

            cudaEventSynchronize(events[0]); CUERR; //wait until h_numFullyUsedReadIds is ready

            d_fullyUsedReadIds2.resize(*h_numFullyUsedReadIds);
            d_segmentIdsOfFullyUsedReadIds2.resize(*h_numFullyUsedReadIds);

            const int possibleNumWarps = newNumTasks;
            const int possibleNumBlocks = SDIV(possibleNumWarps, 128 / 32);
            const int numBlocks = std::min(256, possibleNumBlocks);

            readextendergpukernels::compactUsedIdsOfSelectedTasks<32><<<numBlocks, 128, 0, streams[0]>>>(
                d_newPositionsOfActiveTasks.data(),
                newNumTasks,
                d_fullyUsedReadIds.data(),
                d_fullyUsedReadIds2.data(),
                d_segmentIdsOfFullyUsedReadIds2.data(),
                d_numFullyUsedReadIdsPerAnchor2.data(),
                d_numFullyUsedReadIdsPerAnchorPrefixSum.data(), 
                d_numFullyUsedReadIdsPerAnchorPrefixSum2.data()
            ); CUERR;

            std::swap(d_fullyUsedReadIds, d_fullyUsedReadIds2);
            std::swap(d_numFullyUsedReadIdsPerAnchor, d_numFullyUsedReadIdsPerAnchor2);
            std::swap(d_numFullyUsedReadIdsPerAnchorPrefixSum, d_numFullyUsedReadIdsPerAnchorPrefixSum2);
            std::swap(d_segmentIdsOfFullyUsedReadIds, d_segmentIdsOfFullyUsedReadIds2);

        }

        // {
        //     std::size_t free, total;
        //     cudaMemGetInfo(&free, &total);
        //     std::cerr << "after removeUsedIdsOfFinishedTasks " << free << "\n";
        // }

    }


    std::vector<extension::ExtendResult> constructResults(){
        const int resultMSAColumnPitchInElements = 512; //SDIV(insertSize + insertSizeStddev, 4) * 4;

        #if 1
        nvtx::push_range("constructresultgpumsa", 2);
        {
            const int numFinishedTasks = finishedTasks.size();
            if(numFinishedTasks == 0){
                return std::vector<extension::ExtendResult>{};
            }
            cudaStream_t stream = streams[0];

            h_numCandidatesPerAnchor.resize(numFinishedTasks);
            h_numCandidatesPerAnchorPrefixSum.resize(numFinishedTasks + 1);

            for(int i = 0; i < numFinishedTasks; i++){
                const auto& task = finishedTasks[i];

                h_numCandidatesPerAnchor[i] = task.totalDecodedAnchorsLengths.size() - 1;
            }

            h_numCandidatesPerAnchorPrefixSum[0] = 0;
            std::inclusive_scan(
                h_numCandidatesPerAnchor.begin(),
                h_numCandidatesPerAnchor.end(),
                h_numCandidatesPerAnchorPrefixSum.begin() + 1
            );
            const int numCandidates = h_numCandidatesPerAnchorPrefixSum[numFinishedTasks];
            assert(numCandidates >= 0);

            //if there are no candidates, the resulting sequences will be identical to the input anchors. no computing required
            if(numCandidates == 0){
                h_outputAnchors.resize(numFinishedTasks * resultMSAColumnPitchInElements);
                h_outputAnchorQualities.resize(numFinishedTasks * resultMSAColumnPitchInElements);
                h_anchorSequencesLength.resize(numFinishedTasks);

                for(int i = 0; i < numFinishedTasks; i++){
                    const auto& task = finishedTasks[i];

                    const int num = h_numCandidatesPerAnchor[i];

                    std::copy(
                        task.totalDecodedAnchorsFlat.begin(),
                        task.totalDecodedAnchorsFlat.begin() + decodedSequencePitchInBytes,
                        h_outputAnchors + i * resultMSAColumnPitchInElements
                    );

                    std::fill(
                        h_outputAnchorQualities + i * resultMSAColumnPitchInElements,
                        h_outputAnchorQualities + i * resultMSAColumnPitchInElements + task.totalDecodedAnchorsLengths[0],
                        'I'
                    );

                    h_anchorSequencesLength[i] = task.totalDecodedAnchorsLengths[0];
                }

            }else{

                d_numCandidatesPerAnchor.resizeWithoutCopy(numFinishedTasks, stream);
                d_numCandidatesPerAnchorPrefixSum.resizeWithoutCopy(numFinishedTasks + 1, stream);

                cudaMemcpyAsync(
                    d_numCandidatesPerAnchor.data(),
                    h_numCandidatesPerAnchor.data(),
                    sizeof(int) * numFinishedTasks,
                    H2D,
                    stream
                ); CUERR;

                cudaMemcpyAsync(
                    d_numCandidatesPerAnchorPrefixSum.data(),
                    h_numCandidatesPerAnchorPrefixSum.data(),
                    sizeof(int) * (numFinishedTasks + 1),
                    H2D,
                    stream
                ); CUERR;

                //copy anchor lengths and anchor sequences

                h_outputAnchors.resize(numFinishedTasks * decodedSequencePitchInBytes);
                h_anchorSequencesLength.resize(numFinishedTasks);

                CachedDeviceUVector<int> d_anchorSequencesLength2(numFinishedTasks, stream, *cubAllocator);
                CachedDeviceUVector<unsigned int> d_subjectSequencesData2(numFinishedTasks * encodedSequencePitchInInts, stream, *cubAllocator);
                CachedDeviceUVector<char> d_subjectSequencesDataDecoded2(numFinishedTasks * decodedSequencePitchInBytes, stream, *cubAllocator);

                for(int i = 0; i < numFinishedTasks; i++){
                    const auto& task = finishedTasks[i];

                    const int num = h_numCandidatesPerAnchor[i];

                    std::copy(
                        task.totalDecodedAnchorsFlat.begin(),
                        task.totalDecodedAnchorsFlat.begin() + decodedSequencePitchInBytes,
                        h_outputAnchors + i * decodedSequencePitchInBytes
                    );

                    h_anchorSequencesLength[i] = task.totalDecodedAnchorsLengths[0];
                }

                cudaMemcpyAsync(
                    d_anchorSequencesLength2.data(),
                    h_anchorSequencesLength.data(),
                    sizeof(int) * numFinishedTasks,
                    H2D,
                    stream
                ); CUERR;

                cudaMemcpyAsync(
                    d_subjectSequencesDataDecoded2.data(),
                    h_outputAnchors.data(),
                    sizeof(char) * decodedSequencePitchInBytes * numFinishedTasks,
                    H2D,
                    stream
                ); CUERR;

                readextendergpukernels::encodeSequencesTo2BitKernel<8>
                <<<SDIV(numFinishedTasks, (128 / 8)), 128, 0, streams[0]>>>(
                    d_subjectSequencesData2.data(),
                    d_subjectSequencesDataDecoded2.data(),
                    d_anchorSequencesLength2.data(),
                    decodedSequencePitchInBytes,
                    encodedSequencePitchInInts,
                    numFinishedTasks
                ); CUERR;

                //copy anchor qualities
                // h_outputAnchorQualities.resize(numFinishedTasks * qualityPitchInBytes);
                // d_anchorQualityScores2.resize(numFinishedTasks * qualityPitchInBytes);

                // for(int i = 0; i < numFinishedTasks; i++){
                //     const auto& task = finishedTasks[i];

                //     const int num = h_numCandidatesPerAnchor[i];

                //     std::copy(
                //         task.totalAnchorQualityScoresFlat.begin(),
                //         task.totalAnchorQualityScoresFlat.begin() + qualityPitchInBytes,
                //         h_outputAnchorQualities + i * qualityPitchInBytes
                //     );
                // }

                // cudaMemcpyAsync(
                //     d_anchorQualityScores2.data(),
                //     h_outputAnchorQualities.data(),
                //     sizeof(char) * qualityPitchInBytes * numFinishedTasks,
                //     H2D,
                //     stream
                // ); CUERR;

                //synchronize before reusing pinned buffers
                cudaStreamSynchronize(stream); CUERR;

                //copy "candidate" sequences

                h_outputAnchors.resize(numCandidates * decodedSequencePitchInBytes);
                h_anchorSequencesLength.resize(numCandidates);

                d_candidateSequencesLength.resizeWithoutCopy(numCandidates, stream);
                d_candidateSequencesData.resizeWithoutCopy(numCandidates * encodedSequencePitchInInts, stream);
                CachedDeviceUVector<char> d_candidateSequencesDataDecoded(decodedSequencePitchInBytes * numCandidates, stream, *cubAllocator);

                for(int i = 0; i < numFinishedTasks; i++){
                    const auto& task = finishedTasks[i];

                    const int num = h_numCandidatesPerAnchor[i];
                    const int offset = h_numCandidatesPerAnchorPrefixSum[i];

                    std::copy(
                        task.totalDecodedAnchorsFlat.begin() + decodedSequencePitchInBytes,
                        task.totalDecodedAnchorsFlat.end(),
                        h_outputAnchors + offset * decodedSequencePitchInBytes
                    );

                    std::copy(
                        task.totalDecodedAnchorsLengths.begin() + 1,
                        task.totalDecodedAnchorsLengths.end(),
                        h_anchorSequencesLength + offset
                    );
                }

                cudaMemcpyAsync(
                    d_candidateSequencesLength.data(),
                    h_anchorSequencesLength.data(),
                    sizeof(int) * numCandidates,
                    H2D,
                    stream
                ); CUERR;

                cudaMemcpyAsync(
                    d_candidateSequencesDataDecoded.data(),
                    h_outputAnchors.data(),
                    sizeof(char) * decodedSequencePitchInBytes * numCandidates,
                    H2D,
                    stream
                ); CUERR;

                readextendergpukernels::encodeSequencesTo2BitKernel<8>
                <<<SDIV(numCandidates, (128 / 8)), 128, 0, streams[0]>>>(
                    d_candidateSequencesData.data(),
                    d_candidateSequencesDataDecoded.data(),
                    d_candidateSequencesLength.data(),
                    decodedSequencePitchInBytes,
                    encodedSequencePitchInInts,
                    numCandidates
                ); CUERR;

                //copy "candidate" qualities
                // h_outputAnchorQualities.resize(numCandidates * qualityPitchInBytes);
                // char* d_candidateQualityScores = nullptr;
                // cubAllocator->DeviceAllocate((void**)&d_candidateQualityScores, sizeof(char) * qualityPitchInBytes * numCandidates, stream); CUERR;

                // for(int i = 0; i < numFinishedTasks; i++){
                //     const auto& task = finishedTasks[i];

                //     const int num = h_numCandidatesPerAnchor[i];
                //     const int offset = h_numCandidatesPerAnchorPrefixSum[i];

                //     std::copy(
                //         task.totalDecodedAnchorsFlat.begin() + qualityPitchInBytes,
                //         task.totalDecodedAnchorsFlat.end(),
                //         h_outputAnchorQualities + offset * qualityPitchInBytes
                //     );
                // }

                // cudaMemcpyAsync(
                //     d_candidateQualityScores,
                //     h_outputAnchorQualities.data(),
                //     sizeof(char) * qualityPitchInBytes * numCandidates,
                //     H2D,
                //     stream
                // ); CUERR;

                //synchronize before reusing pinned buffers
                cudaStreamSynchronize(stream); CUERR;

                //sequence data has been transfered to gpu. now set up remaining msa input data

                d_alignment_overlaps.resizeWithoutCopy(numCandidates, stream);
                d_alignment_shifts.resizeWithoutCopy(numCandidates, stream);
                d_alignment_nOps.resizeWithoutCopy(numCandidates, stream);
                d_alignment_best_alignment_flags.resizeWithoutCopy(numCandidates, stream);
                d_isPairedCandidate.resizeWithoutCopy(numCandidates, stream);
                
                helpers::call_fill_kernel_async(d_alignment_overlaps.begin(), numCandidates, 100, stream);
                helpers::call_fill_kernel_async(d_alignment_nOps.begin(), numCandidates, 0, stream);
                helpers::call_fill_kernel_async(d_alignment_best_alignment_flags.begin(), numCandidates, BestAlignment_t::Forward, stream);
                helpers::call_fill_kernel_async(d_isPairedCandidate.begin(), numCandidates, false, stream);

                h_sizeOfGapToMate.resize(numCandidates);
                for(int i = 0; i < numFinishedTasks; i++){
                    const auto& task = finishedTasks[i];

                    const int offset = h_numCandidatesPerAnchorPrefixSum[i];

                    std::copy(
                        task.totalAnchorBeginInExtendedRead.begin() + 1,
                        task.totalAnchorBeginInExtendedRead.end(),
                        h_sizeOfGapToMate + offset
                    );

                    //assert(task.totalAnchorBeginInExtendedRead.back() + task.totalDecodedAnchorsLengths.back() <= insertSize + insertSizeStddev);
                }

                cudaMemcpyAsync(
                    d_alignment_shifts.data(),
                    h_sizeOfGapToMate.data(),
                    sizeof(int) * numCandidates,
                    H2D,
                    stream
                ); CUERR;

                //all input data ready. now set up msa

                

                d_consensusEncoded.resize(numFinishedTasks * resultMSAColumnPitchInElements);
                d_coverage.resize(numFinishedTasks * resultMSAColumnPitchInElements);
                d_msa_column_properties.resize(numFinishedTasks);

                int* d_counts = nullptr;
                cubAllocator->DeviceAllocate((void**)&d_counts, sizeof(int) * numFinishedTasks * 4 * resultMSAColumnPitchInElements, stream); CUERR;

                float* d_weights = nullptr;
                cubAllocator->DeviceAllocate((void**)&d_weights, sizeof(float) * numFinishedTasks * 4 * resultMSAColumnPitchInElements, stream); CUERR;

                int* d_origCoverages = nullptr;
                cubAllocator->DeviceAllocate((void**)&d_origCoverages, sizeof(int) * numFinishedTasks * resultMSAColumnPitchInElements, stream); CUERR;

                float* d_origWeights = nullptr;
                cubAllocator->DeviceAllocate((void**)&d_origWeights, sizeof(float) * numFinishedTasks * resultMSAColumnPitchInElements, stream); CUERR;

                float* d_support = nullptr;
                cubAllocator->DeviceAllocate((void**)&d_support, sizeof(float) * numFinishedTasks * resultMSAColumnPitchInElements, stream); CUERR;

                int* indices1 = nullptr; 
                cubAllocator->DeviceAllocate((void**)&indices1, sizeof(int) * numCandidates, stream); CUERR;

                helpers::lambda_kernel<<<numFinishedTasks, 128, 0, stream>>>(
                    [
                        indices1,
                        d_numCandidatesPerAnchor = d_numCandidatesPerAnchor.data(),
                        d_numCandidatesPerAnchorPrefixSum = d_numCandidatesPerAnchorPrefixSum.data()
                    ] __device__ (){
                        const int num = d_numCandidatesPerAnchor[blockIdx.x];
                        const int offset = d_numCandidatesPerAnchorPrefixSum[blockIdx.x];
                        
                        for(int i = threadIdx.x; i < num; i += blockDim.x){
                            indices1[offset + i] = i;
                        }
                    }
                );


                gpu::GPUMultiMSA multiMSA;

                *h_numAnchors = numFinishedTasks;

                multiMSA.numMSAs = numFinishedTasks;
                multiMSA.columnPitchInElements = resultMSAColumnPitchInElements;
                multiMSA.counts = d_counts;
                multiMSA.weights = d_weights;
                multiMSA.coverages = d_coverage.data();
                multiMSA.consensus = d_consensusEncoded.data();
                multiMSA.support = d_support;
                multiMSA.origWeights = d_origWeights;
                multiMSA.origCoverages = d_origCoverages;
                multiMSA.columnProperties = d_msa_column_properties.data();

                const bool useQualityScoresForMSA = false;

                

                callConstructMultipleSequenceAlignmentsKernel_async(
                    multiMSA,
                    d_alignment_overlaps.data(),
                    d_alignment_shifts.data(),
                    d_alignment_nOps.data(),
                    d_alignment_best_alignment_flags.data(),
                    d_anchorSequencesLength2.data(),
                    d_candidateSequencesLength.data(),
                    indices1,
                    d_numCandidatesPerAnchor.data(),
                    d_numCandidatesPerAnchorPrefixSum.data(),
                    d_subjectSequencesData2.data(),
                    d_candidateSequencesData.data(),
                    d_isPairedCandidate.data(),
                    nullptr, //anchor qualities
                    nullptr, //candidate qualities
                    h_numAnchors.data(), //d_numAnchors
                    goodAlignmentProperties->maxErrorRate,
                    numFinishedTasks,
                    numCandidates,
                    useQualityScoresForMSA,
                    encodedSequencePitchInInts,
                    qualityPitchInBytes,
                    stream,
                    kernelLaunchHandle
                );

                //cubAllocator->DeviceFree(d_candidateQualityScores); CUERR;
                cubAllocator->DeviceFree(d_counts); CUERR;
                cubAllocator->DeviceFree(d_weights); CUERR;
                cubAllocator->DeviceFree(d_origCoverages); CUERR;
                cubAllocator->DeviceFree(d_origWeights); CUERR;
                cubAllocator->DeviceFree(d_support); CUERR;        
                cubAllocator->DeviceFree(indices1); CUERR; 

                //compute quality of consensus
                d_consensusQuality.resize(numFinishedTasks * resultMSAColumnPitchInElements);

                d_subjectSequencesDataDecoded2.resizeWithoutCopy(numFinishedTasks * resultMSAColumnPitchInElements, stream);
                char* d_decodedConsensus = d_subjectSequencesDataDecoded2.data();
                
                h_outputAnchorQualities.resize(numFinishedTasks * resultMSAColumnPitchInElements);
                h_outputAnchors.resize(numFinishedTasks * resultMSAColumnPitchInElements);
                h_anchorSequencesLength.resize(numFinishedTasks);
                
                //compute consensus quality, decoded consensus and compute consensus lengths per anchor
                helpers::lambda_kernel<<<numFinishedTasks, 256, 0, stream>>>(
                    [
                        d_consensusLengths = d_anchorSequencesLength2.data(),
                        d_decodedConsensus = d_decodedConsensus,
                        consensusQuality = d_consensusQuality.data(),
                        support = d_support,
                        coverages = d_coverage.data(),
                        d_encodedConsensus = d_consensusEncoded.data(),
                        msa_column_properties = d_msa_column_properties.data(),
                        d_numCandidatesInMsa = d_numCandidatesPerAnchor.data(),
                        columnPitchInElements = resultMSAColumnPitchInElements,
                        numTasks = numFinishedTasks
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

                        for(int t = blockIdx.x; t < numTasks; t += gridDim.x){

                            const float* const taskSupport = support + t * columnPitchInElements;
                            const int* const taskCoverage = coverages + t * columnPitchInElements;
                            char* const taskConsensusQuality = consensusQuality + t * columnPitchInElements;
                            const int begin = msa_column_properties[t].firstColumn_incl;
                            const int end = msa_column_properties[t].lastColumn_excl;

                            assert(begin >= 0);
                            assert(end < columnPitchInElements);

                            if(threadIdx.x == 0){
                                d_consensusLengths[t] = end - begin;
                            }

                            for(int i = begin + threadIdx.x; i < end; i += blockDim.x){
                                const float support = taskSupport[i];
                                // const float cov = taskCoverage[i];

                                // char q = getQualityChar(support);

                                // //scale down quality depending on coverage
                                // q = char(float(q) * min(1.0f, cov * 1.0f / 5.0f));

                                taskConsensusQuality[i] = getQualityChar(support);
                            }

                            for(int i = begin + threadIdx.x; i < end; i += blockDim.x){
                                const int outpos = i - begin;

                                d_decodedConsensus[t * columnPitchInElements + outpos]
                                    = decodeConsensus(d_encodedConsensus[t * columnPitchInElements + i]);
                            }
                        }
                    }
                ); CUERR;

                cudaMemcpyAsync(
                    h_outputAnchors.data(),
                    d_decodedConsensus,
                    sizeof(char) * numFinishedTasks * resultMSAColumnPitchInElements,
                    D2H,
                    stream
                ); CUERR;

                cudaMemcpyAsync(
                    h_outputAnchorQualities.data(),
                    d_consensusQuality.data(),
                    sizeof(char) * numFinishedTasks * resultMSAColumnPitchInElements,
                    D2H,
                    stream
                ); CUERR;

                cudaMemcpyAsync(
                    h_anchorSequencesLength.data(),
                    d_anchorSequencesLength2.data(),
                    sizeof(int) * numFinishedTasks,
                    D2H,
                    stream
                ); CUERR;

                cudaStreamSynchronize(stream); CUERR;
            }

        }
        nvtx::pop_range();
        #endif

        std::vector<extension::ExtendResult> extendResults;
        extendResults.reserve(finishedTasks.size());

        // int x = 0;

        for(std::size_t t = 0; t < finishedTasks.size(); t++){
            const auto& task = finishedTasks[t];

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

            // std::cerr << "task " << x << ". iteration = " << task.iteration << ", abort = " << task.abort << ", abortReasond = " << extension::to_string(task.abortReason)
            //     << ", matefound = " << task.mateHasBeenFound << ", id = " << task.id << ", myReadid = " << task.myReadId << "\n";

            // x++;

            //construct extended read
            //build msa of all saved totalDecodedAnchors[0]

            const int numsteps = task.totalDecodedAnchorsLengths.size();

            std::string_view decodedAnchor(
                task.totalDecodedAnchorsFlat.data(), 
                task.totalDecodedAnchorsLengths[0]
            );

            std::string_view anchorQuality(
                task.totalAnchorQualityScoresFlat.data(),
                task.totalDecodedAnchorsLengths[0]
            );

            // const int* shifts = task.totalAnchorBeginInExtendedRead.data() + 1;
            // std::vector<float> initialWeights(numsteps-1, 1.0f);

            // MultipleSequenceAlignment::InputData msaInput;
            // msaInput.useQualityScores = false;
            // msaInput.subjectLength = decodedAnchor.length();
            // msaInput.nCandidates = numsteps-1;
            // msaInput.candidatesPitch = decodedSequencePitchInBytes;
            // msaInput.candidateQualitiesPitch = qualityPitchInBytes;
            // msaInput.subject = decodedAnchor.data();
            // msaInput.candidates = task.totalDecodedAnchorsFlat.data() + decodedSequencePitchInBytes;
            // msaInput.subjectQualities = anchorQuality.data();
            // msaInput.candidateQualities = task.totalAnchorQualityScoresFlat.data() + qualityPitchInBytes;
            // msaInput.candidateLengths = task.totalDecodedAnchorsLengths.data() + 1;
            // msaInput.candidateShifts = shifts;
            // msaInput.candidateDefaultWeightFactors = initialWeights.data();

            // MultipleSequenceAlignment msa(qualityConversion);

            // msa.build(msaInput);

            //msa.print(std::cerr);

            const int gpuLength = h_anchorSequencesLength[t];
            // std::string_view gpuConsensus(h_outputAnchors.data() + t * resultMSAColumnPitchInElements, gpuLength);
            // std::string_view gpuConsensusQual(h_outputAnchorQualities.data() + t * resultMSAColumnPitchInElements, gpuLength);
            std::string extendedRead(h_outputAnchors.data() + t * resultMSAColumnPitchInElements, gpuLength);
            std::string extendedReadQuality(h_outputAnchorQualities.data() + t * resultMSAColumnPitchInElements, gpuLength);


            // std::string extendedRead(msa.consensus.begin(), msa.consensus.end());
            // std::string extendedReadQuality(msa.consensus.size(), '\0');
            // std::transform(msa.support.begin(), msa.support.end(), extendedReadQuality.begin(),
            //     [](const float f){
            //         return getQualityChar(f);
            //     }
            // );

            // if(extendedRead != gpuConsensus){
            //     std::cerr << "seq t = " << t << "task.myReadId = " << task.myReadId << ", task.id = " << task.id << ", 'numsteps-1' = " << numsteps-1 << "\n";
            //     std::cerr << "cpu: " << extendedRead << "\n";
            //     std::cerr << "gpu: " << gpuConsensus << "\n";
            //     std::cerr << "cpu: " << extendedReadQuality << "\n";
            //     std::cerr << "gpu: " << gpuConsensusQual << "\n";
            //     assert(false);
            // }

            // if(extendedReadQuality != gpuConsensusQual){
            //     std::cerr << "qual t = " << t << "task.myReadId = " << task.myReadId << ", task.id = " << task.id << ", 'numsteps-1' = " << numsteps-1 << "\n";
            //     std::cerr << "cpu: " << extendedReadQuality << "\n";
            //     std::cerr << "gpu: " << gpuConsensusQual << "\n";
            //     assert(false);
            // }

            

            std::copy(decodedAnchor.begin(), decodedAnchor.end(), extendedRead.begin());
            std::copy(anchorQuality.begin(), anchorQuality.end(), extendedReadQuality.begin());


            //alternative extendedRead. no msa + consensus, just concat
            #if 0

            std::string extendedReadTmp;

            if(numsteps > 1){
                extendedReadTmp.resize(shifts[numsteps - 1] + task.totalDecodedAnchorsLengths.back(), '\0');

                auto b = std::copy(decodedAnchor.begin(), decodedAnchor.end(), extendedReadTmp.begin());

                std::cerr << "debug. copy\n";
                std::copy(
                    decodedAnchor.begin(),
                    decodedAnchor.end(),
                    std::ostream_iterator<char>(std::cerr, "")
                );
                std::cerr << "\n";

                for(int i = 0; i < numsteps - 1; i++){
                    const int currentEnd = std::distance(extendedReadTmp.begin(), b);

                    const int nextLength = task.totalDecodedAnchorsLengths[i];
                    const int nextBegin = shifts[i];

                    std::cerr << nextBegin << " + " << nextLength << " > " << currentEnd << "?\n";

                    if(nextBegin + nextLength > currentEnd){
                        const int copybegin = currentEnd - nextBegin;
                        b = std::copy(
                            task.totalDecodedAnchorsFlat.begin() + (i+1) * decodedSequencePitchInBytes + copybegin,
                            task.totalDecodedAnchorsFlat.begin() + (i+1) * decodedSequencePitchInBytes + copybegin + nextLength,
                            b
                        );

                        std::cerr << "debug. copy\n";
                        std::copy(
                            task.totalDecodedAnchorsFlat.begin() + (i+1) * decodedSequencePitchInBytes + copybegin,
                            task.totalDecodedAnchorsFlat.begin() + (i+1) * decodedSequencePitchInBytes + copybegin + nextLength,
                            std::ostream_iterator<char>(std::cerr, "")
                        );
                        std::cerr << "\n";
                    }
                }

                if(!(b == extendedReadTmp.end())){
                    for(int i = 0; i < numsteps; i++){
                        std::copy(
                            task.totalDecodedAnchorsFlat.begin() + (i) * decodedSequencePitchInBytes,
                            task.totalDecodedAnchorsFlat.begin() + (i+1) * decodedSequencePitchInBytes,
                            std::ostream_iterator<char>(std::cerr, "")
                        );
                        std::cerr << "\n";
                    }

                    std::cerr << "lenghts\n";
                    std::copy(
                        task.totalDecodedAnchorsLengths.begin(),
                        task.totalDecodedAnchorsLengths.end(),
                        std::ostream_iterator<int >(std::cerr, ", ")
                    );
                    std::cerr << "\n";

                    std::cerr << "shifts\n";
                    std::copy(
                        task.totalAnchorBeginInExtendedRead.begin(),
                        task.totalAnchorBeginInExtendedRead.end(),
                        std::ostream_iterator<int >(std::cerr, ", ")
                    );
                    std::cerr << "\n";

                    std::cerr << "extendedReadTmp\n";
                    std::cerr << extendedReadTmp << "\n";

                    std::cerr << "extendedRead\n";
                    std::cerr << extendedRead << "\n";
                }

                assert(b == extendedReadTmp.end());

                // if(extendedReadTmp != extendedRead){
                //     std::cerr << "old: " << extendedRead << "\n";
                //     std::cerr << "new: " << extendedReadTmp << "\n";
                // }
            }else{
                extendedReadTmp = decodedAnchor;
            }

            
            //std::swap(extendedReadTmp, extendedRead);
            #endif




            

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



    //helpers

    void setGpuSegmentIds(
        int* d_segmentIds,
        int numSegments,
        int numElements,
        const int* d_numElementsPerSegment,
        const int* d_numElementsPerSegmentPrefixSum,
        cudaStream_t stream
    ) const {
        cudaMemsetAsync(d_segmentIds, 0, sizeof(int) * numElements, stream); CUERR;
        
        readextendergpukernels::setFirstSegmentIdsKernel<<<SDIV(numSegments, 256), 256, 0, stream>>>(
            d_numElementsPerSegment,
            d_segmentIds,
            d_numElementsPerSegmentPrefixSum,
            numSegments
        );

        std::size_t bytes = 0;

        cudaError_t cubstatus = cub::DeviceScan::InclusiveScan(
            nullptr, 
            bytes, 
            d_segmentIds, 
            d_segmentIds, 
            cub::Max{},
            numElements,
            stream
        );
        assert(cubstatus == cudaSuccess);

        CachedDeviceUVector<char> cubTemp(bytes, stream, *cubAllocator);

        cubstatus = cub::DeviceScan::InclusiveScan(
            cubTemp.data(), 
            bytes, 
            d_segmentIds, 
            d_segmentIds, 
            cub::Max{},
            numElements,
            stream
        );
        assert(cubstatus == cudaSuccess);
    }

    void loadCandidateQualityScores(cudaStream_t stream, char* d_qualityscores){
        char* outputQualityScores = d_qualityscores;

        if(correctionOptions->useQualityScores){
            h_candidateReadIds.resize(totalNumCandidates);
            

            cudaMemcpyAsync(
                h_candidateReadIds.data(),
                d_candidateReadIds.data(),
                sizeof(read_number) * totalNumCandidates,
                D2H,
                stream
            ); CUERR;

            cudaStreamSynchronize(stream); CUERR;


            gpuReadStorage->gatherQualities(
                readStorageHandle,
                outputQualityScores,
                qualityPitchInBytes,
                h_candidateReadIds.data(),
                d_candidateReadIds.data(),
                totalNumCandidates,
                stream
            );

        }else{
            helpers::call_fill_kernel_async(
                outputQualityScores,
                qualityPitchInBytes * totalNumCandidates,
                'I',
                stream
            ); CUERR;
        }
        
    }

    void setStateToFinished(){
        // for(auto&& task : tasks){
        //     addFinishedTask(std::move(task));
        // }
        addSortedFinishedTasks(tasks);
        tasks.clear();

        setState(BatchData::State::Finished);
    }
    
    void addFinishedTask(extension::Task&& task){
        //finished tasks must be stored sorted by pairId. Tasks with same pairId are sorted by id
        auto comp = [](const auto& l, const auto& r){
            return std::tie(l.pairId, l.id) < std::tie(r.pairId, r.id);
        };
        auto where = std::upper_bound(finishedTasks.begin(), finishedTasks.end(), task, comp);
        finishedTasks.insert(where, std::move(task));
    }

    void addSortedFinishedTasks(std::vector<extension::Task>& tasksToAdd){
        //finished tasks must be stored sorted by pairId. Tasks with same pairId are sorted by id

        auto comp = [](const auto& l, const auto& r){
            return std::tie(l.pairId, l.id) < std::tie(r.pairId, r.id);
        };
        assert(std::is_sorted(tasksToAdd.begin(), tasksToAdd.end(), comp));

        std::vector<extension::Task> newFinishedTasks(finishedTasks.size() + tasksToAdd.size());

        std::merge(
            std::make_move_iterator(tasksToAdd.begin()), 
            std::make_move_iterator(tasksToAdd.end()), 
            std::make_move_iterator(finishedTasks.begin()), 
            std::make_move_iterator(finishedTasks.end()), 
            newFinishedTasks.begin(),
            comp
        );

        std::swap(newFinishedTasks, finishedTasks);
    }

    void handleEarlyExitOfTasks4(){

        for(int i = 0; i < numTasks; i++){ 
            const auto& task = tasks[i];
            const int whichtype = task.id % 4;

            //whichtype 0: LR, strand1 searching mate to the right.
            //whichtype 1: LR, strand1 just extend to the right.
            //whichtype 2: RL, strand2 searching mate to the right.
            //whichtype 3: RL, strand2 just extend to the right.

            if(whichtype == 0){
                assert(task.direction == extension::ExtensionDirection::LR);
                assert(task.pairedEnd == true);

                if(task.mateHasBeenFound){
                    for(int k = 1; k <= 4; k++){
                        if(tasks[i + k].pairId == task.pairId){
                            if(tasks[i+k].id == task.id + 1){
                                //disable LR partner task
                                tasks[i + k].abort = true;
                                tasks[i + k].abortReason = extension::AbortReason::PairedAnchorFinished;
                            }else if(tasks[i+k].id == task.id + 2){
                                //disable RL search task
                                tasks[i + k].abort = true;
                                tasks[i + k].abortReason = extension::AbortReason::OtherStrandFoundMate;
                            }
                        }else{
                            break;
                        }
                    }
                }else if(task.abort){
                    for(int k = 1; k <= 4; k++){
                        if(tasks[i + k].pairId == task.pairId){
                            if(tasks[i+k].id == task.id + 1){
                                //disable LR partner task  
                                tasks[i + k].abort = true;
                                tasks[i + k].abortReason = extension::AbortReason::PairedAnchorFinished;
                                break;
                            }
                        }else{
                            break;
                        }
                    }
                }
            }else if(whichtype == 2){
                assert(task.direction == extension::ExtensionDirection::RL);
                assert(task.pairedEnd == true);

                if(task.mateHasBeenFound){
                    if(tasks[i + 1].pairId == task.pairId){
                        if(tasks[i + 1].id == task.id + 1){
                            //disable RL partner task
                            tasks[i + 1].abort = true;
                            tasks[i + 1].abortReason = extension::AbortReason::PairedAnchorFinished;
                        }
                    }

                    for(int k = 1; k <= 2; k++){
                        if(tasks[i - k].pairId == task.pairId){
                            if(tasks[i - k].id == task.id - 2){
                                //disable LR search task
                                tasks[i - k].abort = true;
                                tasks[i - k].abortReason = extension::AbortReason::OtherStrandFoundMate;
                            }
                        }else{
                            break;
                        }
                    }
                    
                }else if(task.abort){
                    if(tasks[i + 1].pairId == task.pairId){
                        if(tasks[i + 1].id == task.id + 1){
                            //disable RL partner task
                            tasks[i + 1].abort = true;
                            tasks[i + 1].abortReason = extension::AbortReason::PairedAnchorFinished;
                        }
                    }
                }
            }
        }
    }






    bool pairedEnd = false;
    State state = State::None;
    int numTasks = 0;
    int numTasksWithMateRemoved = 0;
    int someId = 0;
    int numReadPairs = 0;

    int totalNumCandidates = 0;

    int deviceId{};
    int insertSize{};
    int insertSizeStddev{};
    int maxextensionPerStep{1};
    int minCoverageForExtension{1};
    cub::CachingDeviceAllocator* cubAllocator{};
    const gpu::GpuReadStorage* gpuReadStorage{};
    const gpu::GpuMinhasher* gpuMinhasher{};
    mutable MinhasherHandle minhashHandle{};
    const CorrectionOptions* correctionOptions{};
    const GoodAlignmentProperties* goodAlignmentProperties{};
    const cpu::QualityScoreConversion* qualityConversion{};
    mutable ReadStorageHandle readStorageHandle{};
    mutable gpu::KernelLaunchHandle kernelLaunchHandle{};

    std::size_t encodedSequencePitchInInts = 0;
    std::size_t decodedSequencePitchInBytes = 0;
    std::size_t msaColumnPitchInElements = 0;
    std::size_t qualityPitchInBytes = 0;

    std::size_t outputAnchorPitchInBytes = 0;
    std::size_t outputAnchorQualityPitchInBytes = 0;
    std::size_t decodedMatesRevCPitchInBytes = 0;

    
    PinnedBuffer<read_number> h_candidateReadIds{};

    PinnedBuffer<int> h_segmentIdsOfReadIds{};

    DeviceBuffer<unsigned int> d_anchormatedata{};

    DeviceBuffer<int> d_anchorIndicesWithRemovedMates{};

    PinnedBuffer<int> h_numCandidatesPerAnchor{};
    PinnedBuffer<int> h_numCandidatesPerAnchorPrefixSum{};



    PinnedBuffer<int> h_numAnchors{};
    PinnedBuffer<int> h_numCandidates{};
    DeviceBuffer<int> d_numAnchors{};
    DeviceBuffer<int> d_numCandidates{};
    DeviceBuffer<int> d_numCandidates2{};
    PinnedBuffer<int> h_numAnchorsWithRemovedMates{};

    // ----- candidate data
    CachedDeviceUVector<unsigned int> d_candidateSequencesData{};
    CachedDeviceUVector<int> d_candidateSequencesLength{};    
    CachedDeviceUVector<read_number> d_candidateReadIds{};
    CachedDeviceUVector<bool> d_isPairedCandidate{};
    CachedDeviceUVector<int> d_anchorIndicesOfCandidates{};
    CachedDeviceUVector<int> d_alignment_overlaps{};
    CachedDeviceUVector<int> d_alignment_shifts{};
    CachedDeviceUVector<int> d_alignment_nOps{};
    CachedDeviceUVector<BestAlignment_t> d_alignment_best_alignment_flags{};

    CachedDeviceUVector<int> d_numCandidatesPerAnchor{};
    CachedDeviceUVector<int> d_numCandidatesPerAnchorPrefixSum{};
    // ----- 
    

    // ----- staging buffers for input
    PinnedBuffer<char> h_anchorQualityScores{};
    PinnedBuffer<char> h_subjectSequencesDataDecoded{};
    PinnedBuffer<read_number> h_anchorReadIds{};
    PinnedBuffer<read_number> h_mateReadIds{};
    PinnedBuffer<int> h_anchorSequencesLength{};
    PinnedBuffer<unsigned int> h_inputanchormatedata{};
    PinnedBuffer<int> h_inputMateLengths;
    PinnedBuffer<bool> h_isPairedTask;
    // ----- 

    // ----- input data

    CachedDeviceUVector<unsigned int> d_inputanchormatedata{};
    CachedDeviceUVector<char> d_subjectSequencesDataDecoded{};
    CachedDeviceUVector<char> d_anchorQualityScores{};
    CachedDeviceUVector<int> d_anchorSequencesLength{};
    CachedDeviceUVector<read_number> d_anchorReadIds{};
    CachedDeviceUVector<read_number> d_mateReadIds{};
    CachedDeviceUVector<int> d_inputMateLengths{};
    CachedDeviceUVector<bool> d_isPairedTask{};
    CachedDeviceUVector<unsigned int> d_subjectSequencesData{};

    // -----

    DeviceBuffer<read_number> d_usedReadIds{};
    DeviceBuffer<int> d_numUsedReadIdsPerAnchor{};
    DeviceBuffer<int> d_numUsedReadIdsPerAnchorPrefixSum{};
    DeviceBuffer<int> d_segmentIdsOfUsedReadIds{};
    DeviceBuffer<read_number> d_usedReadIds2{};
    DeviceBuffer<int> d_numUsedReadIdsPerAnchor2{};
    DeviceBuffer<int> d_numUsedReadIdsPerAnchorPrefixSum2{};
    DeviceBuffer<int> d_segmentIdsOfUsedReadIds2{};
    PinnedBuffer<int> h_numUsedReadIds{};

    DeviceBuffer<read_number> d_fullyUsedReadIds{};
    DeviceBuffer<int> d_numFullyUsedReadIdsPerAnchor{};
    DeviceBuffer<int> d_numFullyUsedReadIdsPerAnchorPrefixSum{};
    DeviceBuffer<int> d_segmentIdsOfFullyUsedReadIds{};

    DeviceBuffer<read_number> d_fullyUsedReadIds2{};
    DeviceBuffer<int> d_numFullyUsedReadIdsPerAnchor2{};
    DeviceBuffer<int> d_numFullyUsedReadIdsPerAnchorPrefixSum2{};
    DeviceBuffer<int> d_segmentIdsOfFullyUsedReadIds2{};
    PinnedBuffer<int> h_numFullyUsedReadIds{};
    

    PinnedBuffer<int> h_newPositionsOfActiveTasks{};
    PinnedBuffer<int> d_newPositionsOfActiveTasks{};

    DeviceBuffer<std::uint8_t> d_consensusEncoded; //encoded , 0-4
    DeviceBuffer<int> d_coverage;
    DeviceBuffer<gpu::MSAColumnProperties> d_msa_column_properties;

    DeviceBuffer<char> d_consensusQuality;

    DeviceBuffer<char> d_outputAnchors;
    DeviceBuffer<char> d_outputAnchorQualities;
    DeviceBuffer<bool> d_outputMateHasBeenFound;
    DeviceBuffer<extension::AbortReason> d_abortReasons;
    DeviceBuffer<int> d_outputAnchorLengths{};
    DeviceBuffer<bool> d_isFullyUsedCandidate{};

    PinnedBuffer<int> h_firstTasksOfPairsToCheck;

    PinnedBuffer<int> h_accumExtensionsLengths;
    PinnedBuffer<extension::AbortReason> h_abortReasons;
    PinnedBuffer<char> h_outputAnchors;
    PinnedBuffer<char> h_outputAnchorQualities;
    PinnedBuffer<int> h_outputAnchorLengths;
    PinnedBuffer<bool> h_outputMateHasBeenFound;
    PinnedBuffer<int> h_sizeOfGapToMate;
    PinnedBuffer<bool> h_isFullyUsedCandidate{};

    // tasḱ_accumExtensionLengths;
    // tasḱ_pairedEnd;
    // tasḱ_decodedMateRevC            
    // tasḱ_numRemainingCandidates
    // tasḱ_abortReason
    // tasḱ_mateHasBeenFound
    // tasḱ_currentAnchorLength
    // tasḱ_totalDecodedAnchors
    // tasḱ_totalAnchorQualityScores
    // tasḱ_totalAnchorBeginInExtendedRead         
    // tasḱ_abort
    // tasḱ_iteration
    // tasḱ_direction;
    // tasḱ_myReadId;
    // tasḱ_mateReadId;
    // tasḱ_myLength;
    // tasḱ_mateLength;
    // tasḱ_mateQualityScoresReversed
    // tasḱ_mateHasBeenFound;

    
    std::array<CudaEvent, 1> events{};
    std::array<cudaStream_t, 4> streams{};
    std::vector<extension::Task> tasks;
    std::vector<extension::Task> finishedTasks;

};


}


#endif