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
        readStorageHandle(gpuReadStorage->makeHandle())
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
    void addTasks(TaskIter tasksBegin, TaskIter tasksEnd){
        const int numAdditionalTasks = std::distance(tasksBegin, tasksEnd);
        assert(numAdditionalTasks % 4 == 0);
        if(numAdditionalTasks == 0) return;

        const int currentNumTasks = tasks.size();
        const int newNumTasks = currentNumTasks + numAdditionalTasks;


        void* cubTemp = nullptr;
        std::size_t cubTempSize = 0;
        cudaError_t cubstatus = cudaSuccess;

        h_anchorReadIds.resize(newNumTasks);
        h_mateReadIds.resize(newNumTasks);
        h_subjectSequencesDataDecoded.resize(newNumTasks * decodedSequencePitchInBytes);
        h_anchorSequencesLength.resize(newNumTasks);
        h_anchorQualityScores.resize(newNumTasks * qualityPitchInBytes);
        h_inputanchormatedata.resize(newNumTasks * encodedSequencePitchInInts);


        //d_anchorIndicesWithRemovedMates.resize(newNumTasks); // ???

        d_numUsedReadIdsPerAnchor2.resize(newNumTasks);
        d_numUsedReadIdsPerAnchorPrefixSum.resize(newNumTasks);
        d_numFullyUsedReadIdsPerAnchor2.resize(newNumTasks);
        d_numFullyUsedReadIdsPerAnchorPrefixSum.resize(newNumTasks);

        d_anchorReadIds2.resize(newNumTasks);
        d_mateReadIds2.resize(newNumTasks);
        d_subjectSequencesData2.resize(newNumTasks * encodedSequencePitchInInts);
        d_subjectSequencesDataDecoded2.resize(newNumTasks * decodedSequencePitchInBytes);
        d_anchorSequencesLength2.resize(newNumTasks);
        d_anchorQualityScores2.resize(newNumTasks * qualityPitchInBytes);
        d_inputanchormatedata2.resize(newNumTasks * encodedSequencePitchInInts);

        cudaMemcpyAsync(
            d_anchorReadIds2.data(), 
            d_anchorReadIds.data(),
            sizeof(int) * currentNumTasks,
            D2D,
            streams[0]
        ); CUERR;
        std::swap(d_anchorReadIds, d_anchorReadIds2);

        cudaMemcpyAsync(
            d_mateReadIds2.data(), 
            d_mateReadIds.data(),
            sizeof(int) * currentNumTasks,
            D2D,
            streams[0]
        ); CUERR;
        std::swap(d_mateReadIds, d_mateReadIds2);

        cudaMemcpyAsync(
            d_subjectSequencesDataDecoded2.data(), 
            d_subjectSequencesDataDecoded.data(),
            sizeof(char) * currentNumTasks * decodedSequencePitchInBytes,
            D2D,
            streams[0]
        ); CUERR;
        std::swap(d_subjectSequencesDataDecoded, d_subjectSequencesDataDecoded2);

        cudaMemcpyAsync(
            d_subjectSequencesData2.data(), 
            d_subjectSequencesData.data(),
            sizeof(unsigned int) * currentNumTasks * encodedSequencePitchInInts,
            D2D,
            streams[0]
        ); CUERR;
        std::swap(d_subjectSequencesData, d_subjectSequencesData2);

        cudaMemcpyAsync(
            d_anchorSequencesLength2.data(), 
            d_anchorSequencesLength.data(),
            sizeof(int) * currentNumTasks,
            D2D,
            streams[0]
        ); CUERR;
        std::swap(d_anchorSequencesLength, d_anchorSequencesLength2);

        cudaMemcpyAsync(
            d_anchorQualityScores2.data(), 
            d_anchorQualityScores.data(),
            sizeof(char) * currentNumTasks * qualityPitchInBytes,
            D2D,
            streams[0]
        ); CUERR;
        std::swap(d_anchorQualityScores, d_anchorQualityScores2);

        cudaMemcpyAsync(
            d_inputanchormatedata2.data(), 
            d_inputanchormatedata.data(),
            sizeof(unsigned int) * currentNumTasks * encodedSequencePitchInInts,
            D2D,
            streams[0]
        ); CUERR;
        std::swap(d_inputanchormatedata, d_inputanchormatedata2);

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

        cubAllocator->DeviceAllocate((void**)&cubTemp, cubTempSize, streams[0]);  CUERR;

        cubstatus = cub::DeviceScan::ExclusiveSum(
            cubTemp,
            cubTempSize,
            d_numUsedReadIdsPerAnchor.data(), 
            d_numUsedReadIdsPerAnchorPrefixSum.data(), 
            newNumTasks, 
            streams[0]
        );
        assert(cudaSuccess == cubstatus);

        cubAllocator->DeviceFree(cubTemp); CUERR;



        cubstatus = cub::DeviceScan::ExclusiveSum(
            nullptr,
            cubTempSize,
            d_numFullyUsedReadIdsPerAnchor.data(), 
            d_numFullyUsedReadIdsPerAnchorPrefixSum.data(), 
            newNumTasks, 
            streams[0]
        );
        assert(cudaSuccess == cubstatus);

        cubAllocator->DeviceAllocate((void**)&cubTemp, cubTempSize, streams[0]);  CUERR;

        cubstatus = cub::DeviceScan::ExclusiveSum(
            cubTemp,
            cubTempSize,
            d_numFullyUsedReadIdsPerAnchor.data(), 
            d_numFullyUsedReadIdsPerAnchorPrefixSum.data(), 
            newNumTasks, 
            streams[0]
        );
        assert(cudaSuccess == cubstatus);

        cubAllocator->DeviceFree(cubTemp); CUERR;



        for(int t = 0; t < numAdditionalTasks; t++){
            const auto& task = *(tasksBegin + t);

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
                task.currentQualityScores.begin(),
                task.currentQualityScores.end(),
                h_anchorQualityScores.begin() + t * qualityPitchInBytes
            );
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

        cudaMemcpyAsync(
            d_anchorReadIds.data() + currentNumTasks,
            h_anchorReadIds.data(),
            sizeof(read_number) * numAdditionalTasks,
            H2D,
            streams[0]
        ); CUERR;

        cudaMemcpyAsync(
            d_mateReadIds.data() + currentNumTasks,
            h_mateReadIds.data(),
            sizeof(read_number) * numAdditionalTasks,
            H2D,
            streams[0]
        ); CUERR;

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




        //save tasks and update indices of active tasks

        tasks.insert(tasks.end(), std::make_move_iterator(tasksBegin), std::make_move_iterator(tasksEnd));
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

        nvtx::pop_range();
    }

    void getCandidateReadIds(){
        assert(state == BatchData::State::BeforeHash);

        cudaStream_t stream = streams[0];

        d_numCandidatesPerAnchor.resize(numTasks);
        d_numCandidatesPerAnchorPrefixSum.resize(numTasks + 1);

        int totalNumValues = 0;

        gpuMinhasher->determineNumValues(
            minhashHandle,
            d_subjectSequencesData.get(),
            encodedSequencePitchInInts,
            d_anchorSequencesLength.get(),
            numTasks,
            d_numCandidatesPerAnchor.get(),
            totalNumValues,
            stream
        );

        cudaStreamSynchronize(stream); CUERR;

        d_candidateReadIds.resize(totalNumValues);    

        if(totalNumValues == 0){
            cudaMemsetAsync(d_numCandidatesPerAnchor.get(), 0, sizeof(int) * numTasks , stream); CUERR;
            cudaMemsetAsync(d_numCandidatesPerAnchorPrefixSum.get(), 0, sizeof(int) * (1 + numTasks), stream); CUERR;
            totalNumCandidates = 0;

            for(auto&& task : tasks){
                addFinishedTask(std::move(task));
            }

            setState(BatchData::State::Finished);
            return;
        }

        gpuMinhasher->retrieveValues(
            minhashHandle,
            nullptr,
            numTasks,              
            totalNumValues,
            d_candidateReadIds.get(),
            d_numCandidatesPerAnchor.get(),
            d_numCandidatesPerAnchorPrefixSum.get(),
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
        
        d_anchorIndicesOfCandidates.resize(totalNumCandidates);

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
        
        void* cubtempstorage; cubAllocator->DeviceAllocate((void**)&cubtempstorage, cubBytes, firstStream);   
        
        helpers::call_fill_kernel_async(d_shouldBeKept, totalNumCandidates, false, firstStream);

        d_numCandidatesPerAnchor2.resize(numTasks);

        
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
            cubtempstorage,
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
            cubtempstorage,
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
    
            void* cubtempstream2 = nullptr; cubAllocator->DeviceAllocate((void**)&cubtempstream2, cubtempstream2bytes, secondStream);
                
            assert(d_anchormatedata.data() != nullptr);

            cubstatus = cub::DeviceSelect::Flagged(
                cubtempstream2,
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

            cubAllocator->DeviceFree(cubtempstream2);
        }

        cubAllocator->DeviceFree(d_anchorFlags); CUERR;

        d_numCandidatesPerAnchorPrefixSum2.resize(numTasks + 1);

        // //compute prefix sum of number of candidates per anchor
        helpers::call_set_kernel_async(d_numCandidatesPerAnchorPrefixSum2.data(), 0, 0, firstStream);

        cubstatus = cub::DeviceScan::InclusiveSum(
            cubtempstorage, 
            cubBytes, 
            d_numCandidatesPerAnchor2.data(), 
            d_numCandidatesPerAnchorPrefixSum2.data() + 1, 
            numTasks,
            firstStream
        );
        assert(cubstatus == cudaSuccess);

        #if 0

        //compute segment ids for candidate read ids
        cudaEventSynchronize(events[0]); CUERR; //wait for h_numCandidatesPerAnchor

        for(int i = 0, sum = 0; i < numTasks; i++){
            std::fill(
                h_segmentIdsOfReadIds.data() + sum,
                h_segmentIdsOfReadIds.data() + sum + h_numCandidatesPerAnchor[i],
                i
            );
            sum += h_numCandidatesPerAnchor[i];
        }

        cudaMemcpyAsync(
            d_anchorIndicesOfCandidates.data(),
            h_segmentIdsOfReadIds.data(),
            sizeof(int) * totalNumCandidates,
            H2D,
            firstStream
        ); CUERR;

        #else

        helpers::call_fill_kernel_async(d_anchorIndicesOfCandidates.data(), totalNumCandidates, 0, firstStream);

        readextendergpukernels::setFirstSegmentIdsKernel<<<SDIV(numTasks, 256), 256, 0, firstStream>>>(
            d_numCandidatesPerAnchor2.data(),
            d_anchorIndicesOfCandidates.data(),
            d_numCandidatesPerAnchorPrefixSum2.data(),
            numTasks
        );

        cubstatus = cub::DeviceScan::InclusiveScan(
            cubtempstorage, 
            cubBytes, 
            d_anchorIndicesOfCandidates.data(), 
            d_anchorIndicesOfCandidates.data(), 
            cub::Max{},
            totalNumCandidates,
            firstStream
        );
        assert(cubstatus == cudaSuccess);

        #endif
   

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
            cubtempstorage, 
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

        cubAllocator->DeviceFree(cubtempstorage); CUERR;

        //removeUsedIdsAndMateIds is a compaction step. check early exit.
        if(totalNumCandidates == 0){
            setState(BatchData::State::Finished);
        }else{
            setState(BatchData::State::BeforeComputePairFlags);
        }
    }

    void computePairFlagsGpu() {
        assert(state == BatchData::State::BeforeComputePairFlags);

        cudaStream_t stream = streams[0];
        DEBUGDEVICESYNC

        d_isPairedCandidate.resize(totalNumCandidates);

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

            // int* d_status = nullptr;
            // //cubAllocator->DeviceAllocate((void**)&d_status, sizeof(int) * numChecks); CUERR;
            // cudaMallocHost(&d_status, sizeof(int) * numChecks); CUERR;

            // std::fill(d_status, d_status + numChecks, 0);

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

        d_candidateSequencesLength.resize(totalNumCandidates);
        d_candidateSequencesData.resize(encodedSequencePitchInInts * totalNumCandidates);

        assert(d_candidateReadIds.size() >= totalNumCandidates);

        gpuReadStorage->gatherSequences(
            readStorageHandle,
            d_candidateSequencesData.get(),
            encodedSequencePitchInInts,
            h_candidateReadIds.get(),
            d_candidateReadIds.get(), //device accessible
            totalNumCandidates,
            stream
        );

        gpuReadStorage->gatherSequenceLengths(
            readStorageHandle,
            d_candidateSequencesLength.get(),
            d_candidateReadIds.get(),
            totalNumCandidates,
            stream
        );

        setState(BatchData::State::BeforeEraseData);
    }

    void eraseDataOfRemovedMates(){
        assert(state == BatchData::State::BeforeEraseData);

        if(numTasksWithMateRemoved > 0){

            cudaStream_t stream = streams[0];

            d_candidateSequencesData2.resize(encodedSequencePitchInInts * totalNumCandidates);

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

            cubAllocator->DeviceAllocate((void**)&cubTemp, cubTempSize, stream); CUERR;

            cubstatus = cub::DeviceScan::InclusiveSum(
                cubTemp,
                cubTempSize,
                d_numCandidatesPerAnchor.data(), 
                d_numCandidatesPerAnchorPrefixSum.data() + 1, 
                numTasks, 
                stream
            );
            assert(cudaSuccess == cubstatus);

            cubAllocator->DeviceFree(cubTemp); CUERR;

            cudaMemcpyAsync(
                h_numCandidates.data(),
                d_numCandidatesPerAnchorPrefixSum.data() + numTasks,
                sizeof(int),
                D2H,
                stream
            ); CUERR;

            cudaStreamSynchronize(stream); CUERR;

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
            //     std::cerr << "Offsets after erasedataofremovedmates:\n";
            //     for(int i = 0; i < numTasks+1; i++){
            //         std::cerr << offsets[i] << " ";
            //     }
            //     std::cerr << "\n";

            // }

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


        d_alignment_overlaps.resize(totalNumCandidates);
        d_alignment_shifts.resize(totalNumCandidates);
        d_alignment_nOps.resize(totalNumCandidates);
        d_alignment_best_alignment_flags.resize(totalNumCandidates);

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
                d_alignment_overlaps.get(),
                d_alignment_shifts.get(),
                d_alignment_nOps.get(),
                d_alignment_isValid,
                d_alignment_best_alignment_flags.get(),
                d_subjectSequencesData.get(),
                d_candidateSequencesData.get(),
                d_anchorSequencesLength.get(),
                d_candidateSequencesLength.get(),
                d_numCandidatesPerAnchorPrefixSum.get(),
                d_numCandidatesPerAnchor.get(),
                d_anchorIndicesOfCandidates.get(),
                h_numAnchors.get(),
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

        void* d_tempstorage = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_tempstorage, tempstoragebytes, stream); CUERR;

        callAlignmentKernel(d_tempstorage, tempstoragebytes);

        cubAllocator->DeviceFree(d_tempstorage); CUERR;

        cubAllocator->DeviceFree(d_alignment_isValid); CUERR;

        setState(BatchData::State::BeforeAlignmentFilter);
    }

    void filterAlignments(){
        assert(state == BatchData::State::BeforeAlignmentFilter);

        cudaStream_t stream = streams[0];


        DEBUGDEVICESYNC

        const int numAnchors = numTasks;

        d_numCandidatesPerAnchor2.resize(numTasks);
        h_numCandidates.resize(1);

        d_candidateSequencesData2.resize(encodedSequencePitchInInts * totalNumCandidates);

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

        std::size_t cubTempSize = 0;
        void* cubTemp = nullptr;

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

        DEBUGDEVICESYNC

        cubAllocator->DeviceAllocate((void**)&cubTemp, cubTempSize, stream); CUERR;

        cubstatus = cub::DeviceSelect::Flagged(
            cubTemp, 
            cubTempSize, 
            d_zip_data, 
            d_keepflags, 
            d_zip_data_tmp, 
            h_numCandidates.data(), 
            totalNumCandidates, 
            stream
        );
        assert(cubstatus == cudaSuccess);

        DEBUGDEVICESYNC

        cubAllocator->DeviceFree(cubTemp); CUERR;

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

        cubAllocator->DeviceAllocate((void**)&cubTemp, cubTempSize, stream); CUERR;

        DEBUGDEVICESYNC

        cubstatus = cub::DeviceSelect::Flagged(
            cubTemp,
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

        cubAllocator->DeviceFree(cubTemp); CUERR;

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

        cubAllocator->DeviceAllocate((void**)&cubTemp, cubTempSize, stream); CUERR;

        cubstatus = cub::DeviceScan::InclusiveSum(
            cubTemp,
            cubTempSize,
            d_numCandidatesPerAnchor2.data(), 
            d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            numTasks, 
            stream
        );
        assert(cudaSuccess == cubstatus);

        cubAllocator->DeviceFree(cubTemp); CUERR;

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
            setState(BatchData::State::Finished);
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

        d_numCandidatesPerAnchor2.resize(numTasks);

        int* indices1 = nullptr; 
        int* indices2 = nullptr;
        cubAllocator->DeviceAllocate((void**)&indices1, sizeof(int) * totalNumCandidates, firstStream); CUERR;
        cubAllocator->DeviceAllocate((void**)&indices2, sizeof(int) * totalNumCandidates, firstStream); CUERR;

        

        helpers::lambda_kernel<<<numTasks, 128, 0, firstStream>>>(
            [
                indices1,
                d_numCandidatesPerAnchor = d_numCandidatesPerAnchor.get(),
                d_numCandidatesPerAnchorPrefixSum = d_numCandidatesPerAnchorPrefixSum.get()
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
        multiMSA.coverages = d_coverage.get();
        multiMSA.consensus = d_consensusEncoded.get();
        multiMSA.support = d_support;
        multiMSA.origWeights = d_origWeights;
        multiMSA.origCoverages = d_origCoverages;
        multiMSA.columnProperties = d_msa_column_properties.get();

        const bool useQualityScoresForMSA = true;

        callConstructMultipleSequenceAlignmentsKernel_async(
            multiMSA,
            d_alignment_overlaps.get(),
            d_alignment_shifts.get(),
            d_alignment_nOps.get(),
            d_alignment_best_alignment_flags.get(),
            d_anchorSequencesLength.get(),
            d_candidateSequencesLength.get(),
            indices1, //d_indices,
            d_numCandidatesPerAnchor.get(),
            d_numCandidatesPerAnchorPrefixSum.get(),
            d_subjectSequencesData.get(),
            d_candidateSequencesData.get(),
            d_isPairedCandidate.get(),
            d_anchorQualityScores.get(), //d_anchor_qualities.get(),
            d_candidateQualityScores,
            h_numAnchors.get(), //d_numAnchors
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
            d_numCandidates2.get(),
            multiMSA,
            d_alignment_best_alignment_flags.get(),
            d_alignment_shifts.get(),
            d_alignment_nOps.get(),
            d_alignment_overlaps.get(),
            d_subjectSequencesData.get(),
            d_candidateSequencesData.get(),
            d_isPairedCandidate.get(),
            d_anchorSequencesLength.get(),
            d_candidateSequencesLength.get(),
            d_anchorQualityScores.get(), //d_anchor_qualities.get(),
            d_candidateQualityScores,
            d_shouldBeKept,
            d_numCandidatesPerAnchorPrefixSum.get(),
            h_numAnchors.get(),
            goodAlignmentProperties->maxErrorRate,
            numTasks,
            totalNumCandidates,
            useQualityScoresForMSA, //correctionOptions->useQualityScores,
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            indices1, //d_indices,
            d_numCandidatesPerAnchor.get(),
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
                d_numCandidatesPerAnchor2 = d_numCandidatesPerAnchor2.get(),
                d_numCandidatesPerAnchorPrefixSum = d_numCandidatesPerAnchorPrefixSum.get()
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

        cubAllocator->DeviceAllocate((void**)&cubtemp, cubBytes, firstStream);

        cubstatus = cub::DeviceSelect::Flagged(
            cubtemp,
            cubBytes,
            d_candidateReadIds.data(),
            d_shouldBeKept,
            d_candidateReadIds2,
            h_numCandidates.data(),
            totalNumCandidates,
            firstStream
        );

        assert(cubstatus == cudaSuccess);
        cubAllocator->DeviceFree(cubtemp);

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
            d_numCandidatesPerAnchor2.get(), 
            d_numCandidatesPerAnchorPrefixSum.get() + 1, 
            numTasks, 
            firstStream
        );
        assert(cubstatus == cudaSuccess);

        cubAllocator->DeviceAllocate((void**)&cubtemp, cubBytes, firstStream); CUERR;

        cubstatus = cub::DeviceScan::InclusiveSum(
            cubtemp,
            cubBytes,
            d_numCandidatesPerAnchor2.get(), 
            d_numCandidatesPerAnchorPrefixSum.get() + 1, 
            numTasks, 
            firstStream
        );
        assert(cubstatus == cudaSuccess);
        cubAllocator->DeviceFree(cubtemp); CUERR;

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
        h_inputMateLengths.resize(numTasks);
        h_abortReasons.resize(numTasks);
        h_outputAnchors.resize(numTasks * outputAnchorPitchInBytes);
        h_outputAnchorQualities.resize(numTasks * outputAnchorQualityPitchInBytes);
        h_outputAnchorLengths.resize(numTasks);
        h_isPairedTask.resize(numTasks);
        h_decodedMatesRevC.resize(numTasks * decodedMatesRevCPitchInBytes);
        h_outputMateHasBeenFound.resize(numTasks);
        h_sizeOfGapToMate.resize(numTasks);
        h_isFullyUsedCandidate.resize(totalNumCandidates);

        h_scatterMap.resize(numTasks);

        int* d_accumExtensionsLengths = nullptr;
        int* d_inputMateLengths = nullptr;
        int* d_accumExtensionsLengthsOUT = nullptr;
        bool* d_isPairedTask = nullptr;
        char* d_decodedMatesRevC = nullptr;
        int* d_sizeOfGapToMate = nullptr;

        cubAllocator->DeviceAllocate((void**)&d_accumExtensionsLengths, sizeof(int) * numTasks, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_inputMateLengths, sizeof(int) * numTasks, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_accumExtensionsLengthsOUT, sizeof(int) * numTasks, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_isPairedTask, sizeof(bool) * numTasks, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_decodedMatesRevC, sizeof(char) * numTasks * decodedMatesRevCPitchInBytes, stream); CUERR;
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

        // helpers::call_fill_kernel_async(
        //     thrust::make_zip_iterator(thrust::make_tuple(d_outputMateHasBeenFound, d_abortReasons)),
        //     numTasks,
        //     thrust::make_tuple(false, extension::AbortReason::None),
        //     stream
        // );

        for(int i = 0; i < numTasks; i++){
            const int index = i;
            const auto& task = tasks[index];

            h_accumExtensionsLengths[i] = task.accumExtensionLengths;
            h_inputMateLengths[i] = task.mateLength;
            h_isPairedTask[i] = task.pairedEnd;
        }

        helpers::call_copy_n_kernel(
            thrust::make_zip_iterator(thrust::make_tuple(
                h_accumExtensionsLengths.data(),
                h_inputMateLengths.data(),
                h_isPairedTask.data()
            )),
            numTasks,
            thrust::make_zip_iterator(thrust::make_tuple(
                d_accumExtensionsLengths,
                d_inputMateLengths,
                d_isPairedTask
            )),
            stream
        );

        int numPairedEndTasks = 0;
        for(int i = 0; i < numTasks; i++){
            const int index = i;
            const auto& task = tasks[index];

            if(task.pairedEnd){

                // assert(task.decodedMateRevC.size() <= decodedMatesRevCPitchInBytes);
                // std::copy(task.decodedMateRevC.begin(), task.decodedMateRevC.end(), &h_decodedMatesRevC[i * decodedMatesRevCPitchInBytes]);
                std::copy(task.decodedMateRevC.begin(), task.decodedMateRevC.end(), &h_decodedMatesRevC[numPairedEndTasks * decodedMatesRevCPitchInBytes]);
                h_scatterMap[numPairedEndTasks] = i;
                numPairedEndTasks++;
            }
        }

        char* d_decodedMatesRevCDense = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_decodedMatesRevCDense, sizeof(char) * numTasks * decodedMatesRevCPitchInBytes, stream); CUERR;
        int* d_scatterMap = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_scatterMap, sizeof(int) * numTasks, stream); CUERR;

        cudaMemcpyAsync(
            d_decodedMatesRevCDense,
            h_decodedMatesRevC.data(),
            sizeof(char) * decodedMatesRevCPitchInBytes * numPairedEndTasks,
            H2D,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            d_scatterMap,
            h_scatterMap.data(),
            sizeof(int) * numPairedEndTasks,
            H2D,
            stream
        ); CUERR;

        helpers::lambda_kernel<<<numTasks, 128, 0, stream>>>(
            [
                numPairedEndTasks = numPairedEndTasks,
                decodedMatesRevCPitchInBytes = decodedMatesRevCPitchInBytes,
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
                d_inputMateLengths = (int*)d_inputMateLengths,
                d_abortReasons = d_abortReasons.data(),
                d_accumExtensionsLengthsOUT = (int*)d_accumExtensionsLengthsOUT,
                d_outputAnchors = (char*)d_outputAnchors,
                outputAnchorPitchInBytes = outputAnchorPitchInBytes,
                d_outputAnchorQualities = (char*)d_outputAnchorQualities,
                outputAnchorQualityPitchInBytes = outputAnchorQualityPitchInBytes,
                d_outputAnchorLengths = d_outputAnchorLengths.data(),
                d_isPairedTask = (bool*)d_isPairedTask,
                d_decodedMatesRevC = (char*)d_decodedMatesRevC,
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
        cubAllocator->DeviceFree(d_inputMateLengths); CUERR;
        cubAllocator->DeviceFree(d_accumExtensionsLengthsOUT); CUERR;
        cubAllocator->DeviceFree(d_isPairedTask); CUERR;
        cubAllocator->DeviceFree(d_decodedMatesRevC); CUERR;
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

            void* cubtemp = nullptr;
            cubAllocator->DeviceAllocate((void**)&cubtemp, bytes, stream);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                cubtemp, 
                bytes, 
                d_numUsedReadIdsPerAnchor.data(), 
                d_numUsedReadIdsPerAnchorPrefixSum.data(), 
                numTasks,
                stream
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceFree(cubtemp);

            *h_numUsedReadIds = newsize;

        }

        {
            std::size_t bytes = 0;
            void* cubtemp = nullptr;
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

            cubAllocator->DeviceAllocate((void**)&cubtemp, bytes);

            cubstatus = cub::DeviceSelect::Flagged(
                cubtemp,
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

            cubAllocator->DeviceFree(cubtemp);

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

            cubAllocator->DeviceAllocate((void**)&cubtemp, bytes);

            cubstatus = cub::DeviceSegmentedReduce::Sum(
                cubtemp,
                bytes,
                d_isFullyUsedCandidate.data(),
                d_currentNumFullyUsedreadIdsPerAnchor,
                numTasks,
                d_numCandidatesPerAnchorPrefixSum.data(),
                d_numCandidatesPerAnchorPrefixSum.data() + 1,
                stream
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceFree(cubtemp);

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

            cubAllocator->DeviceAllocate((void**)&cubtemp, bytes, stream);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                cubtemp, 
                bytes, 
                d_currentNumFullyUsedreadIdsPerAnchor, 
                d_currentNumFullyUsedreadIdsPerAnchorPS, 
                numTasks,
                stream
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceFree(cubtemp);


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

            cubAllocator->DeviceAllocate((void**)&cubtemp, bytes, stream);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                cubtemp, 
                bytes, 
                d_numFullyUsedReadIdsPerAnchor.data(), 
                d_numFullyUsedReadIdsPerAnchorPrefixSum.data(), 
                numTasks,
                stream
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceFree(cubtemp);

            *h_numFullyUsedReadIds = newsize;

        }

        setState(BatchData::State::BeforeCopyToHost);
    }
    
    void copyBuffersToHost(){
        assert(state == BatchData::State::BeforeCopyToHost);

        nvtx::push_range("copyBuffersToHost", 8);

        h_candidateReadIds.resize(totalNumCandidates);
        h_numCandidatesPerAnchor.resize(numTasks);
        h_numCandidatesPerAnchorPrefixSum.resize(numTasks + 1);

        h_numCandidatesPerAnchorPrefixSum[0] = 0;
       
        helpers::call_copy_n_kernel(
            thrust::make_zip_iterator(thrust::make_tuple(
                d_numCandidatesPerAnchorPrefixSum.data() + 1,
                d_numCandidatesPerAnchor.data()
            )),
            numTasks,
            thrust::make_zip_iterator(thrust::make_tuple(
                h_numCandidatesPerAnchorPrefixSum.data() + 1,
                h_numCandidatesPerAnchor.data()
            )),
            streams[0]
        );   

        cudaMemcpyAsync(
            h_candidateReadIds.data(),
            d_candidateReadIds.data(),
            sizeof(read_number) * totalNumCandidates,
            D2H,
            streams[0]
        );

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

                if(!task.mateHasBeenFound){
                    const int newlength = h_outputAnchorLengths[i];

                    std::string newseq(h_outputAnchors.data() + i * outputAnchorPitchInBytes, newlength);
                    std::string newq(h_outputAnchorQualities.data() + i * outputAnchorQualityPitchInBytes, newlength);

                    task.currentAnchorLength = newlength;
                    task.accumExtensionLengths = h_accumExtensionsLengths[i];
                    task.totalDecodedAnchors.emplace_back(std::move(newseq));
                    task.totalAnchorQualityScores.emplace_back(std::move(newq));
                    task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);

                    task.currentQualityScores = task.totalAnchorQualityScores.back(); 
                    
                }else{
                    const int sizeofGap = h_sizeOfGapToMate[i];
                    if(sizeofGap == 0){
                        task.accumExtensionLengths = h_accumExtensionsLengths[i];
                        task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);
                        task.totalDecodedAnchors.emplace_back(task.decodedMateRevC);
                        task.totalAnchorQualityScores.emplace_back(task.mateQualityScoresReversed);
                    }else{
                        const int newlength = h_outputAnchorLengths[i];

                        std::string newseq(h_outputAnchors.data() + i * outputAnchorPitchInBytes, newlength);
                        std::string newq(h_outputAnchorQualities.data() + i * outputAnchorQualityPitchInBytes, newlength);

                        task.accumExtensionLengths = h_accumExtensionsLengths[i];
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

        for(int i = 0; i < numTasks; i++){
            if(tasks[i].isActive(insertSize, insertSizeStddev)){
                h_newPositionsOfActiveTasks[newPosSize] = i;

                if(newPosSize != i){
                    tasks[newPosSize] = std::move(tasks[i]);
                }

                newPosSize++;
            }else{
                addFinishedTask(std::move(tasks[i]));
            }
        }
        h_newPositionsOfActiveTasks.resize(newPosSize);

        tasks.erase(tasks.begin() + newPosSize, tasks.end());
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
            setState(BatchData::State::Finished);
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

        std::size_t bytes = 0;
        void* cubtemp = nullptr;
        cudaError_t cubstatus = cudaSuccess;

        //set new decoded anchors
        d_subjectSequencesDataDecoded.resize(newNumTasks * decodedSequencePitchInBytes);

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

        cubAllocator->DeviceAllocate((void**)&cubtemp, bytes);

        cubstatus = cub::DeviceSelect::Flagged(
            cubtemp,
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
        
        cubAllocator->DeviceFree(cubtemp);

        // set new anchor quality scores
        d_anchorQualityScores.resize(newNumTasks * qualityPitchInBytes);

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

        cubAllocator->DeviceAllocate((void**)&cubtemp, bytes);

        cubstatus = cub::DeviceSelect::Flagged(
            cubtemp,
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
        
        cubAllocator->DeviceFree(cubtemp);

        //set new anchorReadIds, mateReadIds, and anchor lengths

        d_anchorReadIds2.resize(newNumTasks);
        d_mateReadIds2.resize(newNumTasks);
        d_anchorSequencesLength.resize(newNumTasks);

        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            bytes,
            thrust::make_zip_iterator(thrust::make_tuple(
                d_anchorReadIds.data(),
                d_mateReadIds.data(),
                d_outputAnchorLengths.data()
            )),
            d_isActive,
            thrust::make_zip_iterator(thrust::make_tuple(
                d_anchorReadIds2.data(),
                d_mateReadIds2.data(),
                d_anchorSequencesLength.data()
            )),
            thrust::make_discard_iterator(),
            numTasks,
            streams[0]
        );
        assert(cubstatus == cudaSuccess);

        cubAllocator->DeviceAllocate((void**)&cubtemp, bytes);

        cubstatus = cub::DeviceSelect::Flagged(
            cubtemp,
            bytes,
            thrust::make_zip_iterator(thrust::make_tuple(
                d_anchorReadIds.data(),
                d_mateReadIds.data(),
                d_outputAnchorLengths.data()
            )),
            d_isActive,
            thrust::make_zip_iterator(thrust::make_tuple(
                d_anchorReadIds2.data(),
                d_mateReadIds2.data(),
                d_anchorSequencesLength.data()
            )),
            thrust::make_discard_iterator(),
            numTasks,
            streams[0]
        );
        assert(cubstatus == cudaSuccess);
        
        cubAllocator->DeviceFree(cubtemp);

        std::swap(d_anchorReadIds, d_anchorReadIds2);
        std::swap(d_mateReadIds, d_mateReadIds2);


        //set new encoded mate data

        d_inputanchormatedata2.resize(newNumTasks * encodedSequencePitchInInts);

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

        cubAllocator->DeviceAllocate((void**)&cubtemp, bytes);

        cubstatus = cub::DeviceSelect::Flagged(
            cubtemp,
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
        
        cubAllocator->DeviceFree(cubtemp);

        std::swap(d_inputanchormatedata, d_inputanchormatedata2);
        
        cubAllocator->DeviceFree(d_isActive);

        //convert new anchors to 2bit representation

        d_subjectSequencesData.resize(newNumTasks * encodedSequencePitchInInts);

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

        d_numCandidatesPerAnchor.resize(newNumTasks);
        d_numCandidatesPerAnchor2.resize(newNumTasks);
        d_numCandidatesPerAnchorPrefixSum.resize(newNumTasks+1);
        d_numCandidatesPerAnchorPrefixSum2.resize(newNumTasks+1);

        d_anchorIndicesWithRemovedMates.resize(newNumTasks);

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

            std::size_t bytes = 0;
            void* cubtemp = nullptr;
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

            cubAllocator->DeviceAllocate((void**)&cubtemp, bytes, streams[0]);

            cubstatus = cub::DeviceReduce::Sum(
                cubtemp, 
                bytes, 
                d_numUsedReadIdsPerAnchor2.data(), 
                h_numUsedReadIds.data(),
                newNumTasks,
                streams[0]
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceFree(cubtemp);

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

            cubAllocator->DeviceAllocate((void**)&cubtemp, bytes, streams[0]);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                cubtemp, 
                bytes, 
                d_numUsedReadIdsPerAnchor2.data(), 
                d_numUsedReadIdsPerAnchorPrefixSum2.data(), 
                newNumTasks,
                streams[0]
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceFree(cubtemp);

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

            std::size_t bytes = 0;
            void* cubtemp = nullptr;
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

            cubAllocator->DeviceAllocate((void**)&cubtemp, bytes, streams[0]);

            cubstatus = cub::DeviceReduce::Sum(
                cubtemp, 
                bytes, 
                d_numFullyUsedReadIdsPerAnchor2.data(), 
                h_numFullyUsedReadIds.data(),
                newNumTasks,
                streams[0]
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceFree(cubtemp);

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

            cubAllocator->DeviceAllocate((void**)&cubtemp, bytes, streams[0]);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                cubtemp, 
                bytes, 
                d_numFullyUsedReadIdsPerAnchor2.data(), 
                d_numFullyUsedReadIdsPerAnchorPrefixSum2.data(), 
                newNumTasks,
                streams[0]
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceFree(cubtemp);

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


    std::vector<extension::ExtendResult> constructResults() const{
        std::vector<extension::ExtendResult> extendResults;
        extendResults.reserve(finishedTasks.size());

        // int x = 0;

        for(const auto& task : finishedTasks){

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

        void* cubtemp = nullptr;
        cubAllocator->DeviceAllocate((void**)&cubtemp, bytes, stream);

        cubstatus = cub::DeviceScan::InclusiveScan(
            cubtemp, 
            bytes, 
            d_segmentIds, 
            d_segmentIds, 
            cub::Max{},
            numElements,
            stream
        );
        assert(cubstatus == cudaSuccess);

        cubAllocator->DeviceFree(cubtemp);
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
    
    void addFinishedTask(extension::Task&& task){
        //finished tasks must be stored sorted by pairId. Tasks with same pairId are sorted by id
        auto comp = [](const auto& l, const auto& r){
            return std::tie(l.pairId, l.id) < std::tie(r.pairId, r.id);
        };
        auto where = std::upper_bound(finishedTasks.begin(), finishedTasks.end(), task, comp);
        finishedTasks.insert(where, std::move(task));
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
    DeviceBuffer<read_number> d_candidateReadIds{};

    PinnedBuffer<int> h_anchorIndicesOfCandidates{};

    PinnedBuffer<bool> h_isPairedCandidate{};
    DeviceBuffer<bool> d_isPairedCandidate{};

    DeviceBuffer<int> d_anchorIndicesOfCandidates{};

    PinnedBuffer<int> h_segmentIdsOfReadIds{};

    DeviceBuffer<unsigned int> d_anchormatedata{};

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

    
    DeviceBuffer<int> d_candidateSequencesLength{};
    

    DeviceBuffer<unsigned int> d_candidateSequencesData2{};

    DeviceBuffer<unsigned int> d_subjectSequencesData{};
    DeviceBuffer<unsigned int> d_subjectSequencesData2{};
    DeviceBuffer<unsigned int> d_candidateSequencesData{};

    

    // ----- staging buffers for input
    PinnedBuffer<char> h_anchorQualityScores{};
    PinnedBuffer<char> h_subjectSequencesDataDecoded{};
    PinnedBuffer<read_number> h_anchorReadIds{};
    PinnedBuffer<read_number> h_mateReadIds{};
    PinnedBuffer<int> h_anchorSequencesLength{};
    PinnedBuffer<unsigned int> h_inputanchormatedata{};
    // ----- 

    // ----- input data

    DeviceBuffer<unsigned int> d_inputanchormatedata{};
    DeviceBuffer<char> d_subjectSequencesDataDecoded{};
    DeviceBuffer<char> d_anchorQualityScores{};
    DeviceBuffer<int> d_anchorSequencesLength{};
    DeviceBuffer<read_number> d_anchorReadIds{};
    DeviceBuffer<read_number> d_mateReadIds{};

    DeviceBuffer<unsigned int> d_inputanchormatedata2{};
    DeviceBuffer<char> d_subjectSequencesDataDecoded2{};
    DeviceBuffer<char> d_anchorQualityScores2{};
    DeviceBuffer<int> d_anchorSequencesLength2{};
    DeviceBuffer<read_number> d_anchorReadIds2{};
    DeviceBuffer<read_number> d_mateReadIds2{};

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
    // tasḱ_currentQualityScores                    
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
    std::array<CudaStream, 4> streams{};
    std::vector<extension::Task> tasks;
    std::vector<extension::Task> finishedTasks;

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
        batchData.d_numCandidatesPerAnchor.resize(batchData.numTasks);
        batchData.d_numCandidatesPerAnchorPrefixSum.resize(batchData.numTasks + 1);

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


#if 0

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
            case BatchData::State::BeforePrepareNextIteration: return ComputeType::GPU;
            case BatchData::State::Finished: return ComputeType::CPU;
            case BatchData::State::None: return ComputeType::CPU;
            default: return ComputeType::CPU;
        };
    }


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

        void* cubtemp = nullptr;
        cubAllocator->DeviceAllocate((void**)&cubtemp, bytes, stream);

        cubstatus = cub::DeviceScan::InclusiveScan(
            cubtemp, 
            bytes, 
            d_segmentIds, 
            d_segmentIds, 
            cub::Max{},
            numElements,
            stream
        );
        assert(cubstatus == cudaSuccess);

        cubAllocator->DeviceFree(cubtemp);
    }

    void updateUsedCandidateIds(BatchData& batchData) const{
        cudaStream_t stream = batchData.streams[0];

        setGpuSegmentIds(
            batchData.d_anchorIndicesOfCandidates.data(),
            batchData.numTasks,
            batchData.totalNumCandidates,
            batchData.d_numCandidatesPerAnchor.data(),
            batchData.d_numCandidatesPerAnchorPrefixSum.data(),
            stream
        );

        {

            read_number* d_newUsedReadIds = nullptr; 
            int* d_newNumUsedreadIdsPerAnchor = nullptr;
            int* d_newSegmentIdsOfUsedReadIds = nullptr;

            const int maxoutputsize = batchData.totalNumCandidates + *batchData.h_numUsedReadIds;

            cubAllocator->DeviceAllocate((void**)&d_newUsedReadIds, sizeof(read_number) * maxoutputsize);
            cubAllocator->DeviceAllocate((void**)&d_newNumUsedreadIdsPerAnchor, sizeof(int) * batchData.numTasks);
            cubAllocator->DeviceAllocate((void**)&d_newSegmentIdsOfUsedReadIds, sizeof(int) * maxoutputsize);

            ThrustCachingAllocator<char> thrustCachingAllocator1(deviceId, cubAllocator, stream);

            // helpers::lambda_kernel<<<1,1,0,stream>>>(
            //     [
            //         foo = batchData.d_numUsedReadIdsPerAnchorPrefixSum.data()
            //     ] __device__ (){
            //         printf("before: %d %d %d %d\n", foo[0], foo[1], foo[2], foo[3]);
            //     }
            // );

            auto d_newUsedReadIds_end = GpuSegmentedSetOperation::set_union(
                thrustCachingAllocator1,
                batchData.d_candidateReadIds.data(),
                batchData.d_numCandidatesPerAnchor.data(),
                batchData.d_numCandidatesPerAnchorPrefixSum.data(),
                batchData.d_anchorIndicesOfCandidates.data(),
                batchData.totalNumCandidates,
                batchData.numTasks,
                batchData.d_usedReadIds.data(),
                batchData.d_numUsedReadIdsPerAnchor.data(),
                batchData.d_numUsedReadIdsPerAnchorPrefixSum.data(),
                batchData.d_segmentIdsOfUsedReadIds.data(),
                *batchData.h_numUsedReadIds,
                batchData.indicesOfActiveTasks.size(),        
                d_newUsedReadIds,
                d_newNumUsedreadIdsPerAnchor,
                d_newSegmentIdsOfUsedReadIds,
                std::max(batchData.numTasks, batchData.numTasks),
                stream
            );

            int newsize = std::distance(d_newUsedReadIds, d_newUsedReadIds_end);

            // helpers::lambda_kernel<<<1, 1024, 0, stream>>>(
            //     [
            //         numTasks = batchData.numTasks,
            //         d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.data(),
            //         d_numCandidatesPerAnchorPrefixSum = batchData.d_numCandidatesPerAnchorPrefixSum.data(),
            //         d_candidateReadIds = batchData.d_candidateReadIds.data(),
            //         d_numCandidatesPerAnchorsize = batchData.d_numCandidatesPerAnchor.size(),
            //         d_numCandidatesPerAnchorPrefixSumsize = batchData.d_numCandidatesPerAnchorPrefixSum.size(),
            //         d_candidateReadIdssize = batchData.d_candidateReadIds.size()
            //     ] __device__ (){

            //         assert(d_numCandidatesPerAnchorsize == numTasks);
            //         assert(d_numCandidatesPerAnchorPrefixSumsize == numTasks + 1);

            //         assert(d_numCandidatesPerAnchorPrefixSum[0] == 0);

            //         int ps = 0;

            //         for(int i = 0; i < numTasks; i++){
            //             assert(d_numCandidatesPerAnchorPrefixSum[i] == ps);
            //             ps += d_numCandidatesPerAnchor[i];
            //         }

            //         for(int i = 0; i < numTasks; i++){
            //             assert(d_numCandidatesPerAnchorPrefixSum[i] + d_numCandidatesPerAnchor[i] <= d_candidateReadIdssize);
            //         }

            //         for(int i = threadIdx.x; i< d_candidateReadIdssize; i += blockDim.x){
            //             assert(d_candidateReadIds[i] < 30085710);
            //         }
            //     }
            // ); CUERR;

            // cudaStreamSynchronize(stream); CUERR;


            // helpers::lambda_kernel<<<1, 1024, 0, stream>>>(
            //     [
            //         numTasks = batchData.numTasks,
            //         d_numUsedReadIdsPerAnchor = batchData.d_numUsedReadIdsPerAnchor.data(),
            //         d_numUsedReadIdsPerAnchorPrefixSum = batchData.d_numUsedReadIdsPerAnchorPrefixSum.data(),
            //         d_usedReadIds = batchData.d_usedReadIds.data(),
            //         d_numUsedReadIdsPerAnchorsize = batchData.d_numUsedReadIdsPerAnchor.size(),
            //         d_numUsedReadIdsPerAnchorPrefixSumsize = batchData.d_numUsedReadIdsPerAnchorPrefixSum.size(),
            //         d_usedReadIdssize = batchData.d_usedReadIds.size()
            //     ] __device__ (){

            //         assert(d_numUsedReadIdsPerAnchorsize == numTasks);
            //         assert(d_numUsedReadIdsPerAnchorPrefixSumsize == numTasks);

            //         assert(d_numUsedReadIdsPerAnchorPrefixSum[0] == 0);

            //         int ps = 0;

            //         for(int i = 0; i < numTasks; i++){
            //             assert(d_numUsedReadIdsPerAnchorPrefixSum[i] == ps);
            //             ps += d_numUsedReadIdsPerAnchor[i];
            //         }

            //         for(int i = 0; i < numTasks; i++){
            //             assert(d_numUsedReadIdsPerAnchorPrefixSum[i] + d_numUsedReadIdsPerAnchor[i] <= d_usedReadIdssize);
            //         }

            //         for(int i = threadIdx.x; i< d_usedReadIdssize; i += blockDim.x){
            //             assert(d_usedReadIds[i] < 30085710);
            //         }
            //     }
            // );
            // auto status = cudaStreamSynchronize(stream);
            // if(cudaSuccess != status){
            //     std::cerr  << "batchData.numTasks = " << batchData.numTasks
            //     << " batchData.d_numUsedReadIdsPerAnchor.size() = " << batchData.d_numUsedReadIdsPerAnchor.size()
            //     << " batchData.d_numUsedReadIdsPerAnchorPrefixSum.size() = " << batchData.d_numUsedReadIdsPerAnchorPrefixSum.size()
            //     << " batchData.d_usedReadIds.size() = " << batchData.d_usedReadIds.size() 
            //     << " *batchData.h_numUsedReadIds = " << *batchData.h_numUsedReadIds << "\n";
            //     CUERR;
            // }





            batchData.d_usedReadIds.resize(newsize);
            batchData.d_segmentIdsOfUsedReadIds.resize(newsize);

            cudaMemcpyAsync(
                batchData.d_usedReadIds.data(),
                d_newUsedReadIds,
                sizeof(read_number) * newsize,
                D2D,
                stream
            );
            cubAllocator->DeviceFree(d_newUsedReadIds);

            cudaMemcpyAsync(
                batchData.d_segmentIdsOfUsedReadIds.data(),
                d_newSegmentIdsOfUsedReadIds,
                sizeof(int) * newsize,
                D2D,
                stream
            );
            cubAllocator->DeviceFree(d_newSegmentIdsOfUsedReadIds);

            cudaMemcpyAsync(
                batchData.d_numUsedReadIdsPerAnchor.data(),
                d_newNumUsedreadIdsPerAnchor,
                sizeof(int) * batchData.numTasks,
                D2D,
                stream
            );
            cubAllocator->DeviceFree(d_newNumUsedreadIdsPerAnchor);

            std::size_t bytes = 0;

            cudaError_t cubstatus = cub::DeviceScan::ExclusiveSum(
                nullptr, 
                bytes, 
                batchData.d_numUsedReadIdsPerAnchor.data(), 
                batchData.d_numUsedReadIdsPerAnchorPrefixSum.data(), 
                batchData.numTasks,
                stream
            );
            assert(cubstatus == cudaSuccess);

            void* cubtemp = nullptr;
            cubAllocator->DeviceAllocate((void**)&cubtemp, bytes, stream);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                cubtemp, 
                bytes, 
                batchData.d_numUsedReadIdsPerAnchor.data(), 
                batchData.d_numUsedReadIdsPerAnchorPrefixSum.data(), 
                batchData.numTasks,
                stream
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceFree(cubtemp);

            *batchData.h_numUsedReadIds = newsize;

            // helpers::lambda_kernel<<<1, 1024, 0, stream>>>(
            //     [
            //         numTasks = batchData.numTasks,
            //         d_numUsedReadIdsPerAnchor = batchData.d_numUsedReadIdsPerAnchor.data(),
            //         d_numUsedReadIdsPerAnchorPrefixSum = batchData.d_numUsedReadIdsPerAnchorPrefixSum.data(),
            //         d_usedReadIds = batchData.d_usedReadIds.data(),
            //         d_numUsedReadIdsPerAnchorsize = batchData.d_numUsedReadIdsPerAnchor.size(),
            //         d_numUsedReadIdsPerAnchorPrefixSumsize = batchData.d_numUsedReadIdsPerAnchorPrefixSum.size(),
            //         d_usedReadIdssize = batchData.d_usedReadIds.size()
            //     ] __device__ (){

            //         assert(d_numUsedReadIdsPerAnchorsize == numTasks);
            //         assert(d_numUsedReadIdsPerAnchorPrefixSumsize == numTasks);

            //         assert(d_numUsedReadIdsPerAnchorPrefixSum[0] == 0);

            //         int ps = 0;

            //         for(int i = 0; i < numTasks; i++){
            //             assert(d_numUsedReadIdsPerAnchorPrefixSum[i] == ps);
            //             ps += d_numUsedReadIdsPerAnchor[i];
            //         }

            //         for(int i = 0; i < numTasks; i++){
            //             assert(d_numUsedReadIdsPerAnchorPrefixSum[i] + d_numUsedReadIdsPerAnchor[i] <= d_usedReadIdssize);
            //         }

            //         for(int i = threadIdx.x; i< d_usedReadIdssize; i += blockDim.x){
            //             assert(d_usedReadIds[i] < 30085710);
            //         }
            //     }
            // ); CUERR;

            // cudaStreamSynchronize(stream); CUERR;

        }

        {
            std::size_t bytes = 0;
            void* cubtemp = nullptr;
            cudaError_t cubstatus = cudaSuccess;

            read_number* d_currentFullyUsedReadIds = nullptr; 
            int* d_currentNumFullyUsedreadIdsPerAnchor = nullptr;
            int* d_currentNumFullyUsedreadIdsPerAnchorPS = nullptr;
            int* d_currentSegmentIdsOfFullyUsedReadIds = nullptr;
            int* h_currentNumFullyUsed = batchData.h_firstTasksOfPairsToCheck.data();
            

            cubAllocator->DeviceAllocate((void**)&d_currentFullyUsedReadIds, sizeof(read_number) * batchData.totalNumCandidates);
            cubAllocator->DeviceAllocate((void**)&d_currentNumFullyUsedreadIdsPerAnchor, sizeof(int) * batchData.numTasks);
            cubAllocator->DeviceAllocate((void**)&d_currentNumFullyUsedreadIdsPerAnchorPS, sizeof(int) * batchData.numTasks);
            cubAllocator->DeviceAllocate((void**)&d_currentSegmentIdsOfFullyUsedReadIds, sizeof(int) * batchData.totalNumCandidates);

            auto candidatesAndSegmentIdsIn = thrust::make_zip_iterator(
                thrust::make_tuple(
                    batchData.d_candidateReadIds.data(),
                    batchData.d_anchorIndicesOfCandidates.data()
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
                batchData.d_isFullyUsedCandidate.data(),
                candidatesAndSegmentIdsOut,
                h_currentNumFullyUsed,
                batchData.totalNumCandidates,
                stream
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceAllocate((void**)&cubtemp, bytes);

            cubstatus = cub::DeviceSelect::Flagged(
                cubtemp,
                bytes,
                candidatesAndSegmentIdsIn,
                batchData.d_isFullyUsedCandidate.data(),
                candidatesAndSegmentIdsOut,
                h_currentNumFullyUsed,
                batchData.totalNumCandidates,
                stream
            );
            assert(cubstatus == cudaSuccess);
            cudaEventRecord(batchData.events[0], stream); CUERR;

            cubAllocator->DeviceFree(cubtemp);

            //compute current number of fully used candidates per segment
            cubstatus = cub::DeviceSegmentedReduce::Sum(
                nullptr,
                bytes,
                batchData.d_isFullyUsedCandidate.data(),
                d_currentNumFullyUsedreadIdsPerAnchor,
                batchData.numTasks,
                batchData.d_numCandidatesPerAnchorPrefixSum.data(),
                batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1,
                stream
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceAllocate((void**)&cubtemp, bytes);

            cubstatus = cub::DeviceSegmentedReduce::Sum(
                cubtemp,
                bytes,
                batchData.d_isFullyUsedCandidate.data(),
                d_currentNumFullyUsedreadIdsPerAnchor,
                batchData.numTasks,
                batchData.d_numCandidatesPerAnchorPrefixSum.data(),
                batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1,
                stream
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceFree(cubtemp);

            //compute prefix sum of current number of fully used candidates per segment

            cubstatus = cub::DeviceScan::ExclusiveSum(
                nullptr, 
                bytes, 
                d_currentNumFullyUsedreadIdsPerAnchor, 
                d_currentNumFullyUsedreadIdsPerAnchorPS, 
                batchData.numTasks,
                stream
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceAllocate((void**)&cubtemp, bytes, stream);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                cubtemp, 
                bytes, 
                d_currentNumFullyUsedreadIdsPerAnchor, 
                d_currentNumFullyUsedreadIdsPerAnchorPS, 
                batchData.numTasks,
                stream
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceFree(cubtemp);


            cudaEventSynchronize(batchData.events[0]); CUERR;



            read_number* d_newFullyUsedReadIds = nullptr; 
            int* d_newNumFullyUsedreadIdsPerAnchor = nullptr;
            int* d_newSegmentIdsOfFullyUsedReadIds = nullptr;

            const int maxoutputsize = batchData.totalNumCandidates + *batchData.h_numFullyUsedReadIds;

            cubAllocator->DeviceAllocate((void**)&d_newFullyUsedReadIds, sizeof(read_number) * maxoutputsize);
            cubAllocator->DeviceAllocate((void**)&d_newNumFullyUsedreadIdsPerAnchor, sizeof(int) * batchData.numTasks);
            cubAllocator->DeviceAllocate((void**)&d_newSegmentIdsOfFullyUsedReadIds, sizeof(int) * maxoutputsize);

            ThrustCachingAllocator<char> thrustCachingAllocator1(deviceId, cubAllocator, stream);

            auto d_newFullyUsedReadIds_end = GpuSegmentedSetOperation::set_union(
                thrustCachingAllocator1,
                d_currentFullyUsedReadIds,
                d_currentNumFullyUsedreadIdsPerAnchor,
                d_currentNumFullyUsedreadIdsPerAnchorPS,
                d_currentSegmentIdsOfFullyUsedReadIds,
                *h_currentNumFullyUsed,
                batchData.numTasks,
                batchData.d_fullyUsedReadIds.data(),
                batchData.d_numFullyUsedReadIdsPerAnchor.data(),
                batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum.data(),
                batchData.d_segmentIdsOfFullyUsedReadIds.data(),
                *batchData.h_numFullyUsedReadIds,
                batchData.indicesOfActiveTasks.size(),        
                d_newFullyUsedReadIds,
                d_newNumFullyUsedreadIdsPerAnchor,
                d_newSegmentIdsOfFullyUsedReadIds,
                std::max(batchData.numTasks, batchData.numTasks),
                stream
            );

            cubAllocator->DeviceFree(d_currentFullyUsedReadIds);
            cubAllocator->DeviceFree(d_currentNumFullyUsedreadIdsPerAnchor);
            cubAllocator->DeviceFree(d_currentNumFullyUsedreadIdsPerAnchorPS);
            cubAllocator->DeviceFree(d_currentSegmentIdsOfFullyUsedReadIds);

            int newsize = std::distance(d_newFullyUsedReadIds, d_newFullyUsedReadIds_end);

            // helpers::lambda_kernel<<<1, 1024, 0, stream>>>(
            //     [
            //         numTasks = batchData.numTasks,
            //         d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.data(),
            //         d_numCandidatesPerAnchorPrefixSum = batchData.d_numCandidatesPerAnchorPrefixSum.data(),
            //         d_candidateReadIds = batchData.d_candidateReadIds.data(),
            //         d_numCandidatesPerAnchorsize = batchData.d_numCandidatesPerAnchor.size(),
            //         d_numCandidatesPerAnchorPrefixSumsize = batchData.d_numCandidatesPerAnchorPrefixSum.size(),
            //         d_candidateReadIdssize = batchData.d_candidateReadIds.size()
            //     ] __device__ (){

            //         assert(d_numCandidatesPerAnchorsize == numTasks);
            //         assert(d_numCandidatesPerAnchorPrefixSumsize == numTasks + 1);

            //         assert(d_numCandidatesPerAnchorPrefixSum[0] == 0);

            //         int ps = 0;

            //         for(int i = 0; i < numTasks; i++){
            //             assert(d_numCandidatesPerAnchorPrefixSum[i] == ps);
            //             ps += d_numCandidatesPerAnchor[i];
            //         }

            //         for(int i = 0; i < numTasks; i++){
            //             assert(d_numCandidatesPerAnchorPrefixSum[i] + d_numCandidatesPerAnchor[i] <= d_candidateReadIdssize);
            //         }

            //         for(int i = threadIdx.x; i< d_candidateReadIdssize; i += blockDim.x){
            //             assert(d_candidateReadIds[i] < 30085710);
            //         }
            //     }
            // ); CUERR;

            // cudaStreamSynchronize(stream); CUERR;


            // helpers::lambda_kernel<<<1, 1024, 0, stream>>>(
            //     [
            //         numTasks = batchData.numTasks,
            //         d_numFullyUsedReadIdsPerAnchor = batchData.d_numFullyUsedReadIdsPerAnchor.data(),
            //         d_numFullyUsedReadIdsPerAnchorPrefixSum = batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum.data(),
            //         d_fullyUsedReadIds = batchData.d_fullyUsedReadIds.data(),
            //         d_numFullyUsedReadIdsPerAnchorsize = batchData.d_numFullyUsedReadIdsPerAnchor.size(),
            //         d_numFullyUsedReadIdsPerAnchorPrefixSumsize = batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum.size(),
            //         d_fullyUsedReadIdssize = batchData.d_fullyUsedReadIds.size()
            //     ] __device__ (){

            //         assert(d_numFullyUsedReadIdsPerAnchorsize == numTasks);
            //         assert(d_numFullyUsedReadIdsPerAnchorPrefixSumsize == numTasks);

            //         assert(d_numFullyUsedReadIdsPerAnchorPrefixSum[0] == 0);

            //         int ps = 0;

            //         for(int i = 0; i < numTasks; i++){
            //             assert(d_numFullyUsedReadIdsPerAnchorPrefixSum[i] == ps);
            //             ps += d_numFullyUsedReadIdsPerAnchor[i];
            //         }

            //         for(int i = 0; i < numTasks; i++){
            //             assert(d_numFullyUsedReadIdsPerAnchorPrefixSum[i] + d_numFullyUsedReadIdsPerAnchor[i] <= d_fullyUsedReadIdssize);
            //         }

            //         for(int i = threadIdx.x; i< d_fullyUsedReadIdssize; i += blockDim.x){
            //             assert(d_fullyUsedReadIds[i] < 30085710);
            //         }
            //     }
            // );
            // auto status = cudaStreamSynchronize(stream);
            // if(cudaSuccess != status){
            //     std::cerr  << "batchData.numTasks = " << batchData.numTasks
            //     << " batchData.d_numFullyUsedReadIdsPerAnchor.size() = " << batchData.d_numFullyUsedReadIdsPerAnchor.size()
            //     << " batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum.size() = " << batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum.size()
            //     << " batchData.d_fullyUsedReadIds.size() = " << batchData.d_fullyUsedReadIds.size() 
            //     << " *batchData.h_numFullyUsedReadIds = " << *batchData.h_numFullyUsedReadIds << "\n";
            //     CUERR;
            // }





            batchData.d_fullyUsedReadIds.resize(newsize);
            batchData.d_segmentIdsOfFullyUsedReadIds.resize(newsize);

            cudaMemcpyAsync(
                batchData.d_fullyUsedReadIds.data(),
                d_newFullyUsedReadIds,
                sizeof(read_number) * newsize,
                D2D,
                stream
            );
            cubAllocator->DeviceFree(d_newFullyUsedReadIds);

            cudaMemcpyAsync(
                batchData.d_segmentIdsOfFullyUsedReadIds.data(),
                d_newSegmentIdsOfFullyUsedReadIds,
                sizeof(int) * newsize,
                D2D,
                stream
            );
            cubAllocator->DeviceFree(d_newSegmentIdsOfFullyUsedReadIds);

            cudaMemcpyAsync(
                batchData.d_numFullyUsedReadIdsPerAnchor.data(),
                d_newNumFullyUsedreadIdsPerAnchor,
                sizeof(int) * batchData.numTasks,
                D2D,
                stream
            );
            cubAllocator->DeviceFree(d_newNumFullyUsedreadIdsPerAnchor);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                nullptr, 
                bytes, 
                batchData.d_numFullyUsedReadIdsPerAnchor.data(), 
                batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum.data(), 
                batchData.numTasks,
                stream
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceAllocate((void**)&cubtemp, bytes, stream);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                cubtemp, 
                bytes, 
                batchData.d_numFullyUsedReadIdsPerAnchor.data(), 
                batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum.data(), 
                batchData.numTasks,
                stream
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceFree(cubtemp);

            *batchData.h_numFullyUsedReadIds = newsize;

            // helpers::lambda_kernel<<<1, 1024, 0, stream>>>(
            //     [
            //         numTasks = batchData.numTasks,
            //         d_numFullyUsedReadIdsPerAnchor = batchData.d_numFullyUsedReadIdsPerAnchor.data(),
            //         d_numFullyUsedReadIdsPerAnchorPrefixSum = batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum.data(),
            //         d_fullyUsedReadIds = batchData.d_fullyUsedReadIds.data(),
            //         d_numFullyUsedReadIdsPerAnchorsize = batchData.d_numFullyUsedReadIdsPerAnchor.size(),
            //         d_numFullyUsedReadIdsPerAnchorPrefixSumsize = batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum.size(),
            //         d_fullyUsedReadIdssize = batchData.d_fullyUsedReadIds.size()
            //     ] __device__ (){

            //         assert(d_numFullyUsedReadIdsPerAnchorsize == numTasks);
            //         assert(d_numFullyUsedReadIdsPerAnchorPrefixSumsize == numTasks);

            //         assert(d_numFullyUsedReadIdsPerAnchorPrefixSum[0] == 0);

            //         int ps = 0;

            //         for(int i = 0; i < numTasks; i++){
            //             assert(d_numFullyUsedReadIdsPerAnchorPrefixSum[i] == ps);
            //             ps += d_numFullyUsedReadIdsPerAnchor[i];
            //         }

            //         for(int i = 0; i < numTasks; i++){
            //             assert(d_numFullyUsedReadIdsPerAnchorPrefixSum[i] + d_numFullyUsedReadIdsPerAnchor[i] <= d_fullyUsedReadIdssize);
            //         }

            //         for(int i = threadIdx.x; i< d_fullyUsedReadIdssize; i += blockDim.x){
            //             assert(d_fullyUsedReadIds[i] < 30085710);
            //         }
            //     }
            // );
            // auto status = cudaStreamSynchronize(stream);
            // if(cudaSuccess != status){
            //     std::cerr  << "batchData.numTasks = " << batchData.numTasks
            //     << " batchData.d_numFullyUsedReadIdsPerAnchor.size() = " << batchData.d_numFullyUsedReadIdsPerAnchor.size()
            //     << " batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum.size() = " << batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum.size()
            //     << " batchData.d_fullyUsedReadIds.size() = " << batchData.d_fullyUsedReadIds.size() 
            //     << " *batchData.h_numFullyUsedReadIds = " << *batchData.h_numFullyUsedReadIds << "\n";
            //     CUERR;
            // }

        }
    }



 
    void performNextStep(BatchData& batchData) const{

        const auto name = BatchData::to_string(batchData.state);

        // {
        //     std::size_t free, total;
        //     cudaMemGetInfo(&free, &total);
        //     std::cerr << "before " << name << " " << free << "\n";
        // }

        switch(batchData.state){
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
            case BatchData::State::BeforePrepareNextIteration: prepareNextIteration(batchData); break;
            case BatchData::State::Finished: break;
            case BatchData::State::None: break;
            default: break;
        };

        // {
        //     std::size_t free, total;
        //     cudaMemGetInfo(&free, &total);
        //     std::cerr << "after " << name << " " << free << "\n";
        // }

    }


    void getCandidateReadIds(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforeHash);

        nvtx::push_range("getCandidateReadIds", 0);

        batchData.getCandidateReadIds();

        //gpuReadHasher.getCandidateReadIds(batchData, batchData.streams[0]);
        batchData.setState(BatchData::State::BeforeRemoveIds);

        nvtx::pop_range();
    }

    void removeUsedIdsAndMateIds(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforeRemoveIds);
        
        nvtx::push_range("removeUsedIdsAndMateIds", 1);

        batchData.removeUsedIdsAndMateIds();

        // #ifdef DO_REMOVE_USED_IDS_AND_MATE_IDS_ON_GPU

        // removeUsedIdsAndMateIds(batchData, batchData.streams[0], batchData.streams[0]);  

        // #else 

        // removeUsedIdsAndMateIdsCPU(batchData, batchData.streams[0], batchData.streams[0]);    
        
        // #endif

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

        // {
        //     std::size_t free, total;
        //     cudaMemGetInfo(&free, &total);
        //     std::cerr << "before computeMSAs " << free << "\n";
        // }

        nvtx::push_range("computeMSAs", 6);

        //Sets batchData.totalNumCandidates to the sum of number of candidates for all tasks. (msa refinement can remove candidates)
        computeMSAs(batchData, batchData.streams[0], batchData.streams[0]);

        nvtx::pop_range();

        // {
        //     std::size_t free, total;
        //     cudaMemGetInfo(&free, &total);
        //     std::cerr << "after computeMSAs " << free << "\n";
        // }

        batchData.setState(BatchData::State::BeforeExtend);
    }

    void computeExtendedSequencesFromMSAs(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforeExtend);

        // {
        //     std::size_t free, total;
        //     cudaMemGetInfo(&free, &total);
        //     std::cerr << "before computeExtendedSequencesFromMSAs " << free << "\n";
        // }

        nvtx::push_range("computeExtendedSequences", 7);

        computeExtendedSequencesFromMSAs(batchData, batchData.streams[0]);

        nvtx::pop_range();

        // {
        //     std::size_t free, total;
        //     cudaMemGetInfo(&free, &total);
        //     std::cerr << "after computeExtendedSequencesFromMSAs " << free << "\n";
        // }

        // {
        //     std::size_t free, total;
        //     cudaMemGetInfo(&free, &total);
        //     std::cerr << "before updateUsedCandidateIds " << free << "\n";
        // }

        nvtx::push_range("updateUsedCandidateIds", 2);

        updateUsedCandidateIds(batchData);

        nvtx::pop_range();

        // {
        //     std::size_t free, total;
        //     cudaMemGetInfo(&free, &total);
        //     std::cerr << "after updateUsedCandidateIds " << free << "\n";
        // }

        batchData.setState(BatchData::State::BeforeCopyToHost);
    }

    void copyBuffersToHost(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforeCopyToHost);

        nvtx::push_range("copyBuffersToHost", 8);

        copyBuffersToHost(batchData, batchData.streams[0], batchData.streams[0]);

        cudaStreamSynchronize(batchData.streams[0]); CUERR;
        cudaStreamSynchronize(batchData.streams[1]); CUERR;

        nvtx::pop_range();

        batchData.setState(BatchData::State::BeforeUnpack);
    }

    void unpackResults(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforeUnpack);

        // {
        //     std::size_t free, total;
        //     cudaMemGetInfo(&free, &total);
        //     std::cerr << "before unpackResults " << free << "\n";
        // }

        nvtx::push_range("unpackResults", 9);

        unpackResultsIntoTasks(batchData);

        nvtx::pop_range();

        batchData.setState(BatchData::State::BeforePrepareNextIteration);

        // {
        //     std::size_t free, total;
        //     cudaMemGetInfo(&free, &total);
        //     std::cerr << "after unpackResults " << free << "\n";
        // }
    }

    void prepareNextIteration(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforePrepareNextIteration);

        //update list of active task indices
        batchData.h_newPositionsOfActiveTasks.resize(batchData.numTasks);
        int newPosSize = 0;

        for(int i = 0; i < batchData.numTasks; i++){
            if(batchData.tasks[batchData.indicesOfActiveTasks[i]].isActive(insertSize, insertSizeStddev)){
                batchData.h_newPositionsOfActiveTasks[newPosSize++] = i;
            }
        }
        batchData.h_newPositionsOfActiveTasks.resize(newPosSize);

       
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

        if(!batchData.isEmpty()){

            batchData.d_newPositionsOfActiveTasks.resize(batchData.h_newPositionsOfActiveTasks.size());

            cudaMemcpyAsync(
                batchData.d_newPositionsOfActiveTasks.data(),
                batchData.h_newPositionsOfActiveTasks.data(),
                sizeof(int) * batchData.h_newPositionsOfActiveTasks.size(),
                H2D,
                batchData.streams[0]
            ); CUERR;

            // {
            //     std::size_t free, total;
            //     cudaMemGetInfo(&free, &total);
            //     std::cerr << "before updateBuffersForNextIteration " << free << "\n";
            // }

            nvtx::push_range("updateBuffersForNextIteration", 6);

            updateBuffersForNextIteration(batchData);

            nvtx::pop_range();

            // {
            //     std::size_t free, total;
            //     cudaMemGetInfo(&free, &total);
            //     std::cerr << "after updateBuffersForNextIteration " << free << "\n";
            // }

        }

        batchData.numTasks = batchData.indicesOfActiveTasks.size();

        if(!batchData.isEmpty()){
            batchData.setState(BatchData::State::BeforeHash);
        }else{
            batchData.setState(BatchData::State::Finished);
        }
        
    }

    void updateBuffersForNextIteration(BatchData& batchData) const{
        nvtx::push_range("removeUsedIdsOfFinishedTasks", 6);

        removeUsedIdsOfFinishedTasks(batchData);

        nvtx::pop_range();

        //compute selection flags of remaining tasks
        const int newNumTasks = batchData.d_newPositionsOfActiveTasks.size();
        bool* d_isActive = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_isActive, batchData.numTasks, batchData.streams[0]);
        cudaMemsetAsync(d_isActive, 0, batchData.numTasks, batchData.streams[0]); CUERR;

        helpers::lambda_kernel<<<SDIV(newNumTasks, 128), 128, 0, batchData.streams[0]>>>(
            [
                d_isActive,
                d_newPositionsOfActiveTasks = batchData.d_newPositionsOfActiveTasks.data(),
                newNumTasks
            ] __device__ (){
                const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                const int stride = blockDim.x * gridDim.x;

                for(int i = tid; i < newNumTasks; i += stride){
                    d_isActive[d_newPositionsOfActiveTasks[i]] = true;
                }
            }
        ); CUERR;

        std::size_t bytes = 0;
        void* cubtemp = nullptr;
        cudaError_t cubstatus = cudaSuccess;

        //set new decoded anchors
        batchData.d_subjectSequencesDataDecoded.resize(newNumTasks * batchData.decodedSequencePitchInBytes);

        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            bytes,
            batchData.d_outputAnchors.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_isActive, batchData.outputAnchorPitchInBytes)
            ),
            batchData.d_subjectSequencesDataDecoded.data(),
            thrust::make_discard_iterator(),
            batchData.numTasks * batchData.outputAnchorPitchInBytes,
            batchData.streams[0]
        );
        assert(cubstatus == cudaSuccess);

        cubAllocator->DeviceAllocate((void**)&cubtemp, bytes);

        cubstatus = cub::DeviceSelect::Flagged(
            cubtemp,
            bytes,
            batchData.d_outputAnchors.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_isActive, batchData.outputAnchorPitchInBytes)
            ),
            batchData.d_subjectSequencesDataDecoded.data(),
            thrust::make_discard_iterator(),
            batchData.numTasks * batchData.outputAnchorPitchInBytes,
            batchData.streams[0]
        );
        assert(cubstatus == cudaSuccess);
        
        cubAllocator->DeviceFree(cubtemp);

        // set new anchor quality scores
        batchData.d_anchorQualityScores.resize(newNumTasks * batchData.qualityPitchInBytes);

        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            bytes,
            batchData.d_outputAnchorQualities.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_isActive, batchData.outputAnchorQualityPitchInBytes)
            ),
            batchData.d_anchorQualityScores.data(),
            thrust::make_discard_iterator(),
            batchData.numTasks * batchData.outputAnchorQualityPitchInBytes,
            batchData.streams[0]
        );
        assert(cubstatus == cudaSuccess);

        cubAllocator->DeviceAllocate((void**)&cubtemp, bytes);

        cubstatus = cub::DeviceSelect::Flagged(
            cubtemp,
            bytes,
            batchData.d_outputAnchorQualities.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_isActive, batchData.outputAnchorQualityPitchInBytes)
            ),
            batchData.d_anchorQualityScores.data(),
            thrust::make_discard_iterator(),
            batchData.numTasks * batchData.outputAnchorQualityPitchInBytes,
            batchData.streams[0]
        );
        assert(cubstatus == cudaSuccess);
        
        cubAllocator->DeviceFree(cubtemp);

        //set new anchorReadIds, mateReadIds, and anchor lengths

        batchData.d_anchorReadIds2.resize(newNumTasks);
        batchData.d_mateReadIds2.resize(newNumTasks);
        batchData.d_anchorSequencesLength.resize(newNumTasks);

        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            bytes,
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.d_anchorReadIds.data(),
                batchData.d_mateReadIds.data(),
                batchData.d_outputAnchorLengths.data()
            )),
            d_isActive,
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.d_anchorReadIds2.data(),
                batchData.d_mateReadIds2.data(),
                batchData.d_anchorSequencesLength.data()
            )),
            thrust::make_discard_iterator(),
            batchData.numTasks,
            batchData.streams[0]
        );
        assert(cubstatus == cudaSuccess);

        cubAllocator->DeviceAllocate((void**)&cubtemp, bytes);

        cubstatus = cub::DeviceSelect::Flagged(
            cubtemp,
            bytes,
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.d_anchorReadIds.data(),
                batchData.d_mateReadIds.data(),
                batchData.d_outputAnchorLengths.data()
            )),
            d_isActive,
            thrust::make_zip_iterator(thrust::make_tuple(
                batchData.d_anchorReadIds2.data(),
                batchData.d_mateReadIds2.data(),
                batchData.d_anchorSequencesLength.data()
            )),
            thrust::make_discard_iterator(),
            batchData.numTasks,
            batchData.streams[0]
        );
        assert(cubstatus == cudaSuccess);
        
        cubAllocator->DeviceFree(cubtemp);

        std::swap(batchData.d_anchorReadIds, batchData.d_anchorReadIds2);
        std::swap(batchData.d_mateReadIds, batchData.d_mateReadIds2);


        //set new encoded mate data

        batchData.d_inputanchormatedata2.resize(newNumTasks * batchData.encodedSequencePitchInInts);

        cubstatus = cub::DeviceSelect::Flagged(
            nullptr,
            bytes,
            batchData.d_inputanchormatedata.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_isActive, batchData.encodedSequencePitchInInts)
            ),
            batchData.d_inputanchormatedata2.data(),
            thrust::make_discard_iterator(),
            batchData.numTasks * batchData.encodedSequencePitchInInts,
            batchData.streams[0]
        );
        assert(cubstatus == cudaSuccess);

        cubAllocator->DeviceAllocate((void**)&cubtemp, bytes);

        cubstatus = cub::DeviceSelect::Flagged(
            cubtemp,
            bytes,
            batchData.d_inputanchormatedata.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_isActive, batchData.encodedSequencePitchInInts)
            ),
            batchData.d_inputanchormatedata2.data(),
            thrust::make_discard_iterator(),
            batchData.numTasks * batchData.encodedSequencePitchInInts,
            batchData.streams[0]
        );
        assert(cubstatus == cudaSuccess);
        
        cubAllocator->DeviceFree(cubtemp);

        std::swap(batchData.d_inputanchormatedata, batchData.d_inputanchormatedata2);
        
        cubAllocator->DeviceFree(d_isActive);

        //convert new anchors to 2bit representation

        batchData.d_subjectSequencesData.resize(newNumTasks * batchData.encodedSequencePitchInInts);

        readextendergpukernels::encodeSequencesTo2BitKernel<8>
        <<<SDIV(newNumTasks, (128 / 8)), 128, 0, batchData.streams[0]>>>(
            batchData.d_subjectSequencesData.data(),
            batchData.d_subjectSequencesDataDecoded.data(),
            batchData.d_anchorSequencesLength.data(),
            batchData.decodedSequencePitchInBytes,
            batchData.encodedSequencePitchInInts,
            newNumTasks
        ); CUERR;


        //shrink remaining buffers
        batchData.d_anchormatedata.resize(newNumTasks * batchData.encodedSequencePitchInInts);

        batchData.d_numCandidatesPerAnchor.resize(newNumTasks);
        batchData.d_numCandidatesPerAnchor2.resize(newNumTasks);
        batchData.d_numCandidatesPerAnchorPrefixSum.resize(newNumTasks+1);
        batchData.d_numCandidatesPerAnchorPrefixSum2.resize(newNumTasks+1);

        batchData.d_anchorIndicesWithRemovedMates.resize(newNumTasks);

    }


    void process(BatchData& batchData) const{
        assert(batchData.state == BatchData::State::BeforeHash);

        const std::size_t initialNumTasks = tasks.size();

        while(batchData.state != BatchData::State::Finished){
            performNextStep(batchData);
        }

        assert(tasks.size() == 0);
        assert(finishedTasks.size() == initialNumTasks);
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

        handleEarlyExitOfTasks4(batchData.tasks, batchData.indicesOfActiveTasks);

        for(int i = 0; i < numActiveTasks; i++){
            auto& task = batchData.tasks[batchData.indicesOfActiveTasks[i]];

            task.iteration++;
        }
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

                // std::cerr << "batchData.d_numCandidatesPerAnchor.size(): " << batchData.d_numCandidatesPerAnchor.size() << "\n";
                // std::cerr << "batchData.d_numCandidatesPerAnchorPrefixSum.size(): " << batchData.d_numCandidatesPerAnchorPrefixSum.size() << "\n";
                // std::cerr << "batchData.d_candidateReadIds.size(): " << batchData.d_candidateReadIds.size() << "\n";
                // std::cerr << "batchData.d_numUsedReadIdsPerAnchor.size(): " << batchData.d_numUsedReadIdsPerAnchor.size() << "\n";
                // std::cerr << "batchData.d_numUsedReadIdsPerAnchorPrefixSum.size(): " << batchData.d_numUsedReadIdsPerAnchorPrefixSum.size() << "\n";
                // std::cerr << "batchData.d_usedReadIds.size(): " << batchData.d_usedReadIds.size() << "\n";
                // std::cerr << "batchData.d_isPairedCandidate.size(): " << batchData.d_isPairedCandidate.size() << "\n";

                // std::cerr << "batchData.d_numCandidatesPerAnchor.data(): " << batchData.d_numCandidatesPerAnchor.data() << "\n";
                // std::cerr << "batchData.d_numCandidatesPerAnchorPrefixSum.data(): " << batchData.d_numCandidatesPerAnchorPrefixSum.data() << "\n";
                // std::cerr << "batchData.d_candidateReadIds.data(): " << batchData.d_candidateReadIds.data() << "\n";
                // std::cerr << "batchData.d_numUsedReadIdsPerAnchor.data(): " << batchData.d_numUsedReadIdsPerAnchor.data() << "\n";
                // std::cerr << "batchData.d_numUsedReadIdsPerAnchorPrefixSum.data(): " << batchData.d_numUsedReadIdsPerAnchorPrefixSum.data() << "\n";
                // std::cerr << "batchData.d_usedReadIds.data(): " << batchData.d_usedReadIds.data() << "\n";
                // std::cerr << "batchData.d_isPairedCandidate.data(): " << batchData.d_isPairedCandidate.data() << "\n";

                

            //     cudaDeviceSynchronize(); CUERR;
            // }

            // std::cerr << "batchData.numTasks = " << batchData.numTasks 
            // << " batchData.d_numUsedReadIdsPerAnchor.size() = " << batchData.d_numUsedReadIdsPerAnchor.size() 
            // << " batchData.d_numUsedReadIdsPerAnchorPrefixSum.size() = " << batchData.d_numUsedReadIdsPerAnchorPrefixSum.size() << "\n";

            // helpers::lambda_kernel<<<1, 1024, 0, stream>>>(
            //     [
            //         numChecks,
            //         d_firstTasksOfPairsToCheck,
            //         numTasks = batchData.numTasks,
            //         d_numUsedReadIdsPerAnchor = batchData.d_numUsedReadIdsPerAnchor.data(),
            //         d_numUsedReadIdsPerAnchorPrefixSum = batchData.d_numUsedReadIdsPerAnchorPrefixSum.data(),
            //         d_usedReadIds = batchData.d_usedReadIds.data(),
            //         d_numUsedReadIdsPerAnchorsize = batchData.d_numUsedReadIdsPerAnchor.size(),
            //         d_numUsedReadIdsPerAnchorPrefixSumsize = batchData.d_numUsedReadIdsPerAnchorPrefixSum.size(),
            //         d_usedReadIdssize = int(batchData.d_usedReadIds.size())
            //     ] __device__ (){

            //         assert(d_numUsedReadIdsPerAnchorsize == numTasks);
            //         assert(d_numUsedReadIdsPerAnchorPrefixSumsize == numTasks);

            //         assert(d_numUsedReadIdsPerAnchorPrefixSum[0] == 0);

            //         for(int i = 0; i < numChecks; i++){
            //             assert(d_firstTasksOfPairsToCheck[i] < numTasks);
            //         }

            //         int ps = 0;

            //         for(int i = 0; i < numTasks; i++){
            //             assert(d_numUsedReadIdsPerAnchorPrefixSum[i] == ps);
            //             ps += d_numUsedReadIdsPerAnchor[i];
            //         }

            //         for(int i = 0; i < numTasks; i++){
            //             assert(d_numUsedReadIdsPerAnchorPrefixSum[i] + d_numUsedReadIdsPerAnchor[i] <= d_usedReadIdssize);
            //         }

            //         for(int i = threadIdx.x; i< d_usedReadIdssize; i += blockDim.x){
            //             if(!(d_usedReadIds[i] < 30085710)){
            //                 printf("i %d, d_usedReadIdssize %d, d_usedReadIds[i] %d\n", i, d_usedReadIdssize, d_usedReadIds[i]);
            //             }
            //             assert(d_usedReadIds[i] < 30085710);
            //         }
            //     }
            // ); CUERR;

            //cudaStreamSynchronize(stream);
            

            dim3 block = 128;
            dim3 grid = numChecks;

            helpers::lambda_kernel<<<grid, block, 0, stream>>>(
                [
                    numChecks,
                    d_firstTasksOfPairsToCheck,
                    d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.data(),
                    d_numCandidatesPerAnchorPrefixSum = batchData.d_numCandidatesPerAnchorPrefixSum.data(), // numTasks + 1
                    d_numCandidatesPerAnchorPrefixSumsize = batchData.d_numCandidatesPerAnchorPrefixSum.size(),
                    d_candidateReadIds = batchData.d_candidateReadIds.data(),
                    d_candidateReadIdssize = batchData.d_candidateReadIds.size(),
                    d_numUsedReadIdsPerAnchor = batchData.d_numUsedReadIdsPerAnchor.data(),
                    d_numUsedReadIdsPerAnchorsize = batchData.d_numUsedReadIdsPerAnchor.size(),
                    d_numUsedReadIdsPerAnchorPrefixSum = batchData.d_numUsedReadIdsPerAnchorPrefixSum.data(), // numTasks
                    d_numUsedReadIdsPerAnchorPrefixSumsize = batchData.d_numUsedReadIdsPerAnchorPrefixSum.size(), // numTasks
                    d_usedReadIds = batchData.d_usedReadIds.data(),
                    d_usedReadIdssize = batchData.d_usedReadIds.size(),

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

    }


    void computePairFlagsCpu(BatchData& batchData, cudaStream_t stream) const{
        //computed in removeUsedIdsAndMateIdsCPU
        assert(false && "computePairFlagsCpu cannot be used");

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
        assert(false && "removeUsedIdsAndMateIdsCPU cannot be used");
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

            batchData.d_anchorIndicesWithRemovedMates.resize(batchData.numTasksWithMateRemoved);

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

        batchData.d_anchorIndicesWithRemovedMates.resize(batchData.numTasks);

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

        batchData.d_numCandidatesPerAnchor2.resize(batchData.numTasks);

        
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

        batchData.d_anchorIndicesWithRemovedMates.resize(batchData.numTasksWithMateRemoved);

        cubAllocator->DeviceFree(d_shouldBeKept); CUERR;

        //std::cerr << "new numTasksWithMateRemoved = " << batchData.numTasksWithMateRemoved << ", totalNumCandidates = " << batchData.totalNumCandidates << "\n";

        if(batchData.numTasksWithMateRemoved > 0){

            batchData.d_anchormatedata.resize(batchData.numTasks * batchData.encodedSequencePitchInInts);

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

        batchData.d_numCandidatesPerAnchorPrefixSum2.resize(batchData.numTasks + 1);

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
            batchData.numTasks,
            batchData.d_fullyUsedReadIds.data(),
            batchData.d_numFullyUsedReadIdsPerAnchor.data(),
            batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum.data(),
            batchData.d_segmentIdsOfFullyUsedReadIds.data(),
            *batchData.h_numFullyUsedReadIds,
            batchData.numTasks,        
            batchData.d_candidateReadIds.data(),
            batchData.d_numCandidatesPerAnchor.data(),
            d_anchorIndicesOfCandidates2,
            batchData.numTasks,
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
        //cubAllocator->DeviceAllocate((void**)&d_abortReasons, sizeof(extension::AbortReason) * batchData.numTasks, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_accumExtensionsLengthsOUT, sizeof(int) * batchData.numTasks, stream); CUERR;
        //cubAllocator->DeviceAllocate((void**)&d_outputAnchors, sizeof(char) * batchData.numTasks * batchData.outputAnchorPitchInBytes, stream); CUERR;
        //cubAllocator->DeviceAllocate((void**)&d_outputAnchorQualities, sizeof(char) * batchData.numTasks * batchData.outputAnchorQualityPitchInBytes, stream); CUERR;
        //cubAllocator->DeviceAllocate((void**)&d_outputAnchorLengths, sizeof(int) * batchData.numTasks, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_isPairedTask, sizeof(bool) * batchData.numTasks, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_decodedMatesRevC, sizeof(char) * batchData.numTasks * batchData.decodedMatesRevCPitchInBytes, stream); CUERR;
        //cubAllocator->DeviceAllocate((void**)&d_outputMateHasBeenFound, sizeof(bool) * batchData.numTasks, stream); CUERR;
        cubAllocator->DeviceAllocate((void**)&d_sizeOfGapToMate, sizeof(int) * batchData.numTasks, stream); CUERR;

        //cubAllocator->DeviceAllocate((void**)&d_isFullyUsedCandidate, sizeof(bool) * batchData.totalNumCandidates, stream); CUERR;
        
        batchData.d_isFullyUsedCandidate.resize(batchData.totalNumCandidates);
        batchData.d_outputAnchors.resize(batchData.numTasks * batchData.outputAnchorPitchInBytes);
        batchData.d_outputAnchorQualities.resize(batchData.numTasks * batchData.outputAnchorQualityPitchInBytes);
        batchData.d_outputMateHasBeenFound.resize(batchData.numTasks);
        batchData.d_abortReasons.resize(batchData.numTasks);
        batchData.d_outputAnchorLengths.resize(batchData.numTasks);
        

        d_isFullyUsedCandidate = batchData.d_isFullyUsedCandidate.data();
        d_outputAnchors = batchData.d_outputAnchors.data();
        d_outputAnchorQualities = batchData.d_outputAnchorQualities.data();
        d_outputMateHasBeenFound = batchData.d_outputMateHasBeenFound.data();
        d_abortReasons = batchData.d_abortReasons.data();
        d_outputAnchorLengths = batchData.d_outputAnchorLengths.data();


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
        //cubAllocator->DeviceFree(d_abortReasons); CUERR;
        cubAllocator->DeviceFree(d_accumExtensionsLengthsOUT); CUERR;
        // cubAllocator->DeviceFree(d_outputAnchors); CUERR;
        // cubAllocator->DeviceFree(d_outputAnchorQualities); CUERR;
        //cubAllocator->DeviceFree(d_outputAnchorLengths); CUERR;
        cubAllocator->DeviceFree(d_isPairedTask); CUERR;
        cubAllocator->DeviceFree(d_decodedMatesRevC); CUERR;
        //cubAllocator->DeviceFree(d_outputMateHasBeenFound); CUERR;
        cubAllocator->DeviceFree(d_sizeOfGapToMate); CUERR;
        //cubAllocator->DeviceFree(d_isFullyUsedCandidate); CUERR;
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

    void removeUsedIdsOfFinishedTasks(BatchData& batchData) const{
        const int newNumTasks = batchData.d_newPositionsOfActiveTasks.size();

        if(newNumTasks == 0) return;

        assert(newNumTasks <= batchData.numTasks);

        // {
        //     std::size_t free, total;
        //     cudaMemGetInfo(&free, &total);
        //     std::cerr << "before removeUsedIdsOfFinishedTasks " << free << "\n";
        // }


        //update used ids

        {

            batchData.d_numUsedReadIdsPerAnchor2.resize(newNumTasks);
            batchData.d_numUsedReadIdsPerAnchorPrefixSum2.resize(newNumTasks);           

            helpers::lambda_kernel<<<SDIV(newNumTasks,256), 256, 0, batchData.streams[0]>>>(
                [
                    indicesOfActiveTasks = batchData.d_newPositionsOfActiveTasks.data(),
                    newNumTasks,
                    d_numUsedReadIdsPerAnchorOut = batchData.d_numUsedReadIdsPerAnchor2.data(),
                    d_numUsedReadIdsPerAnchorIn = batchData.d_numUsedReadIdsPerAnchor.data(),
                    d_numUsedReadIdsPerAnchorOutsize = batchData.d_numUsedReadIdsPerAnchor2.size(),
                    d_numUsedReadIdsPerAnchorInsize = batchData.d_numUsedReadIdsPerAnchor.size()
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

            std::size_t bytes = 0;
            void* cubtemp = nullptr;
            cudaError_t cubstatus = cudaSuccess;
            
            cubstatus = cub::DeviceReduce::Sum(
                nullptr, 
                bytes, 
                batchData.d_numUsedReadIdsPerAnchor2.data(), 
                batchData.h_numUsedReadIds.data(),
                newNumTasks,
                batchData.streams[0]
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceAllocate((void**)&cubtemp, bytes, batchData.streams[0]);

            cubstatus = cub::DeviceReduce::Sum(
                cubtemp, 
                bytes, 
                batchData.d_numUsedReadIdsPerAnchor2.data(), 
                batchData.h_numUsedReadIds.data(),
                newNumTasks,
                batchData.streams[0]
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceFree(cubtemp);

            cudaEventRecord(batchData.events[0], batchData.streams[0]);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                nullptr, 
                bytes, 
                batchData.d_numUsedReadIdsPerAnchor2.data(), 
                batchData.d_numUsedReadIdsPerAnchorPrefixSum2.data(),  
                newNumTasks,
                batchData.streams[0]
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceAllocate((void**)&cubtemp, bytes, batchData.streams[0]);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                cubtemp, 
                bytes, 
                batchData.d_numUsedReadIdsPerAnchor2.data(), 
                batchData.d_numUsedReadIdsPerAnchorPrefixSum2.data(), 
                newNumTasks,
                batchData.streams[0]
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceFree(cubtemp);

            cudaEventSynchronize(batchData.events[0]); CUERR; //wait until h_numUsedReadIds is ready

            batchData.d_usedReadIds2.resize(*batchData.h_numUsedReadIds);
            batchData.d_segmentIdsOfUsedReadIds2.resize(*batchData.h_numUsedReadIds);            

            const int possibleNumWarps = SDIV(newNumTasks, 32);
            const int possibleNumBlocks = SDIV(possibleNumWarps, 128 / 32);
            const int numBlocks = std::min(256, possibleNumBlocks);

            helpers::lambda_kernel<<<numBlocks, 128, 0, batchData.streams[0]>>>(
                [
                    indicesOfActiveTasks = batchData.d_newPositionsOfActiveTasks.data(),
                    newNumTasks,
                    d_usedReadIdsIn = batchData.d_usedReadIds.data(),
                    d_usedReadIdsOut = batchData.d_usedReadIds2.data(),
                    d_segmentIdsOfUsedIdsOut = batchData.d_segmentIdsOfUsedReadIds2.data(),
                    d_numUsedReadIdsPerActiveTask = batchData.d_numUsedReadIdsPerAnchor2.data(),
                    inputOffsets = batchData.d_numUsedReadIdsPerAnchorPrefixSum.data(), 
                    outputOffsets = batchData.d_numUsedReadIdsPerAnchorPrefixSum2.data()
                ] __device__ (){
                    //use one warp per segment

                    const int warpid = (threadIdx.x + blockDim.x * blockIdx.x) / 32;
                    const int numwarps = (blockDim.x * gridDim.x) / 32;
                    const int lane = threadIdx.x % 32;

                    for(int t = warpid; t < newNumTasks; t += numwarps){
                        const int activeIndex = indicesOfActiveTasks[t];
                        const int num = d_numUsedReadIdsPerActiveTask[t];
                        const int inputOffset = inputOffsets[activeIndex];
                        const int outputOffset = outputOffsets[t];

                        for(int i = lane; i < num; i += 32){
                            //copy read id
                            d_usedReadIdsOut[outputOffset + i] = d_usedReadIdsIn[inputOffset + i];
                            //set new segment id
                            d_segmentIdsOfUsedIdsOut[outputOffset + i] = t;
                        }
                    }
                }
            ); CUERR;

            std::swap(batchData.d_usedReadIds, batchData.d_usedReadIds2);
            std::swap(batchData.d_numUsedReadIdsPerAnchor, batchData.d_numUsedReadIdsPerAnchor2);
            std::swap(batchData.d_numUsedReadIdsPerAnchorPrefixSum, batchData.d_numUsedReadIdsPerAnchorPrefixSum2);
            std::swap(batchData.d_segmentIdsOfUsedReadIds, batchData.d_segmentIdsOfUsedReadIds2);
        }

        //update fully used ids
        
        {

            batchData.d_numFullyUsedReadIdsPerAnchor2.resize(newNumTasks);
            batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum2.resize(newNumTasks);

            helpers::lambda_kernel<<<SDIV(newNumTasks,256), 256, 0, batchData.streams[0]>>>(
                [
                    indicesOfActiveTasks = batchData.d_newPositionsOfActiveTasks.data(),
                    newNumTasks,
                    d_numFullyUsedReadIdsPerAnchorOut = batchData.d_numFullyUsedReadIdsPerAnchor2.data(),
                    d_numFullyUsedReadIdsPerAnchorIn = batchData.d_numFullyUsedReadIdsPerAnchor.data(),
                    d_numFullyUsedReadIdsPerAnchorOutsize = batchData.d_numFullyUsedReadIdsPerAnchor2.size(),
                    d_numFullyUsedReadIdsPerAnchorInsize = batchData.d_numFullyUsedReadIdsPerAnchor.size()
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

            std::size_t bytes = 0;
            void* cubtemp = nullptr;
            cudaError_t cubstatus = cudaSuccess;
            
            cubstatus = cub::DeviceReduce::Sum(
                nullptr, 
                bytes, 
                batchData.d_numFullyUsedReadIdsPerAnchor2.data(), 
                batchData.h_numFullyUsedReadIds.data(),
                newNumTasks,
                batchData.streams[0]
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceAllocate((void**)&cubtemp, bytes, batchData.streams[0]);

            cubstatus = cub::DeviceReduce::Sum(
                cubtemp, 
                bytes, 
                batchData.d_numFullyUsedReadIdsPerAnchor2.data(), 
                batchData.h_numFullyUsedReadIds.data(),
                newNumTasks,
                batchData.streams[0]
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceFree(cubtemp);

            cudaEventRecord(batchData.events[0], batchData.streams[0]);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                nullptr, 
                bytes, 
                batchData.d_numFullyUsedReadIdsPerAnchor2.data(), 
                batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum2.data(),  
                newNumTasks,
                batchData.streams[0]
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceAllocate((void**)&cubtemp, bytes, batchData.streams[0]);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                cubtemp, 
                bytes, 
                batchData.d_numFullyUsedReadIdsPerAnchor2.data(), 
                batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum2.data(), 
                newNumTasks,
                batchData.streams[0]
            );
            assert(cubstatus == cudaSuccess);

            cubAllocator->DeviceFree(cubtemp);

            cudaEventSynchronize(batchData.events[0]); CUERR; //wait until h_numFullyUsedReadIds is ready

            batchData.d_fullyUsedReadIds2.resize(*batchData.h_numFullyUsedReadIds);
            batchData.d_segmentIdsOfFullyUsedReadIds2.resize(*batchData.h_numFullyUsedReadIds);

            const int possibleNumWarps = SDIV(newNumTasks, 32);
            const int possibleNumBlocks = SDIV(possibleNumWarps, 128 / 32);
            const int numBlocks = std::min(256, possibleNumBlocks);

            helpers::lambda_kernel<<<numBlocks, 128, 0, batchData.streams[0]>>>(
                [
                    indicesOfActiveTasks = batchData.d_newPositionsOfActiveTasks.data(),
                    newNumTasks,
                    d_fullyUsedReadIdsIn = batchData.d_fullyUsedReadIds.data(),
                    d_fullyUsedReadIdsOut = batchData.d_fullyUsedReadIds2.data(),
                    d_fullyUsedReadIdsInsize = batchData.d_fullyUsedReadIds.size(),
                    d_fullyUsedReadIdsOutsize = batchData.d_fullyUsedReadIds2.size(),
                    d_segmentIdsOfFullyUsedIdsOut = batchData.d_segmentIdsOfFullyUsedReadIds2.data(),
                    d_segmentIdsOfFullyUsedIdsOutsize = batchData.d_segmentIdsOfFullyUsedReadIds2.size(),
                    d_numFullyUsedReadIdsPerActiveTask = batchData.d_numFullyUsedReadIdsPerAnchor2.data(),
                    inputOffsets = batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum.data(), 
                    outputOffsets = batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum2.data()
                ] __device__ (){
                    //use one warp per segment

                    const int warpid = (threadIdx.x + blockDim.x * blockIdx.x) / 32;
                    const int numwarps = (blockDim.x * gridDim.x) / 32;
                    const int lane = threadIdx.x % 32;

                    for(int t = warpid; t < newNumTasks; t += numwarps){
                        const int activeIndex = indicesOfActiveTasks[t];
                        const int num = d_numFullyUsedReadIdsPerActiveTask[t];
                        const int inputOffset = inputOffsets[activeIndex];
                        const int outputOffset = outputOffsets[t];

                        for(int i = lane; i < num; i += 32){
                            //copy read id
                            // assert(inputOffset + i < d_fullyUsedReadIdsInsize);
                            // assert(outputOffset + i < d_fullyUsedReadIdsOutsize);
                            d_fullyUsedReadIdsOut[outputOffset + i] = d_fullyUsedReadIdsIn[inputOffset + i];
                            //set new segment id
                            //assert(outputOffset + i < d_segmentIdsOfFullyUsedIdsOutsize);
                            d_segmentIdsOfFullyUsedIdsOut[outputOffset + i] = t;
                        }
                    }
                }
            ); CUERR;

            std::swap(batchData.d_fullyUsedReadIds, batchData.d_fullyUsedReadIds2);
            std::swap(batchData.d_numFullyUsedReadIdsPerAnchor, batchData.d_numFullyUsedReadIdsPerAnchor2);
            std::swap(batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum, batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum2);
            std::swap(batchData.d_segmentIdsOfFullyUsedReadIds, batchData.d_segmentIdsOfFullyUsedReadIds2);

        }

        // {
        //     std::size_t free, total;
        //     cudaMemGetInfo(&free, &total);
        //     std::cerr << "after removeUsedIdsOfFinishedTasks " << free << "\n";
        // }

    }

};

#endif


}


#endif