#include <gpu/minhashkernels.hpp>
#include <gpu/nvtxtimelinemarkers.hpp>

#include <hpc_helpers.cuh>
#include <config.hpp>


#include <nvToolsExt.h>

#include <cub/cub.cuh>

#include <algorithm>
#include <cassert>
#include <vector>


namespace care{










template<int blocksize>
__global__
void compactDataOfUniqueRanges(
        read_number* __restrict__ output,
        const read_number* __restrict__ input,
        const int* __restrict__ sizesOfUniqueRangesPrefixsum,
        const int* __restrict__ rangesPerSequenceBegins,
        int numSequences){

    for(int sequenceIndex = blockIdx.x; sequenceIndex < numSequences; sequenceIndex += gridDim.x){

        const read_number* blockinput = input + rangesPerSequenceBegins[sequenceIndex];
        read_number* blockoutput = output + sizesOfUniqueRangesPrefixsum[sequenceIndex];
        const int numElements = sizesOfUniqueRangesPrefixsum[sequenceIndex + 1] - sizesOfUniqueRangesPrefixsum[sequenceIndex];

        for(int index = threadIdx.x; index < numElements; index += blocksize){
            blockoutput[index] = blockinput[index];
        }
    }

}


template<int blocksize, int numtempregs>
__global__
void makeUniqueRangesKernel(
        read_number* __restrict__ inoutData, 
        int* __restrict__ sizesOfUniqueRanges, 
        int numSequences,
        const read_number* __restrict__ anchorIds,
        const int* __restrict__ rangesPerSequenceBegins,
        int globalOffset){

    using BlockRadixSort = cub::BlockRadixSort<read_number, blocksize, numtempregs>;
    using BlockLoad = cub::BlockLoad<read_number, blocksize, numtempregs, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockDiscontinuity = cub::BlockDiscontinuity<read_number, blocksize>;
    using BlockScan = cub::BlockScan<int, blocksize>; 

    __shared__ union{
        typename BlockRadixSort::TempStorage sort;
        typename BlockLoad::TempStorage load;
        typename BlockDiscontinuity::TempStorage discontinuity;
        typename BlockScan::TempStorage scan;
    } temp_storage;

    __shared__ read_number anchorId;

    for(int sequenceIndex = blockIdx.x; sequenceIndex < numSequences; sequenceIndex += gridDim.x){

        read_number tempregs[numtempregs];   

        #pragma unroll
        for(int i = 0; i <numtempregs; i++){
            tempregs[i] = std::numeric_limits<read_number>::max();
        }

        const int sizeOfRange = rangesPerSequenceBegins[sequenceIndex + 1] - rangesPerSequenceBegins[sequenceIndex];
        if(sizeOfRange == 0){
            if(threadIdx.x == 0){
                sizesOfUniqueRanges[sequenceIndex] = 0;
            }
        }else{
        
            read_number* const myRange = inoutData + rangesPerSequenceBegins[sequenceIndex] - globalOffset;

            assert(sizeOfRange <= numtempregs * blocksize);            

            BlockLoad(temp_storage.load).Load(
                myRange, 
                tempregs, 
                sizeOfRange
            );

            if(threadIdx.x == 0){
                anchorId = anchorIds[sequenceIndex];
            }

            __syncthreads();

            BlockRadixSort(temp_storage.sort).Sort(tempregs);

            __syncthreads();

            int head_flags[numtempregs];

            BlockDiscontinuity(temp_storage.discontinuity).FlagHeads(
                head_flags, 
                tempregs, 
                cub::Inequality()
            );

            __syncthreads();            

            #pragma unroll
            for(int i = 0; i < numtempregs; i++){
                if(threadIdx.x * numtempregs + i >= sizeOfRange){
                    head_flags[i] = 0;
                }else{
                    if(tempregs[i] == anchorId){
                        head_flags[i] = 0;
                    }
                }
            }

            int prefixsum[numtempregs];
            int numberOfSetHeadFlags = 0;

            BlockScan(temp_storage.scan).ExclusiveSum(head_flags, prefixsum, numberOfSetHeadFlags);

            __syncthreads();

            #pragma unroll
            for(int i = 0; i < numtempregs; i++){
                if(threadIdx.x * numtempregs + i < sizeOfRange && head_flags[i] == 1){
                    myRange[prefixsum[i]] = tempregs[i];
                }
            }

            if(threadIdx.x == 0){
                sizesOfUniqueRanges[sequenceIndex] = numberOfSetHeadFlags;
            }
        }
    }

}



template<int blocksize, int numtempregs>
__device__
void makeUniqueRangeSingleWarp(
        read_number* __restrict__ myRange, 
        int sizeOfRange, 
        int* __restrict__ sizeOfUniqueRange,
        typename cub::BlockRadixSort<read_number, blocksize, numtempregs>::TempStorage& sorttemp,
        typename cub::BlockLoad<read_number, blocksize, numtempregs, cub::BLOCK_LOAD_WARP_TRANSPOSE>::TempStorage& loadtemp){

    using BlockRadixSort = cub::BlockRadixSort<read_number, blocksize, numtempregs>;
    using BlockLoad = cub::BlockLoad<read_number, blocksize, numtempregs, cub::BLOCK_LOAD_WARP_TRANSPOSE>;


    read_number tempregs[numtempregs];   

    #pragma unroll
    for(int i = 0; i <numtempregs; i++){
        tempregs[i] = std::numeric_limits<read_number>::max();
    }

    BlockLoad(loadtemp).Load(
        myRange, 
        tempregs, 
        sizeOfRange
    );

    __syncthreads();

    BlockRadixSort(sorttemp).SortBlockedToStriped(tempregs);

    int numUniqueElements = 0;
 
    #pragma unroll
    for(int i = 0; i < numtempregs; i++){
        const read_number curElement = tempregs[i];

        read_number nextElement = threadIdx.x == 0 ? tempregs[(i+1) % numtempregs] : tempregs[i];
        nextElement = __shfl_sync(0xFFFFFFFF, nextElement, threadIdx.x+1);               

        //find elements which are not equal to their right neighbor and not out of range
        const bool predicate = (curElement != nextElement) && (i * blocksize + threadIdx.x < sizeOfRange);

        const uint32_t mask = __ballot_sync(0xFFFFFFFF, predicate);
        const uint32_t count = __popc(mask);

        //get position
        const uint32_t numPre = __popc(mask & ((1 << threadIdx.x) -1));
        const int position = numUniqueElements + numPre;

        if(predicate){
            myRange[position] = curElement;
        }
        numUniqueElements += count;
    }
    
    if(threadIdx.x == 0){
        *sizeOfUniqueRange = numUniqueElements;
    }
}


template<int blocksize, int numtempregs>
__global__
void makeUniqueRangesKernelWithIntrinsicsSingleWarp(
        read_number* __restrict__ inoutData, 
        int* __restrict__ sizesOfUniqueRanges, 
        int numSequences,
        const int* __restrict__ rangesPerSequenceBegins,
        int globalOffset){

    static_assert(blocksize == 32, "blocksize must be 32 for SingleWarp kernel");

    using BlockRadixSort = cub::BlockRadixSort<read_number, blocksize, numtempregs>;
    using BlockLoad = cub::BlockLoad<read_number, blocksize, numtempregs, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    
    __shared__ union{
        typename BlockRadixSort::TempStorage sort;
        typename BlockLoad::TempStorage load;
    } temp_storage;

    for(int sequenceIndex = blockIdx.x; sequenceIndex < numSequences; sequenceIndex += gridDim.x){

        const int sizeOfRange = rangesPerSequenceBegins[sequenceIndex + 1] - rangesPerSequenceBegins[sequenceIndex];
        assert(sizeOfRange <= numtempregs * blocksize);

        if(sizeOfRange == 0){
            if(threadIdx.x == 0){
                sizesOfUniqueRanges[sequenceIndex] = 0;
            }
        }else{
        
            read_number* const myRange = inoutData + rangesPerSequenceBegins[sequenceIndex] - globalOffset;

            makeUniqueRangeSingleWarp<blocksize, numtempregs>(
                myRange, 
                sizeOfRange, 
                &sizesOfUniqueRanges[sequenceIndex],
                temp_storage.sort,
                temp_storage.load
            );
        }
    }

}





template<int blocksize, int numtempregs>
__global__
void makeUniqueRangesKernelWithIntrinsicsSingleWarpChunked(
        read_number* __restrict__ inoutData, 
        int* __restrict__ sizesOfUniqueRanges, 
        int numSequences,
        const int* __restrict__ rangesPerSequenceBegins,
        int globalOffset){

    static_assert(blocksize == 32, "blocksize must be 32 for SingleWarp kernel");

    using BlockRadixSortFull = cub::BlockRadixSort<read_number, blocksize, numtempregs>;
    using BlockLoadFull = cub::BlockLoad<read_number, blocksize, numtempregs, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

    using BlockRadixSort1 = cub::BlockRadixSort<read_number, blocksize, 1>;
    using BlockLoad1 = cub::BlockLoad<read_number, blocksize, 1, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

    using BlockRadixSort2 = cub::BlockRadixSort<read_number, blocksize, 2>;
    using BlockLoad2 = cub::BlockLoad<read_number, blocksize, 2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

    using BlockRadixSort4 = cub::BlockRadixSort<read_number, blocksize, 4>;
    using BlockLoad4 = cub::BlockLoad<read_number, blocksize, 4, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

    using BlockRadixSort8 = cub::BlockRadixSort<read_number, blocksize, 8>;
    using BlockLoad8 = cub::BlockLoad<read_number, blocksize, 8, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

    using BlockRadixSort16 = cub::BlockRadixSort<read_number, blocksize, 16>;
    using BlockLoad16 = cub::BlockLoad<read_number, blocksize, 16, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

    using BlockRadixSort32 = cub::BlockRadixSort<read_number, blocksize, 32>;
    using BlockLoad32 = cub::BlockLoad<read_number, blocksize, 32, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

    using BlockRadixSort64 = cub::BlockRadixSort<read_number, blocksize, 64>;
    using BlockLoad64 = cub::BlockLoad<read_number, blocksize, 64, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    
    __shared__ union{
        typename BlockRadixSortFull::TempStorage sortFull;
        typename BlockLoadFull::TempStorage loadFull;

        typename BlockRadixSort1::TempStorage sort1;
        typename BlockLoad1::TempStorage load1;

        typename BlockRadixSort2::TempStorage sort2;
        typename BlockLoad2::TempStorage load2;

        typename BlockRadixSort4::TempStorage sort4;
        typename BlockLoad4::TempStorage load4;

        typename BlockRadixSort8::TempStorage sort8;
        typename BlockLoad8::TempStorage load8;

        typename BlockRadixSort16::TempStorage sort16;
        typename BlockLoad16::TempStorage load16;

        typename BlockRadixSort32::TempStorage sort32;
        typename BlockLoad32::TempStorage load32;

        typename BlockRadixSort64::TempStorage sort64;
        typename BlockLoad64::TempStorage load64;
    } temp_storage;

    for(int sequenceIndex = blockIdx.x; sequenceIndex < numSequences; sequenceIndex += gridDim.x){

        const int sizeOfRange = rangesPerSequenceBegins[sequenceIndex + 1] - rangesPerSequenceBegins[sequenceIndex];
        assert(sizeOfRange <= numtempregs * blocksize);

        if(sizeOfRange == 0){
            if(threadIdx.x == 0){
                sizesOfUniqueRanges[sequenceIndex] = 0;
            }
        }else{
            read_number* const myRange = inoutData + rangesPerSequenceBegins[sequenceIndex] - globalOffset;
            int* const sizeOfUniqueRange = sizesOfUniqueRanges + sequenceIndex;

            if(sizeOfRange <= blocksize * 1){
                makeUniqueRangeSingleWarp<blocksize, 1>(
                    myRange, 
                    sizeOfRange, 
                    sizeOfUniqueRange,
                    temp_storage.sort1,
                    temp_storage.load1
                );
            }else if(sizeOfRange <= blocksize * 2){
                makeUniqueRangeSingleWarp<blocksize, 2>(
                    myRange, 
                    sizeOfRange, 
                    sizeOfUniqueRange,
                    temp_storage.sort2,
                    temp_storage.load2
                );
            }else if(sizeOfRange <= blocksize * 4){
                makeUniqueRangeSingleWarp<blocksize, 4>(
                    myRange, 
                    sizeOfRange, 
                    sizeOfUniqueRange,
                    temp_storage.sort4,
                    temp_storage.load4
                );
            }else if(sizeOfRange <= blocksize * 8){
                makeUniqueRangeSingleWarp<blocksize, 8>(
                    myRange, 
                    sizeOfRange, 
                    sizeOfUniqueRange,
                    temp_storage.sort8,
                    temp_storage.load8
                );
            }else if(sizeOfRange <= blocksize * 16){
                makeUniqueRangeSingleWarp<blocksize, 16>(
                    myRange, 
                    sizeOfRange, 
                    sizeOfUniqueRange,
                    temp_storage.sort16,
                    temp_storage.load16
                );
            }else if(sizeOfRange <= blocksize * 32){
                makeUniqueRangeSingleWarp<blocksize, 32>(
                    myRange, 
                    sizeOfRange, 
                    sizeOfUniqueRange,
                    temp_storage.sort32,
                    temp_storage.load32
                );
            }else if(sizeOfRange <= blocksize * 64){
                makeUniqueRangeSingleWarp<blocksize, 64>(
                    myRange, 
                    sizeOfRange, 
                    sizeOfUniqueRange,
                    temp_storage.sort64,
                    temp_storage.load64
                );
            }else{
                makeUniqueRangeSingleWarp<blocksize, numtempregs>(
                    myRange, 
                    sizeOfRange, 
                    sizeOfUniqueRange,
                    temp_storage.sortFull,
                    temp_storage.loadFull
                );
            }

            
        }
    }

}




template<int blocksize, int numtempregs>
__global__
void makeUniqueRangesKernelWithIntrinsicsMultiWarp(
        read_number* __restrict__ inoutData, 
        int* __restrict__ sizesOfUniqueRanges, 
        int numSequences,
        const int* __restrict__ rangesPerSequenceBegins,
        int globalOffset){

    using BlockRadixSort = cub::BlockRadixSort<read_number, blocksize, numtempregs>;
    using BlockLoad = cub::BlockLoad<read_number, blocksize, numtempregs, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using WarpScan = cub::WarpScan<int>;
    
    constexpr int warpsize = 32;
    constexpr int numWarpsPerBlock = blocksize / warpsize;
    static_assert(blocksize % warpsize == 0, "blocksize not multiple of warpsize");

    __shared__ union{
        typename BlockRadixSort::TempStorage sort;
        typename BlockLoad::TempStorage load;
        typename WarpScan::TempStorage warpscan[numWarpsPerBlock];  
    } temp_storage;

    __shared__ read_number rightNeighborPerWarp[numWarpsPerBlock];
    __shared__ int countsPerWarp[numWarpsPerBlock];
    __shared__ int countsPrefixsum[numWarpsPerBlock+1];

    const int warpId = threadIdx.x / warpsize;
    const int laneId = threadIdx.x % warpsize;

    for(int sequenceIndex = blockIdx.x; sequenceIndex < numSequences; sequenceIndex += gridDim.x){

        read_number tempregs[numtempregs];   

        #pragma unroll
        for(int i = 0; i <numtempregs; i++){
            tempregs[i] = std::numeric_limits<read_number>::max();
        }

        const int sizeOfRange = rangesPerSequenceBegins[sequenceIndex + 1] - rangesPerSequenceBegins[sequenceIndex];
        if(sizeOfRange == 0){
            if(threadIdx.x == 0){
                sizesOfUniqueRanges[sequenceIndex] = 0;
            }
        }else{
        
            read_number* const myRange = inoutData + rangesPerSequenceBegins[sequenceIndex] - globalOffset;

            assert(sizeOfRange <= numtempregs * blocksize);

            BlockLoad(temp_storage.load).Load(
                myRange, 
                tempregs, 
                sizeOfRange
            );

            __syncthreads();

            BlockRadixSort(temp_storage.sort).SortBlockedToStriped(tempregs);

            int numUniqueElements = 0;
            #pragma unroll
            for(int i = 0; i < numtempregs; i++){                

                //save input for previous warp in shared memory
                if(laneId == 0){
                    if(warpId == 0){
                        rightNeighborPerWarp[numWarpsPerBlock-1] = tempregs[(i+1) % numtempregs];
                    }else{
                        rightNeighborPerWarp[warpId-1] = tempregs[i];
                    }
                    countsPerWarp[warpId] = 0;
                }
                __syncthreads();

                const read_number curElement = tempregs[i];

                read_number nextElement = laneId == 0 ? rightNeighborPerWarp[warpId] : tempregs[i];
                nextElement = __shfl_sync(0xFFFFFFFF, nextElement, laneId+1);               

                //find elements which are not equal to their right neighbor and not out of range
                const bool predicate = (curElement != nextElement) && (i * blocksize + threadIdx.x < sizeOfRange);

                const uint32_t mask = __ballot_sync(0xFFFFFFFF, predicate);
                const int count = __popc(mask);

                if(laneId == 0){
                    countsPerWarp[warpId] = count;
                }
                __syncthreads();

                if(warpId == 0){
                    int c = laneId < numWarpsPerBlock ? countsPerWarp[laneId] : 0;
                    int warp_aggregate = 0;
                    WarpScan(temp_storage.warpscan[warpId]).ExclusiveSum(c, c, warp_aggregate);
                    //__syncwarp();

                    if(laneId < numWarpsPerBlock){
                        countsPrefixsum[laneId] = c;
                    }
                    if(laneId == 0){
                        countsPrefixsum[numWarpsPerBlock] = warp_aggregate;
                    }
                }

                const int numPre = __popc(mask & ((1 << laneId) -1));

                __syncthreads();

                //get position                
                const int position = numUniqueElements + countsPrefixsum[warpId] + numPre;

                if(predicate){
                    myRange[position] = curElement;
                }
                numUniqueElements += countsPrefixsum[numWarpsPerBlock];
            }

            if(threadIdx.x == 0){
                sizesOfUniqueRanges[sequenceIndex] = numUniqueElements;
            }
        }
    }

}









void makeCompactUniqueRangesGmem(
        MergeRangesGpuHandle<read_number>& handle, 
        const std::pair<const read_number*, const read_number*>* ranges,
        int numRanges, 
        const read_number* d_anchorIds, 
        int rangesPerSequence, 
        int totalNumElements, 
        bool onlyAlloc,
        cudaStream_t stream){

    assert(false && "cannot use gmem merge currently");

    const int numSequences = numRanges / rangesPerSequence;
    if(numSequences == 0){
        return;
    }
    
    
    {
        size_t temp_storage_bytes = 0;
        size_t temp_storage_bytes2 = 0;

        cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr, 
            temp_storage_bytes, 
            handle.d_data.get(), 
            handle.d_results.get(),
            totalNumElements, 
            numSequences, 
            handle.d_rangesBeginPerSequence.get(), 
            handle.d_rangesBeginPerSequence.get() + 1,
            0,
            sizeof(read_number) * 8,
            stream
        );

        using CountIter = cub::CountingInputIterator<int>;
        read_number* d_data = handle.d_data.get();
        auto toHeadFlag = [=] __device__ (int index){
            if(index == 0 || d_data[index-1] != d_data[index]){
                return 1;
            }else{
                return 0;
            }
        };

        using TransformOp = decltype(toHeadFlag);
        using HeadFlagsIter = cub::TransformInputIterator<int, TransformOp, CountIter>;

        HeadFlagsIter head_flags(CountIter{0}, toHeadFlag);

        cub::DeviceSegmentedReduce::Sum(
            nullptr, 
            temp_storage_bytes2, 
            head_flags, 
            handle.d_uniqueRangeLengths.get(),
            numSequences, 
            handle.d_rangesBeginPerSequence.get(), 
            handle.d_rangesBeginPerSequence.get() + 1,
            stream
        );

        temp_storage_bytes = std::max(temp_storage_bytes, temp_storage_bytes2);
    
        cub::DeviceScan::InclusiveSum(
            nullptr, 
            temp_storage_bytes2, 
            handle.d_uniqueRangeLengths.get(), 
            handle.d_uniqueRangeLengthsPrefixsum.get() + 1, 
            numSequences,
            stream
        );

        temp_storage_bytes = std::max(temp_storage_bytes, temp_storage_bytes2);
    
        cub::DeviceSelect::Flagged(
            nullptr, 
            temp_storage_bytes2,
            handle.d_data.get(), 
            head_flags, 
            handle.d_results.get(), 
            cub::DiscardOutputIterator<int>{}, //d_num_selected_out
            totalNumElements,
            stream
        );

        temp_storage_bytes = std::max(temp_storage_bytes, temp_storage_bytes2);
        handle.cubTempStorage.resize(temp_storage_bytes);
    }

    if(onlyAlloc){
        return;
    }

    cudaMemcpyAsync(
        handle.d_rangesBeginPerSequence.get(), 
        handle.h_rangesBeginPerSequence.get(),
        sizeof(int) * (numSequences+1),
        H2D,
        stream
    ); CUERR;

    //copy data of ranges into contiguous pinned memory, then to device async
    {
        auto begin = handle.h_data.get();
        auto end = handle.h_data.get();
        for(int rangeIndex = 0; rangeIndex < numRanges; rangeIndex++){
            end = std::copy(
                ranges[rangeIndex].first, 
                ranges[rangeIndex].second,
                begin
            );
            
            begin = end;
        }
    }

    cudaMemcpyAsync(
        handle.d_data.get(), 
        handle.h_data.get(),
        sizeof(read_number) * totalNumElements,
        H2D,
        stream
    ); CUERR;

    size_t cubTempStorageSize = handle.cubTempStorage.sizeInBytes();

    cub::DeviceSegmentedRadixSort::SortKeys(
        handle.cubTempStorage.get(), 
        cubTempStorageSize, 
        handle.d_data.get(), 
        handle.d_results.get(),
        totalNumElements, 
        numSequences, 
        handle.d_rangesBeginPerSequence.get(), 
        handle.d_rangesBeginPerSequence.get() + 1,
        0,
        sizeof(read_number) * 8,
        stream
    );

    std::swap(handle.d_data, handle.d_results);

    using CountIter = cub::CountingInputIterator<int>;
    read_number* d_data = handle.d_data.get();
    auto toHeadFlag = [=] __device__ (int index){
        if(index == 0 || d_data[index-1] != d_data[index]){
            return 1;
        }else{
            return 0;
        }
    };

    using TransformOp = decltype(toHeadFlag);
    using HeadFlagsIter = cub::TransformInputIterator<int, TransformOp, CountIter>;


    HeadFlagsIter head_flags(CountIter{0}, toHeadFlag);

    cub::DeviceSegmentedReduce::Sum(
        handle.cubTempStorage.get(), 
        cubTempStorageSize, 
        head_flags, 
        handle.d_uniqueRangeLengths.get(),
        numSequences, 
        handle.d_rangesBeginPerSequence.get(), 
        handle.d_rangesBeginPerSequence.get() + 1,
        stream
    );

    cub::DeviceScan::InclusiveSum(
        handle.cubTempStorage.get(), 
        cubTempStorageSize, 
        handle.d_uniqueRangeLengths.get(), 
        handle.d_uniqueRangeLengthsPrefixsum.get() + 1, 
        numSequences,
        stream
    );

    cub::DeviceSelect::Flagged(
        handle.cubTempStorage.get(), 
        cubTempStorageSize,
        handle.d_data.get(), 
        head_flags, 
        handle.d_results.get(), 
        cub::DiscardOutputIterator<int>{}, //d_num_selected_out == uniqueRangelengthsPrefixSum[numSequences]
        totalNumElements,
        stream
    );

       
}



void makeCompactUniqueRangesSmem(
        MergeRangesGpuHandle<read_number>& handle, 
        read_number* d_compactUniqueCandidateIds,
        int* d_candidatesPerAnchor,
        int* d_candidatesPerAnchorPrefixSum,
        read_number* d_candidateIds,
        const std::pair<const read_number*, const read_number*>* ranges,
        int numRanges, 
        const read_number* d_anchorIds, 
        int rangesPerSequence, 
        int totalNumElements, 
        bool onlyAlloc,
        MergeRangesKernelType kernelType,
        cudaStream_t stream){

    const int numSequences = numRanges / rangesPerSequence;
    if(numSequences == 0){
        return;
    }

    {
        size_t temp_storage_bytes = 0;

        cub::DeviceScan::InclusiveSum(
            nullptr, 
            temp_storage_bytes,
            d_candidatesPerAnchor,
            d_candidatesPerAnchorPrefixSum + 1,
            numSequences,
            stream
        ); CUERR;
    
        handle.cubTempStorage.resize(temp_storage_bytes);
    }

    if(onlyAlloc){
        return;
    }

    cudaMemcpyAsync(
        handle.d_rangesBeginPerSequence.get(), 
        handle.h_rangesBeginPerSequence.get(),
        sizeof(int) * (numSequences+1),
        H2D,
        stream
    ); CUERR;

    cudaEventRecord(handle.events.back(), stream); CUERR;

    for(auto& pipelinestream : handle.streams){
        cudaStreamWaitEvent(pipelinestream, handle.events.back(), 0); CUERR;
    }

    {
        const int numstreams = handle.streams.size();

        int elementOffset = 0;
        int sequenceOffset = 0;

        for(int i = 0; i < numstreams; i++){
            const int sequenceBegin = i * numSequences / numstreams;
            const int sequenceEnd = std::min((i+1) * numSequences / numstreams, numSequences);

            const int mynumSequences = sequenceEnd - sequenceBegin;

            int mynumElements = 0;
            int largestNumElementsPerSequence = 0;
            for(int sequenceIndex = sequenceBegin; sequenceIndex < sequenceEnd; sequenceIndex++){
                
                int numElementsForSequence = 0;

                for(int k = 0; k < rangesPerSequence; k++){
                    const int rangeIndex = sequenceIndex * rangesPerSequence + k;    
                    const int sizeOfRange = std::distance(ranges[rangeIndex].first, ranges[rangeIndex].second);
                    numElementsForSequence += sizeOfRange;
                }

                largestNumElementsPerSequence = std::max(largestNumElementsPerSequence, numElementsForSequence);                
            } 

            #define processData(blocksize, numtempregs) \
            { \
                switch(kernelType){ \
                case MergeRangesKernelType::allcub: \
                    makeUniqueRangesKernel<(blocksize), (numtempregs)><<<mynumSequences, (blocksize), 0, handle.streams[i]>>>( \
                        d_candidateIds + elementOffset,  \
                        d_candidatesPerAnchor + sequenceOffset,  \
                        mynumSequences, \
                        d_anchorIds, \
                        handle.d_rangesBeginPerSequence.get() + sequenceOffset, \
                        elementOffset \
                    ); CUERR; \
                    break; \
                case MergeRangesKernelType::popcmultiwarp: \
                    makeUniqueRangesKernelWithIntrinsicsMultiWarp<(blocksize), (numtempregs)><<<mynumSequences, (blocksize), 0, handle.streams[i]>>>( \
                        d_candidateIds + elementOffset,  \
                        d_candidatesPerAnchor + sequenceOffset,  \
                        mynumSequences, \
                        handle.d_rangesBeginPerSequence.get() + sequenceOffset, \
                        elementOffset \
                    ); CUERR; \
                    break; \
                case MergeRangesKernelType::popcsinglewarp: \
                    makeUniqueRangesKernelWithIntrinsicsSingleWarp<32, 64><<<mynumSequences, 32, 0, handle.streams[i]>>>( \
                        d_candidateIds + elementOffset,  \
                        d_candidatesPerAnchor + sequenceOffset,  \
                        mynumSequences, \
                        handle.d_rangesBeginPerSequence.get() + sequenceOffset, \
                        elementOffset \
                    ); CUERR; \
                    break; \
                case MergeRangesKernelType::popcsinglewarpchunked: \
                    makeUniqueRangesKernelWithIntrinsicsSingleWarpChunked<32, 64><<<mynumSequences, 32, 0, handle.streams[i]>>>( \
                        d_candidateIds + elementOffset,  \
                        d_candidatesPerAnchor + sequenceOffset,  \
                        mynumSequences, \
                        handle.d_rangesBeginPerSequence.get() + sequenceOffset, \
                        elementOffset \
                    ); CUERR; \
                    break; \
                default: std::cerr << "unknown kernel type\n"; \
                } \
            }

            if(largestNumElementsPerSequence <= 32){
                constexpr int blocksize = 32;
                constexpr int numtempregs = 1;
                assert(largestNumElementsPerSequence <= blocksize * numtempregs);

                processData(blocksize, numtempregs);
            }else if(largestNumElementsPerSequence <= 64){
                constexpr int blocksize = 64;
                constexpr int numtempregs = 1;
                assert(largestNumElementsPerSequence <= blocksize * numtempregs);

                processData(blocksize, numtempregs);
            }else if(largestNumElementsPerSequence <= 96){
                constexpr int blocksize = 96;
                constexpr int numtempregs = 1;
                assert(largestNumElementsPerSequence <= blocksize * numtempregs);

                processData(blocksize, numtempregs);
            }else if(largestNumElementsPerSequence <= 128){
                constexpr int blocksize = 128;
                constexpr int numtempregs = 1;
                assert(largestNumElementsPerSequence <= blocksize * numtempregs);

                processData(blocksize, numtempregs);
            }else if(largestNumElementsPerSequence <= 256){
                constexpr int blocksize = 128;
                constexpr int numtempregs = 2;
                assert(largestNumElementsPerSequence <= blocksize * numtempregs);

                processData(blocksize, numtempregs);
            }else if(largestNumElementsPerSequence <= 512){
                constexpr int blocksize = 128;
                constexpr int numtempregs = 4;
                assert(largestNumElementsPerSequence <= blocksize * numtempregs);

                processData(blocksize, numtempregs);
            }else if(largestNumElementsPerSequence <= 1024){
                constexpr int blocksize = 128;
                constexpr int numtempregs = 8;
                assert(largestNumElementsPerSequence <= blocksize * numtempregs);

                processData(blocksize, numtempregs);
            }else if(largestNumElementsPerSequence <= 2048){
                constexpr int blocksize = 128;
                constexpr int numtempregs = 16;
                assert(largestNumElementsPerSequence <= blocksize * numtempregs);

                processData(blocksize, numtempregs);
            }else if(largestNumElementsPerSequence <= 4096){
                constexpr int blocksize = 128;
                constexpr int numtempregs = 32;
                assert(largestNumElementsPerSequence <= blocksize * numtempregs);

                processData(blocksize, numtempregs);
            }else{
                constexpr int blocksize = 128;
                constexpr int numtempregs = 64;
                assert(largestNumElementsPerSequence <= blocksize * numtempregs);
                
                processData(blocksize, numtempregs);
            }


            cudaEventRecord(handle.events[i], handle.streams[i]); CUERR;
            cudaStreamWaitEvent(stream, handle.events[i], 0); CUERR;

            elementOffset += mynumElements;
            sequenceOffset += mynumSequences;
        }
        
    }    

    size_t tmpsize = handle.cubTempStorage.sizeInBytes();

    cub::DeviceScan::InclusiveSum(
        handle.cubTempStorage.get(), 
        tmpsize,
        d_candidatesPerAnchor,
        d_candidatesPerAnchorPrefixSum + 1,
        numSequences,
        stream
    ); CUERR;

    compactDataOfUniqueRanges<256><<<numSequences, 256, 0, stream>>>(
        d_compactUniqueCandidateIds,
        d_candidateIds,
        d_candidatesPerAnchorPrefixSum,
        handle.d_rangesBeginPerSequence.get(),
        numSequences
    ); CUERR;

    //cudaEventRecord(handle.events.back(), stream); CUERR;   

}



void mergeRangesGpuAsync(
        MergeRangesGpuHandle<read_number>& handle, 
        read_number* d_compactUniqueCandidateIds,
        int* d_candidatesPerAnchor,
        int* d_candidatesPerAnchorPrefixSum,
        read_number* d_candidateIds,
        const std::pair<const read_number*, const read_number*>* h_ranges, 
        int numRanges, 
        const read_number* d_anchorIds, 
        int rangesPerSequence, 
        cudaStream_t stream,
        MergeRangesKernelType kernelType){
    
    const int numSequences = numRanges / rangesPerSequence;
    if(numSequences == 0){
        return;
    }

    handle.d_rangesBeginPerSequence.resize(numSequences+1);
    handle.h_rangesBeginPerSequence.resize(numSequences+1);

    handle.h_rangesBeginPerSequence[0] = 0;

    //int longestRange = 0;

    //nvtx::push_range("longestrange", 4);
    int maxNumResults = 0;
    for(int i = 0; i < numSequences; i++){   
        int rangeOfSequence = 0;     
        for(int k = 0; k < rangesPerSequence; k++){
            const int rangeIndex = i * rangesPerSequence + k;
            maxNumResults += std::distance(h_ranges[rangeIndex].first, h_ranges[rangeIndex].second);
            //rangeOfSequence += std::distance(h_ranges[rangeIndex].first, h_ranges[rangeIndex].second);
        }
        //longestRange = std::max(longestRange, rangeOfSequence);
        handle.h_rangesBeginPerSequence[i+1] = maxNumResults;
    }
    //nvtx::pop_range();

    //std::cerr << "longestRange = " << longestRange << "\n";
    handle.d_data.resize(maxNumResults);
    handle.h_data.resize(maxNumResults);

    handle.d_results.resize(maxNumResults);
    handle.h_results.resize(maxNumResults);

    handle.h_numresults.resize(1);

    handle.d_uniqueRangeLengths.resize(numSequences);
    handle.h_uniqueRangeLengths.resize(numSequences);

    handle.d_uniqueRangeLengthsPrefixsum.resize(numSequences + 1);
    cudaMemsetAsync(handle.d_uniqueRangeLengthsPrefixsum, 0, sizeof(int), stream); CUERR;


    //TIMERSTARTCPU(makeCompactUniqueRanges);

    if(kernelType == MergeRangesKernelType::devicewide){
        makeCompactUniqueRangesGmem(
            handle, 
            h_ranges,
            numRanges, 
            d_anchorIds,
            rangesPerSequence, 
            maxNumResults, 
            false,
            stream
        );
        
    }else{
        makeCompactUniqueRangesSmem(
            handle, 
            d_compactUniqueCandidateIds,
            d_candidatesPerAnchor,
            d_candidatesPerAnchorPrefixSum,
            d_candidateIds,
            h_ranges,
            numRanges, 
            d_anchorIds,
            rangesPerSequence, 
            maxNumResults, 
            false,
            kernelType,
            stream
        );
    }

}



OperationResult mergeRangesGpu(
        MergeRangesGpuHandle<read_number>& handle, 
        const std::pair<const read_number*, const read_number*>* h_ranges, 
        int numRanges, 
        const read_number* d_anchorIds, 
        int rangesPerSequence, 
        cudaStream_t stream,
        MergeRangesKernelType kernelType){
    
    const int numSequences = numRanges / rangesPerSequence;
    if(numSequences == 0){
        return OperationResult{};
    }

    // mergeRangesGpuAsync(
    //     handle, 
    //     h_ranges, 
    //     numRanges, 
    //     d_anchorIds, 
    //     rangesPerSequence, 
    //     stream,
    //     kernelType
    // );

    cudaMemcpyAsync(
        handle.h_uniqueRangeLengths.get(), 
        handle.d_uniqueRangeLengths.get(), 
        sizeof(int) * numSequences, 
        D2H, 
        stream
    ); CUERR;
    cudaMemcpyAsync(
        handle.h_numresults.get(), 
        handle.d_uniqueRangeLengthsPrefixsum.get() + numSequences, 
        sizeof(int), 
        D2H, 
        stream
    ); CUERR;
    cudaMemcpyAsync(
        handle.h_results.get(), 
        handle.d_results.get(), 
        sizeof(read_number) * handle.d_results.size(), 
        D2H, 
        stream
    ); CUERR; 


    cudaStreamSynchronize(stream); CUERR;

    //TIMERSTOPCPU(makeCompactUniqueRanges);

    OperationResult result;

    result.candidateIds.clear();
    result.candidateIds.resize(*handle.h_numresults.get());

    result.candidateIdsPerSequence.clear();
    result.candidateIdsPerSequence.resize(numSequences, 0);

    std::copy_n(
        handle.h_results.get(), 
        *handle.h_numresults.get(), 
        result.candidateIds.begin()
    );

    std::copy_n(
        handle.h_uniqueRangeLengths.get(), 
        numSequences, 
        result.candidateIdsPerSequence.begin()
    );

    return result;

}



} //namespace care