#ifndef CARE_MINHASHINGKERNELS_CUH
#define CARE_MINHASHINGKERNELS_CUH

#include <cstdint>

#include <gpu/cuda_block_select.cuh>
#include <cub/cub.cuh>

namespace care{
namespace gpu{

    void callMinhashSignaturesKernel(
        std::uint64_t* d_signatures,
        std::size_t signaturesRowPitchElements,
        const unsigned int* d_sequences2Bit,
        std::size_t sequenceRowPitchElements,
        int numSequences,
        const int* d_sequenceLengths,
        int k,
        int numHashFuncs,
        int firstHashFunc,
        cudaStream_t stream
    );

    void callMinhashSignaturesKernel(
        std::uint64_t* d_signatures,
        std::size_t signaturesRowPitchElements,
        const unsigned int* d_sequences2Bit,
        std::size_t sequenceRowPitchElements,
        int numSequences,
        const int* d_sequenceLengths,
        int k,
        int numHashFuncs,
        const int* d_hashFunctionNumbers,
        cudaStream_t stream
    );

    /*
    If toFind[s] exists in segment s, remove it from this segment s 
    by shifting remaining elements to the left.
    segmentSizes[s] will be updated.
    */
    template<class T, int BLOCKSIZE, int ITEMS_PER_THREAD>
    __global__ 
    void findAndRemoveFromSegmentKernel(
        const T* __restrict__ toFind,
        T* items,
        int numSegments,
        int* __restrict__ segmentSizes,
        const int* __restrict__ segmentBeginOffsets
    ){
        constexpr int itemsPerIteration = ITEMS_PER_THREAD * BLOCKSIZE;

        assert(BLOCKSIZE == blockDim.x);

        using BlockLoad = cub::BlockLoad<T, BLOCKSIZE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
        using MyBlockSelect = BlockSelect<T, BLOCKSIZE>;

        __shared__ union{
            typename BlockLoad::TempStorage load;
            typename MyBlockSelect::TempStorage select;
        } temp_storage;

        for(int s = blockIdx.x; s < numSegments; s += gridDim.x){
            const int segmentsize = segmentSizes[s];
            const int beginOffset = segmentBeginOffsets[s];
            const T idToRemove = toFind[s];

            const int numIterations = SDIV(segmentsize, itemsPerIteration);
            T myitems[ITEMS_PER_THREAD];
            int flags[ITEMS_PER_THREAD];

            int numSelectedTotal = 0;
            int remainingItems = segmentsize;
            const T* inputdata = items + beginOffset;
            T* outputdata = items + beginOffset;

            for(int iter = 0; iter < numIterations; iter++){
                const int validItems = min(remainingItems, itemsPerIteration);
                BlockLoad(temp_storage.load).Load(inputdata, myitems, validItems);

                #pragma unroll
                for(int i = 0; i < ITEMS_PER_THREAD; i++){
                    if(threadIdx.x * ITEMS_PER_THREAD + i < validItems && myitems[i] != idToRemove){
                        flags[i] = 1;
                    }else{
                        flags[i] = 0;
                    }
                }

                __syncthreads();

                const int numSelected = MyBlockSelect(temp_storage.select).ForEachFlagged(myitems, flags, validItems,
                    [&](const auto& item, const int& pos){
                        outputdata[pos] = item;
                    }
                );
                assert(numSelected <= validItems);

                numSelectedTotal += numSelected;
                outputdata += numSelected;
                inputdata += validItems;
                remainingItems -= validItems;

                __syncthreads();
            }

            assert(segmentsize >= numSelectedTotal);

            //update segment size
            if(numSelectedTotal != segmentsize){
                if(threadIdx.x == 0){
                    segmentSizes[s] = numSelectedTotal;
                }
            }
        }
    }

    template<class T, int BLOCKSIZE, int ITEMS_PER_THREAD>
    void callFindAndRemoveFromSegmentKernel(
        const T* d_toFind,
        T* d_items,
        int numSegments,
        int* d_segmentSizes,
        const int* d_segmentBeginOffsets,
        cudaStream_t stream
    ){
        if(numSegments <= 0){
            return;
        }

        dim3 block = BLOCKSIZE;
        dim3 grid = numSegments;

        findAndRemoveFromSegmentKernel<T, BLOCKSIZE, ITEMS_PER_THREAD><<<grid, block, 0, stream>>>
            (d_toFind, d_items, numSegments, d_segmentSizes, d_segmentBeginOffsets);
    }
}
}





#endif