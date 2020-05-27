#ifndef CARE_CUDA_UNIQUE_CUH
#define CARE_CUDA_UNIQUE_CUH

#ifdef __NVCC__

namespace care{
namespace gpu{


namespace cudauniquekernels{


    template<int blocksize, int elemsPerThread, class T, class OffsetIterator>
    __global__
    void makeUniqueRangeWithRegSortKernel(
        T* __restrict__ output,
        int* __restrict__ unique_lengths, 
        const T* __restrict__ input,
        int numSegments,
        OffsetIterator begin_offsets, //segment i begins at input[d_begin_offsets[i]]
        OffsetIterator end_offsets, //segment i ends at input[d_end_offsets[i]] (exclusive)
        int begin_bit = 0,
        int end_bit = sizeof(T) * 8        
    ){
    
        using BlockRadixSort = cub::BlockRadixSort<T, blocksize, elemsPerThread>;
        using BlockLoad = cub::BlockLoad<T, blocksize, elemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
        using BlockDiscontinuity = cub::BlockDiscontinuity<T, blocksize>;
        using BlockScan = cub::BlockScan<int, blocksize>; 
    
        __shared__ union{
            typename BlockRadixSort::TempStorage sort;
            typename BlockLoad::TempStorage load;
            typename BlockDiscontinuity::TempStorage discontinuity;
            typename BlockScan::TempStorage scan;
        } temp_storage;
    
        for(int segmentId = blockIdx.x; segmentId < numSegments; segmentId += gridDim.x){
    
            T tempregs[elemsPerThread];   
    
            #pragma unroll
            for(int i = 0; i < elemsPerThread; i++){
                tempregs[i] = std::numeric_limits<T>::max();
            }
    
            const int segmentBegin = begin_offsets[segmentId];
            const int segmentEnd = end_offsets[segmentId];
            const int sizeOfRange = segmentEnd - segmentBegin;

            if(sizeOfRange == 0){
                if(threadIdx.x == 0){
                    unique_lengths[segmentId] = 0;
                }
            }else{
                
                assert(sizeOfRange <= elemsPerThread * blocksize);            
    
                BlockLoad(temp_storage.load).Load(
                    input + segmentBegin, 
                    tempregs, 
                    sizeOfRange
                );

                // if(segmentId == 11){
                //     if(threadIdx.x == 0){
                //         printf("segmentBegin %d, segmentEnd %d\n", segmentBegin, segmentEnd);
                //     }
                //     for(int i = 0; i < blockDim.x; i++){
                //         if(threadIdx.x == i){
                //             for(int k = 0; k < elemsPerThread; k++){
                //                 printf("elem %d: %u\n", threadIdx.x * elemsPerThread + k, tempregs[k]);
                //             }
                //         }
                //         __syncthreads();
                //     }
                // }
        
                __syncthreads();
    
                BlockRadixSort(temp_storage.sort).Sort(tempregs, begin_bit, end_bit);
    
                __syncthreads();
    
                int head_flags[elemsPerThread];
    
                BlockDiscontinuity(temp_storage.discontinuity).FlagHeads(
                    head_flags, 
                    tempregs, 
                    cub::Inequality()
                );
    
                __syncthreads();            
    
                //disable out of bounds elements
                #pragma unroll
                for(int i = 0; i < elemsPerThread; i++){
                    if(threadIdx.x * elemsPerThread + i >= sizeOfRange){
                        head_flags[i] = 0;
                    }
                }
    
                int prefixsum[elemsPerThread];
                int numberOfSetHeadFlags = 0;
    
                BlockScan(temp_storage.scan).ExclusiveSum(head_flags, prefixsum, numberOfSetHeadFlags);
    
                __syncthreads();
    
                #pragma unroll
                for(int i = 0; i < elemsPerThread; i++){
                    if(threadIdx.x * elemsPerThread + i < sizeOfRange && head_flags[i] == 1){
                        output[segmentBegin + prefixsum[i]] = tempregs[i];
                    }
                }
    
                if(threadIdx.x == 0){
                    unique_lengths[segmentId] = numberOfSetHeadFlags;
                }
            }
        }
    
    }



}



struct GpuSegmentedUnique{
    struct HandleData{

    };

    template<class T>
    static void unique(
        //HandleData* handle,
        const T* d_items,
        int numItems,
        T* d_unique_items,
        int* d_unique_lengths,
        int numSegments,
        int* d_begin_offsets,
        int* d_end_offsets,
        int* h_begin_offsets,
        int* h_end_offsets,
        int begin_bit = 0,
        int end_bit = sizeof(T) * 8,
        cudaStream_t stream = 0
    ){

        int sizeOfLargestSegment = 0;
        for(int i = 0; i < numSegments; i++){
            const int segmentSize = h_end_offsets[i] - h_begin_offsets[i];
            sizeOfLargestSegment = std::max(segmentSize, sizeOfLargestSegment);
        }

        callMakeUniqueRangeWithRegSortKernel(
            d_unique_items,
            d_unique_lengths, 
            d_items,
            numSegments,
            d_begin_offsets, 
            d_end_offsets, 
            sizeOfLargestSegment,
            begin_bit,
            end_bit,
            stream
        );
    }

    template<class T, class OffsetIterator>
    static void callMakeUniqueRangeWithRegSortKernel(
        T* d_output,
        int* d_unique_lengths, 
        const T* __restrict__ d_input,
        int numSegments,
        OffsetIterator d_begin_offsets, //segment i begins at input[d_begin_offsets[i]]
        OffsetIterator d_end_offsets, //segment i ends at input[d_end_offsets[i]] (exclusive)
        int sizeOfLargestSegment,
        int begin_bit = 0,
        int end_bit = sizeof(T) * 8,
        cudaStream_t stream = 0
    ){
        #define processData(blocksize, elemsPerThread) \
            cudauniquekernels::makeUniqueRangeWithRegSortKernel<(blocksize), (elemsPerThread)> \
                <<<numSegments, (blocksize), 0, stream>>>( \
                d_output,  \
                d_unique_lengths, \
                d_input,  \
                numSegments, \
                d_begin_offsets, \
                d_end_offsets, \
                begin_bit, \
                end_bit \
            ); CUERR;

        if(sizeOfLargestSegment <= 32){
            constexpr int blocksize = 32;
            constexpr int elemsPerThread = 1;
            assert(sizeOfLargestSegment <= blocksize * elemsPerThread);

            processData(blocksize, elemsPerThread);
        }else if(sizeOfLargestSegment <= 64){
            constexpr int blocksize = 64;
            constexpr int elemsPerThread = 1;
            assert(sizeOfLargestSegment <= blocksize * elemsPerThread);

            processData(blocksize, elemsPerThread);
        }else if(sizeOfLargestSegment <= 96){
            constexpr int blocksize = 96;
            constexpr int elemsPerThread = 1;
            assert(sizeOfLargestSegment <= blocksize * elemsPerThread);

            processData(blocksize, elemsPerThread);
        }else if(sizeOfLargestSegment <= 128){
            constexpr int blocksize = 128;
            constexpr int elemsPerThread = 1;
            assert(sizeOfLargestSegment <= blocksize * elemsPerThread);

            processData(blocksize, elemsPerThread);
        }else if(sizeOfLargestSegment <= 256){
            constexpr int blocksize = 128;
            constexpr int elemsPerThread = 2;
            assert(sizeOfLargestSegment <= blocksize * elemsPerThread);

            processData(blocksize, elemsPerThread);
        }else if(sizeOfLargestSegment <= 512){
            constexpr int blocksize = 128;
            constexpr int elemsPerThread = 4;
            assert(sizeOfLargestSegment <= blocksize * elemsPerThread);

            processData(blocksize, elemsPerThread);
        }else if(sizeOfLargestSegment <= 1024){
            constexpr int blocksize = 128;
            constexpr int elemsPerThread = 8;
            assert(sizeOfLargestSegment <= blocksize * elemsPerThread);

            processData(blocksize, elemsPerThread);
        }else if(sizeOfLargestSegment <= 2048){
            constexpr int blocksize = 128;
            constexpr int elemsPerThread = 16;
            assert(sizeOfLargestSegment <= blocksize * elemsPerThread);

            processData(blocksize, elemsPerThread);
        }else if(sizeOfLargestSegment <= 4096){
            constexpr int blocksize = 128;
            constexpr int elemsPerThread = 32;
            assert(sizeOfLargestSegment <= blocksize * elemsPerThread);

            processData(blocksize, elemsPerThread);
        }else{
            constexpr int blocksize = 128;
            constexpr int elemsPerThread = 64;


            if(sizeOfLargestSegment <= blocksize * elemsPerThread){
            
                processData(blocksize, elemsPerThread);

            }else{
                
                std::cerr << sizeOfLargestSegment << " > " << (blocksize * elemsPerThread) << " , cannot use callMakeUniqueRangeWithRegSortKernel \n";
                assert(false);
            }
        }
    }



};





}
}


#endif // __NVCC__


#endif
