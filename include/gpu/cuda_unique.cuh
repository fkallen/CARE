#ifndef CARE_CUDA_UNIQUE_CUH
#define CARE_CUDA_UNIQUE_CUH

#include <gpu/simpleallocation.cuh>
#include <hpc_helpers.cuh>

#include <memory>

#ifdef __NVCC__

namespace care{
namespace gpu{


namespace cudauniquekernels{

    template<class T, class OffsetIterator>
    __global__
    void checkUniquenessKernel(
        const T* __restrict__ elements_before_unique,
        const T* __restrict__ unique_elements,
        const int* __restrict__ unique_lengths, 
        int numSegments,
        OffsetIterator begin_offsets, //segment i begins at input[d_begin_offsets[i]]
        OffsetIterator end_offsets //segment i ends at input[d_end_offsets[i]] (exclusive)      
    ){
        
        for(int segmentId = blockIdx.x; segmentId < numSegments; segmentId += gridDim.x){
    
            const int segmentBegin = begin_offsets[segmentId];
            const int segmentEnd = end_offsets[segmentId];
            const int sizeOfRange = segmentEnd - segmentBegin;
            const int uniqueSize = unique_lengths[segmentId];

            for(int p = threadIdx.x; p < sizeOfRange; p += blockDim.x){
                
                const T element = elements_before_unique[segmentBegin + p];

                int count = 0;

                for(int i = 0; i < uniqueSize; i++){
                    if(element == unique_elements[segmentBegin + i]){
                        count++;
                    }
                }

                if(count != 1){
                    printf("error segment %d, element %u at original position %d appears %d times,"
                            "sizeOfRange %d, uniqueSize %d\n",
                        segmentId, element, segmentBegin + p, count, sizeOfRange, uniqueSize);
                    assert(false);
                }
            }            
        }    
    }

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

            if(sizeOfRange > elemsPerThread * blocksize){
                continue;
            }

            if(sizeOfRange > 0){
                
                //assert(sizeOfRange <= elemsPerThread * blocksize);            
    
                BlockLoad(temp_storage.load).Load(
                    input + segmentBegin, 
                    tempregs, 
                    sizeOfRange
                );
       
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
    
                //disable out-of-segment elements
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



    template<int blocksize, int elemsPerThread, class T, class OffsetIterator>
    __global__
    void makeUniqueRangeFromSortedRangeKernel(
        T* __restrict__ output,
        int* __restrict__ unique_lengths, 
        const T* __restrict__ input, // each segment must be sorted
        int numSegments,
        OffsetIterator begin_offsets, //segment i begins at input[d_begin_offsets[i]]
        OffsetIterator end_offsets //segment i ends at input[d_end_offsets[i]] (exclusive)  
    ){
    
        using BlockLoad = cub::BlockLoad<T, blocksize, elemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
        using BlockDiscontinuity = cub::BlockDiscontinuity<T, blocksize>;
        using BlockScan = cub::BlockScan<int, blocksize>; 
    
        __shared__ union{
            typename BlockLoad::TempStorage load;
            typename BlockDiscontinuity::TempStorage discontinuity;
            typename BlockScan::TempStorage scan;
        } temp_storage;

        __shared__ T shared_tile_predeccessor_item;
    
        for(int segmentId = blockIdx.x; segmentId < numSegments; segmentId += gridDim.x){
    
            T tempregs[elemsPerThread];   
    
            #pragma unroll
            for(int i = 0; i < elemsPerThread; i++){
                tempregs[i] = std::numeric_limits<T>::max();
            }
    
            const int segmentBegin = begin_offsets[segmentId];
            const int segmentEnd = end_offsets[segmentId];
            const int sizeOfRange = segmentEnd - segmentBegin;

            if(sizeOfRange > 0){

                constexpr int elemsPerBlock = elemsPerThread * blocksize;

                const T* const segmentInput = input + segmentBegin;
                T* const segmentOutput = output + segmentBegin;

                int remainingElementsToProcess = sizeOfRange;

                int uniqueLengthOfSegment = 0;

                const int numIters = SDIV(sizeOfRange, elemsPerBlock);
                
                for(int iter = 0; iter < numIters; iter++){      
                    
                    const int threadElementsOffset = iter * elemsPerBlock + threadIdx.x * elemsPerThread;
    
                    BlockLoad(temp_storage.load).Load(
                        segmentInput + iter * elemsPerBlock, 
                        tempregs, 
                        remainingElementsToProcess
                    );

                    if(iter == 0 && threadIdx.x == 0){
                        shared_tile_predeccessor_item = std::numeric_limits<T>::max();
                    }

                    __syncthreads();
            
                    int head_flags[elemsPerThread];
        
                    BlockDiscontinuity(temp_storage.discontinuity).FlagHeads(
                        head_flags, 
                        tempregs, 
                        cub::Inequality(),
                        shared_tile_predeccessor_item
                    );
        
                    __syncthreads();            
        
                    //disable out-of-segment elements
                    #pragma unroll
                    for(int i = 0; i < elemsPerThread; i++){
                        if(threadElementsOffset + i >= sizeOfRange){
                            head_flags[i] = 0;
                        }
                    }
        
                    int prefixsum[elemsPerThread];
                    int numberOfSetHeadFlags = 0;
        
                    BlockScan(temp_storage.scan).ExclusiveSum(head_flags, prefixsum, numberOfSetHeadFlags);
        
                    __syncthreads();
        
                    #pragma unroll
                    for(int i = 0; i < elemsPerThread; i++){
                        if(threadElementsOffset + i < sizeOfRange && head_flags[i] == 1){
                            segmentOutput[uniqueLengthOfSegment + prefixsum[i]] = tempregs[i];
                        }
                    }
                   
                    //set shared_tile_predeccessor_item for next iteration
                    const int threadOfLastItem = (min(remainingElementsToProcess, elemsPerBlock)-1) / elemsPerThread;
                    const int elemIndex = (min(remainingElementsToProcess, elemsPerBlock)-1) % elemsPerThread;

                    if(threadOfLastItem == threadIdx.x){
                        shared_tile_predeccessor_item = tempregs[elemIndex];
                    }

                    uniqueLengthOfSegment += numberOfSetHeadFlags;
                    remainingElementsToProcess -= elemsPerBlock;

                }

                if(threadIdx.x == 0){
                    unique_lengths[segmentId] = uniqueLengthOfSegment;
                }
            }
        }
    
    }


    //set end offset to begin offset for each segment smaller than minimumSegmentSize
    template<class OffsetIterator>
    __global__
    void setSmallSegmentSizesToZeroKernel(
        int* __restrict__ output_end_offsets,
        int numSegments,
        OffsetIterator input_begin_offsets,
        OffsetIterator input_end_offsets,
        int minimumSegmentSize
    ){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        for(int i = tid; i < numSegments; i += stride){
            const int begin = input_begin_offsets[i];
            const int end = input_end_offsets[i];
            const int segmentSize = end - begin;

            if(segmentSize < minimumSegmentSize){
                output_end_offsets[i] = 0;
            }else{
                output_end_offsets[i] = end;
            }
        }
    }



}



struct GpuSegmentedUnique{
    struct HandleData{
        int gpu;
        SimpleAllocationDevice<char> d_temp_storage;

        HandleData(){
            cudaGetDevice(&gpu); CUERR;
        }

        HandleData(int gpu): gpu(gpu){
        }

        ~HandleData(){
            int curgpu;
            cudaGetDevice(&curgpu); CUERR;
            cudaSetDevice(gpu); CUERR;

            d_temp_storage.destroy();

            cudaSetDevice(curgpu); CUERR;
        }
    };

    using Handle = std::shared_ptr<HandleData>;

    static Handle makeHandle(){
        return std::make_shared<HandleData>();
    }

    static Handle makeHandle(int gpu){
        return std::make_shared<HandleData>(gpu);
    }

    // removes duplicates in each segment, writes results to d_unique_items
    template<class T>
    static void unique(
        Handle& handle,
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

        if(sizeOfLargestSegment == 0){
            return;
        }

        constexpr int maximumSegmentSizeForRegSort = 128 * 64;

        //cudaMemsetAsync(d_unique_lengths, 0, sizeof(int) * numSegments, stream);

        //select segments large than limit
        // cudauniquekernels::setSmallSegmentSizesToZeroKernel<<<SDIV(numSegments, 128), 128, 0, stream>>>(
        //     handle->end_offsets_tmp.get(),
        //     numSegments,
        //     d_begin_offsets,
        //     d_end_offsets,
        //     maximumSegmentSizeForRegSort+1
        // )

        if(sizeOfLargestSegment <= maximumSegmentSizeForRegSort){

            makeUniqueRangeWithRegSort(
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

        }else{
            std::size_t temp_storage_bytes = 0;

            makeUniqueRangeWithGmemSort(
                nullptr,
                temp_storage_bytes,
                d_unique_items,
                d_unique_lengths, 
                d_items,
                numItems,
                numSegments,
                d_begin_offsets, 
                d_end_offsets, 
                sizeOfLargestSegment,
                begin_bit,
                end_bit,
                stream
            );

            handle->d_temp_storage.resize(temp_storage_bytes);

            makeUniqueRangeWithGmemSort(
                handle->d_temp_storage.get(),
                temp_storage_bytes,
                d_unique_items,
                d_unique_lengths, 
                d_items,
                numItems,
                numSegments,
                d_begin_offsets, 
                d_end_offsets, 
                sizeOfLargestSegment,
                begin_bit,
                end_bit,
                stream
            );
        }

        cudauniquekernels::checkUniquenessKernel<<<numSegments, 128, 0, stream>>>(
            d_items,
            d_unique_items,
            d_unique_lengths, 
            numSegments,
            d_begin_offsets,
            d_end_offsets
        );
    }

    template<class T, class OffsetIterator>
    static void makeUniqueRangeWithRegSort(
        T* d_output,
        int* d_unique_lengths, 
        const T* d_input,
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

    template<class T, class OffsetIterator>
    static void makeUniqueRangeWithGmemSort(
        void* d_temp_storage,
        std::size_t& temp_storage_bytes,
        T* d_output,
        int* d_unique_lengths, 
        const T* d_input,
        int numItems,
        int numSegments,
        OffsetIterator d_begin_offsets, //segment i begins at input[d_begin_offsets[i]]
        OffsetIterator d_end_offsets, //segment i ends at input[d_end_offsets[i]] (exclusive)
        int sizeOfLargestSegment,
        int begin_bit = 0,
        int end_bit = sizeof(T) * 8,
        cudaStream_t stream = 0
    ){

        
        std::size_t requiredCubTempStorageSize = 0;
        cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr,
            requiredCubTempStorageSize,
            d_input, //input
            d_output, //output
            numItems,
            numSegments,
            d_begin_offsets,
            d_end_offsets,
            begin_bit,
            end_bit,
            stream
        );

        std::size_t requiredTempStorageSize = requiredCubTempStorageSize;
        requiredTempStorageSize += 512; //padding
        requiredTempStorageSize += sizeof(T) * numItems; //double buffer for sorting

        if(d_temp_storage == nullptr){
            temp_storage_bytes = requiredTempStorageSize;
            return;
        }

        assert(temp_storage_bytes >= requiredTempStorageSize);

        void* d_cub_temp = d_temp_storage;
        const std::size_t offset = SDIV(requiredCubTempStorageSize, 512) * 512;
        T* d_item_temp = (T*)(((char*)d_cub_temp) + offset);

        constexpr int blocksize = 128;
        constexpr int elemsPerThread = 4;

        cub::DeviceSegmentedRadixSort::SortKeys(
            d_cub_temp,
            temp_storage_bytes,
            d_input, //input
            d_item_temp, //output
            numItems,
            numSegments,
            d_begin_offsets,
            d_end_offsets,
            begin_bit,
            end_bit,
            stream
        );

        // generic_kernel<<<1,1,0,stream>>>([=]__device__(){
        //     printf("after sort\n");
        //     for(int i = d_begin_offsets[0]; i < d_end_offsets[0]; i++){
        //         printf("%d %u\n", i, d_item_temp[i]);
        //     }
        // });

        cudauniquekernels::makeUniqueRangeFromSortedRangeKernel<blocksize, elemsPerThread>
                <<<numSegments, blocksize, 0, stream>>>(
            d_output, //output
            d_unique_lengths, 
            d_item_temp, //input
            numSegments,
            d_begin_offsets,
            d_end_offsets 
        );
    }

};





}
}


#endif // __NVCC__


#endif
