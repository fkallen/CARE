#ifndef CARE_CUDA_UNIQUE_CUH
#define CARE_CUDA_UNIQUE_CUH

#include <hpc_helpers.cuh>
#include <memorymanagement.hpp>

#include <gpu/cuda_block_select.cuh>
#include <gpu/cuda_block_unique.cuh>
#include <gpu/cudaerrorcheck.cuh>


#include <memory>
#include <cassert>
#include <cmath>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/device_uvector.hpp>
#include <gpu/rmm_utilities.cuh>

#include <thrust/for_each.h>
#include <thrust/gather.h>

//#include <cuda/functional>

namespace care{
namespace gpu{
namespace cudauniquekernels{

    template<class T>
    struct GetElementSum{
        const T* array1;
        const T* array2;

        __host__ __device__
        GetElementSum(const T* a1, const T* a2) : array1(a1), array2(a2){}

        __host__ __device__
        T operator()(int index){
            return array1[index] + array2[index];
        }
    };


    template<int blocksize, int elemsPerThread, class T, class OffsetIterator>
    __launch_bounds__(blocksize)
    __global__
    void makeUniqueRangeWithRegSortKernel(
        T* __restrict__ output,
        int* __restrict__ unique_lengths, 
        const T* __restrict__ input,
        const int* __restrict__ segmentIds,
        int numSegments,
        OffsetIterator begin_offsets,
        OffsetIterator end_offsets,
        int begin_bit = 0,
        int end_bit = sizeof(T) * 8        
    ){
    
        using BlockRadixSort = cub::BlockRadixSort<T, blocksize, elemsPerThread>;
        using BlockLoad = cub::BlockLoad<T, blocksize, elemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
        using BlockDiscontinuity = cub::BlockDiscontinuity<T, blocksize>;
        using MyBlockUnique = BlockUnique<T, blocksize, elemsPerThread>;
    
        __shared__ union{
            typename BlockRadixSort::TempStorage sort;
            typename BlockLoad::TempStorage load;
            typename BlockDiscontinuity::TempStorage discontinuity;
            T elements[elemsPerThread * blocksize];
        } temp_storage1;

        __shared__ union{
            typename MyBlockUnique::TempStorage unique;
        } temp_storage2;
    
        for(int x = blockIdx.x; x < numSegments; x += gridDim.x){
            const int segmentId = segmentIds[x];
            T tempregs[elemsPerThread];   
    
            #pragma unroll
            for(int i = 0; i < elemsPerThread; i++){
                tempregs[i] = std::numeric_limits<T>::max();
            }
    
            const int segmentBegin = begin_offsets[segmentId];
            const int segmentEnd = end_offsets[segmentId];
            const int segmentSize = segmentEnd - segmentBegin;

            if(segmentSize == 0){
                if(threadIdx.x == 0){
                    unique_lengths[segmentId] = 0;
                }
                continue;
            }

            if(segmentSize > 0){
                
                //assert(segmentSize <= elemsPerThread * blocksize);            
    
                BlockLoad(temp_storage1.load).Load(
                    input + segmentBegin, 
                    tempregs, 
                    segmentSize
                );
       
                __syncthreads();
    
                BlockRadixSort(temp_storage1.sort).Sort(tempregs, begin_bit, end_bit);
    
                __syncthreads();

                const int numUnique = MyBlockUnique(temp_storage2.unique).execute(
                    tempregs, 
                    segmentSize, 
                    &temp_storage1.elements[0]
                    //output + segmentBegin
                );
    
                __syncthreads();
                for(int i = threadIdx.x; i < numUnique; i += blocksize){
                    output[segmentBegin + i] = temp_storage1.elements[i];
                }

                if(threadIdx.x == 0){
                    unique_lengths[segmentId] = numUnique;
                }

                __syncthreads();
            }
        }
    
    }

    template<int blocksize, int elemsPerThread, class T, class OffsetIterator>
    void callMakeUniqueRangeWithRegSortKernel(
        T* d_output,
        int* d_unique_lengths, 
        const T* d_input,
        const int* d_segmentIds,
        int numSegments,
        OffsetIterator begin_offsets,
        OffsetIterator end_offsets,
        int begin_bit,
        int end_bit,
        cudaStream_t stream
    ){
        auto kernel = makeUniqueRangeWithRegSortKernel<blocksize, elemsPerThread, T, OffsetIterator>;

        int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            kernel,
            blocksize, 
            0
        ));
    
        const int maxBlocks = maxBlocksPerSM * numSMs;  
    
        dim3 block(blocksize, 1, 1);
        const int numBlocks = numSegments;
        dim3 grid(std::min(numBlocks, maxBlocks), 1, 1);

        kernel<<<grid, block, 0, stream>>>(
            d_output,
            d_unique_lengths, 
            d_input,
            d_segmentIds,
            numSegments,
            begin_offsets,
            end_offsets,
            begin_bit,
            end_bit     
        );
        CUDACHECKASYNC;
    }

    template<int blocksize, int elemsPerThread, class T, class OffsetIterator>
    __launch_bounds__(blocksize)
    __global__
    void makeUniqueRangeFromSortedRangeKernel(
        T* __restrict__ output,
        int* __restrict__ unique_lengths, 
        const T* __restrict__ input, // each segment must be sorted
        const int* __restrict__ segmentIds,
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
            T elements[blocksize * elemsPerThread];
        } temp_storage;

        __shared__ T shared_tile_predeccessor_item;
    
        for(int x = blockIdx.x; x < numSegments; x += gridDim.x){
            const int segmentId = segmentIds[x];
    
            T tempregs[elemsPerThread];   
    
            #pragma unroll
            for(int i = 0; i < elemsPerThread; i++){
                tempregs[i] = std::numeric_limits<T>::max();
            }
    
            const int segmentBegin = begin_offsets[segmentId];
            const int segmentEnd = end_offsets[segmentId];
            const int segmentSize = segmentEnd - segmentBegin;

            if(segmentSize > 0){

                constexpr int elemsPerBlock = elemsPerThread * blocksize;

                const T* const segmentInput = input + segmentBegin;
                T* const segmentOutput = output + segmentBegin;

                int remainingElementsToProcess = segmentSize;

                int uniqueLengthOfSegment = 0;

                const int numIters = SDIV(segmentSize, elemsPerBlock);
                
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
                        if(threadElementsOffset + i >= segmentSize){
                            head_flags[i] = 0;
                        }
                    }
        
                    int prefixsum[elemsPerThread];
                    int numberOfSetHeadFlags = 0;
        
                    BlockScan(temp_storage.scan).ExclusiveSum(head_flags, prefixsum, numberOfSetHeadFlags);
        
                    __syncthreads();
        
                    #pragma unroll
                    for(int i = 0; i < elemsPerThread; i++){
                        if(threadElementsOffset + i < segmentSize && head_flags[i] == 1){
                            temp_storage.elements[prefixsum[i]] = tempregs[i];
                            //segmentOutput[uniqueLengthOfSegment + prefixsum[i]] = tempregs[i];
                        }
                    }
                   
                    __syncthreads();

                    for(int i = threadIdx.x; i < numberOfSetHeadFlags; i += blocksize){
                        segmentOutput[uniqueLengthOfSegment + i] = temp_storage.elements[i];
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
            }else{
                if(threadIdx.x == 0){
                    unique_lengths[segmentId] = 0;
                }
            }
        }
    
    }

    template<int blocksize, int elemsPerThread, class T, class OffsetIterator>
    void callMakeUniqueRangeFromSortedRangeKernel(
        T* d_output,
        int* d_unique_lengths, 
        const T* d_input,
        const int* d_segmentIds,
        int numSegments,
        OffsetIterator begin_offsets,
        OffsetIterator end_offsets,
        cudaStream_t stream
    ){
        auto kernel = makeUniqueRangeFromSortedRangeKernel<blocksize, elemsPerThread, T, OffsetIterator>;

        int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            kernel,
            blocksize, 
            0
        ));
    
        const int maxBlocks = maxBlocksPerSM * numSMs;  
    
        dim3 block(blocksize, 1, 1);
        const int numBlocks = numSegments;
        dim3 grid(std::min(numBlocks, maxBlocks), 1, 1);

        kernel<<<grid, block, 0, stream>>>(
            d_output,
            d_unique_lengths, 
            d_input,
            d_segmentIds,
            numSegments,
            begin_offsets,
            end_offsets
        );
        CUDACHECKASYNC;
    }


} //namespace cudauniquekernels

struct GpuSegmentedUnique{

    template<class T>
    static void unique(
        T* d_items,
        int numItems,
        T* d_unique_items,
        int* d_unique_lengths,
        int numSegments,
        const int* d_begin_offsets,
        const int* d_end_offsets,
        cudaStream_t stream = 0,
        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
    ){
        auto thrustpolicy = rmm::exec_policy_nosync(stream, mr);

        rmm::device_uvector<int> d_segmentLengths(numSegments, stream, mr);
        thrust::for_each(
            thrustpolicy,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(0) + numSegments,
            [
                d_begin_offsets,
                d_end_offsets,
                d_segmentLengths = d_segmentLengths.data()
            ] __device__ (int index){
                d_segmentLengths[index] = d_end_offsets[index] - d_begin_offsets[index];
            }
        );

        const int numSegmentsRounded = SDIV(numSegments, 128) * 128;
        constexpr int numPartitions = 6;
        constexpr int boundaries[numPartitions]{128,256,512,1024,2048, std::numeric_limits<int>::max()};
        // constexpr int numPartitions = 2;
        // constexpr int boundaries[numPartitions]{2048, std::numeric_limits<int>::max()};
        

        rmm::device_uvector<int> d_allSegmentIds(numSegmentsRounded * numPartitions, stream, mr);
        rmm::device_uvector<int> d_partitionSizes(numPartitions, stream, mr);
        CUDACHECK(cudaMemsetAsync(d_partitionSizes.data(), 0, sizeof(int) * numPartitions, stream));

        thrust::for_each(
            thrustpolicy,
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(0) + numSegments,
            [
                numSegmentsRounded = numSegmentsRounded,
                d_allSegmentIds = d_allSegmentIds.data(),
                d_partitionSizes = d_partitionSizes.data(),
                d_segmentLengths = d_segmentLengths.data()
            ] __device__ (int index){
                //constexpr int boundaries[numPartitions]{128,256,512,1024,2048, std::numeric_limits<int>::max()};
                const int length = d_segmentLengths[index];
                for(int i = 0; i < numPartitions; i++){
                    if(length <= boundaries[i]){
                        const int pos = atomicAdd(d_partitionSizes + i, 1);
                        d_allSegmentIds[i * numSegmentsRounded + pos] = index;
                        break;
                    }
                }
            }
        );

        std::vector<int> h_partitionSizes(numPartitions);
        CUDACHECK(cudaMemcpyAsync(
            h_partitionSizes.data(),
            d_partitionSizes.data(),
            sizeof(int) * numPartitions,
            D2H,
            stream
        ));
        CUDACHECK(cudaStreamSynchronize(stream));
        // for(int i = 0; i < numPartitions; i++){
        //     std::cerr << h_partitionSizes[i] << " ";
        // }
        // std::cerr << "\n";

        cub::DoubleBuffer<T> d_items_dblbuf{d_items, d_unique_items};

        if(h_partitionSizes[numPartitions - 1] > 0){
            const int numLargestSegments = h_partitionSizes[numPartitions - 1];
            const int* d_largestSegmentIds = d_allSegmentIds.data() + numSegmentsRounded * (numPartitions - 1);
            rmm::device_uvector<int> d_largeSegmentLengths(numLargestSegments, stream, mr);
            rmm::device_uvector<int> d_largeSegmentBeginOffsets(numLargestSegments + 1, stream, mr);
            auto d_largeSegmentEndOffsets = thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                // cuda::proclaim_return_type<int>([
                //     d_largeSegmentLengths = d_largeSegmentLengths.data(),
                //     d_largeSegmentBeginOffsets = d_largeSegmentBeginOffsets.data()
                // ] __device__ (int index){
                //     return d_largeSegmentLengths[index] + d_largeSegmentBeginOffsets[index];
                // })
                cudauniquekernels::GetElementSum<int>{d_largeSegmentLengths.data(), d_largeSegmentBeginOffsets.data()}
            );

            thrust::gather(
                thrustpolicy,
                d_largestSegmentIds,
                d_largestSegmentIds + numLargestSegments,
                d_segmentLengths.begin(),
                d_largeSegmentLengths.begin()
            );
            thrust::gather(
                thrustpolicy,
                d_largestSegmentIds,
                d_largestSegmentIds + numLargestSegments,
                d_begin_offsets,
                d_largeSegmentBeginOffsets.begin()
            );

            std::size_t cubTempSize = 0;
            CUDACHECK(cub::DeviceSegmentedSort::SortKeys(
                nullptr,
                cubTempSize,
                d_items_dblbuf,
                numItems,
                numLargestSegments,
                d_largeSegmentBeginOffsets.begin(),
                d_largeSegmentEndOffsets,
                stream
            ));

            rmm::device_uvector<char> d_cubtemp(cubTempSize, stream, mr);

            CUDACHECK(cub::DeviceSegmentedSort::SortKeys(
                d_cubtemp.data(),
                cubTempSize,
                d_items_dblbuf,
                numItems,
                numLargestSegments,
                d_largeSegmentBeginOffsets.begin(),
                d_largeSegmentEndOffsets,
                stream
            ));

            T* const output = d_unique_items; //if Current() == d_unique_items, operations will be inplace

            cudauniquekernels::callMakeUniqueRangeFromSortedRangeKernel<128, 8>(
                output,
                d_unique_lengths, 
                d_items_dblbuf.Current(), //input
                d_largestSegmentIds,
                numLargestSegments,
                d_begin_offsets,
                d_end_offsets,
                stream
            );
        }

        #define runSmallSegment(partitionIndex, blocksize_, itemsPerThread_) { \
            const std::string rangename = "small segments <= " + std::to_string(boundaries[partitionIndex]); \
            nvtx::push_range(rangename, 1); \
            if(h_partitionSizes[partitionIndex] > 0){ \
                constexpr int blocksize = blocksize_; \
                constexpr int itemsPerThread = itemsPerThread_; \
                static_assert(itemsPerThread > 0); \
                cudauniquekernels::callMakeUniqueRangeWithRegSortKernel<blocksize,itemsPerThread>( \
                    d_unique_items, \
                    d_unique_lengths,  \
                    d_items, \
                    d_allSegmentIds.data() + numSegmentsRounded * partitionIndex, \
                    h_partitionSizes[partitionIndex], \
                    d_begin_offsets, \
                    d_end_offsets, \
                    0, \
                    sizeof(T) * 8, \
                    stream \
                ); \
            } \
            nvtx::pop_range(); \
        }

        runSmallSegment(0, 128, 1)
        runSmallSegment(1, 128, 2)
        runSmallSegment(2, 128, 4)
        runSmallSegment(3, 128, 8)
        runSmallSegment(4, 128, 16)

        #undef runSmallSegment
    }

// (
//     std::vector<care::read_number *>, 
//     const std::vector<int>, 
//     std::vector<care::read_number *>, 
//     std::vector<int *>, 
//     const std::vector<int, 
//     std::vector<int *>, 
//     std::vector<int *>, 
//     const std::vector<cudaStream_t>, 
//     const std::vector<int>, 
//     int *)

    template<class T>
    static void unique(
        const std::vector<T*>& vec_d_items,
        const std::vector<int>& vec_numItems,
        const std::vector<T*>& vec_d_unique_items,
        const std::vector<int*>& vec_d_unique_lengths,
        const std::vector<int>& vec_numSegments,
        const std::vector<const int*>& vec_d_begin_offsets,
        const std::vector<const int*>& vec_d_end_offsets,
        const std::vector<cudaStream_t> streams,
        const std::vector<int>& deviceIds,
        int* h_tempstorage // numPartitions * deviceIds.size()
    ){
        constexpr int numPartitions = 6;
        constexpr int boundaries[numPartitions]{128,256,512,1024,2048, std::numeric_limits<int>::max()};
        int oldDeviceId;
        CUDACHECK(cudaGetDevice(&oldDeviceId));

        const int numGpus = deviceIds.size();

        std::vector<rmm::device_uvector<char>> vec_d_buffers;
        std::vector<int*> vec_d_segmentLengths(numGpus, nullptr);
        std::vector<int*> vec_d_allSegmentIds(numGpus, nullptr);
        std::vector<int*> vec_d_partitionSizes(numGpus, nullptr);

        int* const h_partitionsSizesPerGpu = h_tempstorage;

        for(int g = 0; g < numGpus; g++){
            CUDACHECK(cudaSetDevice(deviceIds[g]));
            const int numSegments = vec_numSegments[g];
            const int numSegmentsRounded = SDIV(numSegments, 128) * 128;

            size_t bytes[3]{
                SDIV(sizeof(int) * numSegments, 512) * 512, // d_segmentLengths
                SDIV(sizeof(int) * numSegmentsRounded * numPartitions, 512) * 512, // d_allSegmentIds
                SDIV(sizeof(int) * numPartitions, 512) * 512 // d_partitionSizes
            };
            const size_t totalBytes = bytes[0] + bytes[1] + bytes[2];

            vec_d_buffers.emplace_back(totalBytes, streams[g]);
            vec_d_segmentLengths[g] = reinterpret_cast<int*>(vec_d_buffers.back().data());
            vec_d_allSegmentIds[g] = reinterpret_cast<int*>(vec_d_buffers.back().data() + bytes[0]);
            vec_d_partitionSizes[g] = reinterpret_cast<int*>(vec_d_buffers.back().data() + bytes[0] + bytes[1]);

            thrust::for_each(
                rmm::exec_policy_nosync(streams[g]),
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(0) + numSegments,
                [
                    d_begin_offsets = vec_d_begin_offsets[g],
                    d_end_offsets = vec_d_end_offsets[g],
                    d_segmentLengths = vec_d_segmentLengths[g]
                ] __device__ (int index){
                    d_segmentLengths[index] = d_end_offsets[index] - d_begin_offsets[index];
                }
            );
        }

        for(int g = 0; g < numGpus; g++){
            CUDACHECK(cudaSetDevice(deviceIds[g]));
            auto thrustpolicy = rmm::exec_policy_nosync(streams[g]);
            const int numSegments = vec_numSegments[g];
            const int numSegmentsRounded = SDIV(numSegments, 128) * 128;

            CUDACHECK(cudaMemsetAsync(vec_d_partitionSizes[g], 0, sizeof(int) * numPartitions, streams[g]));

            thrust::for_each(
                thrustpolicy,
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(0) + numSegments,
                [
                    numSegmentsRounded = numSegmentsRounded,
                    d_allSegmentIds = vec_d_allSegmentIds[g],
                    d_partitionSizes = vec_d_partitionSizes[g],
                    d_segmentLengths = vec_d_segmentLengths[g]
                ] __device__ (int index){
                    //constexpr int boundaries[numPartitions]{128,256,512,1024,2048, std::numeric_limits<int>::max()};
                    const int length = d_segmentLengths[index];
                    for(int i = 0; i < numPartitions; i++){
                        if(length <= boundaries[i]){
                            const int pos = atomicAdd(d_partitionSizes + i, 1);
                            d_allSegmentIds[i * numSegmentsRounded + pos] = index;
                            break;
                        }
                    }
                }
            );
        }
        for(int g = 0; g < numGpus; g++){
            CUDACHECK(cudaSetDevice(deviceIds[g]));
            int* const h_partitionSizes = h_partitionsSizesPerGpu + numPartitions * g;
            CUDACHECK(cudaMemcpyAsync(
                h_partitionSizes,
                vec_d_partitionSizes[g],
                sizeof(int) * numPartitions,
                D2H,
                streams[g]
            ));
        }

        std::vector<rmm::device_uvector<char>> vec_d_tempBuffersAfterSync;
        std::vector<int*> vec_d_largeSegmentLengths(numGpus, nullptr);
        std::vector<int*> vec_d_largeSegmentBeginOffsets(numGpus, nullptr);
        std::vector<void*> vec_d_cubSortTemp(numGpus, nullptr);
        std::vector<size_t> vec_d_cubSortTempBytes(numGpus, 0);

        for(int g = 0; g < numGpus; g++){
            CUDACHECK(cudaSetDevice(deviceIds[g]));
            CUDACHECK(cudaStreamSynchronize(streams[g]));

            auto thrustpolicy = rmm::exec_policy_nosync(streams[g]);
            int* const h_partitionSizes = h_partitionsSizesPerGpu + numPartitions * g;
            const int numSegments = vec_numSegments[g];
            const int numSegmentsRounded = SDIV(numSegments, 128) * 128;

            cub::DoubleBuffer<T> d_items_dblbuf{vec_d_items[g], vec_d_unique_items[g]};

            if(h_partitionSizes[numPartitions - 1] > 0){
                const int numLargestSegments = h_partitionSizes[numPartitions - 1];
                const int* d_largestSegmentIds = vec_d_allSegmentIds[g] + numSegmentsRounded * (numPartitions - 1);

                void* allocations[3]{};
                std::size_t allocation_sizes[3]{};
                std::size_t storage_bytes = 0;

                allocation_sizes[0] = sizeof(int) * numLargestSegments; // d_largeSegmentLengths
                allocation_sizes[1] = sizeof(int) * (numLargestSegments + 1); // d_largeSegmentBeginOffsets

                CUDACHECK(cub::DeviceSegmentedSort::SortKeys(
                    nullptr,
                    allocation_sizes[2],
                    d_items_dblbuf,
                    vec_numItems[g],
                    numLargestSegments,
                    (int*)nullptr,
                    (int*)nullptr,
                    streams[g]
                ));

                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    storage_bytes,
                    allocations,
                    allocation_sizes
                ));
                vec_d_tempBuffersAfterSync.emplace_back(storage_bytes, streams[g]);
                CUDACHECK(cub::AliasTemporaries(
                    vec_d_tempBuffersAfterSync.back().data(),
                    storage_bytes,
                    allocations,
                    allocation_sizes
                ));

                vec_d_largeSegmentLengths[g] = reinterpret_cast<int*>(allocations[0]);
                vec_d_largeSegmentBeginOffsets[g] = reinterpret_cast<int*>(allocations[1]);
                vec_d_cubSortTemp[g] = reinterpret_cast<void*>(allocations[2]);
                vec_d_cubSortTempBytes[g] = allocation_sizes[2];

                thrust::gather(
                    thrustpolicy,
                    d_largestSegmentIds,
                    d_largestSegmentIds + numLargestSegments,
                    thrust::make_zip_iterator(
                        vec_d_segmentLengths[g],
                        vec_d_begin_offsets[g]
                    ),
                    thrust::make_zip_iterator(
                        vec_d_largeSegmentLengths[g],
                        vec_d_largeSegmentBeginOffsets[g]
                    )
                );
            }else{
                vec_d_tempBuffersAfterSync.emplace_back(0, streams[g]);
            }
        }

        for(int g = 0; g < numGpus; g++){
            CUDACHECK(cudaSetDevice(deviceIds[g]));
            int* const h_partitionSizes = h_partitionsSizesPerGpu + numPartitions * g;

            cub::DoubleBuffer<T> d_items_dblbuf{vec_d_items[g], vec_d_unique_items[g]};

            if(h_partitionSizes[numPartitions - 1] > 0){
                const int numLargestSegments = h_partitionSizes[numPartitions - 1];

                auto d_largeSegmentEndOffsets = thrust::make_transform_iterator(
                    thrust::make_counting_iterator(0),
                    // cuda::proclaim_return_type<int>([
                    //     d_largeSegmentLengths = vec_d_largeSegmentLengths[g],
                    //     d_largeSegmentBeginOffsets = vec_d_largeSegmentBeginOffsets[g]
                    // ] __device__ (int index){
                    //     return d_largeSegmentLengths[index] + d_largeSegmentBeginOffsets[index];
                    // })
                    cudauniquekernels::GetElementSum<int>{vec_d_largeSegmentLengths[g], vec_d_largeSegmentBeginOffsets[g]}
                );

                CUDACHECK(cub::DeviceSegmentedSort::SortKeys(
                    vec_d_cubSortTemp[g],
                    vec_d_cubSortTempBytes[g],
                    d_items_dblbuf,
                    vec_numItems[g],
                    numLargestSegments,
                    vec_d_largeSegmentBeginOffsets[g],
                    d_largeSegmentEndOffsets,
                    streams[g]
                ));
            }
        }
        for(int g = 0; g < numGpus; g++){
            CUDACHECK(cudaSetDevice(deviceIds[g]));

            int* const h_partitionSizes = h_partitionsSizesPerGpu + numPartitions * g;
            const int numSegments = vec_numSegments[g];
            const int numSegmentsRounded = SDIV(numSegments, 128) * 128;
            cub::DoubleBuffer<T> d_items_dblbuf{vec_d_items[g], vec_d_unique_items[g]};

            if(h_partitionSizes[numPartitions - 1] > 0){
                const int numLargestSegments = h_partitionSizes[numPartitions - 1];
                const int* d_largestSegmentIds = vec_d_allSegmentIds[g] + numSegmentsRounded * (numPartitions - 1);

                T* const output = vec_d_unique_items[g]; //if Current() == d_unique_items, operations will be inplace

                cudauniquekernels::callMakeUniqueRangeFromSortedRangeKernel<128, 8>(
                    output,
                    vec_d_unique_lengths[g], 
                    d_items_dblbuf.Current(), //input
                    d_largestSegmentIds,
                    numLargestSegments,
                    vec_d_begin_offsets[g],
                    vec_d_end_offsets[g],
                    streams[g]
                );
            }
        }

        #define runSmallSegment(partitionIndex, blocksize_, itemsPerThread_) { \
            for(int g = 0; g < numGpus; g++){ \
                CUDACHECK(cudaSetDevice(deviceIds[g])); \
                const std::string rangename = "small segments <= " + std::to_string(boundaries[partitionIndex]); \
                int* const h_partitionSizes = h_partitionsSizesPerGpu + numPartitions * g; \
                if(h_partitionSizes[partitionIndex] > 0){ \
                    const int numSegments = vec_numSegments[g]; \
                    const int numSegmentsRounded = SDIV(numSegments, 128) * 128; \
                    constexpr int blocksize = blocksize_; \
                    constexpr int itemsPerThread = itemsPerThread_; \
                    static_assert(itemsPerThread > 0); \
                    cudauniquekernels::callMakeUniqueRangeWithRegSortKernel<blocksize,itemsPerThread>( \
                        vec_d_unique_items[g], \
                        vec_d_unique_lengths[g],  \
                        vec_d_items[g], \
                        vec_d_allSegmentIds[g] + numSegmentsRounded * partitionIndex, \
                        h_partitionSizes[partitionIndex], \
                        vec_d_begin_offsets[g], \
                        vec_d_end_offsets[g], \
                        0, \
                        sizeof(T) * 8, \
                        streams[g] \
                    ); \
                } \
            } \
        }

        runSmallSegment(0, 128, 1)
        runSmallSegment(1, 128, 2)
        runSmallSegment(2, 128, 4)
        runSmallSegment(3, 128, 8)
        runSmallSegment(4, 128, 16)

        #undef runSmallSegment

        for(int g = 0; g < numGpus; g++){
            CUDACHECK(cudaSetDevice(deviceIds[g]));
            vec_d_buffers[g].release();
            vec_d_tempBuffersAfterSync[g].release();
        }
        
        CUDACHECK(cudaSetDevice(oldDeviceId));
    }

};





}
}

#endif
