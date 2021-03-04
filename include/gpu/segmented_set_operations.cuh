#ifndef SEGMENTED_SET_OPERATIONS_CUH
#define SEGMENTED_SET_OPERATIONS_CUH



#include <thrust/set_operations.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/device_new_allocator.h>

#include <cub/cub.cuh>

#include <iostream>


template<class T>
__global__ void fillSegmentIdsKernel(
    const int* __restrict__ segmentSizes,
    const int* __restrict__ segmentBeginOffsets,
    int numSegments,
    T* __restrict__ output
){
    for(int seg = blockIdx.x; seg < numSegments; seg += gridDim.x){
        const int offset = segmentBeginOffsets[seg];
        const int size = segmentSizes[seg];

        for(int i = threadIdx.x; i < size; i += blockDim.x){
            output[offset + i] = seg;
        }
    }
}

template<class T>
void callFillSegmentIdsKernel(
    const int* d_segmentSizes,
    const int* d_segmentBeginOffsets,
    int numSegments,
    T* d_output,
    cudaStream_t stream
){
    dim3 block = 128;
    dim3 grid = numSegments;

    fillSegmentIdsKernel<<<grid, block, 0, stream>>>(
        d_segmentSizes,
        d_segmentBeginOffsets,
        numSegments,
        d_output
    ); CUERR;
}

__global__
void setOutputSegmentSizesKernel(
    const int* __restrict__ uniqueIds,
    const int* __restrict__ reducedCounts,
    const int* __restrict__ numUnique,
    int* __restrict__ outputSizes
){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    const int n = *numUnique;

    for(int i = tid; i < n; i += stride){
        outputSizes[uniqueIds[i]] = reducedCounts[i];
    }
}

__global__
void initAndSetOutputSegmentSizesSingleBlockKernel(
    const int* __restrict__ uniqueIds,
    const int* __restrict__ reducedCounts,
    const int* __restrict__ numUnique,
    int* __restrict__ outputSizes,
    int numSegments
){
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int n = *numUnique;

    for(int i = tid; i < numSegments; i += stride){
        outputSizes[i] = 0;
    }

    __syncthreads();

    for(int i = tid; i < n; i += stride){
        outputSizes[uniqueIds[i]] = reducedCounts[i];
    }
}

struct GpuSegmentedSetOperation{

    //result = input1 - input2, per segment
    template<class ThrustAllocator, class T, class SegmentIdIter1, class SegmentIdIter2>
    T* difference(
        ThrustAllocator& allocator,
        const T* d_input1,
        const int* d_segmentSizes1,
        const int* d_segmentBeginOffsets1,
        SegmentIdIter1 d_segmentIds1,
        int numElements1,
        const T* d_input2,
        const int* d_segmentSizes2,
        const int* d_segmentBeginOffsets2,
        SegmentIdIter2 d_segmentIds2,
        int numElements2,
        int numSegments,        
        T* d_output,
        int* d_outputSegmentSizes,
        int* d_outputSegmentIds,
        cudaStream_t stream
    ){
        static_assert(sizeof(typename ThrustAllocator::value_type) == 1, "Allocator for GpuSegmentedSetOperation difference must allocate bytes.");

        auto policy = thrust::cuda::par(allocator).on(stream);

        auto comp = [] __device__ (const auto& t1, const auto& t2){
            const int idl = thrust::get<0>(t1);
            const int idr = thrust::get<0>(t2);

            if(idl < idr) return true;
            if(idl > idr) return false;

            return thrust::get<1>(t1) < thrust::get<1>(t2);
        };

        auto first1 = thrust::make_zip_iterator(thrust::make_tuple(d_segmentIds1, d_input1));
        auto last1 = thrust::make_zip_iterator(thrust::make_tuple(d_segmentIds1 + numElements1, d_input1 + numElements1));

        auto first2 = thrust::make_zip_iterator(thrust::make_tuple(d_segmentIds2, d_input2));
        auto last2 = thrust::make_zip_iterator(thrust::make_tuple(d_segmentIds2 + numElements2, d_input2 + numElements2));

        auto outputZip = thrust::make_zip_iterator(thrust::make_tuple(d_outputSegmentIds, d_output));

        auto outputZipEnd = thrust::set_difference(policy, first1, last1, first2, last2, outputZip, comp);

        int outputsize = thrust::distance(outputZip, outputZipEnd);

        std::size_t cubbytes = 0;

        cudaError_t cubstatus = cub::DeviceRunLengthEncode::Encode(
            nullptr,
            cubbytes,
            (int*) nullptr,
            (int*) nullptr,
            (int*) nullptr,
            (int*) nullptr,
            outputsize,
            stream
        );
        assert(cubstatus == cudaSuccess);

        void* temp_allocations[4];
        std::size_t temp_allocation_sizes[4];
        
        temp_allocation_sizes[0] = sizeof(int) * numSegments;
        temp_allocation_sizes[1] = sizeof(int) * numSegments;
        temp_allocation_sizes[2] = sizeof(int);
        temp_allocation_sizes[3] = cubbytes;
        
        std::size_t temp_storage_bytes = 0;
        cubstatus = cub::AliasTemporaries(
            nullptr,
            temp_storage_bytes,
            temp_allocations,
            temp_allocation_sizes
        );
        assert(cubstatus == cudaSuccess);

        auto tempPtr = allocator.allocate(sizeof(char) * temp_storage_bytes);
        cubstatus = cub::AliasTemporaries(
            (void*)thrust::raw_pointer_cast(tempPtr),
            temp_storage_bytes,
            temp_allocations,
            temp_allocation_sizes
        );
        assert(cubstatus == cudaSuccess);


        int* const uniqueIds = (int*)temp_allocations[0];
        int* const reducedCounts = (int*)temp_allocations[1];        
        int* const numRuns = (int*)temp_allocations[2];
        void* const cubtemp = (void*)temp_allocations[3];
        
        cubstatus = cub::DeviceRunLengthEncode::Encode(
            cubtemp,
            cubbytes,
            d_outputSegmentIds,
            uniqueIds,
            reducedCounts,
            numRuns,
            outputsize,
            stream
        );
        assert(cubstatus == cudaSuccess);

        #if 1

        initAndSetOutputSegmentSizesSingleBlockKernel<<<1, 1024, 0, stream>>>(
            uniqueIds,
            reducedCounts,
            numRuns,
            d_outputSegmentSizes,
            numSegments
        );

        #else

        cudaMemsetAsync(
            d_outputSegmentSizes,
            0,
            sizeof(int) * numSegments,
            stream
        );

        setOutputSegmentSizesKernel<<<SDIV(numSegments, 256), 256, 0, stream>>>(
            uniqueIds,
            reducedCounts,
            numRuns,
            d_outputSegmentSizes
        );

        #endif

        //cudaStreamSynchronize(stream); CUERR;

        allocator.deallocate(tempPtr, sizeof(char) * temp_storage_bytes);        

        return d_output + outputsize;
    }

};




#endif