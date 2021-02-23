#ifndef SEGMENTED_SET_OPERATIONS_CUH
#define SEGMENTED_SET_OPERATIONS_CUH



#include <thrust/set_operations.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/device_new_allocator.h>

#include <iostream>


template<class T>
__global__ void fillSegmentIdsKernel(
    const int* segmentSizes,
    const int* segmentBeginOffsets,
    int numSegments,
    T* output
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

        thrust::reduce_by_key(
            policy, 
            d_outputSegmentIds, 
            d_outputSegmentIds + outputsize, 
            thrust::make_constant_iterator(1), 
            thrust::make_discard_iterator(), 
            d_outputSegmentSizes
        );

        cudaStreamSynchronize(stream); CUERR;

        return d_output + outputsize;
    }

    //result = input1 - input2, per segment
    template<class T>
    T* difference(
        const T* d_input1,
        const int* d_segmentSizes1,
        const int* d_segmentBeginOffsets1,
        int numElements1,
        const T* d_input2,
        const int* d_segmentSizes2,
        const int* d_segmentBeginOffsets2,
        int numElements2,
        int numSegments,        
        T* d_output,
        int* d_outputSegmentSizes,
        cudaStream_t stream
    ){
        const int maxOutputSize = numElements1 + numElements2;

        size_t paddedsize1 = SDIV(numElements1, 64) * 64;
        size_t paddedsize2 = SDIV(numElements2, 64) * 64;
        int* segmentIds = nullptr;
        cudaMalloc(&segmentIds, sizeof(int) * paddedsize1 + sizeof(int) * paddedsize2 + sizeof(int) * maxOutputSize); CUERR;

        int* const d_segmentIds1 = segmentIds;
        int* const d_segmentIds2 = d_segmentIds1 + paddedsize1;
        int* const d_outputSegmentIds = d_segmentIds2 + paddedsize2;

        dim3 block = 128;
        dim3 grid = numSegments;

        fillSegmentIdsKernel<<<grid, block, 0, stream>>>(
            d_segmentSizes1,
            d_segmentBeginOffsets1,
            numSegments,
            d_segmentIds1
        ); CUERR;

        fillSegmentIdsKernel<<<grid, block, 0, stream>>>(
            d_segmentSizes2,
            d_segmentBeginOffsets2,
            numSegments,
            d_segmentIds2
        ); CUERR;

        thrust::device_new_allocator<char> allocator{};

        T* retVal = difference(
            allocator,
            d_input1,
            d_segmentSizes1,
            d_segmentBeginOffsets1,
            d_segmentIds1,
            numElements1,
            d_input2,
            d_segmentSizes2,
            d_segmentBeginOffsets2,
            d_segmentIds2,
            numElements2,
            numSegments,        
            d_output,
            d_outputSegmentSizes,
            d_outputSegmentIds,
            stream
        );

        cudaFree(segmentIds); CUERR;

        return retVal;
    }
};




#endif