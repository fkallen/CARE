#include <gpu/util_gpu.cuh>
#include <gpu/cudaerrorcheck.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

/*
    input:
    d_segmentSizes {2,3,4}
    d_segmentSizesPrefixSum {0,2,5}
    numSegments 3
    numElements 9
    output:
    {0,0,1,1,1,2,2,2,2}
*/

void getSegmentIdsPerElement(
    int* d_output,
    const int* d_segmentSizes, 
    const int* d_segmentSizesPrefixSum, 
    int numSegments, 
    int numElements,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr
){
    auto thrustpolicy = rmm::exec_policy_nosync(stream, mr);

    CUDACHECK(cudaMemsetAsync(d_output, 0, sizeof(int) * numElements, stream));
    //must not scatter for empty segments
    thrust::scatter_if(
        thrustpolicy,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(0) + numSegments, 
        d_segmentSizesPrefixSum,
        thrust::make_transform_iterator(
            d_segmentSizes, 
            [] __host__ __device__ (int i){return i > 0;}
        ),
        d_output
    );

    thrust::inclusive_scan(
        thrustpolicy,
        d_output, 
        d_output + numElements, 
        d_output, 
        thrust::maximum<int>{}
    );
}

rmm::device_uvector<int> getSegmentIdsPerElement(
    const int* d_segmentSizes, 
    const int* d_segmentSizesPrefixSum, 
    int numSegments, 
    int numElements,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr
){
    rmm::device_uvector<int> result(numElements, stream, mr);

    getSegmentIdsPerElement(
        result.data(),
        d_segmentSizes,
        d_segmentSizesPrefixSum,
        numSegments,
        numElements,
        stream,
        mr
    );

    return result;
}