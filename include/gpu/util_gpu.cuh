#ifndef CARE_UTIL_GPU_CUH
#define CARE_UTIL_GPU_CUH

#include <rmm/device_uvector.hpp>

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
);

rmm::device_uvector<int> getSegmentIdsPerElement(
    const int* d_segmentSizes, 
    const int* d_segmentSizesPrefixSum, 
    int numSegments, 
    int numElements,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr
);






#endif