#ifndef SEGMENTED_SET_OPERATIONS_CUH
#define SEGMENTED_SET_OPERATIONS_CUH

#include <gpu/cudaerrorcheck.cuh>
#include <gpu/util_gpu.cuh>

#include <thrust/set_operations.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <cub/cub.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <iostream>


namespace gpusegmentedsetoperationkernels{


template<class dummy=void>
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

template<class dummy=void>
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

}

struct ValueSegmentIdComparator{
    template<class T>
    __host__ __device__
    bool operator()(const T& t1, const T& t2){
        const int idl = thrust::get<0>(t1);
        const int idr = thrust::get<0>(t2);

        if(idl < idr) return true;
        if(idl > idr) return false;

        return thrust::get<1>(t1) < thrust::get<1>(t2);
    };
};

struct GpuSegmentedSetOperation{


    //result = input1 - input2, per segment
    template<class T>
    static T* set_difference(
        const T* d_input1,
        const int* d_segmentSizes1,
        const int* d_segmentBeginOffsets1,
        int numElements1,
        int numSegments1,
        const T* d_input2,
        const int* d_segmentSizes2,
        const int* d_segmentBeginOffsets2,
        int numElements2,
        int numSegments2,        
        T* d_output,
        int* d_outputSegmentSizes,
        int numOutputSegments,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr
    ){

        const int expectedNumOutputSegments = numSegments1;
        if(numOutputSegments != expectedNumOutputSegments) 
            throw std::runtime_error("numOutputSegments != expectedNumOutputSegments");

        const int maxOutputElements = numElements1;

        auto executeSetOperation = [](
            auto& policy, 
            auto first1, 
            auto last1, 
            auto first2, 
            auto last2, 
            auto output, 
            auto comp
        ){
            return thrust::set_difference(policy, first1, last1, first2, last2, output, comp);
        };

        return setOperation_impl(
            d_input1,
            d_segmentSizes1,
            d_segmentBeginOffsets1,
            numElements1,
            numSegments1,
            d_input2,
            d_segmentSizes2,
            d_segmentBeginOffsets2,
            numElements2,
            numSegments2,
            d_output,
            d_outputSegmentSizes,
            expectedNumOutputSegments,
            maxOutputElements,
            executeSetOperation,
            stream,
            mr
        );
    }

    //result = input1 \cap input2, per segment
    template<class T>
    static T* set_intersection(
        const T* d_input1,
        const int* d_segmentSizes1,
        const int* d_segmentBeginOffsets1,
        int numElements1,
        int numSegments1,
        const T* d_input2,
        const int* d_segmentSizes2,
        const int* d_segmentBeginOffsets2,
        int numElements2,
        int numSegments2,        
        T* d_output,
        int* d_outputSegmentSizes,
        int numOutputSegments,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr
    ){

        const int expectedNumOutputSegments = std::max(numSegments1, numSegments2);
        if(numOutputSegments != expectedNumOutputSegments) 
            throw std::runtime_error("numOutputSegments != expectedNumOutputSegments");

        const int maxOutputElements = std::max(numElements1, numElements2);

        auto executeSetOperation = [](
            auto& policy, 
            auto first1, 
            auto last1, 
            auto first2, 
            auto last2, 
            auto output, 
            auto comp
        ){
            return thrust::set_intersection(policy, first1, last1, first2, last2, output, comp);
        };

        return setOperation_impl(
            d_input1,
            d_segmentSizes1,
            d_segmentBeginOffsets1,
            numElements1,
            numSegments1,
            d_input2,
            d_segmentSizes2,
            d_segmentBeginOffsets2,
            numElements2,
            numSegments2,
            d_output,
            d_outputSegmentSizes,
            expectedNumOutputSegments,
            maxOutputElements,
            executeSetOperation,
            stream,
            mr
        );
    }

    //result = (input1 \cup input2) - (input1 \cap input2), per segment
    template<class T>
    static T* set_symmetric_difference(
        const T* d_input1,
        const int* d_segmentSizes1,
        const int* d_segmentBeginOffsets1,
        int numElements1,
        int numSegments1,
        const T* d_input2,
        const int* d_segmentSizes2,
        const int* d_segmentBeginOffsets2,
        int numElements2,
        int numSegments2,        
        T* d_output,
        int* d_outputSegmentSizes,
        int numOutputSegments,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr
    ){

        const int expectedNumOutputSegments = std::max(numSegments1, numSegments2);
        if(numOutputSegments != expectedNumOutputSegments) 
            throw std::runtime_error("numOutputSegments != expectedNumOutputSegments");

        const int maxOutputElements = numElements1 + numElements2;

        auto executeSetOperation = [](
            auto& policy, 
            auto first1, 
            auto last1, 
            auto first2, 
            auto last2, 
            auto output, 
            auto comp
        ){
            return thrust::set_symmetric_difference(policy, first1, last1, first2, last2, output, comp);
        };

        return setOperation_impl(
            d_input1,
            d_segmentSizes1,
            d_segmentBeginOffsets1,
            numElements1,
            numSegments1,
            d_input2,
            d_segmentSizes2,
            d_segmentBeginOffsets2,
            numElements2,
            numSegments2,
            d_output,
            d_outputSegmentSizes,
            expectedNumOutputSegments,
            maxOutputElements,
            executeSetOperation,
            stream,
            mr
        );
    }

    //result = input1 \cup input2, per segment
    template<class T>
    static T* set_union(
        const T* d_input1,
        const int* d_segmentSizes1,
        const int* d_segmentBeginOffsets1,
        int numElements1,
        int numSegments1,
        const T* d_input2,
        const int* d_segmentSizes2,
        const int* d_segmentBeginOffsets2,
        int numElements2,
        int numSegments2,        
        T* d_output,
        int* d_outputSegmentSizes,
        int numOutputSegments,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr
    ){

        const int expectedNumOutputSegments = std::max(numSegments1, numSegments2);
        if(numOutputSegments != expectedNumOutputSegments) 
            throw std::runtime_error("numOutputSegments != expectedNumOutputSegments");

        const int maxOutputElements = numElements1 + numElements2;

        auto executeSetOperation = [](
            auto& policy, 
            auto first1, 
            auto last1, 
            auto first2, 
            auto last2, 
            auto output, 
            auto comp
        ){
            return thrust::set_union(policy, first1, last1, first2, last2, output, comp);
        };

        return setOperation_impl(
            d_input1,
            d_segmentSizes1,
            d_segmentBeginOffsets1,
            numElements1,
            numSegments1,
            d_input2,
            d_segmentSizes2,
            d_segmentBeginOffsets2,
            numElements2,
            numSegments2,
            d_output,
            d_outputSegmentSizes,
            expectedNumOutputSegments,
            maxOutputElements,
            executeSetOperation,
            stream,
            mr
        );
    }

    //result = input1 \cap input2, per segment
    template<class T>
    static T* merge(
        const T* d_input1,
        const int* d_segmentSizes1,
        const int* d_segmentBeginOffsets1,
        int numElements1,
        int numSegments1,
        const T* d_input2,
        const int* d_segmentSizes2,
        const int* d_segmentBeginOffsets2,
        int numElements2,
        int numSegments2,        
        T* d_output,
        int* d_outputSegmentSizes,
        int numOutputSegments,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr
    ){

        const int expectedNumOutputSegments = std::max(numSegments1, numSegments2);
        if(numOutputSegments != expectedNumOutputSegments) 
            throw std::runtime_error("numOutputSegments != expectedNumOutputSegments");

        const int maxOutputElements = numElements1 + numElements2;

        auto executeSetOperation = [](
            auto& policy, 
            auto first1, 
            auto last1, 
            auto first2, 
            auto last2, 
            auto output, 
            auto comp
        ){
            return thrust::merge(policy, first1, last1, first2, last2, output, comp);
        };

        return setOperation_impl(
            d_input1,
            d_segmentSizes1,
            d_segmentBeginOffsets1,
            numElements1,
            numSegments1,
            d_input2,
            d_segmentSizes2,
            d_segmentBeginOffsets2,
            numElements2,
            numSegments2,
            d_output,
            d_outputSegmentSizes,
            expectedNumOutputSegments,
            maxOutputElements,
            executeSetOperation,
            stream,
            mr
        );
    }

private:

    //result = input1 OP input2, per segment
    template<class T, class ThrustSetOpFunc>
    static T* setOperation_impl(
        const T* d_input1,
        const int* d_segmentSizes1,
        const int* d_segmentBeginOffsets1,
        int numElements1,
        int numSegments1,
        const T* d_input2,
        const int* d_segmentSizes2,
        const int* d_segmentBeginOffsets2,
        int numElements2,
        int numSegments2,        
        T* d_output,
        int* d_outputSegmentSizes,
        int numOutputSegments,
        int maxOutputElements,
        ThrustSetOpFunc executeSetOperation,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr
    ){
        rmm::device_uvector<int> d_segmentIds1 = getSegmentIdsPerElement(
            d_segmentSizes1,
            d_segmentBeginOffsets1,
            numSegments1,
            numElements1,
            stream,
            mr
        );

        rmm::device_uvector<int> d_segmentIds2 = getSegmentIdsPerElement(
            d_segmentSizes2,
            d_segmentBeginOffsets2,
            numSegments2,
            numElements2,
            stream,
            mr
        );

        rmm::device_uvector<int> d_outputSegmentIds(maxOutputElements, stream, mr);

        auto policy = rmm::exec_policy_nosync(stream, mr);

        auto first1 = thrust::make_zip_iterator(thrust::make_tuple(d_segmentIds1.data(), d_input1));
        auto last1 = thrust::make_zip_iterator(thrust::make_tuple(d_segmentIds1.data() + numElements1, d_input1 + numElements1));

        auto first2 = thrust::make_zip_iterator(thrust::make_tuple(d_segmentIds2.data(), d_input2));
        auto last2 = thrust::make_zip_iterator(thrust::make_tuple(d_segmentIds2.data() + numElements2, d_input2 + numElements2));

        auto outputZip = thrust::make_zip_iterator(thrust::make_tuple(d_outputSegmentIds.data(), d_output));

        auto outputZipEnd = executeSetOperation(policy, first1, last1, first2, last2, outputZip, ValueSegmentIdComparator{});

        int outputsize = thrust::distance(outputZip, outputZipEnd);

        std::size_t cubbytes = 0;

        CUDACHECK(cub::DeviceRunLengthEncode::Encode(
            nullptr,
            cubbytes,
            (int*) nullptr,
            (int*) nullptr,
            (int*) nullptr,
            (int*) nullptr,
            outputsize,
            stream
        ));

        void* temp_allocations[4];
        std::size_t temp_allocation_sizes[4];
        
        temp_allocation_sizes[0] = sizeof(int) * numOutputSegments;
        temp_allocation_sizes[1] = sizeof(int) * numOutputSegments;
        temp_allocation_sizes[2] = sizeof(int);
        temp_allocation_sizes[3] = cubbytes;

        std::size_t temp_storage_bytes = 0;
        CUDACHECK(cub::AliasTemporaries(
            nullptr,
            temp_storage_bytes,
            temp_allocations,
            temp_allocation_sizes
        ));

        rmm::device_uvector<char> d_temp(temp_storage_bytes, stream, mr);
        CUDACHECK(cub::AliasTemporaries(
            (void*)d_temp.data(),
            temp_storage_bytes,
            temp_allocations,
            temp_allocation_sizes
        ));

        int* const uniqueIds = (int*)temp_allocations[0];
        int* const reducedCounts = (int*)temp_allocations[1];        
        int* const numRuns = (int*)temp_allocations[2];
        void* const cubtemp = (void*)temp_allocations[3];
        
        CUDACHECK(cub::DeviceRunLengthEncode::Encode(
            cubtemp,
            cubbytes,
            d_outputSegmentIds.data(),
            uniqueIds,
            reducedCounts,
            numRuns,
            outputsize,
            stream
        ));

        if(numOutputSegments <= 4096){

            gpusegmentedsetoperationkernels::initAndSetOutputSegmentSizesSingleBlockKernel<<<1, 1024, 0, stream>>>(
                uniqueIds,
                reducedCounts,
                numRuns,
                d_outputSegmentSizes,
                numOutputSegments
            ); CUDACHECKASYNC;

        }else{

            CUDACHECK(cudaMemsetAsync(
                d_outputSegmentSizes,
                0,
                sizeof(int) * numOutputSegments,
                stream
            ));

            gpusegmentedsetoperationkernels::setOutputSegmentSizesKernel<<<SDIV(numOutputSegments, 256), 256, 0, stream>>>(
                uniqueIds,
                reducedCounts,
                numRuns,
                d_outputSegmentSizes
            ); CUDACHECKASYNC;

        }

        return d_output + outputsize;
    }

};




#endif