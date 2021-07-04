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
    template<class ThrustAllocator, class T, class SegmentIdIter1, class SegmentIdIter2>
    static T* set_difference(
        ThrustAllocator& allocator,
        const T* d_input1,
        const int* d_segmentSizes1,
        const int* d_segmentBeginOffsets1,
        SegmentIdIter1 d_segmentIds1,
        int numElements1,
        int numSegments1,
        const T* d_input2,
        const int* d_segmentSizes2,
        const int* d_segmentBeginOffsets2,
        SegmentIdIter2 d_segmentIds2,
        int numElements2,
        int numSegments2,        
        T* d_output,
        int* d_outputSegmentSizes,
        int* d_outputSegmentIds,
        int numOutputSegments,
        cudaStream_t stream
    ){

        const int expectedNumOutputSegments = numSegments1;
        assert(numOutputSegments == expectedNumOutputSegments);

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
            allocator,
            d_input1,
            d_segmentSizes1,
            d_segmentBeginOffsets1,
            d_segmentIds1,
            numElements1,
            numSegments1,
            d_input2,
            d_segmentSizes2,
            d_segmentBeginOffsets2,
            d_segmentIds2,
            numElements2,
            numSegments2,
            d_output,
            d_outputSegmentSizes,
            d_outputSegmentIds,
            expectedNumOutputSegments,
            stream,
            executeSetOperation
        );
    }

    //result = input1 \cap input2, per segment
    template<class ThrustAllocator, class T, class SegmentIdIter1, class SegmentIdIter2>
    static T* set_intersection(
        ThrustAllocator& allocator,
        const T* d_input1,
        const int* d_segmentSizes1,
        const int* d_segmentBeginOffsets1,
        SegmentIdIter1 d_segmentIds1,
        int numElements1,
        int numSegments1,
        const T* d_input2,
        const int* d_segmentSizes2,
        const int* d_segmentBeginOffsets2,
        SegmentIdIter2 d_segmentIds2,
        int numElements2,
        int numSegments2,        
        T* d_output,
        int* d_outputSegmentSizes,
        int* d_outputSegmentIds,
        int numOutputSegments,
        cudaStream_t stream
    ){

        const int expectedNumOutputSegments = std::max(numSegments1, numSegments2);
        assert(numOutputSegments == expectedNumOutputSegments);

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
            allocator,
            d_input1,
            d_segmentSizes1,
            d_segmentBeginOffsets1,
            d_segmentIds1,
            numElements1,
            numSegments1,
            d_input2,
            d_segmentSizes2,
            d_segmentBeginOffsets2,
            d_segmentIds2,
            numElements2,
            numSegments2,
            d_output,
            d_outputSegmentSizes,
            d_outputSegmentIds,
            expectedNumOutputSegments,
            stream,
            executeSetOperation
        );
    }

    //result = (input1 \cup input2) - (input1 \cap input2), per segment
    template<class ThrustAllocator, class T, class SegmentIdIter1, class SegmentIdIter2>
    static T* set_symmetric_difference(
        ThrustAllocator& allocator,
        const T* d_input1,
        const int* d_segmentSizes1,
        const int* d_segmentBeginOffsets1,
        SegmentIdIter1 d_segmentIds1,
        int numElements1,
        int numSegments1,
        const T* d_input2,
        const int* d_segmentSizes2,
        const int* d_segmentBeginOffsets2,
        SegmentIdIter2 d_segmentIds2,
        int numElements2,
        int numSegments2,        
        T* d_output,
        int* d_outputSegmentSizes,
        int* d_outputSegmentIds,
        int numOutputSegments,
        cudaStream_t stream
    ){

        const int expectedNumOutputSegments = std::max(numSegments1, numSegments2);
        assert(numOutputSegments == expectedNumOutputSegments);

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
            allocator,
            d_input1,
            d_segmentSizes1,
            d_segmentBeginOffsets1,
            d_segmentIds1,
            numElements1,
            numSegments1,
            d_input2,
            d_segmentSizes2,
            d_segmentBeginOffsets2,
            d_segmentIds2,
            numElements2,
            numSegments2,
            d_output,
            d_outputSegmentSizes,
            d_outputSegmentIds,
            expectedNumOutputSegments,
            stream,
            executeSetOperation
        );
    }

    //result = input1 \cup input2, per segment
    template<class ThrustAllocator, class T, class SegmentIdIter1, class SegmentIdIter2>
    static T* set_union(
        ThrustAllocator& allocator,
        const T* d_input1,
        const int* d_segmentSizes1,
        const int* d_segmentBeginOffsets1,
        SegmentIdIter1 d_segmentIds1,
        int numElements1,
        int numSegments1,
        const T* d_input2,
        const int* d_segmentSizes2,
        const int* d_segmentBeginOffsets2,
        SegmentIdIter2 d_segmentIds2,
        int numElements2,
        int numSegments2,        
        T* d_output,
        int* d_outputSegmentSizes,
        int* d_outputSegmentIds,
        int numOutputSegments,
        cudaStream_t stream
    ){

        const int expectedNumOutputSegments = std::max(numSegments1, numSegments2);
        assert(numOutputSegments == expectedNumOutputSegments);

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
            allocator,
            d_input1,
            d_segmentSizes1,
            d_segmentBeginOffsets1,
            d_segmentIds1,
            numElements1,
            numSegments1,
            d_input2,
            d_segmentSizes2,
            d_segmentBeginOffsets2,
            d_segmentIds2,
            numElements2,
            numSegments2,
            d_output,
            d_outputSegmentSizes,
            d_outputSegmentIds,
            expectedNumOutputSegments,
            stream,
            executeSetOperation
        );
    }

    //result = input1 \cap input2, per segment
    template<class ThrustAllocator, class T, class SegmentIdIter1, class SegmentIdIter2>
    static T* merge(
        ThrustAllocator& allocator,
        const T* d_input1,
        const int* d_segmentSizes1,
        const int* d_segmentBeginOffsets1,
        SegmentIdIter1 d_segmentIds1,
        int numElements1,
        int numSegments1,
        const T* d_input2,
        const int* d_segmentSizes2,
        const int* d_segmentBeginOffsets2,
        SegmentIdIter2 d_segmentIds2,
        int numElements2,
        int numSegments2,        
        T* d_output,
        int* d_outputSegmentSizes,
        int* d_outputSegmentIds,
        int numOutputSegments,
        cudaStream_t stream
    ){

        const int expectedNumOutputSegments = std::max(numSegments1, numSegments2);
        assert(numOutputSegments == expectedNumOutputSegments);

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
            allocator,
            d_input1,
            d_segmentSizes1,
            d_segmentBeginOffsets1,
            d_segmentIds1,
            numElements1,
            numSegments1,
            d_input2,
            d_segmentSizes2,
            d_segmentBeginOffsets2,
            d_segmentIds2,
            numElements2,
            numSegments2,
            d_output,
            d_outputSegmentSizes,
            d_outputSegmentIds,
            expectedNumOutputSegments,
            stream,
            executeSetOperation
        );
    }

private:

    //result = input1 OP input2, per segment
    template<class ThrustAllocator, class T, class SegmentIdIter1, class SegmentIdIter2, class ThrustSetOpFunc>
    static T* setOperation_impl(
        ThrustAllocator& allocator,
        const T* d_input1,
        const int* d_segmentSizes1,
        const int* d_segmentBeginOffsets1,
        SegmentIdIter1 d_segmentIds1,
        int numElements1,
        int numSegments1,
        const T* d_input2,
        const int* d_segmentSizes2,
        const int* d_segmentBeginOffsets2,
        SegmentIdIter2 d_segmentIds2,
        int numElements2,
        int numSegments2,        
        T* d_output,
        int* d_outputSegmentSizes,
        int* d_outputSegmentIds,
        int numOutputSegments,
        cudaStream_t stream,
        ThrustSetOpFunc executeSetOperation
    ){
        static_assert(sizeof(typename ThrustAllocator::value_type) == 1, "Allocator for GpuSegmentedSetOperation difference must allocate bytes.");

        auto policy = thrust::cuda::par(allocator).on(stream);

        // cudaDeviceSynchronize(); CUERR; 

        // std::vector<int> input1(numElements1);
        // cudaMemcpyAsync(
        //     input1.data(),
        //     d_input1,
        //     sizeof(T) * (numElements1),
        //     D2H,
        //     stream
        // ); CUERR;

        // std::vector<int> segmentIds1(numElements1);
        // cudaMemcpyAsync(
        //     segmentIds1.data(),
        //     d_segmentIds1,
        //     sizeof(int) * (numElements1),
        //     D2H,
        //     stream
        // ); CUERR;

        // std::vector<int> segmentSizes1(numSegments1);
        // cudaMemcpyAsync(
        //     segmentSizes1.data(),
        //     d_segmentSizes1,
        //     sizeof(int) * (numSegments1),
        //     D2H,
        //     stream
        // ); CUERR;

        // std::vector<int> segmentBeginOffsets1(numSegments1);
        // cudaMemcpyAsync(
        //     segmentBeginOffsets1.data(),
        //     d_segmentBeginOffsets1,
        //     sizeof(int) * (numSegments1),
        //     D2H,
        //     stream
        // ); CUERR;


        // std::vector<int> input2(numElements2);
        // cudaMemcpyAsync(
        //     input2.data(),
        //     d_input2,
        //     sizeof(T) * (numElements2),
        //     D2H,
        //     stream
        // ); CUERR;

        // std::vector<int> segmentIds2(numElements2);
        // cudaMemcpyAsync(
        //     segmentIds2.data(),
        //     d_segmentIds2,
        //     sizeof(int) * (numElements2),
        //     D2H,
        //     stream
        // ); CUERR;

        // std::vector<int> segmentSizes2(numSegments2);
        // cudaMemcpyAsync(
        //     segmentSizes2.data(),
        //     d_segmentSizes2,
        //     sizeof(int) * (numSegments2),
        //     D2H,
        //     stream
        // ); CUERR;

        // std::vector<int> segmentBeginOffsets2(numSegments2);
        // cudaMemcpyAsync(
        //     segmentBeginOffsets2.data(),
        //     d_segmentBeginOffsets2,
        //     sizeof(int) * (numSegments2),
        //     D2H,
        //     stream
        // ); CUERR;

        // cudaDeviceSynchronize(); CUERR;

        // std::cerr << "input1\n";
        // std::copy(input1.begin(), input1.end(), std::ostream_iterator<T>(std::cerr, " "));
        // std::cerr << "\n";

        // std::cerr << "segmentIds1\n";
        // std::copy(segmentIds1.begin(), segmentIds1.end(), std::ostream_iterator<int>(std::cerr, " "));
        // std::cerr << "\n";

        // std::cerr << "segmentSizes1\n";
        // std::copy(segmentSizes1.begin(), segmentSizes1.end(), std::ostream_iterator<int>(std::cerr, " "));
        // std::cerr << "\n";

        // std::cerr << "segmentBeginOffsets1\n";
        // std::copy(segmentBeginOffsets1.begin(), segmentBeginOffsets1.end(), std::ostream_iterator<int>(std::cerr, " "));
        // std::cerr << "\n";

        // std::cerr << "input2\n";
        // std::copy(input2.begin(), input2.end(), std::ostream_iterator<T>(std::cerr, " "));
        // std::cerr << "\n";

        // std::cerr << "segmentIds2\n";
        // std::copy(segmentIds2.begin(), segmentIds2.end(), std::ostream_iterator<int>(std::cerr, " "));
        // std::cerr << "\n";

        // std::cerr << "segmentSizes2\n";
        // std::copy(segmentSizes2.begin(), segmentSizes2.end(), std::ostream_iterator<int>(std::cerr, " "));
        // std::cerr << "\n";

        // std::cerr << "segmentBeginOffsets2\n";
        // std::copy(segmentBeginOffsets2.begin(), segmentBeginOffsets2.end(), std::ostream_iterator<int>(std::cerr, " "));
        // std::cerr << "\n";





        auto first1 = thrust::make_zip_iterator(thrust::make_tuple(d_segmentIds1, d_input1));
        auto last1 = thrust::make_zip_iterator(thrust::make_tuple(d_segmentIds1 + numElements1, d_input1 + numElements1));

        auto first2 = thrust::make_zip_iterator(thrust::make_tuple(d_segmentIds2, d_input2));
        auto last2 = thrust::make_zip_iterator(thrust::make_tuple(d_segmentIds2 + numElements2, d_input2 + numElements2));

        auto outputZip = thrust::make_zip_iterator(thrust::make_tuple(d_outputSegmentIds, d_output));

        auto outputZipEnd = executeSetOperation(policy, first1, last1, first2, last2, outputZip, ValueSegmentIdComparator{});

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
        
        temp_allocation_sizes[0] = sizeof(int) * numOutputSegments;
        temp_allocation_sizes[1] = sizeof(int) * numOutputSegments;
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

        if(numOutputSegments <= 4096){

            initAndSetOutputSegmentSizesSingleBlockKernel<<<1, 1024, 0, stream>>>(
                uniqueIds,
                reducedCounts,
                numRuns,
                d_outputSegmentSizes,
                numOutputSegments
            );

        }else{

            cudaMemsetAsync(
                d_outputSegmentSizes,
                0,
                sizeof(int) * numOutputSegments,
                stream
            );

            setOutputSegmentSizesKernel<<<SDIV(numOutputSegments, 256), 256, 0, stream>>>(
                uniqueIds,
                reducedCounts,
                numRuns,
                d_outputSegmentSizes
            );

        }

        allocator.deallocate(tempPtr, sizeof(char) * temp_storage_bytes);

        // std::vector<int> output(outputsize);
        // cudaMemcpyAsync(
        //     output.data(),
        //     d_output,
        //     sizeof(T) * (outputsize),
        //     D2H,
        //     stream
        // ); CUERR;

        // std::vector<int> outputSegmentIds(outputsize);
        // cudaMemcpyAsync(
        //     outputSegmentIds.data(),
        //     d_outputSegmentIds,
        //     sizeof(int) * (outputsize),
        //     D2H,
        //     stream
        // ); CUERR;

        // std::vector<int> outputSegmentSizes(numOutputSegments);
        // cudaMemcpyAsync(
        //     outputSegmentSizes.data(),
        //     d_outputSegmentSizes,
        //     sizeof(int) * (numOutputSegments),
        //     D2H,
        //     stream
        // ); CUERR;

        // cudaDeviceSynchronize(); CUERR;

        // std::cerr << "output\n";
        // std::copy(output.begin(), output.end(), std::ostream_iterator<T>(std::cerr, " "));
        // std::cerr << "\n";

        // std::cerr << "outputSegmentIds\n";
        // std::copy(outputSegmentIds.begin(), outputSegmentIds.end(), std::ostream_iterator<int>(std::cerr, " "));
        // std::cerr << "\n";

        // std::cerr << "outputSegmentSizes\n";
        // std::copy(outputSegmentSizes.begin(), outputSegmentSizes.end(), std::ostream_iterator<int>(std::cerr, " "));
        // std::cerr << "\n";        

        return d_output + outputsize;
    }

};




#endif