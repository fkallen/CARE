#ifndef MULTI_GPU_ARRAY_HPP
#define MULTI_GPU_ARRAY_HPP


#include <hpc_helpers.cuh>

#include <gpu/singlegpu2darray.cuh>

#include <memorymanagement.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include <limits>
#include <stdexcept>

#include <cub/cub.cuh>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;


namespace MultiGpu2dArrayKernels{

    __global__
    void printkernel(int* i){
        printf("printkernel: i = %d\n", *i);
    }

    template<class T, class IndexGenerator>
    __global__
    void gatherKernel(
        const T* __restrict__ src, 
        size_t srcRowPitchInBytes, 
        size_t numColumns,
        size_t numRows,
        T* __restrict__ dest, 
        size_t destRowPitchInBytes, 
        IndexGenerator indices, 
        size_t numIndices
    ){
        if(numIndices == 0) return;

        auto group = cg::this_grid();

        TwoDimensionalArray<T> array(
            numRows,
            numColumns,
            srcRowPitchInBytes,
            const_cast<T*>(src)
        );

        array.gather(group, dest, destRowPitchInBytes, indices, numIndices);
    }

    template<class T, class IndexGenerator>
    __global__
    void gatherKernel(
        const T* __restrict__ src, 
        size_t srcRowPitchInBytes, 
        size_t numRows,
        size_t numColumns,
        T* __restrict__ dest, 
        size_t destRowPitchInBytes, 
        IndexGenerator indices, 
        const size_t* __restrict__ numIndicesPtr
    ){
        const size_t numIndices = *numIndicesPtr;

        if(numIndices == 0) return;

        auto group = cg::this_grid();

        TwoDimensionalArray<T> array(
            numRows,
            numColumns,
            srcRowPitchInBytes,
            const_cast<T*>(src)
        );

        array.gather(group, dest, destRowPitchInBytes, indices, numIndices);
    }

    template<class T, class IndexGenerator>
    __global__
    void scatterKernel(
        const T* __restrict__ src, 
        size_t srcRowPitchInBytes, 
        size_t numRows,
        size_t numColumns,
        T* __restrict__ dest, 
        size_t destRowPitchInBytes, 
        IndexGenerator indices, 
        size_t numIndices
    ){
        if(numIndices == 0) return;

        auto group = cg::this_grid();

        TwoDimensionalArray<T> array(
            numRows,
            numColumns,
            destRowPitchInBytes,
            dest
        );

        array.scatter(group, src, srcRowPitchInBytes, indices, numIndices);
    }

    template<class T, class IndexGenerator>
    __global__
    void scatterKernel(
        const T* __restrict__ src, 
        size_t srcRowPitchInBytes, 
        size_t numRows,
        size_t numColumns,
        T* __restrict__ dest, 
        size_t destRowPitchInBytes, 
        IndexGenerator indices, 
        const size_t* __restrict__ numIndicesPtr
    ){
        const size_t numIndices = *numIndicesPtr;
        if(numIndices == 0) return;

        auto group = cg::this_grid();

        TwoDimensionalArray<T> array(
            numRows,
            numColumns,
            destRowPitchInBytes,
            dest
        );

        array.scatter(group, src, srcRowPitchInBytes, indices, numIndices);
    }

    template<class T>
    __global__
    void copyKernel(const T* __restrict__ src, const size_t* __restrict__ numElementsPtr, T* __restrict__ dst){
        const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t stride = gridDim.x * blockDim.x;

        const size_t numElements = *numElementsPtr;

        for(size_t i = tid; i < numElements; i += stride){
            dst[i] = src[i];
        }
    }

    template<class T>
    __global__
    void copyKernel(const T* __restrict__ src, size_t numElements, T* __restrict__ dst){
        const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t stride = gridDim.x * blockDim.x;

        for(size_t i = tid; i < numElements; i += stride){
            dst[i] = src[i];
        }
    }



    template<class IndexType, int maxNumGpus = 8>
    __global__
    void partitionSplitKernel(
        IndexType* __restrict__ splitIndices, // numIndices elements per partition
        size_t* __restrict__ splitDestinationPositions, // numIndices elements per partition
        size_t* __restrict__ numSplitIndicesPerPartition, 
        int numPartitions,
        const size_t* __restrict__ partitionOffsetsPS,
        size_t numIndices,
        const IndexType* __restrict__ indices
    ){

        assert(numPartitions <= maxNumGpus+1);

        using Index_t = size_t;

        auto atomicAggInc = [](auto& group, Index_t* counter){
            Index_t group_res;
            if(group.thread_rank() == 0){
                group_res = atomicAdd((unsigned long long*)counter, (unsigned long long)(group.size()));
            }
            return group.shfl(group_res, 0) + group.thread_rank();
        };

        //save prefixsum in registers
        Index_t reg_partitionOffsetsPS[maxNumGpus+1];

        #pragma unroll
        for(int i = 0; i < maxNumGpus+1; i++){
            if(i < numPartitions+1){
                reg_partitionOffsetsPS[i] = partitionOffsetsPS[i];
            }else{
                reg_partitionOffsetsPS[i] = partitionOffsetsPS[numPartitions];
            }
        }

        const Index_t numIds = numIndices;

        for(Index_t tid = threadIdx.x + blockIdx.x * Index_t(blockDim.x); 
                tid < numIds; 
                tid += Index_t(blockDim.x) * gridDim.x){
            
            const Index_t elementIndex = indices[tid];
            int location = -1;

            #pragma unroll
            for(int i = 0; i < maxNumGpus+1; i++){
                if(i < numPartitions 
                        && reg_partitionOffsetsPS[i] <= elementIndex 
                        && elementIndex < reg_partitionOffsetsPS[i+1]){
                    location = i;
                    break;
                }
            }

#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 700
            if(location != -1){
                auto g = cg::coalesced_threads();
                //partition into groups of threads with the same location. Cannot use a tiled_partition<N> as input, because for-loop may cause a deadlock:
                //"The implementation may cause the calling thread to wait until all the members of the parent group have invoked the operation before resuming execution."
                auto partitionGroup = cg::labeled_partition(g, location);

                const Index_t j = atomicAggInc(partitionGroup, &numSplitIndicesPerPartition[location]);
                splitIndices[location * numIds + j] = elementIndex;
                splitDestinationPositions[location * numIds + j] = tid;
            }
              
#else
            for(int i = 0; i < numPartitions; ++i){
                if(i == location){
                    auto group = cg::coalesced_threads();
                    const Index_t j = atomicAggInc(group, &numSplitIndicesPerPartition[i]);
                    splitIndices[i * numIds + j] = elementIndex;
                    splitDestinationPositions[i * numIds + j] = tid;
                    break;
                }
            }
#endif 

        }

    }

    template<class T>
    struct LinearAccessFunctor{
        HOSTDEVICEQUALIFIER
        LinearAccessFunctor(T* ptr) : data(ptr){}

        HOSTDEVICEQUALIFIER
        T& operator()(size_t pos){
            return data[pos];
        }

        T* data;
    };
}




enum class MultiGpu2dArrayLayout {FirstFit, EvenShare};
enum class MultiGpu2dArrayInitMode {MustFitCompletely, CanDiscardRows};

template<class T, class IndexType = int>
class MultiGpu2dArray{
public:

    template<class W>
    using HostBuffer = helpers::SimpleAllocationPinnedHost<W>;
    template<class W>
    using DeviceBuffer = helpers::SimpleAllocationDevice<W>;
    //using DeviceBuffer = helpers::SimpleAllocationPinnedHost<W>;



    MultiGpu2dArray()
    : alignmentInBytes(sizeof(T)){

    }

    MultiGpu2dArray(
        size_t numRows, 
        size_t numColumns, 
        size_t alignmentInBytes,
        std::vector<int> dDeviceIds, // data resides on these gpus
        std::vector<size_t> memoryLimits,
        MultiGpu2dArrayLayout layout,
        MultiGpu2dArrayInitMode initMode = MultiGpu2dArrayInitMode::MustFitCompletely) 
    : alignmentInBytes(alignmentInBytes), 
        dataDeviceIds(dDeviceIds)
    {

        assert(alignmentInBytes > 0);
        //assert(alignmentInBytes % sizeof(T) == 0);

        init(numRows, numColumns, memoryLimits, layout, initMode);
    }

    ~MultiGpu2dArray(){
        destroy();
    }

    MultiGpu2dArray(const MultiGpu2dArray& rhs)
        : MultiGpu2dArray(rhs.numRows, rhs.numColumns, rhs.alignmentInBytes)
    {
        gpuArrays.clear();
        for(const auto& otherarray : rhs.gpuArrays){
            gpuArrays.emplace_back(otherarray);
        }
    }

    MultiGpu2dArray(MultiGpu2dArray&& other) noexcept
        : MultiGpu2dArray()
    {
        swap(*this, other);
    }

    MultiGpu2dArray& operator=(MultiGpu2dArray other){
        swap(*this, other);

        return *this;
    }

    friend void swap(MultiGpu2dArray& l, MultiGpu2dArray& r) noexcept
    {
        std::swap(l.numRows, r.numRows);
        std::swap(l.numColumns, r.numColumns);
        std::swap(l.alignmentInBytes, r.alignmentInBytes);
        std::swap(l.gpuArrays, r.gpuArrays);
        std::swap(l.dataDeviceIds, r.dataDeviceIds);
        std::swap(l.h_arrayOffsets, r.h_arrayOffsets);
        std::swap(l.h_arrayOffsetsPrefixSum, r.h_arrayOffsetsPrefixSum);
    }

    void destroy(){
        numRows = 0;
        numColumns = 0;
        alignmentInBytes = 0;
        gpuArrays.clear();
        dataDeviceIds.clear();
        h_arrayOffsets.destroy();
        h_arrayOffsetsPrefixSum.destroy();
    }

    struct HandleStruct{
        struct PerDevice{
            using ArgIndexPair = typename cub::ArgIndexInputIterator<IndexType*, size_t>::value_type;

            PerDevice()
                : 
                d_numSelected(1),
                event(cudaEventDisableTiming)
            {        

            }

            DeviceBuffer<size_t> d_multigpuarrayOffsets{};
            DeviceBuffer<size_t> d_multigpuarrayOffsetsPrefixSum{};
            DeviceBuffer<size_t> d_numSelected{};
            CudaStream stream{};
            CudaEvent event{};

            DeviceBuffer<char> d_cubTemp{};
            DeviceBuffer<IndexType> d_indices{};
            DeviceBuffer<IndexType> d_selectedIndices{};
            DeviceBuffer<size_t> d_selectedPositions{};
            DeviceBuffer<ArgIndexPair> d_selectedIndicesWithPositions{};
            DeviceBuffer<char> d_dataCommunicationBuffer{};
        };

        struct PerCaller{
            using ArgIndexPair = typename cub::ArgIndexInputIterator<IndexType*, size_t>::value_type;

            PerCaller()
                : 
                d_numSelected(1)
            {        

            }

            DeviceBuffer<size_t> d_multigpuarrayOffsets{};
            DeviceBuffer<size_t> d_multigpuarrayOffsetsPrefixSum{};
            DeviceBuffer<size_t> d_numSelected{};
            

            DeviceBuffer<char> d_cubTemp{};
            DeviceBuffer<IndexType> d_indices{};
            DeviceBuffer<IndexType> d_selectedIndices{};
            DeviceBuffer<size_t> d_selectedPositions{};
            DeviceBuffer<ArgIndexPair> d_selectedIndicesWithPositions{};
            DeviceBuffer<char> d_dataCommunicationBuffer{};

            CudaEvent event{cudaEventDisableTiming};
            std::vector<CudaEvent> events{};
            std::vector<CudaStream> streams{};
        };

        HandleStruct(){
            cudaGetDevice(&deviceId); CUERR;            
        }

        HandleStruct(HandleStruct&&) = default;
        HandleStruct(const HandleStruct&) = delete;

        void setMaxNumberOfIndices(size_t num){
            maxNumberOfIndices = num;
            maxNumberOfIndicesIsSet = true;

            cudaDeviceSynchronize(); CUERR;
        }

        size_t getMaxNumberOfIndices() const{
            return maxNumberOfIndices;
        }

        bool maxNumberOfIndicesIsSet = false;
        int deviceId = 0;
        size_t maxNumberOfIndices = std::numeric_limits<size_t>::max();

        std::vector<PerDevice> deviceBuffers{};
        PerCaller callerBuffers{};

        MemoryUsage getMemoryInfo() const{
            MemoryUsage result{};

            return result;
        }
    };

    using Handle = std::shared_ptr<HandleStruct>;

    MemoryUsage getMemoryInfo(const Handle& handle) const{
        MemoryUsage result{};

        return handle->getMemoryInfo();
    }

    Handle makeHandle() const{
        Handle handle = std::make_shared<HandleStruct>();

        const int numDataGpus = dataDeviceIds.size();
        for(int d = 0; d < numDataGpus; d++){
            const int deviceId = dataDeviceIds[d];
            cub::SwitchDevice sd(deviceId);

            typename HandleStruct::PerDevice deviceBuffers; 

            deviceBuffers.d_multigpuarrayOffsets.resize(h_arrayOffsets.size());
            deviceBuffers.d_multigpuarrayOffsetsPrefixSum.resize(h_arrayOffsetsPrefixSum.size());

            cudaMemcpy(
                deviceBuffers.d_multigpuarrayOffsets.get(),
                h_arrayOffsets.get(),
                h_arrayOffsets.size() * sizeof(std::size_t),
                H2D
            ); CUERR;

            cudaMemcpy(
                deviceBuffers.d_multigpuarrayOffsetsPrefixSum.get(),
                h_arrayOffsetsPrefixSum.get(),
                h_arrayOffsetsPrefixSum.size() * sizeof(std::size_t),
                H2D
            ); CUERR;

            handle->deviceBuffers.emplace_back(std::move(deviceBuffers));
        }

        if(h_arrayOffsets.size() > 0){

            auto& callerBuffers = handle->callerBuffers;
            callerBuffers.streams.resize(h_arrayOffsets.size());
            
            for(int i = 0; i < int(h_arrayOffsets.size()); i++){
                callerBuffers.events.emplace_back(cudaEventDisableTiming);
            }

            callerBuffers.d_multigpuarrayOffsets.resize(h_arrayOffsets.size()); CUERR;
            callerBuffers.d_multigpuarrayOffsetsPrefixSum.resize(h_arrayOffsetsPrefixSum.size()); CUERR;
            callerBuffers.d_numSelected.resize(h_arrayOffsets.size()); CUERR;

            cudaMemcpy(
                callerBuffers.d_multigpuarrayOffsets.get(),
                h_arrayOffsets.get(),
                h_arrayOffsets.size() * sizeof(std::size_t),
                H2D
            ); CUERR;

            cudaMemcpy(
                callerBuffers.d_multigpuarrayOffsetsPrefixSum.get(),
                h_arrayOffsetsPrefixSum.get(),
                h_arrayOffsetsPrefixSum.size() * sizeof(std::size_t),
                H2D
            ); CUERR;  

        } 

        return handle;
    }

    void gather(
        Handle& handle, 
        T* d_dest, 
        size_t destRowPitchInBytes, 
        const IndexType* d_indices, 
        size_t numIndices, 
        cudaStream_t destStream = 0
    ) const{

        if(getNumRows() == 0) return;

        //TODO maybe perform batched gather to limit memory usage of handle

        gather_internal(
            handle,
            d_dest,
            destRowPitchInBytes,
            d_indices,
            numIndices,
            false,
            destStream
        );
    }

    void gather(
        Handle& handle, 
        T* d_dest, 
        size_t destRowPitchInBytes, 
        const IndexType* d_indices, 
        size_t numIndices, 
        bool mayContainInvalidIndices,
        cudaStream_t destStream = 0
    ) const{

        if(getNumRows() == 0) return;

        //TODO maybe perform batched gather to limit memory usage of handle

        gather_internal(
            handle,
            d_dest,
            destRowPitchInBytes,
            d_indices,
            numIndices,
            mayContainInvalidIndices,
            destStream
        );
    }

    void scatter(
        Handle& handle, 
        const T* d_src, 
        size_t srcRowPitchInBytes, 
        const IndexType* d_indices, 
        size_t numIndices, 
        cudaStream_t srcStream = 0
    ) const{

        if(getNumRows() == 0) return;

        //TODO maybe perform batched scatter to limit memory usage of handle

        scatter_internal(
            handle,
            d_src,
            srcRowPitchInBytes,
            d_indices,
            numIndices,
            false,
            srcStream
        );
    }

    void scatter(
        Handle& handle, 
        const T* d_src, 
        size_t srcRowPitchInBytes, 
        const IndexType* d_indices, 
        size_t numIndices, 
        bool mayContainInvalidIndices,
        cudaStream_t srcStream = 0
    ) const{

        if(getNumRows() == 0) return;

        //TODO maybe perform batched scatter to limit memory usage of handle

        scatter_internal(
            handle,
            d_src,
            srcRowPitchInBytes,
            d_indices,
            numIndices,
            mayContainInvalidIndices,
            srcStream
        );
    }


    void gather(T* d_dest, size_t destRowPitchInBytes, size_t rowBegin, size_t rowEnd, cudaStream_t stream = 0) const{
        assert(false && "not implemented");
    }

    

    void scatter(const T* d_src, size_t srcRowPitchInBytes, size_t rowBegin, size_t rowEnd, cudaStream_t stream = 0) const{
        assert(false && "not implemented");
    }

    size_t getNumRows() const noexcept{
        return numRows;
    }

    size_t getNumColumns() const noexcept{
        return numColumns;
    }

    size_t getAlignmentInBytes() const noexcept{
        return alignmentInBytes;
    }

    bool isSingleGpu() const noexcept{
        return gpuArrays.size() == 1;
    }

    std::vector<size_t> getRowDistribution() const{
        std::vector<size_t> result(h_arrayOffsets.size());
        std::copy_n(h_arrayOffsets.get(), h_arrayOffsets.size(), result.begin());

        return result;
    }

//these should be private, but this would cause problems with cuda device lambdas
public:
    void gather_internal(
        Handle& handle, T* d_dest, 
        size_t destRowPitchInBytes, 
        const IndexType* d_indices, 
        size_t numIndices, 
        bool mayContainInvalidIndices,
        cudaStream_t destStream
    ) const{
        if(numIndices == 0) return;

        int destDeviceId = 0;
        cudaGetDevice(&destDeviceId); CUERR;

        if(isSingleGpu() && !mayContainInvalidIndices){
            if(gpuArrays[0]->getDeviceId() == destDeviceId){ 
                //all data to be gathered resides on the destination device

                auto indexGenerator = [d_indices] __device__ (auto i){
                    return d_indices[i];
                };

                gpuArrays[0]->gather(d_dest, destRowPitchInBytes, indexGenerator, numIndices, destStream);
                return;
            }else{
                //all data resides on a different device

                //using peer access on different device data

                auto indexGenerator = [d_indices] __device__ (auto i){
                    return d_indices[i];
                };

                gpuArrays[0]->gather(d_dest, destRowPitchInBytes, indexGenerator, numIndices, destStream);
            }
        }else{
            
            //perform multisplit to distribute indices and filter out invalid indices. then perform local gathers,
            // and scatter into result array

            const int numGpus = gpuArrays.size();

            auto& callerBuffers = handle->callerBuffers;
            callerBuffers.d_selectedIndices.resize(numIndices * gpuArrays.size());
            callerBuffers.d_selectedPositions.resize(numIndices * gpuArrays.size());

            for(int d = 0; d < numGpus; d++){
                const auto& gpuArray = gpuArrays[d];
                const int deviceId = gpuArray->getDeviceId();

                cub::SwitchDevice sd(deviceId);

                auto& deviceBuffers = handle->deviceBuffers[d];
                deviceBuffers.d_selectedIndices.resize(numIndices);
                deviceBuffers.d_dataCommunicationBuffer.resize(numIndices * destRowPitchInBytes);
            }
            

            cudaMemsetAsync(callerBuffers.d_numSelected.get(), 0, callerBuffers.d_numSelected.sizeInBytes(), destStream); CUERR;

            //helpers::GpuTimer gpuTimer(destStream, "partitionsplit");

            MultiGpu2dArrayKernels::partitionSplitKernel<<<SDIV(numIndices, 128), 128, 0, destStream>>>(
                callerBuffers.d_selectedIndices.get(),
                callerBuffers.d_selectedPositions.get(),
                callerBuffers.d_numSelected.get(),
                gpuArrays.size(),
                callerBuffers.d_multigpuarrayOffsetsPrefixSum.get(),
                numIndices,
                d_indices
            );

            // gpuTimer.stop();
            // gpuTimer.print();
            
            callerBuffers.d_dataCommunicationBuffer.resize(numIndices * destRowPitchInBytes);

            cudaEventRecord(callerBuffers.event, destStream); CUERR;

            for(int d = 0; d < numGpus; d++){
                const auto& gpuArray = gpuArrays[d];
                auto& deviceBuffers = handle->deviceBuffers[d];
                {

                    const int deviceId = gpuArray->getDeviceId();
                    cub::SwitchDevice sd(deviceId);

                    cudaStreamWaitEvent(deviceBuffers.stream, callerBuffers.event, 0); CUERR;

                    const IndexType* d_selectedIndices = callerBuffers.d_selectedIndices.get() + d * numIndices;
                    const size_t* d_numSelected = callerBuffers.d_numSelected.get() + d;

                    //TODO 
                    // if (destDeviceId != deviceId) and interconnect is pci-e, try to copy selected data to device deviceId in an efficient way
                    // For now, it is always accessed via peer interconnect

                    // if(destDeviceId != deviceId && pci-e){
                    //     //copy indices to local buffer
                    //     MultiGpu2dArrayKernels::copyKernel<<<SDIV(numIndices, 128), 128, 0, deviceBuffers.stream>>>(
                    //         callerBuffers.d_selectedIndices.get() + d * numIndices, 
                    //         callerBuffers.d_numSelected.get() + d,
                    //         deviceBuffers.d_selectedIndices.get()
                    //     ); CUERR;

                    //     // MultiGpu2dArrayKernels::copyKernel<<<SDIV(numIndices, 128), 128, 0, deviceBuffers.stream>>>(
                    //     //     callerBuffers.d_selectedPositions.get() + d * numIndices, 
                    //     //     callerBuffers.d_numSelected.get() + d,
                    //     //     deviceBuffers.d_selectedPositions.get()
                    //     // ); CUERR;

                    //     copy(
                    //         deviceBuffers.d_numSelected.get(), 
                    //         deviceId,
                    //         callerBuffers.d_numSelected.get() + d, 
                    //         destDeviceId,                                         
                    //         sizeof(size_t), 
                    //         deviceBuffers.stream
                    //     );

                    //     d_selectedIndices = deviceBuffers.d_selectedIndices.get();
                    //     d_numSelected = deviceBuffers.d_numSelected.get();
                    // }


                    auto gatherIndexGenerator = [
                        d,
                        d_selectedIndices,
                        ps = deviceBuffers.d_multigpuarrayOffsetsPrefixSum.get()
                    ] __device__ (auto i){
                        const IndexType index = d_selectedIndices[i];
                        return index - ps[d]; //transform into local index for this gpuArray
                    };

                    //std::cerr << "gpuArray->gather\n";
                    gpuArray->gather(
                        (T*)deviceBuffers.d_dataCommunicationBuffer.get(), 
                        destRowPitchInBytes, 
                        gatherIndexGenerator, 
                        d_numSelected,
                        numIndices, 
                        deviceBuffers.stream
                    );

                    cudaEventRecord(deviceBuffers.event, deviceBuffers.stream); CUERR;
                }
              
            }
            
            auto processWithDeviceIdCondition = [&](auto condition){

                //copy partial results from gpuArray to local buffer, and scatter into output buffer
                for(int d = 0; d < numGpus; d++){
                    const auto& gpuArray = gpuArrays[d];
                    const int deviceId = gpuArray->getDeviceId();

                    if(!condition(deviceId)){
                        continue;
                    }

                    auto& deviceBuffers = handle->deviceBuffers[d];

                    cudaStreamWaitEvent(destStream, deviceBuffers.event, 0); CUERR;

                    

                    //if the gpuArray lives on the destination device, we can avoid the copy of partial gathered data, and use it directly
                    T* dataToBeScatteredIntoDestinationArray 
                        = (T*)deviceBuffers.d_dataCommunicationBuffer.get();

                    //TODO 
                    // if (destDeviceId != deviceId) and interconnect is pci-e, try to copy commBuffer to destDeviceId in an efficient way
                    // For now, it is always accessed via peer interconnect

                    // if(destDeviceId != deviceId && pci-e){
                    //     dim3 block(128, 1, 1);
                    //     dim3 grid(SDIV(numIndices, block.x), 1, 1);

                    //     auto copyIndexGenerator = [] __device__ (auto i){
                    //         return i;
                    //     };

                    //     //used as copy kernel between devices with peer access. replace with efficient copy for pci-e interconnect
                    //     MultiGpu2dArrayKernels::gatherKernel<<<grid, block, 0, destStream>>>(
                    //         (const T*)deviceBuffers.d_dataCommunicationBuffer.get(), 
                    //         destRowPitchInBytes, 
                    //         getNumColumns(),
                    //         (T*)callerBuffers.d_dataCommunicationBuffer.get(), 
                    //         destRowPitchInBytes, 
                    //         copyIndexGenerator, 
                    //         callerBuffers.d_numSelected.get() + gpuArrayIndex
                    //     );

                    //     dataToBeScatteredIntoDestinationArray
                    //         = (T*)callerBuffers.d_dataCommunicationBuffer.get();

                    //     //cudaDeviceSynchronize(); CUERR;

                    // }

                    dim3 block(128, 1, 1);
                    dim3 grid(SDIV(numIndices, block.x), 1, 1);

                    // auto scatterIndexGenerator = [
                    //     d_selectedPositions = callerBuffers.d_selectedPositions.get() + gpuArrayIndex * numIndices
                    // ] __device__ (auto i){
                    //     const size_t index = d_selectedPositions[i];
                    //     return index; //destination row
                    // };

                    MultiGpu2dArrayKernels::LinearAccessFunctor<std::size_t> scatterIndexGenerator(
                        callerBuffers.d_selectedPositions.get() + d * numIndices
                    );

                    MultiGpu2dArrayKernels::scatterKernel<<<grid, block, 0, destStream>>>(
                        dataToBeScatteredIntoDestinationArray, 
                        destRowPitchInBytes, 
                        getNumRows(),
                        getNumColumns(),
                        d_dest, 
                        destRowPitchInBytes, 
                        scatterIndexGenerator, 
                        callerBuffers.d_numSelected.get() + d
                    ); CUERR;


                    //cudaDeviceSynchronize(); CUERR;
                }
            };

            //first process chunk on same device id. If it exists, this chunk should finish first, giving remote accesses more time to complete
            processWithDeviceIdCondition([destDeviceId](int id){
                return destDeviceId == id;
            });

            processWithDeviceIdCondition([destDeviceId](int id){
                return destDeviceId != id;
            });
        }
    }


    void gather_internal2(
        Handle& handle, T* d_dest, 
        size_t destRowPitchInBytes, 
        const IndexType* d_indices, 
        size_t numIndices, 
        bool mayContainInvalidIndices,
        cudaStream_t destStream
    ) const{
        if(numIndices == 0) return;

        int destDeviceId = 0;
        cudaGetDevice(&destDeviceId); CUERR;

        if(isSingleGpu() && !mayContainInvalidIndices){
            if(gpuArrays[0]->getDeviceId() == destDeviceId){ 
                //all data to be gathered resides on the destination device

                auto indexGenerator = [d_indices] __device__ (auto i){
                    return d_indices[i];
                };

                gpuArrays[0]->gather(d_dest, destRowPitchInBytes, indexGenerator, numIndices, destStream);
                return;
            }else{
                //all data resides on a different device

                //using peer access on different device data

                auto indexGenerator = [d_indices] __device__ (auto i){
                    return d_indices[i];
                };

                gpuArrays[0]->gather(d_dest, destRowPitchInBytes, indexGenerator, numIndices, destStream);
            }
        }else{

            //perform multisplit to distribute indices and filter out invalid indices. then perform local gathers,
            // and scatter into result array

            const int numGpus = gpuArrays.size();

            auto& callerBuffers = handle->callerBuffers;
            callerBuffers.d_selectedIndices.resize(numIndices * gpuArrays.size());
            callerBuffers.d_selectedPositions.resize(numIndices * gpuArrays.size());

            for(int d = 0; d < numGpus; d++){
                const auto& gpuArray = gpuArrays[d];
                const int deviceId = gpuArray->getDeviceId();

                cub::SwitchDevice sd(deviceId);

                auto& deviceBuffers = handle->deviceBuffers[d];
                deviceBuffers.d_selectedIndices.resize(numIndices);
                deviceBuffers.d_dataCommunicationBuffer.resize(numIndices * destRowPitchInBytes);
            }
            

            cudaMemsetAsync(callerBuffers.d_numSelected.get(), 0, callerBuffers.d_numSelected.sizeInBytes(), destStream); CUERR;

            //helpers::GpuTimer gpuTimer(destStream, "partitionsplit");

            MultiGpu2dArrayKernels::partitionSplitKernel<<<SDIV(numIndices, 128), 128, 0, destStream>>>(
                callerBuffers.d_selectedIndices.get(),
                callerBuffers.d_selectedPositions.get(),
                callerBuffers.d_numSelected.get(),
                gpuArrays.size(),
                callerBuffers.d_multigpuarrayOffsetsPrefixSum.get(),
                numIndices,
                d_indices
            );

            // gpuTimer.stop();
            // gpuTimer.print();
            
            callerBuffers.d_dataCommunicationBuffer.resize(numIndices * destRowPitchInBytes);

            cudaEventRecord(callerBuffers.event, destStream); CUERR;

            for(int d = 0; d < numGpus; d++){
                const auto& gpuArray = gpuArrays[d];
                auto& deviceBuffers = handle->deviceBuffers[d];
                {

                    const int deviceId = gpuArray->getDeviceId();
                    cub::SwitchDevice sd(deviceId);

                    cudaStreamWaitEvent(deviceBuffers.stream, callerBuffers.event, 0); CUERR;

                    const IndexType* d_selectedIndices = callerBuffers.d_selectedIndices.get() + d * numIndices;
                    const size_t* d_numSelected = callerBuffers.d_numSelected.get() + d;

                    auto gatherIndexGenerator = [
                        d,
                        d_selectedIndices,
                        ps = deviceBuffers.d_multigpuarrayOffsetsPrefixSum.get()
                    ] __device__ (auto i){
                        const IndexType index = d_selectedIndices[i];
                        return index - ps[d]; //transform into local index for this gpuArray
                    };

                    //std::cerr << "gpuArray->gather\n";
                    gpuArray->gather(
                        (T*)deviceBuffers.d_dataCommunicationBuffer.get(), 
                        destRowPitchInBytes, 
                        gatherIndexGenerator, 
                        d_numSelected,
                        numIndices, 
                        deviceBuffers.stream
                    );

                    cudaEventRecord(deviceBuffers.event, deviceBuffers.stream); CUERR;
                }

                cudaStreamWaitEvent(callerBuffers.streams[d], deviceBuffers.event, 0); CUERR;


                T* dataToBeScatteredIntoDestinationArray 
                        = (T*)deviceBuffers.d_dataCommunicationBuffer.get();

                //TODO 
                // if (destDeviceId != deviceId) and interconnect is pci-e, try to copy commBuffer to destDeviceId in an efficient way
                // For now, it is always accessed via peer interconnect

                // if(destDeviceId != deviceId && pci-e){
                //     dim3 block(128, 1, 1);
                //     dim3 grid(SDIV(numIndices, block.x), 1, 1);

                //     auto copyIndexGenerator = [] __device__ (auto i){
                //         return i;
                //     };

                //     //used as copy kernel between devices with peer access. replace with efficient copy for pci-e interconnect
                //     MultiGpu2dArrayKernels::gatherKernel<<<grid, block, 0, destStream>>>(
                //         (const T*)deviceBuffers.d_dataCommunicationBuffer.get(), 
                //         destRowPitchInBytes, 
                //         getNumColumns(),
                //         (T*)callerBuffers.d_dataCommunicationBuffer.get(), 
                //         destRowPitchInBytes, 
                //         copyIndexGenerator, 
                //         callerBuffers.d_numSelected.get() + gpuArrayIndex
                //     );

                //     dataToBeScatteredIntoDestinationArray
                //         = (T*)callerBuffers.d_dataCommunicationBuffer.get();

                //     //cudaDeviceSynchronize(); CUERR;

                // }

                dim3 block(128, 1, 1);
                dim3 grid(SDIV(numIndices, block.x), 1, 1);

                // auto scatterIndexGenerator = [
                //     d_selectedPositions = callerBuffers.d_selectedPositions.get() + gpuArrayIndex * numIndices
                // ] __device__ (auto i){
                //     const size_t index = d_selectedPositions[i];
                //     return index; //destination row
                // };

                MultiGpu2dArrayKernels::LinearAccessFunctor<std::size_t> scatterIndexGenerator(
                    callerBuffers.d_selectedPositions.get() + d * numIndices
                );

                MultiGpu2dArrayKernels::scatterKernel<<<grid, block, 0, callerBuffers.streams[d]>>>(
                    dataToBeScatteredIntoDestinationArray, 
                    destRowPitchInBytes, 
                    getNumRows(),
                    getNumColumns(),
                    d_dest, 
                    destRowPitchInBytes, 
                    scatterIndexGenerator, 
                    callerBuffers.d_numSelected.get() + d
                ); CUERR; 

                cudaEventRecord(callerBuffers.events[d]); CUERR;                   
            }

            //join multi gpu work to destStream
            for(int d = 0; d < numGpus; d++){
                cudaStreamWaitEvent(destStream, callerBuffers.events[d], 0); CUERR;
            }
            
            // auto processWithDeviceIdCondition = [&](auto condition){

            //     //copy partial results from gpuArray to local buffer, and scatter into output buffer
            //     for(int d = 0; d < numGpus; d++){
            //         const auto& gpuArray = gpuArrays[d];
            //         const int deviceId = gpuArray->getDeviceId();

            //         if(!condition(deviceId)){
            //             continue;
            //         }

            //         auto& deviceBuffers = handle->deviceBuffers[d];

            //         cudaStreamWaitEvent(destStream, deviceBuffers.event, 0); CUERR;

                    

            //         //if the gpuArray lives on the destination device, we can avoid the copy of partial gathered data, and use it directly
            //         T* dataToBeScatteredIntoDestinationArray 
            //             = (T*)deviceBuffers.d_dataCommunicationBuffer.get();

            //         //TODO 
            //         // if (destDeviceId != deviceId) and interconnect is pci-e, try to copy commBuffer to destDeviceId in an efficient way
            //         // For now, it is always accessed via peer interconnect

            //         // if(destDeviceId != deviceId && pci-e){
            //         //     dim3 block(128, 1, 1);
            //         //     dim3 grid(SDIV(numIndices, block.x), 1, 1);

            //         //     auto copyIndexGenerator = [] __device__ (auto i){
            //         //         return i;
            //         //     };

            //         //     //used as copy kernel between devices with peer access. replace with efficient copy for pci-e interconnect
            //         //     MultiGpu2dArrayKernels::gatherKernel<<<grid, block, 0, destStream>>>(
            //         //         (const T*)deviceBuffers.d_dataCommunicationBuffer.get(), 
            //         //         destRowPitchInBytes, 
            //         //         getNumColumns(),
            //         //         (T*)callerBuffers.d_dataCommunicationBuffer.get(), 
            //         //         destRowPitchInBytes, 
            //         //         copyIndexGenerator, 
            //         //         callerBuffers.d_numSelected.get() + gpuArrayIndex
            //         //     );

            //         //     dataToBeScatteredIntoDestinationArray
            //         //         = (T*)callerBuffers.d_dataCommunicationBuffer.get();

            //         //     //cudaDeviceSynchronize(); CUERR;

            //         // }

            //         dim3 block(128, 1, 1);
            //         dim3 grid(SDIV(numIndices, block.x), 1, 1);

            //         // auto scatterIndexGenerator = [
            //         //     d_selectedPositions = callerBuffers.d_selectedPositions.get() + gpuArrayIndex * numIndices
            //         // ] __device__ (auto i){
            //         //     const size_t index = d_selectedPositions[i];
            //         //     return index; //destination row
            //         // };

            //         MultiGpu2dArrayKernels::LinearAccessFunctor<std::size_t> scatterIndexGenerator(
            //             callerBuffers.d_selectedPositions.get() + d * numIndices
            //         );

            //         MultiGpu2dArrayKernels::scatterKernel<<<grid, block, 0, destStream>>>(
            //             dataToBeScatteredIntoDestinationArray, 
            //             destRowPitchInBytes, 
            //             getNumRows(),
            //             getNumColumns(),
            //             d_dest, 
            //             destRowPitchInBytes, 
            //             scatterIndexGenerator, 
            //             callerBuffers.d_numSelected.get() + d
            //         ); CUERR;


            //         //cudaDeviceSynchronize(); CUERR;
            //     }
            // };

            // //first process chunk on same device id. If it exists, this chunk should finish first, giving remote accesses more time to complete
            // processWithDeviceIdCondition([destDeviceId](int id){
            //     return destDeviceId == id;
            // });

            // processWithDeviceIdCondition([destDeviceId](int id){
            //     return destDeviceId != id;
            // });
        }
    }

    void scatter_internal(
        Handle& handle, 
        const T* d_src, 
        size_t srcRowPitchInBytes, 
        const IndexType* d_indices, 
        size_t numIndices, 
        bool mayContainInvalidIndices,
        cudaStream_t srcStream = 0
    ) const{
        if(numIndices == 0) return;

        int srcDeviceId = 0;
        cudaGetDevice(&srcDeviceId); CUERR;

        if(isSingleGpu() && !mayContainInvalidIndices && gpuArrays[0]->getDeviceId() == srcDeviceId){ //if all data should be scattered to same device
            auto indexGenerator = [d_indices] __device__ (auto i){
                return d_indices[i];
            };

            gpuArrays[0]->scatter(d_src, srcRowPitchInBytes, indexGenerator, numIndices, srcStream);
            return;
        }else{

            const int numGpus = gpuArrays.size();
            cub::ArgIndexInputIterator<const IndexType*, size_t> d_indicesWithPosition(d_indices);

            auto& callerBuffers = handle->callerBuffers;
            callerBuffers.d_selectedIndicesWithPositions.resize(numIndices);
            callerBuffers.d_dataCommunicationBuffer.resize(numIndices * srcRowPitchInBytes);

            size_t temp_storage_bytes = 0;
            {
                auto selectOp = [
                    d = numGpus,
                    ps = callerBuffers.d_multigpuarrayOffsetsPrefixSum.get()
                ] __device__ (auto indexPositionPair){
                    //key == position, value == index
                    //select if index belongs to device
                    return (ps[d] <= indexPositionPair.value && indexPositionPair.value < ps[d+1]);
                };

                cub::DeviceSelect::If(
                    nullptr, 
                    temp_storage_bytes, 
                    d_indicesWithPosition, 
                    callerBuffers.d_selectedIndicesWithPositions.get(), 
                    callerBuffers.d_numSelected.get(), 
                    numIndices, 
                    selectOp, 
                    srcStream
                );
            }

            callerBuffers.d_cubTemp.resize(temp_storage_bytes);

            for(int d = 0; d < numGpus; d++){
                const auto& gpuArray = gpuArrays[d];
                const int deviceId = gpuArray->getDeviceId();
                cub::SwitchDevice sddev(deviceId);

                auto& targetDeviceBuffers = handle->deviceBuffers[d];
                targetDeviceBuffers.d_selectedIndicesWithPositions.resize(numIndices);
                targetDeviceBuffers.d_dataCommunicationBuffer.resize(numIndices * srcRowPitchInBytes);
            }

            for(int d = 0; d < numGpus; d++){
                const auto& gpuArray = gpuArrays[d];
                const int deviceId = gpuArray->getDeviceId();

                auto& targetDeviceBuffers = handle->deviceBuffers[d];               

                auto selectOp = [
                    d,
                    ps = callerBuffers.d_multigpuarrayOffsetsPrefixSum.get()
                ] __device__ (auto indexPositionPair){
                    //key == position, value == index
                    //select if index belongs to device
                    return (ps[d] <= indexPositionPair.value && indexPositionPair.value < ps[d+1]);
                };

                //select indices for gpuArray
                cub::DeviceSelect::If(
                    callerBuffers.d_cubTemp.get(), 
                    temp_storage_bytes, 
                    d_indicesWithPosition, 
                    callerBuffers.d_selectedIndicesWithPositions.get(), 
                    callerBuffers.d_numSelected.get(), 
                    numIndices, 
                    selectOp, 
                    srcStream
                );

                //cudaDeviceSynchronize(); CUERR;

                auto gatherIndexGenerator = [
                    d_selectedIndicesWithPositions = callerBuffers.d_selectedIndicesWithPositions.get(), 
                    ps = callerBuffers.d_multigpuarrayOffsetsPrefixSum.get()
                ] __device__ (auto i){
                    const IndexType index = d_selectedIndicesWithPositions[i].key; // position in array
                    return index; 
                };

                //local gather data which should be scattered to current gpuArray
                dim3 block(128, 1, 1);
                dim3 grid(SDIV(numIndices, block.x), 1, 1);

                MultiGpu2dArrayKernels::gatherKernel<<<grid, block, 0, srcStream>>>(
                    d_src, 
                    srcRowPitchInBytes, 
                    getNumRows(),
                    getNumColumns(),
                    (T*)callerBuffers.d_dataCommunicationBuffer.get(), 
                    srcRowPitchInBytes, 
                    gatherIndexGenerator, 
                    callerBuffers.d_numSelected.get()
                ); CUERR;

                //cudaDeviceSynchronize(); CUERR;

                copy(
                    targetDeviceBuffers.d_numSelected.get(), 
                    deviceId, 
                    callerBuffers.d_numSelected.get(), 
                    srcDeviceId, 
                    sizeof(size_t), 
                    srcStream
                );

                copy(
                    targetDeviceBuffers.d_selectedIndicesWithPositions.get(), 
                    deviceId, 
                    callerBuffers.d_selectedIndicesWithPositions.get(), 
                    srcDeviceId, 
                    sizeof(typename cub::ArgIndexInputIterator<IndexType*, size_t>::value_type) * numIndices, 
                    srcStream
                );

                copy(
                    targetDeviceBuffers.d_dataCommunicationBuffer.get(), 
                    deviceId, 
                    callerBuffers.d_dataCommunicationBuffer.get(), 
                    srcDeviceId, 
                    srcRowPitchInBytes * numIndices, 
                    srcStream
                );

                //cudaDeviceSynchronize(); CUERR;

                cudaEventRecord(callerBuffers.event, srcStream); CUERR;

                //cudaDeviceSynchronize(); CUERR;

                //scatter into target gpuArray
                cub::SwitchDevice sd(deviceId);

                cudaStreamWaitEvent(targetDeviceBuffers.stream, callerBuffers.event, 0); CUERR;

                auto scatterIndexGenerator = [
                    d,
                    d_selectedIndicesWithPositions = targetDeviceBuffers.d_selectedIndicesWithPositions.get(),
                    ps = targetDeviceBuffers.d_multigpuarrayOffsetsPrefixSum.get()
                ] __device__ (auto i){
                    const IndexType index = d_selectedIndicesWithPositions[i].value;
                    return index - ps[d]; //transform into local index for this gpuArray
                };

                gpuArray->scatter(
                    (const T*)targetDeviceBuffers.d_dataCommunicationBuffer.get(), 
                    srcRowPitchInBytes, 
                    scatterIndexGenerator, 
                    targetDeviceBuffers.d_numSelected.get(),
                    numIndices, 
                    targetDeviceBuffers.stream
                );

                cudaEventRecord(targetDeviceBuffers.event, targetDeviceBuffers.stream); CUERR;

                //cudaDeviceSynchronize(); CUERR;

                //gpuArray->print();
            }

            //join multi gpu work to srcStream
            for(int d = 0; d < numGpus; d++){
                const auto& gpuArray = gpuArrays[d];
                const int deviceId = gpuArray->getDeviceId();

                cudaStreamWaitEvent(srcStream, handle->deviceBuffers[d].event, 0); CUERR;
            }
        }
    }

    MemoryUsage getMemoryInfo() const{
        MemoryUsage result{};

        result.host = 0;
        for(const auto& ptr : gpuArrays){
            result += ptr->getMemoryInfo();
        }

        return result;
    }

private:
    void copy(
        void* dst, 
        int dstDevice, 
        const void* src, 
        int srcDevice, 
        size_t count, 
        cudaStream_t stream = 0
    ) const{
        if(dstDevice == srcDevice && dst == src){
            //std::cerr << "copy into same buffer on device " << srcDevice << ". return\n";
            return;
        }else{
            //std::cerr << "copy from device " << srcDevice << " to device " << dstDevice << "\n";
        }

        cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream); CUERR;
    }

    void init(
        size_t numRows, 
        size_t numColumns, 
        std::vector<size_t> memoryLimits, 
        MultiGpu2dArrayLayout layout,
        MultiGpu2dArrayInitMode initMode
    ){
        assert(numRows > 0);
        assert(numColumns > 0);
        assert(memoryLimits.size() == dataDeviceIds.size());

        this->numRows = numRows;
        this->numColumns = numColumns;

        h_arrayOffsets.resize(dataDeviceIds.size());
        h_arrayOffsetsPrefixSum.resize(h_arrayOffsets.size() + 1);

        //init gpu arrays

        const int numGpus = dataDeviceIds.size();

        const size_t minbytesPerRow = sizeof(T) * numColumns;
        const size_t rowPitchInBytes = SDIV(minbytesPerRow, getAlignmentInBytes()) * getAlignmentInBytes();

        std::vector<size_t> maxRowsPerGpu(numGpus);

        for(int i = 0; i < numGpus; i++){
            maxRowsPerGpu[i] = memoryLimits[i] / rowPitchInBytes;
        }

        if(layout == MultiGpu2dArrayLayout::EvenShare){
            std::cerr << "Layout::EvenShare not implemented. Will use Layout::FirstFit\n";
        }

        std::vector<size_t> rowsPerGpu(numGpus);

        size_t remaining = numRows;
        for(int i = 0; i < numGpus; i++){
            size_t myrows = std::min(maxRowsPerGpu[i], remaining);
            rowsPerGpu[i] = myrows;

            remaining -= myrows;

            //std::cerr << rowsPerGpu[i] << " ";
        }

        //std::cerr << ", remaining " << remaining << "\n";

        if(initMode == MultiGpu2dArrayInitMode::MustFitCompletely){
            if(remaining > 0){
                throw std::invalid_argument("Cannot fit all array elements into provided memory\n");
            }
        }else{
            //InitMode::CanDiscardRows

            this->numRows -= remaining;
        }

        std::vector<int> usedDeviceIds;

        for(int i = 0; i < numGpus; i++){
            if(rowsPerGpu[i] > 0){
                cub::SwitchDevice sd(dataDeviceIds[i]);
                auto arrayptr = std::make_unique<Gpu2dArrayManaged<T>>(
                    rowsPerGpu[i], getNumColumns(), getAlignmentInBytes()
                );
                gpuArrays.emplace_back(std::move(arrayptr));

                usedDeviceIds.emplace_back(dataDeviceIds[i]);
            }
        }

        dataDeviceIds = usedDeviceIds;

        std::copy(rowsPerGpu.begin(), rowsPerGpu.end(), h_arrayOffsets.get());
        std::partial_sum(rowsPerGpu.begin(), rowsPerGpu.end(), h_arrayOffsetsPrefixSum.get() + 1);
        h_arrayOffsetsPrefixSum[0] = 0;


        for(int i = 0; i < numGpus; i++){
            cub::SwitchDevice sd(dataDeviceIds[i]);

            DeviceBuffer<std::size_t> a(h_arrayOffsets.size());

            cudaMemcpy(
                a.get(),
                h_arrayOffsets.get(),
                h_arrayOffsets.size() * sizeof(std::size_t),
                H2D
            ); CUERR;

            d_offsetsArrayForGpus.emplace_back(std::move(a));

            DeviceBuffer<std::size_t> b(h_arrayOffsetsPrefixSum.size());

            cudaMemcpy(
                b.get(),
                h_arrayOffsetsPrefixSum.get(),
                h_arrayOffsetsPrefixSum.size() * sizeof(std::size_t),
                H2D
            ); CUERR;

            d_offsetsPrefixSumArrayForGpus.emplace_back(std::move(b));
        }

        std::cerr << "multigpuarray offsets prefixsum: ";
        for(int i = 0; i < numGpus+1; i++){
            std::cerr << h_arrayOffsetsPrefixSum[i] << " ";
        }
        std::cerr << "\n";

    }

    // int getDeviceIdIndex(int deviceId) const{
    //     auto it = std::find(dataDeviceIds.begin(), dataDeviceIds.end(), deviceId);
    //     if(it != dataDeviceIds.end()){
    //         return std::distance(dataDeviceIds.begin(), it);
    //     }else{
    //         assert(false);
    //         return -1;
    //     }
    // }

    size_t numRows{};
    size_t numColumns{};
    size_t alignmentInBytes{};
    std::vector<std::unique_ptr<Gpu2dArrayManaged<T>>> gpuArrays{};

    std::vector<int> dataDeviceIds;

    HostBuffer<size_t> h_arrayOffsets;
    HostBuffer<size_t> h_arrayOffsetsPrefixSum;

    std::vector<DeviceBuffer<std::size_t>> d_offsetsArrayForGpus{};
    std::vector<DeviceBuffer<std::size_t>> d_offsetsPrefixSumArrayForGpus{};
};





#endif
