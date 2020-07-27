#ifndef CARE_MULTI_GPU_ARRAY_HPP
#define CARE_MULTI_GPU_ARRAY_HPP

#include <gpu/cuda_raiiwrappers.hpp>
#include <gpu/gpuarray.hpp>
#include <hpc_helpers.cuh>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include <gpu/simpleallocation.cuh>

#include <cub/cub.cuh>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace care{

namespace MultiGpu2dArrayKernels{

    template<class T, class IndexGenerator>
    __global__
    void gatherKernel(
        const T* __restrict__ src, 
        size_t srcRowPitchInBytes, 
        size_t numColumns,
        T* __restrict__ dest, 
        size_t destRowPitchInBytes, 
        IndexGenerator indices, 
        int numIndices
    ){
        auto group = cg::this_grid();

        const size_t elementsToCopy = numIndices * numColumns;

        for(size_t i = group.thread_rank(); i < elementsToCopy; i += group.size()){
            const size_t outputRow = i / numColumns;
            const size_t inputRow = indices(outputRow);
            const size_t column = i % numColumns;
            
            ((T*)(((char*)dest) + outputRow * destRowPitchInBytes))[column] 
                = ((const T*)(((const char*)src) + inputRow * srcRowPitchInBytes))[column];
        }
    }

    template<class T, class IndexGenerator>
    __global__
    void gatherKernel(
        const T* __restrict__ src, 
        size_t srcRowPitchInBytes, 
        size_t numColumns,
        T* __restrict__ dest, 
        size_t destRowPitchInBytes, 
        IndexGenerator indices, 
        const int* __restrict__ numIndicesPtr
    ){
        auto group = cg::this_grid();

        const int numIndices = *numIndicesPtr;

        const size_t elementsToCopy = numIndices * numColumns;

        for(size_t i = group.thread_rank(); i < elementsToCopy; i += group.size()){
            const size_t outputRow = i / numColumns;
            const size_t inputRow = indices(outputRow);
            const size_t column = i % numColumns;
            
            ((T*)(((char*)dest) + outputRow * destRowPitchInBytes))[column] 
                = ((const T*)(((const char*)src) + inputRow * srcRowPitchInBytes))[column];
        }
    }

    template<class T, class IndexGenerator>
    __global__
    void scatterKernel(
        const T* __restrict__ src, 
        size_t srcRowPitchInBytes, 
        size_t numColumns,
        T* __restrict__ dest, 
        size_t destRowPitchInBytes, 
        IndexGenerator indices, 
        int numIndices
    ){
        auto group = cg::this_grid();

        const size_t elementsToCopy = numIndices * numColumns;

        for(size_t i = group.thread_rank(); i < elementsToCopy; i += group.size()){
            const size_t inputRow = i / numColumns;
            const size_t outputRow = indices(inputRow);
            const size_t column = i % numColumns;
            
            ((T*)(((char*)dest) + outputRow * destRowPitchInBytes))[column] 
                = ((const T*)(((const char*)src) + inputRow * srcRowPitchInBytes))[column];
        }
    }

    template<class T, class IndexGenerator>
    __global__
    void scatterKernel(
        const T* __restrict__ src, 
        size_t srcRowPitchInBytes, 
        size_t numColumns,
        T* __restrict__ dest, 
        size_t destRowPitchInBytes, 
        IndexGenerator indices, 
        const int* __restrict__ numIndicesPtr
    ){
        const int numIndices = *numIndicesPtr;

        auto group = cg::this_grid();

        const size_t elementsToCopy = numIndices * numColumns;

        for(size_t i = group.thread_rank(); i < elementsToCopy; i += group.size()){
            const size_t inputRow = i / numColumns;
            const size_t outputRow = indices(inputRow);
            const size_t column = i % numColumns;
            
            ((T*)(((char*)dest) + outputRow * destRowPitchInBytes))[column] 
                = ((const T*)(((const char*)src) + inputRow * srcRowPitchInBytes))[column];
        }
    }
}

template<class T>
class MultiGpu2dArray{
public:

    enum class Layout {FirstFit, EvenShare};

    MultiGpu2dArray()
    : alignmentInBytes(sizeof(T)){

    }

    MultiGpu2dArray(
        size_t numRows, 
        size_t numColumns, 
        size_t alignmentInBytes,
        std::vector<int> dDeviceIds, // data resides on these gpus
        std::vector<int> cDeviceIds,
        std::vector<size_t> memoryLimits,
        Layout layout) // array can be accessed by these gpus
    : alignmentInBytes(alignmentInBytes), 
        dataDeviceIds(dDeviceIds),
        callerDeviceIds(cDeviceIds)
    {

        assert(alignmentInBytes > 0);
        assert(alignmentInBytes % sizeof(T) == 0);

        init(numRows, numColumns, memoryLimits, layout);
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
        std::swap(l.gpuArrays, r.gpuArrays);
    }

    void destroy(){
        numRows = 0;
        numColumns = 0;
        gpuArrays.clear();
    }

    struct HandleStruct{
        struct PerDevice{
            using ArgIndexPair = cub::ArgIndexInputIterator<int*, int>::value_type;

            PerDevice()
                : 
                d_numSelected(1),
                event(cudaEventDisableTiming)
            {        

            }

            SimpleAllocationDevice<size_t> d_multigpuarrayOffsets{};
            SimpleAllocationDevice<size_t> d_multigpuarrayOffsetsPrefixSum{};
            SimpleAllocationDevice<int> d_numSelected{};
            CudaStream stream{};
            CudaEvent event{};

            SimpleAllocationDevice<char> d_cubTemp{};
            SimpleAllocationDevice<int> d_indices{};
            SimpleAllocationDevice<ArgIndexPair> d_selectedIndicesWithPositions{};
            SimpleAllocationDevice<char> d_dataCommunicationBuffer{};
        };

        struct PerCaller{
            PerCaller()
                : event(cudaEventDisableTiming)
            {                
            }

            CudaEvent event;
        };

        HandleStruct() = default;
        HandleStruct(HandleStruct&&) = default;
        HandleStruct(const HandleStruct&) = delete;

        std::map<int, PerDevice> deviceBuffers{};
        std::map<int, PerCaller> callerBuffers{};
    };

    using Handle = std::shared_ptr<HandleStruct>;

    Handle makeHandle() const{
        Handle handle = std::make_shared<HandleStruct>();

        for(const auto deviceId : dataDeviceIds){
            cub::SwitchDevice sd(deviceId);

            auto& deviceBuffers = handle->deviceBuffers[deviceId];

            deviceBuffers.d_multigpuarrayOffsets.resize(h_arrayOffsets.size());
            deviceBuffers.d_multigpuarrayOffsetsPrefixSum.resize(h_arrayOffsetsPrefixSum.size());

            cudaMemcpy(
                deviceBuffers.d_multigpuarrayOffsets.get(),
                h_arrayOffsets.get(),
                h_arrayOffsets.size() * sizeof(std::size_t),
                H2D
            );

            cudaMemcpy(
                deviceBuffers.d_multigpuarrayOffsetsPrefixSum.get(),
                h_arrayOffsetsPrefixSum.get(),
                h_arrayOffsetsPrefixSum.size() * sizeof(std::size_t),
                H2D
            );
        }

        for(const auto deviceId : callerDeviceIds){
            cub::SwitchDevice sd(deviceId);

            auto& deviceBuffers = handle->deviceBuffers[deviceId];

            handle->callerBuffers[deviceId]; //init

            deviceBuffers.d_multigpuarrayOffsets.resize(h_arrayOffsets.size());
            deviceBuffers.d_multigpuarrayOffsetsPrefixSum.resize(h_arrayOffsetsPrefixSum.size());

            cudaMemcpy(
                deviceBuffers.d_multigpuarrayOffsets.get(),
                h_arrayOffsets.get(),
                h_arrayOffsets.size() * sizeof(std::size_t),
                H2D
            );

            cudaMemcpy(
                deviceBuffers.d_multigpuarrayOffsetsPrefixSum.get(),
                h_arrayOffsetsPrefixSum.get(),
                h_arrayOffsetsPrefixSum.size() * sizeof(std::size_t),
                H2D
            );
        }        

        return handle;
    }

    void gather(
        Handle& handle, T* d_dest, 
        size_t destRowPitchInBytes, 
        const int* d_indices, 
        int numIndices, 
        int destDeviceId, 
        cudaStream_t destStream = 0
    ) const{
        if(numIndices == 0) return;

        cub::SwitchDevice sddest(destDeviceId);

        if(false && isSingleGpu() && gpuArrays[0]->getDeviceId() == destDeviceId){ //if all data to be gathered resides on the destination device
            auto indexGenerator = [d_indices] __device__ (auto i){
                return d_indices[i];
            };

            gpuArrays[0]->gather(d_dest, destRowPitchInBytes, indexGenerator, numIndices, destStream);
            return;
        }else{

            auto& callerBuffers = handle->callerBuffers[destDeviceId];
            auto& callerDeviceBuffer = handle->deviceBuffers[destDeviceId];

            cudaEventRecord(callerBuffers.event, destStream); CUERR;

            for(const auto& gpuArray : gpuArrays){
                const int deviceId = gpuArray->getDeviceId();
                cub::SwitchDevice sd(deviceId);

                auto& deviceBuffers = handle->deviceBuffers[deviceId];

                deviceBuffers.d_selectedIndicesWithPositions.resize(numIndices);
                deviceBuffers.d_indices.resize(numIndices);
                deviceBuffers.d_dataCommunicationBuffer.resize(numIndices * destRowPitchInBytes);

                cudaStreamWaitEvent(deviceBuffers.stream, callerBuffers.event, 0); CUERR;

                //copy indices to local buffer
                copy(
                    deviceBuffers.d_indices.get(), 
                    deviceId, 
                    d_indices, 
                    destDeviceId, 
                    sizeof(int) * numIndices,
                    deviceBuffers.stream
                );

                cub::ArgIndexInputIterator<int*, int> d_indicesWithPosition(deviceBuffers.d_indices.get());

                auto selectOp = [
                    deviceIdIndex = getDeviceIdIndex(deviceId), 
                    ps = deviceBuffers.d_multigpuarrayOffsetsPrefixSum.get()
                ] __device__ (auto indexPositionPair){
                    //key == position, value == index
                    return (ps[deviceIdIndex] <= indexPositionPair.value && indexPositionPair.value < ps[deviceIdIndex+1]);
                };

                size_t temp_storage_bytes = 0;
                cub::DeviceSelect::If(
                    nullptr, 
                    temp_storage_bytes, 
                    d_indicesWithPosition, 
                    deviceBuffers.d_selectedIndicesWithPositions.get(), 
                    deviceBuffers.d_numSelected.get(), 
                    numIndices, 
                    selectOp, 
                    deviceBuffers.stream
                );

                deviceBuffers.d_cubTemp.resize(temp_storage_bytes);

                //select indices for gpuArray
                cub::DeviceSelect::If(
                    deviceBuffers.d_cubTemp.get(), 
                    temp_storage_bytes, 
                    d_indicesWithPosition, 
                    deviceBuffers.d_selectedIndicesWithPositions.get(), 
                    deviceBuffers.d_numSelected.get(), 
                    numIndices, 
                    selectOp, 
                    deviceBuffers.stream
                );

                auto gatherIndexGenerator = [
                    deviceIdIndex = getDeviceIdIndex(deviceId), 
                    d_indicesWithPosition, ps = deviceBuffers.d_multigpuarrayOffsetsPrefixSum.get()
                ] __device__ (auto i){
                    const int index = d_indicesWithPosition[i].value;
                    return index - ps[deviceIdIndex]; //transform into local index for this gpuArray
                };

                gpuArray->gather(
                    (T*)deviceBuffers.d_dataCommunicationBuffer.get(), 
                    destRowPitchInBytes, 
                    gatherIndexGenerator, 
                    deviceBuffers.d_numSelected.get(),
                    numIndices, 
                    deviceBuffers.stream
                );

                cudaEventRecord(deviceBuffers.event, deviceBuffers.stream); CUERR;
            }

            //copy partial results from gpuArray to local buffer, and scatter into output buffer
            for(const auto& gpuArray : gpuArrays){
                const int deviceId = gpuArray->getDeviceId();

                auto& deviceBuffers = handle->deviceBuffers[deviceId];

                cudaStreamWaitEvent(destStream, deviceBuffers.event, 0);

                copy(
                    callerDeviceBuffer.d_numSelected.get(), 
                    destDeviceId, 
                    deviceBuffers.d_numSelected.get(), 
                    deviceId,                    
                    sizeof(int), 
                    destStream
                );

                copy(
                    callerDeviceBuffer.d_selectedIndicesWithPositions.get(), 
                    destDeviceId, 
                    deviceBuffers.d_selectedIndicesWithPositions.get(), 
                    deviceId,                    
                    sizeof(cub::ArgIndexInputIterator<int*, int>::value_type) * numIndices, 
                    destStream
                );

                copy(
                    callerDeviceBuffer.d_dataCommunicationBuffer.get(), 
                    destDeviceId, 
                    deviceBuffers.d_dataCommunicationBuffer.get(), 
                    deviceId,                     
                    destRowPitchInBytes * numIndices, 
                    destStream
                );

                dim3 block(128, 1, 1);
                dim3 grid(SDIV(numIndices, block.x), 1, 1);

                auto scatterIndexGenerator = [
                    d_indicesWithPosition = callerDeviceBuffer.d_selectedIndicesWithPositions.get(),
                    ps = deviceBuffers.d_multigpuarrayOffsetsPrefixSum.get()
                ] __device__ (auto i){
                    const int index = d_indicesWithPosition[i].key;
                    return index; //destination row
                };

                MultiGpu2dArrayKernels::scatterKernel<<<grid, block, 0, destStream>>>(
                    (const T*)callerDeviceBuffer.d_dataCommunicationBuffer.get(), 
                    destRowPitchInBytes, 
                    getNumColumns(),
                    d_dest, 
                    destRowPitchInBytes, 
                    scatterIndexGenerator, 
                    callerDeviceBuffer.d_numSelected.get()
                ); CUERR;
            }
        }
    }

    void gather(T* d_dest, size_t destRowPitchInBytes, size_t rowBegin, size_t rowEnd, cudaStream_t stream = 0) const{
        assert(false && "not implemented");
    }

    void scatter(
        Handle& handle, 
        const T* d_src, 
        size_t srcRowPitchInBytes, 
        const int* d_indices, 
        int numIndices, 
        int srcDeviceId, 
        cudaStream_t srcStream = 0
    ) const{
        if(numIndices == 0) return;

        cub::SwitchDevice sdsrc(srcDeviceId);

        if(isSingleGpu() && gpuArrays[0]->getDeviceId() == srcDeviceId){ //if all data to be gathered resides on the destination device
            auto indexGenerator = [d_indices] __device__ (auto i){
                return d_indices[i];
            };

            gpuArrays[0]->scatter(d_src, srcRowPitchInBytes, indexGenerator, numIndices, srcStream);
            return;
        }else{

            auto& callerBuffers = handle->callerBuffers[srcDeviceId];          
            auto& callerDeviceBuffer = handle->deviceBuffers[srcDeviceId];

            for(const auto& gpuArray : gpuArrays){
                const int deviceId = gpuArray->getDeviceId();

                callerDeviceBuffer.d_selectedIndicesWithPositions.resize(numIndices);
                callerDeviceBuffer.d_dataCommunicationBuffer.resize(numIndices * srcRowPitchInBytes);

                auto& targetDeviceBuffers = handle->deviceBuffers[deviceId];

                targetDeviceBuffers.d_selectedIndicesWithPositions.resize(numIndices);
                targetDeviceBuffers.d_dataCommunicationBuffer.resize(numIndices * srcRowPitchInBytes);

                cub::ArgIndexInputIterator<const int*, int> d_indicesWithPosition(d_indices);

                auto selectOp = [
                    deviceIdIndex = getDeviceIdIndex(deviceId), 
                    ps = callerDeviceBuffer.d_multigpuarrayOffsetsPrefixSum.get()
                ] __device__ (auto indexPositionPair){
                    //key == position, value == index
                    return (ps[deviceIdIndex] <= indexPositionPair.key && indexPositionPair.key < ps[deviceIdIndex+1]);
                };

                size_t temp_storage_bytes = 0;
                cub::DeviceSelect::If(
                    nullptr, 
                    temp_storage_bytes, 
                    d_indicesWithPosition, 
                    callerDeviceBuffer.d_selectedIndicesWithPositions.get(), 
                    callerDeviceBuffer.d_numSelected.get(), 
                    numIndices, 
                    selectOp, 
                    srcStream
                );

                callerDeviceBuffer.d_cubTemp.resize(temp_storage_bytes);

                //select indices for gpuArray
                cub::DeviceSelect::If(
                    callerDeviceBuffer.d_cubTemp.get(), 
                    temp_storage_bytes, 
                    d_indicesWithPosition, 
                    callerDeviceBuffer.d_selectedIndicesWithPositions.get(), 
                    callerDeviceBuffer.d_numSelected.get(), 
                    numIndices, 
                    selectOp, 
                    srcStream
                );

                auto gatherIndexGenerator = [
                    d_indicesWithPosition, 
                    ps = callerDeviceBuffer.d_multigpuarrayOffsetsPrefixSum.get()
                ] __device__ (auto i){
                    const int index = d_indicesWithPosition[i].key; // position in array
                    return index; 
                };

                //local gather data which should be scattered to current gpuArray
                dim3 block(128, 1, 1);
                dim3 grid(SDIV(numIndices, block.x), 1, 1);

                MultiGpu2dArrayKernels::gatherKernel<<<grid, block, 0, srcStream>>>(
                    d_src, 
                    srcRowPitchInBytes, 
                    getNumColumns(),
                    (T*)callerDeviceBuffer.d_dataCommunicationBuffer.get(), 
                    srcRowPitchInBytes, 
                    gatherIndexGenerator, 
                    callerDeviceBuffer.d_numSelected.get()
                ); CUERR;

                copy(
                    targetDeviceBuffers.d_numSelected.get(), 
                    deviceId, 
                    callerDeviceBuffer.d_numSelected.get(), 
                    srcDeviceId, 
                    sizeof(int), 
                    srcStream
                );

                copy(
                    targetDeviceBuffers.d_selectedIndicesWithPositions.get(), 
                    deviceId, 
                    callerDeviceBuffer.d_selectedIndicesWithPositions.get(), 
                    srcDeviceId, 
                    sizeof(cub::ArgIndexInputIterator<int*, int>::value_type) * numIndices, 
                    srcStream
                );

                copy(
                    targetDeviceBuffers.d_dataCommunicationBuffer.get(), 
                    deviceId, 
                    callerDeviceBuffer.d_dataCommunicationBuffer.get(), 
                    srcDeviceId, 
                    srcRowPitchInBytes * numIndices, 
                    srcStream
                );

                cudaEventRecord(callerBuffers.event, srcStream); CUERR;

                //scatter into target gpuArray
                cub::SwitchDevice sd(deviceId);

                cudaStreamWaitEvent(targetDeviceBuffers.stream, callerBuffers.event, 0); CUERR;

                auto scatterIndexGenerator = [
                    deviceIdIndex = getDeviceIdIndex(deviceId), 
                    d_indicesWithPosition, ps = targetDeviceBuffers.d_multigpuarrayOffsetsPrefixSum.get()
                ] __device__ (auto i){
                    const int index = d_indicesWithPosition[i].value;
                    return index - ps[deviceIdIndex]; //transform into local index for this gpuArray
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
            }

            //join multi gpu work to srcStream
            for(const auto& gpuArray : gpuArrays){
                const int deviceId = gpuArray->getDeviceId();
                cudaStreamWaitEvent(srcStream, handle->deviceBuffers[deviceId].event, 0); CUERR;
            }
        }
    }

    void scatter(const T* d_src, size_t srcRowPitchInBytes, size_t rowBegin, size_t rowEnd, cudaStream_t stream = 0) const{
        assert(false && "not implemented");
    }

    MemoryUsage getMemoryInfo() const{
        MemoryUsage result;
        result.host = 0;
        
        for(const auto& gpuArray : gpuArrays){
            result += gpuArray->getMemoryInfo();
        }

        return result;
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

private:

    void copy(
        void* dst, 
        int dstDevice, 
        const void* src, 
        int srcDevice, 
        size_t count, 
        cudaStream_t stream = 0
    ) const{
        if(dstDevice == srcDevice && dst == src) return;

        cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream); CUERR;
    }

    void init(
        size_t numRows, 
        size_t numColumns, 
        std::vector<size_t> memoryLimits, 
        Layout layout
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

        if(layout == Layout::EvenShare){
            std::cerr << "Layout::EvenShare not implemented. Will use Layout::FirstFit\n";
        }

        std::vector<size_t> rowsPerGpu(numGpus);

        size_t remaining = numRows;
        for(int i = 0; i < numGpus; i++){
            size_t myrows = std::min(maxRowsPerGpu[i], remaining);
            rowsPerGpu[i] = myrows;

            remaining -= myrows;
        }

        assert(remaining == 0);

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
    }

    int getDeviceIdIndex(int deviceId) const{
        auto it = std::find(dataDeviceIds.begin(), dataDeviceIds.end(), deviceId);
        if(it != dataDeviceIds.end()){
            return std::distance(dataDeviceIds.begin(), it);
        }else{
            assert(false);
            return -1;
        }
    }

    size_t numRows{};
    size_t numColumns{};
    size_t alignmentInBytes{};
    std::vector<std::unique_ptr<Gpu2dArrayManaged<T>>> gpuArrays{};

    std::vector<int> dataDeviceIds;
    std::vector<int> callerDeviceIds;

    SimpleAllocationPinnedHost<size_t> h_arrayOffsets;
    SimpleAllocationPinnedHost<size_t> h_arrayOffsetsPrefixSum;
};





}

#endif