#ifndef CARE_MULTI_GPU_ARRAY_HPP
#define CARE_MULTI_GPU_ARRAY_HPP


#include <gpu/gpuarray.hpp>

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

#include <gpu/simpleallocation.cuh>

#include <cub/cub.cuh>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace care{

namespace MultiGpu2dArrayKernels{

    template<class T, class IndexGenerator>
    void scatter(
        const T* __restrict__ src, 
        size_t srcRowPitchInBytes, 
        size_t numColumns,
        T* __restrict__ dest, 
        size_t destRowPitchInBytes, 
        IndexGenerator indices, 
        const int* numIndicesPtr
    ){
        const int numIndices = *numIndicesPtr;

        auto group = cg::this_grid();

        const size_t elementsToCopy = numIndices * numColumns;

        for(size_t i = group.thread_rank(); i < elementsToCopy; i += group.size()){
            const size_t inputRow = i / numColumns;
            const size_t outputRow = indices(inputRow);
            const size_t column = i % numColumns;
            
            ((T*)(((char*)arraydata) + outputRow * destRowPitchInBytes))[column] 
                = ((const T*)(((const char*)src) + inputRow * srcRowPitchInBytes))[column];
        }
    }
}

template<class T>
class MultiGpu2dArray{
public:

    MultiGpu2dArray()
    : alignmentInBytes(sizeof(T)){

    }

    MultiGpu2dArray(size_t numRows, size_t numColumns, size_t alignmentInBytes)
    : alignmentInBytes(alignmentInBytes){

        assert(alignmentInBytes > 0);
        assert(alignmentInBytes % sizeof(T) == 0);

        init(numRows, numColumns);
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

    void init(size_t numRows, size_t numColumns){
        assert(numRows > 0);
        assert(numColumns > 0);

        this->numRows = numRows;
        this->numColumns = numColumns;

        //init gpu arrays
    }

    void destroy(){
        numRows = 0;
        numColumns = 0;
        gpuArrays.clear();
    }

    struct HandleStruct{
        struct PerDevice{
            using ArgIndexPair = cub::ArgIndexInputIterator<int*, int>::value_type;

            SimpleAllocationDevice<size_t> d_multigpuarrayOffsets;
            SimpleAllocationDevice<size_t> d_multigpuarrayOffsetsPrefixSum;
            SimpleAllocationDevice<int> d_numSelected;
            cudaStream_t stream;
            cudaEvent_t event;

            SimpleAllocationDevice<char> d_cubTemp;            
            SimpleAllocationDevice<ArgIndexPair> d_selectedIndicesWithPositions;
            SimpleAllocationDevice<char> d_gatheredData;
        };

        struct PerDestination{
            cudaEvent_t event;
            SimpleAllocationDevice<char> d_gatheredData;
        };

        std::map<int, PerDevice> deviceBuffers;
        std::map<int, PerDestination> destinationBuffers;
    };

    using Handle = std::shared_ptr<HandleStruct>;

    void gather(Handle& handle, T* d_dest, size_t destRowPitchInBytes, const int* d_indices, int numIndices, int destDeviceId, cudaStream_t destStream = 0) const{
        if(numIndices == 0) return;

        cub::SwitchDevice sddest(destDeviceId);

        if(isSingleGpu() && gpuArray[0]->getDeviceId() == destDeviceId){ //if all data to be gathered resides on the destination device
            auto indexGenerator = [d_indices] __device__ (auto i){
                return d_indices[i];
            };

            gpuArrays[0]->gather(d_dest, destRowPitchInBytes, indexGenerator, numIndices, stream);
            return;
        }else{

            auto& destinationBuffers = handle->destinationBuffers[destDeviceId];
            cudaEventRecord(destinationBuffers.event, stream); CUERR;

            for(const auto& gpuArray : gpuArrays){
                const int deviceId = gpuArray->getDeviceId();
                cub::SwitchDevice sd(deviceId);

                auto& deviceBuffers = handle->deviceBuffers[deviceId];

                deviceBuffers.d_selectedIndicesWithPositions.resize(numIndices);
                deviceBuffers.d_gatheredData.resize(numIndices * destRowPitchInBytes);

                cudaStreamWaitEvent(deviceBuffers.stream, destinationBuffers.event, 0); CUERR;

                cub::ArgIndexInputIterator<int*, int> d_indicesWithPosition(d_indices);

                auto selectOp = [
                    deviceId, ps = deviceBuffers.d_multigpuarrayOffsetsPrefixSum.get()
                ] __device__ (auto indexPositionPair){
                    //key == position, value == index
                    return (ps[deviceId] <= indexPositionPair.key && indexPositionPair.key < ps[deviceId+1]);
                };

                //TODO maybe copy indices to this gpu before selecting

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
                    nullptr, 
                    temp_storage_bytes, 
                    d_indicesWithPosition, 
                    deviceBuffers.d_selectedIndicesWithPositions.get(), 
                    deviceBuffers.d_numSelected.get(), 
                    numIndices, 
                    selectOp, 
                    deviceBuffers.stream
                );

                auto gatherIndexGenerator = [
                    deviceId, d_indicesWithPosition, ps = deviceBuffers.d_multigpuarrayOffsetsPrefixSum.get()
                ] __device__ (auto i){
                    const int index = d_indicesWithPosition[i].value;
                    return index - ps[deviceId]; //transform into local index for this gpuArray
                };

                gpuArray->gather(
                    (T*)deviceBuffers.d_gatheredData.get(), 
                    destRowPitchInBytes, 
                    gatherIndexGenerator, 
                    deviceBuffers.d_numSelected.get(),
                    numIndices, 
                    deviceBuffers.stream
                );

                //TODO maybe copy to target gpu first, then scatter on target gpu

                dim3 block(128, 1, 1);
                dim3 grid(SDIV(numIndices, block.x), 1, 1);

                auto scatterIndexGenerator = [
                    deviceId, d_indicesWithPosition, ps = deviceBuffers.d_multigpuarrayOffsetsPrefixSum.get()
                ] __device__ (auto i){
                    const int index = d_indicesWithPosition[i].key;
                    return index; //destination row
                };

                scatter<<<grid, block, 0, deviceBuffers.stream>>>(
                    deviceBuffers.d_gatheredData.get(), 
                    destRowPitchInBytes, 
                    numColumns,
                    d_dest, 
                    destRowPitchInBytes, 
                    scatterIndexGenerator, 
                    deviceBuffers.d_numSelected.get()
                ); CUERR;

                cudaEventRecord(deviceBuffers.event, deviceBuffers.stream); CUERR;
            }

            for(const auto& gpuArray : gpuArrays){
                const int deviceId = gpuArray->getDeviceId();
                cudaStreamWaitEvent(destStream, handle->deviceBuffers[deviceId].event, 0); CUERR;
            }
        }
    }

    void gather(T* d_dest, size_t destRowPitchInBytes, size_t rowBegin, size_t rowEnd, cudaStream_t stream = 0) const{
        assert(false && "not implemented");
    }

    void scatter(const T* d_src, size_t srcRowPitchInBytes, const int* d_indices, int numIndices, cudaStream_t stream = 0) const{
        if(numIndices == 0) return;

        dim3 block(128, 1, 1);
        dim3 grid(SDIV(numIndices * numColumns, block.x), 1, 1);

        TwoDimensionalArray<T> array = wrapper();

        Gpu2dArrayManagedKernels::scatterkernel<<<grid, block, 0, stream>>>(
            array, 
            d_src, 
            srcRowPitchInBytes, 
            d_indices, 
            numIndices
        );

        CUERR;
    }

    void scatter(const T* d_src, size_t srcRowPitchInBytes, size_t rowBegin, size_t rowEnd, cudaStream_t stream = 0) const{
        const size_t rows = rowEnd - rowBegin;
        if(rows == 0) return;

        dim3 block(128, 1, 1);
        dim3 grid(SDIV(rows * numColumns, block.x), 1, 1);

        TwoDimensionalArray<T> array = wrapper();

        Gpu2dArrayManagedKernels::scatterkernel<<<grid, block, 0, stream>>>(
            array, 
            d_src, 
            srcRowPitchInBytes, 
            rowBegin, 
            rowEnd
        );

        CUERR;
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

    size_t numRows{};
    size_t numColumns{};
    size_t alignmentInBytes{};
    std::vector<std::unique_ptr<Gpu2dArrayManaged>> gpuArrays{};

    SimpleAllocationPinnedHost<size_t> h_arrayOffsets;
    SimpleAllocationPinnedHost<size_t> h_arrayOffsetsPrefixSum;
}





}

#endif