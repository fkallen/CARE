#ifndef SINGLE_GPU_ARRAY_HPP
#define SINGLE_GPU_ARRAY_HPP

#include <hpc_helpers.cuh>
#include <gpu/cudaerrorcheck.cuh>

#include <2darray.hpp>
#include <memorymanagement.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <memory>

#include <cooperative_groups.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;




namespace Gpu2dArrayManagedKernels{

    template<class T, class IndexIterator>
    __global__
    void scatterkernel(TwoDimensionalArray<T> array, const T* __restrict__ src, size_t srcRowPitchInBytes, IndexIterator indices, const size_t* __restrict__ d_numIndices){
        auto gridGroup = cg::this_grid();

        const size_t numIndices = *d_numIndices;

        if(numIndices == 0) return;

        array.scatter(gridGroup, src, srcRowPitchInBytes, indices, numIndices);
    }

    template<class T, class IndexIterator>
    __global__
    void scatterkernel(TwoDimensionalArray<T> array, const T* __restrict__ src, size_t srcRowPitchInBytes, IndexIterator indices, size_t numIndices){
        auto gridGroup = cg::this_grid();

        if(numIndices == 0) return;

        array.scatter(gridGroup, src, srcRowPitchInBytes, indices, numIndices);
    }

    template<class T, class IndexIterator>
    __global__
    void gatherkernel(TwoDimensionalArray<T> array, T* __restrict__ dest, size_t destRowPitchInBytes, IndexIterator indices, const size_t* __restrict__ d_numIndices){
        auto gridGroup = cg::this_grid();

        const size_t numIndices = *d_numIndices;

        if(numIndices == 0) return;

        array.gather(gridGroup, dest, destRowPitchInBytes, indices, numIndices);
    }

    template<class T, class IndexIterator>
    __global__
    void gatherkernel(TwoDimensionalArray<T> array, T* __restrict__ dest, size_t destRowPitchInBytes, IndexIterator indices, size_t numIndices){
        auto gridGroup = cg::this_grid();

        if(numIndices == 0) return;

        array.gather(gridGroup, dest, destRowPitchInBytes, indices, numIndices);
    }

}

template<class T>
class Gpu2dArrayManaged{
public:

    Gpu2dArrayManaged()
    : alignmentInBytes(sizeof(T)){
        CUDACHECK(cudaGetDevice(&deviceId));
    }

    Gpu2dArrayManaged(size_t numRows, size_t numColumns, size_t alignmentInBytes)
    : alignmentInBytes(alignmentInBytes){

        assert(alignmentInBytes > 0);
        //assert(alignmentInBytes % sizeof(T) == 0);

        init(numRows, numColumns);
    }

    ~Gpu2dArrayManaged(){
        destroy();
    }

    Gpu2dArrayManaged(const Gpu2dArrayManaged& rhs)
        : Gpu2dArrayManaged(rhs.numRows, rhs.numColumns, rhs.alignmentInBytes)
    {
        CUDACHECK(cudaMemcpy(arraydata, rhs.arraydata, numRows * rowPitchInBytes, D2D));
    }

    Gpu2dArrayManaged(Gpu2dArrayManaged&& other) noexcept
        : Gpu2dArrayManaged()
    {
        swap(*this, other);
    }

    Gpu2dArrayManaged& operator=(Gpu2dArrayManaged other){
        swap(*this, other);

        return *this;
    }

    friend void swap(Gpu2dArrayManaged& l, Gpu2dArrayManaged& r) noexcept
    {
        std::swap(l.deviceId, r.deviceId);
        std::swap(l.numRows, r.numRows);
        std::swap(l.numColumns, r.numColumns);
        std::swap(l.rowPitchInBytes, r.rowPitchInBytes);
        std::swap(l.arraydata, r.arraydata);
    }

    void init(size_t numRows, size_t numColumns){
        assert(numRows > 0);
        assert(numColumns > 0);

        this->numRows = numRows;
        this->numColumns = numColumns;

        CUDACHECK(cudaGetDevice(&deviceId));

        rowPitchInBytes = computePitch(numColumns, alignmentInBytes);

        if(numRows > 0 && numColumns > 0){

            CUDACHECK(cudaMalloc(&arraydata, numRows * rowPitchInBytes));
            CUDACHECK(cudaMemset(arraydata, 0, numRows * rowPitchInBytes));

        }
    }

    static size_t computePitch(size_t numberOfColumns, size_t alignment) noexcept{
        const size_t minbytesPerRow = sizeof(T) * numberOfColumns;
        return SDIV(minbytesPerRow, alignment) * alignment;
    }

    void destroy(){
        cub::SwitchDevice sd{deviceId};

        CUDACHECK(cudaFree(arraydata));
        
        numRows = 0;
        numColumns = 0;
        alignmentInBytes = 0;
        rowPitchInBytes = 0;
        arraydata = nullptr;
    }

    std::unique_ptr<Gpu2dArrayManaged> makeCopy(int targetDeviceid) const{
        cub::SwitchDevice sd{targetDeviceid};

        auto result = std::make_unique<Gpu2dArrayManaged>();

        result->numRows = numRows;
        result->numColumns = numColumns;
        result->alignmentInBytes = alignmentInBytes;
        result->rowPitchInBytes = rowPitchInBytes;
        result->arraydata = nullptr;

        cudaError_t status = cudaMalloc(&result->arraydata, numRows * rowPitchInBytes);
        if(status != cudaSuccess){
            return nullptr;
        }
        status = cudaMemcpyAsync(result->arraydata, arraydata, numRows * rowPitchInBytes, D2D, cudaStreamPerThread);
        if(status != cudaSuccess){
            return nullptr;
        }
        status = cudaStreamSynchronize(cudaStreamPerThread);

        if(status != cudaSuccess){
            return nullptr;
        }else{
            return result;
        }
    }

    TwoDimensionalArray<T> wrapper() const noexcept{
        TwoDimensionalArray<T> wrapper(numRows, numColumns, rowPitchInBytes, arraydata);

        return wrapper;
    }

    operator TwoDimensionalArray<T>() const{
        return wrapper();
    }

    void print() const{
        T* tmp;
        CUDACHECK(cudaMallocHost(&tmp, numRows * numColumns));
        gather(tmp, numColumns * sizeof(T), thrust::make_counting_iterator<std::size_t>(0), numRows);
        CUDACHECK(cudaDeviceSynchronize());

        for(size_t i = 0; i < numRows; i++){
            for(size_t k = 0; k < numColumns; k++){
                std::cerr << tmp[i * numColumns + k] << " ";
            }
            std::cerr << "\n";
        }

        CUDACHECK(cudaFreeHost(tmp));
    }

    template<class IndexIterator>
    void gather(T* d_dest, size_t destRowPitchInBytes, IndexIterator d_indices, size_t numIndices, cudaStream_t stream = 0) const{
        if(numIndices == 0) return;
        if(getNumRows() == 0) return;

        dim3 block(128, 1, 1);
        dim3 grid(SDIV(numIndices * numColumns, block.x), 1, 1);

        TwoDimensionalArray<T> array = wrapper();

        Gpu2dArrayManagedKernels::gatherkernel<<<grid, block, 0, stream>>>(
            array, 
            d_dest, 
            destRowPitchInBytes, 
            d_indices, 
            numIndices
        );

        CUDACHECKASYNC;
    }

    template<class IndexIterator>
    void gather(T* d_dest, size_t destRowPitchInBytes, IndexIterator d_indices, const size_t* d_numIndices, size_t maxNumIndices, cudaStream_t stream = 0) const{
        if(maxNumIndices == 0) return;
        if(getNumRows() == 0) return;

        dim3 block(128, 1, 1);
        dim3 grid(SDIV(maxNumIndices * numColumns, block.x), 1, 1);

        TwoDimensionalArray<T> array = wrapper();

        Gpu2dArrayManagedKernels::gatherkernel<<<grid, block, 0, stream>>>(
            array, 
            d_dest, 
            destRowPitchInBytes, 
            d_indices, 
            d_numIndices
        );

        CUDACHECKASYNC;
    }

    void gather(T* d_dest, size_t destRowPitchInBytes, size_t rowBegin, size_t rowEnd, cudaStream_t stream = 0) const{
        const size_t rows = rowEnd - rowBegin;

        if(rows == 0) return;
        if(getNumRows() == 0) return;

        dim3 block(128, 1, 1);
        dim3 grid(SDIV(rows * numColumns, block.x), 1, 1);

        TwoDimensionalArray<T> array = wrapper();

        auto IndexIterator = thrust::make_counting_iterator<std::size_t>(rowBegin);

        Gpu2dArrayManagedKernels::gatherkernel<<<grid, block, 0, stream>>>(
            array, 
            d_dest, 
            destRowPitchInBytes, 
            IndexIterator, 
            rows
        );

        CUDACHECKASYNC;
    }


    template<class IndexIterator>
    void scatter(const T* d_src, size_t srcRowPitchInBytes, IndexIterator d_indices, size_t numIndices, cudaStream_t stream = 0) const{
        if(numIndices == 0) return;
        if(getNumRows() == 0) return;

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

        CUDACHECKASYNC;
    }

    template<class IndexIterator>
    void scatter(const T* d_src, size_t srcRowPitchInBytes, IndexIterator d_indices, const size_t* d_numIndices, size_t maxNumIndices, cudaStream_t stream = 0) const{
        if(maxNumIndices == 0) return;
        if(getNumRows() == 0) return;

        dim3 block(128, 1, 1);
        dim3 grid(SDIV(maxNumIndices * numColumns, block.x), 1, 1);

        TwoDimensionalArray<T> array = wrapper();

        Gpu2dArrayManagedKernels::scatterkernel<<<grid, block, 0, stream>>>(
            array, 
            d_src, 
            srcRowPitchInBytes, 
            d_indices, 
            d_numIndices
        );

        CUDACHECKASYNC;
    }

    void scatter(const T* d_src, size_t srcRowPitchInBytes, size_t rowBegin, size_t rowEnd, cudaStream_t stream = 0) const{
        const size_t rows = rowEnd - rowBegin;
        if(rows == 0) return;
        if(getNumRows() == 0) return;

        dim3 block(128, 1, 1);
        dim3 grid(SDIV(rows * numColumns, block.x), 1, 1);

        TwoDimensionalArray<T> array = wrapper();

        auto IndexIterator = thrust::make_counting_iterator<std::size_t>(rowBegin);

        Gpu2dArrayManagedKernels::scatterkernel<<<grid, block, 0, stream>>>(
            array, 
            d_src, 
            srcRowPitchInBytes, 
            IndexIterator, 
            rows
        );

        CUDACHECKASYNC;
    }

    int getDeviceId() const noexcept{
        return deviceId;
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

    T* getGpuData() const noexcept{
        return arraydata;
    }

    size_t getPitch() const noexcept{
        return rowPitchInBytes;
    }

    MemoryUsage getMemoryInfo() const{
        MemoryUsage result{};

        result.host = 0;
        result.device[getDeviceId()] = getPitch() * getNumRows();

        return result;
    }

private:
    int deviceId{};
    size_t numRows{};
    size_t numColumns{};
    size_t alignmentInBytes{};
    size_t rowPitchInBytes{};
    T* arraydata{};   
};




#endif