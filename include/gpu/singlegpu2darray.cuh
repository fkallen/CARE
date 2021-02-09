#ifndef SINGLE_GPU_ARRAY_HPP
#define SINGLE_GPU_ARRAY_HPP

#include <hpc_helpers.cuh>

#include <2darray.hpp>
#include <memorymanagement.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include <cooperative_groups.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;




namespace Gpu2dArrayManagedKernels{

    template<class T, class IndexGenerator>
    __global__
    void scatterkernel(TwoDimensionalArray<T> array, const T* __restrict__ src, size_t srcRowPitchInBytes, IndexGenerator indices, const size_t* __restrict__ d_numIndices){
        auto gridGroup = cg::this_grid();

        const size_t numIndices = *d_numIndices;

        if(numIndices == 0) return;

        array.scatter(gridGroup, src, srcRowPitchInBytes, indices, numIndices);
    }

    template<class T, class IndexGenerator>
    __global__
    void scatterkernel(TwoDimensionalArray<T> array, const T* __restrict__ src, size_t srcRowPitchInBytes, IndexGenerator indices, size_t numIndices){
        auto gridGroup = cg::this_grid();

        if(numIndices == 0) return;

        array.scatter(gridGroup, src, srcRowPitchInBytes, indices, numIndices);
    }

    template<class T, class IndexGenerator>
    __global__
    void gatherkernel(TwoDimensionalArray<T> array, T* __restrict__ dest, size_t destRowPitchInBytes, IndexGenerator indices, const size_t* __restrict__ d_numIndices){
        auto gridGroup = cg::this_grid();

        const size_t numIndices = *d_numIndices;

        if(numIndices == 0) return;

        array.gather(gridGroup, dest, destRowPitchInBytes, indices, numIndices);
    }

    template<class T, class IndexGenerator>
    __global__
    void gatherkernel(TwoDimensionalArray<T> array, T* __restrict__ dest, size_t destRowPitchInBytes, IndexGenerator indices, size_t numIndices){
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
        cudaGetDevice(&deviceId); CUERR;
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
        cudaMemcpy(arraydata, rhs.arraydata, numRows * rowPitchInBytes, D2D); CUERR;
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

        cudaGetDevice(&deviceId); CUERR;

        rowPitchInBytes = computePitch(numColumns, alignmentInBytes);

        if(numRows > 0 && numColumns > 0){

            cudaMalloc(&arraydata, numRows * rowPitchInBytes); CUERR;
            cudaMemset(arraydata, 0, numRows * rowPitchInBytes); CUERR;

        }
    }

    static size_t computePitch(size_t numberOfColumns, size_t alignment) noexcept{
        const size_t minbytesPerRow = sizeof(T) * numberOfColumns;
        return SDIV(minbytesPerRow, alignment) * alignment;
    }

    void destroy(){
        cub::SwitchDevice sd{deviceId};

        cudaFree(arraydata); CUERR;
        
        numRows = 0;
        numColumns = 0;
        alignmentInBytes = 0;
        rowPitchInBytes = 0;
        arraydata = nullptr;
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
        cudaMallocHost(&tmp, numRows * numColumns); CUERR;
        gather(tmp, numColumns * sizeof(T), [=]__device__(auto i){return i;}, numRows);
        cudaDeviceSynchronize(); CUERR;

        for(size_t i = 0; i < numRows; i++){
            for(size_t k = 0; k < numColumns; k++){
                std::cerr << tmp[i * numColumns + k] << " ";
            }
            std::cerr << "\n";
        }

        cudaFreeHost(tmp); CUERR;
    }

    template<class IndexGenerator>
    void gather(T* d_dest, size_t destRowPitchInBytes, IndexGenerator d_indices, size_t numIndices, cudaStream_t stream = 0) const{
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

        CUERR;
    }

    template<class IndexGenerator>
    void gather(T* d_dest, size_t destRowPitchInBytes, IndexGenerator d_indices, const size_t* d_numIndices, size_t maxNumIndices, cudaStream_t stream = 0) const{
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

        CUERR;
    }

    void gather(T* d_dest, size_t destRowPitchInBytes, size_t rowBegin, size_t rowEnd, cudaStream_t stream = 0) const{
        const size_t rows = rowEnd - rowBegin;

        if(rows == 0) return;
        if(getNumRows() == 0) return;

        dim3 block(128, 1, 1);
        dim3 grid(SDIV(rows * numColumns, block.x), 1, 1);

        TwoDimensionalArray<T> array = wrapper();

        auto indexgenerator = [rowBegin] __device__ (auto i){
            return rowBegin + i;
        };

        Gpu2dArrayManagedKernels::gatherkernel<<<grid, block, 0, stream>>>(
            array, 
            d_dest, 
            destRowPitchInBytes, 
            indexgenerator, 
            rows
        );

        CUERR;
    }


    template<class IndexGenerator>
    void scatter(const T* d_src, size_t srcRowPitchInBytes, IndexGenerator d_indices, size_t numIndices, cudaStream_t stream = 0) const{
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

        CUERR;
    }

    template<class IndexGenerator>
    void scatter(const T* d_src, size_t srcRowPitchInBytes, IndexGenerator d_indices, const size_t* d_numIndices, size_t maxNumIndices, cudaStream_t stream = 0) const{
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

        CUERR;
    }

    void scatter(const T* d_src, size_t srcRowPitchInBytes, size_t rowBegin, size_t rowEnd, cudaStream_t stream = 0) const{
        const size_t rows = rowEnd - rowBegin;
        if(rows == 0) return;
        if(getNumRows() == 0) return;

        dim3 block(128, 1, 1);
        dim3 grid(SDIV(rows * numColumns, block.x), 1, 1);

        TwoDimensionalArray<T> array = wrapper();

        auto indexgenerator = [rowBegin] __device__ (auto i){
            return rowBegin + i;
        };

        Gpu2dArrayManagedKernels::scatterkernel<<<grid, block, 0, stream>>>(
            array, 
            d_src, 
            srcRowPitchInBytes, 
            indexgenerator, 
            rows
        );

        CUERR;
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