#ifndef CARE_GPU_ARRAY_HPP
#define CARE_GPU_ARRAY_HPP

#include <hpc_helpers.cuh>
#include <gpu/coopgrouphelpers.hpp>

#include <algorithm>
#include <cassert>
#include <vector>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;



namespace care{


template<class T>
class TwoDimensionalArray{
public:

    using value_type = T;

    TwoDimensionalArray() = default;

    HOSTDEVICEQUALIFIER
    TwoDimensionalArray(
        size_t numRows,
        size_t numColumns,
        size_t rowPitchInBytes,
        T* arraydata
    ) :
        numRows(numRows),
        numColumns(numColumns),
        rowPitchInBytes(rowPitchInBytes),
        arraydata(arraydata){
    }

    //gather rows from array to dest
    template<class Group>
    HOSTDEVICEQUALIFIER
    void gather(Group& group, T* __restrict__ dest, size_t destRowPitchInBytes, const int* __restrict__ indices, int numIndices){
        const size_t elementsToCopy = numIndices * numColumns;

        for(size_t i = group.thread_rank(); i < elementsToCopy; i += group.size()){
            const size_t outputRow = i / numColumns;
            const size_t inputRow = indices[outputRow];
            const size_t column = i % numColumns;
            
            ((T*)(((char*)dest) + outputRow * destRowPitchInBytes))[column] 
                = ((const T*)(((const char*)arraydata) + inputRow * rowPitchInBytes))[column];
        }
    }

    //scatter rows from src into array
    template<class Group>
    HOSTDEVICEQUALIFIER
    void scatter(Group& group, const T* __restrict__ src, size_t srcRowPitchInBytes, const int* __restrict__ indices, int numIndices){
        const size_t elementsToCopy = numIndices * numColumns;

        for(size_t i = group.thread_rank(); i < elementsToCopy; i += group.size()){
            const size_t inputRow = i / numColumns;
            const size_t outputRow = indices[inputRow];
            const size_t column = i % numColumns;
            
            ((T*)(((char*)arraydata) + outputRow * rowPitchInBytes))[column] 
                = ((const T*)(((const char*)src) + inputRow * srcRowPitchInBytes))[column];
        }
    }

private:
    size_t numRows;
    size_t numColumns;
    size_t rowPitchInBytes;
    T* arraydata;
};


namespace Gpu2dArrayManagedKernels{

    template<class T>
    __global__
    void scatterkernel(TwoDimensionalArray<T> array, const T* __restrict__ src, size_t srcRowPitchInBytes, const int* __restrict__ indices, int numIndices){
        auto gridGroup = cg::this_grid();

        array.scatter(gridGroup, src, srcRowPitchInBytes, indices, numIndices);
    }

    template<class T>
    __global__
    void gatherkernel(TwoDimensionalArray<T> array, T* __restrict__ dest, size_t destRowPitchInBytes, const int* __restrict__ indices, int numIndices){
        auto gridGroup = cg::this_grid();

        array.gather(gridGroup, dest, destRowPitchInBytes, indices, numIndices);
    }
}

template<class T>
class Gpu2dArrayManaged{
public:
    Gpu2dArrayManaged() = default;

    Gpu2dArrayManaged(size_t numRows, size_t numColumns, size_t alignment)
        : numRows(numRows), numColumns(numColumns), rowPitchInBytes(rowPitchInBytes)
    {
        assert(alignment % sizeof(T) == 0);

        cudaGetDevice(&deviceId); CUERR;
        
        const size_t minbytesPerRow = sizeof(T) * numColumns;
        rowPitchInBytes = SDIV(minbytesPerRow, alignment) * alignment;

        cudaMalloc(&arraydata, numRows * rowPitchInBytes); CUERR;
        cudaMemset(arraydata, 0, numRows * rowPitchInBytes); CUERR;
    }

    ~Gpu2dArrayManaged(){
        int current;
        cudaGetDevice(&current); CUERR;
        cudaSetDevice(deviceId); CUERR;

        cudaFree(arraydata); CUERR;
        arraydata = nullptr;

        cudaSetDevice(current);
    }

    Gpu2dArrayManaged(const Gpu2dArrayManaged& rhs)
        : Gpu2dArrayManaged(rhs.numRows, rhs.numColumns, rhs.rowPitchInBytes)
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

    TwoDimensionalArray<T> wrapper() const noexcept{
        TwoDimensionalArray<T> wrapper(numRows, numColumns, rowPitchInBytes, arraydata);

        return wrapper;
    }

    operator TwoDimensionalArray<T>() const{
        return wrapper();
    }

    void gather(T* d_dest, size_t destRowPitchInBytes, const int* d_indices, int numIndices, cudaStream_t stream = 0){
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

    void scatter(const T* d_src, size_t srcRowPitchInBytes, const int* d_indices, int numIndices, cudaStream_t stream = 0){
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

private:
    int deviceId;
    size_t numRows;
    size_t numColumns;
    size_t rowPitchInBytes;
    T* arraydata;   
};




template<class T>
class Cpu2dArrayManaged{
public:
    Cpu2dArrayManaged() = default;

    Cpu2dArrayManaged(size_t numRows, size_t numColumns, size_t alignment)
        : numRows(numRows), numColumns(numColumns), rowPitchInBytes(rowPitchInBytes)
    {
        assert(alignment % sizeof(T) == 0);
        
        const size_t minbytesPerRow = sizeof(T) * numColumns;
        rowPitchInBytes = SDIV(minbytesPerRow, alignment) * alignment;

        arraydata.resize(SDIV(rowPitchInBytes, sizeof(T)) * sizeof(T), T{});
    }

    ~Cpu2dArrayManaged(){
    }

    Cpu2dArrayManaged(const Cpu2dArrayManaged& rhs)
        : Cpu2dArrayManaged(rhs.numRows, rhs.numColumns, rhs.rowPitchInBytes)
    {
        std::copy(rhs.arraydata.begin(), rhs.arraydata.end(), arraydata.begin());
    }

    Cpu2dArrayManaged(Cpu2dArrayManaged&& other) noexcept
        : Cpu2dArrayManaged()
    {
        swap(*this, other);
    }

    Cpu2dArrayManaged& operator=(Cpu2dArrayManaged other){
        swap(*this, other);

        return *this;
    }

    friend void swap(Cpu2dArrayManaged& l, Cpu2dArrayManaged& r) noexcept
    {
        std::swap(l.numRows, r.numRows);
        std::swap(l.numColumns, r.numColumns);
        std::swap(l.rowPitchInBytes, r.rowPitchInBytes);
        std::swap(l.arraydata, r.arraydata);
    }

    TwoDimensionalArray<T> wrapper() const noexcept{
        TwoDimensionalArray<T> wrapper(numRows, numColumns, rowPitchInBytes, arraydata.data());

        return wrapper;
    }

    operator TwoDimensionalArray<T>() const{
        return wrapper();
    }

    void gather(T* h_dest, size_t destRowPitchInBytes, const int* h_indices, int numIndices){
        TwoDimensionalArray<T> array = wrapper();
        SingleThreadGroup group{};

        array.gather(
            group, 
            h_dest, 
            destRowPitchInBytes, 
            h_indices, 
            numIndices
        );
    }

    void scatter(const T* h_src, size_t srcRowPitchInBytes, const int* h_indices, int numIndices){
        TwoDimensionalArray<T> array = wrapper();
        SingleThreadGroup group{};

        array.scatter(
            group,
            h_src, 
            srcRowPitchInBytes, 
            h_indices, 
            numIndices
        );
    }

private:
    size_t numRows;
    size_t numColumns;
    size_t rowPitchInBytes;
    mutable std::vector<T> arraydata;   
};







}


#endif

