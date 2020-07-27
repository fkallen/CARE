#ifndef CARE_GPU_ARRAY_HPP
#define CARE_GPU_ARRAY_HPP

#include <hpc_helpers.cuh>
#include <gpu/coopgrouphelpers.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include <cooperative_groups.h>
#include <cub/cub.cuh>

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

    template<class Group, class IndexGenerator>
    HOSTDEVICEQUALIFIER
    void gather(Group& group, T* __restrict__ dest, size_t destRowPitchInBytes, IndexGenerator indices, int numIndices){
        const size_t elementsToCopy = numIndices * numColumns;

        for(size_t i = group.thread_rank(); i < elementsToCopy; i += group.size()){
            const size_t outputRow = i / numColumns;
            const size_t inputRow = indices(outputRow);
            const size_t column = i % numColumns;
            
            ((T*)(((char*)dest) + outputRow * destRowPitchInBytes))[column] 
                = ((const T*)(((const char*)arraydata) + inputRow * rowPitchInBytes))[column];
        }
    }

    //gather rows from array to dest
    template<class Group>
    HOSTDEVICEQUALIFIER
    void gather(Group& group, T* __restrict__ dest, size_t destRowPitchInBytes, size_t rowBegin, size_t rowEnd){
        const size_t numRows = rowEnd - rowBegin;
        const size_t elementsToCopy = numRows * numColumns;

        for(size_t i = group.thread_rank(); i < elementsToCopy; i += group.size()){
            const size_t outputRow = i / numColumns;
            const size_t inputRow = rowBegin + outputRow;
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

    //scatter rows from src into array
    template<class Group>
    HOSTDEVICEQUALIFIER
    void scatter(Group& group, const T* __restrict__ src, size_t srcRowPitchInBytes, size_t rowBegin, size_t rowEnd){
        const size_t numRows = rowEnd - rowBegin;
        const size_t elementsToCopy = numRows * numColumns;

        for(size_t i = group.thread_rank(); i < elementsToCopy; i += group.size()){
            const size_t inputRow = i / numColumns;
            const size_t outputRow = rowBegin + inputRow;
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
    void scatterkernel(TwoDimensionalArray<T> array, const T* __restrict__ src, size_t srcRowPitchInBytes, size_t rowBegin, size_t rowEnd){
        auto gridGroup = cg::this_grid();

        array.scatter(gridGroup, src, srcRowPitchInBytes, rowBegin, rowEnd);
    }

    template<class T, class IndexGenerator>
    __global__
    void gatherkernel(TwoDimensionalArray<T> array, T* __restrict__ dest, size_t destRowPitchInBytes, IndexGenerator indices, const int* __restrict__ d_numIndices){
        auto gridGroup = cg::this_grid();

        const int numIndices = *d_numIndices;

        array.gather(gridGroup, dest, destRowPitchInBytes, indices, numIndices);
    }

    template<class T>
    __global__
    void gatherkernel(TwoDimensionalArray<T> array, T* __restrict__ dest, size_t destRowPitchInBytes, size_t rowBegin, size_t rowEnd){
        auto gridGroup = cg::this_grid();

        array.gather(gridGroup, dest, destRowPitchInBytes, rowBegin, rowEnd);
    }
}

template<class T>
class Gpu2dArrayManaged{
public:
    struct StreamMetaData{
        size_t numRows;
        size_t numColumns;
    };

    Gpu2dArrayManaged()
    : alignmentInBytes(sizeof(T)){
        cudaGetDevice(&deviceId); CUERR;
    }

    Gpu2dArrayManaged(size_t numRows, size_t numColumns, size_t alignmentInBytes)
    : alignmentInBytes(alignmentInBytes){

        assert(alignmentInBytes > 0);
        assert(alignmentInBytes % sizeof(T) == 0);

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
        
        const size_t minbytesPerRow = sizeof(T) * numColumns;
        rowPitchInBytes = SDIV(minbytesPerRow, alignmentInBytes) * alignmentInBytes;

        cudaMalloc(&arraydata, numRows * rowPitchInBytes); CUERR;
        cudaMemset(arraydata, 0, numRows * rowPitchInBytes); CUERR;
    }

    void destroy(){
        int current;
        cudaGetDevice(&current); CUERR;
        cudaSetDevice(deviceId); CUERR;

        cudaFree(arraydata); CUERR;
        
        numRows = 0;
        numColumns = 0;
        alignmentInBytes = 0;
        rowPitchInBytes = 0;
        arraydata = nullptr;

        cudaSetDevice(current);
    }

    TwoDimensionalArray<T> wrapper() const noexcept{
        TwoDimensionalArray<T> wrapper(numRows, numColumns, rowPitchInBytes, arraydata);

        return wrapper;
    }

    operator TwoDimensionalArray<T>() const{
        return wrapper();
    }

    template<class IndexGenerator>
    void gather(T* d_dest, size_t destRowPitchInBytes, IndexGenerator d_indices, int numIndices, cudaStream_t stream = 0) const{
        if(numIndices == 0) return;

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
    void gather(T* d_dest, size_t destRowPitchInBytes, IndexGenerator d_indices, int* d_numIndices, int maxNumIndices, cudaStream_t stream = 0) const{
        
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

        dim3 block(128, 1, 1);
        dim3 grid(SDIV(rows * numColumns, block.x), 1, 1);

        TwoDimensionalArray<T> array = wrapper();

        Gpu2dArrayManagedKernels::gatherkernel<<<grid, block, 0, stream>>>(
            array, 
            d_dest, 
            destRowPitchInBytes, 
            rowBegin, 
            rowEnd
        );

        CUERR;
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

    // void writeMetadataToStream(std::ostream& out) const{
    //     StreamMetaData metaData;
    //     metaData.numRows = numRows;
    //     metaData.numColumns = numColumns;

    //     out.write(reinterpret_cast<const char*>(&metaData), sizeof(StreamMetaData));
    // }

    // void readMetadataFromStream(std::istream& is){
    //     StreamMetaData metaData;
    //     is.read(reinterpret_cast<char*>(&metaData), sizeof(StreamMetaData));

    //     destroy();

    //     init(metaData.numRows, metaData.numColumns);
    // }

    // void skipMetadata(std::istream& is) const{
    //     StreamMetaData metaData;
    //     is.read(reinterpret_cast<char*>(&metaData), sizeof(StreamMetaData));
    // }

    // void writeToStream(std::ostream& out) const{
    //     cub::SwitchDevice sd(deviceId);

    //     StreamMetaData metaData;
    //     metaData.numRows = numRows;
    //     metaData.numColumns = numColumns;

    //     out.write(reinterpret_cast<const char*>(&metaData), sizeof(StreamMetaData));

    //     constexpr size_t mb = 1 << 20;
    //     constexpr size_t buffersize = 16 * mb;

    //     T* h_transferbuffer;
    //     cudaMallocHost(&h_transferbuffer, buffersize); CUERR;

    //     cudaStream_t stream = 0;

    //     const size_t maxRowsInTransferbuffer = buffersize / numColumns;
    //     assert(maxRowsInTransferbuffer > 0);

    //     const size_t numIterations = SDIV(numRows, maxRowsInTransferbuffer);

    //     for(size_t i = 0; i < numIterations; i++){
    //         const size_t rowBegin = i * maxRowsInTransferbuffer;
    //         const size_t rowEnd = std::min((i+1) * maxRowsInTransferbuffer, numRows);
    //         const size_t batchsizerows = rowEnd - rowBegin;

    //         cudaStreamSynchronize(stream);

    //         gather(h_transferbuffer, numColumns * sizeof(T),  rowBegin, rowEnd, stream); CUERR;

    //         cudaStreamSynchronize(stream);

    //         out.write(reinterpret_cast<const char*>(h_transferbuffer), batchsizerows * numColumns * sizeof(T));
    //     }        

    //     cudaFreeHost(h_transferbuffer);
    // }

    // void readDataFromStream(std::istream& in) const{
    //     assert(numRows > 0);
    //     assert(numColumns > 0);

    //     cub::SwitchDevice sd(deviceId);

    //     constexpr size_t mb = 1 << 20;
    //     constexpr size_t buffersize = 16 * mb;

    //     T* h_transferbuffer;
    //     cudaMallocHost(&h_transferbuffer, buffersize); CUERR;

    //     cudaStream_t stream = 0;

    //     const size_t maxRowsInTransferbuffer = buffersize / numColumns;
    //     assert(maxRowsInTransferbuffer > 0);

    //     const size_t numIterations = SDIV(numRows, maxRowsInTransferbuffer);

    //     for(size_t i = 0; i < numIterations; i++){
    //         const size_t rowBegin = i * maxRowsInTransferbuffer;
    //         const size_t rowEnd = std::min((i+1) * maxRowsInTransferbuffer, numRows);
    //         const size_t batchsizerows = rowEnd - rowBegin;

    //         cudaStreamSynchronize(stream);

    //         in.read(reinterpret_cast<char*>(h_transferbuffer), batchsizerows * numColumns * sizeof(T));            

    //         scatter(h_transferbuffer, numColumns * sizeof(T),  rowBegin, rowEnd, stream); CUERR;

    //         cudaStreamSynchronize(stream);            
    //     }        

    //     cudaFreeHost(h_transferbuffer);
    // }

    MemoryUsage getMemoryInfo() const{
        MemoryUsage result;
        result.host = 0;
        result[getDeviceId()] = getNumRows() * rowPitchInBytes;

        return result;
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

private:
    int deviceId{};
    size_t numRows{};
    size_t numColumns{};
    size_t alignmentInBytes{};
    size_t rowPitchInBytes{};
    T* arraydata{};   
};





template<class T>
class Cpu2dArrayManaged{
public:
    struct StreamMetaData{
        size_t numRows;
        size_t numColumns;
        size_t alignmentInBytes;
    };

    Cpu2dArrayManaged() 
    : alignmentInBytes(sizeof(T)){

    }

    Cpu2dArrayManaged(size_t numRows, size_t numColumns, size_t alignmentInBytes)
        : alignmentInBytes(alignmentInBytes){

        assert(alignmentInBytes > 0);
        assert(alignmentInBytes % sizeof(T) == 0);

        init(numRows, numColumns);
    }

    ~Cpu2dArrayManaged(){
        destroy();
    }

    Cpu2dArrayManaged(const Cpu2dArrayManaged& rhs)
        : Cpu2dArrayManaged(rhs.numRows, rhs.numColumns, rhs.alignmentInBytes)
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
        std::swap(l.alignmentInBytes, r.alignmentInBytes);
        std::swap(l.rowPitchInBytes, r.rowPitchInBytes);
        std::swap(l.arraydata, r.arraydata);
    }

    void init(size_t numRows, size_t numColumns){
        assert(numRows > 0);
        assert(numColumns > 0);

        this->numRows = numRows;
        this->numColumns = numColumns;
        
        const size_t minbytesPerRow = sizeof(T) * numColumns;
        rowPitchInBytes = SDIV(minbytesPerRow, alignmentInBytes) * alignmentInBytes;

        arraydata.resize(SDIV(rowPitchInBytes, sizeof(T)) * sizeof(T), T{});
    }

    void destroy(){        
        numRows = 0;
        numColumns = 0;
        alignmentInBytes = 0;
        rowPitchInBytes = 0;

        std::vector<T> tmp;
        std::swap(tmp, arraydata);
    }

    TwoDimensionalArray<T> wrapper() const noexcept{
        TwoDimensionalArray<T> wrapper(numRows, numColumns, rowPitchInBytes, arraydata.data());

        return wrapper;
    }

    operator TwoDimensionalArray<T>() const{
        return wrapper();
    }

    void gather(T* h_dest, size_t destRowPitchInBytes, const int* h_indices, int numIndices){
        if(numIndices == 0) return;

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
        if(numIndices == 0) return;

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

    void gather(T* h_dest, size_t destRowPitchInBytes, size_t rowBegin, size_t rowEnd){
        const size_t rows = rowEnd - rowBegin;
        if(rows == 0) return;

        TwoDimensionalArray<T> array = wrapper();
        SingleThreadGroup group{};

        array.gather(
            group, 
            h_dest, 
            destRowPitchInBytes, 
            rowBegin, 
            rowEnd
        );
    }

    void scatter(const T* h_src, size_t srcRowPitchInBytes, size_t rowBegin, size_t rowEnd){
        const size_t rows = rowEnd - rowBegin;
        if(rows == 0) return;

        TwoDimensionalArray<T> array = wrapper();
        SingleThreadGroup group{};

        array.scatter(
            group,
            h_src, 
            srcRowPitchInBytes, 
            rowBegin, 
            rowEnd
        );
    }

    // void writeMetadataToStream(std::ostream& out) const{
    //     StreamMetaData metaData;
    //     metaData.numRows = numRows;
    //     metaData.numColumns = numColumns;
    //     metaData.alignmentInBytes = alignmentInBytes;

    //     out.write(reinterpret_cast<const char*>(&metaData), sizeof(StreamMetaData));
    // }

    // void readMetadataFromStream(std::istream& is){
    //     StreamMetaData metaData;
    //     is.read(reinterpret_cast<char*>(&metaData), sizeof(StreamMetaData));

    //     destroy();

    //     init(metaData.numRows, metaData.numColumns, metaData.alignmentInBytes);
    // }

    // void skipMetadata(std::istream& is) const{
    //     StreamMetaData metaData;
    //     is.read(reinterpret_cast<char*>(&metaData), sizeof(StreamMetaData));
    // }

    // void writeDataToStream(std::ostream& out) const{
    //     constexpr size_t mb = 1 << 20;
    //     constexpr size_t buffersize = 16 * mb;

    //     std::vector<char> h_transferbuffer(buffersize);

    //     const size_t maxRowsInTransferbuffer = buffersize / numColumns;
    //     assert(maxRowsInTransferbuffer > 0);

    //     const size_t numIterations = SDIV(numRows, maxRowsInTransferbuffer);

    //     for(size_t i = 0; i < numIterations; i++){
    //         const size_t rowBegin = i * maxRowsInTransferbuffer;
    //         const size_t rowEnd = std::min((i+1) * maxRowsInTransferbuffer, numRows);
    //         const size_t batchsizerows = rowEnd - rowBegin;

    //         gather((T*)h_transferbuffer.data(), numColumns * sizeof(T),  rowBegin, rowEnd); CUERR;

    //         out.write(reinterpret_cast<const char*>(h_transferbuffer.data()), batchsizerows * numColumns * sizeof(T));
    //     }
    // }

    // void readDataFromStream(std::istream& in) const{
    //     assert(numRows > 0);
    //     assert(numColumns > 0);

    //     constexpr size_t mb = 1 << 20;
    //     constexpr size_t buffersize = 16 * mb;

    //     std::vector<char> h_transferbuffer(buffersize);

    //     const size_t maxRowsInTransferbuffer = buffersize / numColumns;
    //     assert(maxRowsInTransferbuffer > 0);

    //     const size_t numIterations = SDIV(numRows, maxRowsInTransferbuffer);

    //     for(size_t i = 0; i < numIterations; i++){
    //         const size_t rowBegin = i * maxRowsInTransferbuffer;
    //         const size_t rowEnd = std::min((i+1) * maxRowsInTransferbuffer, numRows);
    //         const size_t batchsizerows = rowEnd - rowBegin;

    //         in.read(reinterpret_cast<char*>(h_transferbuffer.data()), batchsizerows * numColumns * sizeof(T));            

    //         scatter((const T*)h_transferbuffer.data(), numColumns * sizeof(T),  rowBegin, rowEnd); CUERR;       
    //     }        

    // }

    MemoryUsage getMemoryInfo() const{
        MemoryUsage result;
        result.host = getNumRows() * rowPitchInBytes;

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

private:
    size_t numRows{};
    size_t numColumns{};
    size_t alignmentInBytes{};
    size_t rowPitchInBytes{};
    mutable std::vector<T> arraydata{};
};







}


#endif

