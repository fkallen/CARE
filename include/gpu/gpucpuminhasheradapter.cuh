#ifndef CARE_GPUCPUMINHASHERADAPTER_CUH
#define CARE_GPUCPUMINHASHERADAPTER_CUH

#include <gpu/gpuminhasher.cuh>
#include <cpuminhasher.hpp>
#include <minhasherhandle.hpp>

#include <hpc_helpers.cuh>

#include <cub/cub.cuh>

namespace care{
namespace gpu{


class GPUCPUMinhasherAdapter : public CpuMinhasher{
public:
    using Key_t = CpuMinhasher::Key;

public:

    GPUCPUMinhasherAdapter(GpuMinhasher& gpuMinhasher_, MinhasherHandle gpuHandle_, cudaStream_t stream_, cub::CachingDeviceAllocator& cubAllocator_)
        : gpuHandle(gpuHandle_), gpuMinhasher(&gpuMinhasher_), stream(stream_), cubAllocator(&cubAllocator_){

    }

public: //inherited interface

    MinhasherHandle makeMinhasherHandle() const override{
        return gpuMinhasher->makeMinhasherHandle();
    }

    void destroyHandle(MinhasherHandle& handle) const override{
        gpuMinhasher->destroyHandle(handle);
    }

    void determineNumValues(
        MinhasherHandle& queryHandle,
        const unsigned int* h_sequenceData2Bit,
        std::size_t encodedSequencePitchInInts,
        const int* h_sequenceLengths,
        int numSequences,
        int* h_numValuesPerSequence,
        int& totalNumValues
    ) const override{

        unsigned int* d_sequenceData2Bit = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_sequenceData2Bit, encodedSequencePitchInInts * numSequences, stream); CUERR;

        int* d_sequenceLengths;
        cubAllocator->DeviceAllocate((void**)&d_sequenceLengths, sizeof(int) * numSequences, stream); CUERR;

        int* d_numValuesPerSequence;
        cubAllocator->DeviceAllocate((void**)&d_numValuesPerSequence, sizeof(int) * numSequences, stream); CUERR;

        cudaMemcpyAsync(
            d_sequenceData2Bit,
            h_sequenceData2Bit,
            encodedSequencePitchInInts * numSequences,
            H2D,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            d_sequenceLengths,
            h_sequenceLengths,
            sizeof(int) * numSequences,
            H2D,
            stream
        ); CUERR;

        gpuMinhasher->determineNumValues(
            queryHandle,
            d_sequenceData2Bit,
            encodedSequencePitchInInts,
            d_sequenceLengths,
            numSequences,
            d_numValuesPerSequence,
            totalNumValues,
            stream
        );

        cudaMemcpyAsync(
            h_numValuesPerSequence,
            d_numValuesPerSequence,
            sizeof(int) * numSequences,
            D2H,
            stream
        ); CUERR;

        cubAllocator->DeviceFree(d_sequenceData2Bit); CUERR;
        cubAllocator->DeviceFree(d_sequenceLengths); CUERR;
        cubAllocator->DeviceFree(d_numValuesPerSequence); CUERR;

        cudaStreamSynchronize(stream); CUERR;
    }

    void retrieveValues(
        MinhasherHandle& queryHandle,
        const read_number* h_readIds,
        int numSequences,
        int totalNumValues,
        read_number* h_values,
        int* h_numValuesPerSequence,
        int* h_offsets //numSequences + 1
    ) const override{
        read_number* d_readIds = nullptr;
        if(h_readIds != nullptr){
            cubAllocator->DeviceAllocate((void**)&d_readIds, sizeof(read_number) * numSequences, stream); CUERR;

            cudaMemcpyAsync(
                d_readIds,
                h_readIds,
                sizeof(read_number) * numSequences,
                H2D,
                stream
            ); CUERR;
        }

        read_number* d_values = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_values, sizeof(read_number) * totalNumValues, stream); CUERR;

        int* d_numValuesPerSequence = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_numValuesPerSequence, sizeof(int) * numSequences, stream); CUERR;

        int* d_offsets = nullptr;
        cubAllocator->DeviceAllocate((void**)&d_offsets, sizeof(int) * (numSequences + 1), stream); CUERR;

        gpuMinhasher->retrieveValues(
            queryHandle,
            d_readIds,
            numSequences,
            totalNumValues,
            d_values,
            d_numValuesPerSequence,
            d_offsets, //numSequences + 1
            stream
        );

        cudaMemcpyAsync(
            h_values,
            d_values,
            sizeof(read_number) * totalNumValues,
            D2H,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            h_numValuesPerSequence,
            d_numValuesPerSequence,
            sizeof(int) * numSequences,
            D2H,
            stream
        ); CUERR;

        cudaMemcpyAsync(
            h_offsets,
            d_offsets,
            sizeof(int) * (numSequences + 1),
            D2H,
            stream
        ); CUERR;

        cubAllocator->DeviceFree(d_offsets); CUERR;
        cubAllocator->DeviceFree(d_numValuesPerSequence); CUERR;
        cubAllocator->DeviceFree(d_values); CUERR;
        if(d_readIds != nullptr){
            cubAllocator->DeviceFree(d_readIds); CUERR;
        }
        cudaStreamSynchronize(stream); CUERR;
    }

    void compact() override{
        gpuMinhasher->compact(stream);
        cudaStreamSynchronize(stream);
    }

    MemoryUsage getMemoryInfo() const noexcept override{
        return gpuMinhasher->getMemoryInfo();
    }

    MemoryUsage getMemoryInfo(const MinhasherHandle& queryHandle) const noexcept{
        return gpuMinhasher->getMemoryInfo(queryHandle);
    }

    int getNumResultsPerMapThreshold() const noexcept override{
        return gpuMinhasher->getNumResultsPerMapThreshold();
    }
    
    int getNumberOfMaps() const noexcept override{
        return gpuMinhasher->getNumberOfMaps();
    }

    void destroy() override{
        gpuMinhasher->destroy();
    }

private:
    MinhasherHandle gpuHandle;
    GpuMinhasher* gpuMinhasher;
    cudaStream_t stream;
    cub::CachingDeviceAllocator* cubAllocator;

};


} //namespace gpu
} //namespace care




#endif