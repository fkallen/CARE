#ifndef CARE_GPUCPUREADSTORAGEADAPTER_CUH
#define CARE_GPUCPUREADSTORAGEADAPTER_CUH

#include <gpu/gpureadstorage.cuh>
#include <cpureadstorage.hpp>

#include <hpc_helpers.cuh>

#include <cub/cub.cuh>

namespace care{
namespace gpu{

    struct GPUCPUReadStorageAdapter : public CpuReadStorage{
    public:

        GPUCPUReadStorageAdapter() = default;

        GPUCPUReadStorageAdapter(const GpuReadStorage& gpuReadStorage_, ReadStorageHandle gpuHandle_, cudaStream_t stream_, cub::CachingDeviceAllocator& cubAllocator_)
            : gpuHandle(gpuHandle_), gpuReadStorage(&gpuReadStorage_), stream(stream_), cubAllocator(&cubAllocator_){

        }

    public: //inherited interface

        void areSequencesAmbiguous(
            bool* result, 
            const read_number* readIds, 
            int numSequences
        ) const override{

            read_number* d_readIds = nullptr;
            cubAllocator->DeviceAllocate((void**)&d_readIds, sizeof(read_number) * numSequences, stream); CUERR;
            bool* d_result;
            cubAllocator->DeviceAllocate((void**)&d_result, sizeof(bool) * numSequences, stream); CUERR;

            cudaMemcpyAsync(
                d_readIds,
                readIds,
                sizeof(read_number) * numSequences,
                H2D,
                stream
            ); CUERR;

            gpuReadStorage->areSequencesAmbiguous(
                gpuHandle,
                d_result, 
                d_readIds, 
                numSequences, 
                stream
            );

            cudaMemcpyAsync(
                result,
                d_result,
                sizeof(bool) * numSequences,
                D2H,
                stream
            ); CUERR;

            cubAllocator->DeviceFree(d_result); CUERR;
            cubAllocator->DeviceFree(d_readIds); CUERR;

            cudaStreamSynchronize(stream); CUERR;
        }

        void gatherSequences(
            unsigned int* sequence_data,
            size_t outSequencePitchInInts,
            const read_number* readIds,
            int numSequences
        ) const override{
            read_number* d_readIds = nullptr;
            cubAllocator->DeviceAllocate((void**)&d_readIds, sizeof(read_number) * numSequences, stream); CUERR;
            unsigned int* d_sequence_data;
            cubAllocator->DeviceAllocate((void**)&d_sequence_data, sizeof(unsigned int) * outSequencePitchInInts * numSequences, stream); CUERR;

            cudaMemcpyAsync(
                d_readIds,
                readIds,
                sizeof(read_number) * numSequences,
                H2D,
                stream
            ); CUERR;

            gpuReadStorage->gatherSequences(
                gpuHandle,
                d_sequence_data,
                outSequencePitchInInts,
                readIds,
                d_readIds,
                numSequences,
                stream
            );

            cudaMemcpyAsync(
                sequence_data,
                d_sequence_data,
                sizeof(unsigned int) * outSequencePitchInInts * numSequences,
                D2H,
                stream
            ); CUERR;

            cubAllocator->DeviceFree(d_sequence_data); CUERR;
            cubAllocator->DeviceFree(d_readIds); CUERR;

            cudaStreamSynchronize(stream); CUERR;
        }

        void gatherQualities(
            char* quality_data,
            size_t out_quality_pitch,
            const read_number* readIds,
            int numSequences
        ) const override{
            read_number* d_readIds = nullptr;
            cubAllocator->DeviceAllocate((void**)&d_readIds, sizeof(read_number) * numSequences, stream); CUERR;
            char* d_quality_data;
            cubAllocator->DeviceAllocate((void**)&d_quality_data, sizeof(char) * out_quality_pitch * numSequences, stream); CUERR;

            cudaMemcpyAsync(
                d_readIds,
                readIds,
                sizeof(read_number) * numSequences,
                H2D,
                stream
            ); CUERR;

            gpuReadStorage->gatherQualities(
                gpuHandle,
                d_quality_data,
                out_quality_pitch,
                readIds,
                d_readIds,
                numSequences,
                stream
            );

            cudaMemcpyAsync(
                quality_data,
                d_quality_data,
                sizeof(char) * out_quality_pitch * numSequences,
                D2H,
                stream
            ); CUERR;

            cubAllocator->DeviceFree(d_quality_data); CUERR;
            cubAllocator->DeviceFree(d_readIds); CUERR;

            cudaStreamSynchronize(stream); CUERR;
        }

        void gatherSequenceLengths(
            int* lengths,
            const read_number* readIds,
            int numSequences
        ) const override{
            read_number* d_readIds = nullptr;
            cubAllocator->DeviceAllocate((void**)&d_readIds, sizeof(read_number) * numSequences, stream); CUERR;
            int* d_lengths;
            cubAllocator->DeviceAllocate((void**)&d_lengths, sizeof(int) * numSequences, stream); CUERR;

            cudaMemcpyAsync(
                d_readIds,
                readIds,
                sizeof(read_number) * numSequences,
                H2D,
                stream
            ); CUERR;

            gpuReadStorage->gatherSequenceLengths(
                gpuHandle,
                d_lengths,
                d_readIds,
                numSequences,    
                stream
            );

            cudaMemcpyAsync(
                lengths,
                d_lengths,
                sizeof(int) * numSequences,
                D2H,
                stream
            ); CUERR;

            cubAllocator->DeviceFree(d_lengths); CUERR;
            cubAllocator->DeviceFree(d_readIds); CUERR;

            cudaStreamSynchronize(stream); CUERR;
        }

        void getIdsOfAmbiguousReads(
            read_number* ids
        ) const override{
            gpuReadStorage->getIdsOfAmbiguousReads(ids);
        }

        std::int64_t getNumberOfReadsWithN() const override{
            return gpuReadStorage->getNumberOfReadsWithN();
        }

        MemoryUsage getMemoryInfo() const override{
            return gpuReadStorage->getMemoryInfo();
        }

        read_number getNumberOfReads() const override{
            return gpuReadStorage->getNumberOfReads();
        }

        bool canUseQualityScores() const override{
            return gpuReadStorage->canUseQualityScores();
        }

        int getSequenceLengthLowerBound() const override{
            return gpuReadStorage->getSequenceLengthLowerBound();
        }

        int getSequenceLengthUpperBound() const override{
            return gpuReadStorage->getSequenceLengthUpperBound();
        }

        bool isPairedEnd() const override{
            return gpuReadStorage->isPairedEnd();
        }

        // void destroy() override{
        //     return gpuReadStorage->destroy();
        // }

    private:
        mutable ReadStorageHandle gpuHandle;
        const GpuReadStorage* gpuReadStorage;
        cudaStream_t stream;
        cub::CachingDeviceAllocator* cubAllocator;
    };


} //namespace gpu
} //namespace care




#endif