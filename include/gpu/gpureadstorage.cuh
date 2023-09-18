#ifndef CARE_GPUREADSTORAGE_CUH
#define CARE_GPUREADSTORAGE_CUH


#ifdef __NVCC__

#include <config.hpp>
#include <memorymanagement.hpp>
#include <readstoragehandle.hpp>
#include <gpu/asyncresult.cuh>
#include <hpc_helpers.cuh>

#include <cstdint>

#include <rmm/mr/device/device_memory_resource.hpp>

namespace care{

namespace gpu{


class GpuReadStorage{
public:

    virtual ~GpuReadStorage() = default;

    virtual ReadStorageHandle makeHandle() const = 0;

    virtual void destroyHandle(ReadStorageHandle& handle) const = 0;

    virtual void areSequencesAmbiguous(
        ReadStorageHandle& handle,
        bool* d_result, 
        const read_number* d_readIds, 
        int numSequences, 
        cudaStream_t stream
    ) const = 0;

    virtual void gatherSequences(
        ReadStorageHandle& handle,
        unsigned int* d_sequence_data,
        size_t outSequencePitchInInts,
        const AsyncConstBufferWrapper<read_number> h_readIds,
        const read_number* d_readIds,
        int numSequences,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr
    ) const = 0;

    virtual void gatherContiguousSequences(
        ReadStorageHandle& handle,
        unsigned int* d_sequence_data,
        size_t outSequencePitchInInts,
        read_number firstIndex,
        int numSequences,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr
    ) const = 0;

    virtual void gatherQualities(
        ReadStorageHandle& handle,
        char* d_quality_data,
        size_t out_quality_pitch,
        const AsyncConstBufferWrapper<read_number> h_readIds,
        const read_number* d_readIds,
        int numSequences,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr
    ) const = 0;

    virtual void gatherContiguousQualities(
        ReadStorageHandle& handle,
        char* d_quality_data,
        size_t out_quality_pitch,
        read_number firstIndex,
        int numSequences,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr
    ) const = 0;

    virtual void gatherSequenceLengths(
        ReadStorageHandle& handle,
        int* d_lengths,
        const read_number* d_readIds,
        int numSequences,    
        cudaStream_t stream
    ) const = 0;

    virtual void multi_gatherSequences(
        std::vector<ReadStorageHandle>& vec_handle,
        std::vector<unsigned int*>& vec_d_sequence_data,
        size_t outSequencePitchInInts,
        const std::vector<read_number*>& vec_d_readIds,
        const std::vector<int>& vec_numSequences,
        const std::vector<cudaStream_t>& streams,
        const std::vector<int>& deviceIds,
        const std::vector<rmm::mr::device_memory_resource*>& mrs
    ) const = 0;

    virtual void multi_gatherSequences(
        std::vector<ReadStorageHandle>& vec_handle,
        std::vector<unsigned int*>& vec_d_sequence_data,
        size_t outSequencePitchInInts,
        const std::vector<AsyncConstBufferWrapper<read_number>>& vec_h_readIds,
        const std::vector<read_number*>& vec_d_readIds,
        const std::vector<int>& vec_numSequences,
        const std::vector<cudaStream_t>& streams,
        const std::vector<int>& deviceIds,
        const std::vector<rmm::mr::device_memory_resource*>& mrs
    ) const = 0;

    virtual void multi_gatherQualities(
        std::vector<ReadStorageHandle>& vec_handle,
        std::vector<char*>& vec_d_quality_data,
        size_t out_quality_pitch,
        const std::vector<read_number*>& vec_d_readIds,
        const std::vector<int>& vec_numSequences,
        const std::vector<cudaStream_t>& streams,
        const std::vector<int>& deviceIds,
        const std::vector<rmm::mr::device_memory_resource*>& mrs
    ) const = 0;

    virtual void multi_gatherQualities(
        std::vector<ReadStorageHandle>& vec_handle,
        std::vector<char*>& vec_d_quality_data,
        size_t out_quality_pitch,
        const std::vector<AsyncConstBufferWrapper<read_number>>& vec_h_readIds,
        const std::vector<read_number*>& vec_d_readIds,
        const std::vector<int>& vec_numSequences,
        const std::vector<cudaStream_t>& streams,
        const std::vector<int>& deviceIds,
        const std::vector<rmm::mr::device_memory_resource*>& mrs
    ) const = 0;

    virtual void getIdsOfAmbiguousReads(
        read_number* ids
    ) const = 0;

    virtual std::int64_t getNumberOfReadsWithN() const = 0;

    virtual MemoryUsage getMemoryInfo() const = 0;

    virtual MemoryUsage getMemoryInfo(const ReadStorageHandle& handle) const = 0;

    virtual read_number getNumberOfReads() const = 0;

    virtual bool canUseQualityScores() const = 0;

    virtual int getSequenceLengthLowerBound() const = 0;

    virtual int getSequenceLengthUpperBound() const = 0;

    virtual bool isPairedEnd() const = 0;

    //virtual void destroy() = 0;

    virtual bool hasHostSequences() const noexcept = 0;

    virtual bool hasHostQualities() const noexcept = 0;

protected:
    ReadStorageHandle constructHandle(int id) const{
        return ReadStorageHandle{id};
    }

};


}
}

#endif




#endif