#ifndef CARE_GPUMINHASHER_CUH
#define CARE_GPUMINHASHER_CUH


#ifdef __NVCC__

#include <config.hpp>
#include <options.hpp>
#include <gpu/distributedreadstorage.hpp>

#include <cstdint>

namespace care{
namespace gpu{

class GpuMinhasher{
public:
    struct QueryHandle{
        int id;
        const GpuMinhasher* parent;
    };

    using Key = kmer_type;

    virtual ~GpuMinhasher() = default;

    virtual int constructFromReadStorage(
        const RuntimeOptions &runtimeOptions,
        std::uint64_t nReads,
        const DistributedReadStorage& gpuReadStorage,
        int upperBoundSequenceLength,
        int maxNumHashfunctions,
        int hashFunctionOffset = 0
    ) = 0;

    virtual QueryHandle makeQueryHandle() const = 0;

    virtual void query(
        QueryHandle& handle,
        const unsigned int* d_encodedSequences,
        std::size_t encodedSequencePitchInInts,
        const int* d_sequenceLengths,
        int numSequences,
        int deviceId, 
        cudaStream_t stream,
        read_number* d_similarReadIds,
        int* d_similarReadsPerSequence,
        int* d_similarReadsPerSequencePrefixSum
    ) const = 0;

    virtual void queryExcludingSelf(
        QueryHandle& handle,
        const read_number* d_readIds,
        const unsigned int* d_encodedSequences,
        std::size_t encodedSequencePitchInInts,
        const int* d_sequenceLengths,
        int numSequences,
        int deviceId, 
        cudaStream_t stream,
        read_number* d_similarReadIds,
        int* d_similarReadsPerSequence,
        int* d_similarReadsPerSequencePrefixSum
    ) const = 0;

    virtual void compact(cudaStream_t stream) = 0;

    virtual MemoryUsage getMemoryInfo() const noexcept = 0;

    virtual MemoryUsage getMemoryInfo(const QueryHandle& handle) const noexcept = 0;

    virtual int getNumResultsPerMapThreshold() const noexcept = 0;
    
    virtual int getNumberOfMaps() const noexcept = 0;

    virtual void destroy() = 0;

};



} //namespace gpu
} //namespace care


#endif
#endif