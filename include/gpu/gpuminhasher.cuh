#ifndef CARE_GPUMINHASHER_CUH
#define CARE_GPUMINHASHER_CUH


#ifdef __NVCC__

#include <config.hpp>
#include <memorymanagement.hpp>
#include <gpu/distributedreadstorage.hpp>
#include <minhasherhandle.hpp>

#include <cstdint>

namespace care{
namespace gpu{

class GpuMinhasher{
public:

    using Key = kmer_type;

    virtual ~GpuMinhasher() = default;

    virtual MinhasherHandle makeMinhasherHandle() const = 0;

    virtual void destroyHandle(MinhasherHandle& handle) const = 0;

    virtual void determineNumValues(
        MinhasherHandle& queryHandle,
        const unsigned int* d_sequenceData2Bit,
        std::size_t encodedSequencePitchInInts,
        const int* d_sequenceLengths,
        int numSequences,
        int* d_numValuesPerSequence,
        int& totalNumValues,
        cudaStream_t stream
    ) const = 0;

    virtual void retrieveValues(
        MinhasherHandle& queryHandle,
        const read_number* d_readIds,
        int numSequences,
        int totalNumValues,
        read_number* d_values,
        int* d_numValuesPerSequence,
        int* d_offsets, //numSequences + 1
        cudaStream_t stream
    ) const = 0;

    virtual void compact(cudaStream_t stream) = 0;

    virtual MemoryUsage getMemoryInfo() const noexcept = 0;

    virtual MemoryUsage getMemoryInfo(const MinhasherHandle& handle) const noexcept = 0;

    virtual int getNumResultsPerMapThreshold() const noexcept = 0;
    
    virtual int getNumberOfMaps() const noexcept = 0;

    virtual void destroy() = 0;

    virtual bool hasGpuTables() const noexcept = 0;

protected:
    MinhasherHandle constructHandle(int id) const{
        return MinhasherHandle{id};
    }

};



} //namespace gpu
} //namespace care


#endif
#endif