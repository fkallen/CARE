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
    class QueryHandle{
    friend class GpuMinhasher;
    public:

        int getId() const noexcept{
            return id;
        }

    private:
        QueryHandle() = default;
        QueryHandle(int i) : id(i){}

        int id;
        //const GpuMinhasher* parent;
    };

    using Key = kmer_type;

    virtual ~GpuMinhasher() = default;

    virtual QueryHandle makeQueryHandle() const = 0;

    virtual void determineNumValues(
        QueryHandle& queryHandle,
        const unsigned int* d_sequenceData2Bit,
        std::size_t encodedSequencePitchInInts,
        const int* d_sequenceLengths,
        int numSequences,
        int* d_numValuesPerSequence,
        int& totalNumValues,
        cudaStream_t stream
    ) const = 0;

    virtual void retrieveValues(
        QueryHandle& queryHandle,
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

    virtual MemoryUsage getMemoryInfo(const QueryHandle& handle) const noexcept = 0;

    virtual int getNumResultsPerMapThreshold() const noexcept = 0;
    
    virtual int getNumberOfMaps() const noexcept = 0;

    virtual void destroy() = 0;

protected:
    QueryHandle constructHandle(int id) const{
        return QueryHandle{id};
    }

};



} //namespace gpu
} //namespace care


#endif
#endif