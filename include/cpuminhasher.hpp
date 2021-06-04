#ifndef CARE_CPUMINHASHER_HPP
#define CARE_CPUMINHASHER_HPP

#include <config.hpp>
#include <memorymanagement.hpp>
#include <minhasherhandle.hpp>

#include <cstdint>

namespace care{

class CpuMinhasher{
public:

    using Key = kmer_type;

    virtual ~CpuMinhasher() = default;

    virtual MinhasherHandle makeMinhasherHandle() const = 0;

    virtual void destroyHandle(MinhasherHandle& handle) const = 0;

    virtual void determineNumValues(
        MinhasherHandle& queryHandle,
        const unsigned int* h_sequenceData2Bit,
        std::size_t encodedSequencePitchInInts,
        const int* h_sequenceLengths,
        int numSequences,
        int* h_numValuesPerSequence,
        int& totalNumValues
    ) const = 0;

    virtual void retrieveValues(
        MinhasherHandle& queryHandle,
        const read_number* h_readIds,
        int numSequences,
        int totalNumValues,
        read_number* h_values,
        int* h_numValuesPerSequence,
        int* h_offsets //numSequences + 1
    ) const = 0;

    //virtual void compact() = 0;

    virtual MemoryUsage getMemoryInfo() const noexcept = 0;

    virtual MemoryUsage getMemoryInfo(const MinhasherHandle& handle) const noexcept = 0;

    virtual int getNumResultsPerMapThreshold() const noexcept = 0;
    
    virtual int getNumberOfMaps() const noexcept = 0;

    //virtual void destroy() = 0;

protected:
    MinhasherHandle constructHandle(int id) const{
        return MinhasherHandle{id};
    }

};



}





#endif