#ifndef CARE_CPUMINHASHER_HPP
#define CARE_CPUMINHASHER_HPP

#include <config.hpp>
#include <memorymanagement.hpp>
#include <cstdint>

namespace care{

class CpuMinhasher{
public:
    class QueryHandle{
    friend class CpuMinhasher;
    public:

        int getId() const noexcept{
            return id;
        }

    private:
        QueryHandle() = default;
        QueryHandle(int i) : id(i){}

        int id;
    };

    using Key = kmer_type;

    virtual ~CpuMinhasher() = default;

    virtual QueryHandle makeQueryHandle() const = 0;

    virtual void determineNumValues(
        QueryHandle& queryHandle,
        const unsigned int* h_sequenceData2Bit,
        std::size_t encodedSequencePitchInInts,
        const int* h_sequenceLengths,
        int numSequences,
        int* h_numValuesPerSequence,
        int& totalNumValues
    ) const = 0;

    virtual void retrieveValues(
        QueryHandle& queryHandle,
        const read_number* h_readIds,
        int numSequences,
        int totalNumValues,
        read_number* h_values,
        int* h_numValuesPerSequence,
        int* h_offsets //numSequences + 1
    ) const = 0;

    virtual void compact() = 0;

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



}





#endif