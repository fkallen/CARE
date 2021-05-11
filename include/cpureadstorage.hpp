#ifndef CARE_CPUREADSTORAGE_HPP
#define CARE_CPUREADSTORAGE_HPP

#include <config.hpp>
#include <memorymanagement.hpp>
#include <readstoragehandle.hpp>

#include <cstdint>

namespace care{

class CpuReadStorage{
public:

    virtual ~CpuReadStorage() = default;

    virtual ReadStorageHandle makeHandle() const = 0;

    virtual void destroyHandle(ReadStorageHandle& handle) const = 0;

    virtual void areSequencesAmbiguous(
        ReadStorageHandle& handle,
        bool* result, 
        const read_number* readIds, 
        int numSequences
    ) const = 0;

    virtual void gatherSequences(
        ReadStorageHandle& handle,
        unsigned int* sequence_data,
        size_t outSequencePitchInInts,
        const read_number* readIds,
        int numSequences
    ) const = 0;

    virtual void gatherQualities(
        ReadStorageHandle& handle,
        char* quality_data,
        size_t out_quality_pitch,
        const read_number* readIds,
        int numSequences
    ) const = 0;

    virtual void gatherSequenceLengths(
        ReadStorageHandle& handle,
        int* lengths,
        const read_number* readIds,
        int numSequences
    ) const = 0;

    virtual void getIdsOfAmbiguousReads(
        ReadStorageHandle& handle,
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

    virtual void destroy() = 0;

protected:
    ReadStorageHandle constructHandle(int id) const{
        return ReadStorageHandle{id};
    }

};



}

#endif