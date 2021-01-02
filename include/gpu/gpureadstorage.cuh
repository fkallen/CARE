#ifndef CARE_GPUREADSTORAGE_CUH
#define CARE_GPUREADSTORAGE_CUH


#ifdef __NVCC__

#include <config.hpp>
#include <memorymanagement.hpp>

#include <cstdint>

namespace care{

namespace gpu{

class GpuReadStorage{
public:
    class Handle{
    friend class GpuReadStorage;
    public:

        int getId() const noexcept{
            return id;
        }

    private:
        Handle() = default;
        Handle(int i) : id(i){}

        int id;
        //const GpuMinhasher* parent;
    };


    virtual ~GpuReadStorage() = default;

    virtual Handle makeHandle() const = 0;

    virtual void areSequencesAmbiguous(
        Handle& handle,
        bool* d_result, 
        const read_number* d_readIds, 
        int numSequences, 
        cudaStream_t stream
    ) const = 0;

    virtual void gatherSequences(
        Handle& handle,
        unsigned int* d_sequence_data,
        size_t outSequencePitchInInts,
        const read_number* h_readIds,
        const read_number* d_readIds,
        int numSequences,
        cudaStream_t stream
    ) const = 0;

    virtual void gatherQualities(
        Handle& handle,
        char* d_quality_data,
        size_t out_quality_pitch,
        const read_number* h_readIds,
        const read_number* d_readIds,
        int numSequences,
        cudaStream_t stream
    ) const = 0;

    virtual void gatherSequenceLengths(
        Handle& handle,
        int* d_lengths,
        const read_number* d_readIds,
        int numSequences,    
        cudaStream_t stream
    ) const = 0;

    virtual bool readContainsN(read_number readId) const = 0;

    virtual std::int64_t getNumberOfReadsWithN() const = 0;

    virtual MemoryUsage getMemoryInfo() const = 0;

    virtual MemoryUsage getMemoryInfo(const Handle& handle) const = 0;

    virtual read_number getNumberOfReads() const = 0;

    virtual bool canUseQualityScores() const = 0;

    virtual int getSequenceLengthLowerBound() const = 0;

    virtual int getSequenceLengthUpperBound() const = 0;

    virtual void destroy() = 0;

protected:
    Handle constructHandle(int id) const{
        return Handle{id};
    }

};


}
}

#endif




#endif