#ifndef CARE_GPUMINHASHER_CUH
#define CARE_GPUMINHASHER_CUH


#ifdef __NVCC__

#include <config.hpp>
#include <memorymanagement.hpp>
#include <minhasherhandle.hpp>
#include <threadpool.hpp>

#include <cstdint>
#include <vector>

#include <rmm/mr/device/device_memory_resource.hpp>

namespace care{
namespace gpu{

class GpuMinhasher{
public:

    using Key = kmer_type;

    virtual ~GpuMinhasher() = default;

    virtual MinhasherHandle makeMinhasherHandle() const = 0;

    virtual void destroyHandle(MinhasherHandle& handle) const = 0;

    //construction
    virtual void setHostMemoryLimitForConstruction(std::size_t bytes) = 0;
    virtual void setDeviceMemoryLimitsForConstruction(const std::vector<std::size_t>&) = 0;
    virtual int addHashTables(int numAdditionalTables, const int* hashFunctionIds, cudaStream_t stream) = 0;
    virtual void setThreadPool(ThreadPool* tp) = 0;
    virtual void compact(cudaStream_t stream) = 0;
    virtual void constructionIsFinished(cudaStream_t stream) = 0;

    virtual void insert(
        const unsigned int* d_sequenceData2Bit,
        int numSequences,
        const int* d_sequenceLengths,
        std::size_t encodedSequencePitchInInts,
        const read_number* d_readIds,
        const read_number* h_readIds,
        int firstHashfunction,
        int numHashfunctions,
        const int* h_hashFunctionNumbers,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr
    ) = 0;

    //return number of hash tables where insert was unsuccessfull
    virtual int checkInsertionErrors(
        int firstHashfunction,
        int numHashfunctions,
        cudaStream_t stream
    ) = 0;

    //query

    virtual void determineNumValues(
        MinhasherHandle& queryHandle,
        const unsigned int* d_sequenceData2Bit,
        std::size_t encodedSequencePitchInInts,
        const int* d_sequenceLengths,
        int numSequences,
        int* d_numValuesPerSequence,
        int& totalNumValues,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr
    ) const = 0;

    virtual void retrieveValues(
        MinhasherHandle& queryHandle,
        int numSequences,
        int totalNumValues,
        read_number* d_values,
        const int* d_numValuesPerSequence,
        int* d_offsets, //numSequences + 1
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr
    ) const = 0;


    virtual MemoryUsage getMemoryInfo() const noexcept = 0;

    virtual MemoryUsage getMemoryInfo(const MinhasherHandle& handle) const noexcept = 0;

    virtual int getNumResultsPerMapThreshold() const noexcept = 0;
    
    virtual int getNumberOfMaps() const noexcept = 0;

    virtual int getKmerSize() const noexcept = 0;

    //virtual void destroy() = 0;

    virtual bool hasGpuTables() const noexcept = 0;

    virtual void writeToStream(std::ostream& os) const = 0;
    virtual int loadFromStream(std::ifstream& is, int numMapsUpperLimit) = 0;
    virtual bool canWriteToStream() const noexcept = 0;
    virtual bool canLoadFromStream() const noexcept = 0;

protected:
    MinhasherHandle constructHandle(int id) const{
        return MinhasherHandle{id};
    }

};

class GpuMinhasherWithMultiQuery{
public:
    virtual void multi_determineNumValues(
        MinhasherHandle& queryHandle,
        const std::vector<const unsigned int*>& vec_d_sequenceData2Bit,
        std::size_t encodedSequencePitchInInts,
        const std::vector<const int*>& vec_d_sequenceLengths,
        const std::vector<int>& vec_numSequences,
        const std::vector<int*>& vec_d_numValuesPerSequence,
        const std::vector<int*>& vec_totalNumValues,
        const std::vector<cudaStream_t>& callerStreams,
        const std::vector<int>& callerDeviceIds,
        const std::vector<rmm::mr::device_memory_resource*>& mrs
    ) const = 0;

    virtual void multi_retrieveValues(
        MinhasherHandle& queryHandle,
        const std::vector<int>& vec_numSequences,
        const std::vector<const int*>& vec_totalNumValues,
        const std::vector<read_number*>& vec_d_values,
        const std::vector<const int*>& vec_d_numValuesPerSequence,
        const std::vector<int*> vec_d_offsets, //numSequences + 1
        const std::vector<cudaStream_t>& streams,
        const std::vector<int> callerDeviceIds,
        const std::vector<rmm::mr::device_memory_resource*>& mrs
    ) const = 0;
};


} //namespace gpu
} //namespace care


#endif
#endif