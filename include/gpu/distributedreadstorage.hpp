#ifndef CARE_DISTRIBUTED_READ_STORAGE_HPP
#define CARE_DISTRIBUTED_READ_STORAGE_HPP


#ifdef __NVCC__

#include <gpu/distributedarray.hpp>
#include <config.hpp>
#include <sequencefileio.hpp>


#include <cstdint>
#include <limits>

namespace care{
namespace gpu{

struct DistributedReadStorage {
public:

    struct MemoryInfo{
        size_t hostSizeInBytes{};
        std::vector<size_t> deviceSizeInBytes{};
        std::vector<int> deviceIds{};
    };

    struct Statistics{
        int maximumSequenceLength = 0;
        int minimumSequenceLength = std::numeric_limits<int>::max();
    };

    using Length_t = int;

    using GatherHandleSequences = DistributedArray<unsigned int, read_number>::GatherHandle;
    using GatherHandleLengths = DistributedArray<Length_t, read_number>::GatherHandle;
    using GatherHandleQualities = DistributedArray<char, read_number>::GatherHandle;

    std::vector<int> deviceIds;
    read_number numberOfReads;
    int sequenceLengthLimit;
    bool useQualityScores;

    std::vector<read_number> readIdsOfReadsWithUndeterminedBase; //sorted in ascending order

    DistributedArray<unsigned int, read_number> distributedSequenceData2;
    DistributedArray<Length_t, read_number> distributedSequenceLengths2;
    DistributedArray<char, read_number> distributedQualities2;



    Statistics statistics;

	bool hasMoved = false;

    DistributedReadStorage() : DistributedReadStorage({}, 0, false, 0){};

    DistributedReadStorage(const std::vector<int>& deviceIds_, read_number nReads, bool b, int maximum_sequence_length);

    DistributedReadStorage(const DistributedReadStorage& other) = delete;
    DistributedReadStorage& operator=(const DistributedReadStorage& other) = delete;

	DistributedReadStorage(DistributedReadStorage&& other);

	DistributedReadStorage& operator=(DistributedReadStorage&& other);

	MemoryInfo getMemoryInfo() const;

    Statistics getStatistics() const;

	void destroy();

    read_number getNumberOfReads() const;
    bool canUseQualityScores() const;
    int getSequenceLengthLimit() const;
    std::vector<int> getDeviceIds() const;

    void saveToFile(const std::string& filename) const;
    void loadFromFile(const std::string& filename);
    void loadFromFile(const std::string& filename, const std::vector<int>& deviceIds_);

    void setReads(read_number firstIndex, read_number lastIndex_excl, const std::vector<Read>& reads, int numThreads);
    void setReads(const std::vector<read_number>& indices, const std::vector<Read>& reads, int numThreads);

    void setReadContainsN(read_number readId, bool contains);
    bool readContainsN(read_number readId) const;

    GatherHandleSequences makeGatherHandleSequences() const;

    GatherHandleLengths makeGatherHandleLengths() const;

    GatherHandleQualities makeGatherHandleQualities() const;

    void gatherSequenceDataToGpuBufferAsync(
                                const GatherHandleSequences& handle,
                                char* d_sequence_data,
                                size_t out_sequence_pitch,
                                const read_number* h_readIds,
                                const read_number* d_readIds,
                                int nReadIds,
                                int deviceId,
                                cudaStream_t stream,
                                int numCpuThreads) const;

    void gatherSequenceLengthsToGpuBufferAsync(
                                const GatherHandleLengths& handle,
                                int* d_lengths,
                                const read_number* h_readIds,
                                const read_number* d_readIds,
                                int nReadIds,
                                int deviceId,
                                cudaStream_t stream,
                                int numCpuThreads) const;

    void gatherQualitiesToGpuBufferAsync(
                                const GatherHandleQualities& handle,
                                char* d_quality_data,
                                size_t out_quality_pitch,
                                const read_number* h_readIds,
                                const read_number* d_readIds,
                                int nReadIds,
                                int deviceId,
                                cudaStream_t stream,
                                int numCpuThreads) const;

    std::future<void> gatherSequenceDataToHostBufferAsync(
                                const GatherHandleSequences& handle,
                                char* h_sequence_data,
                                size_t out_sequence_pitch,
                                const read_number* h_readIds,
                                int nReadIds,
                                int numCpuThreads) const;

    std::future<void> gatherSequenceLengthsToHostBufferAsync(
                                const GatherHandleLengths& handle,
                                int* h_lengths,
                                const read_number* h_readIds,
                                int nReadIds,
                                int numCpuThreads) const;

    std::future<void> gatherQualitiesToHostBufferAsync(
                                const GatherHandleQualities& handle,
                                char* h_quality_data,
                                size_t out_quality_pitch,
                                const read_number* h_readIds,
                                int nReadIds,
                                int numCpuThreads) const;

    void gatherSequenceDataToHostBuffer(
                                const GatherHandleSequences& handle,
                                char* h_sequence_data,
                                size_t out_sequence_pitch,
                                const read_number* h_readIds,
                                int nReadIds,
                                int numCpuThreads) const;

    void gatherSequenceLengthsToHostBuffer(
                                const GatherHandleLengths& handle,
                                int* h_lengths,
                                const read_number* h_readIds,
                                int nReadIds,
                                int numCpuThreads) const;

    void gatherQualitiesToHostBuffer(
                                const GatherHandleQualities& handle,
                                char* h_quality_data,
                                size_t out_quality_pitch,
                                const read_number* h_readIds,
                                int nReadIds,
                                int numCpuThreads) const;


    private:
        void init(const std::vector<int>& deviceIds, read_number nReads, bool b, int maximum_sequence_length);

        void setSequences(read_number firstIndex, read_number lastIndex_excl, const char* data);
        void setSequences(const std::vector<read_number>& indices, const char* data);
        void setSequenceLengths(read_number firstIndex, read_number lastIndex_excl, const Length_t* data);
        void setSequenceLengths(const std::vector<read_number>& indices, const Length_t* data);
        void setQualities(read_number firstIndex, read_number lastIndex_excl, const char* data);
        void setQualities(const std::vector<read_number>& indices, const char* data);
};




}
}

#endif




#endif
