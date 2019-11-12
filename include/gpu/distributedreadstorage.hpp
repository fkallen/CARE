#ifndef CARE_DISTRIBUTED_READ_STORAGE_HPP
#define CARE_DISTRIBUTED_READ_STORAGE_HPP


#ifdef __NVCC__

#include <gpu/distributedarray.hpp>
#include <gpu/gpulengthstorage.hpp>
#include <config.hpp>
#include <sequencefileio.hpp>

#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>

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

    struct SavedGpuData{
        std::vector<std::vector<char>> sequencedata;
        std::vector<std::vector<char>> qualitydata;

        bool sequenceDataInMemory = false;
        bool qualityDataInMemory = false;

        void clear(){
            *this = SavedGpuData{};
        }
    };

    using Length_t = int;

    using GatherHandleSequences = DistributedArray<unsigned int, read_number>::GatherHandle;
    using GatherHandleLengths = DistributedArray<Length_t, read_number>::GatherHandle;
    using GatherHandleQualities = DistributedArray<char, read_number>::GatherHandle;

    using LengthStore_t = LengthStore<std::uint32_t>;
    using GPULengthStore_t = GPULengthStore<std::uint32_t>;

    bool isReadOnly;
    std::vector<int> deviceIds;
    std::atomic<read_number> numberOfInsertedReads;
    read_number maximumNumberOfReads;
    int sequenceLengthLowerBound;
    int sequenceLengthUpperBound;
    bool useQualityScores;

    std::vector<read_number> readIdsOfReadsWithUndeterminedBase; //sorted in ascending order
    std::mutex mutexUndeterminedBaseReads;
    LengthStore_t lengthStorage;
    GPULengthStore_t gpulengthStorage;
    mutable DistributedArray<unsigned int, read_number> distributedSequenceData2;
    mutable DistributedArray<Length_t, read_number> distributedSequenceLengths2;
    mutable DistributedArray<char, read_number> distributedQualities2;



    Statistics statistics;

	bool hasMoved = false;

    DistributedReadStorage() : DistributedReadStorage({}, 0, false, 0, 0){};

    DistributedReadStorage(const std::vector<int>& deviceIds_, read_number nReads, bool b, 
                            int minimum_sequence_length, int maximum_sequence_length);

    DistributedReadStorage(const DistributedReadStorage& other) = delete;
    DistributedReadStorage& operator=(const DistributedReadStorage& other) = delete;

	DistributedReadStorage(DistributedReadStorage&& other);

	DistributedReadStorage& operator=(DistributedReadStorage&& other);

	MemoryInfo getMemoryInfo() const;

    Statistics getStatistics() const;

	void destroy();

    read_number getNumberOfReads() const;
    read_number getMaximumNumberOfReads() const;
    bool canUseQualityScores() const;
    int getSequenceLengthLowerBound() const;
    int getSequenceLengthUpperBound() const;
    std::vector<int> getDeviceIds() const;

    void saveToFile(const std::string& filename) const;
    void loadFromFile(const std::string& filename);
    void loadFromFile(const std::string& filename, const std::vector<int>& deviceIds_);
    // void writeGpuDataToStreamAndFreeGpuMem(std::ofstream& stream) const;
    // void allocGpuMemAndReadGpuDataFromStream(std::ifstream& stream) const;
    SavedGpuData saveGpuDataAndFreeGpuMem(std::ofstream& stream, std::size_t numBytesMustRemainFree) const;
    void allocGpuMemAndLoadGpuData(std::ifstream& stream, const SavedGpuData& saved) const;

    void setReads(read_number firstIndex, read_number lastIndex_excl, const Read* reads, int numReads);
    void setReads(read_number firstIndex, read_number lastIndex_excl, const std::vector<Read>& reads);
    void setReads(const std::vector<read_number>& indices, const Read* reads, int numReads);
    void setReads(const std::vector<read_number>& indices, const std::vector<Read>& reads);

    void setReadContainsN(read_number readId, bool contains);
    bool readContainsN(read_number readId) const;

    void constructionIsComplete();
    void allowModifications();

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

    void gatherSequenceLengthsToGpuBufferAsyncNew(
                                int* d_lengths,
                                int deviceId,
                                const read_number* d_readIds,
                                int nReadIds,    
                                cudaStream_t stream) const;

    void gatherSequenceLengthsToHostBufferNew(
                                int* lengths,
                                const read_number* readIds,
                                int nReadIds) const;


    private:
        void init(const std::vector<int>& deviceIds, read_number nReads, bool b, 
                    int minimum_sequence_length, int maximum_sequence_length);

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
