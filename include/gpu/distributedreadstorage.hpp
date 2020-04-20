#ifndef CARE_DISTRIBUTED_READ_STORAGE_HPP
#define CARE_DISTRIBUTED_READ_STORAGE_HPP


#ifdef __NVCC__

#include <gpu/distributedarray.hpp>
#include <gpu/gpulengthstorage.hpp>
#include <gpu/gpubitarray.cuh>

#include <memorymanagement.hpp>

#include <config.hpp>
#include <readlibraryio.hpp>

#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <map>

namespace care{
namespace gpu{

struct DistributedReadStorage {
public:

    // struct MemoryInfo{
    //     size_t hostSizeInBytes{};
    //     std::vector<size_t> deviceSizeInBytes{};
    //     std::vector<int> deviceIds{};
    // };

    struct Statistics{
        int maximumSequenceLength = 0;
        int minimumSequenceLength = std::numeric_limits<int>::max();
    };

    struct SavedGpuPartitionData{
        void clear(){
            *this = SavedGpuPartitionData{};
        }
 
        enum class Type {Memory, File};

        Type sequenceDataLocation = Type::File;
        Type qualityDataLocation = Type::File;

        int partitionId = -42;

        std::vector<char> sequenceData;
        std::vector<char> qualityData;
    };

    struct SavedGpuData{
        void clear(){
            *this = SavedGpuData{};
        }

        std::vector<SavedGpuPartitionData> gpuPartitionData;
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
    mutable DistributedArray<unsigned int, read_number> distributedSequenceData;
    mutable DistributedArray<char, read_number> distributedQualities;

    std::map<int, GpuBitArray<read_number>> bitArraysUndeterminedBase;



    Statistics statistics;

	bool hasMoved = false;

    DistributedReadStorage() : DistributedReadStorage({}, 0, false, 0, 0){};

    DistributedReadStorage(const std::vector<int>& deviceIds_, read_number nReads, bool b, 
                            int minimum_sequence_length, int maximum_sequence_length);

    DistributedReadStorage(const std::vector<int>& deviceIds_, const std::vector<SequenceFileProperties>& sequenceFileProperties, bool qualityScores);

    DistributedReadStorage(const DistributedReadStorage& other) = delete;
    DistributedReadStorage& operator=(const DistributedReadStorage& other) = delete;

	DistributedReadStorage(DistributedReadStorage&& other);

	DistributedReadStorage& operator=(DistributedReadStorage&& other);

	MemoryUsage getMemoryInfo() const;

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
    SavedGpuData saveGpuDataAndFreeGpuMem(std::ofstream& stream, std::size_t numUsableBytes) const;
    SavedGpuPartitionData saveGpuPartitionData(
            int deviceId,
            std::ofstream& stream, 
            std::size_t* numBytesMustRemainFree) const;

    void loadGpuPartitionData(int deviceId, 
                                        std::ifstream& stream, 
                                        const SavedGpuPartitionData& saved) const;

    void allocateGpuData(int deviceId) const;
    void deallocateGpuData(int deviceId) const;

    SavedGpuPartitionData saveGpuPartitionDataAndFreeGpuMem(
                    int deviceId,
                    std::ofstream& stream, 
                    std::size_t* numUsableBytes) const;

    void allocGpuMemAndLoadGpuPartitionData(int deviceId, 
                                            std::ifstream& stream, 
                                            const SavedGpuPartitionData& saved) const;

    void allocGpuMemAndLoadGpuData(std::ifstream& stream, const SavedGpuData& saved) const;

    void setReads(ThreadPool* threadPool, read_number firstIndex, read_number lastIndex_excl, const Read* reads, int numReads);
    void setReads(ThreadPool* threadPool, read_number firstIndex, read_number lastIndex_excl, const std::vector<Read>& reads);
    void setReads(ThreadPool* threadPool, const std::vector<read_number>& indices, const Read* reads, int numReads);
    void setReads(ThreadPool* threadPool, const std::vector<read_number>& indices, const std::vector<Read>& reads);

    void setReadContainsN(read_number readId, bool contains);
    bool readContainsN(read_number readId) const;
    std::int64_t getNumberOfReadsWithN() const;

    void readsContainN_async(
        int deviceId,
        bool* d_result, 
        const read_number* d_positions, 
        int nPositions, 
        cudaStream_t stream) const;

    void readsContainN_async(
        int deviceId,
        bool* d_result, 
        const read_number* d_positions, 
        const int* d_nPositions,
        int nPositionsUpperBound, 
        cudaStream_t stream) const;

    void setReadsContainN_async(
        int deviceId,
        bool* d_values, 
        const read_number* d_positions, 
        int nPositions,
        cudaStream_t stream) const;
    
    void setGpuBitArraysFromVector();

    void constructionIsComplete();
    void allowModifications();

    GatherHandleSequences makeGatherHandleSequences() const;

    GatherHandleQualities makeGatherHandleQualities() const;

    void gatherSequenceDataToGpuBufferAsync(
                                ThreadPool* threadPool,
                                const GatherHandleSequences& handle,
                                unsigned int* d_sequence_data,
                                size_t outSequencePitchInInts,
                                const read_number* h_readIds,
                                const read_number* d_readIds,
                                int nReadIds,
                                int deviceId,
                                cudaStream_t stream) const;

    void gatherQualitiesToGpuBufferAsync(
                                ThreadPool* threadPool,
                                const GatherHandleQualities& handle,
                                char* d_quality_data,
                                size_t out_quality_pitch,
                                const read_number* h_readIds,
                                const read_number* d_readIds,
                                int nReadIds,
                                int deviceId,
                                cudaStream_t stream) const;

    void gatherSequenceLengthsToGpuBufferAsync(
                                int* d_lengths,
                                int deviceId,
                                const read_number* d_readIds,
                                int nReadIds,    
                                cudaStream_t stream) const;

    void gatherSequenceLengthsToHostBuffer(
                                int* lengths,
                                const read_number* readIds,
                                int nReadIds) const;


    private:
        void init(const std::vector<int>& deviceIds, read_number nReads, bool b, 
                    int minimum_sequence_length, int maximum_sequence_length);

        void init(const std::vector<int>& deviceIds_, const std::vector<SequenceFileProperties>& sequenceFileProperties, bool qualityScores);

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
