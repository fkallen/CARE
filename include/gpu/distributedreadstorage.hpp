#ifndef CARE_DISTRIBUTED_READ_STORAGE_HPP
#define CARE_DISTRIBUTED_READ_STORAGE_HPP


#ifdef __NVCC__

#include <gpu/distributedarray.hpp>
#include <gpu/gpulengthstorage.hpp>
#include <gpu/gpubitarray.cuh>

#include <memorymanagement.hpp>
#include <cub/util_allocator.cuh>

#include <config.hpp>
#include <options.hpp>
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

    struct ReadInserterHandle{
    public:
        int deviceId;

        cudaStream_t stream1 = nullptr;
        cudaStream_t stream2 = nullptr;
        helpers::SimpleAllocationPinnedHost<char> h_decodedSequences;
        helpers::SimpleAllocationDevice<char> d_decodedSequences;
        helpers::SimpleAllocationPinnedHost<unsigned int> h_encodedSequences;
        helpers::SimpleAllocationDevice<unsigned int> d_encodedSequences;
        helpers::SimpleAllocationPinnedHost<int> h_lengths;
        helpers::SimpleAllocationDevice<int> d_lengths;
        std::vector<char> sequenceData;
        std::vector<int> sequenceLengths;
        std::vector<char> qualityData;

        ReadInserterHandle(){
            cudaGetDevice(&deviceId); CUERR;
            create();
        }

        ReadInserterHandle(int deviceId) : deviceId(deviceId){
            create();
        }

        ~ReadInserterHandle(){
            int cur = 0;
            cudaGetDevice(&cur); CUERR;
            cudaSetDevice(deviceId); CUERR;
            cudaStreamDestroy(stream1); CUERR;
            cudaStreamDestroy(stream2); CUERR;
            cudaSetDevice(cur); CUERR;
        }

        ReadInserterHandle(const ReadInserterHandle&) = delete;
        ReadInserterHandle& operator=(const ReadInserterHandle&) = delete;

        ReadInserterHandle(ReadInserterHandle&& rhs){
            *this = std::move(rhs);
        }

        ReadInserterHandle& operator=(ReadInserterHandle&& rhs){
            std::swap(deviceId, rhs.deviceId);
            std::swap(stream1, rhs.stream1);
            std::swap(stream2, rhs.stream2);
            std::swap(h_decodedSequences, rhs.h_decodedSequences);
            std::swap(d_decodedSequences, rhs.d_decodedSequences);
            std::swap(h_encodedSequences, rhs.h_encodedSequences);
            std::swap(d_encodedSequences, rhs.d_encodedSequences);
            std::swap(h_lengths, rhs.h_lengths);
            std::swap(d_lengths, rhs.d_lengths);
            std::swap(sequenceData, rhs.sequenceData);
            std::swap(sequenceLengths, rhs.sequenceLengths);
            std::swap(qualityData, rhs.qualityData);

            return *this;
        }
        
        void create(){
            int cur = 0;
            cudaGetDevice(&cur); CUERR;
            cudaSetDevice(deviceId); CUERR;
            cudaStreamCreate(&stream1); CUERR;
            cudaStreamCreate(&stream2); CUERR;
            cudaSetDevice(cur); CUERR;
        }    
    };

    // struct ReadInserter{
    // public:
    //     ReadInserter() = default;
    //     ReadInserter(const ReadInserter&) = delete;
    //     ReadInserter& operator=(const ReadInserter&) = delete;
    //     ReadInserter(ReadInserter&&) = default;
    //     ReadInserter& operator=(ReadInserter&&) = default;

    //     void setReads(
    //         ThreadPool* threadPool, 
    //         const read_number* indices, 
    //         const Read* reads, 
    //         int numReads
    //     )
    // private:
    //     struct ReadInserterHandle{
    //     public:
    //         int deviceId;

    //         cudaStream_t stream1 = nullptr;
    //         cudaStream_t stream2 = nullptr;
    //         helpers::SimpleAllocationPinnedHost<char> h_decodedSequences;
    //         helpers::SimpleAllocationDevice<char> d_decodedSequences;
    //         helpers::SimpleAllocationPinnedHost<char> h_encodedSequences;
    //         helpers::SimpleAllocationDevice<char> d_encodedSequences;
    //         helpers::SimpleAllocationPinnedHost<char> h_lengths;
    //         helpers::SimpleAllocationDevice<char> d_lengths;

    //         ReadInserterHandle(){
    //             cudaGetDevice(&deviceId); CUERR;
    //             create();
    //         }

    //         ReadInserterHandle(int deviceId) : deviceId(deviceId){
    //             create();
    //         }

    //         ~ReadInserterHandle(){
    //             int cur = 0;
    //             cudaGetDevice(&cur); CUERR;
    //             cudaSetDevice(deviceId); CUERR;
    //             cudaStreamDestroy(stream1); CUERR;
    //             cudaStreamDestroy(stream2); CUERR;
    //             cudaSetDevice(cur); CUERR;
    //         }

    //         ReadInserterHandle(const ReadInserterHandle&) = delete;
    //         ReadInserterHandle& operator=(const ReadInserterHandle&) = delete;
    //         ReadInserterHandle(ReadInserterHandle&&) = default;
    //         ReadInserterHandle& operator=(ReadInserterHandle&&) = default;
            
    //         void create(){
    //             int cur = 0;
    //             cudaGetDevice(&cur); CUERR;
    //             cudaSetDevice(deviceId); CUERR;
    //             cudaStreamCreate(&stream1); CUERR;
    //             cudaStreamCreate(&stream2); CUERR;
    //             cudaSetDevice(cur); CUERR;
    //         }    
    //     };

        
    //     ReadInserterHandle handle;
    // }; 
    
    using Length_t = int;

    using GatherHandleSequences = DistributedArray<unsigned int, read_number>::GatherHandle;
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

    cub::CachingDeviceAllocator cubCachingAllocator;

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

    void construct(
        std::vector<std::string> inputfiles,
        bool useQualityScores,
        read_number expectedNumberOfReads,
        int expectedMinimumReadLength,
        int expectedMaximumReadLength,
        int threads,
        bool showProgress
    );

    void constructPaired(
        std::vector<std::string> inputfiles,
        bool useQualityScores,
        read_number expectedNumberOfReads,
        int expectedMinimumReadLength,
        int expectedMaximumReadLength,
        int threads,
        bool showProgress
    );

	MemoryUsage getMemoryInfo() const;
    MemoryUsage getMemoryInfoOfGatherHandleSequences(const GatherHandleSequences& handle) const;
    MemoryUsage getMemoryInfoOfGatherHandleQualities(const GatherHandleQualities& handle) const;

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

    

    void setReads(
        ReadInserterHandle& handle,
        ThreadPool* threadPool, 
        const read_number* indices, 
        const Read* reads, 
        int numReads
    );

    auto makeReadInserter(int deviceId = 0){
        auto handleptr = std::make_unique<ReadInserterHandle>(deviceId);

        return  [
                    this, 
                    handleptr = std::move(handleptr)
                ](
                    ThreadPool* threadPool, 
                    const read_number* indices, 
                    const Read* reads, 
                    int numReads
                ){
                    this->setReads(*handleptr, threadPool, indices, reads, numReads);
                };
        
    }

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
        void setSequences(const read_number* indices, const char* data, int numReads);
        void setSequenceLengths(read_number firstIndex, read_number lastIndex_excl, const Length_t* data);
        void setSequenceLengths(const read_number* indices, const Length_t* data, int numReads);
        void setQualities(read_number firstIndex, read_number lastIndex_excl, const char* data);
        void setQualities(const read_number* indices, const char* data, int numReads);
};




}
}

#endif




#endif
