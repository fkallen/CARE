#ifndef DISTRIBUTED_ARRAY_HPP
#define DISTRIBUTED_ARRAY_HPP


#ifdef __NVCC__

#include <gpu/simpleallocation.cuh>

#include <vector>
#include <map>
#include <future>


struct DistributedArray{
public:
    struct GatherHandle{
        SimpleAllocationPinnedHost<size_t> pinnedLocalIndices;
        SimpleAllocationPinnedHost<size_t> pinnedPermutationIndices;
        SimpleAllocationPinnedHost<char> pinnedResultData;
        std::vector<SimpleAllocationDevice<size_t>> deviceLocalReadIdsPerLocation;
        std::vector<SimpleAllocationDevice<char>> dataPerGpu;

        std::map<int, SimpleAllocationDevice<char>> tmpResultsOfDevice;
        std::map<int, SimpleAllocationDevice<size_t>> permutationIndicesOfDevice;

        std::vector<cudaStream_t> streamsPerGpu;
        std::vector<cudaEvent_t> eventsPerGpu;
    };

    int numGpus;
    int numLocations; //numGpus + 1
    int hostLocation; // numLocations - 1
    int preferedLocation;
    size_t numElements;
    size_t sizeOfElement;
    std::vector<int> deviceIds; //device ids which can be used to store data
    std::vector<float> maxFreeMemFraction; // how many elements of data can be stored on wich gpu
    std::vector<size_t> elementsPerLocation; // how many elements are stored on which location
    std::vector<size_t> elementsPerLocationPS; //inclusive prefix sum with leading zero
    std::vector<char*> dataPtrPerLocation; // the storage of each location. dataPtrPerLocation[hostLocation] is the host data. dataPtrPerLocation[gpu] is device data

    DistributedArray(std::vector<int> deviceIds_, std::vector<float> maxFreeMemFraction_, size_t numElements_, size_t sizeOfElement_, int preferedLocation_ = -1);

    ~DistributedArray();

    int getLocation(size_t index) const;

    void set(size_t index, const char* data);

    void set(size_t firstIndex, size_t lastIndex_excl, const char* data);

    //indices must be strictly increasing sequence where indices[i+1] == indices[i]+1
    void set(const std::vector<size_t>& indices, const char* data);

    void get(size_t index, char* result);

    std::unique_ptr<GatherHandle> makeGatherHandle() const;

    void destroyGatherHandle(std::unique_ptr<GatherHandle>&& handle) const;

    std::future<void> gatherReadsInHostMemAsync(const std::unique_ptr<GatherHandle>&, size_t* indices, size_t numIds, char* result) const;

    void gatherReadsInHostMem(const std::unique_ptr<GatherHandle>&, size_t* indices, size_t numIds, char* result) const;

    void gatherReadsInGpuMem(const std::unique_ptr<GatherHandle>&, size_t* indices, size_t* d_indices, size_t numIds, int deviceId, char* d_result) const;

    //the same GatherHandle must not be used in another call until the results of the previous call are calculated
    void gatherReadsInGpuMemAsync(const std::unique_ptr<GatherHandle>&, size_t* indices, size_t* d_indices, size_t numIds, int deviceId, char* d_result, cudaStream_t stream) const;

    //d_result, d_readIds must point to memory of device deviceId. read ids must be local read ids for this device
    void copyDataToGpuBufferAsync(char* d_result, const size_t* d_indices, size_t nIndices, int deviceId, cudaStream_t stream) const;
};





#endif








#endif
