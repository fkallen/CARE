#ifndef DISTRIBUTED_ARRAY_HPP
#define DISTRIBUTED_ARRAY_HPP


#ifdef __NVCC__

#include <gpu/simpleallocation.cuh>

#include <vector>
#include <map>
#include <future>
#include <cassert>

struct DistributedArray{
public:
    struct PeerAccess{
        int numGpus;
        std::vector<int> accessMatrix;

        PeerAccess(){
            cudaGetDeviceCount(&numGpus); CUERR;
            accessMatrix.resize(numGpus * numGpus);
            for(int i = 0; i < numGpus; i++){
                for(int k = 0; k < numGpus; k++){
                    //device i can access device k?
                    cudaDeviceCanAccessPeer(&accessMatrix[i * numGpus + k], i, k);
                }
            }
        }

        bool hasPeerAccess(int device, int peerDevice) const{
            assert(device < numGpus);
            assert(peerDevice < numGpus);
            return accessMatrix[device * numGpus + peerDevice] == 1;
        }

        void enablePeerAccess(int device, int peerDevice) const{
            assert(hasPeerAccess(device, peerDevice));
            int oldId; cudaGetDevice(&oldId); CUERR;
            cudaSetDevice(device); CUERR;
            cudaError_t status = cudaDeviceEnablePeerAccess(peerDevice, 0);
            if(status != cudaSuccess){
                if(status == cudaErrorPeerAccessAlreadyEnabled){
                    std::cerr << "Peer access from " << device << " to " << peerDevice << " has already been enabled\n";
                }else{
                    CUERR;
                }
            }
            cudaSetDevice(oldId); CUERR;
        }

        void disablePeerAccess(int device, int peerDevice) const{
            assert(hasPeerAccess(device, peerDevice));
            int oldId; cudaGetDevice(&oldId); CUERR;
            cudaSetDevice(device); CUERR;
            cudaError_t status = cudaDeviceDisablePeerAccess(peerDevice); CUERR;
            if(status != cudaSuccess){
                if(status == cudaErrorPeerAccessNotEnabled){
                    std::cerr << "Peer access from " << device << " to " << peerDevice << " has not yet been enabled\n";
                }else{
                    CUERR;
                }
            }
            cudaSetDevice(oldId); CUERR;
        }

        void enableAllPeerAccesses(){
            for(int i = 0; i < numGpus; i++){
                for(int k = 0; k < numGpus; k++){
                    if(hasPeerAccess(i, k)){
                        enablePeerAccess(i, k);
                    }
                }
            }
        }

        void disableAllPeerAccesses(){
            for(int i = 0; i < numGpus; i++){
                for(int k = 0; k < numGpus; k++){
                    if(hasPeerAccess(i, k)){
                        disablePeerAccess(i, k);
                    }
                }
            }
        }
    };

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

    bool debug;
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

    void destroyGatherHandle(const std::unique_ptr<GatherHandle>& handle) const;

    std::future<void> gatherReadsInHostMemAsync(const std::unique_ptr<GatherHandle>&, size_t* indices, size_t numIds, char* result) const;

    void gatherReadsInHostMem(const std::unique_ptr<GatherHandle>&, size_t* indices, size_t numIds, char* result) const;

    void gatherReadsInGpuMem(const std::unique_ptr<GatherHandle>&, size_t* indices, size_t* d_indices, size_t numIds, int deviceId, char* d_result) const;

    //the same GatherHandle must not be used in another call until the results of the previous call are calculated
    void gatherReadsInGpuMemAsync(const std::unique_ptr<GatherHandle>&, size_t* indices, size_t* d_indices, size_t numIds, int deviceId, char* d_result, cudaStream_t stream) const;

    //d_result, d_indices must point to memory of device deviceId. d_indices[i] + indexOffset must be a local element index for this device
    void copyDataToGpuBufferAsync(char* d_result, const size_t* d_indices, size_t nIndices, int deviceId, cudaStream_t stream, size_t indexOffset) const;

    //d_result, d_indices must point to memory of device deviceId. d_indices[i] must be a local element index for this device
    void copyDataToGpuBufferAsync(char* d_result, const size_t* d_indices, size_t nIndices, int deviceId, cudaStream_t stream) const;
};





#endif








#endif
