#ifndef DISTRIBUTED_ARRAY_HPP
#define DISTRIBUTED_ARRAY_HPP


#ifdef __NVCC__

#include <gpu/simpleallocation.cuh>
#include <gpu/utility_kernels.cuh>
#include <hpc_helpers.cuh>
#include <gpu/nvtxtimelinemarkers.hpp>
#include <gpu/peeraccess.hpp>
#include <threadpool.hpp>
//#include <util.hpp>
#include <memorymanagement.hpp>

#include <cub/cub.cuh>
#include <cub/iterator/discard_output_iterator.cuh>

#include <algorithm>
#include <numeric>
#include <cassert>
#include <iterator>
#include <vector>
#include <map>
#include <future>
#include <cassert>
#include <omp.h>
#include <fstream>
#include <mutex>
#include <condition_variable>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// selects all elements e in inputIndices for which selector returns true
template<int blocksize, class Index_t, class Selector>
__global__
void distrArrayCompactIndicesOfLocationKernel(
    Index_t* __restrict__ outputIndices,
    Index_t* __restrict__ numOutputIndices,
    const Index_t* __restrict__ inputIndices,
    Index_t numIds,
    Selector selector
){
    constexpr int N = 4;

    using BlockScan = cub::BlockScan<int, blocksize>;
    using BlockLoad = cub::BlockLoad<Index_t, blocksize, N, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

    __shared__ union{
        typename BlockScan::TempStorage scan;
        typename BlockLoad::TempStorage load;
    } tempstorage;

    __shared__ int numSelected;

    if(threadIdx.x == 0){
        numSelected = 0;
    }

    __syncthreads();

    const Index_t iters = SDIV(numIds, N);

    for(int iter = 0; iter < iters; iter++){
        Index_t myelems[N];
        int flags[N];
        int ps[N];
        int block_aggregate;                            
        
        const Index_t begin = blocksize * N * iter;
        const Index_t end = min(blocksize * N * (iter+1), numIds);

        BlockLoad(tempstorage.load).Load(
            inputIndices + begin, 
            myelems,
            end - begin,
            std::numeric_limits<Index_t>::max()
        );

        #pragma unroll
        for(int i = 0; i < N; i++){
            flags[i] = 0;

            if(selector(myelems[i])){
                flags[i] = 1;
            }
        }

        __syncthreads();

        BlockScan(tempstorage.scan).ExclusiveSum(
            flags, 
            ps, 
            block_aggregate
        );

        #pragma unroll
        for(int i = 0; i < N; i++){
            if(flags[i] == 1){
                const Index_t destPos = ps[i] + numSelected;
                outputIndices[destPos] = myelems[i];
            }
        }

        __syncthreads();

        if(threadIdx.x == 0){
            numSelected += block_aggregate;
        }

        __syncthreads();
    }

    if(threadIdx.x == 0){
        *numOutputIndices = numSelected;
    }
}


template<class Index_t, class Value_t>
__global__
void distrArrayGatherKernel(
        Value_t* __restrict__ result, 
        const Value_t* __restrict__ sourceData, 
        const Index_t* __restrict__ indices, 
        Index_t nIndices,
        Index_t indexOffset, 
        size_t resultPitchValueTs,
        size_t numCols){

    for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; i < nIndices * numCols; i += size_t(blockDim.x) * gridDim.x){
        const Index_t outputrow = i / numCols;
        const Index_t inputrow = indices[outputrow] + indexOffset;
        const Index_t col = i % numCols;
        result[size_t(outputrow) * resultPitchValueTs + col] 
                = sourceData[size_t(inputrow) * numCols + col];
    }
}

template<class Index_t, class Value_t>
__global__
void distrArrayGatherKernel2(
        Value_t* __restrict__ result, 
        const Value_t* __restrict__ sourceData, 
        const Index_t* __restrict__ indices, 
        const Index_t* __restrict__ nIndicesPtr,
        Index_t indexOffset, 
        size_t resultPitchValueTs,
        size_t numCols){

    const Index_t nIndices = *nIndicesPtr;

    for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; i < nIndices * numCols; i += size_t(blockDim.x) * gridDim.x){
        const Index_t outputrow = i / numCols;
        const Index_t inputrow = indices[outputrow] + indexOffset;
        const Index_t col = i % numCols;
        result[size_t(outputrow) * resultPitchValueTs + col] 
                = sourceData[size_t(inputrow) * numCols + col];
    }
}

enum class DistributedArrayLayout{
    GPUBlock, GPUEqual
};

template<class Value_t, class Index_t = size_t>
struct DistributedArray{
public:

    

    struct GatherHandleStruct{
        SimpleAllocationPinnedHost<Index_t> pinnedLocalIndices;
        SimpleAllocationPinnedHost<Index_t> pinnedPermutationIndices;
        SimpleAllocationPinnedHost<Value_t> pinnedResultData;
        std::vector<SimpleAllocationDevice<Index_t>> deviceLocalIndicesPerLocation;
        std::vector<SimpleAllocationDevice<Index_t>> numLocalIndicesPerLocation;
        std::vector<SimpleAllocationDevice<Index_t*>> ptrsToNumIndicesPerLocation;
        std::vector<SimpleAllocationDevice<Value_t>> dataPerGpu;

        std::map<int, SimpleAllocationDevice<Value_t>> tmpResultsOfDevice;
        std::map<int, SimpleAllocationDevice<Index_t>> permutationIndicesOfDevice;
        std::map<int, SimpleAllocationDevice<Index_t>> localIndicesOnDevice;

        std::vector<cudaStream_t> streamsPerGpu;
        std::vector<cudaEvent_t> eventsPerGpu;

        cudaEvent_t readyEvent;
        care::ThreadPool::ParallelForHandle pforHandle;


        // -------------------------------------------
        std::vector<SimpleAllocationPinnedHost<Index_t>> pinnedIndicesOfGpuLocation; //numGpus
        std::vector<SimpleAllocationDevice<Index_t>> d_indicesOfGpuLocation; //numGpus

        std::vector<Index_t> indicesOfHostLocation;
        std::vector<std::size_t> numIndicesPerLocation; //numLocations
        std::vector<std::size_t> numIndicesPerLocationPS; //numLocations+1


        std::vector<SimpleAllocationPinnedHost<Index_t>> pinnedDestinationPositionsOfLocation; //numLocations
        std::vector<SimpleAllocationDevice<Index_t>> d_destinationPositionsOfGpuLocation; //numGpus

        std::vector<SimpleAllocationPinnedHost<Value_t>> pinnedGatheredElementsOfLocation; //numLocations
        std::vector<SimpleAllocationDevice<Value_t>> d_gatheredElementsOfGpuLocation; //numGpus

        std::map<int, SimpleAllocationDevice<Value_t>> map_d_tmpResults; //tmp buffers to store all gathered elements on the destination gpu
        std::map<int, SimpleAllocationDevice<Index_t>> map_d_destinationPositionsOfGpu;
        std::map<int, SimpleAllocationDevice<Index_t>> map_d_elementsPerLocationPS;
        std::map<int, SimpleAllocationDevice<Index_t>> map_d_numIndicesPerLocation;
        std::map<int, SimpleAllocationDevice<Index_t>> map_d_numIndicesPerLocationPS;
        std::map<int, std::vector<SimpleAllocationDevice<Index_t>>> map_d_indicesForLocationsVector;
        std::map<int, std::vector<SimpleAllocationDevice<Index_t>>> map_d_destinationPositionsForLocationsVector;

        std::map<int, SimpleAllocationDevice<Index_t*>> map_d_indicesForLocationsPointers;
        std::map<int, SimpleAllocationDevice<Index_t*>> map_d_destinationPositionsForLocationsPointers;

        SimpleAllocationPinnedHost<Index_t*> pinnedPointersForLocations;
        SimpleAllocationPinnedHost<Index_t*> pinnedPointersForLocations2;


        std::vector<cudaStream_t> streamsPerGpuLocation;
        std::vector<cudaEvent_t> eventsPerGpuLocation;

        std::map<int, cudaStream_t> map_streams;
        std::map<int, cudaEvent_t> map_events;

        std::vector<int> registeredDeviceIds;

    };

    using GatherHandle = std::unique_ptr<GatherHandleStruct>;
    using PeerAccess_t = PeerAccess;

    struct SinglePartitionInfo{
        bool isSinglePartition = false;
        int locationId = -1;
    };

    bool debug;
    int numGpus;
    int numLocations; //numGpus + 1
    int hostLocation; // numLocations - 1
    int preferedLocation;
    int maxNumGpus = 32;
    DistributedArrayLayout layout;
    Index_t numRows;
    Index_t numColumns;
    size_t sizeOfElement;
    SinglePartitionInfo singlePartitionInfo;
    std::vector<int> deviceIds; //device ids which can be used to store data
    std::vector<size_t> memoryLimitBytesPerGPU; // how many elements of data can be stored on wich gpu
    std::vector<Index_t> elementsPerLocation; // how many elements are stored on which location
    std::vector<Index_t> elementsPerLocationPS; //inclusive prefix sum with leading zero

    std::vector<Value_t*> dataPtrPerLocation; // the storage of each location. dataPtrPerLocation[hostLocation] is the host data. dataPtrPerLocation[gpu] is device data
    PeerAccess_t peerAccess;

    DistributedArray()
        : DistributedArray({},{}, DistributedArrayLayout::GPUEqual, 0,0,-1){
    }

    DistributedArray(std::vector<int> deviceIds_, 
                    std::vector<size_t> memoryLimitBytesPerGPU_, 
                    DistributedArrayLayout layout_,
                    Index_t numRows_, 
                    Index_t numCols_, 
                    int preferedLocation_ = -1)
                    : debug(false),
                    numGpus(deviceIds_.size()),
                    numLocations(numGpus+1),
                    hostLocation(numLocations-1),
                    preferedLocation(preferedLocation_),
                    layout(layout_),
                    numRows(numRows_),
                    numColumns(numCols_),
                    sizeOfElement(numCols_ * sizeof(Value_t)),
                    deviceIds(std::move(deviceIds_)),
                    memoryLimitBytesPerGPU(std::move(memoryLimitBytesPerGPU_)),
                    peerAccess(PeerAccess_t{}){

        assert(deviceIds.size() == memoryLimitBytesPerGPU.size());

        assert(deviceIds.size() <= maxNumGpus);

        elementsPerLocation.resize(numLocations, 0);
        elementsPerLocationPS.resize(numLocations+1, 0);
        dataPtrPerLocation.resize(numLocations, nullptr);

        peerAccess.enableAllPeerAccesses();

        if(numRows > 0 && numColumns > 0){

            int oldId; cudaGetDevice(&oldId); CUERR;

            size_t totalRequiredMemory = numRows * sizeOfElement;

            bool preferedLocationIsSufficient = false;

            if(preferedLocation != -1 && preferedLocation != hostLocation){
                if(memoryLimitBytesPerGPU[preferedLocation] >= totalRequiredMemory){
                    preferedLocationIsSufficient = true;
                }
            }

            if(preferedLocationIsSufficient){
                wrapperCudaSetDevice(deviceIds[preferedLocation]); CUERR;
                elementsPerLocation[preferedLocation] = numRows;
                cudaMalloc(&dataPtrPerLocation[preferedLocation], totalRequiredMemory); CUERR;
            }else{

                size_t remainingElements = numRows;
                if(layout == DistributedArrayLayout::GPUBlock){
                    for(int gpu = 0; gpu < numGpus && remainingElements > 0; gpu++){
                        wrapperCudaSetDevice(deviceIds[gpu]); CUERR;

                        size_t rows = std::min(remainingElements, memoryLimitBytesPerGPU[gpu] / sizeOfElement);
                        elementsPerLocation[gpu] = rows;
                        if(rows == 0){
                            continue;
                        }

                        cudaMalloc(&(dataPtrPerLocation[gpu]), rows * sizeOfElement); CUERR;

                        remainingElements -= rows;
                    }
                }else if(layout == DistributedArrayLayout::GPUEqual){
                    std::size_t totalPossibleElementsPerGpu = 0;

                    std::vector<std::size_t> possibleElementsPerGpu(numGpus);
                    for(int gpu = 0; gpu < numGpus; gpu++){
                        possibleElementsPerGpu[gpu] = memoryLimitBytesPerGPU[gpu] / sizeOfElement;
                        elementsPerLocation[gpu] = 0;

                        totalPossibleElementsPerGpu += possibleElementsPerGpu[gpu];
                    }

                    for(std::size_t i = 0; i < std::min(totalPossibleElementsPerGpu, std::size_t(numRows)); i++){
                        const int gpu = i % numGpus;
                        elementsPerLocation[gpu]++;
                    }

                    for(int gpu = 0; gpu < numGpus; gpu++){
                        wrapperCudaSetDevice(deviceIds[gpu]); CUERR;
                        cudaMalloc(&(dataPtrPerLocation[gpu]), elementsPerLocation[gpu] * sizeOfElement); CUERR;
                    }

                    if(std::size_t(numRows) > totalPossibleElementsPerGpu){
                        remainingElements = std::size_t(numRows) - totalPossibleElementsPerGpu;
                    }else{
                        remainingElements = 0;
                    }
                }

                //remaining elements are stored in host memory
                if(remainingElements > 0){
                    dataPtrPerLocation[hostLocation] = new Value_t[remainingElements * numColumns];
                    elementsPerLocation[hostLocation] = remainingElements;
                }
            }

            std::partial_sum(elementsPerLocation.begin(), elementsPerLocation.end(), elementsPerLocationPS.begin()+1);

            auto singlepartitioniter = std::find(elementsPerLocation.begin(), elementsPerLocation.end(), numRows);
            singlePartitionInfo.isSinglePartition = singlepartitioniter != elementsPerLocation.end();
            singlePartitionInfo.locationId = std::distance(elementsPerLocation.begin(), singlepartitioniter);

            if(true){
                std::cerr << "DistributedArray:\n";
                std::cerr << "device ids: [";
                std::copy(deviceIds.begin(), deviceIds.end(), std::ostream_iterator<int>(std::cerr, " "));
                std::cerr << "]\n";
                std::cerr << "SinglePartitionInfo: " << singlePartitionInfo.isSinglePartition << ", " << singlePartitionInfo.locationId << '\n';
                std::cerr << "elements per location: [";
                std::copy(elementsPerLocation.begin(), elementsPerLocation.end(), std::ostream_iterator<Index_t>(std::cerr, " "));
                std::cerr << "]\n";
            }

            wrapperCudaSetDevice(oldId); CUERR;
        }
    }

    DistributedArray(const DistributedArray&) = delete;
    DistributedArray(DistributedArray&& rhs){
        operator=(std::move(rhs));
    }

    DistributedArray& operator=(const DistributedArray&) = delete;

    DistributedArray& operator=(DistributedArray&& rhs){
        destroy();
        debug = rhs.debug;
        numGpus = rhs.numGpus;
        numLocations = rhs.numLocations;
        hostLocation = rhs.hostLocation;
        preferedLocation = rhs.preferedLocation;
        layout = rhs.layout;
        numRows = rhs.numRows;
        numColumns = rhs.numColumns;
        sizeOfElement = rhs.sizeOfElement;
        singlePartitionInfo = rhs.singlePartitionInfo;
        deviceIds = std::move(rhs.deviceIds);
        memoryLimitBytesPerGPU = std::move(rhs.memoryLimitBytesPerGPU);
        elementsPerLocation = std::move(rhs.elementsPerLocation);
        elementsPerLocationPS = std::move(rhs.elementsPerLocationPS);
        dataPtrPerLocation = std::move(rhs.dataPtrPerLocation);
        peerAccess = std::move(rhs.peerAccess);

        rhs.numGpus = 0;
        rhs.numLocations = 1;
        rhs.hostLocation = 0;
        rhs.preferedLocation = -1;
        rhs.numRows = 0;
        rhs.numColumns = 0;
        rhs.sizeOfElement = 0;
        rhs.singlePartitionInfo = SinglePartitionInfo{};
        rhs.deviceIds.clear();
        rhs.memoryLimitBytesPerGPU.clear();
        rhs.elementsPerLocation.clear();
        rhs.elementsPerLocationPS.clear();
        rhs.dataPtrPerLocation.clear();

        return *this;
    }

    ~DistributedArray(){
        destroy();
    }

    inline
    cudaError_t wrapperCudaSetDevice(int newId) const{
        // int currentId = -1;
        // cudaGetDevice(&currentId); CUERR;

        // std::cerr << "cudaSetDevice " << currentId << " -> " << newId << '\n';
        cudaError_t res = cudaSetDevice(newId); CUERR;
        return res;
    }



    // template<class Ptr>
    // Ptr offsetPtr(Ptr valuetBasePointer, Index_t rowIndex){
    //     return (Ptr)(((char*)(valuetBasePointer)) + sizeOfElement * rowIndex);
    // }

    const Value_t* offsetPtr(const Value_t* valuetBasePointer, Index_t rowIndex) const {
        const size_t row = rowIndex;
        return (const Value_t*)(((const char*)(valuetBasePointer)) + sizeOfElement * row);
    }

    Value_t* offsetPtr(Value_t* valuetBasePointer, Index_t rowIndex) const {
        const size_t row = rowIndex;
        return (Value_t*)(((char*)(valuetBasePointer)) + sizeOfElement * row);
    }

    int getLocation(Index_t index) const{
    	int location = 0;
    	for(; location < numLocations; location++){
    		if(index < elementsPerLocationPS[location+1])
                        break;
    	}
        if(location >= numLocations){
            std::cerr << location << " " << numLocations << "\n";
            std::copy(elementsPerLocationPS.begin(), elementsPerLocationPS.end(), std::ostream_iterator<Index_t>(std::cerr, " "));
            std::cerr << "\n";
        }
        assert(location < numLocations);
    	return location;
    }

    void set(Index_t index, const Value_t* data){
        int location = getLocation(index);
        Index_t localIndex = index - elementsPerLocationPS[location];
        Value_t* destPtr = offsetPtr(dataPtrPerLocation[location], localIndex);
        if(location == hostLocation){
            std::copy_n(data, numColumns, destPtr);
        }else{
            int oldDevice; cudaGetDevice(&oldDevice); CUERR;
            wrapperCudaSetDevice(deviceIds[location]); CUERR;
            cudaMemcpy(destPtr, data, sizeOfElement, H2D); CUERR;
            wrapperCudaSetDevice(oldDevice); CUERR;
        }
    }

    void set(Index_t firstIndex, Index_t lastIndex_excl, const Value_t* data){
        std::vector<Index_t> indices(lastIndex_excl - firstIndex);
        std::iota(indices.begin(), indices.end(), firstIndex);

        set(indices, data);
    }

    void setSafe(Index_t firstIndex, Index_t lastIndex_excl, const Value_t* data){
        std::vector<Index_t> indices(lastIndex_excl - firstIndex);
        std::iota(indices.begin(), indices.end(), firstIndex);

        setSafe(indices, data);
    }

    //indices must be strictly increasing sequence where indices[i+1] == indices[i]+1
    void set(const std::vector<Index_t>& indices, const Value_t* data){
        {
            bool isOk = true;
            for(size_t i = 0; i < indices.size()-1; i++){
                if(indices[i] + 1 != indices[i+1]){
                    isOk = false;
                    break;
                }
            }

            if(!isOk){
                setSafe(indices, data);
            }
        }
        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        std::vector<int> firstLocalIndices(numLocations, -1);
        std::vector<int> hitsPerLocation(numLocations, 0);
        int previousLocation = -1;
    	for(size_t i = 0; i < size_t(indices.size()); i++){
    		int location = getLocation(indices[i]);
            assert(location >= previousLocation);
            if(location != previousLocation){
                assert(firstLocalIndices[location] == -1);
                Index_t localIndex = indices[i] - elementsPerLocationPS[location];
                firstLocalIndices[location] = localIndex;
            }

    		hitsPerLocation[location]++;
            previousLocation = location;
    	}

    	std::vector<int> hitsPerLocationPrefixSum(numLocations+1,0);
    	std::partial_sum(hitsPerLocation.begin(), hitsPerLocation.end(), hitsPerLocationPrefixSum.begin()+1);


        for(int location = 0; location < numLocations; location++){
            if(hitsPerLocation[location] > 0){
                assert(firstLocalIndices[location] >= 0);

                size_t srcOffset = hitsPerLocationPrefixSum[location] * numColumns;
                const Value_t* srcPtr = data + srcOffset;

                Value_t* destPtr = offsetPtr(dataPtrPerLocation[location], firstLocalIndices[location]);
                int numHits = hitsPerLocation[location];

                std::cerr << "set copy from " << (void*)srcPtr << " to " << (void*)destPtr << "\n";
                std::cerr << indices[0] << " " << indices[1] << " " << indices[2] << '\n';

                if(location == hostLocation){
                    std::copy_n(srcPtr, numColumns * numHits, destPtr);
                }else{
                    wrapperCudaSetDevice(deviceIds[location]); CUERR;
                    cudaMemcpy(destPtr, srcPtr, sizeOfElement * numHits, H2D); CUERR;
                }
            }
        }

        wrapperCudaSetDevice(oldDevice); CUERR;
    }

    void setSafe(const std::vector<Index_t>& indices, const Value_t* data){
        assert(std::is_sorted(indices.begin(), indices.end()));
        assert(!indices.empty());

        int oldDevice; cudaGetDevice(&oldDevice); CUERR;




        if(singlePartitionInfo.isSinglePartition){
            const int locationId = singlePartitionInfo.locationId;

            bool isConsecutiveIndices = true;
            for(size_t k = 0; k < indices.size()-1; k++){
                if(indices[k]+1 != indices[k+1]){
                    isConsecutiveIndices = false;
                    break;
                }
            }

            //std::cerr << "setSafe isConsecutiveIndices? " << isConsecutiveIndices << "\n";

            if(isConsecutiveIndices){
                if(locationId == hostLocation){
                    Index_t start = indices[0];
                    Value_t* destPtr = offsetPtr(dataPtrPerLocation[locationId], start);

                    std::copy_n(data, indices.size() * numColumns, destPtr);
                    return;
                }else{
                    Index_t start = indices[0];
                    Value_t* destPtr = offsetPtr(dataPtrPerLocation[locationId], start);
                    wrapperCudaSetDevice(deviceIds[locationId]); CUERR;
                    cudaMemcpy(destPtr, data, indices.size() * sizeOfElement, H2D); CUERR;
                    wrapperCudaSetDevice(oldDevice); CUERR;
                    return;
                }
            }


        }

        std::vector<int> localIndices(indices.size(), -1);
        std::vector<int> hitsPerLocation(numLocations, 0);

        for(size_t i = 0; i < size_t(indices.size()); i++){
            int location = getLocation(indices[i]);
            Index_t localIndex = indices[i] - elementsPerLocationPS[location];
            localIndices[i] = localIndex;
            hitsPerLocation[location]++;
        }

        std::vector<int> hitsPerLocationPrefixSum(numLocations+1,0);
        std::partial_sum(hitsPerLocation.begin(), hitsPerLocation.end(), hitsPerLocationPrefixSum.begin()+1);

        for(int location = 0; location < numLocations; location++){
            if(hitsPerLocation[location] > 0){
                int numHits = hitsPerLocation[location];
                int psOffset = hitsPerLocationPrefixSum[location];
                int num = 0;
                while(num < numHits){
                    int from = num;
                    int to = num+1;
                    while(to < numHits && localIndices[psOffset+to-1] + 1 == localIndices[psOffset+to]){
                        to++;
                    }

                    //copy elements [psOffset+from, psOffset+to[ to array
                    size_t srcOffset = (psOffset+from) * numColumns;
                    const Value_t* srcPtr = data + srcOffset;
                    Value_t* destPtr = offsetPtr(dataPtrPerLocation[location], localIndices[psOffset+from]);
                    if(location == hostLocation){
                        std::copy_n(srcPtr, size_t(numColumns) * (to-from), destPtr);
                    }else{
                        wrapperCudaSetDevice(deviceIds[location]); CUERR;
                        cudaMemcpy(destPtr, srcPtr, sizeOfElement * (to-from), H2D); CUERR;
                    }

                    num = to;
                }
            }
        }

        wrapperCudaSetDevice(oldDevice); CUERR;
    }

    void get(Index_t index, Value_t* result){
        int location = getLocation(index);
        Index_t localIndex = index - elementsPerLocationPS[location];
        const Value_t* srcPtr = offsetPtr(dataPtrPerLocation[location], localIndex);

        if(location == hostLocation){
            std::copy_n(srcPtr, numColumns, result);
        }else{
            int oldDevice; cudaGetDevice(&oldDevice); CUERR;
            wrapperCudaSetDevice(deviceIds[location]); CUERR;
            cudaMemcpy(result, srcPtr, sizeOfElement, D2H); CUERR;
            wrapperCudaSetDevice(oldDevice); CUERR;
        }
    }

    GatherHandle makeGatherHandle() const{
        auto handle = std::make_unique<GatherHandleStruct>();
        handle->deviceLocalIndicesPerLocation.resize(numLocations);
        handle->numLocalIndicesPerLocation.resize(numLocations);
        handle->ptrsToNumIndicesPerLocation.resize(numLocations);
        handle->dataPerGpu.resize(numGpus);
        handle->streamsPerGpu.resize(numGpus);
        handle->eventsPerGpu.resize(numGpus);
        int oldDevice; cudaGetDevice(&oldDevice); CUERR;
        for(int gpu = 0; gpu < numGpus; gpu++){
            wrapperCudaSetDevice(deviceIds[gpu]); CUERR;
            cudaStreamCreate(&(handle->streamsPerGpu[gpu])); CUERR;
            cudaEventCreate(&(handle->eventsPerGpu[gpu])); CUERR;

            handle->numLocalIndicesPerLocation[gpu].resize(1);
            handle->ptrsToNumIndicesPerLocation[gpu].resize(numLocations);
        }

        std::vector<Index_t*> numPtrs(numLocations);
        for(int loc = 0; loc < numLocations; loc++){
            numPtrs[loc] = handle->numLocalIndicesPerLocation[loc].get();
        }

        for(int gpu = 0; gpu < numGpus; gpu++){
            wrapperCudaSetDevice(deviceIds[gpu]); CUERR;

            cudaMemcpy(          
                handle->ptrsToNumIndicesPerLocation[gpu].get(),
                numPtrs.data(),
                sizeof(Index_t*) * numLocations,
                H2D
            ); CUERR;
        }


        cudaEventCreate(&(handle->readyEvent)); CUERR;

        // -----------------------------------------
        wrapperCudaSetDevice(oldDevice); CUERR;

        handle->pinnedIndicesOfGpuLocation.resize(numGpus);
        handle->d_indicesOfGpuLocation.resize(numGpus);
        handle->numIndicesPerLocation.resize(numLocations);
        handle->numIndicesPerLocationPS.resize(numLocations+1);
        handle->pinnedDestinationPositionsOfLocation.resize(numLocations);
        handle->d_destinationPositionsOfGpuLocation.resize(numGpus);
        handle->pinnedGatheredElementsOfLocation.resize(numLocations);
        handle->d_gatheredElementsOfGpuLocation.resize(numGpus);
        handle->pinnedDestinationPositionsOfLocation.resize(numLocations);
        handle->pinnedPointersForLocations.resize(numLocations);
        handle->pinnedPointersForLocations2.resize(numLocations);

        handle->streamsPerGpuLocation.resize(numGpus);
        handle->eventsPerGpuLocation.resize(numGpus);

        for(int gpu = 0; gpu < numGpus; gpu++){
            registerDeviceIdForHandlenew(handle, deviceIds[gpu]);

            wrapperCudaSetDevice(deviceIds[gpu]); CUERR;
            cudaStreamCreate(&(handle->streamsPerGpuLocation[gpu])); CUERR;
            cudaEventCreateWithFlags(&(handle->eventsPerGpuLocation[gpu]), cudaEventDisableTiming); CUERR;
        }

        wrapperCudaSetDevice(oldDevice); CUERR;

        return handle;
    }

    void registerDeviceIdForHandlenew(const GatherHandle& handle, int deviceId) const{
        auto it = std::find(
            handle->registeredDeviceIds.begin(), 
            handle->registeredDeviceIds.end(), 
            deviceId
        );

        if(it == handle->registeredDeviceIds.end()){
            int oldId = 0;
            cudaGetDevice(&oldId); CUERR;
            wrapperCudaSetDevice(deviceId); CUERR;

            handle->map_d_tmpResults.emplace(deviceId, SimpleAllocationDevice<Value_t>{});
            handle->map_d_destinationPositionsOfGpu.emplace(deviceId, SimpleAllocationDevice<Index_t>{});
            handle->map_d_elementsPerLocationPS.emplace(deviceId, SimpleAllocationDevice<Index_t>(numLocations + 1));

            SimpleAllocationDevice<Index_t> aaa1(numLocations);
            handle->map_d_numIndicesPerLocation.emplace(deviceId, std::move(aaa1));
            SimpleAllocationDevice<Index_t> aaa2(numLocations+1);
            handle->map_d_numIndicesPerLocationPS.emplace(deviceId, std::move(aaa2));

            std::vector<SimpleAllocationDevice<Index_t>> vec1(numLocations + 1);
            handle->map_d_indicesForLocationsVector.emplace(deviceId, std::move(vec1));
            std::vector<SimpleAllocationDevice<Index_t>> vec2(numLocations + 1);
            handle->map_d_destinationPositionsForLocationsVector.emplace(deviceId, std::move(vec2));

            cudaMemcpy(
                handle->map_d_elementsPerLocationPS[deviceId].get(),
                elementsPerLocationPS.data(),
                sizeof(Index_t) * (numLocations+1),
                H2D
            ); CUERR; 

            handle->map_d_indicesForLocationsPointers.emplace(deviceId, SimpleAllocationDevice<Index_t*>(numLocations + 1));
            handle->map_d_destinationPositionsForLocationsPointers.emplace(deviceId, SimpleAllocationDevice<Index_t*>(numLocations + 1));

            cudaStream_t stream;
            cudaStreamCreate(&stream); CUERR;
            handle->map_streams[deviceId] = std::move(stream);

            cudaEvent_t event;
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming); CUERR;
            handle->map_events[deviceId] = std::move(event);       

            handle->registeredDeviceIds.push_back(deviceId);

            wrapperCudaSetDevice(oldId); CUERR;
        }
    }

    void unregisterDeviceIdForHandlenew(const GatherHandle& handle, int deviceId) const{
        auto it = std::find(
            handle->registeredDeviceIds.begin(), 
            handle->registeredDeviceIds.end(), 
            deviceId
        );

        if(it != handle->registeredDeviceIds.end()){
            int oldId = 0;
            cudaGetDevice(&oldId); CUERR;
            wrapperCudaSetDevice(deviceId); CUERR;

            handle->map_d_tmpResults[deviceId].destroy();
            handle->map_d_destinationPositionsOfGpu[deviceId].destroy();
            handle->map_d_elementsPerLocationPS[deviceId].destroy();
            handle->map_d_numIndicesPerLocation[deviceId].destroy();
            handle->map_d_numIndicesPerLocationPS[deviceId].destroy();

            handle->map_d_indicesForLocationsVector[deviceId].clear();
            handle->map_d_destinationPositionsForLocationsVector[deviceId].clear();

            handle->map_d_indicesForLocationsPointers[deviceId].destroy();
            handle->map_d_destinationPositionsForLocationsPointers[deviceId].destroy();
            
            cudaStreamDestroy(handle->map_streams[deviceId]);
            cudaEventDestroy(handle->map_events[deviceId]);   

            handle->registeredDeviceIds.erase(it);

            wrapperCudaSetDevice(oldId); CUERR;
        }
    }

    void destroyGatherHandleStruct(const GatherHandle& handle) const{
        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        handle->pinnedLocalIndices = std::move(SimpleAllocationPinnedHost<Index_t>{});
        handle->pinnedResultData = std::move(SimpleAllocationPinnedHost<Value_t>{});
        handle->pinnedResultData = std::move(SimpleAllocationPinnedHost<Value_t>{});

        for(size_t gpu = 0; gpu < handle->dataPerGpu.size(); gpu++){
            wrapperCudaSetDevice(deviceIds[gpu]); CUERR;

            handle->deviceLocalIndicesPerLocation[gpu] = std::move(SimpleAllocationDevice<Index_t>{});
            handle->dataPerGpu[gpu] = std::move(SimpleAllocationDevice<Value_t>{});
            cudaStreamDestroy(handle->streamsPerGpu[gpu]); CUERR;
            cudaEventDestroy(handle->eventsPerGpu[gpu]); CUERR;
        }

        cudaEventDestroy(handle->readyEvent); CUERR;

        for(auto& pair : handle->tmpResultsOfDevice){
            wrapperCudaSetDevice(pair.first); CUERR;
            pair.second = std::move(SimpleAllocationDevice<Value_t>{});
        }

        for(auto& pair : handle->permutationIndicesOfDevice){
            wrapperCudaSetDevice(pair.first); CUERR;
            pair.second = std::move(SimpleAllocationDevice<Index_t>{});
        }

        //    -----------------

        for(int gpu = 0; gpu < numGpus; gpu++){
            wrapperCudaSetDevice(deviceIds[gpu]); CUERR;
            handle->d_indicesOfGpuLocation[gpu].destroy();
            handle->d_destinationPositionsOfGpuLocation[gpu].destroy();
            handle->d_gatheredElementsOfGpuLocation[gpu].destroy();

            cudaStreamDestroy(handle->streamsPerGpuLocation[gpu]);
            cudaEventDestroy(handle->eventsPerGpuLocation[gpu]);
        }

        while(handle->registeredDeviceIds.size() > 0){
            unregisterDeviceIdForHandlenew(handle, handle->registeredDeviceIds[0]);
        }

        wrapperCudaSetDevice(oldDevice); CUERR;
    }

    std::future<void> gatherElementsInHostMemAsync(const GatherHandle& handle,
                                                    const Index_t* indices,
                                                    Index_t numIds,
                                                    Value_t* result,
                                                    size_t resultPitch // result element i begins at offset i * resultPitch
                                                    ) const{
        assert(resultPitch >= sizeOfElement);

        const auto instance = this;
        auto future = std::async(std::launch::async, [=](const GatherHandle& handle){
                                    instance->gatherElementsInHostMem(handle, indices, numIds, result, resultPitch);
                                }, std::ref(handle));
        return future;
    }

    void gatherElementsInHostMem(const GatherHandle& handle,
                                const Index_t* indices,
                                Index_t numIds,
                                Value_t* result,
                                size_t resultPitch // result element i begins at offset i * resultPitch
                                ) const{

        //assert(resultPitch >= sizeOfElement);

        int oldDevice; cudaGetDevice(&oldDevice); CUERR;


        //fastpath, if all elements of distributed array reside on a single partition
        if(singlePartitionInfo.isSinglePartition){
            if(singlePartitionInfo.locationId == hostLocation){
                if(debug) std::cerr << "single location array fasthpath on host\n";

                for(Index_t k = 0; k < numIds; k++){
                    const Index_t localId = indices[k];
                    const Value_t* srcPtr = offsetPtr(dataPtrPerLocation[hostLocation], localId);
                    Value_t* destPtr = (Value_t*)(((const char*)(result)) + resultPitch * k);
                    std::copy_n(srcPtr, numColumns, destPtr);
                }

                return;
            }else{
                if(debug) std::cerr << "single location array fasthpath on partition " << singlePartitionInfo.locationId << "\n";
                int oldId = 0;
                cudaGetDevice(&oldId); CUERR;

                const int locationId = singlePartitionInfo.locationId;
                const int deviceId = deviceIds[locationId];
                wrapperCudaSetDevice(deviceId); CUERR;

                const cudaStream_t stream = handle->streamsPerGpu[locationId];
                auto& h_indices = handle->pinnedLocalIndices;
                auto& d_indices = handle->deviceLocalIndicesPerLocation[locationId];
                auto& h_result = handle->pinnedResultData;
                auto& d_result = handle->dataPerGpu[locationId];

                h_indices.resize(numIds);
                d_indices.resize(numIds);
                h_result.resize(numIds * numColumns);
                d_result.resize(numIds * numColumns);

                std::copy_n(indices, numIds, h_indices.get());
                cudaMemcpyAsync(d_indices.get(), h_indices.get(), sizeof(Index_t) * numIds, H2D, stream); CUERR;
                copyDataToGpuBufferAsync(d_result.get(), sizeOfElement, deviceId, d_indices, numIds, deviceId, stream, -elementsPerLocationPS[locationId]); CUERR;
                cudaMemcpyAsync(h_result.get(), d_result.get(), sizeof(Value_t) * numIds * numColumns, D2H, stream); CUERR;
                cudaStreamSynchronize(stream); CUERR;

                wrapperCudaSetDevice(oldId); CUERR;
                
                for(Index_t i = 0; i < numIds; i++){
                    const Value_t* srcPtr = offsetPtr(h_result.get(), i);
                    Value_t* destPtr = (Value_t*)(((const char*)(result)) + resultPitch * i);
                    std::copy_n(srcPtr, numColumns, destPtr);
                }
                

                return;
            }
        }

        std::vector<Index_t> hitsPerLocation(numLocations, 0);
        for(Index_t i = 0; i < numIds; i++){
            const int location = getLocation(indices[i]);
            hitsPerLocation[location]++;
        }

        std::vector<Index_t> hitsPerLocationPrefixSum(numLocations+1,0);
        std::partial_sum(hitsPerLocation.begin(), hitsPerLocation.end(), hitsPerLocationPrefixSum.begin()+1);
        std::fill(hitsPerLocation.begin(), hitsPerLocation.end(), 0);

        std::vector<Index_t> permutationIndices(numIds);

        handle->pinnedLocalIndices.resize(numIds);
        handle->pinnedPermutationIndices.resize(numIds);
        for(Index_t i = 0; i < numIds; i++){
            const int location = getLocation(indices[i]);
            const Index_t localIndex = indices[i] - elementsPerLocationPS[location];
            const Index_t tmpresultindex = hitsPerLocationPrefixSum[location] + hitsPerLocation[location];
            handle->pinnedPermutationIndices[i] = tmpresultindex;

            handle->pinnedLocalIndices[tmpresultindex] = localIndex;
            //std::cerr << "read id " << readIds[i] << ", location " << location << ", localReadId " << localReadId << ", index " << index << "\n";
            hitsPerLocation[location]++;
        }
        handle->pinnedResultData.resize(numIds * numColumns);

        // std::cerr << "hitsPerLocation: ";
        // std::copy(hitsPerLocation.begin(), hitsPerLocation.end(), std::ostream_iterator<Index_t>(std::cerr, " "));
        // std::cerr << "\n";

        int oldId = 0;
        cudaGetDevice(&oldId); CUERR;
        //gather from gpus to host
        for(int gpu = 0; gpu < numGpus; gpu++){
            Index_t numHits = hitsPerLocation[gpu];
            if(numHits > 0){
                const int deviceId = deviceIds[gpu];
                const cudaStream_t stream = handle->streamsPerGpu[gpu];

                wrapperCudaSetDevice(deviceIds[gpu]); CUERR;

                auto& myLocalIds = handle->deviceLocalIndicesPerLocation[gpu];
                auto& myResult = handle->dataPerGpu[gpu];


                myLocalIds.resize(numHits);
                cudaMemcpyAsync(myLocalIds.get(),
                                handle->pinnedLocalIndices.get() + hitsPerLocationPrefixSum[gpu],
                                sizeof(Index_t) * numHits,
                                H2D,
                                stream); CUERR;

                myResult.resize(numHits * numColumns);

                copyDataToGpuBufferAsync(myResult.get(), sizeOfElement, deviceId, myLocalIds.get(), numHits, deviceId, stream, 0);

                cudaMemcpyAsync(handle->pinnedResultData.get() + hitsPerLocationPrefixSum[gpu] * numColumns,
                                myResult.get(),
                                numHits * sizeOfElement,
                                D2H,
                                stream); CUERR;

                cudaEventRecord(handle->eventsPerGpu[gpu], stream); CUERR;
            }
        }

        //gather from host to host
        if(hitsPerLocation[hostLocation] > 0){
            const Index_t numHits = hitsPerLocation[hostLocation];
            const Index_t* hostLocalIds = handle->pinnedLocalIndices.get() + hitsPerLocationPrefixSum[hostLocation];
            Value_t* hostResult = offsetPtr(handle->pinnedResultData.get(), hitsPerLocationPrefixSum[hostLocation]);

            for(Index_t k = 0; k < numHits; k++){
                const Index_t localId = hostLocalIds[k];
                const Value_t* srcPtr = offsetPtr(dataPtrPerLocation[hostLocation], localId);
                Value_t* destPtr = offsetPtr(hostResult, k);
                std::copy_n(srcPtr, numColumns, destPtr);
            }
        }

        for(auto& event : handle->eventsPerGpu){
            cudaEventSynchronize(event); CUERR;
        }

        wrapperCudaSetDevice(oldId); CUERR;

        //permute pinnedResultData and store in output array
        for(Index_t dstindex = 0; dstindex < numIds; dstindex++){
            const Index_t srcindex = handle->pinnedPermutationIndices[dstindex];
            const Value_t* srcPtr = offsetPtr(handle->pinnedResultData.get(), srcindex);
            Value_t* destPtr = (Value_t*)(((const char*)(result)) + resultPitch * dstindex);
            std::copy_n(srcPtr, numColumns, destPtr);
        }
    }

    template<class ParallelFor>
    void gatherElementsInGpuMem(ParallelFor&& forLoop,
                                const GatherHandle& handle,
                                const Index_t* indices,
                                const Index_t* d_indices,
                                Index_t numIds,
                                int deviceId,
                                Value_t* d_result,
                                size_t resultPitch // result element i begins at byte offset i * resultPitch
                                ) const {
        assert(resultPitch >= sizeOfElement);

        cudaStream_t stream = 0;
        int oldDevice = 0; cudaGetDevice(&oldDevice); CUERR;
        wrapperCudaSetDevice(deviceId); CUERR;
        cudaStreamCreate(&stream); CUERR;
        gatherElementsInGpuMemAsync(std::forward<ParallelFor>(forLoop), handle, indices, d_indices, numIds, deviceId, d_result, resultPitch, stream);
        cudaStreamSynchronize(stream); CUERR;
        cudaStreamDestroy(stream); CUERR;
        wrapperCudaSetDevice(oldDevice); CUERR;
    }

    //the same GatherHandleStruct must not be used in another call until the results of the previous call are calculated
    template<class ParallelFor>
    void gatherElementsInGpuMemAsync(ParallelFor&& forLoop,
                                    const GatherHandle& handle,
                                    const Index_t* indices,
                                    const Index_t* d_indices,
                                    Index_t numIds,
                                    int resultDeviceId,
                                    Value_t* d_result,
                                    size_t resultPitch, // result element i begins at offset i * resultPitch
                                    cudaStream_t syncstream) const{

        if(singlePartitionInfo.isSinglePartition){
            nvtx::push_range("singlePartitionGather", 0);
            gatherElementsInGpuMemAsyncSinglePartitionMode(
                forLoop,
                handle,
                indices,
                d_indices,
                numIds,
                resultDeviceId,
                d_result,
                resultPitch,
                syncstream
            );
            nvtx::pop_range();

        }else{

            if(elementsPerLocation[hostLocation] == 0){
                nvtx::push_range("nohostGather", 1);

                gatherElementsInGpuMemAsyncNoHostPartition(
                    forLoop,
                    handle,
                    indices,
                    d_indices,
                    numIds,
                    resultDeviceId,
                    d_result,
                    resultPitch,
                    syncstream
                );

                nvtx::pop_range();
            }else{        

                nvtx::push_range("generalGather", 2);

                //gatherElementsInGpuMemAsyncDefault(
                gatherElementsInGpuMemAsyncGeneral(
                //gatherElementsInGpuMemAsyncNewBroken(
                    forLoop,
                    handle,
                    indices,
                    d_indices,
                    numIds,
                    resultDeviceId,
                    d_result,
                    resultPitch,
                    syncstream
                );

                nvtx::pop_range();

            }

        }
    }

    template<class ParallelFor>
    void gatherElementsInGpuMemAsyncSinglePartitionMode(
            ParallelFor&& forLoop,
            const GatherHandle& handle,
            const Index_t* indices,
            const Index_t* d_indices,
            Index_t numIds,
            int resultDeviceId,
            Value_t* d_result,
            size_t resultPitch, // result element i begins at offset i * resultPitch
            cudaStream_t syncstream) const{

        assert(singlePartitionInfo.isSinglePartition);

        if(singlePartitionInfo.locationId == hostLocation){
            // if(debug) std::cerr << "single location array fasthpath on host\n";

            handle->pinnedGatheredElementsOfLocation[hostLocation].resize(numIds * SDIV(resultPitch, sizeof(Value_t)));

            Value_t* const myResult = handle->pinnedGatheredElementsOfLocation[hostLocation].get();

            auto gather = [&](Index_t begin, Index_t end, int /*threadId*/){
                for(Index_t k = begin; k < end; k++){
                    const Index_t localId = indices[k];

                    const Value_t* const srcPtr = offsetPtr(dataPtrPerLocation[hostLocation], localId);
                    Value_t* const destPtr = (Value_t*)(((const char*)(myResult)) + resultPitch * k);

                    std::copy_n(srcPtr, numColumns, destPtr);
                }
            };

            forLoop( 
                Index_t(0), 
                numIds, 
                gather
            );

            int oldDevice; cudaGetDevice(&oldDevice); CUERR;

            wrapperCudaSetDevice(resultDeviceId); CUERR;

            cudaMemcpyAsync(d_result, myResult, numIds * resultPitch, H2D, syncstream); CUERR;

            wrapperCudaSetDevice(oldDevice); CUERR;
        }else{
            //if(debug) std::cerr << "single location array fast path on partition " << singlePartitionInfo.locationId << "\n";

            const int locationId = singlePartitionInfo.locationId;
            const int gpu = locationId;
            const int mydeviceId = deviceIds[gpu];
            cudaStream_t mystream = handle->streamsPerGpu[gpu];
            cudaEvent_t myevent = handle->eventsPerGpu[gpu];

            int oldDevice; cudaGetDevice(&oldDevice); CUERR;
            
            
            if(mydeviceId == resultDeviceId){
                wrapperCudaSetDevice(resultDeviceId); CUERR;

                copyDataToGpuBufferAsync(
                    d_result, 
                    resultPitch, 
                    resultDeviceId, 
                    d_indices, 
                    numIds, 
                    resultDeviceId, 
                    syncstream, 
                    -elementsPerLocationPS[gpu]
                );
            }else{
                registerDeviceIdForHandlenew(handle, resultDeviceId);

                wrapperCudaSetDevice(resultDeviceId);
                cudaEventRecord(handle->map_events[resultDeviceId], syncstream); CUERR;

                wrapperCudaSetDevice(mydeviceId); CUERR;

                auto& myGatherResult = handle->d_gatheredElementsOfGpuLocation[gpu];
                myGatherResult.resize(numIds * numColumns);

                cudaStreamWaitEvent(mystream, handle->map_events[resultDeviceId], 0); CUERR;

                copyDataToGpuBufferAsync(
                    myGatherResult.get(), 
                    sizeof(Value_t) * numColumns, 
                    mydeviceId, 
                    d_indices, 
                    numIds, 
                    mydeviceId, 
                    mystream, 
                    -elementsPerLocationPS[gpu]
                );

                //copy to result array (peer access)
                {
                    assert(resultPitch % sizeof(Value_t) == 0);

                    size_t resultPitchValueTs = resultPitch / sizeof(Value_t);
                    size_t numCols = numColumns;

                    const Value_t* const input = myGatherResult.get();
                    Value_t* const output = d_result;

                    dim3 block(256,1,1);
                    dim3 grid(std::min(320ul, SDIV(numIds * numCols, block.x)),1,1);

                    generic_kernel<<<grid, block, 0, mystream>>>([=] __device__ (){

                        for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; 
                                i < numIds * numCols; 
                                i += size_t(blockDim.x) * gridDim.x){

                            const Index_t inputRow = i / numCols;
                            const Index_t col = i % numCols;
                            const Index_t outputRow = inputRow;
                            
                            output[size_t(outputRow) * resultPitchValueTs + col] 
                                = input[size_t(inputRow) * numCols + col];
                        }
                    }); CUERR;
                }

                cudaEventRecord(myevent, mystream); CUERR;
                cudaStreamWaitEvent(syncstream, myevent, 0); CUERR;
            }

            wrapperCudaSetDevice(oldDevice); CUERR;            
        }
    }


    template<class ParallelFor>
    void gatherElementsInGpuMemAsyncNoHostPartition(
            ParallelFor&& forLoop,
            const GatherHandle& handle,
            const Index_t* indices,
            const Index_t* d_indices,
            Index_t numIds,
            int resultDeviceId,
            Value_t* d_result,
            size_t resultPitch, // result element i begins at offset i * resultPitch
            cudaStream_t syncstream) const{

        if(numIds == 0){
            return;
        }

        assert(elementsPerLocation[hostLocation] == 0);
        assert(!singlePartitionInfo.isSinglePartition); //there is a dedicated function for this case


        registerDeviceIdForHandlenew(handle, resultDeviceId);

        
        auto& d_destination_elementsPerLocationPS = handle->map_d_elementsPerLocationPS[resultDeviceId];
        auto& d_destination_numIndicesPerLocation = handle->map_d_numIndicesPerLocation[resultDeviceId];
        auto& d_destination_numIndicesPerLocationPS = handle->map_d_numIndicesPerLocationPS[resultDeviceId];

        auto& d_indicesForLocationsVector = handle->map_d_indicesForLocationsVector[resultDeviceId];
        auto& d_destinationPositionsForLocationsVector = handle->map_d_destinationPositionsForLocationsVector[resultDeviceId];

        // auto& d_destination_gatheredElementsForLocation = handle->map_d_gatheredElementsForLocation[resultDeviceId];
        // auto& d_destination_posIndexPairsForLocation = handle->map_d_positionIndexPairsForLocation[resultDeviceId];
        // auto& d_destination_cubTemp = handle->map_d_cubTemp[resultDeviceId];

        auto& destination_event = handle->map_events[resultDeviceId];
        auto& destination_stream = handle->map_streams[resultDeviceId];

        wrapperCudaSetDevice(resultDeviceId);
        cudaEventRecord(destination_event, syncstream); CUERR;
        //cudaStreamWaitEvent(destination_stream, destination_event, 0); CUERR;

        for(int i = 0; i < numLocations; i++){
            d_indicesForLocationsVector[i].resize(numIds);
            d_destinationPositionsForLocationsVector[i].resize(numIds);

            // //debugging
            // cudaMemsetAsync(
            //     d_indicesForLocationsVector[i].get(), 
            //     0, 
            //     d_indicesForLocationsVector[i].sizeInBytes(), 
            //     syncstream
            // ); CUERR;

            // //debugging
            // cudaMemsetAsync(
            //     d_destinationPositionsForLocationsVector[i].get(), 
            //     0, 
            //     d_destinationPositionsForLocationsVector[i].sizeInBytes(), 
            //     syncstream
            // ); CUERR;
        }

        for(int i = 0; i < numLocations; i++){
            handle->pinnedPointersForLocations[i] = d_indicesForLocationsVector[i].get();
        }

        cudaMemcpyAsync(
            handle->map_d_indicesForLocationsPointers[resultDeviceId].get(),
            handle->pinnedPointersForLocations.get(),
            sizeof(Index_t*) * numLocations,
            H2D,
            syncstream
        ); CUERR;

        for(int i = 0; i < numLocations; i++){
            handle->pinnedPointersForLocations2[i] = d_destinationPositionsForLocationsVector[i].get();
        }

        cudaMemcpyAsync(
            handle->map_d_destinationPositionsForLocationsPointers[resultDeviceId].get(),
            handle->pinnedPointersForLocations2.get(),
            sizeof(Index_t*) * numLocations,
            H2D,
            syncstream
        ); CUERR;


        // //debugging
        // cudaMemsetAsync(
        //     d_result, 
        //     0, 
        //     resultPitch * numIds, 
        //     syncstream
        // ); CUERR;

        //find indices per location + prefixsum
        {
            int numLocs = numLocations;
            auto* d_elementsPerLocationPSPtr = d_destination_elementsPerLocationPS.get();
            auto* d_indicesPerLocationPtr = d_destination_numIndicesPerLocation.get();
            auto* d_indicesPerLocationPSPtr = d_destination_numIndicesPerLocationPS.get();

            auto* splitIndices = handle->map_d_indicesForLocationsPointers[resultDeviceId].get();
            auto* splitDestinationPositions = handle->map_d_destinationPositionsForLocationsPointers[resultDeviceId].get();

            // cudaDeviceSynchronize(); CUERR;

            // wrapperCudaSetDevice(resultDeviceId);
            // std::cerr << "cudamemsetasync resultDeviceId = " << resultDeviceId 
            //             << "(" << (void*)d_indicesPerLocationPtr << "\n"
            //             << "0\n"
            //             << sizeof(Index_t) * numIds << "\n"
            //             << (void*) syncstream << "\n";
            cudaMemsetAsync(
                d_indicesPerLocationPtr, 
                0, 
                sizeof(Index_t) * numLocations, 
                syncstream
            ); CUERR;

            assert(maxNumGpus <= 32);
            generic_kernel<<<SDIV(numIds, 256), 256, 0, syncstream>>>([=] __device__ (){
                constexpr int maxNumLocations = 33;

                auto atomicAggInc = [](Index_t* counter){
                    auto g = cg::coalesced_threads();
                    int warp_res;
                    if(g.thread_rank() == 0){
                        warp_res = atomicAdd(counter, Index_t(g.size()));
                    }
                    return g.shfl(warp_res, 0) + g.thread_rank();
                };

                __shared__ Index_t shared_elementsPerLocationPS[maxNumLocations + 1];

                for(int tid = threadIdx.x; tid < numLocs+1; tid += blockDim.x){
                    shared_elementsPerLocationPS[tid] = d_elementsPerLocationPSPtr[tid];
                }

                // for(int tid = threadIdx.x + numLocs+1; tid < maxNumLocations+1; tid += blockDim.x){
                //     shared_elementsPerLocationPS[tid] = d_elementsPerLocationPSPtr[numLocs];
                // }

                __syncthreads();
                
                for(Index_t tid = threadIdx.x + blockIdx.x * Index_t(blockDim.x); 
                        tid < numIds; 
                        tid += Index_t(blockDim.x) * gridDim.x){

                    const Index_t element = d_indices[tid];
                    int location = -1;

                    for(int i = 0; i < numLocs; i++){
                        if(shared_elementsPerLocationPS[i] <= element && element < shared_elementsPerLocationPS[i+1]){
                            location = i;
                        }
                    }

                    #pragma unroll
                    for(int i = 0; i < maxNumLocations; ++i){
                        if(i == location){
                            const Index_t j = atomicAggInc(&d_indicesPerLocationPtr[i]);
                            splitIndices[i][j] = element;
                            splitDestinationPositions[i][j] = tid;
                        }
                    }
                }                
            }); CUERR;

            generic_kernel<<<1,1, 0, syncstream>>>([=]__device__(){
                d_indicesPerLocationPSPtr[0] = 0;
                
                for(int i = 0; i < numLocs; i++){
                    d_indicesPerLocationPSPtr[i+1] = d_indicesPerLocationPSPtr[i] + d_indicesPerLocationPtr[i];
                }

                // printf("inputindices\n");
                // for(int i = 0; i < numIds; i++){
                //     printf("%d, ", d_indices[i]);
                // }
                // printf("\n");

                // for(int i = 0; i < numLocs; i++){
                //     printf("indices for location %d:\n", i);

                //     for(int k = 0; k < d_indicesPerLocationPtr[i]; k++){
                //         printf("%d, ", splitIndices[i][k]);
                //     }
                //     printf("\n");

                //     printf("destpositions for location %d:\n", i);

                //     for(int k = 0; k < d_indicesPerLocationPtr[i]; k++){
                //         printf("%d, ", splitDestinationPositions[i][k]);
                //     }
                //     printf("\n");
                // }
            }); CUERR;

            cudaEventRecord(destination_event, syncstream); CUERR;
        }

        for(int gpu = 0; gpu < numGpus; gpu++){
            const int location = gpu;
            if(elementsPerLocation[location] > 0){
                const int gpuDeviceId = deviceIds[gpu];
                cudaStream_t gpuStream = handle->streamsPerGpuLocation[location];
                cudaEvent_t gpuEvent = handle->eventsPerGpuLocation[location];

                wrapperCudaSetDevice(gpuDeviceId);
                cudaStreamWaitEvent(gpuStream, destination_event, 0); CUERR;

                auto& myGatherResult = handle->d_gatheredElementsOfGpuLocation[gpu];
                myGatherResult.resize(numIds * numColumns);

                //gather elements of selected indices
                {
                    const size_t numCols = numColumns;
                    const size_t resultPitchValueTs = numColumns;

                    const Value_t* const sourceArrayDataPtr = dataPtrPerLocation[gpu];
                    const Index_t* const myIndicesPtr = d_indicesForLocationsVector[location].get(); //peer access
                    const Index_t* const myNumIndicesPtr = d_destination_numIndicesPerLocation.get() + location;
                    auto indexOffset = elementsPerLocationPS[location];
                    Value_t* outputPtr = myGatherResult.get();

                    generic_kernel<<<SDIV(numIds, 256), 256, 0, gpuStream>>>(
                        [=] __device__ (){

                            const Index_t nIndices = *myNumIndicesPtr;

                            for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; 
                                    i < nIndices * numCols; 
                                    i += size_t(blockDim.x) * gridDim.x){

                                const Index_t outputrow = i / numCols;
                                const Index_t col = i % numCols;
                                const Index_t inputrow = myIndicesPtr[outputrow] - indexOffset;

                                outputPtr[size_t(outputrow) * resultPitchValueTs + col] 
                                        = sourceArrayDataPtr[size_t(inputrow) * numCols + col];
                            }
                        }
                    ); CUERR;

                    cudaEventRecord(gpuEvent, gpuStream); CUERR;
                }

                wrapperCudaSetDevice(resultDeviceId); CUERR;

                cudaStreamWaitEvent(syncstream, gpuEvent, 0); CUERR;

                //scatter to result array (may be peer access)
                {
                    assert(resultPitch % sizeof(Value_t) == 0);

                    size_t resultPitchValueTs = resultPitch / sizeof(Value_t);
                    size_t numCols = numColumns;

                    const Value_t* const input = myGatherResult.get();
                    const Index_t* const permutIndices = d_destinationPositionsForLocationsVector[gpu].get();
                    const Index_t* const myNumIndicesPtr = d_destination_numIndicesPerLocation.get() + location;
                    Value_t* const output = d_result;

                    dim3 block(256,1,1);
                    dim3 grid(std::min(320ul, SDIV(numIds * numCols, block.x)),1,1);

                    generic_kernel<<<grid, block, 0, syncstream>>>([=] __device__ (){

                        const Index_t nIndices = *myNumIndicesPtr;

                        for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; 
                                i < nIndices * numCols; 
                                i += size_t(blockDim.x) * gridDim.x){

                            const Index_t inputRow = i / numCols;
                            const Index_t col = i % numCols;
                            const Index_t outputRow = permutIndices[inputRow];
                            
                            output[size_t(outputRow) * resultPitchValueTs + col] 
                                = input[size_t(inputRow) * numCols + col];
                        }
                    }); CUERR;
                }
            }
        }


    }                                    

    //the same GatherHandleStruct must not be used in another call until the results of the previous call are calculated
    template<class ParallelFor>
    void gatherElementsInGpuMemAsyncGeneral(ParallelFor&& forLoop,
                                    const GatherHandle& handle,
                                    const Index_t* indices,
                                    const Index_t* d_indices,
                                    Index_t numIds,
                                    int resultDeviceId,
                                    Value_t* d_result,
                                    size_t resultPitch, // result element i begins at offset i * resultPitch
                                    cudaStream_t stream) const{
        if(numIds == 0) return;

        assert(resultPitch % sizeof(Value_t) == 0);

        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        //const int numCpuThreads = care::threadpool.getConcurrency();

        cudaStream_t syncStream = stream;

        registerDeviceIdForHandlenew(handle, resultDeviceId);

        wrapperCudaSetDevice(resultDeviceId);
        cudaEventRecord(handle->map_events[resultDeviceId], syncStream); CUERR;

        auto& tmpResultsOnResultDevice = handle->map_d_tmpResults[resultDeviceId];
        auto& d_destinationPositionsOnResultDevice = handle->map_d_destinationPositionsOfGpu[resultDeviceId];
        d_destinationPositionsOnResultDevice.resize(numIds);
        tmpResultsOnResultDevice.resize(numIds * numColumns);

        for(int gpu = 0; gpu < numGpus; gpu++){
            const int loc = gpu;
            if(elementsPerLocation[loc] > 0){
                handle->pinnedIndicesOfGpuLocation[gpu].resize(numIds);
                handle->pinnedDestinationPositionsOfLocation[gpu].resize(numIds);
            }
        }

        handle->indicesOfHostLocation.resize(numIds);
        handle->pinnedDestinationPositionsOfLocation[hostLocation].resize(numIds);

        std::fill(handle->numIndicesPerLocation.begin(), handle->numIndicesPerLocation.end(), 0);

        for(Index_t i = 0; i < numIds; i++){
            const Index_t index = indices[i];
            const int loc = getLocation(index);

            auto& num = handle->numIndicesPerLocation[loc];
            handle->pinnedDestinationPositionsOfLocation[loc][num] = i;

            if(loc != hostLocation){
                const int gpu = loc;
                handle->pinnedIndicesOfGpuLocation[gpu][num] = index;
            }else{
                handle->indicesOfHostLocation[num] = index;
            }

            num++;
        }

        handle->numIndicesPerLocationPS[0] = 0;
        std::partial_sum(
            handle->numIndicesPerLocation.begin(), 
            handle->numIndicesPerLocation.end(), 
            handle->numIndicesPerLocationPS.begin()+1
        );

        for(int gpu = 0; gpu < numGpus; gpu++){
            const int loc = gpu;
            const int numForLoc = handle->numIndicesPerLocation[loc];
            if(numForLoc > 0){
                const int myDeviceId = deviceIds[gpu];
                wrapperCudaSetDevice(myDeviceId);

                handle->d_indicesOfGpuLocation[gpu].resize(numForLoc);

                cudaMemcpyAsync(
                    handle->d_indicesOfGpuLocation[gpu].get(),
                    handle->pinnedIndicesOfGpuLocation[gpu].get(),
                    sizeof(Index_t) * numForLoc,
                    H2D,
                    handle->streamsPerGpu[gpu]
                ); CUERR;           

                handle->d_destinationPositionsOfGpuLocation[gpu].resize(numForLoc);

                cudaMemcpyAsync(
                    handle->d_destinationPositionsOfGpuLocation[gpu].get(),
                    handle->pinnedDestinationPositionsOfLocation[loc].get(),
                    sizeof(Index_t) * numForLoc,
                    H2D,
                    handle->streamsPerGpu[gpu]
                ); CUERR;  
            }
        }
       

        //cudaDeviceSynchronize(); CUERR;

        for(int gpu = 0; gpu < numGpus; gpu++){
            const int loc = gpu;
            const int numForLoc = handle->numIndicesPerLocation[loc];
            if(numForLoc > 0){
                Index_t* const myDestinationPositionsOnResultDevice
                    = d_destinationPositionsOnResultDevice + handle->numIndicesPerLocationPS[loc];

                Value_t* const myTmpResultsOnResultDevice 
                    = tmpResultsOnResultDevice + handle->numIndicesPerLocationPS[loc];

                //cudaDeviceSynchronize(); CUERR;

                const int myDeviceId = deviceIds[gpu];
                wrapperCudaSetDevice(myDeviceId);

                auto& myGatherResult = handle->d_gatheredElementsOfGpuLocation[gpu];
                myGatherResult.resize(numForLoc * numColumns);

                //gather elements of selected indices
                {
                    const size_t numCols = numColumns;
                    const size_t resultPitchValueTs = numColumns;

                    const Value_t* const sourceArrayDataPtr = dataPtrPerLocation[gpu];
                    const Index_t* const myIndicesPtr = handle->d_indicesOfGpuLocation[gpu].get();
                    auto indexOffset = elementsPerLocationPS[loc];
                    Value_t* outputPtr = myGatherResult.get();

                    generic_kernel<<<SDIV(numIds, 256), 256, 0, handle->streamsPerGpu[gpu]>>>(
                        [=] __device__ (){

                            const Index_t nIndices = numForLoc;

                            for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; 
                                    i < nIndices * numCols; 
                                    i += size_t(blockDim.x) * gridDim.x){

                                const Index_t outputrow = i / numCols;
                                const Index_t col = i % numCols;
                                const Index_t inputrow = myIndicesPtr[outputrow] - indexOffset;

                                outputPtr[size_t(outputrow) * resultPitchValueTs + col] 
                                        = sourceArrayDataPtr[size_t(inputrow) * numCols + col];
                            }
                        }
                    ); CUERR;
                }

                //scatter to result array (may be peer access)
                {
                    assert(resultPitch % sizeof(Value_t) == 0);

                    size_t resultPitchValueTs = resultPitch / sizeof(Value_t);
                    size_t numCols = numColumns;

                    const Value_t* const input = myGatherResult.get();
                    const Index_t* const permutIndices = handle->d_destinationPositionsOfGpuLocation[gpu].get();
                    Value_t* const output = d_result;

                    dim3 block(256,1,1);
                    dim3 grid(std::min(320ul, SDIV(numForLoc * numCols, block.x)),1,1);

                    generic_kernel<<<grid, block, 0, handle->streamsPerGpu[gpu]>>>([=] __device__ (){

                        for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; 
                                i < numForLoc * numCols; 
                                i += size_t(blockDim.x) * gridDim.x){

                            const Index_t inputRow = i / numCols;
                            const Index_t col = i % numCols;
                            const Index_t outputRow = permutIndices[inputRow];
                            
                            output[size_t(outputRow) * resultPitchValueTs + col] 
                                = input[size_t(inputRow) * numCols + col];
                        }
                    }); CUERR;

                    cudaEventRecord(handle->eventsPerGpu[gpu], handle->streamsPerGpu[gpu]); CUERR;
                    cudaStreamWaitEvent(syncStream, handle->eventsPerGpu[gpu], 0); CUERR;
                }

                //cudaDeviceSynchronize(); CUERR;
            }
        }

        //cudaDeviceSynchronize(); CUERR;

        //handle host gathering
        if(handle->numIndicesPerLocation[hostLocation] > 0){
            const int numHostIndices = handle->numIndicesPerLocation[hostLocation];

            handle->pinnedGatheredElementsOfLocation[hostLocation].resize(numHostIndices * numColumns);

            Index_t* const myDestinationPositionsOnResultDevice
                    = d_destinationPositionsOnResultDevice + handle->numIndicesPerLocationPS[hostLocation];

            Value_t* const myTmpResultsOnResultDevice 
                = tmpResultsOnResultDevice + handle->numIndicesPerLocationPS[hostLocation];

            wrapperCudaSetDevice(resultDeviceId);
            cudaMemcpyAsync(
                myDestinationPositionsOnResultDevice, 
                handle->pinnedDestinationPositionsOfLocation[hostLocation].get(),
                sizeof(Index_t) * numHostIndices,
                H2D,
                syncStream
            ); CUERR;

            Value_t* const myResult = handle->pinnedGatheredElementsOfLocation[hostLocation].get();   

            forLoop(
                0, 
                numHostIndices, 
                [&](int begin, int end, int threadId){
                    for(int k = begin; k < end; k++){
                        const Index_t localId = handle->indicesOfHostLocation[k] - elementsPerLocationPS[hostLocation];
                        const Value_t* srcPtr = offsetPtr(dataPtrPerLocation[hostLocation], localId);
                        Value_t* destPtr = myResult + size_t(k) * numColumns;
                        std::copy_n(srcPtr, numColumns, destPtr);
                    }
                }
            );

            cudaMemcpyAsync(
                myTmpResultsOnResultDevice,
                myResult,
                sizeof(Value_t) * numColumns * numHostIndices,
                H2D,
                syncStream
            ); CUERR;

            //on resultgpu scatter the copied data to the correct positions in final output array
            {
                assert(resultPitch % sizeof(Value_t) == 0);

                size_t resultPitchValueTs = resultPitch / sizeof(Value_t);
                size_t numCols = numColumns;

                const Value_t* const input = myTmpResultsOnResultDevice;
                const Index_t* const permutIndices = myDestinationPositionsOnResultDevice;
                Value_t* const output = d_result;

                dim3 block(256,1,1);
                dim3 grid(std::min(320ul, SDIV(numHostIndices * numCols, block.x)),1,1);

                generic_kernel<<<grid, block, 0, syncStream>>>([=] __device__ (){

                    for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; 
                            i < numHostIndices * numCols; 
                            i += size_t(blockDim.x) * gridDim.x){

                        const Index_t inputRow = i / numCols;
                        const Index_t col = i % numCols;
                        const Index_t outputRow = permutIndices[inputRow];
                        
                        output[size_t(outputRow) * resultPitchValueTs + col] 
                            = input[size_t(inputRow) * numCols + col];
                    }
                }); CUERR;
            }
        }

        wrapperCudaSetDevice(oldDevice); CUERR;
    }



    template<class ParallelFor>
    void gatherElementsInGpuMemAsyncNewBroken(ParallelFor&& forLoop,
                                    const GatherHandle& handle,
                                    const Index_t* indices,
                                    const Index_t* d_indices,
                                    Index_t numIds,
                                    int resultDeviceId,
                                    Value_t* d_result,
                                    size_t resultPitch, // result element i begins at offset i * resultPitch
                                    cudaStream_t stream) const{
        if(numIds == 0) return;

        assert(resultPitch % sizeof(Value_t) == 0);

        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        //const int numCpuThreads = care::threadpool.getConcurrency();

        //fastpath, if all elements of distributed array reside in a single partition
        if(singlePartitionInfo.isSinglePartition){
            if(singlePartitionInfo.locationId == hostLocation){
                if(debug) std::cerr << "single location array fasthpath on host\n";

                auto& h_result = handle->pinnedResultData;
                h_result.resize(numIds * SDIV(resultPitch, sizeof(Value_t)));

                auto gather = [&](Index_t begin, Index_t end, int /*threadId*/){
                    for(Index_t k = begin; k < end; k++){
                        const Index_t localId = indices[k];
                        const Value_t* srcPtr = offsetPtr(dataPtrPerLocation[hostLocation], localId);
                        //Value_t* destPtr = offsetPtr(h_result.get(), k);
                        Value_t* destPtr = (Value_t*)(((const char*)(h_result.get())) + resultPitch * k);
                        
                        std::copy_n(srcPtr, numColumns, destPtr);
                    }
                };

                forLoop( 
                    Index_t(0), 
                    numIds, 
                    gather
                );

                int oldDevice; cudaGetDevice(&oldDevice); CUERR;

                wrapperCudaSetDevice(resultDeviceId); CUERR;

                cudaMemcpyAsync(d_result, h_result.get(), numIds * resultPitch, H2D, stream); CUERR;

                wrapperCudaSetDevice(oldDevice); CUERR;

                return;
            }else{
                //if(debug) std::cerr << "single location array fast path on partition " << singlePartitionInfo.locationId << "\n";

                const int locationId = singlePartitionInfo.locationId;
                const int mydeviceId = deviceIds[locationId];
                cudaStream_t mystream = handle->streamsPerGpu[locationId];
                cudaEvent_t myevent = handle->eventsPerGpu[locationId];
            #if 1
                 if(mydeviceId == resultDeviceId){
                     copyDataToGpuBufferAsync(d_result, resultPitch, mydeviceId, d_indices, numIds, mydeviceId, stream, -elementsPerLocationPS[locationId]);

                     return;
                 }
            #else

                wrapperCudaSetDevice(mydeviceId); CUERR;

                Value_t* destptr = d_result;

                Value_t* localGatherPtr = nullptr;
                Index_t* localIndicesPtr = nullptr;
                cudaStream_t localGatherStream = cudaStream_t(0);

                if(resultDeviceId == mydeviceId){
                    localGatherPtr = destptr;
                    localIndicesPtr = const_cast<Index_t*>(d_indices);
                    localGatherStream = stream;
                }else{
                    handle->dataPerGpu[locationId].resize(numIds * numColumns);
                    localGatherPtr = handle->dataPerGpu[locationId].get();

                    handle->deviceLocalIndicesPerLocation[locationId].resize(numIds);
                    localIndicesPtr = handle->deviceLocalIndicesPerLocation[locationId].get();
                    cudaMemcpyPeerAsync(localIndicesPtr,
                                        mydeviceId,
                                        d_indices,
                                        resultDeviceId,
                                        sizeof(Index_t) * numIds,
                                        localGatherStream); CUERR;

                    localGatherStream = mystream;
                }

                if(debug) cudaDeviceSynchronize(); CUERR;

                //local gather on device mydeviceId
                copyDataToGpuBufferAsync(localGatherPtr, resultPitch, mydeviceId, localIndicesPtr, numIds, mydeviceId, localGatherStream, -elementsPerLocationPS[locationId]);

                if(debug) cudaDeviceSynchronize(); CUERR;           

                //send partial results to destination gpu
                
                if(resultDeviceId != mydeviceId){

                    cudaEventRecord(myevent, localGatherStream); CUERR;
                    wrapperCudaSetDevice(resultDeviceId); CUERR;
                    cudaStreamWaitEvent(stream, myevent,0); CUERR; //wait in result stream until partial results are on the host.
                    
                    cudaMemcpyPeerAsync(destptr,
                                        resultDeviceId,
                                        localGatherPtr,
                                        mydeviceId,
                                        sizeOfElement * numIds,
                                        stream); CUERR;

                    if(debug) cudaDeviceSynchronize(); CUERR;
                } 

                return;
            #endif
               
            }
        }

        //cudaDeviceSynchronize(); CUERR;

        cudaStream_t syncStream = stream;

        registerDeviceIdForHandlenew(handle, resultDeviceId);

        wrapperCudaSetDevice(resultDeviceId);
        cudaEventRecord(handle->map_events[resultDeviceId], syncStream); CUERR;

        auto& tmpResultsOnResultDevice = handle->map_d_tmpResults[resultDeviceId];
        auto& d_destinationPositionsOnResultDevice = handle->map_d_destinationPositionsOfGpu[resultDeviceId];
        d_destinationPositionsOnResultDevice.resize(numIds);
        tmpResultsOnResultDevice.resize(numIds * numColumns);

        for(int gpu = 0; gpu < numGpus; gpu++){
            const int loc = gpu;
            if(elementsPerLocation[loc] > 0){
                handle->pinnedIndicesOfGpuLocation[gpu].resize(numIds);
                handle->pinnedDestinationPositionsOfLocation[gpu].resize(numIds);
            }
        }

        std::fill(handle->numIndicesPerLocation.begin(), handle->numIndicesPerLocation.end(), 0);

        for(Index_t i = 0; i < numIds; i++){
            const Index_t index = indices[i];
            const int loc = getLocation(index);

            auto& num = handle->numIndicesPerLocation[loc];
            handle->pinnedDestinationPositionsOfLocation[loc][num] = i;

            if(loc != hostLocation){
                const int gpu = loc;
                handle->pinnedIndicesOfGpuLocation[gpu][num] = index;
            }else{
                handle->indicesOfHostLocation[num] = index;
            }

            num++;
        }

        handle->numIndicesPerLocationPS[0] = 0;
        std::partial_sum(
            handle->numIndicesPerLocation.begin(), 
            handle->numIndicesPerLocation.end(), 
            handle->numIndicesPerLocationPS.begin()+1
        );      

        //cudaDeviceSynchronize(); CUERR;

        for(int gpu = 0; gpu < numGpus; gpu++){
            const int loc = gpu;
            const int numForLoc = handle->numIndicesPerLocation[loc];
            if(numForLoc > 0){
                Index_t* const myDestinationPositionsOnResultDevice
                    = d_destinationPositionsOnResultDevice + handle->numIndicesPerLocationPS[loc];

                Value_t* const myTmpResultsOnResultDevice 
                    = tmpResultsOnResultDevice + handle->numIndicesPerLocationPS[loc];

                wrapperCudaSetDevice(resultDeviceId);
                cudaMemcpyAsync(
                    myDestinationPositionsOnResultDevice, 
                    handle->pinnedDestinationPositionsOfLocation[loc].get(),
                    sizeof(Index_t) * numForLoc,
                    H2D,
                    syncStream
                ); CUERR;

                //cudaDeviceSynchronize(); CUERR;

                const int myDeviceId = deviceIds[gpu];
                wrapperCudaSetDevice(myDeviceId);

                //cudaStreamWaitEvent(handle->streamsPerGpu[gpu], map_events[resultDeviceId], 0); CUERR;

                handle->d_indicesOfGpuLocation[gpu].resize(numForLoc);

                cudaMemcpyAsync(
                    handle->d_indicesOfGpuLocation[gpu].get(),
                    handle->pinnedIndicesOfGpuLocation[gpu].get(),
                    sizeof(Index_t) * numForLoc,
                    H2D,
                    handle->streamsPerGpu[gpu]
                ); CUERR;                


                //cudaDeviceSynchronize(); CUERR;

                auto& myGatherResult = handle->d_gatheredElementsOfGpuLocation[gpu];
                myGatherResult.resize(numForLoc * numColumns);

                //gather elements of selected indices
                {
                    const size_t numCols = numColumns;
                    const size_t resultPitchValueTs = numColumns;

                    const Value_t* const sourceArrayDataPtr = dataPtrPerLocation[gpu];
                    const Index_t* const myIndicesPtr = handle->d_indicesOfGpuLocation[gpu].get();
                    auto indexOffset = elementsPerLocationPS[loc];
                    Value_t* outputPtr = myGatherResult.get();

                    generic_kernel<<<SDIV(numIds, 256), 256, 0, handle->streamsPerGpu[gpu]>>>(
                        [=] __device__ (){

                            const Index_t nIndices = numForLoc;

                            for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; 
                                    i < nIndices * numCols; 
                                    i += size_t(blockDim.x) * gridDim.x){

                                const Index_t outputrow = i / numCols;
                                const Index_t col = i % numCols;
                                const Index_t inputrow = myIndicesPtr[outputrow] - indexOffset;

                                outputPtr[size_t(outputrow) * resultPitchValueTs + col] 
                                        = sourceArrayDataPtr[size_t(inputrow) * numCols + col];
                            }
                        }
                    ); CUERR;
                }

                //copy my gathered elements to result device
                cudaMemcpyPeerAsync(myTmpResultsOnResultDevice,
                    resultDeviceId,
                    myGatherResult.get(),
                    myDeviceId,
                    sizeOfElement * numForLoc,
                    handle->streamsPerGpu[gpu]
                ); CUERR;

                //cudaStreamSynchronize(syncStream); CUERR;

                //cudaDeviceSynchronize(); CUERR;
                
                cudaEventRecord(handle->eventsPerGpu[gpu], handle->streamsPerGpu[gpu]); CUERR;

                //scatter to destination array on resultdevice
                wrapperCudaSetDevice(resultDeviceId);
                cudaStreamWaitEvent(syncStream, handle->eventsPerGpu[gpu], 0); CUERR;

                //cudaDeviceSynchronize(); CUERR;

                //on resultgpu scatter the copied data to the correct positions in final output array
                {
                    assert(resultPitch % sizeof(Value_t) == 0);

                    size_t resultPitchValueTs = resultPitch / sizeof(Value_t);
                    size_t numCols = numColumns;

                    const Value_t* const input = myTmpResultsOnResultDevice;
                    const Index_t* const permutIndices = myDestinationPositionsOnResultDevice;
                    Value_t* const output = d_result;

                    dim3 block(256,1,1);
                    dim3 grid(std::min(320ul, SDIV(numForLoc * numCols, block.x)),1,1);

                    generic_kernel<<<grid, block, 0, syncStream>>>([=] __device__ (){

                        for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; 
                                i < numForLoc * numCols; 
                                i += size_t(blockDim.x) * gridDim.x){

                            const Index_t inputRow = i / numCols;
                            const Index_t col = i % numCols;
                            const Index_t outputRow = permutIndices[inputRow];
                            
                            output[size_t(outputRow) * resultPitchValueTs + col] 
                                = input[size_t(inputRow) * numCols + col];
                        }
                    }); CUERR;
                }

                //cudaDeviceSynchronize(); CUERR;
            }
        }

        //cudaDeviceSynchronize(); CUERR;

        //handle host gathering
        if(handle->numIndicesPerLocation[hostLocation] > 0){
            const int numHostIndices = handle->numIndicesPerLocation[hostLocation];

            Index_t* const myDestinationPositionsOnResultDevice
                    = d_destinationPositionsOnResultDevice + handle->numIndicesPerLocationPS[hostLocation];

            Value_t* const myTmpResultsOnResultDevice 
                = tmpResultsOnResultDevice + handle->numIndicesPerLocationPS[hostLocation];

            wrapperCudaSetDevice(resultDeviceId);
            cudaMemcpyAsync(
                myDestinationPositionsOnResultDevice, 
                handle->pinnedDestinationPositionsOfLocation[hostLocation].get(),
                sizeof(Index_t) * numHostIndices,
                H2D,
                syncStream
            ); CUERR;

            Value_t* const myResult = handle->pinnedGatheredElementsOfLocation[hostLocation].get();   

            forLoop(
                0, 
                numHostIndices, 
                [&](int begin, int end, int threadId){
                    for(int k = begin; k < end; k++){
                        const Index_t localId = handle->indicesOfHostLocation[k] - elementsPerLocationPS[hostLocation];
                        const Value_t* srcPtr = offsetPtr(dataPtrPerLocation[hostLocation], localId);
                        Value_t* destPtr = myResult + size_t(k) * numColumns;
                        std::copy_n(srcPtr, numColumns, destPtr);
                    }
                }
            );

            cudaMemcpyAsync(
                myTmpResultsOnResultDevice,
                myResult,
                sizeof(Value_t) * numColumns * numHostIndices,
                H2D,
                syncStream
            ); CUERR;

            //on resultgpu scatter the copied data to the correct positions in final output array
            {
                assert(resultPitch % sizeof(Value_t) == 0);

                size_t resultPitchValueTs = resultPitch / sizeof(Value_t);
                size_t numCols = numColumns;

                const Value_t* const input = myTmpResultsOnResultDevice;
                const Index_t* const permutIndices = myDestinationPositionsOnResultDevice;
                Value_t* const output = d_result;

                dim3 block(256,1,1);
                dim3 grid(std::min(320ul, SDIV(numHostIndices * numCols, block.x)),1,1);

                generic_kernel<<<grid, block, 0, syncStream>>>([=] __device__ (){

                    for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; 
                            i < numHostIndices * numCols; 
                            i += size_t(blockDim.x) * gridDim.x){

                        const Index_t inputRow = i / numCols;
                        const Index_t col = i % numCols;
                        const Index_t outputRow = permutIndices[inputRow];
                        
                        output[size_t(outputRow) * resultPitchValueTs + col] 
                            = input[size_t(inputRow) * numCols + col];
                    }
                }); CUERR;
            }
        }

        wrapperCudaSetDevice(oldDevice); CUERR;

    }





        //the same GatherHandleStruct must not be used in another call until the results of the previous call are calculated
    template<class ParallelFor>
    void gatherElementsInGpuMemAsyncDefault(ParallelFor&& forLoop,
                                    const GatherHandle& handle,
                                    const Index_t* indices,
                                    const Index_t* d_indices,
                                    Index_t numIds,
                                    int resultDeviceId,
                                    Value_t* d_result,
                                    size_t resultPitch, // result element i begins at offset i * resultPitch
                                    cudaStream_t stream) const{
        if(numIds == 0) return;

        assert(resultPitch % sizeof(Value_t) == 0);

        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        //const int numCpuThreads = care::threadpool.getConcurrency();

        //fastpath, if all elements of distributed array reside in a single partition
        if(singlePartitionInfo.isSinglePartition){
            if(singlePartitionInfo.locationId == hostLocation){
                if(debug) std::cerr << "single location array fasthpath on host\n";

                auto& h_result = handle->pinnedResultData;
                h_result.resize(numIds * SDIV(resultPitch, sizeof(Value_t)));

                auto gather = [&](Index_t begin, Index_t end, int /*threadId*/){
                    for(Index_t k = begin; k < end; k++){
                        const Index_t localId = indices[k];
                        const Value_t* srcPtr = offsetPtr(dataPtrPerLocation[hostLocation], localId);
                        //Value_t* destPtr = offsetPtr(h_result.get(), k);
                        Value_t* destPtr = (Value_t*)(((const char*)(h_result.get())) + resultPitch * k);
                        
                        std::copy_n(srcPtr, numColumns, destPtr);
                    }
                };

                forLoop( 
                    Index_t(0), 
                    numIds, 
                    gather
                );

                int oldDevice; cudaGetDevice(&oldDevice); CUERR;

                wrapperCudaSetDevice(resultDeviceId); CUERR;

                cudaMemcpyAsync(d_result, h_result.get(), numIds * resultPitch, H2D, stream); CUERR;

                wrapperCudaSetDevice(oldDevice); CUERR;

                return;
            }else{
                //if(debug) std::cerr << "single location array fast path on partition " << singlePartitionInfo.locationId << "\n";

                const int locationId = singlePartitionInfo.locationId;
                const int mydeviceId = deviceIds[locationId];
                cudaStream_t mystream = handle->streamsPerGpu[locationId];
                cudaEvent_t myevent = handle->eventsPerGpu[locationId];
                #if 1
                if(mydeviceId == resultDeviceId){
                    copyDataToGpuBufferAsync(d_result, resultPitch, mydeviceId, d_indices, numIds, mydeviceId, stream, -elementsPerLocationPS[locationId]);

                    return;
                }
                #else

                wrapperCudaSetDevice(mydeviceId); CUERR;

                Value_t* destptr = d_result;

                Value_t* localGatherPtr = nullptr;
                Index_t* localIndicesPtr = nullptr;
                cudaStream_t localGatherStream = cudaStream_t(0);

                if(resultDeviceId == mydeviceId){
                    localGatherPtr = destptr;
                    localIndicesPtr = const_cast<Index_t*>(d_indices);
                    localGatherStream = stream;
                }else{
                    handle->dataPerGpu[locationId].resize(numIds * numColumns);
                    localGatherPtr = handle->dataPerGpu[locationId].get();

                    handle->deviceLocalIndicesPerLocation[locationId].resize(numIds);
                    localIndicesPtr = handle->deviceLocalIndicesPerLocation[locationId].get();
                    cudaMemcpyPeerAsync(localIndicesPtr,
                                        mydeviceId,
                                        d_indices,
                                        resultDeviceId,
                                        sizeof(Index_t) * numIds,
                                        localGatherStream); CUERR;

                    localGatherStream = mystream;
                }

                if(debug) cudaDeviceSynchronize(); CUERR;

                //local gather on device mydeviceId
                copyDataToGpuBufferAsync(localGatherPtr, resultPitch, mydeviceId, localIndicesPtr, numIds, mydeviceId, localGatherStream, -elementsPerLocationPS[locationId]);

                if(debug) cudaDeviceSynchronize(); CUERR;           

                //send partial results to destination gpu
                
                if(resultDeviceId != mydeviceId){

                    cudaEventRecord(myevent, localGatherStream); CUERR;
                    wrapperCudaSetDevice(resultDeviceId); CUERR;
                    cudaStreamWaitEvent(stream, myevent,0); CUERR; //wait in result stream until partial results are on the host.
                    
                    cudaMemcpyPeerAsync(destptr,
                                        resultDeviceId,
                                        localGatherPtr,
                                        mydeviceId,
                                        sizeOfElement * numIds,
                                        stream); CUERR;

                    if(debug) cudaDeviceSynchronize(); CUERR;
                } 

                return;
                #endif
               
            }
        }


//TIMERSTARTCPU(countHitsPerLocation);
        //int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        auto deviceIdIter = std::find(deviceIds.begin(), deviceIds.end(), resultDeviceId);
        int deviceIdLocation = -1;
        if(deviceIdIter != deviceIds.end()){
            deviceIdLocation = std::distance(deviceIds.begin(), deviceIdIter);
        }else{
            assert(false);
        }










        
        std::vector<Index_t> hitsPerLocation(numLocations, 0);
        std::vector<int> locationsOfIndices(numIds);
        std::vector<Index_t> localIndices(numIds);

        #if 1   
        const int threadlocoffset = SDIV(numLocations,32) * 32;     
        const int numThreads = forLoop.getNumThreads();
        std::vector<Index_t> hitsPerLocationPerThread(threadlocoffset * numThreads, 0);

        forLoop(
            Index_t(0), 
            numIds, 
            [&](Index_t begin, Index_t end, int threadId){                
                Index_t* hitsptr = hitsPerLocationPerThread.data() + threadlocoffset * threadId;
                for(Index_t i = begin; i < end; i++){
                    const int location = getLocation(indices[i]);
                    locationsOfIndices[i] = location;
                    const Index_t localIndex = indices[i] - elementsPerLocationPS[location];
                    localIndices[i] = localIndex;
                    hitsptr[location]++;
                }
            }
        );

        for(int k = 0; k < numThreads; k++){
            for(int l = 0; l < numLocations; l++){
                hitsPerLocation[l] += hitsPerLocationPerThread[threadlocoffset * k + l];
            }
        }

        // if(hitsPerLocation[0] < hitsPerLocation[1]){
        //     std::cerr << hitsPerLocation[0] << " " << hitsPerLocation[1] << "\n";
        // }

        #else 

        for(Index_t i = 0; i < numIds; i++){
            int location = getLocation(indices[i]);
            hitsPerLocation[location]++;
        }

        #endif


//TIMERSTOPCPU(countsHitPerLocation);

        if(debug){
            std::cerr << "hitsPerLocation: ";
            std::copy(hitsPerLocation.begin(), hitsPerLocation.end(), std::ostream_iterator<Index_t>(std::cerr, " "));
            std::cerr << "\n";
        }

        //shortcut. if all elements reside on the gpu with device id deviceId, perform a simple gather on the gpu, avoiding copies to host and from host.
        bool simpleGatherOnSameDevice = true;

        if(deviceIdLocation != -1){
            for(int i = 0; i < numLocations && simpleGatherOnSameDevice; i++){
                if(i != deviceIdLocation && hitsPerLocation[i] > 0){
                    simpleGatherOnSameDevice = false;
                }
            }
        }

        if(simpleGatherOnSameDevice){
            if(debug) std::cerr << "simpleGatherOnSameDevice " << resultDeviceId << '\n';
            copyDataToGpuBufferAsync(d_result, resultPitch, resultDeviceId, d_indices, numIds, resultDeviceId, stream, -elementsPerLocationPS[deviceIdLocation]);
            return;
        }

//TIMERSTARTCPU(makenewindices);
    	std::vector<Index_t> hitsPerLocationPrefixSum(numLocations+1,0);
    	std::partial_sum(hitsPerLocation.begin(), hitsPerLocation.end(), hitsPerLocationPrefixSum.begin()+1);
        std::fill(hitsPerLocation.begin(), hitsPerLocation.end(), 0);

        handle->pinnedLocalIndices.resize(numIds);
        handle->pinnedPermutationIndices.resize(numIds);
        for(Index_t i = 0; i < numIds; i++){
    		const int location = locationsOfIndices[i]; //getLocation(indices[i]);
            const Index_t localIndex = localIndices[i]; //indices[i] - elementsPerLocationPS[location];
            const Index_t tmpresultindex = hitsPerLocationPrefixSum[location] + hitsPerLocation[location];
            handle->pinnedPermutationIndices[i] = tmpresultindex;

            handle->pinnedLocalIndices[tmpresultindex] = localIndex;
            //std::cerr << "read id " << readIds[i] << ", location " << location << ", localReadId " << localReadId << ", index " << index << "\n";
    		hitsPerLocation[location]++;
    	}
//TIMERSTOPCPU(makenewindices);

//TIMERSTARTCPU(resizedevicevectors)
        handle->pinnedResultData.resize(numIds * numColumns);

        wrapperCudaSetDevice(resultDeviceId); CUERR;

        handle->tmpResultsOfDevice[resultDeviceId].resize(numIds * numColumns);

        handle->permutationIndicesOfDevice[resultDeviceId].resize(numIds);

        cudaMemcpyAsync(handle->permutationIndicesOfDevice[resultDeviceId],
                        handle->pinnedPermutationIndices.get(),
                        sizeof(Index_t) * numIds,
                        H2D,
                        stream); CUERR;

        if(debug) cudaDeviceSynchronize(); CUERR;

//TIMERSTOPCPU(resizedevicevectors);

        //gather from gpus to host
        for(int gpu = 0; gpu < numGpus; gpu++){
            Index_t numHits = hitsPerLocation[gpu];
            if(numHits > 0){
                int mydeviceId = deviceIds[gpu];
                cudaStream_t mystream = handle->streamsPerGpu[gpu];
                cudaEvent_t myevent = handle->eventsPerGpu[gpu];
             
                wrapperCudaSetDevice(mydeviceId); CUERR;

                Value_t* destptr = offsetPtr(handle->tmpResultsOfDevice[resultDeviceId].get(), hitsPerLocationPrefixSum[gpu]);

                Value_t* localGatherPtr = nullptr;
                cudaStream_t localGatherStream = cudaStream_t(0);

                if(resultDeviceId == mydeviceId){
                    localGatherPtr = destptr;
                    localGatherStream = stream;
                }else{
                    handle->dataPerGpu[gpu].resize(numHits * numColumns);
                    localGatherPtr = handle->dataPerGpu[gpu].get();
                    localGatherStream = mystream;
                }
                
                auto& myLocalIds = handle->deviceLocalIndicesPerLocation[gpu];

                myLocalIds.resize(numHits);
                Index_t* localIdsPtr = myLocalIds.get();
                cudaMemcpyAsync(localIdsPtr,
                                handle->pinnedLocalIndices.get() + hitsPerLocationPrefixSum[gpu],
                                sizeof(Index_t) * numHits,
                                H2D,
                                localGatherStream); CUERR;

                if(debug) cudaDeviceSynchronize(); CUERR;

                //local gather on device mydeviceId
                copyDataToGpuBufferAsync(localGatherPtr, sizeOfElement, mydeviceId, localIdsPtr, numHits, mydeviceId, localGatherStream, 0);

                if(debug) cudaDeviceSynchronize(); CUERR;

                //send partial results to destination gpu
                if(resultDeviceId != mydeviceId){
                    cudaEventRecord(myevent, mystream); CUERR;
                    wrapperCudaSetDevice(resultDeviceId); CUERR;
                    cudaStreamWaitEvent(stream, myevent,0); CUERR; //wait in result stream until partial results are on the host.
                    
                    cudaMemcpyPeerAsync(destptr,
                                        resultDeviceId,
                                        localGatherPtr,
                                        mydeviceId,
                                        sizeOfElement * numHits,
                                        stream); CUERR;

                    if(debug){ cudaDeviceSynchronize(); CUERR;}
                }                       
            }
    	}

        //gather from host to host
        if(hitsPerLocation[hostLocation] > 0){
            const Index_t numHits = hitsPerLocation[hostLocation];
            const auto hitsOffset = hitsPerLocationPrefixSum[hostLocation];
            const Index_t* hostLocalIds = handle->pinnedLocalIndices.get() + hitsOffset;            

            forLoop(
                Index_t(0), 
                numHits, 
                [&](Index_t begin, Index_t end, int threadId){
                    for(Index_t k = begin; k < end; k++){
                        const Index_t localId = hostLocalIds[k];
                        const Value_t* srcPtr = offsetPtr(dataPtrPerLocation[hostLocation], localId);
                        Value_t* destPtr = offsetPtr(handle->pinnedResultData.get(), hitsOffset + k);
                        std::copy_n(srcPtr, numColumns, destPtr);
                    }
                }
            );
        }

        wrapperCudaSetDevice(resultDeviceId); CUERR;

        cudaMemcpyAsync(offsetPtr(handle->tmpResultsOfDevice[resultDeviceId].get(), hitsPerLocationPrefixSum[hostLocation]),
                        offsetPtr(handle->pinnedResultData.get(), hitsPerLocationPrefixSum[hostLocation]),
                        hitsPerLocation[hostLocation] * sizeOfElement,
                        H2D,
                        stream); CUERR;

        if(debug) cudaDeviceSynchronize(); CUERR;

        {
            assert(resultPitch % sizeof(Value_t) == 0);

            size_t resultPitchValueTs = resultPitch / sizeof(Value_t);
            size_t numCols = numColumns;
            const Index_t* indices = handle->permutationIndicesOfDevice[resultDeviceId].get();
            const Value_t* src = handle->tmpResultsOfDevice[resultDeviceId].get();
            Value_t* dest = d_result;
            Index_t n = numIds;

            dim3 block(256,1,1);
            dim3 grid(std::min(320ul, SDIV(n * numCols, block.x)),1,1);

            generic_kernel<<<grid, block, 0, stream>>>([=] __device__ (){
                for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; i < n * numCols; i += size_t(blockDim.x) * gridDim.x){
                    const Index_t outputrow = i / numCols;
                    const Index_t inputrow = indices[outputrow];
                    const Index_t col = i % numCols;
                    dest[size_t(outputrow) * resultPitchValueTs + col] = src[size_t(inputrow) * numCols + col];
                }
            }); CUERR;

        }

        if(debug) cudaDeviceSynchronize(); CUERR;

        // cudaStreamSynchronize(stream); CUERR;
        
        // std::cerr << "permutationIndices: ";
        // std::copy(handle->pinnedPermutationIndices.get(), handle->pinnedPermutationIndices.get() + numIds, std::ostream_iterator<int>(std::cerr, " "));
        // std::cerr << "\n";
        
        // // std::vector<int> hostResultPermuted(numIds);
        // // for(int i = 0; i < numIds; i++){
        // //     hostResultPermuted[i] = handle->pinnedResultData[permutationIndices[i]];
        // // }
        
        // std::cerr << "pinnedLocalIndices: ";
        // std::copy(handle->pinnedLocalIndices.get(), handle->pinnedLocalIndices.get() + hitsPerLocationPrefixSum.back(), std::ostream_iterator<int>(std::cerr, " "));
        // std::cerr << "\n";
        
        // // std::cerr << "pinnedResultData: ";
        // // std::copy(handle->pinnedResultData.get(), handle->pinnedResultData.get() + hitsPerLocationPrefixSum.back(), std::ostream_iterator<int>(std::cerr, " "));
        // // std::cerr << "\n";
        
        // // std::cerr << "permutedResultData: ";
        // // std::copy(hostResultPermuted.begin(), hostResultPermuted.end(), std::ostream_iterator<int>(std::cerr, " "));
        // // std::cerr << "\n";

        // std::exit(0);
        wrapperCudaSetDevice(oldDevice); CUERR;
    }











    struct GatherHandleStruct2{
        using PositionIndexPair = typename cub::ArgIndexInputIterator<Index_t*>::value_type;
        SimpleAllocationPinnedHost<Value_t> pinnedResultData;
        SimpleAllocationPinnedHost<PositionIndexPair> pinnedPositionIndexPairsForLocation;
        SimpleAllocationPinnedHost<Index_t> pinnedNumIndicesForHost;
        SimpleAllocationPinnedHost<Index_t> pinnedDestinationPositionsOfHostIndices;

        std::vector<Index_t> indicesForHost;

        std::map<int, SimpleAllocationDevice<Index_t>> map_d_destinationPositionsOfHostIndices;

        std::map<int, SimpleAllocationDevice<Index_t>> map_d_elementsPerLocationPS;
        std::map<int, SimpleAllocationDevice<Index_t>> map_d_numIndicesPerLocation;
        std::map<int, SimpleAllocationDevice<Index_t>> map_d_numIndicesPerLocationPS;
        std::map<int, SimpleAllocationDevice<PositionIndexPair>> map_d_positionIndexPairsForLocation;    
        std::map<int, SimpleAllocationDevice<Value_t>> map_d_gatheredElementsForLocation;
        std::map<int, cudaStream_t> map_cudastreams;
        std::map<int, cudaEvent_t> map_cudaevents;
        std::map<int, cudaEvent_t> map_cudaeventsForHostComm;

        std::vector<int> registeredDeviceIds;

        //cudaEvent_t readyEvent;
        care::ThreadPool::ParallelForHandle pforHandle;
    };

    using GatherHandle2 = std::unique_ptr<GatherHandleStruct2>;

    GatherHandle2 makeGatherHandle2() const{
        auto handle = std::make_unique<GatherHandleStruct2>();

        handle->pinnedNumIndicesForHost.resize(1);

        for(int gpu = 0; gpu < numGpus; gpu++){
            registerDeviceIdForHandle(handle, deviceIds[gpu]);
        }

        return handle;
    }

    void destroyGatherHandle2(const GatherHandle2& handle) const{
        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        handle->pinnedResultData.destroy();
        handle->pinnedPositionIndexPairsForLocation.destroy();
        handle->pinnedNumIndicesForHost.destroy();
        handle->pinnedDestinationPositionsOfHostIndices.destroy();

        for(int deviceId : handle->registeredDeviceIds){
            wrapperCudaSetDevice(deviceId); CUERR;

            handle->map_d_elementsPerLocationPS[deviceId].destroy();
            handle->map_d_numIndicesPerLocation[deviceId].destroy();
            handle->map_d_numIndicesPerLocationPS[deviceId].destroy();
            handle->map_d_indicesForLocation[deviceId].destroy();
            handle->map_d_gatheredElementsForLocation[deviceId].destroy();
            handle->map_d_destinationPositionsOfHostIndices[deviceId].destroy();

            cudaStreamDestroy(handle->map_cudastreams[deviceId]); CUERR;
            cudaEventDestroy(handle->map_cudaevents[deviceId]); CUERR;
            cudaEventDestroy(handle->map_cudaeventsForHostComm[deviceId]); CUERR;
        }

        wrapperCudaSetDevice(oldDevice); CUERR;
    }


    void registerDeviceIdForHandle(GatherHandle2& handle, int deviceId){
        using PositionIndexPair = typename cub::ArgIndexInputIterator<Index_t*>::value_type;

        auto it = std::find(
            handle->registeredDeviceIds.begin(), 
            handle->registeredDeviceIds.end(), 
            deviceId
        );

        if(it == handle->registeredDeviceIds.end()){
            int oldId = 0;
            cudaGetDevice(&oldId); CUERR;
            wrapperCudaSetDevice(deviceId); CUERR;

            SimpleAllocationDevice<Index_t> d_ps(numLocations+1);
            cudaMemcpy(
                d_ps.get(),
                elementsPerLocationPS.data(),
                sizeof(Index_t) * (numLocations+1),
                H2D
            ); CUERR;  

            handle->map_d_elementsPerLocationPS[deviceId] = std::move(d_ps);

            SimpleAllocationDevice<Index_t> d_iPerL(numLocations);
            handle->map_d_numIndicesPerLocation[deviceId] = std::move(d_iPerL);

            SimpleAllocationDevice<Index_t> d_iPerLPS(numLocations+1);
            handle->map_d_numIndicesPerLocationPS[deviceId] = std::move(d_iPerLPS);

            SimpleAllocationDevice<PositionIndexPair> d_posIndexPairsForLocation(numLocations);
            handle->map_d_positionIndexPairsForLocation[deviceId] = std::move(d_posIndexPairsForLocation);

            SimpleAllocationDevice<Value_t> d_gatheredElements;
            handle->map_d_gatheredElementsForLocation[deviceId] = std::move(d_gatheredElements);

            SimpleAllocationDevice<Index_t> d_destpos;
            handle->map_d_destinationPositionsOfHostIndices[deviceId] = std::move(d_destpos);
                    
            cudaStream_t stream;
            cudaStreamCreate(&stream); CUERR;
            handle->map_cudastreams[deviceId] = std::move(stream);

            cudaEvent_t event;
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming); CUERR;
            handle->map_cudaevents[deviceId] = std::move(event);
            
            cudaEvent_t event2;
            cudaEventCreateWithFlags(&event2, cudaEventDisableTiming); CUERR;
            handle->map_cudaeventsForHostComm[deviceId] = std::move(event2);            

            handle->registeredDeviceIds.push_back(deviceId);

            wrapperCudaSetDevice(oldId); CUERR;
        }
    }


    //the same GatherHandleStruct must not be used in another call until the results of the previous call are calculated
    template<class ParallelFor>
    void gatherElementsInGpuMemAsync2(ParallelFor&& forLoop,
                                    GatherHandle2& handle,
                                    const Index_t* indices,
                                    const Index_t* d_indices,
                                    Index_t numIds,
                                    int resultDeviceId,
                                    Value_t* d_result,
                                    size_t resultPitch, // result element i begins at offset i * resultPitch
                                    cudaStream_t syncstream //must be on device resutlDeviceId
                                    ) const{
        if(numIds == 0) return;

        using PositionIndexPair = typename cub::ArgIndexInputIterator<Index_t*>::value_type;

        assert(resultPitch % sizeof(Value_t) == 0);

        //const int numCpuThreads = care::threadpool.getConcurrency();

        //fastpath, if all elements of distributed array reside in a single partition
        if(singlePartitionInfo.isSinglePartition){
            if(singlePartitionInfo.locationId == hostLocation){
                if(debug) std::cerr << "single location array fasthpath on host\n";

                auto& h_result = handle->pinnedResultData;
                h_result.resize(numIds * SDIV(resultPitch, sizeof(Value_t)));

                auto gather = [&](Index_t begin, Index_t end, int /*threadId*/){
                    for(Index_t k = begin; k < end; k++){
                        const Index_t localId = indices[k];
                        const Value_t* srcPtr = offsetPtr(dataPtrPerLocation[hostLocation], localId);
                        //Value_t* destPtr = offsetPtr(h_result.get(), k);
                        Value_t* destPtr = (Value_t*)(((const char*)(h_result.get())) + resultPitch * k);
                        
                        std::copy_n(srcPtr, numColumns, destPtr);
                    }
                };

                forLoop( 
                    Index_t(0), 
                    numIds, 
                    gather
                );

                int oldDevice; cudaGetDevice(&oldDevice); CUERR;

                wrapperCudaSetDevice(resultDeviceId); CUERR;

                cudaMemcpyAsync(d_result, h_result.get(), numIds * resultPitch, H2D, syncstream); CUERR;

                wrapperCudaSetDevice(oldDevice); CUERR;

                return;
            }else{
                //if(debug) std::cerr << "single location array fast path on partition " << singlePartitionInfo.locationId << "\n";

                const int locationId = singlePartitionInfo.locationId;
                const int mydeviceId = deviceIds[locationId];

                if(mydeviceId == resultDeviceId){
                    copyDataToGpuBufferAsync(
                        d_result, 
                        resultPitch, 
                        mydeviceId, 
                        d_indices, 
                        numIds, 
                        mydeviceId, 
                        syncstream, 
                        -elementsPerLocationPS[locationId]
                    );

                    return;
                }               
            }
        }


//TIMERSTARTCPU(countHitsPerLocation);
        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        auto deviceIdIter = std::find(deviceIds.begin(), deviceIds.end(), resultDeviceId);
        int deviceIdLocation = -1;
        if(deviceIdIter != deviceIds.end()){
            deviceIdLocation = std::distance(deviceIds.begin(), deviceIdIter);
        }else{
            assert(false);
        }

        registerDeviceIdForHandle(handle, resultDeviceId);
        

        auto& d_destination_elementsPerLocationPS = handle->map_d_elementsPerLocationPS[resultDeviceId];
        auto& d_destination_numIndicesPerLocation = handle->map_d_numIndicesPerLocation[resultDeviceId];
        auto& d_destination_numIndicesPerLocationPS = handle->map_d_numIndicesPerLocationPS[resultDeviceId];
        auto& d_destination_gatheredElementsForLocation = handle->map_d_gatheredElementsForLocation[resultDeviceId];
        auto& d_destination_posIndexPairsForLocation = handle->map_d_positionIndexPairsForLocation[resultDeviceId];
        auto& d_destination_cubTemp = handle->map_d_cubTemp[resultDeviceId];

        auto& destination_cudaevent = handle->map_cudaevents[resultDeviceId];
        auto& destination_stream = handle->map_cudastreams[resultDeviceId];

        cudaEventRecord(destination_cudaevent, syncstream); CUERR;
        cudaStreamWaitEvent(destination_stream, destination_cudaevent, 0); CUERR;

        //find indices per location + prefixsum
        {
            int numLocs = numLocations;
            auto* d_elementsPerLocationPSPtr = d_destination_elementsPerLocationPS.get();
            auto* d_indicesPerLocationPtr = d_destination_numIndicesPerLocation.get();
            auto* d_indicesPerLocationPSPtr = d_destination_numIndicesPerLocationPS.get();

            cudaMemsetAsync(
                d_indicesPerLocationPtr, 
                0, 
                sizeof(Index_t) * numIds, 
                destination_stream
            ); CUERR;

            assert(maxNumGpus <= 32);
            generic_kernel<<<SDIV(numIds, 256), 256, 0, destination_stream>>>([=] __device__ (){

                __shared__ Index_t shared_elementsPerLocationPS[34]; //maxnumGpus + 2
                __shared__ Index_t shared_numHitsPerLocation[33]; //maxnumGpus + 1

                for(int tid = threadIdx.x; tid < numLocs; tid += blockDim.x){
                    shared_numHitsPerLocation[tid] = 0;
                }

                for(int tid = threadIdx.x; tid < numLocs+1; tid += blockDim.x){
                    shared_elementsPerLocationPS[tid] = d_elementsPerLocationPSPtr[tid];
                }

                __syncthreads();
                
                for(Index_t tid = threadIdx.x + blockIdx.x * Index_t(blockDim.x); 
                        tid < numIds; 
                        tid += Index_t(blockDim.x) * gridDim.x){

                    Index_t element = d_indices[tid];

                    for(int i = 0; i < numLocs; i++){
                        if(shared_elementsPerLocationPS[i] <= element && element < shared_elementsPerLocationPS[i+1]){
                            atomicAdd(shared_numHitsPerLocation + i, 1);
                        }
                    }
                }

                __syncthreads();

                for(int tid = threadIdx.x; tid < numLocs; tid += blockDim.x){
                    atomicAdd(d_indicesPerLocationPtr + tid, shared_numHitsPerLocation[tid]);
                }
                
            }); CUERR;

            generic_kernel<<<1,1, 0, destination_stream>>>([=]__device__(){
                d_indicesPerLocationPSPtr[0] = 0;
                
                for(int i = 0; i < numLocs; i++){
                    d_indicesPerLocationPSPtr[i+1] = d_indicesPerLocationPSPtr[i] + d_indicesPerLocationPtr[i];
                }
            }); CUERR;

            cudaEventRecord(destination_cudaevent, destination_stream); CUERR;
        }

        //if host memory is used by distributed array, calculate the respective indices on destination gpu,
        //then async copy indices to host.
        // if(elementsPerLocation[hostLocation] > 0){
        //     cudaMemcpyAsync(
        //         handle->pinnedNumIndicesForHost.get(),
        //         d_destination_numIndicesPerLocation.get() + hostLocation,
        //         sizeof(Index_t),
        //         destination_stream,
        //         D2H
        //     ); CUERR;

        //     handle->pinnedPositionIndexPairsForLocation.resize(numIds);
        //     d_destination_posIndexPairsForLocation.resize(numIds);

        //     size_t cubBytes = 0;
        //     //determine temp bytes
        //     cub::DeviceSelect::If(
        //         nullptr, 
        //         cubBytes, 
        //         cub::ArgIndexInputIterator<Index_t*>(d_indices), 
        //         d_destination_posIndexPairsForLocation.get(), 
        //         cub::DiscardOutputIterator<>{}, //calculated in d_destination_indicesPerLocation[gpu]
        //         numIds, 
        //         [=] __device__ (auto pair){
        //             //pair.key == position, pair.value == d_indices[pair.key]
        //             return d_destination_elementsPerLocationPS[hostLocation] <= pair.value 
        //                     && pair.value < d_destination_elementsPerLocationPS[hostLocation+1];
        //         },
        //         destination_stream
        //     );

        //     d_destination_cubTemp.resize(cubBytes);

        //     //select indices for host
        //     cub::DeviceSelect::If(
        //         d_destination_cubTemp.get(), 
        //         cubBytes, 
        //         cub::ArgIndexInputIterator<Index_t*>(d_indices), 
        //         d_destination_posIndexPairsForLocation.get(), 
        //         cub::DiscardOutputIterator<>{}, //calculated in d_destination_indicesPerLocation[gpu]
        //         numIds, 
        //         [=] __device__ (auto pair){
        //             //pair.key == position, pair.value == d_indices[pair.key]
        //             return d_destination_elementsPerLocationPS[hostLocation] <= pair.value 
        //                     && pair.value < d_destination_elementsPerLocationPS[hostLocation+1];
        //         },
        //         destination_stream
        //     );

        //     cudaMemcpyAsync(
        //         handle->pinnedPositionIndexPairsForLocation.get(),
        //         d_destination_posIndexPairsForLocation.get(),
        //         d_destination_posIndexPairsForLocation.sizeInBytes(),
        //         destination_stream,
        //         D2H
        //     ); CUERR;

        //     cudaEventRecord(handle->map_cudaeventsForHostComm[resultDeviceId], destination_stream); CUERR;
        // }

        //gather elements locally on the respective partitions, copy to result device. scatter to correct positions
        for(int gpu = 0; gpu < numGpus; gpu++){
            const int location = gpu;
            if(elementsPerLocation[location] > 0){
                const int gpu_deviceId = deviceIds[gpu];

                wrapperCudaSetDevice(gpu_deviceId); CUERR;

                auto& gpu_cudaevent = handle->map_cudaevents[gpu_deviceId];
                auto& gpu_stream = handle->map_cudastreams[gpu_deviceId];

                auto& gpu_elementsPerLocationPS = handle->map_d_elementsPerLocationPS[gpu_deviceId];
                auto& gpu_gatheredElementsForLocation = handle->map_d_gatheredElementsForLocation[gpu_deviceId];
                auto& gpu_posIndexPairsForLocation = handle->map_d_positionIndexPairsForLocation[gpu_deviceId];
                auto& gpu_numIndicesPerLocation = handle->map_d_numIndicesPerLocation[gpu_deviceId];
                auto& gpu_numIndicesPerLocationPS = handle->map_d_numIndicesPerLocationPS[gpu_deviceId];
                auto& gpu_cubTemp = handle->map_d_cubTemp[gpu_deviceId];

                auto* d_destination_elementsPerLocationPSPtr = d_destination_elementsPerLocationPS.get();

                //wait for counting and prefixsum
                cudaStreamWaitEvent(gpu_stream, destination_cudaevent, 0); CUERR;

                //copy num indices per location + prefixsum to gpu, since it is accessed multiple times
                cudaMemcpyPeerAsync(
                    gpu_numIndicesPerLocation.get(),
                    gpu_deviceId,
                    d_destination_numIndicesPerLocation.get(),
                    resultDeviceId,
                    d_destination_numIndicesPerLocation.sizeInBytes(),
                    gpu_stream
                ); CUERR;

                cudaMemcpyPeerAsync(
                    gpu_numIndicesPerLocationPS.get(),
                    gpu_deviceId,
                    d_destination_numIndicesPerLocationPS.get(),
                    resultDeviceId,
                    d_destination_numIndicesPerLocationPS.sizeInBytes(),
                    gpu_stream
                ); CUERR;

                gpu_posIndexPairsForLocation.resize(numIds);

                size_t cubBytes = 0;
                //determine temp bytes
                cub::DeviceSelect::If(
                    nullptr, 
                    cubBytes, 
                    cub::ArgIndexInputIterator<Index_t*>(d_indices), 
                    gpu_posIndexPairsForLocation.get(), 
                    cub::DiscardOutputIterator<>{}, //calculated in d_destination_indicesPerLocation[gpu]
                    numIds, 
                    [=] __device__ (auto pair){
                        //pair.key == position, pair.value == d_indices[pair.key]
                        return d_destination_elementsPerLocationPSPtr[location] <= pair.value 
                                && pair.value < d_destination_elementsPerLocationPSPtr[location+1];
                    },
                    gpu_stream
                );

                gpu_cubTemp.resize(cubBytes);

                //select indices for current gpu
                cub::DeviceSelect::If(
                    gpu_cubTemp.get(), 
                    cubBytes, 
                    cub::ArgIndexInputIterator<Index_t*>(d_indices), 
                    gpu_posIndexPairsForLocation.get(), 
                    cub::DiscardOutputIterator<>{}, //calculated in d_destination_indicesPerLocation[gpu]
                    numIds, 
                    [=] __device__ (auto pair){
                        //pair.key == position, pair.value == d_indices[pair.key]
                        return d_destination_elementsPerLocationPSPtr[location] <= pair.value 
                                && pair.value < d_destination_elementsPerLocationPSPtr[location+1];
                    },
                    gpu_stream
                );

                //gather elements of selected indices
                {
                    //allocate upper bound. number of elements for location is not available on host
                    gpu_gatheredElementsForLocation.resize(numColumns * numIds); 

                    const size_t numCols = numColumns;
                    const size_t resultPitchValueTs = numColumns;
                    Value_t* const outputBasePtr = gpu_gatheredElementsForLocation.get();
                    const Value_t* const sourceArrayDataPtr = dataPtrPerLocation[gpu];
                    const Index_t* const outputOffsetPtr = gpu_numIndicesPerLocationPS.get() + gpu;
                    const PositionIndexPair* const myposIndexPairsPtr = gpu_posIndexPairsForLocation.get();
                    const Index_t* const myNumIndicesPtr = gpu_numIndicesPerLocation.get() + gpu;
                    const Index_t* const indexOffsetPtr = gpu_elementsPerLocationPS.get() + gpu;

                    generic_kernel<<<SDIV(numIds, 128), 128, 0, gpu_stream>>>(
                        [=] __device__ (){

                            const Index_t nIndices = *myNumIndicesPtr;
                            const Index_t indexOffset = *indexOffsetPtr;
                            const Index_t outputOffset = *outputOffsetPtr;
                            Value_t* const outputPtr = outputBasePtr + size_t(outputOffset) * numCols;

                            for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; 
                                    i < nIndices * numCols; 
                                    i += size_t(blockDim.x) * gridDim.x){

                                const Index_t outputrow = i / numCols;
                                const Index_t inputrow = myposIndexPairsPtr[outputrow].value - indexOffset;
                                const Index_t col = i % numCols;
                                outputPtr[size_t(outputrow) * resultPitchValueTs + col] 
                                        = sourceArrayDataPtr[size_t(inputrow) * numCols + col];
                            }
                        }
                    ); CUERR;
                }

                //copy gathered data to target gpu
                if(resultDeviceId != gpu_deviceId){
                    
                    const size_t numCols = numColumns;
                    Value_t* const outputBasePtr = d_destination_gatheredElementsForLocation.get();
                    const Value_t* const inputPtr = gpu_gatheredElementsForLocation[gpu];
                    const Index_t* const outputOffsetPtr = gpu_numIndicesPerLocationPS.get() + gpu;
                    const Index_t* const myNumIndicesPtr = gpu_numIndicesPerLocation.get() + gpu;

                    generic_kernel<<<SDIV(numIds, 256), 256, 0, gpu_stream>>>(
                        [=] __device__ (){

                            const Index_t nIndices = *myNumIndicesPtr;
                            const Index_t outputOffset = *outputOffsetPtr;
                            Value_t* const outputPtr = outputBasePtr + size_t(outputOffset) * numCols;

                            for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; 
                                    i < nIndices * numCols; 
                                    i += size_t(blockDim.x) * gridDim.x){

                                outputPtr[i] = inputPtr[i];
                            }
                        }
                    ); CUERR;
                }

                if(debug){ cudaDeviceSynchronize(); CUERR;}

                cudaEventRecord(gpu_cudaevent, gpu_stream); CUERR;
                
                wrapperCudaSetDevice(gpu_deviceId); CUERR;
                cudaStreamWaitEvent(destination_stream, gpu_cudaevent, 0); CUERR;

                //on resultgpu scatter the copied data to the correct positions in final output array
                {
                    assert(resultPitch % sizeof(Value_t) == 0);

                    size_t resultPitchValueTs = resultPitch / sizeof(Value_t);
                    size_t numCols = numColumns;

                    const Value_t* const inputBasePtr = d_destination_gatheredElementsForLocation.get();
                    const Index_t* const inputOffsetPtr = d_destination_numIndicesPerLocationPS.get() + gpu;

                    Value_t* dest = d_result;
                    const Index_t* const myNumIndicesPtr = d_destination_numIndicesPerLocation.get() + gpu;
                    //peer access
                    const PositionIndexPair* const myposIndexPairsPtr = gpu_posIndexPairsForLocation.get();

                    dim3 block(256,1,1);
                    dim3 grid(std::min(320ul, SDIV(numIds * numCols, block.x)),1,1);

                    generic_kernel<<<grid, block, 0, destination_stream>>>([=] __device__ (){
                        const Index_t nIndices = *myNumIndicesPtr;

                        const Value_t* const src = inputBasePtr + *inputOffsetPtr;

                        for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; 
                                i < nIndices * numCols; 
                                i += size_t(blockDim.x) * gridDim.x){

                            const Index_t inputRow = i / numCols;
                            const Index_t col = i % numCols;
                            const Index_t outputRow = myposIndexPairsPtr[inputRow].key;
                            
                            dest[size_t(outputRow) * resultPitchValueTs + col] 
                                = src[size_t(inputRow) * numCols + col];
                        }
                    }); CUERR;
                }

            }else{
                //do nothing
            }
        }

        //handle data from host
        handle->indicesForHost.resize(numIds);
        handle->pinnedDestinationPositionsOfHostIndices.resize(numIds);

        std::size_t numIndicesNotForHost = 0;
        std::size_t numHostIndices = 0;

        for(Index_t i = 0; i < numIds; i++){
            const Index_t index = indices[i];
            if(index >= elementsPerLocationPS[hostLocation]){
                handle->indicesForHost[numHostIndices] = index;
                handle->pinnedDestinationPositionsOfHostIndices[numHostIndices] = i;
                numHostIndices++;
            }else{
                numIndicesNotForHost++;
            }
        }

        //const std::size_t numHostIndices = std::distance(handle->indicesForHost.begin(), indicesForHostEnd);

        wrapperCudaSetDevice(resultDeviceId);

        if(numHostIndices > 0){
            cudaMemcpyAsync(
                handle->map_d_destinationPositionsOfHostIndices[resultDeviceId].get(),
                handle->pinnedDestinationPositionsOfHostIndices.get(),
                sizeof(Index_t) * numHostIndices,
                H2D,
                syncstream
            ); CUERR;            

            forLoop(
                std::size_t(0), 
                numHostIndices, 
                [&](std::size_t begin, std::size_t end, int threadId){
                    const std::size_t indexoffset = elementsPerLocationPS[hostLocation];
                    
                    for(std::size_t k = begin; k < end; k++){
                        const std::size_t localId = handle->indicesForHost[k] - indexoffset;

                        const Value_t* srcPtr = offsetPtr(dataPtrPerLocation[hostLocation], localId);

                        Value_t* destPtr = handle->pinnedResultData.get() + std::size_t(numColumns) * k;
                        std::copy_n(srcPtr, numColumns, destPtr);
                    }
                }
            );

            cudaMemcpyAsync(
                d_destination_gatheredElementsForLocation.get() + numColumns * numIndicesNotForHost,
                handle->pinnedResultData.get(),
                sizeof(Value_t) * numColumns * numHostIndices,
                H2D,
                syncstream
            ); CUERR;

            //on resultgpu scatter the copied data to the correct positions in final output array
            {
                assert(resultPitch % sizeof(Value_t) == 0);

                size_t resultPitchValueTs = resultPitch / sizeof(Value_t);
                size_t numCols = numColumns;

                const Value_t* const inputBasePtr = d_destination_gatheredElementsForLocation.get();
                const Index_t* const inputOffsetPtr = d_destination_numIndicesPerLocationPS.get() + hostLocation;

                Value_t* dest = d_result;
                const Index_t* const myNumIndicesPtr = d_destination_numIndicesPerLocation.get() + hostLocation;

                const Index_t* const destPositionsPtr = handle->map_d_destinationPositionsOfHostIndices[resultDeviceId].get();

                dim3 block(256,1,1);
                dim3 grid(std::min(320ul, SDIV(numIds * numCols, block.x)),1,1);

                generic_kernel<<<grid, block, 0, syncstream>>>([=] __device__ (){
                    const Index_t nIndices = *myNumIndicesPtr;

                    const Value_t* const src = inputBasePtr + *inputOffsetPtr;

                    for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; 
                            i < nIndices * numCols; 
                            i += size_t(blockDim.x) * gridDim.x){

                        const Index_t inputRow = i / numCols;
                        const Index_t col = i % numCols;
                        const Index_t outputRow = destPositionsPtr[inputRow];
                        
                        dest[size_t(outputRow) * resultPitchValueTs + col] 
                            = src[size_t(inputRow) * numCols + col];
                    }
                }); CUERR;
            }
        }

        // //scan host input indices for any host index
        // bool anyHostIndex = std::any_of(
        //     indices, 
        //     indices + numIds, 
        //     [](auto index){
        //         return index >= elementsPerLocationPS[hostLocation];
        //     }
        // );

        // if(anyHostIndex){
        //     //calculate indices for host, on host. stop if indices for host which were calculated on the gpu
        //     //finished transfering to host.

        //     if(cudaEventNotReady == cudaEventQuery(handle->map_cudaeventsForHostComm[resultDeviceId])){

        //     }


        // }

        wrapperCudaSetDevice(oldDevice); CUERR;
    }






    // d_result points to memory of resultDevice, d_indices points to memory of sourceDevice. d_indices[i] + indexOffset. 
    // Gathers array elements of sourceDevice to resultDevice
    // if resultDevice != sourceDevice, peer access must be enabled
    void copyDataToGpuBufferAsync(Value_t* d_result, size_t resultPitch, int resultDevice, const Index_t* d_indices, Index_t nIndices, int sourceDevice, cudaStream_t stream, Index_t indexOffset) const{
        //assert(resultPitch >= sizeOfElement);
        assert(resultPitch % sizeof(Value_t) == 0);

        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        auto it = std::find(deviceIds.begin(), deviceIds.end(), sourceDevice);
        assert(it != deviceIds.end());

        const int gpu = std::distance(deviceIds.begin(), it);
        //size_t sizeOfElement_ = sizeOfElement;
        const size_t numCols = numColumns;
        const size_t resultPitchValueTs = resultPitch / sizeof(Value_t);

        const Value_t* const gpuData = dataPtrPerLocation[gpu];

        wrapperCudaSetDevice(resultDevice); CUERR;

        dim3 block(256,1,1);
        dim3 grid(std::min(320ul, SDIV(nIndices * numCols, block.x)),1,1);

        // generic_kernel<<<grid, block, 0, stream>>>([=] __device__ (){
        //     for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; i < nIndices * numCols; i += size_t(blockDim.x) * gridDim.x){
        //         const Index_t outputrow = i / numCols;
        //         const Index_t inputrow = d_indices[outputrow] + indexOffset;
        //         const Index_t col = i % numCols;
        //         d_result[size_t(outputrow) * resultPitchValueTs + col] 
        //                 = gpuData[size_t(inputrow) * numCols + col];
        //     }
        // }); CUERR;

        distrArrayGatherKernel<<<grid, block, 0, stream>>>(
            d_result,
            gpuData,
            d_indices,
            nIndices,
            indexOffset,
            resultPitchValueTs,
            numCols
        ); CUERR;

        wrapperCudaSetDevice(oldDevice); CUERR;
    }

        // d_result points to memory of resultDevice, d_indices points to memory of sourceDevice. d_indices[i] + indexOffset. 
    // Gathers array elements of sourceDevice to resultDevice
    // if resultDevice != sourceDevice, peer access must be enabled
    void copyDataToGpuBufferAsync2(
                Value_t* d_result, size_t resultPitch, int resultDevice, const Index_t* d_indices, const Index_t* d_numIndicesPtr,
                Index_t maxNumIndices, int sourceDevice, cudaStream_t stream, Index_t indexOffset) const{

        //assert(resultPitch >= sizeOfElement);
        assert(resultPitch % sizeof(Value_t) == 0);

        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        auto it = std::find(deviceIds.begin(), deviceIds.end(), sourceDevice);
        assert(it != deviceIds.end());

        const int gpu = std::distance(deviceIds.begin(), it);
        //size_t sizeOfElement_ = sizeOfElement;
        const size_t numCols = numColumns;
        const size_t resultPitchValueTs = resultPitch / sizeof(Value_t);

        const Value_t* const gpuData = dataPtrPerLocation[gpu];

        wrapperCudaSetDevice(resultDevice); CUERR;

        dim3 block(256,1,1);
        dim3 grid(std::min(320ul, SDIV(maxNumIndices * numCols, block.x)),1,1);

        // generic_kernel<<<grid, block, 0, stream>>>([=] __device__ (){
        //     for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; i < nIndices * numCols; i += size_t(blockDim.x) * gridDim.x){
        //         const Index_t outputrow = i / numCols;
        //         const Index_t inputrow = d_indices[outputrow] + indexOffset;
        //         const Index_t col = i % numCols;
        //         d_result[size_t(outputrow) * resultPitchValueTs + col] 
        //                 = gpuData[size_t(inputrow) * numCols + col];
        //     }
        // }); CUERR;

        distrArrayGatherKernel2<<<grid, block, 0, stream>>>(
            d_result,
            gpuData,
            d_indices,
            d_numIndicesPtr,
            indexOffset,
            resultPitchValueTs,
            numCols
        ); CUERR;

        wrapperCudaSetDevice(oldDevice); CUERR;
    }

    std::vector<Index_t> getPartitions() const{
        return elementsPerLocation;
    }

    void writeHostPartitionToStream(std::ofstream& stream) const{
        const auto begin = dataPtrPerLocation[hostLocation];
        const std::size_t size = elementsPerLocation[hostLocation] * sizeOfElement;
        stream.write(reinterpret_cast<const char*>(begin), size);
    }

    void readHostPartitionFromStream(std::ifstream& stream) const{
        auto begin = dataPtrPerLocation[hostLocation];
        const std::size_t size = elementsPerLocation[hostLocation] * sizeOfElement;
        stream.read(reinterpret_cast<char*>(begin), size);
    }

    void writeGpuPartitionToStream(int partition, std::ofstream& stream) const{
        assert(0 <= partition);
        assert(partition < numGpus);

        constexpr std::int64_t MB = std::int64_t(1024) * 1024;
        constexpr std::int64_t safety = std::int64_t(64) * MB;
        constexpr std::int64_t maxBytes = std::int64_t(64) * MB;

        int currentId;
        cudaGetDevice(&currentId); CUERR;

        wrapperCudaSetDevice(deviceIds[partition]); CUERR;


        std::int64_t availableBytes = getAvailableMemoryInKB() * 1024;
        if(availableBytes > safety){
            availableBytes -= safety;
        }

        const std::int64_t bytesPerElement = sizeof(Value_t) * sizeOfElement;
        const std::int64_t buffersize = std::max(bytesPerElement, std::min(availableBytes, maxBytes));
        
        Value_t* buffer = nullptr;
        cudaMallocHost(&buffer, buffersize); CUERR;

        const std::int64_t batchsize = buffersize / bytesPerElement;
        const std::int64_t numBatches = SDIV(elementsPerLocation[partition], batchsize);
        for(std::int64_t batch = 0; batch < numBatches; batch++){
            std::int64_t begin = batch * batchsize;
            std::int64_t end = std::min(std::int64_t(elementsPerLocation[partition]), (batch + 1) * batchsize);
            const std::int64_t numElements = end-begin;

            const Value_t* src = offsetPtr(dataPtrPerLocation[partition], begin);
            //TIMERSTARTCPU(writeGpuPartitionToStream_memcpy);
            cudaMemcpy(buffer, src, sizeOfElement * numElements, D2H); CUERR;
            //TIMERSTOPCPU(writeGpuPartitionToStream_memcpy);

            //TIMERSTARTCPU(writeGpuPartitionToStream_file);
            stream.write(reinterpret_cast<const char*>(buffer), sizeOfElement * numElements);
            //TIMERSTOPCPU(writeGpuPartitionToStream_file);
        }

        cudaFreeHost(buffer); CUERR;

        wrapperCudaSetDevice(currentId); CUERR;
    }

    void readGpuPartitionFromStream(int partition, std::ifstream& stream){
        constexpr std::int64_t MB = std::int64_t(1024) * 1024;
        constexpr std::int64_t safety = std::int64_t(64) * MB;
        constexpr std::int64_t maxBytes = std::int64_t(64) * MB;

        std::int64_t availableBytes = getAvailableMemoryInKB() * 1024;
        if(availableBytes > safety){
            availableBytes -= safety;
        }

        const std::int64_t bytesPerElement = sizeof(Value_t) * sizeOfElement;
        const std::int64_t doublebuffersize = std::max(bytesPerElement, std::min(availableBytes, 2*maxBytes));
        const std::int64_t buffersize = doublebuffersize / 2;

        assert(buffersize > 4);

        std::array<Value_t*, 2> buffers;
        cudaMallocHost(&buffers[0], buffersize); CUERR;
        cudaMallocHost(&buffers[1], buffersize); CUERR;
        
        int currentId;
        cudaGetDevice(&currentId); CUERR;
        wrapperCudaSetDevice(deviceIds[partition]); CUERR;

        std::array<cudaStream_t, 2> streams;
        cudaStreamCreate(&streams[0]); CUERR;
        cudaStreamCreate(&streams[1]); CUERR;

        const std::int64_t batchsize = buffersize / bytesPerElement;
        const std::int64_t numBatches = SDIV(elementsPerLocation[partition], batchsize);

        int bufferindex = 0;

        for(std::int64_t batch = 0; batch < numBatches; batch++){
            std::int64_t begin = batch * batchsize;
            std::int64_t end = std::min(std::int64_t(elementsPerLocation[partition]), (batch + 1) * batchsize);
            const std::int64_t numElements = end-begin;

            cudaStreamSynchronize(streams[bufferindex]); CUERR;

            //TIMERSTARTCPU(readGpuPartitionFromStream_file);
            stream.read(reinterpret_cast<char*>(buffers[bufferindex]), sizeOfElement * numElements);
            //TIMERSTOPCPU(readGpuPartitionFromStream_file);

            
            Value_t* dest = offsetPtr(dataPtrPerLocation[partition], begin);
            //TIMERSTARTCPU(readGpuPartitionFromStream_memcpy);
            cudaMemcpyAsync(dest, buffers[bufferindex], sizeOfElement * numElements, H2D, streams[bufferindex]); CUERR; 
            //TIMERSTOPCPU(readGpuPartitionFromStream_memcpy);       

            bufferindex = bufferindex == 0 ? 1 : 0;        
        }

        cudaStreamSynchronize(streams[0]); CUERR;
        cudaStreamSynchronize(streams[1]); CUERR;

        cudaStreamDestroy(streams[0]); CUERR;
        cudaStreamDestroy(streams[1]); CUERR;

        cudaFreeHost(buffers[0]); CUERR;
        cudaFreeHost(buffers[1]); CUERR;

        wrapperCudaSetDevice(currentId); CUERR;
    }

    std::vector<char> writeGpuPartitionToMemory(int partition) const{
        assert(0 <= partition);
        assert(partition < numGpus);

        int currentId;
        cudaGetDevice(&currentId); CUERR;

        wrapperCudaSetDevice(deviceIds[partition]); CUERR;

        std::size_t bytes = getPartitionSizeInBytes(partition);
        std::vector<char> vec(bytes);

        //TIMERSTARTCPU(writeGpuPartitionToMemory_memcpy);
        cudaMemcpy(vec.data(), dataPtrPerLocation[partition], bytes, D2H); CUERR;
        //TIMERSTOPCPU(writeGpuPartitionToMemory_memcpy);

        wrapperCudaSetDevice(currentId); CUERR;

        return vec;
    }

    void readGpuPartitionFromMemory(int partition, const std::vector<char>& savedpartition){

        int currentId;
        cudaGetDevice(&currentId); CUERR;

        wrapperCudaSetDevice(deviceIds[partition]); CUERR;

        std::size_t bytes = savedpartition.size();
        //TIMERSTARTCPU(readGpuPartitionFromMemory_memcpy);
        cudaMemcpy(dataPtrPerLocation[partition], savedpartition.data(), bytes, H2D); CUERR;
        //TIMERSTOPCPU(readGpuPartitionFromMemory_memcpy);

        wrapperCudaSetDevice(currentId); CUERR;
    }

    void writeGpuPartitionsToStream(std::ofstream& stream) const{
        for(int gpu = 0; gpu < numGpus; gpu++){
            writeGpuPartitionToStream(gpu, stream);
        }
    }

    void readGpuPartitionsFromStream(std::ifstream& stream){
        for(int gpu = 0; gpu < numGpus; gpu++){
            readGpuPartitionFromStream(gpu, stream);
        }
    }

    std::vector<std::vector<char>> writeGpuPartitionsToMemory() const{
        std::vector<std::vector<char>> gpuPartitionsOnHost;
        gpuPartitionsOnHost.reserve(numGpus);

        for(int gpu = 0; gpu < numGpus; gpu++){
            auto vec = writeGpuPartitionToMemory(gpu);
            gpuPartitionsOnHost.emplace_back(std::move(vec));
        }

        return gpuPartitionsOnHost;
    }

    void readGpuPartitionsFromMemory(const std::vector<std::vector<char>>& savedpartitions){
        assert(savedpartitions.size() == numGpus);

        for(int gpu = 0; gpu < numGpus; gpu++){
            readGpuPartitionFromMemory(gpu, savedpartitions[gpu]);
        }
    }

    void writeToStream(std::ofstream& stream) const{
        const size_t totalMemory = numRows * sizeOfElement;
        stream.write(reinterpret_cast<const char*>(&totalMemory), sizeof(size_t));

        writeGpuPartitionsToStream(stream);
        writeHostPartitionToStream(stream);
    }

    void readFromStream(std::ifstream& stream){
        size_t totalMemory = 1;
        stream.read(reinterpret_cast<char*>(&totalMemory), sizeof(size_t));
        
        readGpuPartitionsFromStream(stream);
        readHostPartitionFromStream(stream);
    }

    std::size_t getPartitionSizeInBytes(int partition) const{
        assert(0 <= partition);
        assert(partition < numGpus + 1);

        return elementsPerLocation[partition] * sizeOfElement;
    }

    bool isEnoughMemoryForGpuPartitions(std::size_t availablehostbytes) const{
        for(int gpu = 0; gpu < numGpus; gpu++){
            const std::size_t gpubytes = getPartitionSizeInBytes(gpu);
            if(availablehostbytes < gpubytes){
                return false;
            }else{
                availablehostbytes -= gpubytes;
            }
        }
        return true;
    }

    void deallocateGpuPartition(int partition){
        assert(0 <= partition);
        assert(partition < numGpus);

        int currentId;
        cudaGetDevice(&currentId); CUERR;

        wrapperCudaSetDevice(deviceIds[partition]); CUERR;
        cudaFree(dataPtrPerLocation[partition]); CUERR;

        wrapperCudaSetDevice(currentId); CUERR;
    }

    void allocateGpuPartition(int partition){
        assert(0 <= partition);
        assert(partition < numGpus);

        int currentId;
        cudaGetDevice(&currentId); CUERR;

        wrapperCudaSetDevice(deviceIds[partition]); CUERR;
        cudaMalloc(&dataPtrPerLocation[partition], sizeOfElement * elementsPerLocation[partition]); CUERR;

        wrapperCudaSetDevice(currentId); CUERR;
    }

    void deallocateGpuPartitions(){
        for(int gpu = 0; gpu < numGpus; gpu++){
            deallocateGpuPartition(gpu);
        }
    }

    void allocateGpuPartitions(){
        for(int gpu = 0; gpu < numGpus; gpu++){
            allocateGpuPartition(gpu);
        }
    }

private:
    void destroy(){
        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        if(dataPtrPerLocation.size() > 0){

            for(int gpu = 0; gpu < numGpus; gpu++){
                wrapperCudaSetDevice(deviceIds[gpu]); CUERR;
                if(debug) std::cerr << "DistributedArray::destroy device " << deviceIds[gpu] << " cudaFree(" << static_cast<void*>(dataPtrPerLocation[gpu]) << ")\n";
                cudaFree(dataPtrPerLocation[gpu]); CUERR;
            }

            if(debug) std::cerr << "DistributedArray::destroy delete [](" << static_cast<void*>(dataPtrPerLocation[hostLocation]) << ")\n";
            delete [] dataPtrPerLocation[hostLocation];
        }

        wrapperCudaSetDevice(oldDevice); CUERR;
    }
};





#endif








#endif
