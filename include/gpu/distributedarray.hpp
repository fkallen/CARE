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
        std::vector<SimpleAllocationDevice<Value_t>> dataPerGpu;

        std::map<int, SimpleAllocationDevice<Value_t>> tmpResultsOfDevice;
        std::map<int, SimpleAllocationDevice<Index_t>> permutationIndicesOfDevice;
        std::map<int, SimpleAllocationDevice<Index_t>> localIndicesOnDevice;

        std::vector<cudaStream_t> streamsPerGpu;
        std::vector<cudaEvent_t> eventsPerGpu;

        cudaEvent_t readyEvent;
        care::ThreadPool::ParallelForHandle pforHandle;
    };

    using GatherHandle = std::shared_ptr<GatherHandleStruct>;
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

                    if(numRows > totalPossibleElementsPerGpu){
                        remainingElements = numRows - totalPossibleElementsPerGpu;
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
        handle->dataPerGpu.resize(numGpus);
        handle->streamsPerGpu.resize(numGpus);
        handle->eventsPerGpu.resize(numGpus);
        int oldDevice; cudaGetDevice(&oldDevice); CUERR;
        for(int gpu = 0; gpu < numGpus; gpu++){
            wrapperCudaSetDevice(deviceIds[gpu]); CUERR;
            cudaStreamCreate(&(handle->streamsPerGpu[gpu])); CUERR;
            cudaEventCreate(&(handle->eventsPerGpu[gpu])); CUERR;
        }
        cudaEventCreate(&(handle->readyEvent)); CUERR;
        wrapperCudaSetDevice(oldDevice); CUERR;

        return handle;
    }

    void destroyGatherHandleStruct(const GatherHandle& handle) const{
        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        handle->pinnedLocalIndices = std::move(SimpleAllocationPinnedHost<Index_t>{});
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
                copyDataToGpuBufferAsync(d_result.get(), sizeOfElement, d_indices, numIds, deviceId, stream, -elementsPerLocationPS[locationId]); CUERR;
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

                copyDataToGpuBufferAsync(myResult.get(), sizeOfElement, myLocalIds.get(), numHits, deviceId, stream);

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
                                    cudaStream_t stream) const{
        if(numIds == 0) return;

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
                     copyDataToGpuBufferAsync(d_result, resultPitch, d_indices, numIds, mydeviceId, stream, -elementsPerLocationPS[locationId]);

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
                copyDataToGpuBufferAsync(localGatherPtr, resultPitch, localIndicesPtr, numIds, mydeviceId, localGatherStream, -elementsPerLocationPS[locationId]);

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
        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        auto deviceIdIter = std::find(deviceIds.begin(), deviceIds.end(), resultDeviceId);
        int deviceIdLocation = -1;
        if(deviceIdIter != deviceIds.end()){
            deviceIdLocation = std::distance(deviceIds.begin(), deviceIdIter);
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
            copyDataToGpuBufferAsync(d_result, resultPitch, d_indices, numIds, resultDeviceId, stream, -elementsPerLocationPS[deviceIdLocation]);
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
        //
        // std::cerr << "permutationIndices: ";
        // std::copy(handle->pinnedPermutationIndices.get(), handle->pinnedPermutationIndices.get() + numIds, std::ostream_iterator<int>(std::cerr, " "));
        // std::cerr << "\n";
        //
        // std::vector<int> hostResultPermuted(numIds);
        // for(int i = 0; i < numIds; i++){
        //     hostResultPermuted[i] = handle->pinnedResultData[permutationIndices[i]];
        // }
        //
        // std::cerr << "pinnedLocalIndices: ";
        // std::copy(handle->pinnedLocalIndices.get(), handle->pinnedLocalIndices.get() + hitsPerLocationPrefixSum.back(), std::ostream_iterator<int>(std::cerr, " "));
        // std::cerr << "\n";
        //
        // std::cerr << "pinnedResultData: ";
        // std::copy(handle->pinnedResultData.get(), handle->pinnedResultData.get() + hitsPerLocationPrefixSum.back(), std::ostream_iterator<int>(std::cerr, " "));
        // std::cerr << "\n";
        //
        // std::cerr << "permutedResultData: ";
        // std::copy(hostResultPermuted.begin(), hostResultPermuted.end(), std::ostream_iterator<int>(std::cerr, " "));
        // std::cerr << "\n";

        wrapperCudaSetDevice(oldDevice); CUERR;
    }

    //d_result, d_indices must point to memory of device deviceId. d_indices[i] + indexOffset must be a local element index for this device
    void copyDataToGpuBufferAsync(Value_t* d_result, size_t resultPitch, const Index_t* d_indices, Index_t nIndices, int deviceId, cudaStream_t stream, Index_t indexOffset) const{
        // assert(resultPitch >= sizeOfElement);
        assert(resultPitch % sizeof(Value_t) == 0);

        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        wrapperCudaSetDevice(deviceId); CUERR;

        auto it = std::find(deviceIds.begin(), deviceIds.end(), deviceId);
        assert(it != deviceIds.end());

        int gpu = std::distance(deviceIds.begin(), it);
        //size_t sizeOfElement_ = sizeOfElement;
        size_t numCols = numColumns;
        size_t resultPitchValueTs = resultPitch / sizeof(Value_t);

        const Value_t* const gpuData = dataPtrPerLocation[gpu];

        dim3 block(256,1,1);
        dim3 grid(std::min(320ul, SDIV(nIndices * numCols, block.x)),1,1);

        generic_kernel<<<grid, block, 0, stream>>>([=] __device__ (){
            for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; i < nIndices * numCols; i += size_t(blockDim.x) * gridDim.x){
                const Index_t outputrow = i / numCols;
                const Index_t inputrow = d_indices[outputrow] + indexOffset;
                const Index_t col = i % numCols;

                d_result[size_t(outputrow) * resultPitchValueTs + col] 
                        = gpuData[size_t(inputrow) * numCols + col];
            }
        }); CUERR;

        wrapperCudaSetDevice(oldDevice); CUERR;
    }

    //d_result points to memory of resultDevice, d_indices points to memory of sourceDevice. Gathers array elements of sourceDevice to resultDevice
    // if resultDevice != sourceDevice, peer access must be enabled
    void copyDataToGpuBufferAsync(Value_t* d_result, size_t resultPitch, int resultDevice, const Index_t* d_indices, Index_t nIndices, int sourceDevice, cudaStream_t stream, Index_t indexOffset) const{
        //assert(resultPitch >= sizeOfElement);
        assert(resultPitch % sizeof(Value_t) == 0);

        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        auto it = std::find(deviceIds.begin(), deviceIds.end(), sourceDevice);
        assert(it != deviceIds.end());

        int gpu = std::distance(deviceIds.begin(), it);
        //size_t sizeOfElement_ = sizeOfElement;
        size_t numCols = numColumns;
        size_t resultPitchValueTs = resultPitch / sizeof(Value_t);

        const Value_t* const gpuData = dataPtrPerLocation[gpu];

        wrapperCudaSetDevice(resultDevice); CUERR;

        dim3 block(256,1,1);
        dim3 grid(std::min(320ul, SDIV(nIndices * numCols, block.x)),1,1);

        generic_kernel<<<grid, block, 0, stream>>>([=] __device__ (){
            for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; i < nIndices * numCols; i += size_t(blockDim.x) * gridDim.x){
                const Index_t outputrow = i / numCols;
                const Index_t inputrow = d_indices[outputrow] + indexOffset;
                const Index_t col = i % numCols;
                d_result[size_t(outputrow) * resultPitchValueTs + col] 
                        = gpuData[size_t(inputrow) * numCols + col];
            }
        }); CUERR;

        wrapperCudaSetDevice(oldDevice); CUERR;
    }

    //d_result, d_indices must point to memory of device deviceId. d_indices[i] must be a local element index for this device
    void copyDataToGpuBufferAsync(Value_t* d_result, size_t resultPitch, const Index_t* d_indices, Index_t nIndices, int deviceId, cudaStream_t stream) const{
        copyDataToGpuBufferAsync(d_result, resultPitch, d_indices, nIndices, deviceId, stream, 0);
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
        const size_t totalMemory = numRows * numColumns * sizeOfElement;
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
