#ifndef DISTRIBUTED_ARRAY_HPP
#define DISTRIBUTED_ARRAY_HPP


#ifdef __NVCC__

#include <gpu/simpleallocation.cuh>
#include <gpu/utility_kernels.cuh>
#include <hpc_helpers.cuh>
#include <gpu/nvtxtimelinemarkers.hpp>
#include <gpu/peeraccess.hpp>
#include <threadpool.hpp>

#include <algorithm>
#include <numeric>
#include <cassert>
#include <iterator>
#include <vector>
#include <map>
#include <future>
#include <cassert>
#include <omp.h>

#include <mutex>
#include <condition_variable>





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
    };

    using GatherHandle = std::shared_ptr<GatherHandleStruct>;
    using PeerAccess_t = PeerAccessDebug;

    struct SinglePartitionInfo{
        bool isSinglePartition = false;
        int locationId = -1;
    };

    bool debug;
    int numGpus;
    int numLocations; //numGpus + 1
    int hostLocation; // numLocations - 1
    int preferedLocation;
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
        : DistributedArray({},{},0,0,-1){
    }

    DistributedArray(std::vector<int> deviceIds_, std::vector<size_t> memoryLimitBytesPerGPU_, Index_t numRows_, Index_t numCols_, int preferedLocation_ = -1)
                    : debug(false),
                    numGpus(deviceIds_.size()),
                    numLocations(numGpus+1),
                    hostLocation(numLocations-1),
                    preferedLocation(preferedLocation_),
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
                cudaSetDevice(deviceIds[preferedLocation]); CUERR;
                elementsPerLocation[preferedLocation] = numRows;
                cudaMalloc(&dataPtrPerLocation[preferedLocation], totalRequiredMemory); CUERR;
            }else{

                size_t remainingElements = numRows;

                for(int gpu = 0; gpu < numGpus && remainingElements > 0; gpu++){
                    cudaSetDevice(deviceIds[gpu]); CUERR;

                    size_t rows = std::min(remainingElements, memoryLimitBytesPerGPU[gpu] / sizeOfElement);
                    elementsPerLocation[gpu] = rows;
                    if(rows == 0){
                        continue;
                    }

                    cudaMalloc(&(dataPtrPerLocation[gpu]), rows * sizeOfElement); CUERR;

                    remainingElements -= rows;
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

            cudaSetDevice(oldId); CUERR;
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

    // template<class Ptr>
    // Ptr offsetPtr(Ptr valuetBasePointer, Index_t rowIndex){
    //     return (Ptr)(((char*)(valuetBasePointer)) + sizeOfElement * rowIndex);
    // }

    const Value_t* offsetPtr(const Value_t* valuetBasePointer, Index_t rowIndex) const {
        return (const Value_t*)(((const char*)(valuetBasePointer)) + sizeOfElement * rowIndex);
    }

    Value_t* offsetPtr(Value_t* valuetBasePointer, Index_t rowIndex) const {
        return (Value_t*)(((char*)(valuetBasePointer)) + sizeOfElement * rowIndex);
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
            cudaSetDevice(deviceIds[location]); CUERR;
            cudaMemcpy(destPtr, data, sizeOfElement, H2D); CUERR;
            cudaSetDevice(oldDevice); CUERR;
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
                    cudaSetDevice(deviceIds[location]); CUERR;
                    cudaMemcpy(destPtr, srcPtr, sizeOfElement * numHits, H2D); CUERR;
                }
            }
        }

        cudaSetDevice(oldDevice); CUERR;
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
                    cudaSetDevice(deviceIds[locationId]); CUERR;
                    cudaMemcpy(destPtr, data, indices.size() * sizeOfElement, H2D); CUERR;
                    cudaSetDevice(oldDevice); CUERR;
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
                        std::copy_n(srcPtr, numColumns * (to-from), destPtr);
                    }else{
                        cudaSetDevice(deviceIds[location]); CUERR;
                        cudaMemcpy(destPtr, srcPtr, sizeOfElement * (to-from), H2D); CUERR;
                    }

                    num = to;
                }
            }
        }

        cudaSetDevice(oldDevice); CUERR;
    }

    void get(Index_t index, Value_t* result){
        int location = getLocation(index);
        Index_t localIndex = index - elementsPerLocationPS[location];
        const Value_t* srcPtr = offsetPtr(dataPtrPerLocation[location], localIndex);

        if(location == hostLocation){
            std::copy_n(srcPtr, numColumns, result);
        }else{
            int oldDevice; cudaGetDevice(&oldDevice); CUERR;
            cudaSetDevice(deviceIds[location]); CUERR;
            cudaMemcpy(result, srcPtr, sizeOfElement, D2H); CUERR;
            cudaSetDevice(oldDevice); CUERR;
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
            cudaSetDevice(deviceIds[gpu]); CUERR;
            cudaStreamCreate(&(handle->streamsPerGpu[gpu])); CUERR;
            cudaEventCreate(&(handle->eventsPerGpu[gpu])); CUERR;
        }
        cudaEventCreate(&(handle->readyEvent)); CUERR;
        cudaSetDevice(oldDevice); CUERR;

        return handle;
    }

    void destroyGatherHandleStruct(const GatherHandle& handle) const{
        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        handle->pinnedLocalIndices = std::move(SimpleAllocationPinnedHost<Index_t>{});
        handle->pinnedResultData = std::move(SimpleAllocationPinnedHost<Value_t>{});

        for(size_t gpu = 0; gpu < handle->dataPerGpu.size(); gpu++){
            cudaSetDevice(deviceIds[gpu]); CUERR;

            handle->deviceLocalIndicesPerLocation[gpu] = std::move(SimpleAllocationDevice<Index_t>{});
            handle->dataPerGpu[gpu] = std::move(SimpleAllocationDevice<Value_t>{});
            cudaStreamDestroy(handle->streamsPerGpu[gpu]); CUERR;
            cudaEventDestroy(handle->eventsPerGpu[gpu]); CUERR;
        }

        cudaEventDestroy(handle->readyEvent); CUERR;

        for(auto& pair : handle->tmpResultsOfDevice){
            cudaSetDevice(pair.first); CUERR;
            pair.second = std::move(SimpleAllocationDevice<Value_t>{});
        }

        for(auto& pair : handle->permutationIndicesOfDevice){
            cudaSetDevice(pair.first); CUERR;
            pair.second = std::move(SimpleAllocationDevice<Index_t>{});
        }

        cudaSetDevice(oldDevice); CUERR;
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

                const int locationId = singlePartitionInfo.locationId;
                const int deviceId = deviceIds[locationId];
                cudaSetDevice(deviceId); CUERR;

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

        //gather from gpus to host
        for(int gpu = 0; gpu < numGpus; gpu++){
            Index_t numHits = hitsPerLocation[gpu];
            if(numHits > 0){
                const int deviceId = deviceIds[gpu];
                const cudaStream_t stream = handle->streamsPerGpu[gpu];

                cudaSetDevice(deviceIds[gpu]); CUERR;

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

                cudaMemcpyAsync(handle->pinnedResultData.get() + hitsPerLocationPrefixSum[gpu] * sizeOfElement,
                                myResult.get(),
                                sizeOfElement * numHits,
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

        //permute pinnedResultData and store in output array
        for(Index_t dstindex = 0; dstindex < numIds; dstindex++){
            const Index_t srcindex = handle->pinnedPermutationIndices[dstindex];
            const Value_t* srcPtr = offsetPtr(handle->pinnedResultData.get(), srcindex);
            Value_t* destPtr = (Value_t*)(((const char*)(result)) + resultPitch * dstindex);
            std::copy_n(srcPtr, numColumns, destPtr);
        }
    }

    void gatherElementsInGpuMem(const GatherHandle& handle,
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
        cudaSetDevice(deviceId); CUERR;
        cudaStreamCreate(&stream); CUERR;
        gatherElementsInGpuMemAsync(handle, indices, d_indices, numIds, deviceId, d_result, resultPitch, stream);
        cudaStreamSynchronize(stream); CUERR;
        cudaStreamDestroy(stream); CUERR;
        cudaSetDevice(oldDevice); CUERR;
    }

    //the same GatherHandleStruct must not be used in another call until the results of the previous call are calculated
    void gatherElementsInGpuMemAsync(const GatherHandle& handle,
                                    const Index_t* indices,
                                    const Index_t* d_indices,
                                    Index_t numIds,
                                    int resultDeviceId,
                                    Value_t* d_result,
                                    size_t resultPitch, // result element i begins at offset i * resultPitch
                                    cudaStream_t stream) const{
        if(numIds == 0) return;

        const int numCpuThreads = care::threadpool.getConcurrency();

        //fastpath, if all elements of distributed array reside on the gpu with device id deviceId
        if(singlePartitionInfo.isSinglePartition){
            if(singlePartitionInfo.locationId == hostLocation){
                if(debug) std::cerr << "single location array fasthpath on host\n";

                auto& h_result = handle->pinnedResultData;
                h_result.resize(numIds * numColumns);

                care::threadpool.parallelFor(
                    Index_t(0), 
                    numIds, 
                    [&](Index_t begin, Index_t end, int threadId){
                        for(Index_t k = begin; k < end; k++){
                            const Index_t localId = indices[k];
                            const Value_t* srcPtr = offsetPtr(dataPtrPerLocation[hostLocation], localId);
                            Value_t* destPtr = offsetPtr(h_result.get(), k);
                            std::copy_n(srcPtr, numColumns, destPtr);
                        }
                    },
                    numCpuThreads
                );

                int oldDevice; cudaGetDevice(&oldDevice); CUERR;

                cudaSetDevice(resultDeviceId); CUERR;

                cudaMemcpyAsync(d_result, h_result.get(), sizeof(Value_t) * numIds * numColumns, H2D, stream); CUERR;

                cudaSetDevice(oldDevice); CUERR;

                return;
            }else{
                if(debug) std::cerr << "single location array fasthpath on partition " << singlePartitionInfo.locationId << "\n";

                const int locationId = singlePartitionInfo.locationId;
                const int deviceId = deviceIds[locationId];

                copyDataToGpuBufferAsync(d_result, resultPitch, d_indices, numIds, deviceId, stream, -elementsPerLocationPS[locationId]);

                return;
            }
        }


//TIMERSTARTCPU(countHitsPerLocation);
        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        auto deviceIdIter = std::find(deviceIds.begin(), deviceIds.end(), resultDeviceId);
        int deviceIdLocation = -1;
        if(deviceIdIter != deviceIds.end()){
            deviceIdLocation = std::distance(deviceIds.begin(), deviceIdIter);
        }

        const int threadlocoffset = SDIV(numLocations,32) * 32;
        std::vector<Index_t> hitsPerLocation(numLocations, 0);
        std::vector<Index_t> hitsPerLocationPerThread(threadlocoffset * numCpuThreads, 0);

        care::threadpool.parallelFor(
            Index_t(0), 
            numIds, 
            [&](Index_t begin, Index_t end, int threadId){                
                Index_t* hitsptr = hitsPerLocationPerThread.data() + threadlocoffset * threadId;
                for(Index_t i = begin; i < end; i++){
                    int location = getLocation(indices[i]);
                    hitsptr[location]++;
                }
            },
            numCpuThreads
        );

        for(int k = 0; k < numCpuThreads; k++){
            for(int l = 0; l < numLocations; l++){
                hitsPerLocation[l] += hitsPerLocationPerThread[threadlocoffset * k + l];
            }
        }
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

        std::vector<Index_t> permutationIndices(numIds);

        handle->pinnedLocalIndices.resize(numIds);
        handle->pinnedPermutationIndices.resize(numIds);
        for(Index_t i = 0; i < numIds; i++){
    		int location = getLocation(indices[i]);
            Index_t localIndex = indices[i] - elementsPerLocationPS[location];
            Index_t tmpresultindex = hitsPerLocationPrefixSum[location] + hitsPerLocation[location];
            handle->pinnedPermutationIndices[i] = tmpresultindex;

            handle->pinnedLocalIndices[tmpresultindex] = localIndex;
            //std::cerr << "read id " << readIds[i] << ", location " << location << ", localReadId " << localReadId << ", index " << index << "\n";
    		hitsPerLocation[location]++;
    	}
//TIMERSTOPCPU(makenewindices);

//TIMERSTARTCPU(resizedevicevectors)
        handle->pinnedResultData.resize(numIds * numColumns);

        cudaSetDevice(resultDeviceId); CUERR;
        handle->localIndicesOnDevice[resultDeviceId].resize(numIds);
        cudaMemcpyAsync(handle->localIndicesOnDevice[resultDeviceId].get(),
                        handle->pinnedLocalIndices.get(),
                        sizeof(Index_t) * numIds,
                        H2D,
                        stream); CUERR;

        handle->tmpResultsOfDevice[resultDeviceId].resize(numIds * numColumns);

        handle->permutationIndicesOfDevice[resultDeviceId].resize(numIds);

        cudaMemcpyAsync(handle->permutationIndicesOfDevice[resultDeviceId],
                        handle->pinnedPermutationIndices.get(),
                        sizeof(Index_t) * numIds,
                        H2D,
                        stream); CUERR;
//TIMERSTOPCPU(resizedevicevectors);

        //gather from gpus to host
        for(int gpu = 0; gpu < numGpus; gpu++){
            Index_t numHits = hitsPerLocation[gpu];
            if(numHits > 0){
                int mydeviceId = deviceIds[gpu];
                cudaStream_t mystream = handle->streamsPerGpu[gpu];
                cudaEvent_t myevent = handle->eventsPerGpu[gpu];

                if(true && (peerAccess.canAccessPeer(resultDeviceId, mydeviceId) || resultDeviceId == mydeviceId)){
                    if(debug) std::cerr << "use peer access / local access: " << resultDeviceId << " <---- " << mydeviceId << "\n";

                    cudaSetDevice(resultDeviceId); CUERR;

                    Value_t* destptr = offsetPtr(handle->tmpResultsOfDevice[resultDeviceId].get(), hitsPerLocationPrefixSum[gpu]);
                    Index_t* localIdsPtr = handle->localIndicesOnDevice[resultDeviceId].get() + hitsPerLocationPrefixSum[gpu];

                    copyDataToGpuBufferAsync(destptr, sizeOfElement, resultDeviceId, localIdsPtr, numHits, mydeviceId, stream, 0);
                }else{
                    if(debug) std::cerr << "use intermediate host: " << resultDeviceId << " <---- host <---- " << mydeviceId << "\n";


            	    cudaSetDevice(mydeviceId); CUERR;

                    auto& myLocalIds = handle->deviceLocalIndicesPerLocation[gpu];
                    auto& myResult = handle->dataPerGpu[gpu];


            	    myLocalIds.resize(numHits);
                    cudaMemcpyAsync(myLocalIds.get(),
                                    handle->pinnedLocalIndices.get() + hitsPerLocationPrefixSum[gpu],
                                    sizeof(Index_t) * numHits,
                                    H2D,
                                    mystream); CUERR;

            	    myResult.resize(numHits * numColumns);

                    copyDataToGpuBufferAsync(myResult.get(), sizeOfElement, mydeviceId, myLocalIds.get(), numHits, mydeviceId, mystream, 0);

                    Value_t* const myPinnedResults = offsetPtr(handle->pinnedResultData.get(), hitsPerLocationPrefixSum[gpu]);
                    cudaMemcpyAsync(myPinnedResults,
                                    myResult.get(),
                                    sizeOfElement * numHits,
                                    D2H,
                                    mystream); CUERR;

                    cudaEventRecord(myevent, mystream); CUERR;

                    cudaSetDevice(resultDeviceId); CUERR;
                    cudaStreamWaitEvent(stream, myevent,0); CUERR; //wait in result stream until partial results are on the host.

                    //copy partial results to tmp result buffer on device
                    Value_t* mytmpResultsOfDevice = offsetPtr(handle->tmpResultsOfDevice[resultDeviceId].get(), hitsPerLocationPrefixSum[gpu]);
                    cudaMemcpyAsync(mytmpResultsOfDevice,
                                    myPinnedResults,
                                    sizeOfElement * hitsPerLocation[gpu],
                                    H2D,
                                    stream); CUERR;
                }
            }
    	}

        //gather from host to host
        if(hitsPerLocation[hostLocation] > 0){
            const Index_t numHits = hitsPerLocation[hostLocation];
            const auto hitsOffset = hitsPerLocationPrefixSum[hostLocation];
            const Index_t* hostLocalIds = handle->pinnedLocalIndices.get() + hitsOffset;            

            care::threadpool.parallelFor(
                Index_t(0), 
                numHits, 
                [&](Index_t begin, Index_t end, int threadId){
                    for(Index_t k = begin; k < end; k++){
                        const Index_t localId = hostLocalIds[k];
                        const Value_t* srcPtr = offsetPtr(dataPtrPerLocation[hostLocation], localId);
                        Value_t* destPtr = offsetPtr(handle->pinnedResultData.get(), hitsOffset + k);
                        std::copy_n(srcPtr, numColumns, destPtr);
                    }
                },
                numCpuThreads
            );
        }

        cudaSetDevice(resultDeviceId); CUERR;

        cudaMemcpyAsync(offsetPtr(handle->tmpResultsOfDevice[resultDeviceId].get(), hitsPerLocationPrefixSum[hostLocation]),
                        offsetPtr(handle->pinnedResultData.get(), hitsPerLocationPrefixSum[hostLocation]),
                        hitsPerLocation[hostLocation] * sizeOfElement,
                        H2D,
                        stream); CUERR;

        {
            assert(resultPitch % sizeof(Value_t) == 0);

            size_t resultPitchValueTs = resultPitch / sizeof(Value_t);
            size_t numCols = numColumns;
            const Index_t* indices = handle->permutationIndicesOfDevice[resultDeviceId].get();
            const Value_t* src = handle->tmpResultsOfDevice[resultDeviceId].get();
            Value_t* dest = d_result;
            Index_t n = numIds;

            dim3 block(256,1,1);
            dim3 grid(std::min(65535ul, SDIV(n * numCols, block.x)),1,1);

            generic_kernel<<<grid, block, 0, stream>>>([=] __device__ (){
                for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; i < n * numCols; i += size_t(blockDim.x) * gridDim.x){
                    const Index_t outputrow = i / numCols;
                    const Index_t inputrow = indices[outputrow];
                    const Index_t col = i % numCols;
                    dest[outputrow * resultPitchValueTs + col] = src[inputrow * numCols + col];
                }
            }); CUERR;

        }

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

        cudaSetDevice(oldDevice); CUERR;
    }

    //d_result, d_indices must point to memory of device deviceId. d_indices[i] + indexOffset must be a local element index for this device
    void copyDataToGpuBufferAsync(Value_t* d_result, size_t resultPitch, const Index_t* d_indices, Index_t nIndices, int deviceId, cudaStream_t stream, Index_t indexOffset) const{
        // assert(resultPitch >= sizeOfElement);
        assert(resultPitch % sizeof(Value_t) == 0);

        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        cudaSetDevice(deviceId); CUERR;

        auto it = std::find(deviceIds.begin(), deviceIds.end(), deviceId);
        assert(it != deviceIds.end());

        int gpu = std::distance(deviceIds.begin(), it);
        //size_t sizeOfElement_ = sizeOfElement;
        size_t numCols = numColumns;
        size_t resultPitchValueTs = resultPitch / sizeof(Value_t);

        const Value_t* const gpuData = dataPtrPerLocation[gpu];

        dim3 block(256,1,1);
        dim3 grid(std::min(65535ul, SDIV(nIndices * numCols, block.x)),1,1);

        generic_kernel<<<grid, block, 0, stream>>>([=] __device__ (){
            for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; i < nIndices * numCols; i += size_t(blockDim.x) * gridDim.x){
                const Index_t outputrow = i / numCols;
                const Index_t inputrow = d_indices[outputrow] + indexOffset;
                const Index_t col = i % numCols;
                d_result[outputrow * resultPitchValueTs + col] = gpuData[inputrow * numCols + col];
            }
        }); CUERR;

        cudaSetDevice(oldDevice); CUERR;
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

        cudaSetDevice(resultDevice); CUERR;

        dim3 block(256,1,1);
        dim3 grid(std::min(65535ul, SDIV(nIndices * numCols, block.x)),1,1);

        generic_kernel<<<grid, block, 0, stream>>>([=] __device__ (){
            for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; i < nIndices * numCols; i += size_t(blockDim.x) * gridDim.x){
                const Index_t outputrow = i / numCols;
                const Index_t inputrow = d_indices[outputrow] + indexOffset;
                const Index_t col = i % numCols;
                d_result[outputrow * resultPitchValueTs + col] = gpuData[inputrow * numCols + col];
            }
        }); CUERR;

        cudaSetDevice(oldDevice); CUERR;
    }

    //d_result, d_indices must point to memory of device deviceId. d_indices[i] must be a local element index for this device
    void copyDataToGpuBufferAsync(Value_t* d_result, size_t resultPitch, const Index_t* d_indices, Index_t nIndices, int deviceId, cudaStream_t stream) const{
        copyDataToGpuBufferAsync(d_result, resultPitch, d_indices, nIndices, deviceId, stream, 0);
    }

    std::vector<Index_t> getPartitions() const{
        return elementsPerLocation;
    }

private:
    void destroy(){
        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        if(dataPtrPerLocation.size() > 0){

            for(int gpu = 0; gpu < numGpus; gpu++){
                cudaSetDevice(deviceIds[gpu]); CUERR;
                if(debug) std::cerr << "DistributedArray::destroy device " << deviceIds[gpu] << " cudaFree(" << static_cast<void*>(dataPtrPerLocation[gpu]) << ")\n";
                cudaFree(dataPtrPerLocation[gpu]); CUERR;
            }

            if(debug) std::cerr << "DistributedArray::destroy delete [](" << static_cast<void*>(dataPtrPerLocation[hostLocation]) << ")\n";
            delete [] dataPtrPerLocation[hostLocation];
        }

        cudaSetDevice(oldDevice); CUERR;
    }

    template<class Func>
    void maybeAsync(Func&& f){
        #if 0
            f();
        #else
            care::threadpool.enqueue(std::forward<Func>(f));
        #endif
    }

};





#endif








#endif
