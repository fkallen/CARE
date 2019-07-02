#include <gpu/distributedarray.hpp>
#include <gpu/simpleallocation.cuh>
#include <gpu/utility_kernels.cuh>
#include <hpc_helpers.cuh>

#include <vector>
#include <map>
#include <future>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <iterator>


    DistributedArray::DistributedArray(std::vector<int> deviceIds_, std::vector<float> maxFreeMemFraction_, size_t numElements_, size_t sizeOfElement_, int preferedLocation_)
            : debug(true),
            numGpus(deviceIds_.size()),
            numLocations(numGpus+1),
            hostLocation(numLocations-1),
            preferedLocation(preferedLocation_),
            numElements(numElements_),
            sizeOfElement(sizeOfElement_),
            deviceIds(std::move(deviceIds_)),
            maxFreeMemFraction(std::move(maxFreeMemFraction_)),
            peerAccess(PeerAccess{}){

        assert(deviceIds.size() == maxFreeMemFraction.size());
        assert(std::all_of(maxFreeMemFraction.begin(), maxFreeMemFraction.end(), [](float f){return f <= 1.0f;}));
        assert(std::all_of(maxFreeMemFraction.begin(), maxFreeMemFraction.end(), [](float f){return f >= 0.0f;}));

        elementsPerLocation.resize(numLocations, 0);
        elementsPerLocationPS.resize(numLocations+1, 0);
        dataPtrPerLocation.resize(numLocations, nullptr);

        std::vector<size_t> freeMemPerGpu(numGpus);

        int oldId; cudaGetDevice(&oldId); CUERR;

        for(int gpu = 0; gpu < numGpus; gpu++){
            cudaSetDevice(deviceIds[gpu]); CUERR;

            size_t total;
            cudaMemGetInfo(&freeMemPerGpu[gpu], &total); CUERR;
        }

        size_t totalRequiredMemory = numElements * sizeOfElement;

        bool preferedLocationIsSufficient = false;

        if(preferedLocation != -1 && preferedLocation != hostLocation){
            if(freeMemPerGpu[preferedLocation] * maxFreeMemFraction[preferedLocation] <= totalRequiredMemory){
                preferedLocationIsSufficient = true;
            }
        }

        if(preferedLocationIsSufficient){
            cudaSetDevice(deviceIds[preferedLocation]); CUERR;
            elementsPerLocation[preferedLocation] = numElements;
            cudaMalloc(&dataPtrPerLocation[preferedLocation], totalRequiredMemory); CUERR;
        }else{
            std::vector<size_t> freeMemPerGpuTreshold(freeMemPerGpu.size());
            size_t remainingElements = numElements;

            for(int gpu = 0; gpu < numGpus && remainingElements > 0; gpu++){
                cudaSetDevice(deviceIds[gpu]); CUERR;
                freeMemPerGpuTreshold[gpu] = freeMemPerGpu[gpu] * maxFreeMemFraction[gpu];
                size_t elements = std::min(remainingElements, freeMemPerGpuTreshold[gpu] / sizeOfElement);
                elementsPerLocation[gpu] = elements;
                cudaMalloc(&dataPtrPerLocation[gpu], elements * sizeOfElement); CUERR;

                remainingElements -= elements;
            }

            //remaining elements are stored in host memory
            if(remainingElements > 0){
                dataPtrPerLocation[hostLocation] = new char[remainingElements * sizeOfElement];
                elementsPerLocation[hostLocation] = remainingElements;
            }

        }


        std::partial_sum(elementsPerLocation.begin(), elementsPerLocation.end(), elementsPerLocationPS.begin()+1);

        if(debug){
             std::cerr << "device ids: [";
             std::copy(deviceIds.begin(), deviceIds.end(), std::ostream_iterator<int>(std::cerr, " "));
             std::cerr << "]\n";

             std::cerr << "elements per location: [";
             std::copy(elementsPerLocation.begin(), elementsPerLocation.end(), std::ostream_iterator<size_t>(std::cerr, " "));
             std::cerr << "]\n";
        }

        cudaSetDevice(oldId); CUERR;
    }

    DistributedArray::~DistributedArray(){
        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        for(int gpu = 0; gpu < numGpus; gpu++){
            cudaSetDevice(deviceIds[gpu]); CUERR;
            cudaFree(dataPtrPerLocation[gpu]); CUERR;
        }

        delete [] dataPtrPerLocation[hostLocation];

        cudaSetDevice(oldDevice); CUERR;
    }

    int DistributedArray::getLocation(size_t index) const{
    	int location = 0;
    	for(; location < numLocations; location++){
    		if(index < elementsPerLocationPS[location+1])
                        break;
    	}
        assert(location < numLocations);
    	return location;
    }

    void DistributedArray::set(size_t index, const char* data){
        int location = getLocation(index);
        size_t localIndex = index - elementsPerLocationPS[location];
        if(location == hostLocation){
            std::copy_n(data, sizeOfElement, dataPtrPerLocation[hostLocation] + sizeOfElement * localIndex);
        }else{
            int oldDevice; cudaGetDevice(&oldDevice); CUERR;
            cudaSetDevice(deviceIds[location]); CUERR;
            cudaMemcpy(dataPtrPerLocation[location] + sizeOfElement * localIndex, data, sizeOfElement, H2D); CUERR;
            cudaSetDevice(oldDevice); CUERR;
        }
    }

    void DistributedArray::set(size_t firstIndex, size_t lastIndex_excl, const char* data){
        std::vector<size_t> indices(lastIndex_excl - firstIndex);
        std::iota(indices.begin(), indices.end(), firstIndex);

        set(indices, data);
    }

    //indices must be strictly increasing sequence
    void DistributedArray::set(const std::vector<size_t>& indices, const char* data){
        assert(std::is_sorted(indices.begin(), indices.end(),
            [](auto l, auto r){
                if(r == l+1)
                    return true;
                return false;
            }
        ));
        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        std::vector<int> firstLocalIndices(numLocations, -1);
        std::vector<int> hitsPerLocation(numLocations, 0);
        int previousLocation = -1;
    	for(size_t i = 0; i < size_t(indices.size()); i++){
    		int location = getLocation(indices[i]);
            assert(location >= previousLocation);
            if(location != previousLocation){
                assert(firstLocalIndices[location] == -1);
                size_t localIndex = indices[i] - elementsPerLocationPS[location];
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

                size_t srcOffset = hitsPerLocationPrefixSum[location] * sizeOfElement;
                size_t dstOffset = firstLocalIndices[location] * sizeOfElement;
                int numHits = hitsPerLocation[location];

                if(location == hostLocation){
                    std::copy_n(data + srcOffset, sizeOfElement * numHits, dataPtrPerLocation[location] + dstOffset);
                }else{
                    cudaSetDevice(deviceIds[location]); CUERR;
                    cudaMemcpy(dataPtrPerLocation[location] + dstOffset, data + srcOffset, sizeOfElement * numHits, H2D); CUERR;
                }
            }
        }

        cudaSetDevice(oldDevice); CUERR;
    }

    void DistributedArray::get(size_t index, char* result){
        int location = getLocation(index);
        size_t localIndex = index - elementsPerLocationPS[location];
        if(location == hostLocation){
            std::copy_n(dataPtrPerLocation[hostLocation] + sizeOfElement * localIndex, sizeOfElement, result);
        }else{
            int oldDevice; cudaGetDevice(&oldDevice); CUERR;
            cudaSetDevice(deviceIds[location]); CUERR;
            cudaMemcpy(result, dataPtrPerLocation[location] + sizeOfElement * localIndex, sizeOfElement, D2H); CUERR;
            cudaSetDevice(oldDevice); CUERR;
        }
    }



    //d_result, d_indices must point to memory of device deviceId. d_indices[i] + indexOffset must be a local element index for this device
    void DistributedArray::copyDataToGpuBufferAsync(char* d_result, const size_t* d_indices, size_t nIndices, int deviceId, cudaStream_t stream, size_t indexOffset) const{
        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        cudaSetDevice(deviceId); CUERR;

        auto it = std::find(deviceIds.begin(), deviceIds.end(), deviceId);
        assert(it != deviceIds.end());

        int gpu = std::distance(deviceIds.begin(), it);
        size_t sizeOfElement_ = sizeOfElement;

        const char* const gpuData = dataPtrPerLocation[gpu];

        dim3 grid(SDIV(nIndices, 128),1,1);
        dim3 block(128,1,1);

        generic_kernel<<<grid, block,0, stream>>>([=] __device__ (){
            for(size_t k = threadIdx.x + blockDim.x * blockIdx.x; k < nIndices; k += blockDim.x * gridDim.x){
                const size_t index = d_indices[k] + indexOffset;
                for(size_t b = 0; b < sizeOfElement_; b++){
                    d_result[k * sizeOfElement_ + b] = gpuData[index * sizeOfElement_ + b];
                }
            }
        }); CUERR;

        cudaSetDevice(oldDevice); CUERR;
    }

    void DistributedArray::copyDataToGpuBufferAsync(char* d_result, int resultDevice, const size_t* d_indices, size_t nIndices, int sourceDevice, cudaStream_t stream, size_t indexOffset) const{
        int oldDevice; cudaGetDevice(&oldDevice); CUERR;



        auto it = std::find(deviceIds.begin(), deviceIds.end(), sourceDevice);
        assert(it != deviceIds.end());

        int gpu = std::distance(deviceIds.begin(), it);
        size_t sizeOfElement_ = sizeOfElement;

        const char* const gpuData = dataPtrPerLocation[gpu];

        dim3 grid(SDIV(nIndices, 128),1,1);
        dim3 block(128,1,1);

        cudaSetDevice(resultDevice); CUERR;

        generic_kernel<<<grid, block,0, stream>>>([=] __device__ (){
            for(size_t k = threadIdx.x + blockDim.x * blockIdx.x; k < nIndices; k += blockDim.x * gridDim.x){
                const size_t index = d_indices[k] + indexOffset;
                for(size_t b = 0; b < sizeOfElement_; b++){
                    d_result[k * sizeOfElement_ + b] = gpuData[index * sizeOfElement_ + b];
                }
            }
        }); CUERR;

        cudaSetDevice(oldDevice); CUERR;
    }

    //d_result, d_indices must point to memory of device deviceId. d_indices[i] must be a local element index for this device
    void DistributedArray::copyDataToGpuBufferAsync(char* d_result, const size_t* d_indices, size_t nIndices, int deviceId, cudaStream_t stream) const{
        copyDataToGpuBufferAsync(d_result, d_indices, nIndices, deviceId, stream, 0);
    }


    std::unique_ptr<DistributedArray::GatherHandle> DistributedArray::makeGatherHandle() const{
        auto handle = std::make_unique<GatherHandle>();
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
        cudaSetDevice(oldDevice); CUERR;

        return handle;
    }

    void DistributedArray::destroyGatherHandle(const std::unique_ptr<GatherHandle>& handle) const{
        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        handle->pinnedLocalIndices = std::move(SimpleAllocationPinnedHost<size_t>{});
        handle->pinnedResultData = std::move(SimpleAllocationPinnedHost<char>{});

        for(size_t gpu = 0; gpu < handle->dataPerGpu.size(); gpu++){
            cudaSetDevice(deviceIds[gpu]); CUERR;

            handle->deviceLocalIndicesPerLocation[gpu] = std::move(SimpleAllocationDevice<size_t>{});
            handle->dataPerGpu[gpu] = std::move(SimpleAllocationDevice<char>{});
            cudaStreamDestroy(handle->streamsPerGpu[gpu]); CUERR;
            cudaEventDestroy(handle->eventsPerGpu[gpu]); CUERR;
        }

        for(auto& pair : handle->tmpResultsOfDevice){
            cudaSetDevice(pair.first); CUERR;
            pair.second = std::move(SimpleAllocationDevice<char>{});
        }

        for(auto& pair : handle->permutationIndicesOfDevice){
            cudaSetDevice(pair.first); CUERR;
            pair.second = std::move(SimpleAllocationDevice<size_t>{});
        }

        cudaSetDevice(oldDevice); CUERR;
    }

    std::future<void> DistributedArray::gatherReadsInHostMemAsync(const std::unique_ptr<GatherHandle>& handle, size_t* indices, size_t numIds, char* result) const{
        const auto instance = this;
        auto future = std::async(std::launch::async, [=](const std::unique_ptr<GatherHandle>& handle){
                                    instance->gatherReadsInHostMem(handle, indices, numIds, result);
                                }, std::ref(handle));
        return future;
    }

    void DistributedArray::gatherReadsInHostMem(const std::unique_ptr<GatherHandle>& handle, size_t* indices, size_t numIds, char* result) const{
        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

        std::vector<size_t> hitsPerLocation(numLocations, 0);
        for(size_t i = 0; i < numIds; i++){
            int location = getLocation(indices[i]);
            hitsPerLocation[location]++;
        }

        std::vector<size_t> hitsPerLocationPrefixSum(numLocations+1,0);
        std::partial_sum(hitsPerLocation.begin(), hitsPerLocation.end(), hitsPerLocationPrefixSum.begin()+1);
        std::fill(hitsPerLocation.begin(), hitsPerLocation.end(), 0);

        std::vector<size_t> permutationIndices(numIds);

        handle->pinnedLocalIndices.resize(numIds);
        handle->pinnedPermutationIndices.resize(numIds);
        for(size_t i = 0; i < numIds; i++){
            int location = getLocation(indices[i]);
            size_t localIndex = indices[i] - elementsPerLocationPS[location];
            size_t tmpresultindex = hitsPerLocationPrefixSum[location] + hitsPerLocation[location];
            handle->pinnedPermutationIndices[i] = tmpresultindex;

            handle->pinnedLocalIndices[tmpresultindex] = localIndex;
            //std::cerr << "read id " << readIds[i] << ", location " << location << ", localReadId " << localReadId << ", index " << index << "\n";
            hitsPerLocation[location]++;
        }
        handle->pinnedResultData.resize(numIds * sizeOfElement);

        // std::cerr << "hitsPerLocation: ";
        // std::copy(hitsPerLocation.begin(), hitsPerLocation.end(), std::ostream_iterator<size_t>(std::cerr, " "));
        // std::cerr << "\n";

        //gather from gpus to host
        for(int gpu = 0; gpu < numGpus; gpu++){
            size_t numHits = hitsPerLocation[gpu];
            if(numHits > 0){
                int deviceId = deviceIds[gpu];
                cudaStream_t stream = handle->streamsPerGpu[gpu];

                cudaSetDevice(deviceIds[gpu]); CUERR;

                auto& myLocalIds = handle->deviceLocalIndicesPerLocation[gpu];
                auto& myResult = handle->dataPerGpu[gpu];


                myLocalIds.resize(numHits);
                cudaMemcpyAsync(myLocalIds.get(),
                                handle->pinnedLocalIndices.get() + hitsPerLocationPrefixSum[gpu],
                                sizeof(size_t) * numHits,
                                H2D,
                                stream); CUERR;

                myResult.resize(numHits * sizeOfElement);

                copyDataToGpuBufferAsync(myResult.get(), myLocalIds.get(), numHits, deviceId, stream);

                cudaMemcpyAsync(handle->pinnedResultData.get() + hitsPerLocationPrefixSum[gpu] * sizeOfElement,
                                myResult.get(),
                                sizeOfElement * numHits,
                                D2H,
                                stream); CUERR;

                cudaEventRecord(handle->eventsPerGpu[gpu]); CUERR;
            }
        }

        //gather from host to host
        if(hitsPerLocation[hostLocation] > 0){
            size_t numHits = hitsPerLocation[hostLocation];
            const size_t* hostLocalIds = handle->pinnedLocalIndices.get() + hitsPerLocationPrefixSum[hostLocation];
            char* hostResult = handle->pinnedResultData.get() + hitsPerLocationPrefixSum[hostLocation] * sizeOfElement;
            for(size_t k = 0; k < numHits; k++){
                size_t localId = hostLocalIds[k];
                std::copy_n(&dataPtrPerLocation[hostLocation][localId * sizeOfElement], sizeOfElement, &hostResult[k * sizeOfElement]);
            }
        }

        for(auto& event : handle->eventsPerGpu){
            cudaEventSynchronize(event); CUERR;
        }

        //permute pinnedResultData and store in output array
        for(size_t dstindex = 0; dstindex < numIds; dstindex++){
            const size_t srcindex = handle->pinnedPermutationIndices[dstindex];
            std::copy_n(&handle->pinnedResultData[srcindex * sizeOfElement], sizeOfElement, &result[dstindex * sizeOfElement]);
        }
    }

    void DistributedArray::gatherReadsInGpuMem(const std::unique_ptr<GatherHandle>& handle, size_t* indices, size_t* d_indices, size_t numIds, int deviceId, char* d_result) const {
        cudaStream_t stream = 0;
        cudaStreamCreate(&stream); CUERR;
        gatherReadsInGpuMemAsync(handle, indices, d_indices, numIds, deviceId, d_result, stream);
        cudaStreamSynchronize(stream); CUERR;
        cudaStreamDestroy(stream); CUERR;
    }

    //the same GatherHandle must not be used in another call until the results of the previous call are calculated
    void DistributedArray::gatherReadsInGpuMemAsync(const std::unique_ptr<GatherHandle>& handle, size_t* indices, size_t* d_indices, size_t numIds, int deviceId, char* d_result, cudaStream_t stream) const{
        if(numIds == 0) return;

        int oldDevice; cudaGetDevice(&oldDevice); CUERR;

TIMERSTARTCPU(countsHitsPerLocation);
    	std::vector<size_t> hitsPerLocation(numLocations, 0);
    	for(size_t i = 0; i < numIds; i++){
    		int location = getLocation(indices[i]);
    		hitsPerLocation[location]++;
    	}
TIMERSTOPCPU(countsHitsPerLocation);

        if(debug){
            std::cerr << "hitsPerLocation: ";
            std::copy(hitsPerLocation.begin(), hitsPerLocation.end(), std::ostream_iterator<size_t>(std::cerr, " "));
            std::cerr << "\n";
        }

        //shortcut. if all elements reside on the gpu with device id deviceId, perform a simple gather on the gpu, avoiding copies to host and from host.
        bool simpleGatherOnSameDevice = true;
        auto deviceIdIter = std::find(deviceIds.begin(), deviceIds.end(), deviceId);
        int deviceIdLocation = -1;
        if(deviceIdIter != deviceIds.end()){
            deviceIdLocation = std::distance(deviceIds.begin(), deviceIdIter);
            for(int i = 0; i < numLocations && simpleGatherOnSameDevice; i++){
                if(i != deviceIdLocation && hitsPerLocation[i] > 0){
                    simpleGatherOnSameDevice = false;
                }
            }
        }

        if(simpleGatherOnSameDevice){
            if(debug) std::cerr << "simpleGatherOnSameDevice " << deviceId << '\n';
            copyDataToGpuBufferAsync(d_result, d_indices, numIds, deviceId, stream, -elementsPerLocationPS[deviceIdLocation]);
            return;
        }

TIMERSTARTCPU(makenewindices);
    	std::vector<size_t> hitsPerLocationPrefixSum(numLocations+1,0);
    	std::partial_sum(hitsPerLocation.begin(), hitsPerLocation.end(), hitsPerLocationPrefixSum.begin()+1);
        std::fill(hitsPerLocation.begin(), hitsPerLocation.end(), 0);

        std::vector<size_t> permutationIndices(numIds);

        handle->pinnedLocalIndices.resize(numIds);
        handle->pinnedPermutationIndices.resize(numIds);
        for(size_t i = 0; i < numIds; i++){
    		int location = getLocation(indices[i]);
            size_t localIndex = indices[i] - elementsPerLocationPS[location];
            size_t tmpresultindex = hitsPerLocationPrefixSum[location] + hitsPerLocation[location];
            handle->pinnedPermutationIndices[i] = tmpresultindex;

            handle->pinnedLocalIndices[tmpresultindex] = localIndex;
            //std::cerr << "read id " << readIds[i] << ", location " << location << ", localReadId " << localReadId << ", index " << index << "\n";
    		hitsPerLocation[location]++;
    	}
TIMERSTOPCPU(makenewindices);

TIMERSTARTCPU(resizedevicevectors)
        handle->pinnedResultData.resize(numIds * sizeOfElement);

        cudaSetDevice(deviceId); CUERR;
        handle->localIndicesOnDevice[deviceId].resize(numIds);
        cudaMemcpyAsync(handle->localIndicesOnDevice[deviceId].get(),
                        handle->pinnedLocalIndices.get(),
                        sizeof(size_t) * numIds,
                        H2D,
                        stream); CUERR;

        handle->tmpResultsOfDevice[deviceId].resize(numIds * sizeOfElement);

        handle->permutationIndicesOfDevice[deviceId].resize(numIds);

        cudaMemcpyAsync(handle->permutationIndicesOfDevice[deviceId],
                        handle->pinnedPermutationIndices.get(),
                        sizeof(size_t) * numIds,
                        H2D,
                        stream); CUERR;
TIMERSTOPCPU(resizedevicevectors);

        //gather from gpus to host
        for(int gpu = 0; gpu < numGpus; gpu++){
            size_t numHits = hitsPerLocation[gpu];
            if(numHits > 0){
                int mydeviceId = deviceIds[gpu];
                cudaStream_t mystream = handle->streamsPerGpu[gpu];
                cudaEvent_t myevent = handle->eventsPerGpu[gpu];

                if(true && (peerAccess.hasPeerAccess(deviceId, mydeviceId) || deviceId == mydeviceId)){
                    std::cerr << "use peer access / local access: " << deviceId << " <---- " << mydeviceId << "\n";
/*
                    cudaSetDevice(mydeviceId); CUERR;
                    auto& myLocalIds = handle->deviceLocalIndicesPerLocation[gpu];
                    myLocalIds.resize(numHits);
                    cudaMemcpyAsync(myLocalIds.get(),
                                    handle->pinnedLocalIndices.get() + hitsPerLocationPrefixSum[gpu],
                                    sizeof(size_t) * numHits,
                                    H2D,
                                    mystream); CUERR;
                    cudaEventRecord(myevent, mystream); CUERR;
*/
                    cudaSetDevice(deviceId); CUERR;
                    cudaStreamWaitEvent(stream, myevent,0); CUERR; //wait in result stream until remote indices are ready
                    char* destptr = handle->tmpResultsOfDevice[deviceId].get() + hitsPerLocationPrefixSum[gpu] * sizeOfElement;
                    size_t* localIdsPtr = handle->localIndicesOnDevice[deviceId].get() + hitsPerLocationPrefixSum[gpu];
                    //copyDataToGpuBufferAsync(destptr, deviceId, myLocalIds.get(), numHits, mydeviceId, stream, 0);
                    copyDataToGpuBufferAsync(destptr, deviceId, localIdsPtr, numHits, mydeviceId, stream, 0);
                }else{
                    std::cerr << "use intermediate host: " << deviceId << " <---- host <---- " << mydeviceId << "\n";


            	    cudaSetDevice(mydeviceId); CUERR;

                    auto& myLocalIds = handle->deviceLocalIndicesPerLocation[gpu];
                    auto& myResult = handle->dataPerGpu[gpu];


            	    myLocalIds.resize(numHits);
                    cudaMemcpyAsync(myLocalIds.get(),
                                    handle->pinnedLocalIndices.get() + hitsPerLocationPrefixSum[gpu],
                                    sizeof(size_t) * numHits,
                                    H2D,
                                    mystream); CUERR;

            	    myResult.resize(numHits * sizeOfElement);

                    copyDataToGpuBufferAsync(myResult.get(), mydeviceId, myLocalIds.get(), numHits, mydeviceId, mystream, 0);

                    cudaMemcpyAsync(handle->pinnedResultData.get() + hitsPerLocationPrefixSum[gpu] * sizeOfElement,
                                    myResult.get(),
                                    sizeOfElement * numHits,
                                    D2H,
                                    mystream); CUERR;

                    cudaEventRecord(myevent, mystream); CUERR;

                    cudaSetDevice(deviceId); CUERR;
                    cudaStreamWaitEvent(stream, myevent,0); CUERR; //wait in result stream until partial results are on the host.

                    //copy partial results to tmp result buffer on device
                    cudaMemcpyAsync(handle->tmpResultsOfDevice[deviceId].get() + hitsPerLocationPrefixSum[gpu] * sizeOfElement,
                                    handle->pinnedResultData.get() + hitsPerLocationPrefixSum[gpu] * sizeOfElement,
                                    sizeOfElement * hitsPerLocation[gpu],
                                    H2D,
                                    stream);
                }
            }
    	}

        //gather from host to host
        if(hitsPerLocation[hostLocation] > 0){
            size_t numHits = hitsPerLocation[hostLocation];
            const size_t* hostLocalIds = handle->pinnedLocalIndices.get() + hitsPerLocationPrefixSum[hostLocation];
            char* hostResult = handle->pinnedResultData.get() + hitsPerLocationPrefixSum[hostLocation] * sizeOfElement;
            for(size_t k = 0; k < numHits; k++){
                size_t localId = hostLocalIds[k];
                std::copy_n(&dataPtrPerLocation[hostLocation][localId * sizeOfElement], sizeOfElement, &hostResult[k * sizeOfElement]);
            }
        }

        cudaSetDevice(deviceId); CUERR;

        // for(int gpu = 0; gpu < numGpus; gpu++){
        //     cudaStreamWaitEvent(stream, handle->eventsPerGpu[gpu], 0); CUERR;
        // }

        //cudaStreamSynchronize(stream); CUERR;

        // cudaMemcpyAsync(handle->tmpResultsOfDevice[deviceId].get(),
        //                 handle->pinnedResultData.get(),
        //                 sizeOfElement * numIds,
        //                 H2D,
        //                 stream);

        cudaMemcpyAsync(handle->tmpResultsOfDevice[deviceId].get() + hitsPerLocationPrefixSum[hostLocation] * sizeOfElement,
                        handle->pinnedResultData.get() + hitsPerLocationPrefixSum[hostLocation] * sizeOfElement,
                        hitsPerLocation[hostLocation] * sizeOfElement,
                        H2D,
                        stream);

        {
            size_t size = sizeOfElement;
            const size_t* indices = handle->permutationIndicesOfDevice[deviceId].get();
            const char* src = handle->tmpResultsOfDevice[deviceId].get();
            char* dest = d_result;
            size_t n = numIds;

            dim3 block(128,1,1);
            dim3 grid(SDIV(n, block.x),1,1);

            generic_kernel<<<grid, block, 0, stream>>>([=] __device__ (){
                for(size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x){
                    const size_t srcindex = indices[i];
            	    for(size_t b = 0; b < size; b++){
                        dest[i * size + b] = src[srcindex * size + b];
                    }
                }
            });

        }

        /*call_compact_kernel_async((size_t*)d_result,
                                (size_t*)handle->tmpResultsOfDevice[deviceId].get(),
                                handle->permutationIndicesOfDevice[deviceId].get(),
                                numIds,
                                stream);*/




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
