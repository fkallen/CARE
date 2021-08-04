#ifndef DISTRIBUTED_ARRAY_HPP
#define DISTRIBUTED_ARRAY_HPP


#ifdef __NVCC__

#include <hpc_helpers.cuh>
#include <threadpool.hpp>
//#include <util.hpp>
#include <memorymanagement.hpp>

#include <gpu/cudaerrorcheck.cuh>

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


namespace distarraykernels{

    template<class Index_t, int maxNumGpus = 32>
    __global__
    void partitionSplitKernel(
        Index_t* const * __restrict__ splitIndices,
        Index_t* const * __restrict__ splitDestinationPositions,
        Index_t* __restrict__ numSplitIndicesPerLocation,
        int numLocations,
        const Index_t* __restrict__ elementsPerLocationPS,
        const Index_t* __restrict__ numIdsPtr,
        const Index_t* __restrict__ indices
    ){

        assert(numLocations <= maxNumGpus+1);

        auto atomicAggInc = [](Index_t* counter){
            auto g = cg::coalesced_threads();
            Index_t warp_res;
            if(g.thread_rank() == 0){
                warp_res = atomicAdd(counter, Index_t(g.size()));
            }
            return g.shfl(warp_res, 0) + g.thread_rank();
        };

        __shared__ Index_t shared_elementsPerLocationPS[maxNumGpus + 1 + 1];

        for(int tid = threadIdx.x; tid < numLocations+1; tid += blockDim.x){
            shared_elementsPerLocationPS[tid] = elementsPerLocationPS[tid];
        }

        __syncthreads();

        const Index_t numIds = *numIdsPtr;
        
        for(Index_t tid = threadIdx.x + blockIdx.x * Index_t(blockDim.x); 
                tid < numIds; 
                tid += Index_t(blockDim.x) * gridDim.x){

            const Index_t elementIndex = indices[tid];
            int location = -1;

            for(int i = 0; i < numLocations; i++){
                if(shared_elementsPerLocationPS[i] <= elementIndex && elementIndex < shared_elementsPerLocationPS[i+1]){
                    location = i;
                }
            }

            for(int i = 0; i < numLocations; ++i){
                if(i == location){
                    const Index_t j = atomicAggInc(&numSplitIndicesPerLocation[i]);
                    splitIndices[i][j] = elementIndex;
                    splitDestinationPositions[i][j] = tid;
                }
            }
        }
    }

    template<class T>
    __global__
    void exclPrefixSumSingleThreadKernel(
        T* __restrict__ ps, 
        const T* __restrict__ input, 
        int numElements
    ){
        ps[0] = 0;
        for(int i = 0; i < numElements-1; i++){
            ps[i+1] = ps[i] + input[i];
        }
    }

    template<class Index_t, class Value_t>
    __global__
    void gatherKernel(
            Value_t* __restrict__ result, 
            const Value_t* __restrict__ sourceData, 
            const Index_t* __restrict__ indices, 
            const Index_t* __restrict__ nIndicesPtr,
            Index_t indexOffset, 
            size_t resultPitchValueTs,
            size_t numCols
    ){

        const Index_t nIndices = *nIndicesPtr;

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
    void gatherKernel(
            Value_t* __restrict__ result, 
            const Value_t* __restrict__ sourceData, 
            const Index_t* __restrict__ indices, 
            Index_t nIndices,
            Index_t indexOffset, 
            size_t resultPitchValueTs,
            size_t numCols
    ){
        auto standardGather = [&](){

            for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; i < nIndices * numCols; i += size_t(blockDim.x) * gridDim.x){
                const Index_t outputrow = i / numCols;
                const Index_t inputrow = indices[outputrow] + indexOffset;
                const Index_t col = i % numCols;
                result[size_t(outputrow) * resultPitchValueTs + col] 
                        = sourceData[size_t(inputrow) * numCols + col];
            }
        };

        // auto specialGather = [&](){
        //     constexpr int groupsize = 8;
        //     auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());
        //     const int numGroupsInGrid = (blockDim.x * gridDim.x) / groupsize;
        //     const int groupIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;

        //     for(int outputrow = groupIdInGrid; outputrow < nIndices; outputrow += numGroupsInGrid){
        //         const Index_t inputrow = indices[outputrow] + indexOffset;

        //         for(int k = group.thread_rank(); k < numCols; k += group.size()){
        //             result[size_t(outputrow) * resultPitchValueTs + k] 
        //                 = sourceData[size_t(inputrow) * numCols + k];
        //         }
        //     }
        // };

        // auto specialGather2 = [&](){
        //     constexpr int groupsize = 8;
        //     auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());
        //     const int numGroupsInGrid = (blockDim.x * gridDim.x) / groupsize;
        //     const int groupIdInGrid = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;

        //     const int numIntsToCopy = (numCols * sizeof(Value_t)) / sizeof(int);
        //     const int remainingBytes = (numCols * sizeof(Value_t)) - numIntsToCopy * sizeof(int);

        //     for(int outputrow = groupIdInGrid; outputrow < nIndices; outputrow += numGroupsInGrid){
        //         const Index_t inputrow = indices[outputrow] + indexOffset;

        //         for(int k = group.thread_rank(); k < numIntsToCopy; k += group.size()){
        //             ((int*)&result[size_t(outputrow) * resultPitchValueTs])[k] 
        //                 = ((const int*)&sourceData[size_t(inputrow) * numCols])[k];
        //         }

        //         for(int k = group.thread_rank(); k < remainingBytes; k += group.size()){
        //             ((char*)(((int*)&result[size_t(outputrow) * resultPitchValueTs]) + numIntsToCopy))[k] 
        //                 = ((const char*)(((const int*)&sourceData[size_t(inputrow) * numCols]) + numIntsToCopy))[k];
        //         }
        //     }
        // };

        standardGather();
        //specialGather();
        // assert((numCols * sizeof(Value_t)) % sizeof(int) == 0);
        // assert((resultPitchValueTs * sizeof(Value_t)) % sizeof(int) == 0);

        // specialGather2();
    }


    template<class Index_t, class Value_t>
    __global__
    void scatterKernel(
            Value_t* __restrict__ result, 
            const Value_t* __restrict__ sourceData, 
            const Index_t* __restrict__ indices, 
            const Index_t* __restrict__ nIndicesPtr,
            Index_t indexOffset, 
            size_t resultPitchValueTs,
            size_t numCols
    ){

        const Index_t nIndices = *nIndicesPtr;

        for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; i < nIndices * numCols; i += size_t(blockDim.x) * gridDim.x){
            const Index_t inputrow = i / numCols;
            const Index_t outputrow = indices[inputrow] + indexOffset;
            const Index_t col = i % numCols;
            result[size_t(outputrow) * resultPitchValueTs + col] 
                    = sourceData[size_t(inputrow) * numCols + col];
        }
    }

    template<class Index_t, class Value_t>
    __global__
    void copy2Dkernel(
        Value_t* __restrict__ output,
        const Value_t* __restrict__ input,
        Index_t numRows,
        Index_t numColumns,
        size_t inputRowPitchElements,
        size_t outputRowPitchElements
    ){

        for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; 
                i < numRows * numColumns; 
                i += size_t(blockDim.x) * gridDim.x){

            const Index_t inputRow = i / numColumns;
            const Index_t col = i % numColumns;
            const Index_t outputRow = inputRow;
            
            output[size_t(outputRow) * outputRowPitchElements + col] 
                = input[size_t(inputRow) * inputRowPitchElements + col];
        }
    }


    // kernels with parameter object
    template<class Index_t>
    struct PartitionSplitKernelParams{
        Index_t** __restrict__ splitIndices;
        Index_t** __restrict__ splitDestinationPositions;
        Index_t* __restrict__ numSplitIndicesPerLocation;
        int numLocations;
        const Index_t* __restrict__ elementsPerLocationPS;
        const Index_t* __restrict__ numIdsPtr;
        const Index_t* __restrict__ indices;
    };

    template<class Index_t, int maxNumGpus = 32>
    __global__
    void partitionSplitKernel(
        const PartitionSplitKernelParams<Index_t>* __restrict__ params
    ){

        assert(params->numLocations <= maxNumGpus+1);

        auto atomicAggInc = [](Index_t* counter){
            auto g = cg::coalesced_threads();
            int warp_res;
            if(g.thread_rank() == 0){
                warp_res = atomicAdd(counter, Index_t(g.size()));
            }
            return g.shfl(warp_res, 0) + g.thread_rank();
        };

        __shared__ Index_t shared_elementsPerLocationPS[maxNumGpus + 1 + 1];

        for(int tid = threadIdx.x; tid < params->numLocations+1; tid += blockDim.x){
            shared_elementsPerLocationPS[tid] = params->elementsPerLocationPS[tid];
        }

        __syncthreads();

        const Index_t numIds = *params->numIdsPtr;
        
        for(Index_t tid = threadIdx.x + blockIdx.x * Index_t(blockDim.x); 
                tid < numIds; 
                tid += Index_t(blockDim.x) * gridDim.x){

            const Index_t elementIndex = params->indices[tid];
            int location = -1;

            for(int i = 0; i < params->numLocations; i++){
                if(shared_elementsPerLocationPS[i] <= elementIndex && elementIndex < shared_elementsPerLocationPS[i+1]){
                    location = i;
                }
            }

            for(int i = 0; i < params->numLocations; ++i){
                if(i == location){
                    const Index_t j = atomicAggInc(&params->numSplitIndicesPerLocation[i]);
                    params->splitIndices[i][j] = elementIndex;
                    params->splitDestinationPositions[i][j] = tid;
                }
            }
        }
    }

    template<class Value_t>
    struct PrefixSumKernelParams{
        Value_t* __restrict__ output;
        const Value_t* __restrict__ input;
        int numElements;
    };

    template<class Value_t>
    __global__
    void exclPrefixSumSingleThreadKernel(
        const PrefixSumKernelParams<Value_t>* __restrict__ params
    ){
        params->output[0] = 0;
        for(int i = 0; i < params->numElements-1; i++){
            params->output[i+1] = params->output[i] + params->input[i];
        }
    }

    template<class Index_t, class Value_t>
    struct GatherParams{
        Value_t* __restrict__ result;
        const Value_t* __restrict__ sourceData;
        const Index_t* __restrict__ indices; 
        const Index_t* __restrict__ nIndicesPtr;
        Index_t indexOffset;
        size_t resultPitchValueTs;
        size_t numCols;
    };

    template<class Index_t, class Value_t>
    __global__
    void gatherKernel(
        const GatherParams<Index_t,Value_t>* __restrict__ params
    ){

        const Index_t nIndices = *params->nIndicesPtr;

        for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; 
                i < nIndices * params->numCols; 
                i += size_t(blockDim.x) * gridDim.x){

            const Index_t outputrow = i / params->numCols;
            const Index_t inputrow = params->indices[outputrow] + params->indexOffset;
            const Index_t col = i % params->numCols;
            params->result[size_t(outputrow) * params->resultPitchValueTs + col] 
                    = params->sourceData[size_t(inputrow) * params->numCols + col];
        }
    }

    template<class Index_t, class Value_t>
    struct ScatterParams{
        Value_t* __restrict__ result;
        const Value_t* __restrict__ sourceData;
        const Index_t* __restrict__ indices;
        const Index_t* __restrict__ nIndicesPtr;
        Index_t indexOffset;
        size_t resultPitchValueTs;
        size_t numCols;
    };

    template<class Index_t, class Value_t>
    __global__
    void scatterKernel(
        const ScatterParams<Index_t,Value_t>* __restrict__ params
    ){
        const Index_t nIndices = *params->nIndicesPtr;

        for(size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x; 
                i < nIndices * params->numCols; 
                i += size_t(blockDim.x) * gridDim.x){

            const Index_t inputrow = i / params->numCols;
            const Index_t outputrow = params->indices[inputrow] + params->indexOffset;
            const Index_t col = i % params->numCols;
            params->result[size_t(outputrow) * params->resultPitchValueTs + col] 
                    = params->sourceData[size_t(inputrow) * params->numCols + col];
        }
    }

}










enum class DistributedArrayLayout{
    GPUBlock, GPUEqual
};

template<class Value_t, class Index_t = size_t>
struct DistributedArray{
public:

    

    struct GatherHandleStruct{

        std::mutex mutex;
        care::ThreadPool::ParallelForHandle pforHandle;

        helpers::SimpleAllocationPinnedHost<Index_t> pinnedIndicesOfHostLocation;
        helpers::SimpleAllocationPinnedHost<Index_t> numIndicesOfHostLocation;

        helpers::SimpleAllocationPinnedHost<Index_t> pinnedDestinationPositionsOfHostLocation;
        helpers::SimpleAllocationPinnedHost<Value_t> pinnedGatheredElementsOfHostLocation; 

        std::vector<helpers::SimpleAllocationDevice<Value_t>> d_gatheredElementsOfGpuLocation; //numGpus

        std::map<int, helpers::SimpleAllocationDevice<Value_t>> map_d_tmpResults; //tmp buffers to store all gathered elements on the destination gpu
        std::map<int, helpers::SimpleAllocationDevice<Index_t>> map_d_destinationPositionsOfGpu;
        std::map<int, helpers::SimpleAllocationDevice<Index_t>> map_d_elementsPerLocationPS;
        std::map<int, helpers::SimpleAllocationDevice<Index_t>> map_d_numIndicesPerLocation;
        std::map<int, helpers::SimpleAllocationDevice<Index_t>> map_d_numIndicesPerLocationPS;
        std::map<int, std::vector<helpers::SimpleAllocationDevice<Index_t>>> map_d_indicesForLocationsVector;
        std::map<int, std::vector<helpers::SimpleAllocationDevice<Index_t>>> map_d_destinationPositionsForLocationsVector;


        //kernel parameters per device id
        std::map<int, helpers::SimpleAllocationDevice<distarraykernels::PartitionSplitKernelParams<Index_t>>> map_d_splitkernelparams;
        std::map<int, helpers::SimpleAllocationPinnedHost<distarraykernels::PartitionSplitKernelParams<Index_t>>> map_h_splitkernelparams;

        std::map<int, helpers::SimpleAllocationDevice<distarraykernels::PrefixSumKernelParams<Index_t>>> map_d_prefixsumkernelparams;
        std::map<int, helpers::SimpleAllocationPinnedHost<distarraykernels::PrefixSumKernelParams<Index_t>>> map_h_prefixsumkernelparams;

        std::map<int, helpers::SimpleAllocationDevice<distarraykernels::GatherParams<Index_t,Value_t>>> map_d_gatherkernelparams;
        std::map<int, helpers::SimpleAllocationPinnedHost<distarraykernels::GatherParams<Index_t,Value_t>>> map_h_gatherkernelparams;

        using hscatvec_t = std::vector<helpers::SimpleAllocationPinnedHost<distarraykernels::ScatterParams<Index_t,Value_t>>>;
        using dscatvec_t = std::vector<helpers::SimpleAllocationDevice<distarraykernels::ScatterParams<Index_t,Value_t>>>;
        std::map<int, dscatvec_t> map_d_scatterkernelparams;
        std::map<int, hscatvec_t> map_h_scatterkernelparams;


        std::map<int, helpers::SimpleAllocationDevice<char>> map_d_packedpointersAndNumIndicesArg;
        helpers::SimpleAllocationPinnedHost<char> h_packedpointersAndNumIndicesArg;

        std::map<int, helpers::SimpleAllocationDevice<char>> map_d_packedKernelParamsPartPref;
        helpers::SimpleAllocationPinnedHost<char> h_packedKernelParamsPartPref;

        //
        std::map<int, cudaGraphExec_t> map_nohostExecutionGraph;

        std::vector<cudaStream_t> streamsPerGpuLocation;
        std::vector<cudaEvent_t> eventsPerGpuLocation;

        std::map<int, cudaStream_t> map_streams;
        std::map<int, cudaEvent_t> map_events;
        std::map<int, cudaEvent_t> map_d2hevents;

        std::vector<int> registeredDeviceIds;

    };

    using GatherHandle = std::unique_ptr<GatherHandleStruct>;

    MemoryUsage getMemoryInfoOfHandle(const GatherHandle& handle) const{

        MemoryUsage result;
        result.host += handle->pinnedIndicesOfHostLocation.capacityInBytes();
        result.host += handle->numIndicesOfHostLocation.capacityInBytes();
        result.host += handle->pinnedDestinationPositionsOfHostLocation.capacityInBytes();
        result.host += handle->pinnedGatheredElementsOfHostLocation.capacityInBytes();
        result.host += handle->h_packedpointersAndNumIndicesArg.capacityInBytes();
        result.host += handle->h_packedKernelParamsPartPref.capacityInBytes();

        for(size_t i = 0; i < handle->d_gatheredElementsOfGpuLocation.size(); i++){
            result.device[deviceIds[i]] += handle->d_gatheredElementsOfGpuLocation[i].capacityInBytes();
        }

        auto processHostMap = [&](const auto& m){
            for(const auto& pair : m){
                result.host += pair.second.capacityInBytes();
            }
        };

        auto processDeviceMap = [&](const auto& m){
            for(const auto& pair : m){
                result.device[pair.first] += pair.second.capacityInBytes();
            }
        };

        processDeviceMap(handle->map_d_tmpResults);
        processDeviceMap(handle->map_d_destinationPositionsOfGpu);
        processDeviceMap(handle->map_d_elementsPerLocationPS);
        processDeviceMap(handle->map_d_numIndicesPerLocation);
        processDeviceMap(handle->map_d_numIndicesPerLocationPS);
        processDeviceMap(handle->map_d_splitkernelparams);
        processDeviceMap(handle->map_d_prefixsumkernelparams);
        processDeviceMap(handle->map_d_gatherkernelparams);
        processDeviceMap(handle->map_d_packedpointersAndNumIndicesArg);
        processDeviceMap(handle->map_d_packedKernelParamsPartPref);

        processHostMap(handle->map_h_splitkernelparams);
        processHostMap(handle->map_h_prefixsumkernelparams);
        processHostMap(handle->map_h_gatherkernelparams);



        for(const auto& pair : handle->map_d_indicesForLocationsVector){
            for(const auto& buf : pair.second){
                result.device[pair.first] += buf.capacityInBytes();
            }
        }

        for(const auto& pair : handle->map_d_destinationPositionsForLocationsVector){
            for(const auto& buf : pair.second){
                result.device[pair.first] += buf.capacityInBytes();
            }
        }

        for(const auto& pair : handle->map_d_scatterkernelparams){
            for(const auto& buf : pair.second){
                result.device[pair.first] += buf.capacityInBytes();
            }
        }

        for(const auto& pair : handle->map_h_scatterkernelparams){
            for(const auto& buf : pair.second){
                result.host += buf.capacityInBytes();
            }
        }

        return result;
    }

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
                    memoryLimitBytesPerGPU(std::move(memoryLimitBytesPerGPU_)){

        assert(deviceIds.size() == memoryLimitBytesPerGPU.size());

        assert(deviceIds.size() <= maxNumGpus);

        elementsPerLocation.resize(numLocations, 0);
        elementsPerLocationPS.resize(numLocations+1, 0);
        dataPtrPerLocation.resize(numLocations, nullptr);

        if(numRows > 0 && numColumns > 0){

            int oldId; CUDACHECK(cudaGetDevice(&oldId));

            size_t totalRequiredMemory = numRows * sizeOfElement;

            bool preferedLocationIsSufficient = false;

            if(preferedLocation != -1 && preferedLocation != hostLocation){
                if(memoryLimitBytesPerGPU[preferedLocation] >= totalRequiredMemory){
                    preferedLocationIsSufficient = true;
                }
            }

            if(preferedLocationIsSufficient){
                CUDACHECK(wrapperCudaSetDevice(deviceIds[preferedLocation]));
                elementsPerLocation[preferedLocation] = numRows;
                CUDACHECK(cudaMalloc(&dataPtrPerLocation[preferedLocation], totalRequiredMemory));
            }else{

                size_t remainingElements = numRows;
                if(layout == DistributedArrayLayout::GPUBlock){
                    for(int gpu = 0; gpu < numGpus && remainingElements > 0; gpu++){
                        CUDACHECK(wrapperCudaSetDevice(deviceIds[gpu]));

                        size_t rows = std::min(remainingElements, memoryLimitBytesPerGPU[gpu] / sizeOfElement);
                        elementsPerLocation[gpu] = rows;
                        if(rows == 0){
                            continue;
                        }

                        CUDACHECK(cudaMalloc(&(dataPtrPerLocation[gpu]), rows * sizeOfElement));

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
                        CUDACHECK(wrapperCudaSetDevice(deviceIds[gpu]));
                        CUDACHECK(cudaMalloc(&(dataPtrPerLocation[gpu]), elementsPerLocation[gpu] * sizeOfElement));
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

            if(debug){
                std::cerr << "DistributedArray:\n";
                std::cerr << "device ids: [";
                std::copy(deviceIds.begin(), deviceIds.end(), std::ostream_iterator<int>(std::cerr, " "));
                std::cerr << "]\n";
                std::cerr << "SinglePartitionInfo: " << singlePartitionInfo.isSinglePartition << ", " << singlePartitionInfo.locationId << '\n';
                std::cerr << "elements per location: [";
                std::copy(elementsPerLocation.begin(), elementsPerLocation.end(), std::ostream_iterator<Index_t>(std::cerr, " "));
                std::cerr << "]\n";
            }

            CUDACHECK(wrapperCudaSetDevice(oldId));
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
        // cudaGetDevice(&currentId);

        // std::cerr << "cudaSetDevice " << currentId << " -> " << newId << '\n';
        cudaError_t res = cudaSetDevice(newId);
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
            int oldDevice; CUDACHECK(cudaGetDevice(&oldDevice));
            CUDACHECK(wrapperCudaSetDevice(deviceIds[location]));
            CUDACHECK(cudaMemcpy(destPtr, data, sizeOfElement, H2D));
            CUDACHECK(wrapperCudaSetDevice(oldDevice));
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

        const std::size_t n = indices.size();
        setSafe(indices.data(), data, n);
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
        int oldDevice; CUDACHECK(cudaGetDevice(&oldDevice));

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
                    CUDACHECK(wrapperCudaSetDevice(deviceIds[location]));
                    CUDACHECK(cudaMemcpy(destPtr, srcPtr, sizeOfElement * numHits, H2D));
                }
            }
        }

        CUDACHECK(wrapperCudaSetDevice(oldDevice));
    }

    void setSafe(const Index_t* indices, const Value_t* data, std::size_t n){
        setSafeFromHostData(indices, data, n);
    }

    void setSafeFromHostData(const Index_t* indices, const Value_t* data, std::size_t n){
        assert(std::is_sorted(indices, indices + n));

        if(n == 0){
            return;
        }

        int oldDevice; CUDACHECK(cudaGetDevice(&oldDevice));




        if(singlePartitionInfo.isSinglePartition){
            const int locationId = singlePartitionInfo.locationId;

            bool isConsecutiveIndices = true;
            for(size_t k = 0; k < n-1; k++){
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

                    std::copy_n(data, n * numColumns, destPtr);
                    return;
                }else{
                    Index_t start = indices[0];
                    Value_t* destPtr = offsetPtr(dataPtrPerLocation[locationId], start);
                    CUDACHECK(wrapperCudaSetDevice(deviceIds[locationId]));
                    CUDACHECK(cudaMemcpy(destPtr, data, n * sizeOfElement, H2D));
                    CUDACHECK(wrapperCudaSetDevice(oldDevice));
                    return;
                }
            }


        }

        std::vector<int> localIndices(n, -1);
        std::vector<int> hitsPerLocation(numLocations, 0);

        for(size_t i = 0; i < n; i++){
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
                        CUDACHECK(wrapperCudaSetDevice(deviceIds[location]));
                        CUDACHECK(cudaMemcpy(destPtr, srcPtr, sizeOfElement * (to-from), H2D));
                    }

                    num = to;
                }
            }
        }

        CUDACHECK(wrapperCudaSetDevice(oldDevice));
    }



    void get(Index_t index, Value_t* result){
        int location = getLocation(index);
        Index_t localIndex = index - elementsPerLocationPS[location];
        const Value_t* srcPtr = offsetPtr(dataPtrPerLocation[location], localIndex);

        if(location == hostLocation){
            std::copy_n(srcPtr, numColumns, result);
        }else{
            int oldDevice; CUDACHECK(cudaGetDevice(&oldDevice));
            CUDACHECK(wrapperCudaSetDevice(deviceIds[location]));
            CUDACHECK(cudaMemcpy(result, srcPtr, sizeOfElement, D2H));
            CUDACHECK(wrapperCudaSetDevice(oldDevice));
        }
    }

    GatherHandle makeGatherHandle() const{
        auto handle = std::make_unique<GatherHandleStruct>();

        int oldDevice;
        CUDACHECK(cudaGetDevice(&oldDevice));
        //CUDACHECK(wrapperCudaSetDevice(oldDevice));

        handle->d_gatheredElementsOfGpuLocation.resize(numGpus);
        
        handle->numIndicesOfHostLocation.resize(1);

        handle->h_packedpointersAndNumIndicesArg.resize(
            sizeof(Index_t*) * numLocations
            + sizeof(Index_t*) * numLocations
            + std::abs(int(sizeof(Index_t*)) - int(sizeof(Index_t))) // padding
            + sizeof(Index_t)
        );

        handle->h_packedKernelParamsPartPref.resize(
            std::max(
                sizeof(distarraykernels::PartitionSplitKernelParams<Index_t>),
                sizeof(distarraykernels::PrefixSumKernelParams<Index_t>)
            ) * 2
        );

        handle->streamsPerGpuLocation.resize(numGpus);
        handle->eventsPerGpuLocation.resize(numGpus);

        for(int gpu = 0; gpu < numGpus; gpu++){
            registerDeviceIdForHandlenew(handle, deviceIds[gpu]);

            CUDACHECK(wrapperCudaSetDevice(deviceIds[gpu]));
            CUDACHECK(cudaStreamCreate(&(handle->streamsPerGpuLocation[gpu])));
            CUDACHECK(cudaEventCreateWithFlags(&(handle->eventsPerGpuLocation[gpu]), cudaEventDisableTiming));
        }

        CUDACHECK(wrapperCudaSetDevice(oldDevice));

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
            CUDACHECK(cudaGetDevice(&oldId));
            CUDACHECK(wrapperCudaSetDevice(deviceId));

            cudaStream_t stream;
            CUDACHECK(cudaStreamCreate(&stream));
            handle->map_streams[deviceId] = std::move(stream);

            cudaEvent_t event;
            CUDACHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            handle->map_events[deviceId] = std::move(event); 

            cudaEvent_t event2;
            CUDACHECK(cudaEventCreateWithFlags(&event2, cudaEventDisableTiming));
            handle->map_d2hevents[deviceId] = std::move(event2); 
             

            handle->map_d_tmpResults.emplace(deviceId, helpers::SimpleAllocationDevice<Value_t>{});
            handle->map_d_destinationPositionsOfGpu.emplace(deviceId, helpers::SimpleAllocationDevice<Index_t>{});
            handle->map_d_elementsPerLocationPS.emplace(deviceId, helpers::SimpleAllocationDevice<Index_t>(numLocations + 1));

            helpers::SimpleAllocationDevice<Index_t> aaa1(numLocations);
            handle->map_d_numIndicesPerLocation.emplace(deviceId, std::move(aaa1));
            helpers::SimpleAllocationDevice<Index_t> aaa2(numLocations+1);
            handle->map_d_numIndicesPerLocationPS.emplace(deviceId, std::move(aaa2));

            std::vector<helpers::SimpleAllocationDevice<Index_t>> vec1(numLocations + 1);
            handle->map_d_indicesForLocationsVector.emplace(deviceId, std::move(vec1));
            std::vector<helpers::SimpleAllocationDevice<Index_t>> vec2(numLocations + 1);
            handle->map_d_destinationPositionsForLocationsVector.emplace(deviceId, std::move(vec2));

            CUDACHECK(cudaMemcpyAsync(
                handle->map_d_elementsPerLocationPS[deviceId].get(),
                elementsPerLocationPS.data(),
                sizeof(Index_t) * (numLocations+1),
                H2D,
                handle->map_streams[deviceId]
            )); 
            CUDACHECK(cudaStreamSynchronize(handle->map_streams[deviceId]));

            handle->map_d_splitkernelparams.emplace(deviceId, helpers::SimpleAllocationDevice<distarraykernels::PartitionSplitKernelParams<Index_t>>(1));
            handle->map_h_splitkernelparams.emplace(deviceId, helpers::SimpleAllocationPinnedHost<distarraykernels::PartitionSplitKernelParams<Index_t>>(1));

            handle->map_d_prefixsumkernelparams.emplace(deviceId, helpers::SimpleAllocationDevice<distarraykernels::PrefixSumKernelParams<Index_t>>(1));
            handle->map_h_prefixsumkernelparams.emplace(deviceId, helpers::SimpleAllocationPinnedHost<distarraykernels::PrefixSumKernelParams<Index_t>>(1));

            handle->map_d_gatherkernelparams.emplace(deviceId, helpers::SimpleAllocationDevice<distarraykernels::GatherParams<Index_t,Value_t>>(1));
            handle->map_h_gatherkernelparams.emplace(deviceId, helpers::SimpleAllocationPinnedHost<distarraykernels::GatherParams<Index_t,Value_t>>(1));

            using hscatvec_t = std::vector<helpers::SimpleAllocationPinnedHost<distarraykernels::ScatterParams<Index_t,Value_t>>>;
            using dscatvec_t = std::vector<helpers::SimpleAllocationDevice<distarraykernels::ScatterParams<Index_t,Value_t>>>;

            hscatvec_t hscatvec(numGpus);
            for(int k = 0; k < numGpus; k++){
                hscatvec[k].resize(1);
            }
            handle->map_h_scatterkernelparams.emplace(deviceId, std::move(hscatvec)); 
            dscatvec_t dscatvec(numGpus);
            for(int k = 0; k < numGpus; k++){
                dscatvec[k].resize(1);
            }
            handle->map_d_scatterkernelparams.emplace(deviceId, std::move(dscatvec));

            handle->map_d_packedpointersAndNumIndicesArg.emplace(deviceId, helpers::SimpleAllocationDevice<char>());
            handle->map_d_packedpointersAndNumIndicesArg[deviceId].resize(handle->h_packedpointersAndNumIndicesArg.size());

            handle->map_d_packedKernelParamsPartPref.emplace(deviceId, helpers::SimpleAllocationDevice<char>());
            handle->map_d_packedKernelParamsPartPref[deviceId].resize(handle->h_packedKernelParamsPartPref.size());

     

            handle->registeredDeviceIds.push_back(deviceId);

            CUDACHECK(wrapperCudaSetDevice(oldId));
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
            CUDACHECK(cudaGetDevice(&oldId));
            CUDACHECK(wrapperCudaSetDevice(deviceId));

            handle->map_d_tmpResults[deviceId].destroy();
            handle->map_d_destinationPositionsOfGpu[deviceId].destroy();
            handle->map_d_elementsPerLocationPS[deviceId].destroy();
            handle->map_d_numIndicesPerLocation[deviceId].destroy();
            handle->map_d_numIndicesPerLocationPS[deviceId].destroy();

            handle->map_d_indicesForLocationsVector[deviceId].clear();
            handle->map_d_destinationPositionsForLocationsVector[deviceId].clear();

            handle->map_d_numIndices[deviceId].destroy();

            handle->map_d_splitkernelparams[deviceId].destroy();
            handle->map_d_prefixsumkernelparams[deviceId].destroy();
            handle->map_d_gatherkernelparams[deviceId].destroy();
            handle->map_d_scatterkernelparams[deviceId].clear();

            handle->map_h_splitkernelparams[deviceId].destroy();
            handle->map_h_prefixsumkernelparams[deviceId].destroy();
            handle->map_h_gatherkernelparams[deviceId].destroy();
            handle->map_h_scatterkernelparams[deviceId].clear();

            handle->map_d_packedpointersAndNumIndicesArg[deviceId].destroy();
            handle->map_d_packedKernelParamsPartPref[deviceId].destroy();

            auto git = handle->map_nohostExecutionGraph.find(deviceId);
            if(git != handle->map_nohostExecutionGraph.end()){
                CUDACHECK(cudaGraphExecDestroy(git->second));
            }
            
            CUDACHECK(cudaStreamDestroy(handle->map_streams[deviceId]));
            CUDACHECK(cudaEventDestroy(handle->map_events[deviceId]));
            CUDACHECK(cudaEventDestroy(handle->map_d2hevents[deviceId]));

            handle->registeredDeviceIds.erase(it);

            CUDACHECK(wrapperCudaSetDevice(oldId));
        }
    }

    void destroyGatherHandleStruct(const GatherHandle& handle) const{
        int oldDevice; CUDACHECK(cudaGetDevice(&oldDevice));

        handle->pinnedDestinationPositionsOfHostLocation.destroy();
        handle->h_packedpointersAndNumIndicesArg.destroy();
        handle->h_packedKernelParamsPartPref.destroy();

        for(int gpu = 0; gpu < numGpus; gpu++){
            CUDACHECK(wrapperCudaSetDevice(deviceIds[gpu]));

            handle->d_gatheredElementsOfGpuLocation[gpu].destroy();

            CUDACHECK(cudaStreamDestroy(handle->streamsPerGpuLocation[gpu]));
            CUDACHECK(cudaEventDestroy(handle->eventsPerGpuLocation[gpu]));
        }

        while(handle->registeredDeviceIds.size() > 0){
            unregisterDeviceIdForHandlenew(handle, handle->registeredDeviceIds[0]);
        }

        CUDACHECK(wrapperCudaSetDevice(oldDevice));
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

        if(numIds <= 0){
            return;
        }

        //std::lock_guard<std::mutex> l(handle->mutex);

        if(singlePartitionInfo.isSinglePartition){
            nvtx::ScopedRange r("singlePartitionGather", 0);

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

        }else{

            if(elementsPerLocation[hostLocation] == 0){
                nvtx::ScopedRange r("nohostGather", 1);

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

            }else{        

                nvtx::ScopedRange r("generalGather", 2);

                gatherElementsInGpuMemAsyncGeneral(
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

        if(numIds <= 0){
            return;
        }

        if(singlePartitionInfo.locationId == hostLocation){
            // if(debug) std::cerr << "single location array fasthpath on host\n";

            handle->pinnedGatheredElementsOfHostLocation.resize(numIds * SDIV(resultPitch, sizeof(Value_t)));

            Value_t* const myResult = handle->pinnedGatheredElementsOfHostLocation.get();

            int oldDevice; CUDACHECK(cudaGetDevice(&oldDevice));
            CUDACHECK(wrapperCudaSetDevice(resultDeviceId));

            auto gather = [&](Index_t begin, Index_t end, int /*threadId*/){
                #if 0
                for(Index_t k = begin; k < end; k++){
                    const Index_t localId = indices[k];

                    const Value_t* const srcPtr = offsetPtr(dataPtrPerLocation[hostLocation], localId);
                    Value_t* const destPtr = (Value_t*)(((const char*)(myResult)) + resultPitch * k);

                    std::copy_n(srcPtr, numColumns, destPtr);
                }
                #else
                const Index_t myChunksize = end - begin;

                //batches to overlap gathering with cudamemcpyAsync
                constexpr Index_t batchsize = 5000;
                const Index_t numBatches = SDIV(myChunksize, batchsize);

                for(Index_t b = 0; b < numBatches; b++){
                    const Index_t batchbegin = begin + b * batchsize;
                    const Index_t batchend = begin + std::min((b+1) * batchsize, myChunksize);
                    const Index_t currentBatchsize = batchend - batchbegin;

                    for(Index_t k = batchbegin; k < batchend; k++){
                        const Index_t localId = indices[k];

                        const Value_t* const srcPtr = offsetPtr(dataPtrPerLocation[hostLocation], localId);
                        Value_t* const destPtr = (Value_t*)(((char*)(myResult)) + resultPitch * k);

                        std::copy_n(srcPtr, numColumns, destPtr);
                    }

                    CUDACHECK(cudaMemcpyAsync(
                        (((char*)(d_result)) + resultPitch * size_t(batchbegin)),
                        (((const char*)(myResult)) + resultPitch * size_t(batchbegin)),
                        currentBatchsize * resultPitch,
                        H2D,
                        syncstream
                    ));
                }
                #endif
            };

            forLoop( 
                Index_t(0), 
                numIds, 
                gather
            );

            CUDACHECK(wrapperCudaSetDevice(oldDevice));

            // int oldDevice; CUDACHECK(cudaGetDevice(&oldDevice));

            // CUDACHECK(wrapperCudaSetDevice(resultDeviceId));

            // CUDACHECK(cudaMemcpyAsync(d_result, myResult, numIds * resultPitch, H2D, syncstream));

            // CUDACHECK(wrapperCudaSetDevice(oldDevice));
        }else{
            //if(debug) std::cerr << "single location array fast path on partition " << singlePartitionInfo.locationId << "\n";

            const int locationId = singlePartitionInfo.locationId;
            const int gpu = locationId;
            const int mydeviceId = deviceIds[gpu];
            cudaStream_t mystream = handle->streamsPerGpuLocation[gpu];
            cudaEvent_t myevent = handle->eventsPerGpuLocation[gpu];

            int oldDevice; CUDACHECK(cudaGetDevice(&oldDevice));
            
            
            if(mydeviceId == resultDeviceId){
                CUDACHECK(wrapperCudaSetDevice(resultDeviceId));

                distarraykernels::gatherKernel<<<640, 128, 0, syncstream>>>(
                    d_result, 
                    dataPtrPerLocation[gpu], 
                    d_indices, 
                    numIds,
                    -elementsPerLocationPS[gpu], 
                    resultPitch / sizeof(Value_t),
                    numColumns
                ); CUDACHECKASYNC;
            }else{
                registerDeviceIdForHandlenew(handle, resultDeviceId);

                CUDACHECK(wrapperCudaSetDevice(resultDeviceId));
                CUDACHECK(cudaEventRecord(handle->map_events[resultDeviceId], syncstream));

                CUDACHECK(wrapperCudaSetDevice(mydeviceId));

                auto& myGatherResult = handle->d_gatheredElementsOfGpuLocation[gpu];
                myGatherResult.resize(numIds * numColumns);

                CUDACHECK(cudaStreamWaitEvent(mystream, handle->map_events[resultDeviceId], 0));

                distarraykernels::gatherKernel<<<640, 128, 0, mystream>>>(
                    myGatherResult.get(), 
                    dataPtrPerLocation[gpu], 
                    d_indices, 
                    numIds,
                    -elementsPerLocationPS[gpu], 
                    numColumns,
                    numColumns
                ); CUDACHECKASYNC;

                distarraykernels::copy2Dkernel<Index_t, Value_t><<<640, 128, 0, mystream>>>(
                    d_result,
                    myGatherResult.get(),
                    numIds,
                    numColumns,
                    numColumns,
                    resultPitch / sizeof(Value_t)
                ); CUDACHECKASYNC;

                CUDACHECK(cudaEventRecord(myevent, mystream));
                CUDACHECK(cudaStreamWaitEvent(syncstream, myevent, 0));
            }

            CUDACHECK(wrapperCudaSetDevice(oldDevice));            
        }
    }  

    cudaGraphExec_t buildNoHostExecutionGraphByCapture(const GatherHandle& handle, int deviceId) const{
        int oldDeviceId = 0;
        CUDACHECK(cudaGetDevice(&oldDeviceId));
        CUDACHECK(cudaSetDevice(deviceId));

        cudaStream_t capturestream;
        CUDACHECK(cudaStreamCreate(&capturestream));

        CUDACHECK(cudaStreamBeginCapture(capturestream, cudaStreamCaptureModeRelaxed));

        auto& destination_event = handle->map_events[deviceId];
        auto& destination_stream = handle->map_streams[deviceId];
        
        CUDACHECK(cudaEventRecord(destination_event, capturestream));

        for(int gpu = 0; gpu < numGpus; gpu++){
            const int location = gpu;
            if(elementsPerLocation[location] > 0){
                const int gpuDeviceId = deviceIds[gpu];
                cudaStream_t gpuStream = handle->streamsPerGpuLocation[location];
                CUDACHECK(cudaSetDevice(gpuDeviceId));
                CUDACHECK(cudaStreamWaitEvent(gpuStream, destination_event, 0));

                CUDACHECK(cudaMemcpyAsync(
                    handle->map_d_gatherkernelparams[gpu].get(),
                    handle->map_h_gatherkernelparams[gpu].get(),
                    handle->map_h_gatherkernelparams[gpu].sizeInBytes(),
                    H2D,
                    gpuStream
                ));
            }
        }

        CUDACHECK(cudaSetDevice(deviceId));

        CUDACHECK(cudaMemcpyAsync(
            handle->map_d_packedKernelParamsPartPref[deviceId].get(),
            handle->h_packedKernelParamsPartPref.get(),
            handle->h_packedKernelParamsPartPref.sizeInBytes(),
            H2D,
            capturestream
        ));

        for(int gpu = 0; gpu < numGpus; gpu++){
            const int location = gpu;
            if(elementsPerLocation[location] > 0){
                CUDACHECK(cudaMemcpyAsync(
                    handle->map_d_scatterkernelparams[deviceId][gpu].get(),
                    handle->map_h_scatterkernelparams[deviceId][gpu].get(),
                    handle->map_h_scatterkernelparams[deviceId][gpu].sizeInBytes(),
                    H2D,
                    capturestream
                ));
            }
        }

        constexpr std::size_t paramsOffset = std::max(
            sizeof(distarraykernels::PartitionSplitKernelParams<Index_t>),
            sizeof(distarraykernels::PrefixSumKernelParams<Index_t>)
        );

        auto d_partitionsplitkernelParams 
            = (distarraykernels::PartitionSplitKernelParams<Index_t>*)handle->map_d_packedKernelParamsPartPref[deviceId].get();

        auto d_pskernelParams 
            = (distarraykernels::PrefixSumKernelParams<Index_t>*)(((char*)d_partitionsplitkernelParams) + paramsOffset);

        //find indices per location + prefixsum
        helpers::call_fill_kernel_async(
            handle->map_d_numIndicesPerLocation[deviceId].get(), 
            numLocations, 
            Index_t(0), 
            capturestream
        ); CUDACHECKASYNC;


        distarraykernels::partitionSplitKernel<Index_t, 32><<<1000, 256, 0, capturestream>>>(
            d_partitionsplitkernelParams
        ); CUDACHECKASYNC;

        distarraykernels::exclPrefixSumSingleThreadKernel<Index_t><<<1,1,0,capturestream>>>(
            d_pskernelParams
        ); CUDACHECKASYNC;

        CUDACHECK(cudaEventRecord(destination_event, capturestream));

        //gather data from gpu partitions into memory of the respective gpu, 
        for(int gpu = 0; gpu < numGpus; gpu++){
            const int location = gpu;
            if(elementsPerLocation[location] > 0){
                const int gpuDeviceId = deviceIds[gpu];
                cudaStream_t gpuStream = handle->streamsPerGpuLocation[location];
                cudaEvent_t gpuEvent = handle->eventsPerGpuLocation[location];

                CUDACHECK(wrapperCudaSetDevice(gpuDeviceId));
                CUDACHECK(cudaStreamWaitEvent(gpuStream, destination_event, 0));

                distarraykernels::gatherKernel<Index_t, Value_t><<<1000, 256, 0, gpuStream>>>(
                    handle->map_d_gatherkernelparams[gpu].get()
                ); CUDACHECKASYNC;

                CUDACHECK(cudaEventRecord(gpuEvent, gpuStream));
            }
        }
        // then scatter its gathered data to destination array via peer access
        CUDACHECK(wrapperCudaSetDevice(deviceId));

        for(int gpu = 0; gpu < numGpus; gpu++){
            const int location = gpu;
            if(elementsPerLocation[location] > 0){
                cudaEvent_t gpuEvent = handle->eventsPerGpuLocation[location];

                CUDACHECK(wrapperCudaSetDevice(deviceId));

                CUDACHECK(cudaStreamWaitEvent(capturestream, gpuEvent, 0));

                distarraykernels::scatterKernel<Index_t, Value_t><<<1000, 256, 0, capturestream>>>(
                    handle->map_d_scatterkernelparams[deviceId][gpu].get()
                ); CUDACHECKASYNC;
            }
        }

        cudaGraph_t graph;
        CUDACHECK(cudaStreamEndCapture(capturestream, &graph));
        
        cudaGraphExec_t execGraph;
        cudaGraphNode_t errorNode;
        auto logBuffer = std::make_unique<char[]>(1025);
        std::fill_n(logBuffer.get(), 1025, 0);
        cudaError_t status = cudaGraphInstantiate(&execGraph, graph, &errorNode, logBuffer.get(), 1025);
        if(status != cudaSuccess){
            if(logBuffer[1024] != '\0'){
                std::cerr << "cudaGraphInstantiate: truncated error message: ";
                std::copy_n(logBuffer.get(), 1025, std::ostream_iterator<char>(std::cerr, ""));
                std::cerr << "\n";
            }else{
                std::cerr << "cudaGraphInstantiate: error message: ";
                std::cerr << logBuffer.get();
                std::cerr << "\n";
            }
        }
        CUDACHECK(status);

        CUDACHECK(cudaGraphDestroy(graph));
        CUDACHECK(cudaStreamDestroy(capturestream));
        CUDACHECK(cudaSetDevice(oldDeviceId));

        return execGraph;
    }

    cudaGraphExec_t getNoHostExecutionGraph(const GatherHandle& handle, int deviceId) const{
        auto it = handle->map_nohostExecutionGraph.find(deviceId);

        if(it != handle->map_nohostExecutionGraph.end()){
            return it->second;
        }else{
            cudaGraphExec_t execGraph = buildNoHostExecutionGraphByCapture(handle, deviceId);
            handle->map_nohostExecutionGraph[deviceId] = execGraph;
            return execGraph;
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

        // assert(elementsPerLocation[hostLocation] == 0);
        // assert(!singlePartitionInfo.isSinglePartition); //there is a dedicated function for this case

        int oldId = 0;
        CUDACHECK(cudaGetDevice(&oldId));

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

        CUDACHECK(wrapperCudaSetDevice(resultDeviceId));
        CUDACHECK(cudaEventRecord(destination_event, syncstream));

        for(int i = 0; i < numLocations; i++){
            d_indicesForLocationsVector[i].resize(numIds);
            d_destinationPositionsForLocationsVector[i].resize(numIds);
        }

        for(int gpu = 0; gpu < numGpus; gpu++){
            handle->d_gatheredElementsOfGpuLocation[gpu].resize(numIds * numColumns);
        }

        Index_t** const pinnedPointersForLocations = (Index_t**)handle->h_packedpointersAndNumIndicesArg.get();
        Index_t** const pinnedPointersForLocations2 = pinnedPointersForLocations + numLocations;
        Index_t* const pinnedNumIndices = (Index_t*)(((char*)(pinnedPointersForLocations2 + numLocations))
                        + std::abs(int(sizeof(Index_t*)) - int(sizeof(Index_t)))); //proper pointer alignment

        Index_t** const d_indicesForLocationsPointers = (Index_t**)handle->map_d_packedpointersAndNumIndicesArg[resultDeviceId].get();
        Index_t** const d_destinationPositionsForLocationsPointers = d_indicesForLocationsPointers + numLocations;
        Index_t* const d_destination_numIndices = (Index_t*)(((char*)(d_destinationPositionsForLocationsPointers + numLocations))
                        + std::abs(int(sizeof(Index_t*)) - int(sizeof(Index_t)))); //proper pointer alignment

        for(int i = 0; i < numLocations; i++){
            pinnedPointersForLocations[i] = d_indicesForLocationsVector[i].get();
            pinnedPointersForLocations2[i] = d_destinationPositionsForLocationsVector[i].get();
        }

        *pinnedNumIndices = numIds;

        CUDACHECK(cudaMemcpyAsync(
            handle->map_d_packedpointersAndNumIndicesArg[resultDeviceId].get(),
            handle->h_packedpointersAndNumIndicesArg.get(),
            handle->h_packedpointersAndNumIndicesArg.sizeInBytes(),
            H2D,
            syncstream
        ));

        constexpr size_t paramsoffset = std::max(
            sizeof(distarraykernels::PartitionSplitKernelParams<Index_t>),
            sizeof(distarraykernels::PrefixSumKernelParams<Index_t>)
        );

        auto h_destination_partitionSplitKernelParams 
                = (distarraykernels::PartitionSplitKernelParams<Index_t>*)handle->h_packedKernelParamsPartPref.get();

        h_destination_partitionSplitKernelParams->splitIndices = d_indicesForLocationsPointers;
        h_destination_partitionSplitKernelParams->splitDestinationPositions = d_destinationPositionsForLocationsPointers;
        h_destination_partitionSplitKernelParams->numSplitIndicesPerLocation = d_destination_numIndicesPerLocation.get();
        h_destination_partitionSplitKernelParams->numLocations = numLocations;
        h_destination_partitionSplitKernelParams->elementsPerLocationPS = d_destination_elementsPerLocationPS.get();
        h_destination_partitionSplitKernelParams->numIdsPtr = d_destination_numIndices;
        h_destination_partitionSplitKernelParams->indices = d_indices;

        auto h_destination_prefixsumKernelParams 
                = (distarraykernels::PrefixSumKernelParams<Index_t>*)
                    (((char*)h_destination_partitionSplitKernelParams) + paramsoffset);

        h_destination_prefixsumKernelParams->output = d_destination_numIndicesPerLocationPS.get();
        h_destination_prefixsumKernelParams->input = d_destination_numIndicesPerLocation.get();
        h_destination_prefixsumKernelParams->numElements = numLocations;

        

        // auto& h_destination_partitionSplitKernelParams = handle->map_h_splitkernelparams[resultDeviceId][0];

        // h_destination_partitionSplitKernelParams.splitIndices = d_indicesForLocationsPointers;
        // h_destination_partitionSplitKernelParams.splitDestinationPositions = d_destinationPositionsForLocationsPointers;
        // h_destination_partitionSplitKernelParams.numSplitIndicesPerLocation = d_destination_numIndicesPerLocation.get();
        // h_destination_partitionSplitKernelParams.numLocations = numLocations;
        // h_destination_partitionSplitKernelParams.elementsPerLocationPS = d_destination_elementsPerLocationPS.get();
        // h_destination_partitionSplitKernelParams.numIdsPtr = d_destination_numIndices;
        // h_destination_partitionSplitKernelParams.indices = d_indices;

        // auto& h_destination_prefixsumKernelParams = handle->map_h_prefixsumkernelparams[resultDeviceId][0];

        // h_destination_prefixsumKernelParams.output = d_destination_numIndicesPerLocationPS.get();
        // h_destination_prefixsumKernelParams.input = d_destination_numIndicesPerLocation.get();
        // h_destination_prefixsumKernelParams.numElements = numLocations;


        for(int gpu = 0; gpu < numGpus; gpu++){
            const int location = gpu;

            //handle->d_gatheredElementsOfGpuLocation[gpu].resize(numIds * numColumns);

            auto& h_destination_scatterKernelParams = handle->map_h_scatterkernelparams[resultDeviceId][gpu][0];
            h_destination_scatterKernelParams.result = d_result;
            h_destination_scatterKernelParams.sourceData = handle->d_gatheredElementsOfGpuLocation[gpu].get();
            h_destination_scatterKernelParams.indices = d_destinationPositionsForLocationsVector[gpu].get();
            h_destination_scatterKernelParams.nIndicesPtr = d_destination_numIndicesPerLocation.get() + location;
            h_destination_scatterKernelParams.indexOffset = 0;
            h_destination_scatterKernelParams.resultPitchValueTs = resultPitch / sizeof(Value_t);
            h_destination_scatterKernelParams.numCols = numColumns;

            auto& h_gpu_gatherKernelParams = handle->map_h_gatherkernelparams[gpu][0];            
            h_gpu_gatherKernelParams.result = handle->d_gatheredElementsOfGpuLocation[gpu].get();
            h_gpu_gatherKernelParams.sourceData = dataPtrPerLocation[gpu];
            h_gpu_gatherKernelParams.indices = d_indicesForLocationsVector[location].get();
            h_gpu_gatherKernelParams.nIndicesPtr = d_destination_numIndicesPerLocation.get() + location;
            h_gpu_gatherKernelParams.indexOffset = -elementsPerLocationPS[location];
            h_gpu_gatherKernelParams.resultPitchValueTs = numColumns;
            h_gpu_gatherKernelParams.numCols = numColumns;
        }

        #if 1

        cudaGraphExec_t execGraph = getNoHostExecutionGraph(handle, resultDeviceId);
        CUDACHECK(cudaGraphLaunch(execGraph, syncstream));

        #else 

        for(int gpu = 0; gpu < numGpus; gpu++){
            const int location = gpu;
            if(elementsPerLocation[location] > 0){
                const int gpuDeviceId = deviceIds[gpu];
                cudaStream_t gpuStream = handle->streamsPerGpuLocation[location];
                CUDACHECK(cudaSetDevice(gpuDeviceId));
                CUDACHECK(cudaStreamWaitEvent(gpuStream, destination_event, 0));

                CUDACHECK(cudaMemcpyAsync(
                    handle->map_d_gatherkernelparams[gpu].get(),
                    handle->map_h_gatherkernelparams[gpu].get(),
                    handle->map_h_gatherkernelparams[gpu].sizeInBytes(),
                    H2D,
                    gpuStream
                ));
            }
        }

        CUDACHECK(cudaSetDevice(resultDeviceId));

        CUDACHECK(cudaMemcpyAsync(
            handle->map_d_packedKernelParamsPartPref[resultDeviceId].get(),
            handle->h_packedKernelParamsPartPref.get(),
            handle->h_packedKernelParamsPartPref.sizeInBytes(),
            H2D,
            syncstream
        ));

        for(int gpu = 0; gpu < numGpus; gpu++){
            const int location = gpu;
            if(elementsPerLocation[location] > 0){
                CUDACHECK(cudaMemcpyAsync(
                    handle->map_d_scatterkernelparams[resultDeviceId][gpu].get(),
                    handle->map_h_scatterkernelparams[resultDeviceId][gpu].get(),
                    handle->map_h_scatterkernelparams[resultDeviceId][gpu].sizeInBytes(),
                    H2D,
                    syncstream
                ));
            }
        }

        constexpr std::size_t paramsOffset = std::max(
            sizeof(distarraykernels::PartitionSplitKernelParams<Index_t>),
            sizeof(distarraykernels::PrefixSumKernelParams<Index_t>)
        );

        auto d_partitionsplitkernelParams 
            = (distarraykernels::PartitionSplitKernelParams<Index_t>*)handle->map_d_packedKernelParamsPartPref[resultDeviceId].get();

        auto d_pskernelParams 
            = (distarraykernels::PrefixSumKernelParams<Index_t>*)(((char*)d_partitionsplitkernelParams) + paramsOffset);

        //find indices per location + prefixsum
        helpers::call_fill_kernel_async(
            handle->map_d_numIndicesPerLocation[resultDeviceId].get(), 
            numLocations, 
            Index_t(0), 
            syncstream
        ); CUDACHECKASYNC;


        distarraykernels::partitionSplitKernel<Index_t, 32><<<1000, 256, 0, syncstream>>>(
            d_partitionsplitkernelParams
        ); CUDACHECKASYNC;

        distarraykernels::exclPrefixSumSingleThreadKernel<Index_t><<<1,1,0,syncstream>>>(
            d_pskernelParams
        ); CUDACHECKASYNC;

        CUDACHECK(cudaEventRecord(destination_event, syncstream));

        //gather data from gpu partitions into memory of the respective gpu, 
        for(int gpu = 0; gpu < numGpus; gpu++){
            const int location = gpu;
            if(elementsPerLocation[location] > 0){
                const int gpuDeviceId = deviceIds[gpu];
                cudaStream_t gpuStream = handle->streamsPerGpuLocation[location];
                cudaEvent_t gpuEvent = handle->eventsPerGpuLocation[location];

                CUDACHECK(wrapperCudaSetDevice(gpuDeviceId));
                CUDACHECK(cudaStreamWaitEvent(gpuStream, destination_event, 0));

                distarraykernels::gatherKernel<Index_t, Value_t><<<1000, 256, 0, gpuStream>>>(
                    handle->map_d_gatherkernelparams[gpu].get()
                ); CUDACHECKASYNC;

                CUDACHECK(cudaEventRecord(gpuEvent, gpuStream));
            }
        }
        // then scatter its gathered data to destination array via peer access
        CUDACHECK(wrapperCudaSetDevice(resultDeviceId));

        for(int gpu = 0; gpu < numGpus; gpu++){
            const int location = gpu;
            if(elementsPerLocation[location] > 0){
                cudaEvent_t gpuEvent = handle->eventsPerGpuLocation[location];

                CUDACHECK(wrapperCudaSetDevice(resultDeviceId));

                CUDACHECK(cudaStreamWaitEvent(syncstream, gpuEvent, 0));

                distarraykernels::scatterKernel<Index_t, Value_t><<<1000, 256, 0, syncstream>>>(
                    handle->map_d_scatterkernelparams[resultDeviceId][gpu].get()
                ); CUDACHECKASYNC;
            }
        }

        #endif
        CUDACHECK(cudaSetDevice(oldId));

    }    

    //does not need host indices
    template<class ParallelFor>
    void gatherElementsInGpuMemAsyncGeneral(ParallelFor&& forLoop,
                                    const GatherHandle& handle,
                                    const Index_t* d_indices,
                                    Index_t numIds,
                                    int resultDeviceId,
                                    Value_t* d_result,
                                    size_t resultPitch, // result element i begins at offset i * resultPitch
                                    cudaStream_t syncstream) const{
        if(numIds == 0){
            return;
        }

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

        CUDACHECK(wrapperCudaSetDevice(resultDeviceId));
        CUDACHECK(cudaEventRecord(destination_event, syncstream));

        for(int i = 0; i < numLocations; i++){
            d_indicesForLocationsVector[i].resize(numIds);
            d_destinationPositionsForLocationsVector[i].resize(numIds);
        }

        for(int gpu = 0; gpu < numGpus; gpu++){
            handle->d_gatheredElementsOfGpuLocation[gpu].resize(numIds * numColumns);
        }

        handle->pinnedIndicesOfHostLocation.resize(numIds);
        handle->pinnedGatheredElementsOfHostLocation.resize(numIds * numColumns);

        Index_t** const pinnedPointersForLocations = (Index_t**)handle->h_packedpointersAndNumIndicesArg.get();
        Index_t** const pinnedPointersForLocations2 = pinnedPointersForLocations + numLocations;
        Index_t* const pinnedNumIndices = (Index_t*)(((char*)(pinnedPointersForLocations2 + numLocations))
                        + std::abs(int(sizeof(Index_t*)) - int(sizeof(Index_t)))); //proper pointer alignment

        Index_t** const d_indicesForLocationsPointers = (Index_t**)handle->map_d_packedpointersAndNumIndicesArg[resultDeviceId].get();
        Index_t** const d_destinationPositionsForLocationsPointers = d_indicesForLocationsPointers + numLocations;
        Index_t* const d_destination_numIndices = (Index_t*)(((char*)(d_destinationPositionsForLocationsPointers + numLocations))
                        + std::abs(int(sizeof(Index_t*)) - int(sizeof(Index_t)))); //proper pointer alignment

        for(int i = 0; i < numLocations; i++){
            pinnedPointersForLocations[i] = d_indicesForLocationsVector[i].get();
            pinnedPointersForLocations2[i] = d_destinationPositionsForLocationsVector[i].get();
        }
        //fix indicesptr for host indices. kernel will write to it via pcie
        pinnedPointersForLocations[hostLocation] = handle->pinnedIndicesOfHostLocation.get();

        *pinnedNumIndices = numIds;

        CUDACHECK(cudaMemcpyAsync(
            handle->map_d_packedpointersAndNumIndicesArg[resultDeviceId].get(),
            handle->h_packedpointersAndNumIndicesArg.get(),
            handle->h_packedpointersAndNumIndicesArg.sizeInBytes(),
            H2D,
            syncstream
        ));

        constexpr size_t paramsoffset = std::max(
            sizeof(distarraykernels::PartitionSplitKernelParams<Index_t>),
            sizeof(distarraykernels::PrefixSumKernelParams<Index_t>)
        );

        auto h_destination_partitionSplitKernelParams 
                = (distarraykernels::PartitionSplitKernelParams<Index_t>*)handle->h_packedKernelParamsPartPref.get();

        h_destination_partitionSplitKernelParams->splitIndices = d_indicesForLocationsPointers;
        h_destination_partitionSplitKernelParams->splitDestinationPositions = d_destinationPositionsForLocationsPointers;
        h_destination_partitionSplitKernelParams->numSplitIndicesPerLocation = d_destination_numIndicesPerLocation.get();
        h_destination_partitionSplitKernelParams->numLocations = numLocations;
        h_destination_partitionSplitKernelParams->elementsPerLocationPS = d_destination_elementsPerLocationPS.get();
        h_destination_partitionSplitKernelParams->numIdsPtr = d_destination_numIndices;
        h_destination_partitionSplitKernelParams->indices = d_indices;

        auto h_destination_prefixsumKernelParams 
                = (distarraykernels::PrefixSumKernelParams<Index_t>*)
                    (((char*)h_destination_partitionSplitKernelParams) + paramsoffset);

        h_destination_prefixsumKernelParams->output = d_destination_numIndicesPerLocationPS.get();
        h_destination_prefixsumKernelParams->input = d_destination_numIndicesPerLocation.get();
        h_destination_prefixsumKernelParams->numElements = numLocations;

        for(int gpu = 0; gpu < numGpus; gpu++){
            const int location = gpu;

            auto& h_destination_scatterKernelParams = handle->map_h_scatterkernelparams[resultDeviceId][gpu][0];
            h_destination_scatterKernelParams.result = d_result;
            h_destination_scatterKernelParams.sourceData = handle->d_gatheredElementsOfGpuLocation[gpu].get();
            h_destination_scatterKernelParams.indices = d_destinationPositionsForLocationsVector[gpu].get();
            h_destination_scatterKernelParams.nIndicesPtr = d_destination_numIndicesPerLocation.get() + location;
            h_destination_scatterKernelParams.indexOffset = 0;
            h_destination_scatterKernelParams.resultPitchValueTs = resultPitch / sizeof(Value_t);
            h_destination_scatterKernelParams.numCols = numColumns;

            auto& h_gpu_gatherKernelParams = handle->map_h_gatherkernelparams[gpu][0];            
            h_gpu_gatherKernelParams.result = handle->d_gatheredElementsOfGpuLocation[gpu].get();
            h_gpu_gatherKernelParams.sourceData = dataPtrPerLocation[gpu];
            h_gpu_gatherKernelParams.indices = d_indicesForLocationsVector[location].get();
            h_gpu_gatherKernelParams.nIndicesPtr = d_destination_numIndicesPerLocation.get() + location;
            h_gpu_gatherKernelParams.indexOffset = -elementsPerLocationPS[location];
            h_gpu_gatherKernelParams.resultPitchValueTs = numColumns;
            h_gpu_gatherKernelParams.numCols = numColumns;
        }



        for(int gpu = 0; gpu < numGpus; gpu++){
            const int location = gpu;
            if(elementsPerLocation[location] > 0){
                const int gpuDeviceId = deviceIds[gpu];
                cudaStream_t gpuStream = handle->streamsPerGpuLocation[location];
                CUDACHECK(cudaSetDevice(gpuDeviceId));
                CUDACHECK(cudaStreamWaitEvent(gpuStream, destination_event, 0));

                CUDACHECK(cudaMemcpyAsync(
                    handle->map_d_gatherkernelparams[gpu].get(),
                    handle->map_h_gatherkernelparams[gpu].get(),
                    handle->map_h_gatherkernelparams[gpu].sizeInBytes(),
                    H2D,
                    gpuStream
                ));
            }
        }

        CUDACHECK(cudaSetDevice(resultDeviceId));

        CUDACHECK(cudaMemcpyAsync(
            handle->map_d_packedKernelParamsPartPref[resultDeviceId].get(),
            handle->h_packedKernelParamsPartPref.get(),
            handle->h_packedKernelParamsPartPref.sizeInBytes(),
            H2D,
            syncstream
        ));

        for(int gpu = 0; gpu < numGpus; gpu++){
            const int location = gpu;
            if(elementsPerLocation[location] > 0){
                CUDACHECK(cudaMemcpyAsync(
                    handle->map_d_scatterkernelparams[resultDeviceId][gpu].get(),
                    handle->map_h_scatterkernelparams[resultDeviceId][gpu].get(),
                    handle->map_h_scatterkernelparams[resultDeviceId][gpu].sizeInBytes(),
                    H2D,
                    syncstream
                ));
            }
        }

        constexpr std::size_t paramsOffset = std::max(
            sizeof(distarraykernels::PartitionSplitKernelParams<Index_t>),
            sizeof(distarraykernels::PrefixSumKernelParams<Index_t>)
        );

        auto d_partitionsplitkernelParams 
            = (distarraykernels::PartitionSplitKernelParams<Index_t>*)handle->map_d_packedKernelParamsPartPref[resultDeviceId].get();

        auto d_pskernelParams 
            = (distarraykernels::PrefixSumKernelParams<Index_t>*)(((char*)d_partitionsplitkernelParams) + paramsOffset);

        //find indices per location + prefixsum
        helpers::call_fill_kernel_async(
            handle->map_d_numIndicesPerLocation[resultDeviceId].get(), 
            numLocations, 
            Index_t(0), 
            syncstream
        ); CUDACHECKASYNC;


        distarraykernels::partitionSplitKernel<Index_t, 32><<<1000, 256, 0, syncstream>>>(
            d_partitionsplitkernelParams
        ); CUDACHECKASYNC;

        distarraykernels::exclPrefixSumSingleThreadKernel<Index_t><<<1,1,0,syncstream>>>(
            d_pskernelParams
        ); CUDACHECKASYNC;

        CUDACHECK(cudaEventRecord(destination_event, syncstream));

        CUDACHECK(cudaMemcpyAsync(
            handle->numIndicesOfHostLocation.get(),
            d_destination_numIndicesPerLocation.get() + hostLocation,
            sizeof(Index_t),
            D2H,
            syncstream
        ));

        //event record d2h
        CUDACHECK(cudaEventRecord(handle->map_d2hevents[resultDeviceId], syncstream));

        //gather data from gpu partitions into memory of the respective gpu, 
        for(int gpu = 0; gpu < numGpus; gpu++){
            const int location = gpu;
            if(elementsPerLocation[location] > 0){
                const int gpuDeviceId = deviceIds[gpu];
                cudaStream_t gpuStream = handle->streamsPerGpuLocation[location];
                cudaEvent_t gpuEvent = handle->eventsPerGpuLocation[location];

                CUDACHECK(wrapperCudaSetDevice(gpuDeviceId));
                CUDACHECK(cudaStreamWaitEvent(gpuStream, destination_event, 0));

                distarraykernels::gatherKernel<Index_t, Value_t><<<1000, 256, 0, gpuStream>>>(
                    handle->map_d_gatherkernelparams[gpu].get()
                ); CUDACHECKASYNC;

                CUDACHECK(cudaEventRecord(gpuEvent, gpuStream));
            }
        }
        // then scatter its gathered data to destination array via peer access
        CUDACHECK(wrapperCudaSetDevice(resultDeviceId));

        for(int gpu = 0; gpu < numGpus; gpu++){
            const int location = gpu;
            if(elementsPerLocation[location] > 0){
                cudaEvent_t gpuEvent = handle->eventsPerGpuLocation[location];

                CUDACHECK(wrapperCudaSetDevice(resultDeviceId));

                CUDACHECK(cudaStreamWaitEvent(syncstream, gpuEvent, 0));

                distarraykernels::scatterKernel<Index_t, Value_t><<<1000, 256, 0, syncstream>>>(
                    handle->map_d_scatterkernelparams[resultDeviceId][gpu].get()
                ); CUDACHECKASYNC;
            }
        }

        CUDACHECK(cudaEventSynchronize(handle->map_d2hevents[resultDeviceId]));

        const Index_t numIndicesForHost = *handle->numIndicesOfHostLocation.get();
        //std::cerr << "array type " << sizeof(Value_t) << ": numIndicesForHost = " << numIndicesForHost << "\n";
        if(numIndicesForHost > 0){
            CUDACHECK(wrapperCudaSetDevice(resultDeviceId));

            

            Value_t* const myResult = handle->pinnedGatheredElementsOfHostLocation.get();
            const Index_t* const myIndices = handle->pinnedIndicesOfHostLocation.get();

            // std::vector<Index_t> tmpvec(myIndices, myIndices + numIndicesForHost);
            // std::sort(tmpvec.begin(), tmpvec.end());
            // std::cerr << "first 3 host indices: ";
            // for(int i = 0; i < std::min(Index_t(3), numIndicesForHost); i++){
            //     std::cerr << tmpvec[i] << " ";
            // }
            // std::cerr << "\n";

            // std::cerr << "last 3 host indices: ";
            // for(int i = std::max(Index_t(0), numIndicesForHost-3); i < numIndicesForHost; i++){
            //     std::cerr << tmpvec[i] << " ";
            // }
            // std::cerr << "\n";

            auto gather = [&](Index_t begin, Index_t end, int /*threadId*/){
                nvtx::ScopedRange r("generalgather_host", 7);

                for(Index_t k = begin; k < end; k++){
                    const Index_t localId = myIndices[k] - elementsPerLocationPS[hostLocation];

                    const Value_t* const srcPtr = offsetPtr(dataPtrPerLocation[hostLocation], localId);
                    Value_t* const destPtr = myResult + size_t(k) * numColumns;

                    std::copy_n(srcPtr, numColumns, destPtr);
                }

            };

            forLoop( 
                Index_t(0), 
                numIndicesForHost, 
                gather
            );

            // forLoop(
            //     0, 
            //     numHostIndices, 
            //     [&](int begin, int end, int threadId){
            //         for(int k = begin; k < end; k++){
            //             const Index_t localId = handle->pinnedIndicesOfHostLocation[k] - elementsPerLocationPS[hostLocation];
            //             const Value_t* srcPtr = offsetPtr(dataPtrPerLocation[hostLocation], localId);
            //             Value_t* destPtr = myResult + size_t(k) * numColumns;
            //             std::copy_n(srcPtr, numColumns, destPtr);
            //         }
            //     }
            // );

            auto& h_destination_scatterKernelParams = handle->map_h_scatterkernelparams[resultDeviceId][0][0];
            h_destination_scatterKernelParams.result = d_result;
            h_destination_scatterKernelParams.sourceData = myResult;
            h_destination_scatterKernelParams.indices = d_destinationPositionsForLocationsVector[hostLocation].get();;
            h_destination_scatterKernelParams.nIndicesPtr = d_destination_numIndicesPerLocation.get() + hostLocation;
            h_destination_scatterKernelParams.indexOffset = 0;
            h_destination_scatterKernelParams.resultPitchValueTs = resultPitch / sizeof(Value_t);
            h_destination_scatterKernelParams.numCols = numColumns;

            CUDACHECK(cudaMemcpyAsync(
                handle->map_d_scatterkernelparams[resultDeviceId][0].get(),
                handle->map_h_scatterkernelparams[resultDeviceId][0].get(),
                handle->map_h_scatterkernelparams[resultDeviceId][0].sizeInBytes(),
                H2D,
                syncstream
            ));

            distarraykernels::scatterKernel<Index_t, Value_t><<<1000, 256, 0, syncstream>>>(
                handle->map_d_scatterkernelparams[resultDeviceId][0].get()
            ); CUDACHECKASYNC;
        }
    }

    //multiple partitions, including host partition
    template<class ParallelFor>
    void gatherElementsInGpuMemAsyncGeneral(ParallelFor&& forLoop,
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

        assert(!singlePartitionInfo.isSinglePartition); //there is a dedicated function for this case


        registerDeviceIdForHandlenew(handle, resultDeviceId);

        handle->pinnedIndicesOfHostLocation.resize(numIds);        
        handle->pinnedDestinationPositionsOfHostLocation.resize(numIds);

        //handle elements in gpu locations
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

        //handle elements in host location
        Index_t numIndicesForHost = 0;
        for(Index_t i = 0; i < numIds; i++){
            const Index_t index = indices[i];
            const int loc = getLocation(index);

            if(loc == hostLocation){
                handle->pinnedIndicesOfHostLocation[numIndicesForHost] = index;
                handle->pinnedDestinationPositionsOfHostLocation[numIndicesForHost] = i;
                numIndicesForHost++;
            }
        }

        if(numIndicesForHost > 0){
            CUDACHECK(wrapperCudaSetDevice(resultDeviceId));

            handle->pinnedGatheredElementsOfHostLocation.resize(numIndicesForHost * numColumns);
            handle->map_d_tmpResults[resultDeviceId].resize(numIndicesForHost * numColumns);
            handle->map_d_destinationPositionsOfGpu[resultDeviceId].resize(numIndicesForHost);

            *handle->numIndicesOfHostLocation.get() = numIndicesForHost;

            CUDACHECK(cudaMemcpyAsync(
                handle->map_d_numIndicesPerLocation[resultDeviceId].get() + hostLocation,
                handle->numIndicesOfHostLocation.get(),
                sizeof(Index_t),
                H2D,
                syncstream
            ));

            CUDACHECK(cudaMemcpyAsync(
                handle->map_d_destinationPositionsOfGpu[resultDeviceId].get(),
                handle->pinnedDestinationPositionsOfHostLocation.get(),
                sizeof(Index_t) * numIndicesForHost,
                H2D,
                syncstream
            ));

            Value_t* const myResult = handle->pinnedGatheredElementsOfHostLocation.get();
            const Index_t* const myIndices = handle->pinnedIndicesOfHostLocation.get();

            auto gather = [&](Index_t begin, Index_t end, int /*threadId*/){
                nvtx::ScopedRange r("generalgather_host", 7);
                const Index_t myChunksize = end - begin;

                //batches to overlap gather with cudaMemcpyAsync
                constexpr Index_t batchsize = 5000;
                const Index_t numBatches = SDIV(myChunksize, batchsize);

                for(Index_t b = 0; b < numBatches; b++){
                    const Index_t batchbegin = begin + b * batchsize;
                    const Index_t batchend = begin + std::min((b+1) * batchsize, myChunksize);
                    const Index_t currentBatchsize = batchend - batchbegin;

                    for(Index_t k = batchbegin; k < batchend; k++){
                        const Index_t localId = myIndices[k] - elementsPerLocationPS[hostLocation];

                        const Value_t* const srcPtr = offsetPtr(dataPtrPerLocation[hostLocation], localId);
                        Value_t* const destPtr = myResult + size_t(k) * numColumns;

                        std::copy_n(srcPtr, numColumns, destPtr);
                    }

                    CUDACHECK(cudaMemcpyAsync(
                        handle->map_d_tmpResults[resultDeviceId].get() + size_t(batchbegin) * numColumns,
                        myResult + size_t(batchbegin) * numColumns,
                        sizeof(Value_t) * currentBatchsize * numColumns,
                        H2D,
                        syncstream
                    ));
                }

                // for(Index_t k = begin; k < end; k++){
                //     const Index_t localId = myIndices[k] - elementsPerLocationPS[hostLocation];

                //     const Value_t* const srcPtr = offsetPtr(dataPtrPerLocation[hostLocation], localId);
                //     Value_t* const destPtr = myResult + size_t(k) * numColumns;

                //     std::copy_n(srcPtr, numColumns, destPtr);
                // }

            };

            forLoop( 
                Index_t(0), 
                numIndicesForHost, 
                gather
            );

            // CUDACHECK(cudaMemcpyAsync(
            //     handle->map_d_tmpResults[resultDeviceId].get(),
            //     myResult,
            //     sizeof(Value_t) * numIndicesForHost * numColumns,
            //     H2D,
            //     syncstream
            // ));

            distarraykernels::scatterKernel<Index_t, Value_t><<<1000, 256, 0, syncstream>>>(
                d_result, 
                handle->map_d_tmpResults[resultDeviceId].get(), 
                handle->map_d_destinationPositionsOfGpu[resultDeviceId].get(), 
                handle->map_d_numIndicesPerLocation[resultDeviceId].get() + hostLocation,
                0, 
                resultPitch / sizeof(Value_t),
                numColumns
            ); CUDACHECKASYNC;
        }
    }



    std::vector<Index_t> getPartitions() const{
        return elementsPerLocation;
    }

    std::size_t writeHostPartitionToStream(std::ostream& stream) const{
        const auto begin = dataPtrPerLocation[hostLocation];
        const std::size_t size = elementsPerLocation[hostLocation] * sizeOfElement;
        stream.write(reinterpret_cast<const char*>(begin), size);

        return size;
    }

    void readHostPartitionFromStream(std::istream& stream) const{
        auto begin = dataPtrPerLocation[hostLocation];
        const std::size_t size = elementsPerLocation[hostLocation] * sizeOfElement;
        stream.read(reinterpret_cast<char*>(begin), size);
    }

    std::size_t writeGpuPartitionToStream(int partition, std::ostream& stream) const{
        assert(0 <= partition);
        assert(partition < numGpus);

        constexpr std::int64_t MB = std::int64_t(1024) * 1024;
        constexpr std::int64_t safety = std::int64_t(64) * MB;
        constexpr std::int64_t maxBytes = std::int64_t(64) * MB;

        int currentId;
        CUDACHECK(cudaGetDevice(&currentId));

        CUDACHECK(wrapperCudaSetDevice(deviceIds[partition]));


        std::int64_t availableBytes = getAvailableMemoryInKB() * 1024;
        if(availableBytes > safety){
            availableBytes -= safety;
        }

        const std::int64_t bytesPerElement = sizeof(Value_t) * sizeOfElement;
        const std::int64_t buffersize = std::max(bytesPerElement, std::min(availableBytes, maxBytes));
        
        Value_t* buffer = nullptr;
        CUDACHECK(cudaMallocHost(&buffer, buffersize));

        std::size_t writtenBytes = 0;

        const std::int64_t batchsize = buffersize / bytesPerElement;
        const std::int64_t numBatches = SDIV(elementsPerLocation[partition], batchsize);
        for(std::int64_t batch = 0; batch < numBatches; batch++){
            std::int64_t begin = batch * batchsize;
            std::int64_t end = std::min(std::int64_t(elementsPerLocation[partition]), (batch + 1) * batchsize);
            const std::int64_t numElements = end-begin;

            const Value_t* src = offsetPtr(dataPtrPerLocation[partition], begin);
            //TIMERSTARTCPU(writeGpuPartitionToStream_memcpy);
            CUDACHECK(cudaMemcpy(buffer, src, sizeOfElement * numElements, D2H));
            //TIMERSTOPCPU(writeGpuPartitionToStream_memcpy);

            //TIMERSTARTCPU(writeGpuPartitionToStream_file);
            stream.write(reinterpret_cast<const char*>(buffer), sizeOfElement * numElements);
            //TIMERSTOPCPU(writeGpuPartitionToStream_file);

            writtenBytes += sizeOfElement * numElements;
        }

        CUDACHECK(cudaFreeHost(buffer));

        CUDACHECK(wrapperCudaSetDevice(currentId));

        return writtenBytes;
    }

    void readGpuPartitionFromStream(int partition, std::istream& stream){
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
        CUDACHECK(cudaMallocHost(&buffers[0], buffersize));
        CUDACHECK(cudaMallocHost(&buffers[1], buffersize));
        
        int currentId;
        CUDACHECK(cudaGetDevice(&currentId));
        CUDACHECK(wrapperCudaSetDevice(deviceIds[partition]));

        std::array<cudaStream_t, 2> streams;
        CUDACHECK(cudaStreamCreate(&streams[0]));
        CUDACHECK(cudaStreamCreate(&streams[1]));

        const std::int64_t batchsize = buffersize / bytesPerElement;
        const std::int64_t numBatches = SDIV(elementsPerLocation[partition], batchsize);

        int bufferindex = 0;

        for(std::int64_t batch = 0; batch < numBatches; batch++){
            std::int64_t begin = batch * batchsize;
            std::int64_t end = std::min(std::int64_t(elementsPerLocation[partition]), (batch + 1) * batchsize);
            const std::int64_t numElements = end-begin;

            CUDACHECK(cudaStreamSynchronize(streams[bufferindex]));

            //TIMERSTARTCPU(readGpuPartitionFromStream_file);
            stream.read(reinterpret_cast<char*>(buffers[bufferindex]), sizeOfElement * numElements);
            //TIMERSTOPCPU(readGpuPartitionFromStream_file);

            
            Value_t* dest = offsetPtr(dataPtrPerLocation[partition], begin);
            //TIMERSTARTCPU(readGpuPartitionFromStream_memcpy);
            CUDACHECK(cudaMemcpyAsync(dest, buffers[bufferindex], sizeOfElement * numElements, H2D, streams[bufferindex])); 
            //TIMERSTOPCPU(readGpuPartitionFromStream_memcpy);       

            bufferindex = bufferindex == 0 ? 1 : 0;        
        }

        CUDACHECK(cudaStreamSynchronize(streams[0]));
        CUDACHECK(cudaStreamSynchronize(streams[1]));

        CUDACHECK(cudaStreamDestroy(streams[0]));
        CUDACHECK(cudaStreamDestroy(streams[1]));

        CUDACHECK(cudaFreeHost(buffers[0]));
        CUDACHECK(cudaFreeHost(buffers[1]));

        CUDACHECK(wrapperCudaSetDevice(currentId));
    }

    std::vector<char> writeGpuPartitionToMemory(int partition) const{
        assert(0 <= partition);
        assert(partition < numGpus);

        cub::SwitchDevice sd{deviceIds[partition]};

        std::size_t bytes = getPartitionSizeInBytes(partition);
        std::vector<char> vec(bytes);

        CUDACHECK(cudaMemcpy(vec.data(), dataPtrPerLocation[partition], bytes, D2H));

        return vec;
    }

    void readGpuPartitionFromMemory(int partition, const std::vector<char>& savedpartition){

        cub::SwitchDevice sd{deviceIds[partition]};

        std::size_t bytes = savedpartition.size();
        CUDACHECK(cudaMemcpy(dataPtrPerLocation[partition], savedpartition.data(), bytes, H2D));
    }

    std::size_t writeGpuPartitionsToStream(std::ostream& stream) const{
        std::size_t bytes = 0;

        for(int gpu = 0; gpu < numGpus; gpu++){
            bytes += writeGpuPartitionToStream(gpu, stream);
        }

        return bytes;
    }

    void readGpuPartitionsFromStream(std::istream& stream){
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

    std::size_t writeToStream(std::ostream& stream) const{
        const std::size_t totalMemory = numRows * sizeOfElement;
        stream.write(reinterpret_cast<const char*>(&totalMemory), sizeof(std::size_t));

        std::size_t bytes = writeGpuPartitionsToStream(stream);
        bytes += writeHostPartitionToStream(stream);

        return totalMemory + bytes;
    }

    void readFromStream(std::istream& stream){
        std::size_t totalMemory = 1;
        stream.read(reinterpret_cast<char*>(&totalMemory), sizeof(std::size_t));
        
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

        cub::SwitchDevice sd{deviceIds[partition]};

        CUDACHECK(cudaFree(dataPtrPerLocation[partition]));
    }

    void allocateGpuPartition(int partition){
        assert(0 <= partition);
        assert(partition < numGpus);

        cub::SwitchDevice sd{deviceIds[partition]};
        CUDACHECK(cudaMalloc(&dataPtrPerLocation[partition], sizeOfElement * elementsPerLocation[partition]));
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
        int oldDevice; CUDACHECK(cudaGetDevice(&oldDevice));

        if(dataPtrPerLocation.size() > 0){

            for(int gpu = 0; gpu < numGpus; gpu++){
                CUDACHECK(wrapperCudaSetDevice(deviceIds[gpu]));
                if(debug) std::cerr << "DistributedArray::destroy device " << deviceIds[gpu] << " cudaFree(" << static_cast<void*>(dataPtrPerLocation[gpu]) << ")\n";
                CUDACHECK(cudaFree(dataPtrPerLocation[gpu]));
            }

            if(debug) std::cerr << "DistributedArray::destroy delete [](" << static_cast<void*>(dataPtrPerLocation[hostLocation]) << ")\n";
            delete [] dataPtrPerLocation[hostLocation];
        }

        CUDACHECK(wrapperCudaSetDevice(oldDevice));
    }
};





#endif








#endif
