#ifndef MULTI_GPU_ARRAY_HPP
#define MULTI_GPU_ARRAY_HPP


#include <hpc_helpers.cuh>
#include <gpu/cudaerrorcheck.cuh>
#include <gpu/singlegpu2darray.cuh>
#include <gpu/cudagraphhelpers.cuh>
#include <memorymanagement.hpp>
#include <gpu/multigputransfers.cuh>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include <limits>
#include <stdexcept>

#include <cub/cub.cuh>
#include <cooperative_groups.h>

#include <thrust/iterator/transform_iterator.h>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace cg = cooperative_groups;


namespace MultiGpu2dArrayKernels{

    template<class T, class IndexIterator>
    __global__
    void gatherKernel(
        const T* __restrict__ src, 
        size_t srcRowPitchInBytes, 
        size_t numColumns,
        size_t numRows,
        T* __restrict__ dest, 
        size_t destRowPitchInBytes, 
        IndexIterator indices, 
        size_t numIndices
    ){
        if(numIndices == 0) return;

        auto group = cg::this_grid();

        TwoDimensionalArray<T> array(
            numRows,
            numColumns,
            srcRowPitchInBytes,
            const_cast<T*>(src)
        );

        array.gather(group, dest, destRowPitchInBytes, indices, numIndices);
    }

    template<class T, class IndexIterator>
    __global__
    void gatherKernel(
        const T* __restrict__ src, 
        size_t srcRowPitchInBytes, 
        size_t numRows,
        size_t numColumns,
        T* __restrict__ dest, 
        size_t destRowPitchInBytes, 
        IndexIterator indices, 
        const size_t* __restrict__ numIndicesPtr
    ){
        const size_t numIndices = *numIndicesPtr;

        if(numIndices == 0) return;

        auto group = cg::this_grid();

        TwoDimensionalArray<T> array(
            numRows,
            numColumns,
            srcRowPitchInBytes,
            const_cast<T*>(src)
        );

        array.gather(group, dest, destRowPitchInBytes, indices, numIndices);
    }

    template<class T, class IndexIterator>
    __global__
    void scatterKernel(
        const T* __restrict__ src, 
        size_t srcRowPitchInBytes, 
        size_t numRows,
        size_t numColumns,
        T* __restrict__ dest, 
        size_t destRowPitchInBytes, 
        IndexIterator indices, 
        size_t numIndices
    ){
        if(numIndices == 0) return;

        auto group = cg::this_grid();

        TwoDimensionalArray<T> array(
            numRows,
            numColumns,
            destRowPitchInBytes,
            dest
        );

        array.scatter(group, src, srcRowPitchInBytes, indices, numIndices);
    }

    template<class T, class IndexIterator>
    __global__
    void scatterKernel(
        const T* __restrict__ src, 
        size_t srcRowPitchInBytes, 
        size_t numRows,
        size_t numColumns,
        T* __restrict__ dest, 
        size_t destRowPitchInBytes, 
        IndexIterator indices, 
        const size_t* __restrict__ numIndicesPtr
    ){
        const size_t numIndices = *numIndicesPtr;
        if(numIndices == 0) return;

        auto group = cg::this_grid();

        TwoDimensionalArray<T> array(
            numRows,
            numColumns,
            destRowPitchInBytes,
            dest
        );

        array.scatter(group, src, srcRowPitchInBytes, indices, numIndices);
    }

    template<class T>
    __global__
    void copyKernel(const T* __restrict__ src, const size_t* __restrict__ numElementsPtr, T* __restrict__ dst){
        const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t stride = gridDim.x * blockDim.x;

        const size_t numElements = *numElementsPtr;

        for(size_t i = tid; i < numElements; i += stride){
            dst[i] = src[i];
        }
    }

    template<class T>
    __global__
    void copyKernel(const T* __restrict__ src, size_t numElements, T* __restrict__ dst){
        const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t stride = gridDim.x * blockDim.x;

        for(size_t i = tid; i < numElements; i += stride){
            dst[i] = src[i];
        }
    }



    template<class IndexType, int maxNumGpus = 8>
    __global__
    void partitionSplitKernel(
        IndexType* __restrict__ splitIndices, // numIndices elements per partition
        size_t* __restrict__ splitDestinationPositions, // numIndices elements per partition
        size_t* __restrict__ numSplitIndicesPerPartition, 
        int numPartitions,
        const size_t* __restrict__ partitionOffsetsPS,
        size_t numIndices,
        const IndexType* __restrict__ indices
    ){

        assert(numPartitions <= maxNumGpus+1);

        using Index_t = size_t;

        auto atomicAggInc = [](auto& group, Index_t* counter){
            Index_t group_res;
            if(group.thread_rank() == 0){
                group_res = atomicAdd((unsigned long long*)counter, (unsigned long long)(group.size()));
            }
            return group.shfl(group_res, 0) + group.thread_rank();
        };

        //save prefixsum in registers
        Index_t reg_partitionOffsetsPS[maxNumGpus+1];

        #pragma unroll
        for(int i = 0; i < maxNumGpus+1; i++){
            if(i < numPartitions+1){
                reg_partitionOffsetsPS[i] = partitionOffsetsPS[i];
            }else{
                reg_partitionOffsetsPS[i] = partitionOffsetsPS[numPartitions];
            }
        }

        const Index_t numIds = numIndices;

        for(Index_t tid = threadIdx.x + blockIdx.x * Index_t(blockDim.x); 
                tid < numIds; 
                tid += Index_t(blockDim.x) * gridDim.x){
            
            const Index_t elementIndex = indices[tid];
            int location = -1;

            #pragma unroll
            for(int i = 0; i < maxNumGpus+1; i++){
                if(i < numPartitions 
                        && reg_partitionOffsetsPS[i] <= elementIndex 
                        && elementIndex < reg_partitionOffsetsPS[i+1]){
                    location = i;
                    break;
                }
            }

#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 700
            if(location != -1){
                auto g = cg::coalesced_threads();
                //partition into groups of threads with the same location. Cannot use a tiled_partition<N> as input, because for-loop may cause a deadlock:
                //"The implementation may cause the calling thread to wait until all the members of the parent group have invoked the operation before resuming execution."
                auto partitionGroup = cg::labeled_partition(g, location);

                const Index_t j = atomicAggInc(partitionGroup, &numSplitIndicesPerPartition[location]);
                splitIndices[location * numIds + j] = elementIndex;
                splitDestinationPositions[location * numIds + j] = tid;
            }
              
#else
            for(int i = 0; i < numPartitions; ++i){
                if(i == location){
                    auto group = cg::coalesced_threads();
                    const Index_t j = atomicAggInc(group, &numSplitIndicesPerPartition[i]);
                    splitIndices[i * numIds + j] = elementIndex;
                    splitDestinationPositions[i * numIds + j] = tid;
                    break;
                }
            }
#endif 

        }

    }

    template<class IndexType, int maxNumGpus = 8>
    __global__
    void partitionSplitKernel_splitsizeOnly(
        size_t* __restrict__ numSplitIndicesPerPartition, 
        int numPartitions,
        const size_t* __restrict__ partitionOffsetsPS,
        size_t numIndices,
        const IndexType* __restrict__ indices
    ){

        assert(numPartitions <= maxNumGpus+1);

        using Index_t = size_t;

        auto atomicAggInc = [](auto& group, Index_t* counter){
            Index_t group_res;
            if(group.thread_rank() == 0){
                group_res = atomicAdd((unsigned long long*)counter, (unsigned long long)(group.size()));
            }
            return group.shfl(group_res, 0) + group.thread_rank();
        };

        //save prefixsum in registers
        Index_t reg_partitionOffsetsPS[maxNumGpus+1];

        #pragma unroll
        for(int i = 0; i < maxNumGpus+1; i++){
            if(i < numPartitions+1){
                reg_partitionOffsetsPS[i] = partitionOffsetsPS[i];
            }else{
                reg_partitionOffsetsPS[i] = partitionOffsetsPS[numPartitions];
            }
        }

        const Index_t numIds = numIndices;

        for(Index_t tid = threadIdx.x + blockIdx.x * Index_t(blockDim.x); 
                tid < numIds; 
                tid += Index_t(blockDim.x) * gridDim.x){
            
            const Index_t elementIndex = indices[tid];
            int location = -1;

            #pragma unroll
            for(int i = 0; i < maxNumGpus+1; i++){
                if(i < numPartitions 
                        && reg_partitionOffsetsPS[i] <= elementIndex 
                        && elementIndex < reg_partitionOffsetsPS[i+1]){
                    location = i;
                    break;
                }
            }

#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 700
            if(location != -1){
                auto g = cg::coalesced_threads();
                //partition into groups of threads with the same location. Cannot use a tiled_partition<N> as input, because for-loop may cause a deadlock:
                //"The implementation may cause the calling thread to wait until all the members of the parent group have invoked the operation before resuming execution."
                auto partitionGroup = cg::labeled_partition(g, location);

                atomicAggInc(partitionGroup, &numSplitIndicesPerPartition[location]);
            }
              
#else
            for(int i = 0; i < numPartitions; ++i){
                if(i == location){
                    auto group = cg::coalesced_threads();
                    atomicAggInc(group, &numSplitIndicesPerPartition[i]);
                    break;
                }
            }
#endif 

        }

    }

    template<class T>
    struct LinearAccessFunctor{
        HOSTDEVICEQUALIFIER
        LinearAccessFunctor(T* ptr) : data(ptr){}

        HOSTDEVICEQUALIFIER
        T& operator()(size_t pos){
            return data[pos];
        }

        T* data;
    };

    template<class T>
    struct SubtractFunctor{
        T value;
        __host__ __device__
        SubtractFunctor(T val) : value(val){}

        __host__ __device__
        T operator()(const T& x) const noexcept{
            return x - value;
        }
    };
}




enum class MultiGpu2dArrayLayout {FirstFit, EvenShare};
enum class MultiGpu2dArrayInitMode {MustFitCompletely, CanDiscardRows};

template<class T, class IndexType = int>
class MultiGpu2dArray{
public:

    template<class W>
    using HostBuffer = helpers::SimpleAllocationPinnedHost<W>;
    template<class W>
    using DeviceBuffer = helpers::SimpleAllocationDevice<W>;
    //using DeviceBuffer = helpers::SimpleAllocationPinnedHost<W>;



    MultiGpu2dArray()
    : alignmentInBytes(sizeof(T)){

    }

    MultiGpu2dArray(
        size_t numRows, 
        size_t numColumns, 
        size_t alignmentInBytes,
        std::vector<int> dDeviceIds, // data resides on these gpus
        std::vector<size_t> memoryLimits,
        MultiGpu2dArrayLayout layout,
        MultiGpu2dArrayInitMode initMode = MultiGpu2dArrayInitMode::MustFitCompletely) 
    : alignmentInBytes(alignmentInBytes), 
        dataDeviceIds(dDeviceIds)
    {

        assert(alignmentInBytes > 0);
        //assert(alignmentInBytes % sizeof(T) == 0);

        init(numRows, numColumns, memoryLimits, layout, initMode);
    }

    ~MultiGpu2dArray(){
        destroy();
    }

    MultiGpu2dArray(const MultiGpu2dArray& rhs)
        : MultiGpu2dArray(rhs.numRows, rhs.numColumns, rhs.alignmentInBytes)
    {
        gpuArrays.clear();
        for(const auto& otherarray : rhs.gpuArrays){
            gpuArrays.emplace_back(otherarray);
        }
    }

    MultiGpu2dArray(MultiGpu2dArray&& other) noexcept
        : MultiGpu2dArray()
    {
        swap(*this, other);
    }

    MultiGpu2dArray& operator=(MultiGpu2dArray other){
        swap(*this, other);

        return *this;
    }

    friend void swap(MultiGpu2dArray& l, MultiGpu2dArray& r) noexcept
    {
        std::swap(l.numRows, r.numRows);
        std::swap(l.numColumns, r.numColumns);
        std::swap(l.alignmentInBytes, r.alignmentInBytes);
        std::swap(l.gpuArrays, r.gpuArrays);
        std::swap(l.dataDeviceIds, r.dataDeviceIds);
        std::swap(l.usedDeviceIds, r.usedDeviceIds);
        std::swap(l.h_numRowsPerGpu, r.h_numRowsPerGpu);
        std::swap(l.h_numRowsPerGpuPrefixSum, r.h_numRowsPerGpuPrefixSum);
    }

    void destroy(){
        numRows = 0;
        numColumns = 0;
        alignmentInBytes = 0;
        gpuArrays.clear();
        dataDeviceIds.clear();
        usedDeviceIds.clear();
        h_numRowsPerGpu.destroy();
        h_numRowsPerGpuPrefixSum.destroy();
    }

    struct HandleStruct{
        struct PerDevice{
            PerDevice()
            {        
                cudaGetDevice(&deviceId);
            }

            int deviceId{};

            CudaStream stream{};
            CudaEvent event{cudaEventDisableTiming};
        };

        struct PerCaller{           
            CudaEvent event{cudaEventDisableTiming};
            std::vector<CudaEvent> events{};
            std::vector<CudaStream> streams{};
        };

        HandleStruct(){
            CUDACHECK(cudaGetDevice(&deviceId));            
        }

        HandleStruct(HandleStruct&&) = default;
        HandleStruct(const HandleStruct&) = delete;

        int deviceId = 0;
        std::vector<PerDevice> deviceBuffers{};
        PerCaller callerBuffers{};
        helpers::SimpleAllocationPinnedHost<std::size_t> multisplitTemp{};

        MemoryUsage getMemoryInfo() const{
            MemoryUsage result{};

            return result;
        }
    };

    using Handle = std::shared_ptr<HandleStruct>;

    MemoryUsage getMemoryInfo(const Handle& handle) const{
        MemoryUsage result{};

        return handle->getMemoryInfo();
    }

    Handle makeHandle() const{
        Handle handle = std::make_shared<HandleStruct>();

        const int numDistinctGpus = usedDeviceIds.size();
        for(int d = 0; d < numDistinctGpus; d++){
            const int deviceId = usedDeviceIds[d];
            cub::SwitchDevice sd(deviceId);

            typename HandleStruct::PerDevice deviceBuffers; 

            handle->deviceBuffers.emplace_back(std::move(deviceBuffers));
        }

        if(h_numRowsPerGpu.size() > 0){

            auto& callerBuffers = handle->callerBuffers;
            callerBuffers.streams.resize(h_numRowsPerGpu.size());
            
            for(int i = 0; i < int(h_numRowsPerGpu.size()); i++){
                callerBuffers.events.emplace_back(cudaEventDisableTiming);
            }
            CUDACHECKASYNC;
        } 

        handle->multisplitTemp.resize(std::max(usedDeviceIds.size(), dataDeviceIds.size()));

        return handle;
    }

    bool tryReplication(std::vector<std::size_t> memoryLimits){
        if(!isReplicatedSingleGpu && usedDeviceIds.size() == 1 && usedDeviceIds.size() < dataDeviceIds.size()){
            assert(gpuArrays.size() == 1);

            auto memoryUsage = gpuArrays[0]->getMemoryInfo();
            const std::size_t requiredMemory = memoryUsage.device[gpuArrays[0]->getDeviceId()];

            std::vector<std::unique_ptr<Gpu2dArrayManaged<T>>> replicas;

            for(std::size_t i = usedDeviceIds.size(); i < dataDeviceIds.size(); i++){
                if(memoryLimits[i] < requiredMemory){
                    return false;
                }
                replicas.emplace_back(gpuArrays[0]->makeCopy(dataDeviceIds[i]));
            }

            bool ok = std::all_of(replicas.begin(), replicas.end(), [](const auto& uniqueptr){ return bool(uniqueptr); });

            if(ok){
                gpuArrays.insert(gpuArrays.end(), std::make_move_iterator(replicas.begin()), std::make_move_iterator(replicas.end()));
                usedDeviceIds.insert(usedDeviceIds.end(), dataDeviceIds.begin() + usedDeviceIds.size(), dataDeviceIds.end());
            }

            isReplicatedSingleGpu = ok;

            return ok;

        }else{
            return false;
        }
    }

    void gather(
        Handle& handle, 
        T* d_dest, 
        size_t destRowPitchInBytes, 
        const IndexType* d_indices, 
        size_t numIndices, 
        cudaStream_t destStream = 0
    ) const{

        if(getNumRows() == 0) return;

        gather_internal(
            handle,
            d_dest,
            destRowPitchInBytes,
            d_indices,
            numIndices,
            false,
            destStream
        );
    }

    void gather(
        Handle& handle, 
        T* d_dest, 
        size_t destRowPitchInBytes, 
        const IndexType* d_indices, 
        size_t numIndices, 
        bool mayContainInvalidIndices,
        cudaStream_t destStream = 0
    ) const{

        if(getNumRows() == 0) return;

        gather_internal(
            handle,
            d_dest,
            destRowPitchInBytes,
            d_indices,
            numIndices,
            mayContainInvalidIndices,
            destStream
        );
    }

    void scatter(
        Handle& handle, 
        const T* d_src, 
        size_t srcRowPitchInBytes, 
        const IndexType* d_indices, 
        size_t numIndices, 
        cudaStream_t srcStream = 0
    ) const{

        if(getNumRows() == 0) return;

        scatter_internal(
            handle,
            d_src,
            srcRowPitchInBytes,
            d_indices,
            numIndices,
            false,
            srcStream
        );
    }

    void scatter(
        Handle& handle, 
        const T* d_src, 
        size_t srcRowPitchInBytes, 
        const IndexType* d_indices, 
        size_t numIndices, 
        bool mayContainInvalidIndices,
        cudaStream_t srcStream = 0
    ) const{

        if(getNumRows() == 0) return;

        scatter_internal(
            handle,
            d_src,
            srcRowPitchInBytes,
            d_indices,
            numIndices,
            mayContainInvalidIndices,
            srcStream
        );
    }


    void gatherContiguous(Handle& handle, T* d_dest, size_t destRowPitchInBytes, size_t rowBegin, size_t numRowsToGather, cudaStream_t stream = 0) const{
        if(numRowsToGather == 0) return;

        std::size_t rowEnd = rowBegin + numRowsToGather;

        std::vector<std::size_t> numGatherPerDevice(usedDeviceIds.size(), 0);
        std::vector<std::size_t> firstRowPerDevice(usedDeviceIds.size(), 0);

        for(std::size_t i = 0, cur = rowBegin; i < usedDeviceIds.size(); i++){
            if(cur < h_numRowsPerGpuPrefixSum[i+1]){
                auto myEnd = std::min(h_numRowsPerGpuPrefixSum[i+1], rowEnd);
                const auto myNum = myEnd - cur;
                firstRowPerDevice[i] = cur - h_numRowsPerGpuPrefixSum[i];
                numGatherPerDevice[i] = myNum;
                cur += myNum;
            }
        }

        int oldDeviceId = 0;
        CUDACHECK(cudaGetDevice(&oldDeviceId));

        auto& callerBuffers = handle->callerBuffers;
        CUDACHECK(cudaEventRecord(callerBuffers.event, stream));

        #if 1

            for(std::size_t i = 0, offset = 0; i < usedDeviceIds.size(); i++){
                const std::size_t num = numGatherPerDevice[i];
                if(num > 0){
                    const auto& gpuArray = gpuArrays[i];

                    CUDACHECK(cudaStreamWaitEvent(callerBuffers.streams[i], callerBuffers.event));

                    gpuArray->gatherContiguousPeer(
                        oldDeviceId,
                        (T*)(((char*)d_dest) + destRowPitchInBytes * offset),
                        destRowPitchInBytes,
                        firstRowPerDevice[i],
                        num,
                        callerBuffers.streams[i]
                    );

                    CUDACHECK(cudaEventRecord(callerBuffers.events[i], callerBuffers.streams[i]));

                    offset += num;
                }
            }

            for(std::size_t i = 0; i < usedDeviceIds.size(); i++){
                const std::size_t num = numGatherPerDevice[i];
                if(num > 0){                
                    CUDACHECK(cudaStreamWaitEvent(stream, callerBuffers.events[i]));
                }
            }
        #else

        std::vector<rmm::device_uvector<T>> vec_d_gathered;

        for(std::size_t i = 0; i < usedDeviceIds.size(); i++){
            const std::size_t num = numGatherPerDevice[i];
            if(num > 0){
                const auto& gpuArray = gpuArrays[i];
                auto& deviceBuffers = handle->deviceBuffers[i];
                const int deviceId = gpuArray->getDeviceId();

                cub::SwitchDevice sd{deviceId};

                rmm::device_uvector<T> d_gathered(destRowPitchInBytes * num, deviceBuffers.stream.getStream());
                gpuArray->gatherContiguous(
                    d_gathered.data(),
                    destRowPitchInBytes,
                    firstRowPerDevice[i],
                    num,
                    deviceBuffers.stream
                );
                vec_d_gathered.push_back(std::move(d_gathered));
            }else{
                vec_d_gathered.emplace_back(0, stream);
            }
        }

        for(std::size_t i = 0, offset = 0; i < usedDeviceIds.size(); i++){
            const std::size_t num = numGatherPerDevice[i];
            if(num > 0){
                const auto& gpuArray = gpuArrays[i];
                auto& deviceBuffers = handle->deviceBuffers[i];
                const int deviceId = gpuArray->getDeviceId();

                cub::SwitchDevice sd{deviceId};
                CUDACHECK(cudaStreamWaitEvent(deviceBuffers.stream, callerBuffers.event));

                copy(
                    ((char*)d_dest) + destRowPitchInBytes * offset, 
                    oldDeviceId, 
                    vec_d_gathered[i].data(), 
                    deviceId, 
                    destRowPitchInBytes * num, 
                    deviceBuffers.stream
                );

                offset += num;
                vec_d_gathered[i].release();
                CUDACHECK(cudaEventRecord(deviceBuffers.event, deviceBuffers.stream));
                // for(auto id : {0,1}){
                //     cub::SwitchDevice sd{id};
                //     CUDACHECK(cudaDeviceSynchronize());
                // }
            }
        }

        for(std::size_t i = 0; i < usedDeviceIds.size(); i++){
            const std::size_t num = numGatherPerDevice[i];
            if(num > 0){
                
                CUDACHECK(cudaStreamWaitEvent(stream, handle->deviceBuffers[i].event));
            }
        }
        #endif
    }

    void scatterContiguous(Handle& handle, T* d_src, size_t srcRowPitchInBytes, size_t rowBegin, size_t numRowsToScatter, cudaStream_t stream = 0) const{
        if(numRowsToScatter == 0) return;

        std::size_t rowEnd = rowBegin + numRowsToScatter;

        std::vector<std::size_t> numScatterPerDevice(usedDeviceIds.size(), 0);
        std::vector<std::size_t> firstRowPerDevice(usedDeviceIds.size(), 0);

        for(std::size_t i = 0, cur = rowBegin; i < usedDeviceIds.size(); i++){
            if(cur < h_numRowsPerGpuPrefixSum[i+1]){
                auto myEnd = std::min(h_numRowsPerGpuPrefixSum[i+1], rowEnd);
                const auto myNum = myEnd - cur;
                firstRowPerDevice[i] = cur - h_numRowsPerGpuPrefixSum[i];
                numScatterPerDevice[i] = myNum;
                cur += myNum;
            }
        }

        int oldDeviceId = 0;
        CUDACHECK(cudaGetDevice(&oldDeviceId));

        auto& callerBuffers = handle->callerBuffers;
        CUDACHECK(cudaEventRecord(callerBuffers.event, stream));

        #if 1
            for(std::size_t i = 0, offset = 0; i < usedDeviceIds.size(); i++){
                const std::size_t num = numScatterPerDevice[i];
                if(num > 0){
                    const auto& gpuArray = gpuArrays[i];
                    auto& deviceBuffers = handle->deviceBuffers[i];
                    const int deviceId = gpuArray->getDeviceId();

                    CUDACHECK(cudaStreamWaitEvent(callerBuffers.streams[i], callerBuffers.event));

                    gpuArray->scatterContiguousPeer(
                        oldDeviceId,
                        (T*)(((char*)d_src) + srcRowPitchInBytes * offset),
                        srcRowPitchInBytes,
                        firstRowPerDevice[i],
                        num,
                        callerBuffers.streams[i]
                    );

                    CUDACHECK(cudaEventRecord(callerBuffers.events[i], callerBuffers.streams[i]));

                    offset += num;
                }
            }

            for(std::size_t i = 0; i < usedDeviceIds.size(); i++){
                const std::size_t num = numScatterPerDevice[i];
                if(num > 0){                
                    CUDACHECK(cudaStreamWaitEvent(stream, callerBuffers.events[i]));
                }
            }
        #else

        std::vector<rmm::device_uvector<T>> vec_d_scatter;

        //copy chunks to target devices
        for(std::size_t i = 0, offset = 0; i < usedDeviceIds.size(); i++){
            const std::size_t num = numScatterPerDevice[i];
            if(num > 0){
                const auto& gpuArray = gpuArrays[i];
                auto& deviceBuffers = handle->deviceBuffers[i];
                const int deviceId = gpuArray->getDeviceId();

                cub::SwitchDevice sd{deviceId};
                CUDACHECK(cudaStreamWaitEvent(deviceBuffers.stream, callerBuffers.event));
                rmm::device_uvector<T> d_scatter(srcRowPitchInBytes * num, deviceBuffers.stream.getStream());

                copy(
                    d_scatter.data(), 
                    deviceId,
                    ((char*)d_src) + srcRowPitchInBytes * offset,
                    oldDeviceId,
                    srcRowPitchInBytes * num, 
                    deviceBuffers.stream
                );

                offset += num;

                vec_d_scatter.push_back(std::move(d_scatter));
            }else{
                vec_d_scatter.emplace_back(0, stream);
            }
        }

        //scatter on target devices

        for(std::size_t i = 0; i < usedDeviceIds.size(); i++){
            const std::size_t num = numScatterPerDevice[i];
            if(num > 0){
                const auto& gpuArray = gpuArrays[i];
                auto& deviceBuffers = handle->deviceBuffers[i];
                const int deviceId = gpuArray->getDeviceId();

                cub::SwitchDevice sd{deviceId};

                gpuArray->scatterContiguous(
                    vec_d_scatter[i].data(),
                    srcRowPitchInBytes,
                    firstRowPerDevice[i],
                    num,
                    deviceBuffers.stream
                );
                vec_d_scatter[i].release();
                CUDACHECK(cudaEventRecord(deviceBuffers.event, deviceBuffers.stream));
            }
        }

        for(std::size_t i = 0; i < usedDeviceIds.size(); i++){
            const std::size_t num = numScatterPerDevice[i];
            if(num > 0){                
                CUDACHECK(cudaStreamWaitEvent(stream, handle->deviceBuffers[i].event));
            }
        }

        #endif

    }


    void multi_gather(
        std::vector<Handle>& vec_handle, 
        std::vector<T*>& vec_d_dest, 
        const std::vector<size_t>& vec_destRowPitchInBytes, 
        const std::vector<IndexType*>& vec_d_indices, 
        const std::vector<size_t>& vec_numIndices, 
        bool mayContainInvalidIndices,
        const std::vector<cudaStream_t>& destStreams,
        const std::vector<int>& destDeviceIds
    ) const{
        const int numDestGpus = destDeviceIds.size();

        for(int i = 1; i < numDestGpus; i++){
            if(vec_destRowPitchInBytes[i] != vec_destRowPitchInBytes[0]){
                throw std::runtime_error("multi_gather all row pitches must be equal");
            }
        }

        gather_collective_internal_7(
            vec_handle, 
            vec_d_dest, 
            vec_destRowPitchInBytes, 
            vec_d_indices, 
            vec_numIndices, 
            mayContainInvalidIndices,
            destStreams,
            destDeviceIds
        );

        // {
        //     int N = 0;
        //     CUDACHECK(cudaGetDeviceCount(&N));
        //     for(int i = 0; i < N; i++){
        //         cub::SwitchDevice sd(cudaSetDevice(i));
        //         CUDACHECK(cudaDeviceSynchronize());
        //     }
        // }
    }

    void multi_gather(
        std::vector<Handle>& vec_handle, 
        std::vector<T*>& vec_d_dest, 
        const std::vector<size_t>& vec_destRowPitchInBytes, 
        const std::vector<IndexType*>& vec_d_indices, 
        const std::vector<size_t>& vec_numIndices, 
        const std::vector<cudaStream_t>& destStreams,
        const std::vector<int>& destDeviceIds
    ) const{
        const int numDestGpus = destDeviceIds.size();

        for(int i = 1; i < numDestGpus; i++){
            if(vec_destRowPitchInBytes[i] != vec_destRowPitchInBytes[0]){
                throw std::runtime_error("multi_gather all row pitches must be equal");
            }
        }

        gather_collective_internal_7(
            vec_handle, 
            vec_d_dest, 
            vec_destRowPitchInBytes, 
            vec_d_indices, 
            vec_numIndices, 
            false,
            destStreams,
            destDeviceIds
        );

        // {
        //     int N = 0;
        //     CUDACHECK(cudaGetDeviceCount(&N));
        //     for(int i = 0; i < N; i++){
        //         cub::SwitchDevice sd(cudaSetDevice(i));
        //         CUDACHECK(cudaDeviceSynchronize());
        //     }
        // }
    }

    size_t getNumRows() const noexcept{
        return numRows;
    }

    size_t getNumColumns() const noexcept{
        return numColumns;
    }

    size_t getAlignmentInBytes() const noexcept{
        return alignmentInBytes;
    }

    bool isSingleGpu() const noexcept{
        return gpuArrays.size() == 1;
    }

    std::vector<size_t> getRowDistribution() const{
        std::vector<size_t> result(h_numRowsPerGpu.size());
        std::copy_n(h_numRowsPerGpu.get(), h_numRowsPerGpu.size(), result.begin());

        return result;
    }

    MemoryUsage getMemoryInfo() const{
        MemoryUsage result{};

        result.host = 0;
        for(const auto& ptr : gpuArrays){
            result += ptr->getMemoryInfo();
        }

        return result;
    }

private:

    void gather_internal(
        Handle& handle, T* d_dest, 
        size_t destRowPitchInBytes, 
        const IndexType* d_indices, 
        size_t numIndices, 
        bool mayContainInvalidIndices,
        cudaStream_t destStream
    ) const{
        if(numIndices == 0) return;

        int destDeviceId = 0;
        CUDACHECK(cudaGetDevice(&destDeviceId));

        if((isSingleGpu() || isReplicatedSingleGpu) && !mayContainInvalidIndices){
            auto it = std::find(usedDeviceIds.begin(), usedDeviceIds.end(), destDeviceId);
            if(it != usedDeviceIds.end()){
                //all data to be gathered resides on the destination device.
                const int position = std::distance(usedDeviceIds.begin(), it);
                
                gpuArrays[position]->gather(d_dest, destRowPitchInBytes, d_indices, numIndices, destStream);
            }else{
                //array was not constructed on current device. use peer access from any gpu

                const auto& gpuArray = gpuArrays[0];
                auto& deviceBuffers = handle->deviceBuffers[0];
                const int deviceId = gpuArray->getDeviceId();

                CUDACHECK(cudaEventRecord(handle->callerBuffers.event, destStream));
                
                CUDACHECK(cudaSetDevice(deviceId));
                CUDACHECK(cudaStreamWaitEvent(deviceBuffers.stream, handle->callerBuffers.event, 0));

                //copy indices to other gpu
                rmm::device_uvector<IndexType> d_indices_target(numIndices, deviceBuffers.stream.getStream());
                copy(
                    d_indices_target.data(), 
                    deviceId, 
                    d_indices, 
                    destDeviceId, 
                    sizeof(IndexType) * numIndices, 
                    deviceBuffers.stream
                );

                rmm::device_buffer d_temp(destRowPitchInBytes * numIndices, deviceBuffers.stream.getStream());
                gpuArrays[0]->gather(reinterpret_cast<T*>(d_temp.data()), destRowPitchInBytes, d_indices_target.data(), numIndices, deviceBuffers.stream);

                //copy gathered data to us
                copy(
                    d_dest, 
                    destDeviceId, 
                    d_temp.data(), 
                    deviceId, 
                    destRowPitchInBytes * numIndices, 
                    deviceBuffers.stream
                );

                CUDACHECK(cudaEventRecord(deviceBuffers.event, deviceBuffers.stream.getStream()));

                //wait on destStream until gathered data is ready
                CUDACHECK(cudaSetDevice(destDeviceId));
                CUDACHECK(cudaStreamWaitEvent(destStream, deviceBuffers.event, 0));
            }
        }else{

            //perform multisplit to distribute indices and filter out invalid indices. then perform local gathers,
            // and scatter into result array

            const int numGpus = gpuArrays.size();
            const int numDistinctGpus = h_numRowsPerGpu.size();

            if(!isReplicatedSingleGpu){
                assert(numGpus == numDistinctGpus);
            }else{
                assert(1 == numDistinctGpus);
            }

            auto& callerBuffers = handle->callerBuffers;

            MultiSplitResult splits = multiSplit(d_indices, numIndices, handle->multisplitTemp.data(), destStream);

            std::vector<std::size_t> numSelectedPrefixSum(numDistinctGpus,0);
            std::partial_sum(
                splits.h_numSelectedPerGpu.data(),
                splits.h_numSelectedPerGpu.data() + numDistinctGpus - 1,
                numSelectedPrefixSum.begin() + 1
            );

            rmm::device_uvector<char> callerBuffers_d_dataCommunicationBuffer(numIndices * destRowPitchInBytes, destStream);

            CUDACHECK(cudaEventRecord(handle->callerBuffers.event, destStream));

            for(int d = 0; d < numDistinctGpus; d++){
                const int num = splits.h_numSelectedPerGpu[d];
                if(num > 0){
                    CUDACHECK(cudaStreamWaitEvent(callerBuffers.streams[d], handle->callerBuffers.event));
                }
            }


            {
                std::vector<char*> all_rawPtrs(numDistinctGpus, nullptr);
                std::vector<std::size_t> all_allocatedBytes(numDistinctGpus, 0);
                std::vector<IndexType*> all_deviceBuffers_d_selectedIndices(numDistinctGpus, nullptr);
                std::vector<char*> all_deviceBuffers_d_dataCommunicationBuffer(numDistinctGpus, nullptr);
                std::vector<rmm::mr::device_memory_resource*> all_mrs(numDistinctGpus, nullptr);

                
                for(int d = 0; d < numDistinctGpus; d++){
                    const int num = splits.h_numSelectedPerGpu[d];

                    if(num > 0){
                        const auto& gpuArray = gpuArrays[d];
                        auto& deviceBuffers = handle->deviceBuffers[d];
                        const int deviceId = gpuArray->getDeviceId();

                        cub::SwitchDevice sd(deviceId);

                        //allocate remote buffers
                        all_mrs[d] = rmm::mr::get_current_device_resource();
                        size_t bytes[2]{
                            sizeof(IndexType) * num, // d_selectedIndices
                            destRowPitchInBytes * num //d_dataCommunicationBuffer
                        };
                        const size_t totalBytes = bytes[0] + bytes[1];
                        all_rawPtrs[d] = reinterpret_cast<char*>(all_mrs[d]->allocate(totalBytes, deviceBuffers.stream.getStream()));
                        all_allocatedBytes[d] = totalBytes;
                        all_deviceBuffers_d_selectedIndices[d] = reinterpret_cast<IndexType*>(all_rawPtrs[d]);
                        all_deviceBuffers_d_dataCommunicationBuffer[d] = reinterpret_cast<char*>(all_rawPtrs[d] + bytes[0]);

                        //copy selected indices to remote buffer

                        copy(
                            all_deviceBuffers_d_selectedIndices[d], 
                            deviceId, 
                            //splits.d_selectedIndices.data() + d * numIndices, 
                            splits.d_selectedIndices + d * numIndices, 
                            destDeviceId, 
                            sizeof(IndexType) * num, 
                            deviceBuffers.stream
                        );
                    }
                }

                //gather on remote gpus
                for(int d = 0; d < numDistinctGpus; d++){
                    const int num = splits.h_numSelectedPerGpu[d];

                    if(num > 0){
                        const auto& gpuArray = gpuArrays[d];
                        auto& deviceBuffers = handle->deviceBuffers[d];
                        const int deviceId = gpuArray->getDeviceId();

                        cub::SwitchDevice sd(deviceId);                      

                        auto gatherIndexIterator = thrust::make_transform_iterator(
                            all_deviceBuffers_d_selectedIndices[d],
                            MultiGpu2dArrayKernels::SubtractFunctor<IndexType>(h_numRowsPerGpuPrefixSum[d])
                        );

                        gpuArray->gather(
                            (T*)all_deviceBuffers_d_dataCommunicationBuffer[d], 
                            destRowPitchInBytes, 
                            gatherIndexIterator, 
                            num, 
                            deviceBuffers.stream
                        );
                    }
                }

                //copy remote gathered data to caller
                for(int d = 0; d < numDistinctGpus; d++){
                    const int num = splits.h_numSelectedPerGpu[d];

                    if(num > 0){
                        const auto& gpuArray = gpuArrays[d];
                        auto& deviceBuffers = handle->deviceBuffers[d];
                        const int deviceId = gpuArray->getDeviceId();

                        CUDACHECK(cudaSetDevice(deviceId));

                        copy(
                            callerBuffers_d_dataCommunicationBuffer.data() + destRowPitchInBytes * numSelectedPrefixSum[d], 
                            destDeviceId, 
                            all_deviceBuffers_d_dataCommunicationBuffer[d], 
                            deviceId, 
                            destRowPitchInBytes * num, 
                            deviceBuffers.stream
                        );

                        //free temp allocations of this device after data has been copied
                        all_mrs[d]->deallocate(all_rawPtrs[d], all_allocatedBytes[d], deviceBuffers.stream.getStream());

                        CUDACHECK(cudaEventRecord(deviceBuffers.event, deviceBuffers.stream));

                        CUDACHECK(cudaSetDevice(destDeviceId));
                        CUDACHECK(cudaStreamWaitEvent(callerBuffers.streams[d], deviceBuffers.event, 0));
                    }
                }

                //scatter into result array
                for(int d = 0; d < numDistinctGpus; d++){
                    const int num = splits.h_numSelectedPerGpu[d];

                    if(num > 0){

                        dim3 block(128, 1, 1);
                        dim3 grid(SDIV(numIndices, block.x), 1, 1);

                        MultiGpu2dArrayKernels::scatterKernel<<<grid, block, 0, callerBuffers.streams[d]>>>(
                            (const T*)(callerBuffers_d_dataCommunicationBuffer.data() + destRowPitchInBytes * numSelectedPrefixSum[d]), 
                            destRowPitchInBytes, 
                            numIndices,
                            getNumColumns(),
                            d_dest, 
                            destRowPitchInBytes, 
                            //splits.d_selectedPositions.data() + d * numIndices, 
                            splits.d_selectedPositions + d * numIndices, 
                            num
                        ); CUDACHECKASYNC; 

                        CUDACHECK(cudaEventRecord(callerBuffers.events[d], callerBuffers.streams[d]));
                    }
                }

                //join streams
                for(int d = 0; d < numDistinctGpus; d++){
                    const int num = splits.h_numSelectedPerGpu[d];

                    if(num > 0){
                        CUDACHECK(cudaStreamWaitEvent(destStream, callerBuffers.events[d], 0));
                    }                
                }             
            }
        }
    }

    void scatter_internal(
        Handle& handle, 
        const T* d_src, 
        size_t srcRowPitchInBytes, 
        const IndexType* d_indices, 
        size_t numIndices, 
        bool mayContainInvalidIndices,
        cudaStream_t srcStream = 0
    ) const{
        assert(!isReplicatedSingleGpu); //not implemented

        if(numIndices == 0) return;

        int srcDeviceId = 0;
        CUDACHECK(cudaGetDevice(&srcDeviceId));

        if(isSingleGpu() && !mayContainInvalidIndices && gpuArrays[0]->getDeviceId() == srcDeviceId){ //if all data should be scattered to same device

            gpuArrays[0]->scatter(d_src, srcRowPitchInBytes, d_indices, numIndices, srcStream);
            return;
        }else{
            const int numGpus = gpuArrays.size();
            const int numDistinctGpus = h_numRowsPerGpu.size();

            if(!isReplicatedSingleGpu){
                assert(numGpus == numDistinctGpus);
            }else{
                assert(1 == numDistinctGpus);
            }

            auto& callerBuffers = handle->callerBuffers;
            auto* srcMR = rmm::mr::get_current_device_resource();

            rmm::device_uvector<char> callerBuffers_d_dataCommunicationBuffer(numIndices * srcRowPitchInBytes, srcStream, srcMR);

            MultiSplitResult splits = multiSplit(d_indices, numIndices, handle->multisplitTemp.data(), srcStream);

            std::vector<std::size_t> numSelectedPrefixSum(numDistinctGpus,0);
            std::partial_sum(
                splits.h_numSelectedPerGpu.data(),
                splits.h_numSelectedPerGpu.data() + numDistinctGpus - 1,
                numSelectedPrefixSum.begin() + 1
            );

            for(int d = 0; d < numDistinctGpus; d++){
                const int numSelected = splits.h_numSelectedPerGpu[d];
                if(numSelected > 0){
                    const auto& gpuArray = gpuArrays[d];
                    const int deviceId = gpuArray->getDeviceId();
                    auto& targetDeviceBuffers = handle->deviceBuffers[d];  
                    auto* mr = rmm::mr::get_per_device_resource(rmm::cuda_device_id{deviceId});

                    //local gather data which should be scattered to current gpuArray
                    dim3 block(128, 1, 1);
                    dim3 grid(SDIV(numSelected, block.x), 1, 1);

                    MultiGpu2dArrayKernels::gatherKernel<<<grid, block, 0, callerBuffers.streams[d]>>>(
                        d_src, 
                        srcRowPitchInBytes, 
                        numIndices,
                        getNumColumns(),
                        (T*)(callerBuffers_d_dataCommunicationBuffer.data() + numSelectedPrefixSum[d] * srcRowPitchInBytes), 
                        srcRowPitchInBytes, 
                        //splits.d_selectedPositions.data() + numIndices * d, 
                        splits.d_selectedPositions + numIndices * d, 
                        //splits.d_numSelectedPerGpu.data() + d
                        splits.d_numSelectedPerGpu + d
                    ); CUDACHECKASYNC;

                    CUDACHECK(cudaEventRecord(callerBuffers.events[d], callerBuffers.streams[d]));

                    cub::SwitchDevice sd{deviceId};
                    CUDACHECK(cudaStreamWaitEvent(targetDeviceBuffers.stream, callerBuffers.events[d], 0));


                    IndexType* d_target_selected_indices = reinterpret_cast<IndexType*>(mr->allocate(sizeof(IndexType) * numSelected, targetDeviceBuffers.stream.getStream()));
                    char* d_target_communicationBuffer = reinterpret_cast<char*>(mr->allocate(sizeof(char) * numSelected * srcRowPitchInBytes, targetDeviceBuffers.stream.getStream()));
                    std::size_t* d_target_numSelected = reinterpret_cast<std::size_t*>(mr->allocate(sizeof(std::size_t), targetDeviceBuffers.stream.getStream()));

                    copy(
                        d_target_numSelected, 
                        deviceId, 
                        //splits.d_numSelectedPerGpu.data() + d, 
                        splits.d_numSelectedPerGpu + d, 
                        srcDeviceId, 
                        sizeof(size_t), 
                        targetDeviceBuffers.stream
                    );

                    copy(
                        d_target_selected_indices, 
                        deviceId, 
                        //splits.d_selectedIndices.data() + numIndices * d, 
                        splits.d_selectedIndices + numIndices * d, 
                        srcDeviceId, 
                        sizeof(IndexType) * numSelected, 
                        targetDeviceBuffers.stream
                    );

                    copy(
                        d_target_communicationBuffer, 
                        deviceId, 
                        callerBuffers_d_dataCommunicationBuffer.data() + numSelectedPrefixSum[d] * srcRowPitchInBytes, 
                        srcDeviceId, 
                        srcRowPitchInBytes * numSelected, 
                        targetDeviceBuffers.stream
                    );

                    auto scatterIndexIterator = thrust::make_transform_iterator(
                        d_target_selected_indices,
                        MultiGpu2dArrayKernels::SubtractFunctor<IndexType>(h_numRowsPerGpuPrefixSum[d])
                    );

                    gpuArray->scatter(
                        (const T*)d_target_communicationBuffer, 
                        srcRowPitchInBytes, 
                        scatterIndexIterator, 
                        d_target_numSelected,
                        numSelected, 
                        targetDeviceBuffers.stream
                    );

                    mr->deallocate(d_target_selected_indices, sizeof(IndexType) * numSelected, targetDeviceBuffers.stream.getStream());
                    mr->deallocate(d_target_communicationBuffer, sizeof(char) * numSelected * srcRowPitchInBytes, targetDeviceBuffers.stream.getStream());
                    mr->deallocate(d_target_numSelected, sizeof(std::size_t), targetDeviceBuffers.stream.getStream());

                    CUDACHECK(cudaEventRecord(targetDeviceBuffers.event, targetDeviceBuffers.stream));
                }
            }

            for(int d = 0; d < numDistinctGpus; d++){
                const int numSelected = splits.h_numSelectedPerGpu[d];
                if(numSelected > 0){
                    CUDACHECK(cudaStreamWaitEvent(srcStream, handle->deviceBuffers[d].event, 0));
                }
            }
        }
    }

public:
    void gather_collective_internal_1(
        std::vector<Handle>& vec_handle, 
        std::vector<T*>& vec_d_dest, 
        const std::vector<size_t>& vec_destRowPitchInBytes, 
        const std::vector<IndexType*>& vec_d_indices, 
        const std::vector<size_t>& vec_numIndices, 
        const std::vector<cudaStream_t>& destStreams,
        const std::vector<int>& destDeviceIds
    ) const{
        const int numDestGpus = destDeviceIds.size();

        if((isSingleGpu() || isReplicatedSingleGpu)){
            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd(destDeviceId);

                    auto it = std::find(usedDeviceIds.begin(), usedDeviceIds.end(), destDeviceId);
                    if(it != usedDeviceIds.end()){
                        //all data to be gathered resides on the destination device.
                        const int position = std::distance(usedDeviceIds.begin(), it);
                        
                        gpuArrays[position]->gather(
                            vec_d_dest[g], 
                            vec_destRowPitchInBytes[g], 
                            vec_d_indices[g], 
                            vec_numIndices[g], 
                            destStreams[g]
                        );
                    }else{
                        //array was not constructed on current device. use peer access from any gpu

                        const auto& gpuArray = gpuArrays[0];
                        auto& deviceBuffers = vec_handle[g]->deviceBuffers[0];
                        const int deviceId = gpuArray->getDeviceId();

                        CUDACHECK(cudaEventRecord(vec_handle[g]->callerBuffers.event, destStreams[g]));
                        
                        CUDACHECK(cudaSetDevice(deviceId));
                        CUDACHECK(cudaStreamWaitEvent(deviceBuffers.stream, vec_handle[g]->callerBuffers.event, 0));

                        //copy indices to other gpu
                        rmm::device_uvector<IndexType> d_indices_target(vec_numIndices[g], deviceBuffers.stream.getStream());
                        copy(
                            d_indices_target.data(), 
                            deviceId, 
                            vec_d_indices[g], 
                            destDeviceId, 
                            sizeof(IndexType) * vec_numIndices[g], 
                            deviceBuffers.stream
                        );

                        rmm::device_buffer d_temp(vec_destRowPitchInBytes[g] * vec_numIndices[g], deviceBuffers.stream.getStream());
                        gpuArrays[0]->gather(reinterpret_cast<T*>(d_temp.data()), vec_destRowPitchInBytes[g], d_indices_target.data(), vec_numIndices[g], deviceBuffers.stream);

                        //copy gathered data to us
                        copy(
                            vec_d_dest[g], 
                            destDeviceId, 
                            d_temp.data(), 
                            deviceId, 
                            vec_destRowPitchInBytes[g] * vec_numIndices[g], 
                            deviceBuffers.stream
                        );

                        CUDACHECK(cudaEventRecord(deviceBuffers.event, deviceBuffers.stream.getStream()));

                        //wait on destStream until gathered data is ready
                        CUDACHECK(cudaSetDevice(destDeviceId));
                        CUDACHECK(cudaStreamWaitEvent(destStreams[g], deviceBuffers.event, 0));
                    }
                }
            }
        }else{

            //perform multisplit to distribute indices and filter out invalid indices. then perform local gathers,
            // and scatter into result array

            const int numGpus = gpuArrays.size();
            const int numDistinctGpus = h_numRowsPerGpu.size();

            if(!isReplicatedSingleGpu){
                assert(numGpus == numDistinctGpus);
            }else{
                assert(1 == numDistinctGpus);
            }

            // std::vector<MultiSplitResult> splitsPerDestGpu_fooo;
            // for(int g = 0; g < numDestGpus; g++){
            //     if(vec_numIndices[g] > 0){
            //         const int destDeviceId = destDeviceIds[g];
            //         cub::SwitchDevice sd_{destDeviceId};
                    
            //         splitsPerDestGpu_fooo.emplace_back(vec_numIndices[g], numDistinctGpus, destStreams[g]);
            //     }else{
            //         splitsPerDestGpu_fooo.emplace_back(0, numDistinctGpus, destStreams[g]);
            //     }
            // }
            // CudaGraph cudaGraph; 
            // auto func = [&](cudaStream_t stream){
            //     CudaEvent event1{cudaEventDisableTiming};
            //     CUDACHECK(cudaEventRecord(event1, stream));
                
            //     for(int g = 0; g < numDestGpus; g++){
            //         if(vec_numIndices[g] > 0){
            //             const int destDeviceId = destDeviceIds[g];
            //             cub::SwitchDevice sd_{destDeviceId};
                        
            //             auto& handle = vec_handle[g];
            //             auto& result = splitsPerDestGpu_fooo[g];
                        
            //             CUDACHECK(cudaStreamWaitEvent(destStreams[g], event1, 0));
                        
            //             CUDACHECK(cudaMemsetAsync(result.d_numSelectedPerGpu, 0, sizeof(std::size_t) * numDistinctGpus, destStreams[g]));
            //             MultiGpu2dArrayKernels::partitionSplitKernel<<<SDIV(vec_numIndices[g], 128), 128, 0, destStreams[g]>>>(
            //                 result.d_selectedIndices,
            //                 result.d_selectedPositions,
            //                 result.d_numSelectedPerGpu,
            //                 numDistinctGpus,
            //                 h_numRowsPerGpuPrefixSum.data(),
            //                 vec_numIndices[g],
            //                 vec_d_indices[g]
            //             ); CUDACHECKASYNC;
                        
            //             CUDACHECK(cudaMemcpyAsync(
            //                 handle->multisplitTemp.data(),
            //                 result.d_numSelectedPerGpu,
            //                 sizeof(std::size_t) * numDistinctGpus,
            //                 D2H,
            //                 destStreams[g]
            //             ));
                        
            //             CudaEvent event2{cudaEventDisableTiming};
            //             CUDACHECK(cudaEventRecord(event2, destStreams[g]));
            //             CUDACHECK(cudaStreamWaitEvent(stream, event2, 0));
            //         }
            //     }
            // };
            // cudaGraph.capture(func);
            // cudaGraph.execute(destStreams[0]);
            
            std::vector<MultiSplitResult> splitsPerDestGpu;
            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd_{destDeviceId};
                    
                    auto& handle = vec_handle[g];
                    
                    MultiSplitResult splits = multiSplit_no_h_numSelectedPerGpu(
                        vec_d_indices[g], 
                        vec_numIndices[g], 
                        handle->multisplitTemp.data(), 
                        destStreams[g]
                    );
                    splitsPerDestGpu.push_back(std::move(splits));
                }else{
                    splitsPerDestGpu.emplace_back(0, numDistinctGpus, destStreams[g]);
                }
            }

            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd_{destDeviceId};
                    auto& handle = vec_handle[g];
                    CUDACHECK(cudaStreamSynchronize(destStreams[g]));
                    std::copy(handle->multisplitTemp.data(), handle->multisplitTemp.data() + numDistinctGpus, splitsPerDestGpu[g].h_numSelectedPerGpu.begin());
                }else{
                    std::fill(splitsPerDestGpu[g].h_numSelectedPerGpu.begin(), splitsPerDestGpu[g].h_numSelectedPerGpu.end(), 0);
                }
            }

            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd_{destDeviceId};

                    auto& handle = vec_handle[g];
                    auto& callerBuffers = handle->callerBuffers;

                    //MultiSplitResult splits = multiSplit(vec_d_indices[g], vec_numIndices[g], handle->multisplitTemp.data(), destStreams[g]);
                    MultiSplitResult& splits = splitsPerDestGpu[g];

                    std::vector<std::size_t> numSelectedPrefixSum(numDistinctGpus,0);
                    std::partial_sum(
                        splits.h_numSelectedPerGpu.data(),
                        splits.h_numSelectedPerGpu.data() + numDistinctGpus - 1,
                        numSelectedPrefixSum.begin() + 1
                    );

                    rmm::device_uvector<char> callerBuffers_d_dataCommunicationBuffer(vec_numIndices[g] * vec_destRowPitchInBytes[g], destStreams[g]);

                    CUDACHECK(cudaEventRecord(handle->callerBuffers.event, destStreams[g]));

                    for(int d = 0; d < numDistinctGpus; d++){
                        const int num = splits.h_numSelectedPerGpu[d];
                        if(num > 0){
                            CUDACHECK(cudaStreamWaitEvent(callerBuffers.streams[d], handle->callerBuffers.event));
                        }
                    }


                    {
                        std::vector<char*> all_rawPtrs(numDistinctGpus, nullptr);
                        std::vector<std::size_t> all_allocatedBytes(numDistinctGpus, 0);
                        std::vector<IndexType*> all_deviceBuffers_d_selectedIndices(numDistinctGpus, nullptr);
                        std::vector<char*> all_deviceBuffers_d_dataCommunicationBuffer(numDistinctGpus, nullptr);
                        std::vector<rmm::mr::device_memory_resource*> all_mrs(numDistinctGpus, nullptr);
        
                        for(int d = 0; d < numDistinctGpus; d++){
                            const int num = splits.h_numSelectedPerGpu[d];
                            
                            if(num > 0){
                                const auto& gpuArray = gpuArrays[d];
                                auto& deviceBuffers = handle->deviceBuffers[d];
                                const int deviceId = gpuArray->getDeviceId();
                                
                                cub::SwitchDevice sd(deviceId);
                                
                                //allocate remote buffers
                                all_mrs[d] = rmm::mr::get_current_device_resource();
                                size_t bytes[2]{
                                    sizeof(IndexType) * num, // d_selectedIndices
                                    vec_destRowPitchInBytes[g] * num //d_dataCommunicationBuffer
                                };
                                const size_t totalBytes = bytes[0] + bytes[1];
                                all_rawPtrs[d] = reinterpret_cast<char*>(all_mrs[d]->allocate(totalBytes, deviceBuffers.stream.getStream()));
                                all_allocatedBytes[d] = totalBytes;
                                all_deviceBuffers_d_selectedIndices[d] = reinterpret_cast<IndexType*>(all_rawPtrs[d]);
                                all_deviceBuffers_d_dataCommunicationBuffer[d] = reinterpret_cast<char*>(all_rawPtrs[d] + bytes[0]);
        
                        //     }
                        // }

                        // //copy selected indices to remote buffer
                        // for(int d = 0; d < numDistinctGpus; d++){
                        //     const int num = splits.h_numSelectedPerGpu[d];

                        //     if(num > 0){
                        //         const auto& gpuArray = gpuArrays[d];
                        //         auto& deviceBuffers = handle->deviceBuffers[d];
                        //         const int deviceId = gpuArray->getDeviceId();

                        //         cub::SwitchDevice sd(deviceId);

                                //copy selected indices to remote device
                                copy(
                                    all_deviceBuffers_d_selectedIndices[d], 
                                    deviceId, 
                                    splits.d_selectedIndices + d * vec_numIndices[g], 
                                    destDeviceId, 
                                    sizeof(IndexType) * num, 
                                    deviceBuffers.stream
                                );
                            }
                        }

                        //gather on remote gpus
                        for(int d = 0; d < numDistinctGpus; d++){
                            const int num = splits.h_numSelectedPerGpu[d];

                            if(num > 0){
                                const auto& gpuArray = gpuArrays[d];
                                auto& deviceBuffers = handle->deviceBuffers[d];
                                const int deviceId = gpuArray->getDeviceId();

                                cub::SwitchDevice sd(deviceId);                      

                                auto gatherIndexIterator = thrust::make_transform_iterator(
                                    all_deviceBuffers_d_selectedIndices[d],
                                    MultiGpu2dArrayKernels::SubtractFunctor<IndexType>(h_numRowsPerGpuPrefixSum[d])
                                );

                                gpuArray->gather(
                                    (T*)all_deviceBuffers_d_dataCommunicationBuffer[d], 
                                    vec_destRowPitchInBytes[g], 
                                    gatherIndexIterator, 
                                    num, 
                                    deviceBuffers.stream
                                );
                            }
                        }

                        //copy remote gathered data to caller
                        for(int d = 0; d < numDistinctGpus; d++){
                            const int num = splits.h_numSelectedPerGpu[d];

                            if(num > 0){
                                const auto& gpuArray = gpuArrays[d];
                                auto& deviceBuffers = handle->deviceBuffers[d];
                                const int deviceId = gpuArray->getDeviceId();

                                CUDACHECK(cudaSetDevice(deviceId));

                                copy(
                                    callerBuffers_d_dataCommunicationBuffer.data() + vec_destRowPitchInBytes[g] * numSelectedPrefixSum[d], 
                                    destDeviceId, 
                                    all_deviceBuffers_d_dataCommunicationBuffer[d], 
                                    deviceId, 
                                    vec_destRowPitchInBytes[g] * num, 
                                    deviceBuffers.stream
                                );

                                //free temp allocations of this device after data has been copied
                                all_mrs[d]->deallocate(all_rawPtrs[d], all_allocatedBytes[d], deviceBuffers.stream.getStream());


                                CUDACHECK(cudaEventRecord(deviceBuffers.event, deviceBuffers.stream));

                                CUDACHECK(cudaSetDevice(destDeviceId));
                                CUDACHECK(cudaStreamWaitEvent(callerBuffers.streams[d], deviceBuffers.event, 0));
                            }
                        }

                        //scatter into result array
                        for(int d = 0; d < numDistinctGpus; d++){
                            const int num = splits.h_numSelectedPerGpu[d];

                            if(num > 0){

                                dim3 block(128, 1, 1);
                                dim3 grid(SDIV(vec_numIndices[g], block.x), 1, 1);

                                MultiGpu2dArrayKernels::scatterKernel<<<grid, block, 0, callerBuffers.streams[d]>>>(
                                    (const T*)(callerBuffers_d_dataCommunicationBuffer.data() + vec_destRowPitchInBytes[g] * numSelectedPrefixSum[d]), 
                                    vec_destRowPitchInBytes[g], 
                                    vec_numIndices[g],
                                    getNumColumns(),
                                    vec_d_dest[g], 
                                    vec_destRowPitchInBytes[g], 
                                    splits.d_selectedPositions + d * vec_numIndices[g], 
                                    num
                                ); CUDACHECKASYNC; 

                                CUDACHECK(cudaEventRecord(callerBuffers.events[d], callerBuffers.streams[d]));
                            }
                        }

                        //join streams
                        for(int d = 0; d < numDistinctGpus; d++){
                            const int num = splits.h_numSelectedPerGpu[d];

                            if(num > 0){
                                CUDACHECK(cudaStreamWaitEvent(destStreams[g], callerBuffers.events[d], 0));
                            }                
                        }             
                    }
                }
            }

        }
    }

    void gather_collective_internal_2(
        std::vector<Handle>& vec_handle, 
        std::vector<T*>& vec_d_dest, 
        const std::vector<size_t>& vec_destRowPitchInBytes, 
        const std::vector<IndexType*>& vec_d_indices, 
        const std::vector<size_t>& vec_numIndices, 
        const std::vector<cudaStream_t>& destStreams,
        const std::vector<int>& destDeviceIds
    ) const{
        const int numDestGpus = destDeviceIds.size();

        if((isSingleGpu() || isReplicatedSingleGpu)){
            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd(destDeviceId);

                    auto it = std::find(usedDeviceIds.begin(), usedDeviceIds.end(), destDeviceId);
                    if(it != usedDeviceIds.end()){
                        //all data to be gathered resides on the destination device.
                        const int position = std::distance(usedDeviceIds.begin(), it);
                        
                        gpuArrays[position]->gather(
                            vec_d_dest[g], 
                            vec_destRowPitchInBytes[g], 
                            vec_d_indices[g], 
                            vec_numIndices[g], 
                            destStreams[g]
                        );
                    }else{
                        //array was not constructed on current device. use peer access from any gpu

                        const auto& gpuArray = gpuArrays[0];
                        auto& deviceBuffers = vec_handle[g]->deviceBuffers[0];
                        const int deviceId = gpuArray->getDeviceId();

                        CUDACHECK(cudaEventRecord(vec_handle[g]->callerBuffers.event, destStreams[g]));
                        
                        CUDACHECK(cudaSetDevice(deviceId));
                        CUDACHECK(cudaStreamWaitEvent(deviceBuffers.stream, vec_handle[g]->callerBuffers.event, 0));

                        //copy indices to other gpu
                        rmm::device_uvector<IndexType> d_indices_target(vec_numIndices[g], deviceBuffers.stream.getStream());
                        copy(
                            d_indices_target.data(), 
                            deviceId, 
                            vec_d_indices[g], 
                            destDeviceId, 
                            sizeof(IndexType) * vec_numIndices[g], 
                            deviceBuffers.stream
                        );

                        rmm::device_buffer d_temp(vec_destRowPitchInBytes[g] * vec_numIndices[g], deviceBuffers.stream.getStream());
                        gpuArrays[0]->gather(reinterpret_cast<T*>(d_temp.data()), vec_destRowPitchInBytes[g], d_indices_target.data(), vec_numIndices[g], deviceBuffers.stream);

                        //copy gathered data to us
                        copy(
                            vec_d_dest[g], 
                            destDeviceId, 
                            d_temp.data(), 
                            deviceId, 
                            vec_destRowPitchInBytes[g] * vec_numIndices[g], 
                            deviceBuffers.stream
                        );

                        CUDACHECK(cudaEventRecord(deviceBuffers.event, deviceBuffers.stream.getStream()));

                        //wait on destStream until gathered data is ready
                        CUDACHECK(cudaSetDevice(destDeviceId));
                        CUDACHECK(cudaStreamWaitEvent(destStreams[g], deviceBuffers.event, 0));
                    }
                }
            }
        }else{

            //perform multisplit to distribute indices and filter out invalid indices. then perform local gathers,
            // and scatter into result array

            const int numGpus = gpuArrays.size();
            const int numDistinctGpus = h_numRowsPerGpu.size();

            if(!isReplicatedSingleGpu){
                assert(numGpus == numDistinctGpus);
            }else{
                assert(1 == numDistinctGpus);
            }

            // std::vector<MultiSplitResult> splitsPerDestGpu_fooo;
            // for(int g = 0; g < numDestGpus; g++){
            //     if(vec_numIndices[g] > 0){
            //         const int destDeviceId = destDeviceIds[g];
            //         cub::SwitchDevice sd_{destDeviceId};
                    
            //         splitsPerDestGpu_fooo.emplace_back(vec_numIndices[g], numDistinctGpus, destStreams[g]);
            //     }else{
            //         splitsPerDestGpu_fooo.emplace_back(0, numDistinctGpus, destStreams[g]);
            //     }
            // }
            // CudaGraph cudaGraph; 
            // auto func = [&](cudaStream_t stream){
            //     CudaEvent event1{cudaEventDisableTiming};
            //     CUDACHECK(cudaEventRecord(event1, stream));
                
            //     for(int g = 0; g < numDestGpus; g++){
            //         if(vec_numIndices[g] > 0){
            //             const int destDeviceId = destDeviceIds[g];
            //             cub::SwitchDevice sd_{destDeviceId};
                        
            //             auto& handle = vec_handle[g];
            //             auto& result = splitsPerDestGpu_fooo[g];
                        
            //             CUDACHECK(cudaStreamWaitEvent(destStreams[g], event1, 0));
                        
            //             CUDACHECK(cudaMemsetAsync(result.d_numSelectedPerGpu, 0, sizeof(std::size_t) * numDistinctGpus, destStreams[g]));
            //             MultiGpu2dArrayKernels::partitionSplitKernel<<<SDIV(vec_numIndices[g], 128), 128, 0, destStreams[g]>>>(
            //                 result.d_selectedIndices,
            //                 result.d_selectedPositions,
            //                 result.d_numSelectedPerGpu,
            //                 numDistinctGpus,
            //                 h_numRowsPerGpuPrefixSum.data(),
            //                 vec_numIndices[g],
            //                 vec_d_indices[g]
            //             ); CUDACHECKASYNC;
                        
            //             CUDACHECK(cudaMemcpyAsync(
            //                 handle->multisplitTemp.data(),
            //                 result.d_numSelectedPerGpu,
            //                 sizeof(std::size_t) * numDistinctGpus,
            //                 D2H,
            //                 destStreams[g]
            //             ));
                        
            //             CudaEvent event2{cudaEventDisableTiming};
            //             CUDACHECK(cudaEventRecord(event2, destStreams[g]));
            //             CUDACHECK(cudaStreamWaitEvent(stream, event2, 0));
            //         }
            //     }
            // };
            // cudaGraph.capture(func);
            // cudaGraph.execute(destStreams[0]);
            
            std::vector<MultiSplitResult> splitsPerDestGpu;
            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd_{destDeviceId};
                    
                    auto& handle = vec_handle[g];
                    
                    MultiSplitResult splits = multiSplit_no_h_numSelectedPerGpu(
                        vec_d_indices[g], 
                        vec_numIndices[g], 
                        handle->multisplitTemp.data(), 
                        destStreams[g]
                    );
                    splitsPerDestGpu.push_back(std::move(splits));
                }else{
                    splitsPerDestGpu.emplace_back(0, numDistinctGpus, destStreams[g]);
                }
            }

            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd_{destDeviceId};
                    auto& handle = vec_handle[g];
                    CUDACHECK(cudaStreamSynchronize(destStreams[g]));
                    std::copy(handle->multisplitTemp.data(), handle->multisplitTemp.data() + numDistinctGpus, splitsPerDestGpu[g].h_numSelectedPerGpu.begin());
                }else{
                    std::fill(splitsPerDestGpu[g].h_numSelectedPerGpu.begin(), splitsPerDestGpu[g].h_numSelectedPerGpu.end(), 0);
                }
            }

            {
                std::vector<std::vector<size_t>> all_numSelectedPerGpu(numDestGpus);
                for(int g = 0; g < numDestGpus; g++){
                    all_numSelectedPerGpu[g] = splitsPerDestGpu[g].h_numSelectedPerGpu;
                }

                std::vector<std::vector<size_t>> all_numSelectedPerGpu_horizontalPS(numDestGpus, std::vector<size_t>(numDistinctGpus+1));
                for(int g = 0; g < numDestGpus; g++){
                    for(int d = 0; d < numDistinctGpus; d++){
                        all_numSelectedPerGpu_horizontalPS[g][d+1] = all_numSelectedPerGpu_horizontalPS[g][d] + all_numSelectedPerGpu[g][d];
                    }
                }

                std::vector<std::vector<size_t>> all_numSelectedPerGpu_verticalPS(numDestGpus+1, std::vector<size_t>(numDistinctGpus));
                for(int g = 0; g < numDestGpus; g++){
                    for(int d = 0; d < numDistinctGpus; d++){
                        all_numSelectedPerGpu_verticalPS[g+1][d] = all_numSelectedPerGpu_verticalPS[g][d] + all_numSelectedPerGpu[g][d];
                    }
                }

                // std::cout << "nums\n";
                // for(int g = 0; g < numDestGpus; g++){
                //     for(int d = 0; d < numDistinctGpus; d++){
                //         std::cout << all_numSelectedPerGpu[g][d] << " ";
                //     }
                //     std::cout << "\n";
                // }
                // std::cout << "horizontalPS\n";
                // for(int g = 0; g < numDestGpus; g++){
                //     for(int d = 0; d < numDistinctGpus+1; d++){
                //         std::cout << all_numSelectedPerGpu_horizontalPS[g][d] << " ";
                //     }
                //     std::cout << "\n";
                // }
                // std::cout << "verticalPS\n";
                // for(int g = 0; g < numDestGpus+1; g++){
                //     for(int d = 0; d < numDistinctGpus; d++){
                //         std::cout << all_numSelectedPerGpu_verticalPS[g][d] << " ";
                //     }
                //     std::cout << "\n";
                // }

                std::vector<char*> all_rawPtrs(numDistinctGpus, nullptr);
                std::vector<std::size_t> all_allocatedBytes(numDistinctGpus, 0);
                std::vector<IndexType*> all_deviceBuffers_d_selectedIndices(numDistinctGpus, nullptr);
                std::vector<char*> all_deviceBuffers_d_dataCommunicationBuffer(numDistinctGpus, nullptr);
                std::vector<rmm::mr::device_memory_resource*> all_mrs(numDistinctGpus, nullptr);

                for(int d = 0; d < numDistinctGpus; d++){

                        const auto& gpuArray = gpuArrays[d];
                        const int deviceId = gpuArray->getDeviceId();
                        auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];
                        
                        cub::SwitchDevice sd(deviceId);

                        const int numForAll = all_numSelectedPerGpu_verticalPS[numDestGpus][d];
                        
                        //allocate remote buffers
                        all_mrs[d] = rmm::mr::get_current_device_resource();
                        size_t bytes[2]{
                            SDIV(sizeof(IndexType) * numForAll, 512) * 512, // d_selectedIndices
                            SDIV(vec_destRowPitchInBytes[0 /*g*/] * numForAll, 512) * 512 //d_dataCommunicationBuffer
                        };
                        const size_t totalBytes = bytes[0] + bytes[1];
                        all_rawPtrs[d] = reinterpret_cast<char*>(all_mrs[d]->allocate(totalBytes, deviceBuffers.stream.getStream()));
                        all_allocatedBytes[d] = totalBytes;
                        all_deviceBuffers_d_selectedIndices[d] = reinterpret_cast<IndexType*>(all_rawPtrs[d]);
                        all_deviceBuffers_d_dataCommunicationBuffer[d] = reinterpret_cast<char*>(all_rawPtrs[d] + bytes[0]);

                        // std::cout << "Allocate on device " << deviceId << ": " 
                        //     << (void*)all_deviceBuffers_d_selectedIndices[d] << " " << bytes[0] << " bytes, " 
                        //     << (void*)all_deviceBuffers_d_dataCommunicationBuffer[d] << " " << bytes[1] << " bytes\n";

                }
                for(int d = 0; d < numDistinctGpus; d++){
                    const auto& gpuArray = gpuArrays[d];
                    const int deviceId = gpuArray->getDeviceId();
                    auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];
                    
                    cub::SwitchDevice sd(deviceId);
                    CUDACHECK(cudaStreamSynchronize(deviceBuffers.stream.getStream()));
                }

                //copy selected indices to remote devices
                for(int g = 0; g < numDestGpus; g++){
                    const int destDeviceId = destDeviceIds[g];
                    const auto& splits = splitsPerDestGpu[g];
                    auto& handle = vec_handle[g];
                    auto& callerBuffers = handle->callerBuffers;

                    for(int d = 0; d < numDistinctGpus; d++){
                        //send from g to d
                        const auto& gpuArray = gpuArrays[d];
                        const int deviceId = gpuArray->getDeviceId();
                        copy(
                            all_deviceBuffers_d_selectedIndices[d] + all_numSelectedPerGpu_verticalPS[g][d], 
                            deviceId, 
                            splits.d_selectedIndices + d * vec_numIndices[g], 
                            destDeviceId, 
                            sizeof(IndexType) * all_numSelectedPerGpu[g][d], 
                            callerBuffers.streams[d]
                        );
                    }

                }
                for(int g = 0; g < numDestGpus; g++){
                    const int destDeviceId = destDeviceIds[g];
                    auto& handle = vec_handle[g];
                    auto& callerBuffers = handle->callerBuffers;
                    cub::SwitchDevice sd(destDeviceId);

                    for(int d = 0; d < numDistinctGpus; d++){                    
                        CUDACHECK(cudaStreamSynchronize(callerBuffers.streams[d]));
                    }
                }

                //gather on remote gpus
                for(int d = 0; d < numDistinctGpus; d++){
                    const int num = all_numSelectedPerGpu_verticalPS[numDestGpus][d];

                    if(num > 0){
                        const auto& gpuArray = gpuArrays[d];
                        auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];
                        const int deviceId = gpuArray->getDeviceId();

                        cub::SwitchDevice sd(deviceId);                      

                        auto gatherIndexIterator = thrust::make_transform_iterator(
                            all_deviceBuffers_d_selectedIndices[d],
                            MultiGpu2dArrayKernels::SubtractFunctor<IndexType>(h_numRowsPerGpuPrefixSum[d])
                        );

                        gpuArray->gather(
                            (T*)all_deviceBuffers_d_dataCommunicationBuffer[d], 
                            vec_destRowPitchInBytes[0/*g*/], 
                            gatherIndexIterator, 
                            num, 
                            deviceBuffers.stream
                        );
                    }
                }

                for(int d = 0; d < numDistinctGpus; d++){
                    const auto& gpuArray = gpuArrays[d];
                    const int deviceId = gpuArray->getDeviceId();
                    auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];
                    
                    cub::SwitchDevice sd(deviceId);
                    CUDACHECK(cudaStreamSynchronize(deviceBuffers.stream.getStream()));
                }

                std::vector<rmm::device_uvector<char>> vec_callerBuffers_d_dataCommunicationBuffer;
                for(int g = 0; g < numDestGpus; g++){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd(destDeviceId);
                    vec_callerBuffers_d_dataCommunicationBuffer.emplace_back(vec_numIndices[g] * vec_destRowPitchInBytes[g], destStreams[g]);
                    // std::cout << "Allocate vec_callerBuffers_d_dataCommunicationBuffer on destdevice " << g << ", " << (void*)vec_callerBuffers_d_dataCommunicationBuffer.back().data() 
                    //     << " " << vec_numIndices[g] * vec_destRowPitchInBytes[g] << " bytes\n";
                }
                for(int g = 0; g < numDestGpus; g++){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd(destDeviceId);
                    CUDACHECK(cudaStreamSynchronize(destStreams[g]));
                }

                //copy gathered data back
                for(int g = 0; g < numDestGpus; g++){
                    const int destDeviceId = destDeviceIds[g];
                    auto& handle = vec_handle[g];
                    auto& callerBuffers = handle->callerBuffers;

                    cub::SwitchDevice sd(destDeviceId);

                    for(int d = 0; d < numDistinctGpus; d++){
                        //send from d to g
                        const auto& gpuArray = gpuArrays[d];
                        const int deviceId = gpuArray->getDeviceId();
                        // std::cout << "copy " << all_numSelectedPerGpu[g][d] * vec_destRowPitchInBytes[g] 
                        //     << " bytes from " << d << " to " << g 
                        //     << ", src offset " << all_numSelectedPerGpu_verticalPS[g][d] * vec_destRowPitchInBytes[g] 
                        //     << ", dst offset " << all_numSelectedPerGpu_horizontalPS[g][d] * vec_destRowPitchInBytes[g] 
                        //     << "\n";
                        copy(
                            vec_callerBuffers_d_dataCommunicationBuffer[g].data() + all_numSelectedPerGpu_horizontalPS[g][d] * vec_destRowPitchInBytes[g],
                            destDeviceId,
                            (T*)(all_deviceBuffers_d_dataCommunicationBuffer[d] + all_numSelectedPerGpu_verticalPS[g][d] * vec_destRowPitchInBytes[g]),
                            deviceId,
                            all_numSelectedPerGpu[g][d] * vec_destRowPitchInBytes[g],
                            callerBuffers.streams[d]
                        );
                    }
                }

                for(int g = 0; g < numDestGpus; g++){
                    const int destDeviceId = destDeviceIds[g];
                    auto& handle = vec_handle[g];
                    auto& callerBuffers = handle->callerBuffers;
                    cub::SwitchDevice sd(destDeviceId);

                    for(int d = 0; d < numDistinctGpus; d++){                    
                        CUDACHECK(cudaStreamSynchronize(callerBuffers.streams[d]));
                    }
                }
                for(int d = 0; d < numDistinctGpus; d++){

                    const auto& gpuArray = gpuArrays[d];
                    const int deviceId = gpuArray->getDeviceId();
                    auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];
                    
                    cub::SwitchDevice sd(deviceId);

                    all_mrs[d]->deallocate(all_rawPtrs[d], all_allocatedBytes[d], deviceBuffers.stream.getStream());
                }

                for(int g = 0; g < numDestGpus; g++){
                    const int destDeviceId = destDeviceIds[g];
                    auto& handle = vec_handle[g];
                    auto& callerBuffers = handle->callerBuffers;
                    const auto& splits = splitsPerDestGpu[g];
                    cub::SwitchDevice sd(destDeviceId);

                    if(vec_numIndices[g] > 0){                        
                        for(int d = 0; d < numDistinctGpus; d++){          

                            dim3 block(128, 1, 1);
                            dim3 grid(SDIV(vec_numIndices[g], block.x), 1, 1);

                            MultiGpu2dArrayKernels::scatterKernel<<<grid, block, 0, callerBuffers.streams[d]>>>(
                                (const T*)(vec_callerBuffers_d_dataCommunicationBuffer[g].data() + all_numSelectedPerGpu_horizontalPS[g][d] * vec_destRowPitchInBytes[g]), 
                                vec_destRowPitchInBytes[g], 
                                vec_numIndices[g],
                                getNumColumns(),
                                vec_d_dest[g], 
                                vec_destRowPitchInBytes[g], 
                                splits.d_selectedPositions + d * vec_numIndices[g], 
                                all_numSelectedPerGpu[g][d]
                            ); CUDACHECKASYNC; 
                        }
                    }
                }

                for(int g = 0; g < numDestGpus; g++){
                    const int destDeviceId = destDeviceIds[g];
                    auto& handle = vec_handle[g];
                    auto& callerBuffers = handle->callerBuffers;
                    cub::SwitchDevice sd(destDeviceId);

                    for(int d = 0; d < numDistinctGpus; d++){                    
                        CUDACHECK(cudaStreamSynchronize(callerBuffers.streams[d]));
                    }
                }

                
            }
            
        }
    }


    void gather_collective_internal_3(
        std::vector<Handle>& vec_handle, 
        std::vector<T*>& vec_d_dest, 
        const std::vector<size_t>& vec_destRowPitchInBytes, 
        const std::vector<IndexType*>& vec_d_indices, 
        const std::vector<size_t>& vec_numIndices, 
        const std::vector<cudaStream_t>& destStreams,
        const std::vector<int>& destDeviceIds
    ) const{
        const int numDestGpus = destDeviceIds.size();

        if((isSingleGpu() || isReplicatedSingleGpu)){
            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd(destDeviceId);

                    auto it = std::find(usedDeviceIds.begin(), usedDeviceIds.end(), destDeviceId);
                    if(it != usedDeviceIds.end()){
                        //all data to be gathered resides on the destination device.
                        const int position = std::distance(usedDeviceIds.begin(), it);
                        
                        gpuArrays[position]->gather(
                            vec_d_dest[g], 
                            vec_destRowPitchInBytes[g], 
                            vec_d_indices[g], 
                            vec_numIndices[g], 
                            destStreams[g]
                        );
                    }else{
                        //array was not constructed on current device. use peer access from any gpu

                        const auto& gpuArray = gpuArrays[0];
                        auto& deviceBuffers = vec_handle[g]->deviceBuffers[0];
                        const int deviceId = gpuArray->getDeviceId();

                        CUDACHECK(cudaEventRecord(vec_handle[g]->callerBuffers.event, destStreams[g]));
                        
                        CUDACHECK(cudaSetDevice(deviceId));
                        CUDACHECK(cudaStreamWaitEvent(deviceBuffers.stream, vec_handle[g]->callerBuffers.event, 0));

                        //copy indices to other gpu
                        rmm::device_uvector<IndexType> d_indices_target(vec_numIndices[g], deviceBuffers.stream.getStream());
                        copy(
                            d_indices_target.data(), 
                            deviceId, 
                            vec_d_indices[g], 
                            destDeviceId, 
                            sizeof(IndexType) * vec_numIndices[g], 
                            deviceBuffers.stream
                        );

                        rmm::device_buffer d_temp(vec_destRowPitchInBytes[g] * vec_numIndices[g], deviceBuffers.stream.getStream());
                        gpuArrays[0]->gather(reinterpret_cast<T*>(d_temp.data()), vec_destRowPitchInBytes[g], d_indices_target.data(), vec_numIndices[g], deviceBuffers.stream);

                        //copy gathered data to us
                        copy(
                            vec_d_dest[g], 
                            destDeviceId, 
                            d_temp.data(), 
                            deviceId, 
                            vec_destRowPitchInBytes[g] * vec_numIndices[g], 
                            deviceBuffers.stream
                        );

                        CUDACHECK(cudaEventRecord(deviceBuffers.event, deviceBuffers.stream.getStream()));

                        //wait on destStream until gathered data is ready
                        CUDACHECK(cudaSetDevice(destDeviceId));
                        CUDACHECK(cudaStreamWaitEvent(destStreams[g], deviceBuffers.event, 0));
                    }
                }
            }
        }else{

            //perform multisplit to distribute indices and filter out invalid indices. then perform local gathers,
            // and scatter into result array

            const int numGpus = gpuArrays.size();
            const int numDistinctGpus = h_numRowsPerGpu.size();

            if(!isReplicatedSingleGpu){
                assert(numGpus == numDistinctGpus);
            }else{
                assert(1 == numDistinctGpus);
            }

            // std::vector<MultiSplitResult> splitsPerDestGpu_fooo;
            // for(int g = 0; g < numDestGpus; g++){
            //     if(vec_numIndices[g] > 0){
            //         const int destDeviceId = destDeviceIds[g];
            //         cub::SwitchDevice sd_{destDeviceId};
                    
            //         splitsPerDestGpu_fooo.emplace_back(vec_numIndices[g], numDistinctGpus, destStreams[g]);
            //     }else{
            //         splitsPerDestGpu_fooo.emplace_back(0, numDistinctGpus, destStreams[g]);
            //     }
            // }
            // CudaGraph cudaGraph; 
            // auto func = [&](cudaStream_t stream){
            //     CudaEvent event1{cudaEventDisableTiming};
            //     CUDACHECK(cudaEventRecord(event1, stream));
                
            //     for(int g = 0; g < numDestGpus; g++){
            //         if(vec_numIndices[g] > 0){
            //             const int destDeviceId = destDeviceIds[g];
            //             cub::SwitchDevice sd_{destDeviceId};
                        
            //             auto& handle = vec_handle[g];
            //             auto& result = splitsPerDestGpu_fooo[g];
                        
            //             CUDACHECK(cudaStreamWaitEvent(destStreams[g], event1, 0));
                        
            //             CUDACHECK(cudaMemsetAsync(result.d_numSelectedPerGpu, 0, sizeof(std::size_t) * numDistinctGpus, destStreams[g]));
            //             MultiGpu2dArrayKernels::partitionSplitKernel<<<SDIV(vec_numIndices[g], 128), 128, 0, destStreams[g]>>>(
            //                 result.d_selectedIndices,
            //                 result.d_selectedPositions,
            //                 result.d_numSelectedPerGpu,
            //                 numDistinctGpus,
            //                 h_numRowsPerGpuPrefixSum.data(),
            //                 vec_numIndices[g],
            //                 vec_d_indices[g]
            //             ); CUDACHECKASYNC;
                        
            //             CUDACHECK(cudaMemcpyAsync(
            //                 handle->multisplitTemp.data(),
            //                 result.d_numSelectedPerGpu,
            //                 sizeof(std::size_t) * numDistinctGpus,
            //                 D2H,
            //                 destStreams[g]
            //             ));
                        
            //             CudaEvent event2{cudaEventDisableTiming};
            //             CUDACHECK(cudaEventRecord(event2, destStreams[g]));
            //             CUDACHECK(cudaStreamWaitEvent(stream, event2, 0));
            //         }
            //     }
            // };
            // cudaGraph.capture(func);
            // cudaGraph.execute(destStreams[0]);
            
            std::vector<MultiSplitResult> splitsPerDestGpu;
            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd_{destDeviceId};
                    
                    auto& handle = vec_handle[g];
                    
                    MultiSplitResult splits = multiSplit_no_h_numSelectedPerGpu(
                        vec_d_indices[g], 
                        vec_numIndices[g], 
                        handle->multisplitTemp.data(), 
                        destStreams[g]
                    );
                    splitsPerDestGpu.push_back(std::move(splits));
                }else{
                    splitsPerDestGpu.emplace_back(0, numDistinctGpus, destStreams[g]);
                }
            }

            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd_{destDeviceId};
                    auto& handle = vec_handle[g];
                    CUDACHECK(cudaStreamSynchronize(destStreams[g]));
                    std::copy(handle->multisplitTemp.data(), handle->multisplitTemp.data() + numDistinctGpus, splitsPerDestGpu[g].h_numSelectedPerGpu.begin());
                }else{
                    std::fill(splitsPerDestGpu[g].h_numSelectedPerGpu.begin(), splitsPerDestGpu[g].h_numSelectedPerGpu.end(), 0);
                }
            }

            {
                std::vector<std::vector<size_t>> all_numSelectedPerGpu(numDestGpus);
                for(int g = 0; g < numDestGpus; g++){
                    all_numSelectedPerGpu[g] = splitsPerDestGpu[g].h_numSelectedPerGpu;
                }

                std::vector<std::vector<size_t>> all_numSelectedPerGpu_horizontalPS(numDestGpus, std::vector<size_t>(numDistinctGpus+1));
                for(int g = 0; g < numDestGpus; g++){
                    for(int d = 0; d < numDistinctGpus; d++){
                        all_numSelectedPerGpu_horizontalPS[g][d+1] = all_numSelectedPerGpu_horizontalPS[g][d] + all_numSelectedPerGpu[g][d];
                    }
                }

                std::vector<std::vector<size_t>> all_numSelectedPerGpu_verticalPS(numDestGpus+1, std::vector<size_t>(numDistinctGpus));
                for(int g = 0; g < numDestGpus; g++){
                    for(int d = 0; d < numDistinctGpus; d++){
                        all_numSelectedPerGpu_verticalPS[g+1][d] = all_numSelectedPerGpu_verticalPS[g][d] + all_numSelectedPerGpu[g][d];
                    }
                }

                // std::cout << "nums\n";
                // for(int g = 0; g < numDestGpus; g++){
                //     for(int d = 0; d < numDistinctGpus; d++){
                //         std::cout << all_numSelectedPerGpu[g][d] << " ";
                //     }
                //     std::cout << "\n";
                // }
                // std::cout << "horizontalPS\n";
                // for(int g = 0; g < numDestGpus; g++){
                //     for(int d = 0; d < numDistinctGpus+1; d++){
                //         std::cout << all_numSelectedPerGpu_horizontalPS[g][d] << " ";
                //     }
                //     std::cout << "\n";
                // }
                // std::cout << "verticalPS\n";
                // for(int g = 0; g < numDestGpus+1; g++){
                //     for(int d = 0; d < numDistinctGpus; d++){
                //         std::cout << all_numSelectedPerGpu_verticalPS[g][d] << " ";
                //     }
                //     std::cout << "\n";
                // }

                std::vector<char*> all_rawPtrs(numDistinctGpus, nullptr);
                std::vector<std::size_t> all_allocatedBytes(numDistinctGpus, 0);
                std::vector<IndexType*> all_deviceBuffers_d_selectedIndices(numDistinctGpus, nullptr);
                std::vector<char*> all_deviceBuffers_d_dataCommunicationBuffer(numDistinctGpus, nullptr);
                std::vector<rmm::mr::device_memory_resource*> all_mrs(numDistinctGpus, nullptr);

                for(int d = 0; d < numDistinctGpus; d++){

                    const auto& gpuArray = gpuArrays[d];
                    const int deviceId = gpuArray->getDeviceId();
                    auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];
                    
                    cub::SwitchDevice sd(deviceId);

                    const int numForAll = all_numSelectedPerGpu_verticalPS[numDestGpus][d];
                    
                    //allocate remote buffers
                    all_mrs[d] = rmm::mr::get_current_device_resource();
                    size_t bytes[2]{
                        SDIV(sizeof(IndexType) * numForAll, 512) * 512, // d_selectedIndices
                        SDIV(vec_destRowPitchInBytes[0 /*g*/] * numForAll, 512) * 512 //d_dataCommunicationBuffer
                    };
                    const size_t totalBytes = bytes[0] + bytes[1];
                    all_rawPtrs[d] = reinterpret_cast<char*>(all_mrs[d]->allocate(totalBytes, deviceBuffers.stream.getStream()));
                    all_allocatedBytes[d] = totalBytes;
                    all_deviceBuffers_d_selectedIndices[d] = reinterpret_cast<IndexType*>(all_rawPtrs[d]);
                    all_deviceBuffers_d_dataCommunicationBuffer[d] = reinterpret_cast<char*>(all_rawPtrs[d] + bytes[0]);
                    CUDACHECK(cudaEventRecord(deviceBuffers.event, deviceBuffers.stream));

                    // std::cout << "Allocate on device " << deviceId << ": " 
                    //     << (void*)all_deviceBuffers_d_selectedIndices[d] << " " << bytes[0] << " bytes, " 
                    //     << (void*)all_deviceBuffers_d_dataCommunicationBuffer[d] << " " << bytes[1] << " bytes\n";

                }
                // for(int d = 0; d < numDistinctGpus; d++){
                //     const auto& gpuArray = gpuArrays[d];
                //     const int deviceId = gpuArray->getDeviceId();
                //     auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];
                    
                //     cub::SwitchDevice sd(deviceId);
                //     CUDACHECK(cudaStreamSynchronize(deviceBuffers.stream.getStream()));
                // }

                //copy selected indices to remote devices
                for(int g = 0; g < numDestGpus; g++){
                    const int destDeviceId = destDeviceIds[g];
                    const auto& splits = splitsPerDestGpu[g];
                    auto& handle = vec_handle[g];
                    auto& callerBuffers = handle->callerBuffers;
                    cub::SwitchDevice sd(destDeviceId);

                    for(int d = 0; d < numDistinctGpus; d++){
                        if(all_numSelectedPerGpu[g][d] > 0){
                            auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];
                            CUDACHECK(cudaStreamWaitEvent(callerBuffers.streams[d], deviceBuffers.event, 0));

                            //send from g to d
                            const auto& gpuArray = gpuArrays[d];
                            const int deviceId = gpuArray->getDeviceId();
                            copy(
                                all_deviceBuffers_d_selectedIndices[d] + all_numSelectedPerGpu_verticalPS[g][d], 
                                deviceId, 
                                splits.d_selectedIndices + d * vec_numIndices[g], 
                                destDeviceId, 
                                sizeof(IndexType) * all_numSelectedPerGpu[g][d], 
                                callerBuffers.streams[d]
                            );

                            //CUDACHECK(cudaEventRecord(callerBuffers.events[d], callerBuffers.streams[d]));
                        }
                    }

                }
                for(int g = 0; g < numDestGpus; g++){
                    const int destDeviceId = destDeviceIds[g];
                    auto& handle = vec_handle[g];
                    auto& callerBuffers = handle->callerBuffers;
                    cub::SwitchDevice sd(destDeviceId);

                    for(int d = 0; d < numDistinctGpus; d++){                    
                        CUDACHECK(cudaStreamSynchronize(callerBuffers.streams[d]));
                    }
                }

                //gather on remote gpus
                for(int d = 0; d < numDistinctGpus; d++){
                    const int num = all_numSelectedPerGpu_verticalPS[numDestGpus][d];

                    if(num > 0){
                        const auto& gpuArray = gpuArrays[d];
                        auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];
                        const int deviceId = gpuArray->getDeviceId();

                        cub::SwitchDevice sd(deviceId);  
                        //CUDACHECK(cudaStreamWaitEvent(deviceBuffers.stream, callerBuffers.events[d], 0));                    

                        auto gatherIndexIterator = thrust::make_transform_iterator(
                            all_deviceBuffers_d_selectedIndices[d],
                            MultiGpu2dArrayKernels::SubtractFunctor<IndexType>(h_numRowsPerGpuPrefixSum[d])
                        );

                        gpuArray->gather(
                            (T*)all_deviceBuffers_d_dataCommunicationBuffer[d], 
                            vec_destRowPitchInBytes[0/*g*/], 
                            gatherIndexIterator, 
                            num, 
                            deviceBuffers.stream
                        );
                    }
                }

                for(int d = 0; d < numDistinctGpus; d++){
                    const auto& gpuArray = gpuArrays[d];
                    const int deviceId = gpuArray->getDeviceId();
                    auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];
                    
                    cub::SwitchDevice sd(deviceId);
                    CUDACHECK(cudaStreamSynchronize(deviceBuffers.stream.getStream()));
                }

                std::vector<rmm::device_uvector<char>> vec_callerBuffers_d_dataCommunicationBuffer;
                for(int g = 0; g < numDestGpus; g++){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd(destDeviceId);
                    vec_callerBuffers_d_dataCommunicationBuffer.emplace_back(vec_numIndices[g] * vec_destRowPitchInBytes[g], destStreams[g]);
                    // std::cout << "Allocate vec_callerBuffers_d_dataCommunicationBuffer on destdevice " << g << ", " << (void*)vec_callerBuffers_d_dataCommunicationBuffer.back().data() 
                    //     << " " << vec_numIndices[g] * vec_destRowPitchInBytes[g] << " bytes\n";
                }
                for(int g = 0; g < numDestGpus; g++){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd(destDeviceId);
                    CUDACHECK(cudaStreamSynchronize(destStreams[g]));
                }

                //copy gathered data back
                for(int g = 0; g < numDestGpus; g++){
                    const int destDeviceId = destDeviceIds[g];
                    auto& handle = vec_handle[g];
                    auto& callerBuffers = handle->callerBuffers;

                    cub::SwitchDevice sd(destDeviceId);

                    for(int d = 0; d < numDistinctGpus; d++){
                        //send from d to g
                        const auto& gpuArray = gpuArrays[d];
                        const int deviceId = gpuArray->getDeviceId();
                        // std::cout << "copy " << all_numSelectedPerGpu[g][d] * vec_destRowPitchInBytes[g] 
                        //     << " bytes from " << d << " to " << g 
                        //     << ", src offset " << all_numSelectedPerGpu_verticalPS[g][d] * vec_destRowPitchInBytes[g] 
                        //     << ", dst offset " << all_numSelectedPerGpu_horizontalPS[g][d] * vec_destRowPitchInBytes[g] 
                        //     << "\n";
                        copy(
                            vec_callerBuffers_d_dataCommunicationBuffer[g].data() + all_numSelectedPerGpu_horizontalPS[g][d] * vec_destRowPitchInBytes[g],
                            destDeviceId,
                            (T*)(all_deviceBuffers_d_dataCommunicationBuffer[d] + all_numSelectedPerGpu_verticalPS[g][d] * vec_destRowPitchInBytes[g]),
                            deviceId,
                            all_numSelectedPerGpu[g][d] * vec_destRowPitchInBytes[g],
                            callerBuffers.streams[d]
                        );
                    }
                }

                for(int g = 0; g < numDestGpus; g++){
                    const int destDeviceId = destDeviceIds[g];
                    auto& handle = vec_handle[g];
                    auto& callerBuffers = handle->callerBuffers;
                    cub::SwitchDevice sd(destDeviceId);

                    for(int d = 0; d < numDistinctGpus; d++){                    
                        CUDACHECK(cudaStreamSynchronize(callerBuffers.streams[d]));
                    }
                }
                for(int d = 0; d < numDistinctGpus; d++){

                    const auto& gpuArray = gpuArrays[d];
                    const int deviceId = gpuArray->getDeviceId();
                    auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];
                    
                    cub::SwitchDevice sd(deviceId);

                    all_mrs[d]->deallocate(all_rawPtrs[d], all_allocatedBytes[d], deviceBuffers.stream.getStream());
                }

                for(int g = 0; g < numDestGpus; g++){
                    const int destDeviceId = destDeviceIds[g];
                    auto& handle = vec_handle[g];
                    auto& callerBuffers = handle->callerBuffers;
                    const auto& splits = splitsPerDestGpu[g];
                    cub::SwitchDevice sd(destDeviceId);

                    if(vec_numIndices[g] > 0){                        
                        for(int d = 0; d < numDistinctGpus; d++){          

                            dim3 block(128, 1, 1);
                            dim3 grid(SDIV(vec_numIndices[g], block.x), 1, 1);

                            MultiGpu2dArrayKernels::scatterKernel<<<grid, block, 0, callerBuffers.streams[d]>>>(
                                (const T*)(vec_callerBuffers_d_dataCommunicationBuffer[g].data() + all_numSelectedPerGpu_horizontalPS[g][d] * vec_destRowPitchInBytes[g]), 
                                vec_destRowPitchInBytes[g], 
                                vec_numIndices[g],
                                getNumColumns(),
                                vec_d_dest[g], 
                                vec_destRowPitchInBytes[g], 
                                splits.d_selectedPositions + d * vec_numIndices[g], 
                                all_numSelectedPerGpu[g][d]
                            ); CUDACHECKASYNC; 
                        }
                    }
                }

                for(int g = 0; g < numDestGpus; g++){
                    const int destDeviceId = destDeviceIds[g];
                    auto& handle = vec_handle[g];
                    auto& callerBuffers = handle->callerBuffers;
                    cub::SwitchDevice sd(destDeviceId);

                    for(int d = 0; d < numDistinctGpus; d++){                    
                        CUDACHECK(cudaStreamSynchronize(callerBuffers.streams[d]));
                    }
                }

                
            }
            
        }
    }



    void gather_collective_internal_4(
        std::vector<Handle>& vec_handle, 
        std::vector<T*>& vec_d_dest, 
        const std::vector<size_t>& vec_destRowPitchInBytes, 
        const std::vector<IndexType*>& vec_d_indices, 
        const std::vector<size_t>& vec_numIndices, 
        bool mayContainInvalidIndices,
        const std::vector<cudaStream_t>& destStreams,
        const std::vector<int>& destDeviceIds
    ) const{
        const int numDestGpus = destDeviceIds.size();

        if((isSingleGpu() || isReplicatedSingleGpu) && !mayContainInvalidIndices){
            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd(destDeviceId);

                    auto it = std::find(usedDeviceIds.begin(), usedDeviceIds.end(), destDeviceId);
                    if(it != usedDeviceIds.end()){
                        //all data to be gathered resides on the destination device.
                        const int position = std::distance(usedDeviceIds.begin(), it);
                        
                        gpuArrays[position]->gather(
                            vec_d_dest[g], 
                            vec_destRowPitchInBytes[g], 
                            vec_d_indices[g], 
                            vec_numIndices[g], 
                            destStreams[g]
                        );
                    }else{
                        //array was not constructed on current device. use peer access from any gpu

                        const auto& gpuArray = gpuArrays[0];
                        auto& deviceBuffers = vec_handle[g]->deviceBuffers[0];
                        const int deviceId = gpuArray->getDeviceId();

                        CUDACHECK(cudaEventRecord(vec_handle[g]->callerBuffers.event, destStreams[g]));
                        
                        CUDACHECK(cudaSetDevice(deviceId));
                        CUDACHECK(cudaStreamWaitEvent(deviceBuffers.stream, vec_handle[g]->callerBuffers.event, 0));

                        //copy indices to other gpu
                        rmm::device_uvector<IndexType> d_indices_target(vec_numIndices[g], deviceBuffers.stream.getStream());
                        copy(
                            d_indices_target.data(), 
                            deviceId, 
                            vec_d_indices[g], 
                            destDeviceId, 
                            sizeof(IndexType) * vec_numIndices[g], 
                            deviceBuffers.stream
                        );

                        rmm::device_buffer d_temp(vec_destRowPitchInBytes[g] * vec_numIndices[g], deviceBuffers.stream.getStream());
                        gpuArrays[0]->gather(reinterpret_cast<T*>(d_temp.data()), vec_destRowPitchInBytes[g], d_indices_target.data(), vec_numIndices[g], deviceBuffers.stream);

                        //copy gathered data to us
                        copy(
                            vec_d_dest[g], 
                            destDeviceId, 
                            d_temp.data(), 
                            deviceId, 
                            vec_destRowPitchInBytes[g] * vec_numIndices[g], 
                            deviceBuffers.stream
                        );

                        CUDACHECK(cudaEventRecord(deviceBuffers.event, deviceBuffers.stream.getStream()));

                        //wait on destStream until gathered data is ready
                        CUDACHECK(cudaSetDevice(destDeviceId));
                        CUDACHECK(cudaStreamWaitEvent(destStreams[g], deviceBuffers.event, 0));
                    }
                }
            }
        }else{

            //perform multisplit to distribute indices and filter out invalid indices. then perform local gathers,
            // and scatter into result array

            const int numGpus = gpuArrays.size();
            const int numDistinctGpus = h_numRowsPerGpu.size();

            if(!isReplicatedSingleGpu){
                assert(numGpus == numDistinctGpus);
            }else{
                assert(1 == numDistinctGpus);
            }
            
            std::vector<MultiSplitResult> splitsPerDestGpu;
            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd_{destDeviceId};
                    
                    auto& handle = vec_handle[g];
                    
                    MultiSplitResult splits = multiSplit_no_h_numSelectedPerGpu(
                        vec_d_indices[g], 
                        vec_numIndices[g], 
                        handle->multisplitTemp.data(), 
                        destStreams[g]
                    );
                    splitsPerDestGpu.push_back(std::move(splits));
                }else{
                    splitsPerDestGpu.emplace_back(0, numDistinctGpus, destStreams[g]);
                }
            }

            std::vector<rmm::device_uvector<char>> vec_callerBuffers_d_dataCommunicationBuffer;
            std::vector<std::vector<std::size_t>> vec_numSelectedPrefixSum(numDestGpus, std::vector<std::size_t>(numDistinctGpus,0));
            std::vector<std::vector<char*>> vec_all_rawPtrs(numDestGpus, std::vector<char*>(numDistinctGpus, nullptr));
            std::vector<std::vector<std::size_t>> vec_all_allocatedBytes(numDestGpus, std::vector<std::size_t>(numDistinctGpus, 0));
            std::vector<std::vector<IndexType*>> vec_all_deviceBuffers_d_selectedIndices(numDestGpus, std::vector<IndexType*>(numDistinctGpus, nullptr));
            std::vector<std::vector<char*>> vec_all_deviceBuffers_d_dataCommunicationBuffer(numDestGpus, std::vector<char*>(numDistinctGpus, nullptr));
            std::vector<std::vector<rmm::mr::device_memory_resource*>> vec_all_mrs(numDestGpus, std::vector<rmm::mr::device_memory_resource*>(numDistinctGpus, nullptr));

            for(int g = 0; g < numDestGpus; g++){
                const int destDeviceId = destDeviceIds[g];
                cub::SwitchDevice sd_{destDeviceId};
                MultiSplitResult& splits = splitsPerDestGpu[g];
                auto& handle = vec_handle[g];
                auto& callerBuffers = handle->callerBuffers;

                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd_{destDeviceId};
                    auto& handle = vec_handle[g];
                    CUDACHECK(cudaStreamSynchronize(destStreams[g])); //wait for multi-split results (splits.h_numSelectedPerGpu)
                    std::copy(handle->multisplitTemp.data(), handle->multisplitTemp.data() + numDistinctGpus, splitsPerDestGpu[g].h_numSelectedPerGpu.begin());
                    vec_callerBuffers_d_dataCommunicationBuffer.emplace_back(vec_numIndices[g] * vec_destRowPitchInBytes[g], destStreams[g]);

                    // for(int x = 0; x < numDistinctGpus; x++){
                    //     std::cout << splitsPerDestGpu[g].h_numSelectedPerGpu[x] << " ";
                    // }
                    // std::cout << "\n";

                    // for(int d = 0; d < numDistinctGpus; d++){
                    //     const int num = splits.h_numSelectedPerGpu[d];
                    //     if(num > 0){
                    //         CUDACHECK(cudaStreamWaitEvent(callerBuffers.streams[d], handle->callerBuffers.event));
                    //     }
                    // }

                    std::partial_sum(
                        splits.h_numSelectedPerGpu.data(),
                        splits.h_numSelectedPerGpu.data() + numDistinctGpus - 1,
                        vec_numSelectedPrefixSum[g].begin() + 1
                    );

                    for(int d = 0; d < numDistinctGpus; d++){
                        const int num = splits.h_numSelectedPerGpu[d];
                        
                        if(num > 0){
                            const auto& gpuArray = gpuArrays[d];
                            auto& deviceBuffers = handle->deviceBuffers[d];
                            const int deviceId = gpuArray->getDeviceId();
                            
                            cub::SwitchDevice sd(deviceId);
                            
                            //allocate remote buffers
                            vec_all_mrs[g][d] = rmm::mr::get_current_device_resource();
                            size_t bytes[2]{
                                sizeof(IndexType) * num, // d_selectedIndices
                                vec_destRowPitchInBytes[g] * num //d_dataCommunicationBuffer
                            };
                            const size_t totalBytes = bytes[0] + bytes[1];
                            vec_all_rawPtrs[g][d] = reinterpret_cast<char*>(vec_all_mrs[g][d]->allocate(totalBytes, deviceBuffers.stream.getStream()));
                            vec_all_allocatedBytes[g][d] = totalBytes;
                            vec_all_deviceBuffers_d_selectedIndices[g][d] = reinterpret_cast<IndexType*>(vec_all_rawPtrs[g][d]);
                            vec_all_deviceBuffers_d_dataCommunicationBuffer[g][d] = reinterpret_cast<char*>(vec_all_rawPtrs[g][d] + bytes[0]);

                            copy(
                                vec_all_deviceBuffers_d_selectedIndices[g][d], 
                                deviceId, 
                                splits.d_selectedIndices + d * vec_numIndices[g], 
                                destDeviceId, 
                                sizeof(IndexType) * num, 
                                deviceBuffers.stream
                            );
                        }
                    }
                }else{
                    std::fill(splitsPerDestGpu[g].h_numSelectedPerGpu.begin(), splitsPerDestGpu[g].h_numSelectedPerGpu.end(), 0);
                    vec_callerBuffers_d_dataCommunicationBuffer.emplace_back(0, destStreams[g]);
                }

                //record event that allocation of vec_callerBuffers_d_dataCommunicationBuffer is ready
                CUDACHECK(cudaEventRecord(handle->callerBuffers.event, destStreams[g]));
            }
            //std::cout << "\n";


            
            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd_{destDeviceId};

                    auto& handle = vec_handle[g];
                    auto& callerBuffers = handle->callerBuffers;

                    MultiSplitResult& splits = splitsPerDestGpu[g];
                    //gather on remote gpus
                    for(int d = 0; d < numDistinctGpus; d++){
                        const int num = splits.h_numSelectedPerGpu[d];

                        if(num > 0){
                            const auto& gpuArray = gpuArrays[d];
                            auto& deviceBuffers = handle->deviceBuffers[d];
                            const int deviceId = gpuArray->getDeviceId();

                            cub::SwitchDevice sd(deviceId);                      

                            auto gatherIndexIterator = thrust::make_transform_iterator(
                                vec_all_deviceBuffers_d_selectedIndices[g][d],
                                MultiGpu2dArrayKernels::SubtractFunctor<IndexType>(h_numRowsPerGpuPrefixSum[d])
                            );

                            gpuArray->gather(
                                (T*)vec_all_deviceBuffers_d_dataCommunicationBuffer[g][d], 
                                vec_destRowPitchInBytes[g], 
                                gatherIndexIterator, 
                                num, 
                                deviceBuffers.stream
                            );
                        }
                    }
                }
            }

            for(int d = 0; d < numDistinctGpus; d++){
                for(int g = 0; g < numDestGpus; g++){
                    if(vec_numIndices[g] > 0){
                        const int destDeviceId = destDeviceIds[g];                    

                        auto& handle = vec_handle[g];
                        auto& callerBuffers = handle->callerBuffers;

                        MultiSplitResult& splits = splitsPerDestGpu[g];

                    //copy remote gathered data to caller
      
                        const int num = splits.h_numSelectedPerGpu[d];

                        if(num > 0){
                            const auto& gpuArray = gpuArrays[d];
                            auto& deviceBuffers = handle->deviceBuffers[d];
                            const int deviceId = gpuArray->getDeviceId();

                            cub::SwitchDevice sd_{deviceId};

                            //wait for allocation of vec_callerBuffers_d_dataCommunicationBuffer on a destStreams[g]
                            CUDACHECK(cudaStreamWaitEvent(deviceBuffers.stream, handle->callerBuffers.event)); 

                            copy(
                                vec_callerBuffers_d_dataCommunicationBuffer[g].data() + vec_destRowPitchInBytes[g] * vec_numSelectedPrefixSum[g][d], 
                                destDeviceId, 
                                vec_all_deviceBuffers_d_dataCommunicationBuffer[g][d], 
                                deviceId, 
                                vec_destRowPitchInBytes[g] * num, 
                                deviceBuffers.stream
                            );                            

                            //record that all data has been copied from data gpu d to the different caller gpus
                            CUDACHECK(cudaEventRecord(deviceBuffers.event, deviceBuffers.stream));
                        }
                    }
                }
            }

            
            //scatter into result array
            for(int d = 0; d < numDistinctGpus; d++){
                for(int g = 0; g < numDestGpus; g++){
                    if(vec_numIndices[g] > 0){
                        const int destDeviceId = destDeviceIds[g];
                        cub::SwitchDevice sd_{destDeviceId};

                        auto& handle = vec_handle[g];
                        auto& callerBuffers = handle->callerBuffers;

                        MultiSplitResult& splits = splitsPerDestGpu[g];
                        const auto& gpuArray = gpuArrays[d];
                        auto& deviceBuffers = handle->deviceBuffers[d];
                    
                        const int num = splits.h_numSelectedPerGpu[d];

                        if(num > 0){
                            //wait for copy into vec_callerBuffers_d_dataCommunicationBuffer on deviceBuffers.stream
                            CUDACHECK(cudaStreamWaitEvent(callerBuffers.streams[d], deviceBuffers.event, 0));

                            dim3 block(128, 1, 1);
                            dim3 grid(SDIV(vec_numIndices[g], block.x), 1, 1);

                            MultiGpu2dArrayKernels::scatterKernel<<<grid, block, 0, callerBuffers.streams[d]>>>(
                                (const T*)(vec_callerBuffers_d_dataCommunicationBuffer[g].data() + vec_destRowPitchInBytes[g] * vec_numSelectedPrefixSum[g][d]), 
                                vec_destRowPitchInBytes[g], 
                                vec_numIndices[g],
                                getNumColumns(),
                                vec_d_dest[g], 
                                vec_destRowPitchInBytes[g], 
                                splits.d_selectedPositions + d * vec_numIndices[g], 
                                num
                            ); CUDACHECKASYNC; 
                        }
                    }
                }
            }

            // for(int g = 0; g < numDestGpus; g++){
            //     if(vec_numIndices[g] > 0){
            //         const int destDeviceId = destDeviceIds[g];
            //         cub::SwitchDevice sd_{destDeviceId};

            //         auto& handle = vec_handle[g];
            //         auto& callerBuffers = handle->callerBuffers;

            //         MultiSplitResult& splits = splitsPerDestGpu[g];

            //         //scatter into result array
            //         for(int d = 0; d < numDistinctGpus; d++){
            //             const int num = splits.h_numSelectedPerGpu[d];

            //             if(num > 0){

            //                 dim3 block(128, 1, 1);
            //                 dim3 grid(SDIV(vec_numIndices[g], block.x), 1, 1);

            //                 MultiGpu2dArrayKernels::scatterKernel<<<grid, block, 0, callerBuffers.streams[d]>>>(
            //                     (const T*)(vec_callerBuffers_d_dataCommunicationBuffer[g].data() + vec_destRowPitchInBytes[g] * vec_numSelectedPrefixSum[g][d]), 
            //                     vec_destRowPitchInBytes[g], 
            //                     vec_numIndices[g],
            //                     getNumColumns(),
            //                     vec_d_dest[g], 
            //                     vec_destRowPitchInBytes[g], 
            //                     splits.d_selectedPositions + d * vec_numIndices[g], 
            //                     num
            //                 ); CUDACHECKASYNC; 
            //             }
            //         }
            //     }
            // }


            

            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd_{destDeviceId};

                    auto& handle = vec_handle[g];
                    auto& callerBuffers = handle->callerBuffers;

                    MultiSplitResult& splits = splitsPerDestGpu[g];

                    //join streams
                    for(int d = 0; d < numDistinctGpus; d++){
                        const int num = splits.h_numSelectedPerGpu[d];

                        if(num > 0){
                            CUDACHECK(cudaEventRecord(callerBuffers.events[d], callerBuffers.streams[d]));
                            CUDACHECK(cudaStreamWaitEvent(destStreams[g], callerBuffers.events[d], 0));
                        }                
                    }             
                }
            }

            // for(int g = 0; g < numDestGpus; g++){
            //     if(vec_numIndices[g] > 0){
            //         const int destDeviceId = destDeviceIds[g];
            //         cub::SwitchDevice sd_{destDeviceId};

            //         auto& handle = vec_handle[g];
            //         auto& callerBuffers = handle->callerBuffers;

            //         MultiSplitResult& splits = splitsPerDestGpu[g];

            //         //join streams
            //         for(int d = 0; d < numDistinctGpus; d++){
            //             const int num = splits.h_numSelectedPerGpu[d];
            //             if(num > 0){
            //                 CUDACHECK(cudaStreamSynchronize(callerBuffers.streams[d]));
            //             }                
            //         }             
            //     }
            // }

            for(int g = 0; g < numDestGpus; g++){
                const int destDeviceId = destDeviceIds[g];
                cub::SwitchDevice sd_{destDeviceId};
                CUDACHECK(cudaStreamSynchronize(destStreams[g]));
            }

            //deallocate
            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){                    

                    auto& handle = vec_handle[g];
                    auto& callerBuffers = handle->callerBuffers;

                    MultiSplitResult& splits = splitsPerDestGpu[g];
                    
                    for(int d = 0; d < numDistinctGpus; d++){
                        const int num = splits.h_numSelectedPerGpu[d];

                        if(num > 0){
                            const auto& gpuArray = gpuArrays[d];
                            auto& deviceBuffers = handle->deviceBuffers[d];
                            const int deviceId = gpuArray->getDeviceId();

                            cub::SwitchDevice sd_{deviceId};

                            vec_all_mrs[g][d]->deallocate(vec_all_rawPtrs[g][d], vec_all_allocatedBytes[g][d], deviceBuffers.stream.getStream());

                            //scatterKernel in callerBuffers.streams[d] may not have completed yet so we cannot deallocate on deviceBuffers.stream. Deallocate on destStreams[g] instead

                            //vec_all_mrs[g][d]->deallocate(vec_all_rawPtrs[g][d], vec_all_allocatedBytes[g][d], destStreams[g]);
                        }
                    }
                }
            }

            for(int g = 0; g < numDestGpus; g++){
                const int destDeviceId = destDeviceIds[g];
                cub::SwitchDevice sd_{destDeviceId};
                vec_callerBuffers_d_dataCommunicationBuffer[g].release();

                auto toDelete = std::move(splitsPerDestGpu[g]);
            }




        }
    }



    void gather_collective_internal_5(
        std::vector<Handle>& vec_handle, 
        std::vector<T*>& vec_d_dest, 
        const std::vector<size_t>& vec_destRowPitchInBytes, 
        const std::vector<IndexType*>& vec_d_indices, 
        const std::vector<size_t>& vec_numIndices, 
        const std::vector<cudaStream_t>& destStreams,
        const std::vector<int>& destDeviceIds
    ) const{
        const int numDestGpus = destDeviceIds.size();

        if((isSingleGpu() || isReplicatedSingleGpu)){
            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd(destDeviceId);

                    auto it = std::find(usedDeviceIds.begin(), usedDeviceIds.end(), destDeviceId);
                    if(it != usedDeviceIds.end()){
                        //all data to be gathered resides on the destination device.
                        const int position = std::distance(usedDeviceIds.begin(), it);
                        
                        gpuArrays[position]->gather(
                            vec_d_dest[g], 
                            vec_destRowPitchInBytes[g], 
                            vec_d_indices[g], 
                            vec_numIndices[g], 
                            destStreams[g]
                        );
                    }else{
                        //array was not constructed on current device. use peer access from any gpu

                        const auto& gpuArray = gpuArrays[0];
                        auto& deviceBuffers = vec_handle[g]->deviceBuffers[0];
                        const int deviceId = gpuArray->getDeviceId();

                        CUDACHECK(cudaEventRecord(vec_handle[g]->callerBuffers.event, destStreams[g]));
                        
                        CUDACHECK(cudaSetDevice(deviceId));
                        CUDACHECK(cudaStreamWaitEvent(deviceBuffers.stream, vec_handle[g]->callerBuffers.event, 0));

                        //copy indices to other gpu
                        rmm::device_uvector<IndexType> d_indices_target(vec_numIndices[g], deviceBuffers.stream.getStream());
                        copy(
                            d_indices_target.data(), 
                            deviceId, 
                            vec_d_indices[g], 
                            destDeviceId, 
                            sizeof(IndexType) * vec_numIndices[g], 
                            deviceBuffers.stream
                        );

                        rmm::device_buffer d_temp(vec_destRowPitchInBytes[g] * vec_numIndices[g], deviceBuffers.stream.getStream());
                        gpuArrays[0]->gather(reinterpret_cast<T*>(d_temp.data()), vec_destRowPitchInBytes[g], d_indices_target.data(), vec_numIndices[g], deviceBuffers.stream);

                        //copy gathered data to us
                        copy(
                            vec_d_dest[g], 
                            destDeviceId, 
                            d_temp.data(), 
                            deviceId, 
                            vec_destRowPitchInBytes[g] * vec_numIndices[g], 
                            deviceBuffers.stream
                        );

                        CUDACHECK(cudaEventRecord(deviceBuffers.event, deviceBuffers.stream.getStream()));

                        //wait on destStream until gathered data is ready
                        CUDACHECK(cudaSetDevice(destDeviceId));
                        CUDACHECK(cudaStreamWaitEvent(destStreams[g], deviceBuffers.event, 0));
                    }
                }
            }
        }else{

            //perform multisplit to distribute indices and filter out invalid indices. then perform local gathers,
            // and scatter into result array

            const int numGpus = gpuArrays.size();
            const int numDistinctGpus = h_numRowsPerGpu.size();

            if(!isReplicatedSingleGpu){
                assert(numGpus == numDistinctGpus);
            }else{
                assert(1 == numDistinctGpus);
            }
            
            std::vector<MultiSplitResult> splitsPerDestGpu;
            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd_{destDeviceId};
                    
                    auto& handle = vec_handle[g];
                    
                    MultiSplitResult splits = multiSplit_no_h_numSelectedPerGpu(
                        vec_d_indices[g], 
                        vec_numIndices[g], 
                        handle->multisplitTemp.data(), 
                        destStreams[g]
                    );
                    splitsPerDestGpu.push_back(std::move(splits));
                }else{
                    splitsPerDestGpu.emplace_back(0, numDistinctGpus, destStreams[g]);
                }
            }

            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd_{destDeviceId};
                    auto& handle = vec_handle[g];
                    CUDACHECK(cudaStreamSynchronize(destStreams[g]));
                    std::copy(handle->multisplitTemp.data(), handle->multisplitTemp.data() + numDistinctGpus, splitsPerDestGpu[g].h_numSelectedPerGpu.begin());
                }else{
                    std::fill(splitsPerDestGpu[g].h_numSelectedPerGpu.begin(), splitsPerDestGpu[g].h_numSelectedPerGpu.end(), 0);
                }
            }


            std::vector<std::vector<size_t>> all_numSelectedPerGpu(numDestGpus);
            for(int g = 0; g < numDestGpus; g++){
                all_numSelectedPerGpu[g] = splitsPerDestGpu[g].h_numSelectedPerGpu;
            }

            std::vector<std::vector<size_t>> all_numSelectedPerGpu_horizontalPS(numDestGpus, std::vector<size_t>(numDistinctGpus+1));
            for(int g = 0; g < numDestGpus; g++){
                for(int d = 0; d < numDistinctGpus; d++){
                    all_numSelectedPerGpu_horizontalPS[g][d+1] = all_numSelectedPerGpu_horizontalPS[g][d] + all_numSelectedPerGpu[g][d];
                }
            }

            std::vector<std::vector<size_t>> all_numSelectedPerGpu_verticalPS(numDestGpus+1, std::vector<size_t>(numDistinctGpus));
            for(int g = 0; g < numDestGpus; g++){
                for(int d = 0; d < numDistinctGpus; d++){
                    all_numSelectedPerGpu_verticalPS[g+1][d] = all_numSelectedPerGpu_verticalPS[g][d] + all_numSelectedPerGpu[g][d];
                }
            }

            std::vector<char*> all_rawPtrs(numDistinctGpus, nullptr);
            std::vector<std::size_t> all_allocatedBytes(numDistinctGpus, 0);
            std::vector<IndexType*> all_deviceBuffers_d_selectedIndices(numDistinctGpus, nullptr);
            std::vector<char*> all_deviceBuffers_d_dataCommunicationBuffer(numDistinctGpus, nullptr);
            std::vector<rmm::mr::device_memory_resource*> all_mrs(numDistinctGpus, nullptr);

            for(int d = 0; d < numDistinctGpus; d++){

                const auto& gpuArray = gpuArrays[d];
                const int deviceId = gpuArray->getDeviceId();
                auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];
                
                cub::SwitchDevice sd(deviceId);

                const int numForAll = all_numSelectedPerGpu_verticalPS[numDestGpus][d];
                
                //allocate remote buffers
                all_mrs[d] = rmm::mr::get_current_device_resource();
                size_t bytes[2]{
                    SDIV(sizeof(IndexType) * numForAll, 512) * 512, // d_selectedIndices
                    SDIV(vec_destRowPitchInBytes[0 /*g*/] * numForAll, 512) * 512 //d_dataCommunicationBuffer
                };
                const size_t totalBytes = bytes[0] + bytes[1];
                all_rawPtrs[d] = reinterpret_cast<char*>(all_mrs[d]->allocate(totalBytes, deviceBuffers.stream.getStream()));
                all_allocatedBytes[d] = totalBytes;
                all_deviceBuffers_d_selectedIndices[d] = reinterpret_cast<IndexType*>(all_rawPtrs[d]);
                all_deviceBuffers_d_dataCommunicationBuffer[d] = reinterpret_cast<char*>(all_rawPtrs[d] + bytes[0]);

                // std::cout << "Allocate on device " << deviceId << ": " 
                //     << (void*)all_deviceBuffers_d_selectedIndices[d] << " " << bytes[0] << " bytes, " 
                //     << (void*)all_deviceBuffers_d_dataCommunicationBuffer[d] << " " << bytes[1] << " bytes\n";

            }

            for(int d = 0; d < numDistinctGpus; d++){
                const auto& gpuArray = gpuArrays[d];
                const int deviceId = gpuArray->getDeviceId();
                auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];
                
                cub::SwitchDevice sd(deviceId);
                CUDACHECK(cudaStreamSynchronize(deviceBuffers.stream));
            }


            std::vector<rmm::device_uvector<char>> vec_callerBuffers_d_dataCommunicationBuffer;

            for(int g = 0; g < numDestGpus; g++){
                const int destDeviceId = destDeviceIds[g];
                cub::SwitchDevice sd_{destDeviceId};
                MultiSplitResult& splits = splitsPerDestGpu[g];
                auto& handle = vec_handle[g];
                auto& callerBuffers = handle->callerBuffers;

                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd_{destDeviceId};
                    auto& handle = vec_handle[g];
                    // CUDACHECK(cudaStreamSynchronize(destStreams[g]));
                    // std::copy(handle->multisplitTemp.data(), handle->multisplitTemp.data() + numDistinctGpus, splitsPerDestGpu[g].h_numSelectedPerGpu.begin());
                    vec_callerBuffers_d_dataCommunicationBuffer.emplace_back(vec_numIndices[g] * vec_destRowPitchInBytes[g], destStreams[g]);

                    CUDACHECK(cudaEventRecord(handle->callerBuffers.event, destStreams[g]));


                    for(int d = 0; d < numDistinctGpus; d++){
                        const int num = all_numSelectedPerGpu[g][d];
                        
                        if(num > 0){
                            const auto& gpuArray = gpuArrays[d];
                            auto& deviceBuffers = handle->deviceBuffers[d];
                            const int deviceId = gpuArray->getDeviceId();
                            
                            cub::SwitchDevice sd(deviceId);
                            copy(
                                all_deviceBuffers_d_selectedIndices[d] + all_numSelectedPerGpu_verticalPS[g][d], 
                                deviceId, 
                                splits.d_selectedIndices + d * vec_numIndices[g], 
                                destDeviceId, 
                                sizeof(IndexType) * num, 
                                deviceBuffers.stream
                            );
                        }
                    }
                }else{
                    std::fill(splitsPerDestGpu[g].h_numSelectedPerGpu.begin(), splitsPerDestGpu[g].h_numSelectedPerGpu.end(), 0);
                    vec_callerBuffers_d_dataCommunicationBuffer.emplace_back(0, destStreams[g]);
                }
            }


            
            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd_{destDeviceId};

                    auto& handle = vec_handle[g];
                    auto& callerBuffers = handle->callerBuffers;

                    MultiSplitResult& splits = splitsPerDestGpu[g];
                    //gather on remote gpus
                    for(int d = 0; d < numDistinctGpus; d++){
                        const int num = all_numSelectedPerGpu[g][d];

                        if(num > 0){
                            const auto& gpuArray = gpuArrays[d];
                            auto& deviceBuffers = handle->deviceBuffers[d];
                            const int deviceId = gpuArray->getDeviceId();

                            cub::SwitchDevice sd(deviceId);                      

                            auto gatherIndexIterator = thrust::make_transform_iterator(
                                all_deviceBuffers_d_selectedIndices[d] + all_numSelectedPerGpu_verticalPS[g][d],
                                MultiGpu2dArrayKernels::SubtractFunctor<IndexType>(h_numRowsPerGpuPrefixSum[d])
                            );

                            gpuArray->gather(
                                (T*)(all_deviceBuffers_d_dataCommunicationBuffer[d] + vec_destRowPitchInBytes[g] * all_numSelectedPerGpu_verticalPS[g][d]), 
                                vec_destRowPitchInBytes[g], 
                                gatherIndexIterator, 
                                num, 
                                deviceBuffers.stream
                            );
                        }
                    }
                }
            }

            for(int d = 0; d < numDistinctGpus; d++){
                for(int g = 0; g < numDestGpus; g++){
                    if(vec_numIndices[g] > 0){
                        const int destDeviceId = destDeviceIds[g];                    

                        auto& handle = vec_handle[g];
                        auto& callerBuffers = handle->callerBuffers;

                        MultiSplitResult& splits = splitsPerDestGpu[g];

                    //copy remote gathered data to caller
      
                        const int num = all_numSelectedPerGpu[g][d];

                        if(num > 0){
                            const auto& gpuArray = gpuArrays[d];
                            auto& deviceBuffers = handle->deviceBuffers[d];
                            const int deviceId = gpuArray->getDeviceId();

                            cub::SwitchDevice sd_{deviceId};

                            CUDACHECK(cudaStreamWaitEvent(deviceBuffers.stream, handle->callerBuffers.event));

                            copy(
                                vec_callerBuffers_d_dataCommunicationBuffer[g].data() + all_numSelectedPerGpu_horizontalPS[g][d] * vec_destRowPitchInBytes[g],
                                destDeviceId,
                                all_deviceBuffers_d_dataCommunicationBuffer[d] + all_numSelectedPerGpu_verticalPS[g][d] * vec_destRowPitchInBytes[g],
                                deviceId,
                                vec_destRowPitchInBytes[g] * num,
                                deviceBuffers.stream
                            );


                            //free temp allocations of this device after data has been copied
                            //vec_all_mrs[g][d]->deallocate(vec_all_rawPtrs[g][d], vec_all_allocatedBytes[g][d], deviceBuffers.stream.getStream());


                            CUDACHECK(cudaEventRecord(deviceBuffers.event, deviceBuffers.stream));

                            // CUDACHECK(cudaSetDevice(destDeviceId));
                            // CUDACHECK(cudaStreamWaitEvent(callerBuffers.streams[d], deviceBuffers.event, 0));
                        }
                    }
                }
            }

            //scatter into result array
            for(int d = 0; d < numDistinctGpus; d++){
                for(int g = 0; g < numDestGpus; g++){
                    if(vec_numIndices[g] > 0){
                        const int destDeviceId = destDeviceIds[g];
                        cub::SwitchDevice sd_{destDeviceId};

                        auto& handle = vec_handle[g];
                        auto& callerBuffers = handle->callerBuffers;

                        MultiSplitResult& splits = splitsPerDestGpu[g];
                        const auto& gpuArray = gpuArrays[d];
                        auto& deviceBuffers = handle->deviceBuffers[d];
                    
                        const int num = all_numSelectedPerGpu[g][d];

                        if(num > 0){
                            CUDACHECK(cudaStreamWaitEvent(callerBuffers.streams[d], deviceBuffers.event, 0));

                            dim3 block(128, 1, 1);
                            dim3 grid(SDIV(vec_numIndices[g], block.x), 1, 1);

                            MultiGpu2dArrayKernels::scatterKernel<<<grid, block, 0, callerBuffers.streams[d]>>>(
                                (const T*)(vec_callerBuffers_d_dataCommunicationBuffer[g].data() + vec_destRowPitchInBytes[g] * all_numSelectedPerGpu_horizontalPS[g][d]), 
                                vec_destRowPitchInBytes[g], 
                                vec_numIndices[g],
                                getNumColumns(),
                                vec_d_dest[g], 
                                vec_destRowPitchInBytes[g], 
                                splits.d_selectedPositions + d * vec_numIndices[g], 
                                num
                            ); CUDACHECKASYNC; 
                        }
                    }
                }
            }

            

            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd_{destDeviceId};

                    auto& handle = vec_handle[g];
                    auto& callerBuffers = handle->callerBuffers;

                    MultiSplitResult& splits = splitsPerDestGpu[g];

                    //join streams
                    for(int d = 0; d < numDistinctGpus; d++){
                        const int num = splits.h_numSelectedPerGpu[d];
                        if(num > 0){
                            CUDACHECK(cudaStreamSynchronize(callerBuffers.streams[d]));
                        }                
                    }             
                }
            }
            
            //done. deallocate

            for(int d = 0; d < numDistinctGpus; d++){

                const auto& gpuArray = gpuArrays[d];
                const int deviceId = gpuArray->getDeviceId();
                auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];
                all_mrs[d]->deallocate(all_rawPtrs[d], all_allocatedBytes[d], deviceBuffers.stream.getStream());
            }

            // //deallocate
            // for(int g = 0; g < numDestGpus; g++){
            //     if(vec_numIndices[g] > 0){                    

            //         auto& handle = vec_handle[g];
            //         auto& callerBuffers = handle->callerBuffers;

            //         MultiSplitResult& splits = splitsPerDestGpu[g];
                    
            //         for(int d = 0; d < numDistinctGpus; d++){
            //             const int num = splits.h_numSelectedPerGpu[d];

            //             if(num > 0){
            //                 const auto& gpuArray = gpuArrays[d];
            //                 auto& deviceBuffers = handle->deviceBuffers[d];
            //                 const int deviceId = gpuArray->getDeviceId();

            //                 cub::SwitchDevice sd_{deviceId};

            //                 vec_all_mrs[g][d]->deallocate(vec_all_rawPtrs[g][d], vec_all_allocatedBytes[g][d], deviceBuffers.stream.getStream());
            //             }
            //         }
            //     }
            // }


        }
    }






    void gather_collective_internal_6(
        std::vector<Handle>& vec_handle, 
        std::vector<T*>& vec_d_dest, 
        const std::vector<size_t>& vec_destRowPitchInBytes, 
        const std::vector<IndexType*>& vec_d_indices, 
        const std::vector<size_t>& vec_numIndices, 
        const std::vector<cudaStream_t>& destStreams,
        const std::vector<int>& destDeviceIds
    ) const{
        const int numDestGpus = destDeviceIds.size();

        if((isSingleGpu() || isReplicatedSingleGpu)){
            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd(destDeviceId);

                    auto it = std::find(usedDeviceIds.begin(), usedDeviceIds.end(), destDeviceId);
                    if(it != usedDeviceIds.end()){
                        //all data to be gathered resides on the destination device.
                        const int position = std::distance(usedDeviceIds.begin(), it);
                        
                        gpuArrays[position]->gather(
                            vec_d_dest[g], 
                            vec_destRowPitchInBytes[g], 
                            vec_d_indices[g], 
                            vec_numIndices[g], 
                            destStreams[g]
                        );
                    }else{
                        //array was not constructed on current device. use peer access from any gpu

                        const auto& gpuArray = gpuArrays[0];
                        auto& deviceBuffers = vec_handle[g]->deviceBuffers[0];
                        const int deviceId = gpuArray->getDeviceId();

                        CUDACHECK(cudaEventRecord(vec_handle[g]->callerBuffers.event, destStreams[g]));
                        
                        CUDACHECK(cudaSetDevice(deviceId));
                        CUDACHECK(cudaStreamWaitEvent(deviceBuffers.stream, vec_handle[g]->callerBuffers.event, 0));

                        //copy indices to other gpu
                        rmm::device_uvector<IndexType> d_indices_target(vec_numIndices[g], deviceBuffers.stream.getStream());
                        copy(
                            d_indices_target.data(), 
                            deviceId, 
                            vec_d_indices[g], 
                            destDeviceId, 
                            sizeof(IndexType) * vec_numIndices[g], 
                            deviceBuffers.stream
                        );

                        rmm::device_buffer d_temp(vec_destRowPitchInBytes[g] * vec_numIndices[g], deviceBuffers.stream.getStream());
                        gpuArrays[0]->gather(reinterpret_cast<T*>(d_temp.data()), vec_destRowPitchInBytes[g], d_indices_target.data(), vec_numIndices[g], deviceBuffers.stream);

                        //copy gathered data to us
                        copy(
                            vec_d_dest[g], 
                            destDeviceId, 
                            d_temp.data(), 
                            deviceId, 
                            vec_destRowPitchInBytes[g] * vec_numIndices[g], 
                            deviceBuffers.stream
                        );

                        CUDACHECK(cudaEventRecord(deviceBuffers.event, deviceBuffers.stream.getStream()));

                        //wait on destStream until gathered data is ready
                        CUDACHECK(cudaSetDevice(destDeviceId));
                        CUDACHECK(cudaStreamWaitEvent(destStreams[g], deviceBuffers.event, 0));
                    }
                }
            }
        }else{

            //perform multisplit to distribute indices and filter out invalid indices. then perform local gathers,
            // and scatter into result array

            const int numGpus = gpuArrays.size();
            const int numDistinctGpus = h_numRowsPerGpu.size();

            if(!isReplicatedSingleGpu){
                assert(numGpus == numDistinctGpus);
            }else{
                assert(1 == numDistinctGpus);
            }
            
            std::vector<MultiSplitResult> splitsPerDestGpu;
            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd_{destDeviceId};
                    
                    auto& handle = vec_handle[g];
                    
                    MultiSplitResult splits = multiSplit_no_h_numSelectedPerGpu(
                        vec_d_indices[g], 
                        vec_numIndices[g], 
                        handle->multisplitTemp.data(), 
                        destStreams[g]
                    );
                    splitsPerDestGpu.push_back(std::move(splits));
                }else{
                    splitsPerDestGpu.emplace_back(0, numDistinctGpus, destStreams[g]);
                }
            }

            std::vector<std::vector<size_t>> all_numSelectedPerGpu(numDestGpus, std::vector<size_t>(numDistinctGpus));
            std::vector<std::vector<size_t>> all_numSelectedPerGpu_horizontalPS(numDestGpus, std::vector<size_t>(numDistinctGpus+1));
            std::vector<std::vector<size_t>> all_numSelectedPerGpu_verticalPS(numDestGpus+1, std::vector<size_t>(numDistinctGpus));

            std::vector<rmm::device_uvector<char>> vec_callerBuffers_d_dataCommunicationBuffer;

            for(int g = 0; g < numDestGpus; g++){
                const int destDeviceId = destDeviceIds[g];
                cub::SwitchDevice sd_{destDeviceId};
                MultiSplitResult& splits = splitsPerDestGpu[g];
                auto& handle = vec_handle[g];
                auto& callerBuffers = handle->callerBuffers;

                if(vec_numIndices[g] > 0){
                    CUDACHECK(cudaStreamSynchronize(destStreams[g])); //wait for multi-split results
                    std::copy(
                        handle->multisplitTemp.data(), 
                        handle->multisplitTemp.data() + numDistinctGpus, 
                        all_numSelectedPerGpu[g].begin()
                    );

                }else{
                    std::fill(all_numSelectedPerGpu[g].begin(), all_numSelectedPerGpu[g].end(), 0);
                }

                std::partial_sum(
                    all_numSelectedPerGpu[g].begin(),
                    all_numSelectedPerGpu[g].begin() + numDistinctGpus,
                    all_numSelectedPerGpu_horizontalPS[g].begin() + 1
                );
            }

            for(int g = 0; g < numDestGpus; g++){
                for(int d = 0; d < numDistinctGpus; d++){
                    all_numSelectedPerGpu_verticalPS[g+1][d] = all_numSelectedPerGpu_verticalPS[g][d] + all_numSelectedPerGpu[g][d];
                }
            }

            // std::cout << "nums\n";
            // for(int g = 0; g < numDestGpus; g++){
            //     for(int d = 0; d < numDistinctGpus; d++){
            //         std::cout << all_numSelectedPerGpu[g][d] << " ";
            //     }
            //     std::cout << "\n";
            // }
            // std::cout << "horizontalPS\n";
            // for(int g = 0; g < numDestGpus; g++){
            //     for(int d = 0; d < numDistinctGpus+1; d++){
            //         std::cout << all_numSelectedPerGpu_horizontalPS[g][d] << " ";
            //     }
            //     std::cout << "\n";
            // }
            // std::cout << "verticalPS\n";
            // for(int g = 0; g < numDestGpus+1; g++){
            //     for(int d = 0; d < numDistinctGpus; d++){
            //         std::cout << all_numSelectedPerGpu_verticalPS[g][d] << " ";
            //     }
            //     std::cout << "\n";
            // }

            std::vector<rmm::device_uvector<char>> all_deviceBuffers_raw;
            std::vector<IndexType*> all_deviceBuffers_d_selectedIndices(numDistinctGpus, nullptr);
            std::vector<char*> all_deviceBuffers_d_dataCommunicationBuffer(numDistinctGpus, nullptr);


            for(int d = 0; d < numDistinctGpus; d++){

                const auto& gpuArray = gpuArrays[d];
                const int deviceId = gpuArray->getDeviceId();
                auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];
                
                cub::SwitchDevice sd(deviceId);

                const int numForAll = all_numSelectedPerGpu_verticalPS[numDestGpus][d];
                
                //allocate remote buffers
                //all_mrs[d] = rmm::mr::get_current_device_resource();
                size_t bytes[2]{
                    SDIV(sizeof(IndexType) * numForAll, 512) * 512, // d_selectedIndices
                    SDIV(vec_destRowPitchInBytes[0 /*g*/] * numForAll, 512) * 512 //d_dataCommunicationBuffer
                };
                const size_t totalBytes = bytes[0] + bytes[1];
                all_deviceBuffers_raw.emplace_back(totalBytes, deviceBuffers.stream.getStream());
                all_deviceBuffers_d_selectedIndices[d] = reinterpret_cast<IndexType*>(all_deviceBuffers_raw[d].data());
                all_deviceBuffers_d_dataCommunicationBuffer[d] = reinterpret_cast<char*>(all_deviceBuffers_raw[d].data() + bytes[0]);
                CUDACHECK(cudaStreamSynchronize(deviceBuffers.stream)); //wait for allocation
            }


            //copy selected indices to remote devices
            for(int g = 0; g < numDestGpus; g++){
                const int destDeviceId = destDeviceIds[g];
                const auto& splits = splitsPerDestGpu[g];
                auto& handle = vec_handle[g];
                auto& callerBuffers = handle->callerBuffers;
                cub::SwitchDevice sd(destDeviceId);

                for(int d = 0; d < numDistinctGpus; d++){
                    if(all_numSelectedPerGpu[g][d] > 0){
                        //send from g to d
                        const auto& gpuArray = gpuArrays[d];
                        const int deviceId = gpuArray->getDeviceId();
                        copy(
                            all_deviceBuffers_d_selectedIndices[d] + all_numSelectedPerGpu_verticalPS[g][d], 
                            deviceId, 
                            splits.d_selectedIndices + d * vec_numIndices[g], 
                            destDeviceId, 
                            sizeof(IndexType) * all_numSelectedPerGpu[g][d], 
                            callerBuffers.streams[d]
                        );
                    }
                }
            }


            //wait until all transfers to a data gpu are complete, then gather 
            for(int d = 0; d < numDistinctGpus; d++){
                const int num = all_numSelectedPerGpu_verticalPS[numDestGpus][d];

                if(num > 0){

                    for(int g = 0; g < numDestGpus; g++){
                        const int destDeviceId = destDeviceIds[g];
                        auto& handle = vec_handle[g];
                        auto& callerBuffers = handle->callerBuffers;
                        cub::SwitchDevice sd(destDeviceId);

                        CUDACHECK(cudaStreamSynchronize(callerBuffers.streams[d]));
                    }

                    const auto& gpuArray = gpuArrays[d];
                    auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];
                    const int deviceId = gpuArray->getDeviceId();

                    cub::SwitchDevice sd(deviceId);            

                    auto gatherIndexIterator = thrust::make_transform_iterator(
                        all_deviceBuffers_d_selectedIndices[d],
                        MultiGpu2dArrayKernels::SubtractFunctor<IndexType>(h_numRowsPerGpuPrefixSum[d])
                    );

                    gpuArray->gather(
                        (T*)all_deviceBuffers_d_dataCommunicationBuffer[d], 
                        vec_destRowPitchInBytes[0/*g*/], 
                        gatherIndexIterator, 
                        num, 
                        deviceBuffers.stream
                    );

                }
            }

            // //wait until all transfers are complete
            // for(int g = 0; g < numDestGpus; g++){
            //     const int destDeviceId = destDeviceIds[g];
            //     auto& handle = vec_handle[g];
            //     auto& callerBuffers = handle->callerBuffers;
            //     cub::SwitchDevice sd(destDeviceId);

            //     for(int d = 0; d < numDistinctGpus; d++){      
            //         CUDACHECK(cudaStreamSynchronize(callerBuffers.streams[d]));
            //     }
            // }

            // //gather on remote gpus
            // for(int d = 0; d < numDistinctGpus; d++){
            //     const int num = all_numSelectedPerGpu_verticalPS[numDestGpus][d];

            //     if(num > 0){
            //         const auto& gpuArray = gpuArrays[d];
            //         auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];
            //         const int deviceId = gpuArray->getDeviceId();

            //         cub::SwitchDevice sd(deviceId);  
            //         //CUDACHECK(cudaStreamWaitEvent(deviceBuffers.stream, callerBuffers.events[d], 0));                    

            //         auto gatherIndexIterator = thrust::make_transform_iterator(
            //             all_deviceBuffers_d_selectedIndices[d],
            //             MultiGpu2dArrayKernels::SubtractFunctor<IndexType>(h_numRowsPerGpuPrefixSum[d])
            //         );

            //         gpuArray->gather(
            //             (T*)all_deviceBuffers_d_dataCommunicationBuffer[d], 
            //             vec_destRowPitchInBytes[0/*g*/], 
            //             gatherIndexIterator, 
            //             num, 
            //             deviceBuffers.stream
            //         );
            //     }
            // }
            //allocate transfer buffers on callers
            for(int g = 0; g < numDestGpus; g++){
                const int destDeviceId = destDeviceIds[g];
                cub::SwitchDevice sd_{destDeviceId};
                auto& handle = vec_handle[g];
                auto& callerBuffers = handle->callerBuffers;

                if(vec_numIndices[g] > 0){
                    //vec_callerBuffers_d_dataCommunicationBuffer.emplace_back(vec_numIndices[g] * vec_destRowPitchInBytes[g], callerBuffers.streams[0].getStream());destStreams[g]
                    vec_callerBuffers_d_dataCommunicationBuffer.emplace_back(vec_numIndices[g] * vec_destRowPitchInBytes[g], destStreams[g]);

                }else{
                    vec_callerBuffers_d_dataCommunicationBuffer.emplace_back(0, destStreams[g]);
                }
                //CUDACHECK(cudaStreamSynchronize(callerBuffers.streams[0]));
                CUDACHECK(cudaStreamSynchronize(destStreams[g]));
            }
            // wait until gathers are complete
            for(int d = 0; d < numDistinctGpus; d++){
                const auto& gpuArray = gpuArrays[d];
                const int deviceId = gpuArray->getDeviceId();
                auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];
                
                cub::SwitchDevice sd(deviceId);
                CUDACHECK(cudaStreamSynchronize(deviceBuffers.stream.getStream()));
            }

            //copy gathered data back
            for(int d = 0; d < numDistinctGpus; d++){
                for(int g = 0; g < numDestGpus; g++){
                    if(vec_numIndices[g] > 0){
                        const int destDeviceId = destDeviceIds[g];                    

                        auto& handle = vec_handle[g];
                        auto& callerBuffers = handle->callerBuffers;

                        MultiSplitResult& splits = splitsPerDestGpu[g];
    
                        const int num = all_numSelectedPerGpu[g][d];

                        if(num > 0){
                            const auto& gpuArray = gpuArrays[d];
                            auto& deviceBuffers = handle->deviceBuffers[d];
                            const int deviceId = gpuArray->getDeviceId();

                            cub::SwitchDevice sd_{deviceId};

                            copy(
                                vec_callerBuffers_d_dataCommunicationBuffer[g].data() + all_numSelectedPerGpu_horizontalPS[g][d] * vec_destRowPitchInBytes[g],
                                destDeviceId,
                                (T*)(all_deviceBuffers_d_dataCommunicationBuffer[d] + all_numSelectedPerGpu_verticalPS[g][d] * vec_destRowPitchInBytes[g]),
                                deviceId,
                                vec_destRowPitchInBytes[g] * num, 
                                deviceBuffers.stream
                            );                            

                            //record that all data has been copied from data gpu d to the different caller gpus
                            CUDACHECK(cudaEventRecord(deviceBuffers.event, deviceBuffers.stream));
                        }
                    }
                }
            }


            // //DEBUG
            // for(int g = 0; g < numDestGpus; g++){
            //     const int destDeviceId = destDeviceIds[g];
            //     cub::SwitchDevice sd_{destDeviceId};
            //     CUDACHECK(cudaDeviceSynchronize());

            //     helpers::lambda_kernel<<<1,1,0,0>>>(
            //         [
            //             g,
            //             ptr =  (const T*)(vec_callerBuffers_d_dataCommunicationBuffer[g].data()), 
            //             size = all_numSelectedPerGpu_horizontalPS[g][numDistinctGpus] * getNumColumns()
            //         ] __device__ (){
            //         printf("vec_callerBuffers_d_dataCommunicationBuffer g %d\n", g);
            //         for(int i = 0; i <  size; i++){
            //             printf("%d ", ptr[i]);
            //         }
            //         printf("\n");
            //     });

            //     helpers::lambda_kernel<<<1,1,0,0>>>(
            //         [
            //             g,
            //             ptr = splitsPerDestGpu[g].d_selectedPositions + 0 * vec_numIndices[g], 
            //             size = vec_numIndices[g]
            //         ] __device__ (){
            //         printf("splits.d_selectedPositions + 0 * vec_numIndices[g] g %d\n", g);
            //         for(int i = 0; i <  size; i++){
            //             printf("%lu ", ptr[i]);
            //         }
            //         printf("\n");
            //     });

            //     helpers::lambda_kernel<<<1,1,0,0>>>(
            //         [
            //             g,
            //             ptr = splitsPerDestGpu[g].d_selectedPositions + 1 * vec_numIndices[g], 
            //             size = vec_numIndices[g]
            //         ] __device__ (){
            //         printf("splits.d_selectedPositions + 1 * vec_numIndices[g] g %d\n", g);
            //         for(int i = 0; i <  size; i++){
            //             printf("%lu ", ptr[i]);
            //         }
            //         printf("\n");
            //     });

                
            // }

            for(int d = 0; d < numDistinctGpus; d++){
                for(int g = 0; g < numDestGpus; g++){
                    if(vec_numIndices[g] > 0){
                        const int destDeviceId = destDeviceIds[g];
                        cub::SwitchDevice sd_{destDeviceId};

                        auto& handle = vec_handle[g];
                        MultiSplitResult& splits = splitsPerDestGpu[g];
                        const auto& gpuArray = gpuArrays[d];
                        auto& deviceBuffers = handle->deviceBuffers[d];
                    
                        const int num = all_numSelectedPerGpu[g][d];

                        if(num > 0){
                            //wait for copy into vec_callerBuffers_d_dataCommunicationBuffer on deviceBuffers.stream
                            CUDACHECK(cudaStreamWaitEvent(destStreams[g], deviceBuffers.event, 0));
                        }
                    }
                }
            }

            //scatter into result array
            for(int d = 0; d < numDistinctGpus; d++){
                for(int g = 0; g < numDestGpus; g++){
                    if(vec_numIndices[g] > 0){
                        const int destDeviceId = destDeviceIds[g];
                        cub::SwitchDevice sd_{destDeviceId};

                        auto& handle = vec_handle[g];
                        auto& callerBuffers = handle->callerBuffers;

                        MultiSplitResult& splits = splitsPerDestGpu[g];
                        const auto& gpuArray = gpuArrays[d];
                        auto& deviceBuffers = handle->deviceBuffers[d];
                    
                        const int num = all_numSelectedPerGpu[g][d];

                        if(num > 0){
                            dim3 block(128, 1, 1);
                            dim3 grid(SDIV(vec_numIndices[g], block.x), 1, 1);

                            MultiGpu2dArrayKernels::scatterKernel<<<grid, block, 0, destStreams[g]>>>(
                                (const T*)(vec_callerBuffers_d_dataCommunicationBuffer[g].data() + all_numSelectedPerGpu_horizontalPS[g][d] * vec_destRowPitchInBytes[g]), 
                                vec_destRowPitchInBytes[g], 
                                vec_numIndices[g],
                                getNumColumns(),
                                vec_d_dest[g], 
                                vec_destRowPitchInBytes[g], 
                                splits.d_selectedPositions + d * vec_numIndices[g], 
                                num
                            ); CUDACHECKASYNC; 
                        }
                    }
                }
            }      

            // for(int g = 0; g < numDestGpus; g++){
            //     if(vec_numIndices[g] > 0){
            //         const int destDeviceId = destDeviceIds[g];
            //         cub::SwitchDevice sd_{destDeviceId};

            //         auto& handle = vec_handle[g];
            //         auto& callerBuffers = handle->callerBuffers;

            //         //join streams
            //         for(int d = 0; d < numDistinctGpus; d++){
            //             const int num = all_numSelectedPerGpu[g][d];

            //             if(num > 0){
            //                 CUDACHECK(cudaEventRecord(callerBuffers.events[d], callerBuffers.streams[d]));
            //                 CUDACHECK(cudaStreamWaitEvent(destStreams[g], callerBuffers.events[d], 0));
            //             }                
            //         }
            //     }
            // }

            
            // //scatter into result array
            // for(int d = 0; d < numDistinctGpus; d++){
            //     for(int g = 0; g < numDestGpus; g++){
            //         if(vec_numIndices[g] > 0){
            //             const int destDeviceId = destDeviceIds[g];
            //             cub::SwitchDevice sd_{destDeviceId};

            //             auto& handle = vec_handle[g];
            //             auto& callerBuffers = handle->callerBuffers;

            //             MultiSplitResult& splits = splitsPerDestGpu[g];
            //             const auto& gpuArray = gpuArrays[d];
            //             auto& deviceBuffers = handle->deviceBuffers[d];
                    
            //             const int num = all_numSelectedPerGpu[g][d];

            //             if(num > 0){
            //                 //wait for copy into vec_callerBuffers_d_dataCommunicationBuffer on deviceBuffers.stream
            //                 CUDACHECK(cudaStreamWaitEvent(callerBuffers.streams[d], deviceBuffers.event, 0));

            //                 dim3 block(128, 1, 1);
            //                 dim3 grid(SDIV(vec_numIndices[g], block.x), 1, 1);

            //                 MultiGpu2dArrayKernels::scatterKernel<<<grid, block, 0, callerBuffers.streams[d]>>>(
            //                     (const T*)(vec_callerBuffers_d_dataCommunicationBuffer[g].data() + all_numSelectedPerGpu_horizontalPS[g][d] * vec_destRowPitchInBytes[g]), 
            //                     vec_destRowPitchInBytes[g], 
            //                     vec_numIndices[g],
            //                     getNumColumns(),
            //                     vec_d_dest[g], 
            //                     vec_destRowPitchInBytes[g], 
            //                     splits.d_selectedPositions + d * vec_numIndices[g], 
            //                     num
            //                 ); CUDACHECKASYNC; 
            //             }
            //         }
            //     }
            // }      

            // for(int g = 0; g < numDestGpus; g++){
            //     if(vec_numIndices[g] > 0){
            //         const int destDeviceId = destDeviceIds[g];
            //         cub::SwitchDevice sd_{destDeviceId};

            //         auto& handle = vec_handle[g];
            //         auto& callerBuffers = handle->callerBuffers;

            //         //join streams
            //         for(int d = 0; d < numDistinctGpus; d++){
            //             const int num = all_numSelectedPerGpu[g][d];

            //             if(num > 0){
            //                 CUDACHECK(cudaEventRecord(callerBuffers.events[d], callerBuffers.streams[d]));
            //                 CUDACHECK(cudaStreamWaitEvent(destStreams[g], callerBuffers.events[d], 0));
            //             }                
            //         }
            //     }
            // }

            for(int g = 0; g < numDestGpus; g++){
                const int destDeviceId = destDeviceIds[g];
                cub::SwitchDevice sd_{destDeviceId};
                CUDACHECK(cudaStreamSynchronize(destStreams[g]));
            }
            //work on all streams is done. deallocate

            for(int g = 0; g < numDestGpus; g++){
                const int destDeviceId = destDeviceIds[g];
                cub::SwitchDevice sd_{destDeviceId};
                vec_callerBuffers_d_dataCommunicationBuffer[g].release();
            }

            for(int d = 0; d < numDistinctGpus; d++){
                const auto& gpuArray = gpuArrays[d];
                const int deviceId = gpuArray->getDeviceId();
                auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];                
                cub::SwitchDevice sd(deviceId);

                all_deviceBuffers_raw[d].release();
            }
        }
    }


    void gather_collective_internal_7(
        std::vector<Handle>& vec_handle, 
        std::vector<T*>& vec_d_dest, 
        const std::vector<size_t>& vec_destRowPitchInBytes, 
        const std::vector<IndexType*>& vec_d_indices, 
        const std::vector<size_t>& vec_numIndices, 
        bool mayContainInvalidIndices,
        const std::vector<cudaStream_t>& destStreams,
        const std::vector<int>& destDeviceIds
    ) const{
        const int numDestGpus = destDeviceIds.size();

        if((isSingleGpu() || isReplicatedSingleGpu) && !mayContainInvalidIndices){
            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd(destDeviceId);

                    auto it = std::find(usedDeviceIds.begin(), usedDeviceIds.end(), destDeviceId);
                    if(it != usedDeviceIds.end()){
                        //all data to be gathered resides on the destination device.
                        const int position = std::distance(usedDeviceIds.begin(), it);
                        
                        gpuArrays[position]->gather(
                            vec_d_dest[g], 
                            vec_destRowPitchInBytes[g], 
                            vec_d_indices[g], 
                            vec_numIndices[g], 
                            destStreams[g]
                        );
                    }else{
                        //array was not constructed on current device. use peer access from any gpu

                        const auto& gpuArray = gpuArrays[0];
                        auto& deviceBuffers = vec_handle[g]->deviceBuffers[0];
                        const int deviceId = gpuArray->getDeviceId();

                        CUDACHECK(cudaEventRecord(vec_handle[g]->callerBuffers.event, destStreams[g]));
                        
                        CUDACHECK(cudaSetDevice(deviceId));
                        CUDACHECK(cudaStreamWaitEvent(deviceBuffers.stream, vec_handle[g]->callerBuffers.event, 0));

                        //copy indices to other gpu
                        rmm::device_uvector<IndexType> d_indices_target(vec_numIndices[g], deviceBuffers.stream.getStream());
                        copy(
                            d_indices_target.data(), 
                            deviceId, 
                            vec_d_indices[g], 
                            destDeviceId, 
                            sizeof(IndexType) * vec_numIndices[g], 
                            deviceBuffers.stream
                        );

                        rmm::device_buffer d_temp(vec_destRowPitchInBytes[g] * vec_numIndices[g], deviceBuffers.stream.getStream());
                        gpuArrays[0]->gather(reinterpret_cast<T*>(d_temp.data()), vec_destRowPitchInBytes[g], d_indices_target.data(), vec_numIndices[g], deviceBuffers.stream);

                        //copy gathered data to us
                        copy(
                            vec_d_dest[g], 
                            destDeviceId, 
                            d_temp.data(), 
                            deviceId, 
                            vec_destRowPitchInBytes[g] * vec_numIndices[g], 
                            deviceBuffers.stream
                        );

                        CUDACHECK(cudaEventRecord(deviceBuffers.event, deviceBuffers.stream.getStream()));

                        //wait on destStream until gathered data is ready
                        CUDACHECK(cudaSetDevice(destDeviceId));
                        CUDACHECK(cudaStreamWaitEvent(destStreams[g], deviceBuffers.event, 0));
                    }
                }
            }
        }else{

            //perform multisplit to distribute indices and filter out invalid indices. then perform local gathers,
            // and scatter into result array

            const int numGpus = gpuArrays.size();
            const int numDistinctGpus = h_numRowsPerGpu.size();

            if(!isReplicatedSingleGpu){
                assert(numGpus == numDistinctGpus);
            }else{
                assert(1 == numDistinctGpus);
            }
            
            std::vector<MultiSplitResult> splitsPerDestGpu;
            for(int g = 0; g < numDestGpus; g++){
                if(vec_numIndices[g] > 0){
                    const int destDeviceId = destDeviceIds[g];
                    cub::SwitchDevice sd_{destDeviceId};
                    
                    auto& handle = vec_handle[g];
                    
                    MultiSplitResult splits = multiSplit_no_h_numSelectedPerGpu(
                        vec_d_indices[g], 
                        vec_numIndices[g], 
                        handle->multisplitTemp.data(), 
                        destStreams[g]
                    );
                    splitsPerDestGpu.push_back(std::move(splits));
                }else{
                    splitsPerDestGpu.emplace_back(0, numDistinctGpus, destStreams[g]);
                }
            }

            std::vector<std::vector<size_t>> all_numSelectedPerGpu(numDestGpus, std::vector<size_t>(numDistinctGpus));
            std::vector<std::vector<size_t>> all_numSelectedPerGpu_horizontalPS(numDestGpus, std::vector<size_t>(numDistinctGpus+1));
            std::vector<std::vector<size_t>> all_numSelectedPerGpu_verticalPS(numDestGpus+1, std::vector<size_t>(numDistinctGpus));

            std::vector<rmm::device_uvector<char>> vec_callerBuffers_d_dataCommunicationBuffer;

            //allocate transfer buffers on callers
            for(int g = 0; g < numDestGpus; g++){
                const int destDeviceId = destDeviceIds[g];
                cub::SwitchDevice sd_{destDeviceId};
                auto& handle = vec_handle[g];
                auto& callerBuffers = handle->callerBuffers;

                if(vec_numIndices[g] > 0){
                    vec_callerBuffers_d_dataCommunicationBuffer.emplace_back(vec_numIndices[g] * vec_destRowPitchInBytes[g], destStreams[g]);

                }else{
                    vec_callerBuffers_d_dataCommunicationBuffer.emplace_back(0, destStreams[g]);
                }
            }

            for(int g = 0; g < numDestGpus; g++){
                const int destDeviceId = destDeviceIds[g];
                cub::SwitchDevice sd_{destDeviceId};
                MultiSplitResult& splits = splitsPerDestGpu[g];
                auto& handle = vec_handle[g];
                auto& callerBuffers = handle->callerBuffers;

                if(vec_numIndices[g] > 0){
                    CUDACHECK(cudaStreamSynchronize(destStreams[g])); //wait for multi-split results and allocation
                    std::copy(
                        handle->multisplitTemp.data(), 
                        handle->multisplitTemp.data() + numDistinctGpus, 
                        all_numSelectedPerGpu[g].begin()
                    );

                }else{
                    std::fill(all_numSelectedPerGpu[g].begin(), all_numSelectedPerGpu[g].end(), 0);
                }

                std::partial_sum(
                    all_numSelectedPerGpu[g].begin(),
                    all_numSelectedPerGpu[g].begin() + numDistinctGpus,
                    all_numSelectedPerGpu_horizontalPS[g].begin() + 1
                );
            }

            for(int g = 0; g < numDestGpus; g++){
                for(int d = 0; d < numDistinctGpus; d++){
                    all_numSelectedPerGpu_verticalPS[g+1][d] = all_numSelectedPerGpu_verticalPS[g][d] + all_numSelectedPerGpu[g][d];
                }
            }

            // std::cout << "nums\n";
            // for(int g = 0; g < numDestGpus; g++){
            //     for(int d = 0; d < numDistinctGpus; d++){
            //         std::cout << all_numSelectedPerGpu[g][d] << " ";
            //     }
            //     std::cout << "\n";
            // }
            // std::cout << "horizontalPS\n";
            // for(int g = 0; g < numDestGpus; g++){
            //     for(int d = 0; d < numDistinctGpus+1; d++){
            //         std::cout << all_numSelectedPerGpu_horizontalPS[g][d] << " ";
            //     }
            //     std::cout << "\n";
            // }
            // std::cout << "verticalPS\n";
            // for(int g = 0; g < numDestGpus+1; g++){
            //     for(int d = 0; d < numDistinctGpus; d++){
            //         std::cout << all_numSelectedPerGpu_verticalPS[g][d] << " ";
            //     }
            //     std::cout << "\n";
            // }

            std::vector<rmm::device_uvector<char>> all_deviceBuffers_raw;
            std::vector<IndexType*> all_deviceBuffers_d_selectedIndices(numDistinctGpus, nullptr);
            std::vector<char*> all_deviceBuffers_d_dataCommunicationBuffer(numDistinctGpus, nullptr);


            for(int d = 0; d < numDistinctGpus; d++){

                const auto& gpuArray = gpuArrays[d];
                const int deviceId = gpuArray->getDeviceId();
                auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];
                
                cub::SwitchDevice sd(deviceId);

                const int numForAll = all_numSelectedPerGpu_verticalPS[numDestGpus][d];
                
                //allocate remote buffers
                //all_mrs[d] = rmm::mr::get_current_device_resource();
                size_t bytes[2]{
                    SDIV(sizeof(IndexType) * numForAll, 512) * 512, // d_selectedIndices
                    SDIV(vec_destRowPitchInBytes[0 /*g*/] * numForAll, 512) * 512 //d_dataCommunicationBuffer
                };
                const size_t totalBytes = bytes[0] + bytes[1];
                all_deviceBuffers_raw.emplace_back(totalBytes, deviceBuffers.stream.getStream());
                all_deviceBuffers_d_selectedIndices[d] = reinterpret_cast<IndexType*>(all_deviceBuffers_raw[d].data());
                all_deviceBuffers_d_dataCommunicationBuffer[d] = reinterpret_cast<char*>(all_deviceBuffers_raw[d].data() + bytes[0]);
                CUDACHECK(cudaStreamSynchronize(deviceBuffers.stream)); //wait for allocation
            }


            //copy selected indices to remote devices
            {
                std::vector<std::vector<const void*>> srcBuffers(numDestGpus, std::vector<const void*>(numDistinctGpus));
                std::vector<std::vector<void*>> dstBuffers(numDestGpus, std::vector<void*>(numDistinctGpus));
                std::vector<std::vector<size_t>> transferSizesBytes(numDestGpus, std::vector<size_t>(numDistinctGpus));
                std::vector<cudaStream_t> srcStreams(numDestGpus);
                const std::vector<int>& srcDeviceIds = destDeviceIds;
                std::vector<int> dstDeviceIds(numDistinctGpus);
                //send values
                for(int g = 0; g < numDestGpus; g++){
                    srcStreams[g] = destStreams[g];
                    const auto& splits = splitsPerDestGpu[g];
                    for(int d = 0; d < numDistinctGpus; d++){
                        srcBuffers[g][d] = splits.d_selectedIndices + d * vec_numIndices[g];
                        transferSizesBytes[g][d] = sizeof(IndexType) * all_numSelectedPerGpu[g][d];
                        dstBuffers[g][d] = all_deviceBuffers_d_selectedIndices[d] + all_numSelectedPerGpu_verticalPS[g][d];
                    }
                }
                for(int d = 0; d < numDistinctGpus; d++){
                    const auto& gpuArray = gpuArrays[d];
                    dstDeviceIds[d] = gpuArray->getDeviceId();                    
                }
                multigpu_transfer(
                    srcDeviceIds,
                    srcBuffers,
                    transferSizesBytes,
                    srcStreams,
                    dstDeviceIds,
                    dstBuffers
                );
            }

            // wait for transfers
            for(int g = 0; g < numDestGpus; g++){
                cub::SwitchDevice sd(destDeviceIds[g]);
                CUDACHECK(cudaStreamSynchronize(destStreams[g]));
            }
          
            //gather from arrays
            for(int d = 0; d < numDistinctGpus; d++){
                const int num = all_numSelectedPerGpu_verticalPS[numDestGpus][d];

                if(num > 0){
                    const auto& gpuArray = gpuArrays[d];
                    auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];
                    const int deviceId = gpuArray->getDeviceId();

                    cub::SwitchDevice sd(deviceId);            

                    auto gatherIndexIterator = thrust::make_transform_iterator(
                        all_deviceBuffers_d_selectedIndices[d],
                        MultiGpu2dArrayKernels::SubtractFunctor<IndexType>(h_numRowsPerGpuPrefixSum[d])
                    );

                    gpuArray->gather(
                        (T*)all_deviceBuffers_d_dataCommunicationBuffer[d], 
                        vec_destRowPitchInBytes[0/*g*/], 
                        gatherIndexIterator, 
                        num, 
                        deviceBuffers.stream
                    );

                }
            }

            //copy gathered data back
            {
                std::vector<std::vector<const void*>> srcBuffers(numDistinctGpus, std::vector<const void*>(numDestGpus));
                std::vector<std::vector<void*>> dstBuffers(numDistinctGpus, std::vector<void*>(numDestGpus));
                std::vector<std::vector<size_t>> transferSizesBytes(numDistinctGpus, std::vector<size_t>(numDestGpus));
                std::vector<cudaStream_t> srcStreams(numDistinctGpus);
                std::vector<int> srcDeviceIds(numDistinctGpus);
                const std::vector<int>& dstDeviceIds = destDeviceIds;

                for(int d = 0; d < numDistinctGpus; d++){
                    srcStreams[d] = vec_handle[0/*g*/]->deviceBuffers[d].stream;
                    for(int g = 0; g < numDestGpus; g++){
                        srcBuffers[d][g] = all_deviceBuffers_d_dataCommunicationBuffer[d] + all_numSelectedPerGpu_verticalPS[g][d] * vec_destRowPitchInBytes[g];
                        transferSizesBytes[d][g] = vec_destRowPitchInBytes[g] * all_numSelectedPerGpu[g][d];
                        dstBuffers[d][g] = vec_callerBuffers_d_dataCommunicationBuffer[g].data() + all_numSelectedPerGpu_horizontalPS[g][d] * vec_destRowPitchInBytes[g];
                    }
                }
                for(int d = 0; d < numDistinctGpus; d++){
                    const auto& gpuArray = gpuArrays[d];
                    srcDeviceIds[d] = gpuArray->getDeviceId();                    
                }
                multigpu_transfer(
                    srcDeviceIds,
                    srcBuffers,
                    transferSizesBytes,
                    srcStreams,
                    dstDeviceIds,
                    dstBuffers
                );
            }

            //wait for transfers
            for(int d = 0; d < numDistinctGpus; d++){
                CUDACHECK(cudaSetDevice(gpuArrays[d]->getDeviceId()));
                CUDACHECK(cudaStreamSynchronize(vec_handle[0/*g*/]->deviceBuffers[d].stream));
            }

            //scatter into result array
            for(int d = 0; d < numDistinctGpus; d++){
                for(int g = 0; g < numDestGpus; g++){
                    if(vec_numIndices[g] > 0){
                        const int destDeviceId = destDeviceIds[g];
                        cub::SwitchDevice sd_{destDeviceId};

                        auto& handle = vec_handle[g];
                        auto& callerBuffers = handle->callerBuffers;

                        MultiSplitResult& splits = splitsPerDestGpu[g];
                        const auto& gpuArray = gpuArrays[d];
                        auto& deviceBuffers = handle->deviceBuffers[d];
                    
                        const int num = all_numSelectedPerGpu[g][d];

                        if(num > 0){
                            dim3 block(128, 1, 1);
                            dim3 grid(SDIV(vec_numIndices[g], block.x), 1, 1);

                            MultiGpu2dArrayKernels::scatterKernel<<<grid, block, 0, destStreams[g]>>>(
                                (const T*)(vec_callerBuffers_d_dataCommunicationBuffer[g].data() + all_numSelectedPerGpu_horizontalPS[g][d] * vec_destRowPitchInBytes[g]), 
                                vec_destRowPitchInBytes[g], 
                                vec_numIndices[g],
                                getNumColumns(),
                                vec_d_dest[g], 
                                vec_destRowPitchInBytes[g], 
                                splits.d_selectedPositions + d * vec_numIndices[g], 
                                num
                            ); CUDACHECKASYNC; 
                        }
                    }
                }
            }      

            for(int g = 0; g < numDestGpus; g++){
                const int destDeviceId = destDeviceIds[g];
                cub::SwitchDevice sd_{destDeviceId};
                CUDACHECK(cudaStreamSynchronize(destStreams[g]));
            }
            //work on all streams is done. deallocate

            for(int g = 0; g < numDestGpus; g++){
                const int destDeviceId = destDeviceIds[g];
                cub::SwitchDevice sd_{destDeviceId};
                vec_callerBuffers_d_dataCommunicationBuffer[g].release();

                auto toDelete = std::move(splitsPerDestGpu[g]);
            }

            for(int d = 0; d < numDistinctGpus; d++){
                const auto& gpuArray = gpuArrays[d];
                const int deviceId = gpuArray->getDeviceId();
                auto& deviceBuffers = vec_handle[0/*g*/]->deviceBuffers[d];                
                cub::SwitchDevice sd(deviceId);

                all_deviceBuffers_raw[d].release();
            }
        }
    }

private:
    struct MultiSplitResult{
        // rmm::device_uvector<IndexType> d_selectedIndices;
        // rmm::device_uvector<std::size_t> d_selectedPositions;
        // rmm::device_uvector<std::size_t> d_numSelectedPerGpu;
        
        IndexType* d_selectedIndices;
        std::size_t* d_selectedPositions;
        std::size_t* d_numSelectedPerGpu;
        std::vector<std::size_t> h_numSelectedPerGpu;
        rmm::device_uvector<char> d_data;

        MultiSplitResult(
            std::size_t numIndices, 
            std::size_t numBins, 
            cudaStream_t stream, 
            rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
        ) : d_data(0, stream, mr){
            resize(numIndices, numBins, stream);
        }

        void resize(std::size_t numIndices, std::size_t numBins, cudaStream_t stream){
            size_t bytes[3]{
                SDIV(sizeof(IndexType) * numIndices * numBins, 512) * 512,
                SDIV(sizeof(std::size_t) * numIndices * numBins, 512) * 512,
                SDIV(sizeof(std::size_t) * numBins, 512) * 512,
            };
            size_t totalBytes = bytes[0] + bytes[1] + bytes[2];
            d_data.resize(totalBytes, stream);

            d_selectedIndices = reinterpret_cast<IndexType*>(d_data.data());
            d_selectedPositions = reinterpret_cast<std::size_t*>(d_data.data() + bytes[0]);
            d_numSelectedPerGpu = reinterpret_cast<std::size_t*>(d_data.data() + bytes[0] + bytes[1]);

            h_numSelectedPerGpu.resize(numBins);
        }
    };

    MultiSplitResult multiSplit(const IndexType* d_indices, std::size_t numIndices, std::size_t* tempmemory, cudaStream_t stream) const{
        const int numDistinctGpus = h_numRowsPerGpu.size();
        auto* srcMR = rmm::mr::get_current_device_resource();

        MultiSplitResult result(numIndices, numDistinctGpus, stream, srcMR);

        CUDACHECK(cudaMemsetAsync(result.d_numSelectedPerGpu, 0, sizeof(std::size_t) * numDistinctGpus, stream));
        MultiGpu2dArrayKernels::partitionSplitKernel<<<SDIV(numIndices, 128), 128, 0, stream>>>(
            result.d_selectedIndices,
            result.d_selectedPositions,
            result.d_numSelectedPerGpu,
            numDistinctGpus,
            h_numRowsPerGpuPrefixSum.data(),
            numIndices,
            d_indices
        ); CUDACHECKASYNC;

        CUDACHECK(cudaMemcpyAsync(
            tempmemory,
            result.d_numSelectedPerGpu,
            sizeof(std::size_t) * numDistinctGpus,
            D2H,
            stream
        ));
        CUDACHECK(cudaStreamSynchronize(stream));
        std::copy(tempmemory, tempmemory + numDistinctGpus, result.h_numSelectedPerGpu.begin());
        
        return result;
    }

    MultiSplitResult multiSplit_no_h_numSelectedPerGpu(const IndexType* d_indices, std::size_t numIndices, std::size_t* tempmemory, cudaStream_t stream) const{
        const int numDistinctGpus = h_numRowsPerGpu.size();
        auto* srcMR = rmm::mr::get_current_device_resource();

        MultiSplitResult result(numIndices, numDistinctGpus, stream, srcMR);

        CUDACHECK(cudaMemsetAsync(result.d_numSelectedPerGpu, 0, sizeof(std::size_t) * numDistinctGpus, stream));
        MultiGpu2dArrayKernels::partitionSplitKernel<<<SDIV(numIndices, 128), 128, 0, stream>>>(
            result.d_selectedIndices,
            result.d_selectedPositions,
            result.d_numSelectedPerGpu,
            numDistinctGpus,
            h_numRowsPerGpuPrefixSum.data(),
            numIndices,
            d_indices
        ); CUDACHECKASYNC;

        CUDACHECK(cudaMemcpyAsync(
            tempmemory,
            result.d_numSelectedPerGpu,
            sizeof(std::size_t) * numDistinctGpus,
            D2H,
            stream
        ));
        
        return result;
    }

    void copy(
        void* dst, 
        int dstDevice, 
        const void* src, 
        int srcDevice, 
        size_t count, 
        cudaStream_t stream = 0
    ) const{
        if(dstDevice == srcDevice && dst == src){
            //std::cerr << "copy into same buffer on device " << srcDevice << ". return\n";
            return;
        }else{
            //std::cerr << "copy from device " << srcDevice << " to device " << dstDevice << "\n";
        }

        cudaError_t status = cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
        if(status != cudaSuccess){
            cudaGetLastError();
            std::cerr << "dst=" << dst << ", "
            << "dstDevice=" << dstDevice << ", "
            << "src=" << src << ", "
            << "srcDevice=" << srcDevice << ", "
            << "count=" << count << ".";
            int current = 0;
            CUDACHECK(cudaGetDevice(&current));
            std::cout << "current deviceid " << current << "\n";
        }
        CUDACHECK(status);
    }

    void init(
        size_t numRows, 
        size_t numColumns, 
        std::vector<size_t> memoryLimits, 
        MultiGpu2dArrayLayout layout,
        MultiGpu2dArrayInitMode initMode
    ){
        assert(numRows > 0);
        assert(numColumns > 0);
        assert(memoryLimits.size() == dataDeviceIds.size());

        this->numRows = numRows;
        this->numColumns = numColumns;

        //init gpu arrays

        const int numGpus = dataDeviceIds.size();

        const size_t minbytesPerRow = sizeof(T) * numColumns;
        const size_t rowPitchInBytes = SDIV(minbytesPerRow, getAlignmentInBytes()) * getAlignmentInBytes();

        std::vector<size_t> maxRowsPerGpu(numGpus);

        for(int i = 0; i < numGpus; i++){
            maxRowsPerGpu[i] = memoryLimits[i] / rowPitchInBytes;
        }

        std::vector<size_t> rowsPerGpu(numGpus, 0);
        size_t remaining = numRows;

        if(layout == MultiGpu2dArrayLayout::EvenShare){
            std::size_t divided = numRows / numGpus;
            for(int outer = 0; outer < numGpus; outer++){
                for(int i = 0; i < numGpus; i++){
                    size_t myrows = std::min(maxRowsPerGpu[i] - rowsPerGpu[i], std::min(divided, remaining));
                    rowsPerGpu[i] += myrows;

                    remaining -= myrows;
                }
            }
        }else{
            assert(layout ==MultiGpu2dArrayLayout::FirstFit);

            for(int i = 0; i < numGpus; i++){
                size_t myrows = std::min(maxRowsPerGpu[i], remaining);
                rowsPerGpu[i] = myrows;

                remaining -= myrows;
            }
        }

        if(initMode == MultiGpu2dArrayInitMode::MustFitCompletely){
            if(remaining > 0){
                throw std::invalid_argument("Cannot fit all array elements into provided memory\n");
            }
        }else{
            assert(initMode == MultiGpu2dArrayInitMode::CanDiscardRows);

            this->numRows -= remaining;
        }

        usedDeviceIds.clear();
        std::vector<int> rowsPerUsedGpu;

        for(int i = 0; i < numGpus; i++){
            if(rowsPerGpu[i] > 0){
                cub::SwitchDevice sd(dataDeviceIds[i]);
                auto arrayptr = std::make_unique<Gpu2dArrayManaged<T>>(
                    rowsPerGpu[i], getNumColumns(), getAlignmentInBytes()
                );
                gpuArrays.push_back(std::move(arrayptr));
                usedDeviceIds.push_back(dataDeviceIds[i]);
                rowsPerUsedGpu.push_back(rowsPerGpu[i]);
            }
        }

        const int numUsedDeviceIds = usedDeviceIds.size();

        h_numRowsPerGpu.resize(numUsedDeviceIds);
        h_numRowsPerGpuPrefixSum.resize(numUsedDeviceIds + 1);

        std::copy(rowsPerUsedGpu.begin(), rowsPerUsedGpu.end(), h_numRowsPerGpu.get());
        std::partial_sum(rowsPerUsedGpu.begin(), rowsPerUsedGpu.end(), h_numRowsPerGpuPrefixSum.get() + 1);
        h_numRowsPerGpuPrefixSum[0] = 0;
    }

    bool isReplicatedSingleGpu = false;

    size_t numRows{};
    size_t numColumns{};
    size_t alignmentInBytes{};
    std::vector<std::unique_ptr<Gpu2dArrayManaged<T>>> gpuArrays{};

    std::vector<int> dataDeviceIds;
    std::vector<int> usedDeviceIds;

    HostBuffer<size_t> h_numRowsPerGpu;
    HostBuffer<size_t> h_numRowsPerGpuPrefixSum;
};





#endif
