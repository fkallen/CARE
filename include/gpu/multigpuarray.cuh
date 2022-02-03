#ifndef MULTI_GPU_ARRAY_HPP
#define MULTI_GPU_ARRAY_HPP


#include <hpc_helpers.cuh>
#include <gpu/cudaerrorcheck.cuh>
#include <gpu/singlegpu2darray.cuh>

#include <memorymanagement.hpp>

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

namespace cg = cooperative_groups;


namespace MultiGpu2dArrayKernels{

    template<class T, class IndexGenerator>
    __global__
    void gatherKernel(
        const T* __restrict__ src, 
        size_t srcRowPitchInBytes, 
        size_t numColumns,
        size_t numRows,
        T* __restrict__ dest, 
        size_t destRowPitchInBytes, 
        IndexGenerator indices, 
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

    template<class T, class IndexGenerator>
    __global__
    void gatherKernel(
        const T* __restrict__ src, 
        size_t srcRowPitchInBytes, 
        size_t numRows,
        size_t numColumns,
        T* __restrict__ dest, 
        size_t destRowPitchInBytes, 
        IndexGenerator indices, 
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

    template<class T, class IndexGenerator>
    __global__
    void scatterKernel(
        const T* __restrict__ src, 
        size_t srcRowPitchInBytes, 
        size_t numRows,
        size_t numColumns,
        T* __restrict__ dest, 
        size_t destRowPitchInBytes, 
        IndexGenerator indices, 
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

    template<class T, class IndexGenerator>
    __global__
    void scatterKernel(
        const T* __restrict__ src, 
        size_t srcRowPitchInBytes, 
        size_t numRows,
        size_t numColumns,
        T* __restrict__ dest, 
        size_t destRowPitchInBytes, 
        IndexGenerator indices, 
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
            using ArgIndexPair = typename cub::ArgIndexInputIterator<IndexType*, size_t>::value_type;

            PerDevice()
                : 
                d_numSelected(1),
                event(cudaEventDisableTiming)
            {        
                cudaGetDevice(&deviceId);
            }

            int deviceId{};
            DeviceBuffer<size_t> d_multigpuarrayOffsets{};
            DeviceBuffer<size_t> d_multigpuarrayOffsetsPrefixSum{};
            DeviceBuffer<size_t> d_numSelected{};
            CudaStream stream{};
            CudaEvent event{};

            DeviceBuffer<char> d_cubTemp{};
            DeviceBuffer<IndexType> d_indices{};
            DeviceBuffer<IndexType> d_selectedIndices{};
            DeviceBuffer<size_t> d_selectedPositions{};
            DeviceBuffer<ArgIndexPair> d_selectedIndicesWithPositions{};
            DeviceBuffer<char> d_dataCommunicationBuffer{};
        };

        struct PerCaller{
            using ArgIndexPair = typename cub::ArgIndexInputIterator<IndexType*, size_t>::value_type;

            PerCaller()
                : 
                d_numSelected(1)
            {        

            }

            DeviceBuffer<size_t> d_multigpuarrayOffsets{};
            DeviceBuffer<size_t> d_multigpuarrayOffsetsPrefixSum{};
            DeviceBuffer<size_t> d_numSelected{};
            

            DeviceBuffer<char> d_cubTemp{};
            DeviceBuffer<IndexType> d_indices{};
            DeviceBuffer<IndexType> d_selectedIndices{};
            DeviceBuffer<size_t> d_selectedPositions{};
            DeviceBuffer<ArgIndexPair> d_selectedIndicesWithPositions{};
            DeviceBuffer<char> d_dataCommunicationBuffer{};

            HostBuffer<size_t> h_numSelected{};


            CudaEvent event{cudaEventDisableTiming};
            std::vector<CudaEvent> events{};
            std::vector<CudaStream> streams{};
        };

        HandleStruct() : syncevent{cudaEventDisableTiming}{
            CUDACHECK(cudaGetDevice(&deviceId));            
        }

        HandleStruct(HandleStruct&&) = default;
        HandleStruct(const HandleStruct&) = delete;

        ~HandleStruct(){
            // auto info = getMemoryInfo();

            // std::cerr << "MultiGpuArray::HandleStruct: host: " << info.host;
            // for(const auto& pair : info.device){
            //     std::cerr << ", device[" << pair.first << "]: " << pair.second;
            // }
            // std::cerr << "\n";                
        }

        int deviceId = 0;
        CudaEvent syncevent{};
        std::vector<PerDevice> deviceBuffers{};
        PerCaller callerBuffers{};

        MemoryUsage getMemoryInfo() const{
            MemoryUsage result{};

            for(const auto& buffer : deviceBuffers){
                const int deviceId = buffer.deviceId;

                result.device[deviceId] += buffer.d_multigpuarrayOffsets.capacityInBytes();
                result.device[deviceId] += buffer.d_numSelected.capacityInBytes();
                result.device[deviceId] += buffer.d_cubTemp.capacityInBytes();
                result.device[deviceId] += buffer.d_indices.capacityInBytes();
                result.device[deviceId] += buffer.d_selectedIndices.capacityInBytes();
                result.device[deviceId] += buffer.d_selectedPositions.capacityInBytes();
                result.device[deviceId] += buffer.d_selectedIndicesWithPositions.capacityInBytes();
                result.device[deviceId] += buffer.d_dataCommunicationBuffer.capacityInBytes();
            }

            result.device[deviceId] += callerBuffers.d_multigpuarrayOffsets.capacityInBytes();
            result.device[deviceId] += callerBuffers.d_multigpuarrayOffsetsPrefixSum.capacityInBytes();
            result.device[deviceId] += callerBuffers.d_numSelected.capacityInBytes();
            result.device[deviceId] += callerBuffers.d_cubTemp.capacityInBytes();
            result.device[deviceId] += callerBuffers.d_indices.capacityInBytes();
            result.device[deviceId] += callerBuffers.d_selectedIndices.capacityInBytes();
            result.device[deviceId] += callerBuffers.d_selectedPositions.capacityInBytes();
            result.device[deviceId] += callerBuffers.d_selectedIndicesWithPositions.capacityInBytes();           
            result.device[deviceId] += callerBuffers.d_dataCommunicationBuffer.capacityInBytes();

            result.host = callerBuffers.h_numSelected.capacityInBytes();

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

        const int numDataGpus = usedDeviceIds.size();
        for(int d = 0; d < numDataGpus; d++){
            const int deviceId = usedDeviceIds[d];
            cub::SwitchDevice sd(deviceId);

            typename HandleStruct::PerDevice deviceBuffers; 

            deviceBuffers.d_multigpuarrayOffsets.resize(h_numRowsPerGpu.size());
            deviceBuffers.d_multigpuarrayOffsetsPrefixSum.resize(h_numRowsPerGpuPrefixSum.size());

            CUDACHECK(cudaMemcpy(
                deviceBuffers.d_multigpuarrayOffsets.get(),
                h_numRowsPerGpu.get(),
                h_numRowsPerGpu.size() * sizeof(std::size_t),
                H2D
            ));

            CUDACHECK(cudaMemcpy(
                deviceBuffers.d_multigpuarrayOffsetsPrefixSum.get(),
                h_numRowsPerGpuPrefixSum.get(),
                h_numRowsPerGpuPrefixSum.size() * sizeof(std::size_t),
                H2D
            ));

            handle->deviceBuffers.emplace_back(std::move(deviceBuffers));
        }

        if(h_numRowsPerGpu.size() > 0){

            auto& callerBuffers = handle->callerBuffers;
            callerBuffers.streams.resize(h_numRowsPerGpu.size());
            
            for(int i = 0; i < int(h_numRowsPerGpu.size()); i++){
                callerBuffers.events.emplace_back(cudaEventDisableTiming);
            }
            CUDACHECKASYNC;
            callerBuffers.d_multigpuarrayOffsets.resize(h_numRowsPerGpu.size());
            callerBuffers.d_multigpuarrayOffsetsPrefixSum.resize(h_numRowsPerGpuPrefixSum.size());
            callerBuffers.d_numSelected.resize(h_numRowsPerGpu.size());
            callerBuffers.h_numSelected.resize(h_numRowsPerGpu.size());

            CUDACHECK(cudaMemcpy(
                callerBuffers.d_multigpuarrayOffsets.get(),
                h_numRowsPerGpu.get(),
                h_numRowsPerGpu.size() * sizeof(std::size_t),
                H2D
            ));

            CUDACHECK(cudaMemcpy(
                callerBuffers.d_multigpuarrayOffsetsPrefixSum.get(),
                h_numRowsPerGpuPrefixSum.get(),
                h_numRowsPerGpuPrefixSum.size() * sizeof(std::size_t),
                H2D
            ));  

        } 

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

        //TODO maybe perform batched gather to limit memory usage of handle

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

        //TODO maybe perform batched gather to limit memory usage of handle

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

        //TODO maybe perform batched scatter to limit memory usage of handle

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

        //TODO maybe perform batched scatter to limit memory usage of handle

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


    void gather(T* d_dest, size_t destRowPitchInBytes, size_t rowBegin, size_t rowEnd, cudaStream_t stream = 0) const{
        assert(false && "not implemented");
    }

    

    void scatter(const T* d_src, size_t srcRowPitchInBytes, size_t rowBegin, size_t rowEnd, cudaStream_t stream = 0) const{
        assert(false && "not implemented");
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

//these should be private, but this would cause problems with cuda device lambdas
public:

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
                
                auto indexGenerator = [d_indices] __device__ (auto i){
                    return d_indices[i];
                };

                gpuArrays[position]->gather(d_dest, destRowPitchInBytes, indexGenerator, numIndices, destStream);
            }else{
                //array was not constructed on current device. use peer access from any gpu
                auto indexGenerator = [d_indices] __device__ (auto i){
                    return d_indices[i];
                };

                //TODO check if this code path still works with malloc async 
                
                gpuArrays[0]->gather(d_dest, destRowPitchInBytes, indexGenerator, numIndices, destStream);
            }
        }else{

            //perform multisplit to distribute indices and filter out invalid indices. then perform local gathers,
            // and scatter into result array

            bool hasSynchronized = false;
            auto resizeWithSync = [&](auto& data, std::size_t size){
                using W = decltype(*data.get());

                const std::size_t currentCapacity = data.capacityInBytes();
                const std::size_t newbytes = size * sizeof(W);
                if(!hasSynchronized && currentCapacity < newbytes){
                    handle->syncevent.synchronize();
                    hasSynchronized = true;
                    //std::cerr << "SYNC" << "\n";
                }
                data.resize(size);
            };

            const int numGpus = gpuArrays.size();
            const int numDistinctGpus = h_numRowsPerGpu.size();

            if(!isReplicatedSingleGpu){
                assert(numGpus == numDistinctGpus);
            }else{
                assert(1 == numDistinctGpus);
            }

            auto& callerBuffers = handle->callerBuffers;

            resizeWithSync(callerBuffers.d_selectedIndices, numIndices * gpuArrays.size());
            resizeWithSync(callerBuffers.d_selectedPositions, numIndices * gpuArrays.size());
            resizeWithSync(callerBuffers.d_dataCommunicationBuffer, numIndices * destRowPitchInBytes);

            CUDACHECK(cudaMemsetAsync(callerBuffers.d_numSelected.get(), 0, callerBuffers.d_numSelected.sizeInBytes(), destStream));

            //helpers::GpuTimer gpuTimer(destStream, "partitionsplit");

            MultiGpu2dArrayKernels::partitionSplitKernel<<<SDIV(numIndices, 128), 128, 0, destStream>>>(
                callerBuffers.d_selectedIndices.get(),
                callerBuffers.d_selectedPositions.get(),
                callerBuffers.d_numSelected.get(),
                numDistinctGpus,
                callerBuffers.d_multigpuarrayOffsetsPrefixSum.get(),
                numIndices,
                d_indices
            );

            CUDACHECK(cudaMemcpyAsync(
                callerBuffers.h_numSelected.data(), 
                callerBuffers.d_numSelected.data(), 
                sizeof(std::size_t) * numDistinctGpus, 
                D2H, 
                destStream
            ));
            CUDACHECK(cudaStreamSynchronize(destStream));

            std::vector<std::size_t> numSelectedPrefixSum(numDistinctGpus,0);
            std::partial_sum(
                callerBuffers.h_numSelected.data(),
                callerBuffers.h_numSelected.data() + numDistinctGpus - 1,
                numSelectedPrefixSum.begin() + 1
            );

            // auto it = std::find(usedDeviceIds.begin(), usedDeviceIds.end(), destDeviceId);
            // if(isReplicatedSingleGpu){
            //     const int num = callerBuffers.h_numSelected[0];
            //     if(num > 0){
            //         auto indexGenerator = [d_indices = callerBuffers.d_selectedIndices.data()] __device__ (auto i){
            //             return d_indices[i];
            //         };

            //         const int position = (it != usedDeviceIds.end()) ? std::distance(usedDeviceIds.begin(), it) : 0;

            //         auto& deviceBuffers = handle->deviceBuffers[position];
            //         resizeWithSync(deviceBuffers.d_dataCommunicationBuffer, num * destRowPitchInBytes);

            //         gpuArrays[position]->gather(
            //             (T*)deviceBuffers.d_dataCommunicationBuffer.data(), 
            //             destRowPitchInBytes, 
            //             indexGenerator, 
            //             num, 
            //             destStream
            //         );

            //         dim3 block(128, 1, 1);
            //         dim3 grid(SDIV(numIndices, block.x), 1, 1);

            //         MultiGpu2dArrayKernels::LinearAccessFunctor<std::size_t> scatterIndexGenerator(
            //             callerBuffers.d_selectedPositions.get()
            //         );

            //         MultiGpu2dArrayKernels::scatterKernel<<<grid, block, 0, destStream>>>(
            //             (const T*)(deviceBuffers.d_dataCommunicationBuffer.data()), 
            //             destRowPitchInBytes, 
            //             numIndices,
            //             getNumColumns(),
            //             d_dest, 
            //             destRowPitchInBytes, 
            //             scatterIndexGenerator, 
            //             num
            //         ); CUDACHECKASYNC; 
            //     }
            // }else
            {
                //allocate remote buffers
                for(int d = 0; d < numDistinctGpus; d++){
                    const int num = callerBuffers.h_numSelected[d];

                    if(num > 0){
                        const auto& gpuArray = gpuArrays[d];
                        auto& deviceBuffers = handle->deviceBuffers[d];
                        const int deviceId = gpuArray->getDeviceId();

                        cub::SwitchDevice sd(deviceId);

                        resizeWithSync(deviceBuffers.d_selectedIndices, num);
                        resizeWithSync(deviceBuffers.d_dataCommunicationBuffer, num * destRowPitchInBytes);
                    }
                }

                //copy selected indices to remote buffer
                for(int d = 0; d < numDistinctGpus; d++){
                    const int num = callerBuffers.h_numSelected[d];

                    if(num > 0){
                        const auto& gpuArray = gpuArrays[d];
                        auto& deviceBuffers = handle->deviceBuffers[d];
                        const int deviceId = gpuArray->getDeviceId();

                        cub::SwitchDevice sd(deviceId);

                        //copy selected indices to remote device
                        copy(
                            deviceBuffers.d_selectedIndices.data(), 
                            deviceId, 
                            callerBuffers.d_selectedIndices.data() + d * numIndices, 
                            destDeviceId, 
                            sizeof(IndexType) * num, 
                            deviceBuffers.stream
                        );
                    }
                }

                //gather on remote gpus
                for(int d = 0; d < numDistinctGpus; d++){
                    const int num = callerBuffers.h_numSelected[d];

                    if(num > 0){
                        const auto& gpuArray = gpuArrays[d];
                        auto& deviceBuffers = handle->deviceBuffers[d];
                        const int deviceId = gpuArray->getDeviceId();

                        cub::SwitchDevice sd(deviceId);

                        auto gatherIndexGenerator = [
                            d_selectedIndices = deviceBuffers.d_selectedIndices.data(),
                            indexoffset = h_numRowsPerGpuPrefixSum[d]
                        ] __device__ (auto i){
                            const IndexType index = d_selectedIndices[i];
                            return index - indexoffset; //transform into local index for this gpuArray
                        };

                        gpuArray->gather(
                            (T*)deviceBuffers.d_dataCommunicationBuffer.get(), 
                            destRowPitchInBytes, 
                            gatherIndexGenerator, 
                            num, 
                            deviceBuffers.stream
                        );

                        CUDACHECK(cudaEventRecord(deviceBuffers.event, deviceBuffers.stream));
                    }
                }

                //copy remote gathered data to caller
                for(int d = 0; d < numDistinctGpus; d++){
                    const int num = callerBuffers.h_numSelected[d];

                    if(num > 0){
                        const auto& gpuArray = gpuArrays[d];
                        auto& deviceBuffers = handle->deviceBuffers[d];
                        const int deviceId = gpuArray->getDeviceId();

                        CUDACHECK(cudaStreamWaitEvent(callerBuffers.streams[d], deviceBuffers.event, 0));
    
                        copy(
                            callerBuffers.d_dataCommunicationBuffer.data() + destRowPitchInBytes * numSelectedPrefixSum[d], 
                            destDeviceId, 
                            deviceBuffers.d_dataCommunicationBuffer.data(), 
                            deviceId, 
                            destRowPitchInBytes * num, 
                            callerBuffers.streams[d]
                        );
                    }
                }

                //scatter into result array
                for(int d = 0; d < numDistinctGpus; d++){
                    const int num = callerBuffers.h_numSelected[d];

                    if(num > 0){
                        dim3 block(128, 1, 1);
                        dim3 grid(SDIV(numIndices, block.x), 1, 1);

                        MultiGpu2dArrayKernels::LinearAccessFunctor<std::size_t> scatterIndexGenerator(
                            callerBuffers.d_selectedPositions.get() + d * numIndices
                        );

                        MultiGpu2dArrayKernels::scatterKernel<<<grid, block, 0, callerBuffers.streams[d]>>>(
                            (const T*)(callerBuffers.d_dataCommunicationBuffer.data() + destRowPitchInBytes * numSelectedPrefixSum[d]), 
                            destRowPitchInBytes, 
                            numIndices,
                            getNumColumns(),
                            d_dest, 
                            destRowPitchInBytes, 
                            scatterIndexGenerator, 
                            num
                        ); CUDACHECKASYNC; 

                        CUDACHECK(cudaEventRecord(callerBuffers.events[d], callerBuffers.streams[d]));
                    }
                }

                //join streams
                for(int d = 0; d < numDistinctGpus; d++){
                    const int num = callerBuffers.h_numSelected[d];

                    if(num > 0){
                        CUDACHECK(cudaStreamWaitEvent(destStream, callerBuffers.events[d], 0));
                    }                
                }
            }

            handle->syncevent.record(destStream);            
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
            auto indexGenerator = [d_indices] __device__ (auto i){
                return d_indices[i];
            };

            gpuArrays[0]->scatter(d_src, srcRowPitchInBytes, indexGenerator, numIndices, srcStream);
            return;
        }else{

            bool hasSynchronized = false;
            auto resizeWithSync = [&](auto& data, std::size_t size){
                using W = decltype(*data.get());

                const std::size_t currentCapacity = data.capacityInBytes();
                const std::size_t newbytes = size * sizeof(W);
                if(!hasSynchronized && currentCapacity < newbytes){
                    handle->syncevent.synchronize();
                    hasSynchronized = true;
                    //std::cerr << "SYNC" << "\n";
                }
                data.resize(size);
            };

            const int numGpus = gpuArrays.size();
            cub::ArgIndexInputIterator<const IndexType*, size_t> d_indicesWithPosition(d_indices);

            auto& callerBuffers = handle->callerBuffers;
            resizeWithSync(callerBuffers.d_selectedIndicesWithPositions, numIndices);
            resizeWithSync(callerBuffers.d_dataCommunicationBuffer, numIndices * srcRowPitchInBytes);

            size_t temp_storage_bytes = 0;
            {
                auto selectOp = [
                    d = numGpus,
                    ps = callerBuffers.d_multigpuarrayOffsetsPrefixSum.get()
                ] __device__ (auto indexPositionPair){
                    //key == position, value == index
                    //select if index belongs to device
                    return (ps[d] <= indexPositionPair.value && indexPositionPair.value < ps[d+1]);
                };

                CUDACHECK(cub::DeviceSelect::If(
                    nullptr, 
                    temp_storage_bytes, 
                    d_indicesWithPosition, 
                    callerBuffers.d_selectedIndicesWithPositions.get(), 
                    callerBuffers.d_numSelected.get(), 
                    numIndices, 
                    selectOp, 
                    srcStream
                ));
            }

            resizeWithSync(callerBuffers.d_cubTemp, temp_storage_bytes);

            for(int d = 0; d < numGpus; d++){
                const auto& gpuArray = gpuArrays[d];
                const int deviceId = gpuArray->getDeviceId();
                cub::SwitchDevice sddev(deviceId);

                auto& targetDeviceBuffers = handle->deviceBuffers[d];
                resizeWithSync(targetDeviceBuffers.d_selectedIndicesWithPositions, numIndices);
                resizeWithSync(targetDeviceBuffers.d_dataCommunicationBuffer, numIndices * srcRowPitchInBytes);
            }

            for(int d = 0; d < numGpus; d++){
                const auto& gpuArray = gpuArrays[d];
                const int deviceId = gpuArray->getDeviceId();

                auto& targetDeviceBuffers = handle->deviceBuffers[d];               

                auto selectOp = [
                    d,
                    ps = callerBuffers.d_multigpuarrayOffsetsPrefixSum.get()
                ] __device__ (auto indexPositionPair){
                    //key == position, value == index
                    //select if index belongs to device
                    return (ps[d] <= indexPositionPair.value && indexPositionPair.value < ps[d+1]);
                };

                //select indices for gpuArray
                CUDACHECK(cub::DeviceSelect::If(
                    callerBuffers.d_cubTemp.get(), 
                    temp_storage_bytes, 
                    d_indicesWithPosition, 
                    callerBuffers.d_selectedIndicesWithPositions.get(), 
                    callerBuffers.d_numSelected.get(), 
                    numIndices, 
                    selectOp, 
                    srcStream
                ));

                auto gatherIndexGenerator = [
                    d_selectedIndicesWithPositions = callerBuffers.d_selectedIndicesWithPositions.get(), 
                    ps = callerBuffers.d_multigpuarrayOffsetsPrefixSum.get()
                ] __device__ (auto i){
                    const IndexType index = d_selectedIndicesWithPositions[i].key; // position in array
                    return index; 
                };

                //local gather data which should be scattered to current gpuArray
                dim3 block(128, 1, 1);
                dim3 grid(SDIV(numIndices, block.x), 1, 1);

                MultiGpu2dArrayKernels::gatherKernel<<<grid, block, 0, srcStream>>>(
                    d_src, 
                    srcRowPitchInBytes, 
                    getNumRows(),
                    getNumColumns(),
                    (T*)callerBuffers.d_dataCommunicationBuffer.get(), 
                    srcRowPitchInBytes, 
                    gatherIndexGenerator, 
                    callerBuffers.d_numSelected.get()
                ); CUDACHECKASYNC;

                copy(
                    targetDeviceBuffers.d_numSelected.get(), 
                    deviceId, 
                    callerBuffers.d_numSelected.get(), 
                    srcDeviceId, 
                    sizeof(size_t), 
                    srcStream
                );

                copy(
                    targetDeviceBuffers.d_selectedIndicesWithPositions.get(), 
                    deviceId, 
                    callerBuffers.d_selectedIndicesWithPositions.get(), 
                    srcDeviceId, 
                    sizeof(typename cub::ArgIndexInputIterator<IndexType*, size_t>::value_type) * numIndices, 
                    srcStream
                );

                copy(
                    targetDeviceBuffers.d_dataCommunicationBuffer.get(), 
                    deviceId, 
                    callerBuffers.d_dataCommunicationBuffer.get(), 
                    srcDeviceId, 
                    srcRowPitchInBytes * numIndices, 
                    srcStream
                );

                CUDACHECK(cudaEventRecord(callerBuffers.event, srcStream));

                //scatter into target gpuArray
                cub::SwitchDevice sd(deviceId);

                CUDACHECK(cudaStreamWaitEvent(targetDeviceBuffers.stream, callerBuffers.event, 0));

                auto scatterIndexGenerator = [
                    d,
                    d_selectedIndicesWithPositions = targetDeviceBuffers.d_selectedIndicesWithPositions.get(),
                    ps = targetDeviceBuffers.d_multigpuarrayOffsetsPrefixSum.get()
                ] __device__ (auto i){
                    const IndexType index = d_selectedIndicesWithPositions[i].value;
                    return index - ps[d]; //transform into local index for this gpuArray
                };

                gpuArray->scatter(
                    (const T*)targetDeviceBuffers.d_dataCommunicationBuffer.get(), 
                    srcRowPitchInBytes, 
                    scatterIndexGenerator, 
                    targetDeviceBuffers.d_numSelected.get(),
                    numIndices, 
                    targetDeviceBuffers.stream
                );

                CUDACHECK(cudaEventRecord(targetDeviceBuffers.event, targetDeviceBuffers.stream));

                //gpuArray->print();
            }

            //join multi gpu work to srcStream
            for(int d = 0; d < numGpus; d++){
                const auto& gpuArray = gpuArrays[d];
                const int deviceId = gpuArray->getDeviceId();

                CUDACHECK(cudaStreamWaitEvent(srcStream, handle->deviceBuffers[d].event, 0));
            }

            CUDACHECK(handle->syncevent.record(srcStream));
        }
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
            << "count=" << count << "\n";
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

        if(layout == MultiGpu2dArrayLayout::EvenShare){
            std::cerr << "Layout::EvenShare not implemented. Will use Layout::FirstFit\n";
        }

        std::vector<size_t> rowsPerGpu(numGpus);

        size_t remaining = numRows;
        for(int i = 0; i < numGpus; i++){
            size_t myrows = std::min(maxRowsPerGpu[i], remaining);
            rowsPerGpu[i] = myrows;

            remaining -= myrows;

            //std::cerr << rowsPerGpu[i] << " ";
        }

        //std::cerr << ", remaining " << remaining << "\n";

        if(initMode == MultiGpu2dArrayInitMode::MustFitCompletely){
            if(remaining > 0){
                throw std::invalid_argument("Cannot fit all array elements into provided memory\n");
            }
        }else{
            //InitMode::CanDiscardRows

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

        // std::cerr << "multigpuarray offsets prefixsum: ";
        // for(int i = 0; i < numGpus+1; i++){
        //     std::cerr << h_numRowsPerGpuPrefixSum[i] << " ";
        // }
        // std::cerr << "\n";

    }

    // int getDeviceIdIndex(int deviceId) const{
    //     auto it = std::find(dataDeviceIds.begin(), dataDeviceIds.end(), deviceId);
    //     if(it != dataDeviceIds.end()){
    //         return std::distance(dataDeviceIds.begin(), it);
    //     }else{
    //         assert(false);
    //         return -1;
    //     }
    // }

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