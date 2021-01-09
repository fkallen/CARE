#ifndef CARE_MULTIGPUREADSTORAGE_CUH
#define CARE_MULTIGPUREADSTORAGE_CUH

#include <readstorage.hpp>
#include <gpu/gpureadstorage.cuh>

#include <gpu/multigpuarray.cuh>
#include <sequencehelpers.hpp>
#include <lengthstorage.hpp>
#include <gpu/gpulengthstorage.hpp>
#include <gpu/gpubitarray.cuh>
#include <sharedmutex.hpp>


#include <vector>
#include <cstdint>
#include <memory>
#include <map>
#include <limits>
#include <array>

#include <cub/cub.cuh>

namespace care{
namespace gpu{


class MultiGpuReadStorage : public GpuReadStorage {
public:

    template<class W>
    using HostBuffer = helpers::SimpleAllocationPinnedHost<W>;
    template<class W>
    using DeviceBuffer = helpers::SimpleAllocationDevice<W>;

    using IndexType = care::read_number;

    struct TempData{

        TempData() : event{cudaEventDisableTiming}{
            cudaGetDevice(&deviceId); CUERR;
        }

        ~TempData(){
            // auto info = getMemoryInfo();
            // std::cerr << "MultiGpuReadStorage::TempData: host: " << info.host 
            //     << ", device[" << deviceId << "]: " << info.device[deviceId] << "\n";
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage result{};

            result.host += pinnedBuffer.capacityInBytes();
            result.device[deviceId] += tempbuffer.capacityInBytes();

            result += handleSequences->getMemoryInfo();
            result += handleQualities->getMemoryInfo();

            return result;
        }

        int deviceId{};
        CudaEvent event{};
        DeviceBuffer<char> tempbuffer{};
        HostBuffer<char> pinnedBuffer{};
        std::array<CudaStream,2> streams{};

        typename MultiGpu2dArray<unsigned int, IndexType>::Handle handleSequences{};
        typename MultiGpu2dArray<char, IndexType>::Handle handleQualities{};
    };

    MultiGpuReadStorage(
        const cpu::ContiguousReadStorage& cpuReadStorage_, 
        std::vector<int> deviceIds_, 
        std::vector<std::size_t> memoryLimitsPerDevice_,
        std::size_t memoryLimitHost
    ){

        rebuild(
            cpuReadStorage_,
            deviceIds_,
            memoryLimitsPerDevice_,
            memoryLimitHost
        );

    }

    void rebuild(
        const cpu::ContiguousReadStorage& cpuReadStorage_, 
        std::vector<int> deviceIds_, 
        std::vector<std::size_t> memoryLimitsPerDevice,
        std::size_t memoryLimitHost
    ){
        destroyReadData();

        cpuReadStorage = &cpuReadStorage_;
        deviceIds = std::move(deviceIds_);

        const int numGpus = deviceIds.size();
        const std::size_t numReads = cpuReadStorage->getNumberOfReads();
        numberOfAmbiguousReads = cpuReadStorage->getNumberOfReadsWithN();
        numberOfReads = cpuReadStorage->getNumberOfReads();
        useQualityScores = cpuReadStorage->canUseQualityScores();
        sequenceLengthLowerBound = cpuReadStorage->getSequenceLengthLowerBound();
        sequenceLengthUpperBound = cpuReadStorage->getSequenceLengthUpperBound();

        gpuLengthStorage = std::move(GPULengthStore2<std::uint32_t>(cpuReadStorage->getLengthStore(), deviceIds));

        auto memInfoLengths = gpuLengthStorage.getMemoryInfo();

        for(int d = 0; d < numGpus; d++){            
            if(memoryLimitsPerDevice[d] >= memInfoLengths.device[deviceIds[d]]){
                memoryLimitsPerDevice[d] -= memInfoLengths.device[deviceIds[d]];
            }else{
                memoryLimitsPerDevice[d] = 0;
            }
        }

        for(int d = 0; d < numGpus; d++){
            const int deviceId = deviceIds[d];
            cub::SwitchDevice sd(deviceId); CUERR;

            bitArraysUndeterminedBase[deviceId] = makeGpuBitArray<read_number>(numReads);

            if(memoryLimitsPerDevice[d] >= bitArraysUndeterminedBase[deviceId].numAllocatedBytes){
                memoryLimitsPerDevice[d] -= bitArraysUndeterminedBase[deviceId].numAllocatedBytes;
            }else{
                memoryLimitsPerDevice[d] = 0;
            }

            const read_number* ambiguousIds = cpuReadStorage->getAmbiguousIds();

            const int numAmbiguous = cpuReadStorage->getNumberOfReadsWithN();

            if(numAmbiguous > 0){

                constexpr int batchsize = 1000000;
                const int numBatches = SDIV(numAmbiguous, batchsize);

                DeviceBuffer<bool> d_values(batchsize);
                cudaMemset(d_values, 1, sizeof(bool) * batchsize); CUERR;

                DeviceBuffer<read_number> d_positions(batchsize);

                for(int i = 0; i < numBatches; i++){
                    size_t begin = i * batchsize;
                    size_t end = std::min((i+1) * batchsize, numAmbiguous);
                    size_t elements = end - begin;

                    cudaMemcpy(
                        d_positions.data(), 
                        ambiguousIds + begin, 
                        sizeof(read_number) * elements, 
                        H2D
                    ); CUERR;

                    setBitarray<<<SDIV(elements, 128), 128>>>(
                        bitArraysUndeterminedBase[deviceId], 
                        d_values.data(), 
                        d_positions.data(), 
                        elements
                    ); CUERR;

                    cudaDeviceSynchronize(); CUERR;
                }
            }
        }

        

        

        //handle sequences
        const int numColumnsSequences = SequenceHelpers::getEncodedNumInts2Bit(cpuReadStorage->getSequenceLengthUpperBound());
        sequencesGpu = std::move(
            MultiGpu2dArray<unsigned int, IndexType>(
                numReads,
                numColumnsSequences,
                sizeof(unsigned int),
                deviceIds,
                memoryLimitsPerDevice,
                MultiGpu2dArrayLayout::FirstFit,
                MultiGpu2dArrayInitMode::CanDiscardRows
            )
        );

        //std::cerr << "getNumberOfReads(): " << getNumberOfReads() << ", sequencesGpu.getNumRows(): " << sequencesGpu.getNumRows() << "\n";

        {
            std::size_t batchsize = 65000;
            CudaStream stream{};
            auto arrayhandle = sequencesGpu.makeHandle();

            helpers::SimpleAllocationPinnedHost<IndexType> h_indices(batchsize);
            helpers::SimpleAllocationDevice<unsigned int> d_data(batchsize * cpuReadStorage->getSequencePitch() / sizeof(unsigned int));

            for(std::size_t i = 0; i < sequencesGpu.getNumRows(); i += batchsize){
                const std::size_t currentBatchsize = std::min(batchsize, sequencesGpu.getNumRows() - i);
                std::iota(h_indices.begin(), h_indices.begin() + currentBatchsize, i);

                cudaMemcpyAsync(
                    d_data.data(),
                    (const char*)(cpuReadStorage->getSequenceArray()) + (i) * cpuReadStorage->getSequencePitch(),
                    cpuReadStorage->getSequencePitch() * currentBatchsize,
                    H2D,
                    stream
                ); CUERR;

                sequencesGpu.scatter(
                    arrayhandle, 
                    d_data.data(), 
                    cpuReadStorage->getSequencePitch(), 
                    h_indices.data(), 
                    currentBatchsize, 
                    stream
                );

                cudaStreamSynchronize(stream); CUERR;
            }
        }

        auto meminfo = sequencesGpu.getMemoryInfo();
        for(int i = 0; i < numGpus; i++){
            if(memoryLimitsPerDevice[i] > meminfo.device[deviceIds[i]]){
                memoryLimitsPerDevice[i] -= meminfo.device[deviceIds[i]];
            }else{
                memoryLimitsPerDevice[i] = 0;
            }
        }

        if(canUseQualityScores()){

            const int numColumnsQualities = cpuReadStorage->getSequenceLengthUpperBound();
            qualitiesGpu = std::move(
                MultiGpu2dArray<char, IndexType>(
                    numReads,
                    numColumnsQualities,
                    sizeof(char),
                    deviceIds,
                    memoryLimitsPerDevice,
                    MultiGpu2dArrayLayout::FirstFit,
                    MultiGpu2dArrayInitMode::CanDiscardRows
                )
            );

            {
                const std::size_t batchsize = 65000;
                CudaStream stream{};
                auto arrayhandle = qualitiesGpu.makeHandle();

                helpers::SimpleAllocationPinnedHost<IndexType> h_indices(batchsize);
                helpers::SimpleAllocationDevice<char> d_data(batchsize * cpuReadStorage->getQualityPitch());

                for(std::size_t i = 0; i < qualitiesGpu.getNumRows(); i += batchsize){
                    const std::size_t currentBatchsize = std::min(batchsize, qualitiesGpu.getNumRows() - i);
                    std::iota(h_indices.begin(), h_indices.begin() + currentBatchsize, i);

                    cudaMemcpyAsync(
                        d_data.data(),
                        (const char*)(cpuReadStorage->getQualityArray()) + (i) * cpuReadStorage->getQualityPitch(),
                        cpuReadStorage->getQualityPitch() * currentBatchsize,
                        H2D,
                        stream
                    ); CUERR;

                    qualitiesGpu.scatter(
                        arrayhandle, 
                        d_data.data(), 
                        cpuReadStorage->getQualityPitch(), 
                        h_indices.data(), 
                        currentBatchsize, 
                        stream
                    );

                    cudaStreamSynchronize(stream); CUERR;
                }
            }

            //std::cerr << "getNumberOfReads(): " << getNumberOfReads() << ", qualitiesGpu.getNumRows(): " << qualitiesGpu.getNumRows() << "\n";
        }

        numHostSequences = numReads - sequencesGpu.getNumRows();
        numHostQualities = canUseQualityScores() ? numReads - qualitiesGpu.getNumRows() : 0;
        hostSequencePitch = cpuReadStorage->getSequencePitch();
        hostQualityPitch = cpuReadStorage->getQualityPitch();

        std::size_t memoryOfHostSequences = numHostSequences * cpuReadStorage->getSequencePitch();
        std::size_t memoryOfHostQualities = numHostQualities * cpuReadStorage->getQualityPitch();

        if(hasHostSequences() || hasHostQualities()){
            if(memoryLimitHost >= memoryOfHostSequences + memoryOfHostQualities){
                const std::size_t seqpitchints = cpuReadStorage->getSequencePitch() / sizeof(unsigned int);

                hostsequences.resize(numHostSequences * seqpitchints);

                std::copy(
                    cpuReadStorage->getSequenceArray() + seqpitchints * sequencesGpu.getNumRows(),
                    cpuReadStorage->getSequenceArray() + seqpitchints * numReads,
                    hostsequences.begin()
                );

                if(canUseQualityScores()){

                    hostqualities.resize(numHostQualities * cpuReadStorage->getQualityPitch());

                    std::copy(
                        cpuReadStorage->getQualityArray() + cpuReadStorage->getQualityPitch() * qualitiesGpu.getNumRows(),
                        cpuReadStorage->getQualityArray() + cpuReadStorage->getQualityPitch() * numReads,
                        hostqualities.begin()
                    );
                
                }

                cpuReadStorage = nullptr;
                //std::cerr << "GpuReadstorage is standalone\n";
            }else{
                //std::cerr << "GpuReadstorage cannot be standalone. MemoryLimit: " << memoryLimitHost << ", required: " <<  (memoryOfHostSequences + memoryOfHostQualities) << "\n";
            }
        }else{
            cpuReadStorage = nullptr;
            //std::cerr << "GpuReadstorage is standalone\n";
        }
    }

    bool isStandalone() const noexcept{
        return cpuReadStorage == nullptr;
    }

public: //inherited GPUReadStorage interface

    Handle makeHandle() const override {
        auto data = std::make_unique<TempData>();
        data->handleSequences = sequencesGpu.makeHandle();
        data->handleQualities = qualitiesGpu.makeHandle();
        data->event = CudaEvent{cudaEventDisableTiming};

        std::unique_lock<SharedMutex> lock(sharedmutex);
        const int handleid = counter++;
        Handle h = constructHandle(handleid);

        tempdataVector.emplace_back(std::move(data));
        return h;
    }

    void destroyHandle(Handle& handle) const override{

        std::unique_lock<SharedMutex> lock(sharedmutex);

        const int id = handle.getId();
        assert(id < int(tempdataVector.size()));
        
        tempdataVector[id] = nullptr;
        handle = constructHandle(std::numeric_limits<int>::max());
    }

    void areSequencesAmbiguous(
        Handle& handle,
        bool* d_result, 
        const read_number* d_readIds, 
        int numSequences, 
        cudaStream_t stream
    ) const override{

        if(numSequences > 0 && getNumberOfReadsWithN() > 0){

            int deviceId = 0;
            cudaGetDevice(&deviceId); CUERR;

            dim3 block = 256;
            dim3 grid = SDIV(numSequences, block.x);

            readBitarray<<<grid, block, 0, stream>>>(
                d_result, 
                bitArraysUndeterminedBase.at(deviceId), 
                d_readIds, 
                numSequences
            ); CUERR;
        }
    }

    void gatherSequences(
        Handle& handle,
        unsigned int* d_sequence_data,
        size_t outSequencePitchInInts,
        const read_number* h_readIds,
        const read_number* d_readIds,
        int numSequences,
        cudaStream_t stream
    ) const override{
        nvtx::push_range("multigpureadstorage::gatherSequences", 4);

        int deviceId = 0;
        cudaGetDevice(&deviceId); CUERR;

        TempData* tempData = getTempDataFromHandle(handle);
        assert(tempData->deviceId == deviceId);

        
        bool hasSynchronized = false;
        auto resizeWithSync = [&](auto& data, std::size_t size){
            using W = decltype(*data.get());

            const std::size_t currentCapacity = data.capacityInBytes();
            const std::size_t newbytes = size * sizeof(W);
            if(!hasSynchronized && currentCapacity < newbytes){
                tempData->event.synchronize();
                hasSynchronized = true;
                //std::cerr << "SYNC" << "\n";
            }
            data.resize(size);
        };

        auto gpuGather = [&](){

            nvtx::push_range("sequencesGpu.gather", 5);

            sequencesGpu.gather(
                tempData->handleSequences,
                d_sequence_data,
                sizeof(unsigned int) * outSequencePitchInInts,
                d_readIds,
                numSequences,
                hasHostSequences(),
                stream
            );

            nvtx::pop_range();
        };

        auto hostGather = [&](){
            const std::size_t sequencepitch = sizeof(unsigned int) * outSequencePitchInInts;

            constexpr std::size_t memorylimitbatch = 1 << 19; // 512KB

            if(hasGpuSequences()){

                /*
                    - Find subset of h_readIds which are point to data on the host
                    - Gather data on the host
                    - Scatter into device result array
                    
                    Is batched such that the required temporary storage per batch is small
                */

                const int batchsize = std::min(SDIV(memorylimitbatch, (sequencepitch + sizeof(int))), std::size_t(numSequences));

                void* temp_allocations_host[4]{};
                void* temp_allocations_device[4]{};
                std::size_t temp_allocation_sizes[4]{};

                temp_allocation_sizes[0] = sizeof(char) * sequencepitch * batchsize; // gathered host data 1
                temp_allocation_sizes[1] = sizeof(char) * sequencepitch * batchsize; // gathered host data 2
                temp_allocation_sizes[2] = sizeof(int) * batchsize; // output positions 1
                temp_allocation_sizes[3] = sizeof(int) * batchsize; // output positions 2

                std::size_t temp_storage_bytes = 0;
                cudaError_t cubstatus = cub::AliasTemporaries(
                    nullptr,
                    temp_storage_bytes,
                    temp_allocations_device,
                    temp_allocation_sizes
                );
                assert(cubstatus == cudaSuccess);

                resizeWithSync(tempData->tempbuffer, temp_storage_bytes);
                resizeWithSync(tempData->pinnedBuffer, temp_storage_bytes);

                cubstatus = cub::AliasTemporaries(
                    tempData->tempbuffer.data(),
                    temp_storage_bytes,
                    temp_allocations_device,
                    temp_allocation_sizes
                );
                assert(cubstatus == cudaSuccess);

                cubstatus = cub::AliasTemporaries(
                    tempData->pinnedBuffer.data(),
                    temp_storage_bytes,
                    temp_allocations_host,
                    temp_allocation_sizes
                );
                assert(cubstatus == cudaSuccess);

                std::array<std::vector<read_number>, 2> hostindicesarray{};
                hostindicesarray[0].resize(numSequences);
                hostindicesarray[1].resize(numSequences);

                std::array<unsigned int*, 2> h_hostdataArr{(unsigned int*)temp_allocations_host[0], (unsigned int*)temp_allocations_host[1]};
                std::array<unsigned int*, 2> d_hostdataArr{(unsigned int*)temp_allocations_device[0], (unsigned int*)temp_allocations_device[1]};
                std::array<int*, 2> h_outputpositionsArr{(int*)temp_allocations_host[2], (int*)temp_allocations_host[3]};
                std::array<int*, 2> d_outputpositionsArr{(int*)temp_allocations_device[2], (int*)temp_allocations_device[3]};

                for(int i = 0,k = 0, bufferIndex = 0; i < numSequences; i++){

                    if(isHostElementSequence(h_readIds[i])){
                        if(k == 0){
                            cudaStreamSynchronize(tempData->streams[bufferIndex]); CUERR; // protect pinned buffer
                        }
                        h_outputpositionsArr[bufferIndex][k] = i;
                        hostindicesarray[bufferIndex][k] = h_readIds[i];
                        k++;
                    }

                    if(k == batchsize || ((i == numSequences - 1) && k > 0)){
                        cudaMemcpyAsync(
                            d_outputpositionsArr[bufferIndex],
                            h_outputpositionsArr[bufferIndex],
                            sizeof(int) * k,
                            H2D,
                            tempData->streams[bufferIndex]
                        ); CUERR;

                        gatherHostSequences(
                            hostindicesarray[bufferIndex].data(),
                            k,
                            h_hostdataArr[bufferIndex],
                            outSequencePitchInInts
                        );

                        cudaMemcpyAsync(
                            d_hostdataArr[bufferIndex],
                            h_hostdataArr[bufferIndex],
                            sequencepitch * k,
                            H2D,
                            tempData->streams[bufferIndex]
                        ); CUERR;

                        const int intsToProcess = k * outSequencePitchInInts;

                        helpers::lambda_kernel<<<SDIV(intsToProcess, 128), 128, 0, tempData->streams[bufferIndex]>>>(
                            [
                                =, 
                                d_outputpositions = d_outputpositionsArr[bufferIndex],
                                d_gatheredhost = d_hostdataArr[bufferIndex]
                            ] __device__ (){
                                const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                                const int stride = blockDim.x * gridDim.x;

                                for(int i = tid; i < intsToProcess; i += stride){
                                    const int inputrow = i / outSequencePitchInInts;
                                    const int inputcol = i % outSequencePitchInInts;
                                    const int outputcol = inputcol;
                                    const int outputrow = d_outputpositions[inputrow];

                                    d_sequence_data[outputrow * outSequencePitchInInts + outputcol] = d_gatheredhost[inputrow * outSequencePitchInInts + inputcol];
                                }
                            }
                        ); CUERR;

                        bufferIndex = (bufferIndex + 1) % 2;
                        k = 0;
                    }
                }

                cudaEventRecord(tempData->event, tempData->streams[0]); CUERR;
                cudaStreamWaitEvent(stream, tempData->event, 0); CUERR;
                cudaEventRecord(tempData->event, tempData->streams[1]); CUERR;
                cudaStreamWaitEvent(stream, tempData->event, 0); CUERR;

                
            }else{
                const int batchsize = std::min(SDIV(memorylimitbatch, (sequencepitch)), std::size_t(numSequences));

                resizeWithSync(tempData->pinnedBuffer, 2 * batchsize * sequencepitch);

                std::array<unsigned int*, 2> hostpointers{ 
                    (unsigned int*)(tempData->pinnedBuffer.data()), 
                    (unsigned int*)(tempData->pinnedBuffer.data() + batchsize * sequencepitch)
                };
                assert(hostpointers.size() == tempData->streams.size());

                const int numBatches = SDIV(numSequences, batchsize);
                for(int b = 0; b < numBatches; b++){
                    const int bufferIndex = b % 2;

                    const int begin = b * batchsize;
                    const int end = std::min(numSequences, (b+1) * batchsize);
                    const int sizeOfCurrentBatch = end - begin;

                    cudaStreamSynchronize(tempData->streams[bufferIndex]); CUERR; // protect pinned buffer

                    gatherHostSequences(
                        h_readIds + begin,
                        sizeOfCurrentBatch,
                        hostpointers[bufferIndex],
                        outSequencePitchInInts
                    );

                    cudaError_t status = cudaMemcpyAsync(
                        (char*)(d_sequence_data) + sequencepitch * begin,
                        hostpointers[bufferIndex],
                        sequencepitch * sizeOfCurrentBatch,
                        H2D,
                        tempData->streams[bufferIndex]
                    );

                    if(status != cudaSuccess){
                        std::cerr << "gatherSequences("
                            << " d_sequence_data = " << d_sequence_data 
                            << ", outSequencePitchInInts = " << outSequencePitchInInts
                            << ", numSequences\n";

                        std::cerr << "batchsize = " << batchsize << "\n";
                        std::cerr << "bufferIndex = " << bufferIndex << "\n";
                        std::cerr << "sizeOfCurrentBatch = " << sizeOfCurrentBatch << "\n";
                        std::cerr << "hostpointers[bufferIndex] = " << hostpointers[bufferIndex] << "\n";

                        CUERR;
                    }
                }

                cudaEventRecord(tempData->event, tempData->streams[0]); CUERR;
                cudaStreamWaitEvent(stream, tempData->event, 0); CUERR;
                cudaEventRecord(tempData->event, tempData->streams[1]); CUERR;
                cudaStreamWaitEvent(stream, tempData->event, 0); CUERR;

            }
        };

        if(hasGpuSequences()){
            if(hasHostSequences()){

                hostGather();
                cudaStreamSynchronize(stream);
                gpuGather();
            }else{
                cudaStreamSynchronize(stream);
                gpuGather();
            }
        }else{
            hostGather();
        }

        cudaEventRecord(tempData->event, stream); CUERR;

        nvtx::pop_range();
    }

    void gatherQualities(
        Handle& handle,
        char* d_quality_data,
        size_t out_quality_pitch,
        const read_number* h_readIds,
        const read_number* d_readIds,
        int numSequences,
        cudaStream_t stream
    ) const override{
        nvtx::push_range("multigpureadstorage::gatherQualities", 4);

        int deviceId = 0;
        cudaGetDevice(&deviceId); CUERR;

        TempData* tempData = getTempDataFromHandle(handle);
        assert(tempData->deviceId == deviceId);        

        bool hasSynchronized = false;
        auto resizeWithSync = [&](auto& data, std::size_t size){
            using W = decltype(*data.get());

            const std::size_t currentCapacity = data.capacityInBytes();
            const std::size_t newbytes = size * sizeof(W);
            if(!hasSynchronized && currentCapacity < newbytes){
                tempData->event.synchronize();
                hasSynchronized = true;
                //std::cerr << "SYNC" << "\n";
            }
            data.resize(size);
        };

        auto gpuGather = [&](){
            nvtx::push_range("qualitiesGpu.gather", 5);

            qualitiesGpu.gather(
                tempData->handleQualities,
                d_quality_data,
                out_quality_pitch,
                d_readIds,
                numSequences,
                hasHostQualities(),
                stream
            );

            nvtx::pop_range();
        };

        auto hostGather = [&](){

            constexpr std::size_t memorylimitbatch = 1 << 19; // 512KB

            if(hasGpuQualities()){

                /*
                    - Find subset of h_readIds which are point to data on the host
                    - Gather data on the host
                    - Scatter into device result array
                    
                    Is batched such that the required temporary storage per batch is small
                */

                const int batchsize = std::min(SDIV(memorylimitbatch, (out_quality_pitch + sizeof(int))), std::size_t(numSequences));

                void* temp_allocations_host[4]{};
                void* temp_allocations_device[4]{};
                std::size_t temp_allocation_sizes[4]{};

                temp_allocation_sizes[0] = sizeof(char) * out_quality_pitch * batchsize; // gathered host data 1
                temp_allocation_sizes[1] = sizeof(char) * out_quality_pitch * batchsize; // gathered host data 2
                temp_allocation_sizes[2] = sizeof(int) * batchsize; // output positions 1
                temp_allocation_sizes[3] = sizeof(int) * batchsize; // output positions 2

                std::size_t temp_storage_bytes = 0;
                cudaError_t cubstatus = cub::AliasTemporaries(
                    nullptr,
                    temp_storage_bytes,
                    temp_allocations_device,
                    temp_allocation_sizes
                );
                assert(cubstatus == cudaSuccess);

                resizeWithSync(tempData->tempbuffer, temp_storage_bytes);
                resizeWithSync(tempData->pinnedBuffer, temp_storage_bytes);

                cubstatus = cub::AliasTemporaries(
                    tempData->tempbuffer.data(),
                    temp_storage_bytes,
                    temp_allocations_device,
                    temp_allocation_sizes
                );
                assert(cubstatus == cudaSuccess);

                cubstatus = cub::AliasTemporaries(
                    tempData->pinnedBuffer.data(),
                    temp_storage_bytes,
                    temp_allocations_host,
                    temp_allocation_sizes
                );
                assert(cubstatus == cudaSuccess);

                std::array<std::vector<read_number>, 2> hostindicesarray{};
                hostindicesarray[0].resize(numSequences);
                hostindicesarray[1].resize(numSequences);

                std::array<char*, 2> h_hostdataArr{(char*)temp_allocations_host[0], (char*)temp_allocations_host[1]};
                std::array<char*, 2> d_hostdataArr{(char*)temp_allocations_device[0], (char*)temp_allocations_device[1]};
                std::array<int*, 2> h_outputpositionsArr{(int*)temp_allocations_host[2], (int*)temp_allocations_host[3]};
                std::array<int*, 2> d_outputpositionsArr{(int*)temp_allocations_device[2], (int*)temp_allocations_device[3]};

                for(int i = 0,k = 0, bufferIndex = 0; i < numSequences; i++){

                    if(isHostElementQualityScore(h_readIds[i])){
                        if(k == 0){
                            cudaStreamSynchronize(tempData->streams[bufferIndex]); CUERR; // protect pinned buffer
                        }
                        h_outputpositionsArr[bufferIndex][k] = i;
                        hostindicesarray[bufferIndex][k] = h_readIds[i];
                        k++;
                    }

                    if(k == batchsize || ((i == numSequences - 1) && k > 0)){
                        cudaMemcpyAsync(
                            d_outputpositionsArr[bufferIndex],
                            h_outputpositionsArr[bufferIndex],
                            sizeof(int) * k,
                            H2D,
                            tempData->streams[bufferIndex]
                        ); CUERR;

                        gatherHostQualities(
                            hostindicesarray[bufferIndex].data(),
                            k,
                            h_hostdataArr[bufferIndex],
                            out_quality_pitch
                        );

                        cudaMemcpyAsync(
                            d_hostdataArr[bufferIndex],
                            h_hostdataArr[bufferIndex],
                            sizeof(char) * out_quality_pitch * k,
                            H2D,
                            tempData->streams[bufferIndex]
                        ); CUERR;

                        const int charsToProcess = k * out_quality_pitch;

                        helpers::lambda_kernel<<<SDIV(charsToProcess, 128), 128, 0, tempData->streams[bufferIndex]>>>(
                            [
                                =, 
                                d_outputpositions = d_outputpositionsArr[bufferIndex],
                                d_gatheredhost = d_hostdataArr[bufferIndex]
                            ] __device__ (){
                                const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                                const int stride = blockDim.x * gridDim.x;

                                for(int i = tid; i < charsToProcess; i += stride){
                                    const int inputrow = i / out_quality_pitch;
                                    const int inputcol = i % out_quality_pitch;
                                    const int outputcol = inputcol;
                                    const int outputrow = d_outputpositions[inputrow];

                                    d_quality_data[outputrow * out_quality_pitch + outputcol] = d_gatheredhost[inputrow * out_quality_pitch + inputcol];
                                }
                            }
                        ); CUERR;

                        bufferIndex = (bufferIndex + 1) % 2;
                        k = 0;
                    }
                }

                cudaEventRecord(tempData->event, tempData->streams[0]); CUERR;
                cudaStreamWaitEvent(stream, tempData->event, 0); CUERR;
                cudaEventRecord(tempData->event, tempData->streams[1]); CUERR;
                cudaStreamWaitEvent(stream, tempData->event, 0); CUERR;

                
            }else{
                const int batchsize = std::min(SDIV(memorylimitbatch, (out_quality_pitch)), std::size_t(numSequences));

                resizeWithSync(tempData->pinnedBuffer, 2 * batchsize * out_quality_pitch);

                std::array<char*, 2> hostpointers{
                    tempData->pinnedBuffer.data(), 
                    tempData->pinnedBuffer.data() + batchsize * out_quality_pitch
                };
                assert(hostpointers.size() == tempData->streams.size());

                const int numBatches = SDIV(numSequences, batchsize);
                for(int b = 0; b < numBatches; b++){
                    const int bufferIndex = b % 2;

                    const int begin = b * batchsize;
                    const int end = std::min(numSequences, (b+1) * batchsize);
                    const int sizeOfCurrentBatch = end - begin;

                    cudaStreamSynchronize(tempData->streams[bufferIndex]); CUERR; // protect pinned buffer

                    gatherHostQualities(
                        h_readIds + begin,
                        sizeOfCurrentBatch,
                        hostpointers[bufferIndex],
                        out_quality_pitch
                    );

                    cudaMemcpyAsync(
                        d_quality_data + out_quality_pitch * begin,
                        hostpointers[bufferIndex],
                        sizeof(char) * out_quality_pitch * sizeOfCurrentBatch,
                        H2D,
                        tempData->streams[bufferIndex]
                    ); CUERR;
                }

                cudaEventRecord(tempData->event, tempData->streams[0]); CUERR;
                cudaStreamWaitEvent(stream, tempData->event, 0); CUERR;
                cudaEventRecord(tempData->event, tempData->streams[1]); CUERR;
                cudaStreamWaitEvent(stream, tempData->event, 0); CUERR;
            }

        };

        

        if(hasGpuQualities()){
            if(hasHostQualities()){
                hostGather();
                gpuGather();
            }else{
                gpuGather();
            }
        }else{
            hostGather();
        }

        cudaEventRecord(tempData->event, stream); CUERR;

        nvtx::pop_range();
    }

    void gatherSequenceLengths(
        Handle& handle,
        int* d_lengths,
        const read_number* d_readIds,
        int numSequences,    
        cudaStream_t stream
    ) const override{

        gpuLengthStorage.gatherLengthsOnDeviceAsync(
            d_lengths, 
            d_readIds, 
            numSequences, 
            stream
        );

    }

    std::int64_t getNumberOfReadsWithN() const override{
        return numberOfAmbiguousReads;
    }

    MemoryUsage getMemoryInfo() const override{
        MemoryUsage result;

        if(!isStandalone()){
            result += cpuReadStorage->getMemoryInfo();
        }

        result.host += sizeof(unsigned int) * hostsequences.capacity();
        result.host += sizeof(char) * hostqualities.capacity();

        result += sequencesGpu.getMemoryInfo();

        if(canUseQualityScores()){
            result += qualitiesGpu.getMemoryInfo();
        }

        result += gpuLengthStorage.getMemoryInfo();

        return result;
    }

    MemoryUsage getMemoryInfo(const Handle& handle) const override{
        int deviceId = 0;
        cudaGetDevice(&deviceId); CUERR;

        TempData* tempData = getTempDataFromHandle(handle);
 
        return tempData->getMemoryInfo();
    }

    read_number getNumberOfReads() const override{
        return numberOfReads;
    }

    bool canUseQualityScores() const override{
        return useQualityScores;
    }

    int getSequenceLengthLowerBound() const override{
        return sequenceLengthLowerBound;
    }

    int getSequenceLengthUpperBound() const override{
        return sequenceLengthUpperBound;
    }

    void destroy() override{
        destroyReadData();

        destroyTempData();
    }
    
private:
    template<class T, class IndexGenerator>
    void gatherHostStandaloneImpl(
        const T* source,
        std::size_t numColumns,
        size_t sourcePitchBytes,
        IndexGenerator readIds,
        int numReadIds,
        T* destination,
        size_t destinationPitchBytes
    ) const noexcept{
        
        if(numReadIds == 0){
            return;
        }

        constexpr int prefetch_distance = 4;

        for(int i = 0; i < numReadIds && i < prefetch_distance; ++i) {
            const int index = i;
            const read_number nextReadId = readIds(index);
            const T* const nextData = (const T*)(((const char*)source) + sourcePitchBytes * nextReadId);
            __builtin_prefetch(nextData, 0, 0);
        }

        for(int i = 0; i < numReadIds; i++){
            if(i + prefetch_distance < numReadIds) {
                const int index = i + prefetch_distance;
                const read_number nextReadId = readIds(index);
                const T* const nextData = (const T*)(((const char*)source) + sourcePitchBytes * nextReadId);
                __builtin_prefetch(nextData, 0, 0);
            }

            const int index = i;
            const read_number readId = readIds(index);
            const T* const data = (const T*)(((const char*)source) + sourcePitchBytes * readId);

            T* const destData = (T*)(((char*)destination) + destinationPitchBytes * index);
            std::copy_n(data, numColumns, destData);
        }
    }
    
    void gatherHostSequences(
        const read_number* readIds,
        int numSequences,
        unsigned int* outputarray,
        std::size_t outputPitchInInts
    ) const {
        if(!isStandalone()){
            cpu::ContiguousReadStorage::GatherHandle cpuhandle{};

            cpuReadStorage->gatherSequenceData(
                cpuhandle,
                readIds,
                numSequences,
                outputarray,
                outputPitchInInts
            );
        }else{
            //convert readIds into local indices for host partition
            auto indexGenerator = [&](auto i){
                return readIds[i] - sequencesGpu.getNumRows();
            };

            gatherHostStandaloneImpl(
                hostsequences.data(),
                hostSequencePitch / sizeof(unsigned int),
                hostSequencePitch,
                indexGenerator,
                numSequences,
                outputarray,
                outputPitchInInts * sizeof(unsigned int)
            );
        }
    }

    void gatherHostQualities(
        const read_number* readIds,
        int numSequences,
        char* outputarray,
        std::size_t outputPitchInBytes
    ) const {
        if(!isStandalone()){
            cpu::ContiguousReadStorage::GatherHandle cpuhandle{};

            cpuReadStorage->gatherSequenceQualities(
                cpuhandle,
                readIds,
                numSequences,
                outputarray,
                outputPitchInBytes
            );
        }else{

            //convert readIds into local indices for host partition
            auto indexGenerator = [&](auto i){
                return readIds[i] - qualitiesGpu.getNumRows();
            };

            gatherHostStandaloneImpl(
                hostqualities.data(),
                hostQualityPitch,
                hostQualityPitch,
                indexGenerator,
                numSequences,
                outputarray,
                outputPitchInBytes
            );
        }
    }

    TempData* getTempDataFromHandle(const Handle& handle) const{
        std::shared_lock<SharedMutex> lock(sharedmutex);

        assert(handle.getId() < int(tempdataVector.size()));

        return tempdataVector[handle.getId()].get();
    }

    bool hasHostSequences() const noexcept{
        return numHostSequences > 0;
    }

    bool hasHostQualities() const noexcept{
        return canUseQualityScores() && numHostQualities > 0;
    }

    bool hasGpuSequences() const noexcept{
        return sequencesGpu.getNumRows() > 0;
    }

    bool hasGpuQualities() const noexcept{
        return canUseQualityScores() && qualitiesGpu.getNumRows() > 0;
    }

    bool isHostElementSequence(std::size_t index) const noexcept{
        return sequencesGpu.getNumRows() <= index;
    }

    bool isHostElementQualityScore(std::size_t index) const noexcept{
        return qualitiesGpu.getNumRows() <= index;
    }

    void destroyReadData(){
        if(numberOfReads > 0){
            sequencesGpu.destroy();

            if(canUseQualityScores()){
                qualitiesGpu.destroy();
            }

            gpuLengthStorage.destroyGpuData();

            for(auto& pair : bitArraysUndeterminedBase){
                cub::SwitchDevice sd(pair.first); CUERR;
                destroyGpuBitArray(pair.second);
            }
            bitArraysUndeterminedBase.clear();

            auto deallocVector = [](auto& vec){
                using T = typename std::remove_reference<decltype(vec)>::type;
                T tmp{};
                vec.swap(tmp);
            };

            deallocVector(hostsequences);
            deallocVector(hostqualities);
        }

        useQualityScores = false;
        sequenceLengthLowerBound = 0;
        sequenceLengthUpperBound = 0;
        numberOfReads = 0;
        numberOfAmbiguousReads = 0;
        numHostSequences = 0;
        numHostQualities = 0;
    }

    void destroyTempData(){
        auto deallocVector = [](auto& vec){
            using T = typename std::remove_reference<decltype(vec)>::type;
            T tmp{};
            vec.swap(tmp);
        };

        deallocVector(tempdataVector);
    }
    
    bool useQualityScores{};
    int sequenceLengthLowerBound{};
    int sequenceLengthUpperBound{};
    read_number numberOfReads{};
    std::int64_t numberOfAmbiguousReads{};

    std::size_t numHostSequences{};
    std::size_t numHostQualities{};
    std::size_t hostSequencePitch{};
    std::size_t hostQualityPitch{};
    const cpu::ContiguousReadStorage* cpuReadStorage{};

    MultiGpu2dArray<unsigned int, IndexType> sequencesGpu{};
    MultiGpu2dArray<char, IndexType> qualitiesGpu{};
    std::map<int, GpuBitArray<read_number>> bitArraysUndeterminedBase;


    GPULengthStore2<std::uint32_t> gpuLengthStorage{};
    std::vector<unsigned int> hostsequences{};
    std::vector<char> hostqualities{};

    std::vector<int> deviceIds{};

    mutable int counter = 0;
    mutable SharedMutex sharedmutex{};
    mutable std::vector<std::unique_ptr<TempData>> tempdataVector{};
};
    
}
}






#endif