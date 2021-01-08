#ifndef CARE_MULTIGPUREADSTORAGE_CUH
#define CARE_MULTIGPUREADSTORAGE_CUH

#include <readstorage.hpp>
#include <gpu/gpureadstorage.cuh>

#include <2darray.hpp>
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
            auto info = getMemoryInfo();
            std::cerr << "MultiGpuReadStorage::TempData: host: " << info.host 
                << ", device[" << deviceId << "]: " << info.device[deviceId] << "\n";
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
        std::vector<std::size_t> memoryLimitsPerDevice_
    ){

        rebuild(
            cpuReadStorage_,
            deviceIds_,
            memoryLimitsPerDevice_
        );

    }

    void rebuild(
        const cpu::ContiguousReadStorage& cpuReadStorage_, 
        std::vector<int> deviceIds_, 
        std::vector<std::size_t> memoryLimitsPerDevice
    ){
        destroyGpuReadData();

        cpuReadStorage = &cpuReadStorage_;
        deviceIds = std::move(deviceIds_);

        const int numGpus = deviceIds.size();
        const std::size_t numReads = cpuReadStorage->getNumberOfReads();

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

        std::cerr << "getNumberOfReads(): " << getNumberOfReads() << ", sequencesGpu.getNumRows(): " << sequencesGpu.getNumRows() << "\n";

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

            // auto gatherhandle = makeHandle();

            // helpers::SimpleAllocationPinnedHost<unsigned int> mysequencedata(batchsize * cpuReadStorage->getSequencePitch() / sizeof(unsigned int));
            // helpers::SimpleAllocationPinnedHost<unsigned int> cpusequencedata(batchsize * cpuReadStorage->getSequencePitch() / sizeof(unsigned int));

            // batchsize = 1000;

            // for(std::size_t i = 0; i < numReads; i += batchsize){
            //     const std::size_t currentBatchsize = std::min(batchsize, numReads - i);
            //     std::iota(h_indices.begin(), h_indices.begin() + currentBatchsize, i);

            //     gatherSequences(
            //         gatherhandle,
            //         mysequencedata.data(),
            //         cpuReadStorage->getSequencePitch() / sizeof(unsigned int),
            //         h_indices.data(),
            //         h_indices.data(),
            //         batchsize,
            //         stream
            //     );

            //     cudaStreamSynchronize(stream); CUERR;

            //     cpu::ContiguousReadStorage::GatherHandle cpuhandle{};
            //     cpuReadStorage->gatherSequenceData(
            //         cpuhandle,
            //         h_indices.data(),
            //         batchsize,
            //         cpusequencedata.data(),
            //         cpuReadStorage->getSequencePitch() / sizeof(unsigned int)
            //     );

            //     for(std::size_t k = 0; k < cpuReadStorage->getSequencePitch() / sizeof(unsigned int) * batchsize; k++){
            //         if(cpusequencedata[k] != mysequencedata[k]){
            //             std::cerr << "error i = " << i << ", k = " << k << ", pitch = " << cpuReadStorage->getSequencePitch() << "\n";
            //             assert(false);
            //         }
            //     }
            // }

            

            // destroyHandle(gatherhandle);
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

            std::cerr << "getNumberOfReads(): " << getNumberOfReads() << ", qualitiesGpu.getNumRows(): " << qualitiesGpu.getNumRows() << "\n";
        }

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

        tempData->event.synchronize();

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
            care::cpu::ContiguousReadStorage::GatherHandle cpuhandle{};

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

                tempData->tempbuffer.resize(temp_storage_bytes);
                tempData->pinnedBuffer.resize(temp_storage_bytes);

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

                        cpuReadStorage->gatherSequenceData(
                            cpuhandle,
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

                tempData->pinnedBuffer.resize(2 * batchsize * sequencepitch);

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

                    cpuReadStorage->gatherSequenceData(
                        cpuhandle,
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

        tempData->event.synchronize();

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
            care::cpu::ContiguousReadStorage::GatherHandle cpuhandle{};

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

                tempData->tempbuffer.resize(temp_storage_bytes);
                tempData->pinnedBuffer.resize(temp_storage_bytes);

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

                        cpuReadStorage->gatherSequenceQualities(
                            cpuhandle,
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

                tempData->pinnedBuffer.resize(2 * batchsize * out_quality_pitch);

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

                    cpuReadStorage->gatherSequenceQualities(
                        cpuhandle,
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
        return cpuReadStorage->getNumberOfReadsWithN();
    }

    MemoryUsage getMemoryInfo() const override{
        MemoryUsage result;

        result += cpuReadStorage->getMemoryInfo();

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
        return cpuReadStorage->getNumberOfReads();
    }

    bool canUseQualityScores() const override{
        return cpuReadStorage->canUseQualityScores();
    }

    int getSequenceLengthLowerBound() const override{
        return cpuReadStorage->getSequenceLengthLowerBound();
    }

    int getSequenceLengthUpperBound() const override{
        return cpuReadStorage->getSequenceLengthUpperBound();
    }

    void destroy() override{
        destroyGpuReadData();

        destroyTempData();
    }


private:
    TempData* getTempDataFromHandle(const Handle& handle) const{
        std::shared_lock<SharedMutex> lock(sharedmutex);

        assert(handle.getId() < int(tempdataVector.size()));

        return tempdataVector[handle.getId()].get();
    }

    bool hasHostSequences() const noexcept{
        return getNumberOfReads() > sequencesGpu.getNumRows();
    }

    bool hasHostQualities() const noexcept{
        return canUseQualityScores() && getNumberOfReads() > qualitiesGpu.getNumRows();
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

    void destroyGpuReadData(){
        if(cpuReadStorage != nullptr){
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
        }
    }

    void destroyTempData(){
        auto deallocVector = [](auto& vec){
            using T = typename std::remove_reference<decltype(vec)>::type;
            T tmp{};
            vec.swap(tmp);
        };

        deallocVector(tempdataVector);
    }
    
    const cpu::ContiguousReadStorage* cpuReadStorage{};

    MultiGpu2dArray<unsigned int, IndexType> sequencesGpu{};
    MultiGpu2dArray<char, IndexType> qualitiesGpu{};
    std::map<int, GpuBitArray<read_number>> bitArraysUndeterminedBase;


    GPULengthStore2<std::uint32_t> gpuLengthStorage{};

    std::vector<int> deviceIds{};

    mutable int counter = 0;
    mutable SharedMutex sharedmutex{};
    mutable std::vector<std::unique_ptr<TempData>> tempdataVector{};
};
    
}
}






#endif