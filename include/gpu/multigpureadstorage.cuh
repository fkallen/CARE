#ifndef CARE_MULTIGPUREADSTORAGE_CUH
#define CARE_MULTIGPUREADSTORAGE_CUH

#include <cpureadstorage.hpp>
#include <gpu/gpureadstorage.cuh>
#include <gpu/cudaerrorcheck.cuh>
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
            CUDACHECK(cudaGetDevice(&deviceId));
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
        typename MultiGpu2dArray<unsigned int, IndexType>::Handle handleQualities{};
    };

    MultiGpuReadStorage(
        const CpuReadStorage& cpuReadStorage_, 
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
        const CpuReadStorage& cpuReadStorage_, 
        std::vector<int> deviceIds_, 
        std::vector<std::size_t> memoryLimitsPerDevice,
        std::size_t memoryLimitHost
    ){
        assert(deviceIds_.size() > 0);

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
        pairedEnd = cpuReadStorage->isPairedEnd();

        gpuLengthStorage = std::move(
            GPULengthStore3<std::uint32_t>(
                sequenceLengthLowerBound, 
                sequenceLengthUpperBound, 
                numberOfReads, 
                deviceIds
            )
        );

        {
            constexpr std::size_t batchsize = 1000000;
            const std::size_t numBatches = SDIV(numReads, batchsize);

            std::vector<int> lengths(batchsize);
            std::vector<read_number> readIds(batchsize);

            for(std::size_t i = 0; i < numBatches; i++){
                size_t begin = i * batchsize;
                size_t end = std::min((i+1) * batchsize, numReads);
                size_t elements = end - begin;

                std::iota(readIds.begin(), readIds.begin() + elements, begin);

                cpuReadStorage->gatherSequenceLengths(
                    lengths.data(),
                    readIds.data(),
                    elements
                );

                for(std::size_t k = 0; k < elements; k++){
                    gpuLengthStorage.setLength(readIds[k], lengths[k]);
                }
            }

            gpuLengthStorage.finalize();
        }

        

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
            cub::SwitchDevice sd(deviceId);

            bitArraysUndeterminedBase[deviceId] = makeGpuBitArray<read_number>(numReads);

            if(memoryLimitsPerDevice[d] >= bitArraysUndeterminedBase[deviceId].numAllocatedBytes){
                memoryLimitsPerDevice[d] -= bitArraysUndeterminedBase[deviceId].numAllocatedBytes;
            }else{
                memoryLimitsPerDevice[d] = 0;
            }


            const int numAmbiguous = cpuReadStorage->getNumberOfReadsWithN();

            if(numAmbiguous > 0){

                HostBuffer<read_number> h_positions(numAmbiguous);
                cpuReadStorage->getIdsOfAmbiguousReads(
                    h_positions.data()
                );

                constexpr int batchsize = 1000000;
                const int numBatches = SDIV(numAmbiguous, batchsize);

                DeviceBuffer<bool> d_values(batchsize);
                CUDACHECK(cudaMemset(d_values, 1, sizeof(bool) * batchsize));

                DeviceBuffer<read_number> d_positions(batchsize);

                for(int i = 0; i < numBatches; i++){
                    size_t begin = i * batchsize;
                    size_t end = std::min((i+1) * batchsize, numAmbiguous);
                    size_t elements = end - begin;

                    CUDACHECK(cudaMemcpy(
                        d_positions.data(), 
                        h_positions.data() + begin, 
                        sizeof(read_number) * elements, 
                        H2D
                    ));

                    setBitarray<<<SDIV(elements, 128), 128>>>(
                        bitArraysUndeterminedBase[deviceId], 
                        d_values.data(), 
                        d_positions.data(), 
                        elements
                    ); CUDACHECKASYNC;

                    CUDACHECK(cudaDeviceSynchronize());
                }

                ambigReadIds.clear();
                ambigReadIds.insert(ambigReadIds.end(), h_positions.begin(), h_positions.end());
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
            constexpr std::size_t batchsize = 65000;
            constexpr int numbuffers = 2;

            std::array<CudaStream, numbuffers> streams{};
            auto arrayhandle = sequencesGpu.makeHandle();

            std::array<helpers::SimpleAllocationPinnedHost<IndexType>, numbuffers> indexbuffers{};
            std::array<helpers::SimpleAllocationDevice<IndexType>, numbuffers> deviceindexbuffers{};
            std::array<helpers::SimpleAllocationPinnedHost<unsigned int>, numbuffers> hostdatabuffers{};
            std::array<helpers::SimpleAllocationDevice<unsigned int>, numbuffers> devicedatabuffers{};

            std::array<IndexType*, numbuffers> indexarray{};
            std::array<IndexType*, numbuffers> deviceindexarray{};
            std::array<unsigned int*, numbuffers> hostdataarray{};
            std::array<unsigned int*, numbuffers> dataarray{};

            for(int i = 0; i < numbuffers; i++){
                indexbuffers[i].resize(batchsize);
                deviceindexbuffers[i].resize(batchsize);
                hostdatabuffers[i].resize(batchsize * numColumnsSequences);
                devicedatabuffers[i].resize(batchsize * numColumnsSequences);

                indexarray[i] = indexbuffers[i].data();
                deviceindexarray[i] = deviceindexbuffers[i].data();
                hostdataarray[i] = hostdatabuffers[i].data();
                dataarray[i] = devicedatabuffers[i].data();
            }
            
            for(std::size_t i = 0, iteration = 0; i < sequencesGpu.getNumRows(); i += batchsize, iteration++){
                const int bufferIndex = iteration % numbuffers;

                CUDACHECK(cudaStreamSynchronize(streams[bufferIndex]));

                const std::size_t currentBatchsize = std::min(batchsize, sequencesGpu.getNumRows() - i);
                std::iota(indexarray[bufferIndex], indexarray[bufferIndex] + currentBatchsize, i);

                CUDACHECK(cudaMemcpyAsync(
                    deviceindexarray[bufferIndex],
                    indexarray[bufferIndex],
                    sizeof(IndexType) * currentBatchsize,
                    H2D,
                    streams[bufferIndex]
                ));

                cpuReadStorage->gatherSequences(
                    hostdataarray[bufferIndex],
                    numColumnsSequences,
                    indexarray[bufferIndex],
                    currentBatchsize
                );

                CUDACHECK(cudaMemcpyAsync(
                    dataarray[bufferIndex],
                    hostdataarray[bufferIndex],
                    numColumnsSequences * currentBatchsize * sizeof(unsigned int),
                    H2D,
                    streams[bufferIndex]
                ));

                sequencesGpu.scatter(
                    arrayhandle, 
                    dataarray[bufferIndex], 
                    numColumnsSequences * sizeof(unsigned int), 
                    deviceindexarray[bufferIndex], 
                    currentBatchsize, 
                    streams[bufferIndex]
                );

                
            }

            for(auto& stream : streams){
                CUDACHECK(cudaStreamSynchronize(stream));
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

        const int numColumnsQualitiesInts = SDIV(cpuReadStorage->getSequenceLengthUpperBound(), sizeof(unsigned int));

        if(canUseQualityScores()){

            qualitiesGpu = std::move(
                MultiGpu2dArray<unsigned int, IndexType>(
                    numReads,
                    numColumnsQualitiesInts,
                    sizeof(unsigned int),
                    deviceIds,
                    memoryLimitsPerDevice,
                    MultiGpu2dArrayLayout::FirstFit,
                    MultiGpu2dArrayInitMode::CanDiscardRows
                )
            );

            {
                constexpr std::size_t batchsize = 65000;
                constexpr int numbuffers = 2;

                std::array<CudaStream, numbuffers> streams{};
                auto arrayhandle = qualitiesGpu.makeHandle();

                std::array<helpers::SimpleAllocationPinnedHost<IndexType>, numbuffers> indexbuffers{};
                std::array<helpers::SimpleAllocationDevice<IndexType>, numbuffers> deviceindexbuffers{};
                std::array<helpers::SimpleAllocationPinnedHost<unsigned int>, numbuffers> hostdatabuffers{};
                std::array<helpers::SimpleAllocationDevice<unsigned int>, numbuffers> devicedatabuffers{};

                std::array<IndexType*, numbuffers> indexarray{};
                std::array<IndexType*, numbuffers> deviceindexarray{};
                std::array<unsigned int*, numbuffers> hostdataarray{};
                std::array<unsigned int*, numbuffers> dataarray{};

                for(int i = 0; i < numbuffers; i++){
                    indexbuffers[i].resize(batchsize);
                    deviceindexbuffers[i].resize(batchsize);
                    hostdatabuffers[i].resize(batchsize * numColumnsQualitiesInts);
                    devicedatabuffers[i].resize(batchsize * numColumnsQualitiesInts);

                    indexarray[i] = indexbuffers[i].data();
                    deviceindexarray[i] = deviceindexbuffers[i].data();
                    hostdataarray[i] = hostdatabuffers[i].data();
                    dataarray[i] = devicedatabuffers[i].data();
                }

                for(std::size_t i = 0, iteration = 0; i < qualitiesGpu.getNumRows(); i += batchsize, iteration++){
                    const int bufferIndex = iteration % numbuffers;
                    CUDACHECK(cudaStreamSynchronize(streams[bufferIndex]));

                    const std::size_t currentBatchsize = std::min(batchsize, qualitiesGpu.getNumRows() - i);
                    std::iota(indexarray[bufferIndex], indexarray[bufferIndex] + currentBatchsize, i);

                    CUDACHECK(cudaMemcpyAsync(
                        deviceindexarray[bufferIndex],
                        indexarray[bufferIndex],
                        sizeof(IndexType) * currentBatchsize,
                        H2D,
                        streams[bufferIndex]
                    ));

                    cpuReadStorage->gatherQualities(
                        (char*)hostdataarray[bufferIndex],
                        numColumnsQualitiesInts * sizeof(unsigned int),
                        indexarray[bufferIndex],
                        currentBatchsize
                    );

                    CUDACHECK(cudaMemcpyAsync(
                        dataarray[bufferIndex],
                        hostdataarray[bufferIndex],
                        numColumnsQualitiesInts * currentBatchsize * sizeof(unsigned int),
                        H2D,
                        streams[bufferIndex]
                    ));

                    qualitiesGpu.scatter(
                        arrayhandle, 
                        dataarray[bufferIndex], 
                        numColumnsQualitiesInts * sizeof(unsigned int), 
                        deviceindexarray[bufferIndex], 
                        currentBatchsize, 
                        streams[bufferIndex]
                    );
                }

                for(auto& stream : streams){
                    CUDACHECK(cudaStreamSynchronize(stream));
                }
            }

            //std::cerr << "getNumberOfReads(): " << getNumberOfReads() << ", qualitiesGpu.getNumRows(): " << qualitiesGpu.getNumRows() << "\n";
        }

        

        numHostSequences = numReads - sequencesGpu.getNumRows();
        numHostQualities = canUseQualityScores() ? numReads - qualitiesGpu.getNumRows() : 0;
        hostSequencePitch = numColumnsSequences * sizeof(unsigned int);
        hostQualityPitch = numColumnsQualitiesInts * sizeof(unsigned int);

        std::size_t memoryOfHostSequences = numHostSequences * hostSequencePitch;
        std::size_t memoryOfHostQualities = numHostQualities * hostQualityPitch;

        if(hasHostSequences() || hasHostQualities()){
            if(memoryLimitHost >= memoryOfHostSequences + memoryOfHostQualities){
                const std::size_t seqpitchints = numColumnsSequences;

                hostsequences.resize(numHostSequences * seqpitchints);

                const std::size_t numSequencesToCopy = numReads - sequencesGpu.getNumRows();
                const std::size_t batchsize = 100000;
                for(std::size_t i = 0; i < numSequencesToCopy; i += batchsize){
                    const std::size_t currentBatchsize = std::min(batchsize, numSequencesToCopy - i);

                    std::vector<read_number> indices(currentBatchsize);
                    std::iota(indices.begin(), indices.end(), sequencesGpu.getNumRows() + i);

                    cpuReadStorage->gatherSequences(
                        hostsequences.data() + i * numColumnsSequences,
                        numColumnsSequences,
                        indices.data(),
                        currentBatchsize
                    );
                }

                

                if(canUseQualityScores()){

                    hostqualities.resize(numHostQualities * numColumnsQualitiesInts);

                    const std::size_t numQualitiesToCopy = numReads - qualitiesGpu.getNumRows();
                    const std::size_t batchsizeq = 100000;
                    for(std::size_t i = 0; i < numQualitiesToCopy; i += batchsizeq){
                        const std::size_t currentBatchsize = std::min(batchsizeq, numQualitiesToCopy - i);

                        std::vector<read_number> indices(currentBatchsize);
                        std::iota(indices.begin(), indices.end(), qualitiesGpu.getNumRows() + i);

                        cpuReadStorage->gatherQualities(
                            (char*)(hostqualities.data() + i * numColumnsQualitiesInts),
                            numColumnsQualitiesInts * sizeof(unsigned int),
                            indices.data(),
                            currentBatchsize
                        );
                    }
                
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

    bool trySequenceReplication(const std::vector<std::size_t>& memoryLimits){
        return sequencesGpu.tryReplication(memoryLimits);
    }

    bool tryQualityReplication(const std::vector<std::size_t>& memoryLimits){
        return qualitiesGpu.tryReplication(memoryLimits);
    }

public: //inherited GPUReadStorage interface

    ReadStorageHandle makeHandle() const override {
        auto data = std::make_unique<TempData>();
        data->handleSequences = sequencesGpu.makeHandle();
        data->handleQualities = qualitiesGpu.makeHandle();
        data->event = CudaEvent{cudaEventDisableTiming};

        std::unique_lock<SharedMutex> lock(sharedmutex);
        const int handleid = counter++;
        ReadStorageHandle h = constructHandle(handleid);

        tempdataVector.emplace_back(std::move(data));
        return h;
    }

    void destroyHandle(ReadStorageHandle& handle) const override{

        std::unique_lock<SharedMutex> lock(sharedmutex);

        const int id = handle.getId();
        assert(id < int(tempdataVector.size()));
        
        tempdataVector[id] = nullptr;
        handle = constructHandle(std::numeric_limits<int>::max());
    }

    void areSequencesAmbiguous(
        ReadStorageHandle& handle,
        bool* d_result, 
        const read_number* d_readIds, 
        int numSequences, 
        cudaStream_t stream
    ) const override{

        if(numSequences > 0){
            if(getNumberOfReadsWithN() > 0){

                int deviceId = 0;
                CUDACHECK(cudaGetDevice(&deviceId));

                dim3 block = 256;
                dim3 grid = SDIV(numSequences, block.x);

                readBitarray<<<grid, block, 0, stream>>>(
                    d_result, 
                    bitArraysUndeterminedBase.at(deviceId), 
                    d_readIds, 
                    numSequences
                ); CUDACHECKASYNC;
            }else{
                // if there are no stored reads with ambiguous bases, simply fill output with false
                helpers::call_fill_kernel_async(d_result, numSequences, false, stream); CUDACHECKASYNC;
            }
        }else{
            //output buffer is empty
        }
    }

    void gatherSequences(
        ReadStorageHandle& handle,
        unsigned int* d_sequence_data,
        size_t outSequencePitchInInts,
        const AsyncConstBufferWrapper<read_number> h_readIdsAsync,
        const read_number* d_readIds,
        int numSequences,
        cudaStream_t stream
    ) const override{
        if(numSequences == 0) return;
        
        nvtx::push_range("multigpureadstorage::gatherSequences", 4);

        int deviceId = 0;
        CUDACHECK(cudaGetDevice(&deviceId));

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

            h_readIdsAsync.wait();
            const read_number* h_readIds = h_readIdsAsync.data();

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
                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    temp_storage_bytes,
                    temp_allocations_device,
                    temp_allocation_sizes
                ));

                resizeWithSync(tempData->tempbuffer, temp_storage_bytes);
                resizeWithSync(tempData->pinnedBuffer, temp_storage_bytes);

                CUDACHECK(cub::AliasTemporaries(
                    tempData->tempbuffer.data(),
                    temp_storage_bytes,
                    temp_allocations_device,
                    temp_allocation_sizes
                ));

                CUDACHECK(cub::AliasTemporaries(
                    tempData->pinnedBuffer.data(),
                    temp_storage_bytes,
                    temp_allocations_host,
                    temp_allocation_sizes
                ));

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
                            CUDACHECK(cudaStreamSynchronize(tempData->streams[bufferIndex])); // protect pinned buffer
                        }
                        h_outputpositionsArr[bufferIndex][k] = i;
                        hostindicesarray[bufferIndex][k] = h_readIds[i];
                        k++;
                    }

                    if(k == batchsize || ((i == numSequences - 1) && k > 0)){
                        CUDACHECK(cudaMemcpyAsync(
                            d_outputpositionsArr[bufferIndex],
                            h_outputpositionsArr[bufferIndex],
                            sizeof(int) * k,
                            H2D,
                            tempData->streams[bufferIndex]
                        ));

                        gatherHostSequences(
                            tempData,
                            hostindicesarray[bufferIndex].data(),
                            k,
                            h_hostdataArr[bufferIndex],
                            outSequencePitchInInts
                        );

                        CUDACHECK(cudaMemcpyAsync(
                            d_hostdataArr[bufferIndex],
                            h_hostdataArr[bufferIndex],
                            sequencepitch * k,
                            H2D,
                            tempData->streams[bufferIndex]
                        ));

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
                        ); CUDACHECKASYNC;

                        bufferIndex = (bufferIndex + 1) % 2;
                        k = 0;
                    }
                }

                CUDACHECK(cudaEventRecord(tempData->event, tempData->streams[0]));
                CUDACHECK(cudaStreamWaitEvent(stream, tempData->event, 0));
                CUDACHECK(cudaEventRecord(tempData->event, tempData->streams[1]));
                CUDACHECK(cudaStreamWaitEvent(stream, tempData->event, 0));

                
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

                    CUDACHECK(cudaStreamSynchronize(tempData->streams[bufferIndex])); // protect pinned buffer

                    gatherHostSequences(
                        tempData,
                        h_readIds + begin,
                        sizeOfCurrentBatch,
                        hostpointers[bufferIndex],
                        outSequencePitchInInts
                    );

                    CUDACHECK(cudaMemcpyAsync(
                        (char*)(d_sequence_data) + sequencepitch * begin,
                        hostpointers[bufferIndex],
                        sequencepitch * sizeOfCurrentBatch,
                        H2D,
                        tempData->streams[bufferIndex]
                    ));
                }

                CUDACHECK(cudaEventRecord(tempData->event, tempData->streams[0]));
                CUDACHECK(cudaStreamWaitEvent(stream, tempData->event, 0));
                CUDACHECK(cudaEventRecord(tempData->event, tempData->streams[1]));
                CUDACHECK(cudaStreamWaitEvent(stream, tempData->event, 0));

            }
        };

        if(hasGpuSequences()){
            if(hasHostSequences()){

                hostGather();

                gpuGather();
            }else{
                gpuGather();
            }
        }else{
            hostGather();
        }

        CUDACHECK(cudaEventRecord(tempData->event, stream));

        nvtx::pop_range();
    }

    void gatherQualities(
        ReadStorageHandle& handle,
        char* d_quality_data,
        size_t out_quality_pitch,
        const AsyncConstBufferWrapper<read_number> h_readIdsAsync,
        const read_number* d_readIds,
        int numSequences,
        cudaStream_t stream
    ) const override{
        if(numSequences == 0) return;
        
        nvtx::push_range("multigpureadstorage::gatherQualities", 4);

        int deviceId = 0;
        CUDACHECK(cudaGetDevice(&deviceId));

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
                (unsigned int*)d_quality_data,
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

            h_readIdsAsync.wait();
            const read_number* h_readIds = h_readIdsAsync.data();

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
                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    temp_storage_bytes,
                    temp_allocations_device,
                    temp_allocation_sizes
                ));

                resizeWithSync(tempData->tempbuffer, temp_storage_bytes);
                resizeWithSync(tempData->pinnedBuffer, temp_storage_bytes);

                CUDACHECK(cub::AliasTemporaries(
                    tempData->tempbuffer.data(),
                    temp_storage_bytes,
                    temp_allocations_device,
                    temp_allocation_sizes
                ));

                CUDACHECK(cub::AliasTemporaries(
                    tempData->pinnedBuffer.data(),
                    temp_storage_bytes,
                    temp_allocations_host,
                    temp_allocation_sizes
                ));

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
                            CUDACHECK(cudaStreamSynchronize(tempData->streams[bufferIndex])); // protect pinned buffer
                        }
                        h_outputpositionsArr[bufferIndex][k] = i;
                        hostindicesarray[bufferIndex][k] = h_readIds[i];
                        k++;
                    }

                    if(k == batchsize || ((i == numSequences - 1) && k > 0)){
                        CUDACHECK(cudaMemcpyAsync(
                            d_outputpositionsArr[bufferIndex],
                            h_outputpositionsArr[bufferIndex],
                            sizeof(int) * k,
                            H2D,
                            tempData->streams[bufferIndex]
                        ));

                        gatherHostQualities(
                            tempData,
                            hostindicesarray[bufferIndex].data(),
                            k,
                            h_hostdataArr[bufferIndex],
                            out_quality_pitch
                        );

                        CUDACHECK(cudaMemcpyAsync(
                            d_hostdataArr[bufferIndex],
                            h_hostdataArr[bufferIndex],
                            sizeof(char) * out_quality_pitch * k,
                            H2D,
                            tempData->streams[bufferIndex]
                        ));

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
                        ); CUDACHECKASYNC;

                        bufferIndex = (bufferIndex + 1) % 2;
                        k = 0;
                    }
                }

                CUDACHECK(cudaEventRecord(tempData->event, tempData->streams[0]));
                CUDACHECK(cudaStreamWaitEvent(stream, tempData->event, 0));
                CUDACHECK(cudaEventRecord(tempData->event, tempData->streams[1]));
                CUDACHECK(cudaStreamWaitEvent(stream, tempData->event, 0));

                
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

                    CUDACHECK(cudaStreamSynchronize(tempData->streams[bufferIndex])); // protect pinned buffer

                    gatherHostQualities(
                        tempData,
                        h_readIds + begin,
                        sizeOfCurrentBatch,
                        hostpointers[bufferIndex],
                        out_quality_pitch
                    );

                    CUDACHECK(cudaMemcpyAsync(
                        d_quality_data + out_quality_pitch * begin,
                        hostpointers[bufferIndex],
                        sizeof(char) * out_quality_pitch * sizeOfCurrentBatch,
                        H2D,
                        tempData->streams[bufferIndex]
                    ));
                }

                CUDACHECK(cudaEventRecord(tempData->event, tempData->streams[0]));
                CUDACHECK(cudaStreamWaitEvent(stream, tempData->event, 0));
                CUDACHECK(cudaEventRecord(tempData->event, tempData->streams[1]));
                CUDACHECK(cudaStreamWaitEvent(stream, tempData->event, 0));
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

        CUDACHECK(cudaEventRecord(tempData->event, stream));

        nvtx::pop_range();
    }

    void gatherSequenceLengths(
        ReadStorageHandle& handle,
        int* d_lengths,
        const read_number* d_readIds,
        int numSequences,    
        cudaStream_t stream
    ) const override{

        if(numSequences == 0) return;

        gpuLengthStorage.gatherLengthsOnDeviceAsync(
            d_lengths, 
            d_readIds, 
            numSequences, 
            stream
        );

    }

    void getIdsOfAmbiguousReads(
        read_number* ids
    ) const override{
        std::copy(ambigReadIds.begin(), ambigReadIds.end(), ids);
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

    MemoryUsage getMemoryInfo(const ReadStorageHandle& handle) const override{
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

    bool isPairedEnd() const override{
        return pairedEnd;
    }

    void destroy(){
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
        TempData* tempData,
        const read_number* readIds,
        int numSequences,
        unsigned int* outputarray,
        std::size_t outputPitchInInts
    ) const {
        if(numSequences == 0) return;

        if(!isStandalone()){
            cpuReadStorage->gatherSequences(
                outputarray,
                outputPitchInInts,
                readIds,
                numSequences
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
        TempData* tempData,
        const read_number* readIds,
        int numSequences,
        char* outputarray,
        std::size_t outputPitchInBytes
    ) const {
        if(numSequences == 0) return;
        if(!isStandalone()){

            cpuReadStorage->gatherQualities(
                outputarray,
                outputPitchInBytes,
                readIds,
                numSequences
            );
        }else{

            //convert readIds into local indices for host partition
            auto indexGenerator = [&](auto i){
                return readIds[i] - qualitiesGpu.getNumRows();
            };

            gatherHostStandaloneImpl(
                (char*)hostqualities.data(),
                hostQualityPitch,
                hostQualityPitch,
                indexGenerator,
                numSequences,
                outputarray,
                outputPitchInBytes
            );
        }
    }

    TempData* getTempDataFromHandle(const ReadStorageHandle& handle) const{
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

            gpuLengthStorage.destroy();

            for(auto& pair : bitArraysUndeterminedBase){
                cub::SwitchDevice sd(pair.first);
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
    
    bool pairedEnd{};
    bool useQualityScores{};
    int sequenceLengthLowerBound{};
    int sequenceLengthUpperBound{};
    read_number numberOfReads{};
    std::int64_t numberOfAmbiguousReads{};

    std::size_t numHostSequences{};
    std::size_t numHostQualities{};
    std::size_t hostSequencePitch{};
    std::size_t hostQualityPitch{};
    const CpuReadStorage* cpuReadStorage{};

    MultiGpu2dArray<unsigned int, IndexType> sequencesGpu{};
    MultiGpu2dArray<unsigned int, IndexType> qualitiesGpu{};
    std::map<int, GpuBitArray<read_number>> bitArraysUndeterminedBase;
    std::vector<read_number> ambigReadIds{};


    GPULengthStore3<std::uint32_t> gpuLengthStorage{};
    std::vector<unsigned int> hostsequences{};
    std::vector<unsigned int> hostqualities{};

    std::vector<int> deviceIds{};

    mutable int counter = 0;
    mutable SharedMutex sharedmutex{};
    mutable std::vector<std::unique_ptr<TempData>> tempdataVector{};
};
    
}
}






#endif