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
#include <qualityscorecompression.hpp>


#include <vector>
#include <cstdint>
#include <memory>
#include <map>
#include <limits>
#include <array>

#include <cub/cub.cuh>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>

#include <rmm/device_vector.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/mr/device/thrust_allocator_adaptor.hpp>
#include <gpu/rmm_utilities.cuh>

namespace care{
namespace gpu{


namespace multigpureadstoragekernels{

    template<int numBits, class LengthIterator>
    __global__
    void decompressAndScatterEncodedQualitiesKernel(
        char* __restrict__ output,
        std::size_t outputPitchInBytes,
        const unsigned int* __restrict__ encodedQualities,
        std::size_t encodedQualitiesPitchInInts,
        LengthIterator lengths,
        const int* scatterpositions,
        int numSequences
    ){

        constexpr int groupsize = QualityDecompressionGroupSize<numBits>::value;
    
        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int numGroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;

        for(int s = groupId; s < numSequences; s += numGroups){
            const unsigned int* const myCompressed = encodedQualities + encodedQualitiesPitchInInts * s;
            const int outputRow = scatterpositions[s];
            char* const myQuality = output + outputPitchInBytes * outputRow;
            const int myLength = lengths[s];

            QualityCompressor<numBits>::decodeQualityToString(group, myQuality, myCompressed, myLength);
        }
    }


    template<int numBits, class LengthIterator>
    void callDecompressAndScatterEncodedQualitiesKernel(
        char* __restrict__ output,
        std::size_t outputPitchInBytes,
        const unsigned int* __restrict__ encodedQualities,
        std::size_t encodedQualitiesPitchInInts,
        LengthIterator lengths,
        const int* scatterpositions,
        int numSequences,
        cudaStream_t stream
    ){
        constexpr int groupsize = QualityDecompressionGroupSize<numBits>::value;
        constexpr int blocksize = 128;
        constexpr int groupsPerBlock = blocksize / groupsize;

        const int numBlocks = SDIV(numSequences, groupsPerBlock);

        decompressAndScatterEncodedQualitiesKernel<numBits><<<numBlocks, blocksize, 0, stream>>>(
            output, 
            outputPitchInBytes, 
            encodedQualities, 
            encodedQualitiesPitchInInts, 
            lengths, 
            scatterpositions, 
            numSequences
        );

        CUDACHECKASYNC;
    }

    template<class LengthIterator>
    void callDecompressAndScatterEncodedQualitiesKernel(
        char* __restrict__ output,
        std::size_t outputPitchInBytes,
        const unsigned int* __restrict__ encodedQualities,
        std::size_t encodedQualitiesPitchInInts,
        LengthIterator lengths,
        const int* scatterpositions,
        int numSequences,
        int numBits,
        cudaStream_t stream
    ){
        switch(numBits){
            case 1: callDecompressAndScatterEncodedQualitiesKernel<1>(
                output, 
                outputPitchInBytes, 
                encodedQualities, 
                encodedQualitiesPitchInInts, 
                lengths, 
                scatterpositions, 
                numSequences,
                stream
            ); break;

            case 2: callDecompressAndScatterEncodedQualitiesKernel<2>(
                output, 
                outputPitchInBytes, 
                encodedQualities, 
                encodedQualitiesPitchInInts, 
                lengths, 
                scatterpositions, 
                numSequences,
                stream
            ); break;

            case 8: callDecompressAndScatterEncodedQualitiesKernel<8>(
                output, 
                outputPitchInBytes, 
                encodedQualities, 
                encodedQualitiesPitchInInts, 
                lengths, 
                scatterpositions, 
                numSequences,
                stream
            ); break;

            default: assert(false); break;
        }        
    }
}

class MultiGpuReadStorage : public GpuReadStorage {
public:    

    template<class W>
    using HostBuffer = helpers::SimpleAllocationPinnedHost<W>;

    using IndexType = read_number;

    enum class Layout{FirstFit, EvenShare};

    struct TempData{

        TempData() 
        : event{cudaEventDisableTiming},
            d_tempStorage{0, cudaStreamPerThread}
        {
            CUDACHECK(cudaGetDevice(&deviceId));
            CUDACHECK(cudaStreamSynchronize(cudaStreamPerThread));
        }

        ~TempData(){
            cub::SwitchDevice sd(deviceId);
            d_tempStorage.release();
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage result{};

            result.host += pinnedBuffer.capacityInBytes();

            result += handleSequences->getMemoryInfo();
            result += handleQualities->getMemoryInfo();

            return result;
        }

        int deviceId{};
        CudaEvent event{};
        CudaEvent dependencyevent{};
        HostBuffer<char> pinnedBuffer{};
        std::array<CudaStream,2> streams{};

        rmm::device_uvector<char> d_tempStorage;

        typename MultiGpu2dArray<unsigned int, IndexType>::Handle handleSequences{};
        typename MultiGpu2dArray<unsigned int, IndexType>::Handle handleQualities{};
    };

    MultiGpuReadStorage(
        const CpuReadStorage& cpuReadStorage_, 
        std::vector<int> deviceIds_, 
        std::vector<std::size_t> memoryLimitsPerDevice_,
        std::size_t memoryLimitHost,
        int numQualBits,
        Layout gpuDataLayout,
        bool directPeerAccess_
    ) 
        : directPeerAccess(directPeerAccess_)
    {

        rebuild(
            cpuReadStorage_,
            deviceIds_,
            memoryLimitsPerDevice_,
            memoryLimitHost,
            numQualBits,
            gpuDataLayout
        );

    }

    void rebuild(
        const CpuReadStorage& cpuReadStorage_, 
        std::vector<int> deviceIds_, 
        std::vector<std::size_t> memoryLimitsPerDevice,
        std::size_t memoryLimitHost,
        int /*numQualBits*/,
        Layout gpuDataLayout
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
        
        numQualityBits = cpuReadStorage->getQualityBits();

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

            //TODO if numAmbiguous == 0, don't need bitArrays
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

                rmm::device_vector<bool> d_values(batchsize);
                thrust::fill(rmm::exec_policy_nosync(), d_values.begin(), d_values.end(), true);

                for(int i = 0; i < numBatches; i++){
                    size_t begin = i * batchsize;
                    size_t end = std::min((i+1) * batchsize, numAmbiguous);
                    size_t elements = end - begin;

                    setBitarray<<<SDIV(elements, 128), 128>>>(
                        bitArraysUndeterminedBase[deviceId], 
                        thrust::raw_pointer_cast(d_values.data()), 
                        h_positions.data() + begin, 
                        elements
                    ); CUDACHECKASYNC;

                }
                CUDACHECK(cudaDeviceSynchronize());

                ambigReadIds.clear();
                ambigReadIds.insert(ambigReadIds.end(), h_positions.begin(), h_positions.end());
            }
        }

        

        const MultiGpu2dArrayLayout dataLayout = gpuDataLayout == Layout::FirstFit ? MultiGpu2dArrayLayout::FirstFit : MultiGpu2dArrayLayout::EvenShare;

        //handle sequences
        const int numColumnsSequences = SequenceHelpers::getEncodedNumInts2Bit(cpuReadStorage->getSequenceLengthUpperBound());
        sequencesGpu = std::move(
            MultiGpu2dArray<unsigned int, IndexType>(
                numReads,
                numColumnsSequences,
                sizeof(unsigned int),
                deviceIds,
                memoryLimitsPerDevice,
                directPeerAccess,
                dataLayout,
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
            std::array<helpers::SimpleAllocationPinnedHost<unsigned int>, numbuffers> hostdatabuffers{};

            std::array<rmm::device_vector<IndexType>, numbuffers> deviceindexbuffers{};
            std::array<rmm::device_vector<unsigned int>, numbuffers> devicedatabuffers{};

            std::array<IndexType*, numbuffers> indexarray{};
            std::array<unsigned int*, numbuffers> hostdataarray{};
            std::array<unsigned int*, numbuffers> dataarray{};

            for(int i = 0; i < numbuffers; i++){
                indexbuffers[i].resize(batchsize);
                hostdatabuffers[i].resize(batchsize * numColumnsSequences);
                devicedatabuffers[i].resize(batchsize * numColumnsSequences);

                indexarray[i] = indexbuffers[i].data();
                hostdataarray[i] = hostdatabuffers[i].data();
                dataarray[i] = thrust::raw_pointer_cast(devicedatabuffers[i].data());
            }
            CUDACHECK(cudaDeviceSynchronize());
            
            for(std::size_t i = 0, iteration = 0; i < sequencesGpu.getNumRows(); i += batchsize, iteration++){
                const int bufferIndex = iteration % numbuffers;

                CUDACHECK(cudaStreamSynchronize(streams[bufferIndex]));

                const std::size_t currentBatchsize = std::min(batchsize, sequencesGpu.getNumRows() - i);
                //std::iota(indexarray[bufferIndex], indexarray[bufferIndex] + currentBatchsize, i);

                // cpuReadStorage->gatherSequences(
                //     hostdataarray[bufferIndex],
                //     numColumnsSequences,
                //     indexarray[bufferIndex],
                //     currentBatchsize
                // );

                cpuReadStorage->gatherContiguousSequences(
                    hostdataarray[bufferIndex],
                    numColumnsSequences,
                    i,
                    currentBatchsize
                );

                CUDACHECK(cudaMemcpyAsync(
                    dataarray[bufferIndex],
                    hostdataarray[bufferIndex],
                    numColumnsSequences * currentBatchsize * sizeof(unsigned int),
                    H2D,
                    streams[bufferIndex]
                ));

                sequencesGpu.scatterContiguous(
                    arrayhandle, 
                    dataarray[bufferIndex], 
                    numColumnsSequences * sizeof(unsigned int), 
                    //indexarray[bufferIndex][0],
                    i,
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

        //std::fill(memoryLimitsPerDevice.begin(), memoryLimitsPerDevice.end(), 0); //DEBUG

        const int numColumnsCompressedQualitiesInts = QualityCompressionHelper::getNumInts(cpuReadStorage->getSequenceLengthUpperBound(), numQualityBits);

        if(canUseQualityScores()){

            qualitiesGpu = std::move(
                MultiGpu2dArray<unsigned int, IndexType>(
                    numReads,
                    numColumnsCompressedQualitiesInts,
                    sizeof(unsigned int),
                    deviceIds,
                    memoryLimitsPerDevice,
                    directPeerAccess,
                    dataLayout,
                    MultiGpu2dArrayInitMode::CanDiscardRows
                )
            );

            {
                constexpr std::size_t batchsize = 65000;
                constexpr int numbuffers = 2;

                std::array<CudaStream, numbuffers> streams{};
                auto arrayhandle = qualitiesGpu.makeHandle();

                std::array<helpers::SimpleAllocationPinnedHost<IndexType>, numbuffers> indexbuffers{};
                std::array<helpers::SimpleAllocationPinnedHost<unsigned int>, numbuffers> hostdatabuffers{};

                std::array<rmm::device_vector<IndexType>, numbuffers> deviceindexbuffers{};
                std::array<rmm::device_vector<unsigned int>, numbuffers> devicecompresseddatabuffers{};

                std::array<IndexType*, numbuffers> indexarray{};
                std::array<unsigned int*, numbuffers> hostdataarray{};
                std::array<unsigned int*, numbuffers> compresseddataarray{};

                for(int i = 0; i < numbuffers; i++){
                    indexbuffers[i].resize(batchsize);
                    hostdatabuffers[i].resize(batchsize * numColumnsCompressedQualitiesInts);
                    devicecompresseddatabuffers[i].resize(batchsize * numColumnsCompressedQualitiesInts);

                    indexarray[i] = indexbuffers[i].data();
                    hostdataarray[i] = hostdatabuffers[i].data();
                    compresseddataarray[i] = thrust::raw_pointer_cast(devicecompresseddatabuffers[i].data());
                }

                for(std::size_t i = 0, iteration = 0; i < qualitiesGpu.getNumRows(); i += batchsize, iteration++){
                    const int bufferIndex = iteration % numbuffers;
                    CUDACHECK(cudaStreamSynchronize(streams[bufferIndex]));

                    const std::size_t currentBatchsize = std::min(batchsize, qualitiesGpu.getNumRows() - i);
                    // std::iota(indexarray[bufferIndex], indexarray[bufferIndex] + currentBatchsize, i);

                    // cpuReadStorage->gatherEncodedQualities(
                    //     hostdataarray[bufferIndex],
                    //     numColumnsCompressedQualitiesInts,
                    //     indexarray[bufferIndex],
                    //     currentBatchsize
                    // );
                    cpuReadStorage->gatherContiguousEncodedQualities(
                        hostdataarray[bufferIndex],
                        numColumnsCompressedQualitiesInts,
                        i,
                        currentBatchsize
                    );

                    CUDACHECK(cudaMemcpyAsync(
                        compresseddataarray[bufferIndex],
                        hostdataarray[bufferIndex],
                        numColumnsCompressedQualitiesInts * currentBatchsize * sizeof(unsigned int),
                        H2D,
                        streams[bufferIndex]
                    ));

                    qualitiesGpu.scatterContiguous(
                        arrayhandle, 
                        compresseddataarray[bufferIndex], 
                        numColumnsCompressedQualitiesInts * sizeof(unsigned int), 
                        i, 
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
        hostSequencePitchBytes = numColumnsSequences * sizeof(unsigned int);
        hostQualityPitchBytes = numColumnsCompressedQualitiesInts * sizeof(unsigned int);

        std::size_t memoryOfHostSequences = numHostSequences * hostSequencePitchBytes;
        std::size_t memoryOfHostQualities = numHostQualities * hostQualityPitchBytes;

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

                    hostqualities.resize(numHostQualities * numColumnsCompressedQualitiesInts);

                    const std::size_t numQualitiesToCopy = numReads - qualitiesGpu.getNumRows();
                    const std::size_t batchsizeq = 100000;
                    for(std::size_t i = 0; i < numQualitiesToCopy; i += batchsizeq){
                        const std::size_t currentBatchsize = std::min(batchsizeq, numQualitiesToCopy - i);

                        std::vector<read_number> indices(currentBatchsize);
                        std::iota(indices.begin(), indices.end(), qualitiesGpu.getNumRows() + i);

                        cpuReadStorage->gatherEncodedQualities(
                            hostqualities.data() + i * numColumnsCompressedQualitiesInts,
                            numColumnsCompressedQualitiesInts,
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
        data->dependencyevent = CudaEvent{cudaEventDisableTiming};

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
        ReadStorageHandle& /*handle*/,
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
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr
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

        CUDACHECK(cudaEventRecord(tempData->dependencyevent, stream));
        for(int i = 0; i < 2; i++){
            CUDACHECK(cudaStreamWaitEvent(tempData->streams[i], tempData->dependencyevent, 0));
        }

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

                resizeWithSync(tempData->pinnedBuffer, temp_storage_bytes);

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
                std::array<int*, 2> h_outputpositionsArr{(int*)temp_allocations_host[2], (int*)temp_allocations_host[3]};

                rmm::device_uvector<unsigned int> d_gatheredData1(outSequencePitchInInts * batchsize, tempData->streams[0].getStream(), mr);
                rmm::device_uvector<unsigned int> d_gatheredData2(outSequencePitchInInts * batchsize, tempData->streams[1].getStream(), mr);

                rmm::device_uvector<int> d_outputPositions1(batchsize, tempData->streams[0].getStream(), mr);
                rmm::device_uvector<int> d_outputPositions2(batchsize, tempData->streams[1].getStream(), mr);

                std::array<unsigned int*, 2> d_hostdataArr{d_gatheredData1.data(), d_gatheredData2.data()};
                std::array<int*, 2> d_outputpositionsArr{d_outputPositions1.data(), d_outputPositions2.data()};

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

    void multi_gatherSequences(
        std::vector<ReadStorageHandle>& vec_handle,
        std::vector<unsigned int*>& vec_d_sequence_data,
        size_t outSequencePitchInInts,
        const std::vector<read_number*>& vec_d_readIds,
        const std::vector<int>& vec_numSequences,
        const std::vector<cudaStream_t>& streams,
        const std::vector<int>& callerDeviceIds,
        const std::vector<rmm::mr::device_memory_resource*>& /*mrs*/
    ) const override{
        nvtx::ScopedRange sr("multigpureadstorage::multi_gatherSequences", 4);
        if(hasHostSequences()) throw std::runtime_error("multi_gatherSequences cannot be used if some sequences are stored on the host");

        const int numGpus = callerDeviceIds.size();
        int totalNumSequences = 0;
        for(int g = 0; g < numGpus; g++){
            totalNumSequences += vec_numSequences[g];
        }
        if(totalNumSequences == 0) return;

        std::vector<size_t> vec_destRowPitchInBytes(numGpus, sizeof(unsigned int) * outSequencePitchInInts);
        std::vector<size_t> vec_numSequences_sizet(numGpus);
        std::vector<typename MultiGpu2dArray<unsigned int, IndexType>::Handle> vec_arrayhandle(numGpus);
        for(int g = 0; g < numGpus; g++){
            TempData* tempData = getTempDataFromHandle(vec_handle[g]);
            assert(tempData->deviceId == callerDeviceIds[g]);

            vec_arrayhandle[g] = tempData->handleSequences;
            vec_numSequences_sizet[g] = vec_numSequences[g];
        }

        sequencesGpu.multi_gather(
            vec_arrayhandle, 
            vec_d_sequence_data, 
            vec_destRowPitchInBytes, 
            vec_d_readIds, 
            vec_numSequences_sizet, 
            streams,
            callerDeviceIds
        );

    }

    void multi_gatherSequences(
        std::vector<ReadStorageHandle>& vec_handle,
        std::vector<unsigned int*>& vec_d_sequence_data,
        size_t outSequencePitchInInts,
        const std::vector<AsyncConstBufferWrapper<read_number>>& vec_h_readIdsAsync,
        const std::vector<read_number*>& vec_d_readIds,
        const std::vector<int>& vec_numSequences,
        const std::vector<cudaStream_t>& streams,
        const std::vector<int>& callerDeviceIds,
        const std::vector<rmm::mr::device_memory_resource*>& mrs
    ) const override{
        nvtx::ScopedRange sr("multigpureadstorage::multi_gatherSequences", 4);

        const int numGpus = callerDeviceIds.size();
        const int totalNumSequences = std::reduce(vec_numSequences.begin(), vec_numSequences.end());
        if(totalNumSequences == 0) return;

        std::vector<TempData*> vec_tempData(numGpus);
        for(int g = 0; g < numGpus; g++){
            TempData* tempData = getTempDataFromHandle(vec_handle[g]);
            assert(tempData->deviceId == callerDeviceIds[g]);

            vec_tempData[g] = tempData;
        }

        std::vector<bool> vec_hasSynchronized(numGpus, false);
        auto resizeWithSync = [&](int g, auto& data, std::size_t size){
            using W = decltype(*data.get());

            const std::size_t currentCapacity = data.capacityInBytes();
            const std::size_t newbytes = size * sizeof(W);
            if(!vec_hasSynchronized[g] && currentCapacity < newbytes){
                vec_tempData[g]->event.synchronize();
                vec_hasSynchronized[g] = true;
                //std::cerr << "SYNC" << "\n";
            }
            data.resize(size);
        };

        auto gpuGather = [&](){
            std::vector<typename MultiGpu2dArray<unsigned int, IndexType>::Handle> vec_arrayhandle(numGpus);
            std::vector<size_t> vec_destRowPitchInBytes(numGpus, sizeof(unsigned int) * outSequencePitchInInts);
            std::vector<size_t> vec_numSequences_sizet(numGpus);
            for(int g = 0; g < numGpus; g++){
                vec_numSequences_sizet[g] = vec_numSequences[g];
                vec_arrayhandle[g] = vec_tempData[g]->handleSequences;
            }

            sequencesGpu.multi_gather(
                vec_arrayhandle, 
                vec_d_sequence_data, 
                vec_destRowPitchInBytes, 
                vec_d_readIds, 
                vec_numSequences_sizet, 
                hasHostSequences(),
                streams,
                callerDeviceIds
            );
        };

        auto hostGather = [&](int g){
            cub::SwitchDevice sd_(callerDeviceIds[g]);
            cudaStream_t stream = streams[g];
            TempData* const tempData = vec_tempData[g];
            auto* const mr = mrs[g];

            const std::size_t sequencepitch = sizeof(unsigned int) * outSequencePitchInInts;
            const int numSequences = vec_numSequences[g];

            if(numSequences == 0) return;

            constexpr std::size_t memorylimitbatch = 1 << 19; // 512KB

            CUDACHECK(cudaEventRecord(tempData->dependencyevent, stream));
            for(int i = 0; i < 2; i++){
                CUDACHECK(cudaStreamWaitEvent(tempData->streams[i], tempData->dependencyevent, 0));
            }



            vec_h_readIdsAsync[g].wait();
            const read_number* const h_readIds = vec_h_readIdsAsync[g].data();

            if(hasGpuSequences()){

                /*
                    - Find subset of h_readIds which refer to data on the host
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

                resizeWithSync(g, tempData->pinnedBuffer, temp_storage_bytes);

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
                std::array<int*, 2> h_outputpositionsArr{(int*)temp_allocations_host[2], (int*)temp_allocations_host[3]};

                rmm::device_uvector<unsigned int> d_gatheredData1(outSequencePitchInInts * batchsize, tempData->streams[0].getStream(), mr);
                rmm::device_uvector<unsigned int> d_gatheredData2(outSequencePitchInInts * batchsize, tempData->streams[1].getStream(), mr);

                rmm::device_uvector<int> d_outputPositions1(batchsize, tempData->streams[0].getStream(), mr);
                rmm::device_uvector<int> d_outputPositions2(batchsize, tempData->streams[1].getStream(), mr);

                std::array<unsigned int*, 2> d_hostdataArr{d_gatheredData1.data(), d_gatheredData2.data()};
                std::array<int*, 2> d_outputpositionsArr{d_outputPositions1.data(), d_outputPositions2.data()};

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
                                d_sequence_data = vec_d_sequence_data[g],
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

                resizeWithSync(g, tempData->pinnedBuffer, 2 * batchsize * sequencepitch);

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
                        (char*)(vec_d_sequence_data[g]) + sequencepitch * begin,
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

                for(int g = 0; g < numGpus; g++){
                    hostGather(g);
                }

                gpuGather();
            }else{
                gpuGather();
            }
        }else{
            for(int g = 0; g < numGpus; g++){
                hostGather(g);
            }
        }

        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd_(callerDeviceIds[g]);
            cudaStream_t stream = streams[g];
            TempData* const tempData = vec_tempData[g];
            CUDACHECK(cudaEventRecord(tempData->event, stream));
        }

    }

    // void multi_gatherQualities(
    //     std::vector<ReadStorageHandle>& vec_handle,
    //     std::vector<char*>& vec_d_quality_data,
    //     size_t out_quality_pitch,
    //     const std::vector<read_number*>& vec_d_readIds,
    //     const std::vector<int>& vec_numSequences,
    //     const std::vector<cudaStream_t>& streams,
    //     const std::vector<int>& callerDeviceIds,
    //     const std::vector<rmm::mr::device_memory_resource*>& /*mrs*/
    // ) const override{
    //     nvtx::ScopedRange sr("multigpureadstorage::multi_gatherQualities", 4);
    //     if(hasHostQualities()) throw std::runtime_error("multi_gatherQualities cannot be used if some sequences are stored on the host");

    //     const int numGpus = callerDeviceIds.size();
    //     int totalNumSequences = 0;
    //     for(int g = 0; g < numGpus; g++){
    //         totalNumSequences += vec_numSequences[g];
    //     }
    //     if(totalNumSequences == 0) return;

    //     std::vector<size_t> vec_destRowPitchInBytes(numGpus, out_quality_pitch);
    //     std::vector<size_t> vec_numSequences_sizet(numGpus);
    //     std::vector<unsigned int*> vec_d_quality_data_as_uint(numGpus);
    //     std::vector<typename MultiGpu2dArray<unsigned int, IndexType>::Handle> vec_arrayhandle(numGpus);
    //     for(int g = 0; g < numGpus; g++){
    //         TempData* tempData = getTempDataFromHandle(vec_handle[g]);
    //         assert(tempData->deviceId == callerDeviceIds[g]);

    //         vec_arrayhandle[g] = tempData->handleQualities;
    //         vec_numSequences_sizet[g] = vec_numSequences[g];
    //         vec_d_quality_data_as_uint[g] = reinterpret_cast<unsigned int*>(vec_d_quality_data[g]);
    //     }

    //     qualitiesGpu.multi_gather(
    //         vec_arrayhandle, 
    //         vec_d_quality_data_as_uint, 
    //         vec_destRowPitchInBytes, 
    //         vec_d_readIds, 
    //         vec_numSequences_sizet, 
    //         streams,
    //         callerDeviceIds
    //     );

    // }

    void multi_gatherQualities(
        std::vector<ReadStorageHandle>& vec_handle,
        std::vector<char*>& vec_d_quality_data,
        size_t out_quality_pitch,
        const std::vector<AsyncConstBufferWrapper<read_number>>& vec_h_readIdsAsync,
        const std::vector<read_number*>& vec_d_readIds,
        const std::vector<int>& vec_numSequences,
        const std::vector<cudaStream_t>& streams,
        const std::vector<int>& callerDeviceIds,
        const std::vector<rmm::mr::device_memory_resource*>& mrs
    ) const override{
        nvtx::ScopedRange sr("multigpureadstorage::multi_gatherQualities", 4);

        const int numGpus = callerDeviceIds.size();
        const int totalNumSequences = std::reduce(vec_numSequences.begin(), vec_numSequences.end());
        if(totalNumSequences == 0) return;

        std::vector<TempData*> vec_tempData(numGpus);
        for(int g = 0; g < numGpus; g++){
            TempData* tempData = getTempDataFromHandle(vec_handle[g]);
            assert(tempData->deviceId == callerDeviceIds[g]);

            vec_tempData[g] = tempData;
        }

        std::vector<bool> vec_hasSynchronized(numGpus, false);
        auto resizeWithSync = [&](int g, auto& data, std::size_t size){
            using W = decltype(*data.get());

            const std::size_t currentCapacity = data.capacityInBytes();
            const std::size_t newbytes = size * sizeof(W);
            if(!vec_hasSynchronized[g] && currentCapacity < newbytes){
                vec_tempData[g]->event.synchronize();
                vec_hasSynchronized[g] = true;
                //std::cerr << "SYNC" << "\n";
            }
            data.resize(size);
        };

        const int numColumnsCompressedQualitiesInts = qualitiesGpu.getNumColumns();

        std::vector<unsigned int*> vec_d_compressed(numGpus);
        std::vector<unsigned int*> vec_d_gatherdestination(numGpus);
        std::size_t gatherdestinationPitchInBytes = 0;
        if(numQualityBits == 8){
            for(int g = 0; g < numGpus; g++){
                vec_d_gatherdestination[g] = reinterpret_cast<unsigned int*>(vec_d_quality_data[g]);
            }
            gatherdestinationPitchInBytes = out_quality_pitch;
        }else{
            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd(callerDeviceIds[g]);
                constexpr int roundUpTo = 10000;
                const int roundedNumSequences = SDIV(vec_numSequences[g], roundUpTo) * roundUpTo;
                resizeUninitialized(
                    vec_tempData[g]->d_tempStorage,
                    sizeof(unsigned int) * roundedNumSequences * numColumnsCompressedQualitiesInts,
                    streams[g]
                );
                vec_d_compressed[g] = reinterpret_cast<unsigned int*>(vec_tempData[g]->d_tempStorage.data());
                vec_d_gatherdestination[g] = vec_d_compressed[g];
            }
            
            gatherdestinationPitchInBytes = numColumnsCompressedQualitiesInts * sizeof(unsigned int);
        }

        auto gpuGather = [&](){

            std::vector<size_t> vec_destRowPitchInBytes(numGpus, gatherdestinationPitchInBytes);
            std::vector<size_t> vec_numSequences_sizet(numGpus);
            std::vector<unsigned int*> vec_d_quality_data_as_uint(numGpus);
            std::vector<typename MultiGpu2dArray<unsigned int, IndexType>::Handle> vec_arrayhandle(numGpus);
            for(int g = 0; g < numGpus; g++){
                vec_arrayhandle[g] = vec_tempData[g]->handleQualities;
                vec_numSequences_sizet[g] = vec_numSequences[g];
                vec_d_quality_data_as_uint[g] = reinterpret_cast<unsigned int*>(vec_d_gatherdestination[g]);
            }

            qualitiesGpu.multi_gather(
                vec_arrayhandle, 
                vec_d_quality_data_as_uint, 
                vec_destRowPitchInBytes, 
                vec_d_readIds, 
                vec_numSequences_sizet, 
                hasHostQualities(),
                streams,
                callerDeviceIds
            );
        };


        auto hostGather = [&](int g){
            cub::SwitchDevice sd_(callerDeviceIds[g]);
            cudaStream_t stream = streams[g];
            TempData* const tempData = vec_tempData[g];
            auto* const mr = mrs[g];

            const int numSequences = vec_numSequences[g];

            if(numSequences == 0) return;

            constexpr std::size_t memorylimitbatch = 1 << 19; // 512KB

            CUDACHECK(cudaEventRecord(tempData->dependencyevent, stream));
            for(int i = 0; i < 2; i++){
                CUDACHECK(cudaStreamWaitEvent(tempData->streams[i], tempData->dependencyevent, 0));
            }



            vec_h_readIdsAsync[g].wait();
            const read_number* const h_readIds = vec_h_readIdsAsync[g].data();

            if(hasGpuSequences()){

                /*
                    - Find subset of h_readIds which refer to data on the host
                    - Gather data on the host
                    - Scatter into device result array
                    
                    Is batched such that the required temporary storage per batch is small
                */

                const int batchsize = std::min(SDIV(memorylimitbatch, (sizeof(unsigned int) * numColumnsCompressedQualitiesInts + sizeof(int))), std::size_t(numSequences));

                void* temp_allocations_host[4]{};
                void* temp_allocations_device[4]{};
                std::size_t temp_allocation_sizes[4]{};

                temp_allocation_sizes[0] = sizeof(unsigned int) * numColumnsCompressedQualitiesInts * batchsize; // gathered host data 1
                temp_allocation_sizes[1] = sizeof(unsigned int) * numColumnsCompressedQualitiesInts * batchsize; // gathered host data 2
                temp_allocation_sizes[2] = sizeof(int) * batchsize; // output positions 1
                temp_allocation_sizes[3] = sizeof(int) * batchsize; // output positions 2

                std::size_t temp_storage_bytes = 0;
                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    temp_storage_bytes,
                    temp_allocations_device,
                    temp_allocation_sizes
                ));

                resizeWithSync(g, tempData->pinnedBuffer, temp_storage_bytes);

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
                std::array<int*, 2> h_outputpositionsArr{(int*)temp_allocations_host[2], (int*)temp_allocations_host[3]};

                rmm::device_uvector<unsigned int> d_gatheredData1(numColumnsCompressedQualitiesInts * batchsize, tempData->streams[0].getStream(), mr);
                rmm::device_uvector<unsigned int> d_gatheredData2(numColumnsCompressedQualitiesInts * batchsize, tempData->streams[1].getStream(), mr);

                rmm::device_uvector<int> d_outputPositions1(batchsize, tempData->streams[0].getStream(), mr);
                rmm::device_uvector<int> d_outputPositions2(batchsize, tempData->streams[1].getStream(), mr);

                std::array<unsigned int*, 2> d_hostdataArr{d_gatheredData1.data(), d_gatheredData2.data()};
                std::array<int*, 2> d_outputpositionsArr{d_outputPositions1.data(), d_outputPositions2.data()};

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

                        gatherHostQualitiesEncoded(
                            tempData,
                            hostindicesarray[bufferIndex].data(),
                            k,
                            h_hostdataArr[bufferIndex],
                            numColumnsCompressedQualitiesInts
                        );

                        CUDACHECK(cudaMemcpyAsync(
                            d_hostdataArr[bufferIndex],
                            h_hostdataArr[bufferIndex],
                            sizeof(unsigned int) * numColumnsCompressedQualitiesInts * k,
                            H2D,
                            tempData->streams[bufferIndex]
                        ));

                        const int intsToScatter = k * numColumnsCompressedQualitiesInts;

                        helpers::lambda_kernel<<<SDIV(intsToScatter, 128), 128, 0, tempData->streams[bufferIndex]>>>(
                            [
                                intsToScatter,
                                d_gatherdestination = vec_d_gatherdestination[g],
                                gatherdestinationPitchInBytes,
                                numColumnsCompressedQualitiesInts,
                                d_outputpositions = d_outputpositionsArr[bufferIndex],
                                d_gatheredhost = d_hostdataArr[bufferIndex]
                            ] __device__ (){
                                const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                                const int stride = blockDim.x * gridDim.x;

                                for(int i = tid; i < intsToScatter; i += stride){
                                    const int inputrow = i / numColumnsCompressedQualitiesInts;
                                    const int inputcol = i % numColumnsCompressedQualitiesInts;
                                    const int outputcol = inputcol;
                                    const int outputrow = d_outputpositions[inputrow];

                                    ((unsigned int*)(((char*)d_gatherdestination) + outputrow * gatherdestinationPitchInBytes))[outputcol] 
                                        = d_gatheredhost[inputrow * numColumnsCompressedQualitiesInts + inputcol];
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
                const std::size_t compressedMemoryForOneSeq = numColumnsCompressedQualitiesInts * sizeof(unsigned int);
                const int batchsize = std::min(SDIV(memorylimitbatch, compressedMemoryForOneSeq), std::size_t(numSequences));

                resizeWithSync(g, tempData->pinnedBuffer, 2 * batchsize * compressedMemoryForOneSeq);

                std::array<unsigned int*, 2> hostpointers{
                    reinterpret_cast<unsigned int*>(tempData->pinnedBuffer.data()), 
                    reinterpret_cast<unsigned int*>(tempData->pinnedBuffer.data()) + batchsize * numColumnsCompressedQualitiesInts
                };
                assert(hostpointers.size() == tempData->streams.size());

                const int numBatches = SDIV(numSequences, batchsize);
                for(int b = 0; b < numBatches; b++){
                    const int bufferIndex = b % 2;

                    const int begin = b * batchsize;
                    const int end = std::min(numSequences, (b+1) * batchsize);
                    const int sizeOfCurrentBatch = end - begin;

                    CUDACHECK(cudaStreamSynchronize(tempData->streams[bufferIndex])); // protect pinned buffer

                    gatherHostQualitiesEncoded(
                        tempData,
                        h_readIds + begin,
                        sizeOfCurrentBatch,
                        hostpointers[bufferIndex],
                        numColumnsCompressedQualitiesInts
                    );

                    if(gatherdestinationPitchInBytes == sizeof(unsigned int) * numColumnsCompressedQualitiesInts){
                        CUDACHECK(cudaMemcpyAsync(
                            ((char*)vec_d_gatherdestination[g]) + gatherdestinationPitchInBytes * begin,
                            hostpointers[bufferIndex],
                            sizeof(unsigned int) * numColumnsCompressedQualitiesInts * sizeOfCurrentBatch,
                            H2D,
                            tempData->streams[bufferIndex]
                        ));
                    }else{
                        CUDACHECK(cudaMemcpy2DAsync(
                            ((char*)vec_d_gatherdestination[g]) + gatherdestinationPitchInBytes * begin,
                            gatherdestinationPitchInBytes,
                            hostpointers[bufferIndex],
                            sizeof(unsigned int) * numColumnsCompressedQualitiesInts,
                            sizeof(unsigned int) * numColumnsCompressedQualitiesInts,
                            sizeOfCurrentBatch,
                            H2D,
                            tempData->streams[bufferIndex]
                        ));
                    }
                }

                CUDACHECK(cudaEventRecord(tempData->event, tempData->streams[0]));
                CUDACHECK(cudaStreamWaitEvent(stream, tempData->event, 0));
                CUDACHECK(cudaEventRecord(tempData->event, tempData->streams[1]));
                CUDACHECK(cudaStreamWaitEvent(stream, tempData->event, 0));

            }
        };




        if(hasGpuQualities()){
            if(hasHostQualities()){

                for(int g = 0; g < numGpus; g++){
                    hostGather(g);
                }

                gpuGather();
            }else{
                gpuGather();
            }
        }else{
            for(int g = 0; g < numGpus; g++){
                hostGather(g);
            }
        }

        if(numQualityBits != 8){

            const int maxLengthCompressedPitch = numColumnsCompressedQualitiesInts * sizeof(unsigned int) * 8 / numQualityBits;
            const int maxLengthUncompressedPitch = out_quality_pitch;
            const int l = std::min(maxLengthCompressedPitch, maxLengthUncompressedPitch);

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd_(callerDeviceIds[g]);
                cudaStream_t stream = streams[g];

                if(vec_numSequences[g] > 0){

                    callDecompressQualityScoresKernel(
                        vec_d_quality_data[g], 
                        out_quality_pitch,
                        vec_d_compressed[g], 
                        numColumnsCompressedQualitiesInts,
                        thrust::make_constant_iterator(l),
                        vec_numSequences[g],
                        numQualityBits,
                        stream
                    );
                }
            }
        }else{
            //all good
        }

        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd_(callerDeviceIds[g]);
            cudaStream_t stream = streams[g];
            TempData* const tempData = vec_tempData[g];
            CUDACHECK(cudaEventRecord(tempData->event, stream));
        }

        
    }


    void multi_gatherContiguousQualities(
        std::vector<ReadStorageHandle>& vec_handle,
        std::vector<char*>& vec_d_quality_data,
        size_t out_quality_pitch,
        const std::vector<read_number>& vec_firstIndex,
        const std::vector<int>& vec_numSequences,
        const std::vector<cudaStream_t>& streams,
        const std::vector<int>& callerDeviceIds,
        const std::vector<rmm::mr::device_memory_resource*>& mrs
    ) const override{
        nvtx::ScopedRange sr("multigpureadstorage::multi_gatherContiguousQualities", 4);

        const int numGpus = callerDeviceIds.size();
        const int totalNumSequences = std::reduce(vec_numSequences.begin(), vec_numSequences.end());
        if(totalNumSequences == 0) return;

        std::vector<TempData*> vec_tempData(numGpus);
        for(int g = 0; g < numGpus; g++){
            TempData* tempData = getTempDataFromHandle(vec_handle[g]);
            assert(tempData->deviceId == callerDeviceIds[g]);

            vec_tempData[g] = tempData;
        }

        std::vector<bool> vec_hasSynchronized(numGpus, false);
        auto resizeWithSync = [&](int g, auto& data, std::size_t size){
            using W = decltype(*data.get());

            const std::size_t currentCapacity = data.capacityInBytes();
            const std::size_t newbytes = size * sizeof(W);
            if(!vec_hasSynchronized[g] && currentCapacity < newbytes){
                vec_tempData[g]->event.synchronize();
                vec_hasSynchronized[g] = true;
                //std::cerr << "SYNC" << "\n";
            }
            data.resize(size);
        };

        const int numColumnsCompressedQualitiesInts = qualitiesGpu.getNumColumns();

        std::vector<std::size_t> vec_numGatherOnGpu(numGpus);
        std::vector<std::size_t> vec_firstIndexOnGpu(numGpus);
        std::vector<std::size_t> vec_numGatherOnHost(numGpus);
        std::vector<std::size_t> vec_firstIndexOnHost(numGpus);

        for(int g = 0; g < numGpus; g++){
            vec_numGatherOnGpu[g] = vec_firstIndex[g] < qualitiesGpu.getNumRows() ? std::min(qualitiesGpu.getNumRows() - vec_firstIndex[g], std::size_t(vec_numSequences[g])) : 0;
            vec_firstIndexOnGpu[g]  = vec_firstIndex[g] < qualitiesGpu.getNumRows() ? vec_firstIndex[g] : 0;
            vec_numGatherOnHost[g]  = vec_numSequences[g] - vec_numGatherOnGpu[g];
            vec_firstIndexOnHost[g]  = vec_firstIndex[g] + vec_numGatherOnGpu[g];
        }

        std::vector<unsigned int*> vec_d_gatherdestination(numGpus);
        std::size_t gatherdestinationPitchInBytes = 0;
        if(numQualityBits == 8){
            for(int g = 0; g < numGpus; g++){
                vec_d_gatherdestination[g] = reinterpret_cast<unsigned int*>(vec_d_quality_data[g]);
            }
            gatherdestinationPitchInBytes = out_quality_pitch;
        }else{
            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd(callerDeviceIds[g]);
                constexpr int roundUpTo = 10000;
                const int roundedNumSequences = SDIV(vec_numSequences[g], roundUpTo) * roundUpTo;
                resizeUninitialized(
                    vec_tempData[g]->d_tempStorage,
                    sizeof(unsigned int) * roundedNumSequences * numColumnsCompressedQualitiesInts,
                    streams[g]
                );
                vec_d_gatherdestination[g] = reinterpret_cast<unsigned int*>(vec_tempData[g]->d_tempStorage.data());
            }
            
            gatherdestinationPitchInBytes = numColumnsCompressedQualitiesInts * sizeof(unsigned int);
        }

        auto gpuGather = [&](){

            for(int g = 0; g < numGpus; g++){
                if(vec_numGatherOnGpu[g] > 0){
                    cub::SwitchDevice sd_(callerDeviceIds[g]);

                    qualitiesGpu.gatherContiguous(
                        vec_tempData[g]->handleQualities,
                        vec_d_gatherdestination[g],
                        gatherdestinationPitchInBytes,
                        vec_firstIndexOnGpu[g],
                        vec_numGatherOnGpu[g],
                        streams[g]
                    );
                }
            }

            if(numQualityBits != 8){
                for(int g = 0; g < numGpus; g++){
                    if(vec_numGatherOnGpu[g] > 0){
                        cub::SwitchDevice sd_(callerDeviceIds[g]);

                        const int maxLengthCompressedPitch = numColumnsCompressedQualitiesInts * sizeof(unsigned int) * 8 / numQualityBits;
                        const int maxLengthUncompressedPitch = out_quality_pitch;
                        const int l = std::min(maxLengthCompressedPitch, maxLengthUncompressedPitch);

                        callDecompressQualityScoresKernel(
                            vec_d_quality_data[g], 
                            out_quality_pitch,
                            vec_d_gatherdestination[g], 
                            numColumnsCompressedQualitiesInts,
                            thrust::make_constant_iterator(l),
                            vec_numGatherOnGpu[g],
                            numQualityBits,
                            streams[g]
                        );
                    }
                }
            }
        };


        auto hostGather = [&](){
            const std::size_t compressedMemoryForOneSeq = numColumnsCompressedQualitiesInts * sizeof(unsigned int);

            for(int g = 0; g < numGpus; g++){
                if(vec_numGatherOnHost[g] > 0){

                    cub::SwitchDevice sd_(callerDeviceIds[g]);
                    cudaStream_t stream = streams[g];
                    TempData* const tempData = vec_tempData[g];
                    auto* const mr = mrs[g];

                    const int numSequences = vec_numSequences[g];

                    if(numSequences == 0) continue;;

                    constexpr std::size_t memorylimitbatch = 1 << 19; // 512KB
                    const std::size_t batchsize = std::min(SDIV(memorylimitbatch, (compressedMemoryForOneSeq)), std::size_t(vec_numGatherOnHost[g]));

                    CUDACHECK(cudaEventRecord(tempData->dependencyevent, stream));
                    for(int i = 0; i < 2; i++){
                        CUDACHECK(cudaStreamWaitEvent(tempData->streams[i], tempData->dependencyevent, 0));
                    }

                    resizeWithSync(g, tempData->pinnedBuffer, 2 * batchsize * compressedMemoryForOneSeq);

                    std::array<unsigned int*, 2> hostpointers{ 
                        (unsigned int*)(tempData->pinnedBuffer.data()), 
                        (unsigned int*)(tempData->pinnedBuffer.data() + batchsize * compressedMemoryForOneSeq)
                    };
                    assert(hostpointers.size() == tempData->streams.size());

                    const int numBatches = SDIV(vec_numGatherOnHost[g], batchsize);
                    for(int b = 0; b < numBatches; b++){
                        const int bufferIndex = b % 2;

                        const int begin = b * batchsize;
                        const int end = std::min(vec_numGatherOnHost[g], (b+1) * batchsize);
                        const int sizeOfCurrentBatch = end - begin;

                        std::vector<read_number> readIds(sizeOfCurrentBatch);
                        std::iota(readIds.begin(), readIds.end(), vec_firstIndexOnHost[g] + begin);

                        CUDACHECK(cudaStreamSynchronize(tempData->streams[bufferIndex])); // protect pinned buffer

                        gatherHostQualitiesEncoded(
                            tempData,
                            readIds.data(),
                            sizeOfCurrentBatch,
                            hostpointers[bufferIndex],
                            numColumnsCompressedQualitiesInts
                        );

                        if(gatherdestinationPitchInBytes == sizeof(unsigned int) * numColumnsCompressedQualitiesInts){
                            CUDACHECK(cudaMemcpyAsync(
                                ((char*)vec_d_gatherdestination[g]) + gatherdestinationPitchInBytes * begin,
                                hostpointers[bufferIndex],
                                sizeof(unsigned int) * numColumnsCompressedQualitiesInts * sizeOfCurrentBatch,
                                H2D,
                                tempData->streams[bufferIndex]
                            ));
                        }else{
                            CUDACHECK(cudaMemcpy2DAsync(
                                ((char*)vec_d_gatherdestination[g]) + gatherdestinationPitchInBytes * (vec_numGatherOnGpu[g] + begin),
                                gatherdestinationPitchInBytes,
                                hostpointers[bufferIndex],
                                sizeof(unsigned int) * numColumnsCompressedQualitiesInts,
                                sizeof(unsigned int) * numColumnsCompressedQualitiesInts,
                                sizeOfCurrentBatch,
                                H2D,
                                tempData->streams[bufferIndex]
                            ));
                        }
                    }

                    CUDACHECK(cudaEventRecord(tempData->event, tempData->streams[0]));
                    CUDACHECK(cudaStreamWaitEvent(stream, tempData->event, 0));
                    CUDACHECK(cudaEventRecord(tempData->event, tempData->streams[1]));
                    CUDACHECK(cudaStreamWaitEvent(stream, tempData->event, 0));
                    
                    if(numQualityBits != 8){

                        const int maxLengthCompressedPitch = numColumnsCompressedQualitiesInts * sizeof(unsigned int) * 8 / numQualityBits;
                        const int maxLengthUncompressedPitch = out_quality_pitch;
                        const int l = std::min(maxLengthCompressedPitch, maxLengthUncompressedPitch);

                        callDecompressQualityScoresKernel(
                            vec_d_quality_data[g] + out_quality_pitch * vec_numGatherOnGpu[g], 
                            out_quality_pitch,
                            vec_d_gatherdestination[g] + numColumnsCompressedQualitiesInts * vec_numGatherOnGpu[g], 
                            numColumnsCompressedQualitiesInts,
                            thrust::make_constant_iterator(l),
                            vec_numGatherOnHost[g],
                            numQualityBits,
                            stream
                        );
                    }
                }
            }
        };


        gpuGather();
        hostGather();

        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd_(callerDeviceIds[g]);
            cudaStream_t stream = streams[g];
            TempData* const tempData = vec_tempData[g];
            CUDACHECK(cudaEventRecord(tempData->event, stream));
        }

        
    }

    void gatherContiguousSequences(
        ReadStorageHandle& handle,
        unsigned int* d_sequence_data,
        size_t outSequencePitchInInts,
        read_number firstIndex,
        int numSequences,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* /*mr*/
    ) const override{
        if(numSequences == 0) return;
        
        nvtx::push_range("multigpureadstorage::gatherContiguous", 4);

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

        std::size_t numGatherOnGpu = firstIndex < sequencesGpu.getNumRows() ? std::min(sequencesGpu.getNumRows() - firstIndex, std::size_t(numSequences)) : 0;
        std::size_t firstIndexOnGpu = firstIndex < sequencesGpu.getNumRows() ? firstIndex : 0;
        std::size_t numGatherOnHost = numSequences - numGatherOnGpu;
        std::size_t firstIndexOnHost = firstIndex + numGatherOnGpu;

        if(numGatherOnHost > 0){
            CUDACHECK(cudaEventRecord(tempData->dependencyevent, stream));
        }

        if(numGatherOnGpu > 0){
            nvtx::push_range("sequencesGpu.gatherContiguous", 5);

            sequencesGpu.gatherContiguous(
                tempData->handleSequences,
                d_sequence_data,
                sizeof(unsigned int) * outSequencePitchInInts,
                firstIndexOnGpu,
                numGatherOnGpu,
                stream
            );

            nvtx::pop_range();
        }

        if(numGatherOnHost > 0){
            const std::size_t sequencepitch = sizeof(unsigned int) * outSequencePitchInInts;

            constexpr std::size_t memorylimitbatch = 1 << 19; // 512KB

            const std::size_t batchsize = std::min(SDIV(memorylimitbatch, (sequencepitch)), std::size_t(numGatherOnHost));

            resizeWithSync(tempData->pinnedBuffer, 2 * batchsize * sequencepitch);

            std::array<unsigned int*, 2> hostpointers{ 
                (unsigned int*)(tempData->pinnedBuffer.data()), 
                (unsigned int*)(tempData->pinnedBuffer.data() + batchsize * sequencepitch)
            };
            assert(hostpointers.size() == tempData->streams.size());

            
            for(int i = 0; i < 2; i++){
                CUDACHECK(cudaStreamWaitEvent(tempData->streams[i], tempData->dependencyevent, 0));
            }

            const int numBatches = SDIV(numGatherOnHost, batchsize);
            for(int b = 0; b < numBatches; b++){
                const int bufferIndex = b % 2;

                const int begin = b * batchsize;
                const int end = std::min(numGatherOnHost, (b+1) * batchsize);
                const int sizeOfCurrentBatch = end - begin;

                std::vector<read_number> readIds(sizeOfCurrentBatch);
                std::iota(readIds.begin(), readIds.end(), firstIndexOnHost + begin);

                CUDACHECK(cudaStreamSynchronize(tempData->streams[bufferIndex])); // protect pinned buffer

                gatherHostSequences(
                    tempData,
                    readIds.data(),
                    sizeOfCurrentBatch,
                    hostpointers[bufferIndex],
                    outSequencePitchInInts
                );

                CUDACHECK(cudaMemcpyAsync(
                    (char*)(d_sequence_data) + sequencepitch * (numGatherOnGpu + begin),
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
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr
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

        const int numColumnsCompressedQualitiesInts = qualitiesGpu.getNumColumns();

        unsigned int* d_compressed = nullptr;
        unsigned int* d_gatherdestination;
        std::size_t gatherdestinationPitchInBytes = 0;
        if(numQualityBits == 8){
            d_gatherdestination = reinterpret_cast<unsigned int*>(d_quality_data);
            gatherdestinationPitchInBytes = out_quality_pitch;
        }else{
            // constexpr int roundUpTo = 10000;
            // const int roundedNumSequences = SDIV(numSequences, roundUpTo) * roundUpTo;
            // resizeUninitialized(
            //     tempData->d_tempStorage,
            //     sizeof(unsigned int) * roundedNumSequences * numColumnsCompressedQualitiesInts,
            //     stream
            // );
            d_compressed = reinterpret_cast<unsigned int*>(mr->allocate(numSequences * numColumnsCompressedQualitiesInts * sizeof(unsigned int), stream));
            d_gatherdestination = d_compressed;
            gatherdestinationPitchInBytes = numColumnsCompressedQualitiesInts * sizeof(unsigned int);
        }

        CUDACHECK(cudaEventRecord(tempData->dependencyevent, stream));
        for(int i = 0; i < 2; i++){
            CUDACHECK(cudaStreamWaitEvent(tempData->streams[i], tempData->dependencyevent, 0));
        }

        auto gpuGather = [&](){
            nvtx::push_range("qualitiesGpu.gather", 5);

            qualitiesGpu.gather(
                tempData->handleQualities,
                d_gatherdestination,
                gatherdestinationPitchInBytes,
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

                const int batchsize = std::min(SDIV(memorylimitbatch, (sizeof(unsigned int) * numColumnsCompressedQualitiesInts + sizeof(int))), std::size_t(numSequences));

                void* temp_allocations_host[4]{};
                void* temp_allocations_device[4]{};
                std::size_t temp_allocation_sizes[4]{};

                temp_allocation_sizes[0] = sizeof(unsigned int) * numColumnsCompressedQualitiesInts * batchsize; // gathered host data 1
                temp_allocation_sizes[1] = sizeof(unsigned int) * numColumnsCompressedQualitiesInts * batchsize; // gathered host data 2
                temp_allocation_sizes[2] = sizeof(int) * batchsize; // output positions 1
                temp_allocation_sizes[3] = sizeof(int) * batchsize; // output positions 2

                std::size_t temp_storage_bytes = 0;
                CUDACHECK(cub::AliasTemporaries(
                    nullptr,
                    temp_storage_bytes,
                    temp_allocations_device,
                    temp_allocation_sizes
                ));

                resizeWithSync(tempData->pinnedBuffer, temp_storage_bytes);

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
                std::array<int*, 2> h_outputpositionsArr{(int*)temp_allocations_host[2], (int*)temp_allocations_host[3]};

                rmm::device_uvector<unsigned int> d_gatheredData1(numColumnsCompressedQualitiesInts * batchsize, tempData->streams[0].getStream(), mr);
                rmm::device_uvector<unsigned int> d_gatheredData2(numColumnsCompressedQualitiesInts * batchsize, tempData->streams[1].getStream(), mr);

                rmm::device_uvector<int> d_outputPositions1(batchsize, tempData->streams[0].getStream(), mr);
                rmm::device_uvector<int> d_outputPositions2(batchsize, tempData->streams[1].getStream(), mr);

                std::array<unsigned int*, 2> d_hostdataArr{d_gatheredData1.data(), d_gatheredData2.data()};
                std::array<int*, 2> d_outputpositionsArr{d_outputPositions1.data(), d_outputPositions2.data()};

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

                        gatherHostQualitiesEncoded(
                            tempData,
                            hostindicesarray[bufferIndex].data(),
                            k,
                            h_hostdataArr[bufferIndex],
                            numColumnsCompressedQualitiesInts
                        );

                        CUDACHECK(cudaMemcpyAsync(
                            d_hostdataArr[bufferIndex],
                            h_hostdataArr[bufferIndex],
                            sizeof(unsigned int) * numColumnsCompressedQualitiesInts * k,
                            H2D,
                            tempData->streams[bufferIndex]
                        ));

                        const int intsToScatter = k * numColumnsCompressedQualitiesInts;

                        helpers::lambda_kernel<<<SDIV(intsToScatter, 128), 128, 0, tempData->streams[bufferIndex]>>>(
                            [
                                intsToScatter,
                                d_gatherdestination,
                                gatherdestinationPitchInBytes,
                                numColumnsCompressedQualitiesInts,
                                d_outputpositions = d_outputpositionsArr[bufferIndex],
                                d_gatheredhost = d_hostdataArr[bufferIndex]
                            ] __device__ (){
                                const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                                const int stride = blockDim.x * gridDim.x;

                                for(int i = tid; i < intsToScatter; i += stride){
                                    const int inputrow = i / numColumnsCompressedQualitiesInts;
                                    const int inputcol = i % numColumnsCompressedQualitiesInts;
                                    const int outputcol = inputcol;
                                    const int outputrow = d_outputpositions[inputrow];

                                    ((unsigned int*)(((char*)d_gatherdestination) + outputrow * gatherdestinationPitchInBytes))[outputcol] 
                                        = d_gatheredhost[inputrow * numColumnsCompressedQualitiesInts + inputcol];
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
                const std::size_t compressedMemoryForOneSeq = numColumnsCompressedQualitiesInts * sizeof(unsigned int);
                const int batchsize = std::min(SDIV(memorylimitbatch, compressedMemoryForOneSeq), std::size_t(numSequences));

                resizeWithSync(tempData->pinnedBuffer, 2 * batchsize * compressedMemoryForOneSeq);

                std::array<unsigned int*, 2> hostpointers{
                    reinterpret_cast<unsigned int*>(tempData->pinnedBuffer.data()), 
                    reinterpret_cast<unsigned int*>(tempData->pinnedBuffer.data()) + batchsize * numColumnsCompressedQualitiesInts
                };
                assert(hostpointers.size() == tempData->streams.size());

                const int numBatches = SDIV(numSequences, batchsize);
                for(int b = 0; b < numBatches; b++){
                    const int bufferIndex = b % 2;

                    const int begin = b * batchsize;
                    const int end = std::min(numSequences, (b+1) * batchsize);
                    const int sizeOfCurrentBatch = end - begin;

                    CUDACHECK(cudaStreamSynchronize(tempData->streams[bufferIndex])); // protect pinned buffer

                    gatherHostQualitiesEncoded(
                        tempData,
                        h_readIds + begin,
                        sizeOfCurrentBatch,
                        hostpointers[bufferIndex],
                        numColumnsCompressedQualitiesInts
                    );

                    if(gatherdestinationPitchInBytes == sizeof(unsigned int) * numColumnsCompressedQualitiesInts){
                        CUDACHECK(cudaMemcpyAsync(
                            ((char*)d_gatherdestination) + gatherdestinationPitchInBytes * begin,
                            hostpointers[bufferIndex],
                            sizeof(unsigned int) * numColumnsCompressedQualitiesInts * sizeOfCurrentBatch,
                            H2D,
                            tempData->streams[bufferIndex]
                        ));
                    }else{
                        CUDACHECK(cudaMemcpy2DAsync(
                            ((char*)d_gatherdestination) + gatherdestinationPitchInBytes * begin,
                            gatherdestinationPitchInBytes,
                            hostpointers[bufferIndex],
                            sizeof(unsigned int) * numColumnsCompressedQualitiesInts,
                            sizeof(unsigned int) * numColumnsCompressedQualitiesInts,
                            sizeOfCurrentBatch,
                            H2D,
                            tempData->streams[bufferIndex]
                        ));
                    }
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

        if(numQualityBits != 8){

            const int maxLengthCompressedPitch = numColumnsCompressedQualitiesInts * sizeof(unsigned int) * 8 / numQualityBits;
            const int maxLengthUncompressedPitch = out_quality_pitch;
            const int l = std::min(maxLengthCompressedPitch, maxLengthUncompressedPitch);

            callDecompressQualityScoresKernel(
                d_quality_data, 
                out_quality_pitch,
                d_compressed, 
                numColumnsCompressedQualitiesInts,
                thrust::make_constant_iterator(l),
                numSequences,
                numQualityBits,
                stream
            );

            mr->deallocate(d_compressed, numSequences * numColumnsCompressedQualitiesInts * sizeof(unsigned int), stream);
        }else{
            //all good
        }

        CUDACHECK(cudaEventRecord(tempData->event, stream));

        nvtx::pop_range();
    }


    void gatherContiguousQualities(
        ReadStorageHandle& handle,
        char* d_quality_data,
        size_t out_quality_pitch,
        read_number firstIndex,
        int numSequences,
        cudaStream_t stream,
        rmm::mr::device_memory_resource* mr
    ) const override{
        if(numSequences == 0) return;
        
        nvtx::push_range("multigpureadstorage::gatherContiguous", 4);

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

        const int numColumnsCompressedQualitiesInts = qualitiesGpu.getNumColumns();

        std::size_t numGatherOnGpu = firstIndex < qualitiesGpu.getNumRows() ? std::min(qualitiesGpu.getNumRows() - firstIndex, std::size_t(numSequences)) : 0;
        std::size_t firstIndexOnGpu = firstIndex < qualitiesGpu.getNumRows() ? firstIndex : 0;
        std::size_t numGatherOnHost = numSequences - numGatherOnGpu;
        std::size_t firstIndexOnHost = firstIndex + numGatherOnGpu;

        unsigned int* d_gatherdestination;
        std::size_t gatherdestinationPitchInBytes = 0;
        if(numQualityBits == 8){
            d_gatherdestination = reinterpret_cast<unsigned int*>(d_quality_data);
            gatherdestinationPitchInBytes = out_quality_pitch;
        }else{
            d_gatherdestination = reinterpret_cast<unsigned int*>(mr->allocate(numSequences * numColumnsCompressedQualitiesInts * sizeof(unsigned int), stream));
            gatherdestinationPitchInBytes = numColumnsCompressedQualitiesInts * sizeof(unsigned int);
        }

        if(numGatherOnHost > 0){
            CUDACHECK(cudaEventRecord(tempData->dependencyevent, stream));
        }

        if(numGatherOnGpu > 0){
            nvtx::push_range("qualitiesGpu.gatherContiguous", 5);

            qualitiesGpu.gatherContiguous(
                tempData->handleQualities,
                d_gatherdestination,
                gatherdestinationPitchInBytes,
                firstIndexOnGpu,
                numGatherOnGpu,
                stream
            );

            nvtx::pop_range();

            if(numQualityBits != 8){
                const int maxLengthCompressedPitch = numColumnsCompressedQualitiesInts * sizeof(unsigned int) * 8 / numQualityBits;
                const int maxLengthUncompressedPitch = out_quality_pitch;
                const int l = std::min(maxLengthCompressedPitch, maxLengthUncompressedPitch);

                callDecompressQualityScoresKernel(
                    d_quality_data, 
                    out_quality_pitch,
                    d_gatherdestination, 
                    numColumnsCompressedQualitiesInts,
                    thrust::make_constant_iterator(l),
                    numGatherOnGpu,
                    numQualityBits,
                    stream
                );
            }
        }

        if(numGatherOnHost > 0){
            const std::size_t compressedMemoryForOneSeq = numColumnsCompressedQualitiesInts * sizeof(unsigned int);

            constexpr std::size_t memorylimitbatch = 1 << 19; // 512KB

            const std::size_t batchsize = std::min(SDIV(memorylimitbatch, (compressedMemoryForOneSeq)), std::size_t(numGatherOnHost));

            resizeWithSync(tempData->pinnedBuffer, 2 * batchsize * compressedMemoryForOneSeq);

            std::array<unsigned int*, 2> hostpointers{ 
                (unsigned int*)(tempData->pinnedBuffer.data()), 
                (unsigned int*)(tempData->pinnedBuffer.data() + batchsize * compressedMemoryForOneSeq)
            };
            assert(hostpointers.size() == tempData->streams.size());

            
            for(int i = 0; i < 2; i++){
                CUDACHECK(cudaStreamWaitEvent(tempData->streams[i], tempData->dependencyevent, 0));
            }

            const int numBatches = SDIV(numGatherOnHost, batchsize);
            for(int b = 0; b < numBatches; b++){
                const int bufferIndex = b % 2;

                const int begin = b * batchsize;
                const int end = std::min(numGatherOnHost, (b+1) * batchsize);
                const int sizeOfCurrentBatch = end - begin;

                std::vector<read_number> readIds(sizeOfCurrentBatch);
                std::iota(readIds.begin(), readIds.end(), firstIndexOnHost + begin);

                CUDACHECK(cudaStreamSynchronize(tempData->streams[bufferIndex])); // protect pinned buffer

                gatherHostQualitiesEncoded(
                    tempData,
                    readIds.data(),
                    sizeOfCurrentBatch,
                    hostpointers[bufferIndex],
                    numColumnsCompressedQualitiesInts
                );

                if(gatherdestinationPitchInBytes == sizeof(unsigned int) * numColumnsCompressedQualitiesInts){
                    CUDACHECK(cudaMemcpyAsync(
                        ((char*)d_gatherdestination) + gatherdestinationPitchInBytes * begin,
                        hostpointers[bufferIndex],
                        sizeof(unsigned int) * numColumnsCompressedQualitiesInts * sizeOfCurrentBatch,
                        H2D,
                        tempData->streams[bufferIndex]
                    ));
                }else{
                    CUDACHECK(cudaMemcpy2DAsync(
                        ((char*)d_gatherdestination) + gatherdestinationPitchInBytes * (numGatherOnGpu + begin),
                        gatherdestinationPitchInBytes,
                        hostpointers[bufferIndex],
                        sizeof(unsigned int) * numColumnsCompressedQualitiesInts,
                        sizeof(unsigned int) * numColumnsCompressedQualitiesInts,
                        sizeOfCurrentBatch,
                        H2D,
                        tempData->streams[bufferIndex]
                    ));
                }
            }

            CUDACHECK(cudaEventRecord(tempData->event, tempData->streams[0]));
            CUDACHECK(cudaStreamWaitEvent(stream, tempData->event, 0));
            CUDACHECK(cudaEventRecord(tempData->event, tempData->streams[1]));
            CUDACHECK(cudaStreamWaitEvent(stream, tempData->event, 0));

            if(numQualityBits != 8){

                const int maxLengthCompressedPitch = numColumnsCompressedQualitiesInts * sizeof(unsigned int) * 8 / numQualityBits;
                const int maxLengthUncompressedPitch = out_quality_pitch;
                const int l = std::min(maxLengthCompressedPitch, maxLengthUncompressedPitch);

                callDecompressQualityScoresKernel(
                    d_quality_data + out_quality_pitch * numGatherOnGpu, 
                    out_quality_pitch,
                    d_gatherdestination + numColumnsCompressedQualitiesInts * numGatherOnGpu, 
                    numColumnsCompressedQualitiesInts,
                    thrust::make_constant_iterator(l),
                    numGatherOnHost,
                    numQualityBits,
                    stream
                );
            }
        }

        if(numQualityBits != 8){
            mr->deallocate(
                d_gatherdestination, 
                numSequences * numColumnsCompressedQualitiesInts * sizeof(unsigned int), 
                stream
            );
        }

        CUDACHECK(cudaEventRecord(tempData->event, stream));

        nvtx::pop_range();
    }



    void gatherSequenceLengths(
        ReadStorageHandle& /*handle*/,
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
        result.host += sizeof(unsigned int) * hostqualities.capacity();

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


    bool hasHostSequences() const noexcept override{
        return numHostSequences > 0;
    }

    bool hasHostQualities() const noexcept override{
        return canUseQualityScores() && numHostQualities > 0;
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
        TempData* /*tempData*/,
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
                hostSequencePitchBytes / sizeof(unsigned int),
                hostSequencePitchBytes,
                indexGenerator,
                numSequences,
                outputarray,
                outputPitchInInts * sizeof(unsigned int)
            );
        }
    }

    void gatherHostQualitiesEncoded(
        TempData* /*tempData*/,
        const read_number* readIds,
        int numSequences,
        unsigned int* outputarray,
        std::size_t outputPitchInInts
    ) const {
        if(numSequences == 0) return;
        if(!isStandalone()){

            cpuReadStorage->gatherEncodedQualities(
                outputarray,
                outputPitchInInts,
                readIds,
                numSequences
            );
        }else{

            //convert readIds into local indices for host partition
            auto indexGenerator = [&](auto i){
                return readIds[i] - qualitiesGpu.getNumRows();
            };

            gatherHostStandaloneImpl(
                (unsigned int*)hostqualities.data(),
                hostQualityPitchBytes / sizeof(unsigned int),
                hostQualityPitchBytes,
                indexGenerator,
                numSequences,
                outputarray,
                sizeof(unsigned int) * outputPitchInInts
            );
        }
    }

    TempData* getTempDataFromHandle(const ReadStorageHandle& handle) const{
        std::shared_lock<SharedMutex> lock(sharedmutex);

        assert(handle.getId() < int(tempdataVector.size()));

        return tempdataVector[handle.getId()].get();
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
    
    bool directPeerAccess{};
    bool pairedEnd{};
    bool useQualityScores{};
    int sequenceLengthLowerBound{};
    int sequenceLengthUpperBound{};
    int numQualityBits = 8;
    read_number numberOfReads{};
    std::int64_t numberOfAmbiguousReads{};

    std::size_t numHostSequences{};
    std::size_t numHostQualities{};
    std::size_t hostSequencePitchBytes{};
    std::size_t hostQualityPitchBytes{};
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
