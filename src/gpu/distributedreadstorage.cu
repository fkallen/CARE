#if 1


#include <gpu/distributedreadstorage.hpp>
#include <gpu/distributedarray.hpp>
#include <memorymanagement.hpp>

#include <config.hpp>
#include <sequencehelpers.hpp>
#include <readlibraryio.hpp>
#include <threadpool.hpp>
#include <util.hpp>

#include <readstorageconstruction.hpp>

#include <atomic>
#include <fstream>
#include <omp.h>
#include <algorithm>
#include <iterator>
#include <numeric>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#ifdef __NVCC__

namespace care{
namespace gpu{

DistributedReadStorage::DistributedReadStorage(const std::vector<int>& deviceIds_, read_number nReads, bool b, 
                    int minimum_sequence_length, int maximum_sequence_length){

    init(deviceIds_, nReads, b, minimum_sequence_length, maximum_sequence_length);
}

DistributedReadStorage::DistributedReadStorage(const std::vector<int>& deviceIds_, const std::vector<SequenceFileProperties>& sequenceFileProperties, bool qualityScores){

    init(deviceIds_, sequenceFileProperties, qualityScores);
}

void DistributedReadStorage::init(const std::vector<int>& deviceIds_, read_number nReads, bool b, 
                    int minimum_sequence_length, int maximum_sequence_length){
    assert(minimum_sequence_length <= maximum_sequence_length);

    constexpr DistributedArrayLayout layout = DistributedArrayLayout::GPUBlock; //GPUEqual; //GPUBlock

    int oldId; cudaGetDevice(&oldId); CUERR;

    isReadOnly = false;
    deviceIds = deviceIds_;
    numberOfInsertedReads = 0;
    maximumNumberOfReads = nReads;
    sequenceLengthLowerBound = minimum_sequence_length;
    sequenceLengthUpperBound = maximum_sequence_length;
    useQualityScores = b;

    int numGpus = deviceIds.size();

    for(auto& pair : bitArraysUndeterminedBase){
        cudaSetDevice(pair.first); CUERR;
        destroyGpuBitArray(pair.second);
    }
    bitArraysUndeterminedBase.clear();


    constexpr size_t headRoom = gpuReadStorageHeadroomPerGPU; 
    //constexpr size_t headRoom = (size_t(1) << 30) * 11; 

    if(getMaximumNumberOfReads() > 0 && sequenceLengthUpperBound > 0 && sequenceLengthLowerBound >= 0){
        

        lengthStorage = std::move(LengthStore_t(sequenceLengthLowerBound, sequenceLengthUpperBound, getMaximumNumberOfReads()));


        for(int deviceId : deviceIds){
            cudaSetDevice(deviceId); CUERR;
            bitArraysUndeterminedBase[deviceId] = makeGpuBitArray<read_number>(getMaximumNumberOfReads());
        }
        

        std::vector<size_t> freeMemPerGpu(numGpus, 0);
        std::vector<size_t> totalMemPerGpu(numGpus, 0);
        std::vector<size_t> maximumUsableBytesPerGpu(numGpus, 0);

        auto getGpuMemoryInfo = [&](){
            for(int gpu = 0; gpu < numGpus; gpu++){
                cudaSetDevice(deviceIds[gpu]); CUERR;

                cudaMemGetInfo(&freeMemPerGpu[gpu], &totalMemPerGpu[gpu]); CUERR;
            }
        };

        auto updateMemoryLimits = [&](){
            getGpuMemoryInfo();

            for(int gpu = 0; gpu < numGpus; gpu++){
                const size_t usableMem = freeMemPerGpu[gpu] > headRoom ? freeMemPerGpu[gpu] - headRoom : 0;
                maximumUsableBytesPerGpu[gpu] = usableMem;
            }
            // std::cerr << "Usable memory per gpu : ";
            // std::copy(maximumUsableBytesPerGpu.begin(), maximumUsableBytesPerGpu.end(), std::ostream_iterator<size_t>(std::cerr, " "));
            // std::cerr << "\n";
        };

        


        const int intsPerSequence = SequenceHelpers::getEncodedNumInts2Bit(sequenceLengthUpperBound);

        updateMemoryLimits();
        distributedSequenceData = std::move(DistributedArray<unsigned int, read_number>(deviceIds, 
                                                                                        maximumUsableBytesPerGpu, 
                                                                                        layout,
                                                                                        getMaximumNumberOfReads(), 
                                                                                        intsPerSequence));

        if(useQualityScores){
            updateMemoryLimits();
            distributedQualities = std::move(DistributedArray<char, read_number>(deviceIds, 
                                                                                maximumUsableBytesPerGpu, 
                                                                                layout,
                                                                                getMaximumNumberOfReads(), 
                                                                                sequenceLengthUpperBound));
        }

        getGpuMemoryInfo();

        // std::cerr << "Free memory per gpu after construction of distributed readstorage: ";
        // std::copy(freeMemPerGpu.begin(), freeMemPerGpu.end(), std::ostream_iterator<size_t>(std::cerr, " "));
        // std::cerr << "\n";

    }
    cudaSetDevice(oldId); CUERR;
}



void DistributedReadStorage::init(const std::vector<int>& deviceIds_, const std::vector<SequenceFileProperties>& sequenceFileProperties, bool qualityScores){
    

    constexpr DistributedArrayLayout layout = DistributedArrayLayout::GPUEqual; //GPUEqual; //GPUBlock

    assert(sequenceFileProperties.size() > 0);

    int oldId; cudaGetDevice(&oldId); CUERR;

    isReadOnly = false;
    deviceIds = deviceIds_;
    numberOfInsertedReads = 0;
    maximumNumberOfReads = std::accumulate(
        sequenceFileProperties.begin(), 
        sequenceFileProperties.end(), 
        std::uint64_t{0}, 
        [](const auto acc, const auto& e){return acc + e.nReads;}
    );
    sequenceLengthLowerBound = std::min_element(
        sequenceFileProperties.begin(), 
        sequenceFileProperties.end(), 
        [](const auto& l, const auto& r){return l.minSequenceLength < r.minSequenceLength;}
    )->minSequenceLength;
    sequenceLengthUpperBound = std::max_element(
        sequenceFileProperties.begin(), 
        sequenceFileProperties.end(), 
        [](const auto& l, const auto& r){return l.maxSequenceLength < r.maxSequenceLength;}
    )->maxSequenceLength;
    useQualityScores = qualityScores;

    assert(sequenceLengthLowerBound <= sequenceLengthUpperBound);

    int numGpus = deviceIds.size();

    for(auto& pair : bitArraysUndeterminedBase){
        cudaSetDevice(pair.first); CUERR;
        destroyGpuBitArray(pair.second);
    }
    bitArraysUndeterminedBase.clear();


    constexpr size_t headRoom = gpuReadStorageHeadroomPerGPU; 
    //constexpr size_t headRoom = (size_t(1) << 30) * 11; 

    if(getMaximumNumberOfReads() > 0 && sequenceLengthUpperBound > 0 && sequenceLengthLowerBound >= 0){
        

        lengthStorage = std::move(LengthStore_t(sequenceLengthLowerBound, sequenceLengthUpperBound, getMaximumNumberOfReads()));


        for(int deviceId : deviceIds){
            cudaSetDevice(deviceId); CUERR;
            bitArraysUndeterminedBase[deviceId] = makeGpuBitArray<read_number>(getMaximumNumberOfReads());
        }
        

        std::vector<size_t> freeMemPerGpu(numGpus, 0);
        std::vector<size_t> totalMemPerGpu(numGpus, 0);
        std::vector<size_t> maximumUsableBytesPerGpu(numGpus, 0);

        auto getGpuMemoryInfo = [&](){
            for(int gpu = 0; gpu < numGpus; gpu++){
                cudaSetDevice(deviceIds[gpu]); CUERR;

                cudaMemGetInfo(&freeMemPerGpu[gpu], &totalMemPerGpu[gpu]); CUERR;
            }
        };

        auto updateMemoryLimits = [&](){
            getGpuMemoryInfo();

            for(int gpu = 0; gpu < numGpus; gpu++){
                const size_t usableMem = freeMemPerGpu[gpu] > headRoom ? freeMemPerGpu[gpu] - headRoom : 0;
                maximumUsableBytesPerGpu[gpu] = usableMem;
            }
            std::cerr << "Usable memory per gpu : ";
            std::copy(maximumUsableBytesPerGpu.begin(), maximumUsableBytesPerGpu.end(), std::ostream_iterator<size_t>(std::cerr, " "));
            std::cerr << "\n";
        };

        


        const int intsPerSequence = SequenceHelpers::getEncodedNumInts2Bit(sequenceLengthUpperBound);

        updateMemoryLimits();
        distributedSequenceData = std::move(DistributedArray<unsigned int, read_number>(deviceIds, 
                                                                                        maximumUsableBytesPerGpu, 
                                                                                        layout,
                                                                                        getMaximumNumberOfReads(), 
                                                                                        intsPerSequence));

        if(useQualityScores){
            updateMemoryLimits();
            distributedQualities = std::move(DistributedArray<char, read_number>(deviceIds, 
                                                                                maximumUsableBytesPerGpu, 
                                                                                layout,
                                                                                getMaximumNumberOfReads(), 
                                                                                sequenceLengthUpperBound));
        }

        getGpuMemoryInfo();

        std::cerr << "Free memory per gpu after construction of distributed readstorage: ";
        std::copy(freeMemPerGpu.begin(), freeMemPerGpu.end(), std::ostream_iterator<size_t>(std::cerr, " "));
        std::cerr << "\n";

    }
    cudaSetDevice(oldId); CUERR;
}






DistributedReadStorage::DistributedReadStorage(DistributedReadStorage&& other){
    *this = std::move(other);
}

DistributedReadStorage& DistributedReadStorage::operator=(DistributedReadStorage&& rhs){
    isReadOnly = rhs.isReadOnly;
    deviceIds = std::move(rhs.deviceIds);
    read_number tmp = rhs.numberOfInsertedReads;
    numberOfInsertedReads = tmp;
    maximumNumberOfReads = std::move(rhs.maximumNumberOfReads);
    sequenceLengthLowerBound = std::move(rhs.sequenceLengthLowerBound);
    sequenceLengthUpperBound = std::move(rhs.sequenceLengthUpperBound);
    useQualityScores = std::move(rhs.useQualityScores);
    readIdsOfReadsWithUndeterminedBase = std::move(rhs.readIdsOfReadsWithUndeterminedBase);
    bitArraysUndeterminedBase = std::move(rhs.bitArraysUndeterminedBase);
    lengthStorage = std::move(rhs.lengthStorage);
    gpulengthStorage = std::move(rhs.gpulengthStorage);
    distributedSequenceData = std::move(rhs.distributedSequenceData);
    distributedQualities = std::move(rhs.distributedQualities);
    statistics = std::move(rhs.statistics);
    hasMoved = rhs.hasMoved;
    rhs.hasMoved = true;

    return *this;
}


void DistributedReadStorage::construct(
    std::vector<std::string> inputfiles,
    bool useQualityScores,
    read_number expectedNumberOfReads,
    int expectedMinimumReadLength,
    int expectedMaximumReadLength,
    int threads,
    bool showProgress
){

    auto makeInserterFunc = [&](){return makeReadInserter();};
    auto makeReadContainsNFunc = [this](){
        return [=](read_number readId, bool contains){
            this->setReadContainsN(readId, contains);
        };
    };

    constructReadStorageFromFiles(
        inputfiles,
        useQualityScores,
        expectedNumberOfReads,
        expectedMinimumReadLength,
        expectedMaximumReadLength,
        threads,
        showProgress,
        makeInserterFunc,
        makeReadContainsNFunc
    );

    constructionIsComplete();
}

void DistributedReadStorage::constructPaired(
    std::vector<std::string> inputfiles,
    bool useQualityScores,
    read_number expectedNumberOfReads,
    int expectedMinimumReadLength,
    int expectedMaximumReadLength,
    int threads,
    bool showProgress
){

    auto makeInserterFunc = [&](){return makeReadInserter();};
    auto makeReadContainsNFunc = [this](){
        return [&](read_number readId, bool contains){
            this->setReadContainsN(readId, contains);
        };
    };

    constructReadStorageFromPairedEndFiles(
        inputfiles,
        useQualityScores,
        expectedNumberOfReads,
        expectedMinimumReadLength,
        expectedMaximumReadLength,
        threads,
        showProgress,
        makeInserterFunc,
        makeReadContainsNFunc
    );

    constructionIsComplete();        
}

MemoryUsage DistributedReadStorage::getMemoryInfo() const{


    MemoryUsage info;

    auto handlearray = [&](const auto& array){
        const auto partitions = array.getPartitions();

        for(int location = 0; location < distributedSequenceData.numLocations; location++){
            
            size_t bytes = partitions[location] * array.sizeOfElement;
            //std::cerr << "location " << location << " " << bytes << "\n";
            if(location == array.hostLocation){
                info.host += bytes;
            }else{
                const int deviceId = deviceIds[location];
                info.device[deviceId] += bytes;
            }
        }
    };

    handlearray(distributedSequenceData);

    if(canUseQualityScores()){
        handlearray(distributedQualities);
    }

    auto lengthstorageMem = gpulengthStorage.getMemoryInfo();

    info.host += lengthstorageMem.host;
    for(const auto& pair : lengthstorageMem.device){
        info.device[pair.first] += pair.second;
    }

    for(int deviceId : deviceIds){
        info.device[deviceId] += bitArraysUndeterminedBase.at(deviceId).numAllocatedBytes;
    }
       

    return info;
}

MemoryUsage DistributedReadStorage::getMemoryInfoOfGatherHandleSequences(const GatherHandleSequences& handle) const{
    return distributedSequenceData.getMemoryInfoOfHandle(handle);
}

MemoryUsage DistributedReadStorage::getMemoryInfoOfGatherHandleQualities(const GatherHandleQualities& handle) const{
    return distributedQualities.getMemoryInfoOfHandle(handle);
}


DistributedReadStorage::Statistics DistributedReadStorage::getStatistics() const{
    return statistics;
}

void DistributedReadStorage::destroy(){
    int oldId = 0;
    cudaGetDevice(&oldId); CUERR;

    numberOfInsertedReads = 0;
    maximumNumberOfReads = 0;
    sequenceLengthUpperBound = 0;
    std::vector<size_t> fractions(deviceIds.size(), 0);
    lengthStorage = std::move(LengthStore_t{});
    for(auto& pair : bitArraysUndeterminedBase){
        cudaSetDevice(pair.first); CUERR;
        destroyGpuBitArray(pair.second);
    }
    bitArraysUndeterminedBase.clear();
    gpulengthStorage = std::move(GPULengthStore_t{});
    distributedSequenceData = std::move(DistributedArray<unsigned int, read_number>(deviceIds, fractions, DistributedArrayLayout::GPUBlock, 0, 0));
    distributedQualities = std::move(DistributedArray<char, read_number>(deviceIds, fractions, DistributedArrayLayout::GPUBlock, 0, 0));
    statistics = Statistics{};

    cudaSetDevice(oldId); CUERR;
}

read_number DistributedReadStorage::getNumberOfReads() const{
    return numberOfInsertedReads;
}

read_number DistributedReadStorage::getMaximumNumberOfReads() const{
    return maximumNumberOfReads;
}

bool DistributedReadStorage::canUseQualityScores() const{
    return useQualityScores;
}

int DistributedReadStorage::getSequenceLengthLowerBound() const{
    return sequenceLengthLowerBound;
}

int DistributedReadStorage::getSequenceLengthUpperBound() const{
    return sequenceLengthUpperBound;
}

std::vector<int> DistributedReadStorage::getDeviceIds() const{
    return deviceIds;
}




void DistributedReadStorage::setReads(
    ReadInserterHandle& handle,
    ThreadPool* threadPool, 
    const read_number* indices, 
    const Read* reads, 
    int numReads
){
    if(numReads == 0) return;
    
    assert(!isReadOnly);

    //TIMERSTARTCPU(internalinit);
    auto lengthInRange = [&](Length_t length){
        return getSequenceLengthLowerBound() <= length && length <= getSequenceLengthUpperBound();
    };
    assert(numReads > 0);
    assert(std::all_of(indices, indices + numReads, [&](auto i){ return i < getMaximumNumberOfReads();}));
    assert(std::all_of(reads, reads + numReads, [&](const auto& r){ return lengthInRange(Length_t(r.sequence.length()));}));
    
    if(canUseQualityScores()){
        assert(std::all_of(reads, reads + numReads, [&](const auto& r){ return r.sequence.length() == r.quality.length();}));
    }

    auto minmax = std::minmax_element(reads, reads + numReads, [](const auto& r1, const auto& r2){
        return r1.sequence.length() < r2.sequence.length();
    });

    statistics.minimumSequenceLength = std::min(statistics.minimumSequenceLength, int(minmax.first->sequence.length()));
    statistics.maximumSequenceLength = std::max(statistics.maximumSequenceLength, int(minmax.second->sequence.length()));

    read_number maxIndex = *std::max_element(indices, indices + numReads);

    read_number prev_value = numberOfInsertedReads;
    while(prev_value < maxIndex+1 && !numberOfInsertedReads.compare_exchange_weak(prev_value, maxIndex+1)){
        ;
    }      

    const std::size_t encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(getSequenceLengthUpperBound());
    const std::size_t qualityPitch = getSequenceLengthUpperBound();

    const std::size_t decodedSequencePitch = SDIV(statistics.maximumSequenceLength, 128) * 128;

    constexpr std::size_t buffersize = std::size_t{1} << 20;

    const int readsPerIteration = buffersize / decodedSequencePitch;
    const int iterations = SDIV(numReads, readsPerIteration);

    handle.h_decodedSequences.resize(buffersize * 2);
    handle.d_decodedSequences.resize(buffersize * 2);
    handle.h_encodedSequences.resize(encodedSequencePitchInInts * readsPerIteration * 2);
    handle.d_encodedSequences.resize(encodedSequencePitchInInts * readsPerIteration * 2);
    handle.h_lengths.resize(readsPerIteration * 2);
    handle.d_lengths.resize(readsPerIteration * 2);
    

    char* h_decodedSequences = handle.h_decodedSequences.get();
    char* d_decodedSequences = handle.d_decodedSequences.get();
    unsigned int* h_encodedSequences = handle.h_encodedSequences.get();
    unsigned int* d_encodedSequences = handle.d_encodedSequences.get();
    int* h_lengths = handle.h_lengths.get();
    int* d_lengths = handle.d_lengths.get();

    std::array<char*, 2> h_decodedSequences_arr{h_decodedSequences, h_decodedSequences + buffersize};
    std::array<char*, 2> d_decodedSequences_arr{d_decodedSequences, d_decodedSequences + buffersize};
    std::array<unsigned int*, 2> h_encodedSequences_arr{h_encodedSequences, h_encodedSequences + encodedSequencePitchInInts * readsPerIteration};
    std::array<unsigned int*, 2> d_encodedSequences_arr{d_encodedSequences, d_encodedSequences + encodedSequencePitchInInts * readsPerIteration};
    std::array<int*, 2> h_lengths_arr{h_lengths, h_lengths + readsPerIteration};
    std::array<int*, 2> d_lengths_arr{d_lengths, d_lengths + readsPerIteration};

    std::array<cudaStream_t, 2> streams{handle.stream1, handle.stream2};


    std::vector<char>& sequenceData = handle.sequenceData;
    std::vector<Length_t>& sequenceLengths = handle.sequenceLengths;
    std::vector<char>& qualityData = handle.qualityData;

    sequenceData.resize(statistics.maximumSequenceLength * numReads, 0);
    sequenceLengths.resize(numReads, 0);

    if(canUseQualityScores()){
        qualityData.resize(getSequenceLengthUpperBound() * numReads, 0);
    }

    ThreadPool::ParallelForHandle pforHandle;

    int currentbuf = 0;

    for(int iteration = 0; iteration < iterations; iteration++){

        const int begin = iteration * readsPerIteration;
        const int end = std::min((iteration+1) * readsPerIteration, numReads);
        const int chunksize = end - begin;
        
        for(int i = 0; i < chunksize; i++){
            const auto& r = reads[begin + i];

            std::copy(
                r.sequence.begin(), 
                r.sequence.end(), 
                h_decodedSequences_arr[currentbuf] + i * decodedSequencePitch
            );

            h_lengths_arr[currentbuf][i] = r.sequence.size();
        }

        cudaMemcpyAsync(
            d_decodedSequences_arr[currentbuf],
            h_decodedSequences_arr[currentbuf],
            sizeof(char) * chunksize * decodedSequencePitch,
            H2D,
            streams[currentbuf]
        ); CUERR;

        cudaMemcpyAsync(
            d_lengths_arr[currentbuf],
            h_lengths_arr[currentbuf],
            sizeof(int) * chunksize,
            H2D,
            streams[currentbuf]
        ); CUERR;

        // //kernel for encoding

        helpers::lambda_kernel<<<640, 128, 0, streams[currentbuf]>>>(
            [
                =, 
                decodedSequences = d_decodedSequences_arr[currentbuf],
                encodedSequences = d_encodedSequences_arr[currentbuf],
                lengths = d_lengths_arr[currentbuf]
            ] __device__ (){

                //use one thread block tile per sequence to encode the sequence into 2bit format
                constexpr int tilesize = 8;

                assert(decodedSequencePitch % sizeof(unsigned int) == 0);

                auto tile = cg::tiled_partition<tilesize>(cg::this_thread_block());
                const int numTiles = blockDim.x * gridDim.x / tilesize;
                const int tileId = (threadIdx.x + blockIdx.x * blockDim.x) / tilesize;

                auto encode = [](unsigned int& dest, char base){
                    dest = (dest << 2) | SequenceHelpers::encodeBase(base);
                };

                constexpr int basesPerInt = SequenceHelpers::basesPerInt2Bit();

                for(int sequenceIndex = tileId; sequenceIndex < chunksize; sequenceIndex += numTiles){
                    const char* const sequenceinput = decodedSequences + sequenceIndex * decodedSequencePitch;
                    unsigned int* const output = encodedSequences + sequenceIndex * encodedSequencePitchInInts;

                    const int length = lengths[sequenceIndex];
                    const int numRequiredInts = SequenceHelpers::getEncodedNumInts2Bit(length);

                    for(int outputIntIndex = tile.thread_rank(); outputIntIndex < numRequiredInts; outputIntIndex += tile.size()){

                        const char4* const myInput = ((const char4*)sequenceinput) + 4 * outputIntIndex;

                        unsigned int outputdata = 0;

                        #pragma unroll
                        for(int i = 0; i < 4; i++){
                            const char4 data = myInput[i];

                            if(outputIntIndex * basesPerInt + i * 4 + 0 < length){
                                encode(outputdata, data.x);
                            }
                            if(outputIntIndex * basesPerInt + i * 4 + 1 < length){
                                encode(outputdata, data.y);
                            }
                            if(outputIntIndex * basesPerInt + i * 4 + 2 < length){
                                encode(outputdata, data.z);
                            }
                            if(outputIntIndex * basesPerInt + i * 4 + 3 < length){
                                encode(outputdata, data.w);
                            }
                        } 

                        if(outputIntIndex == numRequiredInts - 1){
                            //pack bits of last integer into higher order bits
                            int leftoverbits = 2 * (numRequiredInts * SequenceHelpers::basesPerInt2Bit() - length);
                            if(leftoverbits > 0){
                                outputdata <<= leftoverbits;
                            }
                        }

                        output[outputIntIndex] = outputdata;
                    }

                }
            }
        ); CUERR;

        cudaMemcpyAsync(
            h_encodedSequences_arr[currentbuf],
            d_encodedSequences_arr[currentbuf],
            sizeof(unsigned int) * chunksize * encodedSequencePitchInInts,
            D2H,
            streams[currentbuf]
        ); CUERR;

        if(iteration > 0){
            //process results of previous iteration
            const int otherbuf = 1 - currentbuf;

            const int otherbegin = (iteration-1) * readsPerIteration;
            const int otherend = iteration * readsPerIteration;
            const int otherchunksize = otherend - otherbegin;

            cudaStreamSynchronize(streams[otherbuf]);

            for(int i = 0; i < otherchunksize; i++){
                const auto& r = reads[otherbegin + i];
    
                std::copy(
                    h_encodedSequences_arr[otherbuf] + i * encodedSequencePitchInInts,
                    h_encodedSequences_arr[otherbuf] + (i+1) * encodedSequencePitchInInts,
                    ((unsigned int*)sequenceData.data()) + (otherbegin + i)* encodedSequencePitchInInts
                );
    
                sequenceLengths[otherbegin + i] = h_lengths_arr[otherbuf][i];

                if(canUseQualityScores()){
                    std::copy(r.quality.begin(), r.quality.end(), qualityData.begin() + (otherbegin + i) * qualityPitch);
                }
            }
        } 

        if(iteration == iterations - 1){
            //process results of current (final) iteration

            cudaStreamSynchronize(streams[currentbuf]);

            for(int i = 0; i < chunksize; i++){

                const auto& r = reads[begin + i];
    
                std::copy(
                    h_encodedSequences_arr[currentbuf] + i * encodedSequencePitchInInts,
                    h_encodedSequences_arr[currentbuf] + (i+1) * encodedSequencePitchInInts,
                    ((unsigned int*)sequenceData.data()) + (begin + i)* encodedSequencePitchInInts
                );
    
                sequenceLengths[begin + i] = h_lengths_arr[currentbuf][i];

                if(canUseQualityScores()){
                    std::copy(r.quality.begin(), r.quality.end(), qualityData.begin() + (begin + i) * qualityPitch);
                }
            }

        } 

        currentbuf = 1 - currentbuf;

    }

    setSequences(indices, sequenceData.data(), numReads);
    setSequenceLengths(indices, sequenceLengths.data(), numReads);
    if(canUseQualityScores()){
        setQualities(indices, qualityData.data(), numReads);
    }

}









void DistributedReadStorage::setReadContainsN(read_number readId, bool contains){
    assert(!isReadOnly);

    std::lock_guard<std::mutex> l(mutexUndeterminedBaseReads);

    auto pos = std::lower_bound(readIdsOfReadsWithUndeterminedBase.begin(),
                                        readIdsOfReadsWithUndeterminedBase.end(),
                                        readId);

    if(contains){
        //if readId is not already in the vector, insert it
        if((pos == readIdsOfReadsWithUndeterminedBase.end()) || (pos != readIdsOfReadsWithUndeterminedBase.end() && *pos != readId)){
            readIdsOfReadsWithUndeterminedBase.insert(pos, readId);
        }
    }else{
        if(pos != readIdsOfReadsWithUndeterminedBase.end() && *pos == readId){
            //remove mark
            readIdsOfReadsWithUndeterminedBase.erase(pos);
        }
    }
}

bool DistributedReadStorage::readContainsN(read_number readId) const{

    auto pos = std::lower_bound(readIdsOfReadsWithUndeterminedBase.begin(),
                                        readIdsOfReadsWithUndeterminedBase.end(),
                                        readId);
    bool b2 = readIdsOfReadsWithUndeterminedBase.end() != pos && *pos == readId;

    return b2;
}

std::int64_t DistributedReadStorage::getNumberOfReadsWithN() const{
    return readIdsOfReadsWithUndeterminedBase.size();
}

void DistributedReadStorage::setReadsContainN_async(
    int deviceId,
    bool* d_values, 
    const read_number* d_positions, 
    int nPositions,
    cudaStream_t stream) const{

    if(nPositions > 0){

        int oldId = 0;
        cudaGetDevice(&oldId); CUERR;
        cudaSetDevice(deviceId); CUERR;

        dim3 block = 256;
        dim3 grid = SDIV(nPositions, block.x);

        setBitarray<<<grid, block, 0, stream>>>(
            bitArraysUndeterminedBase.at(deviceId), 
            d_values, 
            d_positions, 
            nPositions
        ); CUERR;

        cudaSetDevice(oldId); CUERR;

    }
}

void DistributedReadStorage::readsContainN_async(
        int deviceId,
        bool* d_result, 
        const read_number* d_positions, 
        int nPositions, 
        cudaStream_t stream) const{

    if(nPositions > 0){

        int oldId = 0;
        cudaGetDevice(&oldId); CUERR;
        cudaSetDevice(deviceId); CUERR;

        dim3 block = 256;
        dim3 grid = SDIV(nPositions, block.x);

        readBitarray<<<grid, block, 0, stream>>>(
            d_result, 
            bitArraysUndeterminedBase.at(deviceId), 
            d_positions, 
            nPositions
        ); CUERR;

        cudaSetDevice(oldId); CUERR;

    }
}

void DistributedReadStorage::readsContainN_async(
        int deviceId,
        bool* d_result, 
        const read_number* d_positions, 
        const int* d_nPositions,
        int nPositionsUpperBound, 
        cudaStream_t stream) const{

    if(nPositionsUpperBound > 0){

        int oldId = 0;
        cudaGetDevice(&oldId); CUERR;
        cudaSetDevice(deviceId); CUERR;

        dim3 block = 256;
        dim3 grid = SDIV(nPositionsUpperBound, block.x);

        readBitarray<<<grid, block, 0, stream>>>(
            d_result, 
            bitArraysUndeterminedBase.at(deviceId), 
            d_positions, 
            d_nPositions
        ); CUERR;

        cudaSetDevice(oldId); CUERR;

    }
}


void DistributedReadStorage::constructionIsComplete(){
    gpulengthStorage = std::move(GPULengthStore_t{std::move(lengthStorage), deviceIds});
    setGpuBitArraysFromVector();
    isReadOnly = true;
}

void DistributedReadStorage::allowModifications(){
    isReadOnly = false;
    gpulengthStorage.extractCpuLengthStorage(lengthStorage);
    gpulengthStorage = std::move(GPULengthStore_t{});
}

void DistributedReadStorage::setSequences(read_number firstIndex, read_number lastIndex_excl, const char* data){
    assert(!isReadOnly);

    distributedSequenceData.setSafe(firstIndex, lastIndex_excl, reinterpret_cast<const unsigned int*>(data));
}

void DistributedReadStorage::setSequences(const read_number* indices, const char* data, int numReads){
    assert(!isReadOnly);

    distributedSequenceData.setSafe(indices, reinterpret_cast<const unsigned int*>(data), numReads);
}

void DistributedReadStorage::setSequenceLengths(read_number firstIndex, read_number lastIndex_excl, const Length_t* data){
    assert(!isReadOnly);

    for(read_number i = firstIndex; i < lastIndex_excl; i++){
        lengthStorage.setLength(i, data[i - firstIndex]);
    }
}

void DistributedReadStorage::setSequenceLengths(const read_number* indices, const Length_t* data, int numReads){
    assert(!isReadOnly);

    for(int i = 0; i < numReads; i++){
        lengthStorage.setLength(indices[i], data[i]);
    }
}

void DistributedReadStorage::setQualities(read_number firstIndex, read_number lastIndex_excl, const char* data){
    assert(!isReadOnly);

    distributedQualities.setSafe(firstIndex, lastIndex_excl, data);
}

void DistributedReadStorage::setQualities(const read_number* indices, const char* data, int numReads){
    assert(!isReadOnly);
    
    distributedQualities.setSafe(indices, data, numReads);
}

DistributedReadStorage::GatherHandleSequences DistributedReadStorage::makeGatherHandleSequences() const{
    return distributedSequenceData.makeGatherHandle();
}

DistributedReadStorage::GatherHandleQualities DistributedReadStorage::makeGatherHandleQualities() const{
    return distributedQualities.makeGatherHandle();
}

void DistributedReadStorage::gatherSequenceDataToGpuBufferAsync(
                            ThreadPool* threadPool,
                            const DistributedReadStorage::GatherHandleSequences& handle,
                            unsigned int* d_sequence_data,
                            size_t outSequencePitchInInts,
                            const read_number* h_readIds,
                            const read_number* d_readIds,
                            int nReadIds,
                            int deviceId,
                            cudaStream_t stream) const{

    if(threadPool != nullptr){

        ParallelForLoopExecutor forLoop(threadPool, &(handle->pforHandle));

        distributedSequenceData.gatherElementsInGpuMemAsync(forLoop,
            handle,
            h_readIds,
            d_readIds,
            nReadIds,
            deviceId,
            d_sequence_data,
            outSequencePitchInInts * sizeof(unsigned int),
            stream);
    }else{
        SequentialForLoopExecutor forLoop;

        distributedSequenceData.gatherElementsInGpuMemAsync(forLoop,
            handle,
            h_readIds,
            d_readIds,
            nReadIds,
            deviceId,
            d_sequence_data,
            outSequencePitchInInts * sizeof(unsigned int),
            stream);
    }
    

}


void DistributedReadStorage::gatherQualitiesToGpuBufferAsync(
                            ThreadPool* threadPool,
                            const GatherHandleQualities& handle,
                            char* d_quality_data,
                            size_t out_quality_pitch,
                            const read_number* h_readIds,
                            const read_number* d_readIds,
                            int nReadIds,
                            int deviceId,
                            cudaStream_t stream) const{

    if(threadPool != nullptr){

        ParallelForLoopExecutor forLoop(threadPool, &(handle->pforHandle));

        distributedQualities.gatherElementsInGpuMemAsync(forLoop, 
                                                            handle,
                                                            h_readIds,
                                                            d_readIds,
                                                            nReadIds,
                                                            deviceId,
                                                            d_quality_data,
                                                            out_quality_pitch,
                                                            stream);
    }else{
        SequentialForLoopExecutor forLoop;

        distributedQualities.gatherElementsInGpuMemAsync(forLoop, 
            handle,
            h_readIds,
            d_readIds,
            nReadIds,
            deviceId,
            d_quality_data,
            out_quality_pitch,
            stream);
    }

}



void DistributedReadStorage::gatherSequenceLengthsToGpuBufferAsync(
                            int* d_lengths,
                            int deviceId,
                            const read_number* d_readIds,
                            int nReadIds,    
                            cudaStream_t stream) const{

    assert(isReadOnly);

    if(nReadIds == 0) return;

    gpulengthStorage.gatherLengthsOnDeviceAsync(d_lengths, 
                                                deviceId,
                                                d_readIds, 
                                                nReadIds,
                                                stream);

}

void DistributedReadStorage::gatherSequenceLengthsToHostBuffer(
                            int* lengths,
                            const read_number* readIds,
                            int nReadIds) const{

    assert(isReadOnly);

    gpulengthStorage.gatherLengthsOnHost(lengths, readIds, nReadIds);

}


void DistributedReadStorage::saveToFile(const std::string& filename) const{
    std::ofstream stream(filename, std::ios::binary);

    read_number inserted = getNumberOfReads();

    stream.write(reinterpret_cast<const char*>(&inserted), sizeof(read_number));
    stream.write(reinterpret_cast<const char*>(&sequenceLengthLowerBound), sizeof(int));
    stream.write(reinterpret_cast<const char*>(&sequenceLengthUpperBound), sizeof(int));
    stream.write(reinterpret_cast<const char*>(&useQualityScores), sizeof(bool));
    stream.write(reinterpret_cast<const char*>(&statistics), sizeof(Statistics));

    auto pos = stream.tellp();

    stream.seekp(sizeof(std::size_t) * 4, std::ios_base::cur);

    std::size_t lengthsBytes = gpulengthStorage.writeCpuLengthStoreToStream(stream);

    std::size_t sequencesBytes = distributedSequenceData.writeToStream(stream);
    std::size_t qualitiesBytes = 0;
    if(useQualityScores){
        distributedQualities.writeToStream(stream);
    }

    //read ids with N
    std::size_t numUndeterminedReads = readIdsOfReadsWithUndeterminedBase.size();
    stream.write(reinterpret_cast<const char*>(&numUndeterminedReads), sizeof(std::size_t));
    stream.write(reinterpret_cast<const char*>(readIdsOfReadsWithUndeterminedBase.data()), numUndeterminedReads * sizeof(read_number));

    std::size_t ambigBytes = sizeof(std::size_t) + numUndeterminedReads * sizeof(read_number);

    stream.seekp(pos);
    stream.write(reinterpret_cast<const char*>(&lengthsBytes), sizeof(std::size_t));
    stream.write(reinterpret_cast<const char*>(&sequencesBytes), sizeof(std::size_t));
    stream.write(reinterpret_cast<const char*>(&qualitiesBytes), sizeof(std::size_t));
    stream.write(reinterpret_cast<const char*>(&ambigBytes), sizeof(std::size_t));
}



void DistributedReadStorage::loadFromFile(const std::string& filename){
    loadFromFile(filename, deviceIds);
}

void DistributedReadStorage::loadFromFile(const std::string& filename, const std::vector<int>& deviceIds_){
    std::ifstream stream(filename, std::ios::binary);
    if(!stream)
        throw std::runtime_error("Cannot open file " + filename);

    destroy();

    // std::size_t lengthsize = sizeof(Length_t);
    // std::size_t loaded_lengthsize;
    // stream.read(reinterpret_cast<char*>(&loaded_lengthsize), sizeof(std::size_t));

    // if(loaded_lengthsize != lengthsize)
    //     throw std::runtime_error("Wrong size of length type!");


    read_number loaded_numberOfReads;
    int loaded_sequenceLengthLowerBound;
    int loaded_sequenceLengthUpperBound;
    bool loaded_hasQualityScores;

    stream.read(reinterpret_cast<char*>(&loaded_numberOfReads), sizeof(read_number));
    stream.read(reinterpret_cast<char*>(&loaded_sequenceLengthLowerBound), sizeof(int));
    stream.read(reinterpret_cast<char*>(&loaded_sequenceLengthUpperBound), sizeof(int));
    stream.read(reinterpret_cast<char*>(&loaded_hasQualityScores), sizeof(bool));

    init(deviceIds_, loaded_numberOfReads, canUseQualityScores(), 
        loaded_sequenceLengthLowerBound, loaded_sequenceLengthUpperBound);

    numberOfInsertedReads = loaded_numberOfReads;

    stream.read(reinterpret_cast<char*>(&statistics), sizeof(Statistics));

    std::size_t lengthsBytes = 0;
    std::size_t sequencesBytes = 0;      
    std::size_t qualitiesBytes = 0;       
    std::size_t ambigBytes = 0;

    stream.read(reinterpret_cast<char*>(&lengthsBytes), sizeof(std::size_t));
    stream.read(reinterpret_cast<char*>(&sequencesBytes), sizeof(std::size_t));
    stream.read(reinterpret_cast<char*>(&qualitiesBytes), sizeof(std::size_t));
    stream.read(reinterpret_cast<char*>(&ambigBytes), sizeof(std::size_t));

    lengthStorage.readFromStream(stream);

    distributedSequenceData.readFromStream(stream);

    if(canUseQualityScores() && loaded_hasQualityScores){
        //std::cerr << "load qualities\n";
        distributedQualities.readFromStream(stream);
    }else if(canUseQualityScores() && !loaded_hasQualityScores){
            //std::cerr << "no q in bin file\n";
            throw std::runtime_error("Quality scores expected in preprocessed reads file to load, but none are present. Abort.");
    }else if(!canUseQualityScores() && loaded_hasQualityScores){
            //std::cerr << "skip qualities\n";
            stream.ignore(qualitiesBytes);
    }else{
        //!canUseQualityScores() && !loaded_hasQualityScores
        //std::cerr << "no q in file, and no q required. Ok\n";
        stream.ignore(qualitiesBytes);
    }

    //read ids with N
    std::size_t numUndeterminedReads = 0;
    stream.read(reinterpret_cast<char*>(&numUndeterminedReads), sizeof(std::size_t));
    readIdsOfReadsWithUndeterminedBase.resize(numUndeterminedReads);
    stream.read(reinterpret_cast<char*>(readIdsOfReadsWithUndeterminedBase.data()), numUndeterminedReads * sizeof(read_number));

    //gpu read ids with N
    setGpuBitArraysFromVector();
    

}



void DistributedReadStorage::setGpuBitArraysFromVector(){
    constexpr size_t chunksize = 1024 * 1024;

    int oldId = 0;
    cudaGetDevice(&oldId); CUERR;

    const size_t numUndeterminedReads = readIdsOfReadsWithUndeterminedBase.size();
    //std::cerr << "setGpuBitArraysFromVector numUndeterminedReads = " << numUndeterminedReads << "\n";
    for(int deviceId : deviceIds){
        cudaSetDevice(deviceId); CUERR;

        bool* d_values;
        cudaMalloc(&d_values, sizeof(bool) * chunksize); CUERR;
        cudaMemset(d_values, 1, sizeof(bool) * chunksize); CUERR;

        read_number* d_positions;
        cudaMalloc(&d_positions, sizeof(read_number) * chunksize); CUERR;

        const int chunks = SDIV(numUndeterminedReads, chunksize);

        for(int i = 0; i < chunks; i++){
            size_t begin = i * chunksize;
            size_t end = std::min((i+1) * chunksize, numUndeterminedReads);

            size_t elements = end - begin;

            cudaMemcpy(
                d_positions, 
                readIdsOfReadsWithUndeterminedBase.data() + begin, 
                sizeof(read_number) * elements, 
                H2D
            ); CUERR;

            setReadsContainN_async(
                deviceId,
                d_values, 
                d_positions, 
                elements,
                0
            );
            cudaDeviceSynchronize(); CUERR;
        }

        cudaFree(d_values); CUERR;
        cudaFree(d_positions); CUERR;
    }
    cudaSetDevice(oldId); CUERR;
}





}
}

#endif

#endif
