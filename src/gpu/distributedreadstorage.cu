#if 1


#include <gpu/distributedreadstorage.hpp>
#include <gpu/distributedarray.hpp>
#include <memorymanagement.hpp>

#include <config.hpp>
#include <sequence.hpp>
#include <readlibraryio.hpp>
#include <threadpool.hpp>
#include <util.hpp>

#include <atomic>
#include <fstream>
#include <omp.h>
#include <algorithm>
#include <iterator>
#include <numeric>

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

        


        const int intsPerSequence = getEncodedNumInts2Bit(sequenceLengthUpperBound);

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

        


        const int intsPerSequence = getEncodedNumInts2Bit(sequenceLengthUpperBound);

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
    constexpr std::array<char, 4> bases = {'A', 'C', 'G', 'T'};

    auto checkRead = [&, this](read_number readIndex, Read& read, int& Ncount){
        const int readLength = int(read.sequence.size());

        if(readIndex >= expectedNumberOfReads){
            throw std::runtime_error("Error! Expected " + std::to_string(expectedNumberOfReads)
                                    + " reads, but file contains at least "
                                    + std::to_string(readIndex+1) + " reads.");
        }

        if(readLength > expectedMaximumReadLength){
            throw std::runtime_error("Error! Expected maximum read length = "
                                    + std::to_string(expectedMaximumReadLength)
                                    + ", but read " + std::to_string(readIndex)
                                    + "has length " + std::to_string(readLength));
        }

        auto isValidBase = [](char c){
            constexpr std::array<char, 10> validBases{'A','C','G','T','a','c','g','t'};
            return validBases.end() != std::find(validBases.begin(), validBases.end(), c);
        };

        const int undeterminedBasesInRead = std::count_if(read.sequence.begin(), read.sequence.end(), [&](char c){
            return !isValidBase(c);
        });

        //nmap[undeterminedBasesInRead]++;

        if(undeterminedBasesInRead > 0){
            setReadContainsN(readIndex, true);
        }

        for(auto& c : read.sequence){
            if(c == 'a') c = 'A';
            else if(c == 'c') c = 'C';
            else if(c == 'g') c = 'G';
            else if(c == 't') c = 'T';
            else if(!isValidBase(c)){
                c = bases[Ncount];
                Ncount = (Ncount + 1) % 4;
            }
        }
    };

    constexpr size_t maxbuffersize = 1000000;
    constexpr int numBuffers = 2;

    std::array<std::vector<read_number>, numBuffers> indicesBuffers;
    std::array<std::vector<Read>, numBuffers> readsBuffers;
    std::array<bool, numBuffers> canBeUsed;
    std::array<std::mutex, numBuffers> mutex;
    std::array<std::condition_variable, numBuffers> cv;

    ThreadPool threadPool(threads);

    for(int i = 0; i < numBuffers; i++){
        indicesBuffers[i].reserve(maxbuffersize);
        readsBuffers[i].reserve(maxbuffersize);
        canBeUsed[i] = true;
    }

    int bufferindex = 0;
    read_number globalReadId = 0;

    auto showProgressFunc = [show = showProgress](auto totalCount, auto seconds){
        if(show){
            std::cout << "Processed " << totalCount << " reads in file. Elapsed time: " 
                            << seconds << " seconds." << std::endl;
        }
    };

    auto updateShowProgressInterval = [](auto duration){
        return duration * 2;
    };

    ProgressThread<read_number> progressThread(
        expectedNumberOfReads, 
        showProgressFunc, 
        updateShowProgressInterval
    );

    for(const auto& inputfile : inputfiles){
        std::cout << "Converting reads of file " << inputfile << ", storing them in memory\n";

        forEachReadInFile(inputfile,
                        [&](auto /*readnum*/, const auto& read){

                if(!canBeUsed[bufferindex]){
                    std::unique_lock<std::mutex> ul(mutex[bufferindex]);
                    if(!canBeUsed[bufferindex]){
                        //std::cerr << "waiting for other buffer\n";
                        cv[bufferindex].wait(ul, [&](){ return canBeUsed[bufferindex]; });
                    }
                }

                auto indicesBufferPtr = &indicesBuffers[bufferindex];
                auto readsBufferPtr = &readsBuffers[bufferindex];
                indicesBufferPtr->emplace_back(globalReadId);
                readsBufferPtr->emplace_back(read);

                ++globalReadId;

                progressThread.addProgress(1);

                if(indicesBufferPtr->size() >= maxbuffersize){
                    canBeUsed[bufferindex] = false;

                    //std::cerr << "launch other thread\n";
                    threadPool.enqueue([&, indicesBufferPtr, readsBufferPtr, bufferindex](){
                        //std::cerr << "buffer " << bufferindex << " running\n";
                        int nmodcounter = 0;

                        for(int i = 0; i < int(readsBufferPtr->size()); i++){
                            read_number readId = (*indicesBufferPtr)[i];
                            auto& read = (*readsBufferPtr)[i];
                            checkRead(readId, read, nmodcounter);
                        }

                        setReads(&threadPool, *indicesBufferPtr, *readsBufferPtr);

                        //TIMERSTARTCPU(clear);
                        indicesBufferPtr->clear();
                        readsBufferPtr->clear();
                        //TIMERSTOPCPU(clear);
                        
                        std::lock_guard<std::mutex> l(mutex[bufferindex]);
                        canBeUsed[bufferindex] = true;
                        cv[bufferindex].notify_one();

                        //std::cerr << "buffer " << bufferindex << " finished\n";
                    });

                    bufferindex = (bufferindex + 1) % numBuffers; //swap buffers
                }

        });
    }

    auto indicesBufferPtr = &indicesBuffers[bufferindex];
    auto readsBufferPtr = &readsBuffers[bufferindex];

    if(int(readsBufferPtr->size()) > 0){
        if(!canBeUsed[bufferindex]){
            std::unique_lock<std::mutex> ul(mutex[bufferindex]);
            if(!canBeUsed[bufferindex]){
                //std::cerr << "waiting for other buffer\n";
                cv[bufferindex].wait(ul, [&](){ return canBeUsed[bufferindex]; });
            }
        }

        int nmodcounter = 0;

        for(int i = 0; i < int(readsBufferPtr->size()); i++){
            read_number readId = (*indicesBufferPtr)[i];
            auto& read = (*readsBufferPtr)[i];
            checkRead(readId, read, nmodcounter);
        }

        setReads(&threadPool, *indicesBufferPtr, *readsBufferPtr);

        indicesBufferPtr->clear();
        readsBufferPtr->clear();
    }

    for(int i = 0; i < numBuffers; i++){
        std::unique_lock<std::mutex> ul(mutex[i]);
        if(!canBeUsed[i]){
            //std::cerr << "Reading file completed. Waiting for buffer " << i << "\n";
            cv[i].wait(ul, [&](){ return canBeUsed[i]; });
        }
    }

    progressThread.finished();

    // std::cerr << "occurences of n/N:\n";
    // for(const auto& p : nmap){
    //     std::cerr << p.first << " " << p.second << '\n';
    // }

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
    handlearray(distributedQualities);

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

void DistributedReadStorage::setReads(ThreadPool* threadPool,
                                    read_number firstIndex, read_number lastIndex_excl, const Read* reads, int numReads){
    assert(!isReadOnly);

    std::vector<read_number> indices(lastIndex_excl-firstIndex);
    std::iota(indices.begin(), indices.end(), firstIndex);

    setReads(threadPool, indices, reads, numReads);
}

void DistributedReadStorage::setReads(ThreadPool* threadPool,
                                    read_number firstIndex, read_number lastIndex_excl, const std::vector<Read>& reads){
    assert(!isReadOnly);

    setReads(threadPool, firstIndex, lastIndex_excl, reads.data(), int(reads.size()));
}

void DistributedReadStorage::setReads(ThreadPool* threadPool,
                                    const std::vector<read_number>& indices, const std::vector<Read>& reads){
    assert(!isReadOnly);

    setReads(threadPool, indices, reads.data(), int(reads.size()));
}

void DistributedReadStorage::setReads(ThreadPool* threadPool,
                                    const std::vector<read_number>& indices, const Read* reads, int numReads){
    assert(!isReadOnly);

    //TIMERSTARTCPU(internalinit);
    auto lengthInRange = [&](Length_t length){
        return getSequenceLengthLowerBound() <= length && length <= getSequenceLengthUpperBound();
    };
    assert(indices.size() > 0);
    assert(numReads == int(indices.size()));
    assert(std::all_of(indices.begin(), indices.end(), [&](auto i){ return i < getMaximumNumberOfReads();}));
    assert(std::all_of(reads, reads + numReads, [&](const auto& r){ return lengthInRange(Length_t(r.sequence.length()));}));
    
    if(canUseQualityScores()){
        assert(std::all_of(reads, reads + numReads, [&](const auto& r){ return r.sequence.length() == r.quality.length();}));
    }

    auto minmax = std::minmax_element(reads, reads + numReads, [](const auto& r1, const auto& r2){
        return r1.sequence.length() < r2.sequence.length();
    });

    statistics.minimumSequenceLength = std::min(statistics.minimumSequenceLength, int(minmax.first->sequence.length()));
    statistics.maximumSequenceLength = std::max(statistics.maximumSequenceLength, int(minmax.second->sequence.length()));

    read_number maxIndex = *std::max_element(indices.begin(), indices.end());

    read_number prev_value = numberOfInsertedReads;
    while(prev_value < maxIndex+1 && !numberOfInsertedReads.compare_exchange_weak(prev_value, maxIndex+1)){
        ;
    }
        
#if 1
    std::vector<char> sequenceData;
    std::vector<Length_t> sequenceLengths;
    std::vector<char> qualityData;

    const size_t encodedSequencePitch = getEncodedNumInts2Bit(getSequenceLengthUpperBound()) * sizeof(int);
    const size_t qualityPitch = getSequenceLengthUpperBound();

    sequenceData.resize(encodedSequencePitch * numReads, 0);
    sequenceLengths.resize(numReads, 0);
    if(canUseQualityScores()){
        qualityData.resize(getSequenceLengthUpperBound() * numReads, 0);
    }

    //TIMERSTOPCPU(internalinit);

    auto prepare = [&](int begin, int end, int /*threadId*/){
        for(int i = begin; i < end; i++){
            const Read& r = reads[i];

            unsigned int* dest = (unsigned int*)&sequenceData[std::size_t(i) * encodedSequencePitch];
            encodeSequence2Bit(dest,
                                    r.sequence.c_str(),
                                    r.sequence.length());
            sequenceLengths[i] = Length_t(r.sequence.length());
            if(canUseQualityScores()){
                std::copy(r.quality.begin(), r.quality.end(), qualityData.begin() + i * qualityPitch);
            }
        }
    };

    ThreadPool::ParallelForHandle pforHandle;

    threadPool->parallelFor(pforHandle, 0, numReads, prepare);

    //TIMERSTOPCPU(internal);

    //TIMERSTARTCPU(internalset);
    setSequences(indices, sequenceData.data());
    setSequenceLengths(indices, sequenceLengths.data());
    if(canUseQualityScores()){
        setQualities(indices, qualityData.data());
    }
    //TIMERSTOPCPU(internalset);
#else 


    const size_t encodedSequencePitchInInts = getEncodedNumInts2Bit(getSequenceLengthUpperBound());
    const std::size_t encodedSequencePitch = encodedSequencePitchInInts * sizeof(int);
    const size_t qualityPitch = getSequenceLengthUpperBound();

    const std::size_t decodedSequencePitch = SDIV(statistics.maximumSequenceLength, 128) * 128;

    constexpr std::size_t buffersize = std::size_t{1} << 20;

    const int readsPerIteration = buffersize / decodedSequencePitch;
    const int iterations = SDIV(numReads, readsPerIteration);

    char* h_decodedSequences;
    char* d_decodedSequences;

    cudaMallocHost(&h_decodedSequences, sizeof(char) * buffersize * 2); CUERR;
    cudaMalloc(&d_decodedSequences, sizeof(char) * buffersize * 2); CUERR;

    std::array<char*, 2> h_decodedSequences_arr{h_decodedSequences, h_decodedSequences + buffersize};
    std::array<char*, 2> d_decodedSequences_arr{d_decodedSequences, d_decodedSequences + buffersize};

    unsigned int* h_encodedSequences;
    unsigned int* d_encodedSequences;

    cudaMallocHost(&h_encodedSequences, sizeof(unsigned int) * encodedSequencePitchInInts * readsPerIteration * 2); CUERR;
    cudaMalloc(&d_encodedSequences, sizeof(unsigned int) * encodedSequencePitchInInts * readsPerIteration * 2); CUERR;

    std::array<unsigned int*, 2> h_encodedSequences_arr{h_encodedSequences, h_encodedSequences + sizeof(unsigned int) * encodedSequencePitchInInts * readsPerIteration};
    std::array<unsigned int*, 2> d_encodedSequences_arr{d_encodedSequences, d_encodedSequences + sizeof(unsigned int) * encodedSequencePitchInInts * readsPerIteration};

    int* h_lengths;
    int* d_lengths;

    cudaMallocHost(&h_lengths, sizeof(int) * readsPerIteration * 2); CUERR;
    cudaMalloc(&d_lengths, sizeof(int) * readsPerIteration * 2); CUERR;

    std::array<int*, 2> h_lengths_arr{h_lengths, h_lengths + readsPerIteration};
    std::array<int*, 2> d_lengths_arr{d_lengths, d_lengths + readsPerIteration};

    std::array<cudaStream_t, 2> streams;
    cudaStreamCreate(&streams[0]); CUERR;
    cudaStreamCreate(&streams[1]); CUERR;

    std::vector<char> sequenceData;
    std::vector<Length_t> sequenceLengths;
    std::vector<char> qualityData;

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

        // cudaMemcpyAsync(
        //     d_decodedSequences_arr[currentbuf],
        //     h_decodedSequences_arr[currentbuf],
        //     sizeof(char) * chunksize * decodedSequencePitch,
        //     H2D,
        //     streams[currentbuf]
        // ); CUERR;

        // cudaMemcpyAsync(
        //     d_lengths_arr[currentbuf],
        //     h_lengths_arr[currentbuf],
        //     sizeof(int) * chunksize,
        //     H2D,
        //     streams[currentbuf]
        // ); CUERR;

        // //kernel for encoding

        // cudaMemcpyAsync(
        //     h_encodedSequences_arr[currentbuf],
        //     d_encodedSequences_arr[currentbuf],
        //     sizeof(unsigned int) * chunksize * encodedSequencePitchInInts,
        //     D2H,
        //     streams[currentbuf]
        // ); CUERR;

        auto prepare = [&](int b, int e, int /*threadId*/){
            for(int i = b; i < e; i++){
                const Read& r = reads[i];
    
                unsigned int* dest = (unsigned int*)&sequenceData[std::size_t(begin + i) * encodedSequencePitch];
                const char* src = h_decodedSequences_arr[currentbuf] + i * decodedSequencePitch;
                encodeSequence2Bit(
                    dest,
                    src,
                    h_lengths_arr[currentbuf][i]
                );
            }
        };
    
        
    
        threadPool->parallelFor(pforHandle, 0, chunksize, prepare);


        //std::copy(h_lengths_arr[currentbuf], h_lengths_arr[currentbuf] + chunksize, sequenceLengths.begin() + begin);

        if(iteration > 0){
            //process results of previous iteration
            const int otherbuf = 1 - currentbuf;

            const int otherbegin = (iteration-1) * readsPerIteration;
            const int otherend = iteration * readsPerIteration;
            const int otherchunksize = end - begin;

            cudaStreamSynchronize(streams[otherbuf]);

            for(int i = 0; i < chunksize; i++){
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

    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaFree(d_lengths); CUERR;
    cudaFreeHost(h_lengths); CUERR;
    cudaFree(d_encodedSequences); CUERR;
    cudaFreeHost(h_encodedSequences); CUERR;
    cudaFree(d_decodedSequences); CUERR;
    cudaFreeHost(h_decodedSequences); CUERR;



    // ThreadPool::ParallelForHandle pforHandle;

    // threadPool->parallelFor(pforHandle, 0, numReads, prepare);

    setSequences(indices, sequenceData.data());
    setSequenceLengths(indices, sequenceLengths.data());
    if(canUseQualityScores()){
        setQualities(indices, qualityData.data());
    }

#endif
}

void DistributedReadStorage::setReadContainsN(read_number readId, bool contains){
    assert(!isReadOnly);

    auto pos = std::lower_bound(readIdsOfReadsWithUndeterminedBase.begin(),
                                        readIdsOfReadsWithUndeterminedBase.end(),
                                        readId);


    if(contains){
        if(pos != readIdsOfReadsWithUndeterminedBase.end()){
            ; //already marked
        }else{
            std::lock_guard<std::mutex> l(mutexUndeterminedBaseReads);
            readIdsOfReadsWithUndeterminedBase.insert(pos, readId);
        }
    }else{
        if(pos != readIdsOfReadsWithUndeterminedBase.end()){
            //remove mark
            std::lock_guard<std::mutex> l(mutexUndeterminedBaseReads);
            readIdsOfReadsWithUndeterminedBase.erase(pos);
        }else{
            ; //already unmarked
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

void DistributedReadStorage::setSequences(const std::vector<read_number>& indices, const char* data){
    assert(!isReadOnly);

    distributedSequenceData.setSafe(indices, reinterpret_cast<const unsigned int*>(data));
}

void DistributedReadStorage::setSequenceLengths(read_number firstIndex, read_number lastIndex_excl, const Length_t* data){
    assert(!isReadOnly);

    for(read_number i = firstIndex; i < lastIndex_excl; i++){
        lengthStorage.setLength(i, data[i - firstIndex]);
    }
}

void DistributedReadStorage::setSequenceLengths(const std::vector<read_number>& indices, const Length_t* data){
    assert(!isReadOnly);

    for(std::size_t i = 0; i < indices.size(); i++){
        lengthStorage.setLength(indices[i], data[i]);
    }
}

void DistributedReadStorage::setQualities(read_number firstIndex, read_number lastIndex_excl, const char* data){
    assert(!isReadOnly);

    distributedQualities.setSafe(firstIndex, lastIndex_excl, data);
}

void DistributedReadStorage::setQualities(const std::vector<read_number>& indices, const char* data){
    assert(!isReadOnly);
    
    distributedQualities.setSafe(indices, data);
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


#if 0

void DistributedReadStorage::saveToFile(const std::string& filename) const{
    std::ofstream stream(filename, std::ios::binary);

    //int ser_id = serialization_id;
    //std::size_t lengthsize = sizeof(Length_t);
    //stream.write(reinterpret_cast<const char*>(&lengthsize), sizeof(std::size_t));

    read_number inserted = getNumberOfReads();

    stream.write(reinterpret_cast<const char*>(&inserted), sizeof(read_number));
    stream.write(reinterpret_cast<const char*>(&sequenceLengthLowerBound), sizeof(int));
    stream.write(reinterpret_cast<const char*>(&sequenceLengthUpperBound), sizeof(int));
    stream.write(reinterpret_cast<const char*>(&useQualityScores), sizeof(bool));
    stream.write(reinterpret_cast<const char*>(&statistics), sizeof(Statistics));

    gpulengthStorage.writeCpuLengthStoreToStream(stream);

    constexpr read_number batchsize = 10000000;
    int numBatches = SDIV(getNumberOfReads(), batchsize);

    {
        auto sequencehandle = makeGatherHandleSequences();
        size_t outputpitch = getEncodedNumInts2Bit(sequenceLengthUpperBound) * sizeof(int);

        size_t totalSequenceMemory = outputpitch * getNumberOfReads();
        stream.write(reinterpret_cast<const char*>(&totalSequenceMemory), sizeof(size_t));

        for(int batch = 0; batch < numBatches; batch++){
            read_number begin = batch * batchsize;
            read_number end = std::min((batch+1) * batchsize, getNumberOfReads());

            std::vector<read_number> indices(end-begin);
            std::iota(indices.begin(), indices.end(), begin);

            size_t databytes = outputpitch * indices.size();
            std::vector<char> data(databytes, 0);

            auto future = gatherSequenceDataToHostBufferAsync(
                                        sequencehandle,
                                        data.data(),
                                        outputpitch,
                                        indices.data(),
                                        indices.size(),
                                        1);

            future.wait();

            stream.write(reinterpret_cast<const char*>(&data[0]), databytes);
        }
    }

    // {
    //     auto lengthhandle = makeGatherHandleLengths();
    //     size_t outputpitch = sizeof(Length_t);

    //     size_t totalLengthMemory = outputpitch * getNumberOfReads();
    //     stream.write(reinterpret_cast<const char*>(&totalLengthMemory), sizeof(size_t));

    //     for(int batch = 0; batch < numBatches; batch++){
    //         read_number begin = batch * batchsize;
    //         read_number end = std::min((batch+1) * batchsize, getNumberOfReads());

    //         std::vector<read_number> indices(end-begin);
    //         std::iota(indices.begin(), indices.end(), begin);

    //         size_t databytes = outputpitch * indices.size();
    //         std::vector<Length_t> data(indices.size(), 0);

    //         auto future = gatherSequenceLengthsToHostBufferAsync(
    //                                     lengthhandle,
    //                                     data.data(),
    //                                     indices.data(),
    //                                     indices.size(),
    //                                     1);

    //         future.wait();

    //         stream.write(reinterpret_cast<const char*>(&data[0]), databytes);
    //     }
    // }

    if(useQualityScores){
        auto qualityhandle = makeGatherHandleQualities();
        size_t outputpitch = sequenceLengthUpperBound;

        size_t totalqualityMemory = outputpitch * getNumberOfReads();
        stream.write(reinterpret_cast<const char*>(&totalqualityMemory), sizeof(size_t));

        for(int batch = 0; batch < numBatches; batch++){
            read_number begin = batch * batchsize;
            read_number end = std::min((batch+1) * batchsize, getNumberOfReads());

            std::vector<read_number> indices(end-begin);
            std::iota(indices.begin(), indices.end(), begin);

            size_t databytes = outputpitch * indices.size();
            std::vector<char> data(databytes, 0);

            auto future = gatherQualitiesToHostBufferAsync(
                                        qualityhandle,
                                        data.data(),
                                        outputpitch,
                                        indices.data(),
                                        indices.size(),
                                        1);

            future.wait();

            stream.write(reinterpret_cast<const char*>(&data[0]), databytes);
        }
    }

    //read ids with N
    std::size_t numUndeterminedReads = readIdsOfReadsWithUndeterminedBase.size();
    stream.write(reinterpret_cast<const char*>(&numUndeterminedReads), sizeof(size_t));
    stream.write(reinterpret_cast<const char*>(readIdsOfReadsWithUndeterminedBase.data()), numUndeterminedReads * sizeof(read_number));
}

#else 

void DistributedReadStorage::saveToFile(const std::string& filename) const{
    std::ofstream stream(filename, std::ios::binary);

    //int ser_id = serialization_id;
    //std::size_t lengthsize = sizeof(Length_t);
    //stream.write(reinterpret_cast<const char*>(&lengthsize), sizeof(std::size_t));

    read_number inserted = getNumberOfReads();

    stream.write(reinterpret_cast<const char*>(&inserted), sizeof(read_number));
    stream.write(reinterpret_cast<const char*>(&sequenceLengthLowerBound), sizeof(int));
    stream.write(reinterpret_cast<const char*>(&sequenceLengthUpperBound), sizeof(int));
    stream.write(reinterpret_cast<const char*>(&useQualityScores), sizeof(bool));
    stream.write(reinterpret_cast<const char*>(&statistics), sizeof(Statistics));

    gpulengthStorage.writeCpuLengthStoreToStream(stream);

    distributedSequenceData.writeToStream(stream);      
    distributedQualities.writeToStream(stream);       

    //read ids with N
    std::size_t numUndeterminedReads = readIdsOfReadsWithUndeterminedBase.size();
    stream.write(reinterpret_cast<const char*>(&numUndeterminedReads), sizeof(size_t));
    stream.write(reinterpret_cast<const char*>(readIdsOfReadsWithUndeterminedBase.data()), numUndeterminedReads * sizeof(read_number));
}



#endif

void DistributedReadStorage::loadFromFile(const std::string& filename){
    loadFromFile(filename, deviceIds);
}


#if 0

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
    bool loaded_useQualityScores;

    stream.read(reinterpret_cast<char*>(&loaded_numberOfReads), sizeof(read_number));
    stream.read(reinterpret_cast<char*>(&loaded_sequenceLengthLowerBound), sizeof(int));
    stream.read(reinterpret_cast<char*>(&loaded_sequenceLengthUpperBound), sizeof(int));
    stream.read(reinterpret_cast<char*>(&loaded_useQualityScores), sizeof(bool));

    init(deviceIds_, loaded_numberOfReads, loaded_useQualityScores, 
        loaded_sequenceLengthLowerBound, loaded_sequenceLengthUpperBound);

    numberOfInsertedReads = loaded_numberOfReads;

    stream.read(reinterpret_cast<char*>(&statistics), sizeof(Statistics));

    lengthStorage.readFromStream(stream);

    constexpr read_number batchsize = 10000000;
    int numBatches = SDIV(loaded_numberOfReads, batchsize);

    {
        size_t seqpitch = getEncodedNumInts2Bit(sequenceLengthUpperBound) * sizeof(int);

        size_t totalSequenceMemory = 1;
        stream.read(reinterpret_cast<char*>(&totalSequenceMemory), sizeof(size_t));

        size_t totalMemoryRead = 0;

        for(int batch = 0; batch < numBatches; batch++){
            read_number begin = batch * batchsize;
            read_number end = std::min((batch+1) * batchsize, loaded_numberOfReads);

            size_t databytes = seqpitch * (end-begin);
            std::vector<char> data(databytes, 0);

            stream.read(reinterpret_cast<char*>(&data[0]), databytes);
            totalMemoryRead += stream.gcount();

            assert(totalMemoryRead <= totalSequenceMemory);

            setSequences(begin, end, data.data());
        }

        assert(totalMemoryRead == totalSequenceMemory);
    }

    // {
    //     size_t lengthpitch = sizeof(Length_t);

    //     size_t totalLengthMemory = 1;
    //     stream.read(reinterpret_cast<char*>(&totalLengthMemory), sizeof(size_t));

    //     size_t totalMemoryRead = 0;

    //     for(int batch = 0; batch < numBatches; batch++){
    //         read_number begin = batch * batchsize;
    //         read_number end = std::min((batch+1) * batchsize, loaded_numberOfReads);

    //         std::vector<Length_t> data((end-begin), 0);

    //         size_t databytes = lengthpitch * (end-begin);
    //         stream.read(reinterpret_cast<char*>(&data[0]), databytes);
    //         totalMemoryRead += stream.gcount();

    //         assert(totalMemoryRead <= totalLengthMemory);

    //         setSequenceLengths(begin, end, data.data());

    //         // auto minmax = std::minmax_element(data.begin(), data.end(), [](const auto& l1, const auto& l2){
    //         //     return l1 < l2;
    //         // });

    //         // statistics.minimumSequenceLength = std::min(statistics.minimumSequenceLength, int(*minmax.first));
    //         // statistics.maximumSequenceLength = std::max(statistics.maximumSequenceLength, int(*minmax.second));
    //     }

    //     assert(totalMemoryRead == totalLengthMemory);
    // }

    if(useQualityScores){
        size_t qualitypitch = sequenceLengthUpperBound;

        size_t totalqualityMemory = 1;
        stream.read(reinterpret_cast<char*>(&totalqualityMemory), sizeof(size_t));

        size_t totalMemoryRead = 0;

        for(int batch = 0; batch < numBatches; batch++){
            read_number begin = batch * batchsize;
            read_number end = std::min((batch+1) * batchsize, loaded_numberOfReads);

            size_t databytes = qualitypitch * (end-begin);
            std::vector<char> data(databytes, 0);

            stream.read(reinterpret_cast<char*>(&data[0]), databytes);
            totalMemoryRead += stream.gcount();

            assert(totalMemoryRead <= totalqualityMemory);

            setQualities(begin, end, data.data());
        }

        assert(totalMemoryRead == totalqualityMemory);
    }

    //read ids with N
    std::size_t numUndeterminedReads = 0;
    stream.read(reinterpret_cast<char*>(&numUndeterminedReads), sizeof(std::size_t));
    readIdsOfReadsWithUndeterminedBase.resize(numUndeterminedReads);
    stream.read(reinterpret_cast<char*>(readIdsOfReadsWithUndeterminedBase.data()), numUndeterminedReads * sizeof(read_number));

}

#else 

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
    bool loaded_useQualityScores;

    stream.read(reinterpret_cast<char*>(&loaded_numberOfReads), sizeof(read_number));
    stream.read(reinterpret_cast<char*>(&loaded_sequenceLengthLowerBound), sizeof(int));
    stream.read(reinterpret_cast<char*>(&loaded_sequenceLengthUpperBound), sizeof(int));
    stream.read(reinterpret_cast<char*>(&loaded_useQualityScores), sizeof(bool));

    init(deviceIds_, loaded_numberOfReads, loaded_useQualityScores, 
        loaded_sequenceLengthLowerBound, loaded_sequenceLengthUpperBound);

    numberOfInsertedReads = loaded_numberOfReads;

    stream.read(reinterpret_cast<char*>(&statistics), sizeof(Statistics));

    lengthStorage.readFromStream(stream);

    distributedSequenceData.readFromStream(stream);
    distributedQualities.readFromStream(stream);

    //read ids with N
    std::size_t numUndeterminedReads = 0;
    stream.read(reinterpret_cast<char*>(&numUndeterminedReads), sizeof(std::size_t));
    readIdsOfReadsWithUndeterminedBase.resize(numUndeterminedReads);
    stream.read(reinterpret_cast<char*>(readIdsOfReadsWithUndeterminedBase.data()), numUndeterminedReads * sizeof(read_number));

    //gpu read ids with N
    setGpuBitArraysFromVector();
    

}



#endif


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


DistributedReadStorage::SavedGpuPartitionData DistributedReadStorage::saveGpuPartitionData(
            int deviceId,
            std::ofstream& stream, 
            std::size_t* numUsableBytes) const{

    auto it = std::find(deviceIds.begin(), deviceIds.end(), deviceId);
    assert(it != deviceIds.end());

    int gpuIndex = std::distance(deviceIds.begin(), it);

    SavedGpuPartitionData saved;
    saved.partitionId = gpuIndex;

    if(*numUsableBytes >= distributedSequenceData.getPartitionSizeInBytes(gpuIndex)){
        saved.sequenceData = distributedSequenceData.writeGpuPartitionToMemory(gpuIndex);
        saved.sequenceDataLocation = SavedGpuPartitionData::Type::Memory;

        *numUsableBytes -= distributedSequenceData.getPartitionSizeInBytes(gpuIndex);
    }else{
        distributedSequenceData.writeGpuPartitionToStream(gpuIndex, stream);
        saved.sequenceDataLocation = SavedGpuPartitionData::Type::File;
    }

    if(useQualityScores){
        if(*numUsableBytes >= distributedQualities.getPartitionSizeInBytes(gpuIndex)){
            saved.qualityData = distributedQualities.writeGpuPartitionToMemory(gpuIndex);
            saved.qualityDataLocation = SavedGpuPartitionData::Type::Memory;

            *numUsableBytes -= distributedQualities.getPartitionSizeInBytes(gpuIndex);
        }else{
            distributedQualities.writeGpuPartitionToStream(gpuIndex, stream);
            saved.qualityDataLocation = SavedGpuPartitionData::Type::File;
        }
    }

    return saved;
}

void DistributedReadStorage::loadGpuPartitionData(int deviceId, 
                                                std::ifstream& stream, 
                                                const DistributedReadStorage::SavedGpuPartitionData& saved) const{
                      
    auto it = std::find(deviceIds.begin(), deviceIds.end(), deviceId);
    assert(it != deviceIds.end());

    int gpuIndex = std::distance(deviceIds.begin(), it);

    if(saved.sequenceDataLocation == SavedGpuPartitionData::Type::Memory){
        distributedSequenceData.readGpuPartitionFromMemory(gpuIndex, saved.sequenceData);
    }else{
        distributedSequenceData.readGpuPartitionFromStream(gpuIndex, stream);
    }

    if(useQualityScores){
        if(saved.qualityDataLocation == SavedGpuPartitionData::Type::Memory){
            distributedQualities.readGpuPartitionFromMemory(gpuIndex, saved.qualityData);
        }else{
            distributedQualities.readGpuPartitionFromStream(gpuIndex, stream);
        }
    }
}

void DistributedReadStorage::allocateGpuData(int deviceId) const{
    auto it = std::find(deviceIds.begin(), deviceIds.end(), deviceId);
    assert(it != deviceIds.end());

    int gpuIndex = std::distance(deviceIds.begin(), it);

    distributedSequenceData.allocateGpuPartition(gpuIndex);
    if(useQualityScores){
        distributedQualities.allocateGpuPartition(gpuIndex);  
    }
}

void DistributedReadStorage::deallocateGpuData(int deviceId) const{
    auto it = std::find(deviceIds.begin(), deviceIds.end(), deviceId);
    assert(it != deviceIds.end());

    int gpuIndex = std::distance(deviceIds.begin(), it);

    distributedSequenceData.deallocateGpuPartition(gpuIndex);
    if(useQualityScores){
        distributedQualities.deallocateGpuPartition(gpuIndex);  
    }
}

DistributedReadStorage::SavedGpuPartitionData DistributedReadStorage::saveGpuPartitionDataAndFreeGpuMem(
                    int deviceId,
                    std::ofstream& stream, 
                    std::size_t* numUsableBytes) const{

    auto result = saveGpuPartitionData(deviceId,
                                        stream, 
                                        numUsableBytes);

    deallocateGpuData(deviceId);

    return result;
}

void DistributedReadStorage::allocGpuMemAndLoadGpuPartitionData(int deviceId, 
                                                                std::ifstream& stream, 
                                                                const DistributedReadStorage::SavedGpuPartitionData& saved) const{
                      
    allocateGpuData(deviceId);

    loadGpuPartitionData(deviceId, 
                        stream, 
                        saved);
}

DistributedReadStorage::SavedGpuData DistributedReadStorage::saveGpuDataAndFreeGpuMem(std::ofstream& stream, 
                                                                                      std::size_t numBytesMustRemainFree) const{
    SavedGpuData saved;
    saved.gpuPartitionData.reserve(deviceIds.size());

    for(int gpu = 0; gpu < int(deviceIds.size()); gpu++){
        saved.gpuPartitionData.emplace_back(saveGpuPartitionDataAndFreeGpuMem(deviceIds[gpu], stream, &numBytesMustRemainFree));
    }

    return saved;
}

void DistributedReadStorage::allocGpuMemAndLoadGpuData(std::ifstream& stream, const DistributedReadStorage::SavedGpuData& saved) const{
    for(const auto& partitionData : saved.gpuPartitionData){
        allocGpuMemAndLoadGpuPartitionData(deviceIds[partitionData.partitionId], 
                                            stream, 
                                            partitionData);
    }
}




}
}

#endif

#endif
