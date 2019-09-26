#if 1


#include <gpu/distributedreadstorage.hpp>
#include <gpu/distributedarray.hpp>

#include <config.hpp>
#include <sequence.hpp>
#include <sequencefileio.hpp>
#include <threadpool.hpp>

#include <fstream>
#include <omp.h>
#include <algorithm>
#include <iterator>

#ifdef __NVCC__

namespace care{
namespace gpu{

DistributedReadStorage::DistributedReadStorage(const std::vector<int>& deviceIds_, read_number nReads, bool b, 
                    int minimum_sequence_length, int maximum_sequence_length){

    init(deviceIds_, nReads, b, minimum_sequence_length, maximum_sequence_length);
}

void DistributedReadStorage::init(const std::vector<int>& deviceIds_, read_number nReads, bool b, 
                    int minimum_sequence_length, int maximum_sequence_length){
    assert(minimum_sequence_length <= maximum_sequence_length);

    isReadOnly = false;
    deviceIds = deviceIds_;
    numberOfReads = nReads;
    sequenceLengthLowerBound = minimum_sequence_length;
    sequenceLengthUpperBound = maximum_sequence_length;
    useQualityScores = b;

    int numGpus = deviceIds.size();

    constexpr size_t headRoom = gpuReadStorageHeadroomPerGPU; 
    //constexpr size_t headRoom = (size_t(1) << 30) * 11; 

    if(numberOfReads > 0 && sequenceLengthUpperBound > 0 && sequenceLengthLowerBound >= 0){
        lengthStorage = std::move(LengthStore(sequenceLengthLowerBound, sequenceLengthUpperBound, numberOfReads));

        std::cout << "Using " << lengthStorage.getRawBitsPerLength() << " bits per read to store its length\n";

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

        int oldId; cudaGetDevice(&oldId); CUERR;


        const int intsPerSequence = getEncodedNumInts2BitHiLo(sequenceLengthUpperBound);

        updateMemoryLimits();
        distributedSequenceData2 = std::move(DistributedArray<unsigned int, read_number>(deviceIds, maximumUsableBytesPerGpu, numberOfReads, intsPerSequence));

        //updateMemoryLimits();
        //distributedSequenceLengths2 = std::move(DistributedArray<Length_t, read_number>(deviceIds, maximumUsableBytesPerGpu, numberOfReads, 1));

        if(useQualityScores){
            updateMemoryLimits();
            distributedQualities2 = std::move(DistributedArray<char, read_number>(deviceIds, maximumUsableBytesPerGpu, numberOfReads, sequenceLengthUpperBound));
        }

        getGpuMemoryInfo();

        std::cerr << "Free memory per gpu after construction of distributed readstorage: ";
        std::copy(freeMemPerGpu.begin(), freeMemPerGpu.end(), std::ostream_iterator<size_t>(std::cerr, " "));
        std::cerr << "\n";

        cudaSetDevice(oldId); CUERR;
    }
}

DistributedReadStorage::DistributedReadStorage(DistributedReadStorage&& other){
    *this = std::move(other);
}

DistributedReadStorage& DistributedReadStorage::operator=(DistributedReadStorage&& rhs){
    isReadOnly = rhs.isReadOnly;
    deviceIds = std::move(rhs.deviceIds);
    numberOfReads = std::move(rhs.numberOfReads);
    sequenceLengthLowerBound = std::move(rhs.sequenceLengthLowerBound);
    sequenceLengthUpperBound = std::move(rhs.sequenceLengthUpperBound);
    useQualityScores = std::move(rhs.useQualityScores);
    readIdsOfReadsWithUndeterminedBase = std::move(rhs.readIdsOfReadsWithUndeterminedBase);
    lengthStorage = std::move(rhs.lengthStorage);
    gpulengthStorage = std::move(rhs.gpulengthStorage);
    distributedSequenceData2 = std::move(rhs.distributedSequenceData2);
    distributedSequenceLengths2 = std::move(rhs.distributedSequenceLengths2);
    distributedQualities2 = std::move(rhs.distributedQualities2);
    statistics = std::move(rhs.statistics);
    hasMoved = rhs.hasMoved;
    rhs.hasMoved = true;

    return *this;
}

DistributedReadStorage::MemoryInfo DistributedReadStorage::getMemoryInfo() const{
    MemoryInfo info;
    info.deviceSizeInBytes.resize(deviceIds.size(),0);
    info.deviceIds = deviceIds;

    auto handlearray = [&](const auto& array){
        const auto partitions = array.getPartitions();

        for(int location = 0; location < distributedSequenceData2.numLocations; location++){
            size_t bytes = partitions[location] * array.sizeOfElement;
            std::cerr << "location " << location << " " << bytes << "\n";
            if(location == array.hostLocation){
                info.hostSizeInBytes += bytes;
            }else{
                info.deviceSizeInBytes[location] += bytes;
            }
        }
    };

    handlearray(distributedSequenceData2);
    handlearray(distributedSequenceLengths2);
    handlearray(distributedQualities2);

    return info;
}


DistributedReadStorage::Statistics DistributedReadStorage::getStatistics() const{
    return statistics;
}

void DistributedReadStorage::destroy(){
    numberOfReads = 0;
    sequenceLengthUpperBound = 0;
    std::vector<size_t> fractions(deviceIds.size(), 0);
    lengthStorage = std::move(LengthStore{});
    gpulengthStorage = std::move(GPULengthStore{});
    distributedSequenceData2 = std::move(DistributedArray<unsigned int, read_number>(deviceIds, fractions, 0, 0));
    distributedSequenceLengths2 = std::move(DistributedArray<Length_t, read_number>(deviceIds, fractions, 0, 0));
    distributedQualities2 = std::move(DistributedArray<char, read_number>(deviceIds, fractions, 0, 0));
    statistics = Statistics{};
}

read_number DistributedReadStorage::getNumberOfReads() const{
    return numberOfReads;
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

void DistributedReadStorage::setReads(read_number firstIndex, read_number lastIndex_excl, const Read* reads, int numReads){
    assert(!isReadOnly);

    std::vector<read_number> indices(lastIndex_excl-firstIndex);
    std::iota(indices.begin(), indices.end(), firstIndex);

    setReads(indices, reads, numReads);
}

void DistributedReadStorage::setReads(read_number firstIndex, read_number lastIndex_excl, const std::vector<Read>& reads){
    assert(!isReadOnly);

    setReads(firstIndex, lastIndex_excl, reads.data(), int(reads.size()));
}

void DistributedReadStorage::setReads(const std::vector<read_number>& indices, const std::vector<Read>& reads){
    assert(!isReadOnly);

    setReads(indices, reads.data(), int(reads.size()));
}

void DistributedReadStorage::setReads(const std::vector<read_number>& indices, const Read* reads, int numReads){
    assert(!isReadOnly);

    //TIMERSTARTCPU(internalinit);
    auto lengthInRange = [&](Length_t length){
        return getSequenceLengthLowerBound() <= length && length <= getSequenceLengthUpperBound();
    };
    assert(indices.size() > 0);
    assert(numReads == int(indices.size()));
    assert(std::all_of(indices.begin(), indices.end(), [&](auto i){ return i < getNumberOfReads();}));
    assert(std::all_of(reads, reads + numReads, [&](const auto& r){ return lengthInRange(Length_t(r.sequence.length()));}));
    assert(std::all_of(reads, reads + numReads, [&](const auto& r){ return r.sequence.length() == r.quality.length();}));

    auto minmax = std::minmax_element(reads, reads + numReads, [](const auto& r1, const auto& r2){
        return r1.sequence.length() < r2.sequence.length();
    });

    statistics.minimumSequenceLength = std::min(statistics.minimumSequenceLength, int(minmax.first->sequence.length()));
    statistics.maximumSequenceLength = std::max(statistics.maximumSequenceLength, int(minmax.second->sequence.length()));

    std::vector<char> sequenceData;
    std::vector<Length_t> sequenceLengths;
    std::vector<char> qualityData;

    const size_t encodedSequencePitch = getEncodedNumInts2BitHiLo(getSequenceLengthUpperBound()) * sizeof(int);
    const size_t qualityPitch = getSequenceLengthUpperBound();

    sequenceData.resize(encodedSequencePitch * numReads, 0);
    sequenceLengths.resize(numReads, 0);
    if(canUseQualityScores()){
        qualityData.resize(getSequenceLengthUpperBound() * numReads, 0);
    }

    //TIMERSTOPCPU(internalinit);

    int chunksToWaitFor = 0;
    int finishedChunks = 0;
    std::mutex mutex;
    std::condition_variable cv;

    auto prepare = [&](int begin, int end){
        for(int i = begin; i < end; i++){
            const Read& r = reads[i];

            unsigned int* dest = (unsigned int*)&sequenceData[std::size_t(i) * encodedSequencePitch];
            encodeSequence2BitHiLo(dest,
                                    r.sequence.c_str(),
                                    r.sequence.length());
            sequenceLengths[i] = Length_t(r.sequence.length());
            if(canUseQualityScores()){
                std::copy(r.quality.begin(), r.quality.end(), qualityData.begin() + i * qualityPitch);
            }
        }
        std::lock_guard<std::mutex> l(mutex);
        finishedChunks++;
        cv.notify_one();
        //std::cerr << indices[0] << " " << indices.back() << " prepare " << begin << " " << end << " finished\n";
    };

    const int chunks = threadpool.getConcurrency();
    const int chunksize = numReads / chunks;
    const int leftover = numReads % chunks;

    int begin = 0;
    int end = chunksize;

    //TIMERSTARTCPU(internal);

    for(int c = 0; c < chunks-1; c++){
        if(c < leftover){
            end++;
        }
        if(end - begin > 0){
            chunksToWaitFor++;

            threadpool.enqueue([=](){
                prepare(begin, end);
            });
            begin = end;
            end = end + chunksize;
        }else{
            break;
        }
    }

    //handle last chunk in current thread.
    if(end - begin > 0){
        chunksToWaitFor++;
        prepare(begin, end);
        begin = end;
        end = end + chunksize;
    }

    {
        std::unique_lock<std::mutex> l(mutex);
        if(finishedChunks < chunksToWaitFor){
            cv.wait(l, [&](){
                return finishedChunks == chunksToWaitFor;
            });
        }
    }

    //TIMERSTOPCPU(internal);

    //TIMERSTARTCPU(internalset);
    setSequences(indices, sequenceData.data());
    setSequenceLengths(indices, sequenceLengths.data());
    if(canUseQualityScores()){
        setQualities(indices, qualityData.data());
    }
    //TIMERSTOPCPU(internalset);
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

void DistributedReadStorage::constructionIsComplete(){
    gpulengthStorage = std::move(GPULengthStore{std::move(lengthStorage), deviceIds});
    isReadOnly = true;
}

void DistributedReadStorage::allowModifications(){
    isReadOnly = false;
    gpulengthStorage.extractCpuLengthStorage(lengthStorage);
    gpulengthStorage = std::move(GPULengthStore{});
}

void DistributedReadStorage::setSequences(read_number firstIndex, read_number lastIndex_excl, const char* data){
    assert(!isReadOnly);

    distributedSequenceData2.setSafe(firstIndex, lastIndex_excl, reinterpret_cast<const unsigned int*>(data));
}

void DistributedReadStorage::setSequences(const std::vector<read_number>& indices, const char* data){
    assert(!isReadOnly);

    distributedSequenceData2.setSafe(indices, reinterpret_cast<const unsigned int*>(data));
}

void DistributedReadStorage::setSequenceLengths(read_number firstIndex, read_number lastIndex_excl, const Length_t* data){
    assert(!isReadOnly);

    //distributedSequenceLengths2.setSafe(firstIndex, lastIndex_excl, data);

    for(read_number i = firstIndex; i < lastIndex_excl; i++){
        lengthStorage.setLength(i, data[i - firstIndex]);
    }
}

void DistributedReadStorage::setSequenceLengths(const std::vector<read_number>& indices, const Length_t* data){
    assert(!isReadOnly);

    //distributedSequenceLengths2.setSafe(indices, data);

    for(std::size_t i = 0; i < indices.size(); i++){
        lengthStorage.setLength(indices[i], data[i]);
    }
}

void DistributedReadStorage::setQualities(read_number firstIndex, read_number lastIndex_excl, const char* data){
    assert(!isReadOnly);

    distributedQualities2.setSafe(firstIndex, lastIndex_excl, data);
}

void DistributedReadStorage::setQualities(const std::vector<read_number>& indices, const char* data){
    assert(!isReadOnly);
    
    distributedQualities2.setSafe(indices, data);
}

DistributedReadStorage::GatherHandleSequences DistributedReadStorage::makeGatherHandleSequences() const{
    return distributedSequenceData2.makeGatherHandle();
}

DistributedReadStorage::GatherHandleLengths DistributedReadStorage::makeGatherHandleLengths() const{
    assert(false);
    return distributedSequenceLengths2.makeGatherHandle();
}

DistributedReadStorage::GatherHandleQualities DistributedReadStorage::makeGatherHandleQualities() const{
    return distributedQualities2.makeGatherHandle();
}

void DistributedReadStorage::gatherSequenceDataToGpuBufferAsync(
                            const DistributedReadStorage::GatherHandleSequences& handle,
                            char* d_sequence_data,
                            size_t out_sequence_pitch,
                            const read_number* h_readIds,
                            const read_number* d_readIds,
                            int nReadIds,
                            int deviceId,
                            cudaStream_t stream,
                            int) const{

    distributedSequenceData2.gatherElementsInGpuMemAsync(handle,
                                                        h_readIds,
                                                        d_readIds,
                                                        nReadIds,
                                                        deviceId,
                                                        (unsigned int*)d_sequence_data,
                                                        out_sequence_pitch,
                                                        stream);

}




void DistributedReadStorage::gatherSequenceLengthsToGpuBufferAsync(
                            const GatherHandleLengths& handle,
                            int* d_lengths,
                            const read_number* h_readIds,
                            const read_number* d_readIds,
                            int nReadIds,
                            int deviceId,
                            cudaStream_t stream,
                            int) const{
    assert(false);
    distributedSequenceLengths2.gatherElementsInGpuMemAsync(handle,
                                                        h_readIds,
                                                        d_readIds,
                                                        nReadIds,
                                                        deviceId,
                                                        d_lengths,
                                                        sizeof(int),
                                                        stream);

}

void DistributedReadStorage::gatherQualitiesToGpuBufferAsync(
                            const GatherHandleQualities& handle,
                            char* d_quality_data,
                            size_t out_quality_pitch,
                            const read_number* h_readIds,
                            const read_number* d_readIds,
                            int nReadIds,
                            int deviceId,
                            cudaStream_t stream,
                            int) const{

    distributedQualities2.gatherElementsInGpuMemAsync(handle,
                                                        h_readIds,
                                                        d_readIds,
                                                        nReadIds,
                                                        deviceId,
                                                        d_quality_data,
                                                        out_quality_pitch,
                                                        stream);

}


std::future<void> DistributedReadStorage::gatherSequenceDataToHostBufferAsync(
                            const GatherHandleSequences& handle,
                            char* h_sequence_data,
                            size_t out_sequence_pitch,
                            const read_number* h_readIds,
                            int nReadIds,
                            int numCpuThreads) const{

    return distributedSequenceData2.gatherElementsInHostMemAsync(handle,
                                                        h_readIds,
                                                        nReadIds,
                                                        (unsigned int*)h_sequence_data,
                                                        out_sequence_pitch);
}

std::future<void> DistributedReadStorage::gatherSequenceLengthsToHostBufferAsync(
                            const GatherHandleLengths& handle,
                            int* h_lengths,
                            const read_number* h_readIds,
                            int nReadIds,
                            int numCpuThreads) const{

    assert(false);

    return distributedSequenceLengths2.gatherElementsInHostMemAsync(handle,
                                                        h_readIds,
                                                        nReadIds,
                                                        h_lengths,
                                                        sizeof(int));

}

std::future<void> DistributedReadStorage::gatherQualitiesToHostBufferAsync(
                            const GatherHandleQualities& handle,
                            char* h_quality_data,
                            size_t out_quality_pitch,
                            const read_number* h_readIds,
                            int nReadIds,
                            int numCpuThreads) const{

    return distributedQualities2.gatherElementsInHostMemAsync(handle,
                                                        h_readIds,
                                                        nReadIds,
                                                        h_quality_data,
                                                        out_quality_pitch);
}


void DistributedReadStorage::gatherSequenceDataToHostBuffer(
                            const GatherHandleSequences& handle,
                            char* h_sequence_data,
                            size_t out_sequence_pitch,
                            const read_number* h_readIds,
                            int nReadIds,
                            int numCpuThreads) const{

    return distributedSequenceData2.gatherElementsInHostMem(handle,
                                                        h_readIds,
                                                        nReadIds,
                                                        (unsigned int*)h_sequence_data,
                                                        out_sequence_pitch);
}

void DistributedReadStorage::gatherSequenceLengthsToHostBuffer(
                            const GatherHandleLengths& handle,
                            int* h_lengths,
                            const read_number* h_readIds,
                            int nReadIds,
                            int numCpuThreads) const{

    assert(false);
    return distributedSequenceLengths2.gatherElementsInHostMem(handle,
                                                        h_readIds,
                                                        nReadIds,
                                                        h_lengths,
                                                        sizeof(int));

}

void DistributedReadStorage::gatherQualitiesToHostBuffer(
                            const GatherHandleQualities& handle,
                            char* h_quality_data,
                            size_t out_quality_pitch,
                            const read_number* h_readIds,
                            int nReadIds,
                            int numCpuThreads) const{

    return distributedQualities2.gatherElementsInHostMem(handle,
                                                        h_readIds,
                                                        nReadIds,
                                                        h_quality_data,
                                                        out_quality_pitch);
}


void DistributedReadStorage::gatherSequenceLengthsToGpuBufferAsyncNew(
                            int* d_lengths,
                            int deviceId,
                            const read_number* d_readIds,
                            int nReadIds,    
                            cudaStream_t stream) const{

    assert(isReadOnly);

    gpulengthStorage.gatherLengthsOnDeviceAsync(d_lengths, 
                                                deviceId,
                                                d_readIds, 
                                                nReadIds,
                                                stream);

}

void DistributedReadStorage::gatherSequenceLengthsToHostBufferNew(
                            int* lengths,
                            const read_number* readIds,
                            int nReadIds) const{

    assert(isReadOnly);

    gpulengthStorage.gatherLengthsOnHost(lengths, readIds, nReadIds);

}


void DistributedReadStorage::saveToFile(const std::string& filename) const{
    std::ofstream stream(filename, std::ios::binary);

    //int ser_id = serialization_id;
    std::size_t lengthsize = sizeof(Length_t);
    stream.write(reinterpret_cast<const char*>(&lengthsize), sizeof(std::size_t));

    stream.write(reinterpret_cast<const char*>(&numberOfReads), sizeof(read_number));
    stream.write(reinterpret_cast<const char*>(&sequenceLengthLowerBound), sizeof(int));
    stream.write(reinterpret_cast<const char*>(&sequenceLengthUpperBound), sizeof(int));
    stream.write(reinterpret_cast<const char*>(&useQualityScores), sizeof(bool));
    stream.write(reinterpret_cast<const char*>(&statistics), sizeof(Statistics));

    constexpr read_number batchsize = 10000000;
    int numBatches = SDIV(numberOfReads, batchsize);

    {
        auto sequencehandle = makeGatherHandleSequences();
        size_t outputpitch = getEncodedNumInts2BitHiLo(sequenceLengthUpperBound) * sizeof(int);

        size_t totalSequenceMemory = outputpitch * numberOfReads;
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

    {
        auto lengthhandle = makeGatherHandleLengths();
        size_t outputpitch = sizeof(Length_t);

        size_t totalLengthMemory = outputpitch * numberOfReads;
        stream.write(reinterpret_cast<const char*>(&totalLengthMemory), sizeof(size_t));

        for(int batch = 0; batch < numBatches; batch++){
            read_number begin = batch * batchsize;
            read_number end = std::min((batch+1) * batchsize, getNumberOfReads());

            std::vector<read_number> indices(end-begin);
            std::iota(indices.begin(), indices.end(), begin);

            size_t databytes = outputpitch * indices.size();
            std::vector<Length_t> data(indices.size(), 0);

            auto future = gatherSequenceLengthsToHostBufferAsync(
                                        lengthhandle,
                                        data.data(),
                                        indices.data(),
                                        indices.size(),
                                        1);

            future.wait();

            stream.write(reinterpret_cast<const char*>(&data[0]), databytes);
        }
    }

    if(useQualityScores){
        auto qualityhandle = makeGatherHandleQualities();
        size_t outputpitch = sequenceLengthUpperBound;

        size_t totalqualityMemory = outputpitch * numberOfReads;
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

void DistributedReadStorage::loadFromFile(const std::string& filename){
    loadFromFile(filename, deviceIds);
}

void DistributedReadStorage::loadFromFile(const std::string& filename, const std::vector<int>& deviceIds_){
    std::ifstream stream(filename, std::ios::binary);
    if(!stream)
        throw std::runtime_error("Cannot open file " + filename);

    destroy();

    std::size_t lengthsize = sizeof(Length_t);
    std::size_t loaded_lengthsize;
    stream.read(reinterpret_cast<char*>(&loaded_lengthsize), sizeof(std::size_t));

    if(loaded_lengthsize != lengthsize)
        throw std::runtime_error("Wrong size of length type!");


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

    stream.read(reinterpret_cast<char*>(&statistics), sizeof(Statistics));

    constexpr read_number batchsize = 10000000;
    int numBatches = SDIV(numberOfReads, batchsize);

    {
        size_t seqpitch = getEncodedNumInts2BitHiLo(sequenceLengthUpperBound) * sizeof(int);

        size_t totalSequenceMemory = 1;
        stream.read(reinterpret_cast<char*>(&totalSequenceMemory), sizeof(size_t));

        size_t totalMemoryRead = 0;

        for(int batch = 0; batch < numBatches; batch++){
            read_number begin = batch * batchsize;
            read_number end = std::min((batch+1) * batchsize, getNumberOfReads());

            size_t databytes = seqpitch * (end-begin);
            std::vector<char> data(databytes, 0);

            stream.read(reinterpret_cast<char*>(&data[0]), databytes);
            totalMemoryRead += stream.gcount();

            assert(totalMemoryRead <= totalSequenceMemory);

            setSequences(begin, end, data.data());
        }

        assert(totalMemoryRead == totalSequenceMemory);
    }

    {
        size_t lengthpitch = sizeof(Length_t);

        size_t totalLengthMemory = 1;
        stream.read(reinterpret_cast<char*>(&totalLengthMemory), sizeof(size_t));

        size_t totalMemoryRead = 0;

        for(int batch = 0; batch < numBatches; batch++){
            read_number begin = batch * batchsize;
            read_number end = std::min((batch+1) * batchsize, getNumberOfReads());

            std::vector<Length_t> data((end-begin), 0);

            size_t databytes = lengthpitch * (end-begin);
            stream.read(reinterpret_cast<char*>(&data[0]), databytes);
            totalMemoryRead += stream.gcount();

            assert(totalMemoryRead <= totalLengthMemory);

            setSequenceLengths(begin, end, data.data());

            auto minmax = std::minmax_element(data.begin(), data.end(), [](const auto& l1, const auto& l2){
                return l1 < l2;
            });

            statistics.minimumSequenceLength = std::min(statistics.minimumSequenceLength, int(*minmax.first));
            statistics.maximumSequenceLength = std::max(statistics.maximumSequenceLength, int(*minmax.second));
        }

        assert(totalMemoryRead == totalLengthMemory);
    }

    if(useQualityScores){
        size_t qualitypitch = sequenceLengthUpperBound;

        size_t totalqualityMemory = 1;
        stream.read(reinterpret_cast<char*>(&totalqualityMemory), sizeof(size_t));

        size_t totalMemoryRead = 0;

        for(int batch = 0; batch < numBatches; batch++){
            read_number begin = batch * batchsize;
            read_number end = std::min((batch+1) * batchsize, getNumberOfReads());

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






}
}

#endif

#endif
