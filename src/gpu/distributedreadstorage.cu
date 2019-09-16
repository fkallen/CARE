#if 1


#include <gpu/distributedreadstorage.hpp>
#include <gpu/distributedarray.hpp>

#include <config.hpp>
#include <sequence.hpp>
#include <sequencefileio.hpp>

#include <fstream>
#include <omp.h>
#include <algorithm>
#include <iterator>

#ifdef __NVCC__

namespace care{
namespace gpu{

DistributedReadStorage::DistributedReadStorage(const std::vector<int>& deviceIds_, read_number nReads, bool b, int maximum_sequence_length){

    init(deviceIds_, nReads, b, maximum_sequence_length);
}

void DistributedReadStorage::init(const std::vector<int>& deviceIds_, read_number nReads, bool b, int maximum_sequence_length){
    deviceIds = deviceIds_;
    numberOfReads = nReads;
    sequenceLengthLimit = maximum_sequence_length;
    useQualityScores = b;

    int numGpus = deviceIds.size();

    if(numberOfReads > 0 && sequenceLengthLimit > 0){
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
                const size_t usableMem = freeMemPerGpu[gpu] > gpuReadStorageHeadroomPerGPU ? freeMemPerGpu[gpu] - gpuReadStorageHeadroomPerGPU : 0;
                maximumUsableBytesPerGpu[gpu] = usableMem;
            }
            std::cerr << "Usable memory per gpu : ";
            std::copy(maximumUsableBytesPerGpu.begin(), maximumUsableBytesPerGpu.end(), std::ostream_iterator<size_t>(std::cerr, " "));
            std::cerr << "\n";
        };

        int oldId; cudaGetDevice(&oldId); CUERR;


        const int intsPerSequence = getEncodedNumInts2BitHiLo(sequenceLengthLimit);

        updateMemoryLimits();
        distributedSequenceData2 = std::move(DistributedArray<unsigned int, read_number>(deviceIds, maximumUsableBytesPerGpu, numberOfReads, intsPerSequence));

        updateMemoryLimits();
        distributedSequenceLengths2 = std::move(DistributedArray<Length_t, read_number>(deviceIds, maximumUsableBytesPerGpu, numberOfReads, 1));

        if(useQualityScores){
            updateMemoryLimits();
            distributedQualities2 = std::move(DistributedArray<char, read_number>(deviceIds, maximumUsableBytesPerGpu, numberOfReads, sequenceLengthLimit));
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
    deviceIds = std::move(rhs.deviceIds);
    numberOfReads = std::move(rhs.numberOfReads);
    sequenceLengthLimit = std::move(rhs.sequenceLengthLimit);
    useQualityScores = std::move(rhs.useQualityScores);
    readIdsOfReadsWithUndeterminedBase = std::move(rhs.readIdsOfReadsWithUndeterminedBase);
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
    sequenceLengthLimit = 0;
    std::vector<size_t> fractions(deviceIds.size(), 0);
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

int DistributedReadStorage::getSequenceLengthLimit() const{
    return sequenceLengthLimit;
}

std::vector<int> DistributedReadStorage::getDeviceIds() const{
    return deviceIds;
}


void DistributedReadStorage::setReads(read_number firstIndex, read_number lastIndex_excl, const std::vector<Read>& reads, int numThreads){
    std::vector<read_number> indices(lastIndex_excl-firstIndex);
    std::iota(indices.begin(), indices.end(), firstIndex);

    setReads(indices, reads, numThreads);
}

void DistributedReadStorage::setReads(const std::vector<read_number>& indices, const std::vector<Read>& reads, int numThreads){

    assert(indices.size() > 0);
    assert(reads.size() == indices.size());
    assert(std::all_of(indices.begin(), indices.end(), [&](auto i){ return i < getNumberOfReads();}));
    assert(std::all_of(reads.begin(), reads.end(), [&](const auto& r){ return Length_t(r.sequence.length()) <= getSequenceLengthLimit();}));
    assert(std::all_of(reads.begin(), reads.end(), [&](const auto& r){ return r.sequence.length() == r.quality.length();}));

    auto minmax = std::minmax_element(reads.begin(), reads.end(), [](const auto& r1, const auto& r2){
        return r1.sequence.length() < r2.sequence.length();
    });

    statistics.minimumSequenceLength = std::min(statistics.minimumSequenceLength, int(minmax.first->sequence.length()));
    statistics.maximumSequenceLength = std::max(statistics.maximumSequenceLength, int(minmax.second->sequence.length()));

    std::vector<char> sequenceData;
    std::vector<Length_t> sequenceLengths;
    std::vector<char> qualityData;

    const size_t numReads = indices.size();

    const size_t encodedSequencePitch = getEncodedNumInts2BitHiLo(getSequenceLengthLimit()) * sizeof(int);
    const size_t qualityPitch = getSequenceLengthLimit();

    sequenceData.resize(encodedSequencePitch * numReads, 0);
    sequenceLengths.resize(numReads, 0);
    if(canUseQualityScores()){
        qualityData.resize(getSequenceLengthLimit() * numReads, 0);
    }

    int oldNumOMPThreads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        oldNumOMPThreads = omp_get_num_threads();
    }

    omp_set_num_threads(numThreads);

    #pragma omp parallel for
    for(size_t i = 0; i < numReads; i++){
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

    omp_set_num_threads(oldNumOMPThreads);

    setSequences(indices, sequenceData.data());
    setSequenceLengths(indices, sequenceLengths.data());
    if(canUseQualityScores()){
        setQualities(indices, qualityData.data());
    }
}

void DistributedReadStorage::setReadContainsN(read_number readId, bool contains){

    auto pos = std::lower_bound(readIdsOfReadsWithUndeterminedBase.begin(),
                                        readIdsOfReadsWithUndeterminedBase.end(),
                                        readId);


    if(contains){
        if(pos != readIdsOfReadsWithUndeterminedBase.end()){
            ; //already marked
        }else{
            readIdsOfReadsWithUndeterminedBase.insert(pos, readId);
        }
    }else{
        if(pos != readIdsOfReadsWithUndeterminedBase.end()){
            //remove mark
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

void DistributedReadStorage::setSequences(read_number firstIndex, read_number lastIndex_excl, const char* data){
    distributedSequenceData2.setSafe(firstIndex, lastIndex_excl, reinterpret_cast<const unsigned int*>(data));
}

void DistributedReadStorage::setSequences(const std::vector<read_number>& indices, const char* data){
    distributedSequenceData2.setSafe(indices, reinterpret_cast<const unsigned int*>(data));
}

void DistributedReadStorage::setSequenceLengths(read_number firstIndex, read_number lastIndex_excl, const Length_t* data){
    distributedSequenceLengths2.setSafe(firstIndex, lastIndex_excl, data);
}

void DistributedReadStorage::setSequenceLengths(const std::vector<read_number>& indices, const Length_t* data){
    distributedSequenceLengths2.setSafe(indices, data);
}

void DistributedReadStorage::setQualities(read_number firstIndex, read_number lastIndex_excl, const char* data){
    distributedQualities2.setSafe(firstIndex, lastIndex_excl, data);
}

void DistributedReadStorage::setQualities(const std::vector<read_number>& indices, const char* data){
    distributedQualities2.setSafe(indices, data);
}

DistributedReadStorage::GatherHandleSequences DistributedReadStorage::makeGatherHandleSequences() const{
    return distributedSequenceData2.makeGatherHandle();
}

DistributedReadStorage::GatherHandleLengths DistributedReadStorage::makeGatherHandleLengths() const{
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
                            int numCpuThreads) const{

    distributedSequenceData2.gatherElementsInGpuMemAsync(handle,
                                                        h_readIds,
                                                        d_readIds,
                                                        nReadIds,
                                                        deviceId,
                                                        (unsigned int*)d_sequence_data,
                                                        out_sequence_pitch,
                                                        stream,
                                                        numCpuThreads);

}




void DistributedReadStorage::gatherSequenceLengthsToGpuBufferAsync(
                            const GatherHandleLengths& handle,
                            int* d_lengths,
                            const read_number* h_readIds,
                            const read_number* d_readIds,
                            int nReadIds,
                            int deviceId,
                            cudaStream_t stream,
                            int numCpuThreads) const{

    distributedSequenceLengths2.gatherElementsInGpuMemAsync(handle,
                                                        h_readIds,
                                                        d_readIds,
                                                        nReadIds,
                                                        deviceId,
                                                        d_lengths,
                                                        sizeof(int),
                                                        stream,
                                                        numCpuThreads);

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
                            int numCpuThreads) const{

    distributedQualities2.gatherElementsInGpuMemAsync(handle,
                                                        h_readIds,
                                                        d_readIds,
                                                        nReadIds,
                                                        deviceId,
                                                        d_quality_data,
                                                        out_quality_pitch,
                                                        stream,
                                                        numCpuThreads);

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


void DistributedReadStorage::saveToFile(const std::string& filename) const{
    std::ofstream stream(filename, std::ios::binary);

    //int ser_id = serialization_id;
    std::size_t lengthsize = sizeof(Length_t);
    stream.write(reinterpret_cast<const char*>(&lengthsize), sizeof(std::size_t));

    stream.write(reinterpret_cast<const char*>(&numberOfReads), sizeof(read_number));
    stream.write(reinterpret_cast<const char*>(&sequenceLengthLimit), sizeof(int));
    stream.write(reinterpret_cast<const char*>(&useQualityScores), sizeof(bool));
    stream.write(reinterpret_cast<const char*>(&statistics), sizeof(Statistics));

    constexpr read_number batchsize = 10000000;
    int numBatches = SDIV(numberOfReads, batchsize);

    {
        auto sequencehandle = makeGatherHandleSequences();
        size_t outputpitch = getEncodedNumInts2BitHiLo(sequenceLengthLimit) * sizeof(int);

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
        size_t outputpitch = sequenceLengthLimit;

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
    int loaded_sequenceLengthLimit;
    bool loaded_useQualityScores;

    stream.read(reinterpret_cast<char*>(&loaded_numberOfReads), sizeof(read_number));
    stream.read(reinterpret_cast<char*>(&loaded_sequenceLengthLimit), sizeof(int));
    stream.read(reinterpret_cast<char*>(&loaded_useQualityScores), sizeof(bool));

    init(deviceIds_, loaded_numberOfReads, loaded_useQualityScores, loaded_sequenceLengthLimit);

    stream.read(reinterpret_cast<char*>(&statistics), sizeof(Statistics));

    constexpr read_number batchsize = 10000000;
    int numBatches = SDIV(numberOfReads, batchsize);

    {
        size_t seqpitch = getEncodedNumInts2BitHiLo(sequenceLengthLimit) * sizeof(int);

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
        size_t qualitypitch = sequenceLengthLimit;

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
