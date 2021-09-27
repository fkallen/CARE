#ifndef CARE_CHUNKEDREADSTORAGE_HPP
#define CARE_CHUNKEDREADSTORAGE_HPP

#include <util.hpp>
#include <config.hpp>
#include <threadpool.hpp>
#include <readlibraryio.hpp>
#include <sequencehelpers.hpp>
#include <dynamic2darray.hpp>
#include <concurrencyhelpers.hpp>
#include <lengthstorage.hpp>
#include <cpureadstorage.hpp>
#include <memorymanagement.hpp>

#include <unordered_set>
#include <vector>
#include <array>
#include <mutex>
#include <algorithm>
#include <string>
#include <iostream>
#include <future>
#include <limits>
#include <map>
#include <set>
#include <numeric>

namespace care{

class ChunkedReadStorage : public CpuReadStorage{
public:
    ChunkedReadStorage(bool pairedEnd_, bool hasQualityScores_) 
    : pairedEnd(pairedEnd_), hasQualityScores(hasQualityScores_){
        offsetsPrefixSum.emplace_back(0);
    }

    ChunkedReadStorage(const ChunkedReadStorage&) = delete;
    ChunkedReadStorage& operator=(const ChunkedReadStorage&) = delete;

    ChunkedReadStorage(ChunkedReadStorage&&) = default;
    ChunkedReadStorage& operator=(ChunkedReadStorage&&) = default;

    void appendConsecutiveReads(
        read_number firstReadId,
        int numReads,
        std::vector<int> sequenceLengths,
        std::vector<unsigned int> encodedSequences,
        std::size_t sequencePitchInInts,
        std::vector<char> qualities,
        std::size_t qualityPitchBytes
    ){
        StoredEncodedSequencesAppend s;
        s.firstReadId = firstReadId;
        s.numReads = numReads;
        s.data.encodedSequencePitchInInts = sequencePitchInInts;
        s.data.encodedSequences = std::move(encodedSequences);

        sequenceStorageAppend.emplace_back(std::move(s));

        StoredSequenceLengthsAppend l;
        l.firstReadId = firstReadId;
        l.numReads = numReads;
        l.sequenceLengths = std::move(sequenceLengths);

        lengthdataAppend.emplace_back(std::move(l));

        if(canUseQualityScores()){

            StoredQualitiesAppend q;
            q.firstReadId = firstReadId;
            q.numReads = numReads;
            q.data.qualityPitchInBytes = qualityPitchBytes;
            q.data.qualities = std::move(qualities);

            qualityStorageAppend.emplace_back(std::move(q));
        }

        totalNumberOfReads += numReads;        
    }

    void appendAmbiguousReadIds(
        const std::vector<read_number>& ambiguousIds
    ){
        ambigReadIds.insert(ambiguousIds.begin(), ambiguousIds.end());
    }

    void appendingFinished(
        std::size_t memoryLimitBytes
    ){
        auto deallocVector = [](auto& vec){
            using W = typename std::remove_reference<decltype(vec)>::type;
            W tmp{};
            vec.swap(tmp);
        };

        if(!canUseQualityScores()){
            deallocVector(qualityStorage);
        }


        auto segmentLessThan = [](const auto& l, const auto& r){
            return l.firstReadId < r.firstReadId;            
        };

        std::sort(sequenceStorageAppend.begin(), sequenceStorageAppend.end(), segmentLessThan);
        std::sort(lengthdataAppend.begin(), lengthdataAppend.end(), segmentLessThan);
        if(canUseQualityScores()){
            std::sort(qualityStorageAppend.begin(), qualityStorageAppend.end(), segmentLessThan);
        }

        const std::size_t chunksTotal = sequenceStorageAppend.size();
        sequenceStorage.reserve(chunksTotal);
        if(canUseQualityScores()){
            qualityStorage.reserve(chunksTotal);
        }

        for(std::size_t chunk = 0; chunk < sequenceStorageAppend.size(); chunk++){
            offsetsPrefixSum.emplace_back(offsetsPrefixSum.back() + sequenceStorageAppend[chunk].numReads);

            sequenceStorage.emplace_back(std::move(sequenceStorageAppend[chunk].data));
            if(canUseQualityScores()){
                qualityStorage.emplace_back(std::move(qualityStorageAppend[chunk].data));
            }
        }

        int minLength = std::numeric_limits<int>::max();
        int maxLength = 0;

        for(const auto& s : lengthdataAppend){
            auto minmax = std::minmax_element(s.sequenceLengths.begin(), s.sequenceLengths.end());
            minLength = std::min(*minmax.first, minLength);
            maxLength = std::max(*minmax.second, maxLength);
        }

        lengthStorage = std::move(LengthStore<std::uint32_t>(minLength, maxLength, totalNumberOfReads));

        for(std::size_t chunk = 0; chunk < lengthdataAppend.size(); chunk++){
            const auto& s = lengthdataAppend[chunk];
            const std::size_t offset = offsetsPrefixSum[chunk];
            for(std::size_t i = 0; i < s.sequenceLengths.size(); i++){
                const std::size_t index = offset + i;
                lengthStorage.setLength(index, s.sequenceLengths[i]);
            }
        }
        deallocVector(lengthdataAppend);
        deallocVector(sequenceStorageAppend);
        deallocVector(qualityStorageAppend);



        compactSequences(memoryLimitBytes);

        if(canUseQualityScores()){
            compactQualities(memoryLimitBytes);
        }

    }

    void loadFromFile(const std::string& filename){
        std::ifstream stream(filename, std::ios::binary);
        if(!stream)
            throw std::runtime_error("Cannot open file " + filename);

        destroy();

        std::size_t loaded_numreads = 0;
        int loaded_sequenceLengthLowerBound = 0;
        int loaded_sequenceLengthUpperBound = 0;
        bool loaded_hasQualityScores = false;        

        stream.read(reinterpret_cast<char*>(&loaded_numreads), sizeof(std::size_t));
        stream.read(reinterpret_cast<char*>(&loaded_sequenceLengthLowerBound), sizeof(int));
        stream.read(reinterpret_cast<char*>(&loaded_sequenceLengthUpperBound), sizeof(int));            
        stream.read(reinterpret_cast<char*>(&loaded_hasQualityScores), sizeof(bool));

        std::size_t lengthsBytes = 0;
        std::size_t sequencesBytes = 0;      
        std::size_t qualitiesBytes = 0;       
        std::size_t ambigBytes = 0;

        stream.read(reinterpret_cast<char*>(&lengthsBytes), sizeof(std::size_t));
        stream.read(reinterpret_cast<char*>(&sequencesBytes), sizeof(std::size_t));
        stream.read(reinterpret_cast<char*>(&qualitiesBytes), sizeof(std::size_t));
        stream.read(reinterpret_cast<char*>(&ambigBytes), sizeof(std::size_t));

        totalNumberOfReads = loaded_numreads;
        offsetsPrefixSum = {0, totalNumberOfReads};

        lengthStorage.readFromStream(stream);

        hasShrinkedSequences = true;
        stream.read(reinterpret_cast<char*>(&encodedSequencePitchInInts), sizeof(std::size_t));
        std::size_t numsequencedataelements = 0;
        stream.read(reinterpret_cast<char*>(&numsequencedataelements), sizeof(std::size_t));
        shrinkedEncodedSequences.resize(numsequencedataelements);
        stream.read(reinterpret_cast<char*>(shrinkedEncodedSequences.data()), numsequencedataelements * sizeof(unsigned int));


        if(canUseQualityScores() && loaded_hasQualityScores){
            //std::cerr << "load qualities\n";

            hasShrinkedQualities = true;
            stream.read(reinterpret_cast<char*>(&qualityPitchInBytes), sizeof(std::size_t));
            std::size_t numqualitydataelements = 0;
            stream.read(reinterpret_cast<char*>(&numqualitydataelements), sizeof(std::size_t));
            shrinkedQualities.resize(numqualitydataelements);
            stream.read(reinterpret_cast<char*>(shrinkedQualities.data()), numqualitydataelements * sizeof(char));

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
        

        std::size_t numAmbigDataElements = 0;
        stream.read(reinterpret_cast<char*>(&numAmbigDataElements), sizeof(std::size_t));

        std::vector<read_number> tmpambig(numAmbigDataElements);
        stream.read(reinterpret_cast<char*>(tmpambig.data()), numAmbigDataElements * sizeof(read_number));

        ambigReadIds.insert(tmpambig.begin(), tmpambig.end());
    }


    void saveToFile(const std::string& filename) const{
        std::ofstream stream(filename, std::ios::binary);

        std::size_t numReads = getNumberOfReads();
        int sequenceLengthUpperBound = getSequenceLengthUpperBound();
        int sequenceLengthLowerBound = getSequenceLengthLowerBound();
        bool qual = canUseQualityScores();

        stream.write(reinterpret_cast<const char*>(&numReads), sizeof(std::size_t));
        stream.write(reinterpret_cast<const char*>(&sequenceLengthUpperBound), sizeof(int));
        stream.write(reinterpret_cast<const char*>(&sequenceLengthLowerBound), sizeof(int));
        stream.write(reinterpret_cast<const char*>(&qual), sizeof(bool));

        auto pos = stream.tellp();

        stream.seekp(sizeof(std::size_t) * 4, std::ios_base::cur);

        std::size_t lengthsBytes = lengthStorage.writeToStream(stream);

        std::size_t writtenSequenceBytes = 0;
        if(hasShrinkedSequences){

            std::size_t pitch = encodedSequencePitchInInts;
            stream.write(reinterpret_cast<const char*>(&pitch), sizeof(std::size_t));
            writtenSequenceBytes += sizeof(std::size_t); //pitch

            std::size_t numelements = shrinkedEncodedSequences.size();
            stream.write(reinterpret_cast<const char*>(&numelements), sizeof(std::size_t));
            writtenSequenceBytes += sizeof(std::size_t); //numdataelements

            stream.write(reinterpret_cast<const char*>(shrinkedEncodedSequences.data()), numelements * sizeof(unsigned int));            
            writtenSequenceBytes += numelements * sizeof(unsigned int); //dataelements

            assert(numelements == getNumberOfReads() * pitch);
        }else{

            std::size_t pitch = 0;
            for(const auto& s : sequenceStorage){
                pitch = std::max(pitch, s.encodedSequencePitchInInts);
            }
            stream.write(reinterpret_cast<const char*>(&pitch), sizeof(std::size_t));
            writtenSequenceBytes += sizeof(std::size_t); //pitch

            auto numelementspos = stream.tellp();
            stream.seekp(sizeof(std::size_t) * 1, std::ios_base::cur);

            std::size_t numelements = 0;

            for(std::size_t c = 0; c < sequenceStorage.size(); c++){
                const auto& s = sequenceStorage[c];
                const std::size_t numsequencesInChunk = s.encodedSequences.size() / s.encodedSequencePitchInInts;
                

                if(s.encodedSequencePitchInInts == pitch){
                    stream.write(reinterpret_cast<const char*>(s.encodedSequences.data()), s.encodedSequences.size() * sizeof(unsigned int));
                    writtenSequenceBytes += s.encodedSequences.size() * sizeof(unsigned int); //dataelements
                    numelements += s.encodedSequences.size();
                }else{
                    std::vector<unsigned int> temp(pitch);

                    for(std::size_t i = 0; i < numsequencesInChunk; i++){
                        std::copy_n(s.encodedSequences.data() + i * s.encodedSequencePitchInInts, s.encodedSequencePitchInInts, temp.begin());
                        stream.write(reinterpret_cast<const char*>(temp.data()), temp.size() * sizeof(unsigned int));
                        writtenSequenceBytes += temp.size() * sizeof(unsigned int); //dataelements
                        numelements += temp.size();
                    }
                }
            }

            assert(numelements == getNumberOfReads() * pitch);

            auto curpos = stream.tellp();
            stream.seekp(numelementspos);
            stream.write(reinterpret_cast<const char*>(&numelements), sizeof(std::size_t));
            writtenSequenceBytes += sizeof(std::size_t); //numdataelements
            stream.seekp(curpos);
        }


        std::size_t writtenQualityBytes = 0;
        if(hasShrinkedQualities){

            std::size_t pitch = qualityPitchInBytes;
            stream.write(reinterpret_cast<const char*>(&pitch), sizeof(std::size_t));
            writtenQualityBytes += sizeof(std::size_t); //pitch

            std::size_t numelements = shrinkedQualities.size();
            stream.write(reinterpret_cast<const char*>(&numelements), sizeof(std::size_t));
            writtenQualityBytes += sizeof(std::size_t); //numdataelements

            stream.write(reinterpret_cast<const char*>(shrinkedQualities.data()), numelements * sizeof(char));            
            writtenQualityBytes += numelements * sizeof(char); //dataelements

            assert(numelements == getNumberOfReads() * pitch);
        }else{

            std::size_t pitch = 0;
            for(const auto& q : qualityStorage){
                pitch = std::max(pitch, q.qualityPitchInBytes);
            }
            stream.write(reinterpret_cast<const char*>(&pitch), sizeof(std::size_t));
            writtenQualityBytes += sizeof(std::size_t); //pitch

            auto numelementspos = stream.tellp();
            stream.seekp(sizeof(std::size_t) * 1, std::ios_base::cur);

            std::size_t numelements = 0;

            for(std::size_t c = 0; c < qualityStorage.size(); c++){
                const auto& q = qualityStorage[c];
                const std::size_t numsequencesInChunk = q.qualities.size() / q.qualityPitchInBytes;
                

                if(q.qualityPitchInBytes == pitch){
                    stream.write(reinterpret_cast<const char*>(q.qualities.data()), q.qualities.size() * sizeof(char));
                    writtenQualityBytes += q.qualities.size() * sizeof(char); //dataelements
                    numelements += q.qualities.size();
                }else{
                    std::vector<char> temp(pitch);

                    for(std::size_t i = 0; i < numsequencesInChunk; i++){
                        std::copy_n(q.qualities.data() + i * q.qualityPitchInBytes, q.qualityPitchInBytes, temp.begin());
                        stream.write(reinterpret_cast<const char*>(temp.data()), temp.size() * sizeof(char));
                        writtenQualityBytes += temp.size() * sizeof(char); //dataelements
                        numelements += temp.size();
                    }
                }
            }

            assert(numelements == getNumberOfReads() * pitch);

            auto curpos = stream.tellp();
            stream.seekp(numelementspos);
            stream.write(reinterpret_cast<const char*>(&numelements), sizeof(std::size_t));
            writtenQualityBytes += sizeof(std::size_t); //numdataelements
            stream.seekp(curpos);
        }

        std::size_t numUndeterminedReads = ambigReadIds.size();
        std::vector<read_number> ambigtemp(ambigReadIds.begin(), ambigReadIds.end());

        stream.write(reinterpret_cast<const char*>(&numUndeterminedReads), sizeof(size_t));
        stream.write(reinterpret_cast<const char*>(ambigtemp.data()), numUndeterminedReads * sizeof(read_number));

        std::size_t ambigBytes = sizeof(std::size_t) + numUndeterminedReads * sizeof(read_number);

        stream.seekp(pos);
        stream.write(reinterpret_cast<const char*>(&lengthsBytes), sizeof(std::size_t));
        stream.write(reinterpret_cast<const char*>(&writtenSequenceBytes), sizeof(std::size_t));
        stream.write(reinterpret_cast<const char*>(&writtenQualityBytes), sizeof(std::size_t));
        stream.write(reinterpret_cast<const char*>(&ambigBytes), sizeof(std::size_t));

    }

public: //inherited interface

    void areSequencesAmbiguous(
        bool* result, 
        const read_number* readIds, 
        int numSequences
    ) const override{
        if(numSequences > 0){
            if(getNumberOfReadsWithN() > 0){
                for(int i = 0; i < numSequences; i++){
                    auto it = ambigReadIds.find(readIds[i]);
                    result[i] = (it != ambigReadIds.end());
                }
            }else{
                // if there are no stored reads with ambiguous bases, simply fill output with false
                std::fill(result, result + numSequences, false);
            }
        }else{
            //output buffer is empty
        }
    }

    void gatherSequences(
        unsigned int* sequence_data,
        size_t outSequencePitchInInts,
        const read_number* readIds,
        int numSequences
    ) const override{
        if(numSequences == 0){
            return;
        }

        constexpr int prefetch_distance = 4;

        if(hasShrinkedSequences){
            for(int i = 0; i < numSequences && i < prefetch_distance; ++i) {
                const int index = i;
                const std::size_t nextReadId = readIds[index];
                const unsigned int* const nextData = shrinkedEncodedSequences.data() + encodedSequencePitchInInts * nextReadId;
                __builtin_prefetch(nextData, 0, 0);
            }

            std::size_t destinationPitchBytes = outSequencePitchInInts * sizeof(unsigned int);

            for(int i = 0; i < numSequences; i++){
                if(i + prefetch_distance < numSequences) {
                    const int index = i + prefetch_distance;
                    const std::size_t nextReadId = readIds[index];
                    const unsigned int* const nextData = shrinkedEncodedSequences.data() + encodedSequencePitchInInts * nextReadId;
                    __builtin_prefetch(nextData, 0, 0);
                }

                const std::size_t readId = readIds[i];

                const unsigned int* const data = shrinkedEncodedSequences.data() + encodedSequencePitchInInts * readId;

                unsigned int* const destData = (unsigned int*)(((char*)sequence_data) + destinationPitchBytes * i);
                std::copy_n(data, encodedSequencePitchInInts, destData);
            }
        }else{

            for(int i = 0; i < numSequences && i < prefetch_distance; ++i) {
                const int index = i;
                const std::size_t nextReadId = readIds[index];
                const unsigned int* const nextData = getPointerToSequenceRow(nextReadId);
                __builtin_prefetch(nextData, 0, 0);
            }

            std::size_t destinationPitchBytes = outSequencePitchInInts * sizeof(unsigned int);

            for(int i = 0; i < numSequences; i++){
                if(i + prefetch_distance < numSequences) {
                    const int index = i + prefetch_distance;
                    const std::size_t nextReadId = readIds[index];
                    const unsigned int* const nextData = getPointerToSequenceRow(nextReadId);
                    __builtin_prefetch(nextData, 0, 0);
                }

                const std::size_t readId = readIds[i];
                const std::size_t chunkIndex = getChunkIndexOfRow(readId);
                const std::size_t rowInChunk = getRowIndexInChunk(chunkIndex, readId);

                const unsigned int* const data = sequenceStorage[chunkIndex].encodedSequences.data()
                    + rowInChunk * sequenceStorage[chunkIndex].encodedSequencePitchInInts;

                unsigned int* const destData = (unsigned int*)(((char*)sequence_data) + destinationPitchBytes * i);
                std::copy_n(data, sequenceStorage[chunkIndex].encodedSequencePitchInInts, destData);
            }

        }
    }

    void gatherQualities(
        char* quality_data,
        size_t out_quality_pitch,
        const read_number* readIds,
        int numSequences
    ) const override{
        if(numSequences == 0){
            return;
        }

        constexpr int prefetch_distance = 4;

        if(hasShrinkedQualities){
            for(int i = 0; i < numSequences && i < prefetch_distance; ++i) {
                const int index = i;
                const std::size_t nextReadId = readIds[index];
                const char* const nextData = shrinkedQualities.data() + qualityPitchInBytes * nextReadId;
                __builtin_prefetch(nextData, 0, 0);
            }

            std::size_t destinationPitchBytes = out_quality_pitch * sizeof(char);

            for(int i = 0; i < numSequences; i++){
                if(i + prefetch_distance < numSequences) {
                    const int index = i + prefetch_distance;
                    const std::size_t nextReadId = readIds[index];
                    const char* const nextData = shrinkedQualities.data() + qualityPitchInBytes * nextReadId;
                    __builtin_prefetch(nextData, 0, 0);
                }

                const std::size_t readId = readIds[i];

                const char* const data = shrinkedQualities.data() + qualityPitchInBytes * readId;

                char* const destData = (char*)(((char*)quality_data) + destinationPitchBytes * i);
                std::copy_n(data, qualityPitchInBytes, destData);
            }
        }else{

            for(int i = 0; i < numSequences && i < prefetch_distance; ++i) {
                const int index = i;
                const std::size_t nextReadId = readIds[index];
                const char* const nextData = getPointerToQualityRow(nextReadId);
                __builtin_prefetch(nextData, 0, 0);
            }

            std::size_t destinationPitchBytes = out_quality_pitch * sizeof(char);

            for(int i = 0; i < numSequences; i++){
                if(i + prefetch_distance < numSequences) {
                    const int index = i + prefetch_distance;
                    const std::size_t nextReadId = readIds[index];
                    const char* const nextData = getPointerToQualityRow(nextReadId);
                    __builtin_prefetch(nextData, 0, 0);
                }

                const std::size_t readId = readIds[i];
                const std::size_t chunkIndex = getChunkIndexOfRow(readId);
                const std::size_t rowInChunk = getRowIndexInChunk(chunkIndex, readId);

                const char* const data = qualityStorage[chunkIndex].qualities.data()
                    + rowInChunk * qualityStorage[chunkIndex].qualityPitchInBytes;

                char* const destData = (char*)(((char*)quality_data) + destinationPitchBytes * i);
                std::copy_n(data, qualityStorage[chunkIndex].qualityPitchInBytes, destData);
            }

        }
    }

    void gatherSequenceLengths(
        int* lengths,
        const read_number* readIds,
        int numSequences
    ) const override{
        if(numSequences == 0) return;
        
        for(int i = 0; i < numSequences; i++){
            lengths[i] = lengthStorage.getLength(readIds[i]);
        }
    }

    void getIdsOfAmbiguousReads(
        read_number* ids
    ) const override{
        std::copy(ambigReadIds.begin(), ambigReadIds.end(), ids);
    }

    std::int64_t getNumberOfReadsWithN() const override{
        return ambigReadIds.size();
    }

    MemoryUsage getMemoryInfo() const override{

        MemoryUsage result{};
        result += lengthStorage.getMemoryInfo();

        result.host += sizeof(std::size_t) * offsetsPrefixSum.capacity();
        result.host += sizeof(unsigned int) * shrinkedEncodedSequences.capacity();
        result.host += sizeof(char) * shrinkedQualities.capacity();

        result.host += sizeof(StoredEncodedSequences) * sequenceStorage.capacity();
        result.host += sizeof(StoredQualities) * qualityStorage.capacity();
        result.host += sizeof(read_number) * ambigReadIds.size();

        for(const auto& s : sequenceStorage){
            result.host += sizeof(unsigned int) * s.encodedSequences.capacity();
        }
        for(const auto& s : qualityStorage){
            result.host += sizeof(char) * s.qualities.capacity();
        }

        result.host += sizeof(StoredEncodedSequencesAppend) * lengthdataAppend.capacity();
        result.host += sizeof(StoredSequenceLengthsAppend) * sequenceStorageAppend.capacity();
        result.host += sizeof(StoredQualitiesAppend) * qualityStorageAppend.size();

        for(const auto& s : lengthdataAppend){
            result.host += sizeof(int) * s.sequenceLengths.capacity();
        }
        for(const auto& s : sequenceStorageAppend){
            result.host += sizeof(unsigned int) * s.data.encodedSequences.capacity();
        }
        for(const auto& s : qualityStorageAppend){
            result.host += sizeof(char) * s.data.qualities.capacity();
        }

        return result;
    }

    read_number getNumberOfReads() const override{
        return totalNumberOfReads;
    }

    bool canUseQualityScores() const override{
        return hasQualityScores;
    }

    int getSequenceLengthLowerBound() const override{
        return lengthStorage.getMinLength();
    }

    int getSequenceLengthUpperBound() const override{
        return lengthStorage.getMaxLength();
    }

    bool isPairedEnd() const override{
        return pairedEnd;
    }

    void destroy() {
        auto deallocVector = [](auto& vec){
            using T = typename std::remove_reference<decltype(vec)>::type;
            T tmp{};
            vec.swap(tmp);
        };

        lengthStorage.destroy();

        deallocVector(offsetsPrefixSum);
        deallocVector(sequenceStorage);
        deallocVector(qualityStorage);
        deallocVector(ambigReadIds);
        deallocVector(shrinkedEncodedSequences);
        deallocVector(shrinkedQualities);
        //deallocVector(tempdataVector);

        hasShrinkedSequences = false;
        encodedSequencePitchInInts = 0;
        hasShrinkedQualities = false;
        qualityPitchInBytes = 0;

        counter = 0;

        offsetsPrefixSum.emplace_back(0);
    }

public:


    bool compactSequences(std::size_t& availableMem){
        std::size_t maxLength = lengthStorage.getMaxLength();
        std::size_t numSequences = totalNumberOfReads;

        encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(maxLength);

        if(availableMem >= sizeof(unsigned int) * numSequences * encodedSequencePitchInInts){
            shrinkedEncodedSequences.resize(numSequences * encodedSequencePitchInInts);
            availableMem -= sizeof(unsigned int) * numSequences * encodedSequencePitchInInts;

            for(std::size_t chunk = 0; chunk < sequenceStorage.size(); chunk++){
                const auto& s = sequenceStorage[chunk];
                const std::size_t offset = offsetsPrefixSum[chunk];
                const std::size_t pitchInts = s.encodedSequencePitchInInts;
                const std::size_t num = s.encodedSequences.size() / pitchInts;

                if(pitchInts != encodedSequencePitchInInts){

                    for(std::size_t i = 0; i < num; i++){
                        std::copy(
                            s.encodedSequences.begin() + i * pitchInts,
                            s.encodedSequences.begin() + (i+1) * pitchInts,
                            shrinkedEncodedSequences.begin() + (offset + i) * encodedSequencePitchInInts
                        );
                    }
                }else{
                    std::copy(s.encodedSequences.begin(), s.encodedSequences.end(), shrinkedEncodedSequences.begin() + offset * encodedSequencePitchInInts);
                }

                availableMem += pitchInts * num * sizeof(unsigned int);
            }

            auto deallocVector = [](auto& vec){
                using W = typename std::remove_reference<decltype(vec)>::type;
                W tmp{};
                vec.swap(tmp);
            };

            deallocVector(sequenceStorage);

            hasShrinkedSequences = true;
            //std::cerr << "shrinked sequences\n";

            return true;
        }else{
            return false;
        }
    }

    bool compactQualities(std::size_t& availableMem){
        std::size_t maxLength = lengthStorage.getMaxLength();
        std::size_t numSequences = totalNumberOfReads;

        qualityPitchInBytes = maxLength;

        if(availableMem >= numSequences * qualityPitchInBytes){
            shrinkedQualities.resize(numSequences * qualityPitchInBytes);
            availableMem -= numSequences * qualityPitchInBytes;

            for(std::size_t chunk = 0; chunk < qualityStorage.size(); chunk++){
                const auto& s = qualityStorage[chunk];
                const std::size_t offset = offsetsPrefixSum[chunk];
                const std::size_t pitchBytes = s.qualityPitchInBytes;
                const std::size_t num = s.qualities.size() / pitchBytes;

                if(pitchBytes != qualityPitchInBytes){

                    for(std::size_t i = 0; i < num; i++){
                        std::copy(
                            s.qualities.begin() + i * pitchBytes,
                            s.qualities.begin() + (i+1) * pitchBytes,
                            shrinkedQualities.begin() + (offset + i) * qualityPitchInBytes
                        );
                    }
                }else{
                    std::copy(s.qualities.begin(), s.qualities.end(), shrinkedQualities.begin() + offset * qualityPitchInBytes);
                }

                availableMem += pitchBytes * num;
            }

            auto deallocVector = [](auto& vec){
                using W = typename std::remove_reference<decltype(vec)>::type;
                W tmp{};
                vec.swap(tmp);
            };
            deallocVector(qualityStorage);

            hasShrinkedQualities = true;
            //std::cerr << "shrinked qualities\n";

            return true;
        }else{
            return false;
        }
    }

   

    void printAmbig(){

        std::vector<read_number> vec(ambigReadIds.begin(), ambigReadIds.end());
        std::sort(vec.begin(), vec.end());
        for(auto x : vec){
            std::cerr << x << " ";
        }
        std::cerr << "\n";
    }

    

private:
    std::size_t getChunkIndexOfRow(std::size_t row) const noexcept{
        auto it = std::lower_bound(offsetsPrefixSum.begin(), offsetsPrefixSum.end(), row + 1);
        std::size_t chunkIndex = std::distance(offsetsPrefixSum.begin(), it) - 1;

        return chunkIndex;
    }

    std::size_t getRowIndexInChunk(std::size_t chunkIndex, std::size_t row) const noexcept{
        return row - offsetsPrefixSum[chunkIndex];
    }

    const unsigned int* getPointerToSequenceRow(std::size_t row) const noexcept{
        const std::size_t chunkIndex = getChunkIndexOfRow(row);
        const std::size_t rowInChunk = getRowIndexInChunk(chunkIndex, row);

        const unsigned int* const data = sequenceStorage[chunkIndex].encodedSequences.data()
            + rowInChunk * sequenceStorage[chunkIndex].encodedSequencePitchInInts;

        return data;
    }

    const char* getPointerToQualityRow(std::size_t row) const noexcept{
        const std::size_t chunkIndex = getChunkIndexOfRow(row);
        const std::size_t rowInChunk = getRowIndexInChunk(chunkIndex, row);

        const char* const data = qualityStorage[chunkIndex].qualities.data()
            + rowInChunk * qualityStorage[chunkIndex].qualityPitchInBytes;

        return data;
    }

    struct StoredEncodedSequences{
        std::size_t encodedSequencePitchInInts = 0;
        std::vector<unsigned int> encodedSequences{};
    };

    struct StoredQualities{
        std::size_t qualityPitchInBytes = 0;
        std::vector<char> qualities{};
    };

    struct StoredEncodedSequencesAppend{
        read_number firstReadId = 0;
        int numReads = 0;
        StoredEncodedSequences data;
    };

    struct StoredSequenceLengthsAppend{
        read_number firstReadId = 0;
        int numReads = 0;
        std::vector<int> sequenceLengths{};
    };

    struct StoredQualitiesAppend{
        read_number firstReadId = 0;
        int numReads = 0;
        StoredQualities data;
    };

    bool pairedEnd{};

    bool hasQualityScores{};
    std::size_t totalNumberOfReads{};

    std::vector<std::size_t> offsetsPrefixSum{};
    std::vector<StoredEncodedSequences> sequenceStorage{};
    LengthStore<std::uint32_t> lengthStorage{};
    std::vector<StoredQualities> qualityStorage{};
    std::unordered_set<read_number> ambigReadIds{};

    std::vector<StoredSequenceLengthsAppend> lengthdataAppend{};
    std::vector<StoredEncodedSequencesAppend> sequenceStorageAppend{};
    std::vector<StoredQualitiesAppend> qualityStorageAppend{};

    bool hasShrinkedSequences = false;
    std::size_t encodedSequencePitchInInts{};
    std::vector<unsigned int> shrinkedEncodedSequences{};

    bool hasShrinkedQualities = false;
    std::size_t qualityPitchInBytes{};
    std::vector<char> shrinkedQualities{};

    mutable int counter = 0;
    mutable SharedMutex sharedmutex{};
    //mutable std::vector<std::unique_ptr<TempData>> tempdataVector{};
};


}





#endif