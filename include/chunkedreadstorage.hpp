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

namespace care{

class ChunkedReadStorage{
public:
    struct StoredEncodedSequences{
        int fileId = 0;
        int batchId = 0;
        std::size_t encodedSequencePitchInInts = 0;
        std::vector<unsigned int> encodedSequences{};
    };

    struct StoredSequenceLengths{
        int fileId = 0;
        int batchId = 0;
        std::vector<int> sequenceLengths{};
    };

    struct StoredQualities{
        int fileId = 0;
        int batchId = 0;
        std::size_t qualityPitchInBytes = 0;
        std::vector<char> qualities{};
    };

    struct StoredAmbigIds{
        int fileId = 0;
        int batchId = 0;
        std::vector<read_number> ids{};
    };
public:
    ChunkedReadStorage(bool hasQualityScores_ = false) : hasQualityScores(hasQualityScores_){

    }

    bool canUseQualityScores() const{
        return hasQualityScores;
    }

    void init(
        std::vector<std::size_t>&& numReadsPerFile_,
        std::vector<StoredEncodedSequences>&& sequenceStorage_,
        std::vector<StoredSequenceLengths>&& lengthStorage_,
        std::vector<StoredQualities>&& qualityStorage_,
        std::vector<StoredAmbigIds>&& ambigIds_,
        std::size_t memoryLimitBytes
    ){
        auto deallocVector = [](auto& vec){
            using W = typename std::remove_reference<decltype(vec)>::type;
            W tmp{};
            vec.swap(tmp);
        };

        numReadsPerFile = std::move(numReadsPerFile_);
        sequenceStorage = std::move(sequenceStorage_);
        std::vector<StoredSequenceLengths> lengthdata = std::move(lengthStorage_);
        qualityStorage = std::move(qualityStorage_);

        if(!canUseQualityScores()){
            deallocVector(qualityStorage);
        }

        std::vector<StoredAmbigIds> ambigStorage = std::move(ambigIds_);

        auto lessThanFileAndBatch = [](const auto& l, const auto& r){
            if(l.fileId < r.fileId) return true;
            if(l.fileId > r.fileId) return false;
            return l.batchId < r.batchId;
        };

        std::sort(sequenceStorage.begin(), sequenceStorage.end(), lessThanFileAndBatch);
        std::sort(lengthdata.begin(), lengthdata.end(), lessThanFileAndBatch);
        if(canUseQualityScores()){
            std::sort(qualityStorage.begin(), qualityStorage.end(), lessThanFileAndBatch);
        }
        std::sort(ambigStorage.begin(), ambigStorage.end(), lessThanFileAndBatch);

        for(const auto& s : ambigStorage){
            ambigReadIdsPerFile[s.fileId].insert(s.ids.begin(), s.ids.end());
        }

        offsetsPrefixSum.reserve(sequenceStorage.size() + 1);
        offsetsPrefixSum.emplace_back(0);

        int minLength = std::numeric_limits<int>::max();
        int maxLength = 0;

        for(const auto& s : lengthdata){
            std::size_t num = offsetsPrefixSum.back() + s.sequenceLengths.size();
            offsetsPrefixSum.emplace_back(num);

            auto minmax = std::minmax_element(s.sequenceLengths.begin(), s.sequenceLengths.end());
            minLength = std::min(*minmax.first, minLength);
            maxLength = std::max(*minmax.second, maxLength);
        }

        lengthStorage = std::move(LengthStore<std::uint32_t>(minLength, maxLength, getNumSequences()));

        for(std::size_t chunk = 0; chunk < lengthdata.size(); chunk++){
            const auto& s = lengthdata[chunk];
            const std::size_t offset = offsetsPrefixSum[chunk];
            for(std::size_t i = 0; i < s.sequenceLengths.size(); i++){
                const std::size_t index = offset + i;
                lengthStorage.setLength(index, s.sequenceLengths[i]);
            }
        }

        compactSequences(memoryLimitBytes);

        if(canUseQualityScores()){
            compactQualities(memoryLimitBytes);
        }

    }

    int getNumFiles() const noexcept{
        return numReadsPerFile.size();
    }

    std::size_t getNumSequences() const noexcept{
        return offsetsPrefixSum.back();
    }

    bool compactSequences(std::size_t& availableMem){
        std::size_t maxLength = lengthStorage.getMaxLength();
        std::size_t numSequences = getNumSequences();

        encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(maxLength);

        if(availableMem >= numSequences * encodedSequencePitchInInts){
            shrinkedEncodedSequences.resize(numSequences * encodedSequencePitchInInts);
            availableMem -= numSequences * encodedSequencePitchInInts * sizeof(unsigned int);

            for(std::size_t chunk = 0; chunk < sequenceStorage.size(); chunk++){
                const auto& s = sequenceStorage[chunk];
                const std::size_t offset = offsetsPrefixSum[chunk];
                const std::size_t pitchInts = s.encodedSequencePitchInInts;
                const std::size_t num = s.encodedSequences.size() / pitchInts;

                for(std::size_t i = 0; i < num; i++){
                    std::copy(
                        s.encodedSequences.begin() + i * pitchInts,
                        s.encodedSequences.begin() + (i+1) * pitchInts,
                        shrinkedEncodedSequences.begin() + (offset + i) * encodedSequencePitchInInts
                    );
                }

                availableMem += pitchInts * num * sizeof(unsigned int);
            }

            hasShrinkedSequences = true;
            std::cerr << "shrinked sequences\n";

            return true;
        }else{
            return false;
        }
    }

    bool compactQualities(std::size_t& availableMem){
        std::size_t maxLength = lengthStorage.getMaxLength();
        std::size_t numSequences = getNumSequences();

        qualityPitchInBytes = maxLength;

        if(availableMem >= numSequences * qualityPitchInBytes){
            shrinkedQualities.resize(numSequences * qualityPitchInBytes);
            availableMem -= numSequences * qualityPitchInBytes;

            for(std::size_t chunk = 0; chunk < qualityStorage.size(); chunk++){
                const auto& s = qualityStorage[chunk];
                const std::size_t offset = offsetsPrefixSum[chunk];
                const std::size_t pitchBytes = s.qualityPitchInBytes;
                const std::size_t num = s.qualities.size() / pitchBytes;

                for(std::size_t i = 0; i < num; i++){
                    std::copy(
                        s.qualities.begin() + i * pitchBytes,
                        s.qualities.begin() + (i+1) * pitchBytes,
                        shrinkedQualities.begin() + (offset + i) * pitchBytes
                    );
                }

                availableMem += pitchBytes * num;
            }

            hasShrinkedQualities = true;
            std::cerr << "shrinked qualities\n";

            return true;
        }else{
            return false;
        }
    }

    void gatherSequenceData(
        const read_number* readIds,
        int numReadIds,
        unsigned int* destination,
        int destinationPitchElements
    ) const noexcept{

        if(numReadIds == 0){
            return;
        }

        constexpr int prefetch_distance = 4;

        if(hasShrinkedSequences){
            for(int i = 0; i < numReadIds && i < prefetch_distance; ++i) {
                const int index = i;
                const std::size_t nextReadId = readIds[index];
                const unsigned int* const nextData = shrinkedEncodedSequences.data() + encodedSequencePitchInInts * nextReadId;
                __builtin_prefetch(nextData, 0, 0);
            }

            std::size_t destinationPitchBytes = destinationPitchElements * sizeof(unsigned int);

            for(int i = 0; i < numReadIds; i++){
                if(i + prefetch_distance < numReadIds) {
                    const int index = i + prefetch_distance;
                    const std::size_t nextReadId = readIds[index];
                    const unsigned int* const nextData = shrinkedEncodedSequences.data() + encodedSequencePitchInInts * nextReadId;
                    __builtin_prefetch(nextData, 0, 0);
                }

                const std::size_t readId = readIds[i];

                const unsigned int* const data = shrinkedEncodedSequences.data() + encodedSequencePitchInInts * readId;

                unsigned int* const destData = (unsigned int*)(((char*)destination) + destinationPitchBytes * i);
                std::copy_n(data, encodedSequencePitchInInts, destData);
            }
        }else{

            for(int i = 0; i < numReadIds && i < prefetch_distance; ++i) {
                const int index = i;
                const std::size_t nextReadId = readIds[index];
                const unsigned int* const nextData = getPointerToSequenceRow(nextReadId);
                __builtin_prefetch(nextData, 0, 0);
            }

            std::size_t destinationPitchBytes = destinationPitchElements * sizeof(unsigned int);

            for(int i = 0; i < numReadIds; i++){
                if(i + prefetch_distance < numReadIds) {
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

                unsigned int* const destData = (unsigned int*)(((char*)destination) + destinationPitchBytes * i);
                std::copy_n(data, sequenceStorage[chunkIndex].encodedSequencePitchInInts, destData);
            }

        }
    }

    void gatherQualities(
        const read_number* readIds,
        int numReadIds,
        unsigned int* destination,
        int destinationPitchElements
    ) const noexcept{

        if(numReadIds == 0){
            return;
        }

        constexpr int prefetch_distance = 4;

        if(hasShrinkedQualities){
            for(int i = 0; i < numReadIds && i < prefetch_distance; ++i) {
                const int index = i;
                const std::size_t nextReadId = readIds[index];
                const char* const nextData = shrinkedQualities.data() + qualityPitchInBytes * nextReadId;
                __builtin_prefetch(nextData, 0, 0);
            }

            std::size_t destinationPitchBytes = destinationPitchElements * sizeof(char);

            for(int i = 0; i < numReadIds; i++){
                if(i + prefetch_distance < numReadIds) {
                    const int index = i + prefetch_distance;
                    const std::size_t nextReadId = readIds[index];
                    const char* const nextData = shrinkedQualities.data() + qualityPitchInBytes * nextReadId;
                    __builtin_prefetch(nextData, 0, 0);
                }

                const std::size_t readId = readIds[i];

                const char* const data = shrinkedQualities.data() + qualityPitchInBytes * readId;

                char* const destData = (char*)(((char*)destination) + destinationPitchBytes * i);
                std::copy_n(data, qualityPitchInBytes, destData);
            }
        }else{

            for(int i = 0; i < numReadIds && i < prefetch_distance; ++i) {
                const int index = i;
                const std::size_t nextReadId = readIds[index];
                const char* const nextData = getPointerToQualityRow(nextReadId);
                __builtin_prefetch(nextData, 0, 0);
            }

            std::size_t destinationPitchBytes = destinationPitchElements * sizeof(char);

            for(int i = 0; i < numReadIds; i++){
                if(i + prefetch_distance < numReadIds) {
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

                char* const destData = (char*)(((char*)destination) + destinationPitchBytes * i);
                std::copy_n(data, qualityStorage[chunkIndex].qualityPitchInBytes, destData);
            }

        }
    }


    void gatherSequenceLengths(
            const read_number* readIds,
            int numReadIds,
            int* destination,
            int destinationPitchElements = 1) const noexcept{

        for(int i = 0; i < numReadIds; i++){
            int* const destLength = destination + i * destinationPitchElements;
            *destLength = lengthStorage.getLength(readIds[i]);
        }            
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

    bool hasQualityScores{};

    std::vector<std::size_t> numReadsPerFile{};
    std::vector<std::size_t> offsetsPrefixSum{};
    std::vector<StoredEncodedSequences> sequenceStorage{};
    LengthStore<std::uint32_t> lengthStorage{};
    std::vector<StoredQualities> qualityStorage{};
    std::map<int, std::set<read_number>> ambigReadIdsPerFile{};

    bool hasShrinkedSequences = false;
    std::size_t encodedSequencePitchInInts{};
    std::vector<unsigned int> shrinkedEncodedSequences{};

    bool hasShrinkedQualities = false;
    std::size_t qualityPitchInBytes{};
    std::vector<char> shrinkedQualities{};
};


}





#endif