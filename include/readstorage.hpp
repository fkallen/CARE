#ifndef READ_STORAGE_HPP
#define READ_STORAGE_HPP

#include <config.hpp>
#include <sequence.hpp>
#include <lengthstorage.hpp>

#include <algorithm>
#include <limits>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
#include <omp.h>
#include <map>
#include <fstream>
#include <memory>
#include <cstring>
#include <mutex>
#include <atomic>


namespace care{

namespace cpu{

    /*
        Data structure to store sequences and their quality scores
    */

   //TODO store read ids of reads which contain a base other than A,C,G,T

    struct ContiguousReadStorage{

        struct Statistics{
            int maximumSequenceLength = 0;
            int minimumSequenceLength = std::numeric_limits<int>::max();

            bool operator==(const Statistics& rhs) const noexcept {
                return maximumSequenceLength == rhs.maximumSequenceLength
                    && minimumSequenceLength == rhs.minimumSequenceLength;
            }
            bool operator!=(const Statistics& rhs) const noexcept{
                return !(operator==(rhs));
            }
        };

        static constexpr bool has_reverse_complement = false;
        static constexpr int serialization_id = 1;

        using Length_t = int;
        using LengthStore_t = LengthStore<std::uint32_t>;

        std::unique_ptr<char[]> h_sequence_data = nullptr;
        std::unique_ptr<Length_t[]> h_sequence_lengths = nullptr;
        std::unique_ptr<char[]> h_quality_data = nullptr;
        int sequenceLengthLowerBound = 0;
        int sequenceLengthUpperBound = 0;
        int maximum_allowed_sequence_bytes = 0;
        bool useQualityScores = false;
        read_number maximumNumberOfSequences = 0;
        std::size_t sequence_data_bytes = 0;
        std::size_t sequence_lengths_bytes = 0;
        std::size_t quality_data_bytes = 0;
        std::vector<read_number> readIdsOfReadsWithUndeterminedBase; //sorted in ascending order
        std::mutex mutexUndeterminedBaseReads;
        Statistics statistics;
        std::atomic<read_number> numberOfInsertedReads{0};
        LengthStore_t lengthStorage;
        

        ContiguousReadStorage() : ContiguousReadStorage(0){}

        ContiguousReadStorage(read_number nSequences) : ContiguousReadStorage(nSequences, false){}

        ContiguousReadStorage(read_number nSequences, bool b) : ContiguousReadStorage(nSequences, b, 0, 0){
        }

        ContiguousReadStorage(read_number nSequences, bool b, int minimum_sequence_length, int maximum_sequence_length)
            : sequenceLengthUpperBound(maximum_sequence_length),
                maximum_allowed_sequence_bytes(getEncodedNumInts2BitHiLo(maximum_sequence_length) * sizeof(unsigned int)),
                useQualityScores(b),
                maximumNumberOfSequences(nSequences){

            sequenceLengthLowerBound = minimum_sequence_length;
            sequenceLengthUpperBound = maximum_sequence_length;
            
            lengthStorage = std::move(LengthStore_t(sequenceLengthLowerBound, sequenceLengthUpperBound, nSequences));

            h_sequence_data.reset(new char[std::size_t(maximumNumberOfSequences) * maximum_allowed_sequence_bytes]);
            sequence_data_bytes = sizeof(char) * std::size_t(maximumNumberOfSequences) * maximum_allowed_sequence_bytes;

            //h_sequence_lengths.reset(new Length_t[std::size_t(maximumNumberOfSequences)]);
            sequence_lengths_bytes = sizeof(Length_t) * std::size_t(maximumNumberOfSequences);

            if(useQualityScores){
                h_quality_data.reset(new char[std::size_t(maximumNumberOfSequences) * sequenceLengthUpperBound]);
                quality_data_bytes = sizeof(char) * std::size_t(maximumNumberOfSequences) * sequenceLengthUpperBound;
            }

            std::fill(&h_sequence_data[0], &h_sequence_data[sequence_data_bytes], 0);
            //std::fill(&h_sequence_lengths[0], &h_sequence_lengths[maximumNumberOfSequences], 0);
            std::fill(&h_quality_data[0], &h_quality_data[quality_data_bytes], 0);
        }

        ContiguousReadStorage(const ContiguousReadStorage& other) = delete;
        ContiguousReadStorage& operator=(const ContiguousReadStorage& other) = delete;

        ContiguousReadStorage(ContiguousReadStorage&& other)
            : h_sequence_data(std::move(other.h_sequence_data)),
              h_sequence_lengths(std::move(other.h_sequence_lengths)),
              h_quality_data(std::move(other.h_quality_data)),
              sequenceLengthLowerBound(other.sequenceLengthLowerBound),
              sequenceLengthUpperBound(other.sequenceLengthUpperBound),
              maximum_allowed_sequence_bytes(other.maximum_allowed_sequence_bytes),
              useQualityScores(other.useQualityScores),
              maximumNumberOfSequences(other.maximumNumberOfSequences),
              sequence_data_bytes(other.sequence_data_bytes),
              sequence_lengths_bytes(other.sequence_lengths_bytes),
              quality_data_bytes(other.quality_data_bytes),
              readIdsOfReadsWithUndeterminedBase(std::move(other.readIdsOfReadsWithUndeterminedBase)),
              statistics(std::move(other.statistics)),
              numberOfInsertedReads(other.numberOfInsertedReads.load()),
              lengthStorage(std::move(other.lengthStorage)){

            other.numberOfInsertedReads = 0;
            other.statistics = Statistics{};

        }

        ContiguousReadStorage& operator=(ContiguousReadStorage&& other){
            h_sequence_data = std::move(other.h_sequence_data);
            h_sequence_lengths = std::move(other.h_sequence_lengths);
            h_quality_data = std::move(other.h_quality_data);
            sequenceLengthLowerBound = other.sequenceLengthLowerBound;
            sequenceLengthUpperBound = other.sequenceLengthUpperBound;
            maximum_allowed_sequence_bytes = other.maximum_allowed_sequence_bytes;
            useQualityScores = other.useQualityScores;
            maximumNumberOfSequences = other.maximumNumberOfSequences;
            sequence_data_bytes = other.sequence_data_bytes;
            sequence_lengths_bytes = other.sequence_lengths_bytes;
            quality_data_bytes = other.quality_data_bytes;
            readIdsOfReadsWithUndeterminedBase = std::move(other.readIdsOfReadsWithUndeterminedBase);
            statistics = std::move(other.statistics);
            numberOfInsertedReads = other.numberOfInsertedReads.load();
            lengthStorage = std::move(other.lengthStorage);

            other.numberOfInsertedReads = 0;
            other.statistics = Statistics{};

            return *this;
        }

        bool operator==(const ContiguousReadStorage& other) const{
            if(sequenceLengthLowerBound != other.sequenceLengthLowerBound)
                return false;
            if(sequenceLengthUpperBound != other.sequenceLengthUpperBound)
                return false;
            if(maximum_allowed_sequence_bytes != other.maximum_allowed_sequence_bytes)
                return false;
            if(useQualityScores != other.useQualityScores)
                return false;
            if(maximumNumberOfSequences != other.maximumNumberOfSequences)
                return false;
            if(useQualityScores != other.useQualityScores)
                return false;
            if(sequence_data_bytes != other.sequence_data_bytes)
                return false;
            if(sequence_lengths_bytes != other.sequence_lengths_bytes)
                return false;
            if(quality_data_bytes != other.quality_data_bytes)
                return false;
            if(readIdsOfReadsWithUndeterminedBase != other.readIdsOfReadsWithUndeterminedBase){
                return false;
            }
            if(statistics != other.statistics){
                return false;
            }

            if(lengthStorage != other.lengthStorage){
                return false;
            }

            if(0 != std::memcmp(h_sequence_data.get(), other.h_sequence_data.get(), sequence_data_bytes))
                return false;
            //if(0 != std::memcmp(h_sequence_lengths.get(), other.h_sequence_lengths.get(), sequence_lengths_bytes))
            //    return false;
            if(0 != std::memcmp(h_quality_data.get(), other.h_quality_data.get(), quality_data_bytes))
                return false;

            return true;
        }

        bool operator!=(const ContiguousReadStorage& other) const{
            return !(*this == other);
        }

        std::size_t size() const{
            //assert(std::size_t(maximumNumberOfSequences) * maximum_allowed_sequence_bytes == sequence_data_bytes);
            //assert(std::size_t(maximumNumberOfSequences) * sizeof(Length_t) == sequence_lengths_bytes);

            std::size_t result = 0;
            result += sequence_data_bytes;
            result += sequence_lengths_bytes;

            if(useQualityScores){
                //assert(std::size_t(maximumNumberOfSequences) * sequenceLengthUpperBound * sizeof(char) == quality_data_bytes);
                result += quality_data_bytes;
            }

            return result;
        }

    	void resize(read_number nReads){
    		assert(getMaximumNumberOfSequences() >= nReads);

            maximumNumberOfSequences = nReads;
    	}

    	void destroy(){
            h_sequence_data.reset();
            h_sequence_lengths.reset();
            h_quality_data.reset();
    	}

        Statistics getStatistics() const{
            return statistics;
        }

        read_number getNumberOfReads() const{
            return numberOfInsertedReads;
        }

        void setReadContainsN(read_number readId, bool contains){

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

        bool readContainsN(read_number readId) const{

            auto pos = std::lower_bound(readIdsOfReadsWithUndeterminedBase.begin(),
                                                readIdsOfReadsWithUndeterminedBase.end(),
                                                readId);
            bool b2 = readIdsOfReadsWithUndeterminedBase.end() != pos && *pos == readId;

            return b2;
        }

private:
        void insertSequence(read_number readNumber, const std::string& sequence){
            auto identity = [](auto i){return i;};

            const int sequencelength = sequence.length();

            unsigned int* dest = (unsigned int*)&h_sequence_data[std::size_t(readNumber) * std::size_t(maximum_allowed_sequence_bytes)];
            encodeSequence2BitHiLo(dest,
                                    sequence.c_str(),
                                    sequence.length(),
                                    identity);

            statistics.minimumSequenceLength = std::min(statistics.minimumSequenceLength, sequencelength);
            statistics.maximumSequenceLength = std::max(statistics.maximumSequenceLength, sequencelength);

            //h_sequence_lengths[readNumber] = Length_t(sequence.length());

            lengthStorage.setLength(readNumber, Length_t(sequence.length()));

            read_number prev_value = numberOfInsertedReads;
            while(prev_value < readNumber+1 && !numberOfInsertedReads.compare_exchange_weak(prev_value, readNumber+1)){
                ;
            }
        }
public:
        void insertRead(read_number readNumber, const std::string& sequence){
            assert(readNumber < getMaximumNumberOfSequences());
            assert(int(sequence.length()) <= sequenceLengthUpperBound);

    		if(useQualityScores){
    			insertRead(readNumber, sequence, std::string(sequence.length(), 'A'));
    		}else{
    			insertSequence(readNumber, sequence);
    		}
    	}

        void insertRead(read_number readNumber, const std::string& sequence, const std::string& quality){
            assert(readNumber < getMaximumNumberOfSequences());
            assert(int(sequence.length()) <= sequenceLengthUpperBound);
            assert(int(quality.length()) <= sequenceLengthUpperBound);
            assert(sequence.length() == quality.length());

    		insertSequence(readNumber, sequence);

    		if(useQualityScores){
                std::memcpy(&h_quality_data[std::size_t(readNumber) * std::size_t(sequenceLengthUpperBound)],
                            quality.c_str(),
                            sizeof(char) * quality.length());
    		}
    	}

        const char* fetchQuality_ptr(read_number readNumber) const{
            if(useQualityScores){
                return &h_quality_data[std::size_t(readNumber) * std::size_t(sequenceLengthUpperBound)];
            }else{
                return nullptr;
            }
        }

        const char* fetchSequenceData_ptr(read_number readNumber) const{
        	return &h_sequence_data[std::size_t(readNumber) * std::size_t(maximum_allowed_sequence_bytes)];
        }

        int fetchSequenceLength(read_number readNumber) const{
            return lengthStorage.getLength(readNumber);
        	//return h_sequence_lengths[readNumber];
        }

        bool hasQualityScores() const{
            return useQualityScores;
        }

    	std::uint64_t getMaximumNumberOfSequences() const{
    		return maximumNumberOfSequences;
    	}

        int getMaximumAllowedSequenceBytes() const{
            return maximum_allowed_sequence_bytes;
        }

        void saveToFile(const std::string& filename) const{
            std::ofstream stream(filename, std::ios::binary);

            read_number inserted = getNumberOfReads();

            //int ser_id = serialization_id;
            std::size_t lengthsize = sizeof(Length_t);
            //stream.write(reinterpret_cast<const char*>(&ser_id), sizeof(int));
            stream.write(reinterpret_cast<const char*>(&lengthsize), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&sequenceLengthUpperBound), sizeof(int));
            stream.write(reinterpret_cast<const char*>(&maximum_allowed_sequence_bytes), sizeof(int));
            stream.write(reinterpret_cast<const char*>(&useQualityScores), sizeof(bool));
            stream.write(reinterpret_cast<const char*>(&maximumNumberOfSequences), sizeof(read_number));
            stream.write(reinterpret_cast<const char*>(&sequence_data_bytes), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&sequence_lengths_bytes), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&quality_data_bytes), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&statistics), sizeof(Statistics));
            stream.write(reinterpret_cast<const char*>(&inserted), sizeof(read_number));

            lengthStorage.writeToStream(stream);

            stream.write(reinterpret_cast<const char*>(&h_sequence_data[0]), sequence_data_bytes);
            //stream.write(reinterpret_cast<const char*>(&h_sequence_lengths[0]), sequence_lengths_bytes);
            stream.write(reinterpret_cast<const char*>(&h_quality_data[0]), quality_data_bytes);

            //read ids with N
            std::size_t numUndeterminedReads = readIdsOfReadsWithUndeterminedBase.size();
            stream.write(reinterpret_cast<const char*>(&numUndeterminedReads), sizeof(size_t));
            stream.write(reinterpret_cast<const char*>(readIdsOfReadsWithUndeterminedBase.data()), numUndeterminedReads * sizeof(read_number));

        }

        void loadFromFile(const std::string& filename){
            std::ifstream stream(filename, std::ios::binary);
            if(!stream)
                throw std::runtime_error("Cannot open file " + filename);

            //int ser_id = serialization_id;
            std::size_t lengthsize = sizeof(Length_t);

            //int loaded_serialization_id = 0;
            std::size_t loaded_lengthsize = 0;
            int loaded_sequenceLengthUpperBound = 0;
            int loaded_maximum_allowed_sequence_bytes = 0;
            bool loaded_useQualityScores = false;
            read_number loaded_maximumNumberOfSequences = 0;
            std::size_t loaded_sequence_data_bytes = 0;
            std::size_t loaded_sequence_lengths_bytes = 0;
            std::size_t loaded_quality_data_bytes = 0;
            read_number inserted = 0;

            //stream.read(reinterpret_cast<char*>(&loaded_serialization_id), sizeof(int));
            stream.read(reinterpret_cast<char*>(&loaded_lengthsize), sizeof(std::size_t));
            stream.read(reinterpret_cast<char*>(&loaded_sequenceLengthUpperBound), sizeof(int));
            stream.read(reinterpret_cast<char*>(&loaded_maximum_allowed_sequence_bytes), sizeof(int));
            stream.read(reinterpret_cast<char*>(&loaded_useQualityScores), sizeof(bool));
            stream.read(reinterpret_cast<char*>(&loaded_maximumNumberOfSequences), sizeof(read_number));
            stream.read(reinterpret_cast<char*>(&loaded_sequence_data_bytes), sizeof(std::size_t));
            stream.read(reinterpret_cast<char*>(&loaded_sequence_lengths_bytes), sizeof(std::size_t));
            stream.read(reinterpret_cast<char*>(&loaded_quality_data_bytes), sizeof(std::size_t));
            stream.read(reinterpret_cast<char*>(&statistics), sizeof(Statistics));
            stream.read(reinterpret_cast<char*>(&inserted), sizeof(read_number));

            numberOfInsertedReads = inserted;

            lengthStorage.readFromStream(stream);

            //if(loaded_serialization_id != ser_id)
            //    throw std::runtime_error("Wrong serialization id!");
            if(loaded_lengthsize != lengthsize)
                throw std::runtime_error("Wrong size of length type!");
            //if(useQualityScores && !loaded_useQualityScores)
            //    throw std::runtime_error("Quality scores are required but not present in binary sequence file!");
            //if(!useQualityScores && loaded_useQualityScores)
            //    std::cerr << "The loaded compressed read file contains quality scores, but program does not use them!\n";

            destroy();

            h_sequence_data.reset((char*)new char[loaded_sequence_data_bytes]);
            //h_sequence_lengths.reset((Length_t*)new char[loaded_sequence_lengths_bytes]);
            h_quality_data.reset((char*)new char[loaded_quality_data_bytes]);

            stream.read(reinterpret_cast<char*>(&h_sequence_data[0]), loaded_sequence_data_bytes);
            //stream.read(reinterpret_cast<char*>(&h_sequence_lengths[0]), loaded_sequence_lengths_bytes);
            stream.read(reinterpret_cast<char*>(&h_quality_data[0]), loaded_quality_data_bytes);

            sequenceLengthUpperBound = loaded_sequenceLengthUpperBound;
            maximum_allowed_sequence_bytes = loaded_maximum_allowed_sequence_bytes;
            maximumNumberOfSequences = loaded_maximumNumberOfSequences;
            useQualityScores = loaded_useQualityScores;
            sequence_data_bytes = loaded_sequence_data_bytes;
            sequence_lengths_bytes = loaded_sequence_lengths_bytes;
            quality_data_bytes = loaded_quality_data_bytes;

                //read ids with N
            std::size_t numUndeterminedReads = 0;
            stream.read(reinterpret_cast<char*>(&numUndeterminedReads), sizeof(std::size_t));
            readIdsOfReadsWithUndeterminedBase.resize(numUndeterminedReads);
            stream.read(reinterpret_cast<char*>(readIdsOfReadsWithUndeterminedBase.data()), numUndeterminedReads * sizeof(read_number));
        }
    };

}



}

#endif
