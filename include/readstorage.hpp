#ifndef READ_STORAGE_HPP
#define READ_STORAGE_HPP

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

namespace care{

namespace cpu{

    struct SequenceStatistics{
        int maxSequenceLength = 0;
        int minSequenceLength = 0;
    };

    /*
        Data structure to store sequences and their quality scores
    */
    template<class sequence_t,
    		 class readId_t>
    struct ReadStorage{

    	using Sequence_t = sequence_t;
    	using ReadId_t = readId_t;

        static constexpr bool has_reverse_complement = false;

    	bool useQualityScores = false;

        std::vector<std::string> qualityscores;
        std::vector<Sequence_t> sequences;

        ReadStorage(ReadId_t nSequences) : ReadStorage(nSequences, false){}

        ReadStorage(ReadId_t nSequences, bool b) : ReadStorage(nSequences, b, -1){
        }

        ReadStorage(ReadId_t nSequences, bool b, int /*maximum_sequence_length*/) : useQualityScores(b){
            sequences.resize(nSequences);
            if(useQualityScores){
                qualityscores.resize(nSequences);
            }
        }

        ReadStorage(const ReadStorage&) = default;
        ReadStorage(ReadStorage&&) = default;

        ReadStorage& operator=(const ReadStorage& other){
            useQualityScores = other.useQualityScores;
            qualityscores = other.qualityscores;
            sequences = other.sequences;
            return *this;
        }

        ReadStorage& operator=(ReadStorage&& other){
            useQualityScores = other.useQualityScores;
            qualityscores = std::move(other.qualityscores);
            sequences = std::move(other.sequences);
            return *this;
        }

        bool operator==(const ReadStorage& other){
            if(useQualityScores != other.useQualityScores)
                return false;
            if(sequences.size() != other.sequences.size())
                return false;
            if(qualityscores.size() != other.qualityscores.size())
                return false;

            for(std::size_t i = 0; i < sequences.size(); i++){
                if(sequences[i] != other.sequences[i])
                    return false;
            }
            for(std::size_t i = 0; i < qualityscores.size(); i++){
                if(qualityscores[i] != other.qualityscores[i])
                    return false;
            }
            return true;
        }

        bool operator!=(const ReadStorage& other){
            return !(*this == other);
        }

        std::size_t size() const{
            std::size_t result = 0;

    		std::map<std::size_t, std::uint64_t> map;

            for(const auto& s : qualityscores){
                result += sizeof(std::string) + s.capacity();
            }

            for(const auto& s : sequences){
    			std::size_t t = sizeof(Sequence_t) + s.getNumBytes();

    			map[t]++;

                result += t;
            }

            for(const auto& p : map){
    			std::cout << p.first << " : " << p.second << std::endl;
    		}

            return result;
        }

        std::size_t sizereal() const{
            std::size_t result = 0;

            for(std::size_t i = 0; i < qualityscores.capacity(); i++){
                result += sizeof(std::string) + qualityscores[i].capacity();
            }

            for(std::size_t i = 0; i < sequences.capacity(); i++){
                result += sizeof(Sequence_t) + sequences[i].getNumBytes();
            }

            return result;
        }

    	void resize(ReadId_t nReads){
    		assert(getNumberOfSequences() >= nReads);

    		sequences.resize(nReads);
    		if(useQualityScores)
    			qualityscores.resize(nReads);
    	}

    	void destroy(){
            std::vector<std::string> tmp1;
            std::vector<Sequence_t> tmp2;

    		qualityscores.swap(tmp1);
    		sequences.swap(tmp2);
    	}

        void insertRead(ReadId_t readNumber, const std::string& sequence){
    		if(useQualityScores){
    			insertRead(readNumber, sequence, std::string(sequence.length(), 'A'));
    		}else{
    			Sequence_t seq(sequence);
    			sequences[readNumber] = std::move(seq);
                //sequences.at(readNumber) = std::move(seq);
    		}
    	}

        void insertRead(ReadId_t readNumber, const std::string& sequence, const std::string& quality){
    		Sequence_t seq(sequence);

    		sequences[readNumber] = std::move(seq);
            //sequences.at(readNumber) = std::move(seq);
    		if(useQualityScores){
    			//qualityscores[readNumber] = std::move(quality);
                qualityscores.at(readNumber) = std::move(quality);
    		}
    	}

    	const std::string* fetchQuality_ptr(ReadId_t readNumber) const{
    		if(useQualityScores){
    			return &(qualityscores[readNumber]);
    		}else{
    			//return nullptr;
                throw std::runtime_error("cannot fetchQuality_ptr if useQualityScores == false");
    		}
    	}
private:
    	const Sequence_t* fetchSequence_ptr(ReadId_t readNumber) const{
    		return &sequences[readNumber];
    	}
public:

       const char* fetchSequenceData_ptr(ReadId_t readNumber) const{
    		auto ptr = fetchSequence_ptr(readNumber);
    		return (const char*)ptr->begin();
       }

       int fetchSequenceLength(ReadId_t readNumber) const{
    		auto ptr = fetchSequence_ptr(readNumber);
    		return ptr->length();
       }

    	std::uint64_t getNumberOfSequences() const{
    		return sequences.size();
    	}

        void saveToFile(const std::string& filename) const{
            std::ofstream stream(filename, std::ios::binary);

            auto writesequence = [&](const Sequence_t& seq){
                const int length = seq.length();
                const int bytes = seq.getNumBytes();
                stream.write(reinterpret_cast<const char*>(&length), sizeof(int));
                stream.write(reinterpret_cast<const char*>(&bytes), sizeof(int));
                stream.write(reinterpret_cast<const char*>(seq.begin()), bytes);
            };

            auto writequality = [&](const std::string& qual){
                const std::size_t bytes = qual.length();
                stream.write(reinterpret_cast<const char*>(&bytes), sizeof(int));
                stream.write(reinterpret_cast<const char*>(qual.c_str()), bytes);
            };

            std::size_t numReads = sequences.size();
            stream.write(reinterpret_cast<const char*>(&numReads), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&useQualityScores), sizeof(bool));

            for(std::size_t i = 0; i < numReads; i++){
                writesequence(sequences[i]);
                if(useQualityScores)
                    writequality(qualityscores[i]);
            }
        }

        void loadFromFile(const std::string& filename){
            std::ifstream stream(filename);
            if(!stream)
                throw std::runtime_error("cannot load binary sequences from file " + filename);

            auto readsequence = [&](){
                int length;
                int bytes;
                stream.read(reinterpret_cast<char*>(&length), sizeof(int));
                stream.read(reinterpret_cast<char*>(&bytes), sizeof(int));

                auto data = std::make_unique<std::uint8_t[]>(bytes);
                stream.read(reinterpret_cast<char*>(data.get()), bytes);

                return Sequence_t{data.get(), length};
            };

            auto readquality = [&](){
                static_assert(sizeof(char) == 1);

                std::size_t bytes;
                stream.read(reinterpret_cast<char*>(&bytes), sizeof(int));
                auto data = std::make_unique<char[]>(bytes);

                stream.read(reinterpret_cast<char*>(data.get()), bytes);

                return std::string{data.get(), bytes};
            };

            std::size_t numReads;
            stream.read(reinterpret_cast<char*>(&numReads), sizeof(std::size_t));
            stream.read(reinterpret_cast<char*>(&useQualityScores), sizeof(bool));

            if(numReads > getNumberOfSequences()){
                throw std::runtime_error("Readstorage was constructed for "
                                        + std::to_string(getNumberOfSequences())
                                        + " sequences, but binary file contains "
                                        + std::to_string(numReads) + " sequences!");
            }

            sequences.clear();
            sequences.reserve(numReads);
            if(useQualityScores){
                qualityscores.clear();
                qualityscores.reserve(numReads);
            }

            for(std::size_t i = 0; i < numReads; i++){
                sequences.emplace_back(readsequence());
                if(useQualityScores)
                    qualityscores.emplace_back(readquality());
            }
        }

        SequenceStatistics getSequenceStatistics() const{
            return getSequenceStatistics(1);
        }

        SequenceStatistics getSequenceStatistics(int numThreads) const{
            int maxSequenceLength = 0;
            int minSequenceLength = std::numeric_limits<int>::max();

            const int oldnumthreads = omp_get_thread_num();

            omp_set_num_threads(numThreads);

            #pragma omp parallel for reduction(max:maxSequenceLength) reduction(min:minSequenceLength)
            for(std::size_t readId = 0; readId < getNumberOfSequences(); readId++){
                const int len = fetchSequenceLength(readId);
                if(len > maxSequenceLength)
                    maxSequenceLength = len;
                if(len < minSequenceLength)
                    minSequenceLength = len;
            }

            omp_set_num_threads(oldnumthreads);

            SequenceStatistics stats;
            stats.minSequenceLength = minSequenceLength;
            stats.maxSequenceLength = maxSequenceLength;

            return stats;
        }

    };

    template<class sequence_t,
    		 class readId_t>
    struct ContiguousReadStorage{

    	using Sequence_t = sequence_t;
    	using ReadId_t = readId_t;

        static constexpr bool has_reverse_complement = false;
        static constexpr int serialization_id = 1;

        using Length_t = int;

        std::unique_ptr<char[]> h_sequence_data = nullptr;
        std::unique_ptr<Length_t[]> h_sequence_lengths = nullptr;
        std::unique_ptr<char[]> h_quality_data = nullptr;
        int maximum_allowed_sequence_length = 0;
        int maximum_allowed_sequence_bytes = 0;
        bool useQualityScores = false;
        ReadId_t num_sequences = 0;
        std::size_t sequence_data_bytes = 0;
        std::size_t sequence_lengths_bytes = 0;
        std::size_t quality_data_bytes = 0;

        ContiguousReadStorage(ReadId_t nSequences) : ContiguousReadStorage(nSequences, false){}

        ContiguousReadStorage(ReadId_t nSequences, bool b) : ContiguousReadStorage(nSequences, b, 0){
        }

        ContiguousReadStorage(ReadId_t nSequences, bool b, int maximum_sequence_length)
            : maximum_allowed_sequence_length(maximum_sequence_length),
                maximum_allowed_sequence_bytes(Sequence_t::getNumBytes(maximum_sequence_length)),
                useQualityScores(b),
                num_sequences(nSequences){


            h_sequence_data.reset(new char[std::size_t(num_sequences) * maximum_allowed_sequence_bytes]);
            sequence_data_bytes = sizeof(char) * std::size_t(num_sequences) * maximum_allowed_sequence_bytes;

            h_sequence_lengths.reset(new Length_t[std::size_t(num_sequences)]);
            sequence_lengths_bytes = sizeof(Length_t) * std::size_t(num_sequences);

            if(useQualityScores){
                h_quality_data.reset(new char[std::size_t(num_sequences) * maximum_allowed_sequence_length]);
                quality_data_bytes = sizeof(char) * std::size_t(num_sequences) * maximum_allowed_sequence_length;
            }

            std::fill(&h_sequence_data[0], &h_sequence_data[sequence_data_bytes], 0);
            std::fill(&h_sequence_lengths[0], &h_sequence_lengths[num_sequences], 0);
            std::fill(&h_quality_data[0], &h_quality_data[quality_data_bytes], 0);
        }

        ContiguousReadStorage(const ContiguousReadStorage& other) = delete;
        ContiguousReadStorage& operator=(const ContiguousReadStorage& other) = delete;

        ContiguousReadStorage(ContiguousReadStorage&& other)
            : h_sequence_data(std::move(other.h_sequence_data)),
              h_sequence_lengths(std::move(other.h_sequence_lengths)),
              h_quality_data(std::move(other.h_quality_data)),
              maximum_allowed_sequence_length(other.maximum_allowed_sequence_length),
              maximum_allowed_sequence_bytes(other.maximum_allowed_sequence_bytes),
              useQualityScores(other.useQualityScores),
              num_sequences(other.num_sequences),
              sequence_data_bytes(other.sequence_data_bytes),
              sequence_lengths_bytes(other.sequence_lengths_bytes),
              quality_data_bytes(other.quality_data_bytes){

        }

        ContiguousReadStorage& operator=(ContiguousReadStorage&& other){
            h_sequence_data = std::move(other.h_sequence_data);
            h_sequence_lengths = std::move(other.h_sequence_lengths);
            h_quality_data = std::move(other.h_quality_data);
            maximum_allowed_sequence_length = other.maximum_allowed_sequence_length;
            maximum_allowed_sequence_bytes = other.maximum_allowed_sequence_bytes;
            useQualityScores = other.useQualityScores;
            num_sequences = other.num_sequences;
            sequence_data_bytes = other.sequence_data_bytes;
            sequence_lengths_bytes = other.sequence_lengths_bytes;
            quality_data_bytes = other.quality_data_bytes;

            return *this;
        }

        bool operator==(const ContiguousReadStorage& other){
            if(maximum_allowed_sequence_length != other.maximum_allowed_sequence_length)
                return false;
            if(maximum_allowed_sequence_bytes != other.maximum_allowed_sequence_bytes)
                return false;
            if(useQualityScores != other.useQualityScores)
                return false;
            if(num_sequences != other.num_sequences)
                return false;
            if(useQualityScores != other.useQualityScores)
                return false;
            if(sequence_data_bytes != other.sequence_data_bytes)
                return false;
            if(sequence_lengths_bytes != other.sequence_lengths_bytes)
                return false;
            if(quality_data_bytes != other.quality_data_bytes)
                return false;

            if(0 != std::memcmp(h_sequence_data.get(), other.h_sequence_data.get(), sequence_data_bytes))
                return false;
            if(0 != std::memcmp(h_sequence_lengths.get(), other.h_sequence_lengths.get(), sequence_lengths_bytes))
                return false;
            if(0 != std::memcmp(h_quality_data.get(), other.h_quality_data.get(), quality_data_bytes))
                return false;

            return true;
        }

        bool operator!=(const ContiguousReadStorage& other){
            return !(*this == other);
        }

        std::size_t size() const{
            //assert(std::size_t(num_sequences) * maximum_allowed_sequence_bytes == sequence_data_bytes);
            //assert(std::size_t(num_sequences) * sizeof(Length_t) == sequence_lengths_bytes);

            std::size_t result = 0;
            result += sequence_data_bytes;
            result += sequence_lengths_bytes;

            if(useQualityScores){
                //assert(std::size_t(num_sequences) * maximum_allowed_sequence_length * sizeof(char) == quality_data_bytes);
                result += quality_data_bytes;
            }

            return result;
        }

    	void resize(ReadId_t nReads){
    		assert(getNumberOfSequences() >= nReads);

            num_sequences = nReads;
    	}

    	void destroy(){
            h_sequence_data.reset();
            h_sequence_lengths.reset();
            h_quality_data.reset();
    	}

private:
        void insertSequence(ReadId_t readNumber, const std::string& sequence){
            Sequence_t seq(sequence);
            std::memcpy(&h_sequence_data[std::size_t(readNumber) * std::size_t(maximum_allowed_sequence_bytes)],
                        seq.begin(),
                        seq.getNumBytes());

            h_sequence_lengths[readNumber] = Length_t(sequence.length());
        }
public:
        void insertRead(ReadId_t readNumber, const std::string& sequence){
            assert(readNumber < getNumberOfSequences());
            assert(sequence.length() <= maximum_allowed_sequence_length);

    		if(useQualityScores){
    			insertRead(readNumber, sequence, std::string(sequence.length(), 'A'));
    		}else{
    			insertSequence(readNumber, sequence);
    		}
    	}

        void insertRead(ReadId_t readNumber, const std::string& sequence, const std::string& quality){
            assert(readNumber < getNumberOfSequences());
            assert(int(sequence.length()) <= maximum_allowed_sequence_length);
            assert(int(quality.length()) <= maximum_allowed_sequence_length);
            assert(sequence.length() == quality.length());

    		insertSequence(readNumber, sequence);

    		if(useQualityScores){
                std::memcpy(&h_quality_data[std::size_t(readNumber) * std::size_t(maximum_allowed_sequence_length)],
                            quality.c_str(),
                            sizeof(char) * quality.length());
    		}
    	}

        const char* fetchQuality_ptr(ReadId_t readNumber) const{
            if(useQualityScores){
                return &h_quality_data[std::size_t(readNumber) * std::size_t(maximum_allowed_sequence_length)];
            }else{
                return nullptr;
            }
        }

       const char* fetchSequenceData_ptr(ReadId_t readNumber) const{
    		return &h_sequence_data[std::size_t(readNumber) * std::size_t(maximum_allowed_sequence_bytes)];
       }

       int fetchSequenceLength(ReadId_t readNumber) const{
    		return h_sequence_lengths[readNumber];
       }

    	std::uint64_t getNumberOfSequences() const{
    		return num_sequences;
    	}

        void saveToFile(const std::string& filename) const{
            std::ofstream stream(filename, std::ios::binary);

            //int ser_id = serialization_id;
            std::size_t lengthsize = sizeof(Length_t);
            //stream.write(reinterpret_cast<const char*>(&ser_id), sizeof(int));
            stream.write(reinterpret_cast<const char*>(&lengthsize), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&maximum_allowed_sequence_length), sizeof(int));
            stream.write(reinterpret_cast<const char*>(&maximum_allowed_sequence_bytes), sizeof(int));
            stream.write(reinterpret_cast<const char*>(&useQualityScores), sizeof(bool));
            stream.write(reinterpret_cast<const char*>(&num_sequences), sizeof(ReadId_t));
            stream.write(reinterpret_cast<const char*>(&sequence_data_bytes), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&sequence_lengths_bytes), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&quality_data_bytes), sizeof(std::size_t));
            stream.write(reinterpret_cast<const char*>(&h_sequence_data[0]), sequence_data_bytes);
            stream.write(reinterpret_cast<const char*>(&h_sequence_lengths[0]), sequence_lengths_bytes);
            stream.write(reinterpret_cast<const char*>(&h_quality_data[0]), quality_data_bytes);

        }

        void loadFromFile(const std::string& filename){
            std::ifstream stream(filename, std::ios::binary);
            if(!stream)
                throw std::runtime_error("Cannot open file " + filename);

            //int ser_id = serialization_id;
            std::size_t lengthsize = sizeof(Length_t);

            //int loaded_serialization_id = 0;
            std::size_t loaded_lengthsize = 0;
            int loaded_maximum_allowed_sequence_length = 0;
            int loaded_maximum_allowed_sequence_bytes = 0;
            bool loaded_useQualityScores = false;
            ReadId_t loaded_num_sequences = 0;
            std::size_t loaded_sequence_data_bytes = 0;
            std::size_t loaded_sequence_lengths_bytes = 0;
            std::size_t loaded_quality_data_bytes = 0;

            //stream.read(reinterpret_cast<char*>(&loaded_serialization_id), sizeof(int));
            stream.read(reinterpret_cast<char*>(&loaded_lengthsize), sizeof(std::size_t));
            stream.read(reinterpret_cast<char*>(&loaded_maximum_allowed_sequence_length), sizeof(int));
            stream.read(reinterpret_cast<char*>(&loaded_maximum_allowed_sequence_bytes), sizeof(int));
            stream.read(reinterpret_cast<char*>(&loaded_useQualityScores), sizeof(bool));
            stream.read(reinterpret_cast<char*>(&loaded_num_sequences), sizeof(ReadId_t));
            stream.read(reinterpret_cast<char*>(&loaded_sequence_data_bytes), sizeof(std::size_t));
            stream.read(reinterpret_cast<char*>(&loaded_sequence_lengths_bytes), sizeof(std::size_t));
            stream.read(reinterpret_cast<char*>(&loaded_quality_data_bytes), sizeof(std::size_t));

            //if(loaded_serialization_id != ser_id)
            //    throw std::runtime_error("Wrong serialization id!");
            if(loaded_lengthsize != lengthsize)
                throw std::runtime_error("Wrong size of length type!");
            if(useQualityScores && !loaded_useQualityScores)
                throw std::runtime_error("Quality scores are required but not present in binary sequence file!");
            if(!useQualityScores && loaded_useQualityScores)
                std::cerr << "The loaded compressed read file contains quality scores, but program does not use them!\n";

            destroy();

            h_sequence_data.reset((char*)new char[loaded_sequence_data_bytes]);
            h_sequence_lengths.reset((Length_t*)new char[loaded_sequence_lengths_bytes]);
            h_quality_data.reset((char*)new char[loaded_quality_data_bytes]);

            stream.read(reinterpret_cast<char*>(&h_sequence_data[0]), loaded_sequence_data_bytes);
            stream.read(reinterpret_cast<char*>(&h_sequence_lengths[0]), loaded_sequence_lengths_bytes);
            stream.read(reinterpret_cast<char*>(&h_quality_data[0]), loaded_quality_data_bytes);

            maximum_allowed_sequence_length = loaded_maximum_allowed_sequence_length;
            maximum_allowed_sequence_bytes = loaded_maximum_allowed_sequence_bytes;
            num_sequences = loaded_num_sequences;
            sequence_data_bytes = loaded_sequence_data_bytes;
            sequence_lengths_bytes = loaded_sequence_lengths_bytes;
            quality_data_bytes = loaded_quality_data_bytes;
        }

        SequenceStatistics getSequenceStatistics() const{
            return getSequenceStatistics(1);
        }

        SequenceStatistics getSequenceStatistics(int numThreads) const{
            int maxSequenceLength = 0;
            int minSequenceLength = std::numeric_limits<int>::max();

            const int oldnumthreads = omp_get_thread_num();

            omp_set_num_threads(numThreads);

            #pragma omp parallel for reduction(max:maxSequenceLength) reduction(min:minSequenceLength)
            for(std::size_t readId = 0; readId < getNumberOfSequences(); readId++){
                const int len = fetchSequenceLength(readId);
                if(len > maxSequenceLength)
                    maxSequenceLength = len;
                if(len < minSequenceLength)
                    minSequenceLength = len;
            }

            omp_set_num_threads(oldnumthreads);

            SequenceStatistics stats;
            stats.minSequenceLength = minSequenceLength;
            stats.maxSequenceLength = maxSequenceLength;

            return stats;
        }

    };

}



}

#endif
