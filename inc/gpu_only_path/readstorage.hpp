#ifndef CARE_GPU_READ_STORAGE_HPP
#define CARE_GPU_READ_STORAGE_HPP

#include "../hpc_helpers.cuh"

#include <iostream>
#include <limits>
#include <random>
#include <cstring>
#include <cstdint>
#include <memory>

namespace care{
namespace gpu{

    template<class sequence_t,
             class readId_t>
    struct ContiguousReadStorage{

        enum class Type{
            None,
            Full,
            Managed
        };

        using Sequence_t = sequence_t;
        using ReadId_t = readId_t;

        static constexpr bool has_reverse_complement = false;

        using Length_t = int;

        std::unique_ptr<char[]> h_sequence_data = nullptr;
        std::unique_ptr<Length_t[]> h_sequence_lengths = nullptr;
        std::unique_ptr<char[]> h_quality_data = nullptr;
        int max_sequence_length = 0;
        int max_sequence_bytes = 0;
        bool useQualityScores = false;
        ReadId_t num_sequences = 0;
        std::size_t sequence_data_bytes = 0;
        std::size_t sequence_lengths_bytes = 0;
        std::size_t quality_data_bytes = 0;

        ContiguousReadStorage::Type sequenceType = ContiguousReadStorage::Type::None;
    	ContiguousReadStorage::Type qualityType = ContiguousReadStorage::Type::None;
        ContiguousReadStorage::Type sequencelengthType = ContiguousReadStorage::Type::None;

        std::string nameOf(ContiguousReadStorage::Type type) const {
            switch(type){
                case ContiguousReadStorage::Type::None: return "ContiguousReadStorage::Type::None";
                case ContiguousReadStorage::Type::Full: return "ContiguousReadStorage::Type::Full";
                case ContiguousReadStorage::Type::Managed: return "ContiguousReadStorage::Type::Managed";
                default: return "Error. ContiguousReadStorage::nameOf default case";
            }
        }

        ContiguousReadStorage(ReadId_t nSequences) : ContiguousReadStorage(nSequences, false){}

        ContiguousReadStorage(ReadId_t nSequences, bool b) : ContiguousReadStorage(nSequences, b, 0){
        }

        ContiguousReadStorage(ReadId_t nSequences, bool b, int maximum_sequence_length)
            : max_sequence_length(maximum_sequence_length),
                max_sequence_bytes(Sequence_t::getNumBytes(maximum_sequence_length)),
                useQualityScores(b),
                num_sequences(nSequences){


            h_sequence_data.reset(new char[std::size_t(num_sequences) * max_sequence_bytes]);
            sequence_data_bytes = sizeof(char) * std::size_t(num_sequences) * max_sequence_bytes;

            h_sequence_lengths.reset(new Length_t[std::size_t(num_sequences)]);
            sequence_lengths_bytes = sizeof(Length_t) * std::size_t(num_sequences);

            if(useQualityScores){
                h_quality_data.reset(new char[std::size_t(num_sequences) * max_sequence_length]);
                quality_data_bytes = sizeof(char) * std::size_t(num_sequences) * max_sequence_length;
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
              max_sequence_length(other.max_sequence_length),
              max_sequence_bytes(other.max_sequence_bytes),
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
            max_sequence_length = other.max_sequence_length;
            max_sequence_bytes = other.max_sequence_bytes;
            useQualityScores = other.useQualityScores;
            num_sequences = other.num_sequences;
            sequence_data_bytes = other.sequence_data_bytes;
            sequence_lengths_bytes = other.sequence_lengths_bytes;
            quality_data_bytes = other.quality_data_bytes;

            return *this;
        }

        bool operator==(const ContiguousReadStorage& other){
            if(max_sequence_length != other.max_sequence_length)
                return false;
            if(max_sequence_bytes != other.max_sequence_bytes)
                return false;
            if(useQualityScores != other.useQualityScores)
                return false;
            if(num_sequences != other.num_sequences)
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
            //assert(std::size_t(num_sequences) * max_sequence_bytes == sequence_data_bytes);
            //assert(std::size_t(num_sequences) * sizeof(Length_t) == sequence_lengths_bytes);

            std::size_t result = 0;
            result += sequence_data_bytes;
            result += sequence_lengths_bytes;

            if(useQualityScores){
                //assert(std::size_t(num_sequences) * max_sequence_length * sizeof(char) == quality_data_bytes);
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
            std::memcpy(&h_sequence_data[readNumber * max_sequence_bytes],
                        seq.begin(),
                        seq.getNumBytes());

            h_sequence_lengths[readNumber] = Length_t(sequence.length());
        }
public:
        void insertRead(ReadId_t readNumber, const std::string& sequence){
            assert(readNumber < getNumberOfSequences());
            assert(sequence.length() <= max_sequence_length);

            if(useQualityScores){
                insertRead(readNumber, sequence, std::string(sequence.length(), 'A'));
            }else{
                insertSequence(readNumber, sequence);
            }
        }

        void insertRead(ReadId_t readNumber, const std::string& sequence, const std::string& quality){
            assert(readNumber < getNumberOfSequences());
            assert(sequence.length() <= max_sequence_length);
            assert(quality.length() <= max_sequence_length);
            assert(sequence.length() == quality.length());

            insertSequence(readNumber, sequence);

            if(useQualityScores){
                std::memcpy(&h_quality_data[readNumber * max_sequence_length],
                            quality.c_str(),
                            sizeof(char) * quality.length());
            }
        }

        const char* fetchQuality2_ptr(ReadId_t readNumber) const{
            if(useQualityScores){
                return &h_quality_data[readNumber * max_sequence_length];
            }else{
                return nullptr;
            }
        }

        const std::string* fetchQuality_ptr(ReadId_t readNumber) const{
            return nullptr;
        }

       const char* fetchSequenceData_ptr(ReadId_t readNumber) const{
            return &h_sequence_data[readNumber * max_sequence_bytes];
       }

       int fetchSequenceLength(ReadId_t readNumber) const{
            return h_sequence_lengths[readNumber];
       }

        std::uint64_t getNumberOfSequences() const{
            return num_sequences;
        }

        void saveToFile(const std::string& filename) const{
            std::cout << "ContiguousReadStorage::saveToFile is not implemented yet!" << std::endl;
            /*std::ofstream stream(filename, std::ios::binary);

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
            }*/
        }

        void loadFromFile(const std::string& filename){
            throw std::runtime_error("ContiguousReadStorage::loadFromFile is not implemented yet!");
            /*std::ifstream stream(filename);
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
            }*/
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
