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


template<class sequence_t,
		 class readId_t>
struct CPUReadStorage{

	using Sequence_t = sequence_t;
	using ReadId_t = readId_t;

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


    CPUReadStorage() : CPUReadStorage(false, 0){}
    CPUReadStorage(bool useQualityScores) : CPUReadStorage(useQualityScores, 0){}
    CPUReadStorage(int max_sequence_length) : CPUReadStorage(false, max_sequence_length){}
    CPUReadStorage(bool useQualityScores, int max_sequence_length)
                : max_sequence_length(max_sequence_length), useQualityScores(useQualityScores){
                    max_sequence_bytes = Sequence_t::getNumBytes(max_sequence_length);
                }


    CPUReadStorage(const CPUReadStorage& other) = delete;

    CPUReadStorage(CPUReadStorage&& other)
        : h_sequence_data(std::move(other.h_sequence_data)),
          h_sequence_lengths(std::move(other.h_sequence_lengths)),
          h_quality_data(std::move(other.h_quality_data)),
          max_sequence_length(other.max_sequence_length),
          max_sequence_bytes(other.max_sequence_bytes),
          useQualityScores(other.useQualityScores),
          num_sequences(other.num_sequences){

    }

    CPUReadStorage& operator=(CPUReadStorage&& other){
        h_sequence_data = std::move(other.h_sequence_data);
        h_sequence_lengths = std::move(other.h_sequence_lengths);
        h_quality_data = std::move(other.h_quality_data);
        max_sequence_length = other.max_sequence_length;
        max_sequence_bytes = other.max_sequence_bytes;
        useQualityScores = other.useQualityScores;
        num_sequences = other.num_sequences;
        return *this;
    }

    std::size_t size() const{
        assert(std::size_t(num_sequences) * max_sequence_bytes == sequence_data_bytes);
        assert(std::size_t(num_sequences) * sizeof(Length_t) == sequence_lengths_bytes);

        std::size_t result = 0;
        result += sequence_data_bytes;
        result += sequence_lengths_bytes;

        if(useQualityScores){
            assert(std::size_t(num_sequences) * max_sequence_length * sizeof(char) == quality_data_bytes);
            result += quality_data_bytes;
        }

        return result;
    }

    void init(ReadId_t nReads){
		clear();

        h_sequence_data.reset(new char[std::size_t(nReads) * Sequence_t::getNumBytes(max_sequence_length)]);
        sequence_data_bytes = sizeof(char) * std::size_t(nReads) * Sequence_t::getNumBytes(max_sequence_length);

        h_sequence_lengths.reset(new Length_t[std::size_t(nReads)]);
        sequence_lengths_bytes = sizeof(Length_t) * std::size_t(nReads);

        if(useQualityScores){
            h_quality_data.reset(new char[std::size_t(nReads) * max_sequence_length]);
            quality_data_bytes = sizeof(char) * std::size_t(nReads) * max_sequence_length;
        }
	}

    void insertRead(ReadId_t readNumber, const std::string& sequence){
        assert(sequence.length() <= max_sequence_length);

		if(useQualityScores){
			insertRead(readNumber, sequence, std::string(sequence.length(), 'A'));
		}else{
			Sequence_t seq(sequence);
            std::memcpy(h_sequence_data + readNumber * max_sequence_bytes,
                        seq.begin(),
                        seq.getNumBytes());
		}
	}

    void insertRead(ReadId_t readNumber, const std::string& sequence, const std::string& quality){
        assert(sequence.length() <= max_sequence_length);
        assert(quality.length() <= max_sequence_length);
        assert(sequence.length() == quality.length());

		Sequence_t seq(sequence);

        std::memcpy(h_sequence_data + readNumber * max_sequence_bytes,
                    seq.begin(),
                    seq.getNumBytes());

		if(useQualityScores){
            std::memcpy(h_quality_data + readNumber * max_sequence_length * sizeof(char),
                        quality.c_str(),
                        sizeof(char) * quality.length());
		}
	}

	const char* fetchSequence_ptr(ReadId_t readNumber) const{
		return h_sequence_data + readNumber * max_sequence_bytes;
	}

    const int fetchSequenceLength(ReadId_t readNumber) const{
		return h_sequence_lengths[readNumber];
	}

    const char* fetchQuality_ptr(ReadId_t readNumber) const{
        if(useQualityScores){
            return h_quality_data + readNumber * max_sequence_length * sizeof(char);
        }else{
            return nullptr;
        }
    }

	void resize(ReadId_t nReads){
        assert(num_sequences >= nReads);

        num_sequences = nReads;
	}

	void clear(){
        h_sequence_data.reset();
        h_sequence_lengths.reset();
        h_quality_data.reset();

        num_sequences = 0;
	}

	void destroy(){
		clear();
	}



};



}

#endif
