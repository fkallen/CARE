#ifndef READ_HPP
#define READ_HPP

#include "binarysequencehelpers.hpp"
#include "ganja/hpc_helpers.cuh"

#include <algorithm>
#include <string>
#include <cstdint>
#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <cassert>

#if 1
template<int MAX_LEN_>
struct FixedSizeSequence{
	static constexpr uint8_t BITS_PER_BASE = 2;
	static constexpr uint8_t BASES_PER_BYTE = (sizeof(uint8_t)*8 / BITS_PER_BASE);

	static constexpr int MAX_LEN = MAX_LEN_;
	static constexpr int DATA_BYTES = SDIV(MAX_LEN, 4);

	uint8_t data[DATA_BYTES];
	int length = 0;

	HOSTDEVICEQUALIFIER
	constexpr int getNumBytes() const{
		return DATA_BYTES;
	}

	HOSTDEVICEQUALIFIER
	int length() const{
		return length;
	}

	HOSTDEVICEQUALIFIER
	FixedSizeSequence(){
		memset(data, 0, DATA_BYTES);
	}

	FixedSizeSequence(const std::string& sequence) : length(sequence.size()){
		if(sequence.size() > MAX_LEN || !encode(sequence.c_str(), sequence.size(), sequence.size(), data, DATA_BYTES, true)){
			throw std::runtime_error("could not encode sequence");
		}
	}

	FixedSizeSequence(const uint8_t* rawdata, int nBases_) : length(nBases_){
		if(nBases_ > MAX_LEN){
			throw std::runtime_error("could not encode sequence");
		}
		std::memcpy(data, rawdata, sizeof(uint8_t) * SDIV(nBases_,4));
	}

	HOSTDEVICEQUALIFIER
	FixedSizeSequence(const FixedSizeSequence& other){
		*this = other;
	}

	HOSTDEVICEQUALIFIER
	FixedSizeSequence& operator=(const FixedSizeSequence& other){
		length = other.length;
		memcpy(data, other.data, sizeof(uint8_t) * DATA_BYTES);
	}

	HOSTDEVICEQUALIFIER
	bool operator==(const FixedSizeSequence& rhs) const
	{
		if(length() != rhs.length()) return false;
		return (memcmp(begin(), rhs.begin(), getNumBytes()) == 0);
	}

	HOSTDEVICEQUALIFIER
	bool operator!=(const FixedSizeSequence& other) const
	{
		return !(*this == other);
	}

	bool operator==(const std::string& other) const
	{
		return toString() == other;
	}

	bool operator!=(const std::string& other) const
	{
		return !(*this == other);
	}

	HOSTDEVICEQUALIFIER
	bool operator<(const FixedSizeSequence& rhs) const{
		const int bases = length();
		const int otherbases = rhs.length();
		if(bases < otherbases) return true;
		if(bases > otherbases) return false;

		return (memcmp(begin(), rhs.begin(), getNumBytes()) < 0);
	}

	HOSTDEVICEQUALIFIER
	char operator[](int i) const{
                const int FIRST_USED_BYTE = getNumBytes() - SDIV(length(), BASES_PER_BYTE);
                const int UNUSED_BYTE_SPACE = (BASES_PER_BYTE - (length() % BASES_PER_BYTE)) % BASES_PER_BYTE;
		const int byte = FIRST_USED_BYTE + (i + UNUSED_BYTE_SPACE) / BASES_PER_BYTE;
		const int basepos = (i + UNUSED_BYTE_SPACE) % BASES_PER_BYTE;

		const uint8_t bits = (data[byte] >> (3-basepos) * 2) & 0x03;
		if(bits == 0x00) return 'A';
		if(bits == 0x01) return 'C';
		if(bits == 0x02) return 'G';
		if(bits == 0x03) return 'T';
		return '_';
	}

	HOSTDEVICEQUALIFIER
	void setBase(int pos, char base){
                const int FIRST_USED_BYTE = getNumBytes() - SDIV(length(), BASES_PER_BYTE);
                const int UNUSED_BYTE_SPACE = (BASES_PER_BYTE - (length() % BASES_PER_BYTE)) % BASES_PER_BYTE;
		const int byte = FIRST_USED_BYTE + (pos + UNUSED_BYTE_SPACE) / BASES_PER_BYTE;
		const int basepos = (pos + UNUSED_BYTE_SPACE) % BASES_PER_BYTE;

		uint8_t mask = 0x03 << (3-basepos)*2;
		uint8_t newbyte = data[byte] & ~mask;

                switch(base) {
                case 'A':
                        newbyte |=  0x00 << ((3-basepos) * 2);
                        break;
                case 'C':
                        newbyte |=  0x01 << ((3-basepos) * 2);
                        break;
                case 'G':
                        newbyte |=  0x02 << ((3-basepos) * 2);
                        break;
                case 'T':
                        newbyte |=  0x03 << ((3-basepos) * 2);
                        break;
		default:printf("error setBase\n");
                        newbyte |=  0x00 << ((3-basepos) * 2);
                        break;
		}
		data[byte] = newbyte;
	}

	std::string toString() const{
		char buf[MAX_LEN+1];
		bool b = decode(data, getNumBytes(), length(), buf, length()+1);
		if(!b)
			throw std::runtime_error("could not decode sequence");
		return std::string(buf);
	}

	FixedSizeSequence reverseComplement() const{
		FixedSizeSequence revcompl;
		revcompl.length = length();

		bool res = encoded_to_reverse_complement_encoded(begin(), getNumBytes(), revcompl.begin(), getNumBytes(), length());
		if(!res)
			throw std::runtime_error("could not get reverse complement of " + toString());

		return revcompl;
	}

	HOSTDEVICEQUALIFIER
	constexpr bool isCompressed() const{
		return true;
	}

	HOSTDEVICEQUALIFIER
	uint8_t* begin() const{
		return const_cast<uint8_t*>(&data[0]);
	}

	HOSTDEVICEQUALIFIER
	uint8_t* end() const{
		return &data[0] + getNumBytes();
	}
};
#endif

struct Sequence {

	Sequence();
	Sequence(const std::string& sequence);
	Sequence(const std::uint8_t* rawdata, int nBases_);
	Sequence(Sequence&& other);
	Sequence(const Sequence& other);
	Sequence& operator=(const Sequence& other);
	Sequence& operator=(Sequence&& other);
	bool operator==(const Sequence& rhs) const;
	bool operator!=(const Sequence& other) const;
	bool operator==(const std::string& other) const;
	bool operator!=(const std::string& other) const;
	bool operator<(const Sequence& rhs) const;
	char operator[](int i) const;
	std::string toString() const;
	Sequence reverseComplement() const;
	int getNumBytes() const;
	int length() const;
	bool isCompressed() const;
	std::uint8_t* begin() const;
	std::uint8_t* end() const;
	friend std::ostream& operator<<(std::ostream& stream, const Sequence& seq);

	static constexpr std::uint8_t BASE_A = 0x00;
	static constexpr std::uint8_t BASE_C = 0x01;
	static constexpr std::uint8_t BASE_G = 0x02;
	static constexpr std::uint8_t BASE_T = 0x03;

	std::pair<std::unique_ptr<std::uint8_t[]>, std::size_t> data;
	int nBases;
};

struct SequencePtrLess{
	bool operator() (const Sequence* lhs, const Sequence* rhs) const{
		const int bases = lhs->length();
		const int otherbases = rhs->length();
		if(bases < otherbases) return true;
		if(bases > otherbases) return false;

		return (std::memcmp(lhs->begin(), rhs->begin(), lhs->getNumBytes()) < 0);
	}
};


struct SequenceGeneral {

	SequenceGeneral();
	SequenceGeneral(const std::string& sequence, bool saveCompressed);
	SequenceGeneral(const std::uint8_t* rawdata, int nBases_, bool isCompressed);
	SequenceGeneral(SequenceGeneral&& other);
	SequenceGeneral(const SequenceGeneral& other);
	SequenceGeneral& operator=(const SequenceGeneral& other);
	SequenceGeneral& operator=(SequenceGeneral&& other);
	bool operator==(const SequenceGeneral& rhs) const;
	bool operator!=(const SequenceGeneral& other) const;
	bool operator==(const std::string& other) const;
	bool operator!=(const std::string& other) const;
	bool operator<(const SequenceGeneral& rhs) const;
	char operator[](int i) const;
	std::string toString() const;
	SequenceGeneral reverseComplement() const;
	int getNumBytes() const;
	int length() const;
	bool isCompressed() const;
	std::uint8_t* begin() const;
	std::uint8_t* end() const;
	friend std::ostream& operator<<(std::ostream& stream, const SequenceGeneral& seq);

	static constexpr std::uint8_t BASE_A = 0x00;
	static constexpr std::uint8_t BASE_C = 0x01;
	static constexpr std::uint8_t BASE_G = 0x02;
	static constexpr std::uint8_t BASE_T = 0x03;

	std::pair<std::unique_ptr<std::uint8_t[]>, std::size_t> data;
	int nBases = 0;
	bool compressed = false;
};

struct SequenceGeneralPtrLess{
	bool operator() (const SequenceGeneral* lhs, const SequenceGeneral* rhs) const{
		const int bases = lhs->length();
		const int otherbases = rhs->length();
		if(bases < otherbases) return true;
		if(bases > otherbases) return false;

		if(lhs->isCompressed() == rhs->isCompressed())
			return (std::memcmp(lhs->begin(), rhs->begin(), lhs->getNumBytes()) < 0);
		else{
			for(int i = 0; i < bases; i++){
				if (lhs->begin()[i] < rhs->begin()[i])
					return true;
				if (lhs->begin()[i] > rhs->begin()[i])
					return false;
			}
			return false;
		}
	}
};


struct Read {
	std::string header = "";
	std::string sequence = "";
	std::string quality = "";

	Read()
	{
	}

	Read(const Read& other)
	{
		*this = other;
	}

	Read(Read&& other)
	{
		*this = std::move(other);
	}

	Read& operator=(Read&& other){
		header = std::move(other.header);
		sequence = std::move(other.sequence);
		quality = std::move(other.quality);
		return *this;
	}

	Read& operator=(const Read& other)
	{
		header = other.header;
		sequence = other.sequence;
		quality = other.quality;
		return *this;
	}

	bool operator==(const Read& other) const
	{
		return (header == other.header && sequence == other.sequence && quality == other.quality);
	}
	bool operator!=(const Read& other) const
	{
		return !(*this == other);
	}

	void reset()
	{
		header.clear();
		sequence.clear();
		quality.clear();
	}
};


struct ReadReader {
protected:
	std::string filename;
public:
	ReadReader(const std::string& filename_) : filename(filename_)
	{
	};
	virtual ~ReadReader()
	{
	}

	//return false if EOF or if error occured while reading file. true otherwise
	// both sequence and sequencenumber are only valid if return value is true
	virtual bool getNextRead(Read* sequence, std::uint32_t* sequencenumber)
	{
		return false;
	};
	virtual void seekToRead(const int sequencenumber)
	{
	}
	virtual void reset()
	{
	}

};

struct ReadWriter {
public:
	ReadWriter(){}

	virtual ~ReadWriter(){}

	virtual void writeRead(std::ostream& stream, const Read& read) const
	{
		throw std::runtime_error("ReadWriter::writeRead not implemented");
	}
};


#endif
