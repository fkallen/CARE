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


struct Sequence {

	Sequence() : nBases(0), compressed(false)
	{
	}

	Sequence(const std::string& sequence, bool saveCompressed) 
		: nBases(sequence.length()), compressed(saveCompressed)
	{
	
		const int size = getNumBytes();

		data.reset(new std::uint8_t[size]);

		if(saveCompressed){
			bool retVal = encode(sequence.c_str(), nBases, nBases, begin(), size, true);

			if (!retVal){
				std::cout << sequence << " " << nBases << " " << size << std::endl;
				throw std::runtime_error("could not encode sequence");
			}
		}else{
			std::copy(sequence.begin(), sequence.end(), begin());
		}
	}

	Sequence(const std::uint8_t* rawdata, int nBases_, bool isCompressed) 
		: nBases(nBases_), compressed(isCompressed)
	{
		const int size = getNumBytes();

		data.reset(new std::uint8_t[size]);
		std::copy(rawdata, rawdata + size, begin());
	}

	Sequence(Sequence&& other)
	{
		*this = std::move(other);
	}

	Sequence(const Sequence& other)
	{
		*this = other;
	}

	Sequence& operator=(const Sequence& other)
	{
		nBases = other.nBases;
		compressed = other.compressed;

		const int size = getNumBytes();
		data.reset(new std::uint8_t[size]);
		std::copy(other.begin(), other.end(), begin());

		return *this;
	}

	Sequence& operator=(Sequence&& other){
		if(this != &other){
			nBases = other.nBases;
			compressed = other.compressed;

			data = std::move(other.data);

			other.nBases = 0;
		}
	        return *this;
	}

	bool operator==(const Sequence& other) const
	{
		return (nBases == other.nBases 
			&& compressed == other.compressed 
			&& 0 == std::memcmp(data.get(), other.data.get(), getNumBytes()));
	}

	bool operator!=(const Sequence& other) const
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

	char operator[](int i) const
	{
		if(compressed){
			const int UNUSED_BYTE_SPACE = (4 - (nBases % 4)) % 4;

			const int byte = (i + UNUSED_BYTE_SPACE) / 4;
			const int basepos = (i + UNUSED_BYTE_SPACE) % 4;

			switch ((data[byte] >> (3 - basepos) * 2) & 0x03) {
			case BASE_A: return 'A';
			case BASE_C: return 'C';
			case BASE_G: return 'G';
			case BASE_T: return 'T';
			default: return '_';         // cannot happen
			}
		}else{
			return char(data[i]);
		}

	}

	std::string toString() const
	{
		if(compressed){
			char c[nBases + 1];
			const int compressedSize = getNumBytes();

			bool retVal = decode(data.get(), compressedSize, nBases, c, nBases + 1);

			if (!retVal) throw std::runtime_error("could not decode compressed sequence");
			return std::string(c);
		}else{
			return std::string(reinterpret_cast<char*>(begin()), nBases);
		}
	}

	bool operator<(const Sequence& r) const{
		const int size = getNumBytes();
		const int othersize = r.getNumBytes();
		if(size < othersize) return true;
		if(size > othersize) return false;

		return (std::memcmp(begin(), r.begin(), size) < 0);
	}

	Sequence reverseComplement() const{
		if(isCompressed()){
			Sequence revcompl;
			revcompl.nBases = getNbases();
			revcompl.compressed = true;
			revcompl.data.reset(new std::uint8_t[getNumBytes()]);

			bool res = encoded_to_reverse_complement_encoded(begin(), getNumBytes(), revcompl.begin(), getNumBytes(), getNbases());
			if(!res)
				throw std::runtime_error("could not get reverse complement of " + toString());
			
			return revcompl;
		}else{
			Sequence revcompl;
			revcompl.nBases = getNbases();
			revcompl.compressed = false;
			revcompl.data.reset(new std::uint8_t[getNumBytes()]);

			std::reverse_copy(begin(), end(), revcompl.begin());

			for(char* p = (char*)revcompl.begin(); p < (char*)revcompl.end(); p++){
				switch(*p){
					case 'A': *p = 'T'; break;
					case 'C': *p = 'G'; break;
					case 'G': *p = 'C'; break;
					case 'T': *p = 'A'; break;
					default : break; // don't change N
				}
			}

			return revcompl;
		}
	}

	int getNumBytes() const{
		if(isCompressed())
			return SDIV(nBases,4);
		else
			return nBases;	
	}

	int getNbases() const{
		return nBases;
	}

	bool isCompressed() const{
		return compressed;
	}

	std::uint8_t* begin() const{
		return data.get();
	}

	std::uint8_t* end() const{
		return data.get() + getNumBytes();
	}

	friend std::ostream& operator<<(std::ostream& stream, const Sequence& seq){
		stream << seq.toString();
		return stream;
	}


	static constexpr std::uint8_t BASE_A = 0x00;
	static constexpr std::uint8_t BASE_C = 0x01;
	static constexpr std::uint8_t BASE_G = 0x02;
	static constexpr std::uint8_t BASE_T = 0x03;

	std::unique_ptr<std::uint8_t[]> data;
	int nBases = 0;
	bool compressed = false;
};

struct SequencePtrLess{
	bool operator() (const Sequence* lhs, const Sequence* rhs) const{
		const int size = lhs->getNumBytes();
		const int othersize = rhs->getNumBytes();
		if(size < othersize) return true;
		if(size > othersize) return false;

		return (std::memcmp(lhs->begin(), rhs->begin(), size) < 0);
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
