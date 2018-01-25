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
	int getNbases() const;
	bool isCompressed() const;
	std::uint8_t* begin() const;
	std::uint8_t* end() const;
	friend std::ostream& operator<<(std::ostream& stream, const Sequence& seq);

	static constexpr std::uint8_t BASE_A = 0x00;
	static constexpr std::uint8_t BASE_C = 0x01;
	static constexpr std::uint8_t BASE_G = 0x02;
	static constexpr std::uint8_t BASE_T = 0x03;

	std::pair<std::unique_ptr<std::uint8_t[]>, std::size_t> data;
	int nBases = 0;
};

struct SequencePtrLess{
	bool operator() (const Sequence* lhs, const Sequence* rhs) const{
		const int bases = lhs->getNbases();
		const int otherbases = rhs->getNbases();
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
	int getNbases() const;
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
		const int bases = lhs->getNbases();
		const int otherbases = rhs->getNbases();
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
