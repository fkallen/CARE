#ifndef READ_HPP
#define READ_HPP

#include "binarysequencehelpers.hpp"
#include "hpc_helpers.cuh"

#include <algorithm>
#include <string>
#include <cstdint>
#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <cassert>

namespace care{

#if 1
template<int MAX_LEN_>
struct FixedSizeSequence{
	static constexpr uint8_t BITS_PER_BASE = 2;
	static constexpr uint8_t BASES_PER_BYTE = (sizeof(uint8_t)*8 / BITS_PER_BASE);

	static constexpr int MAX_LEN = MAX_LEN_;
	static constexpr int DATA_BYTES = SDIV(MAX_LEN, 4);

	uint8_t data[DATA_BYTES];
	int length_ = 0;

	HOSTDEVICEQUALIFIER
	constexpr int getNumBytes() const{
		return DATA_BYTES;
	}

	HOSTDEVICEQUALIFIER
	int length() const{
		return length_;
	}

    HOSTDEVICEQUALIFIER
	int& length(){
		return length_;
	}

	HOSTDEVICEQUALIFIER
	FixedSizeSequence(){
		memset(data, 0, DATA_BYTES);
	}

	FixedSizeSequence(const std::string& sequence) : length_(sequence.size()){
		if(sequence.size() > MAX_LEN || !encode(sequence.c_str(), sequence.size(), sequence.size(), data, DATA_BYTES, true)){
			throw std::runtime_error("could not encode sequence");
		}
	}

	FixedSizeSequence(const uint8_t* rawdata, int nBases_) : length_(nBases_){
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
		length() = other.length();
		memcpy(data, other.data, sizeof(uint8_t) * DATA_BYTES);
        return *this;
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
     // static access to i-th base in data
    static char get(const char* data, int length, int index) noexcept{
        const int byte = index / 4;
        const int basepos = index % 4;
        switch((data[byte] >> (3-basepos) * 2) & 0x03){
            case 0x00: return 'A';
            case 0x01: return 'C';
            case 0x02: return 'G';
            case 0x03: return 'T';
            default:return '_'; //cannot happen
        }
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
		revcompl.length() = length();

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

/*
    Sequence class which stores sequence in consecutive 2bit encoding
*/

struct Sequence {

	Sequence() noexcept;
    ~Sequence() noexcept;
	Sequence(const std::string& sequence) noexcept;
	Sequence(const std::uint8_t* rawdata, int nBases_) noexcept;
	Sequence(Sequence&& other) noexcept;
	Sequence(const Sequence& other) noexcept;
	Sequence& operator=(const Sequence& other) noexcept;
	Sequence& operator=(Sequence&& other) noexcept;
	bool operator==(const Sequence& rhs) const noexcept;
	bool operator!=(const Sequence& other) const noexcept;
	bool operator==(const std::string& other) const noexcept;
	bool operator!=(const std::string& other) const noexcept;
	bool operator<(const Sequence& rhs) const noexcept;
	char operator[](int i) const noexcept;
	std::string toString() const noexcept;
	Sequence reverseComplement() const noexcept;
	int getNumBytes() const noexcept;
	int length() const noexcept;
	bool isCompressed() const noexcept;
	std::uint8_t* begin() const noexcept;
	std::uint8_t* end() const noexcept;
	friend std::ostream& operator<<(std::ostream& stream, const Sequence& seq);

    HOSTDEVICEQUALIFIER
     // static access to i-th base in data
    static char get(const char* data, int length, int index) noexcept{
		const int byte = index / 4;
		const int basepos = index % 4;
		switch((data[byte] >> (3-basepos) * 2) & 0x03){
			case 0x00: return 'A';
			case 0x01: return 'C';
			case 0x02: return 'G';
			case 0x03: return 'T';
			default:return '_'; //cannot happen
		}
	}

    HOSTDEVICEQUALIFIER
    // return number of data bytes for a sequence of length l
    static constexpr int number_of_bytes(int l) noexcept{
        return SDIV(l, 4);
    }

	std::pair<std::unique_ptr<std::uint8_t[]>, std::size_t> data;
	int nBases;
};

struct SequencePtrLess{
	bool operator() (const Sequence* lhs, const Sequence* rhs) const noexcept{
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


static_assert(sizeof(std::uint8_t) == sizeof(char));


struct SequenceStringImpl{
    static std::pair<std::unique_ptr<std::uint8_t[]>, std::size_t> encode(const std::string& sequence){
        int bytes = getNumBytes(sequence.size());

        auto ptr = std::make_unique<std::uint8_t[]>(bytes);

        std::copy(sequence.begin(), sequence.end(), ptr.get());
        return {std::move(ptr), bytes};
    }

    HOSTDEVICEQUALIFIER
    static int getNumBytes(int nBases) noexcept{
        return nBases;
    }

    HOSTDEVICEQUALIFIER
    static char get(const char* data, int nBases, int i) noexcept{
        return data[i];
    }

    static std::string toString(const std::uint8_t* data, int nBases){
        return std::string((const char*)data);
    }

    static std::pair<std::unique_ptr<std::uint8_t[]>, std::size_t> reverseComplement(const std::uint8_t* data, int nBases){
        int bytes = getNumBytes(nBases);

        auto ptr = std::make_unique<std::uint8_t[]>(bytes);

        std::reverse_copy(data, data + bytes, ptr.get());

        for(int i = 0; i < bytes; ++i){
            switch(ptr[i]){
                case 'A': ptr[i] = 'T'; break;
                case 'C': ptr[i] = 'G'; break;
                case 'G': ptr[i] = 'C'; break;
                case 'T': ptr[i] = 'A'; break;
                default : break; // don't change N
            }
        }

        return {std::move(ptr), bytes};
    }

    HOSTDEVICEQUALIFIER
    static constexpr bool isCompressed() noexcept{
        return false;
    }

    HOSTDEVICEQUALIFIER
    static void make_reverse_complement(std::uint8_t* reverseComplement, const std::uint8_t* sequence, int sequencelength){
        int bytes = getNumBytes(sequencelength);

        for(int i = 0; i < bytes; ++i){
            switch(sequence[i]){
                case 'A': reverseComplement[bytes-1-i] = 'T'; break;
                case 'C': reverseComplement[bytes-1-i] = 'G'; break;
                case 'G': reverseComplement[bytes-1-i] = 'C'; break;
                case 'T': reverseComplement[bytes-1-i] = 'A'; break;
                default : break; // don't change N
            }
        }
    };
};


struct Sequence2BitImpl{
    static std::pair<std::unique_ptr<std::uint8_t[]>, std::size_t> encode(const std::string& sequence){
        return encode_2bit(sequence);
    }

    HOSTDEVICEQUALIFIER
    static int getNumBytes(int nBases) noexcept{
        return SDIV(nBases, 4);
    }

    HOSTDEVICEQUALIFIER
    static char get(const char* data, int nBases, int i) noexcept{
        const int byte = i / 4;
        const int basepos = i % 4;
        switch((data[byte] >> (3-basepos) * 2) & 0x03){
            case 0x00: return 'A';
            case 0x01: return 'C';
            case 0x02: return 'G';
            case 0x03: return 'T';
            default:return '_'; //cannot happen
        }
    }

    static std::string toString(const std::uint8_t* data, int nBases){
        return decode_2bit(data, nBases);
    }

    static std::pair<std::unique_ptr<std::uint8_t[]>, std::size_t> reverseComplement(const std::uint8_t* data, int nBases){
        return reverse_complement_2bit(data, nBases);
    }

    HOSTDEVICEQUALIFIER
    static constexpr bool isCompressed() noexcept{
        return true;
    }

    HOSTDEVICEQUALIFIER
    static void make_reverse_complement(std::uint8_t* reverseComplement, const std::uint8_t* sequence, int sequencelength){
        auto make_reverse_complement_byte = [](std::uint8_t in) -> std::uint8_t{
            in = ((in >> 2)  & 0x3333u) | ((in & 0x3333u) << 2);
            in = ((in >> 4)  & 0x0F0Fu) | ((in & 0x0F0Fu) << 4);
            return (std::uint8_t(-1) - in) >> (8 * 1 - (4 << 1));
        };

        const int bytes = getNumBytes(sequencelength);
        const int unusedPositions = bytes * 4 - sequencelength;

        for(int i = 0; i < bytes; i++){
            reverseComplement[i] = make_reverse_complement_byte(sequence[bytes - 1 - i]);
        }

        if(unusedPositions > 0){
            reverseComplement[0] <<= (2 * unusedPositions);
            for(int i = 1; i < bytes; i++){
                reverseComplement[i-1] |= reverseComplement[i] >> (2 * (4-unusedPositions));
                reverseComplement[i] <<= (2 * unusedPositions);
            }
        }
    };
};

struct Sequence2BitHiLoImpl{
    static std::pair<std::unique_ptr<std::uint8_t[]>, std::size_t> encode(const std::string& sequence){
        return encode_2bit_hilo(sequence);
    }

    HOSTDEVICEQUALIFIER
    static int getNumBytes(int nBases) noexcept{
        return int(2 * SDIV(nBases, sizeof(std::uint8_t) * 8));
    }

    HOSTDEVICEQUALIFIER
    static char get(const char* data, int nBases, int i) noexcept{
        const int bytes = getNumBytes(nBases);

        const unsigned char* const hi = (const unsigned char*)data;
        const unsigned char* const lo = hi + bytes/2;

        const int byteIndex = i / 8;
        const int pos = i % 8;
        const unsigned char hibit = (hi[byteIndex] >> (7-pos)) & std::uint8_t(1);
        const unsigned char lobit = (lo[byteIndex] >> (7-pos)) & std::uint8_t(1);
        const unsigned char base = (hibit << 1) | lobit;

        switch(base){
            case 0x00: return 'A';
            case 0x01: return 'C';
            case 0x02: return 'G';
            case 0x03: return 'T';
            default:return '_'; //cannot happen
        }
    }

    static std::string toString(const std::uint8_t* data, int nBases){
        return decode_2bit_hilo(data, nBases);
    }

    static std::pair<std::unique_ptr<std::uint8_t[]>, std::size_t> reverseComplement(const std::uint8_t* data, int nBases){
        const int bytes = getNumBytes(nBases);

        std::unique_ptr<std::uint8_t[]> reverseComplement = std::make_unique<std::uint8_t[]>(bytes);

    	make_reverse_complement(reverseComplement.get(), data, nBases);

        return {std::move(reverseComplement), bytes};
    }

    HOSTDEVICEQUALIFIER
    static constexpr bool isCompressed() noexcept{
        return true;
    }

    HOSTDEVICEQUALIFIER
    static void make_reverse_complement(std::uint8_t* reverseComplement, const std::uint8_t* sequence, int sequencelength){
        const int bytes = getNumBytes(sequencelength);
        const int halfbytes = bytes / 2;
        const int unusedBits = sequencelength % 8;

        const std::uint8_t* hiOrig = sequence;
        const std::uint8_t* loOrig = hiOrig + halfbytes;

        std::uint8_t* hiRevC = reverseComplement;
        std::uint8_t* loRevC = hiRevC + halfbytes;

        auto reverse_complement_byte = [](auto b) {
    		b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
    		b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
    		b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
    		return ~b;
    	};

        for(int i = 0; i < halfbytes; ++i){
            hiRevC[i] = reverse_complement_byte(hiOrig[halfbytes - 1 - i]);
            loRevC[i] = reverse_complement_byte(loOrig[halfbytes - 1 - i]);
        }

        if(unusedBits != 0){
            for(int i = 0; i < halfbytes - 1; ++i){
                hiRevC[i] = (hiRevC[i] << unusedBits) | (hiRevC[i+1] >> (8 - unusedBits));
                loRevC[i] = (loRevC[i] << unusedBits) | (loRevC[i+1] >> (8 - unusedBits));
            }
            hiRevC[halfbytes-1] <<= unusedBits;
            loRevC[halfbytes-1] <<= unusedBits;
        }
    };
};




template<class Impl>
struct SequenceBase {

    using Impl_t = Impl;

	SequenceBase() : nBases(0){
        data.second = 0;
    }

    ~SequenceBase(){}

    SequenceBase(const std::string& sequence)
		: nBases(sequence.length()){
		data = Impl::encode(sequence);
	}

    SequenceBase(const std::uint8_t* rawdata, int nBases_) noexcept
		: nBases(nBases_){

		const int size = getNumBytes();
		data.first = std::make_unique<std::uint8_t[]>(size);
		data.second = size;

		std::copy(rawdata, rawdata + size, begin());
	}

	SequenceBase(SequenceBase&& other) noexcept{
        operator=(other);
    }

	SequenceBase(const SequenceBase& other) : nBases(other.nBases){
		const int size = other.getNumBytes();
		data.first = std::make_unique<std::uint8_t[]>(size);
		data.second = size;

		std::copy(other.begin(), other.end(), begin());
    }

	SequenceBase& operator=(const SequenceBase& other){
        SequenceBase tmp(other);
        swap(*this, tmp);
        return *this;
    }

    SequenceBase& operator=(SequenceBase&& other){
        swap(*this, other);
        return *this;
    }

    bool operator==(const SequenceBase& rhs) const noexcept{
        if(length() != rhs.length()) return false;
        return (std::memcmp(begin(), rhs.begin(), getNumBytes()) == 0);
    }

    bool operator!=(const SequenceBase& other) const noexcept{
        return !(*this == other);
    }

    bool operator==(const std::string& other) const noexcept{
        return toString() == other;
    }

    bool operator!=(const std::string& other) const noexcept{
        return !(*this == other);
    }

    bool operator<(const SequenceBase& rhs) const noexcept{
		const int bases = length();
		const int otherbases = rhs.length();
		if(bases < otherbases) return true;
		if(bases > otherbases) return false;

		return (std::memcmp(begin(), rhs.begin(), getNumBytes()) < 0);
	}

	char operator[](int i) const noexcept{
        return Impl::get((const char*)data.first.get(), nBases, i);
    }

	std::string toString() const{
        return Impl::toString(data.first.get(), nBases);
    }

	SequenceBase reverseComplement() const{
        SequenceBase revCompl;
        revCompl.nBases = nBases;
        revCompl.data = Impl::reverseComplement(data.first.get(), nBases);
        return revCompl;
    }

	int getNumBytes() const noexcept{
        return Impl::getNumBytes(nBases);
    }

	int length() const noexcept{
        return nBases;
    }

	bool isCompressed() const noexcept{
        return Impl::isCompressed();
    }

    std::uint8_t* begin() const noexcept{
        return data.first.get();
    }

    std::uint8_t* end() const noexcept{
        return data.first.get() + getNumBytes();
    }

    HOSTDEVICEQUALIFIER
    static char get(const char* data, int nBases, int i) noexcept{
		return Impl::get(data, nBases, i);
	}

    HOSTDEVICEQUALIFIER
    static int getNumBytes(int nBases) noexcept{
        return Impl::getNumBytes(nBases);
    }

    HOSTDEVICEQUALIFIER
    static void make_reverse_complement(std::uint8_t* reverseComplement, const std::uint8_t* sequence, int sequencelength){
        return Impl::make_reverse_complement(reverseComplement, sequence, sequencelength);
    };

	std::pair<std::unique_ptr<std::uint8_t[]>, int> data;
	int nBases;

public:
    friend void swap(SequenceBase& l, SequenceBase& r) noexcept{
        using std::swap;

        swap(l.data, r.data);
        swap(l.nBases, r.nBases);
    }

    friend std::ostream& operator<<(std::ostream& stream, const SequenceBase& seq){
        stream << seq.toString();
        return stream;
    }
};

using Sequence2Bit = SequenceBase<Sequence2BitImpl>;
using Sequence2BitHiLo = SequenceBase<Sequence2BitHiLoImpl>;
using SequenceString = SequenceBase<SequenceStringImpl>;

}
#endif
