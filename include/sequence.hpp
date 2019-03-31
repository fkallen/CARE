#ifndef READ_HPP
#define READ_HPP

#include <config.hpp>


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
        return std::string((const char*)data, nBases);
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

    HOSTDEVICEQUALIFIER
    static void make_reverse_complement_inplace(std::uint8_t* sequence, int sequencelength){

        auto make_reverse_complement_byte = [](std::uint8_t in) -> std::uint8_t{
            switch(in){
                case 'A': return 'T';
                case 'C': return 'G';
                case 'G': return 'C';
                case 'T': return 'A';
                default :return 'F';
            }
        };

        const int bytes = getNumBytes(sequencelength);

        for(int i = 0; i < bytes/2; i++){
            const std::uint8_t front = make_reverse_complement_byte(sequence[i]);
            const std::uint8_t back = make_reverse_complement_byte(sequence[bytes - 1 - i]);
            sequence[i] = back;
            sequence[bytes - 1 - i] = front;
        }

        if(bytes % 2 == 1){
            const int middleindex = bytes/2;
            sequence[middleindex] = make_reverse_complement_byte(sequence[middleindex]);
        }
    };

};


struct Sequence2BitImpl{
    static constexpr int basesPerInt = sizeof(std::uint32_t) * 8 / 2;

    static constexpr std::uint32_t BASE_A = 0x00000000;
    static constexpr std::uint32_t BASE_C = 0x00000001;
    static constexpr std::uint32_t BASE_G = 0x00000002;
    static constexpr std::uint32_t BASE_T = 0x00000003;

    static std::pair<std::unique_ptr<std::uint8_t[]>, std::size_t> encode(const std::string& sequence){
        //return encode_2bit(sequence);

        const std::size_t l = sequence.length();
        const std::size_t nInts = getNumBytes(l) / sizeof(std::uint32_t);

        std::unique_ptr<std::uint8_t[]> encoded = std::make_unique<std::uint8_t[]>(nInts * sizeof(std::uint32_t));
        std::uint32_t* const data = (std::uint32_t*)encoded.get();

        std::memset(encoded.get(), 0, nInts * sizeof(std::uint32_t));

        for(std::size_t i = 0; i < l; i++){
            const std::size_t intIndex = i / basesPerInt;
            const std::size_t pos = i % basesPerInt;
            switch(sequence[i]) {
            case 'A':
                    data[intIndex] |= BASE_A << (2*((basesPerInt - 1)-pos));
                    break;
            case 'C':
                    data[intIndex] |= BASE_C << (2*((basesPerInt - 1)-pos));
                    break;
            case 'G':
                    data[intIndex] |= BASE_G << (2*((basesPerInt - 1)-pos));
                    break;
            case 'T':
                    data[intIndex] |= BASE_T << (2*((basesPerInt - 1)-pos));
                    break;
            default:
                    data[intIndex] |= BASE_A << (2*((basesPerInt - 1)-pos));
                    break;
            }
        }

        return {std::move(encoded), nInts * sizeof(std::uint32_t)};
    }

    HOSTDEVICEQUALIFIER
    static int getNumBytes(int nBases) noexcept{
        const std::size_t nInts = SDIV(nBases, basesPerInt);

        return nInts * sizeof(std::uint32_t);
    }

    HOSTDEVICEQUALIFIER
    static char get(const char* data, int nBases, int i) noexcept{
        const int intIndex = i / basesPerInt;
        const int pos = i % basesPerInt;
        const std::uint32_t* const udata = (const std::uint32_t*)data;

        return (char)((udata[intIndex] >> ((basesPerInt-1-pos) * 2)) & 0x00000003);
    }

    static std::string toString(const std::uint8_t* data, int nBases){
        //return decode_2bit(data, nBases);

        const std::uint32_t* const udata = (const std::uint32_t*)data;

        std::string sequence;
    	sequence.reserve(nBases);

    	for(int i = 0; i < nBases; i++){
            const int intIndex = i / basesPerInt;
            const int pos = i % basesPerInt;
            const std::uint8_t base = ((udata[intIndex] >> ((basesPerInt-1-pos) * 2)) & 0x00000003);
            switch(base){
                case BASE_A: sequence.push_back('A'); break;
                case BASE_C: sequence.push_back('C'); break;
                case BASE_G: sequence.push_back('G'); break;
                case BASE_T: sequence.push_back('T'); break;
                default: sequence.push_back('_'); break; // cannot happen
            }
    	}

    	return sequence;
    }

    static std::pair<std::unique_ptr<std::uint8_t[]>, std::size_t> reverseComplement(const std::uint8_t* data, int nBases){
        //return reverse_complement_2bit(data, nBases);
        const int bytes = getNumBytes(nBases);

        std::unique_ptr<std::uint8_t[]> reverseComplement = std::make_unique<std::uint8_t[]>(bytes);

        //encoded_to_reverse_complement_encoded(data, bytes, reverseComplement.get(), bytes, nBases);
        make_reverse_complement(reverseComplement.get(), data, nBases);

        return {std::move(reverseComplement), bytes};
    }

    HOSTDEVICEQUALIFIER
    static constexpr bool isCompressed() noexcept{
        return true;
    }

    HOSTDEVICEQUALIFIER
    static void make_reverse_complement(std::uint8_t* reverseComplement, const std::uint8_t* sequence, int sequencelength){

        auto make_reverse_complement_int = [](std::uint32_t s){
            s = ((s >> 2)  & 0x33333333u) | ((s & 0x33333333u) << 2);
            s = ((s >> 4)  & 0x0F0F0F0Fu) | ((s & 0x0F0F0F0Fu) << 4);
            s = ((s >> 8)  & 0x00FF00FFu) | ((s & 0x00FF00FFu) << 8);
            s = ((s >> 16) & 0x0000FFFFu) | ((s & 0x0000FFFFu) << 16);
            return (std::uint32_t(-1) - s) >> (8 * sizeof(s) - (16 << 1));
        };

        const std::uint32_t* const usequence = (const std::uint32_t*)sequence;
        std::uint32_t* const ureverseComplement = (std::uint32_t*)reverseComplement;

        const int nInts = getNumBytes(sequencelength) / sizeof(std::uint32_t);
        const int unusedPositions = nInts * basesPerInt - sequencelength;

        for(int i = 0; i < nInts; i++){
            ureverseComplement[i] = make_reverse_complement_int(usequence[nInts - 1 - i]);
        }

        if(unusedPositions > 0){
            ureverseComplement[0] <<= (2 * unusedPositions);
            for(int i = 1; i < nInts; i++){
                ureverseComplement[i-1] |= ureverseComplement[i] >> (2 * (basesPerInt-unusedPositions));
                ureverseComplement[i] <<= (2 * unusedPositions);
            }
        }
    };

    HOSTDEVICEQUALIFIER
    static void make_reverse_complement_inplace(std::uint8_t* sequence, int sequencelength){

        auto make_reverse_complement_int = [](std::uint32_t s){
            s = ((s >> 2)  & 0x33333333u) | ((s & 0x33333333u) << 2);
            s = ((s >> 4)  & 0x0F0F0F0Fu) | ((s & 0x0F0F0F0Fu) << 4);
            s = ((s >> 8)  & 0x00FF00FFu) | ((s & 0x00FF00FFu) << 8);
            s = ((s >> 16) & 0x0000FFFFu) | ((s & 0x0000FFFFu) << 16);
            return (std::uint32_t(-1) - s) >> (8 * sizeof(s) - (16 << 1));
        };

        std::uint32_t* const usequence = (std::uint32_t*)sequence;

        const int nInts = getNumBytes(sequencelength) / sizeof(std::uint32_t);
        const int unusedPositions = nInts * basesPerInt - sequencelength;

        for(int i = 0; i < nInts/2; i++){
            const std::uint32_t front = make_reverse_complement_int(usequence[i]);
            const std::uint32_t back = make_reverse_complement_int(usequence[nInts - 1 - i]);
            usequence[i] = back;
            usequence[nInts - 1 - i] = front;
        }

        if(nInts % 2 == 1){
            const int middleindex = nInts/2;
            usequence[middleindex] = make_reverse_complement_int(usequence[middleindex]);
        }

        if(unusedPositions > 0){
            usequence[0] <<= (2 * unusedPositions);
            for(int i = 1; i < nInts; i++){
                usequence[i-1] |= usequence[i] >> (2 * (basesPerInt-unusedPositions));
                usequence[i] <<= (2 * unusedPositions);
            }
        }
    };

};

struct Sequence2BitHiLoImpl{
    using PaddingType = unsigned int;
    //using PaddingType = unsigned char;

    static std::pair<std::unique_ptr<std::uint8_t[]>, std::size_t> encode(const std::string& sequence){
        //return encode_2bit_hilo(sequence);
        const int nbases = sequence.size();
        const int bytes = getNumBytes(nbases);

    	std::unique_ptr<std::uint8_t[]> encoded = std::make_unique<std::uint8_t[]>(bytes);

    	std::memset(encoded.get(), 0, bytes);

        unsigned int* const hi = (unsigned int*)encoded.get();
        unsigned int* const lo = (unsigned int*)(encoded.get() + bytes/2);

    	for(int i = 0; i < nbases; i++){
    		const int intIndex = i / (8 * sizeof(unsigned int));
    		const int pos = i % (8 * sizeof(unsigned int));
            //unsigned int mask = 1u << ((8 * sizeof(unsigned int)) - 1 - pos);
            unsigned int mask = 1u << pos;

            switch(sequence[i]) {
            case 'A':
                    hi[intIndex] &= ~mask;
                    lo[intIndex] &= ~mask;
                    break;
            case 'C':
                    hi[intIndex] &= ~mask;
                    lo[intIndex] |= mask;
                    break;
            case 'G':
                    hi[intIndex] |= mask;
                    lo[intIndex] &= ~mask;
                    break;
            case 'T':
                    hi[intIndex] |= mask;
                    lo[intIndex] |= mask;
                    break;
            default:
                    hi[intIndex] &= ~mask;
                    lo[intIndex] &= ~mask;
                    break;
            }
    	}

    	return {std::move(encoded), bytes};
    }

    HOSTDEVICEQUALIFIER
    static int getNumBytes(int nBases) noexcept{
        return int(2 * SDIV(nBases, sizeof(PaddingType) * 8) * sizeof(PaddingType));
    }

    HOSTDEVICEQUALIFIER
    static char get(const char* data, int nBases, int i) noexcept{
        const int bytes = getNumBytes(nBases);

        const unsigned int* const hi = (const unsigned int*)data;
        const unsigned int* const lo = (const unsigned int*)(data + bytes/2);

        const int intIndex = i / (8 * sizeof(unsigned int));
        const int pos = i % (8 * sizeof(unsigned int));
        const unsigned char hibit = (hi[intIndex] >> pos) & 1u;
        const unsigned char lobit = (lo[intIndex] >> pos) & 1u;
        const unsigned char base = (hibit << 1) | lobit;

        /*switch(base){
            case 0x00: return 'A';
            case 0x01: return 'C';
            case 0x02: return 'G';
            case 0x03: return 'T';
            default:return '_'; //cannot happen
        }*/

        return (char)base;
    }

    static std::string toString(const std::uint8_t* data, int nBases){
        //return decode_2bit_hilo(data, nBases);

        constexpr std::uint8_t BASE_A = 0x00;
        constexpr std::uint8_t BASE_C = 0x01;
        constexpr std::uint8_t BASE_G = 0x02;
        constexpr std::uint8_t BASE_T = 0x03;

        std::string sequence;
        sequence.reserve(nBases);

        const int bytes = getNumBytes(nBases);

        const unsigned int* const hi = (const unsigned int*)data;
        const unsigned int* const lo = (const unsigned int*)(data + bytes/2);

#if 1

        for(int i = 0; i < nBases; i++){
            const int intIndex = i / (8 * sizeof(unsigned int));
            const int pos = i % (8 * sizeof(unsigned int));

            const unsigned char hibit = (hi[intIndex] >> pos) & 1u;
            const unsigned char lobit = (lo[intIndex] >> pos) & 1u;
            const unsigned char base = (hibit << 1) | lobit;

            switch(base){
            case BASE_A: sequence.push_back('A'); break;
            case BASE_C: sequence.push_back('C'); break;
            case BASE_G: sequence.push_back('G'); break;
            case BASE_T: sequence.push_back('T'); break;
            default: sequence.push_back('_'); break; // cannot happen
            }
        }

#else
        int previousIntIndex = -1;
        std::uint64_t interleaved = 0;

        //return 64 interleaved bits of two 32 bit numbers
        auto interleave = [](std::uint64_t a, std::uint64_t b){
                  a = (a | (a << 16)) & 0x0000FFFF0000FFFF;
                  a = (a | (a << 8)) & 0x00FF00FF00FF00FF;
                  a = (a | (a << 4)) & 0x0F0F0F0F0F0F0F0F;
                  a = (a | (a << 2)) & 0x3333333333333333;
                  a = (a | (a << 1)) & 0x5555555555555555;

                  b = (b | (b << 16)) & 0x0000FFFF0000FFFF;
                  b = (b | (b << 8)) & 0x00FF00FF00FF00FF;
                  b = (b | (b << 4)) & 0x0F0F0F0F0F0F0F0F;
                  b = (b | (b << 2)) & 0x3333333333333333;
                  b = (b | (b << 1)) & 0x5555555555555555;

                  return (a << 1) | b;
          };

        for(int i = 0; i < nBases; i++){
            const int intIndex = i / (8 * sizeof(unsigned int));

            if(intIndex != previousIntIndex){
                interleaved = interleave(hi[intIndex], lo[intIndex]);
                previousIntIndex = intIndex;
            }

            const int pos = i % (8 * sizeof(unsigned int));

            const unsigned char base = (interleaved >> (2*pos)) & 3;

            switch(base){
            case BASE_A: sequence.push_back('A'); break;
            case BASE_C: sequence.push_back('C'); break;
            case BASE_G: sequence.push_back('G'); break;
            case BASE_T: sequence.push_back('T'); break;
            default: sequence.push_back('_'); break; // cannot happen
            }
        }

#endif

        return sequence;
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

        auto reverse_complement_int = [](auto n) {
            n = ((n >> 1) & 0x55555555) | ((n << 1) & 0xaaaaaaaa);
            n = ((n >> 2) & 0x33333333) | ((n << 2) & 0xcccccccc);
            n = ((n >> 4) & 0x0f0f0f0f) | ((n << 4) & 0xf0f0f0f0);
            n = ((n >> 8) & 0x00ff00ff) | ((n << 8) & 0xff00ff00);
            n = ((n >> 16) & 0x0000ffff) | ((n << 16) & 0xffff0000);
            return ~n;
        };

        const int bytes = getNumBytes(sequencelength);
        const int unusedBitsInt = SDIV(sequencelength, 8 * sizeof(unsigned int)) * 8 * sizeof(unsigned int) - sequencelength;

        const unsigned int* const hiOrig = (const unsigned int*)sequence;
        const unsigned int* const loOrig = (const unsigned int*)(sequence + bytes/2);

        unsigned int* const hiRevC = (unsigned int*)reverseComplement;
        unsigned int* const loRevC = (unsigned int*)(reverseComplement + bytes/2);

        const int intsPerHalf = SDIV(sequencelength, 8 * sizeof(unsigned int));
        for(int i = 0; i < intsPerHalf; ++i){
            hiRevC[i] = reverse_complement_int(hiOrig[intsPerHalf - 1 - i]);
            loRevC[i] = reverse_complement_int(loOrig[intsPerHalf - 1 - i]);
        }

        if(unusedBitsInt != 0){
            for(int i = 0; i < intsPerHalf - 1; ++i){
                hiRevC[i] = (hiRevC[i] >> unusedBitsInt) | (hiRevC[i+1] << (8 * sizeof(unsigned int) - unusedBitsInt));
                loRevC[i] = (loRevC[i] >> unusedBitsInt) | (loRevC[i+1] << (8 * sizeof(unsigned int) - unusedBitsInt));
            }

            hiRevC[intsPerHalf - 1] >>= unusedBitsInt;
            loRevC[intsPerHalf - 1] >>= unusedBitsInt;
        }
    };
#if 1
    HOSTDEVICEQUALIFIER
    static void make_reverse_complement_inplace(std::uint8_t* sequence, int sequencelength){

        auto reverse_complement_int = [](auto n) {
            n = ((n >> 1) & 0x55555555) | ((n << 1) & 0xaaaaaaaa);
            n = ((n >> 2) & 0x33333333) | ((n << 2) & 0xcccccccc);
            n = ((n >> 4) & 0x0f0f0f0f) | ((n << 4) & 0xf0f0f0f0);
            n = ((n >> 8) & 0x00ff00ff) | ((n << 8) & 0xff00ff00);
            n = ((n >> 16) & 0x0000ffff) | ((n << 16) & 0xffff0000);
            return ~n;
        };

        const int bytes = getNumBytes(sequencelength);
        const int unusedBitsInt = SDIV(sequencelength, 8 * sizeof(unsigned int)) * 8 * sizeof(unsigned int) - sequencelength;

        unsigned int* const hi = (unsigned int*)sequence;
        unsigned int* const lo = (unsigned int*)(sequence + bytes/2);

        const int intsPerHalf = SDIV(sequencelength, 8 * sizeof(unsigned int));
        for(int i = 0; i < intsPerHalf/2; ++i){
            const unsigned int hifront = reverse_complement_int(hi[i]);
            const unsigned int hiback = reverse_complement_int(hi[intsPerHalf - 1 - i]);
            hi[i] = hiback;
            hi[intsPerHalf - 1 - i] = hifront;

            const unsigned int lofront = reverse_complement_int(lo[i]);
            const unsigned int loback = reverse_complement_int(lo[intsPerHalf - 1 - i]);
            lo[i] = loback;
            lo[intsPerHalf - 1 - i] = lofront;
        }
        if(intsPerHalf % 2 == 1){
            const int middleindex = intsPerHalf/2;
            hi[middleindex] = reverse_complement_int(hi[middleindex]);
            lo[middleindex] = reverse_complement_int(lo[middleindex]);
        }

        if(unusedBitsInt != 0){
            for(int i = 0; i < intsPerHalf - 1; ++i){
                hi[i] = (hi[i] >> unusedBitsInt) | (hi[i+1] << (8 * sizeof(unsigned int) - unusedBitsInt));
                lo[i] = (lo[i] >> unusedBitsInt) | (lo[i+1] << (8 * sizeof(unsigned int) - unusedBitsInt));
            }

            hi[intsPerHalf - 1] >>= unusedBitsInt;
            lo[intsPerHalf - 1] >>= unusedBitsInt;
        }
    };
#else

HOSTDEVICEQUALIFIER
static void make_reverse_complement_inplace(std::uint8_t* sequence, int sequencelength){

    auto make_reverse_complement_byte = [](auto b) {
        b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
        b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
        b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
        return ~b;
    };

    const int bytes = getNumBytes(sequencelength);
    const int halfbytes = bytes / 2;
    //const int unusedBitsByte = halfbytes*8 - sequencelength;
    const int unusedBitsInLastUsedByte = SDIV(sequencelength, 8) * 8 - sequencelength;

    std::uint8_t* const hiBytes = sequence;
    std::uint8_t* const loBytes = sequence + halfbytes;

    const int usedBytesPerHalf = SDIV(sequencelength, 8);
    for(int i = 0; i < usedBytesPerHalf/2; ++i){
        const std::uint8_t hifront = make_reverse_complement_byte(hiBytes[i]);
        const std::uint8_t hiback = make_reverse_complement_byte(hiBytes[usedBytesPerHalf - 1 - i]);
        hiBytes[i] = hiback;
        hiBytes[usedBytesPerHalf - 1 - i] = hifront;

        const std::uint8_t lofront = make_reverse_complement_byte(loBytes[i]);
        const std::uint8_t loback = make_reverse_complement_byte(loBytes[usedBytesPerHalf - 1 - i]);
        loBytes[i] = loback;
        loBytes[usedBytesPerHalf - 1 - i] = lofront;
    }
    if(usedBytesPerHalf % 2 == 1){
        const int middleindex = usedBytesPerHalf/2;
        hiBytes[middleindex] = make_reverse_complement_byte(hiBytes[middleindex]);
        loBytes[middleindex] = make_reverse_complement_byte(loBytes[middleindex]);
    }

    if(unusedBitsInLastUsedByte != 0){
        for(int i = 0; i < halfbytes - 1; ++i){
            hiBytes[i] = (hiBytes[i] << unusedBitsInLastUsedByte) | (hiBytes[i+1] >> (8 - unusedBitsInLastUsedByte));
            loBytes[i] = (loBytes[i] << unusedBitsInLastUsedByte) | (loBytes[i+1] >> (8 - unusedBitsInLastUsedByte));
        }

        hiBytes[halfbytes - 1] <<= unusedBitsInLastUsedByte;
        loBytes[halfbytes - 1] <<= unusedBitsInLastUsedByte;
    }
};


#endif

};




template<class Impl>
struct SequenceBase {

    using Impl_t = Impl;

	SequenceBase() : nBases(0){
        //data.second = 0;
    }

    ~SequenceBase(){}

    SequenceBase(const std::string& sequence)
		: nBases(sequence.length()){
        auto p = Impl::encode(sequence);
		data = std::move(p.first);
	}

    SequenceBase(const std::uint8_t* rawdata, int nBases_) noexcept
		: nBases(nBases_){

		const int size = getNumBytes();
		data = std::make_unique<std::uint8_t[]>(size);

		std::copy(rawdata, rawdata + size, begin());
	}

	SequenceBase(SequenceBase&& other) noexcept{
        operator=(other);
    }

	SequenceBase(const SequenceBase& other) : nBases(other.nBases){
		const int size = other.getNumBytes();
		data = std::make_unique<std::uint8_t[]>(size);

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
        if(!Impl::isCompressed())
            return Impl::get((const char*)data.get(), nBases, i);
        else{
            constexpr std::uint8_t BASE_A = 0x00;
            constexpr std::uint8_t BASE_C = 0x01;
            constexpr std::uint8_t BASE_G = 0x02;
            constexpr std::uint8_t BASE_T = 0x03;

            switch(Impl::get((const char*)data.get(), nBases, i)){
            case BASE_A: return 'A';
            case BASE_C: return 'C';
            case BASE_G: return 'G';
            case BASE_T: return 'T';
            default: return '_';
            }
        }
    }

	std::string toString() const{
        return Impl::toString(data.get(), nBases);
    }

	SequenceBase reverseComplement() const{
        SequenceBase revCompl;
        revCompl.nBases = nBases;
        auto p = Impl::reverseComplement(data.get(), nBases);
        revCompl.data = std::move(p.first);
        //std::cout << "orig:" << toString() << std::endl;
        //std::cout << "revc:" << revCompl.toString() << std::endl;
        return revCompl;
    }

	int getNumBytes() const noexcept{
        return Impl::getNumBytes(nBases);
    }

	int length() const noexcept{
        return nBases;
    }

    HOSTDEVICEQUALIFIER
	constexpr bool isCompressed() const noexcept{
        return Impl::isCompressed();
    }

    std::uint8_t* begin() const noexcept{
        return data.get();
    }

    std::uint8_t* end() const noexcept{
        return data.get() + getNumBytes();
    }

    HOSTDEVICEQUALIFIER
    static char get(const char* data, int nBases, int i) noexcept{
		return Impl::get(data, nBases, i);
	}

    HOSTDEVICEQUALIFIER
    static char get_as_nucleotide(const char* data, int nBases, int i) noexcept{
        auto to_nuc = [](char c){
            switch(c){
            case 0x00: return 'A';
            case 0x01: return 'C';
            case 0x02: return 'G';
            case 0x03: return 'T';
            default: return 'F';
            }
        };

        char c = Impl::get(data, nBases, i);
        if(Impl::isCompressed()){
            c = to_nuc(c);
        }
        return c;
	}

    HOSTDEVICEQUALIFIER
    static int getNumBytes(int nBases) noexcept{
        return Impl::getNumBytes(nBases);
    }

    HOSTDEVICEQUALIFIER
    static void make_reverse_complement(std::uint8_t* reverseComplement, const std::uint8_t* sequence, int sequencelength){
        return Impl::make_reverse_complement(reverseComplement, sequence, sequencelength);
    };

    HOSTDEVICEQUALIFIER
    static void make_reverse_complement_inplace(std::uint8_t* sequence, int sequencelength){
        return Impl::make_reverse_complement_inplace(sequence, sequencelength);
    };

	//std::pair<std::unique_ptr<std::uint8_t[]>, int> data;
    std::unique_ptr<std::uint8_t[]> data;
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

template<class T>
inline std::string getSequenceType();

template<>
inline std::string getSequenceType<Sequence2Bit>(){
    return "Sequence2Bit";
}

template<>
inline std::string getSequenceType<Sequence2BitHiLo>(){
    return "Sequence2BitHiLo";
}

template<>
inline std::string getSequenceType<SequenceString>(){
    return "SequenceString";
}

}
#endif
