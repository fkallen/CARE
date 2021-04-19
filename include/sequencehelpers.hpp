#ifndef CARE_SEQUENCEHELPERS_HPP
#define CARE_SEQUENCEHELPERS_HPP

#include <config.hpp>

#include "hpc_helpers.cuh"

#include <cstdint>
#include <string>
#include <type_traits>

namespace care{

    template<class T>
    struct EncodedReverseComplement2Bit{
    public:
        template<class I>
        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr T compute(I uinteger) noexcept{
            static_assert(std::is_same<T, I>::value, "types do no match");

            return compute_(uinteger);
        }
    private:
        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr unsigned int compute_(unsigned int n) noexcept{
            n = ((n >> 2)  & 0x33333333u) | ((n & 0x33333333u) << 2);
            n = ((n >> 4)  & 0x0F0F0F0Fu) | ((n & 0x0F0F0F0Fu) << 4);
            n = ((n >> 8)  & 0x00FF00FFu) | ((n & 0x00FF00FFu) << 8);
            n = ((n >> 16) & 0x0000FFFFu) | ((n & 0x0000FFFFu) << 16);
            return ((unsigned int)(-1) - n) >> (8 * sizeof(n) - (16 << 1));
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr std::uint64_t compute_(std::uint64_t n) noexcept{
            n = ((n >> 2)  & 0x3333333333333333ull) | ((n & 0x3333333333333333ull) << 2);
            n = ((n >> 4)  & 0x0F0F0F0F0F0F0F0Full) | ((n & 0x0F0F0F0F0F0F0F0Full) << 4);
            n = ((n >> 8)  & 0x00FF00FF00FF00FFull) | ((n & 0x00FF00FF00FF00FFull) << 8);
            n = ((n >> 16) & 0x0000FFFF0000FFFFull) | ((n & 0x0000FFFF0000FFFFull) << 16);
            n = ((n >> 32) & 0x00000000FFFFFFFFull) | ((n & 0x00000000FFFFFFFFull) << 32);
            return ((std::uint64_t)(-1) - n) >> (8 * sizeof(n) - (32 << 1));
        }        
    };

    struct SequenceHelpers{
    public:

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr char complementBaseDecoded(char in) noexcept{
            switch(in){
                case 'A': return 'T';
                case 'C': return 'G';
                case 'G': return 'C';
                case 'T': return 'A';
            }
            return in;
        };

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr void reverseComplementSequenceDecoded(char* reverseComplement, const char* sequence, int sequencelength) noexcept{
            for(int i = 0; i < sequencelength; ++i){
                switch(sequence[i]){
                    case 'A': reverseComplement[sequencelength-1-i] = 'T'; break;
                    case 'C': reverseComplement[sequencelength-1-i] = 'G'; break;
                    case 'G': reverseComplement[sequencelength-1-i] = 'C'; break;
                    case 'T': reverseComplement[sequencelength-1-i] = 'A'; break;
                    default : break; // don't change N
                }
            }
        }

        INLINEQUALIFIER
        static std::string reverseComplementSequenceDecoded(const char* sequence, int sequencelength){
            std::string rev;
            rev.resize(sequencelength);

            reverseComplementSequenceDecoded(&rev[0], sequence, sequencelength);

            return rev;
        }

        HD_WARNING_DISABLE
        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr void reverseComplementSequenceDecodedInplace(char* sequence, int sequencelength) noexcept{

            for(int i = 0; i < sequencelength/2; i++){
                const char front = complementBaseDecoded(sequence[i]);
                const char back = complementBaseDecoded(sequence[sequencelength - 1 - i]);
                sequence[i] = back;
                sequence[sequencelength - 1 - i] = front;
            }

            if(sequencelength % 2 == 1){
                const int middleindex = sequencelength/2;
                sequence[middleindex] = complementBaseDecoded(sequence[middleindex]);
            }
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr std::uint8_t encodedbaseA() noexcept{
            return 0;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr std::uint8_t encodedbaseC() noexcept{
            return 1;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr std::uint8_t encodedbaseG() noexcept{
            return 2;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr std::uint8_t encodedbaseT() noexcept{
            return 3;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr unsigned int basesPerInt2Bit() noexcept{
            return sizeof(unsigned int) * CHAR_BIT / 2;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr char decodeBase(std::uint8_t enc) noexcept{
            switch(enc){
            case encodedbaseA(): return 'A';
            case encodedbaseC(): return 'C';
            case encodedbaseG(): return 'G';
            case encodedbaseT(): return 'T';
            default: return '_';
            }
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr char encodeBase(char c) noexcept{
            switch(c){
            case 'A': return encodedbaseA();
            case 'C': return encodedbaseC();
            case 'G': return encodedbaseG();
            case 'T': return encodedbaseT();
            default: return encodedbaseA();
            }
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr int getEncodedNumInts2Bit(int sequenceLength) noexcept{
            return SDIV(sequenceLength, basesPerInt2Bit());
        }

        HD_WARNING_DISABLE
        template<class IndexTransformation>
        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr void encodeSequence2Bit(unsigned int* out, const char* sequence, int sequenceLength, IndexTransformation indextrafo) noexcept{

            const int nInts = getEncodedNumInts2Bit(sequenceLength);

            for(int i = 0; i < nInts; i++){
                out[indextrafo(i)] = 0;
            }

            for(int nucIndex = 0; nucIndex < sequenceLength; nucIndex++){
                const int intIndex = nucIndex / basesPerInt2Bit();
                switch(sequence[nucIndex]) {
                case 'A':
                    out[indextrafo(intIndex)] = (out[indextrafo(intIndex)] << 2) | encodedbaseA();
                    break;
                case 'C':
                    out[indextrafo(intIndex)] = (out[indextrafo(intIndex)] << 2) | encodedbaseC();
                    break;
                case 'G':
                    out[indextrafo(intIndex)] = (out[indextrafo(intIndex)] << 2) | encodedbaseG();
                    break;
                case 'T':
                    out[indextrafo(intIndex)] = (out[indextrafo(intIndex)] << 2) | encodedbaseT();
                    break;
                default:
                    out[indextrafo(intIndex)] = (out[indextrafo(intIndex)] << 2) | encodedbaseA();
                    break;
                }
            }
            //pack bits of last integer into higher order bits
            int leftoverbits = 2 * (nInts * basesPerInt2Bit() - sequenceLength);
            if(leftoverbits > 0){
                out[indextrafo(nInts-1)] <<= leftoverbits;
            }
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static void encodeSequence2Bit(unsigned int* outencoded, const char* sequence, int sequenceLength) noexcept{
            auto identity = [](auto i){return i;};
            encodeSequence2Bit(outencoded, sequence, sequenceLength, identity);
        }

        HD_WARNING_DISABLE
        template<class IndexTransformation>
        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr std::uint8_t getEncodedNuc2Bit(const unsigned int* data, int sequenceLength, int i, IndexTransformation indextrafo) noexcept{
            const int intIndex = i / basesPerInt2Bit();
            const int pos = i % basesPerInt2Bit();
            return ((data[indextrafo(intIndex)] >> (30 - 2*pos)) & 3u);
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static std::uint8_t getEncodedNuc2Bit(const unsigned int* encodedsequence,
                                    int length,
                                    int position) noexcept{
            auto identity = [](auto i){return i;};
            return getEncodedNuc2Bit(encodedsequence, length, position, identity);
        }

        HD_WARNING_DISABLE
        template<class IndexTransformation>
        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr void decode2BitSequence(char* sequence, const unsigned int* encoded, int sequenceLength, IndexTransformation indextrafo) noexcept{
            for(int i = 0; i < sequenceLength; i++){
                const std::uint8_t base = getEncodedNuc2Bit(encoded, sequenceLength, i, indextrafo);
                sequence[i] = decodeBase(base);
            }
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static void decode2BitSequence(char* sequence,
                                    const unsigned int* encodedsequence,
                                    int length) noexcept{
            auto identity = [](auto i){return i;};
            decode2BitSequence(sequence, encodedsequence, length, identity);
        }

        template<class IndexTransformation>
        static std::string get2BitString(const unsigned int* encoded, int sequenceLength, IndexTransformation indextrafo){
            std::string s;
            s.resize(sequenceLength);
            decode2BitSequence(&s[0], encoded, sequenceLength, indextrafo);
            return s;
        }

        static std::string get2BitString(const unsigned int* encodedsequence,
                                    int length){
            auto identity = [](auto i){return i;};
            return get2BitString(encodedsequence, length, identity);
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr std::uint8_t getEncodedNucFromInt2Bit(unsigned int data, int pos){
            return ((data >> (30 - 2*pos)) & 3u);
        };

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr unsigned int reverseComplementInt2Bit(unsigned int n) noexcept{
            return EncodedReverseComplement2Bit<unsigned int>::compute(n);
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr std::uint64_t reverseComplementInt2Bit(std::uint64_t n) noexcept{
            return EncodedReverseComplement2Bit<std::uint64_t>::compute(n);
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr std::uint8_t complementBase2Bit(std::uint8_t n) noexcept{
            return (~n & std::uint8_t{3});
        }

        HD_WARNING_DISABLE
        template<class IndexTransformation>
        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr void reverseComplementSequenceInplace2Bit(unsigned int* encodedsequence, int sequenceLength, IndexTransformation indextrafo) noexcept{

            const int nInts = getEncodedNumInts2Bit(sequenceLength);
            const int unusedPositions = nInts * basesPerInt2Bit() - sequenceLength;

            for(int i = 0; i < nInts/2; i++){
                const unsigned int front = reverseComplementInt2Bit(encodedsequence[indextrafo(i)]);
                const unsigned int back = reverseComplementInt2Bit(encodedsequence[indextrafo(nInts - 1 - i)]);
                encodedsequence[indextrafo(i)] = back;
                encodedsequence[indextrafo(nInts - 1 - i)] = front;
            }

            if(nInts % 2 == 1){
                const int middleindex = nInts/2;
                encodedsequence[indextrafo(middleindex)] = reverseComplementInt2Bit(encodedsequence[indextrafo(middleindex)]);
            }

            if(unusedPositions > 0){
                for(int i = 0; i < nInts-1; i++){
                    encodedsequence[indextrafo(i)] = (encodedsequence[indextrafo(i)] << (2*unusedPositions))
                                                | (encodedsequence[indextrafo(i+1)] >> (2 * (basesPerInt2Bit()-unusedPositions)));

                }
                encodedsequence[indextrafo(nInts-1)] <<= (2*unusedPositions);
            }
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static void reverseComplementSequenceInplace2Bit(unsigned int* encodedsequence,
                                    int length) noexcept{
            auto identity = [](auto i){return i;};
            reverseComplementSequenceInplace2Bit(encodedsequence, length, identity);
        }

        HD_WARNING_DISABLE
        template<class RcIndexTransformation, class IndexTransformation>
        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr void reverseComplementSequence2Bit(unsigned int* rcencodedsequence,
                                        const unsigned int* encodedsequence,
                                        int sequenceLength,
                                        RcIndexTransformation rcindextrafo,
                                        IndexTransformation indextrafo) noexcept{

            const int nInts = getEncodedNumInts2Bit(sequenceLength);
            for(int i = 0; i < nInts; i++){
                rcencodedsequence[rcindextrafo(i)] = encodedsequence[indextrafo(i)];
            }

            reverseComplementSequenceInplace2Bit(rcencodedsequence, sequenceLength, rcindextrafo);
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static void reverseComplementSequence2Bit(unsigned int* rcencodedsequence,
                                    const unsigned int* encodedsequence,
                                    int length) noexcept{
            auto identity = [](auto i){return i;};
            reverseComplementSequence2Bit(rcencodedsequence, encodedsequence, length, identity, identity);
        }





        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr int getEncodedNumInts2BitHiLo(int sequenceLength) noexcept{
            return int(2 * SDIV(sequenceLength, sizeof(unsigned int) * CHAR_BIT));
        }

        HD_WARNING_DISABLE
        template<class IndexTransformation>
        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr void encodeSequence2BitHiLo(unsigned int* out, const char* sequence, int sequenceLength, IndexTransformation indextrafo) noexcept{
            const int nInts = getEncodedNumInts2BitHiLo(sequenceLength);

            for(int i = 0; i < nInts; i++){
                out[indextrafo(i)] = 0;
            }

            unsigned int* const hi = out;
            unsigned int* const lo = out + indextrafo(nInts/2);

            for(int i = 0; i < sequenceLength; i++){
                const int intIndex = i / (CHAR_BIT * sizeof(unsigned int));

                switch(sequence[i]) {
                case 'A':
                    hi[indextrafo(intIndex)] = (hi[indextrafo(intIndex)] << 1) | 0;
                    lo[indextrafo(intIndex)] = (lo[indextrafo(intIndex)] << 1) | 0;
                    break;
                case 'C':
                    hi[indextrafo(intIndex)] = (hi[indextrafo(intIndex)] << 1) | 0;
                    lo[indextrafo(intIndex)] = (lo[indextrafo(intIndex)] << 1) | 1;
                    break;
                case 'G':
                    hi[indextrafo(intIndex)] = (hi[indextrafo(intIndex)] << 1) | 1;
                    lo[indextrafo(intIndex)] = (lo[indextrafo(intIndex)] << 1) | 0;
                    break;
                case 'T':
                    hi[indextrafo(intIndex)] = (hi[indextrafo(intIndex)] << 1) | 1;
                    lo[indextrafo(intIndex)] = (lo[indextrafo(intIndex)] << 1) | 1;
                    break;
                default:
                    hi[indextrafo(intIndex)] = (hi[indextrafo(intIndex)] << 1) | 0;
                    lo[indextrafo(intIndex)] = (lo[indextrafo(intIndex)] << 1) | 0;
                    break;
                }
            }
            //pack bits of last hi integer and lo integer into their higher order bits
            const int leftoverbits = nInts/2 * CHAR_BIT * sizeof(unsigned int) - sequenceLength;
            if(leftoverbits > 0){
                hi[indextrafo(nInts/2-1)] <<= leftoverbits;
                lo[indextrafo(nInts/2-1)] <<= leftoverbits;
            }
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static void encodeSequence2BitHiLo(unsigned int* outencoded, const char* sequence, int sequenceLength) noexcept{
            auto identity = [](auto i){return i;};
            encodeSequence2BitHiLo(outencoded, sequence, sequenceLength, identity);
        }

        HD_WARNING_DISABLE
        template<class IndexTransformation>
        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr std::uint8_t getEncodedNuc2BitHiLo(const unsigned int* data, int sequenceLength, int i, IndexTransformation indextrafo) noexcept{
            const int nInts = getEncodedNumInts2BitHiLo(sequenceLength);

            const unsigned int* const hi = data;
            const unsigned int* const lo = data + indextrafo(nInts/2);

            const int intIndex = i / (CHAR_BIT * sizeof(unsigned int));
            const int pos = i % (CHAR_BIT * sizeof(unsigned int));
            const unsigned int hibit = (hi[indextrafo(intIndex)] >> (31 - pos)) & 1u;
            const unsigned int lobit = (lo[indextrafo(intIndex)] >> (31 - pos)) & 1u;
            return (hibit << 1) | lobit;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static std::uint8_t getEncodedNuc2BitHiLo(const unsigned int* encodedsequence,
                                    int length,
                                    int position) noexcept{
            auto identity = [](auto i){return i;};
            return getEncodedNuc2BitHiLo(encodedsequence, length, position, identity);
        }


        HD_WARNING_DISABLE
        template<class IndexTransformation>
        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr void decode2BitHiLoSequence(char* sequence, const unsigned int* encoded, int sequenceLength, IndexTransformation indextrafo) noexcept{
            for(int i = 0; i < sequenceLength; i++){
                const std::uint8_t base = getEncodedNuc2BitHiLo(encoded, sequenceLength, i, indextrafo);

                switch(base){
                case encodedbaseA(): sequence[i] = 'A'; break;
                case encodedbaseC(): sequence[i] = 'C'; break;
                case encodedbaseG(): sequence[i] = 'G'; break;
                case encodedbaseT(): sequence[i] = 'T'; break;
                //default: sequence[i] = '_'; break; // cannot happen
                }
            }
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static void decode2BitHiLoSequence(char* sequence,
                                    const unsigned int* encodedsequence,
                                    int length) noexcept{
            auto identity = [](auto i){return i;};
            decode2BitHiLoSequence(sequence, encodedsequence, length, identity);
        }

        template<class IndexTransformation>
        INLINEQUALIFIER
        static std::string get2BitHiLoString(const unsigned int* encoded, int sequenceLength, IndexTransformation indextrafo){
            std::string s;
            s.resize(sequenceLength);
            decode2BitHiLoSequence(&s[0], encoded, sequenceLength, indextrafo);
            return s;
        }

        INLINEQUALIFIER
        static std::string get2BitHiLoString(const unsigned int* encodedsequence,
                                    int length){
            auto identity = [](auto i){return i;};
            return get2BitHiLoString(encodedsequence, length, identity);
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr unsigned int reverseComplementInt2BitHiLoHalf(unsigned int n) noexcept{
            n = ((n >> 1) & 0x55555555) | ((n << 1) & 0xaaaaaaaa);
            n = ((n >> 2) & 0x33333333) | ((n << 2) & 0xcccccccc);
            n = ((n >> 4) & 0x0f0f0f0f) | ((n << 4) & 0xf0f0f0f0);
            n = ((n >> 8) & 0x00ff00ff) | ((n << 8) & 0xff00ff00);
            n = ((n >> 16) & 0x0000ffff) | ((n << 16) & 0xffff0000);
            return ~n;
        };


        HD_WARNING_DISABLE
        template<class IndexTransformation>
        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr void reverseComplementSequenceInplace2BitHiLo(unsigned int* encodedsequence, int sequenceLength, IndexTransformation indextrafo) noexcept{
            const int ints = getEncodedNumInts2BitHiLo(sequenceLength);
            const int unusedBitsInt = SDIV(sequenceLength, CHAR_BIT * sizeof(unsigned int)) * CHAR_BIT * sizeof(unsigned int) - sequenceLength;

            unsigned int* const hi = encodedsequence;
            unsigned int* const lo = hi + indextrafo(ints/2);

            const int intsPerHalf = SDIV(sequenceLength, CHAR_BIT * sizeof(unsigned int));
            for(int i = 0; i < intsPerHalf/2; ++i){
                const unsigned int hifront = reverseComplementInt2BitHiLoHalf(hi[indextrafo(i)]);
                const unsigned int hiback = reverseComplementInt2BitHiLoHalf(hi[indextrafo(intsPerHalf - 1 - i)]);
                hi[indextrafo(i)] = hiback;
                hi[indextrafo(intsPerHalf - 1 - i)] = hifront;

                const unsigned int lofront = reverseComplementInt2BitHiLoHalf(lo[indextrafo(i)]);
                const unsigned int loback = reverseComplementInt2BitHiLoHalf(lo[indextrafo(intsPerHalf - 1 - i)]);
                lo[indextrafo(i)] = loback;
                lo[indextrafo(intsPerHalf - 1 - i)] = lofront;
            }
            if(intsPerHalf % 2 == 1){
                const int middleindex = intsPerHalf/2;
                hi[indextrafo(middleindex)] = reverseComplementInt2BitHiLoHalf(hi[indextrafo(middleindex)]);
                lo[indextrafo(middleindex)] = reverseComplementInt2BitHiLoHalf(lo[indextrafo(middleindex)]);
            }

            if(unusedBitsInt != 0){
                for(int i = 0; i < intsPerHalf - 1; ++i){
                    hi[indextrafo(i)] = (hi[indextrafo(i)] << unusedBitsInt) | (hi[indextrafo(i+1)] >> (CHAR_BIT * sizeof(unsigned int) - unusedBitsInt));
                    lo[indextrafo(i)] = (lo[indextrafo(i)] << unusedBitsInt) | (lo[indextrafo(i+1)] >> (CHAR_BIT * sizeof(unsigned int) - unusedBitsInt));
                }

                hi[indextrafo(intsPerHalf - 1)] <<= unusedBitsInt;
                lo[indextrafo(intsPerHalf - 1)] <<= unusedBitsInt;
            }
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static void reverseComplementSequenceInplace2BitHiLo(unsigned int* encodedsequence,
                                    int length) noexcept{
            auto identity = [](auto i){return i;};
            reverseComplementSequenceInplace2BitHiLo(encodedsequence, length, identity);
        }

        HD_WARNING_DISABLE
        template<class RcIndexTransformation, class IndexTransformation>
        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr void reverseComplementSequence2BitHiLo(unsigned int* rcencodedsequence,
                                        const unsigned int* encodedsequence,
                                        int sequenceLength,
                                        RcIndexTransformation rcindextrafo,
                                        IndexTransformation indextrafo) noexcept{

            const int nInts = getEncodedNumInts2BitHiLo(sequenceLength);
            for(int i = 0; i < nInts; i++){
                rcencodedsequence[rcindextrafo(i)] = encodedsequence[indextrafo(i)];
            }

            reverseComplementSequenceInplace2BitHiLo(rcencodedsequence, sequenceLength, rcindextrafo);
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static void reverseComplementSequence2BitHiLo(unsigned int* rcencodedsequence,
                                    const unsigned int* encodedsequence,
                                    int length) noexcept{
            auto identity = [](auto i){return i;};
            reverseComplementSequence2BitHiLo(rcencodedsequence, encodedsequence, length, identity, identity);
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr unsigned int extractEvenBits(unsigned int x) noexcept{
            x = x & 0x55555555;
            x = (x | (x >> 1)) & 0x33333333;
            x = (x | (x >> 2)) & 0x0F0F0F0F;
            x = (x | (x >> 4)) & 0x00FF00FF;
            x = (x | (x >> 8)) & 0x0000FFFF;
            return x;
        }
        
        HD_WARNING_DISABLE
        template<class InIndexTransformation, class OutIndexTransformation>
        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr void convert2BitTo2BitHiLo(unsigned int* out,
                                    const unsigned int* in,
                                    int length,
                                    InIndexTransformation inindextrafo,
                                    OutIndexTransformation outindextrafo) noexcept{

            const int inInts = getEncodedNumInts2Bit(length);
            const int outInts = getEncodedNumInts2BitHiLo(length);

            unsigned int* const outHi = out;
            unsigned int* const outLo = out + outindextrafo(outInts/2);

            for(int i = 0; i < outInts; i++){
                out[outindextrafo(i)] = 0;
            }

            for(int i = 0; i < inInts; i++){
                const int inindex = inindextrafo(i);
                const int outindex = outindextrafo(i/2);

                unsigned int even16 = extractEvenBits(in[inindex]);
                unsigned int odd16 = extractEvenBits(in[inindex] >> 1);

                if(i % 2 == 0){
                    outHi[outindex] = odd16 << 16;
                    outLo[outindex] = even16 << 16;
                }else{
                    outHi[outindex] = outHi[outindex] | odd16;
                    outLo[outindex] = outLo[outindex] | even16;
                }
            }
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static void convert2BitTo2BitHiLo(unsigned int* out,
                                    const unsigned int* in,
                                    int length) noexcept{
            auto identity = [](auto i){return i;};
            convert2BitTo2BitHiLo(out, in, length, identity, identity);
        }

        
        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr std::uint8_t convertDNACharToIntNoIf(unsigned char input){
            // 'A' -> 0
            // 'C' -> 1
            // 'G' -> 2
            // 'T' -> 3
            // 'a' -> 0
            // 'c' -> 1
            // 'g' -> 2
            // 't' -> 3

            constexpr float a = 167.f/100776.f;
            constexpr float b = -1845.f/33592.f;
            //constexpr float c = 30395.f/50388.f;
            constexpr float c2 = 30395.f/50387.f;
            constexpr float d = 0.f;

            const float x = (input & 0xDf)-(unsigned char)(65);

            //return a*x*x*x + b*x*x + c*x + d;
            return std::uint8_t(((((a * x) + b) * x) + c2) * x + d);
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        static constexpr char convertIntToDNACharNoIf(std::uint8_t input){
            // 0 -> 'A'
            // 1 -> 'C'
            // 2 -> 'G'
            // 3 -> 'T'

            constexpr float a = 7.f / 6.f;
            constexpr float b = -5.f / 2.f;
            constexpr float c = 10.f / 3.f;
            constexpr float d = 65.f;

            const float x = input;

            return char(((((a * x) + b) * x) + c) * x + d);
        }
    };

    static_assert(SequenceHelpers::convertDNACharToIntNoIf('A') == 0,"");
    static_assert(SequenceHelpers::convertDNACharToIntNoIf('C') == 1,"");
    static_assert(SequenceHelpers::convertDNACharToIntNoIf('G') == 2,"");
    static_assert(SequenceHelpers::convertDNACharToIntNoIf('T') == 3,"");
    static_assert(SequenceHelpers::convertDNACharToIntNoIf('a') == 0,"");
    static_assert(SequenceHelpers::convertDNACharToIntNoIf('c') == 1,"");
    static_assert(SequenceHelpers::convertDNACharToIntNoIf('g') == 2,"");
    static_assert(SequenceHelpers::convertDNACharToIntNoIf('t') == 3,"");

    static_assert(SequenceHelpers::convertIntToDNACharNoIf(0) == 'A',"");
    static_assert(SequenceHelpers::convertIntToDNACharNoIf(1) == 'C',"");
    static_assert(SequenceHelpers::convertIntToDNACharNoIf(2) == 'G',"");
    static_assert(SequenceHelpers::convertIntToDNACharNoIf(3) == 'T',"");


}
#endif
