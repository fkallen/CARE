#ifndef CARE_SEQUENCE_CONVERSION_HPP
#define CARE_SEQUENCE_CONVERSION_HPP

#include <config.hpp>


#include "hpc_helpers.cuh"


namespace care{

    constexpr std::uint32_t encodedbaseA = 0x00000000;
    constexpr std::uint32_t encodedbaseC = 0x00000001;
    constexpr std::uint32_t encodedbaseG = 0x00000002;
    constexpr std::uint32_t encodedbaseT = 0x00000003;

    constexpr int basesPerInt2BitNew = sizeof(unsigned int) * 8 / 2;


    //###########################

    HD_WARNING_DISABLE
    HOSTDEVICEQUALIFIER
    __inline__
    void reverseComplementString(char* reverseComplement, const char* sequence, int sequencelength){
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

    HD_WARNING_DISABLE
    HOSTDEVICEQUALIFIER
    __inline__
    void reverseComplementStringInplace(char* sequence, int sequencelength){

        auto make_reverse_complement_nuc = [](char in){
            switch(in){
                case 'A': return 'T';
                case 'C': return 'G';
                case 'G': return 'C';
                case 'T': return 'A';
                default :return 'F';
            }
        };

        for(int i = 0; i < sequencelength/2; i++){
            const std::uint8_t front = make_reverse_complement_nuc(sequence[i]);
            const std::uint8_t back = make_reverse_complement_nuc(sequence[sequencelength - 1 - i]);
            sequence[i] = back;
            sequence[sequencelength - 1 - i] = front;
        }

        if(sequencelength % 2 == 1){
            const int middleindex = sequencelength/2;
            sequence[middleindex] = make_reverse_complement_nuc(sequence[middleindex]);
        }
    }


    //###########################


    HOSTDEVICEQUALIFIER
    __inline__
    int getEncodedNumInts2BitNew(int sequenceLength){
        return SDIV(sequenceLength, basesPerInt2BitNew);
    }

    HD_WARNING_DISABLE
    template<class IndexTransformation>
    HOSTDEVICEQUALIFIER
    void encodeSequence2BitNew(unsigned int* out, const char* sequence, int sequenceLength, IndexTransformation indextrafo){

        const int nInts = getEncodedNumInts2BitNew(sequenceLength);

        for(int i = 0; i < nInts; i++){
            out[indextrafo(i)] = 0;
        }

        for(int nucIndex = 0; nucIndex < sequenceLength; nucIndex++){
            const int intIndex = nucIndex / basesPerInt2BitNew;
            const int pos = nucIndex % basesPerInt2BitNew;
            switch(sequence[nucIndex]) {
            case 'A':
                out[indextrafo(intIndex)] |= encodedbaseA << (2*pos);
                break;
            case 'C':
                out[indextrafo(intIndex)] |= encodedbaseC << (2*pos);
                break;
            case 'G':
                out[indextrafo(intIndex)] |= encodedbaseG << (2*pos);
                break;
            case 'T':
                out[indextrafo(intIndex)] |= encodedbaseT << (2*pos);
                break;
            default:
                out[indextrafo(intIndex)] |= encodedbaseA << (2*pos);
                break;
            }
        }
    }

    HD_WARNING_DISABLE
    template<class IndexTransformation>
    HOSTDEVICEQUALIFIER
    unsigned int getEncodedNuc2BitNew(const unsigned int* data, int sequenceLength, int i, IndexTransformation indextrafo){
        const int intIndex = i / basesPerInt2BitNew;
        const int pos = i % basesPerInt2BitNew;
        return ((data[indextrafo(intIndex)] >> (2*pos)) & 0x00000003);
    }

    HD_WARNING_DISABLE
    template<class IndexTransformation>
    HOSTDEVICEQUALIFIER
    void decode2BitNewSequence(char* sequence, const unsigned int* encoded, int sequenceLength, IndexTransformation indextrafo){
        for(int i = 0; i < sequenceLength; i++){
            const int intIndex = i / basesPerInt2BitNew;
            const int pos = i % basesPerInt2BitNew;
            const int base = getEncodedNuc2BitNew(encoded, sequenceLength, i, indextrafo);
            switch(base){
            case encodedbaseA: sequence[i] = 'A'; break;
            case encodedbaseC: sequence[i] = 'C'; break;
            case encodedbaseG: sequence[i] = 'G'; break;
            case encodedbaseT: sequence[i] = 'T'; break;
            default: sequence[i] = '_'; break; // cannot happen
            }
        }
    }

    HD_WARNING_DISABLE
    template<class IndexTransformation>
    HOSTDEVICEQUALIFIER
    void reverseComplementInplace2BitNew(unsigned int* encodedsequence, int sequenceLength, IndexTransformation indextrafo){

        auto make_reverse_complement_int = [](unsigned int s){
            s = ((s >> 2)  & 0x33333333u) | ((s & 0x33333333u) << 2);
            s = ((s >> 4)  & 0x0F0F0F0Fu) | ((s & 0x0F0F0F0Fu) << 4);
            s = ((s >> 8)  & 0x00FF00FFu) | ((s & 0x00FF00FFu) << 8);
            s = ((s >> 16) & 0x0000FFFFu) | ((s & 0x0000FFFFu) << 16);
            return ((unsigned int)(-1) - s) >> (8 * sizeof(s) - (16 << 1));
        };

        const int nInts = getEncodedNumInts2BitNew(sequenceLength);
        const int unusedPositions = nInts * basesPerInt2BitNew - sequenceLength;

        for(int i = 0; i < nInts/2; i++){
            const std::uint32_t front = make_reverse_complement_int(encodedsequence[indextrafo(i)]);
            const std::uint32_t back = make_reverse_complement_int(encodedsequence[indextrafo(nInts - 1 - i)]);
            encodedsequence[indextrafo(i)] = back;
            encodedsequence[indextrafo(nInts - 1 - i)] = front;
        }

        if(nInts % 2 == 1){
            const int middleindex = nInts/2;
            encodedsequence[indextrafo(middleindex)] = make_reverse_complement_int(encodedsequence[indextrafo(middleindex)]);
        }

        if(unusedPositions > 0){
            for(int i = 0; i < nInts-1; i++){
                encodedsequence[indextrafo(i)] = (encodedsequence[indextrafo(i)] >> (2*unusedPositions))
                                               | (encodedsequence[indextrafo(i+1)] << (2 * (basesPerInt2BitNew-unusedPositions)));

            }
    	encodedsequence[indextrafo(nInts-1)] >>= (2*unusedPositions);
        }
    }

    HD_WARNING_DISABLE
    template<class RcIndexTransformation, class IndexTransformation>
    HOSTDEVICEQUALIFIER
    void reverseComplement2BitNew(unsigned int* rcencodedsequence,
                                    const unsigned int* encodedsequence,
                                    int sequenceLength,
                                    RcIndexTransformation rcindextrafo,
                                    IndexTransformation indextrafo){

        const int nInts = getEncodedNumInts2BitNew(sequenceLength);
        for(int i = 0; i < nInts; i++){
            rcencodedsequence[rcindextrafo(i)] = encodedsequence[indextrafo(i)];
        }

        reverseComplementInplace2BitNew(rcencodedsequence, sequenceLength, rcindextrafo);
    }



    //###########################

    HOSTDEVICEQUALIFIER
    __inline__
    int getEncodedNumInts2BitHiLo(int sequenceLength){
        return int(2 * SDIV(sequenceLength, sizeof(unsigned int) * 8));
    }

    HD_WARNING_DISABLE
    template<class IndexTransformation>
    HOSTDEVICEQUALIFIER
    void encodeSequence2BitHiLo(unsigned int* out, const char* sequence, int sequenceLength, IndexTransformation indextrafo){
        const int nInts = getEncodedNumInts2BitHiLo(sequenceLength);

        for(int i = 0; i < nInts; i++){
            out[i] = 0;
        }

        unsigned int* const hi = out;
        unsigned int* const lo = out + indextrafo(nInts/2);

        for(int i = 0; i < sequenceLength; i++){
            const int intIndex = i / (8 * sizeof(unsigned int));
            const int pos = i % (8 * sizeof(unsigned int));
            const unsigned int mask = 1u << pos;

            switch(sequence[i]) {
            case 'A':
                hi[indextrafo(intIndex)] &= ~mask;
                lo[indextrafo(intIndex)] &= ~mask;
                break;
            case 'C':
                hi[indextrafo(intIndex)] &= ~mask;
                lo[indextrafo(intIndex)] |= mask;
                break;
            case 'G':
                hi[indextrafo(intIndex)] |= mask;
                lo[indextrafo(intIndex)] &= ~mask;
                break;
            case 'T':
                hi[indextrafo(intIndex)] |= mask;
                lo[indextrafo(intIndex)] |= mask;
                break;
            default:
                hi[indextrafo(intIndex)] &= ~mask;
                lo[indextrafo(intIndex)] &= ~mask;
                break;
            }
        }
    }

    HD_WARNING_DISABLE
    template<class IndexTransformation>
    HOSTDEVICEQUALIFIER
    unsigned int getEncodedNuc2BitHiLo(const unsigned int* data, int sequenceLength, int i, IndexTransformation indextrafo){
        const int nInts = getEncodedNumInts2BitHiLo(sequenceLength);

        const unsigned int* const hi = data;
        const unsigned int* const lo = data + indextrafo(nInts/2);

        const int intIndex = i / (8 * sizeof(unsigned int));
        const int pos = i % (8 * sizeof(unsigned int));
        const unsigned int hibit = (hi[indextrafo(intIndex)] >> pos) & 1u;
        const unsigned int lobit = (lo[indextrafo(intIndex)] >> pos) & 1u;
        const unsigned int base = (hibit << 1) | lobit;

        return base;
    }

    HD_WARNING_DISABLE
    template<class IndexTransformation>
    HOSTDEVICEQUALIFIER
    void decode2BitHiLoSequence(char* sequence, const unsigned int* encoded, int sequenceLength, IndexTransformation indextrafo){
        for(int i = 0; i < sequenceLength; i++){
            const unsigned int base = getEncodedNuc2BitHiLo(encoded, sequenceLength, i, indextrafo);

            switch(base){
            case encodedbaseA: sequence[i] = 'A'; break;
            case encodedbaseC: sequence[i] = 'C'; break;
            case encodedbaseG: sequence[i] = 'G'; break;
            case encodedbaseT: sequence[i] = 'T'; break;
            default: sequence[i] = '_'; break; // cannot happen
            }
        }
    }

    HD_WARNING_DISABLE
    template<class IndexTransformation>
    HOSTDEVICEQUALIFIER
    void reverseComplementInplace2BitHiLo(unsigned int* encodedsequence, int sequenceLength, IndexTransformation indextrafo){
        auto reverse_complement_int = [](auto n) {
            n = ((n >> 1) & 0x55555555) | ((n << 1) & 0xaaaaaaaa);
            n = ((n >> 2) & 0x33333333) | ((n << 2) & 0xcccccccc);
            n = ((n >> 4) & 0x0f0f0f0f) | ((n << 4) & 0xf0f0f0f0);
            n = ((n >> 8) & 0x00ff00ff) | ((n << 8) & 0xff00ff00);
            n = ((n >> 16) & 0x0000ffff) | ((n << 16) & 0xffff0000);
            return ~n;
        };

        const int ints = getEncodedNumInts2BitHiLo(sequenceLength);
        const int unusedBitsInt = SDIV(sequenceLength, 8 * sizeof(unsigned int)) * 8 * sizeof(unsigned int) - sequenceLength;

        unsigned int* const hi = encodedsequence;
        unsigned int* const lo = hi + indextrafo(ints/2);

        const int intsPerHalf = SDIV(sequenceLength, 8 * sizeof(unsigned int));
        for(int i = 0; i < intsPerHalf/2; ++i){
            const unsigned int hifront = reverse_complement_int(hi[indextrafo(i)]);
            const unsigned int hiback = reverse_complement_int(hi[indextrafo(intsPerHalf - 1 - i)]);
            hi[indextrafo(i)] = hiback;
            hi[indextrafo(intsPerHalf - 1 - i)] = hifront;

            const unsigned int lofront = reverse_complement_int(lo[indextrafo(i)]);
            const unsigned int loback = reverse_complement_int(lo[indextrafo(intsPerHalf - 1 - i)]);
            lo[indextrafo(i)] = loback;
            lo[indextrafo(intsPerHalf - 1 - i)] = lofront;
        }
        if(intsPerHalf % 2 == 1){
            const int middleindex = intsPerHalf/2;
            hi[indextrafo(middleindex)] = reverse_complement_int(hi[indextrafo(middleindex)]);
            lo[indextrafo(middleindex)] = reverse_complement_int(lo[indextrafo(middleindex)]);
        }

        if(unusedBitsInt != 0){
            for(int i = 0; i < intsPerHalf - 1; ++i){
                hi[indextrafo(i)] = (hi[indextrafo(i)] >> unusedBitsInt) | (hi[indextrafo(i+1)] << (8 * sizeof(unsigned int) - unusedBitsInt));
                lo[indextrafo(i)] = (lo[indextrafo(i)] >> unusedBitsInt) | (lo[indextrafo(i+1)] << (8 * sizeof(unsigned int) - unusedBitsInt));
            }

            hi[indextrafo(intsPerHalf - 1)] >>= unusedBitsInt;
            lo[indextrafo(intsPerHalf - 1)] >>= unusedBitsInt;
        }
    }

    HD_WARNING_DISABLE
    template<class RcIndexTransformation, class IndexTransformation>
    HOSTDEVICEQUALIFIER
    void reverseComplement2BitHiLo(unsigned int* rcencodedsequence,
                                    const unsigned int* encodedsequence,
                                    int sequenceLength,
                                    RcIndexTransformation rcindextrafo,
                                    IndexTransformation indextrafo){

        const int nInts = getEncodedNumInts2BitHiLo(sequenceLength);
        for(int i = 0; i < nInts; i++){
            rcencodedsequence[rcindextrafo(i)] = encodedsequence[indextrafo(i)];
        }

        reverseComplementInplace2BitHiLo(rcencodedsequence, sequenceLength, rcindextrafo);
    }




}
#endif
