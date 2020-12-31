#ifndef CARE_MINHASHING_HPP
#define CARE_MINHASHING_HPP


#include <config.hpp>
#include <hpc_helpers.cuh>
#include <sequencehelpers.hpp>

#include <array>


namespace care{

/*
    Minhash signature of a single sequence which is not encoded
*/
inline
std::array<std::uint64_t, maximum_number_of_maps> 
calculateMinhashSignature(
    const char* sequence, 
    int sequenceLength, 
    int kmerLength, 
    int numHashFuncs,
    int firstHashFunc
) noexcept{

    const int length = sequenceLength;

    std::array<std::uint64_t, maximum_number_of_maps> minhashSignature;
    std::fill_n(minhashSignature.begin(), numHashFuncs, std::numeric_limits<std::uint64_t>::max());

    if(length < kmerLength) return minhashSignature;

    constexpr int maximum_kmer_length = max_k<kmer_type>::value;
    const kmer_type kmer_mask = std::numeric_limits<kmer_type>::max() >> ((maximum_kmer_length - kmerLength) * 2);
    const int rcshiftamount = (maximum_kmer_length - kmerLength) * 2;

    auto handlekmer = [&](auto fwd, auto rc, int numhashfunc){
        using hasher = hashers::MurmurHash<std::uint64_t>;

        const auto smallest = std::min(fwd, rc);
        const auto hashvalue = hasher::hash(smallest + firstHashFunc + numhashfunc);
        minhashSignature[numhashfunc] = std::min(minhashSignature[numhashfunc], hashvalue);
    };

    kmer_type kmer_encoded = 0;
    kmer_type rc_kmer_encoded = std::numeric_limits<kmer_type>::max();

    auto addBase = [&](char c){
        kmer_encoded <<= 2;
        rc_kmer_encoded >>= 2;
        switch(c) {
        case 'A':
            kmer_encoded |= 0;
            rc_kmer_encoded |= kmer_type(3) << (sizeof(kmer_type) * 8 - 2);
            break;
        case 'C':
            kmer_encoded |= 1;
            rc_kmer_encoded |= kmer_type(2) << (sizeof(kmer_type) * 8 - 2);
            break;
        case 'G':
            kmer_encoded |= 2;
            rc_kmer_encoded |= kmer_type(1) << (sizeof(kmer_type) * 8 - 2);
            break;
        case 'T':
            kmer_encoded |= 3;
            rc_kmer_encoded |= kmer_type(0) << (sizeof(kmer_type) * 8 - 2);
            break;
        default:break;
        }
    };

    for(int i = 0; i < kmerLength - 1; i++){
        addBase(sequence[i]);
    }

    for(int i = kmerLength - 1; i < length; i++){
        addBase(sequence[i]);

        for(int m = 0; m < numHashFuncs; m++){
            handlekmer(kmer_encoded & kmer_mask, 
                        rc_kmer_encoded >> rcshiftamount, 
                        m);
        }
    }

    return minhashSignature;
}



/*
    Minhash signature of a single sequence which is 2bit encoded
*/
inline
std::array<std::uint64_t, maximum_number_of_maps> 
calculateMinhashSignature(
    const unsigned int* sequence, 
    int sequenceLength, 
    int kmerLength, 
    int numHashFuncs,
    int firstHashFunc
) noexcept{

    const int length = sequenceLength;

    std::array<std::uint64_t, maximum_number_of_maps> minhashSignature;
    std::fill_n(minhashSignature.begin(), numHashFuncs, std::numeric_limits<std::uint64_t>::max());

    if(length < kmerLength) return minhashSignature;

    constexpr int maximum_kmer_length = max_k<kmer_type>::value;
    const kmer_type kmer_mask = std::numeric_limits<kmer_type>::max() >> ((maximum_kmer_length - kmerLength) * 2);
    const int rcshiftamount = (maximum_kmer_length - kmerLength) * 2;

    auto handlekmer = [&](auto fwd, auto rc, int numhashfunc){
        using hasher = hashers::MurmurHash<std::uint64_t>;

        const auto smallest = std::min(fwd, rc);
        const auto hashvalue = hasher::hash(smallest + firstHashFunc + numhashfunc);
        minhashSignature[numhashfunc] = std::min(minhashSignature[numhashfunc], hashvalue);
    };

    //Compute the first kmer
    std::uint64_t kmer_encoded = sequence[0];
    if(kmerLength <= 16){
        kmer_encoded >>= (16 - kmerLength) * 2;
    }else{
        kmer_encoded = (kmer_encoded << 32) | sequence[1];
        kmer_encoded >>= (32 - kmerLength) * 2;
    }

    kmer_encoded >>= 2; //k-1 bases, allows easier loop

    std::uint64_t rc_kmer_encoded = SequenceHelpers::reverseComplementInt2Bit(kmer_encoded);

    auto addBase = [&](std::uint64_t encBase){
        kmer_encoded <<= 2;
        rc_kmer_encoded >>= 2;

        const std::uint64_t revcBase = (~encBase) & 3;
        kmer_encoded |= encBase;
        rc_kmer_encoded |= revcBase << (64 - 2);
    };

    constexpr int basesPerInt = SequenceHelpers::basesPerInt2Bit();

    for(int nextSequencePos = kmerLength - 1; nextSequencePos < sequenceLength; nextSequencePos++){
        const int nextIntIndex = nextSequencePos / basesPerInt;
        const int nextPositionInInt = nextSequencePos % basesPerInt;

        const std::uint64_t nextBase = sequence[nextIntIndex] >> (30 - 2 * nextPositionInInt);

        addBase(nextBase);

        for(int m = 0; m < numHashFuncs; m++){
            handlekmer(
                kmer_encoded & kmer_mask, 
                rc_kmer_encoded >> rcshiftamount, 
                m
            );
        }
    }

    return minhashSignature;
}



}


#endif