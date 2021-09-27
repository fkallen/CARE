#ifndef CARE_MINHASHING_HPP
#define CARE_MINHASHING_HPP


#include <config.hpp>
#include <hpc_helpers.cuh>
#include <sequencehelpers.hpp>

#include <array>
#include <algorithm>
#include <cstdint>

namespace care{

/*
    Minhash signature of a single sequence which is not encoded
*/
inline
std::array<kmer_type, maximum_number_of_maps> 
calculateMinhashSignature(
    const char* sequence, 
    int sequenceLength, 
    int kmerLength, 
    int numHashFuncs,
    int firstHashFunc
) noexcept{

    const int length = sequenceLength;

    if(length < kmerLength){
        std::array<kmer_type, maximum_number_of_maps> minhashSignature;
        std::fill_n(minhashSignature.begin(), numHashFuncs, std::numeric_limits<kmer_type>::max());
        return minhashSignature;
    }

    std::array<std::uint64_t, maximum_number_of_maps> minhashSignature64;
    std::fill_n(minhashSignature64.begin(), numHashFuncs, std::numeric_limits<std::uint64_t>::max());

    constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;
    const std::uint64_t kmer_mask = std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - kmerLength) * 2);
    const int rcshiftamount = (maximum_kmer_length - kmerLength) * 2;

    auto handlekmer = [&](std::uint64_t fwd, std::uint64_t rc, int numhashfunc){
        using hasher = hashers::MurmurHash<std::uint64_t>;

        const std::uint64_t smallest = std::min(fwd, rc);
        const std::uint64_t hashvalue = hasher::hash(smallest + firstHashFunc + numhashfunc);
        const std::uint64_t current = minhashSignature64[numhashfunc];
        minhashSignature64[numhashfunc] = std::min(current, hashvalue);
    };

    std::uint64_t kmer_encoded = 0;
    std::uint64_t rc_kmer_encoded = std::numeric_limits<std::uint64_t>::max();

    auto addBase = [&](char c){
        kmer_encoded <<= 2;
        rc_kmer_encoded >>= 2;
        switch(c) {
        case 'A':
            kmer_encoded |= 0;
            rc_kmer_encoded |= std::uint64_t(3) << (sizeof(std::uint64_t) * 8 - 2);
            break;
        case 'C':
            kmer_encoded |= 1;
            rc_kmer_encoded |= std::uint64_t(2) << (sizeof(std::uint64_t) * 8 - 2);
            break;
        case 'G':
            kmer_encoded |= 2;
            rc_kmer_encoded |= std::uint64_t(1) << (sizeof(std::uint64_t) * 8 - 2);
            break;
        case 'T':
            kmer_encoded |= 3;
            rc_kmer_encoded |= std::uint64_t(0) << (sizeof(std::uint64_t) * 8 - 2);
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

    std::array<kmer_type, maximum_number_of_maps> minhashSignature;
    std::transform(
        minhashSignature64.begin(), 
        minhashSignature64.end(),
        minhashSignature.begin(),
        [kmer_mask](std::uint64_t hash){
            return kmer_type(hash & kmer_mask);
        }
    );
    return minhashSignature;
}



/*
    Minhash signature of a single sequence which is 2bit encoded
*/
inline
std::array<kmer_type, maximum_number_of_maps> 
calculateMinhashSignature(
    const unsigned int* sequence, 
    int sequenceLength, 
    int kmerLength, 
    int numHashFuncs,
    int firstHashFunc
) noexcept{

    const int length = sequenceLength;

    if(length < kmerLength){
        std::array<kmer_type, maximum_number_of_maps> minhashSignature;
        std::fill_n(minhashSignature.begin(), numHashFuncs, std::numeric_limits<kmer_type>::max());
        return minhashSignature;
    }

    std::array<std::uint64_t, maximum_number_of_maps> minhashSignature64;
    std::fill_n(minhashSignature64.begin(), numHashFuncs, std::numeric_limits<std::uint64_t>::max());

    constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;
    const std::uint64_t kmer_mask = std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - kmerLength) * 2);
    const int rcshiftamount = (maximum_kmer_length - kmerLength) * 2;

    auto handlekmer = [&](std::uint64_t fwd, std::uint64_t rc, int numhashfunc){
        using hasher = hashers::MurmurHash<std::uint64_t>;

        const std::uint64_t smallest = std::min(fwd, rc);
        const std::uint64_t hashvalue = hasher::hash(smallest + firstHashFunc + numhashfunc);
        const std::uint64_t current = minhashSignature64[numhashfunc];
        minhashSignature64[numhashfunc] = std::min(current, hashvalue);
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

    std::array<kmer_type, maximum_number_of_maps> minhashSignature;
    std::transform(
        minhashSignature64.begin(), 
        minhashSignature64.end(),
        minhashSignature.begin(),
        [kmer_mask](std::uint64_t hash){
            return kmer_type(hash & kmer_mask);
        }
    );
    return minhashSignature;
}



}


#endif