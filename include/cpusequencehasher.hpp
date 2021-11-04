#ifndef CARE_CPUSEQUENCEHASHER_HPP
#define CARE_CPUSEQUENCEHASHER_HPP


#include <config.hpp>
#include <hpc_helpers.cuh>
#include <sequencehelpers.hpp>

#include <array>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <cassert>

namespace care{

template<class HashValueType>
struct CPUSequenceHasher{

    struct TopSmallestHashResult{
        friend class CPUSequenceHasher;

        TopSmallestHashResult(std::size_t numSmallest_) : numSmallest(numSmallest_){ assert(numSmallest <= maximum_number_of_maps); }

        std::size_t numSmallest = 0;
        std::size_t numHashes = 0;
        std::array<HashValueType, maximum_number_of_maps> hashes{}; //sorted

        std::size_t size() const noexcept{ return std::min(std::min(numHashes, numSmallest), hashes.size()); }
        const HashValueType* cbegin() const noexcept{ return hashes.data(); }
        const HashValueType* cend() const noexcept{ return hashes.data() + size(); }
        const HashValueType* data() const noexcept{ return hashes.data(); }
        HashValueType* begin() noexcept{ return hashes.data(); }
        HashValueType* end() noexcept{ return hashes.data() + size(); }
        HashValueType* data() noexcept{ return hashes.data(); }
        HashValueType& operator[](std::size_t i) noexcept{ assert(i < size()); return hashes[i]; }
        const HashValueType& operator[](std::size_t i) const noexcept{ assert(i < size()); return hashes[i]; }

    private:

        void insert(HashValueType element){
            //find position of new element
            auto it = std::lower_bound(hashes.begin(), hashes.begin() + numHashes, element);

            //if element is already present in hashes, it is not inserted
            if((it != hashes.begin() + numHashes) && (*it == element)){
                return;
            }else{
                const std::size_t newsize = std::min(numSmallest, std::min(numHashes + 1, hashes.size()));

                if(size() < newsize){
                    //insert new element
                    std::copy_backward(it, hashes.begin() + numHashes, hashes.begin() + newsize);
                    *it = element;
                }else{
                    if(it != hashes.begin() + numHashes){
                        //insert new element if it is not the largest
                        std::copy_backward(it, hashes.begin() + numHashes - 1, hashes.begin() + newsize);
                        *it = element;
                    }else{
                        //element will not be inserted, too large
                    }
                }

                numHashes = newsize;
            }

            //std::cerr << "insert(" << element << "), numHashes = " << numHashes << "\n";
        }
    };

    TopSmallestHashResult getTopSmallestKmerHashes(
        const unsigned int* sequence2Bit,
        const int sequenceLength,
        int kmerLength,
        int numSmallest,
        bool debug = false
    ){
        assert(sizeof(kmer_type) * 8 / 2 >= std::size_t(kmerLength));

        using hasher = hashers::MurmurHash<std::uint64_t>;

        TopSmallestHashResult result(numSmallest);

        if(sequenceLength < kmerLength){
            return result;
        }

        constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;
        const std::uint64_t kmer_mask = std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - kmerLength) * 2);
        const int rcshiftamount = (maximum_kmer_length - kmerLength) * 2;

        std::uint64_t kmer_encoded = sequence2Bit[0];
        if(kmerLength <= 16){
            kmer_encoded >>= (16 - kmerLength) * 2;
        }else{
            kmer_encoded = (kmer_encoded << 32) | sequence2Bit[1];
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

            const std::uint64_t nextBase = sequence2Bit[nextIntIndex] >> (30 - 2 * nextPositionInInt);

            addBase(nextBase);

            //std::cerr << nextSequencePos << "\n";

            //std::cerr << (kmer_encoded & kmer_mask) << ", ";

            const std::uint64_t smallest = std::min(kmer_encoded & kmer_mask, rc_kmer_encoded >> rcshiftamount);
            const std::uint64_t hashvalue = hasher::hash(smallest) & kmer_mask;

            if(debug){
                std::cerr << (kmer_encoded & kmer_mask) << " , " <<  (rc_kmer_encoded >> rcshiftamount) << " , " << smallest << " : " << (hasher::hash(smallest) & kmer_mask) << "\n";
            }
            result.insert(hashvalue);
        }
        //std::cerr << "\n";

        return result;
    }


    template<class OutputIter>
    OutputIter hashInto(
        OutputIter output,
        const unsigned int* sequence, 
        int sequenceLength, 
        int kmerLength, 
        int numHashFuncs,
        int firstHashFunc
    ){
        constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;
        const std::uint64_t kmer_mask = std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - kmerLength) * 2);

        assert(kmerLength <= maximum_kmer_length);
        
        std::vector<std::uint64_t> hashvalues(numHashFuncs, std::numeric_limits<std::uint64_t>::max());

        if(sequenceLength >= kmerLength){
            SequenceHelpers::forEachEncodedCanonicalKmerFromEncodedSequence(
                sequence,
                sequenceLength,
                kmerLength,
                [&](std::uint64_t kmer, int /*pos*/){
                    using hasher = hashers::MurmurHash<std::uint64_t>;

                    for(int i = 0; i < numHashFuncs; i++){
                        const int hashFuncId = i + firstHashFunc;
                        const auto hashvalue = hasher::hash(kmer + hashFuncId);
                        hashvalues[i] = std::min(hashvalues[i], hashvalue);
                    }
                }
            );
        }

        return std::transform(hashvalues.begin(), hashvalues.begin() + numHashFuncs, output, [&](auto hash){ return HashValueType(hash & kmer_mask); });
    }

    std::vector<HashValueType> hash(
        const unsigned int* sequence, 
        int sequenceLength, 
        int kmerLength, 
        int numHashFuncs,
        int firstHashFunc
    ){
        std::vector<HashValueType> result(numHashFuncs);

        hashInto(
            result.begin(),
            sequence,
            sequenceLength,
            kmerLength,
            numHashFuncs,
            firstHashFunc
        );
        
        return result;
    }

};





}


#endif