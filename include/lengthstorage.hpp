#ifndef CARE_LENGTH_STORAGE_HPP
#define CARE_LENGTH_STORAGE_HPP

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

namespace care{

struct LengthStore{
    using Data_t = std::uint32_t;

    LengthStore() = default;

    LengthStore(int minL, int maxL, std::int64_t numElements_) : minLength(minL), maxLength(maxL), numElements(numElements_){
        assert(minL <= maxL);

        const int diff = maxL - minL;
        if(diff == 0){
            return;
        }

        auto nextLargerPowerOfTwo = [](std::uint32_t n){
            n |= n >> 1;
            n |= n >> 2;
            n |= n >> 4;
            n |= n >> 8;
            n |= n >> 16;
            return n + 1;
        };

        auto nextP2 = nextLargerPowerOfTwo(diff);
        assert(nextP2 != std::numeric_limits<std::uint32_t>::max());

        bitsPerLength = int(std::log2(nextP2));
        bitsMask = 0;
        for(int i = 0; i < bitsPerLength;i++){
            bitsMask = (bitsMask << 1) | 1;
        }

        const std::size_t dataelements = ((bitsPerLength * getNumElements()) + DataTBits - 1) / DataTBits;

        data.resize(dataelements);
    }

    LengthStore(const LengthStore&) = default;
    LengthStore(LengthStore&&) = default;
    LengthStore& operator=(const LengthStore&) = default;
    LengthStore& operator=(LengthStore&&) = default;

    int getMinLength() const{
        return minLength;
    }

    int getMaxLength() const{
        return maxLength;
    }

    std::int64_t getNumElements() const{
        return numElements;
    }

    const Data_t* getRaw() const{
        return data.data();
    }

    int getRawBitsPerLength() const{
        return bitsPerLength;
    }

    std::size_t getRawSizeInBytes() const{
        return data.size() * sizeof(Data_t);
    }

    std::size_t getRawSizeInElements() const{
        return data.size();
    }

    int getLength(std::int64_t index) const{
        assert(index >= 0 && index < getNumElements());

        if(getMinLength() == getMaxLength()){
            return getMinLength();
        }

        const std::uint64_t firstBit = bitsPerLength * index;
        const std::uint64_t lastBitExcl = bitsPerLength * (index+1);
        const std::uint64_t firstuintindex = firstBit / DataTBits;
        const int begin = firstBit - firstuintindex * DataTBits;
        const int endExcl = lastBitExcl - firstuintindex * DataTBits;
        
        const auto first = data[firstuintindex];
        //prevent oob access
        const auto second = firstuintindex == data.size() - 1 ? data[firstuintindex] : data[firstuintindex + 1];
        const Data_t lengthBits = getBits(first, second, begin, endExcl);

        return int(lengthBits) + minLength;
    }

    void setLength(std::int64_t index, int length){
        assert(minLength <= length && length <= maxLength);

        if(getMinLength() == getMaxLength()){
            return;
        }

        const int diff = length - minLength;
        const std::uint64_t firstBit = bitsPerLength * index;
        const std::uint64_t lastBitExcl = bitsPerLength * (index+1);
        const std::uint64_t firstuintindex = firstBit / DataTBits;
        const int begin = firstBit - firstuintindex * DataTBits;
        const int endExcl = lastBitExcl - firstuintindex * DataTBits;
        //std::cerr << index << ": Set [" << begin << ", " << endExcl << ") of index " << firstuintindex << "\n";

        auto& first = data[firstuintindex];
        //prevent oob access
        auto& second = firstuintindex == data.size() - 1 ? data[firstuintindex] : data[firstuintindex + 1];

        setBits(first, second, begin, endExcl, Data_t(diff));
    }

private:   
    //extracts at most 32 bits out of the 64 bits [l,r]. extracts bits [begin,endExcl) from [l,r]
    //bits are enumerated from left to right
    Data_t getBits(Data_t l, Data_t r, int begin, int endExcl) const{

        assert(0 <= begin && begin < endExcl && endExcl <= 2 * LengthStore::DataTBits);
        
        const int lbegin = std::min(LengthStore::DataTBits, begin);
        const int lendExcl = std::min(LengthStore::DataTBits, endExcl);

        const int rbegin = std::max(0, begin - LengthStore::DataTBits);
        const int rendExcl = std::max(0, endExcl - LengthStore::DataTBits);

        Data_t lmask = 0;
        for(int i = lbegin; i < lendExcl; i++){
            lmask = (lmask << 1) | 1;
        }

        Data_t rmask = 0;
        for(int i = rbegin; i < rendExcl; i++){
            rmask = (rmask << 1) | 1;
        }

        const Data_t lpiece = (l >> (LengthStore::DataTBits - lendExcl)) & lmask;
        const Data_t rpiece = (r >> (LengthStore::DataTBits - rendExcl)) & rmask;
        Data_t result = lpiece << (bitsPerLength - (lendExcl - lbegin)) | rpiece;
        return result;
    }

    //set bits [begin, endExcl( of the 64 bits [l,r] to the bitsPerLength rightmost bits of toSet
    //bits are enumerated from left to right
    void setBits(Data_t& l, Data_t& r, int begin, int endExcl, Data_t toSet) const{

        assert(0 <= begin && begin < endExcl && endExcl <= 2 * LengthStore::DataTBits);

        toSet = (toSet & bitsMask);
        
        const int rbegin = std::max(0, begin - LengthStore::DataTBits);
        const int rendExcl = std::max(0, endExcl - LengthStore::DataTBits);

        const int rpartition = rendExcl - rbegin; // the rightmost rpartition bits of toSet belong to r

        if(rpartition > 0){
            Data_t rmask = 0;
            for(int i = rbegin; i < rendExcl; i++){
                rmask |= (1u << (LengthStore::DataTBits - 1 - i));
            }

            r &= ~rmask;

            Data_t rpartitionMask = 0;
            for(int i = 0; i < rpartition; i++){
                rpartitionMask = (rpartitionMask << 1) | 1;
            }

            r |= ((toSet & rpartitionMask) << (LengthStore::DataTBits - rpartition));
        }

        const int lbegin = std::min(LengthStore::DataTBits, begin);
        const int lendExcl = std::min(LengthStore::DataTBits, endExcl);

        Data_t lmask = 0;
        for(int i = lbegin; i < lendExcl; i++){
            lmask |= (1u << (LengthStore::DataTBits - 1 - i));
        }

        l &= ~lmask;

        toSet = toSet >> rpartition;
        toSet = toSet << (LengthStore::DataTBits - lendExcl);

        l |= toSet;
    }

    int DataTBits = 8 * sizeof(Data_t);
    int bitsPerLength = 0;
    int minLength = 0;
    int maxLength = 0;
    Data_t bitsMask = 0;
    std::int64_t numElements = 0;

    std::vector<Data_t> data;
};

}



#endif