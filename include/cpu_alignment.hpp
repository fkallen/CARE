#ifndef CARE_CPU_ALIGNMENT_HPP
#define CARE_CPU_ALIGNMENT_HPP


#include <hostdevicefunctions.cuh>
#include <sequencehelpers.hpp>

#include <config.hpp>

#include <vector>
#include <cassert>
#include <algorithm>

namespace care{
namespace cpu{

    template<class Iter1, class Iter2, class Equal>
    int hammingDistance(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2, Equal isEqual){
        int result = 0;

        while(first1 != last1 && first2 != last2){
            result += isEqual(*first1, *first2) ? 0 : 1;

            ++first1;
            ++first2;
        }

        //positions which do not overlap count as mismatch.
        //at least one of the remaining ranges is empty
        result += std::distance(first1, last1);
        result += std::distance(first2, last2);

        return result;
    }

    template<class Iter1, class Iter2>
    int hammingDistance(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2){
        auto isEqual = [](const auto& l, const auto& r){
            return l == r;
        };

        return hammingDistance(first1, last1, first2, last2, isEqual);
    }

    template<class Iter1, class Iter2, class Equal>
    int hammingDistanceOverlap(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2, Equal isEqual){
        int result = 0;

        while(first1 != last1 && first2 != last2){
            result += isEqual(*first1, *first2) ? 0 : 1;

            ++first1;
            ++first2;
        }

        //positions which do not overlap are excluded

        return result;
    }

    template<class Iter1, class Iter2>
    int hammingDistanceOverlap(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2){
        auto isEqual = [](const auto& l, const auto& r){
            return l == r;
        };

        return hammingDistanceOverlap(first1, last1, first2, last2, isEqual);
    }



    template<class Iter1, class Iter2, class Equal>
    int longestMatch(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2, Equal isEqual){
        int longest = 0;
        int current = 0;

        while(first1 != last1 && first2 != last2){
            if(isEqual(*first1, *first2)){
                current++;
            }else{
                longest = std::max(longest, current);
                current = 0;
            }

            ++first1;
            ++first2;
        }

        longest = std::max(longest, current);

        return longest;
    }

    template<class Iter1, class Iter2>
    int longestMatch(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2){
        auto isEqual = [](const auto& l, const auto& r){
            return l == r;
        };

        return longestMatch(first1, last1, first2, last2, isEqual);
    }





namespace shd{

    enum class ShiftDirection {Left, Right, LeftRight, None};

    struct AlignmentResult{
    	int score;
    	int overlap;
    	int shift;
    	int nOps; //edit distance / number of operations
    	bool isValid;

        int get_score() const { return score;}
        int get_overlap() const { return overlap;}
        int get_shift() const { return shift;}
        int get_nOps() const { return nOps;}
        bool get_isValid() const { return isValid;}

        bool operator==(const AlignmentResult& rhs) const {
            return score == rhs.score && overlap == rhs.overlap && shift == rhs.shift && nOps == rhs.nOps && isValid == rhs.isValid;
        }
        bool operator!=(const AlignmentResult& rhs) const{
            return !(operator==(rhs));
        }
    };

    struct CpuAlignmentHandle{
        std::vector<unsigned int> shiftbuffer;
        std::vector<unsigned int> anchorConversionBuffer;
        std::vector<unsigned int> candidateConversionBuffer;
    };

    AlignmentResult
    cpuShiftedHammingDistancePopcount2BitHiLo(
            CpuAlignmentHandle& handle,
            const unsigned int* anchorHiLo,
            int anchorLength,
            const unsigned int* candidateHiLo,
            int candidateLength,
            int min_overlap,
            float maxErrorRate,
            float min_overlap_ratio) noexcept;


    template<ShiftDirection direction>
    AlignmentResult
    cpuShiftedHammingDistancePopcount2BitHiLoWithDirection(
            CpuAlignmentHandle& handle,
            const unsigned int* anchorHiLo,
            int anchorLength,
            const unsigned int* candidateHiLo,
            int candidateLength,
            int min_overlap,
            float maxErrorRate,
            float min_overlap_ratio) noexcept{

        assert(anchorLength > 0);
        assert(candidateLength > 0);

        auto popcount = [](auto i){return __builtin_popcount(i);};
        auto identity = [](auto i){return i;};

        auto hammingDistanceWithShift = [&](bool doShift, int overlapsize, int max_errors,
                                            unsigned int* shiftptr_hi, unsigned int* shiftptr_lo, auto transfunc1,
                                            int shiftptr_size,
                                            const unsigned int* otherptr_hi, const unsigned int* otherptr_lo,
                                            auto transfunc2){
            if(doShift){
                shiftBitArrayLeftBy<1>(shiftptr_hi, shiftptr_size / 2, transfunc1);
                shiftBitArrayLeftBy<1>(shiftptr_lo, shiftptr_size / 2, transfunc1);
            }
            const int score = hammingdistanceHiLo(shiftptr_hi,
                                                    shiftptr_lo,
                                                    otherptr_hi,
                                                    otherptr_lo,
                                                    overlapsize,
                                                    overlapsize,
                                                    max_errors,
                                                    transfunc1,
                                                    transfunc2,
                                                    popcount);

            return score;
        };

        auto& shiftbuffer = handle.shiftbuffer;

        const int anchorInts = SequenceHelpers::getEncodedNumInts2BitHiLo(anchorLength);
        const int candidateInts = SequenceHelpers::getEncodedNumInts2BitHiLo(candidateLength);
        const int maxInts = std::max(anchorInts, candidateInts);


        shiftbuffer.resize(maxInts);

        const unsigned int* const anchorBackup_hi = anchorHiLo;
        const unsigned int* const anchorBackup_lo = anchorHiLo + anchorInts / 2;

        const unsigned int* const candidateBackup_hi = candidateHiLo;
        const unsigned int* const candidateBackup_lo = candidateHiLo + candidateInts / 2;

        const int totalbases = anchorLength + candidateLength;
        const int minoverlap = std::max(min_overlap, int(float(anchorLength) * min_overlap_ratio));

        int bestScore = totalbases; // score is number of mismatches
        int bestShift = -candidateLength; // shift of query relative to anchor. shift < 0 if query begins before anchor

        auto handle_shift = [&](int shift, int overlapsize,
                                unsigned int* shiftptr_hi, unsigned int* shiftptr_lo, auto transfunc1,
                                int shiftptr_size,
                                const unsigned int* otherptr_hi, const unsigned int* otherptr_lo,
                                auto transfunc2){

            const int max_errors_excl = std::min(int(float(overlapsize) * maxErrorRate),
                                            bestScore - totalbases + 2*overlapsize);

            if(max_errors_excl > 0){

                const int mismatches = hammingDistanceWithShift(shift != 0, overlapsize, max_errors_excl,
                                                        shiftptr_hi,shiftptr_lo, transfunc1,
                                                        shiftptr_size,
                                                        otherptr_hi, otherptr_lo, transfunc2);

                const int score = (mismatches < max_errors_excl ?
                                mismatches + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                                : std::numeric_limits<int>::max()); // too many errors, discard

                if(score < bestScore){
                    bestScore = score;
                    bestShift = shift;
                }

                return true;
            }else{
                return false;
            }
        };

        auto handle_zero_shift = [&](int overlapsize,
                                const unsigned int* ptr_hi, const unsigned int* ptr_lo, auto transfunc1,
                                int /*ptr_size*/,
                                const unsigned int* otherptr_hi, const unsigned int* otherptr_lo,
                                auto transfunc2){

            const int shift = 0;

            const int max_errors_excl = std::min(int(float(overlapsize) * maxErrorRate),
                                            bestScore - totalbases + 2*overlapsize);

            if(max_errors_excl > 0){

                const int mismatches = hammingdistanceHiLo(ptr_hi,
                                                    ptr_lo,
                                                    otherptr_hi,
                                                    otherptr_lo,
                                                    overlapsize,
                                                    overlapsize,
                                                    max_errors_excl,
                                                    transfunc1,
                                                    transfunc2,
                                                    popcount);

                const int score = (mismatches < max_errors_excl ?
                                mismatches + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                                : std::numeric_limits<int>::max()); // too many errors, discard

                if(score < bestScore){
                    bestScore = score;
                    bestShift = shift;
                }

                return true;
            }else{
                return false;
            }
        };

        //shift == 0
        {
            const int shift = 0;
            const int overlapsize = std::min(anchorLength - shift, candidateLength);

            handle_zero_shift(overlapsize,
                    anchorBackup_hi, anchorBackup_lo, identity,
                    anchorInts,
                    candidateBackup_hi, candidateBackup_lo, identity);

        }

        unsigned int* shiftbuffer_hi;
        unsigned int* shiftbuffer_lo;

        if(direction == ShiftDirection::Right || direction == ShiftDirection::LeftRight){
            //compute alignments with shift to the right
            std::copy_n(anchorHiLo, anchorInts, shiftbuffer.begin());
            shiftbuffer_hi = shiftbuffer.data();
            shiftbuffer_lo = shiftbuffer.data() + anchorInts / 2;

            for(int shift = 1; shift < anchorLength - minoverlap + 1; ++shift){
                const int overlapsize = std::min(anchorLength - shift, candidateLength);
                bool b = handle_shift(shift, overlapsize,
                                    shiftbuffer_hi, shiftbuffer_lo, identity,
                                    anchorInts,
                                    candidateBackup_hi, candidateBackup_lo, identity);
                if(!b){
                    break;
                }
            }
        }

        if(direction == ShiftDirection::Left || direction == ShiftDirection::LeftRight){

            std::copy_n(candidateHiLo, candidateInts, shiftbuffer.begin());
            shiftbuffer_hi = shiftbuffer.data();
            shiftbuffer_lo = shiftbuffer.data() + candidateInts / 2;

            for(int shift = -1; shift >= -candidateLength + minoverlap; --shift){
                const int overlapsize = std::min(anchorLength, candidateLength + shift);

                bool b = handle_shift(shift, overlapsize,
                                    shiftbuffer_hi, shiftbuffer_lo, identity,
                                    candidateInts,
                                    anchorBackup_hi, anchorBackup_lo, identity);

                if(!b){
                    break;
                }
            }

        }

        AlignmentResult alignmentresult;
        alignmentresult.isValid = (bestShift != -candidateLength);

        const int candidateoverlapbegin_incl = std::max(-bestShift, 0);
        const int candidateoverlapend_excl = std::min(candidateLength, anchorLength - bestShift);
        const int overlapsize = candidateoverlapend_excl - candidateoverlapbegin_incl;
        const int opnr = bestScore - totalbases + 2*overlapsize;

        alignmentresult.score = bestScore;
        alignmentresult.overlap = overlapsize;
        alignmentresult.shift = bestShift;
        alignmentresult.nOps = opnr;

        return alignmentresult;
    }

    

    template<ShiftDirection direction, class Iter>
    Iter
    cpuShiftedHammingDistancePopcount2BitWithDirection(
            CpuAlignmentHandle& handle,
            Iter destinationBegin,
            const unsigned int* anchor2Bit,
            int anchorLength,
            const unsigned int* candidates2Bit,
            int candidatePitchInInts,
            const int* candidateLengths,
            int numCandidates,
            int min_overlap,
            float maxErrorRate,
            float min_overlap_ratio) noexcept{

        const int newanchorInts = SequenceHelpers::getEncodedNumInts2BitHiLo(anchorLength);

        handle.anchorConversionBuffer.resize(newanchorInts);        

        SequenceHelpers::convert2BitTo2BitHiLo(
            handle.anchorConversionBuffer.data(),
            anchor2Bit,
            anchorLength
        );

        auto curIter = destinationBegin;

        for(int candidateIndex = 0; candidateIndex < numCandidates; candidateIndex++, ++curIter){
            const unsigned int* candidate2Bit = candidates2Bit + candidatePitchInInts * candidateIndex;
            const int candidateLength = candidateLengths[candidateIndex];

            const int candidateIntsHiLo = SequenceHelpers::getEncodedNumInts2BitHiLo(candidateLength);
            handle.candidateConversionBuffer.resize(candidateIntsHiLo);

            SequenceHelpers::convert2BitTo2BitHiLo(
                handle.candidateConversionBuffer.data(),
                candidate2Bit,
                candidateLength
            );

            *curIter = cpuShiftedHammingDistancePopcount2BitHiLoWithDirection<direction>(
                            handle,
                            handle.anchorConversionBuffer.data(),
                            anchorLength,
                            handle.candidateConversionBuffer.data(),
                            candidateLength,
                            min_overlap,
                            maxErrorRate,
                            min_overlap_ratio
                        );
        }

        return curIter;
    }

    template<int foo=0>
    int hammingdistanceHiLoWithShift_funnel(
        const unsigned int* lhi_begin,
        const unsigned int* llo_begin,
        const unsigned int* rhi,
        const unsigned int* rlo,
        int lhi_bitcount,
        int rhi_bitcount,
        int numIntsL,
        int /*numIntsR*/,
        int shiftamount,
        int max_errors_excl
    ){

        const int overlap_bitcount = std::min(std::max(0, lhi_bitcount - shiftamount), rhi_bitcount);

        if(overlap_bitcount == 0)
            return max_errors_excl+1;

        const int partitions = SDIV(overlap_bitcount, (8 * sizeof(unsigned int)));
        const int remaining_bitcount = partitions * sizeof(unsigned int) * 8 - overlap_bitcount;
        const int completeShiftInts = shiftamount / (8 * sizeof(unsigned int));
        const int remainingShift = shiftamount - completeShiftInts * 8 * sizeof(unsigned int);

        auto myfunnelshift = [](unsigned int a, unsigned int b, int shift){
            if(shift == 0) return a;
            return (a << shift) | (b >> (8 * sizeof(unsigned int) - shift));
        };

        int result = 0;

        for(int i = 0; i < partitions - 1 && result < max_errors_excl; i += 1) {
            //compute the shifted values of l
            const unsigned int aaa = lhi_begin[(completeShiftInts + i)];
            const unsigned int aab = lhi_begin[(completeShiftInts + i + 1)];
            const unsigned int a = myfunnelshift(aaa, aab, remainingShift);
            const unsigned int baa = llo_begin[(completeShiftInts + i)];
            const unsigned int bab = llo_begin[(completeShiftInts + i + 1)];
            const unsigned int b = myfunnelshift(baa, bab, remainingShift);
            const unsigned int hixor = a ^ rhi[(i)];
            const unsigned int loxor = b ^ rlo[(i)];
            const unsigned int bits = hixor | loxor;
            result += __builtin_popcount(bits);
        }

        if(result >= max_errors_excl)
            return result;

        const unsigned int mask = remaining_bitcount == 0 ? 0xFFFFFFFF : 0xFFFFFFFF << (remaining_bitcount);
        
        unsigned int a = 0;
        unsigned int b = 0;
        if(completeShiftInts + partitions - 1 < numIntsL - 1){
            unsigned int aaa = lhi_begin[(completeShiftInts + partitions - 1)];
            unsigned int aab = lhi_begin[(completeShiftInts + partitions - 1 + 1)];
            a = myfunnelshift(aaa, aab, remainingShift);
            unsigned int baa = llo_begin[(completeShiftInts + partitions - 1)];
            unsigned int bab = llo_begin[(completeShiftInts + partitions - 1 + 1)];
            b = myfunnelshift(baa, bab, remainingShift);
        }else{
            a = (lhi_begin[(completeShiftInts + partitions - 1)]) << remainingShift;
            b = (llo_begin[(completeShiftInts + partitions - 1)]) << remainingShift;
        }
        const unsigned int hixor = a ^ rhi[(partitions - 1)];
        const unsigned int loxor = b ^ rlo[(partitions - 1)];
        const unsigned int bits = hixor | loxor;
        result += __builtin_popcount(bits & mask);

        return result;
    }


    template<int foo=0>
    AlignmentResult
    cpuShiftedHammingDistancePopcount2BitHiLo3(
            CpuAlignmentHandle& /*handle*/,
            const unsigned int* anchorHiLo,
            int anchorLength,
            const unsigned int* candidateHiLo,
            int candidateLength,
            int min_overlap,
            float maxErrorRate,
            float min_overlap_ratio,
            bool debug = false) noexcept{

        assert(anchorLength > 0);
        assert(candidateLength > 0);

        const int anchorInts = SequenceHelpers::getEncodedNumInts2BitHiLo(anchorLength);
        const int candidateInts = SequenceHelpers::getEncodedNumInts2BitHiLo(candidateLength);

        const unsigned int* const anchor_hi = anchorHiLo;
        const unsigned int* const anchor_lo = anchorHiLo + anchorInts / 2;

        const unsigned int* const candidate_hi = candidateHiLo;
        const unsigned int* const candidate_lo = candidateHiLo + candidateInts / 2;

        const int totalbases = anchorLength + candidateLength;
        const int minoverlap = std::max(min_overlap, int(float(anchorLength) * min_overlap_ratio));

        int bestScore = totalbases; // score is number of mismatches
        int bestShift = -candidateLength; // shift of query relative to anchor. shift < 0 if query begins before anchor
        int bestOverlap = 0;
        int bestHammingDistance = totalbases;

        auto updateBest = [&](int shift, int overlapsize, int hammingDistance, int max_errors_excl){
            //treat non-overlapping positions as mismatches to prefer a greater overlap if hamming distance is equal for multiple shifts
            const int nonoverlapping = anchorLength + candidateLength - 2 * overlapsize;
            const int score = (hammingDistance < max_errors_excl ?
                hammingDistance + nonoverlapping // non-overlapping regions count as mismatches
                : std::numeric_limits<int>::max()); // too many errors, discard

            if(score < bestScore){
                bestScore = score;
                bestShift = shift;
                bestOverlap = overlapsize;
                bestHammingDistance = hammingDistance;
            }                
        };

        for(int shift = 0; shift < anchorLength - minoverlap + 1; shift += 1) {
            const int overlapsize = std::min(anchorLength - shift, candidateLength);
            const int max_errors_excl = std::min(int(float(overlapsize) * maxErrorRate),
                bestScore - totalbases + 2*overlapsize);
        
            if(max_errors_excl > 0){
                const int hammingDistance = hammingdistanceHiLoWithShift_funnel(
                    anchor_hi,
                    anchor_lo,
                    candidate_hi,
                    candidate_lo,
                    anchorLength,
                    candidateLength,
                    anchorInts / 2,
                    candidateInts / 2,
                    shift,
                    max_errors_excl
                );
                if(debug){
                    std::cerr << "shift " << shift << ", hamming " << hammingDistance << "\n";
                }

                updateBest(shift, overlapsize, hammingDistance, max_errors_excl);
            }
        }

        for(int shift = -1; shift >= - candidateLength + minoverlap; shift -= 1) {
            const int overlapsize = std::min(anchorLength, candidateLength + shift);
            const int max_errors_excl = std::min(int(float(overlapsize) * maxErrorRate),
                bestScore - (totalbases) + 2*overlapsize);

            if(max_errors_excl > 0){
                const int hammingDistance = hammingdistanceHiLoWithShift_funnel(
                    candidate_hi,
                    candidate_lo,
                    anchor_hi,
                    anchor_lo,
                    candidateLength,
                    anchorLength,
                    candidateInts / 2,
                    anchorInts / 2,
                    -shift,
                    max_errors_excl
                );

                if(debug){
                    std::cerr << "shift " << shift << ", hamming " << hammingDistance << "\n";
                }

                updateBest(shift, overlapsize, hammingDistance, max_errors_excl); 
            }
        }

        if(debug){
            std::cerr << "new bestScore: " << bestScore << ", totalbases: " << totalbases << ", overlapsize: " << bestOverlap << "\n";
        }

        AlignmentResult alignmentresult;
        alignmentresult.isValid = (bestShift != -candidateLength);
        alignmentresult.score = bestScore;
        alignmentresult.overlap = bestOverlap;
        alignmentresult.shift = bestShift;
        alignmentresult.nOps = alignmentresult.isValid ? bestHammingDistance : 0;

        return alignmentresult;
    }








    template<class Iter>
    Iter
    cpuShiftedHammingDistancePopcount2Bit(
            CpuAlignmentHandle& handle,
            Iter destinationBegin,
            const unsigned int* anchor2Bit,
            int anchorLength,
            const unsigned int* candidates2Bit,
            int candidatePitchInInts,
            const int* candidateLengths,
            int numCandidates,
            int min_overlap,
            float maxErrorRate,
            float min_overlap_ratio) noexcept{

        const int newanchorInts = SequenceHelpers::getEncodedNumInts2BitHiLo(anchorLength);

        //handle.anchorConversionBuffer.resize(newanchorInts);

        std::vector<unsigned int> anchorConversionBuffer(newanchorInts);
        std::vector<unsigned int> candidateConversionBuffer;

        // auto& aConvBuffer = handle.anchorConversionBuffer;
        // auto& cConvBuffer = handle.candidateConversionBuffer;

        auto& aConvBuffer = anchorConversionBuffer;
        auto& cConvBuffer = candidateConversionBuffer;

        SequenceHelpers::convert2BitTo2BitHiLo(
            aConvBuffer.data(),
            anchor2Bit,
            anchorLength
        );

        auto curIter = destinationBegin;

        for(int candidateIndex = 0; candidateIndex < numCandidates; candidateIndex++, ++curIter){
            const unsigned int* candidate2Bit = candidates2Bit + candidatePitchInInts * candidateIndex;
            const int candidateLength = candidateLengths[candidateIndex];

            const int candidateIntsHiLo = SequenceHelpers::getEncodedNumInts2BitHiLo(candidateLength);
            cConvBuffer.resize(candidateIntsHiLo);

            SequenceHelpers::convert2BitTo2BitHiLo(
                cConvBuffer.data(),
                candidate2Bit,
                candidateLength
            );
            *curIter = cpuShiftedHammingDistancePopcount2BitHiLo3(
                handle,
                aConvBuffer.data(),
                anchorLength,
                cConvBuffer.data(),
                candidateLength,
                min_overlap,
                maxErrorRate,
                min_overlap_ratio,
                false //candidateIndex == 33
            );

            // *curIter = cpuShiftedHammingDistancePopcount2BitHiLo2(
            //                 handle,
            //                 aConvBuffer.data(),
            //                 anchorLength,
            //                 cConvBuffer.data(),
            //                 candidateLength,
            //                 min_overlap,
            //                 maxErrorRate,
            //                 min_overlap_ratio,
            //                 false //candidateIndex == 33
            //             );
            // auto res = cpuShiftedHammingDistancePopcount2BitHiLo3(
            //     handle,
            //     aConvBuffer.data(),
            //     anchorLength,
            //     cConvBuffer.data(),
            //     candidateLength,
            //     min_overlap,
            //     maxErrorRate,
            //     min_overlap_ratio,
            //     false //candidateIndex == 33
            // );

            // if(res != *curIter){
            //     std::cerr << "Anchor: " << SequenceHelpers::get2BitString(anchor2Bit, anchorLength) << "\n";
            //     std::cerr << "Cand  : " << SequenceHelpers::get2BitString(candidate2Bit, candidateLength) << "\n";
            //     std::cerr << res.score << " " << res.overlap << " " << res.shift << " " << res.nOps << " " << res.isValid << "\n";
            //     std::cerr << (*curIter).score << " " << (*curIter).overlap << " " << (*curIter).shift << " " << (*curIter).nOps << " " << (*curIter).isValid << "\n";
            //     throw std::runtime_error("error alignment");
            // }
        }

        return curIter;
    }




} //namespace shd

using SHDResult = shd::AlignmentResult;


} //namespace care::cpu

} //namespace care




#endif
