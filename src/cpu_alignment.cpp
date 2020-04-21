#include <cpu_alignment.hpp>
#include <config.hpp>

#include <sequence.hpp>
#include <hostdevicefunctions.cuh>

#include <vector>


namespace care{
namespace cpu{
namespace shd{



    AlignmentResult
    cpuShiftedHammingDistancePopcount2BitHiLo(
            CpuAlignmentHandle& handle,
            const unsigned int* subjectHiLo,
            int subjectLength,
            const unsigned int* candidateHiLo,
            int candidateLength,
            int min_overlap,
            float maxErrorRate,
            float min_overlap_ratio) noexcept{

        assert(subjectLength > 0);
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

        const int subjectInts = getEncodedNumInts2BitHiLo(subjectLength);
        const int candidateInts = getEncodedNumInts2BitHiLo(candidateLength);
        const int maxInts = std::max(subjectInts, candidateInts);


        shiftbuffer.resize(maxInts);

        const unsigned int* const subjectBackup_hi = subjectHiLo;
        const unsigned int* const subjectBackup_lo = subjectHiLo + subjectInts / 2;

        const unsigned int* const candidateBackup_hi = candidateHiLo;
        const unsigned int* const candidateBackup_lo = candidateHiLo + candidateInts / 2;

        const int totalbases = subjectLength + candidateLength;
        const int minoverlap = std::max(min_overlap, int(float(subjectLength) * min_overlap_ratio));

        int bestScore = totalbases; // score is number of mismatches
        int bestShift = -candidateLength; // shift of query relative to subject. shift < 0 if query begins before subject

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

        std::copy_n(subjectHiLo, subjectInts, shiftbuffer.begin());
        unsigned int* shiftbuffer_hi = shiftbuffer.data();
        unsigned int* shiftbuffer_lo = shiftbuffer.data() + subjectInts / 2;


        for(int shift = 0; shift < subjectLength - minoverlap + 1; ++shift){
            const int overlapsize = std::min(subjectLength - shift, candidateLength);
            bool b = handle_shift(shift, overlapsize,
                                shiftbuffer_hi, shiftbuffer_lo, identity,
                                subjectInts,
                                candidateBackup_hi, candidateBackup_lo, identity);
            if(!b){
                break;
            }
        }

        std::copy_n(candidateHiLo, candidateInts, shiftbuffer.begin());
        shiftbuffer_hi = shiftbuffer.data();
        shiftbuffer_lo = shiftbuffer.data() + candidateInts / 2;

        for(int shift = -1; shift >= -candidateLength + minoverlap; --shift){
            const int overlapsize = std::min(subjectLength, candidateLength + shift);

            bool b = handle_shift(shift, overlapsize,
                                shiftbuffer_hi, shiftbuffer_lo, identity,
                                candidateInts,
                                subjectBackup_hi, subjectBackup_lo, identity);

            if(!b){
                break;
            }
        }

        AlignmentResult alignmentresult;
        alignmentresult.isValid = (bestShift != -candidateLength);

        const int candidateoverlapbegin_incl = std::max(-bestShift, 0);
        const int candidateoverlapend_excl = std::min(candidateLength, subjectLength - bestShift);
        const int overlapsize = candidateoverlapend_excl - candidateoverlapbegin_incl;
        const int opnr = bestScore - totalbases + 2*overlapsize;

        alignmentresult.score = bestScore;
        alignmentresult.overlap = overlapsize;
        alignmentresult.shift = bestShift;
        alignmentresult.nOps = opnr;

        return alignmentresult;
    }




}
}
}
