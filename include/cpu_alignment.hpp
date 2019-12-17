#ifndef CARE_CPU_ALIGNMENT_HPP
#define CARE_CPU_ALIGNMENT_HPP


#include <sequence.hpp>

#include <config.hpp>
#include <shiftedhammingdistance_common.hpp>

#include <vector>
#include <cassert>
#include <algorithm>

namespace care{
namespace cpu{

namespace shd{

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

    template<class Accessor>
    AlignmentResult
    cpu_shifted_hamming_distance(const char* subject,
                                int subjectLength,
                                const char* query,
                                int queryLength,
                                int min_overlap,
                                float maxErrorRate,
                                float min_overlap_ratio,
                                Accessor getChar)  noexcept{

        const int totalbases = subjectLength + queryLength;
        const int minoverlap = std::max(min_overlap, int(float(subjectLength) * min_overlap_ratio));
        int bestScore = totalbases; // score is number of mismatches
        int bestShift = -queryLength; // shift of query relative to subject. shift < 0 if query begins before subject

        for(int shift = -queryLength + minoverlap; shift < subjectLength - minoverlap + 1; shift++){
            const int overlapsize = std::min(queryLength, subjectLength - shift) - std::max(-shift, 0);
            const int max_errors = int(float(overlapsize) * maxErrorRate);
            int score = 0;

            for(int j = std::max(-shift, 0); j < std::min(queryLength, subjectLength - shift) && score < max_errors; j++){
                score += getChar(subject, subjectLength, j + shift) != getChar(query, queryLength, j);
            }

            score = (score < max_errors ?
                    score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                    : std::numeric_limits<int>::max()); // too many errors, discard

            if(score < bestScore){
                bestScore = score;
                bestShift = shift;
            }
        }

        AlignmentResult result;
        result.isValid = (bestShift != -queryLength);

        const int queryoverlapbegin_incl = std::max(-bestShift, 0);
        const int queryoverlapend_excl = std::min(queryLength, subjectLength - bestShift);
        const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
        const int opnr = bestScore - totalbases + 2*overlapsize;

        result.score = bestScore;
        result.overlap = overlapsize;
        result.shift = bestShift;
        result.nOps = opnr;

        return result;
    }


    template<class Iter>
    Iter
    cpu_multi_shifted_hamming_distance_popcount(Iter destinationbegin,
                                                const char* subject_charptr,
                                                int subjectLength,
                                                const std::vector<char>& querydata,
                                                const std::vector<int>& queryLengths,
                                                int max_sequence_bytes,
                                                int min_overlap,
                                                float maxErrorRate,
                                                float min_overlap_ratio) noexcept{

        assert(max_sequence_bytes % 4 == 0);

        if(queryLengths.size() == 0) return destinationbegin;

        auto getNumBytes = [] (int sequencelength){
            return sizeof(unsigned int) * getEncodedNumInts2BitHiLo(sequencelength);
        };

        auto popcount = [](auto i){return __builtin_popcount(i);};

        auto identity = [](auto i){return i;};

        auto hammingDistanceWithShift = [&](int shift, int overlapsize, int max_errors,
                                            unsigned int* shiftptr_hi, unsigned int* shiftptr_lo, auto transfunc1,
                                            int shiftptr_size,
                                            const unsigned int* otherptr_hi, const unsigned int* otherptr_lo,
                                            auto transfunc2){

            const int shiftamount = shift == 0 ? 0 : 1;

            shiftBitArrayLeftBy(shiftptr_hi, shiftptr_size / 2, shiftamount, transfunc1);
            shiftBitArrayLeftBy(shiftptr_lo, shiftptr_size / 2, shiftamount, transfunc1);

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


        const int nQueries = int(queryLengths.size());

        Iter destination = destinationbegin;

        const unsigned int* const subject = (const unsigned int*)subject_charptr;
        const int subjectints = getNumBytes(subjectLength) / sizeof(unsigned int);
        const int max_sequence_ints = max_sequence_bytes / sizeof(unsigned int);

        std::vector<unsigned int> shiftbuffer(max_sequence_ints);

        const unsigned int* const subjectBackup_hi = (const unsigned int*)(subject);
        const unsigned int* const subjectBackup_lo = ((const unsigned int*)subject) + subjectints / 2;

        for(int index = 0; index < nQueries; index++){
            const unsigned int* const query = (const unsigned int*)(querydata.data() + max_sequence_bytes * index);
            const int queryLength = queryLengths[index];

            const int queryints = getNumBytes(queryLength) / sizeof(unsigned int);
            const unsigned int* const queryBackup_hi = query;
            const unsigned int* const queryBackup_lo = query + queryints / 2;

            const int totalbases = subjectLength + queryLength;
            const int minoverlap = std::max(min_overlap, int(float(subjectLength) * min_overlap_ratio));


            int bestScore = totalbases; // score is number of mismatches
            int bestShift = -queryLength; // shift of query relative to subject. shift < 0 if query begins before subject

            auto handle_shift = [&](int shift, int overlapsize,
                                    unsigned int* shiftptr_hi, unsigned int* shiftptr_lo, auto transfunc1,
                                    int shiftptr_size,
                                    const unsigned int* otherptr_hi, const unsigned int* otherptr_lo,
                                    auto transfunc2){

                const int max_errors = int(float(overlapsize) * maxErrorRate);

                int score = hammingDistanceWithShift(shift, overlapsize, max_errors,
                                                        shiftptr_hi,shiftptr_lo, transfunc1,
                                                        shiftptr_size,
                                                        otherptr_hi, otherptr_lo, transfunc2);

                score = (score < max_errors ?
                score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                : std::numeric_limits<int>::max()); // too many errors, discard

                if(score < bestScore){
                    bestScore = score;
                    bestShift = shift;
                }
            };

            std::copy(subject, subject + subjectints, shiftbuffer.begin());
            unsigned int* shiftbuffer_hi = shiftbuffer.data();
            unsigned int* shiftbuffer_lo = shiftbuffer.data() + subjectints / 2;

            for(int shift = 0; shift < subjectLength - minoverlap + 1; ++shift){
                const int overlapsize = std::min(subjectLength - shift, queryLength);
                handle_shift(shift, overlapsize,
                                shiftbuffer_hi, shiftbuffer_lo, identity,
                                subjectints,
                                queryBackup_hi, queryBackup_lo, identity);
            }

            std::copy(query, query + queryints, shiftbuffer.begin());
            shiftbuffer_hi = shiftbuffer.data();
            shiftbuffer_lo = shiftbuffer.data() + queryints / 2;

            for(int shift = -1; shift >= -queryLength + minoverlap; --shift){
                const int overlapsize = std::min(subjectLength, queryLength + shift);

                handle_shift(shift, overlapsize,
                                shiftbuffer_hi, shiftbuffer_lo, identity,
                                queryints,
                                subjectBackup_hi, subjectBackup_lo, identity);

            }

            AlignmentResult& alignmentresult = *destination;
            alignmentresult.isValid = (bestShift != -queryLength);

            const int queryoverlapbegin_incl = std::max(-bestShift, 0);
            const int queryoverlapend_excl = std::min(queryLength, subjectLength - bestShift);
            const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
            const int opnr = bestScore - totalbases + 2*overlapsize;

            alignmentresult.score = bestScore;
            alignmentresult.overlap = overlapsize;
            alignmentresult.shift = bestShift;
            alignmentresult.nOps = opnr;

            std::advance(destination, 1);
        }

        return destination;
    }




    template<class Iter>
    Iter
    cpu_multi_shifted_hamming_distance_popcount_new(Iter destinationbegin,
                                                const char* subject_charptr,
                                                int subjectLength,
                                                const std::vector<char>& querydata,
                                                const std::vector<int>& queryLengths,
                                                int max_sequence_bytes,
                                                int min_overlap,
                                                float maxErrorRate,
                                                float min_overlap_ratio) noexcept{

        assert(max_sequence_bytes % 4 == 0);

        if(queryLengths.size() == 0) return destinationbegin;

        auto getNumBytes = [] (int sequencelength){
            return sizeof(unsigned int) * getEncodedNumInts2BitHiLo(sequencelength);
        };

        auto popcount = [](auto i){return __builtin_popcount(i);};

        auto identity = [](auto i){return i;};

        auto hammingDistanceWithShift = [&](int shift, int overlapsize, int max_errors,
                                            unsigned int* shiftptr_hi, unsigned int* shiftptr_lo, auto transfunc1,
                                            int shiftptr_size,
                                            const unsigned int* otherptr_hi, const unsigned int* otherptr_lo,
                                            auto transfunc2){

            const int shiftamount = shift == 0 ? 0 : 1;

            shiftBitArrayLeftBy(shiftptr_hi, shiftptr_size / 2, shiftamount, transfunc1);
            shiftBitArrayLeftBy(shiftptr_lo, shiftptr_size / 2, shiftamount, transfunc1);

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


        const int nQueries = int(queryLengths.size());

        Iter destination = destinationbegin;

        const unsigned int* const subject = (const unsigned int*)subject_charptr;
        const int subjectints = getNumBytes(subjectLength) / sizeof(unsigned int);
        const int max_sequence_ints = max_sequence_bytes / sizeof(unsigned int);

        std::vector<unsigned int> shiftbuffer(max_sequence_ints);

        const unsigned int* const subjectBackup_hi = (const unsigned int*)(subject);
        const unsigned int* const subjectBackup_lo = ((const unsigned int*)subject) + subjectints / 2;

        for(int index = 0; index < nQueries; index++){
            const unsigned int* const query = (const unsigned int*)(querydata.data() + max_sequence_bytes * index);
            const int queryLength = queryLengths[index];

            const int queryints = getNumBytes(queryLength) / sizeof(unsigned int);
            const unsigned int* const queryBackup_hi = query;
            const unsigned int* const queryBackup_lo = query + queryints / 2;

            const int totalbases = subjectLength + queryLength;
            const int minoverlap = std::max(min_overlap, int(float(subjectLength) * min_overlap_ratio));


            int bestScore = totalbases; // score is number of mismatches
            int bestShift = -queryLength; // shift of query relative to subject. shift < 0 if query begins before subject

            auto handle_shift = [&](int shift, int overlapsize,
                                    unsigned int* shiftptr_hi, unsigned int* shiftptr_lo, auto transfunc1,
                                    int shiftptr_size,
                                    const unsigned int* otherptr_hi, const unsigned int* otherptr_lo,
                                    auto transfunc2){

                const int max_errors_excl = std::min(int(float(overlapsize) * maxErrorRate),
                                                bestScore - totalbases + 2*overlapsize);

                if(max_errors_excl > 0){

                    const int mismatches = hammingDistanceWithShift(shift, overlapsize, max_errors_excl,
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

            std::copy(subject, subject + subjectints, shiftbuffer.begin());
            unsigned int* shiftbuffer_hi = shiftbuffer.data();
            unsigned int* shiftbuffer_lo = shiftbuffer.data() + subjectints / 2;


            for(int shift = 0; shift < subjectLength - minoverlap + 1; ++shift){
                const int overlapsize = std::min(subjectLength - shift, queryLength);
                bool b = handle_shift(shift, overlapsize,
                                    shiftbuffer_hi, shiftbuffer_lo, identity,
                                    subjectints,
                                    queryBackup_hi, queryBackup_lo, identity);
                if(!b){
                    break;
                }
            }

            std::copy(query, query + queryints, shiftbuffer.begin());
            shiftbuffer_hi = shiftbuffer.data();
            shiftbuffer_lo = shiftbuffer.data() + queryints / 2;

            for(int shift = -1; shift >= -queryLength + minoverlap; --shift){
                const int overlapsize = std::min(subjectLength, queryLength + shift);

                bool b = handle_shift(shift, overlapsize,
                                    shiftbuffer_hi, shiftbuffer_lo, identity,
                                    queryints,
                                    subjectBackup_hi, subjectBackup_lo, identity);

                if(!b){
                    break;
                }
            }

            AlignmentResult& alignmentresult = *destination;
            alignmentresult.isValid = (bestShift != -queryLength);

            const int queryoverlapbegin_incl = std::max(-bestShift, 0);
            const int queryoverlapend_excl = std::min(queryLength, subjectLength - bestShift);
            const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
            const int opnr = bestScore - totalbases + 2*overlapsize;

            alignmentresult.score = bestScore;
            alignmentresult.overlap = overlapsize;
            alignmentresult.shift = bestShift;
            alignmentresult.nOps = opnr;

            std::advance(destination, 1);
        }

        return destination;
    }



    struct CpuAlignmentHandle{
        std::vector<unsigned int> shiftbuffer;
        std::vector<unsigned int> anchorConversionBuffer;
        std::vector<unsigned int> candidateConversionBuffer;
    };

    AlignmentResult
    cpuShiftedHammingDistancePopcount2BitHiLo(
            CpuAlignmentHandle& handle,
            const unsigned int* subjectHiLo,
            int subjectLength,
            const unsigned int* candidateHiLo,
            int candidateLength,
            int min_overlap,
            float maxErrorRate,
            float min_overlap_ratio) noexcept;

    

    template<class Iter>
    Iter
    cpuShiftedHammingDistancePopcount2Bit(
            CpuAlignmentHandle& handle,
            Iter destinationBegin,
            const unsigned int* subject2Bit,
            int subjectLength,
            const unsigned int* candidates2Bit,
            int candidatePitchInInts,
            const int* candidateLengths,
            int numCandidates,
            int min_overlap,
            float maxErrorRate,
            float min_overlap_ratio) noexcept{

        const int newsubjectInts = getEncodedNumInts2BitHiLo(subjectLength);

        handle.anchorConversionBuffer.resize(newsubjectInts);
        handle.candidateConversionBuffer.resize(candidatePitchInInts);

        convert2BitNewTo2BitHiLo(
            handle.anchorConversionBuffer.data(),
            subject2Bit,
            subjectLength
        );

        auto curIter = destinationBegin;

        for(int candidateIndex = 0; candidateIndex < numCandidates; candidateIndex++, ++curIter){
            const unsigned int* candidate2Bit = candidates2Bit + candidatePitchInInts * candidateIndex;
            const int candidateLength = candidateLengths[candidateIndex];

            convert2BitNewTo2BitHiLo(
                handle.candidateConversionBuffer.data(),
                candidate2Bit,
                candidateLength
            );

            *curIter = cpuShiftedHammingDistancePopcount2BitHiLo(
                            handle,
                            handle.anchorConversionBuffer.data(),
                            subjectLength,
                            handle.candidateConversionBuffer.data(),
                            candidateLength,
                            min_overlap,
                            maxErrorRate,
                            min_overlap_ratio
                        );
        }

        return curIter;
    }




    template<class Iter>
    Iter
    cpu_multi_shifted_hamming_distance_popcount_updated(Iter destinationbegin,
                                                const unsigned int* subjectHiLo,
                                                int subjectLength,
                                                const std::vector<unsigned int>& querydata,
                                                const std::vector<int>& queryLengths,
                                                int pitchIntsPerSequence,
                                                int min_overlap,
                                                float maxErrorRate,
                                                float min_overlap_ratio) noexcept{

        assert(subjectLength > 0);
        if(queryLengths.size() == 0) return destinationbegin;

        auto popcount = [](auto i){return __builtin_popcount(i);};

        auto identity = [](auto i){return i;};

        auto hammingDistanceWithShift = [&](int shift, int overlapsize, int max_errors,
                                            unsigned int* shiftptr_hi, unsigned int* shiftptr_lo, auto transfunc1,
                                            int shiftptr_size,
                                            const unsigned int* otherptr_hi, const unsigned int* otherptr_lo,
                                            auto transfunc2){

            const int shiftamount = shift == 0 ? 0 : 1;

            shiftBitArrayLeftBy(shiftptr_hi, shiftptr_size / 2, shiftamount, transfunc1);
            shiftBitArrayLeftBy(shiftptr_lo, shiftptr_size / 2, shiftamount, transfunc1);

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


        const int nQueries = int(queryLengths.size());

        Iter destination = destinationbegin;

        const unsigned int* const subject = subjectHiLo;
        const int subjectints = getEncodedNumInts2BitHiLo(subjectLength);

        std::vector<unsigned int> shiftbuffer(pitchIntsPerSequence);

        const unsigned int* const subjectBackup_hi = (const unsigned int*)(subject);
        const unsigned int* const subjectBackup_lo = ((const unsigned int*)subject) + subjectints / 2;

        for(int index = 0; index < nQueries; index++){
            const unsigned int* const query = querydata.data() + pitchIntsPerSequence * index;
            const int queryLength = queryLengths[index];

            const int queryints = getEncodedNumInts2BitHiLo(queryLength);
            const unsigned int* const queryBackup_hi = query;
            const unsigned int* const queryBackup_lo = query + queryints / 2;

            const int totalbases = subjectLength + queryLength;
            const int minoverlap = std::max(min_overlap, int(float(subjectLength) * min_overlap_ratio));


            int bestScore = totalbases; // score is number of mismatches
            int bestShift = -queryLength; // shift of query relative to subject. shift < 0 if query begins before subject

            auto handle_shift = [&](int shift, int overlapsize,
                                    unsigned int* shiftptr_hi, unsigned int* shiftptr_lo, auto transfunc1,
                                    int shiftptr_size,
                                    const unsigned int* otherptr_hi, const unsigned int* otherptr_lo,
                                    auto transfunc2){

                const int max_errors_excl = std::min(int(float(overlapsize) * maxErrorRate),
                                                bestScore - totalbases + 2*overlapsize);

                if(max_errors_excl > 0){

                    const int mismatches = hammingDistanceWithShift(shift, overlapsize, max_errors_excl,
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

            std::copy(subject, subject + subjectints, shiftbuffer.begin());
            unsigned int* shiftbuffer_hi = shiftbuffer.data();
            unsigned int* shiftbuffer_lo = shiftbuffer.data() + subjectints / 2;


            for(int shift = 0; shift < subjectLength - minoverlap + 1; ++shift){
                const int overlapsize = std::min(subjectLength - shift, queryLength);
                bool b = handle_shift(shift, overlapsize,
                                    shiftbuffer_hi, shiftbuffer_lo, identity,
                                    subjectints,
                                    queryBackup_hi, queryBackup_lo, identity);
                if(!b){
                    break;
                }
            }

            std::copy(query, query + queryints, shiftbuffer.begin());
            shiftbuffer_hi = shiftbuffer.data();
            shiftbuffer_lo = shiftbuffer.data() + queryints / 2;

            for(int shift = -1; shift >= -queryLength + minoverlap; --shift){
                const int overlapsize = std::min(subjectLength, queryLength + shift);

                bool b = handle_shift(shift, overlapsize,
                                    shiftbuffer_hi, shiftbuffer_lo, identity,
                                    queryints,
                                    subjectBackup_hi, subjectBackup_lo, identity);

                if(!b){
                    break;
                }
            }

            AlignmentResult& alignmentresult = *destination;
            alignmentresult.isValid = (bestShift != -queryLength);

            const int queryoverlapbegin_incl = std::max(-bestShift, 0);
            const int queryoverlapend_excl = std::min(queryLength, subjectLength - bestShift);
            const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
            const int opnr = bestScore - totalbases + 2*overlapsize;

            alignmentresult.score = bestScore;
            alignmentresult.overlap = overlapsize;
            alignmentresult.shift = bestShift;
            alignmentresult.nOps = opnr;

            std::advance(destination, 1);
        }

        return destination;
    }




    AlignmentResult
    cpu_shifted_hamming_distance_popcount(const char* subject,
                                int subjectLength,
                                const char* query,
                                int queryLength,
                                int min_overlap,
                                float maxErrorRate,
                                float min_overlap_ratio) noexcept;


    std::vector<AlignmentResult>
    cpu_multi_shifted_hamming_distance_popcount(const char* subject_charptr,
                                int subjectLength,
                                const std::vector<char>& querydata,
                                const std::vector<int>& queryLengths,
                                int max_sequence_bytes,
                                int min_overlap,
                                float maxErrorRate,
                                float min_overlap_ratio) noexcept;


} //namespace shd

using SHDResult = shd::AlignmentResult;


} //namespace care::cpu

} //namespace care




#endif
