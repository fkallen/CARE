#ifndef CARE_CPU_ALIGNMENT_HPP
#define CARE_CPU_ALIGNMENT_HPP


#include <sequence.hpp>

#include <config.hpp>
#include <shiftedhammingdistance_common.hpp>

#include <vector>

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

    template<class B>
    AlignmentResult
    cpu_shifted_hamming_distance_popcount(const char* subject,
                                int subjectLength,
                                const char* query,
                                int queryLength,
                                int min_overlap,
                                float maxErrorRate,
                                float min_overlap_ratio,
                                B getNumBytes) noexcept{

        auto popcount = [](auto i){return __builtin_popcount(i);};

        auto identity = [](auto i){return i;};

        const int subjectbytes = getNumBytes(subjectLength);
        const int querybytes = getNumBytes(queryLength);
        const int totalbases = subjectLength + queryLength;
        const int minoverlap = std::max(min_overlap, int(float(subjectLength) * min_overlap_ratio));

        int bestScore = totalbases; // score is number of mismatches
        int bestShift = -queryLength; // shift of query relative to subject. shift < 0 if query begins before subject

        std::vector<char> subjectdata(subjectbytes);
        std::vector<char> querydata(querybytes);

        unsigned int* subjectdata_hi = (unsigned int*)subjectdata.data();
        unsigned int* subjectdata_lo = (unsigned int*)(subjectdata.data() + subjectbytes / 2);
        unsigned int* querydata_hi = (unsigned int*)querydata.data();
        unsigned int* querydata_lo = (unsigned int*)(querydata.data() + querybytes / 2);

        std::copy(subject, subject + subjectbytes, subjectdata.begin());
        std::copy(query, query + querybytes, querydata.begin());

        /*
            The goal is to calculate the hamming distance for each shift in
            for(int shift = -queryLength + minoverlap; shift < subjectLength - minoverlap; shift++)

            This loop is split into 3 parts. shift = 0, shift < 0 and shift > 0
        */

        //shift == 0
        {
            const int shift = 0;
            const int overlapsize = std::min(subjectLength, queryLength);
            const int max_errors = int(float(overlapsize) * maxErrorRate);

            int score = hammingdistanceHiLo(subjectdata_hi,
                                subjectdata_lo,
                                querydata_hi,
                                querydata_lo,
                                overlapsize,
                                overlapsize,
                                max_errors,
                                identity,
                                identity,
                                popcount);

                score = (score < max_errors ?
                        score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                        : std::numeric_limits<int>::max()); // too many errors, discard

            if(score < bestScore){
                bestScore = score;
                bestShift = shift;
            }
        }

        // shift < 0
        for(int shift = -1; shift >= -queryLength + minoverlap; --shift){
            const int overlapsize = std::min(subjectLength, queryLength + shift);
            const int max_errors = int(float(overlapsize) * maxErrorRate);

            shiftBitArrayLeftBy((unsigned int*)querydata_hi, querybytes / 2 / sizeof(unsigned int), 1, identity);
            shiftBitArrayLeftBy((unsigned int*)querydata_lo, querybytes / 2 / sizeof(unsigned int), 1, identity);

            int score = hammingdistanceHiLo(subjectdata_hi,
                                subjectdata_lo,
                                querydata_hi,
                                querydata_lo,
                                overlapsize,
                                overlapsize,
                                max_errors,
                                identity,
                                identity,
                                popcount);

                score = (score < max_errors ?
                        score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                        : std::numeric_limits<int>::max()); // too many errors, discard

            if(score < bestScore){
                bestScore = score;
                bestShift = shift;
            }
        }

        //shift > 0

        //load query again from memory since it has been modified by calculations with shift < 0
        std::copy(query, query + querybytes, querydata.begin());

        for(int shift = 1; shift < subjectLength - minoverlap + 1; ++shift){
            const int overlapsize = std::min(subjectLength - shift, queryLength);
            const int max_errors = int(float(overlapsize) * maxErrorRate);

            shiftBitArrayLeftBy((unsigned int*)subjectdata_hi, subjectbytes / 2 / sizeof(unsigned int), 1, identity);
            shiftBitArrayLeftBy((unsigned int*)subjectdata_lo, subjectbytes / 2 / sizeof(unsigned int), 1, identity);

            int score = hammingdistanceHiLo(subjectdata_hi,
                                subjectdata_lo,
                                querydata_hi,
                                querydata_lo,
                                overlapsize,
                                overlapsize,
                                max_errors,
                                identity,
                                identity,
                                popcount);

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







    template<class B>
    std::vector<AlignmentResult>
    cpu_multi_shifted_hamming_distance_popcount(const char* subject_charptr,
                                int subjectLength,
                                const std::vector<char>& querydata,
                                const std::vector<int>& queryLengths,
                                int max_sequence_bytes,
                                int min_overlap,
                                float maxErrorRate,
                                float min_overlap_ratio,
                                B getNumBytes) noexcept{

        assert(max_sequence_bytes % 4 == 0);

        if(queryLengths.size() == 0) return {};

        auto popcount = [](auto i){return __builtin_popcount(i);};

        auto identity = [](auto i){return i;};


        const int nQueries = int(queryLengths.size());

        std::vector<AlignmentResult> results(queryLengths.size());

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

            /*
                The goal is to calculate the hamming distance for each shift in
                for(int shift = -queryLength + minoverlap; shift < subjectLength - minoverlap; shift++)

                This loop is split into 3 parts. shift = 0, shift < 0 and shift > 0
            */

            //shift == 0
            {
                const int shift = 0;
                const int overlapsize = std::min(subjectLength, queryLength);
                const int max_errors = int(float(overlapsize) * maxErrorRate);

                int score = hammingdistanceHiLo(subjectBackup_hi,
                                                subjectBackup_lo,
                                                queryBackup_hi,
                                                queryBackup_lo,
                                                overlapsize,
                                                overlapsize,
                                                max_errors,
                                                identity,
                                                identity,
                                                popcount);

                score = (score < max_errors ?
                        score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                        : std::numeric_limits<int>::max()); // too many errors, discard

                if(score < bestScore){
                    bestScore = score;
                    bestShift = shift;
                }
            }

            std::copy(query, query + queryints, shiftbuffer.begin());
            unsigned int* shiftbuffer_hi = shiftbuffer.data();
            unsigned int* shiftbuffer_lo = shiftbuffer.data() + queryints / 2;

            // shift < 0
            for(int shift = -1; shift >= -queryLength + minoverlap; --shift){
                const int overlapsize = std::min(subjectLength, queryLength + shift);
                const int max_errors = int(float(overlapsize) * maxErrorRate);

                shiftBitArrayLeftBy(shiftbuffer_hi, queryints / 2, 1, identity);
                shiftBitArrayLeftBy(shiftbuffer_lo, queryints / 2, 1, identity);

                int score = hammingdistanceHiLo(subjectBackup_hi,
                                                subjectBackup_lo,
                                                shiftbuffer_hi,
                                                shiftbuffer_lo,
                                                overlapsize,
                                                overlapsize,
                                                max_errors,
                                                identity,
                                                identity,
                                                popcount);

                score = (score < max_errors ?
                        score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                        : std::numeric_limits<int>::max()); // too many errors, discard

                if(score < bestScore){
                    bestScore = score;
                    bestShift = shift;
                }
            }

            //shift > 0
            std::copy(subject, subject + subjectints, shiftbuffer.begin());
            shiftbuffer_hi = shiftbuffer.data();
            shiftbuffer_lo = shiftbuffer.data() + subjectints / 2;

            for(int shift = 1; shift < subjectLength - minoverlap + 1; ++shift){
                const int overlapsize = std::min(subjectLength - shift, queryLength);
                const int max_errors = int(float(overlapsize) * maxErrorRate);

                shiftBitArrayLeftBy(shiftbuffer_hi, subjectints / 2, 1, identity);
                shiftBitArrayLeftBy(shiftbuffer_lo, subjectints / 2, 1, identity);

                int score = hammingdistanceHiLo(shiftbuffer_hi,
                                                shiftbuffer_lo,
                                                queryBackup_hi,
                                                queryBackup_lo,
                                                overlapsize,
                                                overlapsize,
                                                max_errors,
                                                identity,
                                                identity,
                                                popcount);

                score = (score < max_errors ?
                        score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                        : std::numeric_limits<int>::max()); // too many errors, discard

                if(score < bestScore){
                    bestScore = score;
                    bestShift = shift;
                }
            }

            AlignmentResult& alignmentresult = results[index];
            alignmentresult.isValid = (bestShift != -queryLength);

            const int queryoverlapbegin_incl = std::max(-bestShift, 0);
            const int queryoverlapend_excl = std::min(queryLength, subjectLength - bestShift);
            const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
            const int opnr = bestScore - totalbases + 2*overlapsize;

            alignmentresult.score = bestScore;
            alignmentresult.overlap = overlapsize;
            alignmentresult.shift = bestShift;
            alignmentresult.nOps = opnr;
        }

        return results;
    }


} //namespace shd

using SHDResult = shd::AlignmentResult;

template<class Sequence_t>
struct CPUShiftedHammingDistanceChooser;

template<>
struct CPUShiftedHammingDistanceChooser<SequenceString>{
    static SHDResult cpu_shifted_hamming_distance(const char* subject,
                                            int subjectLength,
    										const char* query,
    										int queryLength,
                                            int min_overlap,
                                            float maxErrorRate,
                                            float min_overlap_ratio){

        auto accessor = [] (const char* data, int length, int index){
            return SequenceString::get(data, length, index);
        };

        return shd::cpu_shifted_hamming_distance(subject, subjectLength, query, queryLength,
                            min_overlap, maxErrorRate, min_overlap_ratio, accessor);

    }
};

template<>
struct CPUShiftedHammingDistanceChooser<Sequence2Bit>{
    static SHDResult cpu_shifted_hamming_distance(const char* subject,
                                            int subjectLength,
    										const char* query,
    										int queryLength,
                                            int min_overlap,
                                            float maxErrorRate,
                                            float min_overlap_ratio){

        auto accessor = [] (const char* data, int length, int index){
            return Sequence2Bit::get(data, length, index);
        };

        return shd::cpu_shifted_hamming_distance(subject, subjectLength, query, queryLength,
                            min_overlap, maxErrorRate, min_overlap_ratio, accessor);

    }
};

template<>
struct CPUShiftedHammingDistanceChooser<Sequence2BitHiLo>{
    static SHDResult cpu_shifted_hamming_distance(const char* subject,
                                            int subjectLength,
    										const char* query,
    										int queryLength,
                                            int min_overlap,
                                            float maxErrorRate,
                                            float min_overlap_ratio){

        auto getNumBytes = [] (int nbases){
            return Sequence2BitHiLo::getNumBytes(nbases);
        };

        return shd::cpu_shifted_hamming_distance_popcount(subject, subjectLength, query, queryLength,
                            min_overlap, maxErrorRate, min_overlap_ratio, getNumBytes);

    }
};

} //namespace care::cpu

} //namespace care




#endif
