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

        return shd::cpu_shifted_hamming_distance_popcount(subject, subjectLength, query, queryLength,
                            min_overlap, maxErrorRate, min_overlap_ratio);

    }
};

} //namespace care::cpu

} //namespace care




#endif
