#ifndef CARE_CPU_ALIGNMENT_HPP
#define CARE_CPU_ALIGNMENT_HPP


#include "sequence.hpp"

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
                                int subjectlength,
                                const char* query,
                                int querylength,
                                int min_overlap,
                                float maxErrorRate,
                                float min_overlap_ratio,
                                Accessor getChar)  noexcept{

        const int totalbases = subjectlength + querylength;
        const int minoverlap = std::max(min_overlap, int(float(subjectlength) * min_overlap_ratio));
        int bestScore = totalbases; // score is number of mismatches
        int bestShift = -querylength; // shift of query relative to subject. shift < 0 if query begins before subject

        for(int shift = -querylength + minoverlap; shift < subjectlength - minoverlap + 1; shift++){
            const int overlapsize = std::min(querylength, subjectlength - shift) - std::max(-shift, 0);
            const int max_errors = int(float(overlapsize) * maxErrorRate);
            int score = 0;

            for(int j = std::max(-shift, 0); j < std::min(querylength, subjectlength - shift) && score < max_errors; j++){
                score += getChar(subject, subjectlength, j + shift) != getChar(query, querylength, j);
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
        result.isValid = (bestShift != -querylength);

        const int queryoverlapbegin_incl = std::max(-bestShift, 0);
        const int queryoverlapend_excl = std::min(querylength, subjectlength - bestShift);
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
                                int subjectlength,
                                const char* query,
                                int querylength,
                                int min_overlap,
                                float maxErrorRate,
                                float min_overlap_ratio,
                                B getNumBytes) noexcept{

        auto shiftEncodedBasesLeftBy = [](unsigned int* array, int size, int shiftamount){
            const int completeInts = shiftamount / (8 * sizeof(unsigned int));

            for(int i = 0; i < size - completeInts; ++i){
                array[i] = array[completeInts + i];
            }

            for(int i = size - completeInts; i < size; ++i){
                array[i] = 0;
            }

            shiftamount -= completeInts * 8 * sizeof(unsigned int);

            assert(shiftamount < int(8 * sizeof(unsigned int)));

            for(int i = 0; i < size - completeInts - 1; ++i){
                array[i] = (array[i] >> shiftamount) | (array[i+1] << (8 * sizeof(unsigned int) - shiftamount));
            }
            array[size - completeInts - 1] >>= shiftamount;
        };

        auto hammingdistanceHiLo = [](const unsigned int* lhi,
                                        const unsigned int* llo,
                                        const unsigned int* rhi,
                                        const unsigned int* rlo,
                                        int lhi_bitcount,
                                        int rhi_bitcount,
                                        int max_errors){

        	const int overlap_bitcount = lhi_bitcount < rhi_bitcount ? lhi_bitcount : rhi_bitcount;
            const int partitions = SDIV(overlap_bitcount, (8 * sizeof(unsigned int)));
            const int remaining_bitcount = partitions * sizeof(unsigned int) * 8 - overlap_bitcount;

        	int result = 0;

        	for(int i = 0; i < partitions - 1 && result < max_errors; i++){
        		const int hixor = lhi[i] ^ rhi[i];
        		const int loxor = llo[i] ^ rlo[i];
        		const int bits = hixor | loxor;
        		result += __builtin_popcount(bits);
        	}

            if(result >= max_errors)
                return result;

            //in last partition, we ignore the bits which are not part of the overlap
            const unsigned int mask = remaining_bitcount == 0 ? 0xFFFFFFFF : 0xFFFFFFFF >> (remaining_bitcount);
            const int hixor = lhi[partitions - 1] ^ rhi[partitions - 1];
            const int loxor = llo[partitions - 1] ^ rlo[partitions - 1];
            const int bits = hixor | loxor;
            result += __builtin_popcount(bits & mask);

            return result;
        };

        const int subjectbytes = getNumBytes(subjectlength);
        const int querybytes = getNumBytes(querylength);
        const int totalbases = subjectlength + querylength;
        const int minoverlap = std::max(min_overlap, int(float(subjectlength) * min_overlap_ratio));

        int bestScore = totalbases; // score is number of mismatches
        int bestShift = -querylength; // shift of query relative to subject. shift < 0 if query begins before subject

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
            for(int shift = -querylength + minoverlap; shift < subjectlength - minoverlap; shift++)

            This loop is split into 3 parts. shift = 0, shift < 0 and shift > 0
        */

        //shift == 0
        {
            const int shift = 0;
            const int overlapsize = std::min(subjectlength, querylength);
            const int max_errors = int(float(overlapsize) * maxErrorRate);

            int score = hammingdistanceHiLo(subjectdata_hi,
                                subjectdata_lo,
                                querydata_hi,
                                querydata_lo,
                                overlapsize,
                                overlapsize,
                                max_errors);

                score = (score < max_errors ?
                        score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                        : std::numeric_limits<int>::max()); // too many errors, discard

            if(score < bestScore){
                bestScore = score;
                bestShift = shift;
            }
        }

        // shift < 0
        for(int shift = -1; shift >= -querylength + minoverlap; --shift){
            const int overlapsize = std::min(subjectlength, querylength + shift);
            const int max_errors = int(float(overlapsize) * maxErrorRate);

            shiftEncodedBasesLeftBy((unsigned int*)querydata_hi, querybytes / 2 / sizeof(unsigned int), 1);
            shiftEncodedBasesLeftBy((unsigned int*)querydata_lo, querybytes / 2 / sizeof(unsigned int), 1);

            int score = hammingdistanceHiLo(subjectdata_hi,
                                subjectdata_lo,
                                querydata_hi,
                                querydata_lo,
                                overlapsize,
                                overlapsize,
                                max_errors);

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

        for(int shift = 1; shift < subjectlength - minoverlap + 1; ++shift){
            const int overlapsize = std::min(subjectlength - shift, querylength);
            const int max_errors = int(float(overlapsize) * maxErrorRate);

            shiftEncodedBasesLeftBy((unsigned int*)subjectdata_hi, subjectbytes / 2 / sizeof(unsigned int), 1);
            shiftEncodedBasesLeftBy((unsigned int*)subjectdata_lo, subjectbytes / 2 / sizeof(unsigned int), 1);

            int score = hammingdistanceHiLo(subjectdata_hi,
                                subjectdata_lo,
                                querydata_hi,
                                querydata_lo,
                                overlapsize,
                                overlapsize,
                                max_errors);

                score = (score < max_errors ?
                        score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                        : std::numeric_limits<int>::max()); // too many errors, discard

            if(score < bestScore){
                bestScore = score;
                bestShift = shift;
            }
        }

        AlignmentResult result;
        result.isValid = (bestShift != -querylength);

        const int queryoverlapbegin_incl = std::max(-bestShift, 0);
        const int queryoverlapend_excl = std::min(querylength, subjectlength - bestShift);
        const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
        const int opnr = bestScore - totalbases + 2*overlapsize;

        result.score = bestScore;
        result.overlap = overlapsize;
        result.shift = bestShift;
        result.nOps = opnr;

        return result;
    }

} //namespace shd

using SHDResult = shd::AlignmentResult;

template<class Sequence_t>
struct CPUShiftedHammingDistanceChooser;

template<>
struct CPUShiftedHammingDistanceChooser<SequenceString>{
    static SHDResult cpu_shifted_hamming_distance(const char* subject,
                                            int subjectlength,
    										const char* query,
    										int querylength,
                                            int min_overlap,
                                            float maxErrorRate,
                                            float min_overlap_ratio){

        auto accessor = [] (const char* data, int length, int index){
            return SequenceString::get(data, length, index);
        };

        return shd::cpu_shifted_hamming_distance(subject, subjectlength, query, querylength,
                            min_overlap, maxErrorRate, min_overlap_ratio, accessor);

    }
};

template<>
struct CPUShiftedHammingDistanceChooser<Sequence2Bit>{
    static SHDResult cpu_shifted_hamming_distance(const char* subject,
                                            int subjectlength,
    										const char* query,
    										int querylength,
                                            int min_overlap,
                                            float maxErrorRate,
                                            float min_overlap_ratio){

        auto accessor = [] (const char* data, int length, int index){
            return Sequence2Bit::get(data, length, index);
        };

        return shd::cpu_shifted_hamming_distance(subject, subjectlength, query, querylength,
                            min_overlap, maxErrorRate, min_overlap_ratio, accessor);

    }
};

template<>
struct CPUShiftedHammingDistanceChooser<Sequence2BitHiLo>{
    static SHDResult cpu_shifted_hamming_distance(const char* subject,
                                            int subjectlength,
    										const char* query,
    										int querylength,
                                            int min_overlap,
                                            float maxErrorRate,
                                            float min_overlap_ratio){

        auto getNumBytes = [] (int nbases){
            return Sequence2BitHiLo::getNumBytes(nbases);
        };

        return shd::cpu_shifted_hamming_distance_popcount(subject, subjectlength, query, querylength,
                            min_overlap, maxErrorRate, min_overlap_ratio, getNumBytes);

    }
};

} //namespace care::cpu

} //namespace care




#endif
