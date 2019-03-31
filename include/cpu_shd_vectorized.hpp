#ifndef CARE_CPU_SHD_VECTORIZED
#define CARE_CPU_SHD_VECTORIZED

#include <config.hpp>

#include <xmmintrin.h>
#include <cassert>

#if 0

/*
    assumes all candidate length are subject length are identical
*/
    template<class B>
    std::vector<AlignmentResult>
    cpu_multi_shifted_hamming_distance_popcount_sse(const char* subject_charptr,
                                const std::vector<char>& querydata,
                                const int sequencelength,
                                const int max_sequence_bytes,
                                int min_overlap,
                                float maxErrorRate,
                                float min_overlap_ratio,
                                B getNumBytes) noexcept{

        assert(max_sequence_bytes % 4 == 0);

        if(queryLengths.size() == 0) return {};

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

        constexpr int intlanes = 4;

        const int nQueries = int(queryLengths.size());
        const int sequenceBytes = getNumBytes(sequencelength);
        const int sequenceInts = sequenceBytes / sizeof(unsigned int);
        const unsigned int* const subject = (const unsigned int*)subject_charptr;
        const int max_sequence_ints = max_sequence_bytes / sizeof(unsigned int);

        std::vector<__m128> subjectdatavec(sequenceInts); //subjectdatavec[i] contains the i-th int of subject in each lane
        std::vector<__m128> candidatedatavec(sequenceInts); //candidatedatavec[i] contains the i-th int of the lane-th candidate


        std::vector<AlignmentResult> results(queryLengths.size());



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
                                                max_errors);

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

                shiftEncodedBasesLeftBy(shiftbuffer_hi, queryints / 2, 1);
                shiftEncodedBasesLeftBy(shiftbuffer_lo, queryints / 2, 1);

                int score = hammingdistanceHiLo(subjectBackup_hi,
                                                subjectBackup_lo,
                                                shiftbuffer_hi,
                                                shiftbuffer_lo,
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
            std::copy(subject, subject + subjectints, shiftbuffer.begin());
            shiftbuffer_hi = shiftbuffer.data();
            shiftbuffer_lo = shiftbuffer.data() + subjectints / 2;

            for(int shift = 1; shift < subjectLength - minoverlap + 1; ++shift){
                const int overlapsize = std::min(subjectLength - shift, queryLength);
                const int max_errors = int(float(overlapsize) * maxErrorRate);

                shiftEncodedBasesLeftBy(shiftbuffer_hi, subjectints / 2, 1);
                shiftEncodedBasesLeftBy(shiftbuffer_lo, subjectints / 2, 1);

                int score = hammingdistanceHiLo(shiftbuffer_hi,
                                                shiftbuffer_lo,
                                                queryBackup_hi,
                                                queryBackup_lo,
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

#endif

#endif
