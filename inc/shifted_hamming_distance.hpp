#ifndef SHIFTED_HAMMING_DISTANCE_HPP
#define SHIFTED_HAMMING_DISTANCE_HPP

#include "cudareduce.cuh"
#include "hpc_helpers.cuh"
#include "bestalignment.hpp"
#include "util.hpp"
#include "sequence.hpp"

//#include "util.hpp"

#include <cstdint>
#include <algorithm>
#include <vector>
#include <bitset>

#if __CUDACC_VER_MAJOR__ >= 9
#include <cooperative_groups.h>
using namespace cooperative_groups;
#endif

namespace shd{

struct AlignmentResult{
	int score;
	int subject_begin_incl;
	int query_begin_incl;
	int overlap;
	int shift;
	int nOps; //edit distance / number of operations
	bool isNormalized;
	bool isValid;

    HOSTDEVICEQUALIFIER
    bool operator==(const AlignmentResult& rhs) const noexcept;
    HOSTDEVICEQUALIFIER
    bool operator!=(const AlignmentResult& rhs) const noexcept;
    HOSTDEVICEQUALIFIER
    int get_score() const noexcept;
    HOSTDEVICEQUALIFIER
    int get_subject_begin_incl() const noexcept;
    HOSTDEVICEQUALIFIER
    int get_query_begin_incl() const noexcept;
    HOSTDEVICEQUALIFIER
    int get_overlap() const noexcept;
    HOSTDEVICEQUALIFIER
    int get_shift() const noexcept;
    HOSTDEVICEQUALIFIER
    int get_nOps() const noexcept;
    HOSTDEVICEQUALIFIER
    bool get_isNormalized() const noexcept;
    HOSTDEVICEQUALIFIER
    bool get_isValid() const noexcept;
    HOSTDEVICEQUALIFIER
    int& get_score() noexcept;
    HOSTDEVICEQUALIFIER
    int& get_subject_begin_incl() noexcept;
    HOSTDEVICEQUALIFIER
    int& get_query_begin_incl() noexcept;
    HOSTDEVICEQUALIFIER
    int& get_overlap() noexcept;
    HOSTDEVICEQUALIFIER
    int& get_shift() noexcept;
    HOSTDEVICEQUALIFIER
    int& get_nOps() noexcept;
    HOSTDEVICEQUALIFIER
    bool& get_isNormalized() noexcept;
    HOSTDEVICEQUALIFIER
    bool& get_isValid() noexcept;
};

using Result_t = AlignmentResult;

struct SHDdata{

    void* hostptr;
    void* deviceptr;
    std::size_t allocatedMem;

    /*std::size_t subjectsOffset = 0;
    std::size_t subjectLenghtsOffset = 0;
    std::size_t NqueriesPrefixSumOffset = 0;
    std::size_t queriesOffset = 0;
    std::size_t queryLengthsOffset = 0;
    std::size_t resultsOffset = 0;*/
    std::size_t transfersizeH2D = 0;
    std::size_t transfersizeD2H = 0;

	Result_t* d_results = nullptr;
	char* d_subjectsdata = nullptr;
	char* d_queriesdata = nullptr;
	int* d_subjectlengths = nullptr;
	int* d_querylengths = nullptr;
    int* d_NqueriesPrefixSum = nullptr;
    BestAlignment_t* d_bestAlignmentFlags = nullptr;

	Result_t* h_results = nullptr;
	char* h_subjectsdata = nullptr;
	char* h_queriesdata = nullptr;
	int* h_subjectlengths = nullptr;
	int* h_querylengths = nullptr;
    int* h_NqueriesPrefixSum = nullptr;
    BestAlignment_t* h_bestAlignmentFlags = nullptr;

#ifdef __NVCC__
    static constexpr int n_streams = 1;
    cudaStream_t streams[n_streams];
#endif

	int deviceId = -1;
	std::size_t sequencepitch = 0;
	int max_sequence_length = 0;
	int max_sequence_bytes = 0;
    int min_sequence_length = 0;
    int min_sequence_bytes = 0;
	int n_subjects = 0;
	int n_queries = 0;
    int n_results = 0;
	int max_n_subjects = 0;
	int max_n_queries = 0;

    // if number of alignments to calculate is >= gpuThreshold, use GPU.
    int gpuThreshold = 0;

	void resize(int n_sub, int n_quer);
    void resize(int n_sub, int n_quer, int n_results, double factor = 1.2);

};

//init buffers
void cuda_init_SHDdata(SHDdata& data, int deviceId,
                        int max_sequence_length,
                        int max_sequence_bytes,
                        int gpuThreshold);

//free buffers
void cuda_cleanup_SHDdata(SHDdata& data);

/*
    CPU alignment
*/
template<class Accessor>
Result_t
cpu_shifted_hamming_distance(const char* subject,
                            int subjectlength,
                            const char* query,
                            int querylength,
                            int min_overlap,
                            double maxErrorRate,
                            double min_overlap_ratio,
                            Accessor getChar)  noexcept{

    const int totalbases = subjectlength + querylength;
    const int minoverlap = std::max(min_overlap, int(double(subjectlength) * min_overlap_ratio));
    int bestScore = totalbases; // score is number of mismatches
    int bestShift = -querylength; // shift of query relative to subject. shift < 0 if query begins before subject

    for(int shift = -querylength + minoverlap; shift < subjectlength - minoverlap; shift++){
        const int overlapsize = std::min(querylength, subjectlength - shift) - std::max(-shift, 0);
        const int max_errors = int(double(overlapsize) * maxErrorRate);
        int score = 0;

        for(int j = std::max(-shift, 0); j < std::min(querylength, subjectlength - shift) && score < max_errors; j++){
            score += getChar(subject, subjectlength, j + shift) != getChar(query, querylength, j);
        }

        #if 1
            score = (score < max_errors ?
                    score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                    : std::numeric_limits<int>::max()); // too many errors, discard
        #else
        	score += totalbases - 2*overlapsize;
        #endif

        if(score < bestScore){
            bestScore = score;
            bestShift = shift;
        }
    }

    Result_t result;
    result.isValid = (bestShift != -querylength);

    const int queryoverlapbegin_incl = std::max(-bestShift, 0);
    const int queryoverlapend_excl = std::min(querylength, subjectlength - bestShift);
    const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
    const int opnr = bestScore - totalbases + 2*overlapsize;

    result.score = bestScore;
    result.subject_begin_incl = std::max(0, bestShift);
    result.query_begin_incl = queryoverlapbegin_incl;
    result.overlap = overlapsize;
    result.shift = bestShift;
    result.nOps = opnr;
    result.isNormalized = false;

    return result;
}

#if 1
template<class B>
Result_t
cpu_shifted_hamming_distance_popcount(const char* subject,
                            int subjectlength,
                            const char* query,
                            int querylength,
                            int min_overlap,
                            double maxErrorRate,
                            double min_overlap_ratio,
                            B getNumBytes) noexcept{

    auto shiftBitsLeftBy = [](unsigned char* array, int bytes, int shiftamount){
    	constexpr int maxshiftPerIter = 7;
    	const int iters = SDIV(shiftamount, maxshiftPerIter);
    	for(int iter = 0; iter < iters-1; ++iter){
    		for(int i = 0; i < bytes - 1; ++i){
    			array[i] = (array[i] << maxshiftPerIter) | (array[i+1] >> (8 - maxshiftPerIter));
    		}
    		array[bytes - 1] <<= maxshiftPerIter;

    		shiftamount -= maxshiftPerIter;
    	}

    	for(int i = 0; i < bytes - 1; ++i){
    		array[i] = (array[i] << shiftamount) | (array[i+1] >> (8 - shiftamount));
    	}
    	array[bytes - 1] <<= shiftamount;
    };

    auto hammingdistanceHiLo = [](const int* lhi,
                                    const int* llo,
                                    const int* rhi,
                                    const int* rlo,
                                    int lhi_bitcount,
                                    int rhi_bitcount,
                                    int max_errors){

    	const int overlap_bitcount = lhi_bitcount < rhi_bitcount ? lhi_bitcount : rhi_bitcount;
        const int partitions = (overlap_bitcount / 8) / sizeof(int);

    	int result = 0;

    	for(int i = 0; i < partitions && result < max_errors; i++){
    		const int hixor = lhi[i] ^ rhi[i];
    		const int loxor = llo[i] ^ rlo[i];
    		const int bits = hixor | loxor;
    		result += __builtin_popcount(bits);
    	}

        if(result >= max_errors)
            return result;

        int remaining_bitcount = overlap_bitcount - partitions * sizeof(int) * 8;
    	if(remaining_bitcount != 0){
    		const int charpartitions = (remaining_bitcount / 8) / sizeof(std::uint8_t);

    		const std::uint8_t* const lhichar = (const std::uint8_t*)(lhi + partitions);
    		const std::uint8_t* const llochar = (const std::uint8_t*)(llo + partitions);
    		const std::uint8_t* const rhichar = (const std::uint8_t*)(rhi + partitions);
    		const std::uint8_t* const rlochar = (const std::uint8_t*)(rlo + partitions);

    		for(int i = 0; i < charpartitions; i++){
    			const std::uint8_t hixorchar = lhichar[i] ^ rhichar[i];
    			const std::uint8_t loxorchar = llochar[i] ^ rlochar[i];
    			const std::uint8_t bitschar = hixorchar | loxorchar;
    			result += __builtin_popcount(bitschar);
    		}

    		remaining_bitcount = remaining_bitcount - charpartitions * sizeof(std::uint8_t) * 8;

    		if(remaining_bitcount != 0){
    			std::uint8_t mask = 0xFF << (sizeof(std::uint8_t)*8 - remaining_bitcount);
    			const std::uint8_t hixorchar2 = lhichar[charpartitions] ^ rhichar[charpartitions];
    			const std::uint8_t loxorchar2 = llochar[charpartitions] ^ rlochar[charpartitions];
    			const std::uint8_t bitschar2 = hixorchar2 | loxorchar2;
    			result += __builtin_popcount(bitschar2 & mask);
    		}

    	}

        return result;
    };

    const int subjectbytes = getNumBytes(subjectlength);
    const int querybytes = getNumBytes(querylength);
    const int totalbases = subjectlength + querylength;
    const int minoverlap = std::max(min_overlap, int(double(subjectlength) * min_overlap_ratio));

    int bestScore = totalbases; // score is number of mismatches
    int bestShift = -querylength; // shift of query relative to subject. shift < 0 if query begins before subject

    std::vector<char> subjectdata(subjectbytes);
    std::vector<char> querydata(querybytes);

    int* subjectdata_hi = (int*)subjectdata.data();
    int* subjectdata_lo = (int*)(subjectdata.data() + subjectbytes / 2);
    int* querydata_hi = (int*)querydata.data();
    int* querydata_lo = (int*)(querydata.data() + querybytes / 2);

#if 1

    std::copy(subject, subject + subjectbytes, subjectdata.begin());
    std::copy(query, query + querybytes, querydata.begin());

    //static int counter = 0;
    //counter++;

    /*if(counter == 3 || counter == 4){
        auto b2c = [](char c){
            switch(c){
                case 0x00: return 'A';
                case 0x01: return 'C';
                case 0x02: return 'G';
                case 0x03: return 'T';
                default: return 'F';
            }
        };
        printf("subjectbases: %d, querybases %d\n", subjectlength, querylength);
        for(int i = 0; i < subjectlength; i++){
            printf("%c", b2c(care::Sequence2BitHiLo::get(subjectdata.data(), subjectlength, i)));
        }
        printf("\n");

        for(int i = 0; i < querylength; i++){
            printf("%c", b2c(care::Sequence2BitHiLo::get(querydata.data(), querylength, i)));
        }
        printf("\n");
    }*/

    /*
        The goal is to calculate the hamming distance for each shift in
        for(int shift = -querylength + minoverlap; shift < subjectlength - minoverlap; shift++)

        This loop is split into 3 parts. shift = 0, shift < 0 and shift > 0
    */

    //shift == 0
    {
        const int shift = 0;
        const int overlapsize = std::min(querylength, subjectlength - shift) - std::max(-shift, 0);
        const int max_errors = int(double(overlapsize) * maxErrorRate);

        int score = hammingdistanceHiLo(subjectdata_hi,
                            subjectdata_lo,
                            querydata_hi,
                            querydata_lo,
                            subjectlength - abs(shift),
                            querylength - abs(shift),
                            max_errors);




            score = (score < max_errors ?
                    score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                    : std::numeric_limits<int>::max()); // too many errors, discard

    //    if((counter == 3 || counter == 4)  && shift == -29)printf("shift = %d, score = %d\n", shift, score);

        if(score < bestScore){
            bestScore = score;
            bestShift = shift;
        }
    }

    // shift < 0
    for(int shift = -1; shift >= -querylength + minoverlap; --shift){
        const int overlapsize = std::min(querylength, subjectlength - shift) - std::max(-shift, 0);
        const int max_errors = int(double(overlapsize) * maxErrorRate);

        shiftBitsLeftBy((unsigned char*)querydata_hi, querybytes / 2, 1);
        shiftBitsLeftBy((unsigned char*)querydata_lo, querybytes / 2, 1);

        /*if((counter == 3 || counter == 4) && shift == -29){
            auto b2c = [](char c){
                switch(c){
                    case 0x00: return 'A';
                    case 0x01: return 'C';
                    case 0x02: return 'G';
                    case 0x03: return 'T';
                    default: return 'F';
                }
            };
            printf("subjectbases: %d, querybases %d\n", subjectlength, querylength);
            for(int i = 0; i < subjectlength; i++){
                printf("%c", b2c(care::Sequence2BitHiLo::get(subjectdata.data(), subjectlength, i)));
            }
            printf("\n");

            for(int i = 0; i < querylength; i++){
                printf("%c", b2c(care::Sequence2BitHiLo::get(querydata.data(), querylength, i)));
            }
            printf("\n");
        }*/

        int score = hammingdistanceHiLo(subjectdata_hi,
                            subjectdata_lo,
                            querydata_hi,
                            querydata_lo,
                            subjectlength - abs(shift),
                            querylength - abs(shift),
                            max_errors);

            score = (score < max_errors ?
                    score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                    : std::numeric_limits<int>::max()); // too many errors, discard

        //if((counter == 3 || counter == 4)  && shift == -29)printf("shift = %d, score = %d\n", shift, score);

        if(score < bestScore){
            bestScore = score;
            bestShift = shift;
        }
    }

    //shift > 0

    //load query again from memory since it has been modified by calculations with shift < 0
    std::copy(query, query + querybytes, querydata.begin());

    for(int shift = 1; shift < subjectlength - minoverlap; ++shift){
        const int overlapsize = std::min(querylength, subjectlength - shift) - std::max(-shift, 0);
        const int max_errors = int(double(overlapsize) * maxErrorRate);

        shiftBitsLeftBy((unsigned char*)subjectdata_hi, subjectbytes / 2, 1);
        shiftBitsLeftBy((unsigned char*)subjectdata_lo, subjectbytes / 2, 1);

        /*if(counter == 1 && shift == 38){
            auto b2c = [](char c){
                switch(c){
                    case 0x00: return 'A';
                    case 0x01: return 'C';
                    case 0x02: return 'G';
                    case 0x03: return 'T';
                    default: return 'F';
                }
            };
            printf("subjectbases: %d, querybases %d\n", subjectlength, querylength);
            for(int i = 0; i < subjectlength; i++){
                printf("%c", b2c(care::Sequence2BitHiLo::get(subjectdata.data(), subjectlength, i)));
            }
            printf("\n");

            for(int i = 0; i < querylength; i++){
                printf("%c", b2c(care::Sequence2BitHiLo::get(querydata.data(), querylength, i)));
            }
            printf("\n");
        }*/

        int score = hammingdistanceHiLo(subjectdata_hi,
                            subjectdata_lo,
                            querydata_hi,
                            querydata_lo,
                            subjectlength - abs(shift),
                            querylength - abs(shift),
                            max_errors);

            score = (score < max_errors ?
                    score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                    : std::numeric_limits<int>::max()); // too many errors, discard

        //if((counter == 3 || counter == 4)  && shift == -29)printf("shift = %d, score = %d\n", shift, score);

        if(score < bestScore){
            bestScore = score;
            bestShift = shift;
        }
    }

#else

    for(int shift = -querylength + minoverlap; shift < subjectlength - minoverlap; shift++){
        const int overlapsize = std::min(querylength, subjectlength - shift) - std::max(-shift, 0);
        const int max_errors = int(double(overlapsize) * maxErrorRate);

        std::copy(subject, subject + subjectbytes, subjectdata.begin());
        std::copy(query, query + querybytes, querydata.begin());

        if(shift < 0){
            shiftBitsLeftBy((unsigned char*)querydata_hi, querybytes / 2, -shift);
            shiftBitsLeftBy((unsigned char*)querydata_lo, querybytes / 2, -shift);
        }else{
            shiftBitsLeftBy((unsigned char*)subjectdata_hi, subjectbytes / 2, shift);
            shiftBitsLeftBy((unsigned char*)subjectdata_lo, subjectbytes / 2, shift);
        }

        int score = hammingdistanceHiLo(subjectdata_hi,
                            subjectdata_lo,
                            querydata_hi,
                            querydata_lo,
                            subjectlength - abs(shift),
                            querylength - abs(shift));

        #if 1
            score = (score < max_errors ?
                    score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                    : std::numeric_limits<int>::max()); // too many errors, discard
        #else
        	score += totalbases - 2*overlapsize;
        #endif

        if(score < bestScore){
            bestScore = score;
            bestShift = shift;
        }
    }
#endif

    Result_t result;
    result.isValid = (bestShift != -querylength);

    const int queryoverlapbegin_incl = std::max(-bestShift, 0);
    const int queryoverlapend_excl = std::min(querylength, subjectlength - bestShift);
    const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
    const int opnr = bestScore - totalbases + 2*overlapsize;

    result.score = bestScore;
    result.subject_begin_incl = std::max(0, bestShift);
    result.query_begin_incl = queryoverlapbegin_incl;
    result.overlap = overlapsize;
    result.shift = bestShift;
    result.nOps = opnr;
    result.isNormalized = false;

    return result;
}
#endif



/*
    GPU alignment
*/
#ifdef __NVCC__

template<int BLOCKSIZE, class Accessor>
__global__
void
cuda_shifted_hamming_distance_kernel(Result_t* results,
                              const char* subjectsdata,
                              const int* subjectlengths,
                              const char* queriesdata,
                              const int* querylengths,
                              const int* NqueriesPrefixSum,
                              int Nsubjects,
                              int max_sequence_bytes,
                              size_t sequencepitch,
                              int min_overlap,
                              double maxErrorRate,
                              double min_overlap_ratio,
                              Accessor getChar){
    constexpr int WARPSIZE = 32;
    constexpr int NWARPS = (BLOCKSIZE + WARPSIZE - 1) / WARPSIZE;

    static_assert(sizeof(int2) == sizeof(unsigned long long), "sizeof(int2) != sizeof(unsigned long long)");
    static_assert(BLOCKSIZE % WARPSIZE == 0,
        "BLOCKSIZE must be multiple of WARPSIZE");

    extern __shared__ char smem[];

    //set up shared memory pointers
    char* sharedSubject = (char*)(smem);
    char* sharedQuery = (char*)(sharedSubject + max_sequence_bytes);

    for(unsigned queryIndex = blockIdx.x; queryIndex < NqueriesPrefixSum[Nsubjects]; queryIndex += gridDim.x){

        //find subjectindex
        int subjectIndex = 0;
        for(; subjectIndex < Nsubjects; subjectIndex++){
            if(queryIndex < NqueriesPrefixSum[subjectIndex+1])
                break;
        }

        //save subject in shared memory
        const int subjectbases = subjectlengths[subjectIndex];
        for(int threadid = threadIdx.x; threadid < max_sequence_bytes; threadid += BLOCKSIZE){
            sharedSubject[threadid] = subjectsdata[subjectIndex * sequencepitch + threadid];
        }

        //save query in shared memory
        const int querybases = querylengths[queryIndex];
        for(int threadid = threadIdx.x; threadid < max_sequence_bytes; threadid += BLOCKSIZE){
            sharedQuery[threadid] = queriesdata[queryIndex * sequencepitch + threadid];
        }
        //if(threadIdx.x == 0 && querybases != 100){
        //    printf("sid %d, qid %d, s %d, q %d\n", subjectIndex, queryIndex, subjectbases, querybases);
        //}

        __syncthreads();

        //begin SHD algorithm

        const int minoverlap = max(min_overlap, int(double(subjectbases) * min_overlap_ratio));
        const int totalbases = subjectbases + querybases;

        int bestScore = totalbases; // score is number of mismatches
        int bestShift = -querybases; // shift of query relative to subject. shift < 0 if query begins before subject

        for(int shift = -querybases + minoverlap + threadIdx.x; shift < subjectbases - minoverlap; shift += BLOCKSIZE){
            const int overlapsize = min(querybases, subjectbases - shift) - max(-shift, 0);
            const int max_errors = int(double(overlapsize) * maxErrorRate);
            int score = 0;

            for(int j = max(-shift, 0); j < min(querybases, subjectbases - shift) && score < max_errors; j++){
                score += getChar(sharedSubject, subjectbases, j + shift) != getChar(sharedQuery, querybases, j);
            }
#if 1
            score = (score < max_errors ?
                    score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                    : std::numeric_limits<int>::max()); // too many errors, discard
#else
	    score += totalbases - 2*overlapsize;
#endif
            if(score < bestScore){
                bestScore = score;
                bestShift = shift;
            }
        }

        // perform reduction to find smallest score in block. the corresponding shift is required, too
        // pack both score and shift into int2 and perform int2-reduction by only comparing the score

        int2 myval = make_int2(bestScore, bestShift);

        __shared__ unsigned long long blockreducetmp[NWARPS];

        auto func = [](unsigned long long a, unsigned long long b){
            return (*((int2*)&a)).x < (*((int2*)&b)).x ? a : b;
        };

        #if __CUDACC_VER_MAJOR__ < 9
                unsigned long long tilereduced = reduceTile<32>(*((unsigned long long*)&myval), func);
                int warp = threadIdx.x / WARPSIZE;
                int lane = threadIdx.x % WARPSIZE;
                if(lane == 0)
                    blockreducetmp[warp] = tilereduced;
        #else
                auto tile = tiled_partition<32>(this_thread_block());
                unsigned long long tilereduced = reduceTile(tile,
                                            *((unsigned long long*)&myval),
                                            func);
                int warp = threadIdx.x / WARPSIZE;
                if(tile.thread_rank() == 0)
                    blockreducetmp[warp] = tilereduced;
        #endif

        __syncthreads();

        //make result
        if(threadIdx.x == 0){
            //reduce warp results
            unsigned long long reduced = blockreducetmp[0];
            for(int i = 0; i < NWARPS; i++){
                reduced = func(reduced, blockreducetmp[i]);
            }

            bestScore = ((int2*)&reduced)->x;
            bestShift = ((int2*)&reduced)->y;

            Result_t result;

            result.isValid = (bestShift != -querybases);
            const int queryoverlapbegin_incl = max(-bestShift, 0);
            const int queryoverlapend_excl = min(querybases, subjectbases - bestShift);
            const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
            const int opnr = bestScore - totalbases + 2*overlapsize;

            result.score = bestScore;
            result.subject_begin_incl = max(0, bestShift);
            result.query_begin_incl = queryoverlapbegin_incl;
            result.overlap = overlapsize;
            result.shift = bestShift;
            result.nOps = opnr;
            result.isNormalized = false;

            results[queryIndex] = result;
        }
    }

}


template<class Accessor>
void call_shd_kernel_async(const SHDdata& shddata,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength,
                      Accessor accessor) noexcept{

      const int minoverlap = max(min_overlap, int(double(maxSubjectLength) * min_overlap_ratio));
      const int maxShiftsToCheck = maxSubjectLength+1 + maxQueryLength - 2*minoverlap;
      dim3 block(std::min(256, 32 * SDIV(maxShiftsToCheck, 32)), 1, 1);
      dim3 grid(shddata.n_queries, 1, 1);

      const std::size_t smem = sizeof(char) * 2 * shddata.max_sequence_bytes;

      #define mycall(blocksize) cuda_shifted_hamming_distance_kernel<(blocksize)> \
                                  <<<grid, block, smem, shddata.streams[0]>>>( \
                                  shddata.d_results, \
                                  shddata.d_subjectsdata, shddata.d_subjectlengths, \
                                  shddata.d_queriesdata, shddata.d_querylengths, \
                                  shddata.d_NqueriesPrefixSum, shddata.n_subjects, \
                                  shddata.max_sequence_bytes, \
                                  shddata.sequencepitch, \
                                  min_overlap, \
                                  maxErrorRate, \
                                  min_overlap_ratio, \
                                  accessor); CUERR;

      switch(block.x){
      case 32: mycall(32); break;
      case 64: mycall(64); break;
      case 96: mycall(96); break;
      case 128: mycall(128); break;
      case 160: mycall(160); break;
      case 192: mycall(192); break;
      case 224: mycall(224); break;
      case 256: mycall(256); break;
      default: throw std::runtime_error("Want to call shd kernel with 0 threads due to a bug.");
      }

      #undef mycall
}

template<class Accessor>
void call_shd_kernel(const SHDdata& shddata,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength,
                      Accessor accessor) noexcept{

    call_shd_kernel_async(shddata,
                        min_overlap,
                        maxErrorRate,
                        min_overlap_ratio,
                        maxSubjectLength,
                        maxQueryLength,
                        accessor);

    cudaStreamSynchronize(shddata.streams[0]); CUERR;
}





template<int BLOCKSIZE, class Accessor>
__global__
void
cuda_shifted_hamming_distance_with_revcompl_kernel(Result_t* results,
                              const char* subjectsdata,
                              const int* subjectlengths,
                              const char* queriesdata,
                              const int* querylengths,
                              const int* NqueriesPrefixSum,
                              int Nsubjects,
                              int max_sequence_bytes,
                              size_t sequencepitch,
                              int min_overlap,
                              double maxErrorRate,
                              double min_overlap_ratio,
                              Accessor getChar){
    constexpr int WARPSIZE = 32;
    constexpr int NWARPS = (BLOCKSIZE + WARPSIZE - 1) / WARPSIZE;

    static_assert(sizeof(int2) == sizeof(unsigned long long), "sizeof(int2) != sizeof(unsigned long long)");
    static_assert(BLOCKSIZE % WARPSIZE == 0,
        "BLOCKSIZE must be multiple of WARPSIZE");

    extern __shared__ char smem[];

    //set up shared memory pointers
    char* sharedSubject = (char*)(smem);
    char* sharedQuery = (char*)(sharedSubject + max_sequence_bytes);
    char* sharedQueryRevcompl = (char*)(sharedQuery + max_sequence_bytes);

    const int nQueries = NqueriesPrefixSum[Nsubjects];

    for(unsigned resultIndex = blockIdx.x; resultIndex < nQueries * 2; resultIndex += gridDim.x){

        const int queryIndex = resultIndex < nQueries ? resultIndex : resultIndex - nQueries;

        //find subjectindex
        int subjectIndex = 0;
        for(; subjectIndex < Nsubjects; subjectIndex++){
            if(queryIndex < NqueriesPrefixSum[subjectIndex+1])
                break;
        }

        //save subject in shared memory
        const int subjectbases = subjectlengths[subjectIndex];
        for(int threadid = threadIdx.x; threadid < max_sequence_bytes; threadid += BLOCKSIZE){
            sharedSubject[threadid] = subjectsdata[subjectIndex * sequencepitch + threadid];
        }

        //save query in shared memory
        const int querybases = querylengths[queryIndex];
        for(int threadid = threadIdx.x; threadid < max_sequence_bytes; threadid += BLOCKSIZE){
            sharedQuery[threadid] = queriesdata[queryIndex * sequencepitch + threadid];
        }
        //if(threadIdx.x == 0 && querybases != 100){
        //    printf("sid %d, qid %d, s %d, q %d\n", subjectIndex, queryIndex, subjectbases, querybases);
        //}

        __syncthreads();



        auto make_reverse_complement = [](std::uint8_t* reverseComplement, const std::uint8_t* sequence, int sequencelength){
            auto make_reverse_complement_byte = [](std::uint8_t in) -> std::uint8_t{
                in = ((in >> 2)  & 0x3333u) | ((in & 0x3333u) << 2);
                in = ((in >> 4)  & 0x0F0Fu) | ((in & 0x0F0Fu) << 4);
                return (std::uint8_t(-1) - in) >> (8 * 1 - (4 << 1));
            };

            const int bytes = (sequencelength + 3) / 4;
            const int unusedPositions = bytes * 4 - sequencelength;

            for(int i = 0; i < bytes; i++){
                reverseComplement[i] = make_reverse_complement_byte(sequence[bytes - 1 - i]);
            }

            if(unusedPositions > 0){
                reverseComplement[0] <<= (2 * unusedPositions);
                for(int i = 1; i < bytes; i++){
                    reverseComplement[i-1] |= reverseComplement[i] >> (2 * (4-unusedPositions));
                    reverseComplement[i] <<= (2 * unusedPositions);
                }
            }
        };

        //queryIndex != resultIndex -> reverse complement
        if(queryIndex != resultIndex && threadIdx.x == 0){
            make_reverse_complement((std::uint8_t*)sharedQueryRevcompl, (const std::uint8_t*)sharedQuery, querybases);
        }
        __syncthreads();

        //begin SHD algorithm

        const char* query = queryIndex == resultIndex ? sharedQuery : sharedQueryRevcompl;

        const int minoverlap = max(min_overlap, int(double(subjectbases) * min_overlap_ratio));
        const int totalbases = subjectbases + querybases;

        int bestScore = totalbases; // score is number of mismatches
        int bestShift = -querybases; // shift of query relative to subject. shift < 0 if query begins before subject

        for(int shift = -querybases + minoverlap + threadIdx.x; shift < subjectbases - minoverlap; shift += BLOCKSIZE){
            const int overlapsize = min(querybases, subjectbases - shift) - max(-shift, 0);
            const int max_errors = int(double(overlapsize) * maxErrorRate);
            int score = 0;

            for(int j = max(-shift, 0); j < min(querybases, subjectbases - shift) && score < max_errors; j++){
                score += getChar(sharedSubject, subjectbases, j + shift) != getChar(query, querybases, j);
            }
#if 1
            score = (score < max_errors ?
                    score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                    : std::numeric_limits<int>::max()); // too many errors, discard
#else
	    score += totalbases - 2*overlapsize;
#endif
            if(score < bestScore){
                bestScore = score;
                bestShift = shift;
            }
        }
        // perform reduction to find smallest score in block. the corresponding shift is required, too
        // pack both score and shift into int2 and perform int2-reduction by only comparing the score

        int2 myval = make_int2(bestScore, bestShift);

        __shared__ unsigned long long blockreducetmp[NWARPS];

        auto func = [](unsigned long long a, unsigned long long b){
            return (*((int2*)&a)).x < (*((int2*)&b)).x ? a : b;
        };

        #if __CUDACC_VER_MAJOR__ < 9
                unsigned long long tilereduced = reduceTile<32>(*((unsigned long long*)&myval), func);
                int warp = threadIdx.x / WARPSIZE;
                int lane = threadIdx.x % WARPSIZE;
                if(lane == 0)
                    blockreducetmp[warp] = tilereduced;
        #else
                auto tile = tiled_partition<32>(this_thread_block());
                unsigned long long tilereduced = reduceTile(tile,
                                            *((unsigned long long*)&myval),
                                            func);
                int warp = threadIdx.x / WARPSIZE;
                if(tile.thread_rank() == 0)
                    blockreducetmp[warp] = tilereduced;
        #endif

        __syncthreads();

        //make result
        if(threadIdx.x == 0){
            //reduce warp results
            unsigned long long reduced = blockreducetmp[0];
            for(int i = 0; i < NWARPS; i++){
                reduced = func(reduced, blockreducetmp[i]);
            }

            bestScore = ((int2*)&reduced)->x;
            bestShift = ((int2*)&reduced)->y;

            Result_t result;

            result.isValid = (bestShift != -querybases);
            const int queryoverlapbegin_incl = max(-bestShift, 0);
            const int queryoverlapend_excl = min(querybases, subjectbases - bestShift);
            const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
            const int opnr = bestScore - totalbases + 2*overlapsize;

            result.score = bestScore;
            result.subject_begin_incl = max(0, bestShift);
            result.query_begin_incl = queryoverlapbegin_incl;
            result.overlap = overlapsize;
            result.shift = bestShift;
            result.nOps = opnr;
            result.isNormalized = false;

            results[resultIndex] = result;
        }
    }
}


template<class Accessor>
void call_shd_with_revcompl_kernel_async(const SHDdata& shddata,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength,
                      Accessor accessor) noexcept{

      const int minoverlap = max(min_overlap, int(double(maxSubjectLength) * min_overlap_ratio));
      const int maxShiftsToCheck = maxSubjectLength+1 + maxQueryLength - 2*minoverlap;
      dim3 block(std::min(256, 32 * SDIV(maxShiftsToCheck, 32)), 1, 1);
      dim3 grid(shddata.n_queries*2, 1, 1); // one block per (query and its reverse complement)

      const std::size_t smem = sizeof(char) * 3 * shddata.max_sequence_bytes;

      #define mycall(blocksize) cuda_shifted_hamming_distance_with_revcompl_kernel<(blocksize)> \
                                  <<<grid, block, smem, shddata.streams[0]>>>( \
                                  shddata.d_results, \
                                  shddata.d_subjectsdata, shddata.d_subjectlengths, \
                                  shddata.d_queriesdata, shddata.d_querylengths, \
                                  shddata.d_NqueriesPrefixSum, shddata.n_subjects, \
                                  shddata.max_sequence_bytes, \
                                  shddata.sequencepitch, \
                                  min_overlap, \
                                  maxErrorRate, \
                                  min_overlap_ratio, \
                                  accessor); CUERR;

      switch(block.x){
      case 32: mycall(32); break;
      case 64: mycall(64); break;
      case 96: mycall(96); break;
      case 128: mycall(128); break;
      case 160: mycall(160); break;
      case 192: mycall(192); break;
      case 224: mycall(224); break;
      case 256: mycall(256); break;
      default: throw std::runtime_error("Want to call shd kernel with 0 threads due to a bug.");
      }

      #undef mycall
}

template<class Accessor>
void call_shd_with_revcompl_kernel(const SHDdata& shddata,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength,
                      Accessor accessor) noexcept{

    call_shd_with_revcompl_kernel_async(shddata,
                        min_overlap,
                        maxErrorRate,
                        min_overlap_ratio,
                        maxSubjectLength,
                        maxQueryLength,
                        accessor);

    cudaStreamSynchronize(shddata.streams[0]); CUERR;
}









template<int BLOCKSIZE, class Accessor, class RevCompl>
__global__
void
cuda_shifted_hamming_distance_with_revcompl_kernel(Result_t* results,
                              const char* subjectsdata,
                              const int* subjectlengths,
                              const char* queriesdata,
                              const int* querylengths,
                              const int* NqueriesPrefixSum,
                              int Nsubjects,
                              int max_sequence_bytes,
                              size_t sequencepitch,
                              int min_overlap,
                              double maxErrorRate,
                              double min_overlap_ratio,
                              Accessor getChar,
                              RevCompl make_reverse_complement){
    constexpr int WARPSIZE = 32;
    constexpr int NWARPS = (BLOCKSIZE + WARPSIZE - 1) / WARPSIZE;

    static_assert(sizeof(int2) == sizeof(unsigned long long), "sizeof(int2) != sizeof(unsigned long long)");
    static_assert(BLOCKSIZE % WARPSIZE == 0,
        "BLOCKSIZE must be multiple of WARPSIZE");

    extern __shared__ char smem[];

    //set up shared memory pointers
    char* sharedSubject = (char*)(smem);
    char* sharedQuery = (char*)(sharedSubject + max_sequence_bytes);
    char* sharedQueryRevcompl = (char*)(sharedQuery + max_sequence_bytes);

    const int nQueries = NqueriesPrefixSum[Nsubjects];

    for(unsigned resultIndex = blockIdx.x; resultIndex < nQueries * 2; resultIndex += gridDim.x){

        const int queryIndex = resultIndex < nQueries ? resultIndex : resultIndex - nQueries;

        //find subjectindex
        int subjectIndex = 0;
        for(; subjectIndex < Nsubjects; subjectIndex++){
            if(queryIndex < NqueriesPrefixSum[subjectIndex+1])
                break;
        }

        //save subject in shared memory
        const int subjectbases = subjectlengths[subjectIndex];
        for(int threadid = threadIdx.x; threadid < max_sequence_bytes; threadid += BLOCKSIZE){
            sharedSubject[threadid] = subjectsdata[subjectIndex * sequencepitch + threadid];
        }

        //save query in shared memory
        const int querybases = querylengths[queryIndex];
        for(int threadid = threadIdx.x; threadid < max_sequence_bytes; threadid += BLOCKSIZE){
            sharedQuery[threadid] = queriesdata[queryIndex * sequencepitch + threadid];
        }
        //if(threadIdx.x == 0 && querybases != 100){
        //    printf("sid %d, qid %d, s %d, q %d\n", subjectIndex, queryIndex, subjectbases, querybases);
        //}

        __syncthreads();



        //queryIndex != resultIndex -> reverse complement
        if(queryIndex != resultIndex && threadIdx.x == 0){
            make_reverse_complement((std::uint8_t*)sharedQueryRevcompl, (const std::uint8_t*)sharedQuery, querybases);
        }
        __syncthreads();

        //begin SHD algorithm

        const char* query = queryIndex == resultIndex ? sharedQuery : sharedQueryRevcompl;

        const int minoverlap = max(min_overlap, int(double(subjectbases) * min_overlap_ratio));
        const int totalbases = subjectbases + querybases;

        int bestScore = totalbases; // score is number of mismatches
        int bestShift = -querybases; // shift of query relative to subject. shift < 0 if query begins before subject

        for(int shift = -querybases + minoverlap + threadIdx.x; shift < subjectbases - minoverlap; shift += BLOCKSIZE){
            const int overlapsize = min(querybases, subjectbases - shift) - max(-shift, 0);
            const int max_errors = int(double(overlapsize) * maxErrorRate);
            int score = 0;

            for(int j = max(-shift, 0); j < min(querybases, subjectbases - shift) && score < max_errors; j++){
                score += getChar(sharedSubject, subjectbases, j + shift) != getChar(query, querybases, j);
            }
#if 1
            score = (score < max_errors ?
                    score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                    : std::numeric_limits<int>::max()); // too many errors, discard
#else
	    score += totalbases - 2*overlapsize;
#endif
            if(score < bestScore){
                bestScore = score;
                bestShift = shift;
            }
        }
        // perform reduction to find smallest score in block. the corresponding shift is required, too
        // pack both score and shift into int2 and perform int2-reduction by only comparing the score

        int2 myval = make_int2(bestScore, bestShift);

        __shared__ unsigned long long blockreducetmp[NWARPS];

        auto func = [](unsigned long long a, unsigned long long b){
            return (*((int2*)&a)).x < (*((int2*)&b)).x ? a : b;
        };

        #if __CUDACC_VER_MAJOR__ < 9
                unsigned long long tilereduced = reduceTile<32>(*((unsigned long long*)&myval), func);
                int warp = threadIdx.x / WARPSIZE;
                int lane = threadIdx.x % WARPSIZE;
                if(lane == 0)
                    blockreducetmp[warp] = tilereduced;
        #else
                auto tile = tiled_partition<32>(this_thread_block());
                unsigned long long tilereduced = reduceTile(tile,
                                            *((unsigned long long*)&myval),
                                            func);
                int warp = threadIdx.x / WARPSIZE;
                if(tile.thread_rank() == 0)
                    blockreducetmp[warp] = tilereduced;
        #endif

        __syncthreads();

        //make result
        if(threadIdx.x == 0){
            //reduce warp results
            unsigned long long reduced = blockreducetmp[0];
            for(int i = 0; i < NWARPS; i++){
                reduced = func(reduced, blockreducetmp[i]);
            }

            bestScore = ((int2*)&reduced)->x;
            bestShift = ((int2*)&reduced)->y;

            Result_t result;

            result.isValid = (bestShift != -querybases);
            const int queryoverlapbegin_incl = max(-bestShift, 0);
            const int queryoverlapend_excl = min(querybases, subjectbases - bestShift);
            const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
            const int opnr = bestScore - totalbases + 2*overlapsize;

            result.score = bestScore;
            result.subject_begin_incl = max(0, bestShift);
            result.query_begin_incl = queryoverlapbegin_incl;
            result.overlap = overlapsize;
            result.shift = bestShift;
            result.nOps = opnr;
            result.isNormalized = false;

            results[resultIndex] = result;
        }
    }
}


template<class Accessor, class RevCompl>
void call_shd_with_revcompl_kernel_async(const SHDdata& shddata,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength,
                      Accessor accessor,
                      RevCompl make_reverse_complement){

      const int minoverlap = max(min_overlap, int(double(maxSubjectLength) * min_overlap_ratio));
      const int maxShiftsToCheck = maxSubjectLength+1 + maxQueryLength - 2*minoverlap;
      dim3 block(std::min(256, 32 * SDIV(maxShiftsToCheck, 32)), 1, 1);
      dim3 grid(shddata.n_queries*2, 1, 1); // one block per (query and its reverse complement)

      const std::size_t smem = sizeof(char) * 3 * shddata.max_sequence_bytes;

      #define mycall(blocksize) cuda_shifted_hamming_distance_with_revcompl_kernel<(blocksize)> \
                                  <<<grid, block, smem, shddata.streams[0]>>>( \
                                  shddata.d_results, \
                                  shddata.d_subjectsdata, shddata.d_subjectlengths, \
                                  shddata.d_queriesdata, shddata.d_querylengths, \
                                  shddata.d_NqueriesPrefixSum, shddata.n_subjects, \
                                  shddata.max_sequence_bytes, \
                                  shddata.sequencepitch, \
                                  min_overlap, \
                                  maxErrorRate, \
                                  min_overlap_ratio, \
                                  accessor, \
                                  make_reverse_complement); CUERR;

      switch(block.x){
      case 32: mycall(32); break;
      case 64: mycall(64); break;
      case 96: mycall(96); break;
      case 128: mycall(128); break;
      case 160: mycall(160); break;
      case 192: mycall(192); break;
      case 224: mycall(224); break;
      case 256: mycall(256); break;
      default: throw std::runtime_error("Want to call shd kernel with 0 threads due to a bug.");
      }

      #undef mycall
}

template<class Accessor, class RevCompl>
void call_shd_with_revcompl_kernel(const SHDdata& shddata,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength,
                      Accessor accessor,
                      RevCompl make_reverse_complement) noexcept{

    call_shd_with_revcompl_kernel_async(shddata,
                        min_overlap,
                        maxErrorRate,
                        min_overlap_ratio,
                        maxSubjectLength,
                        maxQueryLength,
                        accessor,
                        make_reverse_complement);

    cudaStreamSynchronize(shddata.streams[0]); CUERR;
}










//############## POPCOUNT SHIFTED HAMMING DISTANCE #################



template<int threads_per_shift, class RevCompl, class B>
__global__
void
cuda_popcount_shifted_hamming_distance_with_revcompl_kernel(Result_t* results,
                              const char* subjectsdata,
                              const int* subjectlengths,
                              const char* queriesdata,
                              const int* querylengths,
                              const int* NqueriesPrefixSum,
                              int Nsubjects,
                              int max_sequence_bytes,
                              size_t sequencepitch,
                              int min_overlap,
                              double maxErrorRate,
                              double min_overlap_ratio,
                              RevCompl make_reverse_complement,
                              B getNumBytes){

    auto make_reverse_complement_inplace = [&](std::uint8_t* sequence, int sequencelength){

        auto reverse_complement_byte = [](auto b) {
            b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
            b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
            b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
            return ~b;
        };

        const int bytes = getNumBytes(sequencelength);
        const int halfbytes = bytes / 2;
        //const int unusedBitsByte = halfbytes*8 - sequencelength;
        const int unusedBitsInLastUsedByte = SDIV(sequencelength, 8) * 8 - sequencelength;

        std::uint8_t* const hiBytes = sequence;
        std::uint8_t* const loBytes = sequence + halfbytes;

        const int usedBytesPerHalf = SDIV(sequencelength, 8);
        for(int i = 0; i < usedBytesPerHalf/2; ++i){
            const std::uint8_t hifront = reverse_complement_byte(hiBytes[i]);
            const std::uint8_t hiback = reverse_complement_byte(hiBytes[usedBytesPerHalf - 1 - i]);
            hiBytes[i] = hiback;
            hiBytes[usedBytesPerHalf - 1 - i] = hifront;

            const std::uint8_t lofront = reverse_complement_byte(loBytes[i]);
            const std::uint8_t loback = reverse_complement_byte(loBytes[usedBytesPerHalf - 1 - i]);
            loBytes[i] = loback;
            loBytes[usedBytesPerHalf - 1 - i] = lofront;
        }
        if(usedBytesPerHalf % 2 == 1){
            const int middleindex = usedBytesPerHalf/2;
            hiBytes[middleindex] = reverse_complement_byte(hiBytes[middleindex]);
            loBytes[middleindex] = reverse_complement_byte(loBytes[middleindex]);
        }

        if(unusedBitsInLastUsedByte != 0){
            for(int i = 0; i < halfbytes - 1; ++i){
                hiBytes[i] = (hiBytes[i] << unusedBitsInLastUsedByte) | (hiBytes[i+1] >> (8 - unusedBitsInLastUsedByte));
                loBytes[i] = (loBytes[i] << unusedBitsInLastUsedByte) | (loBytes[i+1] >> (8 - unusedBitsInLastUsedByte));
            }

            hiBytes[halfbytes - 1] <<= unusedBitsInLastUsedByte;
            loBytes[halfbytes - 1] <<= unusedBitsInLastUsedByte;
        }
    };

    //use threads in tile to shift bit-array array by shiftamounts bits to the left
    auto shiftBitsLeftBy = [](thread_block_tile<threads_per_shift>& tile,
                                unsigned char* array,
                                int bytes, // size of array
                                int shiftamount){
        constexpr int maxshiftPerIter = 7;
        const int iters = SDIV(shiftamount, maxshiftPerIter);

        for(int iter = 0; iter < iters-1; ++iter){
            for(int i = tile.thread_rank(); i < bytes - 1; i += tile.size()){
                unsigned char cur = array[i];
                unsigned char next = array[i+1];

                tile.sync();

                array[i] = (cur << maxshiftPerIter) | (next >> (8 - maxshiftPerIter));

                tile.sync();
            }
            if(tile.thread_rank() == 0)
                array[bytes - 1] <<= maxshiftPerIter;

            tile.sync();

            shiftamount -= maxshiftPerIter;
        }

        for(int i = tile.thread_rank(); i < bytes - 1; i += tile.size()){
            unsigned char cur = array[i];
            unsigned char next = array[i+1];

            tile.sync();

            array[i] = (cur << shiftamount) | (next >> (8 - shiftamount));

            tile.sync();
        }

        if(tile.thread_rank() == 0)
            array[bytes - 1] <<= shiftamount;

        tile.sync();
    };

    //result is returned by thread with tile.thread_rank() == 0
    auto hammingdistanceHiLo = [](thread_block_tile<threads_per_shift>& tile,
                                const int* lhi,
                                const int* llo,
                                const int* rhi,
                                const int* rlo,
                                int lhi_bitcount,
                                int rhi_bitcount,
                                int max_errors){

        const int overlap_bitcount = lhi_bitcount < rhi_bitcount ? lhi_bitcount : rhi_bitcount;
        const int completepartitions = (overlap_bitcount / 8) / sizeof(int);

        int laneresult = 0;

        for(int i = tile.thread_rank(); i < completepartitions; i += tile.size()){
            const int hixor = lhi[i] ^ rhi[i];
            const int loxor = llo[i] ^ rlo[i];
            const int bits = hixor | loxor;
            laneresult += __popc(bits);
        }

        int result = reduceTile(tile, laneresult, [](int l, int r){return l+r;});

        if(result >= max_errors)
            return result;

        int remaining_bitcount = overlap_bitcount - completepartitions * sizeof(int) * 8;
        if(remaining_bitcount != 0){
            const int completecharpartitions = (remaining_bitcount / 8) / sizeof(std::uint8_t);

            const std::uint8_t* const lhichar = (const std::uint8_t*)(lhi + completepartitions);
            const std::uint8_t* const llochar = (const std::uint8_t*)(llo + completepartitions);
            const std::uint8_t* const rhichar = (const std::uint8_t*)(rhi + completepartitions);
            const std::uint8_t* const rlochar = (const std::uint8_t*)(rlo + completepartitions);

            laneresult = 0;

            for(int i = tile.thread_rank(); i < completecharpartitions; i += tile.size()){
                const std::uint8_t hixorchar = lhichar[i] ^ rhichar[i];
                const std::uint8_t loxorchar = llochar[i] ^ rlochar[i];
                const std::uint8_t bitschar = hixorchar | loxorchar;
                laneresult += __popc(bitschar);
            }

            result += reduceTile(tile, laneresult, [](int l, int r){return l+r;});

            remaining_bitcount = remaining_bitcount - completecharpartitions * sizeof(std::uint8_t) * 8;

            if(remaining_bitcount != 0 && tile.thread_rank() == 0){
                std::uint8_t mask = 0xFF << (sizeof(std::uint8_t)*8 - remaining_bitcount);
                const std::uint8_t hixorchar2 = lhichar[completecharpartitions] ^ rhichar[completecharpartitions];
                const std::uint8_t loxorchar2 = llochar[completecharpartitions] ^ rlochar[completecharpartitions];
                const std::uint8_t bitschar2 = hixorchar2 | loxorchar2;
                result += __popc(bitschar2 & mask);
            }
        }

        return result;
    };


    static_assert(threads_per_shift > 0
        && threads_per_shift <= 32
        && power_of_two(threads_per_shift),
        "cuda_popcount_shifted_hamming_distance_with_revcompl_kernel: Invalid threads_per_shift.");

    static_assert(sizeof(int2) == sizeof(unsigned long long), "sizeof(int2) != sizeof(unsigned long long)");

    constexpr int WARPSIZE = 32;

    // max_sequence_bytes * tiles_per_block * 3
    extern __shared__ char smem[];

    //const int warps_per_block = blockDim.x / WARPSIZE;
    //set up tiles
    auto tile = tiled_partition<threads_per_shift>(this_thread_block());
    const int tiles_per_block = blockDim.x / threads_per_shift;
    const int localTileId = threadIdx.x / threads_per_shift;
    const int warpId = threadIdx.x / WARPSIZE;

    //set up shared memory pointers
    char* const sharedSubject = (char*)(smem);
    char* const sharedQuery = (char*)(sharedSubject + max_sequence_bytes * tiles_per_block);
    //char* const sharedQueryRevcompl = (char*)(sharedQuery + max_sequence_bytes * tiles_per_block);

    //set up shared memory per tile
    char* const myTileSubject = sharedSubject + max_sequence_bytes * localTileId;
    char* const myTileQuery = sharedQuery + max_sequence_bytes * localTileId;
    //char* const myTileQueryRevcompl = sharedQueryRevcompl + max_sequence_bytes * localTileId;

    const int nQueries = NqueriesPrefixSum[Nsubjects];

    for(unsigned resultIndex = blockIdx.x; resultIndex < nQueries * 2; resultIndex += gridDim.x){

        const int queryIndex = resultIndex < nQueries ? resultIndex : resultIndex - nQueries;

        //find subjectindex
        int subjectIndex = 0;
        for(; subjectIndex < Nsubjects; subjectIndex++){
            if(queryIndex < NqueriesPrefixSum[subjectIndex+1])
                break;
        }

        //save subject in shared memory
        const int subjectbases = subjectlengths[subjectIndex];
        for(int lane = tile.thread_rank(); lane < max_sequence_bytes; lane += tile.size()){
            myTileSubject[lane] = subjectsdata[subjectIndex * sequencepitch + lane];
        }

        //save query in shared memory
        const int querybases = querylengths[queryIndex];
        for(int lane = tile.thread_rank(); lane < max_sequence_bytes; lane += tile.size()){
            myTileQuery[lane] = queriesdata[queryIndex * sequencepitch + lane];
        }

        tile.sync();

        //queryIndex != resultIndex -> reverse complement
        if(queryIndex != resultIndex && tile.thread_rank() == 0){

            //make_reverse_complement((std::uint8_t*)myTileQueryRevcompl, (const std::uint8_t*)myTileQuery, querybases);

            make_reverse_complement_inplace((std::uint8_t*)myTileQuery,querybases);
        }

        tile.sync();

        //begin SHD algorithm

        //const char* query = queryIndex == resultIndex ? myTileQuery : myTileQueryRevcompl;
        const char* query = myTileQuery;

        const int subjectbytes = getNumBytes(subjectbases);
        const int querybytes = getNumBytes(querybases);
        const int totalbases = subjectbases + querybases;
        const int minoverlap = max(min_overlap, int(double(subjectbases) * min_overlap_ratio));

        //assert(subjectbytes % (2 * sizeof(int)) == 0);
        //assert(querybytes % (2 * sizeof(int)) == 0);

        //only valid for tile.thread_rank() == 0
        int bestScore = totalbases; // score is number of mismatches
        int bestShift = -querybases; // shift of query relative to subject. shift < 0 if query begins before subject

        int* subjectdata_hi = (int*)myTileSubject;
        int* subjectdata_lo = (int*)(myTileSubject + subjectbytes / 2);
        int* querydata_hi = (int*)query;
        int* querydata_lo = (int*)(query + querybytes / 2);

        /*
            The goal is to calculate the hamming distance for each shift in
            for(int shift = -querylength + minoverlap; shift < subjectlength - minoverlap; shift++)

            This loop is split into 3 parts. shift = 0, shift < 0 and shift > 0
        */

        //shift == 0
        if(localTileId == 0){
            const int shift = 0;
            const int overlapsize = min(querybases, subjectbases - shift) - max(-shift, 0);
            const int max_errors = int(double(overlapsize) * maxErrorRate);

            int score = hammingdistanceHiLo(tile,
                                subjectdata_hi,
                                subjectdata_lo,
                                querydata_hi,
                                querydata_lo,
                                subjectbases - abs(shift),
                                querybases - abs(shift),
                                max_errors);

            score = (score < max_errors ?
                    score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                    : std::numeric_limits<int>::max()); // too many errors, discard

            if(tile.thread_rank() == 0 && score < bestScore){
                bestScore = score;
                bestShift = shift;
            }
        }

        // shift > 0
        int shifts = subjectbases - minoverlap - 1;
        int shifts_per_tile = SDIV(shifts, tiles_per_block);
        int firstShift = 1 + localTileId * shifts_per_tile;

        //for(int shift = 1; shift < subjectlength - minoverlap; ++shift){
        for(int shift = firstShift;
            shift < subjectbases - minoverlap && shift < 1 + (localTileId+1) * shifts_per_tile;
            ++shift){

            const int overlapsize = min(querybases, subjectbases - shift) - max(-shift, 0);
            const int max_errors = int(double(overlapsize) * maxErrorRate);

            const int shiftamount = shift == firstShift ? firstShift : 1;

            shiftBitsLeftBy(tile,(unsigned char*)subjectdata_hi, subjectbytes / 2, shiftamount);
            shiftBitsLeftBy(tile,(unsigned char*)subjectdata_lo, subjectbytes / 2, shiftamount);

            int score = hammingdistanceHiLo(tile,
                                subjectdata_hi,
                                subjectdata_lo,
                                querydata_hi,
                                querydata_lo,
                                subjectbases - abs(shift),
                                querybases - abs(shift),
                                max_errors);

            score = (score < max_errors ?
                    score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                    : std::numeric_limits<int>::max()); // too many errors, discard

            if(tile.thread_rank() == 0 && score < bestScore){
                bestScore = score;
                bestShift = shift;
            }
        }

        //load subject again from memory since it has been modified by calculations with shift > 0

        for(int lane = tile.thread_rank(); lane < max_sequence_bytes; lane += tile.size()){
            myTileSubject[lane] = subjectsdata[subjectIndex * sequencepitch + lane];
        }

        tile.sync();

        // shift < 0
        shifts = -(-querybases + minoverlap);
        shifts_per_tile = SDIV(shifts, tiles_per_block);
        firstShift = -1 - localTileId * shifts_per_tile;

        //for(int shift = -1; shift >= -querylength + minoverlap; --shift){
        for(int shift = firstShift;
            shift >= -querybases + minoverlap && shift > -1 - (localTileId+1) * shifts_per_tile;
            --shift){

            const int overlapsize = min(querybases, subjectbases - shift) - max(-shift, 0);
            const int max_errors = int(double(overlapsize) * maxErrorRate);

            const int shiftamount = shift == firstShift ? -firstShift : 1; //firstShift is negative, don't forget the minus sign!

            shiftBitsLeftBy(tile,(unsigned char*)querydata_hi, querybytes / 2, shiftamount);
            shiftBitsLeftBy(tile,(unsigned char*)querydata_lo, querybytes / 2, shiftamount);

            int score = hammingdistanceHiLo(tile,
                                subjectdata_hi,
                                subjectdata_lo,
                                querydata_hi,
                                querydata_lo,
                                subjectbases - abs(shift),
                                querybases - abs(shift),
                                max_errors);

            score = (score < max_errors ?
                    score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                    : std::numeric_limits<int>::max()); // too many errors, discard

            if(tile.thread_rank() == 0 && score < bestScore){
                bestScore = score;
                bestShift = shift;
            }
        }

        /*
            The result per tile is present in thread tile.thread_rank() == 0;
            Each of those threads writes its result to shared memory.
            Then, a min reduce is performed on these results to determine the final result
        */

        // perform reduction to find smallest score in block. the corresponding shift is required, too
        // pack both score and shift into int2 and perform int2-reduction by only comparing the score

        int2 myval = make_int2(bestScore, bestShift);

        //reuse allocated shared memory since sequence data is no longer used
        //unsigned long long* blockreducetmp = (unsigned long long*)smem;
        __shared__ unsigned long long blockreducetmp[16];

        auto func = [](unsigned long long a, unsigned long long b){
            return (*((int2*)&a)).x < (*((int2*)&b)).x ? a : b;
        };

        auto warptile = tiled_partition<32>(this_thread_block());
        unsigned long long warpreduced = reduceTile(warptile,
                                    *((unsigned long long*)&myval),
                                    func);
        if(warptile.thread_rank() == 0)
            blockreducetmp[warpId] = warpreduced;

        __syncthreads();

        //make result
        if(threadIdx.x == 0){
            //reduce warp results

            unsigned long long reduced = blockreducetmp[0];
            for(int i = 1; i < blockDim.x / WARPSIZE; i++){
                reduced = func(reduced, blockreducetmp[i]);
            }

            bestScore = ((int2*)&reduced)->x;
            bestShift = ((int2*)&reduced)->y;

            Result_t result;

            result.isValid = (bestShift != -querybases);
            const int queryoverlapbegin_incl = max(-bestShift, 0);
            const int queryoverlapend_excl = min(querybases, subjectbases - bestShift);
            const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
            const int opnr = bestScore - totalbases + 2*overlapsize;

            result.score = bestScore;
            result.subject_begin_incl = max(0, bestShift);
            result.query_begin_incl = queryoverlapbegin_incl;
            result.overlap = overlapsize;
            result.shift = bestShift;
            result.nOps = opnr;
            result.isNormalized = false;

            results[resultIndex] = result;
        }

        __syncthreads();
    }
}


template<class RevCompl, class B>
void call_popcount_shd_with_revcompl_kernel_async(const SHDdata& shddata,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength,
                      RevCompl make_reverse_complement,
                      B getNumBytes){

        auto nextPow2 = [](unsigned int v){
            v--;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            v++;
            return v;
        };

      const int minoverlap = std::max(min_overlap, int(double(maxSubjectLength) * min_overlap_ratio));

      const int maxSequenceLength = std::max(maxSubjectLength, maxQueryLength);
      const int completeInts = maxSequenceLength / (sizeof(int) * 8);
      const int tilesize = std::min(32u, power_of_two(completeInts) ? completeInts : nextPow2(completeInts));

      const int blocksize = 128;
      const int tiles_per_block = blocksize / tilesize;

      dim3 block(blocksize, 1, 1);
      dim3 grid(shddata.n_queries*2, 1, 1); // one block per (query and its reverse complement)
      //dim3 grid(1, 1, 1); // one block per (query and its reverse complement)

      const std::size_t smem = sizeof(char) * 2 * shddata.max_sequence_bytes * tiles_per_block;

      #define mycall(threads_per_shift) cuda_popcount_shifted_hamming_distance_with_revcompl_kernel<(threads_per_shift)> \
                                  <<<grid, block, smem, shddata.streams[0]>>>( \
                                  shddata.d_results, \
                                  shddata.d_subjectsdata, shddata.d_subjectlengths, \
                                  shddata.d_queriesdata, shddata.d_querylengths, \
                                  shddata.d_NqueriesPrefixSum, shddata.n_subjects, \
                                  shddata.max_sequence_bytes, \
                                  shddata.sequencepitch, \
                                  min_overlap, \
                                  maxErrorRate, \
                                  min_overlap_ratio, \
                                  make_reverse_complement, \
                                  getNumBytes); CUERR;

      switch(tilesize){
      case 1: mycall(1); break;
      case 2: mycall(2); break;
      case 4: mycall(4); break;
      case 8: mycall(8); break;
      case 16: mycall(16); break;
      case 32: mycall(32); break;
      default: throw std::runtime_error("call_popcount_shd_with_revcompl_kernel_async: invalid tilesize, " + std::to_string(tilesize));
      }

      #undef mycall
}

template<class RevCompl, class B>
void call_popcount_shd_with_revcompl_kernel(const SHDdata& shddata,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength,
                      RevCompl make_reverse_complement,
                      B getNumBytes) noexcept{

    call_popcount_shd_with_revcompl_kernel_async(shddata,
                        min_overlap,
                        maxErrorRate,
                        min_overlap_ratio,
                        maxSubjectLength,
                        maxQueryLength,
                        make_reverse_complement,
                        getNumBytes);

    cudaStreamSynchronize(shddata.streams[0]); CUERR;
}





#endif //ifdef __NVCC__

}//namespace shd

#endif
