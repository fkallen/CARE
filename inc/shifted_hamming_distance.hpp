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
    char* d_unpacked_queries = nullptr;

	Result_t* h_results = nullptr;
	char* h_subjectsdata = nullptr;
	char* h_queriesdata = nullptr;
	int* h_subjectlengths = nullptr;
	int* h_querylengths = nullptr;
    int* h_NqueriesPrefixSum = nullptr;
    BestAlignment_t* h_bestAlignmentFlags = nullptr;
    char* h_unpacked_queries = nullptr;

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
    int max_n_results = 0;

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


        score = (score < max_errors ?
                score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                : std::numeric_limits<int>::max()); // too many errors, discard

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

    auto shiftEncodedBasesLeftBy = [](unsigned int* array, int size, int shiftamount){
        const int completeInts = shiftamount / (8 * sizeof(unsigned int));

        /*std::cout << "shiftamount = " << shiftamount << ", completeInts = " << completeInts << std::endl;

        std::cout << "before" << std::endl;
        for(int i = 0; i < size; i++){
            std::cout << std::bitset<32>(array[i]) << " ";
        }
        std::cout << std::endl;*/

        for(int i = 0; i < size - completeInts; ++i){
            array[i] = array[completeInts + i];
        }

        for(int i = size - completeInts; i < size; ++i){
            array[i] = 0;
        }

        /*std::cout << "after copy" << std::endl;
        for(int i = 0; i < size; i++){
            std::cout << std::bitset<32>(array[i]) << " ";
        }
        std::cout << std::endl;*/

        shiftamount -= completeInts * 8 * sizeof(unsigned int);

        assert(shiftamount < 8 * sizeof(unsigned int));

        for(int i = 0; i < size - completeInts - 1; ++i){
            array[i] = (array[i] >> shiftamount) | (array[i+1] << (8 * sizeof(unsigned int) - shiftamount));
        }
        array[size - completeInts - 1] >>= shiftamount;

        /*std::cout << "after" << std::endl;
        for(int i = 0; i < size; i++){
            std::cout << std::bitset<32>(array[i]) << " ";
        }
        std::cout << std::endl;*/
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
    const int minoverlap = std::max(min_overlap, int(double(subjectlength) * min_overlap_ratio));

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

    /*static int counter = 0;
    counter++;

    if(counter == 3 || counter == 4){
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

        //if((counter == 3 || counter == 4))
        //    printf("shift = %d, score = %d\n", shift, score);

        if(score < bestScore){
            bestScore = score;
            bestShift = shift;
        }
    }

    // shift < 0
    for(int shift = -1; shift >= -querylength + minoverlap; --shift){
        const int overlapsize = std::min(querylength, subjectlength - shift) - std::max(-shift, 0);
        const int max_errors = int(double(overlapsize) * maxErrorRate);

        //std::cout << "before shift" << std::endl;
        //std::cout << care::Sequence2BitHiLo((unsigned char*)querydata.data(), querylength) << std::endl;

        shiftEncodedBasesLeftBy((unsigned int*)querydata_hi, querybytes / 2 / sizeof(unsigned int), 1);
        shiftEncodedBasesLeftBy((unsigned int*)querydata_lo, querybytes / 2 / sizeof(unsigned int), 1);

        //std::cout << "after shift" << std::endl;
        //std::cout << care::Sequence2BitHiLo((unsigned char*)querydata.data(), querylength) << std::endl;

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

        /*if((counter == 3 || counter == 4)){
            printf("shift = %d, score = %d\n", shift, score);

            if(shift == -29){
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
            }
        }*/

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

        shiftEncodedBasesLeftBy((unsigned int*)subjectdata_hi, subjectbytes / 2 / sizeof(unsigned int), 1);
        shiftEncodedBasesLeftBy((unsigned int*)subjectdata_lo, subjectbytes / 2 / sizeof(unsigned int), 1);

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

        /*if((counter == 3 || counter == 4)){
            printf("shift = %d, score = %d\n", shift, score);
            if(shift == -29){
                if(counter == 1 || counter == 2){
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
                }
            }
        }*/

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
                              RevCompl make_reverse_complement_inplace){
    constexpr int WARPSIZE = 32;
    constexpr int NWARPS = (BLOCKSIZE + WARPSIZE - 1) / WARPSIZE;

    static_assert(sizeof(int2) == sizeof(unsigned long long), "sizeof(int2) != sizeof(unsigned long long)");
    static_assert(BLOCKSIZE % WARPSIZE == 0,
        "BLOCKSIZE must be multiple of WARPSIZE");

    extern __shared__ char smem[];

    //set up shared memory pointers
    char* sharedSubject = (char*)(smem);
    char* sharedQuery = (char*)(sharedSubject + max_sequence_bytes);

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

        __syncthreads();

        //queryIndex != resultIndex -> reverse complement
        if(queryIndex != resultIndex && threadIdx.x == 0){
            make_reverse_complement_inplace((std::uint8_t*)sharedQuery, querybases);
        }
        __syncthreads();

        //begin SHD algorithm

        const char* query = sharedQuery;

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

            /*if(queryIndex == 0){
            for(int i = 0; i < blockDim.x; i++){
                if(threadIdx.x == i){
                    printf("shift = %d, score = %d\n", shift, score);
                }
            }
        }*/

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

      const std::size_t smem = sizeof(char) * 2 * shddata.max_sequence_bytes;

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


template<int threads_per_shift, class B>
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
                              B getNumBytes){

    auto make_reverse_complement_inplace = [&](unsigned int* sequence, int sequencelength){

        auto reverse_complement_int = [](auto n) {
            n = ((n >> 1) & 0x55555555) | ((n << 1) & 0xaaaaaaaa);
            n = ((n >> 2) & 0x33333333) | ((n << 2) & 0xcccccccc);
            n = ((n >> 4) & 0x0f0f0f0f) | ((n << 4) & 0xf0f0f0f0);
            n = ((n >> 8) & 0x00ff00ff) | ((n << 8) & 0xff00ff00);
            n = ((n >> 16) & 0x0000ffff) | ((n << 16) & 0xffff0000);
            return ~n;
        };

        const int ints = getNumBytes(sequencelength) / sizeof(unsigned int);
        const int unusedBitsInt = SDIV(sequencelength, 8 * sizeof(unsigned int)) * 8 * sizeof(unsigned int) - sequencelength;

        unsigned int* const hi = sequence;
        unsigned int* const lo = sequence + ints/2;

        const int intsPerHalf = SDIV(sequencelength, 8 * sizeof(unsigned int));
        for(int i = 0; i < intsPerHalf/2; ++i){
            const unsigned int hifront = reverse_complement_int(hi[i]);
            const unsigned int hiback = reverse_complement_int(hi[intsPerHalf - 1 - i]);
            hi[i] = hiback;
            hi[intsPerHalf - 1 - i] = hifront;

            const unsigned int lofront = reverse_complement_int(lo[i]);
            const unsigned int loback = reverse_complement_int(lo[intsPerHalf - 1 - i]);
            lo[i] = loback;
            lo[intsPerHalf - 1 - i] = lofront;
        }
        if(intsPerHalf % 2 == 1){
            const int middleindex = intsPerHalf/2;
            hi[middleindex] = reverse_complement_int(hi[middleindex]);
            lo[middleindex] = reverse_complement_int(lo[middleindex]);
        }

        if(unusedBitsInt != 0){
            for(int i = 0; i < intsPerHalf - 1; ++i){
                hi[i] = (hi[i] >> unusedBitsInt) | (hi[i+1] << (8 * sizeof(unsigned int) - unusedBitsInt));
                lo[i] = (lo[i] >> unusedBitsInt) | (lo[i+1] << (8 * sizeof(unsigned int) - unusedBitsInt));
            }

            hi[intsPerHalf - 1] >>= unusedBitsInt;
            lo[intsPerHalf - 1] >>= unusedBitsInt;
        }
    };

    //use threads in tile to shift bit-array array by shiftamounts bits to the left

    auto shiftEncodedBasesLeftBy = [](auto& tile, unsigned int* array, int size, int shiftamount, bool print){
        const int completeInts = shiftamount / (8 * sizeof(unsigned int));
#if 1
        for(int i = tile.thread_rank(); i < size - completeInts; i += tile.size()){
            array[i] = array[completeInts + i];
        }

        tile.sync();

        for(int i = size - completeInts + tile.thread_rank(); i < size; i += tile.size()){
            array[i] = 0;
        }

        tile.sync();

        shiftamount -= completeInts * 8 * sizeof(unsigned int);

        const int roundedIters = SDIV(size - completeInts - 1, tile.size()) * tile.size();
        for(int i = tile.thread_rank(); i < roundedIters; i += tile.size()){
            const unsigned int a = i < size - completeInts - 1 ? array[i] : 0;
            const unsigned int b = i < size - completeInts - 1 ? array[i+1] : 0;

            tile.sync();

            if(i < size - completeInts - 1)
                array[i] = (a >> shiftamount) | (b << (8 * sizeof(unsigned int) - shiftamount));

            tile.sync();
        }
        if(tile.thread_rank() == 0)
            array[size - completeInts - 1] >>= shiftamount;

        tile.sync();
#else
        if(tile.thread_rank() == 0){

            for(int i = 0; i < size - completeInts; i += 1){
                array[i] = array[completeInts + i];
                //if(print){
                //    printf("array1[%d] = %u\n", i, array[i]);
                //}
            }
            for(int i = size - completeInts; i < size; i += 1){
                array[i] = 0;
                //if(threadIdx.x / 4 == 7){
                //    printf("array[%d] = %u\n", i, array[i]);
                //}
            }

            shiftamount -= completeInts * 8 * sizeof(unsigned int);

            for(int i = 0; i < size - completeInts - 1; i += 1){
                //if(threadIdx.x >= 0 && blockIdx.x == 89)
                //    printf("tid %d, i %d, size %d, completeInts %d, shiftamount %d\n", threadIdx.x, i, size, completeInts, shiftamount);

                const unsigned int a = array[i];
                const unsigned int b = array[i+1];


                array[i] = (a >> shiftamount) | (b << (8 * sizeof(unsigned int) - shiftamount));
            }
            array[size - completeInts - 1] >>= shiftamount;
        }
        tile.sync();
#endif
    };

    //result is returned by thread with tile.thread_rank() == 0
    auto hammingdistanceHiLo = [](auto& tile,
                                const unsigned int* lhi,
                                const unsigned int* llo,
                                const unsigned int* rhi,
                                const unsigned int* rlo,
                                int lhi_bitcount,
                                int rhi_bitcount,
                                int max_errors){

        const int overlap_bitcount = lhi_bitcount < rhi_bitcount ? lhi_bitcount : rhi_bitcount;
        const int partitions = SDIV(overlap_bitcount, (8 * sizeof(unsigned int)));
        const int remaining_bitcount = partitions * sizeof(unsigned int) * 8 - overlap_bitcount;
#if 1
        int laneresult = 0;

        for(int i = tile.thread_rank(); i < partitions - 1; i += tile.size()){
            const unsigned int hixor = lhi[i] ^ rhi[i];
            const unsigned int loxor = llo[i] ^ rlo[i];
            const unsigned int bits = hixor | loxor;
            laneresult += __popc(bits);
        }

        // i == partitions - 1
        if(tile.thread_rank() == tile.size() - 1){
            const unsigned int mask = remaining_bitcount == 0 ? 0xFFFFFFFF : 0xFFFFFFFF >> (remaining_bitcount);
            const unsigned int hixor = lhi[partitions - 1] ^ rhi[partitions - 1];
            const unsigned int loxor = llo[partitions - 1] ^ rlo[partitions - 1];
            const unsigned int bits = hixor | loxor;
            laneresult += __popc(bits & mask);
        }

        int result = reduceTile(tile, laneresult, [](int l, int r){return l+r;});
#else
        int result = 0;

        if(tile.thread_rank() == 0){
            for(int i = 0; i < partitions - 1; i += 1){
                const unsigned int hixor = lhi[i] ^ rhi[i];
                const unsigned int loxor = llo[i] ^ rlo[i];
                const unsigned int bits = hixor | loxor;
                result += __popc(bits);
            }

            // i == partitions - 1

            const unsigned int mask = remaining_bitcount == 0 ? 0xFFFFFFFF : 0xFFFFFFFF >> (remaining_bitcount);
            const unsigned int hixor = lhi[partitions - 1] ^ rhi[partitions - 1];
            const unsigned int loxor = llo[partitions - 1] ^ rlo[partitions - 1];
            const unsigned int bits = hixor | loxor;
            result += __popc(bits & mask);
        }

        tile.sync();
#endif
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

    //set up shared memory per tile
    unsigned int* const myTileSubject = (unsigned int*)(sharedSubject + max_sequence_bytes * localTileId);
    unsigned int* const myTileQuery = (unsigned int*)(sharedQuery + max_sequence_bytes * localTileId);

    const int nQueries = NqueriesPrefixSum[Nsubjects];

    assert(max_sequence_bytes % sizeof(unsigned int) == 0);

    const int max_sequence_ints = max_sequence_bytes / sizeof(unsigned int);

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
        for(int lane = tile.thread_rank(); lane < max_sequence_ints; lane += tile.size()){
            myTileSubject[lane] = ((unsigned int*)(subjectsdata + subjectIndex * sequencepitch))[lane];
        }

        //save query in shared memory
        const int querybases = querylengths[queryIndex];
        for(int lane = tile.thread_rank(); lane < max_sequence_ints; lane += tile.size()){
            myTileQuery[lane] = ((unsigned int*)(queriesdata + queryIndex * sequencepitch))[lane];
        }

        tile.sync();

        //queryIndex != resultIndex -> reverse complement
        if(queryIndex != resultIndex && tile.thread_rank() == 0){
            make_reverse_complement_inplace(myTileQuery, querybases);
        }

        tile.sync();

        //begin SHD algorithm

        unsigned int* const query = myTileQuery;

        const int subjectints = getNumBytes(subjectbases) / sizeof(unsigned int);
        const int queryints = getNumBytes(querybases) / sizeof(unsigned int);
        const int totalbases = subjectbases + querybases;
        const int minoverlap = max(min_overlap, int(double(subjectbases) * min_overlap_ratio));

        //will only be valid for tile.thread_rank() == 0
        int bestScore = totalbases; // score is number of mismatches
        int bestShift = -querybases; // shift of query relative to subject. shift < 0 if query begins before subject

        unsigned int* subjectdata_hi = myTileSubject;
        unsigned int* subjectdata_lo = myTileSubject + subjectints / 2;
        unsigned int* querydata_hi = query;
        unsigned int* querydata_lo = query + queryints / 2;

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

            /*if(tile.thread_rank() == 0 && localTileId == 7 && resultIndex == 0){
                auto b2c = [](char c){
                    switch(c){
                        case 0x00: return 'A';
                        case 0x01: return 'C';
                        case 0x02: return 'G';
                        case 0x03: return 'T';
                        default: return 'F';
                    }
                };
                printf("before shift %d resultindex %d, queryindex %d, tileid %d\n", shift, resultIndex, queryIndex, localTileId);
                printf("myTileSubject\n");
                for(int i = 0; i < subjectbases; i++){
                    printf("%c", b2c(care::Sequence2BitHiLo::get((char*)myTileSubject, subjectbases, i)));
                }
                printf("\n");

                printf("query\n");
                for(int i = 0; i < querybases; i++){
                    printf("%c", b2c(care::Sequence2BitHiLo::get((char*)query, querybases, i)));
                }
                printf("\n");
            }
            __syncthreads();; //remove*/

            shiftEncodedBasesLeftBy(tile, subjectdata_hi, subjectints / 2, shiftamount, shift == 36 && localTileId == 7 && resultIndex == 0);
            shiftEncodedBasesLeftBy(tile, subjectdata_lo, subjectints / 2, shiftamount, shift == 36 && localTileId == 7 && resultIndex == 0);

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

        for(int lane = tile.thread_rank(); lane < max_sequence_ints; lane += tile.size()){
            myTileSubject[lane] = ((unsigned int*)(subjectsdata + subjectIndex * sequencepitch))[lane];
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

            shiftEncodedBasesLeftBy(tile, querydata_hi, queryints / 2, shiftamount, false);
            shiftEncodedBasesLeftBy(tile, querydata_lo, queryints / 2, shiftamount, false);

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
        unsigned long long warpreduced = *((unsigned long long*)&myval);

        //reuse allocated shared memory since sequence data is no longer used
        //unsigned long long* blockreducetmp = (unsigned long long*)smem;
        __shared__ unsigned long long blockreducetmp[16];

        auto func = [](unsigned long long a, unsigned long long b){
            return (*((int2*)&a)).x < (*((int2*)&b)).x ? a : b;
        };

        auto warptile = tiled_partition<32>(this_thread_block());

        for (unsigned int offset = warptile.size() / 2; offset > 0; offset /= 2){
            warpreduced = func(warpreduced, warptile.shfl_down(warpreduced, offset));
        }

        //unsigned long long warpreduced = reduceTile(warptile,
        //                            *((unsigned long long*)&myval),
        //                            func);
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


template<class B>
__global__
void
cuda_popcount_shifted_hamming_distance_with_revcompl_kernel2(Result_t* results,
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
                              B getNumBytes){

	auto no_bank_conflict_index = [&](int logical_index)->int{
	    return logical_index * blockDim.x;
	};

    auto make_reverse_complement_inplace = [&](unsigned int* sequence, int sequencelength){

        auto reverse_complement_int = [](auto n) {
            n = ((n >> 1) & 0x55555555) | ((n << 1) & 0xaaaaaaaa);
            n = ((n >> 2) & 0x33333333) | ((n << 2) & 0xcccccccc);
            n = ((n >> 4) & 0x0f0f0f0f) | ((n << 4) & 0xf0f0f0f0);
            n = ((n >> 8) & 0x00ff00ff) | ((n << 8) & 0xff00ff00);
            n = ((n >> 16) & 0x0000ffff) | ((n << 16) & 0xffff0000);
            return ~n;
        };

        const int ints = getNumBytes(sequencelength) / sizeof(unsigned int);
        const int unusedBitsInt = SDIV(sequencelength, 8 * sizeof(unsigned int)) * 8 * sizeof(unsigned int) - sequencelength;

        unsigned int* const hi = sequence;
        unsigned int* const lo = sequence + ints/2;

        const int intsPerHalf = SDIV(sequencelength, 8 * sizeof(unsigned int));
        for(int i = 0; i < intsPerHalf/2; ++i){
            const unsigned int hifront = reverse_complement_int(hi[i]);
            const unsigned int hiback = reverse_complement_int(hi[intsPerHalf - 1 - i]);
            hi[i] = hiback;
            hi[intsPerHalf - 1 - i] = hifront;

            const unsigned int lofront = reverse_complement_int(lo[i]);
            const unsigned int loback = reverse_complement_int(lo[intsPerHalf - 1 - i]);
            lo[i] = loback;
            lo[intsPerHalf - 1 - i] = lofront;
        }
        if(intsPerHalf % 2 == 1){
            const int middleindex = intsPerHalf/2;
            hi[middleindex] = reverse_complement_int(hi[middleindex]);
            lo[middleindex] = reverse_complement_int(lo[middleindex]);
        }

        if(unusedBitsInt != 0){
            for(int i = 0; i < intsPerHalf - 1; ++i){
                hi[i] = (hi[i] >> unusedBitsInt) | (hi[i+1] << (8 * sizeof(unsigned int) - unusedBitsInt));
                lo[i] = (lo[i] >> unusedBitsInt) | (lo[i+1] << (8 * sizeof(unsigned int) - unusedBitsInt));
            }

            hi[intsPerHalf - 1] >>= unusedBitsInt;
            lo[intsPerHalf - 1] >>= unusedBitsInt;
        }
    };

    //use threads in tile to shift bit-array array by shiftamounts bits to the left
    auto shiftEncodedBasesLeftBy = [&](unsigned int* array, int size, int shiftamount){
        const int completeInts = shiftamount / (8 * sizeof(unsigned int));

        for(int i = 0; i < size - completeInts; i += 1){
            array[no_bank_conflict_index(i)] = array[no_bank_conflict_index(completeInts + i)];
        }

        for(int i = size - completeInts; i < size; i += 1){
            array[no_bank_conflict_index(i)] = 0;
        }

        shiftamount -= completeInts * 8 * sizeof(unsigned int);

        for(int i = 0; i < size - completeInts - 1; i += 1){
            const unsigned int a = array[no_bank_conflict_index(i)];
            const unsigned int b = array[no_bank_conflict_index(i+1)];

            array[no_bank_conflict_index(i)] = (a >> shiftamount) | (b << (8 * sizeof(unsigned int) - shiftamount));
        }

        array[no_bank_conflict_index(size - completeInts - 1)] >>= shiftamount;
    };

    auto hammingdistanceHiLo = [&](
                                const unsigned int* lhi,
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

        for(int i = 0; i < partitions - 1 && result < max_errors; i += 1){
        	const unsigned int hixor = lhi[no_bank_conflict_index(i)] ^ rhi[no_bank_conflict_index(i)];
        	const unsigned int loxor = llo[no_bank_conflict_index(i)] ^ rlo[no_bank_conflict_index(i)];
        	const unsigned int bits = hixor | loxor;
        	result += __popc(bits);
        }

        if(result >= max_errors)
        	return result;

        // i == partitions - 1

        const unsigned int mask = remaining_bitcount == 0 ? 0xFFFFFFFF : 0xFFFFFFFF >> (remaining_bitcount);
        const unsigned int hixor = lhi[no_bank_conflict_index(partitions - 1)] ^ rhi[no_bank_conflict_index(partitions - 1)];
        const unsigned int loxor = llo[no_bank_conflict_index(partitions - 1)] ^ rlo[no_bank_conflict_index(partitions - 1)];
        const unsigned int bits = hixor | loxor;
        result += __popc(bits & mask);

        return result;
    };

    static_assert(sizeof(int2) == sizeof(unsigned long long), "sizeof(int2) != sizeof(unsigned long long)");

    constexpr int WARPSIZE = 32;

    // max_sequence_bytes * tiles_per_block * 3
    extern __shared__ unsigned int sharedmemory[];

    //const int warps_per_block = blockDim.x / WARPSIZE;

    const int tiles_per_block = blockDim.x;
    const int localTileId = threadIdx.x;
    const int warpId = threadIdx.x / WARPSIZE;

    //set up shared memory pointers
    char* const sharedSubject = (char*)(sharedmemory);
    char* const sharedQuery = (char*)(((char*)sharedSubject) + max_sequence_bytes * tiles_per_block);

    unsigned int* const subjectBackup = (unsigned int*)(((char*)sharedQuery) + max_sequence_bytes * tiles_per_block);
    unsigned int* const queryBackup = (unsigned int*)(((char*)subjectBackup) + max_sequence_bytes);

    //set up shared memory per tile
    unsigned int* const myTileSubject = (unsigned int*)(sharedSubject) + localTileId;
    unsigned int* const myTileQuery = (unsigned int*)(sharedQuery) + localTileId;

    //printf("%d %d %p %p %p %p %p %p %p\n", max_sequence_bytes, tiles_per_block, sharedmemory,
    //        sharedSubject, sharedQuery, subjectBackup, queryBackup, myTileSubject, myTileQuery);

//if(threadIdx.x == 0){
//	printf("tiles_per_block %d, localTileId %d, warpId %d, max_sequence_bytes %d\n",tiles_per_block, localTileId, warpId, max_sequence_bytes);
//}

    const int nQueries = NqueriesPrefixSum[Nsubjects];

    const int max_sequence_ints = max_sequence_bytes / sizeof(unsigned int);

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
        for(int lane = threadIdx.x; lane < max_sequence_ints; lane += blockDim.x){
            subjectBackup[lane] = ((unsigned int*)(subjectsdata + subjectIndex * sequencepitch))[lane];
        }

        //save query in shared memory
        const int querybases = querylengths[queryIndex];
        for(int lane = threadIdx.x; lane < max_sequence_ints; lane += blockDim.x){
            queryBackup[lane] = ((unsigned int*)(queriesdata + queryIndex * sequencepitch))[lane];
        }

        //queryIndex != resultIndex -> reverse complement
        if(threadIdx.x == 0 && queryIndex != resultIndex){
            make_reverse_complement_inplace(queryBackup, querybases);
        }

        __syncthreads();

        //begin SHD algorithm

        unsigned int* const query = myTileQuery;

        const int subjectints = getNumBytes(subjectbases) / sizeof(unsigned int);
        const int queryints = getNumBytes(querybases) / sizeof(unsigned int);
        const int totalbases = subjectbases + querybases;
        const int minoverlap = max(min_overlap, int(double(subjectbases) * min_overlap_ratio));

        //will only be valid for tile.thread_rank() == 0
        int bestScore = totalbases; // score is number of mismatches
        int bestShift = -querybases; // shift of query relative to subject. shift < 0 if query begins before subject

        unsigned int* subjectdata_hi = myTileSubject;
        unsigned int* subjectdata_lo = myTileSubject + subjectints / 2 * blockDim.x;
        unsigned int* querydata_hi = query;
        unsigned int* querydata_lo = query + queryints / 2 * blockDim.x;

        int previousShift = -querybases + minoverlap + localTileId;

        for(int shift = -querybases + minoverlap + localTileId; shift < subjectbases - minoverlap; shift += tiles_per_block){
            if(shift == -querybases + minoverlap + localTileId){
        		//save subject in shared memory
        		for(int lane = 0; lane < max_sequence_ints; lane += 1){
        		    myTileSubject[no_bank_conflict_index(lane)] = subjectBackup[lane];
        		}

        		//save query in shared memory
        		for(int lane = 0; lane < max_sequence_ints; lane += 1){
        		    myTileQuery[no_bank_conflict_index(lane)] = queryBackup[lane];
        		}
        	}else{
        		unsigned int* storeptr = previousShift > 0 ? myTileSubject : myTileQuery;
        		const unsigned int* loadptr = previousShift > 0 ? subjectBackup : queryBackup;

        		for(int lane = 0; lane < max_sequence_ints; lane += 1){
        		    storeptr[no_bank_conflict_index(lane)] = loadptr[lane];
        		}
        	}
        	const int overlapsize = min(querybases, subjectbases - shift) - max(-shift, 0);
        	const int max_errors = int(double(overlapsize) * maxErrorRate);

        	unsigned int* const shiftptr_hi = shift > 0 ? subjectdata_hi : querydata_hi;
        	unsigned int* const shiftptr_lo = shift > 0 ? subjectdata_lo : querydata_lo;
        	const int size = shift > 0 ? subjectints / 2 : queryints / 2;
        	const int shiftamount = abs(shift);

        	shiftEncodedBasesLeftBy(shiftptr_hi, size, shiftamount);
        	shiftEncodedBasesLeftBy(shiftptr_lo, size, shiftamount);

        	int score = hammingdistanceHiLo(
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

    		if(score < bestScore){
    			bestScore = score;
    			bestShift = shift;
    		}

    		previousShift = shift;
        }

        __syncthreads();

        /*
            The result per tile is present in thread tile.thread_rank() == 0;
            Each of those threads writes its result to shared memory.
            Then, a min reduce is performed on these results to determine the final result
        */

        // perform reduction to find smallest score in block. the corresponding shift is required, too
        // pack both score and shift into int2 and perform int2-reduction by only comparing the score

        int2 myval = make_int2(bestScore, bestShift);
        unsigned long long warpreduced = *((unsigned long long*)&myval);

        //reuse allocated shared memory since sequence data is no longer used
        unsigned long long* blockreducetmp = (unsigned long long*)sharedmemory;
        //__shared__ unsigned long long blockreducetmp[16];

        auto func = [](unsigned long long a, unsigned long long b){
            return (*((int2*)&a)).x < (*((int2*)&b)).x ? a : b;
        };

        auto warptile = tiled_partition<32>(this_thread_block());

        for (unsigned int offset = warptile.size() / 2; offset > 0; offset /= 2){
            warpreduced = func(warpreduced, warptile.shfl_down(warpreduced, offset));
        }

        //unsigned long long warpreduced = reduceTile(warptile,
        //                            *((unsigned long long*)&myval),
        //                            func);
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

template<class B>
void call_popcount_shd_with_revcompl_kernel_async(const SHDdata& shddata,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength,
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

      const int blocksize = 64;
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

template<class B>
void call_popcount_shd_with_revcompl_kernel(const SHDdata& shddata,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength,
                      B getNumBytes) noexcept{

    call_popcount_shd_with_revcompl_kernel_async(shddata,
                        min_overlap,
                        maxErrorRate,
                        min_overlap_ratio,
                        maxSubjectLength,
                        maxQueryLength,
                        getNumBytes);

    cudaStreamSynchronize(shddata.streams[0]); CUERR;
}


template<class B>
void call_popcount_shd_with_revcompl_kernel2_async(const SHDdata& shddata,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength,
                      B getNumBytes){

    assert(shddata.max_sequence_bytes % sizeof(unsigned int) == 0);

      const int minoverlap = std::max(min_overlap, int(double(maxSubjectLength) * min_overlap_ratio));
      const int maxSequenceLength = std::max(maxSubjectLength, maxQueryLength);

      const int blocksize = 64;

      dim3 block(blocksize, 1, 1);
      dim3 grid(shddata.n_queries*2, 1, 1); // one block per (query and its reverse complement)
      //dim3 grid(1, 1, 1); // one block per (query and its reverse complement)

      const std::size_t smem = sizeof(char) * (2 * shddata.max_sequence_bytes * blocksize + 2 * shddata.max_sequence_bytes);

      #define mycall cuda_popcount_shifted_hamming_distance_with_revcompl_kernel2 \
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
                                  getNumBytes); CUERR;

      mycall;

      #undef mycall
}



#endif //ifdef __NVCC__

}//namespace shd

#endif
