#ifndef SHIFTED_HAMMING_DISTANCE_HPP
#define SHIFTED_HAMMING_DISTANCE_HPP

#include "cudareduce.cuh"
#include "hpc_helpers.cuh"
#include "bestalignment.hpp"
#include "util.hpp"
#include "sequence.hpp"

#include <cstdint>
#include <algorithm>
#include <vector>
#include <bitset>

#if __CUDACC_VER_MAJOR__ >= 9
#include <cooperative_groups.h>
using namespace cooperative_groups;
#endif

namespace care{
    namespace msa{
        struct MSAColumnProperties;
    }
}

namespace shd{

struct AlignmentResult{
	int score;
	int overlap;
	int shift;
	int nOps; //edit distance / number of operations
	bool isValid;

    HOSTDEVICEQUALIFIER
    bool operator==(const AlignmentResult& rhs) const noexcept;
    HOSTDEVICEQUALIFIER
    bool operator!=(const AlignmentResult& rhs) const noexcept;
    HOSTDEVICEQUALIFIER
    int get_score() const noexcept;
    HOSTDEVICEQUALIFIER
    int get_overlap() const noexcept;
    HOSTDEVICEQUALIFIER
    int get_shift() const noexcept;
    HOSTDEVICEQUALIFIER
    int get_nOps() const noexcept;
    HOSTDEVICEQUALIFIER
    bool get_isValid() const noexcept;
    HOSTDEVICEQUALIFIER
    int& get_score() noexcept;
    HOSTDEVICEQUALIFIER
    int& get_overlap() noexcept;
    HOSTDEVICEQUALIFIER
    int& get_shift() noexcept;
    HOSTDEVICEQUALIFIER
    int& get_nOps() noexcept;
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
    char* d_unpacked_subjects = nullptr;
    char* d_unpacked_queries = nullptr;

    char* d_multiple_sequence_alignment = nullptr;
    float* d_multiple_sequence_alignment_weights = nullptr;
    char* d_consensus = nullptr;
    float* d_support = nullptr;
    int* d_coverage = nullptr;
    float* d_origWeights = nullptr;
    int* d_origCoverages = nullptr;
    char* d_qualityscores = nullptr;
    care::msa::MSAColumnProperties* d_msa_column_properties = nullptr;

	Result_t* h_results = nullptr;
	char* h_subjectsdata = nullptr;
	char* h_queriesdata = nullptr;
	int* h_subjectlengths = nullptr;
	int* h_querylengths = nullptr;
    int* h_NqueriesPrefixSum = nullptr;
    BestAlignment_t* h_bestAlignmentFlags = nullptr;
    char* h_unpacked_subjects = nullptr;
    char* h_unpacked_queries = nullptr;

    char* h_multiple_sequence_alignment = nullptr;
    float* h_multiple_sequence_alignment_weights = nullptr;
    char* h_consensus = nullptr;
    float* h_support = nullptr;
    int* h_coverage = nullptr;
    float* h_origWeights = nullptr;
    int* h_origCoverages = nullptr;
    char* h_qualityscores = nullptr;
    care::msa::MSAColumnProperties* h_msa_column_properties = nullptr;

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

    std::size_t memSubjects;
    std::size_t memSubjectLengths;
    std::size_t memNqueriesPrefixSum;
    std::size_t memQueries;
    std::size_t memQueryLengths;
    std::size_t memResults;
    std::size_t memBestAlignmentFlags;
    std::size_t memUnpackedSubjects;
    std::size_t memUnpackedQueries;
    std::size_t memMultipleSequenceAlignment;
    std::size_t memMultipleSequenceAlignmentWeights;
    std::size_t memConsensus;
    std::size_t memSupport;
    std::size_t memCoverage;
    std::size_t memQualityScores;
    std::size_t memOrigWeights;
    std::size_t memOrigCoverage;
    std::size_t memMSAColumnProperties;

    std::size_t msa_row_pitch;
    std::size_t msa_weights_row_pitch;

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
    result.overlap = overlapsize;
    result.shift = bestShift;
    result.nOps = opnr;

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
    result.overlap = overlapsize;
    result.shift = bestShift;
    result.nOps = opnr;

    return result;
}

/*
    GPU alignment
*/
#ifdef __NVCC__

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
        if(queryIndex != resultIndex){
            if(threadIdx.x == 0){
                make_reverse_complement_inplace((std::uint8_t*)sharedQuery, querybases);
            }
            __syncthreads();
        }



        //begin SHD algorithm

        const char* query = sharedQuery;

        const int minoverlap = max(min_overlap, int(double(subjectbases) * min_overlap_ratio));
        const int totalbases = subjectbases + querybases;

        int bestScore = totalbases; // score is number of mismatches
        int bestShift = -querybases; // shift of query relative to subject. shift < 0 if query begins before subject

        for(int shift = -querybases + minoverlap + threadIdx.x; shift < subjectbases - minoverlap + 1; shift += BLOCKSIZE){
            const int overlapsize = min(querybases, subjectbases - shift) - max(-shift, 0);
            const int max_errors = int(double(overlapsize) * maxErrorRate);
            int score = 0;

            for(int j = max(-shift, 0); j < min(querybases, subjectbases - shift) && score < max_errors; j++){
                score += getChar(sharedSubject, subjectbases, j + shift) != getChar(query, querybases, j);
            }

            score = (score < max_errors ?
                    score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                    : std::numeric_limits<int>::max()); // too many errors, discard

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

        auto tile = tiled_partition<32>(this_thread_block());
        unsigned long long tilereduced = reduceTile(tile,
                                    *((unsigned long long*)&myval),
                                    func);
        int warp = threadIdx.x / WARPSIZE;
        if(tile.thread_rank() == 0)
            blockreducetmp[warp] = tilereduced;

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
            result.overlap = overlapsize;
            result.shift = bestShift;
            result.nOps = opnr;

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
      const int maxShiftsToCheck = maxSubjectLength + maxQueryLength - 2*minoverlap;
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





#if 1

template<class T>
__device__ __forceinline__
char get2bitenc(const T data, int nBases, int i) noexcept{
    const int byte = i / 4;
    const int basepos = i % 4;
    return (char)((data[byte] >> (3-basepos) * 2) & 0x03);
}

template<class T>
__device__ __forceinline__
void make_reverse_complement_inplace_2bitenc(T sequence, int sequencelength){

    auto getNumBytes = [](int nBases){
        return SDIV(nBases, 4);
    };

    auto make_reverse_complement_byte = [](std::uint8_t in) -> std::uint8_t{
        in = ((in >> 2)  & 0x33) | ((in & 0x33) << 2);
        in = ((in >> 4)  & 0x0F) | ((in & 0x0F) << 4);
        return (std::uint8_t(-1) - in) >> (8 * 1 - (4 << 1));
    };

    const int bytes = getNumBytes(sequencelength);
    const int unusedPositions = bytes * 4 - sequencelength;

    for(int i = 0; i < bytes/2; i++){
        const std::uint8_t front = make_reverse_complement_byte(sequence[i]);
        const std::uint8_t back = make_reverse_complement_byte(sequence[bytes - 1 - i]);
        sequence[i] = back;
        sequence[bytes - 1 - i] = front;
    }

    if(bytes % 2 == 1){
        const int middleindex = bytes/2;
        sequence[middleindex] = make_reverse_complement_byte(sequence[middleindex]);
    }

    if(unusedPositions > 0){
        sequence[0] <<= (2 * unusedPositions);
        for(int i = 1; i < bytes; i++){
            sequence[i-1] |= sequence[i] >> (2 * (4-unusedPositions));
            sequence[i] <<= (2 * unusedPositions);
        }
    }

};

template<class R, class T, class IndexFunc>
struct linearmapper{
    T ptr;
    IndexFunc func;

    __device__
    linearmapper(T ptr, IndexFunc func) : ptr(ptr), func(func){}

    __device__
    R operator[](int i) const{
        return *((R*)(ptr + func(i)));
    }

    __device__
    R& operator[](int i){
        return *((R*)(ptr + func(i)));
    }
};

/*
    Kernel which uses one thread per query, each thread has its own smem to store subject and query
*/
template<int BLOCKSIZE, class Accessor, class RevCompl>
__global__
void
cuda_shifted_hamming_distance_with_revcompl_kernel_new(Result_t* results,
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

    static_assert(sizeof(int2) == sizeof(unsigned long long), "sizeof(int2) != sizeof(unsigned long long)");
    static_assert(BLOCKSIZE % WARPSIZE == 0,
        "BLOCKSIZE must be multiple of WARPSIZE");

    auto smem_index = [](int i) -> int{
        return i * BLOCKSIZE;
    };

    extern __shared__ char smem[];

    //set up shared memory pointers
    char* const sharedSubjects = (char*)(smem);
    char* const sharedQueries = (char*)(sharedSubjects + BLOCKSIZE * max_sequence_bytes);

    char* const mySubject = sharedSubjects + threadIdx.x;
    char* const myQuery = sharedQueries + threadIdx.x;

    auto mySubject_linear_uchar = linearmapper<std::uint8_t, char*, decltype(smem_index)>((char*)mySubject, smem_index);
    auto myQuery_linear_uchar = linearmapper<std::uint8_t, char*, decltype(smem_index)>((char*)myQuery, smem_index);

    const int nQueries = NqueriesPrefixSum[Nsubjects];

    for(unsigned resultIndex = threadIdx.x + blockDim.x * blockIdx.x; resultIndex < nQueries * 2; resultIndex += blockDim.x * gridDim.x){
        const int queryIndex = resultIndex < nQueries ? resultIndex : resultIndex - nQueries;

        int subjectIndex = 0;
        for(; subjectIndex < Nsubjects; subjectIndex++){
            if(queryIndex < NqueriesPrefixSum[subjectIndex+1])
                break;
        }

        const int subjectbases = subjectlengths[subjectIndex];
        const int querybases = querylengths[queryIndex];

        for(int lane = 0; lane < max_sequence_bytes; lane += 1){
            mySubject[smem_index(lane)] = subjectsdata[subjectIndex * sequencepitch + lane];
            myQuery[smem_index(lane)] = queriesdata[queryIndex * sequencepitch + lane];
        }

        if(queryIndex != resultIndex){
            make_reverse_complement_inplace_2bitenc(myQuery_linear_uchar, querybases);
        }

        const int minoverlap = max(min_overlap, int(double(subjectbases) * min_overlap_ratio));
        const int totalbases = subjectbases + querybases;

        int bestScore = totalbases; // score is number of mismatches
        int bestShift = -querybases; // shift of query relative to subject. shift < 0 if query begins before subject

        for(int shift = -querybases + minoverlap; shift < subjectbases - minoverlap; shift += 1){
            const int overlapsize = min(querybases, subjectbases - shift) - max(-shift, 0);
            const int max_errors = int(double(overlapsize) * maxErrorRate);
            int score = 0;

            for(int j = max(-shift, 0); j < min(querybases, subjectbases - shift) && score < max_errors; j++){
                score += get2bitenc(mySubject_linear_uchar, subjectbases, j + shift) != get2bitenc(myQuery_linear_uchar, querybases, j);
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

        result.isValid = (bestShift != -querybases);
        const int queryoverlapbegin_incl = max(-bestShift, 0);
        const int queryoverlapend_excl = min(querybases, subjectbases - bestShift);
        const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
        const int opnr = bestScore - totalbases + 2*overlapsize;

        result.score = bestScore;
        result.overlap = overlapsize;
        result.shift = bestShift;
        result.nOps = opnr;

        results[resultIndex] = result;

    }
}

template<class Accessor, class RevCompl>
void call_shd_with_revcompl_kernel_new_async(const SHDdata& shddata,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength,
                      Accessor accessor,
                      RevCompl make_reverse_complement){

    int blocksize = 64;

    dim3 block(blocksize, 1, 1);
    dim3 grid(SDIV(shddata.n_queries*2, block.x), 1, 1); // one block per (query and its reverse complement)

    const std::size_t smem = sizeof(char) * 2 * blocksize * shddata.max_sequence_bytes;

    #define mycall(blocksize) cuda_shifted_hamming_distance_with_revcompl_kernel_new<(blocksize)> \
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

    switch(blocksize){
    case 32: mycall(32); break;
    case 64: mycall(64); break;
    case 96: mycall(96); break;
    case 128: mycall(128); break;
    case 160: mycall(160); break;
    case 192: mycall(192); break;
    case 224: mycall(224); break;
    case 256: mycall(256); break;
    default: throw std::runtime_error("Want to call shd kernel with invalid block size.");
    }

    #undef mycall
}






/*
    Kernel for fixed length sequences which stores sequences in local array
*/
template<int BLOCKSIZE, class Accessor, class RevCompl>
__global__
void
cuda_shifted_hamming_distance_with_revcompl_kernel_new2(Result_t* results,
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

    constexpr int SEQUENCELENGTH = 101;
    constexpr int SEQUENCEBYTES = SDIV(SEQUENCELENGTH, 4);

    //assert(max_sequence_bytes == SEQUENCEBYTES);

    /*auto get2bitenc_unroll = [&](const char* data, int nBases, int i){
        const int byte = i / 4;
        const int basepos = i % 4;
        return (char)(((unsigned char)data[byte] >> (3-basepos) * 2) & 0x03);
    };*/

    auto make_reverse_complement_byte = [](std::uint8_t in) -> std::uint8_t{
        in = ((in >> 2)  & 0x33) | ((in & 0x33) << 2);
        in = ((in >> 4)  & 0x0F) | ((in & 0x0F) << 4);
        return (std::uint8_t(-1) - in) >> (8 * 1 - (4 << 1));
    };

    auto make_reverse_complement_int = [](std::uint32_t s){
        s = ((s >> 2)  & 0x33333333u) | ((s & 0x33333333u) << 2);
        s = ((s >> 4)  & 0x0F0F0F0Fu) | ((s & 0x0F0F0F0Fu) << 4);
        s = ((s >> 8)  & 0x00FF00FFu) | ((s & 0x00FF00FFu) << 8);
        s = ((s >> 16) & 0x0000FFFFu) | ((s & 0x0000FFFFu) << 16);
        return (std::uint32_t(-1) - s) >> (8 * sizeof(s) - (16 << 1));
    };

    char mySubject[26];
    char myQuery[26];

    const int nQueries = NqueriesPrefixSum[Nsubjects];

    for(unsigned resultIndex = threadIdx.x + blockDim.x * blockIdx.x; resultIndex < nQueries * 2; resultIndex += blockDim.x * gridDim.x){
        const int queryIndex = resultIndex < nQueries ? resultIndex : resultIndex - nQueries;

        int subjectIndex = 0;
        for(; subjectIndex < Nsubjects; subjectIndex++){
            if(queryIndex < NqueriesPrefixSum[subjectIndex+1])
                break;
        }

        #pragma unroll
        for(int lane = 0; lane < SEQUENCEBYTES; lane += 1){
            mySubject[lane] = subjectsdata[subjectIndex * sequencepitch + lane];
            myQuery[lane] = queriesdata[queryIndex * sequencepitch + lane];
        }

        if(queryIndex != resultIndex){
            const int unusedPositions = SEQUENCEBYTES * 4 - SEQUENCELENGTH;

            #pragma unroll
            for(int i = 0; i < SEQUENCEBYTES/2; i++){
                const std::uint8_t front = make_reverse_complement_byte(myQuery[i]);
                const std::uint8_t back = make_reverse_complement_byte(myQuery[SEQUENCEBYTES - 1 - i]);
                myQuery[i] = back;
                myQuery[SEQUENCEBYTES - 1 - i] = front;
            }

            if(SEQUENCEBYTES % 2 == 1){
                const int middleindex = SEQUENCEBYTES/2;
                myQuery[middleindex] = make_reverse_complement_byte(myQuery[middleindex]);
            }

            if(unusedPositions > 0){
                myQuery[0] <<= (2 * unusedPositions);
                #pragma unroll
                for(int i = 1; i < SEQUENCEBYTES; i++){
                    myQuery[i-1] |= myQuery[i] >> (2 * (4-unusedPositions));
                    myQuery[i] <<= (2 * unusedPositions);
                }
            }
        }

        const int minoverlap = max(min_overlap, int(double(SEQUENCELENGTH) * min_overlap_ratio));
        const int totalbases = SEQUENCELENGTH + SEQUENCELENGTH;

        int bestScore = totalbases; // score is number of mismatches
        int bestShift = -SEQUENCELENGTH; // shift of query relative to subject. shift < 0 if query begins before subject

        #pragma unroll 4
        for(int shift = -SEQUENCELENGTH; shift < SEQUENCELENGTH; shift += 1){

            if(shift >= -SEQUENCELENGTH + minoverlap && shift < SEQUENCELENGTH - minoverlap){

                const int overlapsize = min(SEQUENCELENGTH, SEQUENCELENGTH - shift) - max(-shift, 0);
                const int max_errors = int(double(overlapsize) * maxErrorRate);
                int score = 0;

                #pragma unroll 4
                for(int j = max(-shift, 0); j < min(SEQUENCELENGTH, SEQUENCELENGTH - shift); j++){
                    const int byteSub = (j + shift) / 4;
                    const int baseposSub = (j + shift) % 4;
                    const int byteQuer = (j) / 4;
                    const int baseposQuer = (j) % 4;
                    const unsigned char charSub = ((unsigned char)mySubject[byteSub] >> (3-baseposSub) * 2) & 0x03;
                    const unsigned char charQuer = ((unsigned char)myQuery[byteQuer] >> (3-baseposQuer) * 2) & 0x03;

                    score += charSub != charQuer;
                    //if(score >= max_errors)
                    //    break;
                }

                score = (score < max_errors ?
                        score + totalbases - 2*overlapsize // non-overlapping regions count as mismatches
                        : std::numeric_limits<int>::max()); // too many errors, discard

                if(score < bestScore){
                    bestScore = score;
                    bestShift = shift;
                }
            }


        }

        Result_t result;

        result.isValid = (bestShift != -SEQUENCELENGTH);
        const int queryoverlapbegin_incl = max(-bestShift, 0);
        const int queryoverlapend_excl = min(SEQUENCELENGTH, SEQUENCELENGTH - bestShift);
        const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
        const int opnr = bestScore - totalbases + 2*overlapsize;

        result.score = bestScore;
        result.overlap = overlapsize;
        result.shift = bestShift;
        result.nOps = opnr;

        results[resultIndex] = result;

    }
}

template<class Accessor, class RevCompl>
void call_shd_with_revcompl_kernel_new2_async(const SHDdata& shddata,
                      int min_overlap,
                      double maxErrorRate,
                      double min_overlap_ratio,
                      int maxSubjectLength,
                      int maxQueryLength,
                      Accessor accessor,
                      RevCompl make_reverse_complement){

    int blocksize = 128;

    dim3 block(blocksize, 1, 1);
    dim3 grid(SDIV(shddata.n_queries*2, block.x), 1, 1); // one block per (query and its reverse complement)

    const std::size_t smem = 0;

    #define mycall(blocksize) cuda_shifted_hamming_distance_with_revcompl_kernel_new2<(blocksize)> \
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

    switch(blocksize){
    case 32: mycall(32); break;
    case 64: mycall(64); break;
    case 96: mycall(96); break;
    case 128: mycall(128); break;
    case 160: mycall(160); break;
    case 192: mycall(192); break;
    case 224: mycall(224); break;
    case 256: mycall(256); break;
    default: throw std::runtime_error("Want to call shd kernel with invalid block size.");
    }

    #undef mycall
}















#endif





template<class B>
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
            result.overlap = overlapsize;
            result.shift = bestShift;
            result.nOps = opnr;

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

    assert(shddata.max_sequence_bytes % sizeof(unsigned int) == 0);

      const int minoverlap = std::max(min_overlap, int(double(maxSubjectLength) * min_overlap_ratio));
      const int maxSequenceLength = std::max(maxSubjectLength, maxQueryLength);

      const int blocksize = 64;

      dim3 block(blocksize, 1, 1);
      dim3 grid(shddata.n_queries*2, 1, 1); // one block per (query and its reverse complement)
      //dim3 grid(1, 1, 1); // one block per (query and its reverse complement)

      const std::size_t smem = sizeof(char) * (2 * shddata.max_sequence_bytes * blocksize + 2 * shddata.max_sequence_bytes);

      #define mycall cuda_popcount_shifted_hamming_distance_with_revcompl_kernel \
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
