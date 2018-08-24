#ifndef SHIFTED_HAMMING_DISTANCE_HPP
#define SHIFTED_HAMMING_DISTANCE_HPP

#include "cudareduce.cuh"
#include "hpc_helpers.cuh"
#include "bestalignment.hpp"

//#include "util.hpp"

#include <cstdint>
#include <algorithm>

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

#if 0
template<class Accessor>
Result_t
cpu_shifted_hamming_distance_new(const char* subject,
                            int subjectlength,
                            const char* query,
                            int querylength,
                            int min_overlap,
                            double maxErrorRate,
                            double min_overlap_ratio,
                            Accessor getChar
                            B getNumBytes) noexcept{

    const int totalbases = subjectlength + querylength;
    const int minoverlap = std::max(min_overlap, int(double(subjectlength) * min_overlap_ratio));
    int bestScore = totalbases; // score is number of mismatches
    int bestShift = -querylength; // shift of query relative to subject. shift < 0 if query begins before subject

    std::vector<char> subjectdata;
    std::vector<char> querydata;

    subjectdata.reserve(getNumBytes(subjectlength));
    querydata.reserve(getNumBytes(querylength));

    for(int shift = -querylength + minoverlap; shift < subjectlength - minoverlap; shift++){
        const int overlapsize = std::min(querylength, subjectlength - shift) - std::max(-shift, 0);
        const int max_errors = int(double(overlapsize) * maxErrorRate);

        subjectdata.insert(subjectdata.begin(), subject, subject + getNumBytes(subjectlength));
        querydata.insert(querydata.begin(), query, query + getNumBytes(querylength));

        shiftBitsBy(querydata.data(), getNumBytes(querylength), shift);

        int score = hammingdistanceHiLo(subjectdata.data(), querydata.data(), sbases, qbases, getNumBytes(subjectlength));

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





#endif //ifdef __NVCC__

}//namespace shd

#endif
