#ifndef SHIFTED_HAMMING_DISTANCE_HPP
#define SHIFTED_HAMMING_DISTANCE_HPP

#include "cudareduce.cuh"
#include "hpc_helpers.cuh"

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

    bool operator==(const AlignmentResult& rhs) const noexcept;
    bool operator!=(const AlignmentResult& rhs) const noexcept;
    int get_score() const noexcept;
    int get_subject_begin_incl() const noexcept;
    int get_query_begin_incl() const noexcept;
    int get_overlap() const noexcept;
    int get_shift() const noexcept;
    int get_nOps() const noexcept;
    bool get_isNormalized() const noexcept;
    bool get_isValid() const noexcept;
    int& get_score() noexcept;
    int& get_subject_begin_incl() noexcept;
    int& get_query_begin_incl() noexcept;
    int& get_overlap() noexcept;
    int& get_shift() noexcept;
    int& get_nOps() noexcept;
    bool& get_isNormalized() noexcept;
    bool& get_isValid() noexcept;
};

using Result_t = AlignmentResult;

struct SHDdata{

	Result_t* d_results = nullptr;
	char* d_subjectsdata = nullptr;
	char* d_queriesdata = nullptr;
	int* d_subjectlengths = nullptr;
	int* d_querylengths = nullptr;
    int* d_NqueriesPrefixSum = nullptr;

	Result_t* h_results = nullptr;
	char* h_subjectsdata = nullptr;
	char* h_queriesdata = nullptr;
	int* h_subjectlengths = nullptr;
	int* h_querylengths = nullptr;
    int* h_NqueriesPrefixSum = nullptr;

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
	int max_n_subjects = 0;
	int max_n_queries = 0;

    // if number of alignments to calculate is >= gpuThreshold, use GPU.
    int gpuThreshold = 0;

	void resize(int n_sub, int n_quer);
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

        int score = 0;

        for(int j = std::max(-shift, 0); j < std::min(querylength, subjectlength - shift); j++){
            score += getChar(subject, subjectlength, j + shift) != getChar(query, querylength, j);
        }

        score += totalbases - overlapsize;

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
    const int opnr = bestScore - totalbases + overlapsize;

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

        __syncthreads();

        //begin SHD algorithm

        const int minoverlap = max(min_overlap, int(double(subjectbases) * min_overlap_ratio));
        const int totalbases = subjectbases + querybases;

        int bestScore = totalbases; // score is number of mismatches
        int bestShift = -querybases; // shift of query relative to subject. shift < 0 if query begins before subject

        for(int shift = -querybases + minoverlap + threadIdx.x; shift < subjectbases - minoverlap; shift += BLOCKSIZE){
            const int overlapsize = min(querybases, subjectbases - shift) - max(-shift, 0);
            int score = 0;

            for(int j = max(-shift, 0); j < min(querybases, subjectbases - shift); j++){
                score += getChar(sharedSubject, subjectbases, j + shift) != getChar(sharedQuery, querybases, j);
            }
            score += totalbases - overlapsize; // non-overlapping regions count as mismatches

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
            const int opnr = bestScore - totalbases + overlapsize;

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

#endif //ifdef __NVCC__

}//namespace shd

#endif
