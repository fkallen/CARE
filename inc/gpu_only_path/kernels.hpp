#ifndef CARE_GPU_KERNELS_HPP
#define CARE_GPU_KERNELS_HPP

#include <stdexcept>

namespace care{
namespace gpu{

    template<int BLOCKSIZE, class Accessor, class RevCompl>
    __global__
    void
    cuda_shifted_hamming_distance_with_revcompl_kernel(
                                    int* alignment_scores,
                                    int* alignment_overlaps,
                                    int* alignment_shifts,
                                    int* alignment_nOps,
                                    bool* alignment_isValid,
                                    const char* subject_sequences_data,
                                    const char* candidate_sequences_data,
                                    const int* subject_sequences_lengths,
                                    const int* candidate_sequences_lengths,
                                    const int* candidates_per_subject_prefixsum,
                                    int n_subjects,
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

        const int nQueries = candidates_per_subject_prefixsum[n_subjects];

        for(unsigned resultIndex = blockIdx.x; resultIndex < nQueries * 2; resultIndex += gridDim.x){

            const int queryIndex = resultIndex < nQueries ? resultIndex : resultIndex - nQueries;

            //find subjectindex
            int subjectIndex = 0;
            for(; subjectIndex < n_subjects; subjectIndex++){
                if(queryIndex < candidates_per_subject_prefixsum[subjectIndex+1])
                    break;
            }

            //save subject in shared memory
            const int subjectbases = subject_sequences_lengths[subjectIndex];
            for(int threadid = threadIdx.x; threadid < max_sequence_bytes; threadid += BLOCKSIZE){
                sharedSubject[threadid] = subject_sequences_data[subjectIndex * sequencepitch + threadid];
            }

            //save query in shared memory
            const int querybases = candidate_sequences_lengths[queryIndex];
            for(int threadid = threadIdx.x; threadid < max_sequence_bytes; threadid += BLOCKSIZE){
                sharedQuery[threadid] = candidate_sequences_data[queryIndex * sequencepitch + threadid];
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

            for(int shift = -querybases + minoverlap + threadIdx.x; shift < subjectbases - minoverlap; shift += BLOCKSIZE){
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

                const int queryoverlapbegin_incl = max(-bestShift, 0);
                const int queryoverlapend_excl = min(querybases, subjectbases - bestShift);
                const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
                const int opnr = bestScore - totalbases + 2*overlapsize;

                alignment_scores[resultIndex] = bestScore;
                alignment_overlaps[resultIndex] = overlapsize;
                alignment_shifts[resultIndex] = bestShift;
                alignment_nOps[resultIndex] = opnr;
                alignment_isValid[resultIndex] = (bestShift != -querybases);
            }
        }
    }

    template<class Accessor, class RevCompl>
    void call_shd_with_revcompl_kernel_async(
                            int* d_alignment_scores,
                            int* d_alignment_overlaps,
                            int* d_alignment_shifts,
                            int* d_alignment_nOps,
                            bool* d_alignment_isValid,
                            const char* d_subject_sequences_data,
                            const char* d_candidate_sequences_data,
                            const int* d_subject_sequences_lengths,
                            const int* d_candidate_sequences_lengths,
                            const int* d_candidates_per_subject_prefixsum,
                            int n_subjects,
                            int max_sequence_bytes,
                            size_t encodedsequencepitch,
                            int min_overlap,
                            double maxErrorRate,
                            double min_overlap_ratio,
                            Accessor getChar,
                            RevCompl make_reverse_complement_inplace,
                            int n_queries,
                            int maxSubjectLength,
                            int maxQueryLength,
                            cudaStream_t stream){

          const int minoverlap = max(min_overlap, int(double(maxSubjectLength) * min_overlap_ratio));
          const int maxShiftsToCheck = maxSubjectLength + maxQueryLength - 2*minoverlap;
          dim3 block(std::min(256, 32 * SDIV(maxShiftsToCheck, 32)), 1, 1);
          dim3 grid(n_queries*2, 1, 1); // one block per (query and its reverse complement)

          const std::size_t smem = sizeof(char) * 2 * max_sequence_bytes;

          #define mycall(blocksize) cuda_shifted_hamming_distance_with_revcompl_kernel<(blocksize)> \
                                      <<<grid, block, smem, stream>>>( \
                                          d_alignment_scores,
                                          d_alignment_overlaps,
                                          d_alignment_shifts,
                                          d_alignment_nOps,
                                          d_alignment_isValid,
                                          d_subject_sequences_data,
                                          d_candidate_sequences_data,
                                          d_subject_sequences_lengths,
                                          d_candidate_sequences_lengths,
                                          d_candidates_per_subject_prefixsum,
                                          n_subjects, \
                                          max_sequence_bytes, \
                                          encodedsequencepitch, \
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

}
}


#endif
