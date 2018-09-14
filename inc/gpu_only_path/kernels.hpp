#ifndef CARE_GPU_KERNELS_HPP
#define CARE_GPU_KERNELS_HPP

#include "../hpc_helpers.cuh"
#include "bestalignment.hpp"
#include "../qualityscoreweights.hpp"

#include <stdexcept>



namespace care{
namespace gpu{

#ifdef __NVCC__

    void call_msa_init_kernel_async(
                        MSAColumnProperties* d_msa_column_properties,
                        const int* d_alignment_shifts,
                        const BestAlignment_t* d_alignment_best_alignment_flags,
                        const int* d_subject_sequences_lengths,
                        const int* d_candidate_sequences_lengths,
                        const int* d_indices,
                        const int* d_indices_per_subject,
                        const int* d_indices_per_subject_prefixsum,
                        int n_subjects,
                        int n_queries,
                        cudaStream_t stream);

    void call_msa_find_consensus_kernel_async(
                            char* const d_consensus,
                            float* const d_support,
                            int* const d_coverage,
                            float* const d_origWeights,
                            int* const d_origCoverages,
                            const char* const d_multiple_sequence_alignments,
                            const float* const d_multiple_sequence_alignment_weights,
                            const MSAColumnProperties* const d_msa_column_properties,
                            const int* const d_candidates_per_subject_prefixsum,
                            const int* d_indices_per_subject,
                            const int* d_indices_per_subject_prefixsum,
                            int n_subjects,
                            int n_queries,
                            int n_indices,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            int msa_max_column_count,
                            cudaStream_t stream);

    void call_msa_correct_subject_kernel_async(
                            const char* d_consensus,
                            const float* d_support,
                            const int* d_coverage,
                            const int* d_origCoverages,
                            const char* d_multiple_sequence_alignments,
                            const MSAColumnProperties* const d_msa_column_properties,
                            const int* d_indices_per_subject_prefixsum,
                            bool* d_is_high_quality_subject,
                            char* d_corrected_subjects,
                            int n_subjects,
                            int n_queries,
                            int n_indices,
                            size_t sequence_pitch,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            double estimatedErrorrate,
                            double avg_support_threshold,
                            double min_support_threshold,
                            double min_coverage_threshold,
                            int k_region,
                            int maximum_sequence_length,
                            cudaStream_t stream);


    /*
        SHIFTED HAMMING DISTANCE
    */

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



    /*
        FIND BEST ALIGNMENT
    */

    template<class AlignmentComp>
    __global__
    void cuda_find_best_alignment_kernel(
                                        BestAlignment_t* d_alignment_best_alignment_flags,
                                        int* d_alignment_scores
                                        int* d_alignment_overlaps,
                                        int* d_alignment_shifts
                                        int* d_alignment_nOps,
                                        bool* d_alignment_isValid,
                                        const int* d_subject_sequences_lengths,
                                        const int* d_candidate_sequences_lengths,
                                        const int* d_candidates_per_subject_prefixsum,
                                        int n_subjects,
                                        AlignmentComp comp){

        const int n_queries = NqueriesPrefixSum[Nsubjects];

        for(unsigned resultIndex = threadIdx.x + blockDim.x * blockIdx.x; resultIndex < n_queries; resultIndex += gridDim.x * blockDim.x){
            const unsigned fwdIndex = resultIndex;
            const unsigned revcIndex = resultIndex + n_queries;

            const int fwd_alignment_score = d_alignment_scores[fwdIndex];
            const int fwd_alignment_overlap = d_alignment_overlaps[fwdIndex];
            const int fwd_alignment_shift = d_alignment_shifts[fwdIndex];
            const int fwd_alignment_nops = d_alignment_nOps[fwdIndex];
            const bool fwd_alignment_isvalid = d_alignment_isValid[fwdIndex];

            const int revc_alignment_score = d_alignment_scores[revcIndex];
            const int revc_alignment_overlap = d_alignment_overlaps[revcIndex];
            const int revc_alignment_shift = d_alignment_shifts[revcIndex];
            const int revc_alignment_nops = d_alignment_nOps[revcIndex];
            const bool revc_alignment_isvalid = d_alignment_isValid[revcIndex];

            const int querylength = d_candidate_sequences_lengths[resultIndex];

            //find subjectindex
            int subjectIndex = 0;
            for(; subjectIndex < Nsubjects; subjectIndex++){
                if(resultIndex < NqueriesPrefixSum[subjectIndex+1])
                    break;
            }

            const int subjectlength = d_subject_sequences_lengths[subjectIndex];

            const BestAlignment_t flag = comp(fwd_alignment_overlap,
                                                revc_alignment_overlap,
                                                fwd_alignment_nops,
                                                revc_alignment_nops,
                                                fwd_alignment_isvalid,
                                                revc_alignment_isvalid,
                                                subjectlength,
                                                querylength);

            bestAlignmentFlags[resultIndex] = flag;

            d_alignment_scores[resultIndex] = flag == BestAlignment_t::Forward ? fwd_alignment_score : revc_alignment_score;
            d_alignment_overlaps[resultIndex] = flag == BestAlignment_t::Forward ? fwd_alignment_overlap : revc_alignment_overlap;
            d_alignment_shifts[resultIndex] = flag == BestAlignment_t::Forward ? fwd_alignment_shift : revc_alignment_shift;
            d_alignment_nOps[resultIndex] = flag == BestAlignment_t::Forward ? fwd_alignment_nops : revc_alignment_nops;
            d_alignment_isValid[resultIndex] = flag == BestAlignment_t::Forward ? fwd_alignment_isvalid : revc_alignment_isvalid;
        }
    }

    template<class AlignmentComp>
    void call_cuda_find_best_alignment_kernel_async(
                                        BestAlignment_t* d_alignment_best_alignment_flags,
                                        int* d_alignment_scores
                                        int* d_alignment_overlaps,
                                        int* d_alignment_shifts
                                        int* d_alignment_nOps,
                                        bool* d_alignment_isValid,
                                        const int* d_subject_sequences_lengths,
                                        const int* d_candidate_sequences_lengths,
                                        const int* d_candidates_per_subject_prefixsum,
                                        int n_subjects,
                                        AlignmentComp d_comp,
                                        int n_queries,
                                        cudaStream_t stream){

        dim3 block(128,1,1);
        dim3 grid(SDIV(n_queries, block.x), 1, 1);

        cuda_find_best_alignment_kernel<<<grid, block, 0, stream>>>(
                                                d_alignment_best_alignment_flags,
                                                d_alignment_scores
                                                d_alignment_overlaps,
                                                d_alignment_shifts
                                                d_alignment_nOps,
                                                d_alignment_isValid,
                                                d_subject_sequences_lengths,
                                                d_candidate_sequences_lengths,
                                                d_candidates_per_subject_prefixsum,
                                                n_subjects,
                                                d_comp); CUERR;

    }


    template<class Accessor>
    __global__
    void msa_add_sequences_kernel(
                            char* __restrict__ d_multiple_sequence_alignments,
                            float* __restrict__ d_multiple_sequence_alignment_weights,
                            const int* __restrict__ d_alignment_shifts,
                            const BestAlignment_t* __restrict__ d_alignment_best_alignment_flags,
                            const char* __restrict__ d_subject_sequences_data,
                            const char* __restrict__ d_candidate_sequences_data,
                            const int* __restrict__ d_subject_sequences_lengths,
                            const int* __restrict__ d_candidate_sequences_lengths,
                            const char* __restrict__ d_subject_qualities,
                            const char* __restrict__ d_candidate_qualities,
                            const MSAColumnProperties*  __restrict__ d_msa_column_properties,
                            const int* __restrict__ d_candidates_per_subject_prefixsum,
                            const int* __restrict__ d_indices,
                            const int* __restrict__ d_indices_per_subject,
                            const int* __restrict__ d_indices_per_subject_prefixsum,
                            int n_subjects,
                            int n_queries,
                            int n_indices,
                            bool canUseQualityScores,
                            size_t encoded_sequence_pitch,
                            size_t quality_pitch
                            size_t msa_row_pitch,
                            size_t msa_weights_row_pitch,
                            Accessor get_as_nucleotide){

        const size_t msa_weights_row_pitch_floats = msa_weights_row_pitch / sizeof(float);

        //copy each subject into the top row of its multiple sequence alignment
        for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
            const int subjectColumnsBegin_incl = d_msa_column_properties[subjectIndex].subjectColumnsBegin_incl;
            const int candidates_before_this_subject = d_indices_per_subject_prefixsum[subjectIndex];

            const unsigned offset1 = msa_row_pitch * (subjectIndex + candidates_before_this_subject);
            const unsigned offset2 = msa_weights_row_pitch_floats * (subjectIndex + candidates_before_this_subject);

            char* const multiple_sequence_alignment = d_multiple_sequence_alignments + offset1;
            float* const multiple_sequence_alignment_weight = d_multiple_sequence_alignment_weights + offset2;

            const int subjectLength = d_subject_sequences_lengths[subjectIndex];
            const char* const subject = d_subject_sequences_data + subjectIndex * encoded_sequence_pitch;
            const char* const subjectQualityScore = d_subject_qualities + subjectIndex * quality_pitch;

            for(int i = threadIdx.x; i < subjectLength; i+= blockDim.x){
                multiple_sequence_alignment[subjectColumnsBegin_incl + i] = get_as_nucleotide(subject, subjectLength, i);
                multiple_sequence_alignment_weight[subjectColumnsBegin_incl + i] = canUseQualityScores ?
                                                                    (float)d_qscore_to_weight[(unsigned char)subjectQualityScore[i]]
                                                                    : 1.0f;
            }
        }

        // copy each query into the multiple sequence alignment of its subject
        for(unsigned index = blockIdx.x; index < n_indices; index += gridDim.x){
            const int queryIndex = d_indices[index];

            const int shift = d_alignment_shifts[queryIndex];
            const BestAlignment_t flag = d_alignment_best_alignment_flags[queryIndex];
            const int queryLength = d_candidate_sequences_lengths[queryIndex];

            //find subjectindex
            int subjectIndex = 0;
            for(; subjectIndex < n_subjects; subjectIndex++){
                if(queryIndex < d_candidates_per_subject_prefixsum[subjectIndex+1])
                    break;
            }

            const int subjectColumnsBegin_incl = d_msa_column_properties[subjectIndex].subjectColumnsBegin_incl;
            const int localQueryIndex = queryIndex - d_candidates_per_subject_prefixsum[subjectIndex];
            const int defaultcolumnoffset = subjectColumnsBegin_incl + shift;

            const unsigned offset1 = msa_row_pitch * (subjectIndex + candidates_before_this_subject);
            const unsigned offset2 = msa_weights_row_pitch_floats * (subjectIndex + candidates_before_this_subject);

            char* const multiple_sequence_alignment = d_multiple_sequence_alignments + offset1;
            float* const multiple_sequence_alignment_weight = d_multiple_sequence_alignment_weights + offset2;

            const char* const query = d_candidate_sequences_data + queryIndex * sequencepitch;
            const char* const queryQualityScore = d_candidate_qualities + queryIndex * quality_pitch;

            assert(flag != BestAlignment_t::None); // indices should only be pointing to valid alignments

            //copy query into msa
            for(int i = threadIdx.x; i < queryLength; i+= blockDim.x){
                const int globalIndex = defaultcolumnoffset + i;
                const int row = 1 + localQueryIndex;
                multiple_sequence_alignment[row * msa_row_pitch + globalIndex] = get_as_nucleotide(query, queryLength, i);
                multiple_sequence_alignment_weight[row * msa_weights_row_pitch_floats + i]
                                    = canUseQualityScores ?
                                        (float)d_qscore_to_weight[(unsigned char)queryQualityScore[i]]
                                        : 1.0f;
            }
        }
    }

    template<class Accessor>
    void call_msa_add_sequences_kernel_async(
                            char* d_multiple_sequence_alignments,
                            float* d_multiple_sequence_alignment_weights,
                            const int* d_alignment_shifts,
                            const BestAlignment_t* d_alignment_best_alignment_flags,
                            const char* d_subject_sequences_data,
                            const char* d_candidate_sequences_data,
                            const int* d_subject_sequences_lengths,
                            const int* d_candidate_sequences_lengths,
                            const char* d_subject_qualities,
                            const char* d_candidate_qualities,
                            const MSAColumnProperties*  d_msa_column_properties,
                            const int* d_candidates_per_subject_prefixsum,
                            const int* d_indices,
                            const int* d_indices_per_subject,
                            const int* d_indices_per_subject_prefixsum,
                            int n_subjects,
                            int n_queries,
                            int n_indices,
                            bool canUseQualityScores,
                            size_t encoded_sequence_pitch,
                            size_t quality_pitch
                            size_t msa_row_pitch,
                            size_t msa_weights_row_pitch,
                            Accessor get_as_nucleotide,
                            cudaStream_t stream){

            dim3 block(128, 1, 1);
            dim3 grid(n_indices, 1, 1); // one block per candidate which needs to be added to msa

            msa_add_sequences_kernel<<<grid, block, 0, stream>>>(d_multiple_sequence_alignments,
                                                            d_multiple_sequence_alignment_weights,
                                                            d_alignment_shifts,
                                                            d_alignment_best_alignment_flags,
                                                            d_subject_sequences_data,
                                                            d_candidate_sequences_data,
                                                            d_subject_sequences_lengths,
                                                            d_candidate_sequences_lengths,
                                                            d_subject_qualities,
                                                            d_candidate_qualities,
                                                            d_msa_column_properties,
                                                            d_candidates_per_subject_prefixsum,
                                                            d_indices,
                                                            d_indices_per_subject,
                                                            d_indices_per_subject_prefixsum,
                                                            n_subjects,
                                                            n_queries,
                                                            n_indices
                                                            canUseQualityScores,
                                                            encoded_sequence_pitch,
                                                            quality_pitch,
                                                            msa_row_pitch,
                                                            msa_weights_row_pitch); CUERR;

    }



    template<int BLOCKSIZE>
    __global__
    void msa_correct_subject_kernel(
                            const char* __restrict__ d_consensus,
                            const float* __restrict__ d_support,
                            const int* __restrict__ d_coverage,
                            const int* __restrict__ d_origCoverages,
                            const char* __restrict__ d_multiple_sequence_alignments,
                            const MSAColumnProperties* const __restrict__ d_msa_column_properties,
                            const int* __restrict__ d_indices_per_subject_prefixsum,
                            bool* __restrict__ d_is_high_quality_subject,
                            char* __restrict__ d_corrected_subjects,
                            int n_subjects,
                            int n_queries,
                            int n_indices,
                            size_t sequence_pitch,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            double estimatedErrorrate,
                            double avg_support_threshold,
                            double min_support_threshold,
                            double min_coverage_threshold,
                            int k_region){

        using BlockReduceBool = cub::BlockReduce<bool, BLOCKSIZE>;
        using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;
        using BlockReduceFloat = cub::BlockReduce<float, BLOCKSIZE>;

        __shared__ union {
            typename BlockReduceBool::TempStorage boolreduce;
            typename BlockReduceInt::TempStorage intreduce;
            typename BlockReduceFloat::TempStorage floatreduce;
        } temp_storage;

        __shared__ bool broadcastbuffer;

        auto isGoodAvgSupport = [&](double avgsupport){
            return avgsupport >= avg_support_threshold;
        };
        auto isGoodMinSupport = [&](double minsupport){
            return minsupport >= min_support_threshold;
        };
        auto isGoodMinCoverage = [&](double mincoverage){
            return mincoverage >= min_coverage_threshold;
        };

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

        for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
            const float* const my_support = d_support + msa_weights_pitch_floats * subjectIndex;
            const int* const my_coverage = d_coverage + msa_weights_pitch_floats * subjectIndex;
            const int* const my_orig_coverage = d_origCoverages + msa_weights_pitch_floats * subjectIndex;
            const char* const my_consensus = d_consensus + msa_pitch  * subjectIndex;
            char* const my_corrected_subject = d_corrected_subjects + subjectIndex * sequence_pitch;

            const MSAColumnProperties properties = d_msa_column_properties[subjectIndex];
            const int subjectColumnsBegin_incl = properties.subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = properties.subjectColumnsEnd_excl;

            float avg_support = 0;
            float min_support = 1.0f;
            //int max_coverage = 0;
            int min_coverage = std::numeric_limits<int>::max();

            for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                assert(i < properties.columnsToCheck);

                avg_support += my_support[i];
                min_support = min(my_support[i], min_support);
                //max_coverage = max(my_coverage[i], max_coverage);
                min_coverage = min(my_coverage[i], min_coverage);
            }

            avg_support = BlockReduceFloat(temp_storage.floatreduce).Sum(avg_support);
            min_support = BlockReduceFloat(temp_storage.floatreduce).Reduce(min_support, cub::Min());
            //max_coverage = BlockReduceInt(temp_storage.intreduce).Reduce(max_coverage, cub::Max());
            min_coverage = BlockReduceInt(temp_storage.intreduce).Reduce(min_coverage, cub::Min());

            avg_support /= (subjectColumnsEnd_excl - subjectColumnsBegin_incl);

            bool isHQ = isGoodAvgSupport(avg_support) && isGoodMinSupport(min_support) && isGoodMinCoverage(min_coverage);

            if(threadIdx.x == 0){
                broadcastbuffer = isHQ;
                d_is_high_quality_subject[subjectIndex] = isHQ;
            }
            __syncthreads();

            isHQ = broadcastbuffer;

            if(isHQ){
                for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                    const int globalIndex = columnProperties.subjectColumnsBegin_incl + i;
                    my_corrected_subject[i - subjectColumnsBegin_incl] = my_consensus[i];
                }
                if(threadIdx.x == 0){
                    d_subject_is_corrected[subjectIndex] = true;
                }
            }else{
                const unsigned offset1 = msa_pitch * (subjectIndex + d_indices_per_subject_prefixsum[subjectIndex]);
                const char* const my_multiple_sequence_alignment = d_multiple_sequence_alignments + offset1;

                //copy orignal sequence, which is in first row of msa, to corrected sequences
                for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                    my_corrected_subject[i - subjectColumnsBegin_incl] = my_multiple_sequence_alignment[i];
                }

                const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;

                bool foundAColumn = false;
                for(int i = 0; i < subjectlength; i++){
                    const int globalIndex = subjectColumnsBegin_incl + i;

                    if(my_support[globalIndex] > 0.5 && my_orig_coverage[globalIndex] < min_coverage_threshold){
                        double avgsupportkregion = 0;
                        int c = 0;
                        bool kregioncoverageisgood = true;

                        for(int j = i - k_region/2; j <= i + k_region/2 && kregioncoverageisgood; j++){
                            if(j != i && j >= 0 && j < subjectlength){
                                avgsupportkregion += my_support[subjectColumnsBegin_incl + j];
                                kregioncoverageisgood &= (my_coverage[subjectColumnsBegin_incl + j] >= min_coverage_threshold);
                                c++;
                            }
                        }

                        avgsupportkregion /= c;
                        if(kregioncoverageisgood && avgsupportkregion >= 1.0-estimatedErrorrate){
                            my_corrected_subject[i] = my_consensus[globalIndex];
                            foundAColumn = true;
                        }
                    }
                }
                //perform block wide or-reduction on foundAColumn
                foundAColumn = BlockReduceBool(temp_storage.boolreduce).Reduce(foundAColumn, [](bool a, bool b){return a || b});
                if(threadIdx.x == 0){
                    d_subject_is_corrected[subjectIndex] = foundAColumn;
                }
            }
        }
    }

    void call_msa_correct_subject_kernel_async(
                            const char* d_consensus,
                            const float* d_support,
                            const int* d_coverage,
                            const int* d_origCoverages,
                            const char* d_multiple_sequence_alignments,
                            const MSAColumnProperties* const d_msa_column_properties,
                            const int* d_indices_per_subject_prefixsum,
                            bool* d_is_high_quality_subject,
                            char* d_corrected_subjects,
                            int n_subjects,
                            int n_queries,
                            int n_indices,
                            size_t sequence_pitch,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            double estimatedErrorrate,
                            double avg_support_threshold,
                            double min_support_threshold,
                            double min_coverage_threshold,
                            int k_region,
                            int maximum_sequence_length,
                            cudaStream_t stream){

        const int max_block_size = 256;
        const int blocksize = std::min(max_block_size, SDIV(maximum_sequence_length, 32) * 32);

        dim3 block(blocksize, 1, 1);
        dim3 grid(n_subjects);

        #define mycall(blocksize) msa_correct_subject_kernel<(blocksize)> \
                                <<<grid, block, 0, stream>>>( \
                                    d_consensus, \
                                    d_support, \
                                    d_coverage, \
                                    d_origCoverages, \
                                    d_multiple_sequence_alignments, \
                                    d_msa_column_properties, \
                                    d_indices_per_subject_prefixsum, \
                                    d_is_high_quality_subject, \
                                    d_corrected_subjects, \
                                    n_subjects, \
                                    n_queries, \
                                    n_indices, \
                                    sequence_pitch, \
                                    msa_pitch, \
                                    msa_weights_pitch, \
                                    estimatedErrorrate, \
                                    avg_support_threshold, \
                                    min_support_threshold, \
                                    min_coverage_threshold, \
                                    k_region); CUERR;

        assert(blocksize > 0 && blocksize <= max_block_size);

        switch(blocksize){
            case 32: mycall(32); break;
            case 64: mycall(64); break;
            case 96: mycall(96); break;
            case 128: mycall(128); break;
            case 160: mycall(160); break;
            case 192: mycall(192); break;
            case 224: mycall(224); break;
            case 256: mycall(256); break;
            default: mycall(256); break;
        }

    }



#endif

}
}


#endif
