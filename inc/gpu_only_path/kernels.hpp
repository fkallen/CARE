#ifndef CARE_GPU_KERNELS_HPP
#define CARE_GPU_KERNELS_HPP

#include "../hpc_helpers.cuh"
#include "bestalignment.hpp"
#include "msa.hpp"
#include "../qualityscoreweights.hpp"

#include <stdexcept>
#include <cassert>

#ifdef __NVCC__
#include <cub/cub.cuh>
#endif


namespace care{
namespace gpu{

#ifdef __NVCC__

	void call_cuda_filter_alignments_by_mismatchratio_kernel_async(
                                        BestAlignment_t* d_alignment_best_alignment_flags,
                                        const int* d_alignment_overlaps,
                                        const int* d_alignment_nOps,
                                        const int* d_indices,
										const int* d_indices_per_subject,
										const int* d_indices_per_subject_prefixsum,
                                        int n_subjects,
										int n_candidates,
										const int* d_num_indices,
										double mismatchratioBaseFactor,
										double goodAlignmentsCountThreshold,
										cudaStream_t stream);

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
                            const MSAColumnProperties* d_msa_column_properties,
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
                            const MSAColumnProperties* d_msa_column_properties,
                            const int* d_indices_per_subject_prefixsum,
                            bool* d_is_high_quality_subject,
                            char* d_corrected_subjects,
							bool* d_subject_is_corrected,
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

        using BlockReduceInt2 = cub::BlockReduce<int2, BLOCKSIZE>;

        __shared__ union {
            typename BlockReduceInt2::TempStorage reduce;
        } temp_storage;

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

			myval = BlockReduceInt2(temp_storage.reduce).Reduce(myval, [](auto a, auto b){return a.x < b.x ? a : b;});
			__syncthreads();

            //make result
            if(threadIdx.x == 0){
                bestScore = myval.x;
                bestShift = myval.y;

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
                            Accessor accessor,
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
                                          d_alignment_scores, \
                                          d_alignment_overlaps, \
                                          d_alignment_shifts, \
                                          d_alignment_nOps, \
                                          d_alignment_isValid, \
                                          d_subject_sequences_data, \
                                          d_candidate_sequences_data, \
                                          d_subject_sequences_lengths, \
                                          d_candidate_sequences_lengths, \
                                          d_candidates_per_subject_prefixsum, \
                                          n_subjects, \
                                          max_sequence_bytes, \
                                          encodedsequencepitch, \
                                          min_overlap, \
                                          maxErrorRate, \
                                          min_overlap_ratio, \
                                          accessor, \
                                          make_reverse_complement_inplace); CUERR;

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
                                        int* d_alignment_scores,
                                        int* d_alignment_overlaps,
                                        int* d_alignment_shifts,
                                        int* d_alignment_nOps,
                                        bool* d_alignment_isValid,
                                        const int* d_subject_sequences_lengths,
                                        const int* d_candidate_sequences_lengths,
                                        const int* d_candidates_per_subject_prefixsum,
                                        int n_subjects,
										int n_queries,
										double min_overlap_ratio,
                                        int min_overlap,
                                        double maxErrorRate,
                                        AlignmentComp comp){

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
            for(; subjectIndex < n_subjects; subjectIndex++){
                if(resultIndex < d_candidates_per_subject_prefixsum[subjectIndex+1])
                    break;
            }

            const int subjectlength = d_subject_sequences_lengths[subjectIndex];

            /*const BestAlignment_t flag = comp(fwd_alignment_overlap,
                                                revc_alignment_overlap,
                                                fwd_alignment_nops,
                                                revc_alignment_nops,
                                                fwd_alignment_isvalid,
                                                revc_alignment_isvalid,
                                                subjectlength,
                                                querylength);*/

			const BestAlignment_t flag = choose_best_alignment(fwd_alignment_overlap,
																revc_alignment_overlap,
																fwd_alignment_nops,
																revc_alignment_nops,
																fwd_alignment_isvalid,
																revc_alignment_isvalid,
																subjectlength,
																querylength,
																min_overlap_ratio,
																min_overlap,
																maxErrorRate);

            d_alignment_best_alignment_flags[resultIndex] = flag;

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
                                        int* d_alignment_scores,
                                        int* d_alignment_overlaps,
                                        int* d_alignment_shifts,
                                        int* d_alignment_nOps,
                                        bool* d_alignment_isValid,
                                        const int* d_subject_sequences_lengths,
                                        const int* d_candidate_sequences_lengths,
                                        const int* d_candidates_per_subject_prefixsum,
                                        int n_subjects,
										double min_overlap_ratio,
                                        int min_overlap,
                                        double maxErrorRate,
                                        AlignmentComp d_comp,
                                        int n_queries,
                                        cudaStream_t stream){

        dim3 block(128,1,1);
        dim3 grid(SDIV(n_queries, block.x), 1, 1);

        cuda_find_best_alignment_kernel<<<grid, block, 0, stream>>>(
                                                d_alignment_best_alignment_flags,
                                                d_alignment_scores,
                                                d_alignment_overlaps,
                                                d_alignment_shifts,
                                                d_alignment_nOps,
                                                d_alignment_isValid,
                                                d_subject_sequences_lengths,
                                                d_candidate_sequences_lengths,
                                                d_candidates_per_subject_prefixsum,
                                                n_subjects,
												n_queries,
												min_overlap_ratio,
												min_overlap,
												maxErrorRate,
                                                d_comp); CUERR;

    }

    /*
	 *
	 */

	template<int BLOCKSIZE>
    __global__
    void cuda_filter_alignments_by_mismatchratio_kernel(
                                        BestAlignment_t* d_alignment_best_alignment_flags,
                                        const int* d_alignment_overlaps,
                                        const int* d_alignment_nOps,
                                        const int* d_indices,
										const int* d_indices_per_subject,
										const int* d_indices_per_subject_prefixsum,
                                        int n_subjects,
										int n_candidates,
										const int* d_num_indices,
										double mismatchratioBaseFactor,
										double goodAlignmentsCountThreshold){

        using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;

        __shared__ union {
            typename BlockReduceInt::TempStorage intreduce;
			int broadcast[3];
        } temp_storage;

        /*if(threadIdx.x == 0){
            printf("n_subjects %d\n", n_subjects);
        }

        printf("blockIdx.x %d\n", blockIdx.x);*/

        for(int subjectindex = blockIdx.x; subjectindex < n_subjects; subjectindex += gridDim.x){

			const int my_n_indices = d_indices_per_subject[subjectindex];
			const int* my_indices = d_indices + d_indices_per_subject_prefixsum[subjectindex];

            //printf("subjectindex %d\n", subjectindex);

			int counts[3]{0,0,0};

            //if(threadIdx.x == 0){
            //    printf("my_n_indices %d\n", my_n_indices);
            //}

			for(int index = threadIdx.x; index < my_n_indices; index += blockDim.x){
				const int candidate_index = my_indices[index];
				const int alignment_overlap = d_alignment_overlaps[candidate_index];
				const int alignment_nops = d_alignment_nOps[candidate_index];

				const double mismatchratio = double(alignment_nops) / alignment_overlap;
                if(mismatchratio >= 4 * mismatchratioBaseFactor){
                    d_alignment_best_alignment_flags[candidate_index] = BestAlignment_t::None;
                }else{

    				#pragma unroll
    				for(int i = 2; i <= 4; i++){
    					counts[i-2] += (mismatchratio < i * mismatchratioBaseFactor);
    				}
                }
			}

			//accumulate counts over block
			#pragma unroll
			for(int i = 0; i < 3; i++){
				counts[i] = BlockReduceInt(temp_storage.intreduce).Sum(counts[i]);
				__syncthreads();
			}

			//broadcast accumulated counts to block
            if(threadIdx.x == 0){
    			#pragma unroll
    			for(int i = 0; i < 3; i++){
                    temp_storage.broadcast[i] = counts[i];
                    //printf("count[%d] = %d\n", i, counts[i]);
    			}
                //printf("mismatchratioBaseFactor %f, goodAlignmentsCountThreshold %f\n", mismatchratioBaseFactor, goodAlignmentsCountThreshold);
			}

            __syncthreads();

            #pragma unroll
            for(int i = 0; i < 3; i++){
                counts[i] = temp_storage.broadcast[i];
            }

			double mismatchratioThreshold = 0;
			if (counts[0] >= goodAlignmentsCountThreshold) {
				mismatchratioThreshold = 2 * mismatchratioBaseFactor;
			} else if (counts[1] >= goodAlignmentsCountThreshold) {
				mismatchratioThreshold = 3 * mismatchratioBaseFactor;
			} else if (counts[2] >= goodAlignmentsCountThreshold) {
				mismatchratioThreshold = 4 * mismatchratioBaseFactor;
			} else {
				mismatchratioThreshold = -1; //this will invalidate all alignments for subject
			}

			// Invalidate all alignments for subject with mismatchratio >= mismatchratioThreshold
			for(int index = threadIdx.x; index < my_n_indices; index += blockDim.x){
				const int candidate_index = my_indices[index];
				const int alignment_overlap = d_alignment_overlaps[candidate_index];
				const int alignment_nops = d_alignment_nOps[candidate_index];

				const double mismatchratio = double(alignment_nops) / alignment_overlap;

				const bool remove = mismatchratio >= mismatchratioThreshold;
				if(remove)
					d_alignment_best_alignment_flags[candidate_index] = BestAlignment_t::None;
			}
        }
    }






    template<int BLOCKSIZE>
    __global__
    void cuda_filter_alignments_by_mismatchratio_kernel_new(
                                        void* d_temp_storage,
                                        BestAlignment_t* d_alignment_best_alignment_flags,
                                        const int* d_alignment_overlaps,
                                        const int* d_alignment_nOps,
                                        const int* d_indices,
                                        const int* d_indices_per_subject,
                                        const int* d_indices_per_subject_prefixsum,
                                        const int* d_candidates_per_subject_prefixsum,
                                        int* d_candidate_ranges_counts_per_subject,

                                        int n_subjects,
                                        int n_candidates,
                                        const int* d_num_indices,
                                        double binsize,
                                        int min_remaining_candidates_per_subject){

        struct Counts{
            int counts[3]{0,0,0};
            int subject_index = 0;
        };

        //Counts shared_counts[BLOCKSIZE];

        Counts* my_counts = (Counts*)(((char*)d_temp_storage) + n_subjects * sizeof(Counts));

        const int n_indices = *d_num_indices;
        Counts local_counts;
        for(int index = threadIdx.x + blockDim.x * blockIdx.x; index < n_indices; index += blockDim.x * gridDim.x){
            const int candidate_index = d_indices[index];

            //find subjectindex
            int subject_index = 0;
            for(; subject_index < n_subjects; subject_index++){
                if(candidate_index < d_candidates_per_subject_prefixsum[subject_index+1])
                    break;
            }

            assert(subject_index >= local_counts.subject_index);

            //if subjectIndex changed, save local_counts to gmem;
            if(subject_index != local_counts.subject_index){
                my_counts[local_counts.subject_index] = local_counts;
            }

            const int alignment_overlap = d_alignment_overlaps[candidate_index];
            const int alignment_nops = d_alignment_nOps[candidate_index];

            const double mismatchratio = double(alignment_nops) / alignment_overlap;

            assert(mismatchratio < 4 * binsize);

            #pragma unroll
            for(int i = 2; i <= 4; i++){
                local_counts.counts[i-2] += (mismatchratio < i * binsize);
            }
        }

    }









    template<class Accessor, class RevCompl>
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
                            const int* __restrict__ d_alignment_overlaps,
                            const int* __restrict__ d_alignment_nOps,
                            const MSAColumnProperties*  __restrict__ d_msa_column_properties,
                            const int* __restrict__ d_candidates_per_subject_prefixsum,
                            const int* __restrict__ d_indices,
                            const int* __restrict__ d_indices_per_subject,
                            const int* __restrict__ d_indices_per_subject_prefixsum,
                            int n_subjects,
                            int n_queries,
                            int n_indices,
                            bool canUseQualityScores,
                            float desiredAlignmentMaxErrorRate,
                            size_t encoded_sequence_pitch,
                            size_t quality_pitch,
                            size_t msa_row_pitch,
                            size_t msa_weights_row_pitch,
                            Accessor get_as_nucleotide,
                            RevCompl make_unpacked_reverse_complement_inplace){

        auto reverse_float = [](float* sequence, int length){

            for(int i = 0; i < length/2; i++){
                const float front = sequence[i];
                const float back = sequence[length - 1 - i];
                sequence[i] = back;
                sequence[length - 1 - i] = front;
            }

            if(length % 2 == 1){
                ; // when sequencelength is odd, the center remains unchanged
            }
        };

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
            const int localQueryIndex = index - d_indices_per_subject_prefixsum[subjectIndex];
            const int defaultcolumnoffset = subjectColumnsBegin_incl + shift;

			const int candidates_before_this_subject = d_indices_per_subject_prefixsum[subjectIndex];

			//printf("index %d, subjectindex %d, queryindex %d, shift %d, defaultcolumnoffset %d, candidates_before_this_subject %d, localQueryIndex %d\n", index, subjectIndex, queryIndex, shift, defaultcolumnoffset, candidates_before_this_subject, localQueryIndex);

            const unsigned offset1 = msa_row_pitch * (subjectIndex + candidates_before_this_subject);
            const unsigned offset2 = msa_weights_row_pitch_floats * (subjectIndex + candidates_before_this_subject);
			const int rowOffset = (subjectIndex + candidates_before_this_subject);

            char* const multiple_sequence_alignment = d_multiple_sequence_alignments + offset1;
            float* const multiple_sequence_alignment_weight = d_multiple_sequence_alignment_weights + offset2;

            const char* const query = d_candidate_sequences_data + queryIndex * encoded_sequence_pitch;

            //need to use index for adressing d_candidate_qualities instead of queryIndex, because d_candidate_qualities is compact
            const char* const queryQualityScore = d_candidate_qualities + index * quality_pitch;

            //const char* const queryQualityScore = d_candidate_qualities + queryIndex * quality_pitch;

            const int query_alignment_overlap = d_alignment_overlaps[queryIndex];
            const int query_alignment_nops = d_alignment_nOps[queryIndex];

            const double defaultweight = 1.0 - sqrtf(query_alignment_nops
                                                        / (query_alignment_overlap * desiredAlignmentMaxErrorRate));

            assert(flag != BestAlignment_t::None); // indices should only be pointing to valid alignments

            //copy query into msa
			const int row = 1 + localQueryIndex;
            for(int i = threadIdx.x; i < queryLength; i+= blockDim.x){
                const int globalIndex = defaultcolumnoffset + i;

                multiple_sequence_alignment[row * msa_row_pitch + globalIndex] = get_as_nucleotide(query, queryLength, i);

                multiple_sequence_alignment_weight[row * msa_weights_row_pitch_floats + globalIndex]
                                    = canUseQualityScores ?
                                        (float)d_qscore_to_weight[(unsigned char)queryQualityScore[i]] * defaultweight
                                        : 1.0f;
            }

            //__syncthreads(); //remove

            /*if(threadIdx.x == 0){
                printf("query %d\n", queryIndex);
                for(int i = 0; i < queryLength; i+= 1){
                    printf("%c", queryQualityScore[i]);
                }
                printf("\n");

                for(int i = 0; i < queryLength; i+= 1){
                    const int globalIndex = defaultcolumnoffset + i;
                    printf("%f ", multiple_sequence_alignment_weight[row * msa_weights_row_pitch_floats + globalIndex]);
                }
                printf("\n");
            }*/


            __syncthreads(); // need to wait until current row is written by all threads

            if(threadIdx.x == 0 && flag == BestAlignment_t::ReverseComplement){
                make_unpacked_reverse_complement_inplace((std::uint8_t*)multiple_sequence_alignment + row * msa_row_pitch + defaultcolumnoffset,
                                                        queryLength);
                //reverse quality weights. if canUseQualityScores == false, then all weights are 1.0f and do not need to be reversed
                if(canUseQualityScores){
                    reverse_float(multiple_sequence_alignment_weight + row * msa_weights_row_pitch_floats + defaultcolumnoffset, queryLength);
                    /*if(threadIdx.x == 0){
                        for(int i = 0; i < queryLength; i+= 1){
                            const int globalIndex = defaultcolumnoffset + i;
                            printf("%f ", multiple_sequence_alignment_weight[row * msa_weights_row_pitch_floats + globalIndex]);
                        }
                        printf("\n");
                    }*/
                }
            }

            //__syncthreads(); //remove

            //__syncthreads();

            /*for(int i = threadIdx.x; i < queryLength; i+= blockDim.x){
                const int globalIndex = defaultcolumnoffset + i;

                if(multiple_sequence_alignment[row * msa_row_pitch + globalIndex] == 'F'){
                    printf("error row %d, globalIndex %d, defaultcolumnoffset %d, i %d\n", row, globalIndex, defaultcolumnoffset, i);
                    assert(false);
                }
            }*/

        }
    }

    template<class Accessor, class RevCompl>
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
                            const int* d_alignment_overlaps,
                            const int* d_alignment_nOps,
                            const MSAColumnProperties*  d_msa_column_properties,
                            const int* d_candidates_per_subject_prefixsum,
                            const int* d_indices,
                            const int* d_indices_per_subject,
                            const int* d_indices_per_subject_prefixsum,
                            int n_subjects,
                            int n_queries,
                            int n_indices,
                            bool canUseQualityScores,
                            float desiredAlignmentMaxErrorRate,
                            size_t encoded_sequence_pitch,
                            size_t quality_pitch,
                            size_t msa_row_pitch,
                            size_t msa_weights_row_pitch,
                            Accessor get_as_nucleotide,
                            RevCompl make_unpacked_reverse_complement_inplace,
                            cudaStream_t stream){

            dim3 block(128, 1, 1);
            dim3 grid(n_indices, 1, 1); // one block per candidate which needs to be added to msa
            //dim3 grid(1,1,1);

			//dim3 block(1, 1, 1);
            //dim3 grid(1, 1, 1); // one block per candidate which needs to be added to msa

            //std::cout << "call_msa_add_sequences_kernel_async, grid: " << n_indices << std::endl;

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
                                                            d_alignment_overlaps,
                                                            d_alignment_nOps,
                                                            d_msa_column_properties,
                                                            d_candidates_per_subject_prefixsum,
                                                            d_indices,
                                                            d_indices_per_subject,
                                                            d_indices_per_subject_prefixsum,
                                                            n_subjects,
                                                            n_queries,
                                                            n_indices,
                                                            canUseQualityScores,
                                                            desiredAlignmentMaxErrorRate,
                                                            encoded_sequence_pitch,
                                                            quality_pitch,
                                                            msa_row_pitch,
                                                            msa_weights_row_pitch,
															get_as_nucleotide,
                                                            make_unpacked_reverse_complement_inplace); CUERR;
    }



    template<int BLOCKSIZE>
    __global__
    void msa_correct_subject_kernel(
                            const char* __restrict__ d_consensus,
                            const float* __restrict__ d_support,
                            const int* __restrict__ d_coverage,
                            const int* __restrict__ d_origCoverages,
                            const char* __restrict__ d_multiple_sequence_alignments,
                            const MSAColumnProperties* __restrict__ d_msa_column_properties,
                            const int* __restrict__ d_indices_per_subject_prefixsum,
                            bool* __restrict__ d_is_high_quality_subject,
                            char* __restrict__ d_corrected_subjects,
							bool* __restrict__ d_subject_is_corrected,
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
			__syncthreads();

            min_support = BlockReduceFloat(temp_storage.floatreduce).Reduce(min_support, cub::Min());
			__syncthreads();

            //max_coverage = BlockReduceInt(temp_storage.intreduce).Reduce(max_coverage, cub::Max());

            min_coverage = BlockReduceInt(temp_storage.intreduce).Reduce(min_coverage, cub::Min());
			__syncthreads();

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
                for(int i = threadIdx.x; i < subjectLength; i += BLOCKSIZE){
                    const int globalIndex = subjectColumnsBegin_incl + i;

                    if(my_support[globalIndex] > 0.5 && my_orig_coverage[globalIndex] <= min_coverage_threshold){
                        double avgsupportkregion = 0;
                        int c = 0;
                        bool kregioncoverageisgood = true;

                        for(int j = i - k_region/2; j <= i + k_region/2 && kregioncoverageisgood; j++){
                            if(j != i && j >= 0 && j < subjectLength){
                                avgsupportkregion += my_support[subjectColumnsBegin_incl + j];
                                kregioncoverageisgood &= (my_coverage[subjectColumnsBegin_incl + j] >= min_coverage_threshold);
                                c++;
                            }
                        }

                        avgsupportkregion /= c;

						//if(i == 33 || i == 34){
						//	printf("%d %f\n", i, avgsupportkregion);
						//}
                        if(kregioncoverageisgood && avgsupportkregion >= 1.0-estimatedErrorrate){
                            my_corrected_subject[i] = my_consensus[globalIndex];
                            foundAColumn = true;
                        }
                    }
                }
                //perform block wide or-reduction on foundAColumn
                foundAColumn = BlockReduceBool(temp_storage.boolreduce).Reduce(foundAColumn, [](bool a, bool b){return a || b;});
				__syncthreads();

                if(threadIdx.x == 0){
                    d_subject_is_corrected[subjectIndex] = foundAColumn;
                }
            }
        }
    }


    template<int BLOCKSIZE, class RevCompl>
    __global__
    void msa_correct_candidates_kernel(
                            const char* __restrict__ d_consensus,
                            const float* __restrict__ d_support,
                            const int* __restrict__ d_coverage,
                            const int* __restrict__ d_origCoverages,
                            const char* __restrict__ d_multiple_sequence_alignments,
                            const MSAColumnProperties* __restrict__ d_msa_column_properties,
							const int* __restrict__ d_indices,
							const int* __restrict__ d_indices_per_subject,
                            const int* __restrict__ d_indices_per_subject_prefixsum,
                            const int* __restrict__ d_high_quality_subject_indices,
							const int* __restrict__ d_num_high_quality_subject_indices,
							const int* __restrict__ d_alignment_shifts,
							const BestAlignment_t* __restrict__ d_alignment_best_alignment_flags,
							const int* __restrict__ d_candidate_sequences_lengths,
							int* __restrict__ d_num_corrected_candidates,
							char* __restrict__ d_corrected_candidates,
							int* __restrict__ d_indices_of_corrected_candidates,
                            int n_subjects,
                            int n_queries,
                            int n_indices,
                            size_t sequence_pitch,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            double min_support_threshold,
                            double min_coverage_threshold,
                            int new_columns_to_correct,
							RevCompl make_unpacked_reverse_complement_inplace){

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);
		const int num_high_quality_subject_indices = *d_num_high_quality_subject_indices;

        for(unsigned index = blockIdx.x; index < num_high_quality_subject_indices; index += gridDim.x){
			const int subjectIndex = d_high_quality_subject_indices[index];
			const int my_num_candidates = d_indices_per_subject[subjectIndex];

            const float* const my_support = d_support + msa_weights_pitch_floats * subjectIndex;
            const int* const my_coverage = d_coverage + msa_weights_pitch_floats * subjectIndex;
            //const int* const my_orig_coverage = d_origCoverages + msa_weights_pitch_floats * subjectIndex;
            const char* const my_consensus = d_consensus + msa_pitch  * subjectIndex;
            const int* const my_indices = d_indices + d_indices_per_subject_prefixsum[subjectIndex];
            char* const my_corrected_candidates = d_corrected_candidates + d_indices_per_subject_prefixsum[subjectIndex] * sequence_pitch;
			int* const my_indices_of_corrected_candidates = d_indices_of_corrected_candidates + d_indices_per_subject_prefixsum[subjectIndex];

            const MSAColumnProperties properties = d_msa_column_properties[subjectIndex];
            const int subjectColumnsBegin_incl = properties.subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = properties.subjectColumnsEnd_excl;

			//const unsigned offset1 = msa_pitch * (subjectIndex + d_indices_per_subject_prefixsum[subjectIndex]);
			//const char* const my_multiple_sequence_alignment = d_multiple_sequence_alignments + offset1;

			int n_corrected_candidates = 0;

			for(int local_candidate_index = 0; local_candidate_index < my_num_candidates; ++local_candidate_index){
				const int global_candidate_index = my_indices[local_candidate_index];
				const int shift = d_alignment_shifts[global_candidate_index];
				const int candidate_length = d_candidate_sequences_lengths[global_candidate_index];
				const BestAlignment_t bestAlignmentFlag = d_alignment_best_alignment_flags[global_candidate_index];
				const int queryColumnsBegin_incl = shift - properties.startindex;
				const int queryColumnsEnd_excl = queryColumnsBegin_incl + candidate_length;

				//check range condition and length condition
				if(subjectColumnsBegin_incl - new_columns_to_correct <= queryColumnsBegin_incl
					&& queryColumnsBegin_incl <= subjectColumnsBegin_incl + new_columns_to_correct
					&& queryColumnsEnd_excl <= subjectColumnsEnd_excl + new_columns_to_correct){

					double newColMinSupport = 1.0;
					int newColMinCov = std::numeric_limits<int>::max();
					//check new columns left of subject
					for(int columnindex = subjectColumnsBegin_incl - new_columns_to_correct;
						columnindex < subjectColumnsBegin_incl;
						columnindex++){

						assert(columnindex < properties.columnsToCheck);
						if(queryColumnsBegin_incl <= columnindex){
							newColMinSupport = my_support[columnindex] < newColMinSupport ? my_support[columnindex] : newColMinSupport;
							newColMinCov = my_coverage[columnindex] < newColMinCov ? my_coverage[columnindex] : newColMinCov;
						}
					}
					//check new columns right of subject
					for(int columnindex = subjectColumnsEnd_excl;
						columnindex < subjectColumnsEnd_excl + new_columns_to_correct
						&& columnindex < properties.columnsToCheck;
						columnindex++){

						newColMinSupport = my_support[columnindex] < newColMinSupport ? my_support[columnindex] : newColMinSupport;
						newColMinCov = my_coverage[columnindex] < newColMinCov ? my_coverage[columnindex] : newColMinCov;
					}

					if(newColMinSupport >= min_support_threshold
						&& newColMinCov >= min_coverage_threshold){

						for(int i = queryColumnsBegin_incl + threadIdx.x; i < queryColumnsEnd_excl; i += BLOCKSIZE){
							my_corrected_candidates[n_corrected_candidates * sequence_pitch + (i - queryColumnsBegin_incl)] = my_consensus[i];
						}

                        __syncthreads(); // need to wait until all threads have written my_corrected_candidates before calculating reverse complement

						if(threadIdx.x == 0){
							//the forward strand will be returned -> make reverse complement again
							if(bestAlignmentFlag == BestAlignment_t::ReverseComplement){
								make_unpacked_reverse_complement_inplace((std::uint8_t*)(my_corrected_candidates + n_corrected_candidates * sequence_pitch), candidate_length);
							}
							my_indices_of_corrected_candidates[n_corrected_candidates] = global_candidate_index;
							//printf("subjectIndex %d global_candidate_index %d\n", subjectIndex, global_candidate_index);
						}

						++n_corrected_candidates;
					}
				}
			}

			//printf("%d %d\n", subjectIndex, n_corrected_candidates);

			if(threadIdx.x == 0){
				d_num_corrected_candidates[subjectIndex] = n_corrected_candidates;
				//printf("%d %d\n", subjectIndex, n_corrected_candidates);
			}
        }
    }

    template<class RevCompl>
    void call_msa_correct_candidates_kernel_async(
                            const char* d_consensus,
                            const float* d_support,
                            const int* d_coverage,
                            const int* d_origCoverages,
                            const char* d_multiple_sequence_alignments,
                            const MSAColumnProperties* d_msa_column_properties,
							const int* d_indices,
							const int* d_indices_per_subject,
                            const int* d_indices_per_subject_prefixsum,
                            const int* d_high_quality_subject_indices,
							const int* d_num_high_quality_subject_indices,
							const int* d_alignment_shifts,
							const BestAlignment_t* d_alignment_best_alignment_flags,
							const int* d_candidate_sequences_lengths,
							int* d_num_corrected_candidates,
							char* d_corrected_candidates,
							int* d_indices_of_corrected_candidates,
                            int n_subjects,
                            int n_queries,
                            int n_indices,
                            size_t sequence_pitch,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            double min_support_threshold,
                            double min_coverage_threshold,
                            int new_columns_to_correct,
							RevCompl make_unpacked_reverse_complement_inplace,
							int maximum_sequence_length,
							cudaStream_t stream){

		const int max_block_size = 256;
		const int blocksize = std::min(max_block_size, SDIV(maximum_sequence_length, 32) * 32);

		dim3 block(blocksize, 1, 1);
		dim3 grid(n_subjects);

		#define mycall(blocksize) msa_correct_candidates_kernel<(blocksize)> \
								<<<grid, block, 0, stream>>>( \
									d_consensus, \
									d_support, \
									d_coverage, \
									d_origCoverages, \
									d_multiple_sequence_alignments, \
									d_msa_column_properties, \
									d_indices, \
									d_indices_per_subject, \
									d_indices_per_subject_prefixsum, \
									d_high_quality_subject_indices, \
									d_num_high_quality_subject_indices, \
									d_alignment_shifts, \
									d_alignment_best_alignment_flags, \
									d_candidate_sequences_lengths, \
									d_num_corrected_candidates, \
									d_corrected_candidates, \
									d_indices_of_corrected_candidates, \
									n_subjects, \
									n_queries, \
									n_indices, \
									sequence_pitch, \
									msa_pitch, \
									msa_weights_pitch, \
									min_support_threshold, \
									min_coverage_threshold, \
									new_columns_to_correct, \
									make_unpacked_reverse_complement_inplace); CUERR;

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

		#undef mycall
	}



#endif

}
}


#endif
