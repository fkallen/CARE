#include "kernels.hpp"
#include "msa.hpp"
#include "bestalignment.hpp"
#include "../hpc_helpers.cuh"

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
										cudaStream_t stream){

		const int blocksize = 128;

        dim3 block(blocksize, 1, 1);
        dim3 grid(n_subjects);

        #define mycall(blocksize) cuda_filter_alignments_by_mismatchratio_kernel<(blocksize)> \
                                <<<grid, block, 0, stream>>>( \
                                    d_alignment_best_alignment_flags, \
                                    d_alignment_overlaps, \
                                    d_alignment_nOps, \
                                    d_indices, \
                                    d_indices_per_subject, \
                                    d_indices_per_subject_prefixsum, \
                                    n_subjects, \
                                    n_candidates, \
                                    d_num_indices, \
                                    mismatchratioBaseFactor, \
                                    goodAlignmentsCountThreshold); CUERR;

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

        //cudaDeviceSynchronize(); CUERR;

		#undef mycall
    }

	template<int BLOCKSIZE>
    __global__
    void msa_init_kernel(
                        MSAColumnProperties* __restrict__ msa_column_properties,
                        const int* __restrict__ alignment_shifts,
                        const BestAlignment_t* __restrict__ alignment_best_alignment_flags,
                        const int* __restrict__ subject_sequences_lengths,
                        const int* __restrict__ candidate_sequences_lengths,
                        const int* __restrict__ indices,
                        const int* __restrict__ indices_per_subject,
                        const int* __restrict__ indices_per_subject_prefixsum,
                        int n_subjects){

        using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;

        __shared__ union {
            typename BlockReduceInt::TempStorage reduce;
        } temp_storage;

        for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
            MSAColumnProperties* const properties_ptr = msa_column_properties + subjectIndex;

            // We only want to consider the candidates with good alignments. the indices of those were determined in a previous step
            const int num_indices_for_this_subject = indices_per_subject[subjectIndex];
            const int* const indices_for_this_subject = indices + indices_per_subject_prefixsum[subjectIndex];

            const int subjectLength = subject_sequences_lengths[subjectIndex];
            int startindex = 0;
            int endindex = subject_sequences_lengths[subjectIndex];

            for(int index = threadIdx.x; index < num_indices_for_this_subject; index += blockDim.x){
                const int queryIndex = indices_for_this_subject[index];

                const int shift = alignment_shifts[queryIndex];
                const BestAlignment_t flag = alignment_best_alignment_flags[queryIndex];
                const int queryLength = candidate_sequences_lengths[queryIndex];

                if(flag != BestAlignment_t::None){
                    const int queryEndsAt = queryLength + shift;
                    startindex = min(startindex, shift);
                    endindex = max(endindex, queryEndsAt);
                }
            }

			startindex = BlockReduceInt(temp_storage.reduce).Reduce(startindex, cub::Min());
			__syncthreads();

			endindex = BlockReduceInt(temp_storage.reduce).Reduce(endindex, cub::Max());
			__syncthreads();

			if(threadIdx.x == 0){
				MSAColumnProperties my_columnproperties;

				my_columnproperties.startindex = startindex;
				my_columnproperties.endindex = endindex;
				my_columnproperties.columnsToCheck = my_columnproperties.endindex - my_columnproperties.startindex;
				my_columnproperties.subjectColumnsBegin_incl = max(-my_columnproperties.startindex, 0);
				my_columnproperties.subjectColumnsEnd_excl = my_columnproperties.subjectColumnsBegin_incl + subjectLength;

				*properties_ptr = my_columnproperties;
			}

        }
    }

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
                        cudaStream_t stream){

		const int blocksize = 128;

        dim3 block(blocksize, 1, 1);
        dim3 grid(n_subjects, 1, 1);

		#define mycall(blocksize) msa_init_kernel<(blocksize)><<<grid, block, 0, stream>>>(d_msa_column_properties, \
                                                    d_alignment_shifts, \
                                                    d_alignment_best_alignment_flags, \
                                                    d_subject_sequences_lengths, \
                                                    d_candidate_sequences_lengths, \
                                                    d_indices, \
                                                    d_indices_per_subject, \
                                                    d_indices_per_subject_prefixsum, \
                                                    n_subjects); CUERR;

		switch(blocksize){
			case 1: mycall(1); break;
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


    __global__
    void msa_find_consensus_kernel(
                            char* __restrict__ d_consensus,
                            float* __restrict__ d_support,
                            int* __restrict__ d_coverage,
                            float* __restrict__ d_origWeights,
                            int* __restrict__ d_origCoverages,
                            const char* __restrict__ d_multiple_sequence_alignments,
                            const float* __restrict__ d_multiple_sequence_alignment_weights,
                            const MSAColumnProperties* __restrict__ d_msa_column_properties,
                            const int* __restrict__ d_candidates_per_subject_prefixsum,
                            const int* __restrict__ d_indices_per_subject,
                            const int* __restrict__ d_indices_per_subject_prefixsum,
                            int n_subjects,
                            int n_queries,
                            const int* __restrict__ d_num_indices,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            int msa_max_column_count,
                            int blocks_per_msa){

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

        const int localBlockId = blockIdx.x % blocks_per_msa;
		//const int n_indices = *d_num_indices;

        //process multiple sequence alignment of each subject
        //for each column in msa, find consensus and support
        for(unsigned subjectIndex = blockIdx.x / blocks_per_msa; subjectIndex < n_subjects; subjectIndex += gridDim.x / blocks_per_msa){
            const int subjectColumnsBegin_incl = d_msa_column_properties[subjectIndex].subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = d_msa_column_properties[subjectIndex].subjectColumnsEnd_excl;
			const int columnsToCheck = d_msa_column_properties[subjectIndex].columnsToCheck;

            //number of rows in multiple sequence alignment for subject[subjectIndex]
            const int msa_rows = 1 + d_indices_per_subject[subjectIndex];

            const unsigned offset1 = msa_pitch * (subjectIndex + d_indices_per_subject_prefixsum[subjectIndex]);
            const unsigned offset2 = msa_weights_pitch_floats * (subjectIndex + d_indices_per_subject_prefixsum[subjectIndex]);

            const char* const my_multiple_sequence_alignment = d_multiple_sequence_alignments + offset1;
            const float* const my_multiple_sequence_alignment_weight = d_multiple_sequence_alignment_weights + offset2;

            char* const my_consensus = d_consensus + subjectIndex * msa_pitch;
            float* const my_support = d_support + subjectIndex * msa_weights_pitch_floats;
            int* const my_coverage = d_coverage + subjectIndex * msa_weights_pitch_floats;

            float* const my_orig_weights = d_origWeights + subjectIndex * msa_weights_pitch_floats;
            int* const my_orig_coverage = d_origCoverages + subjectIndex * msa_weights_pitch_floats;

            for(int column = localBlockId * blockDim.x + threadIdx.x; column < columnsToCheck; column += blocks_per_msa * blockDim.x){

                float Aw = 0.0f;
                float Cw = 0.0f;
                float Gw = 0.0f;
                float Tw = 0.0f;

                int As = 0;
                int Cs = 0;
                int Gs = 0;
                int Ts = 0;

                int columnCoverage = 0;

                for(int row = 0; row < msa_rows; ++row){
                    const char base = my_multiple_sequence_alignment[row * msa_pitch + column];
                    const float weight = my_multiple_sequence_alignment_weight[row * msa_weights_pitch_floats + column];

					//if(!(base == 'A' || base == 'C' || base == 'G' || base == 'T')){
					//	assert(base == 'A' || base == 'C' || base == 'G' || base == 'T');
					//}

                    Aw += (base == 'A' ? weight : 0);
                    Cw += (base == 'C' ? weight : 0);
                    Gw += (base == 'G' ? weight : 0);
                    Tw += (base == 'T' ? weight : 0);

                    As += (base == 'A');
                    Cs += (base == 'C');
                    Gs += (base == 'G');
                    Ts += (base == 'T');

                    columnCoverage += (base == 'A' || base == 'C' || base == 'G' || base == 'T');
                }

				if(columnCoverage <= 0){
					//assert(columnCoverage > 0);
				}

                const float columnWeight = Aw + Cw + Gw + Tw;
                float consWeight = Aw;
                char cons = 'A';

                cons = Cw > consWeight ? 'C' : cons;
                consWeight = Cw > consWeight ? Cw : consWeight;

                cons = Gw > consWeight ? 'G' : cons;
                consWeight = Gw > consWeight ? Gw : consWeight;

                cons = Tw > consWeight ? 'T' : cons;
                consWeight = Tw > consWeight ? Tw : consWeight;

                my_consensus[column] = cons;
                my_support[column] = consWeight / columnWeight;
				//printf("subject %d, column %d, support %f\n", subjectIndex, column, my_support[column]);
                my_coverage[column] = columnCoverage;

                if(subjectColumnsBegin_incl <= column && column < subjectColumnsEnd_excl){
                    const char subjectbase = my_multiple_sequence_alignment[column];
                    if(subjectbase == 'A'){
                        my_orig_weights[column] = Aw;
                        my_orig_coverage[column] = As;
                    }else if(subjectbase == 'C'){
                        my_orig_weights[column] = Cw;
                        my_orig_coverage[column] = Cs;
                    }else if(subjectbase == 'G'){
                        my_orig_weights[column] = Gw;
                        my_orig_coverage[column] = Gs;
                    }else if(subjectbase == 'T'){
                        my_orig_weights[column] = Tw;
                        my_orig_coverage[column] = Ts;
                    }
                }
            }
        }
    }

    void call_msa_find_consensus_kernel_async(
                            char* d_consensus,
                            float* d_support,
                            int* d_coverage,
                            float* d_origWeights,
                            int* d_origCoverages,
                            const char* d_multiple_sequence_alignments,
                            const float* d_multiple_sequence_alignment_weights,
                            const MSAColumnProperties* d_msa_column_properties,
                            const int* d_candidates_per_subject_prefixsum,
                            const int* d_indices_per_subject,
                            const int* d_indices_per_subject_prefixsum,
                            int n_subjects,
                            int n_queries,
                            const int* d_num_indices,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            int msa_max_column_count,
                            cudaStream_t stream){

        const int blocksize = 128;
        const int blocks_per_msa = 2; //SDIV(msa_max_column_count, blocksize);

        dim3 block(blocksize, 1, 1);
        dim3 grid(n_subjects * blocks_per_msa, 1, 1);

        msa_find_consensus_kernel<<<grid, block, 0, stream>>>(d_consensus,
                                                            d_support,
                                                            d_coverage,
                                                            d_origWeights,
                                                            d_origCoverages,
                                                            d_multiple_sequence_alignments,
                                                            d_multiple_sequence_alignment_weights,
                                                            d_msa_column_properties,
                                                            d_candidates_per_subject_prefixsum,
                                                            d_indices_per_subject,
                                                            d_indices_per_subject_prefixsum,
                                                            n_subjects,
                                                            n_queries,
                                                            d_num_indices,
                                                            msa_pitch,
                                                            msa_weights_pitch,
                                                            msa_max_column_count,
                                                            blocks_per_msa); CUERR;

    }

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
                            const int* d_num_indices,
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
                                    d_subject_is_corrected, \
                                    n_subjects, \
                                    n_queries, \
                                    d_num_indices, \
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
		#undef mycall
    }





#endif

}
}
