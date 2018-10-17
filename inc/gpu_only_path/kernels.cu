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
                            const int* __restrict__ d_num_indices,
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
		//const int n_indices = *d_num_indices;

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
