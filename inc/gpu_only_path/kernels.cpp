#include "kernels.hpp"
#include "msa.hpp"
#include "bestalignment.hpp"
#include "../hpc_helpers.cuh"

#ifdef __NVCC__
#include <cub/cub.cuh>
#endif

namespace care{
namespace gpu{

#ifdef __NVCC__

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



        for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
            MSAColumnProperties* const properties_ptr = msa_column_properties + subjectIndex;
#if 0

            const int queryIndexBegin = candidates_per_subject_prefixsum[subjectIndex];
            const int queryIndexEnd = candidates_per_subject_prefixsum[subjectIndex+1];

            const int subjectLength = subject_sequences_lengths[subjectIndex];
            int startindex = 0;
            int endindex = subject_sequences_lengths[subjectIndex];

            for(int queryIndex = queryIndexBegin+threadIdx.x; queryIndex < queryIndexEnd; queryIndex += blockDim.x){
                const int shift = alignment_shifts[queryIndex];
                const BestAlignment_t flag = alignment_best_alignment_flags[queryIndex];
                const int queryLength = candidate_sequences_lengths[queryIndex];

                if(flag != BestAlignment_t::None){
                    const int shift = result.shift;
                    const int queryEndsAt = queryLength + shift;
                    startindex = min(startindex, shift);
                    endindex = max(startindex, queryEndsAt);
                }
            }
#else
            // We only want to consider the candidates with good alignments. the indices of those were determined in a previous step
            const int num_indices_for_this_subject = indices_per_subject[subject_index];
            const int* const indices_for_this_subject = indices + indices_per_subject_prefixsum[subject_index];

            const int subjectLength = subject_sequences_lengths[subjectIndex];
            int startindex = 0;
            int endindex = subject_sequences_lengths[subjectIndex];

            for(int index = threadIdx.x; index < num_indices_for_this_subject; index += blockDim.x){
                const int queryIndex = indices_for_this_subject[index];

                const int shift = alignment_shifts[queryIndex];
                const BestAlignment_t flag = alignment_best_alignment_flags[queryIndex];
                const int queryLength = candidate_sequences_lengths[queryIndex];

                if(flag != BestAlignment_t::None){
                    const int shift = result.shift;
                    const int queryEndsAt = queryLength + shift;
                    startindex = min(startindex, shift);
                    endindex = max(startindex, queryEndsAt);
                }
            }

#endif
            __shared__ int reductionbuffer1[32];
            __shared__ int reductionbuffer2[32];

            if(threadIdx.x < 32){
                reductionbuffer1[threadIdx.x] = std::numeric_limits<int>::max();
                reductionbuffer2[threadIdx.x] = std::numeric_limits<int>::min();
            }

            //block reduction startindex, endindex
            auto tile = cg::tiled_partition<32>(this_thread_block());
            auto minfunc = [](auto a, auto b){return a < b ? a : b;};
            auto maxfunc = [](auto a, auto b){return a > b ? a : b;};

            const int reduced_startindex = reduceTile(tile, startindex, minfunc);
            const int reduced_endindex = reduceTile(tile, endindex, maxfunc);

            if(tile.thread_rank() == 0){
                reductionbuffer1[threadIdx.x / 32] = reduced_startindex;
                reductionbuffer2[threadIdx.x / 32] = reduced_endindex;
            }
            __syncthreads();

            if(threadIdx.x < 32){
                const int final_reduced_startindex = reduceTile(tile, reductionbuffer1[threadIdx.x], minfunc);
                const int final_reduced_endindex = reduceTile(tile, reductionbuffer2[threadIdx.x], maxfunc);

                if(threadIdx.x == 0){
                    MSAColumnProperties my_columnproperties;

                    my_columnproperties.startindex = final_reduced_startindex;
                    my_columnproperties.endindex = final_reduced_endindex;
                    my_columnproperties.columnsToCheck = my_columnproperties.endindex - my_columnproperties.startindex;
                    my_columnproperties.subjectColumnsBegin_incl = max(-my_columnproperties.startindex, 0);
                    my_columnproperties.subjectColumnsEnd_excl = my_columnproperties.subjectColumnsBegin_incl + subjectLength;

                    *properties_ptr = my_columnproperties;
                }
            }
            __syncthreads();

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

        dim3 block(128, 1, 1);
        dim3 grid(n_subjects, 1, 1);

        msa_init_kernel<<<grid, block, 0, stream>>>(d_msa_column_properties,
                                                    d_alignment_shifts,
                                                    d_alignment_best_alignment_flags,
                                                    d_subject_sequences_lengths,
                                                    d_candidate_sequences_lengths,
                                                    d_indices,
                                                    d_indices_per_subject,
                                                    d_indices_per_subject_prefixsum
                                                    n_subjects); CUERR;
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
                            int n_indices,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            int msa_max_column_count,
                            int blocks_per_msa){

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

        const int localBlockId = blockIdx.x % blocks_per_msa;

        //process multiple sequence alignment of each subject
        //for each column in msa, find consensus and support
        for(unsigned subjectIndex = blockIdx.x / blocks_per_msa; subjectIndex < n_subjects; subjectIndex += gridDim.x / blocks_per_msa){
            const int subjectColumnsBegin_incl = d_msa_column_properties[subjectIndex].subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = d_msa_column_properties[subjectIndex].subjectColumnsEnd_excl;

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
            int* const my_orig_coverage = d_origCoverage + subjectIndex * msa_weights_pitch_floats;

            for(int column = localBlockId * blockDim.x + threadIdx.x; column < msa_max_column_count; column += blocks_per_msa * blockDim.x){

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

                    assert(base == 'A' || base == 'C' || base == 'G' || base == 'T');

                    Aw += (base == 'A' ? weight : 0);
                    Cw += (base == 'C' ? weight : 0);
                    Gw += (base == 'G' ? weight : 0);
                    Tw += (base == 'T' ? weight : 0);

                    As += (base == 'A');
                    Cs += (base == 'C');
                    Gs += (base == 'G');
                    Ts += (base == 'T');

                    columnCoverage += 1;
                }

                assert(columnCoverage > 0);

                const float columnWeight = Aw + Cw + Gw + Tw;
                float consWeight = Aw;
                char cons = 'A';

                cons = Cw > consWeight ? 'C' : cons;
                consWeight = Cw > consWeight ? Cw : consWeight;

                cons = Gw > consWeight ? 'G' : cons;
                consWeight = Gw > consWeight ? Gw : consWeight;

                cons = Tw > consWeight ? 'T' : cons;
                consWeight = Tw > consWeight ? Tw : consWeight;

                my_consensus[i] = cons;
                my_support[i] = consWeight / columnWeight;
                my_coverage[i] = columnCoverage;

                if(subjectColumnsBegin_incl <= i && i < subjectColumnsEnd_excl){
                    const char subjectbase = my_multiple_sequence_alignment[i];
                    if(subjectbase == 'A'){
                        my_orig_weights[i] = Aw;
                        my_orig_coverage[i] = As;
                    }else if(subjectbase == 'C'){
                        my_orig_weights[i] = Cw;
                        my_orig_coverage[i] = Cs;
                    }else if(subjectbase == 'G'){
                        my_orig_weights[i] = Gw;
                        my_orig_coverage[i] = Gs;
                    }else if(subjectbase == 'T'){
                        my_orig_weights[i] = Tw;
                        my_orig_coverage[i] = Ts;
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
                            int n_indices,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            int msa_max_column_count,
                            cudaStream_t stream){

        const int blocksize = 128;
        const int blocks_per_msa = SDIV(msa_max_column_count, blocksize);

        dim3 block(blocksize, 1, 1);
        dim3 grid(n_subjects * blocks_per_msa, 1, 1);

        msa_find_consensus_kernel<<<grid, block, 0, stream>>>(d_consensus,
                                                            d_support,
                                                            d_coverage,
                                                            d_origWeights,
                                                            d_origCoverage,
                                                            d_unpacked_subjects,
                                                            d_multiple_sequence_alignments,
                                                            d_multiple_sequence_alignment_weights,
                                                            d_msa_column_properties,
                                                            d_candidates_per_subject_prefixsum,
                                                            d_indices_per_subject,
                                                            d_indices_per_subject_prefixsum,
                                                            n_subjects,
                                                            n_queries,
                                                            n_indices,
                                                            msa_pitch,
                                                            msa_weights_pitch,
                                                            msa_max_column_count,
                                                            blocks_per_msa); CUERR;

    }


    


#endif

}
}
