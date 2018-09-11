#include "../inc/msa_kernels.hpp"
#include "../inc/hpc_helpers.cuh"
#include "../inc/shifted_hamming_distance.hpp"
#include "../inc/bestalignment.hpp"
#include "../inc/qualityscoreweights.hpp"
#include "../inc/cudareduce.cuh"

#include <limits>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace care{

namespace msa{

#ifdef __NVCC__

    __global__
    void msa_init_kernel(
                        MSAColumnProperties* const __restrict__ columnProperties,
                        const shd::Result_t* const __restrict__ results,
                        const BestAlignment_t* const __restrict__ flags,
                        const int* const __restrict__ subjectLengths,
                        const int* const __restrict__ queryLengths,
                        const int* const __restrict__ NqueriesPrefixSum,
                        int n_subjects){

        for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
            MSAColumnProperties* const properties_ptr = columnProperties + subjectIndex;

            const int queryIndexBegin = NqueriesPrefixSum[subjectIndex];
            const int queryIndexEnd = NqueriesPrefixSum[subjectIndex+1];

            const int subjectLength = subjectLengths[subjectIndex];
            int startindex = 0;
            int endindex = subjectLengths[subjectIndex];

            for(int queryIndex = queryIndexBegin+threadIdx.x; queryIndex < queryIndexEnd; queryIndex += blockDim.x){
                const shd::Result_t result = results[queryIndex];
                const BestAlignment_t flag = flags[queryIndex];
                const int queryLength = queryLengths[queryIndex];

                if(flag != BestAlignment_t::None){
                    const int shift = result.shift;
                    const int queryEndsAt = queryLength + shift;
                    startindex = min(startindex, shift);
                    endindex = max(startindex, queryEndsAt);
                }
            }

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

    __global__
    void msa_add_sequences_kernel(
                            char* const __restrict__ multiple_sequence_alignments,
                            float* const __restrict__ multiple_sequence_alignment_weights,
                            const shd::Result_t* const __restrict__ results,
                            const BestAlignment_t* const __restrict__ flags,
                            const char* const __restrict__ unpacked_subjects,
                            const int* const __restrict__ subjectLengths,
                            const char* const __restrict__ unpacked_queries,
                            const int* const __restrict__ queryLengths,
                            const char* const __restrict__ subjectQualityScores,
                            const char* const __restrict__ queryQualityScores,
                            const MSAColumnProperties* const __restrict__ columnProperties,
                            const int* const __restrict__ NqueriesPrefixSum,
                            int n_subjects,
                            int max_sequence_length,
                            bool canUseQualityScores,
                            size_t sequencepitch,
                            size_t msa_row_pitch,
                            size_t msa_weights_row_pitch){

        const int n_queries = NqueriesPrefixSum[n_subjects];
        const int msa_row_length = max_sequence_length * 3 - 2;
        const size_t msa_weights_row_pitch_floats = msa_weights_row_pitch / sizeof(float);

        //copy each subject into the top row of its multiple sequence alignment
        for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
            const int subjectColumnsBegin_incl = columnProperties[subjectIndex].subjectColumnsBegin_incl;

            const unsigned offset1 = msa_row_pitch * (subjectIndex+NqueriesPrefixSum[subjectIndex]);
            const unsigned offset2 = msa_weights_row_pitch_floats * (subjectIndex+NqueriesPrefixSum[subjectIndex]);

            char* const multiple_sequence_alignment = multiple_sequence_alignments + offset1;
            float* const multiple_sequence_alignment_weight = multiple_sequence_alignment_weights + offset2;

            const int subjectLength = subjectLengths[subjectIndex];
            const char* const subject = unpacked_subjects + subjectIndex * sequencepitch;
            const char* const subjectQualityScore = subjectQualityScores + subjectIndex * sequencepitch;

            for(int i = threadIdx.x; i < subjectLength; i+= blockDim.x){
                multiple_sequence_alignment[subjectColumnsBegin_incl + i] = subject[i];
                multiple_sequence_alignment_weight[subjectColumnsBegin_incl + i] = canUseQualityScores ?
                                                                    (float)d_qscore_to_weight[(unsigned char)subjectQualityScore[i]]
                                                                    : 1.0f;
            }
        }

        // copy each query into the multiple sequence alignment of its subject
        for(unsigned queryIndex = blockIdx.x; queryIndex < n_queries; queryIndex += gridDim.x){

            const shd::Result_t result = results[queryIndex];
            const BestAlignment_t flag = flags[queryIndex];
            const int queryLength = queryLengths[queryIndex];

            //find subjectindex
            int subjectIndex = 0;
            for(; subjectIndex < n_subjects; subjectIndex++){
                if(queryIndex < NqueriesPrefixSum[subjectIndex+1])
                    break;
            }

            const int subjectColumnsBegin_incl = columnProperties[subjectIndex].subjectColumnsBegin_incl;

            const int localQueryIndex = queryIndex - NqueriesPrefixSum[subjectIndex];

            const int defaultcolumnoffset = subjectColumnsBegin_incl + result.shift;

            const unsigned offset1 = msa_row_pitch * (subjectIndex+NqueriesPrefixSum[subjectIndex]);
            const unsigned offset2 = msa_weights_row_pitch_floats * (subjectIndex+NqueriesPrefixSum[subjectIndex]);

            char* const multiple_sequence_alignment = multiple_sequence_alignments + offset1;
            float* const multiple_sequence_alignment_weight = multiple_sequence_alignment_weights + offset2;

            const char* const query = unpacked_queries + queryIndex * sequencepitch;
            const char* const queryQualityScore = queryQualityScores + queryIndex * sequencepitch;

            if(flag == BestAlignment_t::None){
                //if the query could not be aligned, fill its row with 'Z'
                for(int i = threadIdx.x; i < msa_row_length; i+= blockDim.x){
                    const int row = 1 + localQueryIndex;
                    multiple_sequence_alignment[row * msa_row_pitch + i] = 'Z';
                    multiple_sequence_alignment_weight[row * msa_weights_row_pitch_floats + i] = 0.0f;
                }
            }else{
                //copy query into msa
                for(int i = threadIdx.x; i < queryLength; i+= blockDim.x){
                    const int globalIndex = defaultcolumnoffset + i;
                    const int row = 1 + localQueryIndex;
                    multiple_sequence_alignment[row * msa_row_pitch + globalIndex] = query[i];
                    multiple_sequence_alignment_weight[row * msa_weights_row_pitch_floats + i]
                                        = canUseQualityScores ?
                                            (float)d_qscore_to_weight[(unsigned char)queryQualityScore[i]]
                                            : 1.0f;
                }
            }
        }
    }

    __global__
    void msa_find_consensus_kernel(
                            char* const __restrict__ consensus,
                            float* const __restrict__ support,
                            int* const __restrict__ coverage,
                            float* const __restrict__ origWeights,
                            int* const __restrict__ origCoverage,
                            const char* const __restrict__ unpacked_subjects,
                            const char* const __restrict__ multiple_sequence_alignments,
                            const float* const __restrict__ multiple_sequence_alignment_weights,
                            const MSAColumnProperties* const __restrict__ columnProperties,
                            const int* const __restrict__ NqueriesPrefixSum,
                            int n_subjects,
                            int max_sequence_length,
                            size_t sequencepitch,
                            size_t msa_row_pitch,
                            size_t msa_weights_row_pitch,
                            int blocks_per_msa){

        const int msa_row_length = max_sequence_length * 3 - 2;
        const size_t msa_weights_row_pitch_floats = msa_weights_row_pitch / sizeof(float);

        const int localBlockId = blockIdx.x % blocks_per_msa;

        //process multiple sequence alignment of each subject
        //for each column in msa, find consensus and support
        for(unsigned subjectIndex = blockIdx.x / blocks_per_msa; subjectIndex < n_subjects; subjectIndex += gridDim.x / blocks_per_msa){
            const int subjectColumnsBegin_incl = columnProperties[subjectIndex].subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = columnProperties[subjectIndex].subjectColumnsEnd_excl;

            const char* const subject = unpacked_subjects + subjectIndex * sequencepitch;

            const int msa_rows = subjectIndex + (NqueriesPrefixSum[subjectIndex+1] - NqueriesPrefixSum[subjectIndex]);

            const unsigned offset1 = msa_row_pitch * (subjectIndex+NqueriesPrefixSum[subjectIndex]);
            const unsigned offset2 = msa_weights_row_pitch_floats * (subjectIndex+NqueriesPrefixSum[subjectIndex]);

            const char* const multiple_sequence_alignment = multiple_sequence_alignments + offset1;
            const float* const multiple_sequence_alignment_weight = multiple_sequence_alignment_weights + offset2;

            char* const my_consensus = consensus + subjectIndex * msa_row_pitch;
            float* const my_support = support + subjectIndex * msa_weights_row_pitch_floats;
            int* const my_coverage = coverage + subjectIndex * msa_weights_row_pitch_floats;

            float* const my_orig_weights = origWeights + subjectIndex * msa_weights_row_pitch_floats;
            int* const my_orig_coverage = origCoverage + subjectIndex * msa_weights_row_pitch_floats;

            for(int i = localBlockId * blockDim.x + threadIdx.x; i < msa_row_length; i+= blocks_per_msa * blockDim.x){

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
                    const char base = multiple_sequence_alignment[row * msa_row_pitch + i];
                    const float weight = multiple_sequence_alignment_weight[row * msa_weights_row_pitch_floats + i];

                    assert(base == 'A' || base == 'C' || base == 'G' || base == 'T' || base == 'Z');

                    Aw += (base == 'A' ? weight : 0);
                    Cw += (base == 'C' ? weight : 0);
                    Gw += (base == 'G' ? weight : 0);
                    Tw += (base == 'T' ? weight : 0);

                    As += (base == 'A');
                    Cs += (base == 'C');
                    Gs += (base == 'G');
                    Ts += (base == 'T');

                    columnCoverage += (base != 'Z' ? 1 : 0);
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
                    const char subjectbase = subject[i];
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

    void call_msa_init_kernel_async(
                        MSAColumnProperties* d_columnProperties,
                        const shd::Result_t* d_results,
                        const BestAlignment_t* d_flags,
                        const int* d_subjectLengths,
                        const int* d_queryLengths,
                        const int* d_NqueriesPrefixSum,
                        int n_subjects,
                        int n_queries,
                        cudaStream_t stream){

        dim3 block(128, 1, 1);
        dim3 grid(SDIV(n_queries, block.x), 1, 1);

        msa_init_kernel<<<grid, block, 0, stream>>>(d_columnProperties,
                                                    d_results,
                                                    d_flags,
                                                    d_subjectLengths,
                                                    d_queryLengths,
                                                    d_NqueriesPrefixSum,
                                                    n_subjects); CUERR;
    }

    void call_msa_add_sequences_kernel_async(
                            char* d_multiple_sequence_alignments,
                            float* d_multiple_sequence_alignment_weights,
                            const shd::Result_t* d_results,
                            const BestAlignment_t* d_flags,
                            const char* d_unpacked_subjects,
                            const int* d_subjectLengths,
                            const char* d_unpacked_queries,
                            const int* d_queryLengths,
                            const char* d_subjectQualityScores,
                            const char* d_queryQualityScores,
                            const MSAColumnProperties* d_columnProperties,
                            const int* d_NqueriesPrefixSum,
                            int n_subjects,
                            int n_queries,
                            int max_sequence_length,
                            bool canUseQualityScores,
                            size_t sequencepitch,
                            size_t msa_row_pitch,
                            size_t msa_weights_row_pitch,
                            cudaStream_t stream){

        dim3 block(128, 1, 1);
        dim3 grid(SDIV(n_queries, block.x), 1, 1);

        msa_add_sequences_kernel<<<grid, block, 0, stream>>>(d_multiple_sequence_alignments,
                                                        d_multiple_sequence_alignment_weights,
                                                        d_results,
                                                        d_flags,
                                                        d_unpacked_subjects,
                                                        d_subjectLengths,
                                                        d_unpacked_queries,
                                                        d_queryLengths,
                                                        d_subjectQualityScores,
                                                        d_queryQualityScores,
                                                        d_columnProperties,
                                                        d_NqueriesPrefixSum,
                                                        n_subjects,
                                                        max_sequence_length,
                                                        canUseQualityScores,
                                                        sequencepitch,
                                                        msa_row_pitch,
                                                        msa_weights_row_pitch); CUERR;

    }

    void call_msa_find_consensus_kernel_async(
                            char* const d_consensus,
                            float* const d_support,
                            int* const d_coverage,
                            float* const d_origWeights,
                            int* const d_origCoverage,
                            const char* d_unpacked_subjects,
                            const char* const d_multiple_sequence_alignments,
                            const float* const d_multiple_sequence_alignment_weights,
                            const MSAColumnProperties* d_columnProperties,
                            const int* const d_NqueriesPrefixSum,
                            int n_subjects,
                            int n_queries,
                            int max_sequence_length,
                            size_t sequencepitch,
                            size_t msa_row_pitch,
                            size_t msa_weights_row_pitch,
                            cudaStream_t stream){

        const int msa_row_length = max_sequence_length * 3 - 2;
        const int blocksize = 128;
        const int blocks_per_msa = SDIV(msa_row_length, blocksize);

        dim3 block(128, 1, 1);
        dim3 grid(n_queries * blocks_per_msa, 1, 1);

        msa_find_consensus_kernel<<<grid, block, 0, stream>>>(d_consensus,
                                                            d_support,
                                                            d_coverage,
                                                            d_origWeights,
                                                            d_origCoverage,
                                                            d_unpacked_subjects,
                                                            d_multiple_sequence_alignments,
                                                            d_multiple_sequence_alignment_weights,
                                                            d_columnProperties,
                                                            d_NqueriesPrefixSum,
                                                            n_subjects,
                                                            max_sequence_length,
                                                            sequencepitch,
                                                            msa_row_pitch,
                                                            msa_weights_row_pitch,
                                                            blocks_per_msa); CUERR;

    }

#endif

}
}
