#ifndef CARE_MSA_KERNELS_HPP
#define CARE_MSA_KERNELS_HPP

#include "../inc/shifted_hamming_distance.hpp"
#include "../inc/bestalignment.hpp"

namespace care{

#ifdef __NVCC__

    void call_msa_init_kernel_async(
                            char* d_multiple_sequence_alignments,
                            float* const d_multiple_sequence_alignment_weights,
                            const shd::AlignmentResult* d_results,
                            const BestAlignment_t* d_flags,
                            const char* d_unpacked_subjects,
                            const int* d_subjectLengths,
                            const char* d_unpacked_queries,
                            const int* d_queryLengths,
                            const char* const d_subjectQualityScores,
                            const char* const d_queryQualityScores,
                            const int* d_NqueriesPrefixSum,
                            int n_subjects,
                            int n_queries,
                            int max_sequence_length,
                            int subjectColumnsBegin_incl,
                            size_t sequencepitch,
                            size_t msa_row_pitch,
                            size_t msa_weights_row_pitch,
                            cudaStream_t stream);

    void call_msa_find_consensus_kernel_async(
                            char* d_consensus,
                            float* d_support,
                            int* d_coverage,
                            float* d_origWeights,
                            int* d_origCoverage,
                            const char* d_unpacked_subjects,
                            const char* d_multiple_sequence_alignments,
                            const float* d_multiple_sequence_alignment_weights,
                            const int* d_NqueriesPrefixSum,
                            int n_subjects,
                            int n_queries,
                            int max_sequence_length,
                            int subjectColumnsBegin_incl,
                            int subjectColumnsEnd_excl,
                            size_t sequencepitch,
                            size_t msa_row_pitch,
                            size_t msa_weights_row_pitch,
                            cudaStream_t stream);

#endif
}


#endif
