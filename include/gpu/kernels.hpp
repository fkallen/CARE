#ifndef CARE_GPU_KERNELS_HPP
#define CARE_GPU_KERNELS_HPP

#include "../hpc_helpers.cuh"
#include "bestalignment.hpp"
#include <msa.hpp>

#include <config.hpp>

#include <map>


namespace care {
namespace gpu {


#ifdef __NVCC__
struct MSAColumnProperties{
    //int startindex;
    //int endindex;
    //int columnsToCheck;
    int subjectColumnsBegin_incl;
    int subjectColumnsEnd_excl;
    int firstColumn_incl;
    int lastColumn_excl;
};


enum class KernelId {
    PopcountSHDTiled,
	FindBestAlignmentExp,
	FilterAlignmentsByMismatchRatio,
	MSAInitExp,
    MSAUpdateProperties,
	MSAAddSequences,
	MSAFindConsensus,
	MSACorrectSubject,
	MSACorrectCandidates,
    MSAAddSequencesImplicitGlobal,
    MSAAddSequencesImplicitShared,
    MSAAddSequencesImplicitSharedTest,
    MSAAddSequencesImplicitSinglecol,
    MSAFindConsensusImplicit,
    MSACorrectSubjectImplicit,
    MSAFindCandidatesOfDifferentRegion,
};

struct KernelLaunchConfig {
	int threads_per_block;
	int smem;
};

constexpr bool operator<(const KernelLaunchConfig& lhs, const KernelLaunchConfig& rhs){
	return lhs.threads_per_block < rhs.threads_per_block
	       && lhs.smem < rhs.smem;
}

struct KernelProperties {
	int max_blocks_per_SM = 1;
};

struct KernelLaunchHandle {
	int deviceId;
	cudaDeviceProp deviceProperties;
	std::map<KernelId, std::map<KernelLaunchConfig, KernelProperties> > kernelPropertiesMap;
};

KernelLaunchHandle make_kernel_launch_handle(int deviceId);

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
			float mismatchratioBaseFactor,
			float goodAlignmentsCountThreshold,
			cudaStream_t stream,
			KernelLaunchHandle& handle);


void call_cuda_popcount_shifted_hamming_distance_with_revcompl_tiled_kernel_async(
            int* d_alignment_scores,
            int* d_alignment_overlaps,
            int* d_alignment_shifts,
            int* d_alignment_nOps,
            bool* d_alignment_isValid,
            const char* __restrict__ subject_sequences_data,
            const char* __restrict__ candidate_sequences_data,
            const int* __restrict__ subject_sequences_lengths,
            const int* __restrict__ candidate_sequences_lengths,
            const int* d_candidates_per_subject_prefixsum,
            const int* h_candidates_per_subject,
            const int* d_candidates_per_subject,
            int n_subjects,
            int n_queries,
            size_t encodedsequencepitch,
            int max_sequence_bytes,
            int min_overlap,
            float maxErrorRate,
            float min_overlap_ratio,
            cudaStream_t stream,
            KernelLaunchHandle& handle);





void call_cuda_find_best_alignment_kernel_async_exp(
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
            float min_overlap_ratio,
            int min_overlap,
            float estimatedErrorrate,
            cudaStream_t stream,
            KernelLaunchHandle& handle);



void call_cuda_filter_alignments_by_mismatchratio_kernel_async(
            BestAlignment_t* d_alignment_best_alignment_flags,
            const int* d_alignment_overlaps,
            const int* d_alignment_nOps,
            const int* d_candidates_per_subject_prefixsum,
            int n_subjects,
            int n_candidates,
            float mismatchratioBaseFactor,
            float goodAlignmentsCountThreshold,
            cudaStream_t stream,
            KernelLaunchHandle& handle);


void call_msa_init_kernel_async_exp(
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
            cudaStream_t stream,
            KernelLaunchHandle& handle);

void call_msa_update_properties_kernel_async(
            MSAColumnProperties* d_msa_column_properties,
            const int* d_coverage,
            const int* d_indices_per_subject,
            int n_subjects,
            size_t msa_weights_pitch,
            cudaStream_t stream,
            KernelLaunchHandle& handle);



void call_msa_add_sequences_kernel_exp_async(
            char* d_multiple_sequence_alignments,
            float* d_multiple_sequence_alignment_weights,
            const int* d_alignment_shifts,
            const BestAlignment_t* d_alignment_best_alignment_flags,
            const int* d_alignment_overlaps,
            const int* d_alignment_nOps,
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
            const int* h_num_indices,
            const int* d_num_indices,
            bool canUseQualityScores,
            float desiredAlignmentMaxErrorRate,
            int maximum_sequence_length,
            int max_sequence_bytes,
            size_t encoded_sequence_pitch,
            size_t quality_pitch,
            size_t msa_row_pitch,
            size_t msa_weights_row_pitch,
            cudaStream_t stream,
            KernelLaunchHandle& handle);





void call_msa_add_sequences_kernel_implicit_shared_async(
            int* d_counts,
            float* d_weights,
            int* d_coverage,
            const int* d_alignment_shifts,
            const BestAlignment_t* d_alignment_best_alignment_flags,
            const int* d_alignment_overlaps,
            const int* d_alignment_nOps,
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
            const int* d_blocks_per_subject_prefixsum,
            int n_subjects,
            int n_queries,
            const int* h_num_indices,
            const int* d_num_indices,
            float expectedAffectedIndicesFraction,
            bool canUseQualityScores,
            float desiredAlignmentMaxErrorRate,
            int maximum_sequence_length,
            int max_sequence_bytes,
            size_t quality_pitch,
            size_t msa_row_pitch,
            size_t msa_weights_row_pitch,
            cudaStream_t stream,
            KernelLaunchHandle& handle);

void call_msa_add_sequences_kernel_implicit_shared_testwithsubjectselection_async(
    int* d_counts,
    float* d_weights,
    int* d_coverage,
    const int* d_alignment_shifts,
    const BestAlignment_t* d_alignment_best_alignment_flags,
    const int* d_alignment_overlaps,
    const int* d_alignment_nOps,
    const char* d_subject_sequences_data,
    const char* d_candidate_sequences_data,
    const int* d_subject_sequences_lengths,
    const int* d_candidate_sequences_lengths,
    const char* d_subject_qualities,
    const char* d_candidate_qualities,
    const MSAColumnProperties*  d_msa_column_properties,
    const int* d_candidates_per_subject_prefixsum,
    const int* d_active_candidate_indices,
    const int* d_active_candidate_indices_per_subject,
    const int* d_active_candidate_indices_per_subject_prefixsum,
    const int* d_active_subject_indices,
    int n_subjects,
    int n_queries,
    const int* d_num_active_candidate_indices,
    const int* h_num_active_candidate_indices,
    const int* d_num_active_subject_indices,
    const int* h_num_active_subject_indices,
    bool canUseQualityScores,
    float desiredAlignmentMaxErrorRate,
    int maximum_sequence_length,
    int max_sequence_bytes,
    size_t encoded_sequence_pitch,
    size_t quality_pitch,
    size_t msa_row_pitch,
    size_t msa_weights_row_pitch,
    cudaStream_t stream,
    KernelLaunchHandle& handle,
    bool debug);

void call_msa_add_sequences_kernel_implicit_global_async(
            int* d_counts,
            float* d_weights,
            int* d_coverage,
            const int* d_alignment_shifts,
            const BestAlignment_t* d_alignment_best_alignment_flags,
            const int* d_alignment_overlaps,
            const int* d_alignment_nOps,
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
            const int* h_num_indices,
            const int* d_num_indices,
            float expectedAffectedIndicesFraction,
            bool canUseQualityScores,
            float desiredAlignmentMaxErrorRate,
            int maximum_sequence_length,
            int max_sequence_bytes,
            size_t encoded_sequence_pitch,
            size_t quality_pitch,
            size_t msa_row_pitch,
            size_t msa_weights_row_pitch,
            cudaStream_t stream,
            KernelLaunchHandle& handle,
            bool debug = false);

void call_msa_add_sequences_implicit_singlecol_kernel_async(
            int* d_counts,
            float* d_weights,
            int* d_coverage,
            const int* d_alignment_shifts,
            const BestAlignment_t* d_alignment_best_alignment_flags,
            const int* d_alignment_overlaps,
            const int* d_alignment_nOps,
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
            bool canUseQualityScores,
            float desiredAlignmentMaxErrorRate,
            int maximum_sequence_length,
            int max_sequence_bytes,
            size_t encoded_sequence_pitch,
            size_t quality_pitch,
            size_t msa_weights_pitch,
            cudaStream_t stream,
            KernelLaunchHandle& handle,
            const read_number* d_subject_read_ids,
            bool debug = false);

void call_msa_add_sequences_kernel_implicit_async(
            int* d_counts,
            float* d_weights,
            int* d_coverage,
            const int* d_alignment_shifts,
            const BestAlignment_t* d_alignment_best_alignment_flags,
            const int* d_alignment_overlaps,
            const int* d_alignment_nOps,
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
            const int* h_num_indices,
            const int* d_num_indices,
            float expectedAffectedIndicesFraction,
            bool canUseQualityScores,
            float desiredAlignmentMaxErrorRate,
            int maximum_sequence_length,
            int max_sequence_bytes,
            size_t encoded_sequence_pitch,
            size_t quality_pitch,
            size_t msa_row_pitch,
            size_t msa_weights_row_pitch,
            cudaStream_t stream,
            KernelLaunchHandle& handle,
            bool debug = false);


void call_msa_find_consensus_kernel_async(
            char* d_consensus,
            float* d_support,
            int* d_coverage,
            float* d_origWeights,
            int* d_origCoverages,
            int* d_counts,
            float* d_weights,
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
            cudaStream_t stream,
            KernelLaunchHandle& handle);

void call_msa_find_consensus_implicit_kernel_async(
            const int* d_counts,
            const float* d_weights,
            char* d_consensus,
            float* d_support,
            const int* d_coverage,
            float* d_origWeights,
            int* d_origCoverages,
            const char* d_subject_sequences_data,
            const int* d_indices_per_subject,
            const MSAColumnProperties* d_msa_column_properties,
            int n_subjects,
            size_t encoded_sequence_pitch,
            size_t msa_pitch,
            size_t msa_weights_pitch,
            cudaStream_t stream,
            KernelLaunchHandle& handle);

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
			float estimatedErrorrate,
			float avg_support_threshold,
			float min_support_threshold,
			float min_coverage_threshold,
			int k_region,
			int maximum_sequence_length,
			cudaStream_t stream,
			KernelLaunchHandle& handle);

void call_msa_correct_subject_implicit_kernel_async(
            const char* d_consensus,
            const float* d_support,
            const int* d_coverage,
            const int* d_origCoverages,
            const MSAColumnProperties* d_msa_column_properties,
            const int* d_indices_per_subject,
            const char* d_subject_sequences_data,
            bool* d_is_high_quality_subject,
            char* d_corrected_subjects,
            bool* d_subject_is_corrected,
            int n_subjects,
            size_t encoded_sequence_pitch,
            size_t sequence_pitch,
            size_t msa_pitch,
            size_t msa_weights_pitch,
            float estimatedErrorrate,
            float avg_support_threshold,
            float min_support_threshold,
            float min_coverage_threshold,
            int k_region,
            int maximum_sequence_length,
            cudaStream_t stream,
            KernelLaunchHandle& handle);

void call_msa_correct_candidates_kernel_async_exp(
            const char* d_consensus,
            const float* d_support,
            const int* d_coverage,
            const int* d_origCoverages,
            const MSAColumnProperties* d_msa_column_properties,
            const int* d_candidate_sequences_lengths,
            const int* d_indices,
            const int* d_indices_per_subject,
            const int* d_indices_per_subject_prefixsum,
            const int* d_high_quality_subject_indices,
            const int* d_num_high_quality_subject_indices,
            const int* d_alignment_shifts,
            const BestAlignment_t* d_alignment_best_alignment_flags,
            int* d_num_corrected_candidates,
            char* d_corrected_candidates,
            int* d_indices_of_corrected_candidates,
            int n_subjects,
            int n_queries,
            const int* d_num_indices,
            size_t sequence_pitch,
            size_t msa_pitch,
            size_t msa_weights_pitch,
            float min_support_threshold,
            float min_coverage_threshold,
            int new_columns_to_correct,
            int maximum_sequence_length,
            cudaStream_t stream,
            KernelLaunchHandle& handle);

void call_msa_findCandidatesOfDifferentRegion_kernel_async(
            bool* d_shouldBeKept,
            const char* d_subject_sequences_data,
            const char* d_candidate_sequences_data,
            const int* d_subject_sequences_lengths,
            const int* d_candidate_sequences_lengths,
            const int* d_candidates_per_subject_prefixsum,
            const int* d_alignment_shifts,
            const BestAlignment_t* d_alignment_best_alignment_flags,
            int n_subjects,
            int n_candidates,
            int max_sequence_bytes,
            size_t encodedsequencepitch,
            const char* d_consensus,
            const int* d_counts,
            const float* d_weights,
            const MSAColumnProperties* d_msa_column_properties,
            size_t msa_pitch,
            size_t msa_weights_pitch,
            const int* d_indices,
            const int* d_indices_per_subject,
            const int* d_indices_per_subject_prefixsum,
            int dataset_coverage,
            cudaStream_t stream,
            KernelLaunchHandle& handle,
            const unsigned int* d_readids,
            bool debug = false);








#endif //ifdef __NVCC__

}
}


#endif
