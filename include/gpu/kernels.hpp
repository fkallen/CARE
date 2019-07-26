#ifndef CARE_GPU_KERNELS_HPP
#define CARE_GPU_KERNELS_HPP

#include "../hpc_helpers.cuh"
//#include <gpu/bestalignment.hpp>
#include <bestalignment.hpp>
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

struct AlignmentResultPointers{
    int* scores;
    int* overlaps;
    int* shifts;
    int* nOps;
    bool* isValid;
    BestAlignment_t* bestAlignmentFlags;
};

struct MSAPointers{
    char* consensus;
    float* support;
    int* coverage;
    float* origWeights;
    int* origCoverages;
    MSAColumnProperties* msaColumnProperties;
    int* counts;
    float* weights;
};

struct ReadSequencesPointers{
    char* subjectSequencesData;
    char* candidateSequencesData;
    char* subjectSequencesDataTransposed;
    char* candidateSequencesDataTransposed;
    int* subjectSequencesLength;
    int* candidateSequencesLength;
};

struct ReadQualitiesPointers{
    char* subjectQualities;
    char* candidateQualities;
    char* candidateQualitiesTransposed;
};

struct CorrectionResultPointers{
        char* correctedSubjects;
        char* correctedCandidates;
        int* numCorrectedCandidates;
        bool* subjectIsCorrected;
        int* indicesOfCorrectedCandidates;
        bool* isHighQualitySubject;
        int* highQualitySubjectIndices;
        int* numHighQualitySubjectIndices;
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


void call_cuda_popcount_shifted_hamming_distance_with_revcompl_tiled_kernel_async(
            AlignmentResultPointers d_alignmentresultpointers,
            ReadSequencesPointers d_sequencePointers,
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
            AlignmentResultPointers d_alignmentresultpointers,
            ReadSequencesPointers d_sequencePointers,
            const int* d_candidates_per_subject_prefixsum,
            int n_subjects,
            int n_queries,
            float min_overlap_ratio,
            int min_overlap,
            float estimatedErrorrate,
            cudaStream_t stream,
            KernelLaunchHandle& handle);



void call_cuda_filter_alignments_by_mismatchratio_kernel_async(
            AlignmentResultPointers d_alignmentresultpointers,
            const int* d_candidates_per_subject_prefixsum,
            int n_subjects,
            int n_candidates,
            float mismatchratioBaseFactor,
            float goodAlignmentsCountThreshold,
            cudaStream_t stream,
            KernelLaunchHandle& handle);


void call_msa_init_kernel_async_exp(
            MSAPointers d_msapointers,
            AlignmentResultPointers d_alignmentresultpointers,
            ReadSequencesPointers d_sequencePointers,
            const int* d_indices,
            const int* d_indices_per_subject,
            const int* d_indices_per_subject_prefixsum,
            int n_subjects,
            int n_queries,
            cudaStream_t stream,
            KernelLaunchHandle& handle);

void call_msa_update_properties_kernel_async(
            MSAPointers d_msapointers,
            const int* d_indices_per_subject,
            int n_subjects,
            size_t msa_weights_pitch,
            cudaStream_t stream,
            KernelLaunchHandle& handle);


void call_msa_add_sequences_kernel_implicit_shared_async(
            MSAPointers d_msapointers,
            AlignmentResultPointers d_alignmentresultpointers,
            ReadSequencesPointers d_sequencePointers,
            ReadQualitiesPointers d_qualityPointers,
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
            MSAPointers d_msapointers,
            AlignmentResultPointers d_alignmentresultpointers,
            ReadSequencesPointers d_sequencePointers,
            ReadQualitiesPointers d_qualityPointers,
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
            MSAPointers d_msapointers,
            AlignmentResultPointers d_alignmentresultpointers,
            ReadSequencesPointers d_sequencePointers,
            ReadQualitiesPointers d_qualityPointers,
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
            MSAPointers d_msapointers,
            AlignmentResultPointers d_alignmentresultpointers,
            ReadSequencesPointers d_sequencePointers,
            ReadQualitiesPointers d_qualityPointers,
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
            MSAPointers d_msapointers,
            AlignmentResultPointers d_alignmentresultpointers,
            ReadSequencesPointers d_sequencePointers,
            ReadQualitiesPointers d_qualityPointers,
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


void call_msa_find_consensus_implicit_kernel_async(
            MSAPointers d_msapointers,
            ReadSequencesPointers d_sequencePointers,
            const int* d_indices_per_subject,
            int n_subjects,
            size_t encoded_sequence_pitch,
            size_t msa_pitch,
            size_t msa_weights_pitch,
            cudaStream_t stream,
            KernelLaunchHandle& handle);


void call_msa_correct_subject_implicit_kernel_async(
            MSAPointers d_msapointers,
            AlignmentResultPointers d_alignmentresultpointers,
            ReadSequencesPointers d_sequencePointers,
            CorrectionResultPointers d_correctionResultPointers,
            const int* d_indices,
            const int* d_indices_per_subject,
            const int* d_indices_per_subject_prefixsum,
            int n_subjects,
            size_t encoded_sequence_pitch,
            size_t sequence_pitch,
            size_t msa_pitch,
            size_t msa_weights_pitch,
            float estimatedErrorrate,
            float desiredAlignmentMaxErrorRate,
            float avg_support_threshold,
            float min_support_threshold,
            float min_coverage_threshold,
            float max_coverage_threshold,
            int k_region,
            int maximum_sequence_length,
            cudaStream_t stream,
            KernelLaunchHandle& handle);

void call_msa_correct_candidates_kernel_async_exp(
            MSAPointers d_msapointers,
            AlignmentResultPointers d_alignmentresultpointers,
            ReadSequencesPointers d_sequencePointers,
            CorrectionResultPointers d_correctionResultPointers,
            const int* d_indices,
            const int* d_indices_per_subject,
            const int* d_indices_per_subject_prefixsum,
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
            MSAPointers d_msapointers,
            AlignmentResultPointers d_alignmentresultpointers,
            ReadSequencesPointers d_sequencePointers,
            bool* d_shouldBeKept,
            const int* d_candidates_per_subject_prefixsum,
            int n_subjects,
            int n_candidates,
            int max_sequence_bytes,
            size_t encodedsequencepitch,
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
