#ifndef CARE_GPU_KERNELS_HPP
#define CARE_GPU_KERNELS_HPP

#include <hpc_helpers.cuh>
#include <gpu/kernellaunch.hpp>

//#include <gpu/bestalignment.hpp>
#include <bestalignment.hpp>
#include <msa.hpp>
#include <correctionresultprocessing.hpp>

#include <config.hpp>

#include <map>


namespace care {
namespace gpu {


#ifdef __NVCC__

struct AnchorHighQualityFlag{
    char data;

    __host__ __device__
    bool hq() const{
        return data == 1;
    }

    __host__ __device__
    void hq(bool isHq){
        data = isHq ? 1 : 0;
    }
};

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
    unsigned int* subjectSequencesData;
    unsigned int* candidateSequencesData;
    int* subjectSequencesLength;
    int* candidateSequencesLength;
    unsigned int* transposedCandidateSequencesData;
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
        AnchorHighQualityFlag* isHighQualitySubject;
        int* highQualitySubjectIndices;
        int* numHighQualitySubjectIndices;
        int* num_uncorrected_positions_per_subject;
        int* uncorrected_positions_per_subject;
};





void call_popcount_shifted_hamming_distance_kernel_async(
                int* d_alignment_overlaps,
                int* d_alignment_shifts,
                int* d_alignment_nOps,
                bool* d_alignment_isValid,
                BestAlignment_t* d_alignment_best_alignment_flags,
                const unsigned int* d_subjectSequencesData,
                const unsigned int* d_candidateSequencesData,
                const int* d_subjectSequencesLength,
                const int* d_candidateSequencesLength,
    			const int* d_candidates_per_subject_prefixsum,
                const int* h_candidates_per_subject,
                const int* d_candidates_per_subject,
                const int* d_anchorIndicesOfCandidates,
    			int n_subjects,
    			int n_queries,
                int maximumSequenceLength,
                int encodedSequencePitchInInts2Bit,
    			int min_overlap,
    			float maxErrorRate,
                float min_overlap_ratio,
                float estimatedNucleotideErrorRate,
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
            KernelLaunchHandle& handle,
            read_number debugsubjectreadid = read_number(-1));

void callSelectIndicesOfGoodCandidatesKernelAsync(
            int* d_indicesOfGoodCandidates,
            int* d_numIndicesPerAnchor,
            int* d_totalNumIndices,
            const BestAlignment_t* d_alignmentFlags,
            const int* d_candidates_per_subject,
            const int* d_candidates_per_subject_prefixsum,
            const int* d_anchorIndicesOfCandidates,
            int numAnchors,
            int numCandidates,
            cudaStream_t stream,
            KernelLaunchHandle& handle);

void callGetNumCorrectedCandidatesPerAnchorKernel(
            int* d_numIndicesPerAnchor,
            const bool* d_isCorrectedCandidate,
            const int* d_numGoodIndicesPerSubject,
            const int* d_candidates_per_subject_prefixsum,
            const int* d_anchorIndicesOfCandidates,
            int numAnchors,
            int numCandidates,
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
            const int* d_candidates_per_subject_prefixsum,
            int n_subjects,
            int n_queries,
            const bool* d_canExecute,
            cudaStream_t stream,
            KernelLaunchHandle& handle);

void call_msa_update_properties_kernel_async(
            MSAPointers d_msapointers,
            const int* d_indices_per_subject,
            int n_subjects,
            size_t msa_weights_pitch,
            const bool* d_canExecute,
            cudaStream_t stream,
            KernelLaunchHandle& handle);


void call_msa_add_sequences_kernel_implicit_async(
            MSAPointers d_msapointers,
            AlignmentResultPointers d_alignmentresultpointers,
            ReadSequencesPointers d_sequencePointers,
            ReadQualitiesPointers d_qualityPointers,
            const int* d_candidates_per_subject_prefixsum,
            const int* d_indices,
            const int* d_indices_per_subject,
            int n_subjects,
            int n_queries,
            const int* d_num_indices,
            float expectedAffectedIndicesFraction,
            bool canUseQualityScores,
            float desiredAlignmentMaxErrorRate,
            int maximum_sequence_length,
            int encodedSequencePitchInInts,
            size_t quality_pitch,
            size_t msa_row_pitch,
            size_t msa_weights_row_pitch,
            const bool* d_canExecute,
            cudaStream_t stream,
            KernelLaunchHandle& handle,
            bool debug = false);


void call_msa_find_consensus_implicit_kernel_async(
            MSAPointers d_msapointers,
            ReadSequencesPointers d_sequencePointers,
            const int* d_indices_per_subject,
            int n_subjects,
            int encodedSequencePitchInInts,
            size_t msa_pitch,
            size_t msa_weights_pitch,
            const bool* d_canExecute,
            cudaStream_t stream,
            KernelLaunchHandle& handle);


void call_msa_findCandidatesOfDifferentRegion_kernel_async(
            int* d_newIndices,
            int* d_newIndicesPerSubject,
            int* d_newNumIndices,
            MSAPointers d_msapointers,
            AlignmentResultPointers d_alignmentresultpointers,
            ReadSequencesPointers d_sequencePointers,
            bool* d_shouldBeKept,
            const int* d_candidates_per_subject_prefixsum,
            int n_subjects,
            int n_candidates,
            int encodedSequencePitchInInts,
            size_t msa_pitch,
            size_t msa_weights_pitch,
            const int* d_indices,
            const int* d_indices_per_subject,
            float desiredAlignmentMaxErrorRate,
            int dataset_coverage,
            const bool* d_canExecute,
            cudaStream_t stream,            
            KernelLaunchHandle& handle,
            const unsigned int* d_readids,
            bool debug = false);


void call_msa_correct_subject_implicit_kernel_async(
            MSAPointers d_msapointers,
            AlignmentResultPointers d_alignmentresultpointers,
            ReadSequencesPointers d_sequencePointers,
            CorrectionResultPointers d_correctionResultPointers,
            const int* d_indices,
            const int* d_indices_per_subject,
            int n_subjects,
            int encodedSequencePitchInInts,
            size_t sequence_pitch,
            size_t msa_pitch,
            size_t msa_weights_pitch,
            int maximumSequenceLength,
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


void callCompactCandidateCorrectionResultsKernel_async(
            char* __restrict__ d_compactedCorrectedCandidates,
            TempCorrectedSequence::Edit* __restrict__ d_compactedEditsPerCorrectedCandidate,
            const int* __restrict__ d_numCorrectedCandidatesPerAnchor,
            const int* __restrict__ d_numCorrectedCandidatesPerAnchorPrefixsum, //exclusive
            const int* __restrict__ d_high_quality_subject_indices,
            const int* __restrict__ d_num_high_quality_subject_indices,
            const int* __restrict__ d_candidates_per_subject_prefixsum,
            const char* __restrict__ d_correctedCandidates,
            const int* __restrict__ d_correctedCandidateLengths,
            const TempCorrectedSequence::Edit* __restrict__ d_editsPerCorrectedCandidate,
            size_t decodedSequencePitch,
            int numEditsThreshold,
            int n_subjects,
            cudaStream_t stream,
            KernelLaunchHandle& handle);


void callConstructAnchorResultsKernelAsync(
            TempCorrectedSequence::Edit* __restrict__ d_editsPerCorrectedSubject,
            int* __restrict__ d_numEditsPerCorrectedSubject,
            int doNotUseEditsValue,
            const int* __restrict__ d_indicesOfCorrectedSubjects,
            const int* __restrict__ d_numIndicesOfCorrectedSubjects,
            const bool* __restrict__ d_readContainsN,
            const unsigned int* __restrict__ d_uncorrectedSubjects,
            const int* __restrict__ d_subjectLengths,
            const char* __restrict__ d_correctedSubjects,
            int numEditsThreshold,
            size_t encodedSequencePitchInInts,
            size_t decodedSequencePitchInBytes,
            int numSubjects,
            cudaStream_t stream,
            KernelLaunchHandle& handle);


void callFlagCandidatesToBeCorrectedKernel_async(
            bool* __restrict__ d_candidateCanBeCorrected,
            int* __restrict__ d_numCorrectedCandidatesPerAnchor,
            const float* __restrict__ d_support,
            const int* __restrict__ d_coverages,
            const MSAColumnProperties* __restrict__ d_msaColumnProperties,
            const int* __restrict__ d_alignmentShifts,
            const int* __restrict__ d_candidateSequencesLengths,
            const int* __restrict__ d_anchorIndicesOfCandidates,
            const AnchorHighQualityFlag* __restrict__ d_hqflags,
            const int* __restrict__ candidatesPerSubjectPrefixsum,
            const int* __restrict__ localGoodCandidateIndices,
            const int* __restrict__ numLocalGoodCandidateIndicesPerSubject,
            size_t msa_weights_pitch_floats,
            float min_support_threshold,
            float min_coverage_threshold,
            int new_columns_to_correct,
            int n_subjects,
            int n_candidates,
            cudaStream_t stream,
            KernelLaunchHandle& handle);

void callCorrectCandidatesWithGroupKernel2_async(
            char* __restrict__ correctedCandidates,
            TempCorrectedSequence::Edit* __restrict__ d_editsPerCorrectedCandidate,
            int* __restrict__ d_numEditsPerCorrectedCandidate,
            const MSAColumnProperties* __restrict__ msaColumnProperties,
            const char* __restrict__ consensus,
            const float* __restrict__ support,
            const int* __restrict__ shifts,
            const BestAlignment_t* __restrict__ bestAlignmentFlags,
            const unsigned int* __restrict__ candidateSequencesData,
            const int* __restrict__ candidateSequencesLengths,
            const bool* __restrict__ d_candidateContainsN,
            const int* __restrict__ candidateIndicesOfCandidatesToBeCorrected,
            const int* __restrict__ numCandidatesToBeCorrected,
            const int* __restrict__ anchorIndicesOfCandidates,
            int doNotUseEditsValue,
            int numEditsThreshold,
            int n_subjects,
            int n_queries,
            int encodedSequencePitchInInts,
            size_t sequence_pitch,
            size_t msa_pitch,
            size_t msa_weights_pitch,
            int maximum_sequence_length,
            cudaStream_t stream,
            KernelLaunchHandle& handle);            



void callConversionKernel2BitTo2BitHiLoNN(
            const unsigned int* d_inputdata,
            size_t inputpitchInInts,
            unsigned int* d_outputdata,
            size_t outputpitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            cudaStream_t stream,
            KernelLaunchHandle& handle);

void callConversionKernel2BitTo2BitHiLoNT(
            const unsigned int* d_inputdata,
            size_t inputpitchInInts,
            unsigned int* d_outputdata,
            size_t outputpitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            cudaStream_t stream,
            KernelLaunchHandle& handle);

void callConversionKernel2BitTo2BitHiLoTT(
            const unsigned int* d_inputdata,
            size_t inputpitchInInts,
            unsigned int* d_outputdata,
            size_t outputpitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            cudaStream_t stream,
            KernelLaunchHandle& handle);            




#endif //ifdef __NVCC__

}
}


#endif
