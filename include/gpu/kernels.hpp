#ifndef CARE_GPU_KERNELS_HPP
#define CARE_GPU_KERNELS_HPP

#include <hpc_helpers.cuh>
#include <gpu/kernellaunch.hpp>
#include <gpu/gpumsa.cuh>

#include <gpu/minhashingkernels.cuh>

#include <bestalignment.hpp>
#include <correctionresultprocessing.hpp>
#include <gpu/forest_gpu.cuh>

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

struct IsHqAnchor{
    DEVICEQUALIFIER
    bool operator() (const AnchorHighQualityFlag& flag) const{
        return flag.hq();
    }
};









void call_popcount_shifted_hamming_distance_kernel_async(
            void* d_tempstorage,
            size_t& tempstoragebytes,
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
            const int* d_candidates_per_subject,
            const int* d_anchorIndicesOfCandidates,
            const int* d_numAnchors,
            const int* d_numCandidates,
            const bool* d_anchorContainsN,
            bool removeAmbiguousAnchors,
            const bool* d_candidateContainsN,
            bool removeAmbiguousCandidates,
            int maxNumAnchors,
            int maxNumCandidates,
            int maximumSequenceLength,
            int encodedSequencePitchInInts2Bit,
            int min_overlap,
            float maxErrorRate,
            float min_overlap_ratio,
            float estimatedNucleotideErrorRate,
            cudaStream_t stream,
            KernelLaunchHandle& handle);


void call_popcount_rightshifted_hamming_distance_kernel_async(
            void* d_tempstorage,
            size_t& tempstoragebytes,
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
            const int* d_candidates_per_subject,
            const int* d_anchorIndicesOfCandidates,
            const int* d_numAnchors,
            const int* d_numCandidates,
            const bool* d_anchorContainsN,
            bool removeAmbiguousAnchors,
            const bool* d_candidateContainsN,
            bool removeAmbiguousCandidates,
            int maxNumAnchors,
            int maxNumCandidates,
            int maximumSequenceLength,
            int encodedSequencePitchInInts2Bit,
            int min_overlap,
            float maxErrorRate,
            float min_overlap_ratio,
            float estimatedNucleotideErrorRate,
            cudaStream_t stream,
            KernelLaunchHandle& handle);


void callSelectIndicesOfGoodCandidatesKernelAsync(
            int* d_indicesOfGoodCandidates,
            int* d_numIndicesPerAnchor,
            int* d_totalNumIndices,
            const BestAlignment_t* d_alignmentFlags,
            const int* d_candidates_per_subject,
            const int* d_candidates_per_subject_prefixsum,
            const int* d_anchorIndicesOfCandidates,
            const int* d_numAnchors,
            const int* d_numCandidates,
            int maxNumAnchors,
            int maxNumCandidates,
            cudaStream_t stream,
            KernelLaunchHandle& handle);


void call_cuda_filter_alignments_by_mismatchratio_kernel_async(
            BestAlignment_t* d_bestAlignmentFlags,
            const int* d_nOps,
            const int* d_overlaps,
            const int* d_candidates_per_subject_prefixsum,
            const int* d_numAnchors,
            const int* d_numCandidates,
            int maxNumAnchors,
            int maxNumCandidates,
            float mismatchratioBaseFactor,
            float goodAlignmentsCountThreshold,
            cudaStream_t stream,
            KernelLaunchHandle& handle);

// msa construction kernels

void callConstructMultipleSequenceAlignmentsKernel_async(
        GPUMultiMSA multiMSA,
        const int* d_overlaps,
        const int* d_shifts,
        const int* d_nOps,
        const BestAlignment_t* d_bestAlignmentFlags,
        const int* d_anchorLengths,
        const int* d_candidateLengths,
        const int* d_indices,
        const int* d_indices_per_subject,
        const int* d_candidatesPerSubjectPrefixSum,            
        const unsigned int* d_subjectSequencesData,
        const unsigned int* d_candidateSequencesData,
        const char* d_subjectQualities,
        const char* d_candidateQualities,
        const int* d_numAnchors,
        float desiredAlignmentMaxErrorRate,
        int maxNumAnchors,
        int maxNumCandidates,
        bool canUseQualityScores,
        int encodedSequencePitchInInts,
        size_t qualityPitchInBytes,
        cudaStream_t stream,
        KernelLaunchHandle& handle);

void callMsaCandidateRefinementKernel_singleiter_async(
        int* d_newIndices,
        int* d_newNumIndicesPerSubject,
        int* d_newNumIndices,
        GPUMultiMSA multiMSA,
        const BestAlignment_t* d_bestAlignmentFlags,
        const int* d_shifts,
        const int* d_nOps,
        const int* d_overlaps,
        const unsigned int* d_subjectSequencesData,
        const unsigned int* d_candidateSequencesData,
        const int* d_subjectSequencesLength,
        const int* d_candidateSequencesLength,
        const char* d_subjectQualities,
        const char* d_candidateQualities,
        bool* d_shouldBeKept,
        const int* d_candidates_per_subject_prefixsum,
        const int* d_numAnchors,
        float desiredAlignmentMaxErrorRate,
        int maxNumAnchors,
        int maxNumCandidates,
        bool canUseQualityScores,
        size_t encodedSequencePitchInInts,
        size_t qualityPitchInBytes,
        const int* d_indices,
        const int* d_indices_per_subject,
        int dataset_coverage,
        int iteration,
        bool* d_anchorIsFinished,
        cudaStream_t stream,
        KernelLaunchHandle& handle
    );


void callMsaCandidateRefinementKernel_multiiter_async(
    int* d_newIndices,
    int* d_newNumIndicesPerSubject,
    int* d_newNumIndices,
    GPUMultiMSA multiMSA,
    const BestAlignment_t* d_bestAlignmentFlags,
    const int* d_shifts,
    const int* d_nOps,
    const int* d_overlaps,
    const unsigned int* d_subjectSequencesData,
    const unsigned int* d_candidateSequencesData,
    const int* d_subjectSequencesLength,
    const int* d_candidateSequencesLength,
    const char* d_subjectQualities,
    const char* d_candidateQualities,
    bool* d_shouldBeKept,
    const int* d_candidates_per_subject_prefixsum,
    const int* d_numAnchors,
    float desiredAlignmentMaxErrorRate,
    int maxNumAnchors,
    int maxNumCandidates,
    bool canUseQualityScores,
    size_t encodedSequencePitchInInts,
    size_t qualityPitchInBytes,
    int* d_indices,
    int* d_indices_per_subject,
    int dataset_coverage,
    int numIterations,
    cudaStream_t stream,
    KernelLaunchHandle& handle
);

void callMsaFindCandidatesOfDifferentRegionAndRemoveThemViaRebuildKernel_async(
            int* d_newIndices,
            int* d_newNumIndicesPerSubject,
            int* d_newNumIndices,
            MSAColumnProperties* d_msaColumnProperties,
            char* d_consensus,
            int* d_coverage,
            int* d_counts,
            float* d_weights,
            float* d_support,
            int* d_origCoverages,
            float* d_origWeights,
            const BestAlignment_t* d_bestAlignmentFlags,
            const int* d_shifts,
            const int* d_nOps,
            const int* d_overlaps,
            const unsigned int* d_subjectSequencesData,
            const unsigned int* d_candidateSequencesData,
            const unsigned int* d_transposedCandidateSequencesData,
            const int* d_subjectSequencesLength,
            const int* d_candidateSequencesLength,
            const char* d_subjectQualities,
            const char* d_candidateQualities,
            bool* d_shouldBeKept,
            const int* d_candidates_per_subject_prefixsum,
            const int* d_numAnchors,
            const int* d_numCandidates,
            float desiredAlignmentMaxErrorRate,
            int maxNumAnchors,
            int maxNumCandidates,
            bool canUseQualityScores,
            size_t encodedSequencePitchInInts,
            size_t qualityPitchInBytes,
            size_t msaColumnPitchInElements,
            const int* d_indices,
            const int* d_indices_per_subject,
            int dataset_coverage,
            const bool* d_canExecute,
            int iteration,
            const read_number* d_subjectReadIds,
            cudaStream_t stream,
            KernelLaunchHandle& handle);

void callMsaFindCandidatesOfDifferentRegionAndRemoveThemViaDeletionKernel_async(
        int* d_newIndices,
        int* d_newNumIndicesPerSubject,
        int* d_newNumIndices,
        MSAColumnProperties* d_msaColumnProperties,
        char* d_consensus,
        int* d_coverage,
        int* d_counts,
        float* d_weights,
        float* d_support,
        int* d_origCoverages,
        float* d_origWeights,
        const BestAlignment_t* d_bestAlignmentFlags,
        const int* d_shifts,
        const int* d_nOps,
        const int* d_overlaps,
        const unsigned int* d_subjectSequencesData,
        const unsigned int* d_candidateSequencesData,
        const unsigned int* d_transposedCandidateSequencesData,
        const int* d_subjectSequencesLength,
        const int* d_candidateSequencesLength,
        const char* d_subjectQualities,
        const char* d_candidateQualities,
        bool* d_shouldBeKept,
        const int* d_candidates_per_subject_prefixsum,
        const int* d_numAnchors,
        const int* d_numCandidates,
        float desiredAlignmentMaxErrorRate,
        int maxNumAnchors,
        int maxNumCandidates,
        bool canUseQualityScores,
        size_t encodedSequencePitchInInts,
        size_t qualityPitchInBytes,
        size_t msaColumnPitchInElements,
        const int* d_indices,
        const int* d_indices_per_subject,
        int dataset_coverage,
        const bool* d_canExecute,
        int iteration,
        const read_number* d_subjectReadIds,
        cudaStream_t stream,
        KernelLaunchHandle& handle);          

// correction kernels


void call_msaCorrectAnchorsKernel_async(
    char* d_correctedSubjects,
    bool* d_subjectIsCorrected,
    AnchorHighQualityFlag* d_isHighQualitySubject,
    GPUMultiMSA multiMSA,
    const unsigned int* d_subjectSequencesData,
    const unsigned int* d_candidateSequencesData,
    const int* d_candidateSequencesLength,
    const int* d_indices_per_subject,
    const int* d_numAnchors,
    int maxNumAnchors,
    int encodedSequencePitchInInts,
    size_t sequence_pitch,
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
    KernelLaunchHandle& handle
);

void callConstructAnchorResultsKernelAsync(
    TempCorrectedSequence::EncodedEdit* __restrict__ d_editsPerCorrectedSubject,
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
    size_t editsPitchInBytes,
    const int* d_numAnchors,
    int maxNumAnchors,
    cudaStream_t stream,
    KernelLaunchHandle& handle
);

void callFlagCandidatesToBeCorrectedKernel_async(
    bool* d_candidateCanBeCorrected,
    int* d_numCorrectedCandidatesPerAnchor,
    GPUMultiMSA multiMSA,
    const int* d_alignmentShifts,
    const int* d_candidateSequencesLengths,
    const int* d_anchorIndicesOfCandidates,
    const AnchorHighQualityFlag* d_hqflags,
    const int* d_candidatesPerSubjectPrefixsum,
    const int* d_localGoodCandidateIndices,
    const int* d_numLocalGoodCandidateIndicesPerSubject,
    const int* d_numAnchors,
    const int* d_numCandidates,
    float min_support_threshold,
    float min_coverage_threshold,
    int new_columns_to_correct,
    cudaStream_t stream,
    KernelLaunchHandle& handle
);

void callCorrectCandidatesKernel_async(
    char* __restrict__ correctedCandidates,
    TempCorrectedSequence::EncodedEdit* __restrict__ d_editsPerCorrectedCandidate,
    int* __restrict__ d_numEditsPerCorrectedCandidate,
    GPUMultiMSA multiMSA,
    const int* __restrict__ shifts,
    const BestAlignment_t* __restrict__ bestAlignmentFlags,
    const unsigned int* __restrict__ candidateSequencesData,
    const int* __restrict__ candidateSequencesLengths,
    const bool* __restrict__ d_candidateContainsN,
    const int* __restrict__ candidateIndicesOfCandidatesToBeCorrected,
    const int* __restrict__ numCandidatesToBeCorrected,
    const int* __restrict__ anchorIndicesOfCandidates,
    const int* d_numAnchors,
    const int* d_numCandidates,
    int doNotUseEditsValue,
    int numEditsThreshold,
    int encodedSequencePitchInInts,
    size_t decodedSequencePitchInBytes,
    size_t editsPitchInBytes,
    int maximum_sequence_length,
    cudaStream_t stream,
    KernelLaunchHandle& handle
);

void callMsaCorrectAnchorsWithForestKernel(
    char* d_correctedSubjects,
    bool* d_subjectIsCorrected,
    AnchorHighQualityFlag* d_isHighQualitySubject,
    GPUMultiMSA multiMSA,
    GpuForest::Clf gpuForest,
    float forestThreshold,
    const unsigned int* d_subjectSequencesData,
    const int* d_indices_per_subject,
    const int numAnchors,
    int encodedSequencePitchInInts,
    size_t decodedSequencePitchInBytes,
    int maximumSequenceLength,
    float estimatedErrorrate,
    float desiredAlignmentMaxErrorRate,
    float estimatedCoverage,
    float avg_support_threshold,
    float min_support_threshold,
    float min_coverage_threshold,
    float max_coverage_threshold,
    cudaStream_t stream
);


void callConversionKernel2BitTo2BitHiLoNN(
            const unsigned int* d_inputdata,
            size_t inputpitchInInts,
            unsigned int* d_outputdata,
            size_t outputpitchInInts,
            const int* d_sequenceLengths,
            const int* d_numSequences,
            int maxNumSequences,
            cudaStream_t stream,
            KernelLaunchHandle& handle);

void callConversionKernel2BitTo2BitHiLoNT(
            const unsigned int* d_inputdata,
            size_t inputpitchInInts,
            unsigned int* d_outputdata,
            size_t outputpitchInInts,
            const int* d_sequenceLengths,
            const int* d_numSequences,
            int maxNumSequences,
            cudaStream_t stream,
            KernelLaunchHandle& handle);

void callConversionKernel2BitTo2BitHiLoTT(
            const unsigned int* d_inputdata,
            size_t inputpitchInInts,
            unsigned int* d_outputdata,
            size_t outputpitchInInts,
            const int* d_sequenceLengths,
            const int* d_numSequences,
            int maxNumSequences,
            cudaStream_t stream,
            KernelLaunchHandle& handle);            







#endif //ifdef __NVCC__

}
}


#endif
