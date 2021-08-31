#ifndef CARE_GPU_KERNELS_HPP
#define CARE_GPU_KERNELS_HPP

#include <hpc_helpers.cuh>
#include <gpu/kernellaunch.hpp>
#include <gpu/gpumsa.cuh>

#include <gpu/minhashingkernels.cuh>

#include <bestalignment.hpp>
#include <correctedsequence.hpp>
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
            AlignmentOrientation* d_alignment_best_alignment_flags,
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
            AlignmentOrientation* d_alignment_best_alignment_flags,
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
            const AlignmentOrientation* d_alignmentFlags,
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
            AlignmentOrientation* d_bestAlignmentFlags,
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

void callComputeMaximumMsaWidthKernel(
    int* d_result,
    const int* d_shifts,
    const int* d_anchorLengths,
    const int* d_candidateLengths,
    const int* d_indices,
    const int* d_indices_per_subject,
    const int* d_candidatesPerSubjectPrefixSum,
    const int numAnchors,
    cudaStream_t stream
);

void callComputeMsaConsensusQualityKernel(
    char* d_consensusQuality,
    int consensusQualityPitchInBytes,
    GPUMultiMSA multiMSA,
    cudaStream_t stream
);

void callComputeDecodedMsaConsensusKernel(
    char* d_consensus,
    int consensusPitchInBytes,
    GPUMultiMSA multiMSA,
    cudaStream_t stream
);

void callComputeMsaSizesKernel(
    int* d_sizes,
    GPUMultiMSA multiMSA,
    cudaStream_t stream
);

void callConstructMultipleSequenceAlignmentsKernel_async(
        GPUMultiMSA multiMSA,
        const int* d_overlaps,
        const int* d_shifts,
        const int* d_nOps,
        const AlignmentOrientation* d_bestAlignmentFlags,
        const int* d_anchorLengths,
        const int* d_candidateLengths,
        const int* d_indices,
        const int* d_indices_per_subject,
        const int* d_candidatesPerSubjectPrefixSum,            
        const unsigned int* d_subjectSequencesData,
        const unsigned int* d_candidateSequencesData,
        const bool* d_isPairedCandidate,
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
        const AlignmentOrientation* d_bestAlignmentFlags,
        const int* d_shifts,
        const int* d_nOps,
        const int* d_overlaps,
        const unsigned int* d_subjectSequencesData,
        const unsigned int* d_candidateSequencesData,
        const bool* d_isPairedCandidate,
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
    const AlignmentOrientation* d_bestAlignmentFlags,
    const int* d_shifts,
    const int* d_nOps,
    const int* d_overlaps,
    const unsigned int* d_subjectSequencesData,
    const unsigned int* d_candidateSequencesData,
    const bool* d_isPairedCandidate,
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



// correction kernels


void call_msaCorrectAnchorsKernel_async(
    char* d_correctedSubjects,
    bool* d_subjectIsCorrected,
    AnchorHighQualityFlag* d_isHighQualitySubject,
    GPUMultiMSA multiMSA,
    const unsigned int* d_subjectSequencesData,
    const int* d_indices_per_subject,
    const int* d_numAnchors,
    int maxNumAnchors,
    int encodedSequencePitchInInts,
    size_t sequence_pitch,
    float estimatedErrorrate,
    float avg_support_threshold,
    float min_support_threshold,
    float min_coverage_threshold,
    int maximum_sequence_length,
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

void callFlagCandidatesToBeCorrectedWithExcludeFlagsKernel(
    bool* d_candidateCanBeCorrected,
    int* d_numCorrectedCandidatesPerAnchor,
    GPUMultiMSA multiMSA,
    const bool* d_excludeFlags, //candidates with flag == true will not be considered
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
    EncodedCorrectionEdit* __restrict__ d_editsPerCorrectedCandidate,
    int* __restrict__ d_numEditsPerCorrectedCandidate,
    GPUMultiMSA multiMSA,
    const int* __restrict__ shifts,
    const AlignmentOrientation* __restrict__ bestAlignmentFlags,
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
    float estimatedCoverage,
    float avg_support_threshold,
    float min_support_threshold,
    float min_coverage_threshold,
    cudaStream_t stream,
    KernelLaunchHandle& handle
);

void callMsaCorrectCandidatesWithForestKernel(
    char* d_correctedCandidates,
    EncodedCorrectionEdit* d_editsPerCorrectedCandidate,
    int* d_numEditsPerCorrectedCandidate,
    GPUMultiMSA multiMSA,
    GpuForest::Clf gpuForest,
    float forestThreshold,
    float estimatedCoverage,
    const int* d_shifts,
    const AlignmentOrientation* d_bestAlignmentFlags,
    const unsigned int* d_candidateSequencesData,
    const int* d_candidateSequencesLengths,
    const bool* d_candidateContainsN,
    const int* d_candidateIndicesOfCandidatesToBeCorrected,
    const int* d_numCandidatesToBeCorrected,
    const int* d_anchorIndicesOfCandidates,
    const int numCandidates,
    int doNotUseEditsValue,
    int numEditsThreshold,            
    int encodedSequencePitchInInts,
    size_t decodedSequencePitchInBytes,
    size_t editsPitchInBytes,
    int maximum_sequence_length,
    cudaStream_t stream,
    KernelLaunchHandle& handle,
    const read_number* candidateReadIds
);

void callConstructSequenceCorrectionResultsKernel(
    EncodedCorrectionEdit* d_edits,
    int* d_numEditsPerCorrection,
    int doNotUseEditsValue,
    const int* d_indicesOfUncorrectedSequences,
    const int* d_numIndices,
    const bool* d_readContainsN,
    const unsigned int* d_uncorrectedEncodedSequences,
    const int* d_sequenceLengths,
    const char* d_correctedSequences,
    const int numCorrectedSequencesUpperBound, // >= *d_numIndices. d_edits must be large enought to store the edits of this many sequences
    bool isCompactCorrection,
    int numEditsThreshold,
    size_t encodedSequencePitchInInts,
    size_t decodedSequencePitchInBytes,
    size_t editsPitchInBytes,        
    cudaStream_t stream,
    KernelLaunchHandle& handle
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
