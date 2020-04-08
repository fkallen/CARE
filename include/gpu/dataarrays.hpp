#ifndef CARE_GPU_DATA_ARRAYS_HPP
#define CARE_GPU_DATA_ARRAYS_HPP

#include "../hpc_helpers.cuh"
#include "bestalignment.hpp"
#include "msa.hpp"
#include "utility_kernels.cuh"

#include <correctionresultprocessing.hpp>

//#include <gpu/thrust_custom_allocators.hpp>
#include <gpu/simpleallocation.cuh>
#include <gpu/kernels.hpp>
#include <config.hpp>

// #ifdef __NVCC__

// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>

// #include <thrust/fill.h>
// #include <thrust/device_ptr.h>
// #include <thrust/async/for_each.h>

// #include <thrust/system/cuda/experimental/pinned_allocator.h>

// #endif

namespace care {
namespace gpu {

#ifdef __NVCC__




struct DataArrays {
	static constexpr int padding_bytes = 4;
	//static constexpr float allocfactor = 1.1;

	// DataArrays() : DataArrays(0){
	// }

	// DataArrays(int deviceId) : deviceId(deviceId){
	// 	//cudaSetDevice(deviceId);
	// };

    static constexpr int overprovisioningPercent = 0;

    template<class T>
    using DeviceBuffer = SimpleAllocationDevice<T, overprovisioningPercent>;
    
    template<class T>
    using PinnedBuffer = SimpleAllocationPinnedHost<T, overprovisioningPercent>;

	void reset(){

        h_subject_sequences_data.destroy();
        h_candidate_sequences_data.destroy();
        h_transposedCandidateSequencesData.destroy();
        h_subject_sequences_lengths.destroy();
        h_candidate_sequences_lengths.destroy();
        h_candidates_per_subject.destroy();
        h_candidates_per_subject_prefixsum.destroy();
        h_subject_read_ids.destroy();
        h_candidate_read_ids.destroy();

        d_subject_sequences_data.destroy();
        d_candidate_sequences_data.destroy();
        d_transposedCandidateSequencesData.destroy();
        d_subject_sequences_lengths.destroy();
        d_candidate_sequences_lengths.destroy();
        d_candidates_per_subject.destroy();
        d_candidates_per_subject_prefixsum.destroy();
        d_subject_read_ids.destroy();
        d_candidate_read_ids.destroy();

        h_subject_qualities.destroy();
        h_candidate_qualities.destroy();

        d_subject_qualities.destroy();
        d_candidate_qualities.destroy();
        d_candidate_qualities_transposed.destroy();

        h_consensus.destroy();
        h_support.destroy();
        h_coverage.destroy();
        h_origWeights.destroy();
        h_origCoverages.destroy();
        h_msa_column_properties.destroy();
        h_counts.destroy();
        h_weights.destroy();

        d_consensus.destroy();
        d_support.destroy();
        d_coverage.destroy();
        d_origWeights.destroy();
        d_origCoverages.destroy();
        d_msa_column_properties.destroy();
        d_counts.destroy();
        d_weights.destroy();

        h_alignment_scores.destroy();
        h_alignment_overlaps.destroy();
        h_alignment_shifts.destroy();
        h_alignment_nOps.destroy();
        h_alignment_isValid.destroy();
        h_alignment_best_alignment_flags.destroy();

        d_alignment_scores.destroy();
        d_alignment_overlaps.destroy();
        d_alignment_shifts.destroy();
        d_alignment_nOps.destroy();
        d_alignment_isValid.destroy();
        d_alignment_best_alignment_flags.destroy();

        h_corrected_subjects.destroy();
        h_corrected_candidates.destroy();
        h_num_corrected_candidates_per_anchor.destroy();
        h_num_corrected_candidates_per_anchor_prefixsum.destroy();
        h_subject_is_corrected.destroy();
        h_indices_of_corrected_candidates.destroy();
        h_is_high_quality_subject.destroy();
        h_high_quality_subject_indices.destroy();
        h_num_high_quality_subject_indices.destroy();
        h_num_uncorrected_positions_per_subject.destroy();
        h_uncorrected_positions_per_subject.destroy();

        d_corrected_subjects.destroy();
        d_corrected_candidates.destroy();
        d_num_corrected_candidates_per_anchor.destroy();
        d_num_corrected_candidates_per_anchor_prefixsum.destroy();
        d_subject_is_corrected.destroy();
        d_indices_of_corrected_candidates.destroy();
        d_is_high_quality_subject.destroy();
        d_high_quality_subject_indices.destroy();
        d_num_high_quality_subject_indices.destroy();
        d_num_uncorrected_positions_per_subject.destroy();
        d_uncorrected_positions_per_subject.destroy();

        h_indices.destroy();
        h_indices_per_subject.destroy();
        h_num_indices.destroy();

        d_indices.destroy();
        d_indices_per_subject.destroy();
        d_num_indices.destroy();

        d_indices_tmp.destroy();
        d_indices_per_subject_tmp.destroy();
        d_num_indices_tmp.destroy();

        d_indices_of_corrected_subjects.destroy();
        d_num_indices_of_corrected_subjects.destroy();


        h_editsPerCorrectedSubject.destroy();
        h_numEditsPerCorrectedSubject.destroy();
        h_editsPerCorrectedCandidate.destroy();
        h_numEditsPerCorrectedCandidate.destroy();
        h_anchorContainsN.destroy();
        h_candidateContainsN.destroy();

        d_editsPerCorrectedSubject.destroy();
        d_numEditsPerCorrectedSubject.destroy();
        d_editsPerCorrectedCandidate.destroy();
        d_numEditsPerCorrectedCandidate.destroy();
        d_anchorContainsN.destroy();
        d_candidateContainsN.destroy();

        d_canExecute.destroy();
        
        d_tempstorage.destroy();
        d_numAnchors.destroy();
        d_numCandidates.destroy();
        h_numAnchors.destroy();
        h_numCandidates.destroy();
	}

    size_t hostArraysSizeInBytes() const{
        auto f = [](const auto& v){
            return v.size();
        };
        size_t bytes = 0;
        bytes += f(h_subject_sequences_data);
        bytes += f(h_candidate_sequences_data);
        bytes += f(h_transposedCandidateSequencesData);
        bytes += f(h_subject_sequences_lengths);
        bytes += f(h_candidate_sequences_lengths);
        bytes += f(h_candidates_per_subject);
        bytes += f(h_candidates_per_subject_prefixsum);
        bytes += f(h_subject_read_ids);
        bytes += f(h_candidate_read_ids);
        bytes += f(h_anchorIndicesOfCandidates);

        bytes += f(h_subject_qualities);
        bytes += f(h_candidate_qualities);

        bytes += f(h_consensus);
        bytes += f(h_support);
        bytes += f(h_coverage);
        bytes += f(h_origWeights);
        bytes += f(h_origCoverages);
        bytes += f(h_msa_column_properties);
        bytes += f(h_counts);
        bytes += f(h_weights);

        bytes += f(h_alignment_scores);
        bytes += f(h_alignment_overlaps);
        bytes += f(h_alignment_shifts);
        bytes += f(h_alignment_nOps);
        bytes += f(h_alignment_isValid);
        bytes += f(h_alignment_best_alignment_flags);

        bytes += f(h_corrected_subjects);
        bytes += f(h_corrected_candidates);
        bytes += f(h_num_corrected_candidates_per_anchor);
        bytes += f(h_num_corrected_candidates_per_anchor_prefixsum);
        bytes += f(h_subject_is_corrected);
        bytes += f(h_indices_of_corrected_candidates);
        bytes += f(h_is_high_quality_subject);
        bytes += f(h_high_quality_subject_indices);
        bytes += f(h_num_high_quality_subject_indices);
        bytes += f(h_num_uncorrected_positions_per_subject);
        bytes += f(h_uncorrected_positions_per_subject);

        bytes += f(h_indices);
        bytes += f(h_indices_per_subject);
        bytes += f(h_num_indices);

        bytes += f(h_editsPerCorrectedSubject);
        bytes += f(h_numEditsPerCorrectedSubject);
        bytes += f(h_editsPerCorrectedCandidate);
        bytes += f(h_numEditsPerCorrectedCandidate);
        bytes += f(h_anchorContainsN);
        bytes += f(h_candidateContainsN);


        return bytes;
	}

    size_t deviceArraysSizeInBytes() const{
        auto f = [](const auto& v){
            return v.size();
        };
        size_t bytes = 0;

        bytes += f(d_subject_sequences_data);
        bytes += f(d_candidate_sequences_data);
        bytes += f(d_transposedCandidateSequencesData);
        bytes += f(d_subject_sequences_lengths);
        bytes += f(d_candidate_sequences_lengths);
        bytes += f(d_candidates_per_subject);
        bytes += f(d_candidates_per_subject_prefixsum);
        bytes += f(d_subject_read_ids);
        bytes += f(d_candidate_read_ids);
        bytes += f(d_anchorIndicesOfCandidates);

        bytes += f(d_subject_qualities);
        bytes += f(d_candidate_qualities);
        bytes += f(d_candidate_qualities_transposed);
        //bytes += f(d_candidate_qualities_tmp);

        bytes += f(d_consensus);
        bytes += f(d_support);
        bytes += f(d_coverage);
        bytes += f(d_origWeights);
        bytes += f(d_origCoverages);
        bytes += f(d_msa_column_properties);
        bytes += f(d_counts);
        bytes += f(d_weights);

        bytes += f(d_alignment_scores);
        bytes += f(d_alignment_overlaps);
        bytes += f(d_alignment_shifts);
        bytes += f(d_alignment_nOps);
        bytes += f(d_alignment_isValid);
        bytes += f(d_alignment_best_alignment_flags);

        bytes += f(d_corrected_subjects);
        bytes += f(d_corrected_candidates);
        bytes += f(d_num_corrected_candidates_per_anchor);
        bytes += f(d_num_corrected_candidates_per_anchor_prefixsum);
        bytes += f(d_subject_is_corrected);
        bytes += f(d_indices_of_corrected_candidates);
        bytes += f(d_is_high_quality_subject);
        bytes += f(d_high_quality_subject_indices);
        bytes += f(d_num_high_quality_subject_indices);
        bytes += f(d_num_uncorrected_positions_per_subject);
        bytes += f(d_uncorrected_positions_per_subject);

        bytes += f(d_indices);
        bytes += f(d_indices_per_subject);
        bytes += f(d_num_indices);
        bytes += f(d_indices_tmp);
        bytes += f(d_indices_per_subject_tmp);
        bytes += f(d_num_indices_tmp);

        bytes += f(d_indices_of_corrected_subjects);
        bytes += f(d_num_indices_of_corrected_subjects);

        bytes += f(d_editsPerCorrectedSubject);
        bytes += f(d_numEditsPerCorrectedSubject);
        bytes += f(d_editsPerCorrectedCandidate);
        bytes += f(d_numEditsPerCorrectedCandidate);
        bytes += f(d_anchorContainsN);
        bytes += f(d_candidateContainsN);

        bytes += f(d_canExecute);
        
        bytes += f(d_tempstorage);


        return bytes;
	}

    size_t hostArraysCapacityInBytes() const{
        auto f = [](const auto& v){
            return v.capacity();
        };
        size_t bytes = 0;
        bytes += f(h_subject_sequences_data);
        bytes += f(h_candidate_sequences_data);
        bytes += f(h_transposedCandidateSequencesData);
        bytes += f(h_subject_sequences_lengths);
        bytes += f(h_candidate_sequences_lengths);
        bytes += f(h_candidates_per_subject);
        bytes += f(h_candidates_per_subject_prefixsum);
        bytes += f(h_subject_read_ids);
        bytes += f(h_candidate_read_ids);
        bytes += f(h_anchorIndicesOfCandidates);

        bytes += f(h_subject_qualities);
        bytes += f(h_candidate_qualities);

        bytes += f(h_consensus);
        bytes += f(h_support);
        bytes += f(h_coverage);
        bytes += f(h_origWeights);
        bytes += f(h_origCoverages);
        bytes += f(h_msa_column_properties);
        bytes += f(h_counts);
        bytes += f(h_weights);

        bytes += f(h_alignment_scores);
        bytes += f(h_alignment_overlaps);
        bytes += f(h_alignment_shifts);
        bytes += f(h_alignment_nOps);
        bytes += f(h_alignment_isValid);
        bytes += f(h_alignment_best_alignment_flags);

        bytes += f(h_corrected_subjects);
        bytes += f(h_corrected_candidates);
        bytes += f(h_num_corrected_candidates_per_anchor);
        bytes += f(h_num_corrected_candidates_per_anchor_prefixsum);
        bytes += f(h_subject_is_corrected);
        bytes += f(h_indices_of_corrected_candidates);
        bytes += f(h_is_high_quality_subject);
        bytes += f(h_high_quality_subject_indices);
        bytes += f(h_num_high_quality_subject_indices);
        bytes += f(h_num_uncorrected_positions_per_subject);
        bytes += f(h_uncorrected_positions_per_subject);

        bytes += f(h_indices);
        bytes += f(h_indices_per_subject);
        bytes += f(h_num_indices);

        bytes += f(h_editsPerCorrectedSubject);
        bytes += f(h_numEditsPerCorrectedSubject);
        bytes += f(h_editsPerCorrectedCandidate);
        bytes += f(h_numEditsPerCorrectedCandidate);
        bytes += f(h_anchorContainsN);
        bytes += f(h_candidateContainsN);



        return bytes;
	}

    size_t deviceArraysCapacityInBytes() const{
        auto f = [](const auto& v){
            return v.capacity();
        };
        size_t bytes = 0;

        bytes += f(d_subject_sequences_data);
        bytes += f(d_candidate_sequences_data);
        bytes += f(d_transposedCandidateSequencesData);
        bytes += f(d_subject_sequences_lengths);
        bytes += f(d_candidate_sequences_lengths);
        bytes += f(d_candidates_per_subject);
        bytes += f(d_candidates_per_subject_prefixsum);
        bytes += f(d_subject_read_ids);
        bytes += f(d_candidate_read_ids);
        bytes += f(d_anchorIndicesOfCandidates);

        bytes += f(d_subject_qualities);
        bytes += f(d_candidate_qualities);
        bytes += f(d_candidate_qualities_transposed);
        //bytes += f(d_candidate_qualities_tmp);

        bytes += f(d_consensus);
        bytes += f(d_support);
        bytes += f(d_coverage);
        bytes += f(d_origWeights);
        bytes += f(d_origCoverages);
        bytes += f(d_msa_column_properties);
        bytes += f(d_counts);
        bytes += f(d_weights);

        bytes += f(d_alignment_scores);
        bytes += f(d_alignment_overlaps);
        bytes += f(d_alignment_shifts);
        bytes += f(d_alignment_nOps);
        bytes += f(d_alignment_isValid);
        bytes += f(d_alignment_best_alignment_flags);

        bytes += f(d_corrected_subjects);
        bytes += f(d_corrected_candidates);
        bytes += f(d_num_corrected_candidates_per_anchor);
        bytes += f(d_num_corrected_candidates_per_anchor_prefixsum);
        bytes += f(d_subject_is_corrected);
        bytes += f(d_indices_of_corrected_candidates);
        bytes += f(d_is_high_quality_subject);
        bytes += f(d_high_quality_subject_indices);
        bytes += f(d_num_high_quality_subject_indices);
        bytes += f(d_num_uncorrected_positions_per_subject);
        bytes += f(d_uncorrected_positions_per_subject);

        bytes += f(d_indices);
        bytes += f(d_indices_per_subject);       
        bytes += f(d_num_indices);
        bytes += f(d_indices_tmp);
        bytes += f(d_indices_per_subject_tmp);
        bytes += f(d_num_indices_tmp);

        bytes += f(d_indices_of_corrected_subjects);
        bytes += f(d_num_indices_of_corrected_subjects);
        bytes += f(d_editsPerCorrectedSubject);
        bytes += f(d_numEditsPerCorrectedSubject);
        bytes += f(d_editsPerCorrectedCandidate);
        bytes += f(d_numEditsPerCorrectedCandidate);
        bytes += f(d_anchorContainsN);
        bytes += f(d_candidateContainsN);

        bytes += f(d_canExecute);
        
        bytes += f(d_tempstorage);

        return bytes;
	}

	// alignment input
    
    DeviceBuffer<char> d_tempstorage;
    PinnedBuffer<int> h_numAnchors;
    PinnedBuffer<int> h_numCandidates;
    DeviceBuffer<int> d_numAnchors;
    DeviceBuffer<int> d_numCandidates;

    PinnedBuffer<unsigned int> h_subject_sequences_data;
    PinnedBuffer<unsigned int> h_candidate_sequences_data;
    PinnedBuffer<unsigned int> h_transposedCandidateSequencesData;
    PinnedBuffer<int> h_subject_sequences_lengths;
    PinnedBuffer<int> h_candidate_sequences_lengths;
    PinnedBuffer<int> h_candidates_per_subject;
    PinnedBuffer<int> h_candidates_per_subject_prefixsum;
    PinnedBuffer<read_number> h_subject_read_ids;
    PinnedBuffer<read_number> h_candidate_read_ids;
    PinnedBuffer<int> h_anchorIndicesOfCandidates; // candidate i belongs to anchor anchorIndicesOfCandidates[i]

    DeviceBuffer<unsigned int> d_subject_sequences_data;
    DeviceBuffer<unsigned int> d_candidate_sequences_data;
    DeviceBuffer<unsigned int> d_transposedCandidateSequencesData;
    DeviceBuffer<int> d_subject_sequences_lengths;
    DeviceBuffer<int> d_candidate_sequences_lengths;
    DeviceBuffer<int> d_candidates_per_subject;
    DeviceBuffer<int> d_candidates_per_subject_prefixsum;
    DeviceBuffer<read_number> d_subject_read_ids;
    DeviceBuffer<read_number> d_candidate_read_ids;
    DeviceBuffer<int> d_anchorIndicesOfCandidates; // candidate i belongs to anchor anchorIndicesOfCandidates[i]

	//indices

    PinnedBuffer<int> h_indices;
    PinnedBuffer<int> h_indices_per_subject;
    PinnedBuffer<int> h_num_indices;

    DeviceBuffer<int> d_indices;
    DeviceBuffer<int> d_indices_per_subject;
    DeviceBuffer<int> d_num_indices;
    DeviceBuffer<int> d_indices_tmp;
    DeviceBuffer<int> d_indices_per_subject_tmp;
    DeviceBuffer<int> d_num_indices_tmp;

    PinnedBuffer<int> h_indices_of_corrected_subjects;
    PinnedBuffer<int> h_num_indices_of_corrected_subjects;

    DeviceBuffer<int> d_indices_of_corrected_subjects;
    DeviceBuffer<int> d_num_indices_of_corrected_subjects;


    PinnedBuffer<TempCorrectedSequence::Edit> h_editsPerCorrectedSubject;
    PinnedBuffer<int> h_numEditsPerCorrectedSubject;
    PinnedBuffer<TempCorrectedSequence::Edit> h_editsPerCorrectedCandidate;
    PinnedBuffer<int> h_numEditsPerCorrectedCandidate;
    PinnedBuffer<bool> h_anchorContainsN;
    PinnedBuffer<bool> h_candidateContainsN;

    DeviceBuffer<TempCorrectedSequence::Edit> d_editsPerCorrectedSubject;
    DeviceBuffer<int> d_numEditsPerCorrectedSubject;
    DeviceBuffer<TempCorrectedSequence::Edit> d_editsPerCorrectedCandidate;
    DeviceBuffer<int> d_numEditsPerCorrectedCandidate;
    DeviceBuffer<bool> d_anchorContainsN;
    DeviceBuffer<bool> d_candidateContainsN;


    PinnedBuffer<char> h_subject_qualities;
    PinnedBuffer<char> h_candidate_qualities;

    DeviceBuffer<char> d_subject_qualities;
    DeviceBuffer<char> d_candidate_qualities;
    DeviceBuffer<char> d_candidate_qualities_transposed;

	//correction results output

    PinnedBuffer<char> h_corrected_subjects;
    PinnedBuffer<char> h_corrected_candidates;
    PinnedBuffer<int> h_num_corrected_candidates_per_anchor;
    PinnedBuffer<int> h_num_corrected_candidates_per_anchor_prefixsum;
    PinnedBuffer<int> h_num_total_corrected_candidates;
    PinnedBuffer<bool> h_subject_is_corrected;
    PinnedBuffer<int> h_indices_of_corrected_candidates;
    PinnedBuffer<int> h_num_uncorrected_positions_per_subject;
    PinnedBuffer<int> h_uncorrected_positions_per_subject;

    DeviceBuffer<char> d_corrected_subjects;
    DeviceBuffer<char> d_corrected_candidates;
    DeviceBuffer<int> d_num_corrected_candidates_per_anchor;
    DeviceBuffer<int> d_num_corrected_candidates_per_anchor_prefixsum;
    DeviceBuffer<int> d_num_total_corrected_candidates;
    DeviceBuffer<bool> d_subject_is_corrected;
    DeviceBuffer<int> d_indices_of_corrected_candidates;
    DeviceBuffer<int> d_num_uncorrected_positions_per_subject;
    DeviceBuffer<int> d_uncorrected_positions_per_subject;

    PinnedBuffer<AnchorHighQualityFlag> h_is_high_quality_subject;
    PinnedBuffer<int> h_high_quality_subject_indices;
    PinnedBuffer<int> h_num_high_quality_subject_indices;

    DeviceBuffer<AnchorHighQualityFlag> d_is_high_quality_subject;
    DeviceBuffer<int> d_high_quality_subject_indices;
    DeviceBuffer<int> d_num_high_quality_subject_indices;

    char* d_compactCorrectedCandidates = nullptr;
    TempCorrectedSequence::Edit* d_compactEditsPerCorrectedCandidate = nullptr;


	//alignment results

    PinnedBuffer<int> h_alignment_scores;
    PinnedBuffer<int> h_alignment_overlaps;
    PinnedBuffer<int> h_alignment_shifts;
    PinnedBuffer<int> h_alignment_nOps;
    PinnedBuffer<bool> h_alignment_isValid;
    PinnedBuffer<BestAlignment_t> h_alignment_best_alignment_flags;

    DeviceBuffer<int> d_alignment_scores;
    DeviceBuffer<int> d_alignment_overlaps;
    DeviceBuffer<int> d_alignment_shifts;
    DeviceBuffer<int> d_alignment_nOps;
    DeviceBuffer<bool> d_alignment_isValid;
    DeviceBuffer<BestAlignment_t> d_alignment_best_alignment_flags;

    DeviceBuffer<bool> d_canExecute;


	// multiple sequence alignment

    PinnedBuffer<char> h_consensus;
    PinnedBuffer<float> h_support;
    PinnedBuffer<int> h_coverage;
    PinnedBuffer<float> h_origWeights;
    PinnedBuffer<int> h_origCoverages;
    PinnedBuffer<MSAColumnProperties> h_msa_column_properties;
    PinnedBuffer<int> h_counts;
    PinnedBuffer<float> h_weights;

    DeviceBuffer<char> d_consensus;
    DeviceBuffer<float> d_support;
    DeviceBuffer<int> d_coverage;
    DeviceBuffer<float> d_origWeights;
    DeviceBuffer<int> d_origCoverages;
    DeviceBuffer<MSAColumnProperties> d_msa_column_properties;
    DeviceBuffer<int> d_counts;
    DeviceBuffer<float> d_weights;

    void printMemoryUsage() const{
        // auto handleArray = [](auto& host, const auto& device){
        //     assert(host.sizeInBytes() == device.sizeInBytes());
        //     cudaMemcpy(host,
        //                 device,
        //                 device.sizeInBytes(),
        //                 D2H); CUERR;
        // };

        #define handlearray(x){auto& host = h_##x; const auto& device = d_##x; \
            assert(host.sizeInBytes() == device.sizeInBytes());\
            assert(host.capacityInBytes() == device.capacityInBytes());\
            std::cerr << #x << " capacity in bytes: " << host.capacityInBytes() << '\n';\
        }

        handlearray(subject_sequences_data);
        handlearray(candidate_sequences_data);
        handlearray(transposedCandidateSequencesData);
        handlearray(subject_sequences_lengths);
        handlearray(candidate_sequences_lengths);
        handlearray(candidates_per_subject);
        handlearray(candidates_per_subject_prefixsum);
        handlearray(subject_read_ids);
        handlearray(candidate_read_ids);
        handlearray(indices);
        handlearray(indices_per_subject);
        handlearray(num_indices);

        handlearray(anchorIndicesOfCandidates);

        handlearray(subject_qualities);
        handlearray(candidate_qualities);

        handlearray(corrected_subjects);
        handlearray(corrected_candidates);
        handlearray(num_corrected_candidates_per_anchor);
        handlearray(subject_is_corrected);
        handlearray(indices_of_corrected_candidates);
        handlearray(num_uncorrected_positions_per_subject);
        handlearray(uncorrected_positions_per_subject);

        handlearray(is_high_quality_subject);
        handlearray(high_quality_subject_indices);
        handlearray(num_high_quality_subject_indices);

        handlearray(alignment_scores);
        handlearray(alignment_overlaps);
        handlearray(alignment_shifts);
        handlearray(alignment_nOps);
        handlearray(alignment_isValid);
        handlearray(alignment_best_alignment_flags);

        handlearray(consensus);
        handlearray(support);
        handlearray(coverage);
        handlearray(origWeights);
        handlearray(origCoverages);
        handlearray(msa_column_properties);
        handlearray(counts);
        handlearray(weights);

        #undef handlearray
    }

    std::size_t getMemoryUsageInBytes() const{
        // auto handleArray = [](auto& host, const auto& device){
        //     assert(host.sizeInBytes() == device.sizeInBytes());
        //     cudaMemcpy(host,
        //                 device,
        //                 device.sizeInBytes(),
        //                 D2H); CUERR;
        // };

        std::size_t bytes = 0;

        #define handlearray(x){auto& host = h_##x; const auto& device = d_##x; \
            assert(host.sizeInBytes() == device.sizeInBytes());\
            assert(host.capacityInBytes() == device.capacityInBytes());\
            bytes += host.capacityInBytes();\
        }

        handlearray(subject_sequences_data);
        handlearray(candidate_sequences_data);
        handlearray(transposedCandidateSequencesData);
        handlearray(subject_sequences_lengths);
        handlearray(candidate_sequences_lengths);
        handlearray(candidates_per_subject);
        handlearray(candidates_per_subject_prefixsum);
        handlearray(subject_read_ids);
        handlearray(candidate_read_ids);
        handlearray(indices);
        handlearray(indices_per_subject);
        handlearray(num_indices);

        handlearray(anchorIndicesOfCandidates);

        handlearray(subject_qualities);
        handlearray(candidate_qualities);

        handlearray(corrected_subjects);
        handlearray(corrected_candidates);
        handlearray(num_corrected_candidates_per_anchor);
        handlearray(subject_is_corrected);
        handlearray(indices_of_corrected_candidates);
        handlearray(num_uncorrected_positions_per_subject);
        handlearray(uncorrected_positions_per_subject);

        handlearray(is_high_quality_subject);
        handlearray(high_quality_subject_indices);
        handlearray(num_high_quality_subject_indices);

        handlearray(alignment_scores);
        handlearray(alignment_overlaps);
        handlearray(alignment_shifts);
        handlearray(alignment_nOps);
        handlearray(alignment_isValid);
        handlearray(alignment_best_alignment_flags);

        handlearray(consensus);
        handlearray(support);
        handlearray(coverage);
        handlearray(origWeights);
        handlearray(origCoverages);
        handlearray(msa_column_properties);
        handlearray(counts);
        handlearray(weights);

        #undef handlearray

        return bytes;
    }

};

    #endif

}
}




#endif
