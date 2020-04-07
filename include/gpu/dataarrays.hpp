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


#if 0
    void resizeAnchorSequenceData(int numAnchors, int maximumSequenceBytes){
        const size_t pitch = SDIV(maximumSequenceBytes, padding_bytes) * padding_bytes;
        h_subject_read_ids.resize(numAnchors);
        h_subject_sequences_data.resize(numAnchors * (pitch / sizeof(int)));
        h_subject_sequences_lengths.resize(numAnchors);
        d_subject_read_ids.resize(numAnchors);
        d_subject_sequences_data.resize(numAnchors * (pitch / sizeof(int)));
        d_subject_sequences_lengths.resize(numAnchors);
    }

	void set_problem_dimensions(int n_sub, int n_quer, int max_seq_length, int max_seq_bytes, int min_overlap_, float min_overlap_ratio_, bool useQualityScores_){
        // n_subjects = n_sub;
		// n_queries = n_quer;
		// maximum_sequence_length = max_seq_length;
        // maximum_sequence_bytes = max_seq_bytes;
		// min_overlap = std::max(1, std::max(min_overlap_, int(maximum_sequence_length * min_overlap_ratio_)));
        // useQualityScores = useQualityScores_;

		// encoded_sequence_pitch = SDIV(maximum_sequence_bytes, padding_bytes) * padding_bytes;
		// quality_pitch = SDIV(max_seq_length * sizeof(char), padding_bytes) * padding_bytes;
		// sequence_pitch = SDIV(max_seq_length * sizeof(char), padding_bytes) * padding_bytes;

		// //sequence input data

        // h_subject_sequences_data.resize(n_sub * encoded_sequence_pitch);
        // h_candidate_sequences_data.resize(n_quer * encoded_sequence_pitch);
        // h_transposedCandidateSequencesData.resize(n_quer * encoded_sequence_pitch);
        // h_subject_sequences_lengths.resize(n_sub);
        // h_candidate_sequences_lengths.resize(n_quer);
        // h_candidates_per_subject.resize(n_sub);
        // h_candidates_per_subject_prefixsum.resize((n_sub + 1));
        // h_subject_read_ids.resize(n_sub);
        // h_candidate_read_ids.resize(n_quer);

        // d_subject_sequences_data.resize(n_sub * encoded_sequence_pitch);
        // d_candidate_sequences_data.resize(n_quer * encoded_sequence_pitch);
        // d_transposedCandidateSequencesData.resize(n_quer * encoded_sequence_pitch);
        // d_subject_sequences_lengths.resize(n_sub);
        // d_candidate_sequences_lengths.resize(n_quer);
        // d_candidates_per_subject.resize(n_sub);
        // d_candidates_per_subject_prefixsum.resize((n_sub + 1));
        // d_subject_read_ids.resize(n_sub);
        // d_candidate_read_ids.resize(n_quer);

		// //alignment output

		// h_alignment_scores.resize(2*n_quer);
        // h_alignment_overlaps.resize(2*n_quer);
        // h_alignment_shifts.resize(2*n_quer);
        // h_alignment_nOps.resize(2*n_quer);
        // h_alignment_isValid.resize(2*n_quer);
        // h_alignment_best_alignment_flags.resize(n_quer);

        // d_alignment_scores.resize(2*n_quer);
        // d_alignment_overlaps.resize(2*n_quer);
        // d_alignment_shifts.resize(2*n_quer);
        // d_alignment_nOps.resize(2*n_quer);
        // d_alignment_isValid.resize(2*n_quer);
        // d_alignment_best_alignment_flags.resize(n_quer);

		// // candidate indices

        // h_indices.resize(n_quer);
        // h_indices_per_subject.resize(n_sub);
        // h_indices_per_subject_prefixsum.resize((n_sub + 1));
        // h_num_indices.resize(1);

        // d_indices.resize(n_quer);
        // d_indices_per_subject.resize(n_sub);
        // d_indices_per_subject_prefixsum.resize((n_sub + 1));
        // d_num_indices.resize(1);
        // d_num_indices_tmp.resize(1);

		// //qualitiy scores
		// if(useQualityScores) {
        //     h_subject_qualities.resize(n_sub * quality_pitch);
        //     h_candidate_qualities.resize(n_quer * quality_pitch);

        //     d_subject_qualities.resize(n_sub * quality_pitch);
        //     d_candidate_qualities.resize(n_quer * quality_pitch);
        //     d_candidate_qualities_transposed.resize(n_quer * quality_pitch);
        //     d_candidate_qualities_tmp.resize(n_quer * quality_pitch);
		// }


		// //correction results

        // h_corrected_subjects.resize(n_sub * sequence_pitch);
        // h_corrected_candidates.resize(n_quer * sequence_pitch);
        // h_num_corrected_candidates_per_anchor.resize(n_sub);
        // h_subject_is_corrected.resize(n_sub);
        // h_indices_of_corrected_candidates.resize(n_quer);
        // h_num_uncorrected_positions_per_subject.resize(n_sub);
        // h_uncorrected_positions_per_subject.resize(n_sub * max_seq_length);

        // d_corrected_subjects.resize(n_sub * sequence_pitch);
        // d_corrected_candidates.resize(n_quer * sequence_pitch);
        // d_num_corrected_candidates_per_anchor.resize(n_sub);
        // d_subject_is_corrected.resize(n_sub);
        // d_indices_of_corrected_candidates.resize(n_quer);
        // d_num_uncorrected_positions_per_subject.resize(n_sub);
        // d_uncorrected_positions_per_subject.resize(n_sub * max_seq_length);

        // h_is_high_quality_subject.resize(n_sub);
        // h_high_quality_subject_indices.resize(n_sub);
        // h_num_high_quality_subject_indices.resize(1);

        // d_is_high_quality_subject.resize(n_sub);
        // d_high_quality_subject_indices.resize(n_sub);
        // d_num_high_quality_subject_indices.resize(1);

		// //multiple sequence alignment

        // int msa_max_column_count = (3*max_seq_length - 2*min_overlap_);
        // msa_pitch = SDIV(sizeof(char)*msa_max_column_count, padding_bytes) * padding_bytes;
        // msa_weights_pitch = SDIV(sizeof(float)*msa_max_column_count, padding_bytes) * padding_bytes;
        // size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

        // h_consensus.resize(n_sub * msa_pitch);
        // h_support.resize(n_sub * msa_weights_pitch_floats);
        // h_coverage.resize(n_sub * msa_weights_pitch_floats);
        // h_origWeights.resize(n_sub * msa_weights_pitch_floats);
        // h_origCoverages.resize(n_sub * msa_weights_pitch_floats);
        // h_msa_column_properties.resize(n_sub);
        // h_counts.resize(n_sub * 4 * msa_weights_pitch_floats);
        // h_weights.resize(n_sub * 4 * msa_weights_pitch_floats);

        // d_consensus.resize(n_sub * msa_pitch);
        // d_support.resize(n_sub * msa_weights_pitch_floats);
        // d_coverage.resize(n_sub * msa_weights_pitch_floats);
        // d_origWeights.resize(n_sub * msa_weights_pitch_floats);
        // d_origCoverages.resize(n_sub * msa_weights_pitch_floats);
        // d_msa_column_properties.resize(n_sub);
        // d_counts.resize(n_sub * 4 * msa_weights_pitch_floats);
        // d_weights.resize(n_sub * 4 * msa_weights_pitch_floats);


        // d_canExecute.resize(1);

        // size_t hostSizeBytes = hostArraysSizeInBytes();
        // size_t deviceSizeBytes = deviceArraysSizeInBytes();
        // size_t hostCapacityBytes = hostArraysCapacityInBytes();
        // size_t deviceCapacityBytes = deviceArraysCapacityInBytes();
        //
        // auto MB = [](auto bytes){
        //     return bytes / 1024. / 1024;
        // };
        //
        // std::cerr << "Resize: Host " << MB(hostSizeBytes) << " " << MB(hostCapacityBytes);
        // std::cerr << " Device " << MB(deviceSizeBytes) << " " << MB(deviceCapacityBytes) << '\n';
	}
#endif

	void set_cub_temp_storage_size(std::size_t newsize){
		d_cub_temp_storage.resize(newsize);
	}

	void zero_gpu(cudaStream_t stream){
        // cudaMemsetAsync(d_consensus, 0, d_consensus.sizeInBytes(), stream); CUERR;
        // cudaMemsetAsync(d_support, 0, d_support.sizeInBytes(), stream); CUERR;
        // cudaMemsetAsync(d_coverage, 0, d_coverage.sizeInBytes(), stream); CUERR;
        // cudaMemsetAsync(d_origWeights, 0, d_origWeights.sizeInBytes(), stream); CUERR;
        // cudaMemsetAsync(d_origCoverages, 0, d_origCoverages.sizeInBytes(), stream); CUERR;
        // cudaMemsetAsync(d_msa_column_properties, 0, d_msa_column_properties.sizeInBytes(), stream); CUERR;
        // cudaMemsetAsync(d_counts, 0, d_counts.sizeInBytes(), stream); CUERR;
        // cudaMemsetAsync(d_weights, 0, d_weights.sizeInBytes(), stream); CUERR;

        //cudaMemsetAsync(d_corrected_subjects, 0, d_corrected_subjects.sizeInBytes(), stream); CUERR;
        //cudaMemsetAsync(d_corrected_candidates, 0, d_corrected_candidates.sizeInBytes(), stream); CUERR;
        //cudaMemsetAsync(d_num_corrected_candidates_per_anchor, 0, d_num_corrected_candidates_per_anchor.sizeInBytes(), stream); CUERR;
        //cudaMemsetAsync(d_subject_is_corrected, 0, d_subject_is_corrected.sizeInBytes(), stream); CUERR;
        //cudaMemsetAsync(d_indices_of_corrected_candidates, 0, d_indices_of_corrected_candidates.sizeInBytes(), stream); CUERR;
        //cudaMemsetAsync(d_is_high_quality_subject, 0, d_is_high_quality_subject.sizeInBytes(), stream); CUERR;
        //cudaMemsetAsync(d_high_quality_subject_indices, 0, d_high_quality_subject_indices.sizeInBytes(), stream); CUERR;
        //cudaMemsetAsync(d_num_high_quality_subject_indices, 0, d_num_high_quality_subject_indices.sizeInBytes(), stream); CUERR;
        //cudaMemsetAsync(d_num_uncorrected_positions_per_subject, 0, d_num_uncorrected_positions_per_subject.sizeInBytes(), stream); CUERR;
        //cudaMemsetAsync(d_uncorrected_positions_per_subject, 0, d_uncorrected_positions_per_subject.sizeInBytes(), stream); CUERR;


        // cudaMemsetAsync(d_alignment_scores, 0, d_alignment_scores.sizeInBytes(), stream); CUERR;
        // cudaMemsetAsync(d_alignment_overlaps, 0, d_alignment_overlaps.sizeInBytes(), stream); CUERR;
        // cudaMemsetAsync(d_alignment_shifts, 0, d_alignment_shifts.sizeInBytes(), stream); CUERR;
        // cudaMemsetAsync(d_alignment_nOps, 0, d_alignment_nOps.sizeInBytes(), stream); CUERR;
        // cudaMemsetAsync(d_alignment_isValid, 0, d_alignment_isValid.sizeInBytes(), stream); CUERR;
        // cudaMemsetAsync(d_alignment_best_alignment_flags, 0, d_alignment_best_alignment_flags.sizeInBytes(), stream); CUERR;

        // cudaMemsetAsync(d_subject_sequences_data, 0, d_subject_sequences_data.sizeInBytes(), stream); CUERR;
        // cudaMemsetAsync(d_candidate_sequences_data, 0, d_candidate_sequences_data.sizeInBytes(), stream); CUERR;
        // cudaMemsetAsync(d_subject_sequences_lengths, 0, d_subject_sequences_lengths.sizeInBytes(), stream); CUERR;
        // cudaMemsetAsync(d_candidate_sequences_lengths, 0, d_candidate_sequences_lengths.sizeInBytes(), stream); CUERR;
        // cudaMemsetAsync(d_candidates_per_subject, 0, d_candidates_per_subject.sizeInBytes(), stream); CUERR;
        // cudaMemsetAsync(d_candidates_per_subject_prefixsum, 0, d_candidates_per_subject_prefixsum.sizeInBytes(), stream); CUERR;
        // cudaMemsetAsync(d_subject_read_ids, 0, d_subject_read_ids.sizeInBytes(), stream); CUERR;
        // cudaMemsetAsync(d_candidate_read_ids, 0, d_candidate_read_ids.sizeInBytes(), stream); CUERR;

        // if(useQualityScores){
        //     cudaMemsetAsync(d_subject_qualities, 0, d_subject_qualities.sizeInBytes(), stream); CUERR;
        //     cudaMemsetAsync(d_candidate_qualities, 0, d_candidate_qualities.sizeInBytes(), stream); CUERR;
        //     cudaMemsetAsync(d_candidate_qualities_transposed, 0, d_candidate_qualities.sizeInBytes(), stream); CUERR;
        //     cudaMemsetAsync(d_candidate_qualities_tmp, 0, d_candidate_qualities_tmp.sizeInBytes(), stream); CUERR;
        // }
	}

	void reset(){

        h_subject_sequences_data = std::move(PinnedBuffer<unsigned int>{});
        h_candidate_sequences_data = std::move(PinnedBuffer<unsigned int>{});
        h_transposedCandidateSequencesData = std::move(PinnedBuffer<unsigned int>{});
        h_subject_sequences_lengths = std::move(PinnedBuffer<int>{});
        h_candidate_sequences_lengths = std::move(PinnedBuffer<int>{});
        h_candidates_per_subject = std::move(PinnedBuffer<int>{});
        h_candidates_per_subject_prefixsum = std::move(PinnedBuffer<int>{});
        h_subject_read_ids = std::move(PinnedBuffer<read_number>{});
        h_candidate_read_ids = std::move(PinnedBuffer<read_number>{});

        d_subject_sequences_data = std::move(DeviceBuffer<unsigned int>{});
        d_candidate_sequences_data = std::move(DeviceBuffer<unsigned int>{});
        d_transposedCandidateSequencesData = std::move(DeviceBuffer<unsigned int>{});
        d_subject_sequences_lengths = std::move(DeviceBuffer<int>{});
        d_candidate_sequences_lengths = std::move(DeviceBuffer<int>{});
        d_candidates_per_subject = std::move(DeviceBuffer<int>{});
        d_candidates_per_subject_prefixsum = std::move(DeviceBuffer<int>{});
        d_subject_read_ids = std::move(DeviceBuffer<read_number>{});
        d_candidate_read_ids = std::move(DeviceBuffer<read_number>{});

        h_subject_qualities = std::move(PinnedBuffer<char>{});
        h_candidate_qualities = std::move(PinnedBuffer<char>{});

        d_subject_qualities = std::move(DeviceBuffer<char>{});
        d_candidate_qualities = std::move(DeviceBuffer<char>{});
        d_candidate_qualities_transposed = std::move(DeviceBuffer<char>{});
        //d_candidate_qualities_tmp = std::move(DeviceBuffer<char>{});

        h_consensus = std::move(PinnedBuffer<char>{});
        h_support = std::move(PinnedBuffer<float>{});
        h_coverage = std::move(PinnedBuffer<int>{});
        h_origWeights = std::move(PinnedBuffer<float>{});
        h_origCoverages = std::move(PinnedBuffer<int>{});
        h_msa_column_properties = std::move(PinnedBuffer<MSAColumnProperties>{});
        h_counts = std::move(PinnedBuffer<int>{});
        h_weights = std::move(PinnedBuffer<float>{});

        d_consensus = std::move(DeviceBuffer<char>{});
        d_support = std::move(DeviceBuffer<float>{});
        d_coverage = std::move(DeviceBuffer<int>{});
        d_origWeights = std::move(DeviceBuffer<float>{});
        d_origCoverages = std::move(DeviceBuffer<int>{});
        d_msa_column_properties = std::move(DeviceBuffer<MSAColumnProperties>{});
        d_counts = std::move(DeviceBuffer<int>{});
        d_weights = std::move(DeviceBuffer<float>{});

        h_alignment_scores = std::move(PinnedBuffer<int>{});
        h_alignment_overlaps = std::move(PinnedBuffer<int>{});
        h_alignment_shifts = std::move(PinnedBuffer<int>{});
        h_alignment_nOps = std::move(PinnedBuffer<int>{});
        h_alignment_isValid = std::move(PinnedBuffer<bool>{});
        h_alignment_best_alignment_flags = std::move(PinnedBuffer<BestAlignment_t>{});

        d_alignment_scores = std::move(DeviceBuffer<int>{});
        d_alignment_overlaps = std::move(DeviceBuffer<int>{});
        d_alignment_shifts = std::move(DeviceBuffer<int>{});
        d_alignment_nOps = std::move(DeviceBuffer<int>{});
        d_alignment_isValid = std::move(DeviceBuffer<bool>{});
        d_alignment_best_alignment_flags = std::move(DeviceBuffer<BestAlignment_t>{});

        h_corrected_subjects = std::move(PinnedBuffer<char>{});
        h_corrected_candidates = std::move(PinnedBuffer<char>{});
        h_num_corrected_candidates_per_anchor = std::move(PinnedBuffer<int>{});
        h_num_corrected_candidates_per_anchor_prefixsum = std::move(PinnedBuffer<int>{});
        h_subject_is_corrected = std::move(PinnedBuffer<bool>{});
        h_indices_of_corrected_candidates = std::move(PinnedBuffer<int>{});
        h_is_high_quality_subject = std::move(PinnedBuffer<AnchorHighQualityFlag>{});
        h_high_quality_subject_indices = std::move(PinnedBuffer<int>{});
        h_num_high_quality_subject_indices = std::move(PinnedBuffer<int>{});
        h_num_uncorrected_positions_per_subject = std::move(PinnedBuffer<int>{});
        h_uncorrected_positions_per_subject= std::move(PinnedBuffer<int>{});

        d_corrected_subjects = std::move(DeviceBuffer<char>{});
        d_corrected_candidates = std::move(DeviceBuffer<char>{});
        d_num_corrected_candidates_per_anchor = std::move(DeviceBuffer<int>{});
        d_num_corrected_candidates_per_anchor_prefixsum = std::move(DeviceBuffer<int>{});
        d_subject_is_corrected = std::move(DeviceBuffer<bool>{});
        d_indices_of_corrected_candidates = std::move(DeviceBuffer<int>{});
        d_is_high_quality_subject = std::move(DeviceBuffer<AnchorHighQualityFlag>{});
        d_high_quality_subject_indices = std::move(DeviceBuffer<int>{});
        d_num_high_quality_subject_indices = std::move(DeviceBuffer<int>{});
        d_num_uncorrected_positions_per_subject = std::move(DeviceBuffer<int>{});
        d_uncorrected_positions_per_subject= std::move(DeviceBuffer<int>{});

        h_indices = std::move(PinnedBuffer<int>{});
        h_indices_per_subject = std::move(PinnedBuffer<int>{});
        h_num_indices = std::move(PinnedBuffer<int>{});

        d_indices = std::move(DeviceBuffer<int>{});
        d_indices_per_subject = std::move(DeviceBuffer<int>{});
        d_num_indices = std::move(DeviceBuffer<int>{});

        d_indices_tmp = std::move(DeviceBuffer<int>{});
        d_indices_per_subject_tmp = std::move(DeviceBuffer<int>{});
        d_num_indices_tmp = std::move(DeviceBuffer<int>{});

        d_indices_of_corrected_subjects = std::move(DeviceBuffer<int>{});
        d_num_indices_of_corrected_subjects = std::move(DeviceBuffer<int>{});


        h_editsPerCorrectedSubject = std::move(PinnedBuffer<TempCorrectedSequence::Edit>{});
        h_numEditsPerCorrectedSubject = std::move(PinnedBuffer<int>{});
        h_editsPerCorrectedCandidate = std::move(PinnedBuffer<TempCorrectedSequence::Edit>{});
        h_numEditsPerCorrectedCandidate = std::move(PinnedBuffer<int>{});
        h_anchorContainsN = std::move(PinnedBuffer<bool>{});
        h_candidateContainsN = std::move(PinnedBuffer<bool>{});

        d_editsPerCorrectedSubject = std::move(DeviceBuffer<TempCorrectedSequence::Edit>{});
        d_numEditsPerCorrectedSubject = std::move(DeviceBuffer<int>{});
        d_editsPerCorrectedCandidate = std::move(DeviceBuffer<TempCorrectedSequence::Edit>{});
        d_numEditsPerCorrectedCandidate = std::move(DeviceBuffer<int>{});
        d_anchorContainsN = std::move(DeviceBuffer<bool>{});
        d_candidateContainsN = std::move(DeviceBuffer<bool>{});


        d_cub_temp_storage = std::move(DeviceBuffer<char>{});

        d_canExecute = std::move(DeviceBuffer<bool>{});
        
        d_tempstorage.destroy();
        d_numAnchors.destroy();
        d_numCandidates.destroy();
        h_numAnchors.destroy();
        h_numCandidates.destroy();

		// n_subjects = 0;
		// n_queries = 0;
		// n_indices = 0;
		// maximum_sequence_length = 0;
        // maximum_sequence_bytes = 0;
		// min_overlap = 1;

		// encoded_sequence_pitch = 0;
		// quality_pitch = 0;
		// sequence_pitch = 0;
		// msa_pitch = 0;
		// msa_weights_pitch = 0;
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

        bytes += f(d_cub_temp_storage);

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

        bytes += f(d_cub_temp_storage);

        bytes += f(d_canExecute);
        
        bytes += f(d_tempstorage);

        return bytes;
	}

	// int deviceId = -1;

	// int n_subjects = 0;
	// int n_queries = 0;
	// int n_indices = 0;
	// int maximum_sequence_length = 0;
    // int maximum_sequence_bytes = 0;
	// int min_overlap = 1;
    // bool useQualityScores = false;

	// alignment input

    ReadSequencesPointers getHostSequencePointers() const{
        ReadSequencesPointers pointers{
            h_subject_sequences_data.get(),
            h_candidate_sequences_data.get(),
            h_subject_sequences_lengths.get(),
            h_candidate_sequences_lengths.get(),
            h_transposedCandidateSequencesData.get()
        };
        return pointers;
    }

    ReadSequencesPointers getDeviceSequencePointers() const{
        ReadSequencesPointers pointers{
            d_subject_sequences_data.get(),
            d_candidate_sequences_data.get(),
            d_subject_sequences_lengths.get(),
            d_candidate_sequences_lengths.get(),
            d_transposedCandidateSequencesData.get()
        };
        return pointers;
    }
    
    DeviceBuffer<char> d_tempstorage;
    PinnedBuffer<int> h_numAnchors;
    PinnedBuffer<int> h_numCandidates;
    DeviceBuffer<int> d_numAnchors;
    DeviceBuffer<int> d_numCandidates;

	//std::size_t encoded_sequence_pitch = 0;

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


    ReadQualitiesPointers getHostQualityPointers() const{
        ReadQualitiesPointers pointers{
            h_subject_qualities.get(),
            h_candidate_qualities.get(),
            nullptr, //candidateQualitiesTransposed
        };
        return pointers;
    }

    ReadQualitiesPointers getDeviceQualityPointers() const{
        ReadQualitiesPointers pointers{
            d_subject_qualities.get(),
            d_candidate_qualities.get(),
            d_candidate_qualities_transposed.get(), //candidateQualitiesTransposed
        };
        return pointers;
    }

    //std::size_t quality_pitch = 0;

    PinnedBuffer<char> h_subject_qualities;
    PinnedBuffer<char> h_candidate_qualities;

    DeviceBuffer<char> d_subject_qualities;
    DeviceBuffer<char> d_candidate_qualities;
    DeviceBuffer<char> d_candidate_qualities_transposed;
    //DeviceBuffer<char> d_candidate_qualities_tmp;

	//correction results output

    CorrectionResultPointers getHostCorrectionResultPointers() const{
        CorrectionResultPointers pointers{
            h_corrected_subjects.get(),
            h_corrected_candidates.get(),
            h_num_corrected_candidates_per_anchor.get(),
            h_subject_is_corrected.get(),
            h_indices_of_corrected_candidates.get(),
            h_is_high_quality_subject.get(),
            h_high_quality_subject_indices.get(),
            h_num_high_quality_subject_indices.get(),
            h_num_uncorrected_positions_per_subject.get(),
            h_uncorrected_positions_per_subject.get(),
        };
        return pointers;
    }

    CorrectionResultPointers getDeviceCorrectionResultPointers() const{
        CorrectionResultPointers pointers{
            d_corrected_subjects.get(),
            d_corrected_candidates.get(),
            d_num_corrected_candidates_per_anchor.get(),
            d_subject_is_corrected.get(),
            d_indices_of_corrected_candidates.get(),
            d_is_high_quality_subject.get(),
            d_high_quality_subject_indices.get(),
            d_num_high_quality_subject_indices.get(),
            d_num_uncorrected_positions_per_subject.get(),
            d_uncorrected_positions_per_subject.get(),
        };
        return pointers;
    }

	//std::size_t sequence_pitch = 0;

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

    AlignmentResultPointers getHostAlignmentResultPointers() const{
        AlignmentResultPointers pointers{
            h_alignment_scores.get(),
            h_alignment_overlaps.get(),
            h_alignment_shifts.get(),
            h_alignment_nOps.get(),
            h_alignment_isValid.get(),
            h_alignment_best_alignment_flags.get(),
        };
        return pointers;
    }

    AlignmentResultPointers getDeviceAlignmentResultPointers() const{
        AlignmentResultPointers pointers{
            d_alignment_scores.get(),
            d_alignment_overlaps.get(),
            d_alignment_shifts.get(),
            d_alignment_nOps.get(),
            d_alignment_isValid.get(),
            d_alignment_best_alignment_flags.get(),
        };
        return pointers;
    }

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

	//tmp storage for cub
    DeviceBuffer<char> d_cub_temp_storage;

    DeviceBuffer<bool> d_canExecute;


	// multiple sequence alignment

	//std::size_t msa_pitch = 0;
	//std::size_t msa_weights_pitch = 0;

    MSAPointers getHostMSAPointers() const{
        MSAPointers ptrs{
            h_consensus.get(),
            h_support.get(),
            h_coverage.get(),
            h_origWeights.get(),
            h_origCoverages.get(),
            h_msa_column_properties.get(),
            h_counts.get(),
            h_weights.get(),
        };
        return  ptrs;
    }

    MSAPointers getDeviceMSAPointers() const{
        MSAPointers ptrs{
            d_consensus.get(),
            d_support.get(),
            d_coverage.get(),
            d_origWeights.get(),
            d_origCoverages.get(),
            d_msa_column_properties.get(),
            d_counts.get(),
            d_weights.get(),
        };
        return  ptrs;
    }

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

    void copyEverythingToHostForDebugging(){
        // auto handleArray = [](auto& host, const auto& device){
        //     assert(host.sizeInBytes() == device.sizeInBytes());
        //     cudaMemcpy(host,
        //                 device,
        //                 device.sizeInBytes(),
        //                 D2H); CUERR;
        // };

        #define handlearray(x){auto& host = h_##x; const auto& device = d_##x; \
            assert(host.sizeInBytes() == device.sizeInBytes()); \
            cudaMemcpy(host, \
                        device, \
                        device.sizeInBytes(), \
                        D2H); CUERR;          \
        }

        cudaDeviceSynchronize(); CUERR;

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

        cudaDeviceSynchronize(); CUERR;

        #undef handlearray
    }


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
