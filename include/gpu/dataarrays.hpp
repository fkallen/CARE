#ifndef CARE_GPU_DATA_ARRAYS_HPP
#define CARE_GPU_DATA_ARRAYS_HPP

#include "../hpc_helpers.cuh"
#include "bestalignment.hpp"
#include "msa.hpp"
#include "utility_kernels.cuh"
#include <gpu/thrust_custom_allocators.hpp>
#include <gpu/simpleallocation.cuh>
#include <gpu/kernels.hpp>
#include <config.hpp>

#ifdef __NVCC__

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/async/for_each.h>

#include <thrust/system/cuda/experimental/pinned_allocator.h>

#endif

namespace care {
namespace gpu {

#ifdef __NVCC__


struct DataArrays {
	static constexpr int padding_bytes = 4;
	static constexpr float allocfactor = 1.1;

	DataArrays() : DataArrays(0){
	}

	DataArrays(int deviceId) : deviceId(deviceId){
		//cudaSetDevice(deviceId);
	};

    void printActiveDataOfSubject(int subjectIndex, std::ostream& out){
        assert(subjectIndex < n_subjects);
#if 1
        size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

        const int numIndices = h_indices_per_subject[subjectIndex];
        const int* indices = h_indices + h_indices_per_subject_prefixsum[subjectIndex];
        const int subjectColumnsBegin_incl = h_msa_column_properties[subjectIndex].subjectColumnsBegin_incl;
        const int subjectColumnsEnd_excl = h_msa_column_properties[subjectIndex].subjectColumnsEnd_excl;
        const int firstColumn_incl = h_msa_column_properties[subjectIndex].firstColumn_incl;
        const int lastColumn_excl = h_msa_column_properties[subjectIndex].lastColumn_excl;
        const int columnsToCheck = lastColumn_excl - firstColumn_incl;

        const char* consensus = &h_consensus[subjectIndex * msa_pitch];
        const int* countsA = &h_counts[4* msa_weights_pitch_floats * subjectIndex + 0*msa_weights_pitch_floats];
        const int* countsC = &h_counts[4* msa_weights_pitch_floats * subjectIndex + 1*msa_weights_pitch_floats];
        const int* countsG = &h_counts[4* msa_weights_pitch_floats * subjectIndex + 2*msa_weights_pitch_floats];
        const int* countsT = &h_counts[4* msa_weights_pitch_floats * subjectIndex + 3*msa_weights_pitch_floats];
        const float* weightsA = &h_weights[4* msa_weights_pitch_floats * subjectIndex + 0*msa_weights_pitch_floats];
        const float* weightsC = &h_weights[4* msa_weights_pitch_floats * subjectIndex + 1*msa_weights_pitch_floats];
        const float* weightsG = &h_weights[4* msa_weights_pitch_floats * subjectIndex + 2*msa_weights_pitch_floats];
        const float* weightsT = &h_weights[4* msa_weights_pitch_floats * subjectIndex + 3*msa_weights_pitch_floats];

        const float* support = &h_support[msa_weights_pitch_floats * subjectIndex];
        const float* origWeights = &h_origWeights[msa_weights_pitch_floats * subjectIndex];
        const int* origCoverages = &h_origCoverages[msa_weights_pitch_floats * subjectIndex];

        const bool subject_is_corrected = h_subject_is_corrected[subjectIndex];
        const bool is_high_quality_subject = h_is_high_quality_subject[subjectIndex];

        const int numCandidates = h_candidates_per_subject_prefixsum[subjectIndex+1] - h_candidates_per_subject_prefixsum[subjectIndex];
        //std::ostream_iterator<double>(std::cout, " ")
        out << "subjectIndex: " << subjectIndex << '\n';
        out << "Subject: ";
        for(int i = 0; i < numCandidates; i++){

        }

        out << "numIndices: " << numIndices << '\n';
        out << "indices:\n";
        std::copy(indices, indices + numIndices, std::ostream_iterator<int>(out, " "));
        out << '\n';

        out << "subjectColumnsBegin_incl: " << subjectColumnsBegin_incl
                << ", subjectColumnsEnd_excl: " << subjectColumnsEnd_excl
                << ", columnsToCheck: " << columnsToCheck << '\n';

        out << "shifts:\n";
        for(int i = 0; i < numIndices; i++){
            out << h_alignment_shifts[indices[i]] << ", ";
        }
        out << '\n';

        out << "consensus:\n";
        std::copy(consensus, consensus + columnsToCheck, std::ostream_iterator<char>(out, ""));
        out << '\n';

        out << "countsA:\n";
        std::copy(countsA, countsA + columnsToCheck, std::ostream_iterator<int>(out, " "));
        out << '\n';

        out << "countsC:\n";
        std::copy(countsC, countsC + columnsToCheck, std::ostream_iterator<int>(out, " "));
        out << '\n';

        out << "countsG:\n";
        std::copy(countsG, countsG + columnsToCheck, std::ostream_iterator<int>(out, " "));
        out << '\n';

        out << "countsT:\n";
        std::copy(countsT, countsT + columnsToCheck, std::ostream_iterator<int>(out, " "));
        out << '\n';

        out << "weightsA:\n";
        std::copy(weightsA, weightsA + columnsToCheck, std::ostream_iterator<float>(out, " "));
        out << '\n';

        out << "weightsC:\n";
        std::copy(weightsC, weightsC + columnsToCheck, std::ostream_iterator<float>(out, " "));
        out << '\n';

        out << "weightsG:\n";
        std::copy(weightsG, weightsG + columnsToCheck, std::ostream_iterator<float>(out, " "));
        out << '\n';

        out << "weightsT:\n";
        std::copy(weightsT, weightsT + columnsToCheck, std::ostream_iterator<float>(out, " "));
        out << '\n';

        out << "support:\n";
        std::copy(support, support + columnsToCheck, std::ostream_iterator<float>(out, " "));
        out << '\n';

        out << "origWeights:\n";
        std::copy(origWeights, origWeights + columnsToCheck, std::ostream_iterator<float>(out, " "));
        out << '\n';

        out << "origCoverages:\n";
        std::copy(origCoverages, origCoverages + columnsToCheck, std::ostream_iterator<int>(out, " "));
        out << '\n';

        out << "subject_is_corrected: " << subject_is_corrected << '\n';
        out << "is_high_quality_subject: " << is_high_quality_subject << '\n';
#endif
    }

	void set_problem_dimensions(int n_sub, int n_quer, int max_seq_length, int max_seq_bytes, int min_overlap_, float min_overlap_ratio_, bool useQualityScores_){
        n_subjects = n_sub;
		n_queries = n_quer;
		maximum_sequence_length = max_seq_length;
        maximum_sequence_bytes = max_seq_bytes;
		min_overlap = std::max(1, std::max(min_overlap_, int(maximum_sequence_length * min_overlap_ratio_)));
        useQualityScores = useQualityScores_;

		encoded_sequence_pitch = SDIV(maximum_sequence_bytes, padding_bytes) * padding_bytes;
		quality_pitch = SDIV(max_seq_length * sizeof(char), padding_bytes) * padding_bytes;
		sequence_pitch = SDIV(max_seq_length * sizeof(char), padding_bytes) * padding_bytes;

		//sequence input data

        h_subject_sequences_data.resize(n_sub * encoded_sequence_pitch * allocfactor);
        h_candidate_sequences_data.resize(n_quer * encoded_sequence_pitch * allocfactor);
        h_subject_sequences_lengths.resize(n_sub * allocfactor);
        h_candidate_sequences_lengths.resize(n_quer * allocfactor);
        h_candidates_per_subject.resize(n_sub * allocfactor);
        h_candidates_per_subject_prefixsum.resize((n_sub + 1) * allocfactor);
        h_subject_read_ids.resize(n_sub * allocfactor);
        h_candidate_read_ids.resize(n_quer * allocfactor);

        d_subject_sequences_data.resize(n_sub * encoded_sequence_pitch * allocfactor);
        d_candidate_sequences_data.resize(n_quer * encoded_sequence_pitch * allocfactor);
        d_subject_sequences_lengths.resize(n_sub * allocfactor);
        d_candidate_sequences_lengths.resize(n_quer * allocfactor);
        d_candidates_per_subject.resize(n_sub * allocfactor);
        d_candidates_per_subject_prefixsum.resize((n_sub + 1) * allocfactor);
        d_subject_read_ids.resize(n_sub * allocfactor);
        d_candidate_read_ids.resize(n_quer * allocfactor);

        d_subject_sequences_data_transposed.resize(n_sub * encoded_sequence_pitch * allocfactor);
        d_candidate_sequences_data_transposed.resize(n_quer * encoded_sequence_pitch * allocfactor);

		//alignment output

		h_alignment_scores.resize(2*n_quer * allocfactor);
        h_alignment_overlaps.resize(2*n_quer * allocfactor);
        h_alignment_shifts.resize(2*n_quer * allocfactor);
        h_alignment_nOps.resize(2*n_quer * allocfactor);
        h_alignment_isValid.resize(2*n_quer * allocfactor);
        h_alignment_best_alignment_flags.resize(n_quer * allocfactor);

        d_alignment_scores.resize(2*n_quer * allocfactor);
        d_alignment_overlaps.resize(2*n_quer * allocfactor);
        d_alignment_shifts.resize(2*n_quer * allocfactor);
        d_alignment_nOps.resize(2*n_quer * allocfactor);
        d_alignment_isValid.resize(2*n_quer * allocfactor);
        d_alignment_best_alignment_flags.resize(n_quer * allocfactor);

		// candidate indices

        h_indices.resize(n_quer * allocfactor);
        h_indices_per_subject.resize(n_sub * allocfactor);
        h_indices_per_subject_prefixsum.resize((n_sub + 1) * allocfactor);
        h_num_indices.resize(1);

        d_indices.resize(n_quer * allocfactor);
        d_indices_per_subject.resize(n_sub * allocfactor);
        d_indices_per_subject_prefixsum.resize((n_sub + 1) * allocfactor);
        d_num_indices.resize(1);

		//qualitiy scores
		if(useQualityScores) {
            h_subject_qualities.resize(n_sub * quality_pitch * allocfactor);
            h_candidate_qualities.resize(n_quer * quality_pitch * allocfactor);

            d_subject_qualities.resize(n_sub * quality_pitch * allocfactor);
            d_candidate_qualities.resize(n_quer * quality_pitch * allocfactor);
            d_candidate_qualities_transposed.resize(n_quer * quality_pitch * allocfactor);
            d_candidate_qualities_tmp.resize(n_quer * quality_pitch * allocfactor);
		}


		//correction results

        h_corrected_subjects.resize(n_sub * sequence_pitch * allocfactor);
        h_corrected_candidates.resize(n_quer * sequence_pitch * allocfactor);
        h_num_corrected_candidates.resize(n_sub * allocfactor);
        h_subject_is_corrected.resize(n_sub * allocfactor);
        h_indices_of_corrected_candidates.resize(n_quer * allocfactor);

        d_corrected_subjects.resize(n_sub * sequence_pitch * allocfactor);
        d_corrected_candidates.resize(n_quer * sequence_pitch * allocfactor);
        d_num_corrected_candidates.resize(n_sub * allocfactor);
        d_subject_is_corrected.resize(n_sub * allocfactor);
        d_indices_of_corrected_candidates.resize(n_quer * allocfactor);

        h_is_high_quality_subject.resize(n_sub * allocfactor);
        h_high_quality_subject_indices.resize(n_sub * allocfactor);
        h_num_high_quality_subject_indices.resize(1);

        d_is_high_quality_subject.resize(n_sub * allocfactor);
        d_high_quality_subject_indices.resize(n_sub * allocfactor);
        d_num_high_quality_subject_indices.resize(1);

		//multiple sequence alignment

        int msa_max_column_count = (3*max_seq_length - 2*min_overlap_);
        msa_pitch = SDIV(sizeof(char)*msa_max_column_count, padding_bytes) * padding_bytes;
        msa_weights_pitch = SDIV(sizeof(float)*msa_max_column_count, padding_bytes) * padding_bytes;
        size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

        h_consensus.resize(n_sub * msa_pitch * allocfactor);
        h_support.resize(n_sub * msa_weights_pitch_floats * allocfactor);
        h_coverage.resize(n_sub * msa_weights_pitch_floats * allocfactor);
        h_origWeights.resize(n_sub * msa_weights_pitch_floats * allocfactor);
        h_origCoverages.resize(n_sub * msa_weights_pitch_floats * allocfactor);
        h_msa_column_properties.resize(n_sub * allocfactor);
        h_counts.resize(n_sub * 4 * msa_weights_pitch_floats * allocfactor);
        h_weights.resize(n_sub * 4 * msa_weights_pitch_floats * allocfactor);

        d_consensus.resize(n_sub * msa_pitch * allocfactor);
        d_support.resize(n_sub * msa_weights_pitch_floats * allocfactor);
        d_coverage.resize(n_sub * msa_weights_pitch_floats * allocfactor);
        d_origWeights.resize(n_sub * msa_weights_pitch_floats * allocfactor);
        d_origCoverages.resize(n_sub * msa_weights_pitch_floats * allocfactor);
        d_msa_column_properties.resize(n_sub * allocfactor);
        d_counts.resize(n_sub * 4 * msa_weights_pitch_floats * allocfactor);
        d_weights.resize(n_sub * 4 * msa_weights_pitch_floats * allocfactor);

	}


	void set_cub_temp_storage_size(std::size_t newsize){
		d_cub_temp_storage.resize(newsize * allocfactor);
	}

	void zero_gpu(cudaStream_t stream){
        cudaMemsetAsync(d_consensus, 0, d_consensus.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_support, 0, d_support.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_coverage, 0, d_coverage.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_origWeights, 0, d_origWeights.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_origCoverages, 0, d_origCoverages.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_msa_column_properties, 0, d_msa_column_properties.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_counts, 0, d_counts.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_weights, 0, d_weights.sizeInBytes(), stream); CUERR;

        cudaMemsetAsync(d_corrected_subjects, 0, d_corrected_subjects.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_corrected_candidates, 0, d_corrected_candidates.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_num_corrected_candidates, 0, d_num_corrected_candidates.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_subject_is_corrected, 0, d_subject_is_corrected.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_indices_of_corrected_candidates, 0, d_indices_of_corrected_candidates.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_is_high_quality_subject, 0, d_is_high_quality_subject.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_high_quality_subject_indices, 0, d_high_quality_subject_indices.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_num_high_quality_subject_indices, 0, d_num_high_quality_subject_indices.sizeInBytes(), stream); CUERR;

        cudaMemsetAsync(d_alignment_scores, 0, d_alignment_scores.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_alignment_overlaps, 0, d_alignment_overlaps.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_alignment_shifts, 0, d_alignment_shifts.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_alignment_nOps, 0, d_alignment_nOps.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_alignment_isValid, 0, d_alignment_isValid.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_alignment_best_alignment_flags, 0, d_alignment_best_alignment_flags.sizeInBytes(), stream); CUERR;

        cudaMemsetAsync(d_subject_sequences_data, 0, d_subject_sequences_data.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_candidate_sequences_data, 0, d_candidate_sequences_data.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_subject_sequences_lengths, 0, d_subject_sequences_lengths.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_candidate_sequences_lengths, 0, d_candidate_sequences_lengths.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_candidates_per_subject, 0, d_candidates_per_subject.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_candidates_per_subject_prefixsum, 0, d_candidates_per_subject_prefixsum.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_subject_read_ids, 0, d_subject_read_ids.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_candidate_read_ids, 0, d_candidate_read_ids.sizeInBytes(), stream); CUERR;

        if(useQualityScores){
            cudaMemsetAsync(d_subject_qualities, 0, d_subject_qualities.sizeInBytes(), stream); CUERR;
            cudaMemsetAsync(d_candidate_qualities, 0, d_candidate_qualities.sizeInBytes(), stream); CUERR;
            cudaMemsetAsync(d_candidate_qualities_transposed, 0, d_candidate_qualities.sizeInBytes(), stream); CUERR;
            cudaMemsetAsync(d_candidate_qualities_tmp, 0, d_candidate_qualities_tmp.sizeInBytes(), stream); CUERR;
        }
	}

	void reset(){

        h_subject_sequences_data = std::move(SimpleAllocationPinnedHost<char>{});
        h_candidate_sequences_data = std::move(SimpleAllocationPinnedHost<char>{});
        h_subject_sequences_lengths = std::move(SimpleAllocationPinnedHost<int>{});
        h_candidate_sequences_lengths = std::move(SimpleAllocationPinnedHost<int>{});
        h_candidates_per_subject = std::move(SimpleAllocationPinnedHost<int>{});
        h_candidates_per_subject_prefixsum = std::move(SimpleAllocationPinnedHost<int>{});
        h_subject_read_ids = std::move(SimpleAllocationPinnedHost<read_number>{});
        h_candidate_read_ids = std::move(SimpleAllocationPinnedHost<read_number>{});

        d_subject_sequences_data = std::move(SimpleAllocationDevice<char>{});
        d_candidate_sequences_data = std::move(SimpleAllocationDevice<char>{});
        d_subject_sequences_data_transposed = std::move(SimpleAllocationDevice<char>{});
        d_candidate_sequences_data_transposed = std::move(SimpleAllocationDevice<char>{});
        d_subject_sequences_lengths = std::move(SimpleAllocationDevice<int>{});
        d_candidate_sequences_lengths = std::move(SimpleAllocationDevice<int>{});
        d_candidates_per_subject = std::move(SimpleAllocationDevice<int>{});
        d_candidates_per_subject_prefixsum = std::move(SimpleAllocationDevice<int>{});
        d_subject_read_ids = std::move(SimpleAllocationDevice<read_number>{});
        d_candidate_read_ids = std::move(SimpleAllocationDevice<read_number>{});

        h_subject_qualities = std::move(SimpleAllocationPinnedHost<char>{});
        h_candidate_qualities = std::move(SimpleAllocationPinnedHost<char>{});

        d_subject_qualities = std::move(SimpleAllocationDevice<char>{});
        d_candidate_qualities = std::move(SimpleAllocationDevice<char>{});
        d_candidate_qualities_transposed = std::move(SimpleAllocationDevice<char>{});
        d_candidate_qualities_tmp = std::move(SimpleAllocationDevice<char>{});

        h_consensus = std::move(SimpleAllocationPinnedHost<char>{});
        h_support = std::move(SimpleAllocationPinnedHost<float>{});
        h_coverage = std::move(SimpleAllocationPinnedHost<int>{});
        h_origWeights = std::move(SimpleAllocationPinnedHost<float>{});
        h_origCoverages = std::move(SimpleAllocationPinnedHost<int>{});
        h_msa_column_properties = std::move(SimpleAllocationPinnedHost<MSAColumnProperties>{});
        h_counts = std::move(SimpleAllocationPinnedHost<int>{});
        h_weights = std::move(SimpleAllocationPinnedHost<float>{});

        d_consensus = std::move(SimpleAllocationDevice<char>{});
        d_support = std::move(SimpleAllocationDevice<float>{});
        d_coverage = std::move(SimpleAllocationDevice<int>{});
        d_origWeights = std::move(SimpleAllocationDevice<float>{});
        d_origCoverages = std::move(SimpleAllocationDevice<int>{});
        d_msa_column_properties = std::move(SimpleAllocationDevice<MSAColumnProperties>{});
        d_counts = std::move(SimpleAllocationDevice<int>{});
        d_weights = std::move(SimpleAllocationDevice<float>{});

        h_alignment_scores = std::move(SimpleAllocationPinnedHost<int>{});
        h_alignment_overlaps = std::move(SimpleAllocationPinnedHost<int>{});
        h_alignment_shifts = std::move(SimpleAllocationPinnedHost<int>{});
        h_alignment_nOps = std::move(SimpleAllocationPinnedHost<int>{});
        h_alignment_isValid = std::move(SimpleAllocationPinnedHost<bool>{});
        h_alignment_best_alignment_flags = std::move(SimpleAllocationPinnedHost<BestAlignment_t>{});

        d_alignment_scores = std::move(SimpleAllocationDevice<int>{});
        d_alignment_overlaps = std::move(SimpleAllocationDevice<int>{});
        d_alignment_shifts = std::move(SimpleAllocationDevice<int>{});
        d_alignment_nOps = std::move(SimpleAllocationDevice<int>{});
        d_alignment_isValid = std::move(SimpleAllocationDevice<bool>{});
        d_alignment_best_alignment_flags = std::move(SimpleAllocationDevice<BestAlignment_t>{});

        h_corrected_subjects = std::move(SimpleAllocationPinnedHost<char>{});
        h_corrected_candidates = std::move(SimpleAllocationPinnedHost<char>{});
        h_num_corrected_candidates = std::move(SimpleAllocationPinnedHost<int>{});
        h_subject_is_corrected = std::move(SimpleAllocationPinnedHost<bool>{});
        h_indices_of_corrected_candidates = std::move(SimpleAllocationPinnedHost<int>{});
        h_is_high_quality_subject = std::move(SimpleAllocationPinnedHost<bool>{});
        h_high_quality_subject_indices = std::move(SimpleAllocationPinnedHost<int>{});
        h_num_high_quality_subject_indices = std::move(SimpleAllocationPinnedHost<int>{});

        d_corrected_subjects = std::move(SimpleAllocationDevice<char>{});
        d_corrected_candidates = std::move(SimpleAllocationDevice<char>{});
        d_num_corrected_candidates = std::move(SimpleAllocationDevice<int>{});
        d_subject_is_corrected = std::move(SimpleAllocationDevice<bool>{});
        d_indices_of_corrected_candidates = std::move(SimpleAllocationDevice<int>{});
        d_is_high_quality_subject = std::move(SimpleAllocationDevice<bool>{});
        d_high_quality_subject_indices = std::move(SimpleAllocationDevice<int>{});
        d_num_high_quality_subject_indices = std::move(SimpleAllocationDevice<int>{});

        h_indices = std::move(SimpleAllocationPinnedHost<int>{});
        h_indices_per_subject = std::move(SimpleAllocationPinnedHost<int>{});
        h_indices_per_subject_prefixsum = std::move(SimpleAllocationPinnedHost<int>{});
        h_num_indices = std::move(SimpleAllocationPinnedHost<int>{});

        d_indices = std::move(SimpleAllocationDevice<int>{});
        d_indices_per_subject = std::move(SimpleAllocationDevice<int>{});
        d_indices_per_subject_prefixsum = std::move(SimpleAllocationDevice<int>{});
        d_num_indices = std::move(SimpleAllocationDevice<int>{});

        d_cub_temp_storage = std::move(SimpleAllocationDevice<char>{});

		n_subjects = 0;
		n_queries = 0;
		n_indices = 0;
		maximum_sequence_length = 0;
        maximum_sequence_bytes = 0;
		min_overlap = 1;

		encoded_sequence_pitch = 0;
		quality_pitch = 0;
		sequence_pitch = 0;
		msa_pitch = 0;
		msa_weights_pitch = 0;
	}

	int deviceId = -1;

	int n_subjects = 0;
	int n_queries = 0;
	int n_indices = 0;
	int maximum_sequence_length = 0;
    int maximum_sequence_bytes = 0;
	int min_overlap = 1;
    bool useQualityScores = false;

	//subject indices

    SimpleAllocationPinnedHost<bool> h_is_high_quality_subject;
    SimpleAllocationPinnedHost<int> h_high_quality_subject_indices;
    SimpleAllocationPinnedHost<int> h_num_high_quality_subject_indices;

    SimpleAllocationDevice<bool> d_is_high_quality_subject;
    SimpleAllocationDevice<int> d_high_quality_subject_indices;
    SimpleAllocationDevice<int> d_num_high_quality_subject_indices;

	// alignment input

    struct

	std::size_t encoded_sequence_pitch = 0;

    SimpleAllocationPinnedHost<char> h_subject_sequences_data;
    SimpleAllocationPinnedHost<char> h_candidate_sequences_data;
    SimpleAllocationPinnedHost<int> h_subject_sequences_lengths;
    SimpleAllocationPinnedHost<int> h_candidate_sequences_lengths;
    SimpleAllocationPinnedHost<int> h_candidates_per_subject;
    SimpleAllocationPinnedHost<int> h_candidates_per_subject_prefixsum;
    SimpleAllocationPinnedHost<read_number> h_subject_read_ids;
    SimpleAllocationPinnedHost<read_number> h_candidate_read_ids;

    SimpleAllocationDevice<char> d_subject_sequences_data;
    SimpleAllocationDevice<char> d_candidate_sequences_data;
    SimpleAllocationDevice<char> d_subject_sequences_data_transposed;
    SimpleAllocationDevice<char> d_candidate_sequences_data_transposed;
    SimpleAllocationDevice<int> d_subject_sequences_lengths;
    SimpleAllocationDevice<int> d_candidate_sequences_lengths;
    SimpleAllocationDevice<int> d_candidates_per_subject;
    SimpleAllocationDevice<int> d_candidates_per_subject_prefixsum;
    SimpleAllocationDevice<read_number> d_subject_read_ids;
    SimpleAllocationDevice<read_number> d_candidate_read_ids;

	//indices

    SimpleAllocationPinnedHost<int> h_indices;
    SimpleAllocationPinnedHost<int> h_indices_per_subject;
    SimpleAllocationPinnedHost<int> h_indices_per_subject_prefixsum;
    SimpleAllocationPinnedHost<int> h_num_indices;

    SimpleAllocationDevice<int> d_indices;
    SimpleAllocationDevice<int> d_indices_per_subject;
    SimpleAllocationDevice<int> d_indices_per_subject_prefixsum;
    SimpleAllocationDevice<int> d_num_indices;



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

    std::size_t quality_pitch = 0;

    SimpleAllocationPinnedHost<char> h_subject_qualities;
    SimpleAllocationPinnedHost<char> h_candidate_qualities;

    SimpleAllocationDevice<char> d_subject_qualities;
    SimpleAllocationDevice<char> d_candidate_qualities;
    SimpleAllocationDevice<char> d_candidate_qualities_transposed;
    SimpleAllocationDevice<char> d_candidate_qualities_tmp;

	//correction results output

	std::size_t sequence_pitch = 0;

    SimpleAllocationPinnedHost<char> h_corrected_subjects;
    SimpleAllocationPinnedHost<char> h_corrected_candidates;
    SimpleAllocationPinnedHost<int> h_num_corrected_candidates;
    SimpleAllocationPinnedHost<bool> h_subject_is_corrected;
    SimpleAllocationPinnedHost<int> h_indices_of_corrected_candidates;

    SimpleAllocationDevice<char> d_corrected_subjects;
    SimpleAllocationDevice<char> d_corrected_candidates;
    SimpleAllocationDevice<int> d_num_corrected_candidates;
    SimpleAllocationDevice<bool> d_subject_is_corrected;
    SimpleAllocationDevice<int> d_indices_of_corrected_candidates;


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

    SimpleAllocationPinnedHost<int> h_alignment_scores;
    SimpleAllocationPinnedHost<int> h_alignment_overlaps;
    SimpleAllocationPinnedHost<int> h_alignment_shifts;
    SimpleAllocationPinnedHost<int> h_alignment_nOps;
    SimpleAllocationPinnedHost<bool> h_alignment_isValid;
    SimpleAllocationPinnedHost<BestAlignment_t> h_alignment_best_alignment_flags;

    SimpleAllocationDevice<int> d_alignment_scores;
    SimpleAllocationDevice<int> d_alignment_overlaps;
    SimpleAllocationDevice<int> d_alignment_shifts;
    SimpleAllocationDevice<int> d_alignment_nOps;
    SimpleAllocationDevice<bool> d_alignment_isValid;
    SimpleAllocationDevice<BestAlignment_t> d_alignment_best_alignment_flags;

	//tmp storage for cub
    SimpleAllocationDevice<char> d_cub_temp_storage;


	// multiple sequence alignment

	std::size_t msa_pitch = 0;
	std::size_t msa_weights_pitch = 0;

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

    SimpleAllocationPinnedHost<char> h_consensus;
    SimpleAllocationPinnedHost<float> h_support;
    SimpleAllocationPinnedHost<int> h_coverage;
    SimpleAllocationPinnedHost<float> h_origWeights;
    SimpleAllocationPinnedHost<int> h_origCoverages;
    SimpleAllocationPinnedHost<MSAColumnProperties> h_msa_column_properties;
    SimpleAllocationPinnedHost<int> h_counts;
    SimpleAllocationPinnedHost<float> h_weights;

    SimpleAllocationDevice<char> d_consensus;
    SimpleAllocationDevice<float> d_support;
    SimpleAllocationDevice<int> d_coverage;
    SimpleAllocationDevice<float> d_origWeights;
    SimpleAllocationDevice<int> d_origCoverages;
    SimpleAllocationDevice<MSAColumnProperties> d_msa_column_properties;
    SimpleAllocationDevice<int> d_counts;
    SimpleAllocationDevice<float> d_weights;

};

    #endif

}
}




#endif
