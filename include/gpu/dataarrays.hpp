#ifndef CARE_GPU_DATA_ARRAYS_HPP
#define CARE_GPU_DATA_ARRAYS_HPP

#include "../hpc_helpers.cuh"
#include "bestalignment.hpp"
#include "msa.hpp"

#ifdef __NVCC__

#include <thrust/fill.h>
#include <thrust/device_ptr.h>

#endif

namespace care {
namespace gpu {

#ifdef __NVCC__

template<class Sequence_t, class ReadId_t>
struct DataArrays {
	static constexpr int padding_bytes = 4;
	static constexpr float allocfactor = 1.1;

	DataArrays() : DataArrays(0){
	}

	DataArrays(int deviceId) : deviceId(deviceId){
		//cudaSetDevice(deviceId);
	};

	void allocCandidateIds(int n_quer){
		memCandidateIds = SDIV(sizeof(ReadId_t) * n_quer, padding_bytes) * padding_bytes;

		std::size_t required_size = memCandidateIds;

		if(required_size > candidate_ids_allocation_size) {
			cudaFree(d_candidate_read_ids); CUERR;
			cudaFreeHost(h_candidate_read_ids); CUERR;

			cudaMalloc(&d_candidate_read_ids, size_t(required_size * allocfactor)); CUERR;
			cudaMallocHost(&h_candidate_read_ids, size_t(required_size * allocfactor)); CUERR;

			candidate_ids_allocation_size = required_size;
		}

		candidate_ids_usable_size = required_size;
	}

	void set_problem_dimensions(int n_sub, int n_quer, int max_seq_length, int min_overlap_, float min_overlap_ratio_, bool useQualityScores){

		encoded_sequence_pitch = SDIV(Sequence_t::getNumBytes(max_seq_length), padding_bytes) * padding_bytes;
		quality_pitch = SDIV(max_seq_length * sizeof(char), padding_bytes) * padding_bytes;
		sequence_pitch = SDIV(max_seq_length * sizeof(char), padding_bytes) * padding_bytes;
		int msa_max_column_count = (3*max_seq_length - 2*min_overlap);
		msa_pitch = SDIV(sizeof(char)*msa_max_column_count, padding_bytes) * padding_bytes;
		msa_weights_pitch = SDIV(sizeof(float)*msa_max_column_count, padding_bytes) * padding_bytes;

		//alignment input
		memSubjects = n_sub * encoded_sequence_pitch;
		memSubjectLengths = SDIV(n_sub * sizeof(int), padding_bytes) * padding_bytes;
		memNqueriesPrefixSum = SDIV((n_sub+1) * sizeof(int), padding_bytes) * padding_bytes;
		memQueries = n_quer * encoded_sequence_pitch;
		memQueryLengths = SDIV(n_quer * sizeof(int), padding_bytes) * padding_bytes;
		memSubjectIds = SDIV(sizeof(ReadId_t) * n_sub, padding_bytes) * padding_bytes;
		//memCandidateIds = SDIV(sizeof(ReadId_t) * n_quer, padding_bytes) * padding_bytes;

		std::size_t required_alignment_transfer_data_allocation_size = memSubjects
		                                                               + memSubjectLengths
		                                                               + memNqueriesPrefixSum
		                                                               + memQueries
		                                                               + memQueryLengths
		                                                               + memSubjectIds;
		//+ memCandidateIds;

		if(required_alignment_transfer_data_allocation_size > alignment_transfer_data_allocation_size) {
			//std::cout << "A" << std::endl;
			cudaFree(alignment_transfer_data_device); CUERR;
			cudaMalloc(&alignment_transfer_data_device, std::size_t(required_alignment_transfer_data_allocation_size * allocfactor)); CUERR;
			cudaFreeHost(alignment_transfer_data_host); CUERR;
			cudaMallocHost(&alignment_transfer_data_host, std::size_t(required_alignment_transfer_data_allocation_size * allocfactor)); CUERR;

			alignment_transfer_data_allocation_size = std::size_t(required_alignment_transfer_data_allocation_size * allocfactor);
		}

		alignment_transfer_data_usable_size = required_alignment_transfer_data_allocation_size;

		h_subject_sequences_data = (char*)alignment_transfer_data_host;
		h_candidate_sequences_data = (char*)(((char*)h_subject_sequences_data) + memSubjects);
		h_subject_sequences_lengths = (int*)(((char*)h_candidate_sequences_data) + memQueries);
		h_candidate_sequences_lengths = (int*)(((char*)h_subject_sequences_lengths) + memSubjectLengths);
		h_candidates_per_subject_prefixsum = (int*)(((char*)h_candidate_sequences_lengths) + memQueryLengths);
		h_subject_read_ids = (ReadId_t*)(((char*)h_candidates_per_subject_prefixsum) + memNqueriesPrefixSum);
		//h_candidate_read_ids = (ReadId_t*)(((char*)h_subject_read_ids) + memSubjectIds);

		d_subject_sequences_data = (char*)alignment_transfer_data_device;
		d_candidate_sequences_data = (char*)(((char*)d_subject_sequences_data) + memSubjects);
		d_subject_sequences_lengths = (int*)(((char*)d_candidate_sequences_data) + memQueries);
		d_candidate_sequences_lengths = (int*)(((char*)d_subject_sequences_lengths) + memSubjectLengths);
		d_candidates_per_subject_prefixsum = (int*)(((char*)d_candidate_sequences_lengths) + memQueryLengths);
		d_subject_read_ids = (ReadId_t*)(((char*)d_candidates_per_subject_prefixsum) + memNqueriesPrefixSum);
		//d_candidate_read_ids = (ReadId_t*)(((char*)d_subject_read_ids) + memSubjectIds);

		//alignment output
		std::size_t memAlignmentScores = SDIV((2*n_quer) * sizeof(int), padding_bytes) * padding_bytes;
		std::size_t memAlignmentOverlaps = SDIV((2*n_quer) * sizeof(int), padding_bytes) * padding_bytes;
		std::size_t memAlignmentShifts = SDIV((2*n_quer) * sizeof(int), padding_bytes) * padding_bytes;
		std::size_t memAlignmentnOps = SDIV((2*n_quer) * sizeof(int), padding_bytes) * padding_bytes;
		std::size_t memAlignmentisValid = SDIV((2*n_quer) * sizeof(bool), padding_bytes) * padding_bytes;
		std::size_t memAlignmentBestAlignmentFlags = SDIV((n_quer) * sizeof(BestAlignment_t), padding_bytes) * padding_bytes;

		std::size_t required_alignment_result_data_allocation_size = memAlignmentScores
		                                                             + memAlignmentOverlaps
		                                                             + memAlignmentShifts
		                                                             + memAlignmentnOps
		                                                             + memAlignmentisValid
		                                                             + memAlignmentBestAlignmentFlags;

		if(required_alignment_result_data_allocation_size > alignment_result_data_allocation_size) {
			//std::cout << "B" << std::endl;
			cudaFree(alignment_result_data_device); CUERR;
			cudaMalloc(&alignment_result_data_device, std::size_t(required_alignment_result_data_allocation_size * allocfactor)); CUERR;
			cudaFreeHost(alignment_result_data_host); CUERR;
			cudaMallocHost(&alignment_result_data_host, std::size_t(required_alignment_result_data_allocation_size * allocfactor)); CUERR;


			alignment_result_data_allocation_size = std::size_t(required_alignment_result_data_allocation_size * allocfactor);
		}

		alignment_result_data_usable_size = required_alignment_result_data_allocation_size;

		h_alignment_scores = (int*)alignment_result_data_host;
		h_alignment_overlaps = (int*)(((char*)h_alignment_scores) + memAlignmentScores);
		h_alignment_shifts = (int*)(((char*)h_alignment_overlaps) + memAlignmentOverlaps);
		h_alignment_nOps = (int*)(((char*)h_alignment_shifts) + memAlignmentShifts);
		h_alignment_isValid = (bool*)(((char*)h_alignment_nOps) + memAlignmentnOps);
		h_alignment_best_alignment_flags = (BestAlignment_t*)(((char*)h_alignment_isValid) + memAlignmentisValid);

		d_alignment_scores = (int*)alignment_result_data_device;
		d_alignment_overlaps = (int*)(((char*)d_alignment_scores) + memAlignmentScores);
		d_alignment_shifts = (int*)(((char*)d_alignment_overlaps) + memAlignmentOverlaps);
		d_alignment_nOps = (int*)(((char*)d_alignment_shifts) + memAlignmentShifts);
		d_alignment_isValid = (bool*)(((char*)d_alignment_nOps) + memAlignmentnOps);
		d_alignment_best_alignment_flags = (BestAlignment_t*)(((char*)d_alignment_isValid) + memAlignmentisValid);


		//indices of hq subjects
		std::size_t memSubjectIndices = SDIV((n_sub) * sizeof(int), padding_bytes) * padding_bytes;
		std::size_t memHQSubjectIndices = SDIV((n_sub) * sizeof(int), padding_bytes) * padding_bytes;
		std::size_t memIsHQSubject = SDIV((n_sub) * sizeof(bool), padding_bytes) * padding_bytes;
		std::size_t memNumHQSubjectIndices = sizeof(int);

		std::size_t required_subject_indices_data_allocation_size = memSubjectIndices
		                                                            + memHQSubjectIndices
		                                                            + memIsHQSubject
		                                                            + memNumHQSubjectIndices;

		if(required_subject_indices_data_allocation_size > subject_indices_data_allocation_size) {
			//std::cout << "C" << " " << n_sub << " " << required_subject_indices_data_allocation_size << " >= " <<  subject_indices_data_allocation_size << std::endl;
			cudaFree(subject_indices_data_device); CUERR;
			cudaMalloc(&subject_indices_data_device, std::size_t(required_subject_indices_data_allocation_size * allocfactor)); CUERR;

			cudaFreeHost(subject_indices_data_host); CUERR;
			cudaMallocHost(&subject_indices_data_host, std::size_t(required_subject_indices_data_allocation_size * allocfactor)); CUERR;

			subject_indices_data_allocation_size = required_subject_indices_data_allocation_size;
		}

		subject_indices_data_usable_size = required_subject_indices_data_allocation_size;

		h_subject_indices = (int*)subject_indices_data_host;
		h_high_quality_subject_indices = (int*)(((char*)h_subject_indices) + memSubjectIndices);
		h_is_high_quality_subject = (bool*)(((char*)h_high_quality_subject_indices) + memHQSubjectIndices);
		h_num_high_quality_subject_indices = (int*)(((char*)h_is_high_quality_subject) + memIsHQSubject);

		d_subject_indices = (int*)subject_indices_data_device;
		d_high_quality_subject_indices = (int*)(((char*)d_subject_indices) + memSubjectIndices);
		d_is_high_quality_subject = (bool*)(((char*)d_high_quality_subject_indices) + memHQSubjectIndices);
		d_num_high_quality_subject_indices = (int*)(((char*)d_is_high_quality_subject) + memIsHQSubject);


		// candidate indices
		if(d_num_indices == nullptr) {
			cudaMalloc(&d_num_indices, sizeof(int)); CUERR;
		}
		if(h_num_indices == nullptr) {
			cudaMallocHost(&h_num_indices, sizeof(int)); CUERR;
		}

		std::size_t memIndices = SDIV(n_quer * sizeof(int), padding_bytes) * padding_bytes;
		std::size_t memIndicesPerSubject = SDIV(n_sub* sizeof(int), padding_bytes) * padding_bytes;
		std::size_t memIndicesPerSubjectPrefixSum = SDIV((n_sub+1)* sizeof(int), padding_bytes) * padding_bytes;

		std::size_t required_indices_transfer_data_allocation_size = memIndices
		                                                             + memIndicesPerSubject
		                                                             + memIndicesPerSubjectPrefixSum;

		if(required_indices_transfer_data_allocation_size > indices_transfer_data_allocation_size) {
			//std::cout << "D" << std::endl;
			cudaFree(indices_transfer_data_device); CUERR;
			cudaMalloc(&indices_transfer_data_device, std::size_t(required_indices_transfer_data_allocation_size * allocfactor)); CUERR;
			cudaFreeHost(indices_transfer_data_host); CUERR;
			cudaMallocHost(&indices_transfer_data_host, std::size_t(required_indices_transfer_data_allocation_size * allocfactor)); CUERR;

			indices_transfer_data_allocation_size = std::size_t(required_indices_transfer_data_allocation_size * allocfactor);
		}

		indices_transfer_data_usable_size = required_indices_transfer_data_allocation_size;

		h_indices = (int*)indices_transfer_data_host;
		h_indices_per_subject = (int*)(((char*)h_indices) + memIndices);
		h_indices_per_subject_prefixsum = (int*)(((char*)h_indices_per_subject) + memIndicesPerSubject);

		d_indices = (int*)indices_transfer_data_device;
		d_indices_per_subject = (int*)(((char*)d_indices) + memIndices);
		d_indices_per_subject_prefixsum = (int*)(((char*)d_indices_per_subject) + memIndicesPerSubject);

		//qualitiy scores
		if(useQualityScores) {
			std::size_t memCandidateQualities = n_quer * quality_pitch;
			std::size_t memSubjectQualities = n_sub * quality_pitch;

			std::size_t required_qualities_transfer_data_allocation_size = memCandidateQualities
			                                                               + memSubjectQualities;

			if(required_qualities_transfer_data_allocation_size > qualities_transfer_data_allocation_size) {
				//std::cout << "E" << std::endl;
				cudaFree(qualities_transfer_data_device); CUERR;
				cudaMalloc(&qualities_transfer_data_device, std::size_t(required_qualities_transfer_data_allocation_size * allocfactor)); CUERR;
				cudaFreeHost(qualities_transfer_data_host); CUERR;
				cudaMallocHost(&qualities_transfer_data_host, std::size_t(required_qualities_transfer_data_allocation_size * allocfactor)); CUERR;

				qualities_transfer_data_allocation_size = std::size_t(required_qualities_transfer_data_allocation_size * allocfactor);
			}

			qualities_transfer_data_usable_size = required_qualities_transfer_data_allocation_size;

			h_candidate_qualities = (char*)qualities_transfer_data_host;
			h_subject_qualities = (char*)(((char*)h_candidate_qualities) + memCandidateQualities);

			d_candidate_qualities = (char*)qualities_transfer_data_device;
			d_subject_qualities = (char*)(((char*)d_candidate_qualities) + memCandidateQualities);
		}


		//correction results

		std::size_t memCorrectedSubjects = n_sub * sequence_pitch;
		std::size_t memCorrectedCandidates = n_quer * sequence_pitch;
		std::size_t memNumCorrectedCandidates = SDIV(n_sub * sizeof(int), padding_bytes) * padding_bytes;
		std::size_t memSubjectIsCorrected = SDIV(n_sub * sizeof(bool), padding_bytes) * padding_bytes;
		std::size_t memIndicesOfCorrectedCandidates = SDIV(n_quer * sizeof(int), padding_bytes) * padding_bytes;

		std::size_t required_correction_results_transfer_data_allocation_size = memCorrectedSubjects
		                                                                        + memCorrectedCandidates
		                                                                        + memNumCorrectedCandidates
		                                                                        + memSubjectIsCorrected
		                                                                        + memIndicesOfCorrectedCandidates;

		if(required_correction_results_transfer_data_allocation_size > correction_results_transfer_data_allocation_size) {
			//std::cout << "F" << std::endl;
			cudaFree(correction_results_transfer_data_device); CUERR;
			cudaMalloc(&correction_results_transfer_data_device, std::size_t(required_correction_results_transfer_data_allocation_size * allocfactor)); CUERR;
			cudaFreeHost(correction_results_transfer_data_host); CUERR;
			cudaMallocHost(&correction_results_transfer_data_host, std::size_t(required_correction_results_transfer_data_allocation_size * allocfactor)); CUERR;

			correction_results_transfer_data_allocation_size = std::size_t(required_correction_results_transfer_data_allocation_size * allocfactor);
		}

		correction_results_transfer_data_usable_size = required_correction_results_transfer_data_allocation_size;

		h_corrected_subjects = (char*)correction_results_transfer_data_host;
		h_corrected_candidates = (char*)(((char*)h_corrected_subjects) + memCorrectedSubjects);
		h_num_corrected_candidates = (int*)(((char*)h_corrected_candidates) + memCorrectedCandidates);
		h_subject_is_corrected = (bool*)(((char*)h_num_corrected_candidates) + memNumCorrectedCandidates);
		h_indices_of_corrected_candidates = (int*)(((char*)h_subject_is_corrected) + memSubjectIsCorrected);

		d_corrected_subjects = (char*)correction_results_transfer_data_device;
		d_corrected_candidates = (char*)(((char*)d_corrected_subjects) + memCorrectedSubjects);
		d_num_corrected_candidates = (int*)(((char*)d_corrected_candidates) + memCorrectedCandidates);
		d_subject_is_corrected = (bool*)(((char*)d_num_corrected_candidates) + memNumCorrectedCandidates);
		d_indices_of_corrected_candidates = (int*)(((char*)d_subject_is_corrected) + memSubjectIsCorrected);


		//multiple sequence alignment

		std::size_t memMultipleSequenceAlignment = (n_sub + n_quer) * msa_pitch;
		std::size_t memMultipleSequenceAlignmentWeights = (n_sub + n_quer) * msa_weights_pitch;
		std::size_t memConsensus = n_sub * msa_pitch;
		std::size_t memSupport = n_sub * msa_weights_pitch;
		std::size_t memCoverage = n_sub * msa_weights_pitch;
		std::size_t memOrigWeights = n_sub * msa_weights_pitch;
		std::size_t memOrigCoverage = n_sub * msa_weights_pitch;
		std::size_t memMSAColumnProperties = SDIV(n_sub * sizeof(MSAColumnProperties), padding_bytes) * padding_bytes;
		std::size_t memIsHighQualityMSA = SDIV(n_sub *  sizeof(bool), padding_bytes) * padding_bytes;

		std::size_t required_msa_data_allocation_size = memMultipleSequenceAlignment
		                                                + memMultipleSequenceAlignmentWeights
		                                                + memConsensus
		                                                + memSupport
		                                                + memCoverage
		                                                + memOrigWeights
		                                                + memOrigCoverage
		                                                + memMSAColumnProperties
		                                                + memIsHighQualityMSA;

		if(required_msa_data_allocation_size > msa_data_allocation_size) {
			//std::cout << "G" << std::endl;
			cudaFree(msa_data_device); CUERR;
			cudaMalloc(&msa_data_device, std::size_t(required_msa_data_allocation_size * allocfactor)); CUERR;
			cudaFreeHost(msa_data_host); CUERR;
			cudaMallocHost(&msa_data_host, std::size_t(required_msa_data_allocation_size * allocfactor)); CUERR;

			msa_data_allocation_size = std::size_t(required_msa_data_allocation_size * allocfactor);
		}

		msa_data_usable_size = required_msa_data_allocation_size;

		h_multiple_sequence_alignments = (char*)msa_data_host;
		h_multiple_sequence_alignment_weights = (float*)(((char*)h_multiple_sequence_alignments) + memMultipleSequenceAlignment);
		h_consensus = (char*)(((char*)h_multiple_sequence_alignment_weights) + memMultipleSequenceAlignmentWeights);
		h_support = (float*)(((char*)h_consensus) + memConsensus);
		h_coverage = (int*)(((char*)h_support) + memSupport);
		h_origWeights = (float*)(((char*)h_coverage) + memCoverage);
		h_origCoverages = (int*)(((char*)h_origWeights) + memOrigWeights);
		h_msa_column_properties = (MSAColumnProperties*)(((char*)h_origCoverages) + memOrigCoverage);

		d_multiple_sequence_alignments = (char*)msa_data_device;
		d_multiple_sequence_alignment_weights = (float*)(((char*)d_multiple_sequence_alignments) + memMultipleSequenceAlignment);
		d_consensus = (char*)(((char*)d_multiple_sequence_alignment_weights) + memMultipleSequenceAlignmentWeights);
		d_support = (float*)(((char*)d_consensus) + memConsensus);
		d_coverage = (int*)(((char*)d_support) + memSupport);
		d_origWeights = (float*)(((char*)d_coverage) + memCoverage);
		d_origCoverages = (int*)(((char*)d_origWeights) + memOrigWeights);
		d_msa_column_properties = (MSAColumnProperties*)(((char*)d_origCoverages) + memOrigCoverage);





		n_subjects = n_sub;
		n_queries = n_quer;
		maximum_sequence_length = max_seq_length;
		min_overlap = std::max(1, std::max(min_overlap_, int(maximum_sequence_length * min_overlap_ratio_)));
	}


	void set_tmp_storage_size(std::size_t newsize){
		if(newsize > tmp_storage_allocation_size) {
			cudaFree(d_temp_storage); CUERR;
			cudaMalloc(&d_temp_storage, std::size_t(newsize * allocfactor)); CUERR;
			tmp_storage_allocation_size = std::size_t(newsize * allocfactor);
		}

		tmp_storage_usable_size = newsize;
	}

	void zero_cpu(){
		std::memset(msa_data_host, 0, msa_data_usable_size);
		std::memset(correction_results_transfer_data_host, 0, correction_results_transfer_data_usable_size);
		std::memset(qualities_transfer_data_host, 0, qualities_transfer_data_usable_size);
		std::memset(indices_transfer_data_host, 0, indices_transfer_data_usable_size);
		//std::fill((int*)indices_transfer_data_host, (int*)(((char*)indices_transfer_data_host) + indices_transfer_data_usable_size), -1);
		std::memset(h_num_indices, 0, sizeof(int));
		std::memset(subject_indices_data_host, 0, subject_indices_data_usable_size);
		std::memset(alignment_result_data_host, 0, alignment_result_data_usable_size);
		std::memset(alignment_transfer_data_host, 0, alignment_transfer_data_usable_size);
		//std::memset(h_candidate_read_ids, 0, candidate_ids_usable_size);
	}

	void zero_gpu(cudaStream_t stream){
		cudaMemsetAsync(msa_data_device, 0, msa_data_usable_size, stream); CUERR;
		cudaMemsetAsync(correction_results_transfer_data_device, 0, correction_results_transfer_data_usable_size, stream); CUERR;
		cudaMemsetAsync(qualities_transfer_data_device, 0, qualities_transfer_data_usable_size, stream); CUERR;
		//cudaMemsetAsync(indices_transfer_data_device, 0, indices_transfer_data_usable_size, stream); CUERR;
		/*thrust::fill(thrust::cuda::par.on(stream),
		            thrust::device_ptr<int>((int*)indices_transfer_data_device),
		            thrust::device_ptr<int>((int*)(((char*)indices_transfer_data_device) + indices_transfer_data_usable_size)),
		            -1);*/
		cudaMemsetAsync(d_num_indices, 0, sizeof(int), stream); CUERR;
		cudaMemsetAsync(subject_indices_data_device, 0, subject_indices_data_usable_size, stream); CUERR;
		cudaMemsetAsync(alignment_result_data_device, 0, alignment_result_data_usable_size, stream); CUERR;
		cudaMemsetAsync(alignment_transfer_data_device, 0, alignment_transfer_data_usable_size, stream); CUERR;
		//cudaMemsetAsync(d_candidate_read_ids, 0, candidate_ids_usable_size); CUERR;
	}

	void fill_d_indices(int val, cudaStream_t stream){
		thrust::fill(thrust::cuda::par.on(stream),
					thrust::device_ptr<int>((int*)indices_transfer_data_device),
					thrust::device_ptr<int>((int*)(((char*)indices_transfer_data_device) + indices_transfer_data_usable_size)),
					val);
	}

	void reset(){
		auto& a = *this;


		cudaFree(a.alignment_transfer_data_device); CUERR;
		cudaFreeHost(a.alignment_transfer_data_host); CUERR;
		cudaFree(a.alignment_result_data_device); CUERR;
		cudaFreeHost(a.alignment_result_data_host); CUERR;
		cudaFree(a.subject_indices_data_device); CUERR;
		cudaFreeHost(a.subject_indices_data_host); CUERR;
		cudaFree(a.d_num_indices); CUERR;
		cudaFreeHost(a.h_num_indices); CUERR;
		cudaFree(a.indices_transfer_data_device); CUERR;
		cudaFreeHost(a.indices_transfer_data_host); CUERR;
		cudaFree(a.qualities_transfer_data_device); CUERR;
		cudaFreeHost(a.qualities_transfer_data_host); CUERR;
		cudaFree(a.correction_results_transfer_data_device); CUERR;
		cudaFreeHost(a.correction_results_transfer_data_host); CUERR;
		cudaFree(a.msa_data_device); CUERR;
		cudaFreeHost(a.msa_data_host); CUERR;
		cudaFree(a.d_temp_storage); CUERR;
		cudaFree(a.d_candidate_read_ids); CUERR;
		cudaFreeHost(a.h_candidate_read_ids); CUERR;

		a.subject_indices_data_device = nullptr;
		a.subject_indices_data_host = nullptr;
		a.h_subject_indices = nullptr;
		a.h_high_quality_subject_indices = nullptr;
		a.h_is_high_quality_subject = nullptr;
		a.h_num_high_quality_subject_indices = nullptr;
		a.d_subject_indices = nullptr;
		a.d_high_quality_subject_indices = nullptr;
		a.d_is_high_quality_subject = nullptr;
		a.d_num_high_quality_subject_indices = nullptr;
		a.alignment_transfer_data_host = nullptr;
		a.alignment_transfer_data_device = nullptr;
		a.h_subject_sequences_data = nullptr;
		a.h_candidate_sequences_data = nullptr;
		a.h_subject_sequences_lengths = nullptr;
		a.h_candidate_sequences_lengths = nullptr;
		a.h_candidates_per_subject_prefixsum = nullptr;
		a.h_subject_read_ids = nullptr;
		a.h_candidate_read_ids = nullptr;
		a.d_subject_sequences_data = nullptr;
		a.d_candidate_sequences_data = nullptr;
		a.d_subject_sequences_lengths = nullptr;
		a.d_candidate_sequences_lengths = nullptr;
		a.d_candidates_per_subject_prefixsum = nullptr;
		a.d_subject_read_ids = nullptr;
		a.d_candidate_read_ids = nullptr;
		a.indices_transfer_data_host = nullptr;
		a.indices_transfer_data_device = nullptr;
		a.h_indices = nullptr;
		a.h_indices_per_subject = nullptr;
		a.h_indices_per_subject_prefixsum = nullptr;
		a.d_indices = nullptr;
		a.d_indices_per_subject = nullptr;
		a.d_indices_per_subject_prefixsum = nullptr;
		a.h_num_indices = nullptr;
		a.d_num_indices = nullptr;
		a.qualities_transfer_data_host = nullptr;
		a.qualities_transfer_data_device = nullptr;
		a.h_candidate_qualities = nullptr;
		a.h_subject_qualities = nullptr;
		a.d_candidate_qualities = nullptr;
		a.d_subject_qualities = nullptr;
		a.correction_results_transfer_data_host = nullptr;
		a.correction_results_transfer_data_device = nullptr;
		a.h_corrected_subjects = nullptr;
		a.h_corrected_candidates = nullptr;
		a.h_num_corrected_candidates = nullptr;
		a.h_subject_is_corrected = nullptr;
		a.h_indices_of_corrected_candidates = nullptr;
		a.d_corrected_subjects = nullptr;
		a.d_corrected_candidates = nullptr;
		a.d_num_corrected_candidates = nullptr;
		a.d_subject_is_corrected = nullptr;
		a.d_indices_of_corrected_candidates = nullptr;
		a.alignment_result_data_host = nullptr;
		a.alignment_result_data_device = nullptr;
		a.h_alignment_scores = nullptr;
		a.h_alignment_overlaps = nullptr;
		a.h_alignment_shifts = nullptr;
		a.h_alignment_nOps = nullptr;
		a.h_alignment_isValid = nullptr;
		a.h_alignment_best_alignment_flags = nullptr;
		a.d_alignment_scores = nullptr;
		a.d_alignment_overlaps = nullptr;
		a.d_alignment_shifts = nullptr;
		a.d_alignment_nOps = nullptr;
		a.d_alignment_isValid = nullptr;
		a.d_alignment_best_alignment_flags = nullptr;
		a.d_temp_storage = nullptr;
		a.msa_data_device = nullptr;
		a.msa_data_host = nullptr;
		a.d_multiple_sequence_alignments = nullptr;
		a.d_multiple_sequence_alignment_weights = nullptr;
		a.d_consensus = nullptr;
		a.d_support = nullptr;
		a.d_coverage = nullptr;
		a.d_origWeights = nullptr;
		a.d_origCoverages = nullptr;
		a.d_msa_column_properties = nullptr;
		a.h_multiple_sequence_alignments = nullptr;
		a.h_multiple_sequence_alignment_weights = nullptr;
		a.h_consensus = nullptr;
		a.h_support = nullptr;
		a.h_coverage = nullptr;
		a.h_origWeights = nullptr;
		a.h_origCoverages = nullptr;
		a.h_msa_column_properties = nullptr;
		a.d_candidate_read_ids = nullptr;
		a.h_candidate_read_ids = nullptr;

		a.n_subjects = 0;
		a.n_queries = 0;
		a.n_indices = 0;
		a.maximum_sequence_length = 0;
		a.min_overlap = 1;
		a.subject_indices_data_allocation_size = 0;
		a.subject_indices_data_usable_size = 0;
		a.alignment_transfer_data_allocation_size = 0;
		a.alignment_transfer_data_usable_size = 0;
		a.encoded_sequence_pitch = 0;
		a.indices_transfer_data_allocation_size = 0;
		a.indices_transfer_data_usable_size = 0;
		a.qualities_transfer_data_allocation_size = 0;
		a.qualities_transfer_data_usable_size = 0;
		a.quality_pitch = 0;
		a.correction_results_transfer_data_allocation_size = 0;
		a.correction_results_transfer_data_usable_size = 0;
		a.sequence_pitch = 0;
		a.alignment_result_data_allocation_size = 0;
		a.alignment_result_data_usable_size = 0;
		a.tmp_storage_allocation_size = 0;
		a.tmp_storage_usable_size = 0;
		a.msa_data_allocation_size = 0;
		a.msa_data_usable_size = 0;
		a.msa_pitch = 0;
		a.msa_weights_pitch = 0;
		a.candidate_ids_allocation_size = 0;
		a.candidate_ids_usable_size = 0;
	}

	int deviceId = -1;

	int n_subjects = 0;
	int n_queries = 0;
	int n_indices = 0;
	int maximum_sequence_length = 0;
	int min_overlap = 1;

	//subject indices

	std::size_t subject_indices_data_allocation_size = 0;
	std::size_t subject_indices_data_usable_size = 0;
	void* subject_indices_data_host = nullptr;
	void* subject_indices_data_device = nullptr;
	int* h_subject_indices = nullptr;
	int* h_high_quality_subject_indices = nullptr;
	bool* h_is_high_quality_subject = nullptr;
	int* h_num_high_quality_subject_indices = nullptr;
	int* d_subject_indices = nullptr;
	int* d_high_quality_subject_indices = nullptr;
	bool* d_is_high_quality_subject = nullptr;
	int* d_num_high_quality_subject_indices = nullptr;

	// alignment input
	std::size_t memSubjects;
	std::size_t memSubjectLengths;
	std::size_t memNqueriesPrefixSum;
	std::size_t memQueries;
	std::size_t memQueryLengths;
	std::size_t memSubjectIds;
	std::size_t memCandidateIds;

	void* alignment_transfer_data_host = nullptr;
	void* alignment_transfer_data_device = nullptr;

	std::size_t alignment_transfer_data_allocation_size = 0;
	std::size_t alignment_transfer_data_usable_size = 0;
	std::size_t candidate_ids_allocation_size = 0;
	std::size_t candidate_ids_usable_size = 0;
	std::size_t encoded_sequence_pitch = 0;

	char* h_subject_sequences_data = nullptr;
	char* h_candidate_sequences_data = nullptr;
	int* h_subject_sequences_lengths = nullptr;
	int* h_candidate_sequences_lengths = nullptr;
	int* h_candidates_per_subject_prefixsum = nullptr;
	ReadId_t* h_subject_read_ids = nullptr;
	ReadId_t* h_candidate_read_ids = nullptr;

	char* d_subject_sequences_data = nullptr;
	char* d_candidate_sequences_data = nullptr;
	int* d_subject_sequences_lengths = nullptr;
	int* d_candidate_sequences_lengths = nullptr;
	int* d_candidates_per_subject_prefixsum = nullptr;
	ReadId_t* d_subject_read_ids = nullptr;
	ReadId_t* d_candidate_read_ids = nullptr;

	//indices
	void* indices_transfer_data_host = nullptr;
	void* indices_transfer_data_device = nullptr;
	std::size_t indices_transfer_data_allocation_size = 0;
	std::size_t indices_transfer_data_usable_size = 0;

	int* h_indices = nullptr;
	int* h_indices_per_subject = nullptr;
	int* h_indices_per_subject_prefixsum = nullptr;

	int* d_indices = nullptr;
	int* d_indices_per_subject = nullptr;
	int* d_indices_per_subject_prefixsum = nullptr;

	int* h_num_indices = nullptr;
	int* d_num_indices = nullptr;

	//qualities input
	void* qualities_transfer_data_host = nullptr;
	void* qualities_transfer_data_device = nullptr;
	std::size_t qualities_transfer_data_allocation_size = 0;
	std::size_t qualities_transfer_data_usable_size = 0;
	std::size_t quality_pitch = 0;

	char* h_candidate_qualities = nullptr;
	char* h_subject_qualities = nullptr;

	char* d_candidate_qualities = nullptr;
	char* d_subject_qualities = nullptr;

	//correction results output

	void* correction_results_transfer_data_host = nullptr;
	void* correction_results_transfer_data_device = nullptr;
	std::size_t correction_results_transfer_data_allocation_size = 0;
	std::size_t correction_results_transfer_data_usable_size = 0;
	std::size_t sequence_pitch = 0;

	char* h_corrected_subjects = nullptr;
	char* h_corrected_candidates = nullptr;
	int* h_num_corrected_candidates = nullptr;
	bool* h_subject_is_corrected = nullptr;
	int* h_indices_of_corrected_candidates = nullptr;

	char* d_corrected_subjects = nullptr;
	char* d_corrected_candidates = nullptr;
	int* d_num_corrected_candidates = nullptr;
	bool* d_subject_is_corrected = nullptr;
	int* d_indices_of_corrected_candidates = nullptr;


	//alignment results
	void* alignment_result_data_host = nullptr;
	void* alignment_result_data_device = nullptr;
	std::size_t alignment_result_data_allocation_size = 0;
	std::size_t alignment_result_data_usable_size = 0;

	int* h_alignment_scores = nullptr;
	int* h_alignment_overlaps = nullptr;
	int* h_alignment_shifts = nullptr;
	int* h_alignment_nOps = nullptr;
	bool* h_alignment_isValid = nullptr;
	BestAlignment_t* h_alignment_best_alignment_flags = nullptr;

	int* d_alignment_scores = nullptr;
	int* d_alignment_overlaps = nullptr;
	int* d_alignment_shifts = nullptr;
	int* d_alignment_nOps = nullptr;
	bool* d_alignment_isValid = nullptr;
	BestAlignment_t* d_alignment_best_alignment_flags = nullptr;

	//tmp storage
	std::size_t tmp_storage_allocation_size = 0;
	std::size_t tmp_storage_usable_size = 0;
	char* d_temp_storage = nullptr;



	// multiple sequence alignment
	void* msa_data_device = nullptr;
	void* msa_data_host = nullptr;
	std::size_t msa_data_allocation_size = 0;
	std::size_t msa_data_usable_size = 0;
	std::size_t msa_pitch = 0;
	std::size_t msa_weights_pitch = 0;

	//need host msa for debuging mostly
	char* h_multiple_sequence_alignments = nullptr;
	float* h_multiple_sequence_alignment_weights = nullptr;
	char* h_consensus = nullptr;
	float* h_support = nullptr;
	int* h_coverage = nullptr;
	float* h_origWeights = nullptr;
	int* h_origCoverages = nullptr;
	MSAColumnProperties* h_msa_column_properties = nullptr;

	char* d_multiple_sequence_alignments = nullptr;
	float* d_multiple_sequence_alignment_weights = nullptr;
	char* d_consensus = nullptr;
	float* d_support = nullptr;
	int* d_coverage = nullptr;
	float* d_origWeights = nullptr;
	int* d_origCoverages = nullptr;
	MSAColumnProperties* d_msa_column_properties = nullptr;

};

    #endif

}
}




#endif
