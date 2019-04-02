#ifndef CARE_GPU_DATA_ARRAYS_HPP
#define CARE_GPU_DATA_ARRAYS_HPP

#include "../hpc_helpers.cuh"
#include "bestalignment.hpp"
#include "msa.hpp"
#include "utility_kernels.cuh"

#include <config.hpp>

#ifdef __NVCC__

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/async/for_each.h>

#endif

namespace care {
namespace gpu {

#ifdef __NVCC__

enum class DataLocation {Host, Device};

template<DataLocation location, class T>
struct ThrustVectorSelection;

template<class T>
struct ThrustVectorSelection<DataLocation::Host, T>{
    using Type = thrust::host_vector<T>;
};

template<class T>
struct ThrustVectorSelection<DataLocation::Device, T>{
    using Type = thrust::device_vector<T>;
};




template<DataLocation location>
struct BatchSequenceQualityData{
    using ThrustVectorChar = typename ThrustVectorSelection<location, char>::Type;

    ThrustVectorChar subject_qualities; // at least n_candidates * encoded_sequence_pitch elements
    ThrustVectorChar candidate_qualities; // at least n_candidates * encoded_sequence_pitch elements

    int n_subjects;
    int n_candidates;
    int quality_pitch;

    BatchSequenceQualityData() : BatchSequenceQualityData(0,0,0){}

    BatchSequenceQualityData(int n_sub, int n_cand, int qual_pitch){
        resize(n_sub, n_cand, qual_pitch);
    }

    void resize(int n_sub, int n_cand, int qual_pitch){
        n_subjects = n_sub;
        n_candidates = n_cand;
        quality_pitch = qual_pitch;

        subject_qualities.resize(n_subjects * qual_pitch);
        candidate_qualities.resize(n_candidates * qual_pitch);
    }

    char* getSubjectQualities() const{
        return thrust::raw_pointer_cast(subject_qualities.data());
    }

    char* getCandidateQualities() const{
        return thrust::raw_pointer_cast(candidate_qualities.data());
    }

    size_t getSubjectQualitiesSize() const{
        return subject_qualities.size();
    }

    size_t getCandidateQualitiesSize() const{
        return candidate_qualities.size();
    }

    int getQualityPitch() const{
        return quality_pitch;
    }
};

template<DataLocation location>
struct BatchSequenceData{
    using ThrustVectorReadNumber = typename ThrustVectorSelection<location, read_number>::Type;
    using ThrustVectorChar = typename ThrustVectorSelection<location, char>::Type;
    using ThrustVectorInt = typename ThrustVectorSelection<location, int>::Type;

    ThrustVectorReadNumber subject_read_ids;
    ThrustVectorReadNumber candidate_read_ids;
    ThrustVectorInt subject_sequences_lengths;
    ThrustVectorInt candidate_sequences_lengths;
    ThrustVectorChar subject_sequences_data;
    ThrustVectorChar candidate_sequences_data;

    ThrustVectorInt candidates_per_subject;
    ThrustVectorInt candidates_per_subject_prefixsum;

    int n_subjects;
    int n_candidates;

    int encoded_sequence_pitch;

    BatchSequenceData() : BatchSequenceData(0,0,0){}
    BatchSequenceData(int n_subjects, int n_candidates, int encoded_sequence_pitch){
        resize(n_subjects, n_candidates, encoded_sequence_pitch);
    }

    void resize(int n_sub, int n_cand, int encoded_seq_pitch){
        n_subjects = n_sub;
        n_candidates = n_cand;
        encoded_sequence_pitch = encoded_seq_pitch;

        subject_read_ids.resize(n_subjects);
        candidate_read_ids.resize(n_candidates);
        subject_sequences_lengths.resize(n_subjects);
        candidate_sequences_lengths.resize(n_candidates);
        subject_sequences_data.resize(n_subjects * encoded_sequence_pitch);
        candidate_sequences_data.resize(n_candidates * encoded_sequence_pitch);
        candidates_per_subject.resize(n_subjects);
        candidates_per_subject_prefixsum.resize(n_subjects+1);
    }

    read_number* getSubjectReadIds() const{
        return thrust::raw_pointer_cast(subject_read_ids.data());
    }

    read_number* getCandidateReadIds() const{
        return thrust::raw_pointer_cast(candidate_read_ids.data());
    }

    int* getSubjectSequencesLengths() const{
        return thrust::raw_pointer_cast(subject_sequences_lengths.data());
    }

    int* getCandidateSequencesLengths() const{
        return thrust::raw_pointer_cast(candidate_sequences_lengths.data());
    }

    char* getSubjectSequencesData() const{
        return thrust::raw_pointer_cast(subject_sequences_data.data());
    }

    char* getCandidateSequencesData() const{
        return thrust::raw_pointer_cast(candidate_sequences_data.data());
    }

    int* getCandidatesPerSubject() const{
        return thrust::raw_pointer_cast(candidates_per_subject.data());
    }

    int* getCandidatesPerSubjectPrefixSum() const{
        return thrust::raw_pointer_cast(candidates_per_subject_prefixsum.data());
    }

    size_t getSubjectReadIdsSize() const{
        return subject_read_ids.size();
    }

    size_t getCandidateReadIdsSize() const{
        return candidate_read_ids.size();
    }

    size_t getSubjectSequencesLengthsSize() const{
        return subject_sequences_lengths.size();
    }

    size_t getCandidateSequencesLengthsSize() const{
        return candidate_sequences_lengths.size();
    }

    size_t getSubjectSequencesDataSize() const{
        return subject_sequences_data.size();
    }

    size_t getCandidateSequencesDataSize() const{
        return candidate_sequences_data.size();
    }

    size_t getCandidatesPerSubjectSize() const{
        return candidates_per_subject.size();
    }

    size_t getCandidatesPerSubjectPrefixSumSize() const{
        return candidates_per_subject_prefixsum.size();
    }
};

template<DataLocation location>
struct BatchAlignmentResults{
    using ThrustVectorBool = typename ThrustVectorSelection<location, bool>::Type;
    using ThrustVectorBestAl = typename ThrustVectorSelection<location, BestAlignment_t>::Type;
    using ThrustVectorInt = typename ThrustVectorSelection<location, int>::Type;

    ThrustVectorInt alignment_scores;
    ThrustVectorInt alignment_overlaps;
    ThrustVectorInt alignment_shifts;
    ThrustVectorInt alignment_nOps;
    ThrustVectorBool alignment_isValid;
    ThrustVectorBestAl alignment_best_alignment_flags;

    int n_alignments;

    BatchAlignmentResults() : BatchAlignmentResults(0){}
    BatchAlignmentResults(int n_alignments){
        resize(n_alignments);
    }

    void resize(int num_alignments){
        n_alignments = num_alignments;

        alignment_scores.resize(n_alignments);
        alignment_overlaps.resize(n_alignments);
        alignment_shifts.resize(n_alignments);
        alignment_nOps.resize(n_alignments);
        alignment_isValid.resize(n_alignments);
        alignment_best_alignment_flags.resize(n_alignments);
    }

    int* getAlignmentScores() const{
        return thrust::raw_pointer_cast(alignment_scores.data());
    }

    int* getAlignmentOverlaps() const{
        return thrust::raw_pointer_cast(alignment_overlaps.data());
    }

    int* getAlignmentShifts() const{
        return thrust::raw_pointer_cast(alignment_shifts.data());
    }

    int* getAlignmentNops() const{
        return thrust::raw_pointer_cast(alignment_nOps.data());
    }

    bool* getValidityFlags() const{
        return thrust::raw_pointer_cast(alignment_isValid.data());
    }

    BestAlignment_t* getBestAlignmentFlags() const{
        return thrust::raw_pointer_cast(alignment_best_alignment_flags.data());
    }

    size_t getAlignmentScoresSize() const{
        return alignment_scores.size();
    }

    size_t getAlignmentOverlapsSize() const{
        return alignment_overlaps.size();
    }

    size_t getAlignmentShiftsSize() const{
        return alignment_shifts.size();
    }

    size_t getAlignmentNopsSize() const{
        return alignment_nOps.size();
    }

    size_t getValidityFlagsSize() const{
        return alignment_isValid.size();
    }

    size_t getBestAlignmentFlagsSize() const{
        return alignment_best_alignment_flags.size();
    }
};


template<DataLocation location>
struct BatchCorrectionResults{
    using ThrustVectorChar = typename ThrustVectorSelection<location, char>::Type;
    using ThrustVectorBool = typename ThrustVectorSelection<location, bool>::Type;
    using ThrustVectorInt = typename ThrustVectorSelection<location, int>::Type;

    ThrustVectorChar corrected_subjects;
    ThrustVectorChar corrected_candidates;
    ThrustVectorInt num_corrected_candidates_per_subject;
    ThrustVectorBool subject_is_corrected;
    ThrustVectorInt indices_of_corrected_candidates;

    int n_subjects;
    int n_candidates;
    int encoded_sequence_pitch;

    BatchCorrectionResults() : BatchCorrectionResults(0,0,0){}
    BatchCorrectionResults(int n_sub, int n_cand, int encoded_seq_pitch){
        resize(n_sub, n_cand, encoded_seq_pitch);
    }

    void resize(int n_sub, int n_cand, int encoded_seq_pitch){
        n_subjects = n_sub;
        n_candidates = n_cand;
        encoded_sequence_pitch = encoded_seq_pitch;

        corrected_subjects.resize(n_subjects * encoded_sequence_pitch);
        corrected_candidates.resize(n_candidates * encoded_sequence_pitch);
        num_corrected_candidates_per_subject.resize(n_subjects);
        subject_is_corrected.resize(n_subjects);
        indices_of_corrected_candidates.resize(n_candidates);
    }

    char* getCorrectedSubjects() const{
        return thrust::raw_pointer_cast(corrected_subjects.data());
    }

    char* getCorrectedCandidates() const{
        return thrust::raw_pointer_cast(corrected_candidates.data());
    }

    int* getNumCorrectedCandidatesPerSubject() const{
        return thrust::raw_pointer_cast(num_corrected_candidates_per_subject.data());
    }

    bool* getSubjectIsCorrectedFlags() const{
        return thrust::raw_pointer_cast(subject_is_corrected.data());
    }

    int* getIndicesOfCorrectedCandidates() const{
        return thrust::raw_pointer_cast(indices_of_corrected_candidates.data());
    }

    size_t getCorrectedSubjectsSize() const{
        return corrected_subjects.size();
    }

    size_t getCorrectedCandidatesSize() const{
        return corrected_candidates.size();
    }

    size_t getNumCorrectedCandidatesPerSubjectSize() const{
        return num_corrected_candidates_per_subject.size();
    }

    size_t getSubjectIsCorrectedFlagsSize() const{
        return subject_is_corrected.size();
    }

    size_t getIndicesOfCorrectedCandidatesSize() const{
        return indices_of_corrected_candidates.size();
    }
};

template<DataLocation location>
struct BatchMSAData{
    using ThrustVectorChar = typename ThrustVectorSelection<location, char>::Type;
    using ThrustVectorMSAColumnProperties = typename ThrustVectorSelection<location, MSAColumnProperties>::Type;
    using ThrustVectorInt = typename ThrustVectorSelection<location, int>::Type;
    using ThrustVectorFloat = typename ThrustVectorSelection<location, float>::Type;

    static constexpr int padding_bytes = 4;

    ThrustVectorChar multiple_sequence_alignments;
    ThrustVectorFloat multiple_sequence_alignment_weights;
    ThrustVectorChar consensus;
    ThrustVectorFloat support;
    ThrustVectorInt coverage;
    ThrustVectorFloat origWeights;
    ThrustVectorInt origCoverages;
    ThrustVectorMSAColumnProperties msa_column_properties;
    ThrustVectorInt counts;
    ThrustVectorFloat weights;

    int n_subjects;
    int n_candidates;
    int max_msa_columns;
    int row_pitch_char;
    int row_pitch_int;
    int row_pitch_float;



    BatchMSAData() : BatchMSAData(0,0,0){}
    BatchMSAData(int n_sub, int n_cand, int max_msa_cols){
        resize(n_sub, n_cand, max_msa_cols);
    }

    void resize(int n_sub, int n_cand, int max_msa_cols){
        n_subjects = n_sub;
        n_candidates = n_cand;
        max_msa_columns = max_msa_cols;
        row_pitch_char = SDIV(sizeof(char)*max_msa_columns, padding_bytes) * padding_bytes;
        row_pitch_int = SDIV(sizeof(int)*max_msa_columns, padding_bytes) * padding_bytes;
        row_pitch_float = SDIV(sizeof(float)*max_msa_columns, padding_bytes) * padding_bytes;

        multiple_sequence_alignments.resize((n_subjects + n_candidates) * row_pitch_char);
        multiple_sequence_alignment_weights.resize((n_subjects + n_candidates) * row_pitch_float);
        consensus.resize(n_subjects * row_pitch_char);
        support.resize(n_subjects * row_pitch_float);
        coverage.resize(n_subjects * row_pitch_int);
        origWeights.resize(n_subjects * row_pitch_float);
        origCoverages.resize(n_subjects * row_pitch_int);
        msa_column_properties.resize(n_subjects);
        counts.resize(n_subjects * 4 * row_pitch_int);
        weights.resize(n_subjects * 4 * row_pitch_float);
    }

    char* getMSASequenceMatrix() const{
        return thrust::raw_pointer_cast(multiple_sequence_alignments.data());
    }

    float* getMSAWeightMatrix() const{
        return thrust::raw_pointer_cast(multiple_sequence_alignment_weights.data());
    }

    char* getConsensus() const{
        return thrust::raw_pointer_cast(consensus.data());
    }

    float* getSupport() const{
        return thrust::raw_pointer_cast(support.data());
    }

    int* getCoverage() const{
        return thrust::raw_pointer_cast(coverage.data());
    }

    float* getOrigWeights() const{
        return thrust::raw_pointer_cast(origWeights.data());
    }

    int* getOrigCoverages() const{
        return thrust::raw_pointer_cast(origCoverages.data());
    }

    MSAColumnProperties* getMSAColumnProperties() const{
        return thrust::raw_pointer_cast(msa_column_properties.data());
    }

    int* getCounts() const{
        return thrust::raw_pointer_cast(counts.data());
    }

    float* getWeights() const{
        return thrust::raw_pointer_cast(weights.data());
    }

    size_t getMSASequenceMatrixSize() const{
        return multiple_sequence_alignments.size();
    }

    size_t getMSAWeightMatrixSize() const{
        return multiple_sequence_alignment_weights.size();
    }

    size_t getConsensusSize() const{
        return consensus.size();
    }

    size_t getSupportSize() const{
        return support.size();
    }

    size_t getCoverageSize() const{
        return coverage.size();
    }

    size_t getOrigWeightsSize() const{
        return origWeights.size();
    }

    size_t getOrigCoveragesSize() const{
        return origCoverages.size();
    }

    size_t getMSAColumnPropertiesSize() const{
        return msa_column_properties.size();
    }

    size_t getCountsSize() const{
        return counts.size();
    }

    size_t getWeightsSize() const{
        return weights.size();
    }

    int getRowPitchChar() const{
        return row_pitch_char;
    }

    int getRowPitchInt() const{
        return row_pitch_int;
    }

    int getRowPitchFloat() const{
        return row_pitch_float;
    }
};

template<DataLocation location>
struct BatchTmpData{
    using ThrustVectorChar = typename ThrustVectorSelection<location, char>::Type;

    ThrustVectorChar data;

    BatchTmpData() : BatchTmpData(0){}
    BatchTmpData(size_t bytes){
        resize(bytes);
    }

    void resize(size_t bytes){
        data.resize(bytes);
    }

    template<class T>
    T* get() const{
        return static_cast<T*>(thrust::raw_pointer_cast(data.data()));
    }

    void getSize() const{
        return data.size();
    }


};

template<DataLocation location>
struct BatchData{
    BatchSequenceData<location> sequenceData;
    BatchSequenceQualityData<location> qualityData;
    BatchAlignmentResults<location> alignmentResults;
    BatchMSAData<location> msaData;
    BatchCorrectionResults<location> correctionResults;

    BatchTmpData<location> cubTemp;

    std::array<BatchTmpData<location>, 4> tmpStorage;

    int n_subjects;
    int n_candidates;
    int maximum_sequence_length;
    int maximum_sequence_bytes;


    BatchData() : BatchData(0,0,0,0,0,0,0){}

    BatchData(int n_sub, int n_cand, int encoded_seq_pitch, int qual_pitch, int max_seq_length, int max_seq_bytes, int max_msa_cols){
        resize(n_sub, n_cand, encoded_seq_pitch, qual_pitch, max_seq_length, max_seq_bytes, max_msa_cols);
    }

    void resize(int n_sub, int n_cand, int encoded_seq_pitch, int qual_pitch, int max_seq_length, int max_seq_bytes, int max_msa_cols){
        n_subjects = n_sub;
        n_candidates = n_cand;
        maximum_sequence_length = max_seq_length;
        maximum_sequence_bytes = max_seq_bytes;

        sequenceData.resize(n_sub, n_cand, encoded_seq_pitch);
        qualityData.resize(n_sub, n_cand, qual_pitch);
        alignmentResults.resize(n_cand * 2);
        msaData.resize(n_sub, n_cand, max_msa_cols);
        correctionResults.resize(n_sub, n_cand, encoded_seq_pitch);

        for(auto& x : tmpStorage){
            x.resize(n_cand);
        }
    }
};

struct DataArrays {
	static constexpr int padding_bytes = 4;
	static constexpr float allocfactor = 1.1;

	DataArrays() : DataArrays(0){
	}

	DataArrays(int deviceId) : deviceId(deviceId){
		//cudaSetDevice(deviceId);
	};

	void allocCandidateIds(int n_quer){
		memCandidateIds = SDIV(sizeof(read_number) * n_quer, padding_bytes) * padding_bytes;

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

	void set_problem_dimensions(int n_sub, int n_quer, int max_seq_length, int max_seq_bytes, int min_overlap_, float min_overlap_ratio_, bool useQualityScores){
        n_subjects = n_sub;
		n_queries = n_quer;
		maximum_sequence_length = max_seq_length;
        maximum_sequence_bytes = max_seq_bytes;
		min_overlap = std::max(1, std::max(min_overlap_, int(maximum_sequence_length * min_overlap_ratio_)));

		encoded_sequence_pitch = SDIV(maximum_sequence_bytes, padding_bytes) * padding_bytes;
		quality_pitch = SDIV(max_seq_length * sizeof(char), padding_bytes) * padding_bytes;
		sequence_pitch = SDIV(max_seq_length * sizeof(char), padding_bytes) * padding_bytes;
		int msa_max_column_count = (3*max_seq_length - 2*min_overlap_);
		msa_pitch = SDIV(sizeof(char)*msa_max_column_count, padding_bytes) * padding_bytes;
		msa_weights_pitch = SDIV(sizeof(float)*msa_max_column_count, padding_bytes) * padding_bytes;

		//alignment input
		memSubjects = n_sub * encoded_sequence_pitch;
		memSubjectLengths = SDIV(n_sub * sizeof(int), padding_bytes) * padding_bytes;
		memNqueriesPrefixSum = SDIV((n_sub+1) * sizeof(int), padding_bytes) * padding_bytes;
        memTilesPrefixSum = SDIV((n_sub+1) * sizeof(int), padding_bytes) * padding_bytes;
		memQueries = n_quer * encoded_sequence_pitch;
		memQueryLengths = SDIV(n_quer * sizeof(int), padding_bytes) * padding_bytes;
		memSubjectIds = SDIV(sizeof(read_number) * n_sub, padding_bytes) * padding_bytes;
		//memCandidateIds = SDIV(sizeof(read_number) * n_quer, padding_bytes) * padding_bytes;

		std::size_t required_alignment_transfer_data_allocation_size = memSubjects
		                                                               + memSubjectLengths
		                                                               + memNqueriesPrefixSum
                                                                       + memTilesPrefixSum
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
        h_tiles_per_subject_prefixsum = (int*)(((char*)h_candidates_per_subject_prefixsum) + memNqueriesPrefixSum);
		h_subject_read_ids = (read_number*)(((char*)h_tiles_per_subject_prefixsum) + memTilesPrefixSum);
		//h_candidate_read_ids = (read_number*)(((char*)h_subject_read_ids) + memSubjectIds);

		d_subject_sequences_data = (char*)alignment_transfer_data_device;
		d_candidate_sequences_data = (char*)(((char*)d_subject_sequences_data) + memSubjects);
		d_subject_sequences_lengths = (int*)(((char*)d_candidate_sequences_data) + memQueries);
		d_candidate_sequences_lengths = (int*)(((char*)d_subject_sequences_lengths) + memSubjectLengths);
		d_candidates_per_subject_prefixsum = (int*)(((char*)d_candidate_sequences_lengths) + memQueryLengths);
        d_tiles_per_subject_prefixsum = (int*)(((char*)d_candidates_per_subject_prefixsum) + memNqueriesPrefixSum);
		d_subject_read_ids = (read_number*)(((char*)d_tiles_per_subject_prefixsum) + memTilesPrefixSum);
		//d_candidate_read_ids = (read_number*)(((char*)d_subject_read_ids) + memSubjectIds);

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
		//std::size_t memIsHighQualityMSA = SDIV(n_sub *  sizeof(bool), padding_bytes) * padding_bytes;
        std::size_t memCounts = n_sub * msa_weights_pitch;
        std::size_t memWeights = n_sub * msa_weights_pitch;
        std::size_t memAllCounts = n_sub * msa_weights_pitch * 4;
        std::size_t memAllWeights = n_sub * msa_weights_pitch * 4;
		std::size_t required_msa_data_allocation_size = memMultipleSequenceAlignment
		                                                + memMultipleSequenceAlignmentWeights
		                                                + memConsensus
		                                                + memSupport
		                                                + memCoverage
		                                                + memOrigWeights
		                                                + memOrigCoverage
		                                                + memMSAColumnProperties
                                                        + 4 * memCounts
                                                        + 4 * memWeights
                                                        + memAllCounts
                                                        + memAllWeights;
		                                                //+ memIsHighQualityMSA;

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
        h_counts = (int*)(((char*)h_msa_column_properties) + memMSAColumnProperties);
        h_weights = (float*)(((char*)h_counts) + memAllCounts);

		d_multiple_sequence_alignments = (char*)msa_data_device;
		d_multiple_sequence_alignment_weights = (float*)(((char*)d_multiple_sequence_alignments) + memMultipleSequenceAlignment);
		d_consensus = (char*)(((char*)d_multiple_sequence_alignment_weights) + memMultipleSequenceAlignmentWeights);
		d_support = (float*)(((char*)d_consensus) + memConsensus);
		d_coverage = (int*)(((char*)d_support) + memSupport);
		d_origWeights = (float*)(((char*)d_coverage) + memCoverage);
		d_origCoverages = (int*)(((char*)d_origWeights) + memOrigWeights);
		d_msa_column_properties = (MSAColumnProperties*)(((char*)d_origCoverages) + memOrigCoverage);
        d_counts = (int*)(((char*)d_msa_column_properties) + memMSAColumnProperties);
        d_weights = (float*)(((char*)d_counts) + memAllCounts);



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

        cudaMemsetAsync(d_multiple_sequence_alignments, char(0xFC), (n_subjects + n_queries) * msa_pitch, stream);

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
		/*thrust::fill(thrust::cuda::par.on(stream),
					thrust::device_ptr<int>((int*)indices_transfer_data_device),
					thrust::device_ptr<int>((int*)(((char*)indices_transfer_data_device) + indices_transfer_data_usable_size)),
					val);*/

        /*thrust::async::for_each(thrust::cuda::par.on(stream),
					thrust::device_ptr<int>((int*)indices_transfer_data_device),
					thrust::device_ptr<int>((int*)(((char*)indices_transfer_data_device) + indices_transfer_data_usable_size)),
					[val] __device__ (int& i){i = val;});*/
        assert(indices_transfer_data_usable_size % sizeof(int) == 0);

        const int elements = indices_transfer_data_usable_size / sizeof(int);
        call_fill_kernel_async((int*)indices_transfer_data_device, elements, val, stream);
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
        a.h_tiles_per_subject_prefixsum = nullptr;
		a.h_subject_read_ids = nullptr;
		a.h_candidate_read_ids = nullptr;
		a.d_subject_sequences_data = nullptr;
		a.d_candidate_sequences_data = nullptr;
		a.d_subject_sequences_lengths = nullptr;
		a.d_candidate_sequences_lengths = nullptr;
		a.d_candidates_per_subject_prefixsum = nullptr;
        a.d_tiles_per_subject_prefixsum = nullptr;
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
        a.h_counts = nullptr;
        a.h_weights = nullptr;
        a.d_counts = nullptr;
        a.d_weights = nullptr;

		a.n_subjects = 0;
		a.n_queries = 0;
		a.n_indices = 0;
		a.maximum_sequence_length = 0;
        a.maximum_sequence_bytes = 0;
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
    int maximum_sequence_bytes = 0;
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
    std::size_t memTilesPrefixSum;
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
    int* h_tiles_per_subject_prefixsum = nullptr;
	read_number* h_subject_read_ids = nullptr;
	read_number* h_candidate_read_ids = nullptr;

	char* d_subject_sequences_data = nullptr;
	char* d_candidate_sequences_data = nullptr;
	int* d_subject_sequences_lengths = nullptr;
	int* d_candidate_sequences_lengths = nullptr;
	int* d_candidates_per_subject_prefixsum = nullptr;
    int* d_tiles_per_subject_prefixsum = nullptr;
	read_number* d_subject_read_ids = nullptr;
	read_number* d_candidate_read_ids = nullptr;

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
    int* h_counts = nullptr;
    float* h_weights = nullptr;

	char* d_multiple_sequence_alignments = nullptr;
	float* d_multiple_sequence_alignment_weights = nullptr;
	char* d_consensus = nullptr;
	float* d_support = nullptr;
	int* d_coverage = nullptr;
	float* d_origWeights = nullptr;
	int* d_origCoverages = nullptr;
	MSAColumnProperties* d_msa_column_properties = nullptr;
    int* d_counts = nullptr;
    float* d_weights = nullptr;

};

    #endif

}
}




#endif
