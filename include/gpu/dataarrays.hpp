#ifndef CARE_GPU_DATA_ARRAYS_HPP
#define CARE_GPU_DATA_ARRAYS_HPP

#include "../hpc_helpers.cuh"
#include "bestalignment.hpp"
#include "msa.hpp"
#include "utility_kernels.cuh"
#include <gpu/thrust_custom_allocators.hpp>

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


namespace detail{

	enum class DataLocation {Host, PinnedHost, Device};

	template<DataLocation location, class T>
	struct ThrustVectorSelection;

	template<class T>
	struct ThrustVectorSelection<DataLocation::Host, T>{
		using Type = thrust::host_vector<T>;
	};

	template<class T>
	struct ThrustVectorSelection<DataLocation::PinnedHost, T>{
		using Type = thrust::host_vector<T, thrust::system::cuda::experimental::pinned_allocator<T>>;
	};

	template<class T>
	struct ThrustVectorSelection<DataLocation::Device, T>{
		using Type = thrust::device_vector<T, ThrustUninitializedDeviceAllocator<T>>;
	};

	template<DataLocation location, class T>
	struct SimpleAllocator;

	template<class T>
	struct SimpleAllocator<DataLocation::Host, T>{
		T* allocate(size_t elements){
			T* ptr{};
			ptr = new T[elements];
			return ptr;
		}

		void deallocate(T* ptr){
			delete [] ptr;
		}
	};

	template<class T>
	struct SimpleAllocator<DataLocation::PinnedHost, T>{
		T* allocate(size_t elements){
			T* ptr{};
			cudaMallocHost(&ptr, elements * sizeof(T)); CUERR;
			return ptr;
		}

		void deallocate(T* ptr){
			cudaFreeHost(ptr); CUERR;
		}
	};

	template<class T>
	struct SimpleAllocator<DataLocation::Device, T>{
		T* allocate(size_t elements){
			T* ptr;
			cudaMalloc(&ptr, elements * sizeof(T)); CUERR;
			return ptr;
		}

		void deallocate(T* ptr){
			cudaFree(ptr); CUERR;
		}
	};

    template<DataLocation location, class T>
    struct SimpleAllocationAccessor;

    template<class T>
    struct SimpleAllocationAccessor<DataLocation::Host, T>{
        static const T& access(const T* array, size_t i){
            return array[i];
        }
        static T& access(T* array, size_t i){
            return array[i];
        }
    };

    template<class T>
    struct SimpleAllocationAccessor<DataLocation::PinnedHost, T>{
        static const T& access(const T* array, size_t i){
            return array[i];
        }
        static T& access(T* array, size_t i){
            return array[i];
        }
    };

    template<class T>
    struct SimpleAllocationAccessor<DataLocation::Device, T>{
        //cannot access device memory on host, return default value
        static T access(const T* array, size_t i){
            return T{};
        }
        static T access(T* array, size_t i){
            return T{};
        }
    };

	template<DataLocation location, class T>
	struct SimpleAllocation{
		using Allocator = SimpleAllocator<location, T>;
        using Accessor = SimpleAllocationAccessor<location, T>;

		T* data_{};
		size_t size_{};
		size_t capacity_{};

		SimpleAllocation() : SimpleAllocation(0){}
		SimpleAllocation(size_t size){
			resize(size);
		}

        SimpleAllocation(const SimpleAllocation&) = delete;
        SimpleAllocation& operator=(const SimpleAllocation&) = delete;

        SimpleAllocation(SimpleAllocation&& rhs) noexcept{
            *this = std::move(rhs);
        }

        SimpleAllocation& operator=(SimpleAllocation&& rhs) noexcept{
            Allocator alloc;
            alloc.deallocate(data_);

            data_ = rhs.data_;
            size_ = rhs.size_;
            capacity_ = rhs.capacity_;

            rhs.data_ = nullptr;
            rhs.size_ = 0;
            rhs.capacity_ = 0;

            return *this;
        }

        ~SimpleAllocation(){
            Allocator alloc;
            alloc.deallocate(data_);
        }

        friend void swap(SimpleAllocation& l, SimpleAllocation& r) noexcept{
    		using std::swap;

    		swap(l.data_, r.data_);
    		swap(l.size_, r.size_);
    		swap(l.capacity_, r.capacity_);
    	}

        T& operator[](size_t i){
            return Accessor::access(get(), i);
        }

        const T& operator[](size_t i) const{
            return Accessor::access(get(), i);
        }

        T& at(size_t i){
            if(i < size()){
                return Accessor::access(get(), i);
            }else{
                throw std::out_of_range("SimpleAllocation::at out-of-bounds access.");
            }
        }

        const T& at(size_t i) const{
            if(i < size()){
                return Accessor::access(get(), i);
            }else{
                throw std::out_of_range("SimpleAllocation::at out-of-bounds access.");
            }
        }

        T* operator+(size_t i) const{
            return get() + i;
        }

        operator T*(){
            return get();
        }

        operator const T*() const{
            return get();
        }


        //size is number of elements of type T
		void resize(size_t newsize){
			if(capacity_ < newsize){
				Allocator alloc;
				alloc.deallocate(data_);
				data_ = alloc.allocate(newsize);
				capacity_ = newsize;
			}
			size_ = newsize;
		}

		T* get() const{
			return data_;
		}

		size_t size() const{
			return size_;
		}

		size_t& sizeRef(){
			return size_;
		}

        size_t sizeInBytes() const{
            return size_ * sizeof(T);
        }
	};

}

template<class T>
using SimpleAllocationHost = detail::SimpleAllocation<detail::DataLocation::Host, T>;

template<class T>
using SimpleAllocationPinnedHost = detail::SimpleAllocation<detail::DataLocation::PinnedHost, T>;

template<class T>
using SimpleAllocationDevice = detail::SimpleAllocation<detail::DataLocation::Device, T>;








template<detail::DataLocation location>
struct BatchSequenceQualityData{
    ////using ThrustVectorChar = typename detail::ThrustVectorSelection<location, char>::Type;

    //ThrustVectorChar subject_qualities; // at least n_candidates * encoded_sequence_pitch elements
    //ThrustVectorChar candidate_qualities; // at least n_candidates * encoded_sequence_pitch elements

	detail::SimpleAllocation<location, char> subject_qualities;
	detail::SimpleAllocation<location, char> candidate_qualities;

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
        return subject_qualities.get();
    }

    char* getCandidateQualities() const{
        return candidate_qualities.get();
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

template<detail::DataLocation location>
struct BatchSequenceData{
    ////using ThrustVectorReadNumber = typename detail::ThrustVectorSelection<location, read_number>::Type;
    ////using ThrustVectorChar = typename detail::ThrustVectorSelection<location, char>::Type;
    ////using ThrustVectorInt = typename detail::ThrustVectorSelection<location, int>::Type;

    detail::SimpleAllocation<location, read_number> subject_read_ids;
    detail::SimpleAllocation<location, read_number> candidate_read_ids;
    detail::SimpleAllocation<location, int> subject_sequences_lengths;
    detail::SimpleAllocation<location, int> candidate_sequences_lengths;
    detail::SimpleAllocation<location, char>  subject_sequences_data;
    detail::SimpleAllocation<location, char>  candidate_sequences_data;

    detail::SimpleAllocation<location, int>  candidates_per_subject;
    detail::SimpleAllocation<location, int>  candidates_per_subject_prefixsum;

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
        return subject_read_ids.get();
    }

    read_number* getCandidateReadIds() const{
        return candidate_read_ids.get();
    }

    int* getSubjectSequencesLengths() const{
        return subject_sequences_lengths.get();
    }

    int* getCandidateSequencesLengths() const{
        return candidate_sequences_lengths.get();
    }

    char* getSubjectSequencesData() const{
        return subject_sequences_data.get();
    }

    char* getCandidateSequencesData() const{
        return candidate_sequences_data.get();
    }

    int* getCandidatesPerSubject() const{
        return candidates_per_subject.get();
    }

    int* getCandidatesPerSubjectPrefixSum() const{
        return candidates_per_subject_prefixsum.get();
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

template<detail::DataLocation location>
struct BatchAlignmentResults{
    //using ThrustVectorBool = typename detail::ThrustVectorSelection<location, bool>::Type;
    //using ThrustVectorBestAl = typename detail::ThrustVectorSelection<location, BestAlignment_t>::Type;
    //using ThrustVectorInt = typename detail::ThrustVectorSelection<location, int>::Type;

    detail::SimpleAllocation<location, int> alignment_scores;
    detail::SimpleAllocation<location, int> alignment_overlaps;
    detail::SimpleAllocation<location, int> alignment_shifts;
    detail::SimpleAllocation<location, int> alignment_nOps;
    detail::SimpleAllocation<location, bool> alignment_isValid;
    detail::SimpleAllocation<location, BestAlignment_t> alignment_best_alignment_flags;

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
        return alignment_scores.get();
    }

    int* getAlignmentOverlaps() const{
        return alignment_overlaps.get();
    }

    int* getAlignmentShifts() const{
        return alignment_shifts.get();
    }

    int* getAlignmentNops() const{
        return alignment_nOps.get();
    }

    bool* getValidityFlags() const{
        return alignment_isValid.get();
    }

    BestAlignment_t* getBestAlignmentFlags() const{
        return alignment_best_alignment_flags.get();
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


template<detail::DataLocation location>
struct BatchCorrectionResults{
    //using ThrustVectorChar = typename detail::ThrustVectorSelection<location, char>::Type;
    //using ThrustVectorBool = typename detail::ThrustVectorSelection<location, bool>::Type;
    //using ThrustVectorInt = typename detail::ThrustVectorSelection<location, int>::Type;

    detail::SimpleAllocation<location, char> corrected_subjects;
    detail::SimpleAllocation<location, char> corrected_candidates;
    detail::SimpleAllocation<location, int> num_corrected_candidates_per_subject;
    detail::SimpleAllocation<location, bool> subject_is_corrected;
    detail::SimpleAllocation<location, int> indices_of_corrected_candidates;

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
        return corrected_subjects.get();
    }

    char* getCorrectedCandidates() const{
        return corrected_candidates.get();
    }

    int* getNumCorrectedCandidatesPerSubject() const{
        return num_corrected_candidates_per_subject.get();
    }

    bool* getSubjectIsCorrectedFlags() const{
        return subject_is_corrected.get();
    }

    int* getIndicesOfCorrectedCandidates() const{
        return indices_of_corrected_candidates.get();
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

template<detail::DataLocation location>
struct BatchMSAData{
    //using ThrustVectorChar = typename detail::ThrustVectorSelection<location, char>::Type;
    //using ThrustVectorMSAColumnProperties = typename detail::ThrustVectorSelection<location, MSAColumnProperties>::Type;
    //using ThrustVectorInt = typename detail::ThrustVectorSelection<location, int>::Type;
    //using ThrustVectorFloat = typename detail::ThrustVectorSelection<location, float>::Type;

    static constexpr int padding_bytes = 4;

    detail::SimpleAllocation<location, char> multiple_sequence_alignments;
    detail::SimpleAllocation<location, float> multiple_sequence_alignment_weights;
    detail::SimpleAllocation<location, char> consensus;
    detail::SimpleAllocation<location, float> support;
    detail::SimpleAllocation<location, int> coverage;
    detail::SimpleAllocation<location, float> origWeights;
    detail::SimpleAllocation<location, int> origCoverages;
    detail::SimpleAllocation<location, MSAColumnProperties> msa_column_properties;
    detail::SimpleAllocation<location, int> counts;
    detail::SimpleAllocation<location, float> weights;

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
        return multiple_sequence_alignments.get();
    }

    float* getMSAWeightMatrix() const{
        return multiple_sequence_alignment_weights.get();
    }

    char* getConsensus() const{
        return consensus.get();
    }

    float* getSupport() const{
        return support.get();
    }

    int* getCoverage() const{
        return coverage.get();
    }

    float* getOrigWeights() const{
        return origWeights.get();
    }

    int* getOrigCoverages() const{
        return origCoverages.get();
    }

    MSAColumnProperties* getMSAColumnProperties() const{
        return msa_column_properties.get();
    }

    int* getCounts() const{
        return counts.get();
    }

    float* getWeights() const{
        return weights.get();
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




template<detail::DataLocation location>
struct BatchData{
    BatchSequenceData<location> sequenceData;
    BatchSequenceQualityData<location> qualityData;
    BatchAlignmentResults<location> alignmentResults;
    BatchMSAData<location> msaData;
    BatchCorrectionResults<location> correctionResults;

    detail::SimpleAllocation<location, char> cubTemp;

    std::array<detail::SimpleAllocation<location, char>, 4> tmpStorage;

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

    void destroy(){
        sequenceData = std::move(BatchSequenceData<location>{});
        qualityData = std::move(BatchSequenceQualityData<location>{});
        alignmentResults = std::move(BatchAlignmentResults<location>{});
        msaData = std::move(BatchMSAData<location>{});
        correctionResults = std::move(BatchCorrectionResults<location>{});
        cubTemp = std::move(detail::SimpleAllocation<location, char>{});

        for(auto& s : tmpStorage){
            s = std::move(detail::SimpleAllocation<location, char>{});
        }

        n_subjects = 0;
        n_candidates = 0;
        maximum_sequence_length = 0;
        maximum_sequence_bytes = 0;
    }
};

using BatchDataHost = BatchData<detail::DataLocation::Host>;
using BatchDataPinnedHost = BatchData<detail::DataLocation::PinnedHost>;
using BatchDataDevice = BatchData<detail::DataLocation::Device>;





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
#if 0
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
        h_candidates_per_subject = (int*)(((char*)h_candidate_sequences_lengths) + memQueryLengths);
		h_candidates_per_subject_prefixsum = (int*)(((char*)h_candidates_per_subject) + memNqueriesPrefixSum);
        h_tiles_per_subject_prefixsum = (int*)(((char*)h_candidates_per_subject_prefixsum) + memNqueriesPrefixSum);
		h_subject_read_ids = (read_number*)(((char*)h_tiles_per_subject_prefixsum) + memTilesPrefixSum);
		//h_candidate_read_ids = (read_number*)(((char*)h_subject_read_ids) + memSubjectIds);

		d_subject_sequences_data = (char*)alignment_transfer_data_device;
		d_candidate_sequences_data = (char*)(((char*)d_subject_sequences_data) + memSubjects);
		d_subject_sequences_lengths = (int*)(((char*)d_candidate_sequences_data) + memQueries);
		d_candidate_sequences_lengths = (int*)(((char*)d_subject_sequences_lengths) + memSubjectLengths);
        d_candidates_per_subject = (int*)(((char*)d_candidate_sequences_lengths) + memQueryLengths);
		d_candidates_per_subject_prefixsum = (int*)(((char*)d_candidates_per_subject) + memNqueriesPrefixSum);
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
            h_subject_qualities.resize(n_sub * quality_pitch);
            h_candidate_qualities.resize(n_quer * quality_pitch);

            d_subject_qualities.resize(n_sub * quality_pitch);;
            d_candidate_qualities.resize(n_quer * quality_pitch);
            d_candidate_qualities_tmp.resize(n_quer * quality_pitch);
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


	void set_tmp_storage_size(std::size_t newsize){
		if(newsize > tmp_storage_allocation_size) {
			cudaFree(d_temp_storage); CUERR;
			cudaMalloc(&d_temp_storage, std::size_t(newsize * allocfactor)); CUERR;
			tmp_storage_allocation_size = std::size_t(newsize * allocfactor);
		}

		tmp_storage_usable_size = newsize;
	}

	void zero_gpu(cudaStream_t stream){
        cudaMemsetAsync(d_consensus.get(), 0, d_consensus.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_support.get(), 0, d_support.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_coverage.get(), 0, d_coverage.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_origWeights.get(), 0, d_origWeights.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_origCoverages.get(), 0, d_origCoverages.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_msa_column_properties.get(), 0, d_msa_column_properties.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_counts.get(), 0, d_counts.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_weights.get(), 0, d_weights.sizeInBytes(), stream); CUERR;

		cudaMemsetAsync(correction_results_transfer_data_device, 0, correction_results_transfer_data_usable_size, stream); CUERR;

        cudaMemsetAsync(d_subject_qualities.get(), 0, d_subject_qualities.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_candidate_qualities.get(), 0, d_candidate_qualities.sizeInBytes(), stream); CUERR;
        cudaMemsetAsync(d_candidate_qualities_tmp.get(), 0, d_candidate_qualities_tmp.sizeInBytes(), stream); CUERR;

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
		cudaFree(a.correction_results_transfer_data_device); CUERR;
		cudaFreeHost(a.correction_results_transfer_data_host); CUERR;
		cudaFree(a.d_temp_storage); CUERR;
		cudaFree(a.d_candidate_read_ids); CUERR;
		cudaFreeHost(a.h_candidate_read_ids); CUERR;

        h_subject_qualities = std::move(SimpleAllocationPinnedHost<char>{});
        h_candidate_qualities = std::move(SimpleAllocationPinnedHost<char>{});

        d_subject_qualities = std::move(SimpleAllocationDevice<char>{});
        d_candidate_qualities = std::move(SimpleAllocationDevice<char>{});
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
        a.h_candidates_per_subject = nullptr;
		a.h_candidates_per_subject_prefixsum = nullptr;
        a.h_tiles_per_subject_prefixsum = nullptr;
		a.h_subject_read_ids = nullptr;
		a.h_candidate_read_ids = nullptr;
		a.d_subject_sequences_data = nullptr;
		a.d_candidate_sequences_data = nullptr;
		a.d_subject_sequences_lengths = nullptr;
		a.d_candidate_sequences_lengths = nullptr;
        a.d_candidates_per_subject = nullptr;
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
		a.quality_pitch = 0;
		a.correction_results_transfer_data_allocation_size = 0;
		a.correction_results_transfer_data_usable_size = 0;
		a.sequence_pitch = 0;
		a.alignment_result_data_allocation_size = 0;
		a.alignment_result_data_usable_size = 0;
		a.tmp_storage_allocation_size = 0;
		a.tmp_storage_usable_size = 0;

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
    int* h_candidates_per_subject = nullptr;
	int* h_candidates_per_subject_prefixsum = nullptr;
    int* h_tiles_per_subject_prefixsum = nullptr;
	read_number* h_subject_read_ids = nullptr;
	read_number* h_candidate_read_ids = nullptr;

	char* d_subject_sequences_data = nullptr;
	char* d_candidate_sequences_data = nullptr;
	int* d_subject_sequences_lengths = nullptr;
	int* d_candidate_sequences_lengths = nullptr;
    int* d_candidates_per_subject = nullptr;
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

	std::size_t quality_pitch = 0;

    SimpleAllocationPinnedHost<char> h_subject_qualities;
    SimpleAllocationPinnedHost<char> h_candidate_qualities;

    SimpleAllocationDevice<char> d_subject_qualities;
    SimpleAllocationDevice<char> d_candidate_qualities;
    SimpleAllocationDevice<char> d_candidate_qualities_tmp;

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

	std::size_t msa_pitch = 0;
	std::size_t msa_weights_pitch = 0;

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
