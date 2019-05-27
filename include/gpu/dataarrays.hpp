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

	std::size_t quality_pitch = 0;

    SimpleAllocationPinnedHost<char> h_subject_qualities;
    SimpleAllocationPinnedHost<char> h_candidate_qualities;

    SimpleAllocationDevice<char> d_subject_qualities;
    SimpleAllocationDevice<char> d_candidate_qualities;
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
