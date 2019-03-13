#ifndef CARE_GPU_KERNEL_SELECTION_HPP
#define CARE_GPU_KERNEL_SELECTION_HPP

#include "kernels.hpp"
#include <cassert>

namespace care {
namespace gpu {


    #ifdef __NVCC__

template<class Sequence_t, class ReadId_t>
struct ShiftedHammingDistanceChooserExp;

template<class ReadId_t>
struct ShiftedHammingDistanceChooserExp<Sequence2BitHiLo, ReadId_t> {

	template<class GPUReadStorage_t>
	static void callKernelAsync(int* d_alignment_scores,
				int* d_alignment_overlaps,
				int* d_alignment_shifts,
				int* d_alignment_nOps,
				bool* d_alignment_isValid,
				const ReadId_t* d_subject_read_ids,
				const ReadId_t* d_candidate_read_ids,
				const char* d_subject_sequences_data,
				const char* d_candidate_sequences_data,
				const int* d_subject_sequences_lengths,
				const int* d_candidate_sequences_lengths,
				const int* d_candidates_per_subject_prefixsum,
				int n_subjects,
				int n_queries,
				int max_sequence_bytes,
				size_t encodedsequencepitch,
				int min_overlap,
				float maxErrorRate,
				float min_overlap_ratio,
				const GPUReadStorage_t* gpuReadStorage,
				const typename GPUReadStorage_t::GPUData& gpuReadStorageGpuData,
				bool useGpuReadStorage,
				cudaStream_t stream,
				KernelLaunchHandle& kernelLaunchHandle){

		//the kernel expects length to be an int
		static_assert(std::numeric_limits<typename GPUReadStorage_t::Length_t>::max() <= std::numeric_limits<int>::max());

		assert(!useGpuReadStorage || (useGpuReadStorage && gpuReadStorage != nullptr));
		//assert(!useGpuReadStorage || (useGpuReadStorage && gpuReadStorage->max_sequence_bytes == max_sequence_bytes));

		const char* d_sequence_data = gpuReadStorageGpuData.d_sequence_data;
		const typename GPUReadStorage_t::Length_t* d_sequence_lengths = gpuReadStorageGpuData.d_sequence_lengths;
		const std::size_t readstorage_sequence_pitch = std::size_t(gpuReadStorage->getSequencePitch());

		auto getNumBytes = [] __device__ (int sequencelength){
			return Sequence2BitHiLo::getNumBytes(sequencelength);
		};

		auto getSubjectPtr_sparse = [=] __device__ (ReadId_t subjectIndex){
			const ReadId_t subjectReadId = d_subject_read_ids[subjectIndex];
			const char* result = d_sequence_data + std::size_t(subjectReadId) * readstorage_sequence_pitch;
			return result;
		};

		auto getCandidatePtr_sparse = [=] __device__ (ReadId_t candidateIndex){
			const ReadId_t candidateReadId = d_candidate_read_ids[candidateIndex];
			const char* result = d_sequence_data + std::size_t(candidateReadId) * readstorage_sequence_pitch;
			return result;
		};

		auto getSubjectLength_sparse = [=] __device__ (ReadId_t subjectIndex){
			const ReadId_t subjectReadId = d_subject_read_ids[subjectIndex];
			const int length = d_sequence_lengths[subjectReadId];
			return length;
		};

		auto getCandidateLength_sparse = [=] __device__ (ReadId_t candidateIndex){
			const ReadId_t candidateReadId = d_candidate_read_ids[candidateIndex];
			const int length = d_sequence_lengths[candidateReadId];
			return length;
		};

		auto getSubjectPtr_dense = [=] __device__ (ReadId_t subjectIndex){
			const char* result = d_subject_sequences_data + std::size_t(subjectIndex) * encodedsequencepitch;
			return result;
		};

		auto getCandidatePtr_dense = [=] __device__ (ReadId_t candidateIndex){
			const char* result = d_candidate_sequences_data + std::size_t(candidateIndex) * encodedsequencepitch;
			return result;
		};

		auto getSubjectLength_dense = [=] __device__ (ReadId_t subjectIndex){
			const int length = d_subject_sequences_lengths[subjectIndex];
			return length;
		};

		auto getCandidateLength_dense = [=] __device__ (ReadId_t candidateIndex){
			const int length = d_candidate_sequences_lengths[candidateIndex];
			return length;
		};

		auto callKernel = [&](auto subjectpointer, auto candidatepointer, auto subjectlength, auto candidatelength){
                      call_cuda_popcount_shifted_hamming_distance_with_revcompl_kernel_async(
								  d_alignment_scores,
								  d_alignment_overlaps,
								  d_alignment_shifts,
								  d_alignment_nOps,
								  d_alignment_isValid,
								  d_candidates_per_subject_prefixsum,
								  n_subjects,
								  n_queries,
								  max_sequence_bytes,
								  min_overlap,
								  maxErrorRate,
								  min_overlap_ratio,
								  getNumBytes,
								  subjectpointer,
								  candidatepointer,
								  subjectlength,
								  candidatelength,
								  stream,
								  kernelLaunchHandle);
				  };

		if(!useGpuReadStorage || !gpuReadStorageGpuData.isValidSequenceData()) {
			callKernel( getSubjectPtr_dense,
						getCandidatePtr_dense,
						getSubjectLength_dense,
						getCandidateLength_dense);
		}else{

			callKernel( getSubjectPtr_sparse,
						getCandidatePtr_sparse,
						getSubjectLength_sparse,
						getCandidateLength_sparse);

		}
	}
};










template<class Sequence_t, class ReadId_t>
struct ShiftedHammingDistanceTiledChooser;

template<class ReadId_t>
struct ShiftedHammingDistanceTiledChooser<Sequence2BitHiLo, ReadId_t> {

	template<class GPUReadStorage_t>
	static void callKernelAsync(int* d_alignment_scores,
				int* d_alignment_overlaps,
				int* d_alignment_shifts,
				int* d_alignment_nOps,
				bool* d_alignment_isValid,
				const ReadId_t* d_subject_read_ids,
				const ReadId_t* d_candidate_read_ids,
				const char* d_subject_sequences_data,
				const char* d_candidate_sequences_data,
				const int* d_subject_sequences_lengths,
				const int* d_candidate_sequences_lengths,
				const int* d_candidates_per_subject_prefixsum,
                const int* h_tiles_per_subject_prefixsum,
                const int* d_tiles_per_subject_prefixsum,
                int tilesize,
				int n_subjects,
				int n_queries,
				int max_sequence_bytes,
				size_t encodedsequencepitch,
				int min_overlap,
				float maxErrorRate,
				float min_overlap_ratio,
				const GPUReadStorage_t* gpuReadStorage,
				const typename GPUReadStorage_t::GPUData& gpuReadStorageGpuData,
				bool useGpuReadStorage,
				cudaStream_t stream,
				KernelLaunchHandle& kernelLaunchHandle){

		//the kernel expects length to be an int
		static_assert(std::numeric_limits<typename GPUReadStorage_t::Length_t>::max() <= std::numeric_limits<int>::max());

		assert(!useGpuReadStorage || (useGpuReadStorage && gpuReadStorage != nullptr));
		//assert(!useGpuReadStorage || (useGpuReadStorage && gpuReadStorage->max_sequence_bytes == max_sequence_bytes));

		const char* d_sequence_data = gpuReadStorageGpuData.d_sequence_data;
		const typename GPUReadStorage_t::Length_t* d_sequence_lengths = gpuReadStorageGpuData.d_sequence_lengths;
		const std::size_t readstorage_sequence_pitch = std::size_t(gpuReadStorage->getSequencePitch());

		auto getNumBytes = [] __device__ (int sequencelength){
			return Sequence2BitHiLo::getNumBytes(sequencelength);
		};

		auto getSubjectPtr_sparse = [=] __device__ (ReadId_t subjectIndex){
			const ReadId_t subjectReadId = d_subject_read_ids[subjectIndex];
			const char* result = d_sequence_data + std::size_t(subjectReadId) * readstorage_sequence_pitch;
			return result;
		};

		auto getCandidatePtr_sparse = [=] __device__ (ReadId_t candidateIndex){
			const ReadId_t candidateReadId = d_candidate_read_ids[candidateIndex];
			const char* result = d_sequence_data + std::size_t(candidateReadId) * readstorage_sequence_pitch;
			return result;
		};

		auto getSubjectLength_sparse = [=] __device__ (ReadId_t subjectIndex){
			const ReadId_t subjectReadId = d_subject_read_ids[subjectIndex];
			const int length = d_sequence_lengths[subjectReadId];
			return length;
		};

		auto getCandidateLength_sparse = [=] __device__ (ReadId_t candidateIndex){
			const ReadId_t candidateReadId = d_candidate_read_ids[candidateIndex];
			const int length = d_sequence_lengths[candidateReadId];
			return length;
		};

		auto getSubjectPtr_dense = [=] __device__ (ReadId_t subjectIndex){
			const char* result = d_subject_sequences_data + std::size_t(subjectIndex) * encodedsequencepitch;
			return result;
		};

		auto getCandidatePtr_dense = [=] __device__ (ReadId_t candidateIndex){
			const char* result = d_candidate_sequences_data + std::size_t(candidateIndex) * encodedsequencepitch;
			return result;
		};

		auto getSubjectLength_dense = [=] __device__ (ReadId_t subjectIndex){
			const int length = d_subject_sequences_lengths[subjectIndex];
			return length;
		};

		auto getCandidateLength_dense = [=] __device__ (ReadId_t candidateIndex){
			const int length = d_candidate_sequences_lengths[candidateIndex];
			return length;
		};

		auto callKernel = [&](auto subjectpointer, auto candidatepointer, auto subjectlength, auto candidatelength){
                      call_cuda_popcount_shifted_hamming_distance_with_revcompl_tiled_kernel_async(
								  d_alignment_scores,
								  d_alignment_overlaps,
								  d_alignment_shifts,
								  d_alignment_nOps,
								  d_alignment_isValid,
								  d_candidates_per_subject_prefixsum,
                                  h_tiles_per_subject_prefixsum,
                                  d_tiles_per_subject_prefixsum,
                                  tilesize,
								  n_subjects,
								  n_queries,
								  max_sequence_bytes,
								  min_overlap,
								  maxErrorRate,
								  min_overlap_ratio,
								  getNumBytes,
								  subjectpointer,
								  candidatepointer,
								  subjectlength,
								  candidatelength,
								  stream,
								  kernelLaunchHandle);
				  };

		if(!useGpuReadStorage || !gpuReadStorageGpuData.isValidSequenceData()) {
			callKernel( getSubjectPtr_dense,
						getCandidatePtr_dense,
						getSubjectLength_dense,
						getCandidateLength_dense);
		}else{

			callKernel( getSubjectPtr_sparse,
						getCandidatePtr_sparse,
						getSubjectLength_sparse,
						getCandidateLength_sparse);

		}
	}
};









template<class ReadId_t>
struct FindBestAlignmentChooserExp {

	template<class GPUReadStorage_t>
	static void callKernelAsync(
				BestAlignment_t* d_alignment_best_alignment_flags,
				int* d_alignment_scores,
				int* d_alignment_overlaps,
				int* d_alignment_shifts,
				int* d_alignment_nOps,
				bool* d_alignment_isValid,
				const int* d_candidates_per_subject_prefixsum,
				float min_overlap_ratio,
				int min_overlap,
				float estimatedErrorrate,
				const ReadId_t* d_subject_read_ids,
				const ReadId_t* d_candidate_read_ids,
				const int* d_subject_sequences_lengths,
				const int* d_candidate_sequences_lengths,
				int n_subjects,
				int n_queries,
				const GPUReadStorage_t* gpuReadStorage,
				const typename GPUReadStorage_t::GPUData& gpuReadStorageGpuData,
				bool useGpuReadStorage,
				cudaStream_t stream,
				KernelLaunchHandle& kernelLaunchHandle){

		assert(!useGpuReadStorage || (useGpuReadStorage && gpuReadStorage != nullptr));

		const typename GPUReadStorage_t::Length_t* d_sequence_lengths = gpuReadStorageGpuData.d_sequence_lengths;

		auto getSubjectLength_sparse = [=] __device__ (ReadId_t subjectIndex){
			const ReadId_t subjectReadId = d_subject_read_ids[subjectIndex];
			const int length = d_sequence_lengths[subjectReadId];
			return length;
		};

		auto getCandidateLength_sparse = [=] __device__ (ReadId_t resultIndex){
			const ReadId_t candidateReadId = d_candidate_read_ids[resultIndex];
			const int length = d_sequence_lengths[candidateReadId];
			return length;
		};

		auto getSubjectLength_dense = [=] __device__ (ReadId_t subjectIndex){
			const int length = d_subject_sequences_lengths[subjectIndex];
			return length;
		};

		auto getCandidateLength_dense = [=] __device__ (ReadId_t resultIndex){
			const int length = d_candidate_sequences_lengths[resultIndex];
			return length;
		};

		auto best_alignment_comp = [=] __device__ (int fwd_alignment_overlap,
					int revc_alignment_overlap,
					int fwd_alignment_nops,
					int revc_alignment_nops,
					bool fwd_alignment_isvalid,
					bool revc_alignment_isvalid,
					int subjectlength,
					int querylength)->BestAlignment_t{

			return choose_best_alignment(fwd_alignment_overlap,
						revc_alignment_overlap,
						fwd_alignment_nops,
						revc_alignment_nops,
						fwd_alignment_isvalid,
						revc_alignment_isvalid,
						subjectlength,
						querylength,
						min_overlap_ratio,
						min_overlap,
						estimatedErrorrate * 4.0f);
		};

		auto callKernel = [&](auto subjectlength, auto querylength){
					  call_cuda_find_best_alignment_kernel_async_exp(
								  d_alignment_best_alignment_flags,
								  d_alignment_scores,
								  d_alignment_overlaps,
								  d_alignment_shifts,
								  d_alignment_nOps,
								  d_alignment_isValid,
								  d_candidates_per_subject_prefixsum,
								  n_subjects,
								  n_queries,
								  min_overlap_ratio,
								  min_overlap,
								  best_alignment_comp,
								  subjectlength,
								  querylength,
								  stream,
								  kernelLaunchHandle);
				  };

		if(!useGpuReadStorage) {
			callKernel( getSubjectLength_dense,
						getCandidateLength_dense);
		}else{
			if(gpuReadStorageGpuData.isValidSequenceData()) {
				callKernel( getSubjectLength_sparse,
							getCandidateLength_sparse);
			}else{
				callKernel( getSubjectLength_dense,
							getCandidateLength_dense);
			}
		}
	}
};


template<class ReadId_t>
struct MSAInitChooserExp {

	template<class GPUReadStorage_t>
	static void callKernelAsync(
				MSAColumnProperties*  d_msa_column_properties,                         //
				const int* d_alignment_shifts, //
				const BestAlignment_t* d_alignment_best_alignment_flags, //
				const ReadId_t* d_subject_read_ids, //
				const ReadId_t* d_candidate_read_ids, //
				const int* d_subject_sequences_lengths, //
				const int* d_candidate_sequences_lengths, //
				const int* d_indices, //
				const int* d_indices_per_subject, //
				const int* d_indices_per_subject_prefixsum, //
				int n_subjects, //
				int n_queries, //
				const GPUReadStorage_t* gpuReadStorage, //
				const typename GPUReadStorage_t::GPUData& gpuReadStorageGpuData,
				bool useGpuReadStorage, //
				cudaStream_t stream,
				KernelLaunchHandle& kernelLaunchHandle){ //

		assert(!useGpuReadStorage || (useGpuReadStorage && gpuReadStorage != nullptr));

		const typename GPUReadStorage_t::Length_t* d_sequence_lengths = gpuReadStorageGpuData.d_sequence_lengths;

		auto getSubjectLength_sparse = [=] __device__ (ReadId_t subjectIndex){
			const ReadId_t subjectReadId = d_subject_read_ids[subjectIndex];
			const int length = d_sequence_lengths[subjectReadId];
			return length;
		};

		auto getCandidateLength_sparse = [=] __device__ (ReadId_t subjectIndex, ReadId_t localCandidateIndex){
			const int* const indices_for_this_subject = d_indices + d_indices_per_subject_prefixsum[subjectIndex];
			const int index = indices_for_this_subject[localCandidateIndex];
			const ReadId_t candidateReadId = d_candidate_read_ids[index];
			const int length = d_sequence_lengths[candidateReadId];
			return length;
		};

		auto getSubjectLength_dense = [=] __device__ (ReadId_t subjectIndex){
			const int length = d_subject_sequences_lengths[subjectIndex];
			return length;
		};

		auto getCandidateLength_dense = [=] __device__ (ReadId_t subjectIndex, ReadId_t localCandidateIndex){
			const int* const indices_for_this_subject = d_indices + d_indices_per_subject_prefixsum[subjectIndex];
			const int index = indices_for_this_subject[localCandidateIndex];
			const int length = d_candidate_sequences_lengths[index];
			return length;
		};

		auto callKernel = [&](auto subjectlength, auto querylength){
					  call_msa_init_kernel_async_exp(
								  d_msa_column_properties,
								  d_alignment_shifts,
								  d_alignment_best_alignment_flags,
								  d_indices,
								  d_indices_per_subject,
								  d_indices_per_subject_prefixsum,
								  n_subjects,
								  n_queries,
								  subjectlength,
								  querylength,
								  stream,
								  kernelLaunchHandle);
				  };

		if(!useGpuReadStorage) {
			callKernel( getSubjectLength_dense,
						getCandidateLength_dense);
		}else{
			if(gpuReadStorageGpuData.isValidSequenceData()) {
				callKernel( getSubjectLength_sparse,
							getCandidateLength_sparse);
			}else{
				callKernel( getSubjectLength_dense,
							getCandidateLength_dense);
			}
		}
	}
};




template<class Sequence_t, class ReadId_t>
struct MSAAddSequencesChooserExp {

	template<class GPUReadStorage_t>
	static void callKernelAsync(
				char* d_multiple_sequence_alignments,
				float* d_multiple_sequence_alignment_weights,
				const int* d_alignment_shifts,
				const BestAlignment_t* d_alignment_best_alignment_flags,
				const ReadId_t* d_subject_read_ids,
				const ReadId_t* d_candidate_read_ids,
				const char* d_subject_sequences_data,
				const char* d_candidate_sequences_data,
				const int* d_subject_sequences_lengths,
				const int* d_candidate_sequences_lengths,
				const char* d_subject_qualities,
				const char* d_candidate_qualities,
				const int* d_alignment_overlaps,
				const int* d_alignment_nOps,
				const MSAColumnProperties*  d_msa_column_properties,
				const int* d_candidates_per_subject_prefixsum,
				const int* d_indices,
				const int* d_indices_per_subject,
				const int* d_indices_per_subject_prefixsum,
				int n_subjects,
				int n_queries,
                const int* h_num_indices,
				const int* d_num_indices,
				bool canUseQualityScores,
				float desiredAlignmentMaxErrorRate,
				int maximum_sequence_length,
				int max_sequence_bytes,
				size_t encoded_sequence_pitch,
				size_t quality_pitch,
				size_t msa_row_pitch,
				size_t msa_weights_row_pitch,
				const GPUReadStorage_t* gpuReadStorage,
				const typename GPUReadStorage_t::GPUData& gpuReadStorageGpuData,
				bool useGpuReadStorage,
				cudaStream_t stream,
				KernelLaunchHandle& kernelLaunchHandle){

		assert(!useGpuReadStorage || (useGpuReadStorage && gpuReadStorage != nullptr));
		//assert(!useGpuReadStorage || (useGpuReadStorage && gpuReadStorage->max_sequence_bytes == max_sequence_bytes));

		auto nucleotide_accessor = [] __device__ (const char* data, int length, int index){
			//return Sequence_t::get_as_nucleotide(data, length, index);
            return Sequence_t::get(data, length, index);
		};

		auto make_unpacked_reverse_complement_inplace = [] __device__ (std::uint8_t* sequence, int sequencelength){
			return care::SequenceString::make_reverse_complement_inplace(sequence, sequencelength);
		};

		const char* d_sequence_data = gpuReadStorageGpuData.d_sequence_data;
		const typename GPUReadStorage_t::Length_t* d_sequence_lengths = gpuReadStorageGpuData.d_sequence_lengths;
		const char* d_quality_data = gpuReadStorageGpuData.isValidQualityData() ?
		                             gpuReadStorageGpuData.d_quality_data :
		                             nullptr;
		const std::size_t readstorage_sequence_pitch = std::size_t(gpuReadStorage->getSequencePitch());
		const std::size_t readstorage_quality_pitch = std::size_t(gpuReadStorage->getQualityPitch());

		auto getSubjectPtr_sparse = [=] __device__ (ReadId_t subjectIndex){
			const ReadId_t subjectReadId = d_subject_read_ids[subjectIndex];
			const char* result = d_sequence_data + std::size_t(subjectReadId) * readstorage_sequence_pitch;
			return result;
		};

		auto getCandidatePtr_sparse = [=] __device__ (ReadId_t candidateIndex){
			const ReadId_t candidateReadId = d_candidate_read_ids[candidateIndex];
			const char* result = d_sequence_data + std::size_t(candidateReadId) * readstorage_sequence_pitch;
			return result;
		};

		auto getSubjectQualityPtr_sparse = [=] __device__ (ReadId_t subjectIndex){
			const ReadId_t subjectReadId = d_subject_read_ids[subjectIndex];
			const char* result = d_quality_data + std::size_t(subjectReadId) * readstorage_quality_pitch;
			return result;
		};

		auto getCandidateQualityPtr_sparse = [=] __device__ (ReadId_t localCandidateIndex){
			const int candidateIndex = d_indices[localCandidateIndex];
			const ReadId_t candidateReadId = d_candidate_read_ids[candidateIndex];
			const char* result = d_quality_data + std::size_t(candidateReadId) * readstorage_quality_pitch;
			return result;
		};

		auto getSubjectLength_sparse = [=] __device__ (ReadId_t subjectIndex){
			const ReadId_t subjectReadId = d_subject_read_ids[subjectIndex];
			const int length = d_sequence_lengths[subjectReadId];
			return length;
		};

		auto getCandidateLength_sparse = [=] __device__ (ReadId_t localCandidateIndex){
			const int candidateIndex = d_indices[localCandidateIndex];
			const ReadId_t candidateReadId = d_candidate_read_ids[candidateIndex];
			const int length = d_sequence_lengths[candidateReadId];
			return length;
		};

		auto getSubjectPtr_dense = [=] __device__ (ReadId_t subjectIndex){
			const char* result = d_subject_sequences_data + std::size_t(subjectIndex) * encoded_sequence_pitch;
			return result;
		};

		auto getCandidatePtr_dense = [=] __device__ (ReadId_t candidateIndex){
			const char* result = d_candidate_sequences_data + std::size_t(candidateIndex) * encoded_sequence_pitch;
			return result;
		};

		auto getSubjectQualityPtr_dense = [=] __device__ (ReadId_t subjectIndex){
			const char* result = d_subject_qualities + std::size_t(subjectIndex) * quality_pitch;
			return result;
		};

		auto getCandidateQualityPtr_dense = [=] __device__ (ReadId_t localCandidateIndex){
			const char* result = d_candidate_qualities + std::size_t(localCandidateIndex) * quality_pitch;
			return result;
		};

		auto getSubjectLength_dense = [=] __device__ (ReadId_t subjectIndex){
			const int length = d_subject_sequences_lengths[subjectIndex];
			return length;
		};

		auto getCandidateLength_dense = [=] __device__ (ReadId_t localCandidateIndex){
			const int candidateIndex = d_indices[localCandidateIndex];
			const int length = d_candidate_sequences_lengths[candidateIndex];
			return length;
		};

		auto callKernel = [&](auto subjectptr, auto candidateptr, auto subjectquality, auto candidatequality, auto subjectlength, auto querylength){
					  call_msa_add_sequences_kernel_exp_async(
								  d_multiple_sequence_alignments,
								  d_multiple_sequence_alignment_weights,
								  d_alignment_shifts,
								  d_alignment_best_alignment_flags,
								  d_subject_sequences_lengths,
								  d_candidate_sequences_lengths,
								  d_alignment_overlaps,
								  d_alignment_nOps,
								  d_msa_column_properties,
								  d_candidates_per_subject_prefixsum,
								  d_indices,
								  d_indices_per_subject,
								  d_indices_per_subject_prefixsum,
								  n_subjects,
								  n_queries,
                                  h_num_indices,
								  d_num_indices,
								  canUseQualityScores,
								  desiredAlignmentMaxErrorRate,
								  maximum_sequence_length,
								  max_sequence_bytes,
								  quality_pitch,
								  msa_row_pitch,
								  msa_weights_row_pitch,
								  nucleotide_accessor,
								  make_unpacked_reverse_complement_inplace,
								  subjectptr,
								  candidateptr,
								  subjectquality,
								  candidatequality,
								  subjectlength,
								  querylength,
								  stream,
								  kernelLaunchHandle);
				  };

		if(!useGpuReadStorage) {
			callKernel( getSubjectPtr_dense,
						getCandidatePtr_dense,
						getSubjectQualityPtr_dense,
						getCandidateQualityPtr_dense,
						getSubjectLength_dense,
						getCandidateLength_dense);
		}else{
			if(gpuReadStorageGpuData.isValidSequenceData()) {
				if(gpuReadStorageGpuData.isValidQualityData()) {
					callKernel( getSubjectPtr_sparse,
								getCandidatePtr_sparse,
								getSubjectQualityPtr_sparse,
								getCandidateQualityPtr_sparse,
								getSubjectLength_sparse,
								getCandidateLength_sparse);
				}else{
					callKernel( getSubjectPtr_sparse,
								getCandidatePtr_sparse,
								getSubjectQualityPtr_dense,
								getCandidateQualityPtr_dense,
								getSubjectLength_sparse,
								getCandidateLength_sparse);
				}
			}else{
				if(gpuReadStorageGpuData.isValidQualityData()) {
					callKernel( getSubjectPtr_dense,
								getCandidatePtr_dense,
								getSubjectQualityPtr_sparse,
								getCandidateQualityPtr_sparse,
								getSubjectLength_dense,
								getCandidateLength_dense);
				}else{
					callKernel( getSubjectPtr_dense,
								getCandidatePtr_dense,
								getSubjectQualityPtr_dense,
								getCandidateQualityPtr_dense,
								getSubjectLength_dense,
								getCandidateLength_dense);
				}
			}
		}
	}
};






template<class Sequence_t, class ReadId_t>
struct MSAAddSequencesChooserImplicit {

	template<class GPUReadStorage_t>
	static void callKernelAsync(
                int* d_counts,
                float* d_weights,
                int* d_coverage,
                float* d_origWeights,
                int* d_origCoverages,
				const int* d_alignment_shifts,
				const BestAlignment_t* d_alignment_best_alignment_flags,
				const ReadId_t* d_subject_read_ids,
				const ReadId_t* d_candidate_read_ids,
				const char* d_subject_sequences_data,
				const char* d_candidate_sequences_data,
				const int* d_subject_sequences_lengths,
				const int* d_candidate_sequences_lengths,
				const char* d_subject_qualities,
				const char* d_candidate_qualities,
				const int* d_alignment_overlaps,
				const int* d_alignment_nOps,
				const MSAColumnProperties*  d_msa_column_properties,
				const int* d_candidates_per_subject_prefixsum,
				const int* d_indices,
				const int* d_indices_per_subject,
				const int* d_indices_per_subject_prefixsum,
				int n_subjects,
				int n_queries,
                const int* h_num_indices,
				const int* d_num_indices,
				bool canUseQualityScores,
				float desiredAlignmentMaxErrorRate,
				int maximum_sequence_length,
				int max_sequence_bytes,
				size_t encoded_sequence_pitch,
				size_t quality_pitch,
				size_t msa_row_pitch,
				size_t msa_weights_row_pitch,
				const GPUReadStorage_t* gpuReadStorage,
				const typename GPUReadStorage_t::GPUData& gpuReadStorageGpuData,
				bool useGpuReadStorage,
				cudaStream_t stream,
				KernelLaunchHandle& kernelLaunchHandle){

		assert(!useGpuReadStorage || (useGpuReadStorage && gpuReadStorage != nullptr));
		//assert(!useGpuReadStorage || (useGpuReadStorage && gpuReadStorage->max_sequence_bytes == max_sequence_bytes));

		auto nucleotide_accessor = [] __device__ (const char* data, int length, int index){
			//return Sequence_t::get_as_nucleotide(data, length, index);
            return Sequence_t::get(data, length, index);
		};

		auto make_unpacked_reverse_complement_inplace = [] __device__ (std::uint8_t* sequence, int sequencelength){
			return care::SequenceString::make_reverse_complement_inplace(sequence, sequencelength);
		};

		const char* d_sequence_data = gpuReadStorageGpuData.d_sequence_data;
		const typename GPUReadStorage_t::Length_t* d_sequence_lengths = gpuReadStorageGpuData.d_sequence_lengths;
		const char* d_quality_data = gpuReadStorageGpuData.isValidQualityData() ?
		                             gpuReadStorageGpuData.d_quality_data :
		                             nullptr;
		const std::size_t readstorage_sequence_pitch = std::size_t(gpuReadStorage->getSequencePitch());
		const std::size_t readstorage_quality_pitch = std::size_t(gpuReadStorage->getQualityPitch());

		auto getSubjectPtr_sparse = [=] __device__ (ReadId_t subjectIndex){
			const ReadId_t subjectReadId = d_subject_read_ids[subjectIndex];
			const char* result = d_sequence_data + std::size_t(subjectReadId) * readstorage_sequence_pitch;
			return result;
		};

		auto getCandidatePtr_sparse = [=] __device__ (ReadId_t candidateIndex){
			const ReadId_t candidateReadId = d_candidate_read_ids[candidateIndex];
			const char* result = d_sequence_data + std::size_t(candidateReadId) * readstorage_sequence_pitch;
			return result;
		};

		auto getSubjectQualityPtr_sparse = [=] __device__ (ReadId_t subjectIndex){
			const ReadId_t subjectReadId = d_subject_read_ids[subjectIndex];
			const char* result = d_quality_data + std::size_t(subjectReadId) * readstorage_quality_pitch;
			return result;
		};

		auto getCandidateQualityPtr_sparse = [=] __device__ (ReadId_t localCandidateIndex){
			const int candidateIndex = d_indices[localCandidateIndex];
			const ReadId_t candidateReadId = d_candidate_read_ids[candidateIndex];
			const char* result = d_quality_data + std::size_t(candidateReadId) * readstorage_quality_pitch;
			return result;
		};

		auto getCandidateLength_sparse = [=] __device__ (ReadId_t localCandidateIndex){
			const int candidateIndex = d_indices[localCandidateIndex];
			const ReadId_t candidateReadId = d_candidate_read_ids[candidateIndex];
			const int length = d_sequence_lengths[candidateReadId];
			return length;
		};

		auto getSubjectPtr_dense = [=] __device__ (ReadId_t subjectIndex){
			const char* result = d_subject_sequences_data + std::size_t(subjectIndex) * encoded_sequence_pitch;
			return result;
		};

		auto getCandidatePtr_dense = [=] __device__ (ReadId_t candidateIndex){
			const char* result = d_candidate_sequences_data + std::size_t(candidateIndex) * encoded_sequence_pitch;
			return result;
		};

		auto getSubjectQualityPtr_dense = [=] __device__ (ReadId_t subjectIndex){
			const char* result = d_subject_qualities + std::size_t(subjectIndex) * quality_pitch;
			return result;
		};

		auto getCandidateQualityPtr_dense = [=] __device__ (ReadId_t localCandidateIndex){
			const char* result = d_candidate_qualities + std::size_t(localCandidateIndex) * quality_pitch;
			return result;
		};

		auto getCandidateLength_dense = [=] __device__ (ReadId_t localCandidateIndex){
			const int candidateIndex = d_indices[localCandidateIndex];
			const int length = d_candidate_sequences_lengths[candidateIndex];
			return length;
		};

		auto callKernel = [&](auto subjectptr, auto candidateptr, auto subjectquality, auto candidatequality, auto querylength){
					  call_msa_add_sequences_kernel_implicit_async(
                                  d_counts,
                                  d_weights,
                                  d_coverage,
                                  d_origWeights,
                                  d_origCoverages,
								  d_alignment_shifts,
								  d_alignment_best_alignment_flags,
								  d_subject_sequences_lengths,
								  d_candidate_sequences_lengths,
								  d_alignment_overlaps,
								  d_alignment_nOps,
								  d_msa_column_properties,
								  d_candidates_per_subject_prefixsum,
								  d_indices,
								  d_indices_per_subject,
								  d_indices_per_subject_prefixsum,
								  n_subjects,
								  n_queries,
                                  h_num_indices,
								  d_num_indices,
								  canUseQualityScores,
								  desiredAlignmentMaxErrorRate,
								  maximum_sequence_length,
								  max_sequence_bytes,
								  quality_pitch,
								  msa_row_pitch,
								  msa_weights_row_pitch,
								  nucleotide_accessor,
								  make_unpacked_reverse_complement_inplace,
								  subjectptr,
								  candidateptr,
								  subjectquality,
								  candidatequality,
								  querylength,
								  stream,
								  kernelLaunchHandle);
				  };

		if(!useGpuReadStorage) {
			callKernel( getSubjectPtr_dense,
						getCandidatePtr_dense,
						getSubjectQualityPtr_dense,
						getCandidateQualityPtr_dense,
						getCandidateLength_dense);
		}else{
			if(gpuReadStorageGpuData.isValidSequenceData()) {
				if(gpuReadStorageGpuData.isValidQualityData()) {
					callKernel( getSubjectPtr_sparse,
								getCandidatePtr_sparse,
								getSubjectQualityPtr_sparse,
								getCandidateQualityPtr_sparse,
								getCandidateLength_sparse);
				}else{
					callKernel( getSubjectPtr_sparse,
								getCandidatePtr_sparse,
								getSubjectQualityPtr_dense,
								getCandidateQualityPtr_dense,
								getCandidateLength_sparse);
				}
			}else{
				if(gpuReadStorageGpuData.isValidQualityData()) {
					callKernel( getSubjectPtr_dense,
								getCandidatePtr_dense,
								getSubjectQualityPtr_sparse,
								getCandidateQualityPtr_sparse,
								getCandidateLength_dense);
				}else{
					callKernel( getSubjectPtr_dense,
								getCandidatePtr_dense,
								getSubjectQualityPtr_dense,
								getCandidateQualityPtr_dense,
								getCandidateLength_dense);
				}
			}
		}
	}
};



template<class Sequence_t, class ReadId_t>
struct MSAFindConsensusChooserImplicit {

	template<class GPUReadStorage_t>
	static void callKernelAsync(
                int* d_counts,
                float* d_weights,
                char* d_consensus,
                float* d_support,
                int* d_coverage,
                float* d_origWeights,
                int* d_origCoverages,
				const ReadId_t* d_subject_read_ids,
				const char* d_subject_sequences_data,
				const MSAColumnProperties*  d_msa_column_properties,
				int n_subjects,
				int max_sequence_bytes,
				size_t encoded_sequence_pitch,
				size_t msa_row_pitch,
				size_t msa_weights_row_pitch,
				const GPUReadStorage_t* gpuReadStorage,
				const typename GPUReadStorage_t::GPUData& gpuReadStorageGpuData,
				bool useGpuReadStorage,
				cudaStream_t stream,
				KernelLaunchHandle& kernelLaunchHandle){

		assert(!useGpuReadStorage || (useGpuReadStorage && gpuReadStorage != nullptr));
		//assert(!useGpuReadStorage || (useGpuReadStorage && gpuReadStorage->max_sequence_bytes == max_sequence_bytes));

		auto nucleotide_accessor = [] __device__ (const char* data, int length, int index){
			//return Sequence_t::get_as_nucleotide(data, length, index);
            return Sequence_t::get(data, length, index);
		};

		const char* d_sequence_data = gpuReadStorageGpuData.d_sequence_data;
		const std::size_t readstorage_sequence_pitch = std::size_t(gpuReadStorage->getSequencePitch());

        //sparse -> read from gpuReadStorage
		auto getSubjectPtr_sparse = [=] __device__ (ReadId_t subjectIndex){
			const ReadId_t subjectReadId = d_subject_read_ids[subjectIndex];
			const char* result = d_sequence_data + std::size_t(subjectReadId) * readstorage_sequence_pitch;
			return result;
		};

		auto getSubjectPtr_dense = [=] __device__ (ReadId_t subjectIndex){
			const char* result = d_subject_sequences_data + std::size_t(subjectIndex) * encoded_sequence_pitch;
			return result;
		};

		auto callKernel = [&](auto subjectptr){
            call_msa_find_consensus_implicit_kernel_async(
                                    d_counts,
                                    d_weights,
                                    d_consensus,
                                    d_support,
                                    d_coverage,
                                    d_origWeights,
                                    d_origCoverages,
                                    d_msa_column_properties,
                                    n_subjects,
                                    msa_row_pitch,
		                            msa_weights_row_pitch,
                                    nucleotide_accessor,
                                    subjectptr,
                                    stream,
                                    kernelLaunchHandle);
        };

		if(!useGpuReadStorage) {
			callKernel( getSubjectPtr_dense);
		}else{
			if(gpuReadStorageGpuData.isValidSequenceData()) {
				callKernel(getSubjectPtr_sparse);
			}else{
				callKernel(getSubjectPtr_dense);
			}
		}
	}
};




template<class Sequence_t, class ReadId_t>
struct MSACorrectSubjectChooserImplicit {

	template<class GPUReadStorage_t>
	static void callKernelAsync(
                char* d_consensus,
                float* d_support,
                int* d_coverage,
                int* d_origCoverages,
				const MSAColumnProperties*  d_msa_column_properties,
                bool* d_is_high_quality_subject,
                char* d_corrected_subjects,
                bool* d_subject_is_corrected,
				int n_subjects,
				size_t encoded_sequence_pitch,
				size_t msa_row_pitch,
				size_t msa_weights_row_pitch,
                float estimatedErrorrate,
                float avg_support_threshold,
                float min_support_threshold,
                float min_coverage_threshold,
                int k_region,
                int maximum_sequence_length,
                const ReadId_t* d_subject_read_ids,
				const char* d_subject_sequences_data,
				const GPUReadStorage_t* gpuReadStorage,
				const typename GPUReadStorage_t::GPUData& gpuReadStorageGpuData,
				bool useGpuReadStorage,
				cudaStream_t stream,
				KernelLaunchHandle& kernelLaunchHandle){

		assert(!useGpuReadStorage || (useGpuReadStorage && gpuReadStorage != nullptr));

		auto nucleotide_accessor = [] __device__ (const char* data, int length, int index){
			//return Sequence_t::get_as_nucleotide(data, length, index);
            return Sequence_t::get(data, length, index);
		};

		const char* d_sequence_data = gpuReadStorageGpuData.d_sequence_data;
		const std::size_t readstorage_sequence_pitch = std::size_t(gpuReadStorage->getSequencePitch());

        //sparse -> read from gpuReadStorage
		auto getSubjectPtr_sparse = [=] __device__ (ReadId_t subjectIndex){
			const ReadId_t subjectReadId = d_subject_read_ids[subjectIndex];
			const char* result = d_sequence_data + std::size_t(subjectReadId) * readstorage_sequence_pitch;
			return result;
		};

		auto getSubjectPtr_dense = [=] __device__ (ReadId_t subjectIndex){
			const char* result = d_subject_sequences_data + std::size_t(subjectIndex) * encoded_sequence_pitch;
			return result;
		};

		auto callKernel = [&](auto subjectptr){
            call_msa_correct_subject_implicit_kernel_async(
                                    d_consensus,
                                    d_support,
                                    d_coverage,
                                    d_origCoverages,
                                    d_msa_column_properties,
                                    d_is_high_quality_subject,
                                    d_corrected_subjects,
                                    d_subject_is_corrected,
                                    n_subjects,
                                    encoded_sequence_pitch,
                                    msa_row_pitch,
		                            msa_weights_row_pitch,
                                    estimatedErrorrate,
                                    avg_support_threshold,
                                    min_support_threshold,
                                    min_coverage_threshold,
                                    k_region,
                                    maximum_sequence_length,
                                    nucleotide_accessor,
                                    subjectptr,
                                    stream,
                                    kernelLaunchHandle);
        };

		if(!useGpuReadStorage) {
			callKernel(getSubjectPtr_dense);
		}else{
			if(gpuReadStorageGpuData.isValidSequenceData()) {
				callKernel(getSubjectPtr_sparse);
			}else{
				callKernel(getSubjectPtr_dense);
			}
		}
	}
};




template<class ReadId_t>
struct MSACorrectCandidatesChooserExp {

	template<class GPUReadStorage_t>
	static void callKernelAsync(
				const char* d_consensus,
				const float* d_support,
				const int* d_coverage,
				const int* d_origCoverages,
				const char* d_multiple_sequence_alignments,
				const MSAColumnProperties* d_msa_column_properties,
				const int* d_indices,
				const int* d_indices_per_subject,
				const int* d_indices_per_subject_prefixsum,
				const int* d_high_quality_subject_indices,
				const int* d_num_high_quality_subject_indices,
				const int* d_alignment_shifts,
				const BestAlignment_t* d_alignment_best_alignment_flags,
				const ReadId_t* d_candidate_read_ids,
				const int* d_candidate_sequences_lengths,
				int* d_num_corrected_candidates,
				char* d_corrected_candidates,
				int* d_indices_of_corrected_candidates,
				int n_subjects,
				int n_queries,
				const int* d_num_indices,
				size_t sequence_pitch,
				size_t msa_pitch,
				size_t msa_weights_pitch,
				float min_support_threshold,
				float min_coverage_threshold,
				int new_columns_to_correct,
				int maximum_sequence_length,
				const GPUReadStorage_t* gpuReadStorage,
				const typename GPUReadStorage_t::GPUData& gpuReadStorageGpuData,
				bool useGpuReadStorage,
				cudaStream_t stream,
				KernelLaunchHandle& kernelLaunchHandle){

		assert(!useGpuReadStorage || (useGpuReadStorage && gpuReadStorage != nullptr));

		const typename GPUReadStorage_t::Length_t* d_sequence_lengths = gpuReadStorageGpuData.d_sequence_lengths;

		auto make_unpacked_reverse_complement_inplace = [] __device__ (std::uint8_t* sequence, int sequencelength){
			return care::SequenceString::make_reverse_complement_inplace(sequence, sequencelength);
		};

		auto getCandidateLength_sparse = [=] __device__ (ReadId_t subjectIndex, ReadId_t localCandidateIndex){
			const int* const my_indices = d_indices + d_indices_per_subject_prefixsum[subjectIndex];
			const int index = my_indices[localCandidateIndex];
			const ReadId_t candidateReadId = d_candidate_read_ids[index];
			const int length = d_sequence_lengths[candidateReadId];
			return length;
		};

		auto getCandidateLength_dense = [=] __device__ (ReadId_t subjectIndex, ReadId_t localCandidateIndex){
			const int* const my_indices = d_indices + d_indices_per_subject_prefixsum[subjectIndex];
			const int index = my_indices[localCandidateIndex];
			const int length = d_candidate_sequences_lengths[index];
			return length;
		};

		auto callKernel = [&](auto revcompl, auto candidatelength){
					  call_msa_correct_candidates_kernel_async_exp(
								  d_consensus,
								  d_support,
								  d_coverage,
								  d_origCoverages,
								  d_multiple_sequence_alignments,
								  d_msa_column_properties,
								  d_indices,
								  d_indices_per_subject,
								  d_indices_per_subject_prefixsum,
								  d_high_quality_subject_indices,
								  d_num_high_quality_subject_indices,
								  d_alignment_shifts,
								  d_alignment_best_alignment_flags,
								  d_num_corrected_candidates,
								  d_corrected_candidates,
								  d_indices_of_corrected_candidates,
								  n_subjects,
								  n_queries,
								  d_num_indices,
								  sequence_pitch,
								  msa_pitch,
								  msa_weights_pitch,
								  min_support_threshold,
								  min_coverage_threshold,
								  new_columns_to_correct,
								  revcompl,
								  candidatelength,
								  maximum_sequence_length,
								  stream,
								  kernelLaunchHandle);
				  };

		if(!useGpuReadStorage) {
			callKernel( make_unpacked_reverse_complement_inplace,
						getCandidateLength_dense);
		}else{
			if(gpuReadStorageGpuData.isValidSequenceData()) {
				callKernel( make_unpacked_reverse_complement_inplace,
							getCandidateLength_sparse);
			}else{
				callKernel( make_unpacked_reverse_complement_inplace,
							getCandidateLength_dense);
			}
		}
	}
};

#endif

}
}




#endif
