#ifndef CARE_GPU_KERNELS_HPP
#define CARE_GPU_KERNELS_HPP

#include "../hpc_helpers.cuh"
#include "bestalignment.hpp"
#include <gpu/qualityscoreweights.hpp>
#include "../msa.hpp"

#include <stdexcept>
#include <cassert>
#include <map>

#ifdef __NVCC__
#include <cub/cub.cuh>
#endif


namespace care {
namespace gpu {


#ifdef __NVCC__
using MSAColumnProperties = care::cpu::MultipleSequenceAlignment::ColumnProperties;

enum class KernelId {
	PopcountSHD,
    PopcountSHDtransposed,
	FindBestAlignmentExp,
	FilterAlignmentsByMismatchRatio,
	MSAInitExp,
	MSAAddSequences,
	MSAFindConsensus,
	MSACorrectSubject,
	MSACorrectCandidates,
    MSAAddSequencesImplicit,
    MSAFindConsensusImplicit,
    MSACorrectSubjectImplicit,
};

struct KernelLaunchConfig {
	int threads_per_block;
	int smem;
};

constexpr bool operator<(const KernelLaunchConfig& lhs, const KernelLaunchConfig& rhs){
	return lhs.threads_per_block < rhs.threads_per_block
	       && lhs.smem < rhs.smem;
}

struct KernelProperties {
	int max_blocks_per_SM = 1;
};

struct KernelLaunchHandle {
	int deviceId;
	cudaDeviceProp deviceProperties;
	std::map<KernelId, std::map<KernelLaunchConfig, KernelProperties> > kernelPropertiesMap;
};

KernelLaunchHandle make_kernel_launch_handle(int deviceId);

void call_cuda_filter_alignments_by_mismatchratio_kernel_async(
			BestAlignment_t* d_alignment_best_alignment_flags,
			const int* d_alignment_overlaps,
			const int* d_alignment_nOps,
			const int* d_indices,
			const int* d_indices_per_subject,
			const int* d_indices_per_subject_prefixsum,
			int n_subjects,
			int n_candidates,
			const int* d_num_indices,
			float mismatchratioBaseFactor,
			float goodAlignmentsCountThreshold,
			cudaStream_t stream,
			KernelLaunchHandle& handle);

void call_msa_find_consensus_kernel_async(
                        char* d_consensus,
                        float* d_support,
                        int* d_coverage,
                        float* d_origWeights,
                        int* d_origCoverages,
                        int* d_countsA,
                        int* d_countsC,
                        int* d_countsG,
                        int* d_countsT,
                        float* d_weightsA,
                        float* d_weightsC,
                        float* d_weightsG,
                        float* d_weightsT,
                        const char* d_multiple_sequence_alignments,
                        const float* d_multiple_sequence_alignment_weights,
                        const MSAColumnProperties* d_msa_column_properties,
                        const int* d_candidates_per_subject_prefixsum,
                        const int* d_indices_per_subject,
                        const int* d_indices_per_subject_prefixsum,
                        int n_subjects,
                        int n_queries,
                        const int* d_num_indices,
                        size_t msa_pitch,
                        size_t msa_weights_pitch,
                        int msa_max_column_count,
                        cudaStream_t stream,
                        KernelLaunchHandle& handle);

void call_msa_correct_subject_kernel_async(
			const char* d_consensus,
			const float* d_support,
			const int* d_coverage,
			const int* d_origCoverages,
			const char* d_multiple_sequence_alignments,
			const MSAColumnProperties* d_msa_column_properties,
			const int* d_indices_per_subject_prefixsum,
			bool* d_is_high_quality_subject,
			char* d_corrected_subjects,
			bool* d_subject_is_corrected,
			int n_subjects,
			int n_queries,
			const int* d_num_indices,
			size_t sequence_pitch,
			size_t msa_pitch,
			size_t msa_weights_pitch,
			float estimatedErrorrate,
			float avg_support_threshold,
			float min_support_threshold,
			float min_coverage_threshold,
			int k_region,
			int maximum_sequence_length,
			cudaStream_t stream,
			KernelLaunchHandle& handle);


/*
    SHIFTED HAMMING DISTANCE
 */
#if 0
template<int BLOCKSIZE, class Accessor, class RevCompl>
__global__
void
cuda_shifted_hamming_distance_with_revcompl_kernel(
			int* alignment_scores,
			int* alignment_overlaps,
			int* alignment_shifts,
			int* alignment_nOps,
			bool* alignment_isValid,
			const char* subject_sequences_data,
			const char* candidate_sequences_data,
			const int* subject_sequences_lengths,
			const int* candidate_sequences_lengths,
			const int* candidates_per_subject_prefixsum,
			int n_subjects,
			int n_queries,
			int max_sequence_bytes,
			size_t sequencepitch,
			int min_overlap,
			float maxErrorRate,
			float min_overlap_ratio,
			Accessor getChar,
			RevCompl make_reverse_complement_inplace){

	using BlockReduceInt2 = cub::BlockReduce<int2, BLOCKSIZE>;

	__shared__ union {
		typename BlockReduceInt2::TempStorage reduce;
	} temp_storage;

	extern __shared__ char smem[];

	//set up shared memory pointers
	char* sharedSubject = (char*)(smem);
	char* sharedQuery = (char*)(sharedSubject + max_sequence_bytes);

	const int nQueries = n_queries;

	for(unsigned resultIndex = blockIdx.x; resultIndex < nQueries * 2; resultIndex += gridDim.x) {

		const int queryIndex = resultIndex < nQueries ? resultIndex : resultIndex - nQueries;

		//find subjectindex
		int subjectIndex = 0;
		for(; subjectIndex < n_subjects; subjectIndex++) {
			if(queryIndex < candidates_per_subject_prefixsum[subjectIndex+1])
				break;
		}

		//save subject in shared memory
		const int subjectbases = subject_sequences_lengths[subjectIndex];
		for(int threadid = threadIdx.x; threadid < max_sequence_bytes; threadid += BLOCKSIZE) {
			sharedSubject[threadid] = subject_sequences_data[subjectIndex * sequencepitch + threadid];
		}

		//save query in shared memory
		const int querybases = candidate_sequences_lengths[queryIndex];
		for(int threadid = threadIdx.x; threadid < max_sequence_bytes; threadid += BLOCKSIZE) {
			sharedQuery[threadid] = candidate_sequences_data[queryIndex * sequencepitch + threadid];
		}

		__syncthreads();
		//queryIndex != resultIndex -> reverse complement
		if(queryIndex != resultIndex) {
			if(threadIdx.x == 0) {
				make_reverse_complement_inplace((std::uint8_t*)sharedQuery, querybases);
			}
			__syncthreads();
		}

		//begin SHD algorithm

		const char* query = sharedQuery;

		const int minoverlap = max(min_overlap, int(float(subjectbases) * min_overlap_ratio));
		const int totalbases = subjectbases + querybases;

		int bestScore = totalbases;                 // score is number of mismatches
		int bestShift = -querybases;                 // shift of query relative to subject. shift < 0 if query begins before subject

		for(int shift = -querybases + minoverlap + threadIdx.x; shift < subjectbases - minoverlap + 1; shift += BLOCKSIZE) {
			const int overlapsize = min(querybases, subjectbases - shift) - max(-shift, 0);
			const int max_errors = int(float(overlapsize) * maxErrorRate);
			int score = 0;

			for(int j = max(-shift, 0); j < min(querybases, subjectbases - shift) && score < max_errors; j++) {
				score += getChar(sharedSubject, subjectbases, j + shift) != getChar(query, querybases, j);
			}

			score = (score < max_errors ?
			         score + totalbases - 2*overlapsize                         // non-overlapping regions count as mismatches
			         : std::numeric_limits<int>::max());                         // too many errors, discard

			if(score < bestScore) {
				bestScore = score;
				bestShift = shift;
			}
		}
		// perform reduction to find smallest score in block. the corresponding shift is required, too
		// pack both score and shift into int2 and perform int2-reduction by only comparing the score

		int2 myval = make_int2(bestScore, bestShift);

		myval = BlockReduceInt2(temp_storage.reduce).Reduce(myval, [](auto a, auto b){
					return a.x < b.x ? a : b;
				});
		__syncthreads();

		//make result
		if(threadIdx.x == 0) {
			bestScore = myval.x;
			bestShift = myval.y;

			const int queryoverlapbegin_incl = max(-bestShift, 0);
			const int queryoverlapend_excl = min(querybases, subjectbases - bestShift);
			const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
			const int opnr = bestScore - totalbases + 2*overlapsize;

			alignment_scores[resultIndex] = bestScore;
			alignment_overlaps[resultIndex] = overlapsize;
			alignment_shifts[resultIndex] = bestShift;
			alignment_nOps[resultIndex] = opnr;
			alignment_isValid[resultIndex] = (bestShift != -querybases);
		}
	}
}

template<class Accessor, class RevCompl>
void call_shd_with_revcompl_kernel_async(
			int* d_alignment_scores,
			int* d_alignment_overlaps,
			int* d_alignment_shifts,
			int* d_alignment_nOps,
			bool* d_alignment_isValid,
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
			Accessor accessor,
			RevCompl make_reverse_complement_inplace,
			int maxSubjectLength,
			int maxQueryLength,
			cudaStream_t stream){

	#define mycall(blocksize) cuda_shifted_hamming_distance_with_revcompl_kernel<(blocksize)> \
	        <<<grid, block, smem, stream>>>( \
		d_alignment_scores, \
		d_alignment_overlaps, \
		d_alignment_shifts, \
		d_alignment_nOps, \
		d_alignment_isValid, \
		d_subject_sequences_data, \
		d_candidate_sequences_data, \
		d_subject_sequences_lengths, \
		d_candidate_sequences_lengths, \
		d_candidates_per_subject_prefixsum, \
		n_subjects, \
		n_queries, \
		max_sequence_bytes, \
		encodedsequencepitch, \
		min_overlap, \
		maxErrorRate, \
		min_overlap_ratio, \
		accessor, \
		make_reverse_complement_inplace); CUERR;

/*
 #define getsms(blocksize) {\
                        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM, \
                                                                        cuda_shifted_hamming_distance_with_revcompl_kernel<(blocksize), Accessor, RevCompl>, \
                                                                        blocksize, smem); CUERR;}

 */


#define getsms(blocksize) {max_blocks_per_SM = 12; \
}

	const int minoverlap = max(min_overlap, int(float(maxSubjectLength) * min_overlap_ratio));
	const int maxShiftsToCheck = maxSubjectLength + maxQueryLength - 2*minoverlap;
	const std::size_t smem = sizeof(char) * 2 * max_sequence_bytes;

	const int blocksize = std::min(256, 32 * SDIV(maxShiftsToCheck, 32));

	int deviceId;
	cudaGetDevice(&deviceId); CUERR;

	int SMs;
	cudaDeviceGetAttribute(&SMs, cudaDevAttrMultiProcessorCount, deviceId); CUERR;

	int max_blocks_per_SM = 1;

	switch(blocksize) {
	case 32: getsms(32); break;
	case 64: getsms(64); break;
	case 96: getsms(96); break;
	case 128: getsms(128); break;
	case 160: getsms(160); break;
	case 192: getsms(192); break;
	case 224: getsms(224); break;
	case 256: getsms(256); break;
	default: throw std::runtime_error("Want to call shd kernel with 0 threads due to a bug.");
	}

	int max_blocks_per_device = SMs * max_blocks_per_SM;

	dim3 block(blocksize, 1, 1);
	dim3 grid(std::min(n_queries*2, max_blocks_per_device), 1, 1);         // one block per (query and its reverse complement)

	switch(blocksize) {
	case 32: mycall(32); break;
	case 64: mycall(64); break;
	case 96: mycall(96); break;
	case 128: mycall(128); break;
	case 160: mycall(160); break;
	case 192: mycall(192); break;
	case 224: mycall(224); break;
	case 256: mycall(256); break;
	default: throw std::runtime_error("Want to call shd kernel with 0 threads due to a bug.");
	}

	  #undef mycall
	  #undef getsms
}


#endif






template<int BLOCKSIZE, class B, class GetSubjectPtr, class GetCandidatePtr, class GetSubjectLength, class GetCandidateLength>
__global__
void
cuda_popcount_shifted_hamming_distance_with_revcompl_kernel(
			int* alignment_scores,
			int* alignment_overlaps,
			int* alignment_shifts,
			int* alignment_nOps,
			bool* alignment_isValid,
			//const int* subject_sequences_lengths,
			//const int* candidate_sequences_lengths,
			const int* candidates_per_subject_prefixsum,
			int n_subjects,
			int n_candidates,
			int max_sequence_bytes,
			int min_overlap,
			float maxErrorRate,
			float min_overlap_ratio,
			B getNumBytes,
			GetSubjectPtr getSubjectPtr,
			GetCandidatePtr getCandidatePtr,
			GetSubjectLength getSubjectLength,
			GetCandidateLength getCandidateLength){

	auto no_bank_conflict_index = [](int logical_index) -> int {
					      return logical_index * BLOCKSIZE;
				      };

	auto make_reverse_complement_inplace = [&](unsigned int* sequence, int sequencelength){

						       auto reverse_complement_int = [](auto n) {
											     n = ((n >> 1) & 0x55555555) | ((n << 1) & 0xaaaaaaaa);
											     n = ((n >> 2) & 0x33333333) | ((n << 2) & 0xcccccccc);
											     n = ((n >> 4) & 0x0f0f0f0f) | ((n << 4) & 0xf0f0f0f0);
											     n = ((n >> 8) & 0x00ff00ff) | ((n << 8) & 0xff00ff00);
											     n = ((n >> 16) & 0x0000ffff) | ((n << 16) & 0xffff0000);
											     return ~n;
										     };

						       const int ints = getNumBytes(sequencelength) / sizeof(unsigned int);
						       const int unusedBitsInt = SDIV(sequencelength, 8 * sizeof(unsigned int)) * 8 * sizeof(unsigned int) - sequencelength;

						       unsigned int* const hi = sequence;
						       unsigned int* const lo = sequence + ints/2;

						       const int intsPerHalf = SDIV(sequencelength, 8 * sizeof(unsigned int));
						       for(int i = 0; i < intsPerHalf/2; ++i) {
							       const unsigned int hifront = reverse_complement_int(hi[i]);
							       const unsigned int hiback = reverse_complement_int(hi[intsPerHalf - 1 - i]);
							       hi[i] = hiback;
							       hi[intsPerHalf - 1 - i] = hifront;

							       const unsigned int lofront = reverse_complement_int(lo[i]);
							       const unsigned int loback = reverse_complement_int(lo[intsPerHalf - 1 - i]);
							       lo[i] = loback;
							       lo[intsPerHalf - 1 - i] = lofront;
						       }
						       if(intsPerHalf % 2 == 1) {
							       const int middleindex = intsPerHalf/2;
							       hi[middleindex] = reverse_complement_int(hi[middleindex]);
							       lo[middleindex] = reverse_complement_int(lo[middleindex]);
						       }

						       if(unusedBitsInt != 0) {
							       for(int i = 0; i < intsPerHalf - 1; ++i) {
								       hi[i] = (hi[i] >> unusedBitsInt) | (hi[i+1] << (8 * sizeof(unsigned int) - unusedBitsInt));
								       lo[i] = (lo[i] >> unusedBitsInt) | (lo[i+1] << (8 * sizeof(unsigned int) - unusedBitsInt));
							       }

							       hi[intsPerHalf - 1] >>= unusedBitsInt;
							       lo[intsPerHalf - 1] >>= unusedBitsInt;
						       }
					       };


	auto shiftEncodedBasesLeftBy = [&](unsigned int* array, int size, int shiftamount){
					       const int completeInts = shiftamount / (8 * sizeof(unsigned int));

					       for(int i = 0; i < size - completeInts; i += 1) {
						       array[no_bank_conflict_index(i)] = array[no_bank_conflict_index(completeInts + i)];
					       }

					       for(int i = size - completeInts; i < size; i += 1) {
						       array[no_bank_conflict_index(i)] = 0;
					       }

					       shiftamount -= completeInts * 8 * sizeof(unsigned int);

					       for(int i = 0; i < size - completeInts - 1; i += 1) {
						       const unsigned int a = array[no_bank_conflict_index(i)];
						       const unsigned int b = array[no_bank_conflict_index(i+1)];

						       array[no_bank_conflict_index(i)] = (a >> shiftamount) | (b << (8 * sizeof(unsigned int) - shiftamount));
					       }

					       array[no_bank_conflict_index(size - completeInts - 1)] >>= shiftamount;
				       };

        auto hammingdistanceHiLo2 = [&](
            //the i-th entry of l must be located at l[no_bank_conflict_index(i)]
        	const unsigned int* lhi,
        	const unsigned int* llo,
            //the i-th entry of r must be located at r[i]
        	const unsigned int* rhi,
        	const unsigned int* rlo,
        	int lhi_bitcount,
        	int rhi_bitcount,
        	int max_errors){

		   const int overlap_bitcount = min(lhi_bitcount, rhi_bitcount);

		   if(overlap_bitcount == 0)
			   return max_errors+1;

		   const int partitions = SDIV(overlap_bitcount, (8 * sizeof(unsigned int)));
		   const int remaining_bitcount = partitions * sizeof(unsigned int) * 8 - overlap_bitcount;

		   int result = 0;

		   for(int i = 0; i < partitions - 1 && result < max_errors; i += 1) {
			   const unsigned int hixor = lhi[no_bank_conflict_index(i)] ^ rhi[i];
			   const unsigned int loxor = llo[no_bank_conflict_index(i)] ^ rlo[i];
			   const unsigned int bits = hixor | loxor;
			   result += __popc(bits);
		   }

		   if(result >= max_errors)
			   return result;

		   // i == partitions - 1

		   const unsigned int mask = remaining_bitcount == 0 ? 0xFFFFFFFF : 0xFFFFFFFF >> (remaining_bitcount);
		   const unsigned int hixor = lhi[no_bank_conflict_index(partitions - 1)] ^ rhi[partitions - 1];
		   const unsigned int loxor = llo[no_bank_conflict_index(partitions - 1)] ^ rlo[partitions - 1];
		   const unsigned int bits = hixor | loxor;
		   result += __popc(bits & mask);

		   return result;
	   };

	using BlockReduceInt2 = cub::BlockReduce<int2, BLOCKSIZE>;

	__shared__ union {
		typename BlockReduceInt2::TempStorage reduce;
	} temp_storage;

	// sizeof(char) * (2 * max_sequence_bytes * blocksize + 2 * max_sequence_bytes);
	extern __shared__ unsigned int sharedmemory[];

	//set up shared memory pointers

    unsigned int* const subjectBackup = (unsigned int*)((char*)sharedmemory);
    unsigned int* const queryBackup = (unsigned int*)(((char*)subjectBackup) + max_sequence_bytes);
    unsigned int* const mySequence = ((unsigned int*)(((char*)queryBackup) + max_sequence_bytes)) + threadIdx.x;

	const int max_sequence_ints = max_sequence_bytes / sizeof(unsigned int);

	for(unsigned resultIndex = blockIdx.x; resultIndex < n_candidates * 2; resultIndex += gridDim.x) {

		const int queryIndex = resultIndex < n_candidates ? resultIndex : resultIndex - n_candidates;

		//find subjectindex
		int subjectIndex = 0;
		for(; subjectIndex < n_subjects; subjectIndex++) {
			if(queryIndex < candidates_per_subject_prefixsum[subjectIndex+1])
				break;
		}


		const int subjectbases = getSubjectLength(subjectIndex);
		const char* subjectptr = getSubjectPtr(subjectIndex);
        const int querybases = getCandidateLength(queryIndex);
		const char* candidateptr = getCandidatePtr(queryIndex);

        //save subject in shared memory
		for(int lane = threadIdx.x; lane < max_sequence_ints; lane += BLOCKSIZE) {
			subjectBackup[lane] = ((unsigned int*)(subjectptr))[lane];
		}

		//save query in shared memory
		for(int lane = threadIdx.x; lane < max_sequence_ints; lane += BLOCKSIZE) {
			queryBackup[lane] = ((unsigned int*)(candidateptr))[lane];
		}

		//queryIndex != resultIndex -> reverse complement
		if(threadIdx.x == 0 && queryIndex != resultIndex) {
			make_reverse_complement_inplace(queryBackup, querybases);
		}

		__syncthreads();

		//begin SHD algorithm

		const int subjectints = getNumBytes(subjectbases) / sizeof(unsigned int);
		const int queryints = getNumBytes(querybases) / sizeof(unsigned int);
		const int totalbases = subjectbases + querybases;
		const int minoverlap = max(min_overlap, int(float(subjectbases) * min_overlap_ratio));

		int bestScore = totalbases;                 // score is number of mismatches
		int bestShift = -querybases;                 // shift of query relative to subject. shift < 0 if query begins before subject

        //initialize threadlocal smem array with subject
        for(int lane = 0; lane < max_sequence_ints; lane += 1) {
            mySequence[no_bank_conflict_index(lane)] = subjectBackup[lane];
        }

        unsigned int* mySequence_hi = mySequence;
        unsigned int* mySequence_lo = mySequence + subjectints / 2 * BLOCKSIZE;

        int previousShift = std::numeric_limits<int>::min();

        for(int shift = threadIdx.x; shift < subjectbases - minoverlap + 1; shift += BLOCKSIZE) {
            const int overlapsize = min(querybases, subjectbases - shift) - max(-shift, 0);
            const int max_errors = int(float(overlapsize) * maxErrorRate);

            unsigned int* const shiftptr_hi = mySequence_hi;
            unsigned int* const shiftptr_lo = mySequence_lo;
            const int size = subjectints / 2;
            const int shiftamount = previousShift == std::numeric_limits<int>::min()
                                        ? shift
                                        : BLOCKSIZE;

            shiftEncodedBasesLeftBy(shiftptr_hi, size, shiftamount);
            shiftEncodedBasesLeftBy(shiftptr_lo, size, shiftamount);

            int score = hammingdistanceHiLo2(
                        mySequence_hi,
                        mySequence_lo,
                        queryBackup,
                        queryBackup + queryints / 2,
                        max(0, subjectbases - abs(shift)),
                        max(0, querybases - abs(shift)),
                        max_errors);

            score = (score < max_errors ?
                     score + totalbases - 2*overlapsize                         // non-overlapping regions count as mismatches
                     : std::numeric_limits<int>::max());                         // too many errors, discard

            if(score < bestScore) {
                bestScore = score;
                bestShift = shift;
            }

            previousShift = shift;
        }

        //initialize threadlocal smem array with query
        for(int lane = 0; lane < max_sequence_ints; lane += 1) {
            mySequence[no_bank_conflict_index(lane)] = queryBackup[lane];
        }

        mySequence_hi = mySequence;
        mySequence_lo = mySequence + queryints / 2 * BLOCKSIZE;

        previousShift = std::numeric_limits<int>::min();

        for(int shift = -1-int(threadIdx.x); shift >= -querybases + minoverlap; shift -= BLOCKSIZE) {
            const int overlapsize = min(querybases, subjectbases - shift) - max(-shift, 0);
            const int max_errors = int(float(overlapsize) * maxErrorRate);

            unsigned int* const shiftptr_hi = mySequence_hi;
            unsigned int* const shiftptr_lo = mySequence_lo;
            const int size = queryints / 2;
            const int shiftamount = previousShift == std::numeric_limits<int>::min()
                                        ? abs(shift) : BLOCKSIZE;

            shiftEncodedBasesLeftBy(shiftptr_hi, size, shiftamount);
            shiftEncodedBasesLeftBy(shiftptr_lo, size, shiftamount);

            int score = hammingdistanceHiLo2(
                            mySequence_hi,
                            mySequence_lo,
                            subjectBackup,
                            subjectBackup + subjectints / 2,
                            max(0, querybases - abs(shift)),
                            max(0, subjectbases - abs(shift)),
                            max_errors);

            score = (score < max_errors ?
                     score + totalbases - 2*overlapsize                         // non-overlapping regions count as mismatches
                     : std::numeric_limits<int>::max());                         // too many errors, discard

            if(score < bestScore) {
                bestScore = score;
                bestShift = shift;
            }

            previousShift = shift;
        }

		int2 myval = make_int2(bestScore, bestShift);

		myval = BlockReduceInt2(temp_storage.reduce).Reduce(myval, [](auto a, auto b){
					return a.x < b.x ? a : b;
				});
		__syncthreads();

		//make result
		if(threadIdx.x == 0) {
			bestScore = myval.x;
			bestShift = myval.y;

			const int queryoverlapbegin_incl = max(-bestShift, 0);
			const int queryoverlapend_excl = min(querybases, subjectbases - bestShift);
			const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
			const int opnr = bestScore - totalbases + 2*overlapsize;

			alignment_scores[resultIndex] = bestScore;
			alignment_overlaps[resultIndex] = overlapsize;
			alignment_shifts[resultIndex] = bestShift;
			alignment_nOps[resultIndex] = opnr;
			alignment_isValid[resultIndex] = (bestShift != -querybases);
		}
	}
}


template<class B, class GetSubjectPtr, class GetCandidatePtr, class GetSubjectLength, class GetCandidateLength>
void call_cuda_popcount_shifted_hamming_distance_with_revcompl_kernel_async(
			int* d_alignment_scores,
			int* d_alignment_overlaps,
			int* d_alignment_shifts,
			int* d_alignment_nOps,
			bool* d_alignment_isValid,
			const int* d_candidates_per_subject_prefixsum,
			int n_subjects,
			int n_queries,
			int max_sequence_bytes,
			int min_overlap,
			float maxErrorRate,
			float min_overlap_ratio,
			B getNumBytes,
			GetSubjectPtr getSubjectPtr,
			GetCandidatePtr getCandidatePtr,
			GetSubjectLength getSubjectLength,
			GetCandidateLength getCandidateLength,
			cudaStream_t stream,
			KernelLaunchHandle& handle){

	    #define mycall(blocksize) cuda_popcount_shifted_hamming_distance_with_revcompl_kernel<(blocksize)> \
	        <<<grid, block, smem, stream>>>( \
    		d_alignment_scores, \
    		d_alignment_overlaps, \
    		d_alignment_shifts, \
    		d_alignment_nOps, \
    		d_alignment_isValid, \
    		d_candidates_per_subject_prefixsum, \
    		n_subjects, \
    		n_queries, \
    		max_sequence_bytes, \
    		min_overlap, \
    		maxErrorRate, \
    		min_overlap_ratio, \
    		getNumBytes, \
    		getSubjectPtr, \
    		getCandidatePtr, \
    		getSubjectLength, \
    		getCandidateLength); CUERR;

    	const int blocksize = 32;
    	const std::size_t smem = sizeof(char) * (max_sequence_bytes * blocksize + 2 * max_sequence_bytes);

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::PopcountSHD);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    		#define getProp(blocksize) { \
            		KernelLaunchConfig kernelLaunchConfig; \
            		kernelLaunchConfig.threads_per_block = (blocksize); \
            		kernelLaunchConfig.smem = sizeof(char) * (max_sequence_bytes * (blocksize) + 2 * max_sequence_bytes); \
            		KernelProperties kernelProperties; \
            		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
            					cuda_popcount_shifted_hamming_distance_with_revcompl_kernel<(blocksize), B, \
            					                                                                GetSubjectPtr, GetCandidatePtr, \
            					                                                                GetSubjectLength, GetCandidateLength>, \
            					kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
            		mymap[kernelLaunchConfig] = kernelProperties; \
            }

    		getProp(32);
    		getProp(64);
    		getProp(96);
    		getProp(128);
    		getProp(160);
    		getProp(192);
    		getProp(224);
    		getProp(256);

    		const auto& kernelProperties = mymap[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

    		handle.kernelPropertiesMap[KernelId::PopcountSHD] = std::move(mymap);

    		#undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    	}

    	dim3 block(blocksize, 1, 1);
    	dim3 grid(std::min(n_queries*2, max_blocks_per_device), 1, 1);         // one block per candidate
        //dim3 grid(32,1,1);

    	switch(blocksize) {
    	case 32: mycall(32); break;
    	case 64: mycall(64); break;
    	case 96: mycall(96); break;
    	case 128: mycall(128); break;
    	case 160: mycall(160); break;
    	case 192: mycall(192); break;
    	case 224: mycall(224); break;
    	case 256: mycall(256); break;
    	default: throw std::runtime_error("Want to call cuda_popcount_shifted_hamming_distance_with_revcompl_kernel with wrong blocksize due to a bug.");
    	}

	    #undef mycall

}






template<int BLOCKSIZE, class B, class GetSubjectPtr, class GetCandidatePtr, class GetSubjectLength, class GetCandidateLength>
__global__
void
cuda_popcount_shifted_hamming_distance_with_revcompl_kernel_transposed(
			int* alignment_scores,
			int* alignment_overlaps,
			int* alignment_shifts,
			int* alignment_nOps,
			bool* alignment_isValid,
			//const int* subject_sequences_lengths,
			//const int* candidate_sequences_lengths,
			const int* candidates_per_subject_prefixsum,
			int n_subjects,
			int n_candidates,
			int max_sequence_bytes,
			int min_overlap,
			float maxErrorRate,
			float min_overlap_ratio,
			B getNumBytes,
			GetSubjectPtr getSubjectPtr,
			GetCandidatePtr getCandidatePtr,
			GetSubjectLength getSubjectLength,
			GetCandidateLength getCandidateLength){

	auto no_bank_conflict_index = [](int logical_index) -> int {
					      return logical_index * BLOCKSIZE;
	};

	auto make_reverse_complement_inplace = [&](unsigned int* sequence, int sequencelength){

						       auto reverse_complement_int = [](auto n) {
											     n = ((n >> 1) & 0x55555555) | ((n << 1) & 0xaaaaaaaa);
											     n = ((n >> 2) & 0x33333333) | ((n << 2) & 0xcccccccc);
											     n = ((n >> 4) & 0x0f0f0f0f) | ((n << 4) & 0xf0f0f0f0);
											     n = ((n >> 8) & 0x00ff00ff) | ((n << 8) & 0xff00ff00);
											     n = ((n >> 16) & 0x0000ffff) | ((n << 16) & 0xffff0000);
											     return ~n;
										     };

						       const int ints = getNumBytes(sequencelength) / sizeof(unsigned int);
						       const int unusedBitsInt = SDIV(sequencelength, 8 * sizeof(unsigned int)) * 8 * sizeof(unsigned int) - sequencelength;

						       unsigned int* const hi = sequence;
						       unsigned int* const lo = &sequence[no_bank_conflict_index(ints/2)];

						       const int intsPerHalf = SDIV(sequencelength, 8 * sizeof(unsigned int));
						       for(int i = 0; i < intsPerHalf/2; ++i) {
							       const unsigned int hifront = reverse_complement_int(hi[no_bank_conflict_index(i)]);
							       const unsigned int hiback = reverse_complement_int(hi[no_bank_conflict_index(intsPerHalf - 1 - i)]);
							       hi[no_bank_conflict_index(i)] = hiback;
							       hi[no_bank_conflict_index(intsPerHalf - 1 - i)] = hifront;

							       const unsigned int lofront = reverse_complement_int(lo[no_bank_conflict_index(i)]);
							       const unsigned int loback = reverse_complement_int(lo[no_bank_conflict_index(intsPerHalf - 1 - i)]);
							       lo[no_bank_conflict_index(i)] = loback;
							       lo[no_bank_conflict_index(intsPerHalf - 1 - i)] = lofront;
						       }
						       if(intsPerHalf % 2 == 1) {
							       const int middleindex = intsPerHalf/2;
							       hi[no_bank_conflict_index(middleindex)] = reverse_complement_int(hi[no_bank_conflict_index(middleindex)]);
							       lo[no_bank_conflict_index(middleindex)] = reverse_complement_int(lo[no_bank_conflict_index(middleindex)]);
						       }

						       if(unusedBitsInt != 0) {
							       for(int i = 0; i < intsPerHalf - 1; ++i) {
								       hi[no_bank_conflict_index(i)] = (hi[no_bank_conflict_index(i)] >> unusedBitsInt) | (hi[no_bank_conflict_index(i+1)] << (8 * sizeof(unsigned int) - unusedBitsInt));
								       lo[no_bank_conflict_index(i)] = (lo[no_bank_conflict_index(i)] >> unusedBitsInt) | (lo[no_bank_conflict_index(i+1)] << (8 * sizeof(unsigned int) - unusedBitsInt));
							       }

							       hi[no_bank_conflict_index(intsPerHalf - 1)] >>= unusedBitsInt;
							       lo[no_bank_conflict_index(intsPerHalf - 1)] >>= unusedBitsInt;
						       }
					       };

	auto shiftEncodedBasesLeftBy = [&](unsigned int* array, int size, int shiftamount){
					       const int completeInts = shiftamount / (8 * sizeof(unsigned int));

					       for(int i = 0; i < size - completeInts; i += 1) {
						       array[no_bank_conflict_index(i)] = array[no_bank_conflict_index(completeInts + i)];
					       }

					       for(int i = size - completeInts; i < size; i += 1) {
						       array[no_bank_conflict_index(i)] = 0;
					       }

					       shiftamount -= completeInts * 8 * sizeof(unsigned int);

					       for(int i = 0; i < size - completeInts - 1; i += 1) {
						       const unsigned int a = array[no_bank_conflict_index(i)];
						       const unsigned int b = array[no_bank_conflict_index(i+1)];

						       array[no_bank_conflict_index(i)] = (a >> shiftamount) | (b << (8 * sizeof(unsigned int) - shiftamount));
					       }

					       array[no_bank_conflict_index(size - completeInts - 1)] >>= shiftamount;
				       };

	auto hammingdistanceHiLo = [&](
		const unsigned int* lhi,
		const unsigned int* llo,
		const unsigned int* rhi,
		const unsigned int* rlo,
		int lhi_bitcount,
		int rhi_bitcount,
		int max_errors){

					   const int overlap_bitcount = min(lhi_bitcount, rhi_bitcount);

					   if(overlap_bitcount == 0)
						   return max_errors+1;

					   const int partitions = SDIV(overlap_bitcount, (8 * sizeof(unsigned int)));
					   const int remaining_bitcount = partitions * sizeof(unsigned int) * 8 - overlap_bitcount;

					   int result = 0;

					   for(int i = 0; i < partitions - 1 && result < max_errors; i += 1) {
						   const unsigned int hixor = lhi[no_bank_conflict_index(i)] ^ rhi[no_bank_conflict_index(i)];
						   const unsigned int loxor = llo[no_bank_conflict_index(i)] ^ rlo[no_bank_conflict_index(i)];
						   const unsigned int bits = hixor | loxor;
						   result += __popc(bits);
					   }

					   if(result >= max_errors)
						   return result;

					   // i == partitions - 1

					   const unsigned int mask = remaining_bitcount == 0 ? 0xFFFFFFFF : 0xFFFFFFFF >> (remaining_bitcount);
					   const unsigned int hixor = lhi[no_bank_conflict_index(partitions - 1)] ^ rhi[no_bank_conflict_index(partitions - 1)];
					   const unsigned int loxor = llo[no_bank_conflict_index(partitions - 1)] ^ rlo[no_bank_conflict_index(partitions - 1)];
					   const unsigned int bits = hixor | loxor;
					   result += __popc(bits & mask);

					   return result;
				   };

	// sizeof(char) * (2 * max_sequence_bytes * blocksize);
	extern __shared__ unsigned int sharedmemory[];


	//set up shared memory pointers
	char* const sharedSubject = (char*)(sharedmemory);
	char* const sharedQuery = (char*)(((char*)sharedSubject) + max_sequence_bytes * BLOCKSIZE);

	//set up shared memory per tile
	unsigned int* const mySubject = (unsigned int*)(sharedSubject) + threadIdx.x;
	unsigned int* const myQuery = (unsigned int*)(sharedQuery) + threadIdx.x;

	const int max_sequence_ints = max_sequence_bytes / sizeof(unsigned int);

	for(unsigned resultIndex = blockIdx.x * BLOCKSIZE + threadIdx.x; resultIndex < n_candidates * 2; resultIndex += BLOCKSIZE * gridDim.x) {

		const int queryIndex = resultIndex < n_candidates ? resultIndex : resultIndex - n_candidates;

		//find subjectindex
		int subjectIndex = 0;
		for(; subjectIndex < n_subjects; subjectIndex++) {
			if(queryIndex < candidates_per_subject_prefixsum[subjectIndex+1])
				break;
		}

		//save subject in shared memory
		const int subjectbases = getSubjectLength(subjectIndex);
		const char* subjectptr = getSubjectPtr(subjectIndex);
		const char* candidateptr = getCandidatePtr(queryIndex);

		for(int lane = 0; lane < max_sequence_ints; lane += 1) {
			mySubject[no_bank_conflict_index(lane)] = ((unsigned int*)(subjectptr))[lane];
		}

		//save query in shared memory
		const int querybases = getCandidateLength(queryIndex);

		for(int lane = 0; lane < max_sequence_ints; lane += 1) {
			myQuery[no_bank_conflict_index(lane)] = ((unsigned int*)(candidateptr))[lane];
		}

		//queryIndex != resultIndex -> reverse complement
		if(queryIndex != resultIndex) {
			make_reverse_complement_inplace(myQuery, querybases);
		}

		__syncthreads();

		//begin SHD algorithm

		const int subjectints = getNumBytes(subjectbases) / sizeof(unsigned int);
		const int queryints = getNumBytes(querybases) / sizeof(unsigned int);
		const int totalbases = subjectbases + querybases;
		const int minoverlap = max(min_overlap, int(float(subjectbases) * min_overlap_ratio));

		int bestScore = totalbases;                 // score is number of mismatches
		int bestShift = -querybases;                 // shift of query relative to subject. shift < 0 if query begins before subject

		unsigned int* const subjectdata_hi = mySubject;
		unsigned int* const subjectdata_lo = mySubject + subjectints / 2 * BLOCKSIZE;
		unsigned int* const querydata_hi = myQuery;
		unsigned int* const querydata_lo = myQuery + queryints / 2 * BLOCKSIZE;

        auto hammingdistanceofshift = [&](int shift, unsigned int* const shiftptr_hi, unsigned int* const shiftptr_lo, const int size){
            const int overlapsize = min(querybases, subjectbases - shift) - max(-shift, 0);
			const int max_errors = int(float(overlapsize) * maxErrorRate);
            const int shiftamount = abs(shift);
            shiftEncodedBasesLeftBy(shiftptr_hi, size, 1);
			shiftEncodedBasesLeftBy(shiftptr_lo, size, 1);
            int score = hammingdistanceHiLo(
						subjectdata_hi,
						subjectdata_lo,
						querydata_hi,
						querydata_lo,
						max(0, subjectbases - abs(shift)),
						max(0, querybases - abs(shift)),
						max_errors);

			score = (score < max_errors ?
			         score + totalbases - 2*overlapsize                         // non-overlapping regions count as mismatches
			         : std::numeric_limits<int>::max());                         // too many errors, discard

			return score;
        };

        for(int shift = 1; shift < subjectbases - minoverlap + 1; shift += 1){
            const int score = hammingdistanceofshift(shift, subjectdata_hi, subjectdata_lo, subjectints / 2);
            /*
                bestScore = (score < bestScore) * score + (score >= bestScore) * bestScore
                bestShift = (score < bestScore) * shift + (score >= bestScore) * bestShift
            */
            if(score < bestScore) {
				bestScore = score;
				bestShift = shift;
			}
        }

        //shift > 0 means that subject was shifted. now, restore subject to original sequence
        //since query is shifted for negative shift amounts.
        for(int lane = 0; lane < max_sequence_ints; lane += 1) {
            mySubject[lane] = ((unsigned int*)(subjectptr))[lane];
        }

        for(int shift = 0; shift >= -querybases + minoverlap; shift -= 1){
            const int score = hammingdistanceofshift(shift, querydata_hi, querydata_lo, queryints / 2);
            if(score < bestScore) {
				bestScore = score;
				bestShift = shift;
			}
        }

		const int queryoverlapbegin_incl = max(-bestShift, 0);
		const int queryoverlapend_excl = min(querybases, subjectbases - bestShift);
		const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
		const int opnr = bestScore - totalbases + 2*overlapsize;

		alignment_scores[resultIndex] = bestScore;
		alignment_overlaps[resultIndex] = overlapsize;
		alignment_shifts[resultIndex] = bestShift;
		alignment_nOps[resultIndex] = opnr;
		alignment_isValid[resultIndex] = (bestShift != -querybases);
	}
}





template<class B, class GetSubjectPtr, class GetCandidatePtr, class GetSubjectLength, class GetCandidateLength>
void call_cuda_popcount_shifted_hamming_distance_with_revcompl_kernel_transposed_async(
			int* d_alignment_scores,
			int* d_alignment_overlaps,
			int* d_alignment_shifts,
			int* d_alignment_nOps,
			bool* d_alignment_isValid,
			const int* d_candidates_per_subject_prefixsum,
			int n_subjects,
			int n_queries,
			int max_sequence_bytes,
			int min_overlap,
			float maxErrorRate,
			float min_overlap_ratio,
			B getNumBytes,
			GetSubjectPtr getSubjectPtr,
			GetCandidatePtr getCandidatePtr,
			GetSubjectLength getSubjectLength,
			GetCandidateLength getCandidateLength,
			cudaStream_t stream,
			KernelLaunchHandle& handle){

	    #define mycall(blocksize) cuda_popcount_shifted_hamming_distance_with_revcompl_kernel_transposed<(blocksize)> \
	        <<<grid, block, smem, stream>>>( \
    		d_alignment_scores, \
    		d_alignment_overlaps, \
    		d_alignment_shifts, \
    		d_alignment_nOps, \
    		d_alignment_isValid, \
    		d_candidates_per_subject_prefixsum, \
    		n_subjects, \
    		n_queries, \
    		max_sequence_bytes, \
    		min_overlap, \
    		maxErrorRate, \
    		min_overlap_ratio, \
    		getNumBytes, \
    		getSubjectPtr, \
    		getCandidatePtr, \
    		getSubjectLength, \
    		getCandidateLength); CUERR;

    	const int blocksize = 64;
    	const std::size_t smem = sizeof(char) * (2*max_sequence_bytes * blocksize + 2 * max_sequence_bytes);

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::PopcountSHDtransposed);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    		#define getProp(blocksize) { \
            		KernelLaunchConfig kernelLaunchConfig; \
            		kernelLaunchConfig.threads_per_block = (blocksize); \
            		kernelLaunchConfig.smem = sizeof(char) * (max_sequence_bytes * (blocksize) + 2 * max_sequence_bytes); \
            		KernelProperties kernelProperties; \
            		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
            					cuda_popcount_shifted_hamming_distance_with_revcompl_kernel_transposed<(blocksize), B, \
            					                                                                GetSubjectPtr, GetCandidatePtr, \
            					                                                                GetSubjectLength, GetCandidateLength>, \
            					kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
            		mymap[kernelLaunchConfig] = kernelProperties; \
            }

    		getProp(32);
    		getProp(64);
    		getProp(96);
    		getProp(128);
    		getProp(160);
    		getProp(192);
    		getProp(224);
    		getProp(256);

    		const auto& kernelProperties = mymap[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

    		handle.kernelPropertiesMap[KernelId::PopcountSHDtransposed] = std::move(mymap);

    		#undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    	}

    	dim3 block(blocksize, 1, 1);
    	dim3 grid(std::min(n_queries*2, max_blocks_per_device), 1, 1);         // one block per candidate

    	switch(blocksize) {
    	case 32: mycall(32); break;
    	case 64: mycall(64); break;
    	case 96: mycall(96); break;
    	case 128: mycall(128); break;
    	case 160: mycall(160); break;
    	case 192: mycall(192); break;
    	case 224: mycall(224); break;
    	case 256: mycall(256); break;
    	default: throw std::runtime_error("Want to call cuda_popcount_shifted_hamming_distance_with_revcompl_kernel_transposed with wrong blocksize due to a bug.");
    	}

	    #undef mycall

}



















template<class AlignmentComp, class GetSubjectLength, class GetCandidateLength>
__global__
void cuda_find_best_alignment_kernel_exp(
			BestAlignment_t* d_alignment_best_alignment_flags,
			int* d_alignment_scores,
			int* d_alignment_overlaps,
			int* d_alignment_shifts,
			int* d_alignment_nOps,
			bool* d_alignment_isValid,
			const int* d_candidates_per_subject_prefixsum,
			int n_subjects,
			int n_queries,
			float min_overlap_ratio,
			int min_overlap,
			AlignmentComp comp,
			GetSubjectLength getSubjectLength,
			GetCandidateLength getCandidateLength){

	for(unsigned resultIndex = threadIdx.x + blockDim.x * blockIdx.x; resultIndex < n_queries; resultIndex += gridDim.x * blockDim.x) {
		const unsigned fwdIndex = resultIndex;
		const unsigned revcIndex = resultIndex + n_queries;

		const int fwd_alignment_score = d_alignment_scores[fwdIndex];
		const int fwd_alignment_overlap = d_alignment_overlaps[fwdIndex];
		const int fwd_alignment_shift = d_alignment_shifts[fwdIndex];
		const int fwd_alignment_nops = d_alignment_nOps[fwdIndex];
		const bool fwd_alignment_isvalid = d_alignment_isValid[fwdIndex];

		const int revc_alignment_score = d_alignment_scores[revcIndex];
		const int revc_alignment_overlap = d_alignment_overlaps[revcIndex];
		const int revc_alignment_shift = d_alignment_shifts[revcIndex];
		const int revc_alignment_nops = d_alignment_nOps[revcIndex];
		const bool revc_alignment_isvalid = d_alignment_isValid[revcIndex];

		//const int querylength = d_candidate_sequences_lengths[resultIndex];
		const int querylength = getCandidateLength(resultIndex);

		//find subjectindex
		int subjectIndex = 0;
		for(; subjectIndex < n_subjects; subjectIndex++) {
			if(resultIndex < d_candidates_per_subject_prefixsum[subjectIndex+1])
				break;
		}

		//const int subjectlength = d_subject_sequences_lengths[subjectIndex];
		const int subjectlength = getSubjectLength(subjectIndex);

		const BestAlignment_t flag = comp(fwd_alignment_overlap,
					revc_alignment_overlap,
					fwd_alignment_nops,
					revc_alignment_nops,
					fwd_alignment_isvalid,
					revc_alignment_isvalid,
					subjectlength,
					querylength);

		d_alignment_best_alignment_flags[resultIndex] = flag;

		d_alignment_scores[resultIndex] = flag == BestAlignment_t::Forward ? fwd_alignment_score : revc_alignment_score;
		d_alignment_overlaps[resultIndex] = flag == BestAlignment_t::Forward ? fwd_alignment_overlap : revc_alignment_overlap;
		d_alignment_shifts[resultIndex] = flag == BestAlignment_t::Forward ? fwd_alignment_shift : revc_alignment_shift;
		d_alignment_nOps[resultIndex] = flag == BestAlignment_t::Forward ? fwd_alignment_nops : revc_alignment_nops;
		d_alignment_isValid[resultIndex] = flag == BestAlignment_t::Forward ? fwd_alignment_isvalid : revc_alignment_isvalid;
	}
}

template<class AlignmentComp, class GetSubjectLength, class GetCandidateLength>
void call_cuda_find_best_alignment_kernel_async_exp(
			BestAlignment_t* d_alignment_best_alignment_flags,
			int* d_alignment_scores,
			int* d_alignment_overlaps,
			int* d_alignment_shifts,
			int* d_alignment_nOps,
			bool* d_alignment_isValid,
			const int* d_candidates_per_subject_prefixsum,
			int n_subjects,
			int n_queries,
			float min_overlap_ratio,
			int min_overlap,
			AlignmentComp d_comp,
			GetSubjectLength getSubjectLength,
			GetCandidateLength getCandidateLength,
			cudaStream_t stream,
			KernelLaunchHandle& handle){

	const int blocksize = 128;
	const std::size_t smem = 0;

	int max_blocks_per_device = 1;

	KernelLaunchConfig kernelLaunchConfig;
	kernelLaunchConfig.threads_per_block = blocksize;
	kernelLaunchConfig.smem = smem;

	auto iter = handle.kernelPropertiesMap.find(KernelId::FindBestAlignmentExp);
	if(iter == handle.kernelPropertiesMap.end()) {

		std::map<KernelLaunchConfig, KernelProperties> mymap;

	    #define getProp(blocksize) { \
		KernelLaunchConfig kernelLaunchConfig; \
		kernelLaunchConfig.threads_per_block = (blocksize); \
		kernelLaunchConfig.smem = 0; \
		KernelProperties kernelProperties; \
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
					cuda_find_best_alignment_kernel_exp<AlignmentComp, GetSubjectLength, GetCandidateLength>, \
					kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
		mymap[kernelLaunchConfig] = kernelProperties; \
}

		getProp(32);
		getProp(64);
		getProp(96);
		getProp(128);
		getProp(160);
		getProp(192);
		getProp(224);
		getProp(256);

		const auto& kernelProperties = mymap[kernelLaunchConfig];
		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

		handle.kernelPropertiesMap[KernelId::FindBestAlignmentExp] = std::move(mymap);

	    #undef getProp
	}else{
		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
	}

	dim3 block(blocksize,1,1);
	dim3 grid(std::min(max_blocks_per_device, SDIV(n_queries, blocksize)), 1, 1);

	cuda_find_best_alignment_kernel_exp<<<grid, block, smem, stream>>>(
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
		d_comp,
		getSubjectLength,
		getCandidateLength); CUERR;

}





template<int BLOCKSIZE, class IndexGenerator_t>
__global__
void cuda_filter_alignments_by_mismatchratio_kernel(
			BestAlignment_t* d_alignment_best_alignment_flags,
			const int* d_alignment_overlaps,
			const int* d_alignment_nOps,
			IndexGenerator_t d_indices,
			const int* d_indices_per_subject,
			const int* d_indices_per_subject_prefixsum,
			int n_subjects,
			int n_candidates,
			const int* d_num_indices,
			float mismatchratioBaseFactor,
			float goodAlignmentsCountThreshold){

	using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;

	__shared__ union {
		typename BlockReduceInt::TempStorage intreduce;
		int broadcast[3];
	} temp_storage;

	/*if(threadIdx.x == 0){
	    printf("n_subjects %d\n", n_subjects);
	   }

	   printf("blockIdx.x %d\n", blockIdx.x);*/

	for(int subjectindex = blockIdx.x; subjectindex < n_subjects; subjectindex += gridDim.x) {

		//const int my_n_indices = d_indices_per_subject[subjectindex];
		const int my_n_indices = d_indices_per_subject_prefixsum[subjectindex+1] - d_indices_per_subject_prefixsum[subjectindex];

		//const int* my_indices = d_indices + d_indices_per_subject_prefixsum[subjectindex];
		const int firstIndex = d_indices_per_subject_prefixsum[subjectindex];

		//printf("subjectindex %d\n", subjectindex);

		int counts[3]{0,0,0};

		//if(threadIdx.x == 0){
		//    printf("my_n_indices %d\n", my_n_indices);
		//}

		for(int index = threadIdx.x; index < my_n_indices; index += blockDim.x) {
			//const int candidate_index = my_indices[index];
			const int candidate_index = d_indices[firstIndex + index];
			if(d_alignment_best_alignment_flags[candidate_index] != BestAlignment_t::None) {

				const int alignment_overlap = d_alignment_overlaps[candidate_index];
				const int alignment_nops = d_alignment_nOps[candidate_index];

				const float mismatchratio = float(alignment_nops) / alignment_overlap;
				if(mismatchratio >= 4 * mismatchratioBaseFactor) {
					d_alignment_best_alignment_flags[candidate_index] = BestAlignment_t::None;
				}else{

						#pragma unroll
					for(int i = 2; i <= 4; i++) {
						counts[i-2] += (mismatchratio < i * mismatchratioBaseFactor);
					}
				}

			}
		}

		//accumulate counts over block
			#pragma unroll
		for(int i = 0; i < 3; i++) {
			counts[i] = BlockReduceInt(temp_storage.intreduce).Sum(counts[i]);
			__syncthreads();
		}

		//broadcast accumulated counts to block
		if(threadIdx.x == 0) {
			#pragma unroll
			for(int i = 0; i < 3; i++) {
				temp_storage.broadcast[i] = counts[i];
				//printf("count[%d] = %d\n", i, counts[i]);
			}
			//printf("mismatchratioBaseFactor %f, goodAlignmentsCountThreshold %f\n", mismatchratioBaseFactor, goodAlignmentsCountThreshold);
		}

		__syncthreads();

	    #pragma unroll
		for(int i = 0; i < 3; i++) {
			counts[i] = temp_storage.broadcast[i];
		}

		float mismatchratioThreshold = 0;
		if (counts[0] >= goodAlignmentsCountThreshold) {
			mismatchratioThreshold = 2 * mismatchratioBaseFactor;
		} else if (counts[1] >= goodAlignmentsCountThreshold) {
			mismatchratioThreshold = 3 * mismatchratioBaseFactor;
		} else if (counts[2] >= goodAlignmentsCountThreshold) {
			mismatchratioThreshold = 4 * mismatchratioBaseFactor;
		} else {
			mismatchratioThreshold = -1.0f;                         //this will invalidate all alignments for subject
			//mismatchratioThreshold = 4 * mismatchratioBaseFactor; //use alignments from every bin
			//mismatchratioThreshold = 1.1f;
		}

		// Invalidate all alignments for subject with mismatchratio >= mismatchratioThreshold
		for(int index = threadIdx.x; index < my_n_indices; index += blockDim.x) {
			//const int candidate_index = my_indices[index];
			const int candidate_index = d_indices[firstIndex + index];
			if(d_alignment_best_alignment_flags[candidate_index] != BestAlignment_t::None) {

				const int alignment_overlap = d_alignment_overlaps[candidate_index];
				const int alignment_nops = d_alignment_nOps[candidate_index];

				const float mismatchratio = float(alignment_nops) / alignment_overlap;

				const bool remove = mismatchratio >= mismatchratioThreshold;
				if(remove)
					d_alignment_best_alignment_flags[candidate_index] = BestAlignment_t::None;

			}
		}
	}
}

template<class IndexGenerator_t>
void call_cuda_filter_alignments_by_mismatchratio_kernel_async(
			BestAlignment_t* d_alignment_best_alignment_flags,
			const int* d_alignment_overlaps,
			const int* d_alignment_nOps,
			IndexGenerator_t d_indices,
			const int* d_indices_per_subject,
			const int* d_indices_per_subject_prefixsum,
			int n_subjects,
			int n_candidates,
			const int* d_num_indices,
			float mismatchratioBaseFactor,
			float goodAlignmentsCountThreshold,
			cudaStream_t stream,
			KernelLaunchHandle& handle){

	const int blocksize = 128;
	const std::size_t smem = 0;

	int max_blocks_per_device = 1;

	KernelLaunchConfig kernelLaunchConfig;
	kernelLaunchConfig.threads_per_block = blocksize;
	kernelLaunchConfig.smem = smem;

	auto iter = handle.kernelPropertiesMap.find(KernelId::FilterAlignmentsByMismatchRatio);
	if(iter == handle.kernelPropertiesMap.end()) {

		std::map<KernelLaunchConfig, KernelProperties> mymap;

	    #define getProp(blocksize) { \
		KernelLaunchConfig kernelLaunchConfig; \
		kernelLaunchConfig.threads_per_block = (blocksize); \
		kernelLaunchConfig.smem = 0; \
		KernelProperties kernelProperties; \
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
					cuda_filter_alignments_by_mismatchratio_kernel<(blocksize), IndexGenerator_t>, \
					kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
		mymap[kernelLaunchConfig] = kernelProperties; \
}

		getProp(32);
		getProp(64);
		getProp(96);
		getProp(128);
		getProp(160);
		getProp(192);
		getProp(224);
		getProp(256);

		const auto& kernelProperties = mymap[kernelLaunchConfig];
		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

		handle.kernelPropertiesMap[KernelId::FilterAlignmentsByMismatchRatio] = std::move(mymap);

	    #undef getProp
	}else{
		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
	}

	dim3 block(blocksize, 1, 1);
	dim3 grid(std::min(max_blocks_per_device, n_subjects));

	#define mycall(blocksize) cuda_filter_alignments_by_mismatchratio_kernel<(blocksize)> \
	        <<<grid, block, smem, stream>>>( \
		d_alignment_best_alignment_flags, \
		d_alignment_overlaps, \
		d_alignment_nOps, \
		d_indices, \
		d_indices_per_subject, \
		d_indices_per_subject_prefixsum, \
		n_subjects, \
		n_candidates, \
		d_num_indices, \
		mismatchratioBaseFactor, \
		goodAlignmentsCountThreshold); CUERR;

	switch(blocksize) {
	case 32: mycall(32); break;
	case 64: mycall(64); break;
	case 96: mycall(96); break;
	case 128: mycall(128); break;
	case 160: mycall(160); break;
	case 192: mycall(192); break;
	case 224: mycall(224); break;
	case 256: mycall(256); break;
	default: mycall(256); break;
	}

		#undef mycall
}



template<int BLOCKSIZE, class GetSubjectLength, class GetCandidateLength>
__global__
void msa_init_kernel_exp(
			MSAColumnProperties* __restrict__ msa_column_properties,
			const int* __restrict__ alignment_shifts,
			const BestAlignment_t* __restrict__ alignment_best_alignment_flags,
			const int* __restrict__ indices,
			const int* __restrict__ indices_per_subject,
			const int* __restrict__ indices_per_subject_prefixsum,
			int n_subjects,
			GetSubjectLength getSubjectLength,
			GetCandidateLength getCandidateLength){

	using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;

	__shared__ union {
		typename BlockReduceInt::TempStorage reduce;
	} temp_storage;

	for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x) {
		MSAColumnProperties* const properties_ptr = msa_column_properties + subjectIndex;

		// We only want to consider the candidates with good alignments. the indices of those were determined in a previous step
		const int num_indices_for_this_subject = indices_per_subject[subjectIndex];
		const int* const indices_for_this_subject = indices + indices_per_subject_prefixsum[subjectIndex];

		const int subjectLength = getSubjectLength(subjectIndex);
		int startindex = 0;
		int endindex = getSubjectLength(subjectIndex);

		for(int index = threadIdx.x; index < num_indices_for_this_subject; index += blockDim.x) {
			const int queryIndex = indices_for_this_subject[index];

			const int shift = alignment_shifts[queryIndex];
			const BestAlignment_t flag = alignment_best_alignment_flags[queryIndex];
			const int queryLength = getCandidateLength(subjectIndex, index);

			if(flag != BestAlignment_t::None) {
				const int queryEndsAt = queryLength + shift;
				startindex = min(startindex, shift);
				endindex = max(endindex, queryEndsAt);
			}
		}

		startindex = BlockReduceInt(temp_storage.reduce).Reduce(startindex, cub::Min());
		__syncthreads();

		endindex = BlockReduceInt(temp_storage.reduce).Reduce(endindex, cub::Max());
		__syncthreads();

		if(threadIdx.x == 0) {
			MSAColumnProperties my_columnproperties;

			my_columnproperties.startindex = startindex;
			my_columnproperties.endindex = endindex;
			my_columnproperties.columnsToCheck = my_columnproperties.endindex - my_columnproperties.startindex;
			my_columnproperties.subjectColumnsBegin_incl = max(-my_columnproperties.startindex, 0);
			my_columnproperties.subjectColumnsEnd_excl = my_columnproperties.subjectColumnsBegin_incl + subjectLength;

			*properties_ptr = my_columnproperties;
		}

	}
}

template<class GetSubjectLength, class GetCandidateLength>
void call_msa_init_kernel_async_exp(
			MSAColumnProperties* d_msa_column_properties,
			const int* d_alignment_shifts,
			const BestAlignment_t* d_alignment_best_alignment_flags,
			const int* d_indices,
			const int* d_indices_per_subject,
			const int* d_indices_per_subject_prefixsum,
			int n_subjects,
			int n_queries,
			GetSubjectLength getSubjectLength,
			GetCandidateLength getCandidateLength,
			cudaStream_t stream,
			KernelLaunchHandle& handle){

	const int blocksize = 128;
	const std::size_t smem = 0;

	int max_blocks_per_device = 1;

	KernelLaunchConfig kernelLaunchConfig;
	kernelLaunchConfig.threads_per_block = blocksize;
	kernelLaunchConfig.smem = smem;

	auto iter = handle.kernelPropertiesMap.find(KernelId::MSAInitExp);
	if(iter == handle.kernelPropertiesMap.end()) {

		std::map<KernelLaunchConfig, KernelProperties> mymap;

	    #define getProp(blocksize) { \
		KernelLaunchConfig kernelLaunchConfig; \
		kernelLaunchConfig.threads_per_block = (blocksize); \
		kernelLaunchConfig.smem = 0; \
		KernelProperties kernelProperties; \
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
					msa_init_kernel_exp<(blocksize), GetSubjectLength, GetCandidateLength>, \
					kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
		mymap[kernelLaunchConfig] = kernelProperties; \
}

		getProp(32);
		getProp(64);
		getProp(96);
		getProp(128);
		getProp(160);
		getProp(192);
		getProp(224);
		getProp(256);

		const auto& kernelProperties = mymap[kernelLaunchConfig];
		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

		handle.kernelPropertiesMap[KernelId::MSAInitExp] = std::move(mymap);

	    #undef getProp
	}else{
		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
	}

	dim3 block(blocksize, 1, 1);
	dim3 grid(std::min(max_blocks_per_device, n_subjects), 1, 1);

		#define mycall(blocksize) msa_init_kernel_exp<(blocksize)><<<grid, block, 0, stream>>>(d_msa_column_properties, \
	                                                                                               d_alignment_shifts, \
	                                                                                               d_alignment_best_alignment_flags, \
	                                                                                               d_indices, \
	                                                                                               d_indices_per_subject, \
	                                                                                               d_indices_per_subject_prefixsum, \
	                                                                                               n_subjects, \
	                                                                                               getSubjectLength, \
	                                                                                               getCandidateLength); CUERR;

	switch(blocksize) {
	case 1: mycall(1); break;
	case 32: mycall(32); break;
	case 64: mycall(64); break;
	case 96: mycall(96); break;
	case 128: mycall(128); break;
	case 160: mycall(160); break;
	case 192: mycall(192); break;
	case 224: mycall(224); break;
	case 256: mycall(256); break;
	default: mycall(256); break;
	}

		#undef mycall


}



template<class Accessor, class RevCompl, class GetSubjectPtr, class GetCandidatePtr, class GetSubjectQualityPtr, class GetCandidateQualityPtr,
         class GetSubjectLength, class GetCandidateLength>
__global__
void msa_add_sequences_kernel_exp(
			char* __restrict__ d_multiple_sequence_alignments,
			float* __restrict__ d_multiple_sequence_alignment_weights,
			const int* __restrict__ d_alignment_shifts,
			const BestAlignment_t* __restrict__ d_alignment_best_alignment_flags,
			const int* __restrict__ d_subject_sequences_lengths,
			const int* __restrict__ d_candidate_sequences_lengths,
			const int* __restrict__ d_alignment_overlaps,
			const int* __restrict__ d_alignment_nOps,
			const MSAColumnProperties*  __restrict__ d_msa_column_properties,
			const int* __restrict__ d_candidates_per_subject_prefixsum,
			const int* __restrict__ d_indices,
			const int* __restrict__ d_indices_per_subject,
			const int* __restrict__ d_indices_per_subject_prefixsum,
			int n_subjects,
			int n_queries,
			const int* __restrict__ d_num_indices,
			bool canUseQualityScores,
			float desiredAlignmentMaxErrorRate,
			int maximum_sequence_length,
			int max_sequence_bytes,
			size_t quality_pitch,
			size_t msa_row_pitch,
			size_t msa_weights_row_pitch,
			Accessor get,
			RevCompl make_unpacked_reverse_complement_inplace,
			GetSubjectPtr getSubjectPtr,
			GetCandidatePtr getCandidatePtr,
			GetSubjectQualityPtr getSubjectQualityPtr,
			GetCandidateQualityPtr getCandidateQualityPtr,
			GetSubjectLength getSubjectLength,
			GetCandidateLength getCandidateLength){

    /*auto reverse_float = [](float* sequence, int length){

        for(int i = 0; i < length/2; i++) {
            const float front = sequence[i];
            const float back = sequence[length - 1 - i];
            sequence[i] = back;
            sequence[length - 1 - i] = front;
        }

        if(length % 2 == 1) {
            ;                         // when sequencelength is odd, the center remains unchanged
        }
    };*/

	extern __shared__ float sharedmem[];

	float* const sharedWeights = (float*)sharedmem;
	char* const sharedSequence = (char*)(sharedWeights + maximum_sequence_length);

	const size_t msa_weights_row_pitch_floats = msa_weights_row_pitch / sizeof(float);
	const int n_indices = *d_num_indices;

	//copy each subject into the top row of its multiple sequence alignment
	for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x) {
		const int subjectColumnsBegin_incl = d_msa_column_properties[subjectIndex].subjectColumnsBegin_incl;
		const int candidates_before_this_subject = d_indices_per_subject_prefixsum[subjectIndex];

		const unsigned offset1 = msa_row_pitch * (subjectIndex + candidates_before_this_subject);
		const unsigned offset2 = msa_weights_row_pitch_floats * (subjectIndex + candidates_before_this_subject);

		char* const multiple_sequence_alignment = d_multiple_sequence_alignments + offset1;
		float* const multiple_sequence_alignment_weight = d_multiple_sequence_alignment_weights + offset2;

		const int subjectLength = getSubjectLength(subjectIndex);
		const char* const subject = getSubjectPtr(subjectIndex);
		const char* const subjectQualityScore = getSubjectQualityPtr(subjectIndex);

		for(int i = threadIdx.x; i < subjectLength; i+= blockDim.x) {
			multiple_sequence_alignment[subjectColumnsBegin_incl + i] = get(subject, subjectLength, i);
			multiple_sequence_alignment_weight[subjectColumnsBegin_incl + i] = canUseQualityScores ?
			                                                                   (float)d_qscore_to_weight[(unsigned char)subjectQualityScore[i]]
			                                                                   : 1.0f;
		}
	}

	// copy each query into the multiple sequence alignment of its subject
	for(unsigned index = blockIdx.x; index < n_indices; index += gridDim.x) {

		const int queryIndex = d_indices[index];

		const int shift = d_alignment_shifts[queryIndex];
		const BestAlignment_t flag = d_alignment_best_alignment_flags[queryIndex];

		//find subjectindex
		int subjectIndex = 0;
		for(; subjectIndex < n_subjects; subjectIndex++) {
			if(queryIndex < d_candidates_per_subject_prefixsum[subjectIndex+1])
				break;
		}

		const int subjectColumnsBegin_incl = d_msa_column_properties[subjectIndex].subjectColumnsBegin_incl;
		const int localQueryIndex = index - d_indices_per_subject_prefixsum[subjectIndex];
		const int defaultcolumnoffset = subjectColumnsBegin_incl + shift;

		const int candidates_before_this_subject = d_indices_per_subject_prefixsum[subjectIndex];

		//printf("index %d, subjectindex %d, queryindex %d, shift %d, defaultcolumnoffset %d, candidates_before_this_subject %d, localQueryIndex %d\n", index, subjectIndex, queryIndex, shift, defaultcolumnoffset, candidates_before_this_subject, localQueryIndex);

		const unsigned offset1 = msa_row_pitch * (subjectIndex + candidates_before_this_subject);
		const unsigned offset2 = msa_weights_row_pitch_floats * (subjectIndex + candidates_before_this_subject);
		//const int rowOffset = (subjectIndex + candidates_before_this_subject);

		char* const multiple_sequence_alignment = d_multiple_sequence_alignments + offset1;
		float* const multiple_sequence_alignment_weight = d_multiple_sequence_alignment_weights + offset2;

		const char* const query = getCandidatePtr(queryIndex);
		const int queryLength = getCandidateLength(index);
		const char* const queryQualityScore = getCandidateQualityPtr(index);

		const int query_alignment_overlap = d_alignment_overlaps[queryIndex];
		const int query_alignment_nops = d_alignment_nOps[queryIndex];

		const float defaultweight = 1.0f - sqrtf(query_alignment_nops
					/ (query_alignment_overlap * desiredAlignmentMaxErrorRate));

		assert(flag != BestAlignment_t::None);                 // indices should only be pointing to valid alignments
#if 0
		//copy query into msa
		const int row = 1 + localQueryIndex;
		for(int i = threadIdx.x; i < queryLength; i+= blockDim.x) {
			const int globalIndex = defaultcolumnoffset + i;

			multiple_sequence_alignment[row * msa_row_pitch + globalIndex] = get_as_nucleotide(query, queryLength, i);

			multiple_sequence_alignment_weight[row * msa_weights_row_pitch_floats + globalIndex]
			        = canUseQualityScores ?
			          (float)d_qscore_to_weight[(unsigned char)queryQualityScore[i]] * defaultweight
			          : 1.0f;
		}

		__syncthreads();                 // need to wait until current row is written by all threads

		if(threadIdx.x == 0 && flag == BestAlignment_t::ReverseComplement) {
			make_unpacked_reverse_complement_inplace((std::uint8_t*)multiple_sequence_alignment + row * msa_row_pitch + defaultcolumnoffset,
						queryLength);
			//reverse quality weights. if canUseQualityScores == false, then all weights are 1.0f and do not need to be reversed
			if(canUseQualityScores) {
				reverse_float(multiple_sequence_alignment_weight + row * msa_weights_row_pitch_floats + defaultcolumnoffset, queryLength);
			}
		}
#else
		//copy query into msa
		if(flag == BestAlignment_t::Forward) {
			const int row = 1 + localQueryIndex;
			for(int i = threadIdx.x; i < queryLength; i+= blockDim.x) {
				const int globalIndex = defaultcolumnoffset + i;

				multiple_sequence_alignment[row * msa_row_pitch + globalIndex] = get(query, queryLength, i);

				multiple_sequence_alignment_weight[row * msa_weights_row_pitch_floats + globalIndex]
				        = canUseQualityScores ?
				          (float)d_qscore_to_weight[(unsigned char)queryQualityScore[i]] * defaultweight
				          : 1.0f;
			}
		}else{
			for(int i = threadIdx.x; i < queryLength; i+= blockDim.x) {
				sharedSequence[i] = get(query, queryLength, i);
				sharedWeights[i] = canUseQualityScores ?
				                   (float)d_qscore_to_weight[(unsigned char)queryQualityScore[i]] * defaultweight
				                   : 1.0f;
			}

			__syncthreads();

			/*if(threadIdx.x == 0) {
				//make_unpacked_reverse_complement_inplace((std::uint8_t*)sharedSequence, queryLength);

                auto make_reverse_complement_byte = [](std::uint8_t in) -> std::uint8_t{
                    constexpr std::uint8_t mask = 0x03;
                    return (~in & mask);
                };

                const int bytes = queryLength;

                for(int i = 0; i < bytes/2; i++){
                    const std::uint8_t front = make_reverse_complement_byte(sharedSequence[i]);
                    const std::uint8_t back = make_reverse_complement_byte(sharedSequence[bytes - 1 - i]);
                    sharedSequence[i] = back;
                    sharedSequence[bytes - 1 - i] = front;
                }

                if(bytes % 2 == 1){
                    const int middleindex = bytes/2;
                    sharedSequence[middleindex] = make_reverse_complement_byte(sharedSequence[middleindex]);
                }

				//reverse quality weights. if canUseQualityScores == false, then all weights are 1.0f and do not need to be reversed
				if(canUseQualityScores) {
					reverse_float(sharedWeights, queryLength);
				}
			}

			__syncthreads();*/
            auto make_reverse_complement_byte = [](std::uint8_t in) -> std::uint8_t{
                constexpr std::uint8_t mask = 0x03;
                return (~in & mask);
            };

            const int bytes = queryLength;
            //make reverse complement of shared sequence
            for(int i = threadIdx.x; i < bytes/2; i += blockDim.x){
                const std::uint8_t front = make_reverse_complement_byte(sharedSequence[i]);
                const std::uint8_t back = make_reverse_complement_byte(sharedSequence[bytes - 1 - i]);
                sharedSequence[i] = back;
                sharedSequence[bytes - 1 - i] = front;
            }

            if(bytes % 2 == 1){
                const int middleindex = bytes/2;
                sharedSequence[middleindex] = make_reverse_complement_byte(sharedSequence[middleindex]);
            }
            //reverse quality weights. if canUseQualityScores == false, then all weights are 1.0f and do not need to be reversed
            if(canUseQualityScores){
                const int floats = queryLength;
                for(int i = threadIdx.x; i < floats/2; i += blockDim.x){
                    const float front = sharedWeights[i];
                    const float back = sharedWeights[bytes - 1 - i];
                    sharedWeights[i] = back;
                    sharedWeights[bytes - 1 - i] = front;
                }
                //for odd length, the middle position remains unchanged
            }

            __syncthreads();


			const int row = 1 + localQueryIndex;
			for(int i = threadIdx.x; i < queryLength; i+= blockDim.x) {
				const int globalIndex = defaultcolumnoffset + i;

				multiple_sequence_alignment[row * msa_row_pitch + globalIndex] = sharedSequence[i];

				multiple_sequence_alignment_weight[row * msa_weights_row_pitch_floats + globalIndex] = sharedWeights[i];
			}

			__syncthreads();
		}

#endif

	}
}

template<class Accessor, class RevCompl, class GetSubjectPtr, class GetCandidatePtr, class GetSubjectQualityPtr, class GetCandidateQualityPtr,
         class GetSubjectLength, class GetCandidateLength>
void call_msa_add_sequences_kernel_exp_async(
			char* d_multiple_sequence_alignments,
			float* d_multiple_sequence_alignment_weights,
			const int* d_alignment_shifts,
			const BestAlignment_t* d_alignment_best_alignment_flags,
			const int* d_subject_sequences_lengths,
			const int* d_candidate_sequences_lengths,
			const int* d_alignment_overlaps,
			const int* d_alignment_nOps,
			const MSAColumnProperties*  d_msa_column_properties,
			const int* d_candidates_per_subject_prefixsum,
			const int* d_indices,
			const int* d_indices_per_subject,
			const int* d_indices_per_subject_prefixsum,
			int n_subjects,
			int n_queries,
			const int* d_num_indices,
			bool canUseQualityScores,
			float desiredAlignmentMaxErrorRate,
			int maximum_sequence_length,
			int max_sequence_bytes,
			size_t quality_pitch,
			size_t msa_row_pitch,
			size_t msa_weights_row_pitch,
			Accessor get_as_nucleotide,
			RevCompl make_unpacked_reverse_complement_inplace,
			GetSubjectPtr getSubjectPtr,
			GetCandidatePtr getCandidatePtr,
			GetSubjectQualityPtr getSubjectQualityPtr,
			GetCandidateQualityPtr getCandidateQualityPtr,
			GetSubjectLength getSubjectLength,
			GetCandidateLength getCandidateLength,
			cudaStream_t stream,
			KernelLaunchHandle& handle){




	const int blocksize = 128;
	const std::size_t smem = sizeof(char) * maximum_sequence_length + sizeof(float) * maximum_sequence_length;

	int max_blocks_per_device = 1;

	KernelLaunchConfig kernelLaunchConfig;
	kernelLaunchConfig.threads_per_block = blocksize;
	kernelLaunchConfig.smem = smem;

	auto iter = handle.kernelPropertiesMap.find(KernelId::MSAAddSequences);
	if(iter == handle.kernelPropertiesMap.end()) {

		std::map<KernelLaunchConfig, KernelProperties> mymap;

	    #define getProp(blocksize) { \
		KernelLaunchConfig kernelLaunchConfig; \
		kernelLaunchConfig.threads_per_block = (blocksize); \
		kernelLaunchConfig.smem = sizeof(char) * maximum_sequence_length + sizeof(float) * maximum_sequence_length; \
		KernelProperties kernelProperties; \
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
					msa_add_sequences_kernel_exp<Accessor, RevCompl, GetSubjectPtr, GetCandidatePtr, \
					                             GetSubjectQualityPtr, GetCandidateQualityPtr, \
					                             GetSubjectLength, GetCandidateLength>, \
					kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
		mymap[kernelLaunchConfig] = kernelProperties; \
}

		getProp(32);
		getProp(64);
		getProp(96);
		getProp(128);
		getProp(160);
		getProp(192);
		getProp(224);
		getProp(256);

		const auto& kernelProperties = mymap[kernelLaunchConfig];
		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM * 2;

		handle.kernelPropertiesMap[KernelId::MSAAddSequences] = std::move(mymap);

	    #undef getProp
	}else{
		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM * 2;
		//std::cout << max_blocks_per_device << " = " << handle.deviceProperties.multiProcessorCount << " * " << kernelProperties.max_blocks_per_SM << std::endl;
	}

	dim3 block(blocksize, 1, 1);
	//d_num_indices blocks will perform work. n_queries is an upper bound of d_num_indices
	dim3 grid(std::min(n_queries, max_blocks_per_device), 1, 1);

	msa_add_sequences_kernel_exp<<<grid, block, smem, stream>>>(d_multiple_sequence_alignments,
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
	                                                            d_num_indices,
	                                                            canUseQualityScores,
	                                                            desiredAlignmentMaxErrorRate,
	                                                            maximum_sequence_length,
	                                                            max_sequence_bytes,
	                                                            quality_pitch,
	                                                            msa_row_pitch,
	                                                            msa_weights_row_pitch,
	                                                            get_as_nucleotide,
	                                                            make_unpacked_reverse_complement_inplace,
	                                                            getSubjectPtr,
	                                                            getCandidatePtr,
	                                                            getSubjectQualityPtr,
	                                                            getCandidateQualityPtr,
	                                                            getSubjectLength,
	                                                            getCandidateLength); CUERR;
}














template<int BLOCKSIZE, class RevCompl, class GetCandidateLength>
__global__
void msa_correct_candidates_kernel_exp(
			const char* __restrict__ d_consensus,
			const float* __restrict__ d_support,
			const int* __restrict__ d_coverage,
			const int* __restrict__ d_origCoverages,
			const char* __restrict__ d_multiple_sequence_alignments,
			const MSAColumnProperties* __restrict__ d_msa_column_properties,
			const int* __restrict__ d_indices,
			const int* __restrict__ d_indices_per_subject,
			const int* __restrict__ d_indices_per_subject_prefixsum,
			const int* __restrict__ d_high_quality_subject_indices,
			const int* __restrict__ d_num_high_quality_subject_indices,
			const int* __restrict__ d_alignment_shifts,
			const BestAlignment_t* __restrict__ d_alignment_best_alignment_flags,
			int* __restrict__ d_num_corrected_candidates,
			char* __restrict__ d_corrected_candidates,
			int* __restrict__ d_indices_of_corrected_candidates,
			int n_subjects,
			int n_queries,
			const int* __restrict__ d_num_indices,
			size_t sequence_pitch,
			size_t msa_pitch,
			size_t msa_weights_pitch,
			float min_support_threshold,
			float min_coverage_threshold,
			int new_columns_to_correct,
			RevCompl make_unpacked_reverse_complement_inplace,
			GetCandidateLength getCandidateLength){

	const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);
	const int num_high_quality_subject_indices = *d_num_high_quality_subject_indices;
	//const int n_indices = *d_num_indices;

	for(unsigned index = blockIdx.x; index < num_high_quality_subject_indices; index += gridDim.x) {
		const int subjectIndex = d_high_quality_subject_indices[index];
		const int my_num_candidates = d_indices_per_subject[subjectIndex];

		const float* const my_support = d_support + msa_weights_pitch_floats * subjectIndex;
		const int* const my_coverage = d_coverage + msa_weights_pitch_floats * subjectIndex;
		//const int* const my_orig_coverage = d_origCoverages + msa_weights_pitch_floats * subjectIndex;
		const char* const my_consensus = d_consensus + msa_pitch  * subjectIndex;
		const int* const my_indices = d_indices + d_indices_per_subject_prefixsum[subjectIndex];
		char* const my_corrected_candidates = d_corrected_candidates + d_indices_per_subject_prefixsum[subjectIndex] * sequence_pitch;
		int* const my_indices_of_corrected_candidates = d_indices_of_corrected_candidates + d_indices_per_subject_prefixsum[subjectIndex];

		const MSAColumnProperties properties = d_msa_column_properties[subjectIndex];
		const int subjectColumnsBegin_incl = properties.subjectColumnsBegin_incl;
		const int subjectColumnsEnd_excl = properties.subjectColumnsEnd_excl;

		int n_corrected_candidates = 0;

		for(int local_candidate_index = 0; local_candidate_index < my_num_candidates; ++local_candidate_index) {
			const int global_candidate_index = my_indices[local_candidate_index];
			const int shift = d_alignment_shifts[global_candidate_index];
			//const int candidate_length = d_candidate_sequences_lengths[global_candidate_index];
			const int candidate_length = getCandidateLength(subjectIndex, local_candidate_index);
			const BestAlignment_t bestAlignmentFlag = d_alignment_best_alignment_flags[global_candidate_index];
			const int queryColumnsBegin_incl = shift - properties.startindex;
			const int queryColumnsEnd_excl = queryColumnsBegin_incl + candidate_length;

			//check range condition and length condition
			if(subjectColumnsBegin_incl - new_columns_to_correct <= queryColumnsBegin_incl
			   && queryColumnsBegin_incl <= subjectColumnsBegin_incl + new_columns_to_correct
			   && queryColumnsEnd_excl <= subjectColumnsEnd_excl + new_columns_to_correct) {

				float newColMinSupport = 1.0f;
				int newColMinCov = std::numeric_limits<int>::max();
				//check new columns left of subject
				for(int columnindex = subjectColumnsBegin_incl - new_columns_to_correct;
				    columnindex < subjectColumnsBegin_incl;
				    columnindex++) {

					assert(columnindex < properties.columnsToCheck);
					if(queryColumnsBegin_incl <= columnindex) {
						newColMinSupport = my_support[columnindex] < newColMinSupport ? my_support[columnindex] : newColMinSupport;
						newColMinCov = my_coverage[columnindex] < newColMinCov ? my_coverage[columnindex] : newColMinCov;
					}
				}
				//check new columns right of subject
				for(int columnindex = subjectColumnsEnd_excl;
				    columnindex < subjectColumnsEnd_excl + new_columns_to_correct
				    && columnindex < properties.columnsToCheck;
				    columnindex++) {

					newColMinSupport = my_support[columnindex] < newColMinSupport ? my_support[columnindex] : newColMinSupport;
					newColMinCov = my_coverage[columnindex] < newColMinCov ? my_coverage[columnindex] : newColMinCov;
				}

				if(newColMinSupport >= min_support_threshold
				   && newColMinCov >= min_coverage_threshold) {

					for(int i = queryColumnsBegin_incl + threadIdx.x; i < queryColumnsEnd_excl; i += BLOCKSIZE) {
						my_corrected_candidates[n_corrected_candidates * sequence_pitch + (i - queryColumnsBegin_incl)] = my_consensus[i];
					}

					__syncthreads();                                         // need to wait until all threads have written my_corrected_candidates before calculating reverse complement

					if(threadIdx.x == 0) {
						//the forward strand will be returned -> make reverse complement again
						if(bestAlignmentFlag == BestAlignment_t::ReverseComplement) {
							make_unpacked_reverse_complement_inplace((std::uint8_t*)(my_corrected_candidates + n_corrected_candidates * sequence_pitch), candidate_length);
						}
						my_indices_of_corrected_candidates[n_corrected_candidates] = global_candidate_index;
						//printf("subjectIndex %d global_candidate_index %d\n", subjectIndex, global_candidate_index);
					}

					++n_corrected_candidates;
				}
			}
		}

		//printf("%d %d\n", subjectIndex, n_corrected_candidates);

		if(threadIdx.x == 0) {
			d_num_corrected_candidates[subjectIndex] = n_corrected_candidates;
			//printf("%d %d\n", subjectIndex, n_corrected_candidates);
		}
	}
}

template<class RevCompl, class GetCandidateLength>
void call_msa_correct_candidates_kernel_async_exp(
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
			RevCompl make_unpacked_reverse_complement_inplace,
			GetCandidateLength getCandidateLength,
			int maximum_sequence_length,
			cudaStream_t stream,
			KernelLaunchHandle& handle){


	const int max_block_size = 256;
	const int blocksize = std::min(max_block_size, SDIV(maximum_sequence_length, 32) * 32);
	const std::size_t smem = 0;

	int max_blocks_per_device = 1;

	KernelLaunchConfig kernelLaunchConfig;
	kernelLaunchConfig.threads_per_block = blocksize;
	kernelLaunchConfig.smem = smem;

	auto iter = handle.kernelPropertiesMap.find(KernelId::MSACorrectCandidates);
	if(iter == handle.kernelPropertiesMap.end()) {

		std::map<KernelLaunchConfig, KernelProperties> mymap;

	    #define getProp(blocksize) { \
		KernelLaunchConfig kernelLaunchConfig; \
		kernelLaunchConfig.threads_per_block = (blocksize); \
		kernelLaunchConfig.smem = 0; \
		KernelProperties kernelProperties; \
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
					msa_correct_candidates_kernel_exp<(blocksize), RevCompl, GetCandidateLength>, \
					kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
		mymap[kernelLaunchConfig] = kernelProperties; \
}

		getProp(32);
		getProp(64);
		getProp(96);
		getProp(128);
		getProp(160);
		getProp(192);
		getProp(224);
		getProp(256);

		const auto& kernelProperties = mymap[kernelLaunchConfig];
		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

		handle.kernelPropertiesMap[KernelId::MSACorrectCandidates] = std::move(mymap);

	    #undef getProp
	}else{
		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
	}

	dim3 block(blocksize, 1, 1);
	dim3 grid(std::min(max_blocks_per_device, n_subjects));

		#define mycall(blocksize) msa_correct_candidates_kernel_exp<(blocksize)> \
	        <<<grid, block, 0, stream>>>( \
		d_consensus, \
		d_support, \
		d_coverage, \
		d_origCoverages, \
		d_multiple_sequence_alignments, \
		d_msa_column_properties, \
		d_indices, \
		d_indices_per_subject, \
		d_indices_per_subject_prefixsum, \
		d_high_quality_subject_indices, \
		d_num_high_quality_subject_indices, \
		d_alignment_shifts, \
		d_alignment_best_alignment_flags, \
		d_num_corrected_candidates, \
		d_corrected_candidates, \
		d_indices_of_corrected_candidates, \
		n_subjects, \
		n_queries, \
		d_num_indices, \
		sequence_pitch, \
		msa_pitch, \
		msa_weights_pitch, \
		min_support_threshold, \
		min_coverage_threshold, \
		new_columns_to_correct, \
		make_unpacked_reverse_complement_inplace, \
		getCandidateLength); CUERR;

	assert(blocksize > 0 && blocksize <= max_block_size);

	switch(blocksize) {
	case 32: mycall(32); break;
	case 64: mycall(64); break;
	case 96: mycall(96); break;
	case 128: mycall(128); break;
	case 160: mycall(160); break;
	case 192: mycall(192); break;
	case 224: mycall(224); break;
	case 256: mycall(256); break;
	default: mycall(256); break;
	}

		#undef mycall
}





template<class Accessor, class RevCompl, class GetSubjectPtr, class GetCandidatePtr, class GetSubjectQualityPtr, class GetCandidateQualityPtr,
         class GetSubjectLength, class GetCandidateLength>
__global__
void msa_add_sequences_kernel_implicit(
            int* __restrict__ d_countsA,
            int* __restrict__ d_countsC,
            int* __restrict__ d_countsG,
            int* __restrict__ d_countsT,
            float* __restrict__ d_weightsA,
            float* __restrict__ d_weightsC,
            float* __restrict__ d_weightsG,
            float* __restrict__ d_weightsT,
            int* __restrict__ d_coverage,
            float* __restrict__ d_origWeights,
            int* __restrict__ d_origCoverages,
			const int* __restrict__ d_alignment_shifts,
			const BestAlignment_t* __restrict__ d_alignment_best_alignment_flags,
			const int* __restrict__ d_subject_sequences_lengths,
			const int* __restrict__ d_candidate_sequences_lengths,
			const int* __restrict__ d_alignment_overlaps,
			const int* __restrict__ d_alignment_nOps,
			const MSAColumnProperties*  __restrict__ d_msa_column_properties,
			const int* __restrict__ d_candidates_per_subject_prefixsum,
			const int* __restrict__ d_indices,
			const int* __restrict__ d_indices_per_subject,
			const int* __restrict__ d_indices_per_subject_prefixsum,
			int n_subjects,
			int n_queries,
			const int* __restrict__ d_num_indices,
			bool canUseQualityScores,
			float desiredAlignmentMaxErrorRate,
			int maximum_sequence_length,
			int max_sequence_bytes,
			size_t quality_pitch,
			size_t msa_row_pitch,
			size_t msa_weights_row_pitch,
			Accessor get,
			RevCompl make_unpacked_reverse_complement_inplace,
			GetSubjectPtr getSubjectPtr,
			GetCandidatePtr getCandidatePtr,
			GetSubjectQualityPtr getSubjectQualityPtr,
			GetCandidateQualityPtr getCandidateQualityPtr,
			GetSubjectLength getSubjectLength,
			GetCandidateLength getCandidateLength){

	extern __shared__ float sharedmem[];

    constexpr char A_enc = 0x00;
    constexpr char C_enc = 0x01;
    constexpr char G_enc = 0x02;
    constexpr char T_enc = 0x03;

	float* const sharedWeights = (float*)sharedmem;
	char* const sharedSequence = (char*)(sharedWeights + maximum_sequence_length);

	const size_t msa_weights_row_pitch_floats = msa_weights_row_pitch / sizeof(float);
	const int n_indices = *d_num_indices;

    //add subjects
	for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x) {
		const int subjectColumnsBegin_incl = d_msa_column_properties[subjectIndex].subjectColumnsBegin_incl;
		const int subjectLength = getSubjectLength(subjectIndex);
		const char* const subject = getSubjectPtr(subjectIndex);
		const char* const subjectQualityScore = getSubjectQualityPtr(subjectIndex);
        const int shift = 0;

        int* const my_countsA = d_countsA + subjectIndex * msa_weights_row_pitch_floats;
        int* const my_countsC = d_countsC + subjectIndex * msa_weights_row_pitch_floats;
        int* const my_countsG = d_countsG + subjectIndex * msa_weights_row_pitch_floats;
        int* const my_countsT = d_countsT + subjectIndex * msa_weights_row_pitch_floats;
        float* const my_weightsA = d_weightsA + subjectIndex * msa_weights_row_pitch_floats;
        float* const my_weightsC = d_weightsC + subjectIndex * msa_weights_row_pitch_floats;
        float* const my_weightsG = d_weightsG + subjectIndex * msa_weights_row_pitch_floats;
        float* const my_weightsT = d_weightsT + subjectIndex * msa_weights_row_pitch_floats;
        int* const my_coverage = d_coverage + subjectIndex * msa_weights_row_pitch_floats;

        for(int i = threadIdx.x; i < subjectLength; i+= blockDim.x){
            const int globalIndex = subjectColumnsBegin_incl + shift + i;
            const char base = get(subject, subjectLength, i);
            const float weight = canUseQualityScores ? d_qscore_to_weight[(unsigned char)subjectQualityScore[i]] : 1.0f;
            switch(base){
                case A_enc: atomicAdd(my_countsA + globalIndex, 1); atomicAdd(my_weightsA + globalIndex, weight);break;
                case C_enc: atomicAdd(my_countsC + globalIndex, 1); atomicAdd(my_weightsC + globalIndex, weight);break;
                case G_enc: atomicAdd(my_countsG + globalIndex, 1); atomicAdd(my_weightsG + globalIndex, weight);break;
                case T_enc: atomicAdd(my_countsT + globalIndex, 1); atomicAdd(my_weightsT + globalIndex, weight);break;
                default: assert(false); break;
            }
            my_coverage[globalIndex]++;
        }
	}


    //add candidates
    for(unsigned index = blockIdx.x; index < n_indices; index += gridDim.x) {
        const int queryIndex = d_indices[index];

        const int shift = d_alignment_shifts[queryIndex];
        const BestAlignment_t flag = d_alignment_best_alignment_flags[queryIndex];

        //find subjectindex
        int subjectIndex = 0;
        for(; subjectIndex < n_subjects; subjectIndex++) {
            if(queryIndex < d_candidates_per_subject_prefixsum[subjectIndex+1])
                break;
        }

        const int subjectColumnsBegin_incl = d_msa_column_properties[subjectIndex].subjectColumnsBegin_incl;
        const int localQueryIndex = index - d_indices_per_subject_prefixsum[subjectIndex];
        const int defaultcolumnoffset = subjectColumnsBegin_incl + shift;

        int* const my_countsA = d_countsA + subjectIndex * msa_weights_row_pitch_floats;
        int* const my_countsC = d_countsC + subjectIndex * msa_weights_row_pitch_floats;
        int* const my_countsG = d_countsG + subjectIndex * msa_weights_row_pitch_floats;
        int* const my_countsT = d_countsT + subjectIndex * msa_weights_row_pitch_floats;
        float* const my_weightsA = d_weightsA + subjectIndex * msa_weights_row_pitch_floats;
        float* const my_weightsC = d_weightsC + subjectIndex * msa_weights_row_pitch_floats;
        float* const my_weightsG = d_weightsG + subjectIndex * msa_weights_row_pitch_floats;
        float* const my_weightsT = d_weightsT + subjectIndex * msa_weights_row_pitch_floats;
        int* const my_coverage = d_coverage + subjectIndex * msa_weights_row_pitch_floats;

        const char* const query = getCandidatePtr(queryIndex);
		const int queryLength = getCandidateLength(index);
		const char* const queryQualityScore = getCandidateQualityPtr(index);

		const int query_alignment_overlap = d_alignment_overlaps[queryIndex];
		const int query_alignment_nops = d_alignment_nOps[queryIndex];

		const float defaultweight = 1.0f - sqrtf(query_alignment_nops
					/ (query_alignment_overlap * desiredAlignmentMaxErrorRate));

        assert(flag != BestAlignment_t::None);                 // indices should only be pointing to valid alignments

		//copy query into msa
		if(flag == BestAlignment_t::Forward) {
            for(int i = threadIdx.x; i < queryLength; i+= blockDim.x){
                const int globalIndex = defaultcolumnoffset + i;
                const char base = get(query, queryLength, i);
                const float weight = canUseQualityScores ? d_qscore_to_weight[(unsigned char)queryQualityScore[i]] * defaultweight : 1.0f;
                switch(base){
                    case A_enc: atomicAdd(my_countsA + globalIndex, 1); atomicAdd(my_weightsA + globalIndex, weight);break;
                    case C_enc: atomicAdd(my_countsC + globalIndex, 1); atomicAdd(my_weightsC + globalIndex, weight);break;
                    case G_enc: atomicAdd(my_countsG + globalIndex, 1); atomicAdd(my_weightsG + globalIndex, weight);break;
                    case T_enc: atomicAdd(my_countsT + globalIndex, 1); atomicAdd(my_weightsT + globalIndex, weight);break;
                    default: assert(false); break;
                }
                my_coverage[globalIndex]++;
            }
		}else{
            for(int i = threadIdx.x; i < queryLength; i+= blockDim.x) {
				sharedSequence[i] = get(query, queryLength, i);
				sharedWeights[i] = canUseQualityScores ?
				                   (float)d_qscore_to_weight[(unsigned char)queryQualityScore[i]] * defaultweight
				                   : 1.0f;
			}

			__syncthreads();

            auto make_reverse_complement_byte = [](std::uint8_t in) -> std::uint8_t{
                constexpr std::uint8_t mask = 0x03;
                return (~in & mask);
            };

            const int bytes = queryLength;
            //make reverse complement of shared sequence
            for(int i = threadIdx.x; i < bytes/2; i += blockDim.x){
                const std::uint8_t front = make_reverse_complement_byte(sharedSequence[i]);
                const std::uint8_t back = make_reverse_complement_byte(sharedSequence[bytes - 1 - i]);
                sharedSequence[i] = back;
                sharedSequence[bytes - 1 - i] = front;
            }

            if(bytes % 2 == 1){
                const int middleindex = bytes/2;
                sharedSequence[middleindex] = make_reverse_complement_byte(sharedSequence[middleindex]);
            }
            //reverse quality weights. if canUseQualityScores == false, then all weights are 1.0f and do not need to be reversed
            if(canUseQualityScores){
                const int floats = queryLength;
                for(int i = threadIdx.x; i < floats/2; i += blockDim.x){
                    const float front = sharedWeights[i];
                    const float back = sharedWeights[bytes - 1 - i];
                    sharedWeights[i] = back;
                    sharedWeights[bytes - 1 - i] = front;
                }
                //for odd length, the middle position remains unchanged
            }

            __syncthreads();


            for(int i = threadIdx.x; i < queryLength; i+= blockDim.x){
                const int globalIndex = defaultcolumnoffset + i;
                const char base = sharedSequence[i];
                const float weight = sharedWeights[i];
                switch(base){
                    case A_enc: atomicAdd(my_countsA + globalIndex, 1); atomicAdd(my_weightsA + globalIndex, weight);break;
                    case C_enc: atomicAdd(my_countsC + globalIndex, 1); atomicAdd(my_weightsC + globalIndex, weight);break;
                    case G_enc: atomicAdd(my_countsG + globalIndex, 1); atomicAdd(my_weightsG + globalIndex, weight);break;
                    case T_enc: atomicAdd(my_countsT + globalIndex, 1); atomicAdd(my_weightsT + globalIndex, weight);break;
                    default: assert(false); break;
                }
                my_coverage[globalIndex]++;
            }

			__syncthreads();
        }


	}
}

template<class Accessor, class RevCompl, class GetSubjectPtr, class GetCandidatePtr, class GetSubjectQualityPtr, class GetCandidateQualityPtr,
         class GetSubjectLength, class GetCandidateLength>
void call_msa_add_sequences_kernel_implicit_async(
            int* d_countsA,
            int* d_countsC,
            int* d_countsG,
            int* d_countsT,
            float* d_weightsA,
            float* d_weightsC,
            float* d_weightsG,
            float* d_weightsT,
            int* d_coverage,
            float* d_origWeights,
            int* d_origCoverages,
			const int* d_alignment_shifts,
			const BestAlignment_t* d_alignment_best_alignment_flags,
			const int* d_subject_sequences_lengths,
			const int* d_candidate_sequences_lengths,
			const int* d_alignment_overlaps,
			const int* d_alignment_nOps,
			const MSAColumnProperties*  d_msa_column_properties,
			const int* d_candidates_per_subject_prefixsum,
			const int* d_indices,
			const int* d_indices_per_subject,
			const int* d_indices_per_subject_prefixsum,
			int n_subjects,
			int n_queries,
			const int* d_num_indices,
			bool canUseQualityScores,
			float desiredAlignmentMaxErrorRate,
			int maximum_sequence_length,
			int max_sequence_bytes,
			size_t quality_pitch,
			size_t msa_row_pitch,
			size_t msa_weights_row_pitch,
			Accessor get_as_nucleotide,
			RevCompl make_unpacked_reverse_complement_inplace,
			GetSubjectPtr getSubjectPtr,
			GetCandidatePtr getCandidatePtr,
			GetSubjectQualityPtr getSubjectQualityPtr,
			GetCandidateQualityPtr getCandidateQualityPtr,
			GetSubjectLength getSubjectLength,
			GetCandidateLength getCandidateLength,
			cudaStream_t stream,
			KernelLaunchHandle& handle){




	const int blocksize = 128;
	const std::size_t smem = sizeof(char) * maximum_sequence_length + sizeof(float) * maximum_sequence_length;

	int max_blocks_per_device = 1;

	KernelLaunchConfig kernelLaunchConfig;
	kernelLaunchConfig.threads_per_block = blocksize;
	kernelLaunchConfig.smem = smem;

	auto iter = handle.kernelPropertiesMap.find(KernelId::MSAAddSequencesImplicit);
	if(iter == handle.kernelPropertiesMap.end()) {

		std::map<KernelLaunchConfig, KernelProperties> mymap;

	    #define getProp(blocksize) { \
		KernelLaunchConfig kernelLaunchConfig; \
		kernelLaunchConfig.threads_per_block = (blocksize); \
		kernelLaunchConfig.smem = sizeof(char) * maximum_sequence_length + sizeof(float) * maximum_sequence_length; \
		KernelProperties kernelProperties; \
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
					msa_add_sequences_kernel_implicit<Accessor, RevCompl, GetSubjectPtr, GetCandidatePtr, \
					                             GetSubjectQualityPtr, GetCandidateQualityPtr, \
					                             GetSubjectLength, GetCandidateLength>, \
					kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
		mymap[kernelLaunchConfig] = kernelProperties; \
}

		getProp(32);
		getProp(64);
		getProp(96);
		getProp(128);
		getProp(160);
		getProp(192);
		getProp(224);
		getProp(256);

		const auto& kernelProperties = mymap[kernelLaunchConfig];
		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM * 2;

		handle.kernelPropertiesMap[KernelId::MSAAddSequencesImplicit] = std::move(mymap);

	    #undef getProp
	}else{
		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM * 2;
		//std::cout << max_blocks_per_device << " = " << handle.deviceProperties.multiProcessorCount << " * " << kernelProperties.max_blocks_per_SM << std::endl;
	}

	dim3 block(blocksize, 1, 1);
	//d_num_indices blocks will perform work. n_queries is an upper bound of d_num_indices
	dim3 grid(std::min(n_queries, max_blocks_per_device), 1, 1);

	msa_add_sequences_kernel_implicit<<<grid, block, smem, stream>>>(
                                                                d_countsA,
                                                                d_countsC,
                                                                d_countsG,
                                                                d_countsT,
                                                                d_weightsA,
                                                                d_weightsC,
                                                                d_weightsG,
                                                                d_weightsT,
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
	                                                            d_num_indices,
	                                                            canUseQualityScores,
	                                                            desiredAlignmentMaxErrorRate,
	                                                            maximum_sequence_length,
	                                                            max_sequence_bytes,
	                                                            quality_pitch,
	                                                            msa_row_pitch,
	                                                            msa_weights_row_pitch,
	                                                            get_as_nucleotide,
	                                                            make_unpacked_reverse_complement_inplace,
	                                                            getSubjectPtr,
	                                                            getCandidatePtr,
	                                                            getSubjectQualityPtr,
	                                                            getCandidateQualityPtr,
	                                                            getSubjectLength,
	                                                            getCandidateLength); CUERR;
}










template<class Accessor, class GetSubjectPtr>
__global__
void msa_find_consensus_implicit_kernel(
                        int* __restrict__ d_countsA,
                        int* __restrict__ d_countsC,
                        int* __restrict__ d_countsG,
                        int* __restrict__ d_countsT,
                        float* __restrict__ d_weightsA,
                        float* __restrict__ d_weightsC,
                        float* __restrict__ d_weightsG,
                        float* __restrict__ d_weightsT,
                        char* __restrict__ d_consensus,
                        float* __restrict__ d_support,
                        int* __restrict__ d_coverage,
                        float* __restrict__ d_origWeights,
                        int* __restrict__ d_origCoverages,
                        const MSAColumnProperties* __restrict__ d_msa_column_properties,
                        int n_subjects,
                        size_t msa_pitch,
                        size_t msa_weights_pitch,
                        int msa_max_column_count,
                        int blocks_per_msa,
                        Accessor get,
                        GetSubjectPtr getSubjectPtr){

    constexpr char A_enc = 0x00;
    constexpr char C_enc = 0x01;
    constexpr char G_enc = 0x02;
    constexpr char T_enc = 0x03;

    const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

    const int localBlockId = blockIdx.x % blocks_per_msa;
    //const int n_indices = *d_num_indices;

    //process multiple sequence alignment of each subject
    //for each column in msa, find consensus and support
    for(unsigned subjectIndex = blockIdx.x / blocks_per_msa; subjectIndex < n_subjects; subjectIndex += gridDim.x / blocks_per_msa){
        const int subjectColumnsBegin_incl = d_msa_column_properties[subjectIndex].subjectColumnsBegin_incl;
        const int subjectColumnsEnd_excl = d_msa_column_properties[subjectIndex].subjectColumnsEnd_excl;
        const int columnsToCheck = d_msa_column_properties[subjectIndex].columnsToCheck;

        const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
        const char* const subject = getSubjectPtr(subjectIndex);

        char* const my_consensus = d_consensus + subjectIndex * msa_pitch;
        float* const my_support = d_support + subjectIndex * msa_weights_pitch_floats;

        float* const my_orig_weights = d_origWeights + subjectIndex * msa_weights_pitch_floats;
        int* const my_orig_coverage = d_origCoverages + subjectIndex * msa_weights_pitch_floats;

        const float* const my_weightsA = d_weightsA + subjectIndex * msa_weights_pitch_floats;
        const float* const my_weightsC = d_weightsC + subjectIndex * msa_weights_pitch_floats;
        const float* const my_weightsG = d_weightsG + subjectIndex * msa_weights_pitch_floats;
        const float* const my_weightsT = d_weightsT + subjectIndex * msa_weights_pitch_floats;

        for(int column = localBlockId * blockDim.x + threadIdx.x; column < columnsToCheck; column += blocks_per_msa * blockDim.x){
            const float wA = my_weightsA[column];
            const float wC = my_weightsC[column];
            const float wG = my_weightsG[column];
            const float wT = my_weightsT[column];
            char cons = 'A';
            float consWeight = wA;
            if(wC > consWeight){
                cons = 'C';
                consWeight = wC;
            }
            if(wG > consWeight){
                cons = 'G';
                consWeight = wG;
            }
            if(wT > consWeight){
                cons = 'T';
                consWeight = wT;
            }
            my_consensus[column] = cons;
            const float columnWeight = wA + wC + wG + wT;
            my_support[column] = consWeight / columnWeight;

            if(subjectColumnsBegin_incl <= column && column < subjectColumnsEnd_excl){
                const int* const my_countsA = d_countsA + subjectIndex * msa_weights_pitch_floats;
                const int* const my_countsC = d_countsC + subjectIndex * msa_weights_pitch_floats;
                const int* const my_countsG = d_countsG + subjectIndex * msa_weights_pitch_floats;
                const int* const my_countsT = d_countsT + subjectIndex * msa_weights_pitch_floats;

                const int localIndex = column - subjectColumnsBegin_incl;
                const char subjectbase = get(subject, subjectLength, localIndex);

                if(subjectbase == A_enc){
                    my_orig_weights[column] = wA;
                    my_orig_coverage[column] = my_countsA[column];
                }else if(subjectbase == C_enc){
                    my_orig_weights[column] = wC;
                    my_orig_coverage[column] = my_countsC[column];;
                }else if(subjectbase == G_enc){
                    my_orig_weights[column] = wG;
                    my_orig_coverage[column] = my_countsG[column];
                }else if(subjectbase == T_enc){
                    my_orig_weights[column] = wT;
                    my_orig_coverage[column] = my_countsT[column];
                }
            }
        }
    }
}

template<class Accessor, class GetSubjectPtr>
void call_msa_find_consensus_implicit_kernel_async(
                        int* d_countsA,
                        int* d_countsC,
                        int* d_countsG,
                        int* d_countsT,
                        float* d_weightsA,
                        float* d_weightsC,
                        float* d_weightsG,
                        float* d_weightsT,
                        char* d_consensus,
                        float* d_support,
                        int* d_coverage,
                        float* d_origWeights,
                        int* d_origCoverages,
                        const MSAColumnProperties* d_msa_column_properties,
                        int n_subjects,
                        size_t msa_pitch,
                        size_t msa_weights_pitch,
                        Accessor get,
                        GetSubjectPtr getSubjectPtr,
                        cudaStream_t stream,
                        KernelLaunchHandle& handle){


    const int blocksize = 128;
    const std::size_t smem = 0;

    int max_blocks_per_device = 1;

    KernelLaunchConfig kernelLaunchConfig;
    kernelLaunchConfig.threads_per_block = blocksize;
    kernelLaunchConfig.smem = smem;

    auto iter = handle.kernelPropertiesMap.find(KernelId::MSAFindConsensusImplicit);
    if(iter == handle.kernelPropertiesMap.end()){

        std::map<KernelLaunchConfig, KernelProperties> mymap;

        #define getProp(blocksize) { \
            KernelLaunchConfig kernelLaunchConfig; \
            kernelLaunchConfig.threads_per_block = (blocksize); \
            kernelLaunchConfig.smem = 0; \
            KernelProperties kernelProperties; \
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                                                            msa_find_consensus_implicit_kernel<Accessor, GetSubjectPtr>, \
                                                            kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
            mymap[kernelLaunchConfig] = kernelProperties; \
        }

        getProp(32);
        getProp(64);
        getProp(96);
        getProp(128);
        getProp(160);
        getProp(192);
        getProp(224);
        getProp(256);

        const auto& kernelProperties = mymap[kernelLaunchConfig];
        max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

        handle.kernelPropertiesMap[KernelId::MSAFindConsensusImplicit] = std::move(mymap);

        #undef getProp
    }else{
        std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
        const KernelProperties& kernelProperties = map[kernelLaunchConfig];
        max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    }


    const int blocks_per_msa = 2; //SDIV(msa_max_column_count, blocksize);

    dim3 block(blocksize, 1, 1);
    dim3 grid(std::min(max_blocks_per_device, n_subjects * blocks_per_msa), 1, 1);

    msa_find_consensus_implicit_kernel<<<grid, block, 0, stream>>>(
                                                        d_countsA,
                                                        d_countsC,
                                                        d_countsG,
                                                        d_countsT,
                                                        d_weightsA,
                                                        d_weightsC,
                                                        d_weightsG,
                                                        d_weightsT,
                                                        d_consensus,
                                                        d_support,
                                                        d_coverage,
                                                        d_origWeights,
                                                        d_origCoverages,
                                                        d_msa_column_properties,
                                                        n_subjects,
                                                        msa_pitch,
                                                        msa_weights_pitch,
                                                        blocks_per_msa,
                                                        get,
                                                        getSubjectPtr); CUERR;

}



template<int BLOCKSIZE, class Accessor, class GetSubjectPtr>
__global__
void msa_correct_subject_implicit_kernel(
                        const char* __restrict__ d_consensus,
                        const float* __restrict__ d_support,
                        const int* __restrict__ d_coverage,
                        const int* __restrict__ d_origCoverages,
                        const MSAColumnProperties* __restrict__ d_msa_column_properties,
                        bool* __restrict__ d_is_high_quality_subject,
                        char* __restrict__ d_corrected_subjects,
                        bool* __restrict__ d_subject_is_corrected,
                        int n_subjects,
                        size_t sequence_pitch,
                        size_t msa_pitch,
                        size_t msa_weights_pitch,
                        float estimatedErrorrate,
                        float avg_support_threshold,
                        float min_support_threshold,
                        float min_coverage_threshold,
                        int k_region,
                        Accessor get,
                        GetSubjectPtr getSubjectPtr){

    using BlockReduceBool = cub::BlockReduce<bool, BLOCKSIZE>;
    using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;
    using BlockReduceFloat = cub::BlockReduce<float, BLOCKSIZE>;

    __shared__ union {
        typename BlockReduceBool::TempStorage boolreduce;
        typename BlockReduceInt::TempStorage intreduce;
        typename BlockReduceFloat::TempStorage floatreduce;
    } temp_storage;

    __shared__ bool broadcastbuffer;

    auto isGoodAvgSupport = [&](float avgsupport){
        return avgsupport >= avg_support_threshold;
    };
    auto isGoodMinSupport = [&](float minsupport){
        return minsupport >= min_support_threshold;
    };
    auto isGoodMinCoverage = [&](float mincoverage){
        return mincoverage >= min_coverage_threshold;
    };

    constexpr char A_enc = 0x00;
    constexpr char C_enc = 0x01;
    constexpr char G_enc = 0x02;
    constexpr char T_enc = 0x03;

    auto to_nuc = [](char c){
        switch(c){
        case A_enc: return 'A';
        case C_enc: return 'C';
        case G_enc: return 'G';
        case T_enc: return 'T';
        default: return 'F';
        }
    };

    const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

    for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
        const float* const my_support = d_support + msa_weights_pitch_floats * subjectIndex;
        const int* const my_coverage = d_coverage + msa_weights_pitch_floats * subjectIndex;
        const int* const my_orig_coverage = d_origCoverages + msa_weights_pitch_floats * subjectIndex;
        const char* const my_consensus = d_consensus + msa_pitch  * subjectIndex;
        char* const my_corrected_subject = d_corrected_subjects + subjectIndex * sequence_pitch;

        const MSAColumnProperties properties = d_msa_column_properties[subjectIndex];
        const int subjectColumnsBegin_incl = properties.subjectColumnsBegin_incl;
        const int subjectColumnsEnd_excl = properties.subjectColumnsEnd_excl;

        float avg_support = 0;
        float min_support = 1.0f;
        //int max_coverage = 0;
        int min_coverage = std::numeric_limits<int>::max();

        for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
            assert(i < properties.columnsToCheck);

            avg_support += my_support[i];
            min_support = min(my_support[i], min_support);
            //max_coverage = max(my_coverage[i], max_coverage);
            min_coverage = min(my_coverage[i], min_coverage);
        }

        avg_support = BlockReduceFloat(temp_storage.floatreduce).Sum(avg_support);
        __syncthreads();

        min_support = BlockReduceFloat(temp_storage.floatreduce).Reduce(min_support, cub::Min());
        __syncthreads();

        //max_coverage = BlockReduceInt(temp_storage.intreduce).Reduce(max_coverage, cub::Max());

        min_coverage = BlockReduceInt(temp_storage.intreduce).Reduce(min_coverage, cub::Min());
        __syncthreads();

        avg_support /= (subjectColumnsEnd_excl - subjectColumnsBegin_incl);

        bool isHQ = isGoodAvgSupport(avg_support) && isGoodMinSupport(min_support) && isGoodMinCoverage(min_coverage);

        if(threadIdx.x == 0){
            broadcastbuffer = isHQ;
            d_is_high_quality_subject[subjectIndex] = isHQ;
        }
        __syncthreads();

        isHQ = broadcastbuffer;

        if(isHQ){
            for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                my_corrected_subject[i - subjectColumnsBegin_incl] = to_nuc(my_consensus[i]);
            }
            if(threadIdx.x == 0){
                d_subject_is_corrected[subjectIndex] = true;
            }
        }else{
            //decode orignal sequence and copy to corrected sequence
            const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
            const char* const subject = getSubjectPtr(subjectIndex);
            for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                my_corrected_subject[i - subjectColumnsBegin_incl] = to_nuc(get(subject, subjectLength, i));
            }

            bool foundAColumn = false;
            for(int i = threadIdx.x; i < subjectLength; i += BLOCKSIZE){
                const int globalIndex = subjectColumnsBegin_incl + i;

                if(my_corrected_subject[i] != my_consensus[globalIndex]
                            && my_support[globalIndex] > 0.5f
                            && my_orig_coverage[globalIndex] <= min_coverage_threshold){
                    float avgsupportkregion = 0;
                    int c = 0;
                    bool kregioncoverageisgood = true;

                    for(int j = i - k_region/2; j <= i + k_region/2 && kregioncoverageisgood; j++){
                        if(j != i && j >= 0 && j < subjectLength){
                            avgsupportkregion += my_support[subjectColumnsBegin_incl + j];
                            kregioncoverageisgood &= (my_coverage[subjectColumnsBegin_incl + j] >= min_coverage_threshold);
                            //kregioncoverageisgood &= (my_coverage[subjectColumnsBegin_incl + j] >= 1);
                            c++;
                        }
                    }

                    avgsupportkregion /= c;

                    //if(i == 33 || i == 34){
                    //	printf("%d %f\n", i, avgsupportkregion);
                    //}
                    if(kregioncoverageisgood && avgsupportkregion >= 1.0f-estimatedErrorrate){
                        my_corrected_subject[i] = my_consensus[globalIndex];
                        foundAColumn = true;
                    }
                }
            }
            //perform block wide or-reduction on foundAColumn
            foundAColumn = BlockReduceBool(temp_storage.boolreduce).Reduce(foundAColumn, [](bool a, bool b){return a || b;});
            __syncthreads();

            if(threadIdx.x == 0){
                d_subject_is_corrected[subjectIndex] = foundAColumn;
            }
        }
    }
}

template<class Accessor, class GetSubjectPtr>
void call_msa_correct_subject_implicit_kernel_async(
                        const char* d_consensus,
                        const float* d_support,
                        const int* d_coverage,
                        const int* d_origCoverages,
                        const MSAColumnProperties* d_msa_column_properties,
                        const int* d_indices_per_subject_prefixsum,
                        bool* d_is_high_quality_subject,
                        char* d_corrected_subjects,
                        bool* d_subject_is_corrected,
                        int n_subjects,
                        size_t sequence_pitch,
                        size_t msa_pitch,
                        size_t msa_weights_pitch,
                        float estimatedErrorrate,
                        float avg_support_threshold,
                        float min_support_threshold,
                        float min_coverage_threshold,
                        int k_region,
                        int maximum_sequence_length,
                        Accessor get,
                        GetSubjectPtr getSubjectPtr,
                        cudaStream_t stream,
                        KernelLaunchHandle& handle){

    const int max_block_size = 256;
    const int blocksize = std::min(max_block_size, SDIV(maximum_sequence_length, 32) * 32);
    const std::size_t smem = 0;

    int max_blocks_per_device = 1;

    KernelLaunchConfig kernelLaunchConfig;
    kernelLaunchConfig.threads_per_block = blocksize;
    kernelLaunchConfig.smem = smem;

    auto iter = handle.kernelPropertiesMap.find(KernelId::MSACorrectSubjectImplicit);
    if(iter == handle.kernelPropertiesMap.end()){

        std::map<KernelLaunchConfig, KernelProperties> mymap;

        #define getProp(blocksize) { \
            KernelLaunchConfig kernelLaunchConfig; \
            kernelLaunchConfig.threads_per_block = (blocksize); \
            kernelLaunchConfig.smem = 0; \
            KernelProperties kernelProperties; \
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                                                            msa_correct_subject_implicit_kernel<(blocksize), Accessor, GetSubjectPtr>, \
                                                            kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
            mymap[kernelLaunchConfig] = kernelProperties; \
        }

        getProp(32);
        getProp(64);
        getProp(96);
        getProp(128);
        getProp(160);
        getProp(192);
        getProp(224);
        getProp(256);

        const auto& kernelProperties = mymap[kernelLaunchConfig];
        max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

        handle.kernelPropertiesMap[KernelId::MSACorrectSubjectImplicit] = std::move(mymap);

        #undef getProp
    }else{
        std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
        const KernelProperties& kernelProperties = map[kernelLaunchConfig];
        max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    }



    dim3 block(blocksize, 1, 1);
    dim3 grid(std::min(n_subjects, max_blocks_per_device));

    #define mycall(blocksize) msa_correct_subject_implicit_kernel<(blocksize)> \
                            <<<grid, block, 0, stream>>>( \
                                d_consensus, \
                                d_support, \
                                d_coverage, \
                                d_origCoverages, \
                                d_msa_column_properties, \
                                d_indices_per_subject_prefixsum, \
                                d_is_high_quality_subject, \
                                d_corrected_subjects, \
                                d_subject_is_corrected, \
                                n_subjects, \
                                sequence_pitch, \
                                msa_pitch, \
                                msa_weights_pitch, \
                                estimatedErrorrate, \
                                avg_support_threshold, \
                                min_support_threshold, \
                                min_coverage_threshold, \
                                k_region, \
                                get, \
                                getSubjectPtr); CUERR;

    assert(blocksize > 0 && blocksize <= max_block_size);

    switch(blocksize){
        case 32: mycall(32); break;
        case 64: mycall(64); break;
        case 96: mycall(96); break;
        case 128: mycall(128); break;
        case 160: mycall(160); break;
        case 192: mycall(192); break;
        case 224: mycall(224); break;
        case 256: mycall(256); break;
        default: mycall(256); break;
    }
    #undef mycall
}




#endif //ifdef __NVCC__

}
}


#endif
