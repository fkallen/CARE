#ifndef ALIGNMENT_SEMI_GLOBAL_HPP
#define ALIGNMENT_SEMI_GLOBAL_HPP

#include "alignment.hpp"
#include "hpc_helpers.cuh"

#include <cstdio>
#include <algorithm>
#include <limits>
#include <cassert>

AlignResult cpu_semi_global_align(const char* subject, const char* query, int ns, int nq, bool subjectIsEncoded, bool queryIsEncoded,
	int SCORE_EQUAL, int SCORE_SUBSTITUTE, int SCORE_INSERT, int SCORE_DELETE);











#ifdef __CUDACC__

//TODO delete definition
size_t cuda_semi_global_align_getSharedMemSize(int maxLengthR1, int maxLengthR2);
//TODO delete definition
size_t cuda_semi_global_align_multiwarp_getSharedMemSize(int maxLengthR1, int maxLengthR2, int threadsPerBlock);

void call_cuda_semi_global_align_kernel_async(AlignResultCompact* result_out, AlignOp* ops_out,
				const int max_ops, 
				const int nsubjects, const int ncandidates,
				const char* r1, const char* r2, 
				const int* r1bytesPrefixSum, const int* r1lengths, const int* encodedr1,
				const int* r2bytesPrefixSum, const int* r2lengths, const int* encodedr2,
				const int* candidatesPerSubjectPrefixSum,
				int maxLengthR1, int maxLengthR2, int SCORE_EQUAL, int SCORE_SUBSTITUTE,
				int SCORE_INSERT, int SCORE_DELETE, cudaStream_t stream = 0);

void call_cuda_semi_global_align_kernel(AlignResultCompact* result_out, AlignOp* ops_out,
				const int max_ops, 
				const int nsubjects, const int ncandidates,
				const char* r1, const char* r2, 
				const int* r1bytesPrefixSum, const int* r1lengths, const int* encodedr1,
				const int* r2bytesPrefixSum, const int* r2lengths, const int* encodedr2,
				const int* candidatesPerSubjectPrefixSum,
				int maxLengthR1, int maxLengthR2, int SCORE_EQUAL, int SCORE_SUBSTITUTE,
				int SCORE_INSERT, int SCORE_DELETE, cudaStream_t stream = 0);

//TODO delete definition
__global__
void cuda_semi_global_align_kernel(AlignResultCompact* result_out, AlignOp* ops_out,
				const int max_ops, 
				const int nsubjects, const int ncandidates,
				const char* r1, const char* r2, 
				const int* r1bytesPrefixSum, const int* r1lengths, const int* encodedr1,
				const int* r2bytesPrefixSum, const int* r2lengths, const int* encodedr2,
				const int* candidatesPerSubjectPrefixSum,
				int maxLengthR1, int maxLengthR2, const int SCORE_EQUAL, const int SCORE_SUBSTITUTE,
				const int SCORE_INSERT, const int SCORE_DELETE);


#endif // #ifdef __CUDACC__



#endif
