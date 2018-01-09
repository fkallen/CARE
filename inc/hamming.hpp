#ifndef HAMMING_HPP
#define HAMMING_HPP

#include "alignment.hpp"

AlignResult cpu_shifted_hamming_distance(const char* subject, const char* query, int ns, int nq, bool subjectIsEncoded, bool queryIsEncoded);

#ifdef __NVCC__

void call_cuda_shifted_hamming_distance_kernel(AlignResultCompact* result_out, AlignOp* ops_out,
					int maxops,
					const char* subjects, 
					int nsubjects,				
					const int* subjectBytesPrefixSum, 
					const int* subjectLengths, 
					const int* isEncodedSubjects,
					const char* queries,
					int nqueries,
					const int* queryBytesPrefixSum, 
					const int* queryLengths, 
					const int* isEncodedQueries,
					const int* queriesPerSubjectPrefixSum,
					int maxSubjectLength,
					int maxQueryLength,
					cudaStream_t stream = 0);

void call_cuda_shifted_hamming_distance_kernel_async(AlignResultCompact* result_out, AlignOp* ops_out,
					int maxops,
					const char* subjects, 
					int nsubjects,				
					const int* subjectBytesPrefixSum, 
					const int* subjectLengths, 
					const int* isEncodedSubjects,
					const char* queries,
					int nqueries,
					const int* queryBytesPrefixSum, 
					const int* queryLengths, 
					const int* isEncodedQueries,
					const int* queriesPerSubjectPrefixSum,
					int maxSubjectLength,
					int maxQueryLength,
					cudaStream_t stream = 0);

#endif


#endif
