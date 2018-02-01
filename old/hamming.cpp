#include "../inc/hamming.hpp"
#include "../inc/alignment.hpp"
#include "../inc/ganja/hpc_helpers.cuh"
#include "../inc/cudareduce.cuh"

#include <limits>
#include <cassert>

/*
	Shifted hamming distance on CPU
*/


		HOSTDEVICEQUALIFIER
		char twobitToChar(char bits){
			if(bits == 0x00) return 'A';
			if(bits == 0x01) return 'C';
			if(bits == 0x02) return 'G';
			if(bits == 0x03) return 'T';
			return '_';
		}


template<class Accessor1, class Accessor2>
AlignResult cpu_shifted_hamming_distance_internal(const char* subject, const char* query, int ns, int nq){

	Accessor1 s{subject, ns};
	Accessor2 q{query, nq};

	const int totalbases = ns + nq;
	int bestScore = totalbases; // score is number of mismatches
	int bestShift = -nq; // shift of query relative to subject. shift < 0 if query begins before subject

	for(int shift = -nq + 1; shift < ns; shift++){
		const int overlapsize = std::min(nq, ns - shift) - std::max(-shift, 0);

		int score = 0;

		for(int j = std::max(-shift, 0); j < std::min(nq, ns - shift); j++){
			score += (s[j + shift] != q[j]) ;		
		}

		score += totalbases - overlapsize;

		if(score < bestScore){
			bestScore = score;
			bestShift = shift;
		}
	}

	AlignResult result;
	result.arc.isValid = (bestShift != -nq);

	const int queryoverlapbegin_incl = std::max(-bestShift, 0);
	const int queryoverlapend_excl = std::min(nq, ns - bestShift);
	const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;

	// check best configuration again and save position of substitutions
	for(int j = queryoverlapbegin_incl; j < queryoverlapend_excl; j++){
		if(s[j + bestShift] != q[j]){
			result.operations.emplace_back(j + bestShift, ALIGNTYPE_SUBSTITUTE, q[j]);
		}		
	}

	result.arc.score = bestScore;
	result.arc.subject_begin_incl = std::max(0, bestShift);
	result.arc.query_begin_incl = queryoverlapbegin_incl;
	result.arc.overlap = overlapsize;
	result.arc.shift = bestShift;
	result.arc.nOps = result.operations.size();
	result.arc.isNormalized = false;

	return result;
}

AlignResult cpu_shifted_hamming_distance(const char* subject, const char* query, int ns, int nq, bool encr1, bool encr2){
	if(encr1 && encr2)
		return cpu_shifted_hamming_distance_internal<EncodedAccessor,EncodedAccessor>(subject, query, ns, nq);
	else if (encr1 && !encr2)
		return cpu_shifted_hamming_distance_internal<EncodedAccessor,OrdinaryAccessor>(subject, query, ns, nq);
	else if (!encr1 && encr2)
		return cpu_shifted_hamming_distance_internal<OrdinaryAccessor,EncodedAccessor>(subject, query, ns, nq);
	else
		return cpu_shifted_hamming_distance_internal<OrdinaryAccessor,OrdinaryAccessor>(subject, query, ns, nq);	
}


/*
	Shifted hamming distance on GPU
*/

#ifdef __NVCC__



namespace{
__device__
char cuda_ordinary_accessor(const char* data, int index){
	return data[index];
}

__device__
char cuda_encoded_accessor(const char* data, int bases, int index){
	const int UNUSED_BYTE_SPACE((4 - (bases % 4)) % 4);
	const int byte = (index + UNUSED_BYTE_SPACE) / 4;
	const int basepos = (index + UNUSED_BYTE_SPACE) % 4;

	#define BASE_A 0x00
	#define BASE_C 0x01
	#define BASE_G 0x02
	#define BASE_T 0x03

	switch((data[byte] >> (3-basepos) * 2) & 0x03) {
                case BASE_A: return 'A';
                case BASE_C: return 'C';
                case BASE_G: return 'G';
                case BASE_T: return 'T';
		default: return '_'; // cannot happen
	}
	
	#undef BASE_A
	#undef BASE_C
	#undef BASE_G
	#undef BASE_T
}

}



size_t cuda_shifted_hamming_distance_getSharedMemSize(int maxSubjectLength, int maxQueryLength){

	size_t smem = 0;
	smem += sizeof(char) * (maxSubjectLength + maxQueryLength);

	return smem;
}

#if 0
__device__ 
AlignResultCompact cuda_shifted_hamming_distance_enc_enc(const char* subject, const char* query, int subjectbases, int querybases){

	const int totalbases = subjectbases + querybases;
	int bestScore = totalbases; // score is number of mismatches
	int bestShift = -querybases; // shift of query relative to subject. shift < 0 if query begins before subject

	for(int shift = -querybases + 1 + threadIdx.x; shift < subjectbases; shift += blockDim.x){
		//const int queryoverlapbegin_incl = max(-shift, 0);
		//const int queryoverlapend_excl = min(querybases, subjectbases - shift);
		//const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
		const int overlapsize = min(querybases, subjectbases - shift) - max(-shift, 0);
		int score = 0;

		//for(int j =queryoverlapbegin_incl; j < queryoverlapend_excl; j++){
		for(int j = max(-shift, 0); j < min(querybases, subjectbases - shift); j++){
			score += cuda_encoded_accessor(subject, subjectbases, j + shift) != cuda_encoded_accessor(query, querybases, j);
		}

		score += totalbases - overlapsize;

		if(score < bestScore){
			bestScore = score;
			bestShift = shift;
		}
	}

	// perform reduction to find smallest score in block. the corresponding shift is required, too
	// pack both score and shift into int2 and perform int2-reduction by only comparing the score

	static_assert(sizeof(int2) == sizeof(unsigned long long), "sizeof(int2) != sizeof(unsigned long long)");

	int2 myval = make_int2(bestScore, bestShift);
	int2 reduced;
	blockreduce<128>(
		(unsigned long long*)&reduced, 
		*((unsigned long long*)&myval), 
		[](unsigned long long a, unsigned long long b){
			return (*((int2*)&a)).x < (*((int2*)&b)).x ? a : b;
		}
	);

	bestScore = reduced.x;
	bestShift = reduced.y;

	//make result
	if(threadIdx.x == 0){

		AlignResultCompact result;

		result.isValid = (bestShift != -querybases);
		const int queryoverlapbegin_incl = max(-bestShift, 0);
		const int queryoverlapend_excl = min(querybases, subjectbases - bestShift);
		const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;		

		// check best configuration again and save position of substitutions
		int opnr = 0;
		for(int j = queryoverlapbegin_incl; j < queryoverlapend_excl; j++){
			const bool mismatch = cuda_encoded_accessor(subject, subjectbases, j + bestShift) != cuda_encoded_accessor(query, querybases, j);

			if(mismatch){
				AlignOp op(j + bestShift, ALIGNTYPE_SUBSTITUTE, encr2 ? cuda_encoded_accessor(sr2, querybases, j) : cuda_ordinary_accessor(sr2, j));
				my_ops_out[opnr] = op;
				opnr++;
			}		
		}

		result.score = bestShift;
		result.subject_begin_incl = max(0, bestShift);
		result.subject_end_excl = result.subject_begin_incl + overlapsize;
		result.query_begin_incl = queryoverlapbegin_incl;
		result.query_end_excl = queryoverlapend_excl;
		result.nOps = opnr;
		result.isNormalized = false;
		return result;
	}
}
#endif

template<bool enc1, bool enc2>
__device__
void cuda_hamming_core(const char* subject, const char* query, int subjectbases, int querybases, int& bestScore, int& bestShift){
	const int totalbases = subjectbases + querybases;
//if(threadIdx.x == 0){
    //printf("subjectbases %d querybases %d totalbases %d\n", subjectbases, querybases, totalbases);
	for(int shift = -querybases + 1 + threadIdx.x; shift < subjectbases; shift += blockDim.x){
    //for(int shift = -querybases + 1; shift < subjectbases; shift ++){
		//const int queryoverlapbegin_incl = max(-shift, 0);
		//const int queryoverlapend_excl = min(querybases, subjectbases - shift);
		//const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
		const int overlapsize = min(querybases, subjectbases - shift) - max(-shift, 0);
		int score = 0;

		//for(int j =queryoverlapbegin_incl; j < queryoverlapend_excl; j++){
		for(int j = max(-shift, 0); j < min(querybases, subjectbases - shift); j++){
				if(enc1 && enc2)
					score += cuda_encoded_accessor(subject, subjectbases, j + shift) != cuda_encoded_accessor(query, querybases, j);
				else if (enc1 && !enc2)
					score += cuda_encoded_accessor(subject, subjectbases, j + shift) != cuda_ordinary_accessor(query, j);
				else if (!enc1 && enc2)
					score += cuda_ordinary_accessor(subject, j + shift) != cuda_encoded_accessor(query, querybases, j);
				else if (!enc1 && !enc2)
					score += cuda_ordinary_accessor(subject, j + shift) != cuda_ordinary_accessor(query, j);	
		}

		score += totalbases - overlapsize;
        //printf("score %d overlapsize %d\n", score, overlapsize);
		if(score < bestScore){
			bestScore = score;
			bestShift = shift;
		}
	}
	//printf("over\n");
//}
}




__global__
void cuda_shifted_hamming_distance_general(AlignResultCompact* result_out, AlignOp* ops_out,
					int max_ops,
					const char* subjects, 
					int nsubjects,				
					const int* subjectBytesPrefixSum, 
					const int* subjectLengths, 
					const int* isEncodedSubjects,
					const char* queries,
					int ncandidates,
					const int* queryBytesPrefixSum, 
					const int* queryLengths, 
					const int* isEncodedQueries,
					const int* queriesPerSubjectPrefixSum,
					int maxSubjectLength){

	extern __shared__ char smem[];

	/* set up shared memory */
	char* sr1 = (char*)(smem);
	char* sr2 = (char*)(sr1 + maxSubjectLength);

	for(int globalCandidateId = blockIdx.x; globalCandidateId < ncandidates; globalCandidateId += gridDim.x){

		//setup batch. get correct pointers, store subject and query in shared mem,...

		AlignResultCompact * const my_result_out = result_out + globalCandidateId;
		AlignOp * const my_ops_out = ops_out + max_ops * globalCandidateId;

		int subjectId = 0;
		for(int i = 0; i < nsubjects; i++)
			if(globalCandidateId >= queriesPerSubjectPrefixSum[i])
				subjectId = i;

		const char * subject = subjects + subjectBytesPrefixSum[subjectId];
		const int subjectbytes = subjectBytesPrefixSum[subjectId+1] - subjectBytesPrefixSum[subjectId];
		const int subjectbases = subjectLengths[subjectId];
		const bool encr1 = isEncodedSubjects[subjectId] == 1;

		for(int threadid = threadIdx.x; threadid < subjectbytes; threadid += blockDim.x){
			sr1[threadid] = subject[threadid];			
		}

		const char * query = queries + queryBytesPrefixSum[globalCandidateId];
		const int querybytes = queryBytesPrefixSum[globalCandidateId+1] - queryBytesPrefixSum[globalCandidateId];
		const int querybases = queryLengths[globalCandidateId];
		const bool encr2 = isEncodedQueries[globalCandidateId] == 1;

		for(int threadid = threadIdx.x; threadid < querybytes; threadid += blockDim.x){
			sr2[threadid] = query[threadid];
		}

		__syncthreads(); //setup complete

		//begin SHD algorithm

		const int totalbases = subjectbases + querybases;
		int bestScore = totalbases; // score is number of mismatches
		int bestShift = -querybases; // shift of query relative to subject. shift < 0 if query begins before subject

		if(encr1 && encr2)
			cuda_hamming_core<true, true>(sr1, sr2, subjectbases, querybases, bestScore, bestShift);
		else if (encr1 && !encr2)
			cuda_hamming_core<true, false>(sr1, sr2, subjectbases, querybases, bestScore, bestShift);
		else if (!encr1 && encr2)
			cuda_hamming_core<false, true>(sr1, sr2, subjectbases, querybases, bestScore, bestShift);
		else if (!encr1 && !encr2)
			cuda_hamming_core<false, false>(sr1, sr2, subjectbases, querybases, bestScore, bestShift);	

		//if(globalCandidateId == 0){
		//	printf("tid: %d, bestscore %d, bestshift %d\n", threadIdx.x, bestScore, bestShift);
		//}

		// perform reduction to find smallest score in block. the corresponding shift is required, too
		// pack both score and shift into int2 and perform int2-reduction by only comparing the score

		static_assert(sizeof(int2) == sizeof(unsigned long long), "sizeof(int2) != sizeof(unsigned long long)");
	
		int2 myval = make_int2(bestScore, bestShift);
		int2 reduced;
		blockreduce<128>(
			(unsigned long long*)&reduced, 
			*((unsigned long long*)&myval), 
			[](unsigned long long a, unsigned long long b){
				return (*((int2*)&a)).x < (*((int2*)&b)).x ? a : b;
			}
		);

		bestScore = reduced.x;
		bestShift = reduced.y;
	
		//make result
		if(threadIdx.x == 0){

			AlignResultCompact result;

			result.isValid = (bestShift != -querybases);
			const int queryoverlapbegin_incl = max(-bestShift, 0);
			const int queryoverlapend_excl = min(querybases, subjectbases - bestShift);
			const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;		

			// check best configuration again and save position of substitutions
			int opnr = 0;
			/*for(int j = queryoverlapbegin_incl; j < queryoverlapend_excl; j++){
				bool mismatch = false;
				if(encr1 && encr2)
					mismatch = cuda_encoded_accessor(sr1, subjectbases, j + bestShift) != cuda_encoded_accessor(sr2, querybases, j);
				else if (encr1 && !encr2)
					mismatch = cuda_encoded_accessor(sr1, subjectbases, j + bestShift) != cuda_ordinary_accessor(sr2, j);
				else if (!encr1 && encr2)
					mismatch = cuda_ordinary_accessor(sr1, j + bestShift) != cuda_encoded_accessor(sr2, querybases, j);
				else if (!encr1 && !encr2)
					mismatch = cuda_ordinary_accessor(sr1, j + bestShift) != cuda_ordinary_accessor(sr2, j);

				if(mismatch){
					AlignOp op(j + bestShift, ALIGNTYPE_SUBSTITUTE, encr2 ? cuda_encoded_accessor(sr2, querybases, j) : cuda_ordinary_accessor(sr2, j));
					my_ops_out[opnr] = op;
					opnr++;
				}		
			}*/

			//assert(bestScore - totalbases + overlapsize == opnr && "opnr check fails");
			opnr = bestScore - totalbases + overlapsize;

			result.score = bestScore;
			result.subject_begin_incl = max(0, bestShift);
			result.query_begin_incl = queryoverlapbegin_incl;
			result.overlap = overlapsize;
			result.shift = bestShift;
			result.nOps = opnr;
			result.isNormalized = false;

			*my_result_out = result;
		}
	}
}




__global__
void cuda_shifted_hamming_distance(AlignResultCompact* result_out, AlignOp* ops_out,
					int max_ops,
					const char* subjects, 
					int nsubjects,				
					const int* subjectBytesPrefixSum, 
					const int* subjectLengths, 
					const int* isEncodedSubjects,
					const char* queries,
					int ncandidates,
					const int* queryBytesPrefixSum, 
					const int* queryLengths, 
					const int* isEncodedQueries,
					const int* queriesPerSubjectPrefixSum,
					int maxSubjectLength){

	extern __shared__ char smem[];

	/* set up shared memory */
	char* sr1 = (char*)(smem);
	char* sr2 = (char*)(sr1 + maxSubjectLength);

	for(int globalCandidateId = blockIdx.x; globalCandidateId < ncandidates; globalCandidateId += gridDim.x){

		//setup batch. get correct pointers, store subject and query in shared mem,...

		AlignResultCompact * const my_result_out = result_out + globalCandidateId;
		AlignOp * const my_ops_out = ops_out + max_ops * globalCandidateId;

		int subjectId = 0;
		for(int i = 0; i < nsubjects; i++)
			if(globalCandidateId >= queriesPerSubjectPrefixSum[i])
				subjectId = i;

		const char * subject = subjects + subjectBytesPrefixSum[subjectId];
		const int subjectbytes = subjectBytesPrefixSum[subjectId+1] - subjectBytesPrefixSum[subjectId];
		const int subjectbases = subjectLengths[subjectId];

		for(int threadid = threadIdx.x; threadid < subjectbytes; threadid += blockDim.x){
			sr1[threadid] = subject[threadid];			
		}

		const char * query = queries + queryBytesPrefixSum[globalCandidateId];
		const int querybytes = queryBytesPrefixSum[globalCandidateId+1] - queryBytesPrefixSum[globalCandidateId];
		const int querybases = queryLengths[globalCandidateId];

		for(int threadid = threadIdx.x; threadid < querybytes; threadid += blockDim.x){
			sr2[threadid] = query[threadid];
		}	

		__syncthreads(); //setup complete

		//begin SHD algorithm

		const int totalbases = subjectbases + querybases;
		int bestScore = totalbases; // score is number of mismatches
		int bestShift = -querybases; // shift of query relative to subject. shift < 0 if query begins before subject

		cuda_hamming_core<true, true>(sr1, sr2, subjectbases, querybases, bestScore, bestShift);	

		//if(globalCandidateId == 0){
		//	printf("tid: %d, bestscore %d, bestshift %d\n", threadIdx.x, bestScore, bestShift);
		//}

		// perform reduction to find smallest score in block. the corresponding shift is required, too
		// pack both score and shift into int2 and perform int2-reduction by only comparing the score

		static_assert(sizeof(int2) == sizeof(unsigned long long), "sizeof(int2) != sizeof(unsigned long long)");
        
        //printf("thread %d, bestScore %d, bestShift %d\n", threadIdx.x, bestScore, bestShift);
	
		int2 myval = make_int2(bestScore, bestShift);
		int2 reduced;
		blockreduce<128>(
			(unsigned long long*)&reduced, 
			*((unsigned long long*)&myval), 
			[](unsigned long long a, unsigned long long b){
				return (*((int2*)&a)).x < (*((int2*)&b)).x ? a : b;
			}
		);

		bestScore = reduced.x;
		bestShift = reduced.y;
	
		//make result
		if(threadIdx.x == 0){

			AlignResultCompact result;

			result.isValid = (bestShift != -querybases);
			const int queryoverlapbegin_incl = max(-bestShift, 0);
			const int queryoverlapend_excl = min(querybases, subjectbases - bestShift);
			const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;		

			// check best configuration again and save position of substitutions
			int opnr = 0;
			/*for(int j = queryoverlapbegin_incl; j < queryoverlapend_excl; j++){
				const bool mismatch = cuda_encoded_accessor(sr1, subjectbases, j + bestShift) != cuda_encoded_accessor(sr2, querybases, j);

				if(mismatch){
					AlignOp op(j + bestShift, ALIGNTYPE_SUBSTITUTE, cuda_encoded_accessor(sr2, querybases, j));
					my_ops_out[opnr] = op;
					opnr++;
				}		
			}*/

			opnr = bestScore - totalbases + overlapsize;

			result.score = bestScore;
			result.subject_begin_incl = max(0, bestShift);
			result.query_begin_incl = queryoverlapbegin_incl;
			result.overlap = overlapsize;
			result.shift = bestShift;
			result.nOps = opnr;
			result.isNormalized = false;

			/*if(blockIdx.x == 0 && threadIdx.x == 0){
				printf("%d %d %d %d %d %d\n", globalCandidateId, queryoverlapbegin_incl, overlapsize, bestScore, opnr, totalbases);
				for(int i = 0; i < subjectbases; i++){
					printf("%c", cuda_encoded_accessor(sr1, subjectbases, i));
				}
				printf("\n");
				for(int i = 0; i < querybases; i++){
					printf("%c", cuda_encoded_accessor(sr2, querybases, i));
				}
				printf("\n");
			}*/

			*my_result_out = result;
		}
	}
}


//wrapper functions to call kernels

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
					cudaStream_t stream){

	call_cuda_shifted_hamming_distance_kernel_async(result_out, ops_out, maxops, subjects, nsubjects, subjectBytesPrefixSum, subjectLengths, isEncodedSubjects, 
							    queries, nqueries, queryBytesPrefixSum, queryLengths, isEncodedQueries, 
							    queriesPerSubjectPrefixSum, maxSubjectLength, maxQueryLength,
							    stream);
	CUERR;
	cudaStreamSynchronize(stream);
	CUERR;
}

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
					cudaStream_t stream){

	dim3 block(std::min(128, 32 * SDIV(maxQueryLength+1, 32)), 1, 1);
	dim3 grid(nqueries, 1, 1);

	/*dim3 block(32,1,1);
	dim3 grid(1,1,1);*/

	size_t smem = cuda_shifted_hamming_distance_getSharedMemSize(maxSubjectLength, maxQueryLength);

	cuda_shifted_hamming_distance<<<grid, block, smem, stream>>>(result_out, ops_out, maxops, subjects, nsubjects, 
									subjectBytesPrefixSum, subjectLengths, isEncodedSubjects, 
									queries, nqueries, queryBytesPrefixSum, queryLengths, isEncodedQueries, 
									queriesPerSubjectPrefixSum, maxSubjectLength);
	CUERR;
}




#endif
