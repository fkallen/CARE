#include "../inc/ganja/hpc_helpers.cuh"
#include "../inc/cudareduce.cuh"

#include "../inc/shd.hpp"


namespace hammingtools{

	namespace alignment{


		HOSTDEVICEQUALIFIER
		char encoded_accessor(const char* data, int bases, int index){
			const int unusedspaceinfirstbyte((4 - (bases % 4)) % 4); //multiple of 2 bits
			const int byte = (index + unusedspaceinfirstbyte) / 4;
			const int basepos = (index + unusedspaceinfirstbyte) % 4;

			return (data[byte] >> (3-basepos) * 2) & 0x03;
		}

		AlignResultCompact cpu_shifted_hamming_distance(const char* subject, const char* query, int ns, int nq){
			const int totalbases = ns + nq;
			int bestScore = totalbases; // score is number of mismatches
			int bestShift = -nq; // shift of query relative to subject. shift < 0 if query begins before subject

			for(int shift = -nq + 1; shift < ns; shift++){
				const int overlapsize = std::min(nq, ns - shift) - std::max(-shift, 0);

				int score = 0;

				for(int j = std::max(-shift, 0); j < std::min(nq, ns - shift); j++){
					score += encoded_accessor(subject, ns, j + shift) != encoded_accessor(query, nq, j);	
				}

				score += totalbases - overlapsize;

				if(score < bestScore){
					bestScore = score;
					bestShift = shift;
				}
			}

			AlignResultCompact result;
			result.isValid = (bestShift != -nq);

			const int queryoverlapbegin_incl = std::max(-bestShift, 0);
			const int queryoverlapend_excl = std::min(nq, ns - bestShift);
			const int overlapsize = queryoverlapend_excl - queryoverlapbegin_incl;
			const int opnr = bestScore - totalbases + overlapsize;

			result.score = bestScore;
			result.subject_begin_incl = std::max(0, bestShift);
			result.query_begin_incl = queryoverlapbegin_incl;
			result.overlap = overlapsize;
			result.shift = bestShift;
			result.nOps = opnr;
			result.isNormalized = false;

			return result;
		}

#ifdef __NVCC__

		__global__
		void cuda_shifted_hamming_distance(SHDdata buffers){

			extern __shared__ char smem[];

			/* set up shared memory */
			char* sr1 = (char*)(smem);
			char* sr2 = (char*)(sr1 + buffers.sequencelength);

			for(int globalQueryId = blockIdx.x; globalQueryId < buffers.n_queries; globalQueryId += gridDim.x){

				//setup batch. get correct pointers, store subject and query in shared mem,...

				AlignResultCompact * const my_result_out = buffers.d_results + globalQueryId;

				int subjectId = 0;
				int queriesPerSubjectPrefixSum = 0;
				for(int i = 0; i < buffers.n_subjects; i++){
					queriesPerSubjectPrefixSum += buffers.d_queriesPerSubject[i];
					if(globalQueryId < queriesPerSubjectPrefixSum){
						subjectId = i;
						break;
					}
				}

				const char* subject = buffers.d_subjectsdata + subjectId * buffers.sequencepitch;
				const char* query = buffers.d_queriesdata + globalQueryId * buffers.sequencepitch;

				for(int threadid = threadIdx.x; threadid < buffers.sequencepitch; threadid += blockDim.x){
					sr1[threadid] = subject[threadid];
					sr2[threadid] = query[threadid];			
				}

				__syncthreads(); //setup complete

				//begin SHD algorithm

				const int subjectbases = buffers.sequencelength;
				const int querybases = buffers.sequencelength;

				const int totalbases = subjectbases + querybases;
				int bestScore = totalbases; // score is number of mismatches
				int bestShift = -querybases; // shift of query relative to subject. shift < 0 if query begins before subject

				for(int shift = -querybases + 1 + threadIdx.x; shift < subjectbases; shift += blockDim.x){
					const int overlapsize = min(querybases, subjectbases - shift) - max(-shift, 0);
					int score = 0;

					for(int j = max(-shift, 0); j < min(querybases, subjectbases - shift); j++){
						score += encoded_accessor(subject, subjectbases, j + shift) != encoded_accessor(query, querybases, j);	
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
					const int opnr = bestScore - totalbases + overlapsize;

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


		size_t shd_kernel_getSharedMemSize(const SHDdata& buffer){

			size_t smem = 0;
			smem += sizeof(char) * 2 * buffer.sequencelength;

			return smem;
		}

		//wrapper functions to call kernels

		void call_shd_kernel(const SHDdata& buffer){

			call_shd_kernel_async(buffer);
			CUERR;
			cudaStreamSynchronize(buffer.stream);
			CUERR;
		}

		void call_shd_kernel_async(const SHDdata& buffer){

			dim3 block(std::min(512, 32 * SDIV(buffer.sequencelength+1, 32)), 1, 1);
			dim3 grid(buffer.n_queries, 1, 1);

			size_t smem = shd_kernel_getSharedMemSize(buffer);

			cuda_shifted_hamming_distance<<<grid, block, smem, buffer.stream>>>(buffer);
			CUERR;
		}
#endif






	}
}
