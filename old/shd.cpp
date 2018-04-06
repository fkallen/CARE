#include "../inc/hpc_helpers.cuh"
#include "../inc/cudareduce.cuh"

#include "../inc/shd.hpp"
#include "../inc/options.hpp"

namespace care{
namespace hammingtools{

	namespace alignment{

		HOSTDEVICEQUALIFIER
		char encoded_accessor(const char* data, int bases, int index){
			const int byte = index / 4;
			const int basepos = index % 4;
			return (data[byte] >> (3-basepos) * 2) & 0x03;
		}

		HOSTDEVICEQUALIFIER
		char twobitToChar(char bits){
			if(bits == 0x00) return 'A';
			if(bits == 0x01) return 'C';
			if(bits == 0x02) return 'G';
			if(bits == 0x03) return 'T';
			return '_';
		}

		AlignResultCompact cpu_shifted_hamming_distance(const GoodAlignmentProperties& prop, const char* subject, const char* query, int ns, int nq){
			const int totalbases = ns + nq;
            const int minoverlap = std::max(prop.min_overlap, int(double(ns) * prop.min_overlap_ratio));
			int bestScore = totalbases; // score is number of mismatches
			int bestShift = -nq; // shift of query relative to subject. shift < 0 if query begins before subject

			for(int shift = -nq + minoverlap; shift < ns - minoverlap; shift++){
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



		template<int BLOCKSIZE>
		__global__
		void cuda_shifted_hamming_distance(const shdparams buffers){
            constexpr int WARPSIZE = 32;
            constexpr int NWARPS = (BLOCKSIZE + WARPSIZE - 1) / WARPSIZE;

            static_assert(sizeof(int2) == sizeof(unsigned long long), "sizeof(int2) != sizeof(unsigned long long)");
            static_assert(BLOCKSIZE % WARPSIZE == 0,
                "BLOCKSIZE must be multiple of WARPSIZE");

			extern __shared__ char smem[];

			/* set up shared memory */
			char* sr1 = (char*)(smem);
			char* sr2 = (char*)(sr1 + buffers.max_sequence_bytes);

			const int subjectbases = buffers.subjectlength;
			const char* subject = buffers.subjectdata;
			for(int threadid = threadIdx.x; threadid < buffers.max_sequence_bytes; threadid += BLOCKSIZE){
				sr1[threadid] = subject[threadid];
			}

            const int minoverlap = max(buffers.props.min_overlap, int(double(subjectbases) * buffers.props.min_overlap_ratio));

			for(int queryId = blockIdx.x; queryId < buffers.n_queries; queryId += gridDim.x){

				const char* query = buffers.queriesdata + queryId * buffers.sequencepitch;

				for(int threadid = threadIdx.x; threadid < buffers.max_sequence_bytes; threadid += BLOCKSIZE){
					sr2[threadid] = query[threadid];
				}

				__syncthreads();

				//begin SHD algorithm

				const int querybases = buffers.querylengths[queryId];

				const int totalbases = subjectbases + querybases;

				int bestScore = totalbases; // score is number of mismatches
				int bestShift = -querybases; // shift of query relative to subject. shift < 0 if query begins before subject

				for(int shift = -querybases + minoverlap + threadIdx.x; shift < subjectbases - minoverlap; shift += BLOCKSIZE){
					const int overlapsize = min(querybases, subjectbases - shift) - max(-shift, 0);
					int score = 0;

					for(int j = max(-shift, 0); j < min(querybases, subjectbases - shift); j++){
						score += encoded_accessor(sr1, subjectbases, j + shift) != encoded_accessor(sr2, querybases, j);
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

                __shared__ unsigned long long blockreducetmp[NWARPS];

                auto func = [](unsigned long long a, unsigned long long b){
                    return (*((int2*)&a)).x < (*((int2*)&b)).x ? a : b;
                };

#if __CUDACC_VER_MAJOR__ < 9
				unsigned long long tilereduced = reduceTile<32>(*((unsigned long long*)&myval), func);
                int warp = threadIdx.x / WARPSIZE;
                int lane = threadIdx.x % WARPSIZE;
                if(lane == 0)
                    blockreducetmp[warp] = tilereduced;

#else
                auto tile = tiled_partition<32>(this_thread_block());
                unsigned long long tilereduced = reduceTile(tile,
                                            *((unsigned long long*)&myval),
                                            func);
                int warp = threadIdx.x / WARPSIZE;
                if(tile.thread_rank() == 0)
                    blockreducetmp[warp] = tilereduced;
#endif

                __syncthreads();


				/*blockreduce<BLOCKSIZE>(
					(unsigned long long*)&reduced,
					*((unsigned long long*)&myval),
					[](unsigned long long a, unsigned long long b){
						return (*((int2*)&a)).x < (*((int2*)&b)).x ? a : b;
					}
				);

				bestScore = reduced.x;
				bestShift = reduced.y;*/

				//make result
				if(threadIdx.x == 0){
                    //reduce warp results
                    unsigned long long reduced = blockreducetmp[0];
                    for(int i = 0; i < NWARPS; i++){
                        reduced = func(reduced, blockreducetmp[i]);
                    }

                    bestScore = ((int2*)&reduced)->x;
    				bestShift = ((int2*)&reduced)->y;

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

					buffers.results[queryId] = result;
				}
			}
		}

		size_t shd_kernel_getSharedMemSize(const shdparams& buffer){

			size_t smem = 0;
			smem += sizeof(char) * 2 * buffer.max_sequence_bytes;

			return smem;
		}

		//wrapper functions to call kernels

		void call_shd_kernel(const shdparams& buffer, int maxQueryLength, cudaStream_t stream){
			call_shd_kernel_async(buffer, maxQueryLength, stream);

			cudaStreamSynchronize(stream); CUERR;
		}

		void call_shd_kernel_async(const shdparams& buffer, int maxQueryLength, cudaStream_t stream){
            const int minoverlap = max(buffer.props.min_overlap, int(double(buffer.subjectlength) * buffer.props.min_overlap_ratio));
            const int maxShiftsToCheck = buffer.subjectlength+1 + maxQueryLength - 2*minoverlap; //FIXME
			dim3 block(std::min(256, 32 * SDIV(maxShiftsToCheck, 32)), 1, 1);
			//dim3 block(256, 1, 1);
			dim3 grid(buffer.n_queries, 1, 1);

			size_t smem = shd_kernel_getSharedMemSize(buffer);

			switch(block.x){
			case 32: cuda_shifted_hamming_distance<32><<<grid, block, smem, stream>>>(buffer); CUERR; break;
			case 64: cuda_shifted_hamming_distance<64><<<grid, block, smem, stream>>>(buffer); CUERR; break;
			case 96: cuda_shifted_hamming_distance<96><<<grid, block, smem, stream>>>(buffer); CUERR; break;
			case 128: cuda_shifted_hamming_distance<128><<<grid, block, smem, stream>>>(buffer); CUERR; break;
			case 160: cuda_shifted_hamming_distance<160><<<grid, block, smem, stream>>>(buffer); CUERR; break;
			case 192: cuda_shifted_hamming_distance<192><<<grid, block, smem, stream>>>(buffer); CUERR; break;
			case 224: cuda_shifted_hamming_distance<224><<<grid, block, smem, stream>>>(buffer); CUERR; break;
			case 256: cuda_shifted_hamming_distance<256><<<grid, block, smem, stream>>>(buffer); CUERR; break;
			default: std::cout << "error call_shd_kernel_async\n"; break;
			}
		}

#endif






	}
}
}
