#include "../inc/sga.hpp"


#ifdef __NVCC__
#include <cooperative_groups.h>
using namespace cooperative_groups;
#endif

#include <cstdint>
#include <limits>
#include <iostream>
#include <iomanip>


#define WARPSIZE 32

namespace graphtools{

	namespace alignment{

		HOSTDEVICEQUALIFIER
		char encoded_accessor(const char* data, int bases, int index){
			const int unusedspaceinfirstbyte((4 - (bases % 4)) % 4); //multiple of 2 bits
			const int byte = (index + unusedspaceinfirstbyte) / 4;
			const int basepos = (index + unusedspaceinfirstbyte) % 4;

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

		AlignResult cpu_semi_global_align_internal(const char* r1, const char* r2, int r1length, int r2bases, 
							const int SCORE_EQUAL, const int SCORE_SUBSTITUTE,
							const int SCORE_INSERT, const int SCORE_DELETE){

			assert(r1length < std::numeric_limits<short>::max());
			assert(r2bases < std::numeric_limits<short>::max());

		//#define CPU_SEMI_GLOBAL_ALIGN_SCOREDEBUGGING

			int scores[r1length + 1][r2bases + 1];
			char prevs[r1length + 1][r2bases + 1];


			// init
			for (int col = 0; col < r2bases + 1; ++col) {
				scores[0][col] = 0;
			}

			// row 0 was filled by column loop
			for (int row = 1; row < r1length + 1; ++row) {
				scores[row][0] = 0;
			}

			// fill matrix
			for (int row = 1; row < r1length + 1; ++row) {
				for (int col = 1; col < r2bases + 1; ++col) {
					// calc entry [row][col]

					const bool ismatch = encoded_accessor(r1, r1length, row - 1) == encoded_accessor(r2, r2bases, col - 1);
					const int matchscore = scores[row - 1][col - 1] + (ismatch ? SCORE_EQUAL : SCORE_SUBSTITUTE);
					const int insscore = scores[row][col - 1] + SCORE_INSERT;
					const int delscore = scores[row - 1][col] + SCORE_DELETE;

					int maximum = 0;
					if (matchscore < delscore) {
						maximum = delscore;
						prevs[row][col] = ALIGNTYPE_DELETE;
					}else{
						maximum = matchscore;
						prevs[row][col] = ismatch ? ALIGNTYPE_MATCH : ALIGNTYPE_SUBSTITUTE;
					}
					if (maximum < insscore) {
						maximum = insscore;
						prevs[row][col] = ALIGNTYPE_INSERT;
					}

					scores[row][col] = maximum;
				}
			}
#if 0
			for (int row = 0; row < r1length + 1; ++row) {
				if(row != 1 && (row-1)%32 == 0){
					std::cout << std::endl;
					std::cout << std::endl;
					for (int col = 1; col < r2bases + 1; ++col) {
						std::cout << "____";
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
				for (int col = 0; col < r2bases + 1; ++col) {
					if((col-1)%32 == 0) std::cout << " | ";
					std::cout << std::setw(4) << scores[row][col];
				}
					
				std::cout << std::endl;
			}
#endif
			// extract best alignment

			int currow = r1length;
			int curcol = r2bases;
			int maximum = std::numeric_limits<int>::min();

			for (int row = 1; row < r1length + 1; ++row) {
				if(scores[row][r2bases] > maximum){
					//short oldmax = maximum;
					maximum = scores[row][r2bases];
					currow = row;
					curcol = r2bases;

					/*std::cout << "row = " << row << ": \n"
						<< scores[row][r2bases] << " > " << oldmax << "\n"
						<< " update currow " << currow << ", curcol " << curcol << std::endl;*/
				}
			}

			for (int col = 1; col < r2bases + 1; ++col) {
				if(scores[r1length][col] > maximum){
					//short oldmax = maximum;
					maximum = scores[r1length][col];
					currow = r1length;
					curcol = col;

					/*std::cout << "col = " << col << ": \n"
						<< scores[r1length][col] << " > " << oldmax << "\n"
						<< " update currow " << currow << ", curcol " << curcol << std::endl;*/
				}
			}
			//std::cout << "currow " << currow << ", curcol " << curcol << std::endl;

			AlignResult alignresult;
			std::vector<AlignOp> operations;

			const int subject_end_excl = currow;

			alignresult.arc.score = maximum;
			alignresult.arc.isNormalized = false;

			while(currow != 0 && curcol != 0){
				switch (prevs[currow][curcol]) {
				case ALIGNTYPE_MATCH: //printf("m\n");
					curcol -= 1;
					currow -= 1;
					break;
				case ALIGNTYPE_SUBSTITUTE: //printf("s\n");

					operations.push_back(AlignOp{(short)(currow-1), ALIGNTYPE_SUBSTITUTE, twobitToChar(encoded_accessor(r2, r2bases, curcol - 1))});

					curcol -= 1;
					currow -= 1;
					break;
				case ALIGNTYPE_DELETE:  //printf("d\n");

					operations.push_back(AlignOp{(short)(currow-1), ALIGNTYPE_DELETE, twobitToChar(encoded_accessor(r1, r1length, currow - 1))});

					curcol -= 0;
					currow -= 1;
					break;
				case ALIGNTYPE_INSERT:  //printf("i\n");

					operations.push_back(AlignOp{(short)currow, ALIGNTYPE_INSERT, twobitToChar(encoded_accessor(r2, r2bases, curcol - 1))});

					curcol -= 1;
					currow -= 0;
					break;
				default : // code should not reach here
					throw std::runtime_error("alignment backtrack error");
				}
			}

			alignresult.operations.resize(operations.size());
			std::reverse_copy(operations.begin(), operations.end(), alignresult.operations.begin());

			alignresult.arc.subject_begin_incl = currow;
			alignresult.arc.query_begin_incl = curcol;
			alignresult.arc.isValid = true;
			alignresult.arc.overlap = subject_end_excl - alignresult.arc.subject_begin_incl;
			alignresult.arc.shift = alignresult.arc.subject_begin_incl == 0 ? -alignresult.arc.query_begin_incl : alignresult.arc.subject_begin_incl;
			alignresult.arc.nOps = operations.size();
			alignresult.arc.isNormalized = false;

			return alignresult;
		}



		AlignResult cpu_semi_global_alignment(const AlignerDataArrays& buffers, const char* subject, const char* query, int ns, int nq){

			return cpu_semi_global_align_internal(subject, query, ns, nq, 
						buffers.ALIGNMENTSCORE_MATCH, buffers.ALIGNMENTSCORE_SUB, 
						buffers.ALIGNMENTSCORE_INS, buffers.ALIGNMENTSCORE_DEL);
		}




	#ifdef __NVCC__
	
		size_t cuda_semi_global_alignment_getSharedMemSize(const sgaparams& buffers){


			size_t smem = 0;
			smem += sizeof(char) * (buffers.max_sequence_bytes + buffers.max_sequence_bytes); //current subject and current query
			smem += sizeof(char) * (buffers.max_sequence_length+1)*SDIV(buffers.max_sequence_length+1, 4); // prevs
			smem += sizeof(short); // padding for pointer alignment
			smem += sizeof(short) * 3*2*(buffers.max_sequence_length + 1); // wavefronts
			smem += sizeof(short) * 4; //best(row/col), best(row/col)score

			return smem;
		}

		/*
		align multiple queries to multiple subjects
		*/
		__global__
		void cuda_semi_global_alignment_kernel(const sgaparams buffers){

			const unsigned int WAVE_ROW_SIZE = 2*(buffers.max_sequence_length + 1);

			extern __shared__ char smem[];

			const int prevcolumncount = SDIV(buffers.max_sequence_length + 1, 4);

			/* set up shared memory */
			char* sr1 = (char*)(smem);
			char* sr2 = (char*)(sr1 + buffers.max_sequence_bytes);
			char* prevs = (char*)(sr2 + buffers.max_sequence_bytes);
			char* tmp = (char*)(prevs + (buffers.max_sequence_length + 1)*prevcolumncount);			
			unsigned long offset = (unsigned long)tmp % sizeof(short); //ensure correct pointer alignment for shorts
			if(offset != 0)
				tmp += sizeof(short) - offset;

			short* scores = (short*)(tmp);
			short* bestrow = (short*)(scores + 3 * WAVE_ROW_SIZE);
			short* bestcol = (short*)(bestrow + 1);
			short* bestrowscore = (short*)(bestcol + 1);
			short* bestcolscore = (short*)(bestrowscore + 1);
			
			const int subjectbases = buffers.subjectlength;
			const char* subject = buffers.subjectdata;
			for(int threadid = threadIdx.x; threadid < buffers.max_sequence_bytes; threadid += blockDim.x){
				sr1[threadid] = subject[threadid];		
			}			

			for(int queryId = blockIdx.x; queryId < buffers.n_queries; queryId += gridDim.x){

				const char* query = buffers.queriesdata + queryId * buffers.sequencepitch;
				const int querybases = buffers.querylengths[queryId];
				for(int threadid = threadIdx.x; threadid < buffers.max_sequence_bytes; threadid += blockDim.x){
					sr2[threadid] = query[threadid];			
				}
				
				AlignResultCompact * const my_result_out = buffers.results + queryId;
				AlignOp * const my_ops_out = buffers.ops + buffers.max_ops_per_alignment * queryId;


				for (int l = threadIdx.x; l < 2*(buffers.max_sequence_length + 1); l += blockDim.x) {
					scores[0*WAVE_ROW_SIZE+l] = 0;
					scores[1*WAVE_ROW_SIZE+l] = 0;
					scores[2*WAVE_ROW_SIZE+l] = 0;
				}

				for(int i = 0; i < buffers.max_sequence_length + 1; i++){
					for (int j = threadIdx.x; j < prevcolumncount; j += blockDim.x) {
						prevs[i * prevcolumncount + j] = 0;
					}
				}


				if (threadIdx.x == 0) {
					*bestrow = 0;
					*bestcol = 0;
					*bestrowscore = SHRT_MIN;
					*bestcolscore = SHRT_MIN;
				}

				__syncthreads();

				for (int baseRow = 2; baseRow < 2*subjectbases+1; ++baseRow) {
		
					const int targetrow = baseRow % 3;
					const int indelrow = (targetrow == 0 ? 2 : targetrow - 1);
					const int matchrow = (indelrow == 0 ? 2 : indelrow - 1);

					// find all threads inside boundary and calculate the score
					for(int threadid = threadIdx.x; threadid < querybases + 1; threadid += blockDim.x){

						// row and col in 2d space
						const int myrow = baseRow - threadid;
						const int mycol = threadid;

						// calculate entry [baseRow - threadid][threadid]
						if((myrow > 0) && (myrow < subjectbases + 1) 
							&& (mycol > 0) && (mycol < querybases + 1)){ 

							bool ismatch = encoded_accessor(sr1, subjectbases, myrow - 1) == encoded_accessor(sr2, querybases, mycol - 1);

							const short matchscore = scores[matchrow * WAVE_ROW_SIZE + threadid - 1] 
										+ (ismatch ? buffers.ALIGNMENTSCORE_MATCH : buffers.ALIGNMENTSCORE_SUB);
							const short insscore = scores[indelrow * WAVE_ROW_SIZE + threadid - 1] + buffers.ALIGNMENTSCORE_INS;
							const short delscore = scores[indelrow * WAVE_ROW_SIZE + threadid] + buffers.ALIGNMENTSCORE_DEL;

							short maximum = 0;
							AlignType prev;
							unsigned int colbyteindex = mycol / 4;
							unsigned int colbitindex = mycol % 4;
		
							if (matchscore < delscore) {
								maximum = delscore;
								prev = ALIGNTYPE_DELETE;
							}else{
								maximum = matchscore;
								prev = ismatch ? ALIGNTYPE_MATCH : ALIGNTYPE_SUBSTITUTE;
							}
							if (maximum < insscore) {
								maximum = insscore;
								prev = ALIGNTYPE_INSERT;
							}

							prevs[myrow * prevcolumncount + colbyteindex] |= (prev << 2*(3-colbitindex));

							scores[targetrow * WAVE_ROW_SIZE + threadid] = maximum;

							// update best score in last row
							if (myrow == subjectbases) {
								if (*bestcolscore < maximum) {
									*bestcolscore = maximum;
									*bestcol = mycol;
									//printf("qborder %d : %d\n", mycol - 1, maximum);
								}
							}

							// update best score in last column
							if (mycol == querybases) {
								if (*bestrowscore < maximum) {
									*bestrowscore = maximum;
									*bestrow = myrow;
									//printf("sborder %d : %d\n", myrow - 1, maximum);
								}
							}
						}
					}

					__syncthreads();
				}

				// get alignment and alignment score
				if (threadIdx.x == 0) {

					short currow;
					short curcol;

					AlignResultCompact result;

					if (*bestcolscore > *bestrowscore) {
						currow = subjectbases;
						curcol = *bestcol;
						result.score = *bestcolscore;
					}else{
						currow = *bestrow;
						curcol = querybases;
						result.score = *bestrowscore;
					}
		
					const int subject_end_excl = currow;

					//printf("currow %d, curcol %d\n", currow, curcol);

					int nOps = 0;
					bool isValid = true;
					AlignOp currentOp;

					while(currow != 0 && curcol != 0){
						unsigned int colbyteindex = curcol / 4;
						unsigned int colbitindex = curcol % 4;
						switch((prevs[currow * prevcolumncount + colbyteindex] >> 2*(3-colbitindex)) & 0x3){
			
						case ALIGNTYPE_MATCH:
							curcol -= 1;
							currow -= 1;
							break;
						case ALIGNTYPE_SUBSTITUTE:

							currentOp.position = currow - 1;
							currentOp.type = ALIGNTYPE_SUBSTITUTE;
							currentOp.base = twobitToChar(encoded_accessor(sr2, querybases, curcol - 1));

							my_ops_out[nOps] = currentOp;
							++nOps;

							curcol -= 1;
							currow -= 1;
							break;
						case ALIGNTYPE_DELETE:

							currentOp.position = currow - 1;
							currentOp.type = ALIGNTYPE_DELETE;
							currentOp.base = twobitToChar(encoded_accessor(sr1, subjectbases, currow - 1));

							my_ops_out[nOps] = currentOp;
							++nOps;

							curcol -= 0;
							currow -= 1;
							break;
						case ALIGNTYPE_INSERT:

							currentOp.position = currow;
							currentOp.type = ALIGNTYPE_INSERT;
							currentOp.base = twobitToChar(encoded_accessor(sr2, querybases, curcol - 1));

							my_ops_out[nOps] = currentOp;
							++nOps;

							curcol -= 1;
							currow -= 0;
							break;
						default : // code should not reach here
							isValid = false;
							printf("alignment backtrack error");
						}
					}
					result.subject_begin_incl = currow;
					result.query_begin_incl = curcol;
					result.overlap = subject_end_excl - result.subject_begin_incl;
					result.shift = result.subject_begin_incl == 0 ? -result.query_begin_incl : result.subject_begin_incl;
					result.nOps = nOps;
					result.isNormalized = false;
					result.isValid = isValid;

					*my_result_out = result;
				}
			}
		}


		void call_cuda_semi_global_alignment_kernel_async(const sgaparams& buffers, cudaStream_t stream){

				size_t smem = cuda_semi_global_alignment_getSharedMemSize(buffers);
				dim3 block(std::min(512, 32 * SDIV(buffers.max_sequence_length+1, 32)), 1, 1);
				dim3 grid(buffers.n_queries, 1, 1);
				//dim3 grid(std::min(buffers.max_blocks, buffers.n_queries), 1, 1);
				// start kernel

				cuda_semi_global_alignment_kernel
					<<<grid, block, smem, stream>>>(buffers);
				CUERR;
		}

		void call_cuda_semi_global_alignment_kernel(const sgaparams& buffers, cudaStream_t stream){

				call_cuda_semi_global_alignment_kernel_async(buffers, stream);

				cudaStreamSynchronize(stream); CUERR;
		}

		

		/*
		align multiple queries to subject
		*/
		__global__
		void cuda_semi_global_alignment_warps_kernel(const sgaparams buffers){
			
			union BestscoreData{
				int data;
				struct{
						short bestScore;
						short bestIndex;
				};
			};
			
			extern __shared__ char smem[];
			
			/*if(threadIdx.x + blockDim.x * blockIdx.x == 0){
					printf("nqueries : %d\n", buffers.n_queries);
					for(int i = 0; i < buffers.n_queries; i++){
						printf("length %d : %d\n", i, buffers.querylengths[i]);
					}
			}*/
				
			auto block = this_thread_block();
			auto warp = tiled_partition<WARPSIZE>(block);
			int warpId = block.thread_rank() / WARPSIZE;
			int nwarps = block.size() / WARPSIZE;		
						
			//set up shared memory pointers
			int prevcolumncount = SDIV(buffers.max_sequence_length, 32);
			
			char* sr1 = (char*)(smem);
			char* sr2 = (char*)(sr1 + buffers.max_sequence_bytes);
			char* tmp = (char*)(sr2 + buffers.max_sequence_bytes);	
			unsigned long long offset = (unsigned long long)tmp % sizeof(unsigned long long); //ensure correct pointer alignment for unsigned long long
			if(offset != 0)
				tmp += sizeof(unsigned long long) - offset;
			unsigned long long* prevs = (unsigned long long*)(tmp);
			tmp = (char*)(prevs + buffers.max_sequence_length * prevcolumncount);	
			offset = (unsigned long long)tmp % sizeof(short); //ensure correct pointer alignment for short
			if(offset != 0)
				tmp += sizeof(short) - offset;
			short* scoreBorderSubject = (short*)(tmp);
			short* scoreBorderQuery = (short*)(scoreBorderSubject + buffers.max_sequence_length);


			//scoreBorderDiag stores diagonal for calculation of tile [subjecttileId, querytileId]
			//number of rows and columns is increased by 1 to avoid boundary check at bottom end / right end
			short* scoreBorderDiag = (short*)(scoreBorderQuery + buffers.max_sequence_length);
			tmp = (char*)(scoreBorderDiag + 2 * (SDIV(buffers.max_sequence_length, 32)+1));	
			offset = (unsigned long long)tmp % sizeof(BestscoreData); //ensure correct pointer alignment for short
			if(offset != 0)
				tmp += sizeof(BestscoreData) - offset;			
			BestscoreData* bestRowData = (BestscoreData*)(tmp);
			BestscoreData* bestColData = (BestscoreData*)(bestRowData + nwarps);		

			const char* subject = buffers.subjectdata;
			int subjectbases = buffers.subjectlength;
	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
			int subjecttiles = SDIV(subjectbases, WARPSIZE);
			int maxTotalTiles = subjecttiles * (SDIV(buffers.max_sequence_length, 32)+1);
			
			for(int threadid = block.thread_rank(); threadid < buffers.max_sequence_bytes; threadid += blockDim.x){
				sr1[threadid] = subject[threadid];		
			}

			for(int queryId = blockIdx.x; queryId < buffers.n_queries; queryId += gridDim.x){
				//if(threadIdx.x == 0) printf("queryId %d\n", queryId);
				const char* query = buffers.queriesdata + queryId * buffers.sequencepitch;
				int querybases = buffers.querylengths[queryId];
				int querytiles = SDIV(querybases, WARPSIZE);
				
				for(int threadid = block.thread_rank(); threadid < buffers.max_sequence_bytes; threadid += blockDim.x){
					sr2[threadid] = query[threadid];			
				}	
				
				for(int i = block.thread_rank(); i < buffers.max_sequence_length; i += blockDim.x){
					scoreBorderSubject[i] = 0;
					scoreBorderQuery[i] = 0;
				}
				
				for(int i = block.thread_rank(); i < maxTotalTiles; i+= blockDim.x){
					scoreBorderDiag[i] = 0;
				}
				
				block.sync(); //finish shared mem writes
				
				int subjecttileId = warpId;
				int subjecttileBases = subjecttileId < subjecttiles - 1 ? WARPSIZE : subjectbases - subjecttileId * WARPSIZE;
				int globalsubjectpos = subjecttileId * WARPSIZE + warp.thread_rank();
				char subjectbase = twobitToChar(encoded_accessor(sr1, subjectbases, globalsubjectpos));

				//tiles move from left to right
				for(int tilediagonal = 0; tilediagonal < 2*querytiles-1; tilediagonal++){
					int querytileId = tilediagonal - warpId;
					int querytileBases = querytileId < querytiles - 1 ? WARPSIZE : querybases - querytileId * WARPSIZE;
					//printf("tilediag %d querytileId %d warpId %d\n", tilediagonal, querytileId, warpId);

					// thread diagonals which cross the last valid row/column in tile must have
					// validBorderDiagonalndexBegin <= threaddiagonal < validBorderDiagonalndexEndexcl
					int validBorderDiagonalndexBegin = -1;
					int validBorderDiagonalndexEndexcl = -1;
					if(subjecttileBases < querytileBases){
						validBorderDiagonalndexBegin = subjecttileBases - 1;
						validBorderDiagonalndexEndexcl = validBorderDiagonalndexBegin + querytileBases;
					}else{
						validBorderDiagonalndexBegin = querytileBases - 1;
						validBorderDiagonalndexEndexcl = validBorderDiagonalndexBegin + subjecttileBases;
					}
					
					if(0 <= querytileId && querytileId < querytiles){
						assert(subjecttileId + querytileId == tilediagonal);
						
						//relax diagonals in tile [subjecttileId, querytileId]						
						
						int globalquerypos = querytileId * WARPSIZE + warp.thread_rank();
						
						//initialize tile data
						
						char querybase = twobitToChar(encoded_accessor(sr2, querybases, globalquerypos));
						short borderScoreSubject = scoreBorderSubject[globalsubjectpos];
						short borderScoreQuery = scoreBorderQuery[globalquerypos];
						//short borderScoreDiag = subjecttileId == 0 ? 0 : scoreBorderDiag[subjecttileId-1];
						short borderScoreDiag = scoreBorderDiag[subjecttileId * (querytiles+1) + querytileId];

//if(tilediagonal == 3 && querytileId == 0){ printf("thread %d: borderScoreQuery %d\n", warp.thread_rank(), borderScoreQuery);}								

						
						short scoreLeft = 0;
						short scoreLeftLeft = 0;
						short scoreDiag = 0;
						short scoreUp = 0;
						short scoreCur = 0;							
						unsigned long long myprev = 0;
						
						//diagonal moves from left to right
						for(int threaddiagonal = 0; threaddiagonal < 2 * WARPSIZE - 1; threaddiagonal++){
							int localquerypos = threaddiagonal - warp.thread_rank();
							int localsubjectpos = warp.thread_rank();
							globalquerypos = querytileId * WARPSIZE + localquerypos;
							
							//if(threadIdx.x == 0){
							//		printf("globalquerypos %d \n", globalquerypos);
							//}
							
							//fetch scores of cells left, above, diag, which are in registers of other threads in the warp								
							scoreLeft = scoreCur;
							scoreDiag = warp.shfl_up(scoreLeftLeft, 1);
							scoreUp = warp.shfl_up(scoreCur, 1);
							short scoreUpTmp = warp.shfl(borderScoreQuery, localquerypos);
							short scoreDiagTmp = warp.shfl(borderScoreQuery, localquerypos-1);
													
							//fetch next base
							char myquerybase = warp.shfl(querybase, localquerypos);
							
							//printf("b thread %d, threaddiagonal %d qb %c\n", warp.thread_rank(), threaddiagonal, myquerybase);
							
							if(0 <= localquerypos && localquerypos < WARPSIZE){
								assert(threaddiagonal == localquerypos + localsubjectpos);
								
								//first row and col need to fetch from border, which is also cached in registers
								if(localquerypos == 0){
									scoreLeft = borderScoreSubject;
								}
								if(localsubjectpos == 0){
									scoreUp = scoreUpTmp;
									//printf("access localquerypos %d\n", localquerypos);
									if(localquerypos == 0){
										scoreDiag = borderScoreDiag;
									}else{
										assert(globalquerypos != 0);
//if(tilediagonal == 3 && querytileId == 0){ printf("td %d, thread %d: scoreDiag = scoreBorderQuery[globalquerypos-1] %d, %d\n", threaddiagonal, warp.thread_rank(), globalquerypos-1, scoreDiag );}	
										scoreDiag = scoreDiagTmp; //scoreBorderQuery[globalquerypos-1];
									}
								}
								
								//printf("c thread %d, tilediagonal %d, threaddiagonal %d\n", block.thread_rank(), tilediagonal, threaddiagonal);
								
								bool ismatch = subjectbase == myquerybase;
								//printf("qid %d, tile %d %d, diag %d warp %d, thread %d\n", queryId, subjecttileId, querytileId, threaddiagonal, warpId, warp.thread_rank());

								short matchscore = scoreDiag 
											+ (ismatch ? buffers.ALIGNMENTSCORE_MATCH : buffers.ALIGNMENTSCORE_SUB);
								short insscore = scoreLeft + buffers.ALIGNMENTSCORE_INS;
								short delscore = scoreUp + buffers.ALIGNMENTSCORE_DEL;									
								//AlignType type;
								if (matchscore < delscore) {
									scoreCur = delscore;
									myprev = (myprev << 2) | ALIGNTYPE_DELETE;
									//type = ALIGNTYPE_DELETE;
								}else{
									scoreCur = matchscore;
									myprev <<= 2;
									myprev |= ismatch ? ALIGNTYPE_MATCH : ALIGNTYPE_SUBSTITUTE;
									//type = ismatch ? ALIGNTYPE_MATCH : ALIGNTYPE_SUBSTITUTE;
								}
								if (scoreCur < insscore) {
									scoreCur = insscore;
									myprev = (myprev << 2) | ALIGNTYPE_INSERT;
									//type = ALIGNTYPE_INSERT;
								}
//if(tilediagonal == 3 && querytileId == 0){ printf("td %d, thread %d: l %d, u %d, d %d, score %d\n", threaddiagonal, warp.thread_rank(), scoreLeft, scoreUp, scoreDiag, scoreCur);}								
								
								if(threaddiagonal >= validBorderDiagonalndexBegin 
								&& threaddiagonal < validBorderDiagonalndexEndexcl 
								&& localquerypos < querytileBases 
								&& localsubjectpos < subjecttileBases){
									coalesced_group activethreads = coalesced_threads();
									// on a valid diagonal, the active thread with smallest id (the thread in the top right) writes the subject border
									// the active thread with the greatest id (the thread in the bottom left) writes the query border
									if(activethreads.thread_rank() == 0){
											scoreBorderSubject[globalsubjectpos] = scoreCur;
//if(tilediagonal == 3 && querytileId == 0){ printf("td %d, thread %d write sborder %d %d\n", threaddiagonal, warp.thread_rank(), globalsubjectpos, scoreCur);}
									}
									if(activethreads.thread_rank() == activethreads.size()-1){
											scoreBorderQuery[globalquerypos] = scoreCur;
//if(tilediagonal == 3 && querytileId == 0){ printf("td %d, thread %d write qborder %d %d\n", threaddiagonal, warp.thread_rank(), globalquerypos, scoreCur);}

									}
								}								
							}
							
							scoreLeftLeft = scoreLeft;
							
							warp.sync(); //diagonal finished
						}
						
						//write prev ops							
						prevs[globalsubjectpos * prevcolumncount + querytileId] = myprev;
						

						
						if(warp.thread_rank() == warp.size() - 1){
							//write diagonal entry which is used by tile [subjecttileId+1, querytileId+1]
							scoreBorderDiag[(subjecttileId+1) * (querytiles+1) + (querytileId+1)] = scoreCur;	
						}								
					}
					
					block.sync(); //tile finished
#if 0
					if(block.thread_rank() == 0){
						printf("scoreBorderSubject after tile diag %d.\n", tilediagonal);
						for(int i = 0; i < buffers.max_sequence_length; i++){
							printf("%4d", scoreBorderSubject[i]);
						}
						printf("\n");
						printf("scoreBorderQuery after tile diag %d.\n", tilediagonal);
						for(int i = 0; i < buffers.max_sequence_length; i++){
							printf("%4d", scoreBorderQuery[i]);
						}
						printf("\n");
					}
					block.sync();
#endif
				}
				
				//borders contain scores of last row / last column. perform max reduction to find alignment begin
				
				BestscoreData rowData;
				rowData.bestScore = SHRT_MIN;
				rowData.bestIndex = block.thread_rank();
				BestscoreData colData;
				colData.bestScore = SHRT_MIN;
				colData.bestIndex = block.thread_rank();
				
				if(block.thread_rank() < subjectbases){
					//printf("sborder %d : %d\n", block.thread_rank(), scoreBorderSubject[block.thread_rank()]);
					rowData.bestScore = scoreBorderSubject[block.thread_rank()];				
				}
				block.sync();
				if(block.thread_rank() < querybases){
					//printf("qborder %d : %d\n", block.thread_rank(), scoreBorderQuery[block.thread_rank()]);
					colData.bestScore = scoreBorderQuery[block.thread_rank()];				
				}				
				
				for (int i = warp.size() / 2; i > 0; i /= 2) {
					BestscoreData otherdata;
					otherdata.data = warp.shfl_down(rowData.data, i);
					rowData = rowData.bestScore > otherdata.bestScore ? rowData : otherdata;
				}
				for (int i = warp.size() / 2; i > 0; i /= 2) {
					BestscoreData otherdata;
					otherdata.data = warp.shfl_down(colData.data, i);
					colData = colData.bestScore > otherdata.bestScore ? colData : otherdata;
				}
				if(warp.thread_rank() == 0){
					bestRowData[warpId] = rowData;
					bestColData[warpId] = colData;
				}
				block.sync();
				
				if(block.thread_rank() == 0){
					
					AlignResultCompact * my_result_out = buffers.results + queryId;
					AlignOp * my_ops_out = buffers.ops + buffers.max_ops_per_alignment * queryId;
				
					//perform final reduction step of warp results, then backtrack alignment
					rowData = bestRowData[0];
					colData = bestColData[0];
					for(int i = 1; i < nwarps; i++){
						rowData = bestRowData[i].bestScore > rowData.bestScore ? bestRowData[i] : rowData;
						colData = bestColData[i].bestScore > colData.bestScore ? bestColData[i] : colData;
					}
					
					short currow;
					short curcol;

					AlignResultCompact result;

					if (colData.bestScore > rowData.bestScore) {
						currow = subjectbases-1;
						curcol = colData.bestIndex;
						result.score = colData.bestScore;
					}else{
						currow = rowData.bestIndex;
						curcol = querybases-1;
						result.score = rowData.bestScore;
					}
					
					//printf("queryId %d, currow %d, curcol %d , score %d\n", queryId, currow, curcol, result.score);
		
					int subject_end_excl = currow+1;

					//printf("currow %d, curcol %d\n", currow, curcol);

					int nOps = 0;
					bool isValid = true;
					AlignOp currentOp;
					bool onceflag = false;
					unsigned long long previousPrev = 0;
					char previousType = 0;
					while(currow != -1 && curcol != -1){
						unsigned int colbyteindex = curcol / 32;
						unsigned int colbitindex = curcol % 32;
						previousPrev = prevs[currow * prevcolumncount + colbyteindex];
						if(onceflag){
							printf("queryId %d, subjectbases %d querybases %d, currow %d, curcol %d , score %d, prevs %lu %d\n", queryId, subjectbases, querybases,  currow, curcol, result.score, prevs[currow * prevcolumncount + colbyteindex], (prevs[currow * prevcolumncount + colbyteindex] >> 2*(31-colbitindex)) & 0x3);
							onceflag = false;
						}
						switch((previousPrev >> 2*(31-colbitindex)) & 0x3){
			
						case ALIGNTYPE_MATCH:
							curcol -= 1;
							currow -= 1;
							previousType = ALIGNTYPE_MATCH;
							break;
						case ALIGNTYPE_SUBSTITUTE:

							currentOp.position = currow;
							currentOp.type = ALIGNTYPE_SUBSTITUTE;
							currentOp.base = twobitToChar(encoded_accessor(sr2, querybases, curcol));

							my_ops_out[nOps] = currentOp;
							++nOps;

							curcol -= 1;
							currow -= 1;
							previousType = ALIGNTYPE_SUBSTITUTE;
							break;
						case ALIGNTYPE_DELETE:

							currentOp.position = currow;
							currentOp.type = ALIGNTYPE_DELETE;
							currentOp.base = twobitToChar(encoded_accessor(sr1, subjectbases, currow));

							my_ops_out[nOps] = currentOp;
							++nOps;

							curcol -= 0;
							currow -= 1;
							previousType = ALIGNTYPE_DELETE;
							break;
						case ALIGNTYPE_INSERT:

							currentOp.position = currow+1;
							currentOp.type = ALIGNTYPE_INSERT;
							currentOp.base = twobitToChar(encoded_accessor(sr2, querybases, curcol));

							my_ops_out[nOps] = currentOp;
							++nOps;

							curcol -= 1;
							currow -= 0;
							previousType = ALIGNTYPE_INSERT;
							break;
						default : // code should not reach here
							isValid = false;
							printf("alignment backtrack error");
						}
					}
					//undo last advance to get correct curcol and currow
					switch(previousType){		
					case ALIGNTYPE_MATCH:
						curcol += 1;
						currow += 1;
						break;
					case ALIGNTYPE_SUBSTITUTE:
						curcol += 1;
						currow += 1;
						break;
					case ALIGNTYPE_DELETE:
						curcol += 0;
						currow += 1;
						break;
					case ALIGNTYPE_INSERT:
						curcol += 1;
						currow += 0;
						break;
					default : break;
					}
					
					result.subject_begin_incl = max(currow, 0);
					result.query_begin_incl = max(curcol, 0);
					result.overlap = subject_end_excl - result.subject_begin_incl;
					result.shift = result.subject_begin_incl == 0 ? -result.query_begin_incl : result.subject_begin_incl;
					result.nOps = nOps;
					result.isNormalized = false;
					result.isValid = isValid;

					//printf("currow %d, curcol %d, subject_begin_incl %d, query_begin_incl %d, overlap %d, shift %d, nOps %d\n", currow, curcol, result.subject_begin_incl, result.query_begin_incl, result.overlap, result.shift, result.nOps);

					*my_result_out = result;					
					
					
				}	
			}
		}
		
		void call_cuda_semi_global_alignment_warps_kernel_async(const sgaparams& buffers, cudaStream_t stream){

				size_t smem = cuda_semi_global_alignment_warps_getSharedMemSize(buffers);
				dim3 block(32 * SDIV(buffers.max_sequence_length, 32), 1, 1);
				dim3 grid(buffers.n_queries, 1, 1);
				//dim3 grid(1, 1, 1);
				//dim3 grid(min(104,buffers.n_queries), 1, 1);
				// start kernel

				//printf("launch [%d %d %d] [%d %d %d] %lu\n", grid.x, grid.y, grid.z, block.x, block.y, block.z, smem);
				cuda_semi_global_alignment_warps_kernel
					<<<grid, block, smem, stream>>>(buffers);
				CUERR;
		}

		void call_cuda_semi_global_alignment_warps_kernel(const sgaparams& buffers, cudaStream_t stream){

				call_cuda_semi_global_alignment_warps_kernel_async(buffers, stream);

				cudaStreamSynchronize(stream); CUERR;
		}
		
		
		size_t cuda_semi_global_alignment_warps_getSharedMemSize(const sgaparams& buffers){
			int nwarps = SDIV(buffers.max_sequence_length, 32);

			size_t smem = 0;
			smem += sizeof(char) * (buffers.max_sequence_bytes + buffers.max_sequence_bytes); //current subject and current query
			smem += sizeof(unsigned long long); // padding for prevs
			smem += sizeof(unsigned long long) * buffers.max_sequence_length * nwarps; // prevs
			smem += sizeof(short); // padding for pointer alignment
			smem += sizeof(short) * 2 * buffers.max_sequence_length; //border
			smem += sizeof(short) * 2 * (SDIV(buffers.max_sequence_length, 32) + 1); // border diags
			smem += sizeof(int);
			smem += sizeof(int) * 2 * nwarps; // bestscore data

			return smem;
		}
		
		
		
		
		
		
		
		
		
		
		
		
		

		size_t cuda_semi_global_alignment_getSharedMemSize(const AlignerDataArrays& buffers){


			size_t smem = 0;
			smem += sizeof(char) * (buffers.max_sequence_bytes + buffers.max_sequence_bytes); //current subject and current query
			smem += sizeof(char) * (buffers.max_sequence_length+1)*SDIV(buffers.max_sequence_length+1, 4); // prevs
			smem += sizeof(short); // padding for pointer alignment
			smem += sizeof(short) * 3*2*(buffers.max_sequence_length + 1); // wavefronts
			smem += sizeof(short) * 4; //best(row/col), best(row/col)score

			return smem;
		}

		/*
		align multiple queries to multiple subjects
		*/
		__global__
		void cuda_semi_global_alignment_kernel(const AlignerDataArrays buffers){

			const unsigned int WAVE_ROW_SIZE = 2*(buffers.max_sequence_length + 1);

			extern __shared__ char smem[];

			const int prevcolumncount = SDIV(buffers.max_sequence_length + 1, 4);

			/* set up shared memory */
			char* sr1 = (char*)(smem);
			char* sr2 = (char*)(sr1 + buffers.max_sequence_bytes);
			char* prevs = (char*)(sr2 + buffers.max_sequence_bytes);
			char* tmp = (char*)(prevs + (buffers.max_sequence_length + 1)*prevcolumncount);			
			unsigned long offset = (unsigned long)tmp % sizeof(short); //ensure correct pointer alignment for shorts
			if(offset != 0)
				tmp += sizeof(short) - offset;

			short* scores = (short*)(tmp);
			short* bestrow = (short*)(scores + 3 * WAVE_ROW_SIZE);
			short* bestcol = (short*)(bestrow + 1);
			short* bestrowscore = (short*)(bestcol + 1);
			short* bestcolscore = (short*)(bestrowscore + 1);

			for(int globalQueryId = blockIdx.x; globalQueryId < buffers.n_queries; globalQueryId += gridDim.x){

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
				const int subjectbases = buffers.d_lengths[subjectId];
				const int querybases = buffers.d_lengths[buffers.n_subjects + globalQueryId];
				AlignResultCompact * const my_result_out = buffers.d_results + globalQueryId;
				AlignOp * const my_ops_out = buffers.d_ops + buffers.max_ops_per_alignment * globalQueryId;

				//set up shared memory

				for(int threadid = threadIdx.x; threadid < buffers.max_sequence_bytes; threadid += blockDim.x){
					sr1[threadid] = subject[threadid];
					sr2[threadid] = query[threadid];			
				}

				for (int l = threadIdx.x; l < 2*(buffers.max_sequence_length + 1); l += blockDim.x) {
					scores[0*WAVE_ROW_SIZE+l] = 0;
					scores[1*WAVE_ROW_SIZE+l] = 0;
					scores[2*WAVE_ROW_SIZE+l] = 0;
				}

				for(int i = 0; i < buffers.max_sequence_length + 1; i++){
					for (int j = threadIdx.x; j < prevcolumncount; j += blockDim.x) {
						prevs[i * prevcolumncount + j] = 0;
					}
				}


				if (threadIdx.x == 0) {
					*bestrow = 0;
					*bestcol = 0;
					*bestrowscore = SHRT_MIN;
					*bestcolscore = SHRT_MIN;
				}

				__syncthreads();



				for (int baseRow = 2; baseRow < 2*subjectbases+1; ++baseRow) {
		
					const int targetrow = baseRow % 3;
					const int indelrow = (targetrow == 0 ? 2 : targetrow - 1);
					const int matchrow = (indelrow == 0 ? 2 : indelrow - 1);

					// find all threads inside boundary and calculate the score
					for(int threadid = threadIdx.x; threadid < querybases + 1; threadid += blockDim.x){

						// row and col in 2d space
						const int myrow = baseRow - threadid;
						const int mycol = threadid;

						// calculate entry [baseRow - threadid][threadid]
						if((myrow > 0) && (myrow < subjectbases + 1) 
							&& (mycol > 0) && (mycol < querybases + 1)){ 

							bool ismatch = encoded_accessor(sr1, subjectbases, myrow - 1) == encoded_accessor(sr2, querybases, mycol - 1);

							const short matchscore = scores[matchrow * WAVE_ROW_SIZE + threadid - 1] 
										+ (ismatch ? buffers.ALIGNMENTSCORE_MATCH : buffers.ALIGNMENTSCORE_SUB);
							const short insscore = scores[indelrow * WAVE_ROW_SIZE + threadid - 1] + buffers.ALIGNMENTSCORE_INS;
							const short delscore = scores[indelrow * WAVE_ROW_SIZE + threadid] + buffers.ALIGNMENTSCORE_DEL;

							short maximum = 0;
							AlignType prev;
							unsigned int colbyteindex = mycol / 4;
							unsigned int colbitindex = mycol % 4;
		
							if (matchscore < delscore) {
								maximum = delscore;
								prev = ALIGNTYPE_DELETE;
							}else{
								maximum = matchscore;
								prev = ismatch ? ALIGNTYPE_MATCH : ALIGNTYPE_SUBSTITUTE;
							}
							if (maximum < insscore) {
								maximum = insscore;
								prev = ALIGNTYPE_INSERT;
							}

							prevs[myrow * prevcolumncount + colbyteindex] |= (prev << 2*(3-colbitindex));

							scores[targetrow * WAVE_ROW_SIZE + threadid] = maximum;

							// update best score in last row
							if (myrow == subjectbases) {
								if (*bestcolscore < maximum) {
									*bestcolscore = maximum;
									*bestcol = mycol;
								}
							}

							// update best score in last column
							if (mycol == querybases) {
								if (*bestrowscore < maximum) {
									*bestrowscore = maximum;
									*bestrow = myrow;
								}
							}
						}
					}

					__syncthreads();
				}

				// get alignment and alignment score
				if (threadIdx.x == 0) {

					short currow;
					short curcol;

					AlignResultCompact result;

					if (*bestcolscore > *bestrowscore) {
						currow = subjectbases;
						curcol = *bestcol;
						result.score = *bestcolscore;
					}else{
						currow = *bestrow;
						curcol = querybases;
						result.score = *bestrowscore;
					}
		
					const int subject_end_excl = currow;

					//printf("currow %d, curcol %d\n", currow, curcol);

					int nOps = 0;
					bool isValid = true;
					AlignOp currentOp;

					while(currow != 0 && curcol != 0){
						unsigned int colbyteindex = curcol / 4;
						unsigned int colbitindex = curcol % 4;
						switch((prevs[currow * prevcolumncount + colbyteindex] >> 2*(3-colbitindex)) & 0x3){
			
						case ALIGNTYPE_MATCH:
							curcol -= 1;
							currow -= 1;
							break;
						case ALIGNTYPE_SUBSTITUTE:

							currentOp.position = currow - 1;
							currentOp.type = ALIGNTYPE_SUBSTITUTE;
							currentOp.base = twobitToChar(encoded_accessor(sr2, querybases, curcol - 1));

							my_ops_out[nOps] = currentOp;
							++nOps;

							curcol -= 1;
							currow -= 1;
							break;
						case ALIGNTYPE_DELETE:

							currentOp.position = currow - 1;
							currentOp.type = ALIGNTYPE_DELETE;
							currentOp.base = twobitToChar(encoded_accessor(sr1, subjectbases, currow - 1));

							my_ops_out[nOps] = currentOp;
							++nOps;

							curcol -= 0;
							currow -= 1;
							break;
						case ALIGNTYPE_INSERT:

							currentOp.position = currow;
							currentOp.type = ALIGNTYPE_INSERT;
							currentOp.base = twobitToChar(encoded_accessor(sr2, querybases, curcol - 1));

							my_ops_out[nOps] = currentOp;
							++nOps;

							curcol -= 1;
							currow -= 0;
							break;
						default : // code should not reach here
							isValid = false;
							printf("alignment backtrack error");
						}
					}
					result.subject_begin_incl = currow;
					result.query_begin_incl = curcol;
					result.overlap = subject_end_excl - result.subject_begin_incl;
					result.shift = result.subject_begin_incl == 0 ? -result.query_begin_incl : result.subject_begin_incl;
					result.nOps = nOps;
					result.isNormalized = false;
					result.isValid = isValid;

					*my_result_out = result;
				}
			}
		}

		void call_cuda_semi_global_alignment_kernel_async(const AlignerDataArrays& buffers){

				size_t smem = cuda_semi_global_alignment_getSharedMemSize(buffers);
				dim3 block(std::min(512, 32 * SDIV(buffers.max_sequence_length+1, 32)), 1, 1);
				//dim3 grid(std::min(buffers.max_blocks, buffers.n_queries), 1, 1);
				dim3 grid(buffers.n_queries, 1, 1);
				// start kernel
				
				cuda_semi_global_alignment_kernel
					<<<grid, block, smem, buffers.stream>>>(buffers);
				CUERR;
		}

		void call_cuda_semi_global_alignment_kernel(const AlignerDataArrays& buffers){

				call_cuda_semi_global_alignment_kernel_async(buffers);

				cudaStreamSynchronize(buffers.stream); CUERR;
		}





//----------------------------below is unused------------------------------------------

		size_t cuda_semi_global_align_multiwarp_getSharedMemSize(int maxLengthR1, int maxLengthR2, int threadsPerBlock){


			constexpr int WARP_SIZE2 = 32;

			union ShuffleData{
				short scoreCurUp[2];
				int data;
			};

			int maxlen = max(maxLengthR1, maxLengthR2);

			size_t smem = 0;
			smem += sizeof(char) * (maxLengthR1 + maxLengthR2);
			smem += sizeof(char) * (maxLengthR1+1)*SDIV(maxLengthR2+1, 4);
			smem += sizeof(short); // padding for pointer alignment
			smem += sizeof(short) * 4;
			smem += sizeof(ShuffleData); // padding for pointer alignment
			smem += sizeof(ShuffleData) * SDIV(threadsPerBlock, WARP_SIZE2);

			return smem;
		}

		/*
			Align multiple queries to single subject. scores in registers, warp shuffle stuff

		*/

		__global__
		void cuda_semi_global_align_multiwarp_kernel(AlignResultCompact* result_out, AlignOp* ops_out,
						const int alignments, const int max_ops, const char* r1, const char* r2, 
						const int r1bytes, const int r1length, 
						const int* r2bytesPrefixSum, const int* r2lengths,
						int maxLengthR1, int maxLengthR2, const int SCORE_EQUAL, const int SCORE_SUBSTITUTE,
			const int SCORE_INSERT, const int SCORE_DELETE)
		{
			constexpr int WARP_SIZE2 = 32;
			constexpr short MIN_SHORT = SHRT_MIN;

			union ShuffleData{
				short scoreCurUp[2];
				int data;
			};

			extern __shared__ char smem[];

			char* sr1 = (char*)(smem);
			char* sr2 = (char*)(sr1 + maxLengthR1);
			char* prevs = (char*)(sr2 + maxLengthR2);

			char* tmp = (char*)(prevs + (maxLengthR1 + 1)*SDIV(maxLengthR2 + 1, 4));

			//ensure correct pointer alignment
			unsigned long offset = (unsigned long)tmp % sizeof(short);
			if(offset != 0)
				tmp += sizeof(short) - offset;

			short* bestrow = (short*)(tmp);
			short* bestcol = (short*)(bestrow + 1);
			short* bestrowscore = (short*)(bestcol + 1);
			short* bestcolscore = (short*)(bestrowscore + 1);

			tmp = (char*)(bestcolscore + 1);

			//ensure correct pointer alignment
			offset = (unsigned long)tmp % sizeof(ShuffleData);
			if(offset != 0)
				tmp += sizeof(ShuffleData) - offset;

			ShuffleData* border = (ShuffleData*)(tmp);


		//#define CUDA_SEMI_GLOBAL_ALIGN_MULTIWARP_SCOREDEBUGGING

		#ifdef CUDA_SEMI_GLOBAL_ALIGN_MULTIWARP_SCOREDEBUGGING
			__shared__ short scores[maxLengthR1+1][maxLengthR2+1]; // for debugging
		#endif

			const int tid = threadIdx.x;
			const int bid = blockIdx.x;
			const int lane = tid % WARP_SIZE2;
			const int warpId = tid / WARP_SIZE2;
			const int warps = SDIV(blockDim.x, WARP_SIZE2);

			//copy subject (r1) into shared memory
			for(int i = tid; i < r1bytes; i += blockDim.x){
				sr1[i] = r1[i];			
			}


			for(int alignmentId = bid; alignmentId < alignments; alignmentId += gridDim.x){

		#ifdef CUDA_SEMI_GLOBAL_ALIGN_MULTIWARP_SCOREDEBUGGING
				if(tid == 0)
					for(int i = 0; i < maxLengthR1+1; i++)
						for(int j = 0; j < maxLengthR2+1; j++)
							scores[i][j] = 0;
		#endif

				AlignResultCompact * const my_result_out = result_out + alignmentId;
				AlignOp * const my_ops_out = ops_out + max_ops * bid;

				const char * my_r2 = r2 + r2bytesPrefixSum[alignmentId];
				const int r2bytes = r2bytesPrefixSum[alignmentId+1] - r2bytesPrefixSum[alignmentId];
				const int r2bases = r2lengths[alignmentId];

				if(blockDim.x < r2bases + 1)
					continue; // not enough threads to compute alignment. cannot use strided loops

				short scoreLeft = 0;
				short scoreDiag = 0;
				ShuffleData cu;
				cu.data = 0;

				if (tid == 0) {
					*bestrow = 0;
					*bestcol = 0;
					*bestrowscore = MIN_SHORT;
					*bestcolscore = MIN_SHORT;

					for(int i = 0; i < warps; i++){
						border[i].data = 0;
					}
				}


				for(int i = 0;i < maxLengthR1 + 1; i++){
					for (int j = threadIdx.x; j < SDIV(maxLengthR2 + 1, 4); j += blockDim.x) {
						prevs[i * SDIV(maxLengthR2 + 1, 4) + j] = 0;
					}
				}

				// copy candidate (my_r2) into shared memory

				for(int i = tid; i < r2bytes; i += blockDim.x){
					sr2[i] = my_r2[i];
				}

				__syncthreads();

				// pad number of iters to avoid branch divergence in shuffle instructions
				const int stridedIters = blockDim.x * SDIV(r2bases + 1, blockDim.x); 

				for (int baseRow = 2; baseRow < r1length + 1 + r2bases; ++baseRow) {

					for(int col = tid; col < stridedIters; col += blockDim.x){

						const int myrow = baseRow - col;
						const int mycol = col;

						const int previousLane = lane > 0 ? lane - 1 : 0;

						ShuffleData othercu;
		#if __CUDACC_VER_MAJOR__ < 9        
						othercu.data = __shfl(cu.data, previousLane);
		#else
						othercu.data = __shfl_sync(__activemask(), cu.data, previousLane);
		#endif
						scoreLeft = othercu.scoreCurUp[0];
						scoreDiag = othercu.scoreCurUp[1];
						cu.scoreCurUp[1] = cu.scoreCurUp[0];
				

						// lane 0 from warpid > 0 needs to fetch from border
						if(warpId > 0 && lane == 0){					
							scoreLeft = border[warpId-1].scoreCurUp[0];
							scoreDiag = border[warpId-1].scoreCurUp[1];
						}

						// additional check because of padded number of iterations
						if(mycol < r2bases + 1){

							// boundary condition
							if((myrow > 0) && (myrow < r1length + 1) 
								&& (mycol > 0) && (mycol < r2bases + 1)){ 

								bool ismatch = encoded_accessor(sr1, r1length, myrow - 1) == encoded_accessor(sr2, r2bases, mycol - 1);


								const short matchscore = scoreDiag + (ismatch ? SCORE_EQUAL : SCORE_SUBSTITUTE);
								const short insscore = scoreLeft + SCORE_INSERT;
								const short delscore = cu.scoreCurUp[1] + SCORE_DELETE;//scoreUp + SCORE_DELETE;

								short maximum = 0;
								AlignType prev;
								unsigned int colbyteindex = mycol / 4;
								unsigned int colbitindex = mycol % 4;

								if (matchscore < delscore) {
									maximum = delscore;
									prev = ALIGNTYPE_DELETE;
								}else{
									maximum = matchscore;
									prev = ismatch ? ALIGNTYPE_MATCH : ALIGNTYPE_SUBSTITUTE;
								}
								if (maximum < insscore) {
									maximum = insscore;
									prev = ALIGNTYPE_INSERT;
								}

								prevs[myrow * SDIV(maxLengthR2 + 1, 4) + colbyteindex] |= (prev << 2*(3-colbitindex));


								//scoreCur = maximum;
								cu.scoreCurUp[0] = maximum;
		#ifdef CUDA_SEMI_GLOBAL_ALIGN_MULTIWARP_SCOREDEBUGGING
								scores[myrow][mycol] = maximum;
		#endif
								// update best score in last row
								if (myrow == r1length) {
									if (*bestcolscore < maximum) {
										*bestcolscore = maximum;
										*bestcol = mycol;
									}
								}

								// update best score in last column
								if (mycol == r2bases) {
									if (*bestrowscore < maximum) {
										*bestrowscore = maximum;
										*bestrow = myrow;
									}
								}
							}
						}
					}
					__syncthreads();

					// wavefront is calculated
					// update border for next iteration. no need for padded number of iters
					for(int col = tid; col < r2bases + 1; col += blockDim.x){
						//const int myrow = baseRow - col;
						//const int mycol = col;
						// no need for boundary condition? if last lane is out of bounds, its border is not used by the next warp
						/*if((myrow > 0) && (myrow < r1length + 1) 
							&& (mycol > 0) && (mycol < r2bases + 1)){*/
		 
							if(lane == WARP_SIZE2 - 1){
								border[warpId] = cu;
							}
						//}
					}

					__syncthreads();

				}

		// get alignment and alignment score
				if (tid == 0) {
		#ifdef CUDA_SEMI_GLOBAL_ALIGN_MULTIWARP_SCOREDEBUGGING
		printf("warpscores\n");
				printf("%3s %3s ", "", "");
				for(int c = 0; c < r2bases; c++)
					printf("%3c ", cuda_ordinary_accessor(sr2, c));
				printf("\n");
				for(int r = 0; r < r1length+1; r++){
					if(r == 0)  printf("%3s ", "");
					else  printf("%3c ", cuda_ordinary_accessor(sr1, r - 1));
			
					for(int c = 0; c < r2bases+1; c++){
						printf("%3d ", scores[r][c]);
					}
					printf("\n");
				}
		#endif

					short currow;
					short curcol;

					AlignResultCompact result;

					if (*bestcolscore > *bestrowscore) {
						currow = r1length;
						curcol = *bestcol;
						result.score = *bestcolscore;
					}else{
						currow = *bestrow;
						curcol = r2bases;
						result.score = *bestrowscore;
					}
			
					const int subject_end_excl = currow;

					int nOps = 0;
					bool isValid = 0;
					AlignOp currentOp;

					while(currow != 0 && curcol != 0){

						unsigned int colbyteindex = curcol / 4;
						unsigned int colbitindex = curcol % 4;
						switch((prevs[currow * SDIV(maxLengthR2 + 1, 4) + colbyteindex] >> 2*(3-colbitindex)) & 0x3){
			
						case ALIGNTYPE_MATCH:
							curcol -= 1;
							currow -= 1;
							break;
						case ALIGNTYPE_SUBSTITUTE:

							currentOp.position = currow - 1;
							currentOp.type = ALIGNTYPE_SUBSTITUTE;
							currentOp.base = twobitToChar(encoded_accessor(sr2, r2bases, curcol - 1));

							my_ops_out[nOps] = currentOp;
							++nOps;

							curcol -= 1;
							currow -= 1;
							break;
						case ALIGNTYPE_DELETE:

							currentOp.position = currow - 1;
							currentOp.type = ALIGNTYPE_DELETE;
							currentOp.base = twobitToChar(encoded_accessor(sr1, r1length, currow - 1));

							my_ops_out[nOps] = currentOp;
							++nOps;

							curcol -= 0;
							currow -= 1;
							break;
						case ALIGNTYPE_INSERT:

							currentOp.position = currow;
							currentOp.type = ALIGNTYPE_INSERT;
							currentOp.base = twobitToChar(encoded_accessor(sr2, r2bases, curcol - 1));

							my_ops_out[nOps] = currentOp;
							++nOps;

							curcol -= 1;
							currow -= 0;
							break;
						default : // code should not reach here
							isValid = false;
							printf("alignment backtrack error");
						}
					}

					result.subject_begin_incl = currow;
					result.query_begin_incl = curcol;
					result.overlap = subject_end_excl - result.subject_begin_incl;
					result.shift = result.subject_begin_incl == 0 ? -result.query_begin_incl : result.subject_begin_incl;
					result.nOps = nOps;
					result.isNormalized = false;
					result.isValid = isValid;

					*my_result_out = result;
				}
			}
		}




#endif

	}
}




