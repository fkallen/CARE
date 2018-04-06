#include "../inc/alignment_semi_global.hpp"
#include "../inc/alignment.hpp"
#include "../inc/hpc_helpers.cuh"

#include <stdio.h>
#include <algorithm>
#include <assert.h>


template<class Accessor1, class Accessor2>
AlignResult cpu_semi_global_align_internal(const char* r1_, const char* r2_, int r1length, int r2bases, const int SCORE_EQUAL, const int SCORE_SUBSTITUTE,
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

	Accessor1 r1{r1_, r1length};
	Accessor2 r2{r2_, r2bases};

	// fill matrix
	for (int row = 1; row < r1length + 1; ++row) {
		for (int col = 1; col < r2bases + 1; ++col) {
			// calc entry [row][col]

			const bool ismatch = r1[row - 1] == r2[col - 1];
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
#ifdef CPU_SEMI_GLOBAL_ALIGN_SCOREDEBUGGING
	printf("cpuscores\n");
	printf("%3s %3s ", "", "");
	for(int c = 0; c < r2bases; c++)
		printf("%3c ", r2[c]);
	printf("\n");
	for(int r = 0; r < r1length+1; r++){
		if(r == 0)  printf("%3s ", "");
		else  printf("%3c ", r1[r-1]);
	
		for(int c = 0; c < r2bases+1; c++){
			printf("%3d ", scores[r][c]);
		}
		printf("\n");
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

			operations.push_back(AlignOp{(short)(currow-1), ALIGNTYPE_SUBSTITUTE, r2[curcol - 1]});

			curcol -= 1;
			currow -= 1;
			break;
		case ALIGNTYPE_DELETE:  //printf("d\n");

			operations.push_back(AlignOp{(short)(currow-1), ALIGNTYPE_DELETE, r1[currow - 1]});

			curcol -= 0;
			currow -= 1;
			break;
		case ALIGNTYPE_INSERT:  //printf("i\n");

			operations.push_back(AlignOp{(short)currow, ALIGNTYPE_INSERT, r2[curcol - 1]});

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



AlignResult cpu_semi_global_align(const char* subject, const char* query, int ns, int nq, bool encr1, bool encr2,
	int SCORE_EQUAL, int SCORE_SUBSTITUTE, int SCORE_INSERT, int SCORE_DELETE){

	if(encr1 && encr2)
		return cpu_semi_global_align_internal<EncodedAccessor,EncodedAccessor>(subject, query, ns, nq, SCORE_EQUAL, SCORE_SUBSTITUTE, SCORE_INSERT, SCORE_DELETE);
	else if (encr1 && !encr2)
		return cpu_semi_global_align_internal<EncodedAccessor,OrdinaryAccessor>(subject, query, ns, nq, SCORE_EQUAL, SCORE_SUBSTITUTE, SCORE_INSERT, SCORE_DELETE);
	else if (!encr1 && encr2)
		return cpu_semi_global_align_internal<OrdinaryAccessor,EncodedAccessor>(subject, query, ns, nq, SCORE_EQUAL, SCORE_SUBSTITUTE, SCORE_INSERT, SCORE_DELETE);
	else
		return cpu_semi_global_align_internal<OrdinaryAccessor,OrdinaryAccessor>(subject, query, ns, nq, SCORE_EQUAL, SCORE_SUBSTITUTE, SCORE_INSERT, SCORE_DELETE);
}








#ifdef __CUDACC__

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




size_t cuda_semi_global_align_getSharedMemSize(int maxLengthR1, int maxLengthR2){
	int maxlen = max(maxLengthR1, maxLengthR2);

	size_t smem = 0;
	smem += sizeof(char) * 2 * (maxLengthR1 + maxLengthR2); //sr1, sr2
	smem += sizeof(char) * (maxLengthR1+1)*SDIV(maxLengthR2+1, 4); // prevs
	smem += sizeof(short); // padding for pointer alignment
	smem += sizeof(short) * 3*2*(maxlen + 1); // wavefronts
	smem += sizeof(short) * 4; //best(row/col), best(row/col)score

	return smem;
}


/*
align multiple queries to multiple subjects
*/
__global__
void cuda_semi_global_align_kernel(AlignResultCompact* result_out, AlignOp* ops_out,
				const int max_ops, 
				const int nsubjects, const int ncandidates,
				const char* r1, const char* r2, 
				const int* r1bytesPrefixSum, const int* r1lengths, const int* encodedr1,
				const int* r2bytesPrefixSum, const int* r2lengths, const int* encodedr2,
				const int* candidatesPerSubjectPrefixSum,
				int maxLengthR1, int maxLengthR2, const int SCORE_EQUAL, const int SCORE_SUBSTITUTE,
	const int SCORE_INSERT, const int SCORE_DELETE)
{

	const unsigned int MAX_READ_LENGTH = max(maxLengthR1, maxLengthR2);
	const unsigned int WAVE_ROW_SIZE = 2*(MAX_READ_LENGTH + 1);

	extern __shared__ char smem[];

	/* set up shared memory */
	char* sr1 = (char*)(smem);
	char* sr2 = (char*)(sr1 + maxLengthR1);
	char* prevs = (char*)(sr2 + maxLengthR2);

	char* tmp = (char*)(prevs + (maxLengthR1 + 1)*SDIV(maxLengthR2 + 1, 4));

	//ensure correct pointer alignment for shorts
	unsigned long offset = (unsigned long)tmp % sizeof(short);
	if(offset != 0)
		tmp += sizeof(short) - offset;

	short* scores = (short*)(tmp);
	short* bestrow = (short*)(scores + 3 * WAVE_ROW_SIZE);
	short* bestcol = (short*)(bestrow + 1);
	short* bestrowscore = (short*)(bestcol + 1);
	short* bestcolscore = (short*)(bestrowscore + 1);


//#define CUDA_SEMI_GLOBAL_ALIGN_SCOREDEBUGGING
#ifdef CUDA_SEMI_GLOBAL_ALIGN_SCOREDEBUGGING
	__shared__ short scores2d[maxLengthR1+1][maxLengthR2+1];
#endif
	for(int globalCandidateId = blockIdx.x; globalCandidateId < ncandidates; globalCandidateId += gridDim.x){

		int subjectId = 0;
		for(int i = 0; i < nsubjects; i++)
			if(globalCandidateId >= candidatesPerSubjectPrefixSum[i])
				subjectId = i;

		const char * const my_r1 = r1 + r1bytesPrefixSum[subjectId];

		const int r1bytes = r1bytesPrefixSum[subjectId+1] - r1bytesPrefixSum[subjectId];
		const int r1bases = r1lengths[subjectId];
		const bool encr1 = encodedr1[subjectId] == 1;

		for(int threadid = threadIdx.x; threadid < r1bytes; threadid += blockDim.x){
			sr1[threadid] = my_r1[threadid];			
		}
	

#ifdef CUDA_SEMI_GLOBAL_ALIGN_SCOREDEBUGGING
			if(threadIdx.x == 0)
				for(int i = 0; i < maxLengthR1+1; i++)
					for(int j = 0; j < maxLengthR2+1; j++)
						scores2d[i][j] = 0;
#endif

			AlignResultCompact * const my_result_out = result_out + globalCandidateId;
			AlignOp * const my_ops_out = ops_out + max_ops * globalCandidateId;

			const char * const my_r2 = r2 + r2bytesPrefixSum[globalCandidateId];

			const int r2bytes = r2bytesPrefixSum[globalCandidateId+1] - r2bytesPrefixSum[globalCandidateId];
			const int r2bases = r2lengths[globalCandidateId];
			const bool encr2 = encodedr2[globalCandidateId] == 1;

			for (int l = threadIdx.x; l < 2*(MAX_READ_LENGTH + 1); l += blockDim.x) {
				scores[0*WAVE_ROW_SIZE+l] = 0;
				scores[1*WAVE_ROW_SIZE+l] = 0;
				scores[2*WAVE_ROW_SIZE+l] = 0;
			}

			for(int i = 0; i < maxLengthR1 + 1; i++){
				for (int j = threadIdx.x; j < SDIV(maxLengthR2 + 1, 4); j += blockDim.x) {
					prevs[i * SDIV(maxLengthR2 + 1, 4) + j] = 0;
				}
			}

			for(int threadid = threadIdx.x; threadid < r2bytes; threadid += blockDim.x){
				sr2[threadid] = my_r2[threadid];
			}

			if (threadIdx.x == 0) {
				*bestrow = 0;
				*bestcol = 0;
				*bestrowscore = -32767;
				*bestcolscore = -32767;
			}

			__syncthreads();

			for (int baseRow = 2; baseRow < r1bases + 1 + r1bases; ++baseRow) {
			
				const int targetrow = baseRow % 3;
				const int indelrow = (targetrow == 0 ? 2 : targetrow - 1);
				const int matchrow = (indelrow == 0 ? 2 : indelrow - 1);

				// find all threads inside boundary and calculate the score
				for(int threadid = threadIdx.x; threadid < r2bases + 1; threadid += blockDim.x){

					// row and col in 2d space
					const int myrow = baseRow - threadid;
					const int mycol = threadid;

					// calculate entry [baseRow - threadid][threadid]
					if((myrow > 0) && (myrow < r1bases + 1) 
						&& (mycol > 0) && (mycol < r2bases + 1)){ 

						bool ismatch;
						if(encr1 && encr2)
							ismatch = cuda_encoded_accessor(sr1, r1bases, myrow - 1) == cuda_encoded_accessor(sr2, r2bases, mycol - 1);
						else if (encr1 && !encr2)
							ismatch = cuda_encoded_accessor(sr1, r1bases, myrow - 1) == cuda_ordinary_accessor(sr2, mycol - 1);
						else if (!encr1 && encr2)
							ismatch = cuda_ordinary_accessor(sr1, myrow - 1) == cuda_encoded_accessor(sr2, r2bases, mycol - 1);
						else if (!encr1 && !encr2)
							ismatch = cuda_ordinary_accessor(sr1, myrow - 1) == cuda_ordinary_accessor(sr2, mycol - 1);

						const short matchscore = scores[matchrow * WAVE_ROW_SIZE + threadid - 1] + (ismatch ? SCORE_EQUAL : SCORE_SUBSTITUTE);
						const short insscore = scores[indelrow * WAVE_ROW_SIZE + threadid - 1] + SCORE_INSERT;
						const short delscore = scores[indelrow * WAVE_ROW_SIZE + threadid] + SCORE_DELETE;

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

						scores[targetrow * WAVE_ROW_SIZE + threadid] = maximum;
	#ifdef CUDA_SEMI_GLOBAL_ALIGN_SCOREDEBUGGING
						scores2d[myrow][mycol] = maximum;
	#endif


						// update best score in last row
						if (myrow == r1bases) {
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

				__syncthreads();
			}

			// get alignment and alignment score
			if (threadIdx.x == 0) {
	#ifdef CUDA_SEMI_GLOBAL_ALIGN_SCOREDEBUGGING
				printf("normalgpuscores\n");
				printf("%3s %3s ", "", "");
				for(int c = 0; c < r2bases; c++)
					printf("%3c ", cuda_ordinary_accessor(sr2, c));
				printf("\n");
				for(int r = 0; r < r1length+1; r++){
					if(r == 0)  printf("%3s ", "");
					else  printf("%3c ", cuda_ordinary_accessor(sr1, r - 1));
			
					for(int c = 0; c < r2bases+1; c++){
						printf("%3d ", scores2d[r][c]);
					}
					printf("\n");
				}
	#endif

				short currow;
				short curcol;

				AlignResultCompact result;

				if (*bestcolscore > *bestrowscore) {
					currow = r1bases;
					curcol = *bestcol;
					result.score = *bestcolscore;
				}else{
					currow = *bestrow;
					curcol = r2bases;
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
					switch((prevs[currow * SDIV(maxLengthR2 + 1, 4) + colbyteindex] >> 2*(3-colbitindex)) & 0x3){
				
					case ALIGNTYPE_MATCH:
						curcol -= 1;
						currow -= 1;
						break;
					case ALIGNTYPE_SUBSTITUTE:

						currentOp.position = currow - 1;
						currentOp.type = ALIGNTYPE_SUBSTITUTE;

						if(encr2)
							currentOp.base = cuda_encoded_accessor(sr2, r2bases, curcol - 1);
						else
							currentOp.base = cuda_ordinary_accessor(sr2, curcol - 1);

						my_ops_out[nOps] = currentOp;
						++nOps;

						curcol -= 1;
						currow -= 1;
						break;
					case ALIGNTYPE_DELETE:

						currentOp.position = currow - 1;
						currentOp.type = ALIGNTYPE_DELETE;

						if(encodedr1)
							currentOp.base = cuda_encoded_accessor(sr1, r1bases, currow - 1);
						else
							currentOp.base = cuda_ordinary_accessor(sr1, currow - 1);

						my_ops_out[nOps] = currentOp;
						++nOps;

						curcol -= 0;
						currow -= 1;
						break;
					case ALIGNTYPE_INSERT:

						currentOp.position = currow;
						currentOp.type = ALIGNTYPE_INSERT;

						if(encr2)
							currentOp.base = cuda_encoded_accessor(sr2, r2bases, curcol - 1);
						else
							currentOp.base = cuda_ordinary_accessor(sr2, curcol - 1);

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


void call_cuda_semi_global_align_kernel_async(AlignResultCompact* result_out, AlignOp* ops_out,
				const int max_ops, 
				const int nsubjects, const int ncandidates,
				const char* r1, const char* r2, 
				const int* r1bytesPrefixSum, const int* r1lengths, const int* encodedr1,
				const int* r2bytesPrefixSum, const int* r2lengths, const int* encodedr2,
				const int* candidatesPerSubjectPrefixSum,
				int maxLengthR1, int maxLengthR2, int SCORE_EQUAL, int SCORE_SUBSTITUTE,
				int SCORE_INSERT, int SCORE_DELETE, cudaStream_t stream){

		size_t smem = cuda_semi_global_align_getSharedMemSize(maxLengthR1, maxLengthR2);

		dim3 block(std::min(512, 32 * SDIV(maxLengthR2+1, 32)), 1, 1);
		dim3 grid(ncandidates, 1, 1);

		// start kernel

		cuda_semi_global_align_kernel
			<<<grid, block, smem, stream>>>(
				result_out,
				ops_out,
				max_ops, 
				nsubjects, 
				ncandidates,
				r1, 
				r2,
				r1bytesPrefixSum,
				r1lengths,
				encodedr1,
				r2bytesPrefixSum,
				r2lengths,
				encodedr2,
				candidatesPerSubjectPrefixSum,
				maxLengthR1, maxLengthR2,
				SCORE_EQUAL, SCORE_SUBSTITUTE, 
				SCORE_INSERT, SCORE_DELETE);
		CUERR;
}

void call_cuda_semi_global_align_kernel(AlignResultCompact* result_out, AlignOp* ops_out,
				const int max_ops, 
				const int nsubjects, const int ncandidates,
				const char* r1, const char* r2, 
				const int* r1bytesPrefixSum, const int* r1lengths, const int* encodedr1,
				const int* r2bytesPrefixSum, const int* r2lengths, const int* encodedr2,
				const int* candidatesPerSubjectPrefixSum,
				int maxLengthR1, int maxLengthR2, int SCORE_EQUAL, int SCORE_SUBSTITUTE,
				int SCORE_INSERT, int SCORE_DELETE, cudaStream_t stream){

		call_cuda_semi_global_align_kernel_async(
				result_out,
				ops_out,
				max_ops, 
				nsubjects, 
				ncandidates,
				r1, 
				r2,
				r1bytesPrefixSum,
				r1lengths,
				encodedr1,
				r2bytesPrefixSum,
				r2lengths,
				encodedr2,
				candidatesPerSubjectPrefixSum,
				maxLengthR1, maxLengthR2,
				SCORE_EQUAL, SCORE_SUBSTITUTE, 
				SCORE_INSERT, SCORE_DELETE, stream);

		cudaStreamSynchronize(stream); CUERR;
}






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
				const int r1bytes, const int r1length, const bool encodedr1,
				const int* r2bytesPrefixSum, const int* r2lengths,
				const int* encodedr2,
				int maxLengthR1, int maxLengthR2, const int SCORE_EQUAL, const int SCORE_SUBSTITUTE,
	const int SCORE_INSERT, const int SCORE_DELETE)
{
	constexpr int WARP_SIZE2 = 32;
	constexpr short MIN_SHORT = -32767;

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
		const bool encr2 = encodedr2[alignmentId] == 1;

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

						bool ismatch = false;

						//find out if r1[myrow-1] == r2[mycol-1]
						if(encodedr1 && encr2)
							ismatch = cuda_encoded_accessor(sr1, r1length, myrow - 1) == cuda_encoded_accessor(sr2, r2bases, mycol - 1);
						else if (encodedr1 && !encr2)
							ismatch = cuda_encoded_accessor(sr1, r1length, myrow - 1) == cuda_ordinary_accessor(sr2, mycol - 1);
						else if (!encodedr1 && encr2)
							ismatch = cuda_ordinary_accessor(sr1, myrow - 1) == cuda_encoded_accessor(sr2, r2bases, mycol - 1);
						else if (!encodedr1 && !encr2)
							ismatch = cuda_ordinary_accessor(sr1, myrow - 1) == cuda_ordinary_accessor(sr2, mycol - 1);

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

					if(encr2)
						currentOp.base = cuda_encoded_accessor(sr2, r2bases, curcol - 1);
					else
						currentOp.base = cuda_ordinary_accessor(sr2, curcol - 1);

					my_ops_out[nOps] = currentOp;
					++nOps;

					curcol -= 1;
					currow -= 1;
					break;
				case ALIGNTYPE_DELETE:

					currentOp.position = currow - 1;
					currentOp.type = ALIGNTYPE_DELETE;

					if(encodedr1)
						currentOp.base = cuda_encoded_accessor(sr1, r1length, currow - 1);
					else
						currentOp.base = cuda_ordinary_accessor(sr1, currow - 1);

					my_ops_out[nOps] = currentOp;
					++nOps;

					curcol -= 0;
					currow -= 1;
					break;
				case ALIGNTYPE_INSERT:

					currentOp.position = currow;
					currentOp.type = ALIGNTYPE_INSERT;

					if(encr2)
						currentOp.base = cuda_encoded_accessor(sr2, r2bases, curcol - 1);
					else
						currentOp.base = cuda_ordinary_accessor(sr2, curcol - 1);

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
#endif // #ifdef __CUDACC__

