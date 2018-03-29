#ifndef CARE_SEMI_GLOBAL_ALIGNMENT_IMPL_HPP
#define CARE_SEMI_GLOBAL_ALIGNMENT_IMPL_HPP

#include "alignment.hpp"
#include "options.hpp"
#include "semi_global_alignment.hpp"

#include <limits>
#include <vector>

namespace care{

/*
        CPU alignment
*/

template<class Accessor>
AlignResult
cpu_semi_global_alignment_impl(const SGAdata* buffers,
                               const AlignmentOptions& alignmentOptions,
                               const char* subject,
                               const char* query,
                               int subjectbases,
                               int querybases,
                               Accessor getChar){

    assert(subjectbases < std::numeric_limits<short>::max());
    assert(querybases < std::numeric_limits<short>::max());

//#define CPU_SEMI_GLOBAL_ALIGN_SCOREDEBUGGING

    int scores[subjectbases + 1][querybases + 1];
    char prevs[subjectbases + 1][querybases + 1];


    // init
    for (int col = 0; col < querybases + 1; ++col) {
        scores[0][col] = 0;
    }

    // row 0 was filled by column loop
    for (int row = 1; row < subjectbases + 1; ++row) {
        scores[row][0] = 0;
    }

    // fill matrix
    for (int row = 1; row < subjectbases + 1; ++row) {
        for (int col = 1; col < querybases + 1; ++col) {
            // calc entry [row][col]

            const bool ismatch = getChar(subject, subjectbases, row - 1) == getChar(query, querybases, col - 1);
            const int matchscore = scores[row - 1][col - 1]
                        + (ismatch ? alignmentOptions.alignmentscore_match : alignmentOptions.alignmentscore_sub);
            const int insscore = scores[row][col - 1] + alignmentOptions.alignmentscore_ins;
            const int delscore = scores[row - 1][col] + alignmentOptions.alignmentscore_del;

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
    for (int row = 0; row < subjectbases + 1; ++row) {
        if(row != 1 && (row-1)%32 == 0){
            std::cout << std::endl;
            std::cout << std::endl;
            for (int col = 1; col < querybases + 1; ++col) {
                std::cout << "____";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for (int col = 0; col < querybases + 1; ++col) {
            if((col-1)%32 == 0) std::cout << " | ";
            std::cout << std::setw(4) << scores[row][col];
        }

        std::cout << std::endl;
    }
#endif
    // extract best alignment

    int currow = subjectbases;
    int curcol = querybases;
    int maximum = std::numeric_limits<int>::min();

    for (int row = 1; row < subjectbases + 1; ++row) {
        if(scores[row][querybases] > maximum){
            //short oldmax = maximum;
            maximum = scores[row][querybases];
            currow = row;
            curcol = querybases;

            /*std::cout << "row = " << row << ": \n"
                << scores[row][querybases] << " > " << oldmax << "\n"
                << " update currow " << currow << ", curcol " << curcol << std::endl;*/
        }
    }

    for (int col = 1; col < querybases + 1; ++col) {
        if(scores[subjectbases][col] > maximum){
            //short oldmax = maximum;
            maximum = scores[subjectbases][col];
            currow = subjectbases;
            curcol = col;

            /*std::cout << "col = " << col << ": \n"
                << scores[subjectbases][col] << " > " << oldmax << "\n"
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

            operations.push_back(AlignOp{(short)(currow-1), ALIGNTYPE_SUBSTITUTE, getChar(query, querybases, curcol - 1)});

            curcol -= 1;
            currow -= 1;
            break;
        case ALIGNTYPE_DELETE:  //printf("d\n");

            operations.push_back(AlignOp{(short)(currow-1), ALIGNTYPE_DELETE, getChar(subject, subjectbases, currow - 1)});

            curcol -= 0;
            currow -= 1;
            break;
        case ALIGNTYPE_INSERT:  //printf("i\n");

            operations.push_back(AlignOp{(short)currow, ALIGNTYPE_INSERT, getChar(query, querybases, curcol - 1)});

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


/*
    GPU alignment
*/

#ifdef __NVCC__

struct sgaparams{
    int max_sequence_length;
    int max_ops_per_alignment;
    int sequencepitch;
    int n_queries;
    int subjectlength;
    int alignmentscore_match = 1;
    int alignmentscore_sub = -1;
    int alignmentscore_ins = -1;
    int alignmentscore_del = -1;
    const int* __restrict__ querylengths;
    const char* __restrict__ subjectdata;
    const char* __restrict__ queriesdata;
    AlignResultCompact* __restrict__ results;
    AlignOp* __restrict__ ops;
};


template<int MAX_SEQUENCE_LENGTH=128, class Accessor>
__global__
void cuda_semi_global_alignment_kernel(const sgaparams buffers, Accessor getChar){

    static_assert(MAX_SEQUENCE_LENGTH % 32 == 0, "MAX_SEQUENCE_LENGTH must be divisible by 32");

    constexpr int prevsPerInt = (sizeof(int)*8/2);

    __shared__ char subject_shared[MAX_SEQUENCE_LENGTH/4];
    __shared__ char query_shared[MAX_SEQUENCE_LENGTH/4];
    __shared__ int prevs[MAX_SEQUENCE_LENGTH*(MAX_SEQUENCE_LENGTH / prevsPerInt)];
    __shared__ short scores[3 * MAX_SEQUENCE_LENGTH];
    __shared__ short bestrow;
    __shared__ short bestcol;
    __shared__ short bestrowscore;
    __shared__ short bestcolscore;

    const int subjectbases = buffers.subjectlength;
    const char* subject = buffers.subjectdata;
    for(int threadid = threadIdx.x; threadid < MAX_SEQUENCE_LENGTH/4; threadid += blockDim.x){
        subject_shared[threadid] = subject[threadid];
    }

    for(int queryId = blockIdx.x; queryId < buffers.n_queries; queryId += gridDim.x){

        const char* query = buffers.queriesdata + queryId * buffers.sequencepitch;
        const int querybases = buffers.querylengths[queryId];
        for(int threadid = threadIdx.x; threadid < MAX_SEQUENCE_LENGTH/4; threadid += blockDim.x){
            query_shared[threadid] = query[threadid];
        }

        AlignResultCompact * const my_result_out = buffers.results + queryId;
        AlignOp * const my_ops_out = buffers.ops + buffers.max_ops_per_alignment * queryId;


        for (int l = threadIdx.x; l < MAX_SEQUENCE_LENGTH; l += blockDim.x) {
            scores[0*MAX_SEQUENCE_LENGTH+l] = 0;
            scores[1*MAX_SEQUENCE_LENGTH+l] = 0;
            scores[2*MAX_SEQUENCE_LENGTH+l] = 0;
        }

        for(int i = 0; i < MAX_SEQUENCE_LENGTH + 1; i++){
            for (int j = threadIdx.x; j < MAX_SEQUENCE_LENGTH / prevsPerInt; j += blockDim.x) {
                prevs[i * MAX_SEQUENCE_LENGTH / prevsPerInt + j] = 0;
            }
        }


        if (threadIdx.x == 0) {
            bestrow = 0;
            bestcol = 0;
            bestrowscore = SHRT_MIN;
            bestcolscore = SHRT_MIN;
        }

        __syncthreads();

        const int globalsubjectpos = threadIdx.x;
        const char subjectbase = getChar(subject_shared, subjectbases, globalsubjectpos);
        int calculatedCells = 0;
        int myprev = 0;

        for (int threaddiagonal = 0; threaddiagonal < subjectbases + querybases - 1; threaddiagonal++) {

            const int targetrow = threaddiagonal % 3;
            const int indelrow = (targetrow == 0 ? 2 : targetrow - 1);
            const int matchrow = (indelrow == 0 ? 2 : indelrow - 1);

            const int globalquerypos = threaddiagonal - threadIdx.x;
            const char querybase = globalquerypos < querybases ? getChar(query_shared, querybases, globalquerypos) : 'F';

            const short scoreDiag = globalsubjectpos == 0 ? 0 : scores[matchrow * MAX_SEQUENCE_LENGTH + threadIdx.x - 1];
            const short scoreLeft = scores[indelrow * MAX_SEQUENCE_LENGTH + threadIdx.x];
            const short scoreUp = globalsubjectpos == 0 ? 0 :  scores[indelrow * MAX_SEQUENCE_LENGTH + threadIdx.x - 1];

            if(globalsubjectpos >= 0 && globalsubjectpos < MAX_SEQUENCE_LENGTH
                && globalquerypos >= 0 && globalquerypos < MAX_SEQUENCE_LENGTH){

                const bool ismatch = subjectbase == querybase;
                const short matchscore = scoreDiag
                            + (ismatch ? buffers.alignmentscore_match : buffers.alignmentscore_sub);
                const short insscore = scoreUp + buffers.alignmentscore_ins;
                const short delscore = scoreLeft + buffers.alignmentscore_del;

                short maximum = 0;
                const unsigned int colindex = globalquerypos / prevsPerInt;

                if (matchscore < delscore) {
                    maximum = delscore;

                    int t = ALIGNTYPE_DELETE;
                    t <<= 2*(prevsPerInt-1- (globalquerypos % prevsPerInt));
                    myprev |= t;
                }else{
                    maximum = matchscore;

                    int t = ismatch ? ALIGNTYPE_MATCH : ALIGNTYPE_SUBSTITUTE;
                    t <<= 2*(prevsPerInt-1- (globalquerypos % prevsPerInt));
                    myprev |= t;
                }
                if (maximum < insscore) {
                    maximum = insscore;

                    int t = ALIGNTYPE_INSERT;
                    t <<= 2*(prevsPerInt-1- (globalquerypos % prevsPerInt));
                    myprev |= t;
                }

                calculatedCells++;
                if(calculatedCells == prevsPerInt || threaddiagonal == subjectbases + querybases - 2){
                    calculatedCells = 0;
                    prevs[globalsubjectpos * (MAX_SEQUENCE_LENGTH / prevsPerInt) + colindex] = myprev;
                    myprev = 0;
                }

                scores[targetrow * MAX_SEQUENCE_LENGTH + threadIdx.x] = maximum;

                if (globalsubjectpos == subjectbases-1) {
                    if (bestcolscore < maximum) {
                        bestcolscore = maximum;
                        bestcol = globalquerypos;
                        //printf("qborder %d : %d\n", mycol - 1, maximum);
                    }
                }

                // update best score in last column
                if (globalquerypos == querybases-1) {
                    if (bestrowscore < maximum) {
                        bestrowscore = maximum;
                        bestrow = globalsubjectpos;
                        //printf("sborder %d : %d\n", myrow - 1, maximum);
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

            if (bestcolscore > bestrowscore) {
                currow = subjectbases-1;
                curcol = bestcol;
                result.score = bestcolscore;
            }else{
                currow = bestrow;
                curcol = querybases-1;
                result.score = bestrowscore;
            }

            const int subject_end_excl = currow + 1;

            //printf("currow %d, curcol %d\n", currow, curcol);

            int nOps = 0;
            bool isValid = true;
            AlignOp currentOp;
            char previousType = 0;
            while(currow != -1 && curcol != -1){
                const unsigned int colIntIndex = curcol / prevsPerInt;
                const unsigned int col2Bitindex = curcol % prevsPerInt;

                switch((prevs[currow * (MAX_SEQUENCE_LENGTH / prevsPerInt) + colIntIndex] >> 2*(prevsPerInt-1-col2Bitindex)) & 0x3){

                case ALIGNTYPE_MATCH:
                    curcol -= 1;
                    currow -= 1;
                    previousType = ALIGNTYPE_MATCH;

                    break;
                case ALIGNTYPE_SUBSTITUTE:
                    currentOp.position = currow;
                    currentOp.type = ALIGNTYPE_SUBSTITUTE;
                    currentOp.base = getChar(query_shared, querybases, curcol);

                    my_ops_out[nOps] = currentOp;
                    ++nOps;

                    curcol -= 1;
                    currow -= 1;
                    previousType = ALIGNTYPE_SUBSTITUTE;

                    break;
                case ALIGNTYPE_DELETE:
                    currentOp.position = currow;
                    currentOp.type = ALIGNTYPE_DELETE;
                    currentOp.base = getChar(subject_shared, subjectbases, currow);

                    my_ops_out[nOps] = currentOp;
                    ++nOps;

                    curcol -= 0;
                    currow -= 1;
                    previousType = ALIGNTYPE_DELETE;

                    break;
                case ALIGNTYPE_INSERT:
                    currentOp.position = currow+1;
                    currentOp.type = ALIGNTYPE_INSERT;
                    currentOp.base = getChar(query_shared, querybases, curcol);

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

//----------------------------below is unused------------------------------------------

#if 0



/*
align multiple queries to subject
*/
template<int MAX_SEQUENCE_LENGTH=128>
__global__
//__launch_bounds__(128, 8)
void cuda_semi_global_alignment_warps_kernel(const sgaparams buffers){

    static_assert(MAX_SEQUENCE_LENGTH % 32 == 0, "MAX_SEQUENCE_LENGTH must be divisible by 32");

    union BestscoreData{
        int data;
        struct{
                short bestScore;
                short bestIndex;
        };
    };

    union prevunion{
        unsigned long long data;
        struct{
            unsigned int hi;
            unsigned int lo;
        };
    };



    /*if(threadIdx.x + blockDim.x * blockIdx.x == 0){
            printf("nqueries : %d\n", buffers.n_queries);
            for(int i = 0; i < buffers.n_queries; i++){
                printf("length %d : %d\n", i, buffers.querylengths[i]);
            }
    }*/

    auto block = this_thread_block();
    auto warp = tiled_partition<WARPSIZE>(block);
    const int warpId = block.thread_rank() / WARPSIZE;
    const int nwarps = block.size() / WARPSIZE;

    //set up shared memory pointers
#if 1
#if 0
    const int prevcolumncount = SDIV(MAX_SEQUENCE_LENGTH, 32);

    char* const subject_shared = (char*)(smem2);
    char* const query_shared = (char*)(subject_shared + MAX_SEQUENCE_LENGTH / 4);
    char* tmp = (char*)(query_shared + MAX_SEQUENCE_LENGTH / 4);
    unsigned long long offset = (unsigned long long)tmp % sizeof(unsigned long long); //ensure correct pointer alignment for unsigned long long
    if(offset != 0)
        tmp += sizeof(unsigned long long) - offset;
    unsigned long long* const prevs = (unsigned long long*)(tmp);
    tmp = (char*)(prevs + MAX_SEQUENCE_LENGTH * prevcolumncount);
    offset = (unsigned long long)tmp % sizeof(short); //ensure correct pointer alignment for short
    if(offset != 0)
        tmp += sizeof(short) - offset;
    short* const scoreBorderSubject = (short*)(tmp);
    short* const scoreBorderQuery = (short*)(scoreBorderSubject + MAX_SEQUENCE_LENGTH);


    //scoreBorderDiag stores diagonal for calculation of tile [subjecttileId, querytileId]
    //number of rows and columns is increased by 1 to avoid boundary check at bottom end / right end
    short* const scoreBorderDiag = (short*)(scoreBorderQuery + MAX_SEQUENCE_LENGTH);
    tmp = (char*)(scoreBorderDiag + 2 * (SDIV(MAX_SEQUENCE_LENGTH, 32)+1));
    offset = (unsigned long long)tmp % sizeof(BestscoreData); //ensure correct pointer alignment for short
    if(offset != 0)
        tmp += sizeof(BestscoreData) - offset;
    BestscoreData* const bestRowData = (BestscoreData*)(tmp);
    BestscoreData* const bestColData = (BestscoreData*)(bestRowData + nwarps);
#else

    __shared__ char subject_shared[MAX_SEQUENCE_LENGTH / 4];
    __shared__ char query_shared[MAX_SEQUENCE_LENGTH / 4];
    __shared__ unsigned long long prevs[MAX_SEQUENCE_LENGTH * MAX_SEQUENCE_LENGTH / 32];
    __shared__ short scoreBorderSubject[MAX_SEQUENCE_LENGTH];
    __shared__ short scoreBorderQuery[MAX_SEQUENCE_LENGTH];
    __shared__ short scoreBorderDiag[(MAX_SEQUENCE_LENGTH/WARPSIZE + 1) * (MAX_SEQUENCE_LENGTH/WARPSIZE + 1)];
    __shared__ BestscoreData bestRowData[MAX_SEQUENCE_LENGTH/WARPSIZE];
    __shared__ BestscoreData bestColData[MAX_SEQUENCE_LENGTH/WARPSIZE];

#endif
#else
    extern __shared__ unsigned long long smem2[];
    const int prevcolumncount = SDIV(buffers.max_sequence_length, 32);

    unsigned long long* prevs = (unsigned long long*)(smem2);
    BestscoreData* bestRowData = (BestscoreData*)(prevs + buffers.max_sequence_length * prevcolumncount);
    BestscoreData* bestColData = (BestscoreData*)(bestRowData + nwarps);
    short* scoreBorderSubject = (short*)(bestColData + nwarps);
    short* scoreBorderQuery = (short*)(scoreBorderSubject + buffers.max_sequence_length);
    short* scoreBorderDiag = (short*)(scoreBorderQuery + buffers.max_sequence_length);
    char* subject_shared = (char*)(scoreBorderDiag + 2 * (SDIV(buffers.max_sequence_length, 32)+1));
    char* query_shared = (char*)(subject_shared + buffers.max_sequence_bytes);

#endif

    const char* const subject = buffers.subjectdata;
    const int subjectbases = buffers.subjectlength;

    const int subjecttiles = SDIV(subjectbases, WARPSIZE);

    for(int threadid = block.thread_rank(); threadid < MAX_SEQUENCE_LENGTH / 4; threadid += blockDim.x){
        subject_shared[threadid] = subject[threadid];
    }

    for(int queryId = blockIdx.x; queryId < buffers.n_queries; queryId += gridDim.x){
        //if(threadIdx.x == 0) printf("queryId %d\n", queryId);
        const char* const query = buffers.queriesdata + queryId * buffers.sequencepitch;
        const int querybases = buffers.querylengths[queryId];
        const int querytiles = SDIV(querybases, WARPSIZE);

        for(int threadid = block.thread_rank(); threadid < MAX_SEQUENCE_LENGTH / 4; threadid += blockDim.x){
            query_shared[threadid] = query[threadid];
        }

        for(int i = block.thread_rank(); i < MAX_SEQUENCE_LENGTH; i += blockDim.x){
            scoreBorderSubject[i] = 0;
            scoreBorderQuery[i] = 0;
        }

        for(int i = block.thread_rank(); i < ((MAX_SEQUENCE_LENGTH / 32) + 1) * ((MAX_SEQUENCE_LENGTH / 32) + 1); i+= blockDim.x){
            scoreBorderDiag[i] = 0;
        }

        block.sync(); //finish shared mem writes

        const int subjecttileId = warpId;
        const int subjecttileBases = subjecttileId < subjecttiles - 1 ? WARPSIZE : subjectbases - subjecttileId * WARPSIZE;
        const int globalsubjectpos = subjecttileId * WARPSIZE + warp.thread_rank();
        const char subjectbase = sgadetail::twobitToChar(sgadetail::encoded_accessor(subject_shared, subjectbases, globalsubjectpos));

        //tiles move from left to right
        for(int tilediagonal = 0; tilediagonal < 2*querytiles-1; tilediagonal++){
            const int querytileId = tilediagonal - warpId;
            const int querytileBases = querytileId < querytiles - 1 ? WARPSIZE : querybases - querytileId * WARPSIZE;
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

                const char querybase = globalquerypos < querybases ? sgadetail::twobitToChar(sgadetail::encoded_accessor(query_shared, querybases, globalquerypos)) : 'F';
                const short borderScoreSubject = scoreBorderSubject[globalsubjectpos];
                const short borderScoreQuery = scoreBorderQuery[globalquerypos];
                const short borderScoreDiag = scoreBorderDiag[subjecttileId * (querytiles+1) + querytileId];

                short scoreLeft = 0;
                short scoreLeftLeft = 0;
                short scoreDiag = 0;
                short scoreUp = 0;
                short scoreCur = 0;
                prevunion myprev{0};

                //diagonal moves from left to right
                for(int threaddiagonal = 0; threaddiagonal < 2 * WARPSIZE - 1; threaddiagonal++){
                    int localquerypos = threaddiagonal - warp.thread_rank();
                    int localsubjectpos = warp.thread_rank();
                    globalquerypos = querytileId * WARPSIZE + localquerypos;

                    //fetch scores of cells left, above, diag, which are in registers of other threads in the warp
                    scoreLeft = scoreCur;
                    scoreDiag = warp.shfl_up(scoreLeftLeft, 1);
                    scoreUp = warp.shfl_up(scoreCur, 1);
                    short scoreUpTmp = warp.shfl(borderScoreQuery, localquerypos);
                    short scoreDiagTmp = warp.shfl(borderScoreQuery, localquerypos-1);

                    //fetch next base
                    char myquerybase = warp.shfl(querybase, localquerypos);

                    if(0 <= localquerypos && localquerypos < WARPSIZE){
                        assert(threaddiagonal == localquerypos + localsubjectpos);

                        //first row and col need to fetch from border, which is also cached in registers
                        if(localquerypos == 0){
                            scoreLeft = borderScoreSubject;
                        }
                        if(localsubjectpos == 0){
                            scoreUp = scoreUpTmp;
                            if(localquerypos == 0){
                                scoreDiag = borderScoreDiag;
                            }else{
                                scoreDiag = scoreDiagTmp;
                            }
                        }

                        const bool ismatch = subjectbase == myquerybase;

                        const short matchscore = scoreDiag
                                    + (ismatch ? buffers.alignmentscore_match : buffers.alignmentscore_sub);
                        const short insscore = scoreLeft + buffers.alignmentscore_ins;
                        const short delscore = scoreUp + buffers.alignmentscore_del;
                        //AlignType type;
                        if (matchscore < delscore) {
                            scoreCur = delscore;
                            myprev.data = (myprev.data << 2) | ALIGNTYPE_DELETE;
                            //type = ALIGNTYPE_DELETE;
                        }else{
                            scoreCur = matchscore;
                            myprev.data <<= 2;
                            myprev.data |= ismatch ? ALIGNTYPE_MATCH : ALIGNTYPE_SUBSTITUTE;
                            //type = ismatch ? ALIGNTYPE_MATCH : ALIGNTYPE_SUBSTITUTE;
                        }
                        if (scoreCur < insscore) {
                            scoreCur = insscore;
                            myprev.data = (myprev.data << 2) | ALIGNTYPE_INSERT;
                            //type = ALIGNTYPE_INSERT;
                        }

                        if(threaddiagonal >= validBorderDiagonalndexBegin
                        && threaddiagonal < validBorderDiagonalndexEndexcl
                        && localquerypos < querytileBases
                        && localsubjectpos < subjecttileBases){
                            coalesced_group activethreads = coalesced_threads();
                            // on a valid diagonal, the active thread with smallest id (the thread in the top right) writes the subject border
                            // the active thread with the greatest id (the thread in the bottom left) writes the query border
                            if(activethreads.thread_rank() == 0){
                                    scoreBorderSubject[globalsubjectpos] = scoreCur;
                            }
                            if(activethreads.thread_rank() == activethreads.size()-1){
                                    scoreBorderQuery[globalquerypos] = scoreCur;
                            }
                        }
                    }

                    scoreLeftLeft = scoreLeft;

                    warp.sync(); //diagonal finished
                }

                //write prev ops
                //prevs[globalsubjectpos * (MAX_SEQUENCE_LENGTH / 32) + querytileId] = myprev;

                //store transposed to reduce smem bank conflicts
                //prevs[querytileId * MAX_SEQUENCE_LENGTH + globalsubjectpos] = myprev;

                //store transposed and split into two 32 bit values to reduce smem bank conflicts
                ((int*)prevs)[querytileId * MAX_SEQUENCE_LENGTH + globalsubjectpos] = myprev.hi;
                ((int*)prevs + querytiles * MAX_SEQUENCE_LENGTH)[querytileId * MAX_SEQUENCE_LENGTH + globalsubjectpos] = myprev.lo;



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
            //unsigned long long previousPrev = 0;
            prevunion previousPrev{0};
            char previousType = 0;
            while(currow != -1 && curcol != -1){
                unsigned int colIntIndex = curcol / 32;
                unsigned int col2Bitindex = curcol % 32;
                //previousPrev = prevs[currow * (MAX_SEQUENCE_LENGTH / 32) + colIntIndex];
                //previousPrev = prevs[colIntIndex * MAX_SEQUENCE_LENGTH + currow];
                previousPrev.hi = ((int*)prevs)[colIntIndex * MAX_SEQUENCE_LENGTH + currow];
                previousPrev.lo = ((int*)prevs + querytiles * MAX_SEQUENCE_LENGTH)[colIntIndex * MAX_SEQUENCE_LENGTH + currow];
                if(onceflag){
                    //printf("queryId %d, subjectbases %d querybases %d, currow %d, curcol %d , score %d, prevs %lu %d\n", queryId, subjectbases, querybases,  currow, curcol, result.score, prevs[currow * prevcolumncount + colIntIndex], (prevs[currow * prevcolumncount + colIntIndex] >> 2*(31-col2Bitindex)) & 0x3);
                    onceflag = false;
                }
                switch((previousPrev.data >> 2*(31-col2Bitindex)) & 0x3){

                case ALIGNTYPE_MATCH:
                    curcol -= 1;
                    currow -= 1;
                    previousType = ALIGNTYPE_MATCH;
                    break;
                case ALIGNTYPE_SUBSTITUTE:

                    currentOp.position = currow;
                    currentOp.type = ALIGNTYPE_SUBSTITUTE;
                    currentOp.base = sgadetail::twobitToChar(sgadetail::encoded_accessor(query_shared, querybases, curcol));

                    my_ops_out[nOps] = currentOp;
                    ++nOps;

                    curcol -= 1;
                    currow -= 1;
                    previousType = ALIGNTYPE_SUBSTITUTE;
                    break;
                case ALIGNTYPE_DELETE:

                    currentOp.position = currow;
                    currentOp.type = ALIGNTYPE_DELETE;
                    currentOp.base = sgadetail::twobitToChar(sgadetail::encoded_accessor(subject_shared, subjectbases, currow));

                    my_ops_out[nOps] = currentOp;
                    ++nOps;

                    curcol -= 0;
                    currow -= 1;
                    previousType = ALIGNTYPE_DELETE;
                    break;
                case ALIGNTYPE_INSERT:

                    currentOp.position = currow+1;
                    currentOp.type = ALIGNTYPE_INSERT;
                    currentOp.base = sgadetail::twobitToChar(sgadetail::encoded_accessor(query_shared, querybases, curcol));

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

        dim3 block(buffers.max_sequence_length, 1, 1);
        dim3 grid(buffers.n_queries, 1, 1);

        switch(buffers.max_sequence_length){
        case 32: cuda_semi_global_alignment_warps_kernel<32><<<grid, block, 0, stream>>>(buffers); break;
        case 64: cuda_semi_global_alignment_warps_kernel<64><<<grid, block, 0, stream>>>(buffers); break;
        case 96: cuda_semi_global_alignment_warps_kernel<96><<<grid, block, 0, stream>>>(buffers); break;
        case 128: cuda_semi_global_alignment_warps_kernel<128><<<grid, block, 0, stream>>>(buffers); break;
        case 160: cuda_semi_global_alignment_warps_kernel<160><<<grid, block, 0, stream>>>(buffers); break;
        case 192: cuda_semi_global_alignment_warps_kernel<192><<<grid, block, 0, stream>>>(buffers); break;
        case 224: cuda_semi_global_alignment_warps_kernel<224><<<grid, block, 0, stream>>>(buffers); break;
        case 256: cuda_semi_global_alignment_warps_kernel<256><<<grid, block, 0, stream>>>(buffers); break;
        case 288: cuda_semi_global_alignment_warps_kernel<288><<<grid, block, 0, stream>>>(buffers); break;
        case 320: cuda_semi_global_alignment_warps_kernel<320><<<grid, block, 0, stream>>>(buffers); break;
        default: assert(false); break;
        }

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


#endif


#endif

}

#endif
