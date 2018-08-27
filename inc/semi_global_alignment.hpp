#ifndef SEMI_GLOBAL_ALIGNMENT_HPP
#define SEMI_GLOBAL_ALIGNMENT_HPP

#include "hpc_helpers.cuh"
#include "bestalignment.hpp"

#include <algorithm>
#include <limits>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>

namespace sga{

struct AlignOp {

    enum class Type : char{
    	match, sub, ins, del
    };

	short position;
	Type type;
	char base;

	HOSTDEVICEQUALIFIER
	AlignOp(){}

	HOSTDEVICEQUALIFIER
	AlignOp(short p, Type t, char b)
		: position(p), type(t), base(b){}

    HOSTDEVICEQUALIFIER
	AlignOp(const AlignOp& other){
        *this = other;
    }

    HOSTDEVICEQUALIFIER
	AlignOp(AlignOp&& other){
        *this = std::move(other);
    }

    HOSTDEVICEQUALIFIER
	AlignOp& operator=(const AlignOp& other){
        position = other.position;
        type = other.type;
        base = other.base;
        return *this;
    }

    HOSTDEVICEQUALIFIER
	AlignOp& operator=(AlignOp&& other){
        position = other.position;
        type = other.type;
        base = other.base;
        return *this;
    }

	HOSTDEVICEQUALIFIER
	bool operator==(const AlignOp& other) const{
		return (position == other.position && type == other.type && base == other.base);
	}

	HOSTDEVICEQUALIFIER
	bool operator!=(const AlignOp& other) const{
		return !(*this == other);
	}
};

struct AlignmentAttributes{
	int score;
	int subject_begin_incl;
	int query_begin_incl;
	int overlap;
	int shift;
	int nOps; //edit distance / number of operations
	bool isNormalized;
	bool isValid;

    HOSTDEVICEQUALIFIER
    bool operator==(const AlignmentAttributes& rhs) const;
    HOSTDEVICEQUALIFIER
    bool operator!=(const AlignmentAttributes& rhs) const;
    HOSTDEVICEQUALIFIER
    int get_score() const;
    HOSTDEVICEQUALIFIER
    int get_subject_begin_incl() const;
    HOSTDEVICEQUALIFIER
    int get_query_begin_incl() const;
    HOSTDEVICEQUALIFIER
    int get_overlap() const;
    HOSTDEVICEQUALIFIER
    int get_shift() const;
    HOSTDEVICEQUALIFIER
    int get_nOps() const;
    HOSTDEVICEQUALIFIER
    bool get_isNormalized() const;
    HOSTDEVICEQUALIFIER
    bool get_isValid() const;
    HOSTDEVICEQUALIFIER
    int& get_score();
    HOSTDEVICEQUALIFIER
    int& get_subject_begin_incl();
    HOSTDEVICEQUALIFIER
    int& get_query_begin_incl();
    HOSTDEVICEQUALIFIER
    int& get_overlap();
    HOSTDEVICEQUALIFIER
    int& get_shift();
    HOSTDEVICEQUALIFIER
    int& get_nOps();
    HOSTDEVICEQUALIFIER
    bool& get_isNormalized();
    HOSTDEVICEQUALIFIER
    bool& get_isValid();
};

struct AlignmentResult{
    using Op_t = AlignOp;

    AlignmentAttributes attributes;
	std::vector<Op_t> operations;

    bool operator==(const AlignmentResult& rhs) const;
    bool operator!=(const AlignmentResult& rhs) const;

    HOSTDEVICEQUALIFIER
    int get_score() const;
    HOSTDEVICEQUALIFIER
    int get_subject_begin_incl() const;
    HOSTDEVICEQUALIFIER
    int get_query_begin_incl() const;
    HOSTDEVICEQUALIFIER
    int get_overlap() const;
    HOSTDEVICEQUALIFIER
    int get_shift() const;
    HOSTDEVICEQUALIFIER
    int get_nOps() const;
    HOSTDEVICEQUALIFIER
    bool get_isNormalized() const;
    HOSTDEVICEQUALIFIER
    bool get_isValid() const;
    HOSTDEVICEQUALIFIER
    int& get_score();
    HOSTDEVICEQUALIFIER
    int& get_subject_begin_incl();
    HOSTDEVICEQUALIFIER
    int& get_query_begin_incl();
    HOSTDEVICEQUALIFIER
    int& get_overlap();
    HOSTDEVICEQUALIFIER
    int& get_shift();
    HOSTDEVICEQUALIFIER
    int& get_nOps();
    HOSTDEVICEQUALIFIER
    bool& get_isNormalized();
    HOSTDEVICEQUALIFIER
    bool& get_isValid();
};

using Result_t = AlignmentResult;
using Op_t = AlignOp;
using Attributes_t = AlignmentAttributes;


struct SGAdata{
    void* hostptr;
    void* deviceptr;
    std::size_t allocatedMem;

    std::size_t transfersizeH2D = 0;
    std::size_t transfersizeD2H = 0;

    Op_t* d_ops = nullptr;
    Op_t* h_ops = nullptr;

    Attributes_t* d_results = nullptr;
    char* d_subjectsdata = nullptr;
    char* d_queriesdata = nullptr;
    int* d_subjectlengths = nullptr;
    int* d_querylengths = nullptr;
    int* d_NqueriesPrefixSum = nullptr;
    BestAlignment_t* d_bestAlignmentFlags = nullptr;

    Attributes_t* h_results = nullptr;
    char* h_subjectsdata = nullptr;
    char* h_queriesdata = nullptr;
    int* h_subjectlengths = nullptr;
    int* h_querylengths = nullptr;
    int* h_NqueriesPrefixSum = nullptr;
    BestAlignment_t* h_bestAlignmentFlags = nullptr;

#ifdef __NVCC__
    static constexpr int n_streams = 1;
    cudaStream_t streams[n_streams];
#endif

    int deviceId;
    size_t sequencepitch = 0;
    int max_sequence_length = 0;
    int max_sequence_bytes = 0;
    int max_ops_per_alignment = 0;
    int n_subjects = 0;
    int n_queries = 0;
    int n_results = 0;
    int max_n_subjects = 0;
    int max_n_queries = 0;
    int max_n_results = 0;

    // if number of alignments to calculate is >= gpuThreshold, use GPU.
    int gpuThreshold = 0;

    void resize(int n_sub, int n_quer);
    void resize(int n_sub, int n_quer, int n_results, double factor = 1.2);
};


void cuda_init_SGAdata(SGAdata& data,
                       int deviceId,
                       int max_sequence_length,
                       int max_sequence_bytes,
                       int gpuThreshold);

void cuda_cleanup_SGAdata(SGAdata& data);

/*
        CPU alignment
*/

template<class Accessor>
Result_t
cpu_semi_global_alignment(const char* subject,
                            int subjectbases,
                            const char* query,
                            int querybases,
                            int score_match,
                            int score_sub,
                            int score_ins,
                            int score_del,
                            Accessor getChar){

    using Score_t = std::int64_t;

    const int numrows = subjectbases + 1;
    const int numcols = querybases + 1;

    std::vector<Score_t> scores(numrows * numcols);
    std::vector<char> prevs(numrows * numcols);

    // init
    for (int col = 0; col < numcols; ++col) {
        scores[(0) * numcols + (col)] = Score_t(0);
    }

    // row 0 was filled by column loop
    for (int row = 1; row < numrows; ++row) {
        scores[(row) * numcols + (0)] = Score_t(0);
    }

    // fill matrix
    for (int row = 1; row < numrows; ++row) {
        for (int col = 1; col < numcols; ++col) {
            // calc entry [row][col]

            const bool ismatch = getChar(subject, subjectbases, row - 1) == getChar(query, querybases, col - 1);
            const Score_t matchscore = scores[(row - 1) * numcols + (col - 1)]
                        + (ismatch ? Score_t(score_match) : Score_t(score_sub));
            const Score_t insscore = scores[(row) * numcols + (col - 1)] + Score_t(score_ins);
            const Score_t delscore = scores[(row - 1) * numcols + (col)] + Score_t(score_del);

            int maximum = 0;
            if (matchscore < delscore) {
                maximum = delscore;
                prevs[(row) * numcols + (col)] = char(AlignOp::Type::del);
            }else{
                maximum = matchscore;
                prevs[(row) * numcols + (col)] = ismatch ? char(AlignOp::Type::match) : char(AlignOp::Type::sub);
            }
            if (maximum < insscore) {
                maximum = insscore;
                prevs[(row) * numcols + (col)] = char(AlignOp::Type::ins);
            }

            scores[(row) * numcols + (col)] = maximum;
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
    Score_t maximum = std::numeric_limits<Score_t>::min();

    for (int row = 1; row < numrows; ++row) {
        if(scores[(row) * numcols + (querybases)] > maximum){
            //short oldmax = maximum;
            maximum = scores[(row) * numcols + (querybases)];
            currow = row;
            curcol = querybases;

            /*std::cout << "row = " << row << ": \n"
                << scores[row][querybases] << " > " << oldmax << "\n"
                << " update currow " << currow << ", curcol " << curcol << std::endl;*/
        }
    }

    for (int col = 1; col < numcols; ++col) {
        if(scores[(subjectbases) * numcols + (col)] > maximum){
            //short oldmax = maximum;
            maximum = scores[(subjectbases) * numcols + (col)];
            currow = subjectbases;
            curcol = col;

            /*std::cout << "col = " << col << ": \n"
                << scores[subjectbases][col] << " > " << oldmax << "\n"
                << " update currow " << currow << ", curcol " << curcol << std::endl;*/
        }
    }
    //std::cout << "currow " << currow << ", curcol " << curcol << std::endl;

    auto base_to_char = [](char b){
        constexpr char BASE_A = 0x00;
        constexpr char BASE_C = 0x01;
        constexpr char BASE_G = 0x02;
        constexpr char BASE_T = 0x03;

        switch(b){
            case BASE_A: return 'A';
            case BASE_C: return 'C';
            case BASE_G: return 'G';
            case BASE_T: return 'T';
            case 'A':
            case 'C':
            case 'G':
            case 'T': return b;
            default: return 'F';
        }
    };

    Result_t alignresult;
    alignresult.operations.reserve(currow + curcol);

    const int subject_end_excl = currow;

    alignresult.get_score() = maximum;
    alignresult.get_isNormalized() = false;

    while(currow != 0 && curcol != 0){
        switch (AlignOp::Type(prevs[(currow) * numcols + (curcol)])) {
        case AlignOp::Type::match: //printf("m\n");
            curcol -= 1;
            currow -= 1;
            break;
        case AlignOp::Type::sub: //printf("s\n");

            alignresult.operations.emplace_back((short)(currow-1), AlignOp::Type::sub, base_to_char(getChar(query, querybases, curcol - 1)));

            curcol -= 1;
            currow -= 1;
            break;
        case AlignOp::Type::del:  //printf("d\n");

            alignresult.operations.emplace_back((short)(currow-1), AlignOp::Type::del, base_to_char(getChar(subject, subjectbases, currow - 1)));

            curcol -= 0;
            currow -= 1;
            break;
        case AlignOp::Type::ins:  //printf("i\n");

            alignresult.operations.emplace_back((short)currow, AlignOp::Type::ins, base_to_char(getChar(query, querybases, curcol - 1)));

            curcol -= 1;
            currow -= 0;
            break;
        }
    }

    std::reverse(alignresult.operations.begin(), alignresult.operations.end());

    alignresult.get_subject_begin_incl() = currow;
    alignresult.get_query_begin_incl() = curcol;
    alignresult.get_isValid() = true;
    alignresult.get_overlap() = subject_end_excl - alignresult.get_subject_begin_incl();
    alignresult.get_shift() = alignresult.get_subject_begin_incl() == 0
                                ? -alignresult.get_query_begin_incl()
                                : alignresult.get_subject_begin_incl();
    alignresult.get_nOps() = alignresult.operations.size();
    alignresult.get_isNormalized() = false;

    return alignresult;
}






template<class Accessor>
Result_t
cpu_semi_global_alignment_new(const char* subject,
                            int subjectbases,
                            const char* query,
                            int querybases,
                            int min_overlap,
                            double maxErrorRate,
                            double min_overlap_ratio,
                            int score_match,
                            int score_sub,
                            int score_ins,
                            int score_del,
                            Accessor getChar){

    using Score_t = std::int64_t;

    const int minoverlap = std::max(min_overlap, int(double(subjectbases) * min_overlap_ratio));

    const int numrows = subjectbases + 1;
    const int numcols = querybases + 1;

    std::vector<Score_t> scores(numrows * numcols);
    std::vector<char> prevs(numrows * numcols);

    // init
    for (int col = 0; col < numcols; ++col) {
        scores[(0) * numcols + (col)] = Score_t(0);
    }

    // row 0 was filled by column loop
    for (int row = 1; row < numrows; ++row) {
        scores[(row) * numcols + (0)] = Score_t(0);
    }

    // fill matrix
    for (int row = 1; row < numrows; ++row) {
        for (int col = 1; col < numcols; ++col) {
            // calc entry [row][col]

            const bool ismatch = getChar(subject, subjectbases, row - 1) == getChar(query, querybases, col - 1);
            const Score_t matchscore = scores[(row - 1) * numcols + (col - 1)]
                        + (ismatch ? Score_t(score_match) : Score_t(score_sub));
            const Score_t insscore = scores[(row) * numcols + (col - 1)] + Score_t(score_ins);
            const Score_t delscore = scores[(row - 1) * numcols + (col)] + Score_t(score_del);

            int maximum = 0;
            if (matchscore < delscore) {
                maximum = delscore;
                prevs[(row) * numcols + (col)] = char(AlignOp::Type::del);
            }else{
                maximum = matchscore;
                prevs[(row) * numcols + (col)] = ismatch ? char(AlignOp::Type::match) : char(AlignOp::Type::sub);
            }
            if (maximum < insscore) {
                maximum = insscore;
                prevs[(row) * numcols + (col)] = char(AlignOp::Type::ins);
            }

            scores[(row) * numcols + (col)] = maximum;
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
    Score_t maximum = std::numeric_limits<Score_t>::min();

    for (int row = 1; row < numrows; ++row) {
        if(scores[(row) * numcols + (querybases)] > maximum){
            //short oldmax = maximum;
            maximum = scores[(row) * numcols + (querybases)];
            currow = row;
            curcol = querybases;

            /*std::cout << "row = " << row << ": \n"
                << scores[row][querybases] << " > " << oldmax << "\n"
                << " update currow " << currow << ", curcol " << curcol << std::endl;*/
        }
    }

    for (int col = 1; col < numcols; ++col) {
        if(scores[(subjectbases) * numcols + (col)] > maximum){
            //short oldmax = maximum;
            maximum = scores[(subjectbases) * numcols + (col)];
            currow = subjectbases;
            curcol = col;

            /*std::cout << "col = " << col << ": \n"
                << scores[subjectbases][col] << " > " << oldmax << "\n"
                << " update currow " << currow << ", curcol " << curcol << std::endl;*/
        }
    }
    //std::cout << "currow " << currow << ", curcol " << curcol << std::endl;

    auto base_to_char = [](char b){
        constexpr char BASE_A = 0x00;
        constexpr char BASE_C = 0x01;
        constexpr char BASE_G = 0x02;
        constexpr char BASE_T = 0x03;

        switch(b){
            case BASE_A: return 'A';
            case BASE_C: return 'C';
            case BASE_G: return 'G';
            case BASE_T: return 'T';
            case 'A':
            case 'C':
            case 'G':
            case 'T': return b;
            default: return 'F';
        }
    };

    Result_t alignresult;
    alignresult.operations.reserve(currow + curcol);

    const int subject_end_excl = currow;

    alignresult.get_score() = maximum;
    alignresult.get_isNormalized() = false;

    while(currow != 0 && curcol != 0){
        switch (AlignOp::Type(prevs[(currow) * numcols + (curcol)])) {
        case AlignOp::Type::match: //printf("m\n");
            curcol -= 1;
            currow -= 1;
            break;
        case AlignOp::Type::sub: //printf("s\n");

            alignresult.operations.emplace_back((short)(currow-1), AlignOp::Type::sub, base_to_char(getChar(query, querybases, curcol - 1)));

            curcol -= 1;
            currow -= 1;
            break;
        case AlignOp::Type::del:  //printf("d\n");

            alignresult.operations.emplace_back((short)(currow-1), AlignOp::Type::del, base_to_char(getChar(subject, subjectbases, currow - 1)));

            curcol -= 0;
            currow -= 1;
            break;
        case AlignOp::Type::ins:  //printf("i\n");

            alignresult.operations.emplace_back((short)currow, AlignOp::Type::ins, base_to_char(getChar(query, querybases, curcol - 1)));

            curcol -= 1;
            currow -= 0;
            break;
        }
    }

    std::reverse(alignresult.operations.begin(), alignresult.operations.end());

    alignresult.get_subject_begin_incl() = currow;
    alignresult.get_query_begin_incl() = curcol;
    alignresult.get_isValid() = true;
    alignresult.get_overlap() = subject_end_excl - alignresult.get_subject_begin_incl();
    alignresult.get_shift() = alignresult.get_subject_begin_incl() == 0
                                ? -alignresult.get_query_begin_incl()
                                : alignresult.get_subject_begin_incl();
    alignresult.get_nOps() = alignresult.operations.size();
    alignresult.get_isNormalized() = false;

    return alignresult;
}






/*
    GPU alignment
*/

#ifdef __NVCC__

template<int MAX_SEQUENCE_LENGTH, class Accessor>
__global__
void cuda_semi_global_alignment_kernel(AlignmentAttributes* results,
                                       AlignOp* ops,
                                       const int max_ops_per_alignment,
                                       const char* subjectsdata,
                                       const int* subjectlengths,
                                       const char* queriesdata,
                                       const int* querylengths,
                                       const int* NqueriesPrefixSum,
                                       const int Nsubjects,
                                       const size_t sequencepitch,
                                       const int score_match,
                                       const int score_sub,
                                       const int score_ins,
                                       const int score_del,
                                       Accessor getChar){

    static_assert(MAX_SEQUENCE_LENGTH % 32 == 0, "MAX_SEQUENCE_LENGTH must be divisible by 32");

    using Score_t = short;

    constexpr int MAX_SEQUENCE_BYTES = MAX_SEQUENCE_LENGTH; //uses some more smem, but simpler code.

    constexpr int prevsPerInt = (sizeof(int)*8/2);

    __shared__ char subject_shared[MAX_SEQUENCE_BYTES];
    __shared__ char query_shared[MAX_SEQUENCE_BYTES];
    __shared__ int prevs[MAX_SEQUENCE_LENGTH*(MAX_SEQUENCE_LENGTH / prevsPerInt)];
    __shared__ Score_t scores[3 * MAX_SEQUENCE_LENGTH];
    __shared__ Score_t bestrow;
    __shared__ Score_t bestcol;
    __shared__ Score_t bestrowscore;
    __shared__ Score_t bestcolscore;

    for(unsigned queryIndex = blockIdx.x; queryIndex < NqueriesPrefixSum[Nsubjects]; queryIndex += gridDim.x){

        //find subjectindex
        int subjectIndex = 0;
        for(; subjectIndex < Nsubjects; subjectIndex++){
            if(queryIndex < NqueriesPrefixSum[subjectIndex+1])
                break;
        }

        //save subject in shared memory
        const int subjectbases = subjectlengths[subjectIndex];
        for(int threadid = threadIdx.x; threadid < MAX_SEQUENCE_BYTES; threadid += blockDim.x){
            subject_shared[threadid] = subjectsdata[subjectIndex * sequencepitch + threadid];
        }

        //save query in shared memory
        const int querybases = querylengths[queryIndex];
        for(int threadid = threadIdx.x; threadid < MAX_SEQUENCE_BYTES; threadid += blockDim.x){
            query_shared[threadid] = queriesdata[queryIndex * sequencepitch + threadid];
        }

        for (int l = threadIdx.x; l < MAX_SEQUENCE_LENGTH; l += blockDim.x) {
            scores[0*MAX_SEQUENCE_LENGTH+l] = Score_t(0);
            scores[1*MAX_SEQUENCE_LENGTH+l] = Score_t(0);
            scores[2*MAX_SEQUENCE_LENGTH+l] = Score_t(0);
        }

        for(int i = 0; i < MAX_SEQUENCE_LENGTH + 1; i++){
            for (int j = threadIdx.x; j < MAX_SEQUENCE_LENGTH / prevsPerInt; j += blockDim.x) {
                prevs[i * MAX_SEQUENCE_LENGTH / prevsPerInt + j] = 0;
            }
        }

        if (threadIdx.x == 0) {
            bestrow = Score_t(0);
            bestcol = Score_t(0);
            bestrowscore = std::numeric_limits<Score_t>::min();
            bestcolscore = std::numeric_limits<Score_t>::min();
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

            const Score_t scoreDiag = globalsubjectpos == 0 ? 0 : scores[matchrow * MAX_SEQUENCE_LENGTH + threadIdx.x - 1];
            const Score_t scoreLeft = scores[indelrow * MAX_SEQUENCE_LENGTH + threadIdx.x];
            const Score_t scoreUp = globalsubjectpos == 0 ? 0 :  scores[indelrow * MAX_SEQUENCE_LENGTH + threadIdx.x - 1];

            if(globalsubjectpos >= 0 && globalsubjectpos < MAX_SEQUENCE_LENGTH
                && globalquerypos >= 0 && globalquerypos < MAX_SEQUENCE_LENGTH){

                const bool ismatch = subjectbase == querybase;
                const Score_t matchscore = scoreDiag
                            + (ismatch ? Score_t(score_match) : Score_t(score_sub));
                const Score_t insscore = scoreUp + Score_t(score_ins);
                const Score_t delscore = scoreLeft + Score_t(score_del);

                Score_t maximum = 0;
                const unsigned int colindex = globalquerypos / prevsPerInt;

                if (matchscore < delscore) {
                    maximum = delscore;

                    int t = int(AlignOp::Type::del);
                    t <<= 2*(prevsPerInt-1- (globalquerypos % prevsPerInt));
                    myprev |= t;
                }else{
                    maximum = matchscore;

                    int t = int(ismatch ? AlignOp::Type::match : AlignOp::Type::sub);
                    t <<= 2*(prevsPerInt-1- (globalquerypos % prevsPerInt));
                    myprev |= t;
                }
                if (maximum < insscore) {
                    maximum = insscore;

                    int t = int(AlignOp::Type::ins);
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

            Attributes_t result;

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

            Attributes_t * const my_result_out = results + queryIndex;
            AlignOp * const my_ops_out = ops + max_ops_per_alignment * queryIndex;

            int nOps = 0;
            bool isValid = true;
            AlignOp currentOp;
            AlignOp::Type previousType;
            while(currow != -1 && curcol != -1){
                const unsigned int colIntIndex = curcol / prevsPerInt;
                const unsigned int col2Bitindex = curcol % prevsPerInt;

                switch(AlignOp::Type((prevs[currow * (MAX_SEQUENCE_LENGTH / prevsPerInt) + colIntIndex] >> 2*(prevsPerInt-1-col2Bitindex)) & 0x3)){

                case AlignOp::Type::match:
                    curcol -= 1;
                    currow -= 1;
                    previousType = AlignOp::Type::match;

                    break;
                case AlignOp::Type::sub:
                    currentOp.position = currow;
                    currentOp.type = AlignOp::Type::sub;
                    currentOp.base = getChar(query_shared, querybases, curcol);

                    my_ops_out[nOps] = currentOp;
                    ++nOps;

                    curcol -= 1;
                    currow -= 1;
                    previousType = AlignOp::Type::sub;

                    break;
                case AlignOp::Type::del:
                    currentOp.position = currow;
                    currentOp.type = AlignOp::Type::del;
                    currentOp.base = getChar(subject_shared, subjectbases, currow);

                    my_ops_out[nOps] = currentOp;
                    ++nOps;

                    curcol -= 0;
                    currow -= 1;
                    previousType = AlignOp::Type::del;

                    break;
                case AlignOp::Type::ins:
                    currentOp.position = currow+1;
                    currentOp.type = AlignOp::Type::ins;
                    currentOp.base = getChar(query_shared, querybases, curcol);

                    my_ops_out[nOps] = currentOp;
                    ++nOps;

                    curcol -= 1;
                    currow -= 0;
                    previousType = AlignOp::Type::ins;

                    break;
                default : // code should not reach here
                    isValid = false;
                    printf("alignment backtrack error");
                }
            }
            switch(previousType){
            case AlignOp::Type::match:
                curcol += 1;
                currow += 1;
                break;
            case AlignOp::Type::sub:
                curcol += 1;
                currow += 1;
                break;
            case AlignOp::Type::del:
                curcol += 0;
                currow += 1;
                break;
            case AlignOp::Type::ins:
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

template<class Accessor>
void call_semi_global_alignment_kernel_async(const SGAdata& sgadata,
                                            const int score_match,
                                            const int score_sub,
                                            const int score_ins,
                                            const int score_del,
                                            const int maxSubjectLength,
                                            const int maxQueryLength,
                                            Accessor accessor){

    const int maxSequenceLength = SDIV(std::max(maxSubjectLength, maxQueryLength), 32) * 32;
    dim3 block(maxSequenceLength, 1, 1);
    dim3 grid(sgadata.n_queries, 1, 1);

    #define mycall(length) cuda_semi_global_alignment_kernel<(length)> \
                            <<<grid, block, 0, sgadata.streams[0]>>>(sgadata.d_results, \
                            sgadata.d_ops, \
                            sgadata.max_ops_per_alignment, \
                            sgadata.d_subjectsdata, \
                            sgadata.d_subjectlengths, \
                            sgadata.d_queriesdata, \
                            sgadata.d_querylengths, \
                            sgadata.d_NqueriesPrefixSum, \
                            sgadata.n_subjects, \
                            sgadata.sequencepitch, \
                            score_match, \
                            score_sub, \
                            score_ins, \
                            score_del, \
                            accessor); CUERR;

    // start kernel
    switch(maxSequenceLength){
    case 32: mycall(32); break;
    case 64: mycall(64); break;
    case 96: mycall(96); break;
    case 128: mycall(128); break;
    case 160: mycall(160); break;
    case 192: mycall(192); break;
    case 224: mycall(224); break;
    case 256: mycall(256); break;
    case 288: mycall(288); break;
    case 320: mycall(320); break;
    default: throw std::runtime_error("cannot use cuda semi global alignment for sequences longer than 320.");
    }

    #undef mycall
}

template<class Accessor>
void call_semi_global_alignment_kernel(const SGAdata& sgadata,
                                            const int score_match,
                                            const int score_sub,
                                            const int score_ins,
                                            const int score_del,
                                            const int maxSubjectLength,
                                            const int maxQueryLength,
                                            Accessor accessor){

    call_semi_global_alignment_kernel_async(sgadata,
                                            score_match,
                                            score_sub,
                                            score_ins,
                                            score_del,
                                            maxSubjectLength,
                                            maxQueryLength,
                                            accessor);

    cudaStreamSynchronize(sgadata.streams[0]); CUERR;
}




template<int MAX_SEQUENCE_LENGTH, class Accessor, class RevCompl>
__global__
void cuda_semi_global_alignment_with_revcompl_kernel(AlignmentAttributes* results,
                                       AlignOp* ops,
                                       const int max_ops_per_alignment,
                                       const char* subjectsdata,
                                       const int* subjectlengths,
                                       const char* queriesdata,
                                       const int* querylengths,
                                       const int* NqueriesPrefixSum,
                                       const int Nsubjects,
                                       const size_t sequencepitch,
                                       const int score_match,
                                       const int score_sub,
                                       const int score_ins,
                                       const int score_del,
                                       Accessor getChar,
                                       RevCompl make_reverse_complement){

    static_assert(MAX_SEQUENCE_LENGTH % 32 == 0, "MAX_SEQUENCE_LENGTH must be divisible by 32");

    using Score_t = short;

    constexpr int MAX_SEQUENCE_BYTES = MAX_SEQUENCE_LENGTH; //uses some more smem, but simpler code.

    constexpr int prevsPerInt = (sizeof(int)*8/2);

    __shared__ char subject_shared[MAX_SEQUENCE_BYTES];
    __shared__ char query_shared[MAX_SEQUENCE_BYTES];
    __shared__ char query_revcompl_shared[MAX_SEQUENCE_BYTES];
    __shared__ int prevs[MAX_SEQUENCE_LENGTH*(MAX_SEQUENCE_LENGTH / prevsPerInt)];
    __shared__ Score_t scores[3 * MAX_SEQUENCE_LENGTH];
    __shared__ Score_t bestrow;
    __shared__ Score_t bestcol;
    __shared__ Score_t bestrowscore;
    __shared__ Score_t bestcolscore;

    const int nQueries = NqueriesPrefixSum[Nsubjects];

    for(unsigned resultIndex = blockIdx.x; resultIndex < nQueries * 2; resultIndex += gridDim.x){

        const int queryIndex = resultIndex < nQueries ? resultIndex : resultIndex - nQueries;

        //find subjectindex
        int subjectIndex = 0;
        for(; subjectIndex < Nsubjects; subjectIndex++){
            if(queryIndex < NqueriesPrefixSum[subjectIndex+1])
                break;
        }

        //save subject in shared memory
        const int subjectbases = subjectlengths[subjectIndex];
        for(int threadid = threadIdx.x; threadid < MAX_SEQUENCE_BYTES; threadid += blockDim.x){
            subject_shared[threadid] = subjectsdata[subjectIndex * sequencepitch + threadid];
        }

        //save query in shared memory
        const int querybases = querylengths[queryIndex];
        for(int threadid = threadIdx.x; threadid < MAX_SEQUENCE_BYTES; threadid += blockDim.x){
            query_shared[threadid] = queriesdata[queryIndex * sequencepitch + threadid];
        }

        for (int l = threadIdx.x; l < MAX_SEQUENCE_LENGTH; l += blockDim.x) {
            scores[0*MAX_SEQUENCE_LENGTH+l] = Score_t(0);
            scores[1*MAX_SEQUENCE_LENGTH+l] = Score_t(0);
            scores[2*MAX_SEQUENCE_LENGTH+l] = Score_t(0);
        }

        for(int i = 0; i < MAX_SEQUENCE_LENGTH + 1; i++){
            for (int j = threadIdx.x; j < MAX_SEQUENCE_LENGTH / prevsPerInt; j += blockDim.x) {
                prevs[i * MAX_SEQUENCE_LENGTH / prevsPerInt + j] = 0;
            }
        }

        if (threadIdx.x == 0) {
            bestrow = Score_t(0);
            bestcol = Score_t(0);
            bestrowscore = std::numeric_limits<Score_t>::min();
            bestcolscore = std::numeric_limits<Score_t>::min();
        }

        __syncthreads();

        //queryIndex != resultIndex -> reverse complement
        if(queryIndex != resultIndex && threadIdx.x == 0){
            make_reverse_complement((std::uint8_t*)query_revcompl_shared, (const std::uint8_t*)query_shared, querybases);
        }
        __syncthreads();

        const char* query = queryIndex == resultIndex ? query_shared : query_revcompl_shared;

        const int globalsubjectpos = threadIdx.x;
        const char subjectbase = getChar(subject_shared, subjectbases, globalsubjectpos);
        int calculatedCells = 0;
        int myprev = 0;

        for (int threaddiagonal = 0; threaddiagonal < subjectbases + querybases - 1; threaddiagonal++) {

            const int targetrow = threaddiagonal % 3;
            const int indelrow = (targetrow == 0 ? 2 : targetrow - 1);
            const int matchrow = (indelrow == 0 ? 2 : indelrow - 1);

            const int globalquerypos = threaddiagonal - threadIdx.x;
            const char querybase = globalquerypos < querybases ? getChar(query, querybases, globalquerypos) : 'F';

            const Score_t scoreDiag = globalsubjectpos == 0 ? 0 : scores[matchrow * MAX_SEQUENCE_LENGTH + threadIdx.x - 1];
            const Score_t scoreLeft = scores[indelrow * MAX_SEQUENCE_LENGTH + threadIdx.x];
            const Score_t scoreUp = globalsubjectpos == 0 ? 0 :  scores[indelrow * MAX_SEQUENCE_LENGTH + threadIdx.x - 1];

            if(globalsubjectpos >= 0 && globalsubjectpos < MAX_SEQUENCE_LENGTH
                && globalquerypos >= 0 && globalquerypos < MAX_SEQUENCE_LENGTH){

                const bool ismatch = subjectbase == querybase;
                const Score_t matchscore = scoreDiag
                            + (ismatch ? Score_t(score_match) : Score_t(score_sub));
                const Score_t insscore = scoreUp + Score_t(score_ins);
                const Score_t delscore = scoreLeft + Score_t(score_del);

                Score_t maximum = 0;
                const unsigned int colindex = globalquerypos / prevsPerInt;

                if (matchscore < delscore) {
                    maximum = delscore;

                    int t = int(AlignOp::Type::del);
                    t <<= 2*(prevsPerInt-1- (globalquerypos % prevsPerInt));
                    myprev |= t;
                }else{
                    maximum = matchscore;

                    int t = int(ismatch ? AlignOp::Type::match : AlignOp::Type::sub);
                    t <<= 2*(prevsPerInt-1- (globalquerypos % prevsPerInt));
                    myprev |= t;
                }
                if (maximum < insscore) {
                    maximum = insscore;

                    int t = int(AlignOp::Type::ins);
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

            Attributes_t result;

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

            Attributes_t * const my_result_out = results + resultIndex;
            AlignOp * const my_ops_out = ops + max_ops_per_alignment * resultIndex;

            int nOps = 0;
            bool isValid = true;
            AlignOp currentOp;
            AlignOp::Type previousType;

            auto base_to_char = [](char b){
                constexpr char BASE_A = 0x00;
                constexpr char BASE_C = 0x01;
                constexpr char BASE_G = 0x02;
                constexpr char BASE_T = 0x03;

                switch(b){
                    case BASE_A: return 'A';
                    case BASE_C: return 'C';
                    case BASE_G: return 'G';
                    case BASE_T: return 'T';
                    case 'A':
                    case 'C':
                    case 'G':
                    case 'T': return b;
                    default: return 'F';
                }
            };

            while(currow != -1 && curcol != -1){
                const unsigned int colIntIndex = curcol / prevsPerInt;
                const unsigned int col2Bitindex = curcol % prevsPerInt;

                switch(AlignOp::Type((prevs[currow * (MAX_SEQUENCE_LENGTH / prevsPerInt) + colIntIndex] >> 2*(prevsPerInt-1-col2Bitindex)) & 0x3)){

                case AlignOp::Type::match:
                    curcol -= 1;
                    currow -= 1;
                    previousType = AlignOp::Type::match;

                    break;
                case AlignOp::Type::sub:
                    currentOp.position = currow;
                    currentOp.type = AlignOp::Type::sub;
                    currentOp.base = base_to_char(getChar(query, querybases, curcol));

                    my_ops_out[nOps] = currentOp;
                    ++nOps;

                    curcol -= 1;
                    currow -= 1;
                    previousType = AlignOp::Type::sub;

                    break;
                case AlignOp::Type::del:
                    currentOp.position = currow;
                    currentOp.type = AlignOp::Type::del;
                    currentOp.base = base_to_char(getChar(subject_shared, subjectbases, currow));

                    my_ops_out[nOps] = currentOp;
                    ++nOps;

                    curcol -= 0;
                    currow -= 1;
                    previousType = AlignOp::Type::del;

                    break;
                case AlignOp::Type::ins:
                    currentOp.position = currow+1;
                    currentOp.type = AlignOp::Type::ins;
                    currentOp.base = base_to_char(getChar(query, querybases, curcol));

                    my_ops_out[nOps] = currentOp;
                    ++nOps;

                    curcol -= 1;
                    currow -= 0;
                    previousType = AlignOp::Type::ins;

                    break;
                default : // code should not reach here
                    isValid = false;
                    printf("alignment backtrack error");
                }
            }
            switch(previousType){
            case AlignOp::Type::match:
                curcol += 1;
                currow += 1;
                break;
            case AlignOp::Type::sub:
                curcol += 1;
                currow += 1;
                break;
            case AlignOp::Type::del:
                curcol += 0;
                currow += 1;
                break;
            case AlignOp::Type::ins:
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

template<class Accessor, class RevCompl>
void call_semi_global_alignment_with_revcompl_kernel_async(const SGAdata& sgadata,
                                            const int score_match,
                                            const int score_sub,
                                            const int score_ins,
                                            const int score_del,
                                            const int maxSubjectLength,
                                            const int maxQueryLength,
                                            Accessor accessor,
                                            RevCompl make_reverse_complement){

    const int maxSequenceLength = SDIV(std::max(maxSubjectLength, maxQueryLength), 32) * 32;
    dim3 block(maxSequenceLength, 1, 1);
    dim3 grid(sgadata.n_queries * 2, 1, 1);

    #define mycall(length) cuda_semi_global_alignment_with_revcompl_kernel<(length)> \
                            <<<grid, block, 0, sgadata.streams[0]>>>(sgadata.d_results, \
                            sgadata.d_ops, \
                            sgadata.max_ops_per_alignment, \
                            sgadata.d_subjectsdata, \
                            sgadata.d_subjectlengths, \
                            sgadata.d_queriesdata, \
                            sgadata.d_querylengths, \
                            sgadata.d_NqueriesPrefixSum, \
                            sgadata.n_subjects, \
                            sgadata.sequencepitch, \
                            score_match, \
                            score_sub, \
                            score_ins, \
                            score_del, \
                            accessor, \
                            make_reverse_complement); CUERR;

    // start kernel
    switch(maxSequenceLength){
    case 32: mycall(32); break;
    case 64: mycall(64); break;
    case 96: mycall(96); break;
    case 128: mycall(128); break;
    case 160: mycall(160); break;
    case 192: mycall(192); break;
    case 224: mycall(224); break;
    case 256: mycall(256); break;
    case 288: mycall(288); break;
    case 320: mycall(320); break;
    default: throw std::runtime_error("cannot use cuda semi global alignment for sequences longer than 320.");
    }

    #undef mycall
}

template<class Accessor, class RevCompl>
void call_semi_global_alignment_with_revcompl_kernel(const SGAdata& sgadata,
                                            const int score_match,
                                            const int score_sub,
                                            const int score_ins,
                                            const int score_del,
                                            const int maxSubjectLength,
                                            const int maxQueryLength,
                                            Accessor accessor,
                                            RevCompl make_reverse_complement){

    call_semi_global_alignment_with_revcompl_kernel_async(sgadata,
                                            score_match,
                                            score_sub,
                                            score_ins,
                                            score_del,
                                            maxSubjectLength,
                                            maxQueryLength,
                                            accessor,
                                            make_reverse_complement);

    cudaStreamSynchronize(sgadata.streams[0]); CUERR;
}




#endif //ifdef __NVCC__

} //namespace sga

#endif
