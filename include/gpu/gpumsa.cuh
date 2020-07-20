 
#ifndef CARE_GPU_MSA_CUH
#define CARE_GPU_MSA_CUH

#ifdef __NVCC__

#include <config.hpp>
#include <hostdevicefunctions.cuh>

#include <bestalignment.hpp>
#include <gpu/kernels.hpp> // MSAColumnProperties

#include <sequence.hpp>
#include <hpc_helpers.cuh>

#include <cassert>
#include <cstdint>
#include <type_traits>

#include <cub/cub.cuh>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace care{
namespace gpu{
/*
struct MSAColumnProperties{
    //int startindex;
    //int endindex;
    //int columnsToCheck;
    int subjectColumnsBegin_incl;
    int subjectColumnsEnd_excl;
    int firstColumn_incl;
    int lastColumn_excl;
};
*/
    struct GpuSingleMSA{
    public:

        template<class ThreadGroup>
        __device__ __forceinline__
        void checkAfterBuild(ThreadGroup& group, int subjectIndex = -1){
    
            const int firstColumn_incl = columnProperties->firstColumn_incl;
            const int lastColumn_excl = columnProperties->lastColumn_excl;
    
            for(int column = firstColumn_incl + group.thread_rank(); 
                    column < lastColumn_excl; 
                    column += group.size()){

                const int* const mycounts = counts + column;
                const float* const myweights = weights + column;
                float sumOfWeights = 0.0f;
    
                #pragma unroll
                for(int k = 0; k < 4; k++){
                    const int count = mycounts[k * columnPitchInElements];
                    const float weight = myweights[k * columnPitchInElements];

                    if(count > 0 && weight <= 0.0f){
                        printf("msa check failed! subjectIndex %d, column %d, base %d, count %d, weight %f\n",
                            subjectIndex, column, k, count, weight);
                        assert(false);
                    }
    
                    if(count <= 0 && weight > 0.0f){
                        printf("msa check failed! subjectIndex %d, column %d, base %d, count %d, weight %f\n",
                            subjectIndex, column, k, count, weight);
                        assert(false);
                    }
    
                    sumOfWeights += weight;
                }
    
                if(sumOfWeights == 0){
                    printf("s %d c %d\n", subjectIndex, column);
                    assert(sumOfWeights != 0);
                }
            }
        }

        template<
            class ThreadGroup, 
            class GroupReduceIntMin, 
            class GroupReduceIntMax
        >
        __device__ __forceinline__
        void initColumnProperties(
                ThreadGroup& group,
                GroupReduceIntMin& groupReduceIntMin,
                GroupReduceIntMax& groupReduceIntMax,
                const int* __restrict__ goodCandidateIndices,
                int numGoodCandidates,
                const int* __restrict__ shifts,
                const BestAlignment_t* __restrict__ alignmentFlags,
                const int subjectLength,
                const int* __restrict__ candidateLengths
        ){

            int startindex = 0;
            int endindex = subjectLength;

            for(int k = group.thread_rank(); k < numGoodCandidates; k += group.size()) {
                const int localCandidateIndex = goodCandidateIndices[k];

                const int shift = shifts[localCandidateIndex];
                const BestAlignment_t flag = alignmentFlags[localCandidateIndex];
                const int queryLength = candidateLengths[localCandidateIndex];

                assert(flag != BestAlignment_t::None);

                const int queryEndsAt = queryLength + shift;
                startindex = min(startindex, shift);
                endindex = max(endindex, queryEndsAt);
            }

            startindex = groupReduceIntMin(startindex);
            endindex = groupReduceIntMax(endindex);

            group.sync();


            if(group.thread_rank() == 0) {
                MSAColumnProperties my_columnproperties;

                my_columnproperties.subjectColumnsBegin_incl = max(-startindex, 0);
                my_columnproperties.subjectColumnsEnd_excl = my_columnproperties.subjectColumnsBegin_incl + subjectLength;
                my_columnproperties.firstColumn_incl = 0;
                my_columnproperties.lastColumn_excl = endindex - startindex;

                *columnProperties = my_columnproperties;
            }
        }

        template<class ThreadGroup>
        __device__ __forceinline__
        void updateColumnProperties(ThreadGroup& group){

            const int firstColumn_incl = columnProperties->firstColumn_incl;
            const int lastColumn_excl = columnProperties->lastColumn_excl;
            const int numColumnsToCheck = lastColumn_excl - firstColumn_incl;

            int newFirstColumn_incl = -1;
            int newLastColumn_excl = -1;

            for(int i = group.thread_rank(); i < numColumnsToCheck-1; i += group.size()){
                const int column = firstColumn_incl + i;

                const int thisCoverage = coverages[column];
                const int nextCoverage = coverages[column+1];
                assert(thisCoverage >= 0);
                assert(nextCoverage >= 0);

                if(thisCoverage == 0 && nextCoverage > 0){
                    newFirstColumn_incl = column+1;
                }

                if(thisCoverage > 0 && nextCoverage == 0){
                    newLastColumn_excl = column+1;
                }
            }

            group.sync();

            //there can be at most one thread for which this is true
            if(newFirstColumn_incl != -1){
                columnProperties->firstColumn_incl = newFirstColumn_incl;
            }
            //there can be at most one thread for which this is true
            if(newLastColumn_excl != -1){
                columnProperties->lastColumn_excl = newLastColumn_excl;
            }
            
            group.sync();
        }

        template<bool doAdd, class ThreadGroup>
        __device__ 
        void msaAddOrDeleteASequence2Bit(
            ThreadGroup& group,
            const unsigned int* __restrict__ sequence, //not transposed
            int sequenceLength, 
            bool isForward,
            int columnStart,
            float overlapweight,
            const char* __restrict__ quality, //not transposed
            bool canUseQualityScores
        ){

            auto getEncodedNucFromInt2Bit = [](unsigned int data, int pos){
                return ((data >> (30 - 2*pos)) & 0x00000003);
            };

            constexpr int nucleotidesPerInt2Bit = 16;
            const int fullInts = sequenceLength / nucleotidesPerInt2Bit;

            for(int intIndex = group.thread_rank(); intIndex < fullInts; intIndex += group.size()){
                const unsigned int currentDataInt = sequence[intIndex];

                for(int k = 0; k < 4; k++){
                    alignas(4) char currentFourQualities[4];

                    assert(size_t(&currentFourQualities[0]) % 4 == 0);

                    if(canUseQualityScores){
                        *((int*)&currentFourQualities[0]) = ((const int*)quality)[intIndex * 4 + k];
                    }

                    for(int l = 0; l < 4; l++){
                        const int posInInt = k * 4 + l;

                        unsigned int encodedBaseAsInt = getEncodedNucFromInt2Bit(currentDataInt, posInInt);
                        if(!isForward){
                            //reverse complement
                            encodedBaseAsInt = (~encodedBaseAsInt & 0x00000003);
                        }
                        const float weight = canUseQualityScores ? getQualityWeight(currentFourQualities[l]) * overlapweight : overlapweight;

                        assert(weight != 0);
                        const int rowOffset = encodedBaseAsInt * columnPitchInElements;
                        const int columnIndex = columnStart 
                                + (isForward ? (intIndex * 16 + posInInt) : sequenceLength - 1 - (intIndex * 16 + posInInt));
                        
                        atomicAdd(counts + rowOffset + columnIndex, doAdd ? 1 : -1);
                        float n = atomicAdd(weights + rowOffset + columnIndex, doAdd ? weight : -weight);
                        atomicAdd(coverages + columnIndex, doAdd ? 1 : -1);
                    }
                }
            }

            //add remaining positions
            if(sequenceLength % nucleotidesPerInt2Bit != 0){
                const unsigned int currentDataInt = sequence[fullInts];
                const int maxPos = sequenceLength - fullInts * 16;

                for(int posInInt = group.thread_rank(); posInInt < maxPos; posInInt += group.size()){
                    unsigned int encodedBaseAsInt = getEncodedNucFromInt2Bit(currentDataInt, posInInt);
                    if(!isForward){
                        //reverse complement
                        encodedBaseAsInt = (~encodedBaseAsInt & 0x00000003);
                    }
                    const float weight = canUseQualityScores ? getQualityWeight(quality[fullInts * 16 + posInInt]) * overlapweight : overlapweight;

                    assert(weight != 0);
                    const int rowOffset = encodedBaseAsInt * columnPitchInElements;
                    const int columnIndex = columnStart 
                        + (isForward ? (fullInts * 16 + posInInt) : sequenceLength - 1 - (fullInts * 16 + posInInt));
                        atomicAdd(counts + rowOffset + columnIndex, doAdd ? 1 : -1);
                        atomicAdd(weights + rowOffset + columnIndex, doAdd ? weight : -weight);
                        atomicAdd(coverages + columnIndex, doAdd ? 1 : -1);
                } 
            }
        }

        template<class ThreadGroup>
        __device__ __forceinline__
        void constructFromSequences(
            ThreadGroup& group,
            const int* __restrict__ myShifts,
            const int* __restrict__ myOverlaps,
            const int* __restrict__ myNops,
            const BestAlignment_t* __restrict__ myAlignmentFlags,
            const unsigned int* __restrict__ myAnchorSequenceData,
            const char* __restrict__ myAnchorQualityData,
            const unsigned int* __restrict__ myCandidateSequencesData,
            const char* __restrict__ myCandidateQualities,
            const int* __restrict__ myCandidateLengths,
            const int* __restrict__ myIndices,
            int numIndices,
            bool canUseQualityScores, 
            size_t encodedSequencePitchInInts,
            size_t qualityPitchInBytes,
            float desiredAlignmentMaxErrorRate,
            int subjectIndex
        ){   
            constexpr int threadsPerSequence = 8;
            auto tile = cg::tiled_partition<threadsPerSequence>(group);
            const int tileIdInGroup = group.thread_rank() / threadsPerSequence;
            const int numTilesInGroup = group.size() / threadsPerSequence;

            for(int column = group.thread_rank(); column < columnPitchInElements; column += group.size()){
                for(int i = 0; i < 4; i++){
                    counts[i * columnPitchInElements + column] = 0;
                    weights[i * columnPitchInElements + column] = 0.0f;
                }

                coverages[column] = 0;
            }
            
            group.sync();

            
            const int subjectColumnsBegin_incl = columnProperties->subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = columnProperties->subjectColumnsEnd_excl;

            const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
            const unsigned int* const subject = myAnchorSequenceData;
            const char* const subjectQualityScore = myAnchorQualityData;
                            
            // for(int i = group.thread_rank(); i < subjectLength; i += group.size()){
            //     const int columnIndex = subjectColumnsBegin_incl + i;
            //     const unsigned int encbase = getEncodedNuc2Bit(subject, subjectLength, i);
            //     const float weight = canUseQualityScores ? getQualityWeight(subjectQualityScore[i]) : 1.0f;
            //     const int rowOffset = int(encbase) * columnPitchInElements;

            //     atomicAdd(counts + rowOffset + columnIndex, 1);
            //     atomicAdd(weights + rowOffset + columnIndex, weight);
            //     atomicAdd(coverages + columnIndex, 1);
            // }

            //add anchor

            if(tileIdInGroup == 0){
                msaAddOrDeleteASequence2Bit<true>(
                    tile,
                    subject, 
                    subjectLength, 
                    true,
                    subjectColumnsBegin_incl,
                    1.0f,
                    subjectQualityScore,
                    canUseQualityScores
                );
            }

            for(int indexInList = tileIdInGroup; indexInList < numIndices; indexInList += numTilesInGroup){

                const int localCandidateIndex = myIndices[indexInList];
                const int shift = myShifts[localCandidateIndex];
                const BestAlignment_t flag = myAlignmentFlags[localCandidateIndex];

                const int queryLength = myCandidateLengths[localCandidateIndex];
                const unsigned int* const query = myCandidateSequencesData 
                    + std::size_t(localCandidateIndex) * encodedSequencePitchInInts;

                const char* const queryQualityScore = myCandidateQualities 
                    + std::size_t(localCandidateIndex) * qualityPitchInBytes;

                const int query_alignment_overlap = myOverlaps[localCandidateIndex];
                const int query_alignment_nops = myNops[localCandidateIndex];

                const float overlapweight = calculateOverlapWeight(
                    subjectLength, 
                    query_alignment_nops, 
                    query_alignment_overlap,
                    desiredAlignmentMaxErrorRate
                );

                assert(overlapweight <= 1.0f);
                assert(overlapweight >= 0.0f);
                assert(flag != BestAlignment_t::None); // indices should only be pointing to valid alignments

                const int defaultcolumnoffset = subjectColumnsBegin_incl + shift;

                const bool isForward = flag == BestAlignment_t::Forward;

                msaAddOrDeleteASequence2Bit<true>(
                    tile,
                    query, 
                    queryLength, 
                    isForward,
                    defaultcolumnoffset,
                    overlapweight,
                    queryQualityScore,
                    canUseQualityScores
                );
            }
        }


        template<class ThreadGroup, class Selector>
        __device__ __forceinline__
        void removeCandidates(
            ThreadGroup& group,
            Selector shouldBeRemoved, //remove candidate myIndices[i] if shouldBeRemoved(i) == true
            const int* __restrict__ myShifts,
            const int* __restrict__ myOverlaps,
            const int* __restrict__ myNops,
            const BestAlignment_t* __restrict__ myAlignmentFlags,
            const unsigned int* __restrict__ myCandidateSequencesData, //not transposed
            const char* __restrict__ myCandidateQualities, //not transposed
            const int* __restrict__ myCandidateLengths,
            const int* __restrict__ myIndices,
            int numIndices,
            bool canUseQualityScores, 
            size_t encodedSequencePitchInInts,
            size_t qualityPitchInBytes,
            float desiredAlignmentMaxErrorRate
        ){  
    

            const int subjectColumnsBegin_incl = columnProperties->subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = columnProperties->subjectColumnsEnd_excl;

            constexpr int threadsPerSequence = 8;
            auto tile = cg::tiled_partition<threadsPerSequence>(group);
            const int tileIdInGroup = group.thread_rank() / threadsPerSequence;
            const int numTilesInGroup = group.size() / threadsPerSequence;
                            
            for(int indexInList = tileIdInGroup; indexInList < numIndices; indexInList += numTilesInGroup){

                if(shouldBeRemoved(indexInList)){

                    const int localCandidateIndex = myIndices[indexInList];
                    const int shift = myShifts[localCandidateIndex];
                    const BestAlignment_t flag = myAlignmentFlags[localCandidateIndex];

                    const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
                    const int queryLength = myCandidateLengths[localCandidateIndex];
                    const unsigned int* const query = myCandidateSequencesData + localCandidateIndex * encodedSequencePitchInInts;

                    const char* const queryQualityScore = myCandidateQualities + std::size_t(localCandidateIndex) * qualityPitchInBytes;

                    const int query_alignment_overlap = myOverlaps[localCandidateIndex];
                    const int query_alignment_nops = myNops[localCandidateIndex];

                    const float overlapweight = calculateOverlapWeight(
                        subjectLength, 
                        query_alignment_nops, 
                        query_alignment_overlap,
                        desiredAlignmentMaxErrorRate
                    );

                    assert(overlapweight <= 1.0f);
                    assert(overlapweight >= 0.0f);
                    assert(flag != BestAlignment_t::None);                 // indices should only be pointing to valid alignments

                    const int defaultcolumnoffset = subjectColumnsBegin_incl + shift;

                    const bool isForward = flag == BestAlignment_t::Forward;

                    msaAddOrDeleteASequence2Bit<false>(
                        tile,
                        query, 
                        queryLength, 
                        isForward,
                        defaultcolumnoffset,
                        overlapweight,
                        queryQualityScore,
                        canUseQualityScores
                    );

                }
            }
        }

        template<class ThreadGroup>
        __device__ __forceinline__
        void findConsensus(
            ThreadGroup& group,
            const unsigned int* __restrict__ myAnchorSequenceData, 
            int encodedSequencePitchInInts,
            int subjectIndex = -1
        ){

            const int subjectColumnsBegin_incl = columnProperties->subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = columnProperties->subjectColumnsEnd_excl;
            const int firstColumn_incl = columnProperties->firstColumn_incl;
            const int lastColumn_excl = columnProperties->lastColumn_excl;

            if(lastColumn_excl > columnPitchInElements){
                if(group.thread_rank() == 0){
                    printf("%d, %d %d\n", subjectIndex, lastColumn_excl, columnPitchInElements);
                }
                group.sync();
            }
            assert(lastColumn_excl <= columnPitchInElements);

            const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
            const unsigned int* const subject = myAnchorSequenceData;

            //set columns to zero which are not in range firstColumn_incl <= column && column < lastColumn_excl

            for(int column = group.thread_rank(); 
                    column < firstColumn_incl; 
                    column += group.size()){

                support[column] = 0;
                origWeights[column] = 0;
                origCoverages[column] = 0;
            }

            const int leftoverRight = columnPitchInElements - lastColumn_excl;

            for(int i = group.thread_rank(); i < leftoverRight; i += group.size()){
                const int column = lastColumn_excl + i;

                support[column] = 0;
                origWeights[column] = 0;
                origCoverages[column] = 0;
            }

            for(int column = group.thread_rank(); 
                    column < firstColumn_incl; 
                    column += group.size()){
                    
                consensus[column] = 5;
            }

            for(int i = group.thread_rank(); i < leftoverRight; i += group.size()){
                const int column = lastColumn_excl + i;

                consensus[column] = 5;
            }

            const int* const myCountsA = counts + 0 * columnPitchInElements;
            const int* const myCountsC = counts + 1 * columnPitchInElements;
            const int* const myCountsG = counts + 2 * columnPitchInElements;
            const int* const myCountsT = counts + 3 * columnPitchInElements;

            const float* const myWeightsA = weights + 0 * columnPitchInElements;
            const float* const myWeightsC = weights + 1 * columnPitchInElements;
            const float* const myWeightsG = weights + 2 * columnPitchInElements;
            const float* const myWeightsT = weights + 3 * columnPitchInElements;

            const int numOuterIters = SDIV(lastColumn_excl, 4);

            for(int outerIter = group.thread_rank(); outerIter < numOuterIters; outerIter += group.size()){

                alignas(4) char consensusArray[4];

                #pragma unroll 
                for(int i = 0; i < 4; i++){
                    const int column = outerIter * 4 + i;

                    if(firstColumn_incl <= column && column < lastColumn_excl){

                        const int ca = myCountsA[column];
                        const int cc = myCountsC[column];
                        const int cg = myCountsG[column];
                        const int ct = myCountsT[column];
                        const float wa = myWeightsA[column];
                        const float wc = myWeightsC[column];
                        const float wg = myWeightsG[column];
                        const float wt = myWeightsT[column];

                        char cons = 5;
                        float consWeight = 0.0f;
                        if(wa > consWeight){
                            cons = 0;
                            consWeight = wa;
                        }
                        if(wc > consWeight){
                            cons = 1;
                            consWeight = wc;
                        }
                        if(wg > consWeight){
                            cons = 2;
                            consWeight = wg;
                        }
                        if(wt > consWeight){
                            cons = 3;
                            consWeight = wt;
                        }

                        consensusArray[i] = cons;

                        const float columnWeight = wa + wc + wg + wt;
                        if(columnWeight == 0){
                            printf("s %d c %d\n", subjectIndex, column);
                            assert(columnWeight != 0);
                        }

                        support[column] = consWeight / columnWeight;

                        if(subjectColumnsBegin_incl <= column && column < subjectColumnsEnd_excl){
                            constexpr unsigned int A_enc = 0x00;
                            constexpr unsigned int C_enc = 0x01;
                            constexpr unsigned int G_enc = 0x02;
                            constexpr unsigned int T_enc = 0x03;

                            const int localIndex = column - subjectColumnsBegin_incl;
                            const unsigned int encNuc = getEncodedNuc2Bit(subject, subjectLength, localIndex);

                            if(encNuc == A_enc){
                                origWeights[column] = wa;
                                origCoverages[column] = ca;
                            }else if(encNuc == C_enc){
                                origWeights[column] = wc;
                                origCoverages[column] = cc;
                            }else if(encNuc == G_enc){
                                origWeights[column] = wg;
                                origCoverages[column] = cg;
                            }else if(encNuc == T_enc){
                                origWeights[column] = wt;
                                origCoverages[column] = ct;
                            }
                        }
                    }
                }

                *((char4*)(consensus + 4*outerIter)) = *((const char4*)(&consensusArray[0]));
            }
        }

        template<
            class ThreadGroup, 
            class GroupReduceBool, 
            class GroupReduceInt2
        >
        __device__ __forceinline__
        void flagCandidatesOfDifferentRegion(
            ThreadGroup& group,
            GroupReduceBool& groupReduceBool,
            GroupReduceInt2& groupReduceInt2,
            int* __restrict__ myNewIndicesPtr,
            int* __restrict__ myNewNumIndicesPerSubjectPtr,
            const unsigned int* __restrict__ myAnchorSequenceData,
            const int subjectLength,
            const unsigned int* __restrict__ myCandidateSequencesData,
            const int* __restrict__ myCandidateLengths,
            const BestAlignment_t* myAlignmentFlags,
            const int* __restrict__ myShifts,
            const int* __restrict__ myNops,
            const int* __restrict__ myOverlaps,
            bool* __restrict__ myShouldBeKept,
            float desiredAlignmentMaxErrorRate,
            int subjectIndex,
            int encodedSequencePitchInInts,
            const int* __restrict__ myIndices,
            const int myNumIndices,
            int dataset_coverage
        ){
            
            static_assert(std::is_same<ThreadGroup, cg::thread_block>::value, 
                "Can only use thread_block for msa::flagCandidatesOfDifferentRegion"); //because of shared per group               

            auto is_significant_count = [](int count, int coverage){
                if(int(coverage * 0.3f) <= count)
                    return true;
                return false;
            };        

            // auto to_nuc = [](unsigned int c){
            //     constexpr unsigned int A_enc = 0x00;
            //     constexpr unsigned int C_enc = 0x01;
            //     constexpr unsigned int G_enc = 0x02;
            //     constexpr unsigned int T_enc = 0x03;

            //     switch(c){
            //     case A_enc: return 'A';
            //     case C_enc: return 'C';
            //     case G_enc: return 'G';
            //     case T_enc: return 'T';
            //     default: return 'F';
            //     }
            // };

            __shared__ bool broadcastbufferbool;
            __shared__ int broadcastbufferint4[4];
            __shared__ int smemcounts[1];

            const unsigned int* const subjectptr = myAnchorSequenceData;

            const int subjectColumnsBegin_incl = columnProperties->subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = columnProperties->subjectColumnsEnd_excl;

            //check if subject and consensus differ at at least one position

            bool hasMismatchToConsensus = false;

            for(int pos = group.thread_rank(); pos < subjectLength && !hasMismatchToConsensus; pos += group.size()){
                const int column = subjectColumnsBegin_incl + pos;
                const char consbase = consensus[column];
                const char subjectbase = getEncodedNuc2Bit(subjectptr, subjectLength, pos);

                hasMismatchToConsensus |= (consbase != subjectbase);
            }

            hasMismatchToConsensus = groupReduceBool(hasMismatchToConsensus, [](auto l, auto r){
                return l || r;
            });

            if(group.thread_rank() == 0){
                broadcastbufferbool = hasMismatchToConsensus;
            }
            group.sync();

            hasMismatchToConsensus = broadcastbufferbool;

            //if subject and consensus differ at at least one position, check columns in msa

            if(hasMismatchToConsensus){
                int col = std::numeric_limits<int>::max();
                bool foundColumn = false;
                char foundBase = 'F';
                int foundBaseIndex = std::numeric_limits<int>::max();
                int consindex = std::numeric_limits<int>::max();

                const int* const myCountsA = counts + 0 * columnPitchInElements;
                const int* const myCountsC = counts + 1 * columnPitchInElements;
                const int* const myCountsG = counts + 2 * columnPitchInElements;
                const int* const myCountsT = counts + 3 * columnPitchInElements;

                for(int columnindex = subjectColumnsBegin_incl + group.thread_rank(); 
                        columnindex < subjectColumnsEnd_excl && !foundColumn; 
                        columnindex += group.size()){

                    int regcounts[4];
                    regcounts[0] = myCountsA[columnindex];
                    regcounts[1] = myCountsC[columnindex];
                    regcounts[2] = myCountsG[columnindex];
                    regcounts[3] = myCountsT[columnindex];

                    const char consbase = consensus[columnindex];
                    consindex = consbase;

                    assert(0 <= consindex && consindex < 4);

                    //find out if there is a non-consensus base with significant coverage
                    int significantBaseIndex = -1;

                    #pragma unroll
                    for(int i = 0; i < 4; i++){
                        if(i != consindex){
                            const bool significant = is_significant_count(regcounts[i], dataset_coverage);

                            significantBaseIndex = significant ? i : significantBaseIndex;
                        }
                    }

                    if(significantBaseIndex != -1){
                        foundColumn = true;
                        col = columnindex;
                        foundBaseIndex = significantBaseIndex;
                    }
                }

                int2 packed{col, foundBaseIndex};
                //find packed value with smallest col
                packed = groupReduceInt2(packed, [](auto l, auto r){
                    if(l.x < r.x){
                        return l;
                    }else{
                        return r;
                    }
                });

                if(group.thread_rank() == 0){
                    if(packed.x != std::numeric_limits<int>::max()){
                        broadcastbufferint4[0] = 1;
                        broadcastbufferint4[1] = packed.x;
                        broadcastbufferint4[2] = packed.y;
                        broadcastbufferint4[3] = packed.y;
                    }else{
                        broadcastbufferint4[0] = 0;
                    }
                }

                group.sync();

                foundColumn = (1 == broadcastbufferint4[0]);
                col = broadcastbufferint4[1];
                foundBase = broadcastbufferint4[2];
                foundBaseIndex = broadcastbufferint4[3];

                if(foundColumn){
                    
                    auto discard_rows = [&](bool keepMatching){
                        
                        for(int k = group.thread_rank(); k < myNumIndices; k += group.size()){
                            const int localCandidateIndex = myIndices[k];
                            const unsigned int* const candidateptr = myCandidateSequencesData + std::size_t(localCandidateIndex) * encodedSequencePitchInInts;
                            const int candidateLength = myCandidateLengths[localCandidateIndex];
                            const int shift = myShifts[localCandidateIndex];
                            const BestAlignment_t alignmentFlag = myAlignmentFlags[localCandidateIndex];

                            //check if row is affected by column col
                            const int row_begin_incl = subjectColumnsBegin_incl + shift;
                            const int row_end_excl = row_begin_incl + candidateLength;
                            const bool notAffected = (col < row_begin_incl || row_end_excl <= col);
                            char base = 5;
                            if(!notAffected){
                                if(alignmentFlag == BestAlignment_t::Forward){
                                    base = getEncodedNuc2Bit(candidateptr, candidateLength, (col - row_begin_incl));
                                }else{
                                    //all candidates of MSA must not have alignmentflag None
                                    assert(alignmentFlag == BestAlignment_t::ReverseComplement); 

                                    const unsigned int forwardbaseEncoded = getEncodedNuc2Bit(candidateptr, candidateLength, row_end_excl-1 - col);
                                    base = (~forwardbaseEncoded & 0x03);
                                }
                            }

                            if(notAffected || (!(keepMatching ^ (base == foundBase)))){
                                myShouldBeKept[k] = true; //same region
                            }else{
                                myShouldBeKept[k] = false; //different region
                            }
                        }
                        #if 1
                        //check that no candidate which should be removed has very good alignment.
                        //if there is such a candidate, none of the candidates will be removed.
                        bool veryGoodAlignment = false;
                        for(int k = group.thread_rank(); k < myNumIndices && !veryGoodAlignment; k += group.size()){
                            if(!myShouldBeKept[k]){
                                const int localCandidateIndex = myIndices[k];
                                const int nOps = myNops[localCandidateIndex];
                                const int overlapsize = myOverlaps[localCandidateIndex];
                                const float overlapweight = calculateOverlapWeight(
                                    subjectLength, 
                                    nOps, 
                                    overlapsize,
                                    desiredAlignmentMaxErrorRate
                                );
                                assert(overlapweight <= 1.0f);
                                assert(overlapweight >= 0.0f);

                                if(fgeq(overlapweight, 0.90f)){
                                    veryGoodAlignment = true;
                                }
                            }
                        }

                        veryGoodAlignment = groupReduceBool(veryGoodAlignment, [](auto l, auto r){
                            return l || r;
                        });

                        if(group.thread_rank() == 0){
                            broadcastbufferbool = veryGoodAlignment;
                        }
                        group.sync();

                        veryGoodAlignment = broadcastbufferbool;

                        if(veryGoodAlignment){
                            for(int k = group.thread_rank(); k < myNumIndices; k += group.size()){
                                myShouldBeKept[k] = true;
                            }
                        }
                        #endif

                        //select indices of candidates to keep and write them to new indices
                        if(group.thread_rank() == 0){
                            smemcounts[0] = 0;
                        }
                        group.sync();

                        const int limit = SDIV(myNumIndices, group.size()) * group.size();
                        for(int k = group.thread_rank(); k < limit; k += group.size()){
                            bool keep = false;
                            if(k < myNumIndices){
                                keep = myShouldBeKept[k];
                            }                               
                
                            if(keep){
                                cg::coalesced_group g = cg::coalesced_threads();
                                int outputPos;
                                if (g.thread_rank() == 0) {
                                    outputPos = atomicAdd(&smemcounts[0], g.size());
                                }
                                outputPos = g.thread_rank() + g.shfl(outputPos, 0);
                                myNewIndicesPtr[outputPos] = myIndices[k];
                            }                        
                        }

                        group.sync();

                        if(group.thread_rank() == 0){
                            *myNewNumIndicesPerSubjectPtr = smemcounts[0];
                        }

                        group.sync();

                    };

                    //compare found base to original base
                    const char originalbase = getEncodedNuc2Bit(subjectptr, subjectLength, col - subjectColumnsBegin_incl);

                    if(originalbase == foundBase){
                        //discard all candidates whose base in column col differs from foundBase
                        discard_rows(true);
                    }else{
                        //discard all candidates whose base in column col matches foundBase
                        discard_rows(false);
                    }

                }else{
                    //did not find a significant columns

                    //remove no candidate
                    for(int k = group.thread_rank(); k < myNumIndices; k += group.size()){
                        myNewIndicesPtr[k] = myIndices[k];
                    }
                    if(group.thread_rank() == 0){
                        *myNewNumIndicesPerSubjectPtr = myNumIndices;
                    }
                }

            }else{
                //no mismatch between consensus and subject

                //remove no candidate
                for(int k = group.thread_rank(); k < myNumIndices; k += group.size()){
                    myShouldBeKept[k] = true;
                }

                for(int k = group.thread_rank(); k < myNumIndices; k += group.size()){
                    myNewIndicesPtr[k] = myIndices[k];
                }
                if(group.thread_rank() == 0){
                    *myNewNumIndicesPerSubjectPtr = myNumIndices;
                }
            }
        }

    public:
        int columnPitchInElements;
        int* counts;
        float* weights;
        int* coverages;
        char* consensus;
        float* support;
        float* origWeights;
        int* origCoverages;
        MSAColumnProperties* columnProperties;
    };

    struct GPUMultiMSA{
    public:
        HOSTDEVICEQUALIFIER
        GpuSingleMSA getSingleMSA(int msaIndex) const{
            GpuSingleMSA msa;

            msa.columnPitchInElements = columnPitchInElements;
            msa.counts = getCountsOfMSA(msaIndex);
            msa.weights = getWeightsOfMSA(msaIndex);
            msa.coverages = getCoveragesOfMSA(msaIndex);
            msa.consensus = getConsensusOfMSA(msaIndex);
            msa.support = getSupportOfMSA(msaIndex);
            msa.origWeights = getOrigWeightsOfMSA(msaIndex);
            msa.origCoverages = getOrigCoveragesOfMSA(msaIndex);
            msa.columnProperties = getColumnPropertiesOfMSA(msaIndex);

            return msa;
        }

        HOSTDEVICEQUALIFIER
        int* getCountsOfMSA(int msaIndex) const{
            return counts + std::size_t(columnPitchInElements) * 4 * msaIndex;
        }

        HOSTDEVICEQUALIFIER
        float* getWeightsOfMSA(int msaIndex) const{
            return weights + std::size_t(columnPitchInElements) * 4 * msaIndex;
        }

        HOSTDEVICEQUALIFIER
        int* getCoveragesOfMSA(int msaIndex) const{
            return coverages + std::size_t(columnPitchInElements) * msaIndex;
        }

        HOSTDEVICEQUALIFIER
        char* getConsensusOfMSA(int msaIndex) const{
            return consensus + std::size_t(columnPitchInElements) * msaIndex;
        }

        HOSTDEVICEQUALIFIER
        float* getSupportOfMSA(int msaIndex) const{
            return support + std::size_t(columnPitchInElements) * msaIndex;
        }

        HOSTDEVICEQUALIFIER
        float* getOrigWeightsOfMSA(int msaIndex) const{
            return origWeights + std::size_t(columnPitchInElements) * msaIndex;
        }

        HOSTDEVICEQUALIFIER
        int* getOrigCoveragesOfMSA(int msaIndex) const{
            return origCoverages + std::size_t(columnPitchInElements) * msaIndex;
        }

        HOSTDEVICEQUALIFIER
        MSAColumnProperties* getColumnPropertiesOfMSA(int msaIndex) const{
            return columnProperties + msaIndex;
        }

    public:
        int numMSAs;
        int columnPitchInElements;
        int* counts;
        float* weights;
        int* coverages;
        char* consensus;
        float* support;
        float* origWeights;
        int* origCoverages;
        MSAColumnProperties* columnProperties;
    };


    
}
}


#endif

#endif