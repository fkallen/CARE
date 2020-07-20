//#define NDEBUG

#include <gpu/kernels.hpp>
#include <hostdevicefunctions.cuh>

//#include <gpu/bestalignment.hpp>
#include <bestalignment.hpp>
#include <gpu/utility_kernels.cuh>

//#include <msa.hpp>
#include <sequence.hpp>

#include <gpu/gpumsa.cuh>


#include <hpc_helpers.cuh>
#include <config.hpp>

#include <cassert>


#include <cub/cub.cuh>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include <thrust/binary_search.h>


namespace care{
namespace gpu{

    enum class MemoryType{
        Global,
        Shared
    };

    enum class SequenceLayout{
        Transposed,
        Linear
    };


    __device__ __forceinline__
    void checkBuiltMSA(
            const MSAColumnProperties* __restrict__ msaColumnProperties,
            const int* __restrict__ counts,
            const float* __restrict__ weights,
            size_t msaColumnPitchInElements,
            int subjectIndex){

        const int firstColumn_incl = msaColumnProperties->firstColumn_incl;
        const int lastColumn_excl = msaColumnProperties->lastColumn_excl;

        for(int column = firstColumn_incl + threadIdx.x; column < lastColumn_excl; column += blockDim.x){
            const int* const mycounts = counts + column;
            const float* const myweights = weights + column;
            float sumOfWeights = 0.0f;

            for(int k = 0; k < 4; k++){
                const int count = mycounts[k * msaColumnPitchInElements];
                const float weight = myweights[k * msaColumnPitchInElements];
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

    template<int BLOCKSIZE>
    __device__ __forceinline__
    void msaInit(
            typename cub::BlockReduce<int, BLOCKSIZE>::TempStorage& tempReduce,
            MSAColumnProperties* __restrict__ msaColumnProperties,
            const int* __restrict__ goodCandidateIndices,
            int numGoodCandidates,
            const int* __restrict__ shifts,
            const BestAlignment_t* __restrict__ alignmentFlags,
            const int subjectLength,
            const int* __restrict__ candidateLengths
            ){

        using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;

        int startindex = 0;
        int endindex = subjectLength;

        for(int k = threadIdx.x; k < numGoodCandidates; k += BLOCKSIZE) {
            const int localCandidateIndex = goodCandidateIndices[k];

            const int shift = shifts[localCandidateIndex];
            const BestAlignment_t flag = alignmentFlags[localCandidateIndex];
            const int queryLength = candidateLengths[localCandidateIndex];

            assert(flag != BestAlignment_t::None);

            const int queryEndsAt = queryLength + shift;
            startindex = min(startindex, shift);
            endindex = max(endindex, queryEndsAt);
        }

        startindex = BlockReduceInt(tempReduce).Reduce(startindex, cub::Min());
        __syncthreads();

        endindex = BlockReduceInt(tempReduce).Reduce(endindex, cub::Max());
        __syncthreads();

        if(threadIdx.x == 0) {
            MSAColumnProperties my_columnproperties;

            my_columnproperties.subjectColumnsBegin_incl = max(-startindex, 0);
            my_columnproperties.subjectColumnsEnd_excl = my_columnproperties.subjectColumnsBegin_incl + subjectLength;
            my_columnproperties.firstColumn_incl = 0;
            my_columnproperties.lastColumn_excl = endindex - startindex;

            *msaColumnProperties = my_columnproperties;
        }
    }

    template<int BLOCKSIZE>
    __device__ __forceinline__
    void msaUpdatePropertiesAfterSequenceRemovalSingleBlock(
            MSAColumnProperties* __restrict__ myMsaColumnProperties,
            const int* __restrict__ myCoverage){

        const int firstColumn_incl = myMsaColumnProperties->firstColumn_incl;
        const int lastColumn_excl = myMsaColumnProperties->lastColumn_excl;
        const int numColumnsToCheck = lastColumn_excl - firstColumn_incl;

        int newFirstColumn_incl = -1;
        int newLastColumn_excl = -1;

        for(int i = threadIdx.x; i < numColumnsToCheck-1; i += BLOCKSIZE){
            const int column = firstColumn_incl + i;

            const int thisCoverage = myCoverage[column];
            const int nextCoverage = myCoverage[column+1];
            assert(thisCoverage >= 0);
            assert(nextCoverage >= 0);

            if(thisCoverage == 0 && nextCoverage > 0){
                newFirstColumn_incl = column+1;
            }

            if(thisCoverage > 0 && nextCoverage == 0){
                newLastColumn_excl = column+1;
            }
        }

        __syncthreads();

        __shared__ int checkcount[2];
        if(threadIdx.x == 0){
            checkcount[0] = 0;
            checkcount[1] = 0;
        }
        __syncthreads();

        //there can be at most one thread for which this is true
        if(newFirstColumn_incl != -1){    
            atomicAdd(&checkcount[0], 1);
            myMsaColumnProperties->firstColumn_incl = newFirstColumn_incl;
        }
        //there can be at most one thread for which this is true
        if(newLastColumn_excl != -1){
            atomicAdd(&checkcount[1], 1);
            myMsaColumnProperties->lastColumn_excl = newLastColumn_excl;
        }
        
        __syncthreads();
        if(threadIdx.x == 0){
            assert(checkcount[0] <= 1);
            assert(checkcount[1] <= 1);
        }
    }





    template<bool isTransposedSequence, bool doAdd>
    __device__ 
    void msaAddOrDeleteASequence2Bit(
            int* __restrict__ counts,
            float* __restrict__ weights,
            int* __restrict__ coverages,
            int msaColumnPitchInElements,
            const unsigned int* sequence, 
            int sequenceLength, 
            bool isForward,
            int columnStart,
            float overlapweight,
            const char* quality,
            bool canUseQualityScores,
            int sequenceelementoffset){

        auto getEncodedNucFromInt2Bit = [](unsigned int data, int pos){
            return ((data >> (30 - 2*pos)) & 0x00000003);
        };

        constexpr int nucleotidesPerInt2Bit = 16;
        const int fullInts = sequenceLength / nucleotidesPerInt2Bit;

        for(int intIndex = 0; intIndex < fullInts; intIndex++){
            const unsigned int currentDataInt = sequence[intIndex * sequenceelementoffset];

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
                    const int rowOffset = encodedBaseAsInt * msaColumnPitchInElements;
                    const int columnIndex = columnStart 
                            + (isForward ? (intIndex * 16 + posInInt) : sequenceLength - 1 - (intIndex * 16 + posInInt));
                    
                    atomicAdd(counts + rowOffset + columnIndex, doAdd ? 1 : -1);
                    float n = atomicAdd(weights + rowOffset + columnIndex, doAdd ? weight : -weight);
                    atomicAdd(coverages + columnIndex, doAdd ? 1 : -1);

                    // if(columnIndex == 26){
                    //     float aaa[4]{0,0,0,0};
                    //     aaa[encodedBaseAsInt] = weight;

                    //     printf("column 26 add qscore %c, qual weight %.8f overlap weight %.8f, %.10f %.10f %.10f %.10f. new %.10f\n", 
                    //         currentFourQualities[l], getQualityWeight(currentFourQualities[l]),overlapweight,aaa[0], aaa[1], aaa[2], aaa[3], n + weight);
                    // }
                }
            }
        }

        //add remaining positions
        if(sequenceLength % nucleotidesPerInt2Bit != 0){
            const unsigned int currentDataInt = sequence[fullInts * sequenceelementoffset];
            const int maxPos = sequenceLength - fullInts * 16;
            for(int posInInt = 0; posInInt < maxPos; posInInt++){
                unsigned int encodedBaseAsInt = getEncodedNucFromInt2Bit(currentDataInt, posInInt);
                if(!isForward){
                    //reverse complement
                    encodedBaseAsInt = (~encodedBaseAsInt & 0x00000003);
                }
                const float weight = canUseQualityScores ? getQualityWeight(quality[fullInts * 16 + posInInt]) * overlapweight : overlapweight;

                assert(weight != 0);
                const int rowOffset = encodedBaseAsInt * msaColumnPitchInElements;
                const int columnIndex = columnStart 
                    + (isForward ? (fullInts * 16 + posInInt) : sequenceLength - 1 - (fullInts * 16 + posInInt));
                    atomicAdd(counts + rowOffset + columnIndex, doAdd ? 1 : -1);
                    atomicAdd(weights + rowOffset + columnIndex, doAdd ? weight : -weight);
                    atomicAdd(coverages + columnIndex, doAdd ? 1 : -1);
            } 
        }
    }


    template<bool isTransposedSequence, bool doAdd, class ThreadGroup>
    __device__ 
    void msaAddOrDeleteASequence2Bit(
            ThreadGroup& threadGroup,
            int* __restrict__ counts,
            float* __restrict__ weights,
            int* __restrict__ coverages,
            int msaColumnPitchInElements,
            const unsigned int* sequence, 
            int sequenceLength, 
            bool isForward,
            int columnStart,
            float overlapweight,
            const char* quality,
            bool canUseQualityScores,
            int sequenceelementoffset){

        auto getEncodedNucFromInt2Bit = [](unsigned int data, int pos){
            return ((data >> (30 - 2*pos)) & 0x00000003);
        };

        constexpr int nucleotidesPerInt2Bit = 16;
        const int fullInts = sequenceLength / nucleotidesPerInt2Bit;

        for(int intIndex = threadGroup.thread_rank(); intIndex < fullInts; intIndex += threadGroup.size()){
        //for(int intIndex = 0; intIndex < fullInts; intIndex++){
            const unsigned int currentDataInt = sequence[intIndex * sequenceelementoffset];

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
                    const int rowOffset = encodedBaseAsInt * msaColumnPitchInElements;
                    const int columnIndex = columnStart 
                            + (isForward ? (intIndex * 16 + posInInt) : sequenceLength - 1 - (intIndex * 16 + posInInt));
                    
                    atomicAdd(counts + rowOffset + columnIndex, doAdd ? 1 : -1);
                    float n = atomicAdd(weights + rowOffset + columnIndex, doAdd ? weight : -weight);
                    atomicAdd(coverages + columnIndex, doAdd ? 1 : -1);

                    // if(columnIndex == 26){
                    //     float aaa[4]{0,0,0,0};
                    //     aaa[encodedBaseAsInt] = weight;

                    //     printf("column 26 add qscore %c, qual weight %.8f overlap weight %.8f, %.10f %.10f %.10f %.10f. new %.10f\n", 
                    //         currentFourQualities[l], getQualityWeight(currentFourQualities[l]),overlapweight,aaa[0], aaa[1], aaa[2], aaa[3], n + weight);
                    // }
                }
            }
        }

        //add remaining positions
        if(sequenceLength % nucleotidesPerInt2Bit != 0){
            const unsigned int currentDataInt = sequence[fullInts * sequenceelementoffset];
            const int maxPos = sequenceLength - fullInts * 16;
            //for(int posInInt = 0; posInInt < maxPos; posInInt++){
            for(int posInInt = threadGroup.thread_rank(); posInInt < maxPos; posInInt += threadGroup.size()){
                unsigned int encodedBaseAsInt = getEncodedNucFromInt2Bit(currentDataInt, posInInt);
                if(!isForward){
                    //reverse complement
                    encodedBaseAsInt = (~encodedBaseAsInt & 0x00000003);
                }
                const float weight = canUseQualityScores ? getQualityWeight(quality[fullInts * 16 + posInInt]) * overlapweight : overlapweight;

                assert(weight != 0);
                const int rowOffset = encodedBaseAsInt * msaColumnPitchInElements;
                const int columnIndex = columnStart 
                    + (isForward ? (fullInts * 16 + posInInt) : sequenceLength - 1 - (fullInts * 16 + posInInt));
                    atomicAdd(counts + rowOffset + columnIndex, doAdd ? 1 : -1);
                    atomicAdd(weights + rowOffset + columnIndex, doAdd ? weight : -weight);
                    atomicAdd(coverages + columnIndex, doAdd ? 1 : -1);
            } 
        }
    }



    template<int BLOCKSIZE, bool isTransposedSequence, bool doAdd>
    __device__ 
    void msaAddOrDeleteASequence2Bit_wholeblock(
            int* __restrict__ counts,
            float* __restrict__ weights,
            int* __restrict__ coverages,
            int msaColumnPitchInElements,
            const unsigned int* sequence, 
            int sequenceLength, 
            bool isForward,
            int columnStart,
            float overlapweight,
            const char* quality,
            bool canUseQualityScores,
            int sequenceelementoffset){

        auto getEncodedNucFromInt2Bit = [](unsigned int data, int pos){
            return ((data >> (30 - 2*pos)) & 0x00000003);
        };

        for(int k = threadIdx.x; k < sequenceLength; k += BLOCKSIZE){
		char qual = 'A';
            if(canUseQualityScores){
                qual = quality[k];
            }
            const unsigned int currentDataInt = sequence[(k / 16) * sequenceelementoffset];
            unsigned int encodedBase = getEncodedNucFromInt2Bit(currentDataInt, k % 16);
            if(!isForward){
                encodedBase = (~encodedBase & 0x00000003);
            }
            const float weight = canUseQualityScores ? getQualityWeight(qual) * overlapweight : overlapweight;
            assert(weight != 0);
            const int rowOffset = encodedBase * msaColumnPitchInElements;
            const int columnIndex = columnStart + (isForward ? k : sequenceLength - 1 - k);
//            counts[rowOffset + columnIndex] += (doAdd ? 1 : -1);
//            weights[rowOffset + columnIndex] += (doAdd ? weight : -weight);
//            coverages[columnIndex] += (doAdd ? 1 : -1);
            atomicAdd(counts + rowOffset + columnIndex, 1);
            atomicAdd(weights + rowOffset + columnIndex, weight);
            atomicAdd(coverages + columnIndex, 1);

       }
   }



    template<int BLOCKSIZE>
    __device__ __forceinline__
    void addSequencesToMSASingleBlock(
            int* __restrict__ inputcounts,
            float* __restrict__ inputweights,
            int* __restrict__ inputcoverages,
            const MSAColumnProperties* __restrict__ myMsaColumnProperties,
            const int* __restrict__ myShifts,
            const int* __restrict__ myOverlaps,
            const int* __restrict__ myNops,
            const BestAlignment_t* __restrict__ myAlignmentFlags,
            const unsigned int* __restrict__ myAnchorSequenceData,
            const char* __restrict__ myAnchorQualityData,
            const unsigned int* __restrict__ myTransposedCandidateSequencesData,
            const char* __restrict__ myCandidateQualities,
            const int* __restrict__ myCandidateLengths,
            const int* __restrict__ myIndices,
            int numIndices,
            size_t elementOffsetForTransposedCandidates,
            bool canUseQualityScores, 
            size_t msaColumnPitchInElements,
            size_t encodedSequencePitchInInts,
            size_t qualityPitchInBytes,
            float desiredAlignmentMaxErrorRate,
            int subjectIndex){  

        constexpr bool candidatesAreTransposed = true;

        int* const mycounts = inputcounts;
        float* const myweights = inputweights;
        int* const mycoverages = inputcoverages;        

        for(int column = threadIdx.x; column < msaColumnPitchInElements * 4; column += BLOCKSIZE){
            mycounts[column] = 0;
            myweights[column] = 0;
        }

        for(int column = threadIdx.x; column < msaColumnPitchInElements; column += BLOCKSIZE){
            mycoverages[column] = 0;
        }   
        
        __syncthreads();

        //add subject
        const int subjectColumnsBegin_incl = myMsaColumnProperties->subjectColumnsBegin_incl;
        const int subjectColumnsEnd_excl = myMsaColumnProperties->subjectColumnsEnd_excl;

        const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
        const unsigned int* const subject = myAnchorSequenceData;
        const char* const subjectQualityScore = myAnchorQualityData;
                        
        for(int i = threadIdx.x; i < subjectLength; i += BLOCKSIZE){
            const int columnIndex = subjectColumnsBegin_incl + i;
            const unsigned int encbase = getEncodedNuc2Bit(subject, subjectLength, i);
            const float weight = canUseQualityScores ? getQualityWeight(subjectQualityScore[i]) : 1.0f;
            const int rowOffset = int(encbase) * msaColumnPitchInElements;

            atomicAdd(mycounts + rowOffset + columnIndex, 1);
            atomicAdd(myweights + rowOffset + columnIndex, weight);
            atomicAdd(mycoverages + columnIndex, 1);
        }

        for(int indexInList = threadIdx.x; indexInList < numIndices; indexInList += BLOCKSIZE){

            const int localCandidateIndex = myIndices[indexInList];
            const int shift = myShifts[localCandidateIndex];
            const BestAlignment_t flag = myAlignmentFlags[localCandidateIndex];

            const int queryLength = myCandidateLengths[localCandidateIndex];
            const unsigned int* const query = myTransposedCandidateSequencesData + localCandidateIndex;

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

            // if(subjectIndex == 2){
            //     for(int x = 0; x < BLOCKSIZE; x++){
            //         if(threadIdx.x == x){
            //             printf("subject 2, indexInList %d, localCandidateIndex %d, isForward %d, shift %d\n",  
            //                     indexInList, localCandidateIndex, isForward, shift);
            //             for(int i = 0; i < queryLength; i += 1){
            //                 const unsigned int encbase = getEncodedNuc2Bit(query, queryLength, i, [&](auto p){return p * elementOffsetForTransposedCandidates;});
            //                 printf("%d ", encbase);
            //             }
            //             printf("\n");
            //         }
            //         __syncthreads();
            //     }
                
                
            // }

            msaAddOrDeleteASequence2Bit<candidatesAreTransposed, true>(
                mycounts,
                myweights,
                mycoverages,
                msaColumnPitchInElements,
                query, 
                queryLength, 
                isForward,
                defaultcolumnoffset,
                overlapweight,
                queryQualityScore,
                canUseQualityScores,
                elementOffsetForTransposedCandidates
            );
        }
    }



    template<int BLOCKSIZE>
    __device__ __forceinline__
    void addSequencesToMSASingleBlockNotTranposed(
            int* __restrict__ inputcounts,
            float* __restrict__ inputweights,
            int* __restrict__ inputcoverages,
            const MSAColumnProperties* __restrict__ myMsaColumnProperties,
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
            size_t elementOffsetForTransposedCandidates,
            bool canUseQualityScores, 
            size_t msaColumnPitchInElements,
            size_t encodedSequencePitchInInts,
            size_t qualityPitchInBytes,
            float desiredAlignmentMaxErrorRate,
            int subjectIndex){  

        constexpr bool candidatesAreTransposed = false;

        int* const mycounts = inputcounts;
        float* const myweights = inputweights;
        int* const mycoverages = inputcoverages;        

        for(int column = threadIdx.x; column < msaColumnPitchInElements * 4; column += BLOCKSIZE){
            mycounts[column] = 0;
            myweights[column] = 0;
        }

        for(int column = threadIdx.x; column < msaColumnPitchInElements; column += BLOCKSIZE){
            mycoverages[column] = 0;
        }   
        
        __syncthreads();

        //add subject
        const int subjectColumnsBegin_incl = myMsaColumnProperties->subjectColumnsBegin_incl;
        const int subjectColumnsEnd_excl = myMsaColumnProperties->subjectColumnsEnd_excl;

        const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
        const unsigned int* const subject = myAnchorSequenceData;
        const char* const subjectQualityScore = myAnchorQualityData;
                        
        for(int i = threadIdx.x; i < subjectLength; i += BLOCKSIZE){
            const int columnIndex = subjectColumnsBegin_incl + i;
            const unsigned int encbase = getEncodedNuc2Bit(subject, subjectLength, i);
            const float weight = canUseQualityScores ? getQualityWeight(subjectQualityScore[i]) : 1.0f;
            const int rowOffset = int(encbase) * msaColumnPitchInElements;

            atomicAdd(mycounts + rowOffset + columnIndex, 1);
            atomicAdd(myweights + rowOffset + columnIndex, weight);
            atomicAdd(mycoverages + columnIndex, 1);
        }

        constexpr int threadsPerSequence = 8;
        auto tile = cg::tiled_partition<threadsPerSequence>(cg::this_thread_block());
        const int tileIdInBlock = threadIdx.x / threadsPerSequence;
        const int numTilesInBlock = blockDim.x / threadsPerSequence;

        for(int indexInList = tileIdInBlock; indexInList < numIndices; indexInList += numTilesInBlock){
         //    const int indexInList = k / threadsPerSequence;
        //for(int indexInList = threadIdx.x; indexInList < numIndices; indexInList += BLOCKSIZE){


            const int localCandidateIndex = myIndices[indexInList];
            const int shift = myShifts[localCandidateIndex];
            const BestAlignment_t flag = myAlignmentFlags[localCandidateIndex];

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

            // msaAddOrDeleteASequence2Bit<candidatesAreTransposed, true>(
            //     mycounts,
            //     myweights,
            //     mycoverages,
            //     msaColumnPitchInElements,
            //     query, 
            //     queryLength, 
            //     isForward,
            //     defaultcolumnoffset,
            //     overlapweight,
            //     queryQualityScore,
            //     canUseQualityScores,
            //     1
            // );

            msaAddOrDeleteASequence2Bit<candidatesAreTransposed, true>(
                tile,
                mycounts,
                myweights,
                mycoverages,
                msaColumnPitchInElements,
                query, 
                queryLength, 
                isForward,
                defaultcolumnoffset,
                overlapweight,
                queryQualityScore,
                canUseQualityScores,
                1
            );
        }
    }







    template<int BLOCKSIZE>
    __device__ __forceinline__
    void addSequencesToMSASingleBlockLessAtomic(
            int* __restrict__ inputcounts,
            float* __restrict__ inputweights,
            int* __restrict__ inputcoverages,
            const MSAColumnProperties* __restrict__ myMsaColumnProperties,
            const int* __restrict__ myShifts,
            const int* __restrict__ myOverlaps,
            const int* __restrict__ myNops,
            const BestAlignment_t* __restrict__ myAlignmentFlags,
            const unsigned int* __restrict__ myAnchorSequenceData,
            const char* __restrict__ myAnchorQualityData,
            const unsigned int* __restrict__ myTransposedCandidateSequencesData,
            const char* __restrict__ myCandidateQualities,
            const int* __restrict__ myCandidateLengths,
            const int* __restrict__ myIndices,
            int numIndices,
            size_t elementOffsetForTransposedCandidates,
            bool canUseQualityScores, 
            size_t msaColumnPitchInElements,
            size_t encodedSequencePitchInInts,
            size_t qualityPitchInBytes,
            float desiredAlignmentMaxErrorRate,
            int subjectIndex){  

        constexpr bool candidatesAreTransposed = true;

        int* const mycounts = inputcounts;
        float* const myweights = inputweights;
        int* const mycoverages = inputcoverages;        

        for(int column = threadIdx.x; column < msaColumnPitchInElements * 4; column += BLOCKSIZE){
            mycounts[column] = 0;
            myweights[column] = 0;
        }

        for(int column = threadIdx.x; column < msaColumnPitchInElements; column += BLOCKSIZE){
            mycoverages[column] = 0;
        }   
        
        //__syncthreads();

        //add subject
        const int subjectColumnsBegin_incl = myMsaColumnProperties->subjectColumnsBegin_incl;
        const int subjectColumnsEnd_excl = myMsaColumnProperties->subjectColumnsEnd_excl;

        const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
        const unsigned int* const subject = myAnchorSequenceData;
        const char* const subjectQualityScore = myAnchorQualityData;
                        
        for(int i = threadIdx.x; i < subjectLength; i += BLOCKSIZE){
            const int columnIndex = subjectColumnsBegin_incl + i;
            const unsigned int encbase = getEncodedNuc2Bit(subject, subjectLength, i);
            const float weight = canUseQualityScores ? getQualityWeight(subjectQualityScore[i]) : 1.0f;
            const int rowOffset = int(encbase) * msaColumnPitchInElements;

            mycounts[rowOffset + columnIndex] = 1;
            myweights[rowOffset + columnIndex] = weight;
            mycoverages[columnIndex] = 1;
        }
        __syncthreads();

        const int firstColumn_incl = myMsaColumnProperties->firstColumn_incl;
        const int lastColumn_excl = myMsaColumnProperties->lastColumn_excl;

        //const int numOuterIters = SDIV(lastColumn_excl, 4);

        //for each column in msa
        for(int column = threadIdx.x; column < lastColumn_excl; column += BLOCKSIZE){

            if(firstColumn_incl <= column){

                int countA = 0;
                int countC = 0;
                int countG = 0;
                int countT = 0;
                float weightA = 0;
                float weightC = 0;
                float weightG = 0;
                float weightT = 0;
                int coverage = 0;

                //for each candidate
                for(int indexInList = 0; indexInList < numIndices; indexInList += 1){
                    const int localCandidateIndex = myIndices[indexInList];
                    const int shift = myShifts[localCandidateIndex];
                    const int queryLength = myCandidateLengths[localCandidateIndex];

                    //if candidate occupies the column, update column
                    if(subjectColumnsBegin_incl + shift <= column && column < subjectColumnsBegin_incl + shift + queryLength){
                    
                        const BestAlignment_t flag = myAlignmentFlags[localCandidateIndex];
            
                        const unsigned int* const query = myTransposedCandidateSequencesData + localCandidateIndex;
            
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
            
                        //const int defaultcolumnoffset = subjectColumnsBegin_incl + shift;
            
                        const bool isForward = flag == BestAlignment_t::Forward;

                        
                        int positionInSequence = column - subjectColumnsBegin_incl - shift;
                        if(!isForward){
                            positionInSequence = queryLength - 1 - positionInSequence;
                        }                        

                        char quality = 'A';

                        auto getEncodedNucFromInt2Bit = [](unsigned int data, int pos){
                            return ((data >> (30 - 2*pos)) & 0x00000003);
                        };

                        const unsigned int currentDataInt = query[(positionInSequence / 16) * elementOffsetForTransposedCandidates];
                        unsigned int encodedBaseAsInt = getEncodedNucFromInt2Bit(currentDataInt, positionInSequence % 16);
                        if(!isForward){
                            //reverse complement
                            encodedBaseAsInt = (~encodedBaseAsInt & 0x00000003);
                        }
                        if(canUseQualityScores){
                            quality = queryQualityScore[positionInSequence];
                        }

                        
                        const float weight = canUseQualityScores ? getQualityWeight(quality) * overlapweight : overlapweight;

                        assert(weight != 0);

                        if(encodedBaseAsInt == 0){
                            countA++;
                            weightA += weight;
                        }else if(encodedBaseAsInt == 1){
                            countC++;
                            weightC += weight;
                        }else if(encodedBaseAsInt == 2){
                            countG++;
                            weightG += weight;
                        }else{
                            countT++;
                            weightT += weight;
                        }

                        coverage += 1;
                    }
                }

#if 0

                if(countA > 0){
                    mycounts[0 * msaColumnPitchInElements + column] += countA;
                }
                if(countC > 0){
                    mycounts[1 * msaColumnPitchInElements + column] += countC;
                }
                if(countG > 0){
                    mycounts[2 * msaColumnPitchInElements + column] += countG;
                }
                if(countT > 0){
                    mycounts[3 * msaColumnPitchInElements + column] += countT;
                }
                if(weightA > 0){
                    myweights[0 * msaColumnPitchInElements + column] += weightA;
                }
                if(weightC > 0){
                    myweights[1 * msaColumnPitchInElements + column] += weightC;
                }
                if(weightG > 0){
                    myweights[2 * msaColumnPitchInElements + column] += weightG;
                }
                if(weightT > 0){
                    myweights[3 * msaColumnPitchInElements + column] += weightT;
                }
                if(coverage > 0){
                    mycoverages[column] += coverage;
                }

#else 


                    mycounts[0 * msaColumnPitchInElements + column] += countA;
                    mycounts[1 * msaColumnPitchInElements + column] += countC;
                    mycounts[2 * msaColumnPitchInElements + column] += countG;
                    mycounts[3 * msaColumnPitchInElements + column] += countT;
                    myweights[0 * msaColumnPitchInElements + column] += weightA;
                    myweights[1 * msaColumnPitchInElements + column] += weightC;
                    myweights[2 * msaColumnPitchInElements + column] += weightG;
                    myweights[3 * msaColumnPitchInElements + column] += weightT;
                    mycoverages[column] += coverage;
#endif
            }
        }

    }








    template<int BLOCKSIZE, class Selector>
    __device__ __forceinline__
    void removeCandidatesFromMSASingleBlock(
            Selector shouldBeRemoved,
            int* __restrict__ inputcounts,
            float* __restrict__ inputweights,
            int* __restrict__ inputcoverages,
            const MSAColumnProperties* __restrict__ myMsaColumnProperties,
            const int* __restrict__ myShifts,
            const int* __restrict__ myOverlaps,
            const int* __restrict__ myNops,
            const BestAlignment_t* __restrict__ myAlignmentFlags,
            const unsigned int* __restrict__ myTransposedCandidateSequencesData,
            const char* __restrict__ myCandidateQualities,
            const int* __restrict__ myCandidateLengths,
            const int* __restrict__ myIndices,
            int numIndices,
            size_t elementOffsetForTransposedCandidates,
            bool canUseQualityScores, 
            size_t msaColumnPitchInElements,
            size_t encodedSequencePitchInInts,
            size_t qualityPitchInBytes,
            float desiredAlignmentMaxErrorRate,
            int subjectIndex){  

        constexpr bool candidatesAreTransposed = true;

        int* const mycounts = inputcounts;
        float* const myweights = inputweights;
        int* const mycoverages = inputcoverages;        

        const int subjectColumnsBegin_incl = myMsaColumnProperties->subjectColumnsBegin_incl;
        const int subjectColumnsEnd_excl = myMsaColumnProperties->subjectColumnsEnd_excl;
                        
        for(int indexInList = threadIdx.x; indexInList < numIndices; indexInList += BLOCKSIZE){

            if(shouldBeRemoved(indexInList)){

                const int localCandidateIndex = myIndices[indexInList];
                const int shift = myShifts[localCandidateIndex];
                const BestAlignment_t flag = myAlignmentFlags[localCandidateIndex];

                const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
                const int queryLength = myCandidateLengths[localCandidateIndex];
                const unsigned int* const query = myTransposedCandidateSequencesData + localCandidateIndex;

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

                msaAddOrDeleteASequence2Bit<candidatesAreTransposed, false>(
                    mycounts,
                    myweights,
                    mycoverages,
                    msaColumnPitchInElements,
                    query, 
                    queryLength, 
                    isForward,
                    defaultcolumnoffset,
                    overlapweight,
                    queryQualityScore,
                    canUseQualityScores,
                    elementOffsetForTransposedCandidates
                );
            }
        }
    }


    template<int BLOCKSIZE, class Selector>
    __device__ __forceinline__
    void removeCandidatesFromMSASingleBlockNotTransposed(
            Selector shouldBeRemoved,
            int* __restrict__ inputcounts,
            float* __restrict__ inputweights,
            int* __restrict__ inputcoverages,
            const MSAColumnProperties* __restrict__ myMsaColumnProperties,
            const int* __restrict__ myShifts,
            const int* __restrict__ myOverlaps,
            const int* __restrict__ myNops,
            const BestAlignment_t* __restrict__ myAlignmentFlags,
            const unsigned int* __restrict__ myCandidateSequencesData,
            const char* __restrict__ myCandidateQualities,
            const int* __restrict__ myCandidateLengths,
            const int* __restrict__ myIndices,
            int numIndices,
            size_t elementOffsetForTransposedCandidates,
            bool canUseQualityScores, 
            size_t msaColumnPitchInElements,
            size_t encodedSequencePitchInInts,
            size_t qualityPitchInBytes,
            float desiredAlignmentMaxErrorRate,
            int subjectIndex){  

        constexpr bool candidatesAreTransposed = true;

        int* const mycounts = inputcounts;
        float* const myweights = inputweights;
        int* const mycoverages = inputcoverages;        

        const int subjectColumnsBegin_incl = myMsaColumnProperties->subjectColumnsBegin_incl;
        const int subjectColumnsEnd_excl = myMsaColumnProperties->subjectColumnsEnd_excl;

        constexpr int threadsPerSequence = 8;
        auto tile = cg::tiled_partition<threadsPerSequence>(cg::this_thread_block());
        const int tileIdInBlock = threadIdx.x / threadsPerSequence;
        const int numTilesInBlock = blockDim.x / threadsPerSequence;
                        
        //for(int indexInList = threadIdx.x; indexInList < numIndices; indexInList += BLOCKSIZE){
        for(int indexInList = tileIdInBlock; indexInList < numIndices; indexInList += numTilesInBlock){

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

                msaAddOrDeleteASequence2Bit<candidatesAreTransposed, false>(
                    tile,
                    mycounts,
                    myweights,
                    mycoverages,
                    msaColumnPitchInElements,
                    query, 
                    queryLength, 
                    isForward,
                    defaultcolumnoffset,
                    overlapweight,
                    queryQualityScore,
                    canUseQualityScores,
                    1
                );
            }
        }
    }







    template<int BLOCKSIZE>
    __device__ __forceinline__
    void findConsensusSingleBlock(
            float* __restrict__ my_support,
            float* __restrict__ my_orig_weights,
            int* __restrict__ my_orig_coverage,
            char* __restrict__ my_consensus,
            const MSAColumnProperties* __restrict__ myMsaColumnProperties, 
            const int* __restrict__ myCounts,
            const float* __restrict__ myWeights,  
            const unsigned int* __restrict__ myAnchorSequenceData, 
            int subjectIndex,
            int encodedSequencePitchInInts, 
            size_t msaColumnPitchInElements){

        const int subjectColumnsBegin_incl = myMsaColumnProperties->subjectColumnsBegin_incl;
        const int subjectColumnsEnd_excl = myMsaColumnProperties->subjectColumnsEnd_excl;
        const int firstColumn_incl = myMsaColumnProperties->firstColumn_incl;
        const int lastColumn_excl = myMsaColumnProperties->lastColumn_excl;

        if(lastColumn_excl > msaColumnPitchInElements){
            if(threadIdx.x == 0){
                printf("%d, %d %lu\n", subjectIndex, lastColumn_excl, msaColumnPitchInElements);
            }
            __syncthreads();
        }
        assert(lastColumn_excl <= msaColumnPitchInElements);

        const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
        const unsigned int* const subject = myAnchorSequenceData;

        //set columns to zero which are not in range firstColumn_incl <= column && column < lastColumn_excl

        for(int column = threadIdx.x; 
                column < firstColumn_incl; 
                column += BLOCKSIZE){

            my_support[column] = 0;
            my_orig_weights[column] = 0;
            my_orig_coverage[column] = 0;
        }

        const int leftoverRight = msaColumnPitchInElements - lastColumn_excl;

        for(int i = threadIdx.x; i < leftoverRight; i += BLOCKSIZE){
            const int column = lastColumn_excl + i;

            my_support[column] = 0;
            my_orig_weights[column] = 0;
            my_orig_coverage[column] = 0;
        }

        for(int column = threadIdx.x; 
            column < firstColumn_incl; 
            column += BLOCKSIZE){
                
            my_consensus[column] = 5;
        }

        for(int i = threadIdx.x; i < leftoverRight; i += BLOCKSIZE){
            const int column = lastColumn_excl + i;

            my_consensus[column] = 5;
        }

        const int* const myCountsA = myCounts + 0 * msaColumnPitchInElements;
        const int* const myCountsC = myCounts + 1 * msaColumnPitchInElements;
        const int* const myCountsG = myCounts + 2 * msaColumnPitchInElements;
        const int* const myCountsT = myCounts + 3 * msaColumnPitchInElements;

        const float* const my_weightsA = myWeights + 0 * msaColumnPitchInElements;
        const float* const my_weightsC = myWeights + 1 * msaColumnPitchInElements;
        const float* const my_weightsG = myWeights + 2 * msaColumnPitchInElements;
        const float* const my_weightsT = myWeights + 3 * msaColumnPitchInElements;
#if 1
        const int numOuterIters = SDIV(lastColumn_excl, 4);

        // auto getEncodedNucFromInt2Bit = [](unsigned int data, int pos){
        //     return ((data >> (30 - 2*pos)) & 0x00000003);
        // };

#if 1
        for(int outerIter = threadIdx.x; outerIter < numOuterIters; outerIter += BLOCKSIZE){

            // const int pos1 = (outerIter * 4) - subjectColumnsBegin_incl);
            // const int pos2 = (outerIter * 4 + 3) - subjectColumnsBegin_incl);
            // const int intIndex1InEncodedSequence = (max(0, pos1/ 16);
            // const int intIndex2InEncodedSequence = (max(0, pos2/ 16);
            // const unsigned int encoded1 = subject[intIndex1InEncodedSequence];
            // const unsigned int encoded2 = subject[intIndex2InEncodedSequence];
            // const unsigned int encodedBases16 = encoded1;

            alignas(4) char consensusArray[4];

            #pragma unroll 
            for(int i = 0; i < 4; i++){
                const int column = outerIter * 4 + i;

                if(firstColumn_incl <= column && column < lastColumn_excl){

                    const int ca = myCountsA[column];
                    const int cc = myCountsC[column];
                    const int cg = myCountsG[column];
                    const int ct = myCountsT[column];
                    const float wa = my_weightsA[column];
                    const float wc = my_weightsC[column];
                    const float wg = my_weightsG[column];
                    const float wt = my_weightsT[column];

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
                    //assert(weightPerCountSum != 0);
                    my_support[column] = consWeight / columnWeight;
                    //my_support[column] = consWeightPerCount / weightPerCountSum;


                    if(subjectColumnsBegin_incl <= column && column < subjectColumnsEnd_excl){
                        constexpr unsigned int A_enc = 0x00;
                        constexpr unsigned int C_enc = 0x01;
                        constexpr unsigned int G_enc = 0x02;
                        constexpr unsigned int T_enc = 0x03;

                        const int localIndex = column - subjectColumnsBegin_incl;
                        const unsigned int encNuc = getEncodedNuc2Bit(subject, subjectLength, localIndex);

                        // const unsigned int encNuc2 = getEncodedNucFromInt2Bit(encodedBases16, localIndex % 16);

                        // if(intIndexInEncodedSequence != localIndex / 16){
                        //     printf("outerIter %d, i %d, intIndexInEncodedSequence %d, localIndex %d, column %d, "
                        //         "subjectColumnsBegin_incl %d\n", 
                        //         outerIter, i, intIndexInEncodedSequence, localIndex, column, subjectColumnsBegin_incl);
                        // }
                        // assert(intIndexInEncodedSequence == localIndex / 16);

                        // assert(encNuc == encNuc2);

                        if(encNuc == A_enc){
                            my_orig_weights[column] = wa;
                            my_orig_coverage[column] = ca;
                        }else if(encNuc == C_enc){
                            my_orig_weights[column] = wc;
                            my_orig_coverage[column] = cc;
                        }else if(encNuc == G_enc){
                            my_orig_weights[column] = wg;
                            my_orig_coverage[column] = cg;
                        }else if(encNuc == T_enc){
                            my_orig_weights[column] = wt;
                            my_orig_coverage[column] = ct;
                        }
                    }
                }
            }

            *((char4*)(my_consensus + 4*outerIter)) = *((const char4*)(&consensusArray[0]));
        }

#else        
        for(int outerIter = threadIdx.x; outerIter < numOuterIters; outerIter += BLOCKSIZE){

            int regCountsA[4];
            int regCountsC[4];
            int regCountsG[4];
            int regCountsT[4];
            float regWeightsA[4];
            float regWeightsC[4];
            float regWeightsG[4];
            float regWeightsT[4];

            *((int4*)&regCountsA[0]) = *((const int4*)(myCountsA + 4 * outerIter));
            *((int4*)&regCountsC[0]) = *((const int4*)(myCountsC + 4 * outerIter));
            *((int4*)&regCountsG[0]) = *((const int4*)(myCountsG + 4 * outerIter));
            *((int4*)&regCountsT[0]) = *((const int4*)(myCountsT + 4 * outerIter));
            *((float4*)&regWeightsA[0]) = *((const float4*)(my_weightsA + 4 * outerIter));
            *((float4*)&regWeightsC[0]) = *((const float4*)(my_weightsC + 4 * outerIter));
            *((float4*)&regWeightsG[0]) = *((const float4*)(my_weightsG + 4 * outerIter));
            *((float4*)&regWeightsT[0]) = *((const float4*)(my_weightsT + 4 * outerIter));



            #pragma unroll 
            for(int i = 0; i < 4; i++){
                const int column = outerIter * 4 + i;

                if(firstColumn_incl <= column && column < lastColumn_excl){

                    // const int ca = myCountsA[column];
                    // const int cc = myCountsC[column];
                    // const int cg = myCountsG[column];
                    // const int ct = myCountsT[column];
                    // const float wa = my_weightsA[column];
                    // const float wc = my_weightsC[column];
                    // const float wg = my_weightsG[column];
                    // const float wt = my_weightsT[column];

                    const int ca = regCountsA[i];
                    const int cc = regCountsC[i];
                    const int cg = regCountsG[i];
                    const int ct = regCountsT[i];
                    const float wa = regWeightsA[i];
                    const float wc = regWeightsC[i];
                    const float wg = regWeightsG[i];
                    const float wt = regWeightsT[i];

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
                    my_consensus[column] = cons;
                    const float columnWeight = wa + wc + wg + wt;
                    if(columnWeight == 0){
                        printf("s %d c %d\n", subjectIndex, column);
                        assert(columnWeight != 0);
                    }
                    //assert(weightPerCountSum != 0);
                    my_support[column] = consWeight / columnWeight;
                    //my_support[column] = consWeightPerCount / weightPerCountSum;


                    if(subjectColumnsBegin_incl <= column && column < subjectColumnsEnd_excl){
                        constexpr unsigned int A_enc = 0x00;
                        constexpr unsigned int C_enc = 0x01;
                        constexpr unsigned int G_enc = 0x02;
                        constexpr unsigned int T_enc = 0x03;

                        const int localIndex = column - subjectColumnsBegin_incl;
                        const unsigned int encNuc = getEncodedNuc2Bit(subject, subjectLength, localIndex);

                        if(encNuc == A_enc){
                            my_orig_weights[column] = wa;
                            my_orig_coverage[column] = ca;
                        }else if(encNuc == C_enc){
                            my_orig_weights[column] = wc;
                            my_orig_coverage[column] = cc;
                        }else if(encNuc == G_enc){
                            my_orig_weights[column] = wg;
                            my_orig_coverage[column] = cg;
                        }else if(encNuc == T_enc){
                            my_orig_weights[column] = wt;
                            my_orig_coverage[column] = ct;
                        }
                    }
                }
            }
        }

#endif

#else 

        const int numColumnsToCheck = lastColumn_excl - firstColumn_incl;

        for(int i = threadIdx.x; i < numColumnsToCheck; i += BLOCKSIZE){
            const int column = firstColumn_incl + i;

            const int ca = myCountsA[column];
            const int cc = myCountsC[column];
            const int cg = myCountsG[column];
            const int ct = myCountsT[column];
            const float wa = my_weightsA[column];
            const float wc = my_weightsC[column];
            const float wg = my_weightsG[column];
            const float wt = my_weightsT[column];

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
            my_consensus[column] = cons;
            const float columnWeight = wa + wc + wg + wt;
            if(columnWeight == 0){
                printf("s %d c %d\n", subjectIndex, column);
                assert(columnWeight != 0);
            }
            //assert(weightPerCountSum != 0);
            my_support[column] = consWeight / columnWeight;
            //my_support[column] = consWeightPerCount / weightPerCountSum;


            if(subjectColumnsBegin_incl <= column && column < subjectColumnsEnd_excl){
                constexpr unsigned int A_enc = 0x00;
                constexpr unsigned int C_enc = 0x01;
                constexpr unsigned int G_enc = 0x02;
                constexpr unsigned int T_enc = 0x03;

                const int localIndex = column - subjectColumnsBegin_incl;
                const unsigned int encNuc = getEncodedNuc2Bit(subject, subjectLength, localIndex);

                if(encNuc == A_enc){
                    my_orig_weights[column] = wa;
                    my_orig_coverage[column] = ca;
                }else if(encNuc == C_enc){
                    my_orig_weights[column] = wc;
                    my_orig_coverage[column] = cc;
                }else if(encNuc == G_enc){
                    my_orig_weights[column] = wg;
                    my_orig_coverage[column] = cg;
                }else if(encNuc == T_enc){
                    my_orig_weights[column] = wt;
                    my_orig_coverage[column] = ct;
                }
            }
        }

#endif        
    }



    template<int BLOCKSIZE>
    __device__ __forceinline__
    void findCandidatesOfDifferentRegionSingleBlock(
            int2* smem,
            int* __restrict__ myNewIndicesPtr,
            int* __restrict__ myNewNumIndicesPerSubjectPtr,
            const MSAColumnProperties* __restrict__ myMsaColumnProperties,
            const char* __restrict__ myConsensus,
            const int* __restrict__ myCounts,
            const float* __restrict__ myWeights,
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
            size_t msaColumnPitchInElements,
            const int* __restrict__ myIndices,
            const int myNumIndices,
            int dataset_coverage){

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

        using BlockReduceBool = cub::BlockReduce<bool, BLOCKSIZE>;
        using BlockReduceInt2 = cub::BlockReduce<int2, BLOCKSIZE>;

        typename BlockReduceBool::TempStorage* temp_storage_boolreduce = (typename BlockReduceBool::TempStorage*) smem;
        typename BlockReduceInt2::TempStorage* temp_storage_int2reduce = (typename BlockReduceInt2::TempStorage*) smem;

        __shared__ bool broadcastbufferbool;
        __shared__ int broadcastbufferint4[4];
        __shared__ int counts[1];

        const unsigned int* const subjectptr = myAnchorSequenceData;

        const int subjectColumnsBegin_incl = myMsaColumnProperties->subjectColumnsBegin_incl;
        const int subjectColumnsEnd_excl = myMsaColumnProperties->subjectColumnsEnd_excl;

        //check if subject and consensus differ at at least one position

        bool hasMismatchToConsensus = false;

        for(int pos = threadIdx.x; pos < subjectLength && !hasMismatchToConsensus; pos += BLOCKSIZE){
            const int column = subjectColumnsBegin_incl + pos;
            const char consbase = myConsensus[column];
            const char subjectbase = getEncodedNuc2Bit(subjectptr, subjectLength, pos);

            hasMismatchToConsensus |= (consbase != subjectbase);
        }

        hasMismatchToConsensus = 
            BlockReduceBool(*temp_storage_boolreduce)
                .Reduce(hasMismatchToConsensus, [](auto l, auto r){return l || r;});

        if(threadIdx.x == 0){
            broadcastbufferbool = hasMismatchToConsensus;
        }
        __syncthreads();

        hasMismatchToConsensus = broadcastbufferbool;

        //if subject and consensus differ at at least one position, check columns in msa

        if(hasMismatchToConsensus){
            int col = std::numeric_limits<int>::max();
            bool foundColumn = false;
            char foundBase = 'F';
            int foundBaseIndex = std::numeric_limits<int>::max();
            int consindex = std::numeric_limits<int>::max();

            const int* const myCountsA = myCounts + 0 * msaColumnPitchInElements;
            const int* const myCountsC = myCounts + 1 * msaColumnPitchInElements;
            const int* const myCountsG = myCounts + 2 * msaColumnPitchInElements;
            const int* const myCountsT = myCounts + 3 * msaColumnPitchInElements;

            for(int columnindex = subjectColumnsBegin_incl + threadIdx.x; 
                    columnindex < subjectColumnsEnd_excl && !foundColumn; 
                    columnindex += BLOCKSIZE){

                int counts[4];
                counts[0] = myCountsA[columnindex];
                counts[1] = myCountsC[columnindex];
                counts[2] = myCountsG[columnindex];
                counts[3] = myCountsT[columnindex];

                const char consbase = myConsensus[columnindex];
                consindex = consbase;

                assert(0 <= consindex && consindex < 4);

                //find out if there is a non-consensus base with significant coverage
                int significantBaseIndex = -1;

                #pragma unroll
                for(int i = 0; i < 4; i++){
                    if(i != consindex){
                        const bool significant = is_significant_count(counts[i], dataset_coverage);

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
            packed = BlockReduceInt2(*temp_storage_int2reduce).Reduce(packed, [](auto l, auto r){
                if(l.x < r.x){
                    return l;
                }else{
                    return r;
                }
            });

            if(threadIdx.x == 0){
                if(packed.x != std::numeric_limits<int>::max()){
                    broadcastbufferint4[0] = 1;
                    broadcastbufferint4[1] = packed.x;
                    broadcastbufferint4[2] = packed.y;
                    broadcastbufferint4[3] = packed.y;
                }else{
                    broadcastbufferint4[0] = 0;
                }
            }

            __syncthreads();

            foundColumn = (1 == broadcastbufferint4[0]);
            col = broadcastbufferint4[1];
            foundBase = broadcastbufferint4[2];
            foundBaseIndex = broadcastbufferint4[3];

            if(foundColumn){
                
                auto discard_rows = [&](bool keepMatching){
                    
                    for(int k = threadIdx.x; k < myNumIndices; k += BLOCKSIZE){
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
                    for(int k = threadIdx.x; k < myNumIndices && !veryGoodAlignment; k += BLOCKSIZE){
                        if(!myShouldBeKept[+ k]){
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

                    veryGoodAlignment = BlockReduceBool(*temp_storage_boolreduce)
                        .Reduce(veryGoodAlignment, [](auto l, auto r){return l || r;});

                    if(threadIdx.x == 0){
                        broadcastbufferbool = veryGoodAlignment;
                    }
                    __syncthreads();

                    veryGoodAlignment = broadcastbufferbool;

                    if(veryGoodAlignment){
                        for(int k = threadIdx.x; k < myNumIndices; k += blockDim.x){
                            myShouldBeKept[k] = true;
                        }
                    }
                    #endif

                    //select indices of candidates to keep and write them to new indices
                    if(threadIdx.x == 0){
                        counts[0] = 0;
                    }
                    __syncthreads();

                    const int limit = SDIV(myNumIndices, BLOCKSIZE) * BLOCKSIZE;
                    for(int k = threadIdx.x; k < limit; k += BLOCKSIZE){
                        bool keep = false;
                        if(k < myNumIndices){
                            keep = myShouldBeKept[k];
                        }                               
            
                        if(keep){
                            cg::coalesced_group g = cg::coalesced_threads();
                            int outputPos;
                            if (g.thread_rank() == 0) {
                                outputPos = atomicAdd(&counts[0], g.size());
                            }
                            outputPos = g.thread_rank() + g.shfl(outputPos, 0);
                            myNewIndicesPtr[outputPos] = myIndices[k];
                        }                        
                    }

                    __syncthreads();

                    if(threadIdx.x == 0){
                        *myNewNumIndicesPerSubjectPtr = counts[0];
                    }

                    __syncthreads();

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
                for(int k = threadIdx.x; k < myNumIndices; k += blockDim.x){
                    myNewIndicesPtr[k] = myIndices[k];
                }
                if(threadIdx.x == 0){
                    *myNewNumIndicesPerSubjectPtr = myNumIndices;
                }
            }

        }else{
            //no mismatch between consensus and subject

            //remove no candidate
            for(int k = threadIdx.x; k < myNumIndices; k += blockDim.x){
                myShouldBeKept[k] = true;
            }

            for(int k = threadIdx.x; k < myNumIndices; k += blockDim.x){
                myNewIndicesPtr[k] = myIndices[k];
            }
            if(threadIdx.x == 0){
                *myNewNumIndicesPerSubjectPtr = myNumIndices;
            }
        }
    }

  


    #ifdef __CUDACC_DEBUG__

        #define buildMSA3Kernel_MIN_BLOCKS   1
        
    #else

        #if __CUDA_ARCH__ >= 610
            #define buildMSA3Kernel_MIN_BLOCKS   8
        #else
            #define buildMSA3Kernel_MIN_BLOCKS   4
        #endif

    #endif



    template<int BLOCKSIZE, MemoryType addSequencesMemType>
    __launch_bounds__(BLOCKSIZE, buildMSA3Kernel_MIN_BLOCKS)
    __global__
    void buildMSA3Kernel(
            MSAColumnProperties* __restrict__ msaColumnProperties,
            int* __restrict__ coverage,
            int* __restrict__ counts,
            float* __restrict__ weights,
            float* __restrict__ d_support,
            float* __restrict__ d_origWeights,
            int* __restrict__ d_origCoverages,
            char* __restrict__ d_consensus,          
            const int* __restrict__ overlaps,
            const int* __restrict__ shifts,
            const int* __restrict__ nOps,
            const BestAlignment_t* __restrict__ bestAlignmentFlags,
            const int* __restrict__ anchorLengths,
            const int* __restrict__ candidateLengths,
            const int* __restrict__ indices,
            const int* __restrict__ indices_per_subject,
            const int* __restrict__ candidatesPerSubjectPrefixSum,            
            const unsigned int* __restrict__ subjectSequencesData,
            const unsigned int* __restrict__ candidateSequencesData,
            const char* __restrict__ subjectQualities,
            const char* __restrict__ candidateQualities,
            const int* __restrict__ d_numAnchors,
            const int*,
            float desiredAlignmentMaxErrorRate,
            bool canUseQualityScores,
            int encodedSequencePitchInInts,
            size_t qualityPitchInBytes,
            size_t msaColumnPitchInElements,
            const bool* __restrict__ canExecute){

        constexpr bool useSmemForAddSequences = (addSequencesMemType == MemoryType::Shared);

        extern __shared__ float sharedmem[];
        __shared__ MSAColumnProperties shared_columnProperties;

        using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;            

        if(*canExecute){

            auto tbGroup = cg::this_thread_block();

            const int n_subjects = *d_numAnchors;

            typename BlockReduceInt::TempStorage* const cubTempStorage = (typename BlockReduceInt::TempStorage*)sharedmem;

            float* const shared_weights = sharedmem;
            int* const shared_counts = (int*)(shared_weights + 4 * msaColumnPitchInElements);
            int* const shared_coverages = (int*)(shared_counts + 4 * msaColumnPitchInElements);

            GPUMultiMSA multiMSA;

            multiMSA.numMSAs = n_subjects;
            multiMSA.columnPitchInElements = msaColumnPitchInElements;
            multiMSA.counts = counts;
            multiMSA.weights = weights;
            multiMSA.coverages = coverage;
            multiMSA.consensus = d_consensus;
            multiMSA.support = d_support;
            multiMSA.origWeights = d_origWeights;
            multiMSA.origCoverages = d_origCoverages;
            multiMSA.columnProperties = msaColumnProperties;

            for(int subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
                const int myNumGoodCandidates = indices_per_subject[subjectIndex];

                if(myNumGoodCandidates > 0){

                    tbGroup.sync(); //wait for smem of previous iteration

                    GpuSingleMSA msa = multiMSA.getSingleMSA(subjectIndex);

                    if(useSmemForAddSequences){
                        msa.counts = shared_counts;
                        msa.weights = shared_weights;
                        msa.coverages = shared_coverages;
                    }

                    const int globalCandidateOffset = candidatesPerSubjectPrefixSum[subjectIndex];

                    const int* const myOverlaps = overlaps + globalCandidateOffset;
                    const int* const myShifts = shifts + globalCandidateOffset;
                    const int* const myNops = nOps + globalCandidateOffset;
                    const BestAlignment_t* const myAlignmentFlags = bestAlignmentFlags + globalCandidateOffset;
                    const int subjectLength = anchorLengths[subjectIndex];
                    const int* const myCandidateLengths = candidateLengths + globalCandidateOffset;
                    const int* const myIndices = indices + globalCandidateOffset;

                    const unsigned int* const myAnchorSequenceData = subjectSequencesData + std::size_t(subjectIndex) * encodedSequencePitchInInts;
                    const unsigned int* const myCandidateSequencesData = candidateSequencesData + size_t(globalCandidateOffset) * encodedSequencePitchInInts;
                    const char* const myAnchorQualityData = subjectQualities + std::size_t(subjectIndex) * qualityPitchInBytes;
                    const char* const myCandidateQualities = candidateQualities + size_t(globalCandidateOffset) * qualityPitchInBytes;

                    MSAColumnProperties columnProperties;

                    msa.columnProperties = &columnProperties;

                    auto groupReduceIntMin = [&](int data){
                        data = BlockReduceInt(*cubTempStorage).Reduce(data, cub::Min());
                        tbGroup.sync();
                        return data;
                    };

                    auto groupReduceIntMax = [&](int data){                        
                        data = BlockReduceInt(*cubTempStorage).Reduce(data, cub::Max());
                        tbGroup.sync();
                        return data;
                    };

                    msa.initColumnProperties(
                        tbGroup,
                        groupReduceIntMin,
                        groupReduceIntMax,
                        myIndices,
                        myNumGoodCandidates,
                        myShifts,
                        myAlignmentFlags,
                        subjectLength,
                        myCandidateLengths
                    );

                    //only thread 0 has valid column properties. 
                    //save to global memory and broadcast to all threads in block
                    if(tbGroup.thread_rank() == 0){
                        auto* globalDest = multiMSA.getColumnPropertiesOfMSA(subjectIndex);
                        *globalDest = columnProperties;
                        shared_columnProperties = columnProperties;
                    }

                    tbGroup.sync();

                    columnProperties = shared_columnProperties;

                    const int columnsToCheck = columnProperties.lastColumn_excl;

                    assert(columnsToCheck <= msaColumnPitchInElements);

                    msa.constructFromSequences(
                        tbGroup,
                        myShifts,
                        myOverlaps,
                        myNops,
                        myAlignmentFlags,
                        myAnchorSequenceData,
                        myAnchorQualityData,
                        myCandidateSequencesData,
                        myCandidateQualities,
                        myCandidateLengths,
                        myIndices,
                        myNumGoodCandidates,
                        canUseQualityScores, 
                        encodedSequencePitchInInts,
                        qualityPitchInBytes,
                        desiredAlignmentMaxErrorRate,
                        subjectIndex
                    );

                    tbGroup.sync();
            
                    msa.checkAfterBuild(tbGroup, subjectIndex);

                    msa.findConsensus(
                        tbGroup,
                        myAnchorSequenceData, 
                        encodedSequencePitchInInts, 
                        subjectIndex
                    );

                    if(useSmemForAddSequences){
                        // copy from counts and weights and coverages from shared to global
                        int* const gmemCounts = multiMSA.getCountsOfMSA(subjectIndex);
                        float* const gmemWeights = multiMSA.getWeightsOfMSA(subjectIndex);
                        int* const gmemCoverages = multiMSA.getCoveragesOfMSA(subjectIndex);

                        for(int index = tbGroup.thread_rank(); index < columnsToCheck; index += tbGroup.size()){
                            for(int k = 0; k < 4; k++){
                                const int* const srcCounts = msa.counts + k * msaColumnPitchInElements + index;
                                int* const destCounts = gmemCounts + k * msaColumnPitchInElements + index;
            
                                const float* const srcWeights = msa.weights + k * msaColumnPitchInElements + index;
                                float* const destWeights = gmemWeights + k * msaColumnPitchInElements + index;
            
                                *destCounts = *srcCounts;
                                *destWeights = *srcWeights;
                            }
                            gmemCoverages[index] = msa.coverages[index];
                        }
                    }
                } 
            }
        }

    }





    template<int BLOCKSIZE, MemoryType addSequencesMemType>
    __global__
    void msa_findCandidatesOfDifferentRegionAndRemoveThemViaRebuild_kernel(
            int* __restrict__ d_newIndices,
            int* __restrict__ d_newNumIndicesPerSubject,
            int* __restrict__ d_newNumIndices,
            MSAColumnProperties* __restrict__ msaColumnProperties,
            char* __restrict__ consensus,
            int* __restrict__ coverage,
            int* __restrict__ counts,
            float* __restrict__ weights,
            float* __restrict__ support,
            int* __restrict__ origCoverages,
            float* __restrict__ origWeights,
            const BestAlignment_t* __restrict__ bestAlignmentFlags,
            const int* __restrict__ shifts,
            const int* __restrict__ nOps,
            const int* __restrict__ overlaps,
            const unsigned int* __restrict__ subjectSequencesData,
            const unsigned int* __restrict__ candidateSequencesData,
            const unsigned int* __restrict__ transposedCandidateSequencesData,
            const int* __restrict__ subjectSequencesLength,
            const int* __restrict__ candidateSequencesLength,
            const char* __restrict__ subjectQualities,
            const char* __restrict__ candidateQualities,
            bool* __restrict__ d_shouldBeKept,
            const int* __restrict__ d_candidates_per_subject_prefixsum,
            const int* __restrict__ d_numAnchors,
            const int*,
            float desiredAlignmentMaxErrorRate,
            bool canUseQualityScores,
            size_t encodedSequencePitchInInts,
            size_t qualityPitchInBytes,
            size_t msaColumnPitchInElements,
            const int* __restrict__ d_indices,
            const int* __restrict__ d_indices_per_subject,
            int dataset_coverage,
            const bool* __restrict__ canExecute,
            int iteration,
            bool* __restrict__ d_anchorIsFinished){

        if(*canExecute){

            using BlockReduceBool = cub::BlockReduce<bool, BLOCKSIZE>;
            using BlockReduceInt2 = cub::BlockReduce<int2, BLOCKSIZE>;
            using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;

            __shared__ union{
                typename BlockReduceBool::TempStorage boolreduce;
                typename BlockReduceInt2::TempStorage int2reduce;
                typename BlockReduceInt::TempStorage intreduce;
            } temp_storage;

            __shared__ MSAColumnProperties shared_columnProperties;

            extern __shared__ float externsharedmem[];

            constexpr bool useSmemMSA = (addSequencesMemType == MemoryType::Shared);

            float* const shared_weights = externsharedmem;
            int* const shared_counts = (int*)(shared_weights + 4 * msaColumnPitchInElements);
            int* const shared_coverages = (int*)(shared_counts + 4 * msaColumnPitchInElements);

            const int n_subjects = *d_numAnchors;

            GPUMultiMSA multiMSA;

            multiMSA.numMSAs = n_subjects;
            multiMSA.columnPitchInElements = msaColumnPitchInElements;
            multiMSA.counts = counts;
            multiMSA.weights = weights;
            multiMSA.coverages = coverage;
            multiMSA.consensus = consensus;
            multiMSA.support = support;
            multiMSA.origWeights = origWeights;
            multiMSA.origCoverages = origCoverages;
            multiMSA.columnProperties = msaColumnProperties;

            auto tbGroup = cg::this_thread_block();

            for(int subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
                const bool myAnchorIsFinished = d_anchorIsFinished[subjectIndex];
                const int myNumIndices = d_indices_per_subject[subjectIndex];

                if(myAnchorIsFinished){
                    if(threadIdx.x == 0){
                        atomicAdd(d_newNumIndices, myNumIndices);
                    }
                }else{               

                    if(myNumIndices > 0){
                        GpuSingleMSA msa = multiMSA.getSingleMSA(subjectIndex);
                        msa.columnProperties = &shared_columnProperties;

                        tbGroup.sync(); //wait for previous iteration

                        if(threadIdx.x == 0){
                            shared_columnProperties = msaColumnProperties[subjectIndex];
                        }
                        tbGroup.sync();

                        const int globalOffset = d_candidates_per_subject_prefixsum[subjectIndex];
                        const int* const myIndices = d_indices + globalOffset;
                        const int* const myNumIndicesPerSubjectPtr = d_indices_per_subject + subjectIndex;

                        int* const myNewIndicesPtr = d_newIndices + globalOffset;
                        int* const myNewNumIndicesPerSubjectPtr = d_newNumIndicesPerSubject + subjectIndex;

                        bool* const myShouldBeKept = d_shouldBeKept + globalOffset;                    

                        const BestAlignment_t* const myAlignmentFlags = bestAlignmentFlags + globalOffset;
                        const int* const myShifts = shifts + globalOffset;
                        const int* const myNops = nOps + globalOffset;
                        const int* const myOverlaps = overlaps + globalOffset;

                        const unsigned int* const myAnchorSequenceData = subjectSequencesData 
                            + std::size_t(subjectIndex) * encodedSequencePitchInInts;
                        const unsigned int* const myCandidateSequencesData = candidateSequencesData 
                            + std::size_t(globalOffset) * encodedSequencePitchInInts;

                        const char* const myAnchorQualityData = subjectQualities + std::size_t(subjectIndex) * qualityPitchInBytes;
                        const char* const myCandidateQualities = candidateQualities 
                            + size_t(globalOffset) * qualityPitchInBytes;

                        const int subjectLength = subjectSequencesLength[subjectIndex];
                        const int* const myCandidateLengths = candidateSequencesLength + globalOffset;

                        const int* const srcIndices = myIndices;
                        int* const destIndices = myNewIndicesPtr;

                        const int* const srcNumIndices = myNumIndicesPerSubjectPtr;
                        int* const destNumIndices = myNewNumIndicesPerSubjectPtr;

                        auto groupReduceBool = [&](bool b, auto comp){
                            b = BlockReduceBool(temp_storage.boolreduce).Reduce(b, comp);
                            return b;
                        };

                        auto groupReduceInt2 = [&](int2 b, auto comp){
                            b = BlockReduceInt2(temp_storage.int2reduce).Reduce(b, comp);
                            return b;
                        };

                        msa.flagCandidatesOfDifferentRegion(
                            tbGroup,
                            groupReduceBool,
                            groupReduceInt2,
                            destIndices,
                            destNumIndices,
                            myAnchorSequenceData,
                            subjectLength,
                            myCandidateSequencesData,
                            myCandidateLengths,
                            myAlignmentFlags,
                            myShifts,
                            myNops,
                            myOverlaps,
                            myShouldBeKept,
                            desiredAlignmentMaxErrorRate,
                            subjectIndex,
                            encodedSequencePitchInInts,
                            srcIndices,
                            *srcNumIndices,
                            dataset_coverage
                        );

                        tbGroup.sync();

                        const int myNewNumIndices = *destNumIndices;

                        if(tbGroup.thread_rank()== 0){
                            atomicAdd(d_newNumIndices, myNewNumIndices);
                        }

                        assert(myNewNumIndices <= myNumIndices);
                        if(myNewNumIndices > 0 && myNewNumIndices < myNumIndices){

                            if(useSmemMSA){
                                msa.counts = shared_counts;
                                msa.weights = shared_weights;
                                msa.coverages = shared_coverages;
                            }

                            auto groupReduceIntMin = [&](int data){
                                data = BlockReduceInt(temp_storage.intreduce).Reduce(data, cub::Min());
                                tbGroup.sync();
                                return data;
                            };
        
                            auto groupReduceIntMax = [&](int data){                        
                                data = BlockReduceInt(temp_storage.intreduce).Reduce(data, cub::Max());
                                tbGroup.sync();
                                return data;
                            };
        
                            msa.initColumnProperties(
                                tbGroup,
                                groupReduceIntMin,
                                groupReduceIntMax,
                                destIndices,
                                *destNumIndices,
                                myShifts,
                                myAlignmentFlags,
                                subjectLength,
                                myCandidateLengths
                            );

                            //only thread 0 has valid column properties. save to global memory and broadcast to all threads in block
                            if(tbGroup.thread_rank() == 0){
                                msaColumnProperties[subjectIndex] = shared_columnProperties;
                            }

                            tbGroup.sync();

                            const int columnsToCheck = shared_columnProperties.lastColumn_excl;

                            assert(columnsToCheck <= msaColumnPitchInElements);

                            msa.constructFromSequences(
                                tbGroup,
                                myShifts,
                                myOverlaps,
                                myNops,
                                myAlignmentFlags,
                                myAnchorSequenceData,
                                myAnchorQualityData,
                                myCandidateSequencesData,
                                myCandidateQualities,
                                myCandidateLengths,
                                destIndices,
                                *destNumIndices,
                                canUseQualityScores, 
                                encodedSequencePitchInInts,
                                qualityPitchInBytes,
                                desiredAlignmentMaxErrorRate,
                                subjectIndex
                            );

                            tbGroup.sync();
                    
                            msa.checkAfterBuild(tbGroup, subjectIndex);

                            msa.findConsensus(
                                tbGroup,
                                myAnchorSequenceData, 
                                encodedSequencePitchInInts, 
                                subjectIndex
                            );

                            tbGroup.sync();

                            if(useSmemMSA){
                                // copy from counts and weights and coverages from shared to global
                                int* const gmemCounts = multiMSA.getCountsOfMSA(subjectIndex);
                                float* const gmemWeights = multiMSA.getWeightsOfMSA(subjectIndex);
                                int* const gmemCoverages = multiMSA.getCoveragesOfMSA(subjectIndex);
        
                                for(int index = tbGroup.thread_rank(); index < columnsToCheck; index += tbGroup.size()){
                                    for(int k = 0; k < 4; k++){
                                        const int* const srcCounts = msa.counts + k * msaColumnPitchInElements + index;
                                        int* const destCounts = gmemCounts + k * msaColumnPitchInElements + index;
                    
                                        const float* const srcWeights = msa.weights + k * msaColumnPitchInElements + index;
                                        float* const destWeights = gmemWeights + k * msaColumnPitchInElements + index;
                    
                                        *destCounts = *srcCounts;
                                        *destWeights = *srcWeights;
                                    }
                                    gmemCoverages[index] = msa.coverages[index];
                                }

                                tbGroup.sync();
                            }

                            
                    
                        }else{
                            if(threadIdx.x == 0){
                                d_anchorIsFinished[subjectIndex] = true;
                            }
                        }
                    }else{
                        if(threadIdx.x == 0){
                            d_newNumIndicesPerSubject[subjectIndex] = 0;
                            d_anchorIsFinished[subjectIndex] = true;
                        }
                        ; //nothing else to do if there are no candidates in msa
                    }
                }
            }
        }
    }



    #ifdef __CUDACC_DEBUG__

        #define findCandidatesOfDifferentRegionAndRemoveThemViaDeletion2_multiiter_MIN_BLOCKS   1
        
    #else

        #if __CUDA_ARCH__ >= 610
            #define findCandidatesOfDifferentRegionAndRemoveThemViaDeletion2_multiiter_MIN_BLOCKS   4
        #else
            #define findCandidatesOfDifferentRegionAndRemoveThemViaDeletion2_multiiter_MIN_BLOCKS   4
        #endif

    #endif


    template<int BLOCKSIZE, MemoryType memoryType>
    __launch_bounds__(BLOCKSIZE, findCandidatesOfDifferentRegionAndRemoveThemViaDeletion2_multiiter_MIN_BLOCKS)
    __global__
    void msa_findCandidatesOfDifferentRegionAndRemoveThemViaDeletion2_multiiteration_kernel(
            int* __restrict__ d_newIndices,
            int* __restrict__ d_newNumIndicesPerSubject,
            int* __restrict__ d_newNumIndices,
            MSAColumnProperties* __restrict__ msaColumnProperties,
            char* __restrict__ consensus,
            int* __restrict__ coverage,
            int* __restrict__ counts,
            float* __restrict__ weights,
            float* __restrict__ support,
            int* __restrict__ origCoverages,
            float* __restrict__ origWeights,
            const BestAlignment_t* __restrict__ bestAlignmentFlags,
            const int* __restrict__ shifts,
            const int* __restrict__ nOps,
            const int* __restrict__ overlaps,
            const unsigned int* __restrict__ subjectSequencesData,
            const unsigned int* __restrict__ candidateSequencesData,
            const unsigned int* __restrict__ transposedCandidateSequencesData,
            const int* __restrict__ subjectSequencesLength,
            const int* __restrict__ candidateSequencesLength,
            const char* __restrict__ subjectQualities,
            const char* __restrict__ candidateQualities,
            bool* __restrict__ d_shouldBeKept,
            const int* __restrict__ d_candidates_per_subject_prefixsum,
            const int* __restrict__ d_numAnchors,
            const int*,
            float desiredAlignmentMaxErrorRate,
            bool canUseQualityScores,
            size_t encodedSequencePitchInInts,
            size_t qualityPitchInBytes,
            size_t msaColumnPitchInElements,
            int* __restrict__ d_indices,
            int* __restrict__ d_indices_per_subject,
            int dataset_coverage,
            const bool* __restrict__ canExecute,
            int numRefinementIterations,
            bool*){

        constexpr bool useSmemMSA = (memoryType == MemoryType::Shared);

        using BlockReduceBool = cub::BlockReduce<bool, BLOCKSIZE>;
        using BlockReduceInt2 = cub::BlockReduce<int2, BLOCKSIZE>;

        extern __shared__ float externsharedmem[];
        __shared__ MSAColumnProperties shared_columnProperties;

        __shared__ union{
            typename BlockReduceBool::TempStorage boolreduce;
            typename BlockReduceInt2::TempStorage int2reduce;
        } temp_storage;      

        if(*canExecute){

            assert(numRefinementIterations > 0);

            auto tbGroup = cg::this_thread_block();

            float* const shared_weights = externsharedmem;
            int* const shared_counts = (int*)(shared_weights + 4 * msaColumnPitchInElements);
            int* const shared_coverages = (int*)(shared_counts + 4 * msaColumnPitchInElements);            

            const int n_subjects = *d_numAnchors;

            GPUMultiMSA multiMSA;

            multiMSA.numMSAs = n_subjects;
            multiMSA.columnPitchInElements = msaColumnPitchInElements;
            multiMSA.counts = counts;
            multiMSA.weights = weights;
            multiMSA.coverages = coverage;
            multiMSA.consensus = consensus;
            multiMSA.support = support;
            multiMSA.origWeights = origWeights;
            multiMSA.origCoverages = origCoverages;
            multiMSA.columnProperties = msaColumnProperties;

            for(int subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
                int myNumIndices = d_indices_per_subject[subjectIndex];                            

                if(myNumIndices > 0){

                    tbGroup.sync();

                    if(threadIdx.x == 0){
                        shared_columnProperties = msaColumnProperties[subjectIndex];
                    }
                    tbGroup.sync();

                    const int globalOffset = d_candidates_per_subject_prefixsum[subjectIndex];
                    int* const myIndices = d_indices + globalOffset;
                    int* const myNumIndicesPerSubjectPtr = d_indices_per_subject + subjectIndex;

                    int* const myNewIndicesPtr = d_newIndices + globalOffset;
                    int* const myNewNumIndicesPerSubjectPtr = d_newNumIndicesPerSubject + subjectIndex;

                    bool* const myShouldBeKept = d_shouldBeKept + globalOffset;                    

                    GpuSingleMSA msa = multiMSA.getSingleMSA(subjectIndex);
                    msa.columnProperties = &shared_columnProperties;

                    if(useSmemMSA){
                        msa.counts = shared_counts;
                        msa.weights = shared_weights;
                        msa.coverages = shared_coverages;
                    }

                    if(useSmemMSA){
                        //load counts weights and coverages from gmem to smem

                        const int* const gmemCounts = multiMSA.getCountsOfMSA(subjectIndex);
                        const float* const gmemWeights = multiMSA.getWeightsOfMSA(subjectIndex);
                        const int* const gmemCoverages = multiMSA.getCoveragesOfMSA(subjectIndex);

                        for(int k = tbGroup.thread_rank(); k < msaColumnPitchInElements; k += tbGroup.size()){
                            for(int i = 0; i < 4; i++){
                                shared_counts[k + i * msaColumnPitchInElements] 
                                    = gmemCounts[k + i * msaColumnPitchInElements];
                                shared_weights[k + i * msaColumnPitchInElements] 
                                    = gmemWeights[k + i * msaColumnPitchInElements];
                            }
                            shared_coverages[k] = gmemCoverages[k];
                        }
                    }

                    auto storeSmemMSAToGmem = [&](){
                        int* const gmemCounts = multiMSA.getCountsOfMSA(subjectIndex);
                        float* const gmemWeights = multiMSA.getWeightsOfMSA(subjectIndex);
                        int* const gmemCoverages = multiMSA.getCoveragesOfMSA(subjectIndex);

                        for(int k = tbGroup.thread_rank(); k < msaColumnPitchInElements; k += tbGroup.size()){
                            for(int i = 0; i < 4; i++){
                                gmemCounts[k + i * msaColumnPitchInElements] 
                                    = shared_counts[k + i * msaColumnPitchInElements];
                                gmemWeights[k + i * msaColumnPitchInElements] 
                                    = shared_weights[k + i * msaColumnPitchInElements];
                            }
                            gmemCoverages[k] = shared_coverages[k];
                        }
                    };

                    const BestAlignment_t* const myAlignmentFlags = bestAlignmentFlags + globalOffset;
                    const int* const myShifts = shifts + globalOffset;
                    const int* const myNops = nOps + globalOffset;
                    const int* const myOverlaps = overlaps + globalOffset;

                    const unsigned int* const myAnchorSequenceData = subjectSequencesData 
                        + std::size_t(subjectIndex) * encodedSequencePitchInInts;
                    const unsigned int* const myCandidateSequencesData = candidateSequencesData 
                        + std::size_t(globalOffset) * encodedSequencePitchInInts;

                    const char* const myCandidateQualities = candidateQualities 
                        + std::size_t(globalOffset) * qualityPitchInBytes;

                    const int subjectLength = subjectSequencesLength[subjectIndex];
                    const int* const myCandidateLengths = candidateSequencesLength + globalOffset;

                    for(int refinementIteration = 0; 
                            refinementIteration < numRefinementIterations; 
                            refinementIteration++){

                        auto finalizeRefinement = [&](int newNumIndicesPerSubject){
                            //copy indices to correct output array
                            if(refinementIteration % 2 == 1){
                                for(int i = tbGroup.thread_rank(); i < myNumIndices; i += tbGroup.size()){
                                    myNewIndicesPtr[i] = myIndices[i];
                                }
                                if(tbGroup.thread_rank() == 0){
                                    *myNewNumIndicesPerSubjectPtr = *myNumIndicesPerSubjectPtr;
                                }
                            }

                            if(tbGroup.thread_rank() == 0){
                                atomicAdd(d_newNumIndices, newNumIndicesPerSubject);
                            }
                        };

                        int* const srcIndices = (refinementIteration % 2 == 0) ?
                                myIndices : myNewIndicesPtr;
                        int* const destIndices = (refinementIteration % 2 == 0) ?
                                myNewIndicesPtr : myIndices;

                        int* const srcNumIndices = (refinementIteration % 2 == 0) ?
                            myNumIndicesPerSubjectPtr : myNewNumIndicesPerSubjectPtr;
                        int* const destNumIndices = (refinementIteration % 2 == 0) ?
                            myNewNumIndicesPerSubjectPtr : myNumIndicesPerSubjectPtr;

                        tbGroup.sync();

                        auto groupReduceBool = [&](bool b, auto comp){
                            b = BlockReduceBool(temp_storage.boolreduce).Reduce(b, comp);
                            return b;
                        };

                        auto groupReduceInt2 = [&](int2 b, auto comp){
                            b = BlockReduceInt2(temp_storage.int2reduce).Reduce(b, comp);
                            return b;
                        };
                        

                        msa.flagCandidatesOfDifferentRegion(
                            tbGroup,
                            groupReduceBool,
                            groupReduceInt2,
                            destIndices,
                            destNumIndices,
                            myAnchorSequenceData,
                            subjectLength,
                            myCandidateSequencesData,
                            myCandidateLengths,
                            myAlignmentFlags,
                            myShifts,
                            myNops,
                            myOverlaps,
                            myShouldBeKept,
                            desiredAlignmentMaxErrorRate,
                            subjectIndex,
                            encodedSequencePitchInInts,
                            srcIndices,
                            *srcNumIndices,
                            dataset_coverage
                        );

                        tbGroup.sync();

                        const int myNewNumIndices = *destNumIndices;
                        
                        assert(myNewNumIndices <= myNumIndices);
                        if(myNewNumIndices > 0 && myNewNumIndices < myNumIndices){
                            auto selector = [&](int i){
                                return !myShouldBeKept[i];
                            };

                            msa.removeCandidates(
                                tbGroup,
                                selector,
                                myShifts,
                                myOverlaps,
                                myNops,
                                myAlignmentFlags,
                                myCandidateSequencesData, //not transposed
                                myCandidateQualities, //not transposed
                                myCandidateLengths,
                                srcIndices,
                                *srcNumIndices,
                                canUseQualityScores, 
                                encodedSequencePitchInInts,
                                qualityPitchInBytes,
                                desiredAlignmentMaxErrorRate
                            );

                            tbGroup.sync();

                            msa.updateColumnProperties(tbGroup);

                            tbGroup.sync();

                            //msa.checkAfterBuild(tbGroup, subjectIndex);

                            assert(shared_columnProperties.firstColumn_incl != -1);
                            assert(shared_columnProperties.lastColumn_excl != -1);

                            msa.findConsensus(
                                tbGroup,
                                myAnchorSequenceData, 
                                encodedSequencePitchInInts, 
                                subjectIndex
                            );

                            if(tbGroup.thread_rank() == 0){
                                msaColumnProperties[subjectIndex] = shared_columnProperties;
                            }

                            tbGroup.sync();

                            myNumIndices = myNewNumIndices;

                            if(refinementIteration == numRefinementIterations - 1){
                                //copy shared mem msa back to gmem

                                if(useSmemMSA){                                            
                                    storeSmemMSAToGmem();
                                }

                                finalizeRefinement(myNewNumIndices);
                            }

                        }else{
                            assert(myNewNumIndices == myNumIndices);

                            if(useSmemMSA){                                 
                                if(refinementIteration > 0){ // if iteration 0 fails, no changes were made
                                    storeSmemMSAToGmem();
                                }
                            }

                            finalizeRefinement(myNewNumIndices);

                            break; //stop refinement
                        }

                        
                    }
                }else{
                    if(tbGroup.thread_rank() == 0){
                        d_newNumIndicesPerSubject[subjectIndex] = 0;
                    }
                    ; //nothing else to do if there are no candidates in msa
                }                
            }
        }
    }









    //####################   KERNEL DISPATCH   ####################
    
    


    void callMsaFindCandidatesOfDifferentRegionAndRemoveThemViaDeletion2MultiIterationKernel_async(
        int* d_newIndices,
        int* d_newNumIndicesPerSubject,
        int* d_newNumIndices,
        MSAColumnProperties* d_msaColumnProperties,
        char* d_consensus,
        int* d_coverage,
        int* d_counts,
        float* d_weights,
        float* d_support,
        int* d_origCoverages,
        float* d_origWeights,
        const BestAlignment_t* d_bestAlignmentFlags,
        const int* d_shifts,
        const int* d_nOps,
        const int* d_overlaps,
        const unsigned int* d_subjectSequencesData,
        const unsigned int* d_candidateSequencesData,
        const unsigned int* d_transposedCandidateSequencesData,
        const int* d_subjectSequencesLength,
        const int* d_candidateSequencesLength,
        const char* d_subjectQualities,
        const char* d_candidateQualities,
        bool* d_shouldBeKept,
        const int* d_candidates_per_subject_prefixsum,
        const int* d_numAnchors,
        const int* d_numCandidates,
        float desiredAlignmentMaxErrorRate,
        int maxNumAnchors,
        int maxNumCandidates,
        bool canUseQualityScores,
        size_t encodedSequencePitchInInts,
        size_t qualityPitchInBytes,
        size_t msaColumnPitchInElements,
        int* d_indices,
        int* d_indices_per_subject,
        int dataset_coverage,
        const bool* d_canExecute,
        int numIteration,
        bool* d_anchorIsFinished,
        cudaStream_t stream,
        KernelLaunchHandle& handle
    ){

        call_fill_kernel_async(
            d_newNumIndices,
            1,
            0,
            stream
        );

        constexpr int blocksize = 128;

        constexpr MemoryType memType = MemoryType::Shared;

        constexpr bool usesSmem = memType == MemoryType::Shared;

        const std::size_t smemAddSequences = (usesSmem ? 
                                                sizeof(float) * 4 * msaColumnPitchInElements // weights
                                                    + sizeof(int) * 4 * msaColumnPitchInElements // counts
                                                    + sizeof(int) * msaColumnPitchInElements // coverages
                                                : 0);

        const std::size_t smem = smemAddSequences;

        constexpr auto kernelId = KernelId::MSAFindCandidatesOfDifferentRegionAndRemoveThemViaDeletion;

        int max_blocks_per_device = 1;

        KernelLaunchConfig kernelLaunchConfig;
        kernelLaunchConfig.threads_per_block = blocksize;
        kernelLaunchConfig.smem = smem;

        auto iter = handle.kernelPropertiesMap.find(kernelId);
        if(iter == handle.kernelPropertiesMap.end()) {

            std::map<KernelLaunchConfig, KernelProperties> mymap;

            KernelProperties kernelProperties;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &kernelProperties.max_blocks_per_SM,
                msa_findCandidatesOfDifferentRegionAndRemoveThemViaDeletion2_multiiteration_kernel<blocksize, memType>,
                kernelLaunchConfig.threads_per_block, 
                kernelLaunchConfig.smem
            ); CUERR;

            mymap[kernelLaunchConfig] = kernelProperties;
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

            handle.kernelPropertiesMap[kernelId] = std::move(mymap);
        }else{
            std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
            const KernelProperties& kernelProperties = map[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
        }

        dim3 block(blocksize, 1, 1);
        //dim3 grid(maxNumAnchors, 1, 1);
        dim3 grid(max_blocks_per_device, 1, 1);


        msa_findCandidatesOfDifferentRegionAndRemoveThemViaDeletion2_multiiteration_kernel<blocksize, memType>
                <<<grid, block, smem, stream>>>(
            d_newIndices,
            d_newNumIndicesPerSubject,
            d_newNumIndices,
            d_msaColumnProperties,
            d_consensus,
            d_coverage,
            d_counts,
            d_weights,
            d_support,
            d_origCoverages,
            d_origWeights,
            d_bestAlignmentFlags,
            d_shifts,
            d_nOps,
            d_overlaps,
            d_subjectSequencesData,
            d_candidateSequencesData,
            d_transposedCandidateSequencesData,
            d_subjectSequencesLength,
            d_candidateSequencesLength,
            d_subjectQualities,
            d_candidateQualities,
            d_shouldBeKept,
            d_candidates_per_subject_prefixsum,
            d_numAnchors,
            d_numCandidates,
            desiredAlignmentMaxErrorRate,
            canUseQualityScores,
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            msaColumnPitchInElements,
            d_indices,
            d_indices_per_subject,
            dataset_coverage,
            d_canExecute,
            numIteration,
            d_anchorIsFinished
        );
    }





    void callMsaFindCandidatesOfDifferentRegionAndRemoveThemViaRebuildKernel_async(
            int* d_newIndices,
            int* d_newNumIndicesPerSubject,
            int* d_newNumIndices,
            MSAColumnProperties* d_msaColumnProperties,
            char* d_consensus,
            int* d_coverage,
            int* d_counts,
            float* d_weights,
            float* d_support,
            int* d_origCoverages,
            float* d_origWeights,
            const BestAlignment_t* d_bestAlignmentFlags,
            const int* d_shifts,
            const int* d_nOps,
            const int* d_overlaps,
            const unsigned int* d_subjectSequencesData,
            const unsigned int* d_candidateSequencesData,
            const unsigned int* d_transposedCandidateSequencesData,
            const int* d_subjectSequencesLength,
            const int* d_candidateSequencesLength,
            const char* d_subjectQualities,
            const char* d_candidateQualities,
            bool* d_shouldBeKept,
            const int* d_candidates_per_subject_prefixsum,
            const int* d_numAnchors,
            const int* d_numCandidates,
            float desiredAlignmentMaxErrorRate,
            int maxNumAnchors,
            int maxNumCandidates,
            bool canUseQualityScores,
            size_t encodedSequencePitchInInts,
            size_t qualityPitchInBytes,
            size_t msaColumnPitchInElements,
            const int* d_indices,
            const int* d_indices_per_subject,
            int dataset_coverage,
            const bool* d_canExecute,
            int iteration,
            bool* d_anchorIsFinished,
            cudaStream_t stream,
            KernelLaunchHandle& handle){

        call_fill_kernel_async(
            d_newNumIndices,
            1,
            0,
            stream
        );

        constexpr int blocksize = 128;

        constexpr MemoryType addSequencesMemType = MemoryType::Shared;

        constexpr bool addSequencesUsesSmem = addSequencesMemType == MemoryType::Shared;

        const std::size_t smemAddSequences = (addSequencesUsesSmem ? 
                                                sizeof(float) * 4 * msaColumnPitchInElements // weights
                                                    + sizeof(int) * 4 * msaColumnPitchInElements // counts
                                                    + sizeof(int) * msaColumnPitchInElements // coverages
                                                : 0);

        const std::size_t smem = smemAddSequences;

        constexpr auto kernelId = KernelId::MSAFindCandidatesOfDifferentRegionAndRemoveThemViaRebuild;

        int max_blocks_per_device = 1;

        KernelLaunchConfig kernelLaunchConfig;
        kernelLaunchConfig.threads_per_block = blocksize;
        kernelLaunchConfig.smem = smem;

        auto iter = handle.kernelPropertiesMap.find(kernelId);
        if(iter == handle.kernelPropertiesMap.end()) {

            std::map<KernelLaunchConfig, KernelProperties> mymap;

            KernelProperties kernelProperties;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &kernelProperties.max_blocks_per_SM,
                msa_findCandidatesOfDifferentRegionAndRemoveThemViaRebuild_kernel<blocksize, addSequencesMemType>,
                kernelLaunchConfig.threads_per_block, 
                kernelLaunchConfig.smem
            ); CUERR;

            mymap[kernelLaunchConfig] = kernelProperties;
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

            handle.kernelPropertiesMap[kernelId] = std::move(mymap);
        }else{
            std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
            const KernelProperties& kernelProperties = map[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
        }

        dim3 block(blocksize, 1, 1);
        //dim3 grid(maxNumAnchors, 1, 1);
        dim3 grid(max_blocks_per_device, 1, 1);


        msa_findCandidatesOfDifferentRegionAndRemoveThemViaRebuild_kernel<blocksize, addSequencesMemType><<<grid, block, smem, stream>>>(
            d_newIndices,
            d_newNumIndicesPerSubject,
            d_newNumIndices,
            d_msaColumnProperties,
            d_consensus,
            d_coverage,
            d_counts,
            d_weights,
            d_support,
            d_origCoverages,
            d_origWeights,
            d_bestAlignmentFlags,
            d_shifts,
            d_nOps,
            d_overlaps,
            d_subjectSequencesData,
            d_candidateSequencesData,
            d_transposedCandidateSequencesData,
            d_subjectSequencesLength,
            d_candidateSequencesLength,
            d_subjectQualities,
            d_candidateQualities,
            d_shouldBeKept,
            d_candidates_per_subject_prefixsum,
            d_numAnchors,
            d_numCandidates,
            desiredAlignmentMaxErrorRate,
            canUseQualityScores,
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            msaColumnPitchInElements,
            d_indices,
            d_indices_per_subject,
            dataset_coverage,
            d_canExecute,
            iteration,
            d_anchorIsFinished
        );
    }





    void callBuildMSA3Kernel_async(
        MSAColumnProperties* d_msaColumnProperties,
        int* d_coverage,
        int* d_counts,
        float* d_weights,
        float* d_support,
        float* d_origWeights,
        int* d_origCoverages,
        char* d_consensus,          
        const int* d_overlaps,
        const int* d_shifts,
        const int* d_nOps,
        const BestAlignment_t* d_bestAlignmentFlags,
        const int* d_anchorLengths,
        const int* d_candidateLengths,
        const int* d_indices,
        const int* d_indices_per_subject,
        const int* d_candidatesPerSubjectPrefixSum,            
        const unsigned int* d_subjectSequencesData,
        const unsigned int* d_candidateSequencesTransposedData,
        const char* d_subjectQualities,
        const char* d_candidateQualities,
        const int* d_numAnchors,
        const int* d_numCandidates,
        float desiredAlignmentMaxErrorRate,
        int maxNumAnchors,
        int maxNumCandidates,
        bool canUseQualityScores,
        int encodedSequencePitchInInts,
        size_t qualityPitchInBytes,
        size_t msaColumnPitchInElements,
        const bool* d_canExecute,
        cudaStream_t stream,
        KernelLaunchHandle& handle){
            

    constexpr MemoryType addSequencesMemType = MemoryType::Shared;
    constexpr bool addSequencesUsesSmem = addSequencesMemType == MemoryType::Shared;
    constexpr int BLOCKSIZE = 128;

    using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;
    using BlockReduceIntStorage = typename BlockReduceInt::TempStorage;

    const std::size_t smemCub = sizeof(BlockReduceIntStorage);
    const std::size_t smemAddSequences = (addSequencesUsesSmem ? 
                                            sizeof(float) * 4 * msaColumnPitchInElements // weights
                                                + sizeof(int) * 4 * msaColumnPitchInElements // counts
                                                + sizeof(int) * msaColumnPitchInElements // coverages
                                            : 0);

    const std::size_t smem = std::max(smemCub, smemAddSequences);

    int max_blocks_per_device = 1;

    KernelLaunchConfig kernelLaunchConfig;
    kernelLaunchConfig.threads_per_block = BLOCKSIZE;
    kernelLaunchConfig.smem = smem;

    auto iter = handle.kernelPropertiesMap.find(KernelId::MSABuild3);
    if(iter == handle.kernelPropertiesMap.end()) {

        std::map<KernelLaunchConfig, KernelProperties> mymap;

        KernelProperties kernelProperties;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &kernelProperties.max_blocks_per_SM,
            buildMSA3Kernel<BLOCKSIZE, addSequencesMemType>,
            kernelLaunchConfig.threads_per_block, 
            kernelLaunchConfig.smem
        ); CUERR;

        mymap[kernelLaunchConfig] = kernelProperties;
        max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

        handle.kernelPropertiesMap[KernelId::MSABuild3] = std::move(mymap);
    }else{
        std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
        const KernelProperties& kernelProperties = map[kernelLaunchConfig];
        max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    }

    dim3 block(BLOCKSIZE, 1, 1);
    //dim3 grid(maxNumAnchors, 1, 1);
    dim3 grid(max_blocks_per_device, 1, 1);
    
    buildMSA3Kernel<BLOCKSIZE, addSequencesMemType><<<grid, block, smem, stream>>>(
        d_msaColumnProperties,
        d_coverage,
        d_counts,
        d_weights,
        d_support,
        d_origWeights,
        d_origCoverages,
        d_consensus,          
        d_overlaps,
        d_shifts,
        d_nOps,
        d_bestAlignmentFlags,
        d_anchorLengths,
        d_candidateLengths,
        d_indices,
        d_indices_per_subject,
        d_candidatesPerSubjectPrefixSum,            
        d_subjectSequencesData,
        d_candidateSequencesTransposedData,
        d_subjectQualities,
        d_candidateQualities,
        d_numAnchors,
        d_numCandidates,
        desiredAlignmentMaxErrorRate,
        canUseQualityScores,
        encodedSequencePitchInInts,
        qualityPitchInBytes,
        msaColumnPitchInElements,
        d_canExecute
    ); CUERR;



}





}
}
