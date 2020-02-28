//#define NDEBUG

#include <gpu/kernels.hpp>
#include <gpu/devicefunctionsforkernels.cuh>

//#include <gpu/bestalignment.hpp>
#include <bestalignment.hpp>
#include <gpu/utility_kernels.cuh>
#include <gpu/cubcachingallocator.cuh>

#include <msa.hpp>
#include <sequence.hpp>




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
            size_t msa_weights_row_pitch_floats,
            int subjectIndex){

        const int firstColumn_incl = msaColumnProperties->firstColumn_incl;
        const int lastColumn_excl = msaColumnProperties->lastColumn_excl;

        for(int column = firstColumn_incl + threadIdx.x; column < lastColumn_excl; column += blockDim.x){
            const int* const mycounts = counts + column;
            const float* const myweights = weights + column;
            float sumOfWeights = 0.0f;

            for(int k = 0; k < 4; k++){
                const int count = mycounts[k * msa_weights_row_pitch_floats];
                const float weight = myweights[k * msa_weights_row_pitch_floats];
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

        for(int k = threadIdx.x; k < numGoodCandidates; k += blockDim.x) {
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



    template<bool isTransposedSequence>
    __device__ 
    void addASequence2BitToMSA(
            int* __restrict__ counts,
            float* __restrict__ weights,
            int* __restrict__ coverages,
            int rowPitchInElements,
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
                    const int rowOffset = encodedBaseAsInt * rowPitchInElements;
                    const int columnIndex = columnStart 
                            + (isForward ? (intIndex * 16 + posInInt) : sequenceLength - 1 - (intIndex * 16 + posInInt));
                    
                    atomicAdd(counts + rowOffset + columnIndex, 1);
                    atomicAdd(weights + rowOffset + columnIndex, weight);
                    atomicAdd(coverages + columnIndex, 1);
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
                const int rowOffset = encodedBaseAsInt * rowPitchInElements;
                const int columnIndex = columnStart 
                    + (isForward ? (fullInts * 16 + posInInt) : sequenceLength - 1 - (fullInts * 16 + posInInt));
                atomicAdd(counts + rowOffset + columnIndex, 1);
                atomicAdd(weights + rowOffset + columnIndex, weight);
                atomicAdd(coverages + columnIndex, 1);
            } 
        }
    }

    template<MemoryType memType>
    __device__ __forceinline__
    void addSequencesToMSASingleBlock(
            float* __restrict__ sharedmem,
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
            size_t msa_weights_row_pitch_floats,
            size_t encodedSequencePitchInInts,
            size_t qualityPitchInBytes,
            int subjectIndex){  

        constexpr bool candidatesAreTransposed = true;
        constexpr bool useSmem = memType == MemoryType::Shared;

        float* const shared_weights = sharedmem;
        int* const shared_counts = (int*)(shared_weights + 4 * msa_weights_row_pitch_floats);
        int* const shared_coverages = (int*)(shared_counts + 4 * msa_weights_row_pitch_floats);

        int* const mycounts = (useSmem ? shared_counts : inputcounts);
        float* const myweights = (useSmem ? shared_weights : inputweights);
        int* const mycoverages = (useSmem ? shared_coverages : inputcoverages);        

        for(int column = threadIdx.x; column < msa_weights_row_pitch_floats * 4; column += blockDim.x){
            mycounts[column] = 0;
            myweights[column] = 0;
        }

        for(int column = threadIdx.x; column < msa_weights_row_pitch_floats; column += blockDim.x){
            mycoverages[column] = 0;
        }   
        
        __syncthreads();

        //add subject
        const int subjectColumnsBegin_incl = myMsaColumnProperties->subjectColumnsBegin_incl;
        const int subjectColumnsEnd_excl = myMsaColumnProperties->subjectColumnsEnd_excl;
        const int columnsToCheck = myMsaColumnProperties->lastColumn_excl;

        assert(columnsToCheck <= msa_weights_row_pitch_floats);

        const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
        const unsigned int* const subject = myAnchorSequenceData;
        const char* const subjectQualityScore = myAnchorQualityData;
                        
        for(int i = threadIdx.x; i < subjectLength; i+= blockDim.x){
            const int columnIndex = subjectColumnsBegin_incl + i;
            const unsigned int encbase = getEncodedNuc2Bit(subject, subjectLength, i);
            const float weight = canUseQualityScores ? getQualityWeight(subjectQualityScore[i]) : 1.0f;
            const int rowOffset = int(encbase) * msa_weights_row_pitch_floats;

            atomicAdd(mycounts + rowOffset + columnIndex, 1);
            atomicAdd(myweights + rowOffset + columnIndex, weight);
            atomicAdd(mycoverages + columnIndex, 1);
        }

        for(int indexInList = threadIdx.x; indexInList < numIndices; indexInList += blockDim.x){

            const int localCandidateIndex = myIndices[indexInList];
            const int shift = myShifts[localCandidateIndex];
            const BestAlignment_t flag = myAlignmentFlags[localCandidateIndex];

            const int queryLength = myCandidateLengths[localCandidateIndex];
            const unsigned int* const query = myTransposedCandidateSequencesData + localCandidateIndex;

            const char* const queryQualityScore = myCandidateQualities + std::size_t(localCandidateIndex) * qualityPitchInBytes;

            const int query_alignment_overlap = myOverlaps[localCandidateIndex];
            const int query_alignment_nops = myNops[localCandidateIndex];

            const float overlapweight = calculateOverlapWeight(subjectLength, query_alignment_nops, query_alignment_overlap);

            assert(overlapweight <= 1.0f);
            assert(overlapweight >= 0.0f);
            assert(flag != BestAlignment_t::None);                 // indices should only be pointing to valid alignments

            const int defaultcolumnoffset = subjectColumnsBegin_incl + shift;

            const bool isForward = flag == BestAlignment_t::Forward;

            addASequence2BitToMSA<candidatesAreTransposed>(
                mycounts,
                myweights,
                mycoverages,
                msa_weights_row_pitch_floats,
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

        if(useSmem){

            __syncthreads(); //wait until msa is ready

            checkBuiltMSA(
                myMsaColumnProperties,
                mycounts,
                myweights,
                msa_weights_row_pitch_floats,
                subjectIndex
            ); 

            // copy from shared to global

            int* const gmemCoverage  = inputcoverages;
            for(int index = threadIdx.x; index < columnsToCheck; index += blockDim.x){
                for(int k = 0; k < 4; k++){
                    const int* const srcCounts = mycounts + k * msa_weights_row_pitch_floats + index;
                    int* const destCounts = inputcounts + k * msa_weights_row_pitch_floats + index;

                    const float* const srcWeights = myweights + k * msa_weights_row_pitch_floats + index;
                    float* const destWeights = inputweights + k * msa_weights_row_pitch_floats + index;

                    *destCounts = *srcCounts;
                    *destWeights = *srcWeights;
                    //atomicAdd(destCounts ,*srcCounts);
                    //atomicAdd(destWeights, *srcWeights);
                }
                gmemCoverage[index] = mycoverages[index];
                //atomicAdd(gmemCoverage + index, mycoverage[index]);
            }

            __syncthreads(); //wait until smem can be reused
        }else{

            __syncthreads(); //wait until msa is ready

            //check msa

            checkBuiltMSA(
                myMsaColumnProperties,
                mycounts,
                myweights,
                msa_weights_row_pitch_floats,
                subjectIndex
            ); 
        }
    }


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
            size_t msa_pitch,
            size_t msa_weights_pitch_floats){

        const int subjectColumnsBegin_incl = myMsaColumnProperties->subjectColumnsBegin_incl;
        const int subjectColumnsEnd_excl = myMsaColumnProperties->subjectColumnsEnd_excl;
        const int firstColumn_incl = myMsaColumnProperties->firstColumn_incl;
        const int lastColumn_excl = myMsaColumnProperties->lastColumn_excl;

        assert(lastColumn_excl <= msa_weights_pitch_floats);

        const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
        const unsigned int* const subject = myAnchorSequenceData;

        //set columns to zero which are not in range firstColumn_incl <= column && column < lastColumn_excl

        for(int column = threadIdx.x; 
                column < firstColumn_incl; 
                column += blockDim.x){

            my_support[column] = 0;
            my_orig_weights[column] = 0;
            my_orig_coverage[column] = 0;
        }

        for(int column = threadIdx.x; 
                lastColumn_excl <= column && column < msa_weights_pitch_floats; 
                column += blockDim.x){

            my_support[column] = 0;
            my_orig_weights[column] = 0;
            my_orig_coverage[column] = 0;
        }

        for(int column = threadIdx.x; 
            column < firstColumn_incl; 
            column += blockDim.x){
                
            my_consensus[column] = 0;
        }

        for(int column = threadIdx.x; 
                lastColumn_excl <= column && column < msa_weights_pitch_floats; 
                column += blockDim.x){

            my_consensus[column] = 0;
        }

        const int* const myCountsA = myCounts + 0 * msa_weights_pitch_floats;
        const int* const myCountsC = myCounts + 1 * msa_weights_pitch_floats;
        const int* const myCountsG = myCounts + 2 * msa_weights_pitch_floats;
        const int* const myCountsT = myCounts + 3 * msa_weights_pitch_floats;

        const float* const my_weightsA = myWeights + 0 * msa_weights_pitch_floats;
        const float* const my_weightsC = myWeights + 1 * msa_weights_pitch_floats;
        const float* const my_weightsG = myWeights + 2 * msa_weights_pitch_floats;
        const float* const my_weightsT = myWeights + 3 * msa_weights_pitch_floats;

        for(int column = threadIdx.x; firstColumn_incl <= column && column < lastColumn_excl; column += blockDim.x){
            const int ca = myCountsA[column];
            const int cc = myCountsC[column];
            const int cg = myCountsG[column];
            const int ct = myCountsT[column];
            const float wa = my_weightsA[column];
            const float wc = my_weightsC[column];
            const float wg = my_weightsG[column];
            const float wt = my_weightsT[column];

            char cons = 'F';
            float consWeight = 0.0f;
            if(wa > consWeight){
                cons = 'A';
                consWeight = wa;
            }
            if(wc > consWeight){
                cons = 'C';
                consWeight = wc;
            }
            if(wg > consWeight){
                cons = 'G';
                consWeight = wg;
            }
            if(wt > consWeight){
                cons = 'T';
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
                constexpr char A_enc = 0x00;
                constexpr char C_enc = 0x01;
                constexpr char G_enc = 0x02;
                constexpr char T_enc = 0x03;
                const int localIndex = column - subjectColumnsBegin_incl;
                const char subjectbase = getEncodedNuc2Bit(subject, subjectLength, localIndex);

                if(subjectbase == A_enc){
                    my_orig_weights[column] = wa;
                    my_orig_coverage[column] = ca;
                }else if(subjectbase == C_enc){
                    my_orig_weights[column] = wc;
                    my_orig_coverage[column] = cc;
                }else if(subjectbase == G_enc){
                    my_orig_weights[column] = wg;
                    my_orig_coverage[column] = cg;
                }else if(subjectbase == T_enc){
                    my_orig_weights[column] = wt;
                    my_orig_coverage[column] = ct;
                }
            }
        }
    }


  

    template<int BLOCKSIZE>
    __global__
    void msaInitKernel(
                MSAColumnProperties* __restrict__ msaColumnProperties,
                const int* __restrict__ alignmentShifts,
                const BestAlignment_t* __restrict__ bestAlignmentFlags,
                const int* __restrict__ anchorLengths,
                const int* __restrict__ candidateLengths,
                const int* __restrict__ indices,
                const int* __restrict__ indices_per_subject,
                const int* __restrict__ candidatesPerSubjectPrefixSum,
                int n_subjects,
                const bool* __restrict__ canExecute){

        if(*canExecute){

            using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;            

            __shared__ union {
                typename BlockReduceInt::TempStorage reduce;
            } temp_storage;

            for(int subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x) {
                
                // We only want to consider the candidates with good alignments. the indices of those were determined in a previous step
                const int myNumGoodCandidates = indices_per_subject[subjectIndex];
                
                if(myNumGoodCandidates > 0){
                    MSAColumnProperties* const properties_ptr = msaColumnProperties + subjectIndex;

                    const int globalOffset = candidatesPerSubjectPrefixSum[subjectIndex];

                    const int* const myIndicesPtr = indices + globalOffset;
                    const int* const myShiftsPtr = alignmentShifts + globalOffset;
                    const BestAlignment_t* const myAlignmentFlagsPtr = bestAlignmentFlags + globalOffset;
                    const int* const myCandidateLengthsPtr = candidateLengths + globalOffset;

                    const int subjectLength = anchorLengths[subjectIndex];

                    msaInit<BLOCKSIZE>(
                        temp_storage.reduce,
                        properties_ptr,
                        myIndicesPtr,
                        myNumGoodCandidates,
                        myShiftsPtr,
                        myAlignmentFlagsPtr,
                        subjectLength,
                        myCandidateLengthsPtr
                    );
                }
            }
        }
    }


    __global__
    void msa_update_properties_kernel(
                MSAPointers d_msapointers,
                const int* __restrict__ d_indices_per_subject,
                size_t msa_weights_pitch,
                int n_subjects,
                const bool* __restrict__ canExecute){

        if(*canExecute){

            const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

            for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x) {
                MSAColumnProperties* const properties_ptr = d_msapointers.msaColumnProperties + subjectIndex;
                const int firstColumn_incl = properties_ptr->firstColumn_incl;
                const int lastColumn_excl = properties_ptr->lastColumn_excl;

                // We only want to consider the candidates with good alignments. the indices of those were determined in a previous step
                const int num_indices_for_this_subject = d_indices_per_subject[subjectIndex];

                if(num_indices_for_this_subject > 0){
                    const int* const my_coverage = d_msapointers.coverage + subjectIndex * msa_weights_pitch_floats;

                    for(int column = threadIdx.x; firstColumn_incl <= column && column < lastColumn_excl-1; column += blockDim.x){
                        assert(my_coverage[column] >= 0);

                        if(my_coverage[column] == 0 && my_coverage[column+1] > 0){
                            properties_ptr->firstColumn_incl = column+1;
                        }

                        if(my_coverage[column] > 0 && my_coverage[column+1] == 0){
                            properties_ptr->lastColumn_excl = column+1;
                        }
                    }

                }else{
                    //clear MSA
                    if(threadIdx.x == 0) {
                        MSAColumnProperties my_columnproperties;

                        my_columnproperties.subjectColumnsBegin_incl = 0;
                        my_columnproperties.subjectColumnsEnd_excl = 0;
                        my_columnproperties.firstColumn_incl = 0;
                        my_columnproperties.lastColumn_excl = 0;

                        *properties_ptr = my_columnproperties;
                    }
                }
            }
        }
    }









    template<MemoryType memType>
    __global__
    void msa_add_sequences_kernel_singleblock(
            int* __restrict__ coverage,
            const MSAColumnProperties* __restrict__ msaColumnProperties,
            int* __restrict__ counts,
            float* __restrict__ weights,
            const int* __restrict__ overlaps,
            const int* __restrict__ shifts,
            const int* __restrict__ nOps,
            const BestAlignment_t* __restrict__ bestAlignmentFlags,
            const unsigned int* __restrict__ subjectSequencesData,
            const unsigned int* __restrict__ candidateSequencesTransposedData,
            const int* __restrict__ subjectSequencesLength,
            const int* __restrict__ candidateSequencesLength,
            const char* __restrict__ subjectQualities,
            const char* __restrict__ candidateQualities,
            const int* __restrict__ d_candidates_per_subject_prefixsum,
            const int* __restrict__ d_indices,
            const int* __restrict__ d_indices_per_subject,
            int n_subjects,
            int n_queries,
            bool canUseQualityScores,
            int encodedSequencePitchInInts,
            size_t qualityPitchInBytes,
            size_t msa_row_pitch,
            size_t msa_weights_row_pitch_floats,
            const bool* __restrict__ canExecute){


        if(*canExecute){

            extern __shared__ float sharedmem[];
            
            for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x) {
                const int numGoodCandidates = d_indices_per_subject[subjectIndex];

                if(numGoodCandidates > 0){

                    const int globalCandidateOffset = d_candidates_per_subject_prefixsum[subjectIndex];

                    int* const mycounts = counts + subjectIndex * 4 * msa_weights_row_pitch_floats;
                    float* const myweights = weights + subjectIndex * 4 * msa_weights_row_pitch_floats;
                    int* const mycoverages = coverage + subjectIndex * msa_weights_row_pitch_floats;
                    const MSAColumnProperties* const myMsaColumnProperties = msaColumnProperties + subjectIndex;

                    const int* const myShifts = shifts + globalCandidateOffset;
                    const int* const myOverlaps = overlaps + globalCandidateOffset;
                    const int* const myNops = nOps + globalCandidateOffset;

                    const BestAlignment_t* const myAlignmentFlags = bestAlignmentFlags + globalCandidateOffset;
                    const unsigned int* const myAnchorSequenceData = subjectSequencesData + std::size_t(subjectIndex) * encodedSequencePitchInInts;
                    const char* const myAnchorQualityData = subjectQualities + std::size_t(subjectIndex) * qualityPitchInBytes;
                    const unsigned int* const myCandidateSequencesData = candidateSequencesTransposedData + size_t(globalCandidateOffset);

                    const char* const myCandidateQualities = candidateQualities 
                                                                        + size_t(globalCandidateOffset) * qualityPitchInBytes;

                    const int* const myCandidateLengths = candidateSequencesLength + globalCandidateOffset;

                    const int* const myIndices = d_indices + globalCandidateOffset;

                    addSequencesToMSASingleBlock<memType>(
                        sharedmem,
                        mycounts,
                        myweights,
                        mycoverages,
                        myMsaColumnProperties,
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
                        numGoodCandidates,
                        n_queries,
                        canUseQualityScores, 
                        msa_weights_row_pitch_floats,
                        encodedSequencePitchInInts,
                        qualityPitchInBytes,
                        subjectIndex
                    );
                }                
            }
        }
    }



    template<MemoryType memType>
    __global__
    void msa_add_sequences_kernel_multiblock(
            int* __restrict__ coverage,
            MSAColumnProperties* __restrict__ msaColumnProperties,
            int* __restrict__ counts,
            float* __restrict__ weights,
            const int* __restrict__ overlaps,
            const int* __restrict__ shifts,
            const int* __restrict__ nOps,
            const BestAlignment_t* __restrict__ bestAlignmentFlags,
            const unsigned int* __restrict__ subjectSequencesData,
            const unsigned int* __restrict__ candidateSequencesTransposedData,
            const int* __restrict__ subjectSequencesLength,
            const int* __restrict__ candidateSequencesLength,
            const char* __restrict__ subjectQualities,
            const char* __restrict__ candidateQualities,
            const int* __restrict__ d_candidates_per_subject_prefixsum,
            const int* __restrict__ d_indices,
            const int* __restrict__ d_indices_per_subject,
            const int* __restrict__ d_blocksPerSubjectPrefixSum,
            int n_subjects,
            int n_queries,
            bool canUseQualityScores,
            int encodedSequencePitchInInts,
            size_t qualityPitchInBytes,
            size_t msa_row_pitch,
            size_t msa_weights_row_pitch_floats,
            const bool* __restrict__ canExecute){

        constexpr bool candidatesAreTransposed = true;
        constexpr bool useSmem = memType == MemoryType::Shared;

        auto get = [] (const char* data, int length, int index){
            return getEncodedNuc2Bit((const unsigned int*)data, length, index, [](auto i){return i;});
        };

        if(*canExecute){

            extern __shared__ float sharedmem[];


            //const size_t msa_weights_row_pitch_floats = msa_weights_row_pitch / sizeof(float);
            const int smemsizefloats = 4 * msa_weights_row_pitch_floats + 4 * msa_weights_row_pitch_floats;

            float* const shared_weights = sharedmem;
            int* const shared_counts = (int*)(shared_weights + 4 * msa_weights_row_pitch_floats);

            const int requiredLogicalBlocks = d_blocksPerSubjectPrefixSum[n_subjects];
            
            for(int logicalBlockId = blockIdx.x; logicalBlockId < requiredLogicalBlocks; logicalBlockId += gridDim.x) {

                const auto iterator = thrust::lower_bound(
                    thrust::seq,
                    d_blocksPerSubjectPrefixSum,
                    d_blocksPerSubjectPrefixSum + n_subjects + 1,
                    logicalBlockId + 1
                );

                const int subjectIndex = thrust::distance(d_blocksPerSubjectPrefixSum, iterator)-1;

                const int numBlocksForThisSubject = d_blocksPerSubjectPrefixSum[subjectIndex+1] - d_blocksPerSubjectPrefixSum[subjectIndex];
                const int blockIdForThisSubject = logicalBlockId - d_blocksPerSubjectPrefixSum[subjectIndex];

                const int numGoodCandidates = d_indices_per_subject[subjectIndex];

                if(numGoodCandidates > 0){

                    const int globalCandidateOffset = d_candidates_per_subject_prefixsum[subjectIndex];    
                    
                    if(useSmem){
                        //clear shared memory
                        for(int i = threadIdx.x; i < smemsizefloats; i += blockDim.x){
                            sharedmem[i] = 0;
                        }
                        __syncthreads();
                    }

                    int* const mycounts = (useSmem ? shared_counts : counts + subjectIndex * 4 * msa_weights_row_pitch_floats);
                    float* const myweights = (useSmem ? shared_weights : weights + subjectIndex * 4 * msa_weights_row_pitch_floats);
                    int* const my_coverage = coverage + subjectIndex * msa_weights_row_pitch_floats;

                    //add subject
                    const int subjectColumnsBegin_incl = msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
                    const int subjectColumnsEnd_excl = msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;
                    const int columnsToCheck = msaColumnProperties[subjectIndex].lastColumn_excl;

                    assert(columnsToCheck <= msa_weights_row_pitch_floats);

                    const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
                    const unsigned int* const subject = subjectSequencesData + std::size_t(subjectIndex) * encodedSequencePitchInInts;
                    const char* const subjectQualityScore = subjectQualities + std::size_t(subjectIndex) * qualityPitchInBytes;
                              
                    if(blockIdForThisSubject == 0){
                        for(int i = threadIdx.x; i < subjectLength; i+= blockDim.x){
                            const int columnIndex = subjectColumnsBegin_incl + i;
                            const char base = get((const char*)subject, subjectLength, i);
                            const float weight = canUseQualityScores ? getQualityWeight(subjectQualityScore[i]) : 1.0f;
                            const int rowOffset = int(base) * msa_weights_row_pitch_floats;

                            atomicAdd(mycounts + rowOffset + columnIndex, 1);
                            atomicAdd(myweights + rowOffset + columnIndex, weight);
                            atomicAdd(my_coverage + columnIndex, 1);
                        }
                    }

                    const int* const myShifts = shifts + globalCandidateOffset;
                    const int* const myOverlaps = overlaps + globalCandidateOffset;
                    const int* const myNops = nOps + globalCandidateOffset;

                    const BestAlignment_t* const myAlignmentFlags = bestAlignmentFlags + globalCandidateOffset;
                    const int* const myCandidateLengths = candidateSequencesLength + globalCandidateOffset;

                    const unsigned int* const myCandidateSequencesData = candidateSequencesTransposedData + size_t(globalCandidateOffset);

                    const char* const myCandidateQualities = candidateQualities 
                                                                        + size_t(globalCandidateOffset) * qualityPitchInBytes; 
                                                                        
                    const int* const myIndices = d_indices + globalCandidateOffset;

                    for(int indexInList = threadIdx.x + blockIdForThisSubject * blockDim.x; 
                            indexInList < numGoodCandidates; 
                            indexInList += numBlocksForThisSubject * blockDim.x){

                        const int localCandidateIndex = myIndices[indexInList];
                        const int shift = myShifts[localCandidateIndex];
                        const BestAlignment_t flag = myAlignmentFlags[localCandidateIndex];

                        const int queryLength = myCandidateLengths[localCandidateIndex];
                        const unsigned int* const query = myCandidateSequencesData 
                                + (candidatesAreTransposed ? localCandidateIndex 
                                                            : localCandidateIndex * encodedSequencePitchInInts);

                        const char* const queryQualityScore = myCandidateQualities + std::size_t(localCandidateIndex) * qualityPitchInBytes;

                        const int query_alignment_overlap = myOverlaps[localCandidateIndex];
                        const int query_alignment_nops = myNops[localCandidateIndex];

                        const float overlapweight = calculateOverlapWeight(subjectLength, query_alignment_nops, query_alignment_overlap);

                        assert(overlapweight <= 1.0f);
                        assert(overlapweight >= 0.0f);
                        assert(flag != BestAlignment_t::None);                 // indices should only be pointing to valid alignments

                        const int defaultcolumnoffset = subjectColumnsBegin_incl + shift;

                        const bool isForward = flag == BestAlignment_t::Forward;

                        addASequence2BitToMSA<candidatesAreTransposed>(
                            mycounts,
                            myweights,
                            my_coverage,
                            msa_weights_row_pitch_floats,
                            query, 
                            queryLength, 
                            isForward,
                            defaultcolumnoffset,
                            overlapweight,
                            queryQualityScore,
                            canUseQualityScores,
                            (candidatesAreTransposed ? n_queries : 1)
                        );
                    }

                    __syncthreads();

                    if(useSmem){

                        for(int index = threadIdx.x; index < columnsToCheck; index += blockDim.x){
                            for(int k = 0; k < 4; k++){
                                const int* const srcCounts = shared_counts + k * msa_weights_row_pitch_floats + index;
                                int* const destCounts = counts + 4 * msa_weights_row_pitch_floats * subjectIndex + k * msa_weights_row_pitch_floats + index;
                                const float* const srcWeights = shared_weights + k * msa_weights_row_pitch_floats + index;
                                float* const destWeights = weights + 4 * msa_weights_row_pitch_floats * subjectIndex + k * msa_weights_row_pitch_floats + index;
                                atomicAdd(destCounts ,*srcCounts);
                                atomicAdd(destWeights, *srcWeights);
                            }
                        }

                        __syncthreads();
                    }
                }                
            }
        }
    }



    __global__
    void check_built_msa_kernel(const MSAColumnProperties* __restrict__ msaColumnProperties,
                                const int* __restrict__ counts,
                                const float* __restrict__ weights,
                                const int* __restrict__ d_indices_per_subject,
                                int nSubjects,
                                size_t msa_weights_row_pitch,
                                const bool* __restrict__ canExecute){

        if(*canExecute){

            const size_t msa_weights_row_pitch_floats = msa_weights_row_pitch / sizeof(float);

            for(int subjectIndex = blockIdx.x; subjectIndex < nSubjects; subjectIndex += gridDim.x){
                if(d_indices_per_subject[subjectIndex] > 0){
                    checkBuiltMSA(
                        msaColumnProperties + subjectIndex,
                        counts + 4 * msa_weights_row_pitch_floats * subjectIndex,
                        weights + 4 * msa_weights_row_pitch_floats * subjectIndex,
                        msa_weights_row_pitch_floats,
                        subjectIndex
                    );                      
                }
            }
        }
    }

    template<int BLOCKSIZE, int blocks_per_msa>
    __global__
    void msaFindConsensusKernel(
            const MSAColumnProperties* __restrict__ d_msaColumnProperties,
            const int* __restrict__ d_counts,
            const float* __restrict__ d_weights,
            float* __restrict__ d_support,
            const int* __restrict__ d_coverage,
            float* __restrict__ d_origWeights,
            int* __restrict__ d_origCoverages,
            char* __restrict__ d_consensus,
            const unsigned int* __restrict__ subjectSequencesData,
            const int* __restrict__ d_indices_per_subject,
            int n_subjects,
            int encodedSequencePitchInInts,
            size_t msa_pitch,
            size_t msa_weights_pitch,
            const bool* __restrict__ canExecute){

        static_assert(blocks_per_msa == 1, "");

        if(*canExecute){

            constexpr char A_enc = 0x00;
            constexpr char C_enc = 0x01;
            constexpr char G_enc = 0x02;
            constexpr char T_enc = 0x03;

            // using BlockReduceFloat = cub::BlockReduce<float, BLOCKSIZE>;

            // __shared__ union {
            //     typename BlockReduceFloat::TempStorage floatreduce;
            // } temp_storage;

            // __shared__ float avgCountPerWeight[4];

            auto get = [] (const char* data, int length, int index){
                return getEncodedNuc2Bit((const unsigned int*)data, length, index, [](auto i){return i;});
            };

            auto getSubjectPtr = [&] (int subjectIndex){
                const unsigned int* result = subjectSequencesData + std::size_t(subjectIndex) * encodedSequencePitchInInts;
                return result;
            };

            const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

            const int localBlockId = blockIdx.x % blocks_per_msa;
            //const int n_indices = *d_num_indices;

            //process multiple sequence alignment of each subject
            //for each column in msa, find consensus and support
            for(int subjectIndex = blockIdx.x / blocks_per_msa; subjectIndex < n_subjects; subjectIndex += gridDim.x / blocks_per_msa){
                if(d_indices_per_subject[subjectIndex] > 0){

                    char* const my_consensus = d_consensus + subjectIndex * msa_pitch;
                    float* const my_support = d_support + subjectIndex * msa_weights_pitch_floats;
                    float* const my_orig_weights = d_origWeights + subjectIndex * msa_weights_pitch_floats;
                    int* const my_orig_coverage = d_origCoverages + subjectIndex * msa_weights_pitch_floats;

                    const MSAColumnProperties* const myColumnProperties = d_msaColumnProperties + subjectIndex;

                    const int* const myCounts = d_counts + 4 * msa_weights_pitch_floats * subjectIndex;
                    const float* const myWeights = d_weights + 4 * msa_weights_pitch_floats * subjectIndex;

                    const unsigned int* const anchorData = subjectSequencesData + std::size_t(subjectIndex) * encodedSequencePitchInInts;

                    findConsensusSingleBlock(
                        my_support,
                        my_orig_weights,
                        my_orig_coverage,
                        my_consensus,
                        myColumnProperties, 
                        myCounts,
                        myWeights,      
                        anchorData, 
                        subjectIndex,
                        encodedSequencePitchInInts, 
                        msa_pitch,
                        msa_weights_pitch_floats
                    );
                }
            }
        }
    }


/*
        This kernel inspects a msa and identifies candidates which could originate
        from a different genome region than the subject.

        the output element d_shouldBeKept[i] indicates whether
        the candidate referred to by d_indices[i] should remain in the msa
    */


    template<int BLOCKSIZE>
    __global__
    void msa_findCandidatesOfDifferentRegion_kernel(
            int* __restrict__ d_newIndices,
            int* __restrict__ d_newNumIndicesPerSubject,
            int* __restrict__ d_newNumIndices,

            MSAPointers d_msapointers,
            AlignmentResultPointers d_alignmentresultpointers,
            ReadSequencesPointers d_sequencePointers,
            bool* __restrict__ d_shouldBeKept,
            const int* __restrict__ d_candidates_per_subject_prefixsum,
            int n_subjects,
            int n_candidates,
            int encodedSequencePitchInInts,
            size_t msa_pitch,
            size_t msa_weights_pitch,
            const int* __restrict__ d_indices,
            const int* __restrict__ d_indices_per_subject,
            int dataset_coverage,
            const bool* __restrict__ canExecute,
            const unsigned int* d_readids,
            bool debug = false){

        if(*canExecute){

            // if(threadIdx.x + blockIdx.x * blockDim.x == 0){
            //     printf("msa_findCandidatesOfDifferentRegion_kernel\n");
            // }

            auto getNumBytes = [] (int sequencelength){
                return sizeof(unsigned int) * getEncodedNumInts2Bit(sequencelength);
            };

            auto get = [] (const char* data, int length, int index){
                return getEncodedNuc2Bit((const unsigned int*)data, length, index, [](auto i){return i;});
            };

            auto getSubjectPtr = [&] (int subjectIndex){
                const unsigned int* result = d_sequencePointers.subjectSequencesData + std::size_t(subjectIndex) * encodedSequencePitchInInts;
                return result;
            };

            auto getCandidatePtr = [&] (int candidateIndex){
                const unsigned int* result = d_sequencePointers.candidateSequencesData + std::size_t(candidateIndex) * encodedSequencePitchInInts;
                return result;
            };

            auto getSubjectLength = [&] (int subjectIndex){
                const int length = d_sequencePointers.subjectSequencesLength[subjectIndex];
                return length;
            };

            auto getCandidateLength = [&] (int candidateIndex){
                const int length = d_sequencePointers.candidateSequencesLength[candidateIndex];
                return length;
            };

            auto is_significant_count = [](int count, int coverage){
                /*if(ceil(estimatedErrorrate * coverage)*2 <= count ){
                    return true;
                }
                return false;*/
                if(int(coverage * 0.3f) <= count)
                    return true;
                return false;

            };

            const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);
            //const char index_to_base[4]{'A','C','G','T'};

            constexpr char A_enc = 0x00;
            constexpr char C_enc = 0x01;
            constexpr char G_enc = 0x02;
            constexpr char T_enc = 0x03;

            auto to_nuc = [](char c){
                switch(c){
                case A_enc: return 'A';
                case C_enc: return 'C';
                case G_enc: return 'G';
                case T_enc: return 'T';
                default: return 'F';
                }
            };

            using BlockReduceBool = cub::BlockReduce<bool, BLOCKSIZE>;
            using BlockReduceInt2 = cub::BlockReduce<int2, BLOCKSIZE>;

            __shared__ union{
                typename BlockReduceBool::TempStorage boolreduce;
                typename BlockReduceInt2::TempStorage int2reduce;
            } temp_storage;

            __shared__ bool broadcastbufferbool;
            __shared__ int broadcastbufferint4[4];
            __shared__ int totalIndices;
            __shared__ int counts[1];

            if(threadIdx.x == 0){
                totalIndices = 0;
            }
            __syncthreads();

            for(int subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){

                const int globalOffset = d_candidates_per_subject_prefixsum[subjectIndex];

                const int* myIndices = d_indices + globalOffset;
                const int myNumIndices = d_indices_per_subject[subjectIndex];

                int* const myNewIndicesPtr = d_newIndices + globalOffset;
                int* const myNewNumIndicesPerSubjectPtr = d_newNumIndicesPerSubject + subjectIndex;

                if(debug && threadIdx.x == 0){
                    //printf("myNumIndices %d\n", myNumIndices);
                }

                if(myNumIndices > 0){

                    const unsigned int* const subjectptr = getSubjectPtr(subjectIndex);
                    const int subjectLength = getSubjectLength(subjectIndex);

                    const char* myConsensus = d_msapointers.consensus + subjectIndex * msa_pitch;

                    const int subjectColumnsBegin_incl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
                    const int subjectColumnsEnd_excl = d_msapointers.msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;

                    //check if subject and consensus differ at at least one position

                    bool hasMismatchToConsensus = false;

                    for(int pos = threadIdx.x; pos < subjectLength && !hasMismatchToConsensus; pos += blockDim.x){
                        const int column = subjectColumnsBegin_incl + pos;
                        const char consbase = myConsensus[column];
                        const char subjectbase = to_nuc(get((const char*)subjectptr, subjectLength, pos));

                        hasMismatchToConsensus |= (consbase != subjectbase);
                    }

                    hasMismatchToConsensus = BlockReduceBool(temp_storage.boolreduce).Reduce(hasMismatchToConsensus, [](auto l, auto r){return l || r;});

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

                        const int* const myCountsA = d_msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 0 * msa_weights_pitch_floats;
                        const int* const myCountsC = d_msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 1 * msa_weights_pitch_floats;
                        const int* const myCountsG = d_msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 2 * msa_weights_pitch_floats;
                        const int* const myCountsT = d_msapointers.counts + 4 * msa_weights_pitch_floats * subjectIndex + 3 * msa_weights_pitch_floats;

                        const float* const myWeightsA = d_msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 0 * msa_weights_pitch_floats;
                        const float* const myWeightsC = d_msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 1 * msa_weights_pitch_floats;
                        const float* const myWeightsG = d_msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 2 * msa_weights_pitch_floats;
                        const float* const myWeightsT = d_msapointers.weights + 4 * msa_weights_pitch_floats * subjectIndex + 3 * msa_weights_pitch_floats;

                        const float* const mySupport = d_msapointers.support + subjectIndex * msa_weights_pitch_floats;

                        for(int columnindex = subjectColumnsBegin_incl + threadIdx.x; columnindex < subjectColumnsEnd_excl && !foundColumn; columnindex += blockDim.x){
                            int counts[4];
                            counts[0] = myCountsA[columnindex];
                            counts[1] = myCountsC[columnindex];
                            counts[2] = myCountsG[columnindex];
                            counts[3] = myCountsT[columnindex];

                            float weights[4];
                            weights[0] = myWeightsA[columnindex];
                            weights[1] = myWeightsC[columnindex];
                            weights[2] = myWeightsG[columnindex];
                            weights[3] = myWeightsT[columnindex];

                            const float support = mySupport[columnindex];

                            const char consbase = myConsensus[columnindex];
                            consindex = -1;

                            switch(consbase){
                                case 'A': consindex = 0;break;
                                case 'C': consindex = 1;break;
                                case 'G': consindex = 2;break;
                                case 'T': consindex = 3;break;
                            }

                            // char consensusByCount = 'A';
                            // int maxCount = counts[0];
                            // if(counts[1] > maxCount){
                            //     consensusByCount = 'C';
                            //     maxCount = counts[1];
                            // }
                            // if(counts[2] > maxCount){
                            //     consensusByCount = 'G';
                            //     maxCount = counts[2];
                            // }
                            // if(counts[3] > maxCount){
                            //     consensusByCount = 'T';
                            //     maxCount = counts[3];
                            // }
                            //
                            // if(consbase != consensusByCount){
                            //     printf("bycounts %c %.6f %.6f %.6f %.6f,\nbyweight %c %.6f %.6f %.6f %.6f\n\n",
                            //             consensusByCount, float(counts[0]), float(counts[1]), float(counts[2]), float(counts[3]),
                            //             consbase, weights[0], weights[1], weights[2], weights[3]);
                            // }

                            //find out if there is a non-consensus base with significant coverage
                            int significantBaseIndex = -1;

                            #pragma unroll
                            for(int i = 0; i < 4; i++){
                                if(i != consindex){
                                    //const bool significant = is_significant_count(counts[i], dataset_coverage);
                                    //const int columnCoverage = counts[0] + counts[1] +counts[2] + counts[3];

                                    const bool significant = is_significant_count(counts[i], dataset_coverage);

                                    //const bool significant = weights[i] / support >= 0.5f;

                                    significantBaseIndex = significant ? i : significantBaseIndex;
                                }
                            }

                            if(significantBaseIndex != -1){
                                foundColumn = true;
                                col = columnindex;
                                foundBaseIndex = significantBaseIndex;

                                // if(debug){
                                //     printf("found col %d, baseIndex %d\n", col, foundBaseIndex);
                                // }
                            }
                        }

                        int2 packed{col, foundBaseIndex};
                        //find packed value with smallest col
                        packed = BlockReduceInt2(temp_storage.int2reduce).Reduce(packed, [](auto l, auto r){
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
                                broadcastbufferint4[2] = to_nuc(packed.y);
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

                        // if(debug && threadIdx.x == 0 /*&& d_readids[subjectIndex] == 207*/){
                        //     printf("reduced: found a column: %d, found col %d, found base %c, baseIndex %d\n", foundColumn, col, foundBase, foundBaseIndex);
                        // }

                        if(foundColumn){

                            //compare found base to original base
                            const char originalbase = to_nuc(get((const char*)subjectptr, subjectLength, col - subjectColumnsBegin_incl));

                            /*int counts[4];

                            counts[0] = myCountsA[col];
                            counts[1] = myCountsC[col];
                            counts[2] = myCountsG[col];
                            counts[3] = myCountsT[col];*/

                            auto discard_rows = [&](bool keepMatching){

                                for(int k = threadIdx.x; k < myNumIndices; k += blockDim.x){
                                    const int localCandidateIndex = myIndices[k];
                                    const int globalCandidateIndex = globalOffset + localCandidateIndex;
                                    const unsigned int* const candidateptr = getCandidatePtr(globalCandidateIndex);
                                    const int candidateLength = getCandidateLength(globalCandidateIndex);
                                    const int shift = d_alignmentresultpointers.shifts[globalCandidateIndex];
                                    const BestAlignment_t alignmentFlag = d_alignmentresultpointers.bestAlignmentFlags[globalCandidateIndex];

                                    //check if row is affected by column col
                                    const int row_begin_incl = subjectColumnsBegin_incl + shift;
                                    const int row_end_excl = row_begin_incl + candidateLength;
                                    const bool notAffected = (col < row_begin_incl || row_end_excl <= col);
                                    char base = 'F';
                                    if(!notAffected){
                                        if(alignmentFlag == BestAlignment_t::Forward){
                                            base = to_nuc(get((const char*)candidateptr, candidateLength, (col - row_begin_incl)));
                                        }else{
                                            assert(alignmentFlag == BestAlignment_t::ReverseComplement); //all candidates of MSA must not have alignmentflag None
                                            const char forwardbaseEncoded = get((const char*)candidateptr, candidateLength, row_end_excl-1 - col);
                                            base = to_nuc((~forwardbaseEncoded & 0x03));
                                        }
                                    }

                                    if(notAffected || (!(keepMatching ^ (base == foundBase)))){
                                        d_shouldBeKept[globalOffset + k] = true; //same region
                                    }else{
                                        d_shouldBeKept[globalOffset + k] = false; //different region
                                    }
                                }
    #if 1
                                //check that no candidate which should be removed has very good alignment.
                                //if there is such a candidate, none of the candidates will be removed.
                                bool veryGoodAlignment = false;
                                for(int k = threadIdx.x; k < myNumIndices && !veryGoodAlignment; k += blockDim.x){
                                    if(!d_shouldBeKept[globalOffset + k]){
                                        const int localCandidateIndex = myIndices[k];
                                        const int globalCandidateIndex = globalOffset + localCandidateIndex;
                                        const int nOps = d_alignmentresultpointers.nOps[globalCandidateIndex];
                                        const int overlapsize = d_alignmentresultpointers.overlaps[globalCandidateIndex];
                                        const float overlapweight = calculateOverlapWeight(subjectLength, nOps, overlapsize);
                                        assert(overlapweight <= 1.0f);
                                        assert(overlapweight >= 0.0f);

                                        if(overlapweight >= 0.90f){
                                            veryGoodAlignment = true;
                                        }
                                    }
                                }

                                veryGoodAlignment = BlockReduceBool(temp_storage.boolreduce).Reduce(veryGoodAlignment, [](auto l, auto r){return l || r;});

                                if(threadIdx.x == 0){
                                    broadcastbufferbool = veryGoodAlignment;
                                }
                                __syncthreads();

                                veryGoodAlignment = broadcastbufferbool;

                                if(veryGoodAlignment){
                                    for(int k = threadIdx.x; k < myNumIndices; k += blockDim.x){
                                        d_shouldBeKept[globalOffset + k] = true;
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
                                        keep = d_shouldBeKept[globalOffset + k];
                                    }                               
                        
                                    if(keep){
                                        cg::coalesced_group g = cg::coalesced_threads();
                                        int outputPos;
                                        if (g.thread_rank() == 0) {
                                            outputPos = atomicAdd(&counts[0], g.size());
                                            atomicAdd(&totalIndices, g.size());
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
                                totalIndices += myNumIndices;
                            }
                        }

                    }else{
                        //no mismatch between consensus and subject

                        //remove no candidate
                        for(int k = threadIdx.x; k < myNumIndices; k += blockDim.x){
                            myNewIndicesPtr[k] = myIndices[k];
                        }
                        if(threadIdx.x == 0){
                            *myNewNumIndicesPerSubjectPtr = myNumIndices;
                            totalIndices += myNumIndices;
                        }
                    }
                }else{
                    ; //nothing to do if there are no candidates in msa
                }
            }

            __syncthreads();

            if(threadIdx.x == 0){
                atomicAdd(d_newNumIndices, totalIndices);
            }
        }
    }










    //####################   KERNEL DISPATCH   ####################
    
    

    void call_msa_init_kernel_async_exp(
            MSAColumnProperties* d_msaColumnProperties,
            const int* d_alignmentShifts,
            const BestAlignment_t* d_bestAlignmentFlags,
            const int* d_anchorLengths,
            const int* d_candidateLengths,
            const int* d_indices,
            const int* d_indices_per_subject,
            const int* d_candidatesPerSubjectPrefixSum,
            int n_subjects,
            int n_candidates,
            const bool* d_canExecute,
            cudaStream_t stream,
            KernelLaunchHandle& handle){


    	constexpr int blocksize = 128;
    	const std::size_t smem = 0;

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::MSAInitExp);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    	    #define getProp(blocksize) { \
                KernelLaunchConfig kernelLaunchConfig; \
                kernelLaunchConfig.threads_per_block = (blocksize); \
                kernelLaunchConfig.smem = 0; \
                KernelProperties kernelProperties; \
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                    msaInitKernel<(blocksize)>, \
                            kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
                mymap[kernelLaunchConfig] = kernelProperties; \
            }

    		getProp(32);
    		getProp(64);
    		getProp(96);
    		getProp(128);
    		getProp(160);
    		getProp(192);
    		getProp(224);
    		getProp(256);

    		const auto& kernelProperties = mymap[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

    		handle.kernelPropertiesMap[KernelId::MSAInitExp] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    	}

    	dim3 block(blocksize, 1, 1);
        dim3 grid(std::min(max_blocks_per_device, n_subjects), 1, 1);

		#define mycall(blocksize) msaInitKernel<(blocksize)> \
                <<<grid, block, 0, stream>>>( \
                    d_msaColumnProperties, \
                    d_alignmentShifts, \
                    d_bestAlignmentFlags, \
                    d_anchorLengths, \
                    d_candidateLengths, \
                    d_indices, \
                    d_indices_per_subject, \
                    d_candidatesPerSubjectPrefixSum, \
                    n_subjects, \
                    d_canExecute); CUERR;

    	switch(blocksize) {
    	case 1: mycall(1); break;
    	case 32: mycall(32); break;
    	case 64: mycall(64); break;
    	case 96: mycall(96); break;
    	case 128: mycall(128); break;
    	case 160: mycall(160); break;
    	case 192: mycall(192); break;
    	case 224: mycall(224); break;
    	case 256: mycall(256); break;
    	default: mycall(256); break;
    	}

    		#undef mycall


    }

    void call_msa_update_properties_kernel_async(
                    MSAPointers d_msapointers,
                    const int* d_indices_per_subject,
                    int n_subjects,
                    size_t msa_weights_pitch,
                    const bool* d_canExecute,
                    cudaStream_t stream,
                    KernelLaunchHandle& handle){

    	const int blocksize = 128;
    	const std::size_t smem = 0;

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::MSAUpdateProperties);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    		KernelLaunchConfig kernelLaunchConfig;
    		kernelLaunchConfig.threads_per_block = (blocksize);
    		kernelLaunchConfig.smem = 0;
    		KernelProperties kernelProperties;
    		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM,
            					msa_update_properties_kernel,
            					kernelLaunchConfig.threads_per_block,
                                kernelLaunchConfig.smem); CUERR;
    		mymap[kernelLaunchConfig] = kernelProperties;

    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

    		handle.kernelPropertiesMap[KernelId::MSAUpdateProperties] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    	}

    	dim3 block(blocksize, 1, 1);
    	dim3 grid(std::min(max_blocks_per_device, n_subjects), 1, 1);

        msa_update_properties_kernel<<<grid, block, 0, stream>>>(d_msapointers,
                                                                d_indices_per_subject,
                                                                msa_weights_pitch,
                                                                n_subjects,
                                                                d_canExecute); CUERR;




    }


    

    

    template<MemoryType memType>
    void call_msaAddSequencesKernelSingleBlock_async(
            const MSAColumnProperties* d_msaColumnProperties,
            int* d_coverage,
            int* d_counts,
            float* d_weights,
            const int* d_overlaps,
            const int* d_shifts,
            const int* d_nOps,
            const BestAlignment_t* d_bestAlignmentFlags,
            const unsigned int* d_subjectSequencesData,
            const unsigned int* d_candidateSequencesTransposedData,
            const int* d_subjectSequencesLength,
            const int* d_candidateSequencesLength,
            const char* d_subjectQualities,
            const char* d_candidateQualities,
            const int* d_candidates_per_subject_prefixsum,
            const int* d_indices,
            const int* d_indices_per_subject,
            int n_subjects,
            int n_queries,
            bool canUseQualityScores,
            int encodedSequencePitchInInts,
            size_t qualityPitchInBytes,
            size_t msa_row_pitch,
            size_t msa_weights_row_pitch,
            const bool* d_canExecute,
            cudaStream_t stream,
            KernelLaunchHandle& handle){

        const KernelId kernelId = (memType == MemoryType::Global
                     ? KernelId::MSAAddSequencesGlobalSingleBlock
                     : KernelId::MSAAddSequencesSharedSingleBlock);

        const size_t msa_weights_row_pitch_floats = msa_weights_row_pitch / sizeof(float);

        constexpr int blocksize = 128;
        
        const std::size_t smem = (memType == MemoryType::Global ? 0
                                    : sizeof(float) * 4 * msa_weights_row_pitch_floats // weights
                                        + sizeof(int) * 4 * msa_weights_row_pitch_floats // counts
                                        + sizeof(int) * msa_weights_row_pitch_floats); // coverages

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(kernelId);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    	    #define getProp(blocksize) { \
                    KernelLaunchConfig kernelLaunchConfig; \
                    kernelLaunchConfig.threads_per_block = (blocksize); \
                    kernelLaunchConfig.smem = smem; \
                    KernelProperties kernelProperties; \
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                        msa_add_sequences_kernel_singleblock<memType>, \
                                kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
                    mymap[kernelLaunchConfig] = kernelProperties; \
            }

            getProp(1);
    		getProp(32);
    		getProp(64);
    		getProp(96);
    		getProp(128);
    		getProp(160);
    		getProp(192);
    		getProp(224);
    		getProp(256);

    		const auto& kernelProperties = mymap[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM * 2;

    		handle.kernelPropertiesMap[kernelId] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM * 2;
    		//std::cout << max_blocks_per_device << " = " << handle.deviceProperties.multiProcessorCount << " * " << kernelProperties.max_blocks_per_SM << std::endl;
    	}

    	dim3 block(blocksize, 1, 1);
        dim3 grid(std::min(n_subjects, max_blocks_per_device), 1, 1);

        // void msa_add_sequences_kernel_singleblock(
        //     int* __restrict__ coverage,
        //     MSAColumnProperties* __restrict__ msaColumnProperties,
        //     int* __restrict__ counts,
        //     float* __restrict__ weights,
        //     const int* __restrict__ overlaps,
        //     const int* __restrict__ shifts,
        //     const int* __restrict__ nOps,
        //     const BestAlignment_t* __restrict__ bestAlignmentFlags,
        //     const unsigned int* __restrict__ subjectSequencesData,
        //     const unsigned int* __restrict__ candidateSequencesTransposedData,
        //     const int* __restrict__ subjectSequencesLength,
        //     const int* __restrict__ candidateSequencesLength,
        //     const char* __restrict__ subjectQualities,
        //     const char* __restrict__ candidateQualities,
        //     const int* __restrict__ d_candidates_per_subject_prefixsum,
        //     const int* __restrict__ d_indices,
        //     const int* __restrict__ d_indices_per_subject,
        //     int n_subjects,
        //     int n_queries,
        //     bool canUseQualityScores,
        //     int encodedSequencePitchInInts,
        //     size_t qualityPitchInBytes,
        //     size_t msa_row_pitch,
        //     size_t msa_weights_row_pitch_floats,
        //     const bool* __restrict__ canExecute){

        msa_add_sequences_kernel_singleblock<memType>
                <<<grid, block, smem, stream>>>(
            d_coverage,
            d_msaColumnProperties,
            d_counts,
            d_weights,
            d_overlaps,
            d_shifts,
            d_nOps,
            d_bestAlignmentFlags,
            d_subjectSequencesData,
            d_candidateSequencesTransposedData,
            d_subjectSequencesLength,
            d_candidateSequencesLength,
            d_subjectQualities,
            d_candidateQualities,
            d_candidates_per_subject_prefixsum,
            d_indices,
            d_indices_per_subject,
            n_subjects,
            n_queries,
            canUseQualityScores,
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            msa_row_pitch,
            msa_weights_row_pitch_floats,
            d_canExecute
        ); CUERR;

    }



    template<MemoryType memType>
    void call_msaAddSequencesKernelMultiBlock_async(
                const MSAColumnProperties* d_msaColumnProperties,
                int* d_coverage,
                int* d_counts,
                float* d_weights,
                const int* d_overlaps,
                const int* d_shifts,
                const int* d_nOps,
                const BestAlignment_t* d_bestAlignmentFlags,
                const unsigned int* d_subjectSequencesData,
                const unsigned int* d_candidateSequencesTransposedData,
                const int* d_subjectSequencesLength,
                const int* d_candidateSequencesLength,
                const char* d_subjectQualities,
                const char* d_candidateQualities,
    			const int* d_candidates_per_subject_prefixsum,
    			const int* d_indices,
    			const int* d_indices_per_subject,
    			int n_subjects,
    			int n_queries,
    			bool canUseQualityScores,
                int encodedSequencePitchInInts,
    			size_t qualityPitchInBytes,
    			size_t msa_row_pitch,
                size_t msa_weights_row_pitch,
                const bool* d_canExecute,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle){

        const KernelId kernelId = (memType == MemoryType::Global
                     ? KernelId::MSAAddSequencesGlobalMultiBlock
                     : KernelId::MSAAddSequencesSharedMultiBlock);

        // set counts, weights, and coverages to zero for subjects with valid indices
        generic_kernel<<<n_subjects, 128, 0, stream>>>([=] __device__ (){
            if(*d_canExecute){
                for(int subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
                    if(d_indices_per_subject[subjectIndex] > 0){
                        const size_t msa_weights_pitch_floats = msa_weights_row_pitch / sizeof(float);

                        int* const mycounts = d_counts + msa_weights_pitch_floats * 4 * subjectIndex;
                        float* const myweights = d_weights + msa_weights_pitch_floats * 4 * subjectIndex;
                        int* const mycoverages = d_coverage + msa_weights_pitch_floats * subjectIndex;

                        for(int column = threadIdx.x; column < msa_weights_pitch_floats * 4; column += blockDim.x){
                            mycounts[column] = 0;
                            myweights[column] = 0;
                        }

                        for(int column = threadIdx.x; column < msa_weights_pitch_floats; column += blockDim.x){
                            mycoverages[column] = 0;
                        }
                    }
                }
            }
        }); CUERR;

        const size_t msa_weights_row_pitch_floats = msa_weights_row_pitch / sizeof(float);

        constexpr int blocksize = 128;
        
        const std::size_t smem = (memType == MemoryType::Global ? 0
                                    : sizeof(float) * 4 * msa_weights_row_pitch_floats // weights
                                        + sizeof(int) * 4 * msa_weights_row_pitch_floats); // counts

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(kernelId);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    	    #define getProp(blocksize) { \
                    KernelLaunchConfig kernelLaunchConfig; \
                    kernelLaunchConfig.threads_per_block = (blocksize); \
                    kernelLaunchConfig.smem = smem; \
                    KernelProperties kernelProperties; \
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                        msa_add_sequences_kernel_multiblock<memType>, \
                                kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
                    mymap[kernelLaunchConfig] = kernelProperties; \
            }

            getProp(1);
    		getProp(32);
    		getProp(64);
    		getProp(96);
    		getProp(128);
    		getProp(160);
    		getProp(192);
    		getProp(224);
    		getProp(256);

    		const auto& kernelProperties = mymap[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM * 2;

    		handle.kernelPropertiesMap[kernelId] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM * 2;
    		//std::cout << max_blocks_per_device << " = " << handle.deviceProperties.multiProcessorCount << " * " << kernelProperties.max_blocks_per_SM << std::endl;
        }
        
        int* d_blocksPerSubjectPrefixSum;
        cubCachingAllocator.DeviceAllocate((void**)&d_blocksPerSubjectPrefixSum, sizeof(int) * (n_subjects+1), stream);  CUERR;

        // calculate blocks per subject prefixsum
        auto getBlocksPerSubject = [=] __device__ (int indices_for_subject){
            return SDIV(indices_for_subject, blocksize);
        };
        cub::TransformInputIterator<int,decltype(getBlocksPerSubject), const int*>
            d_blocksPerSubject(d_indices_per_subject,
                          getBlocksPerSubject);

        void* tempstorage = nullptr;
        size_t tempstoragesize = 0;

        cub::DeviceScan::InclusiveSum(nullptr,
                    tempstoragesize,
                    d_blocksPerSubject,
                    d_blocksPerSubjectPrefixSum+1,
                    n_subjects,
                    stream); CUERR;

        cubCachingAllocator.DeviceAllocate((void**)&tempstorage, tempstoragesize, stream);  CUERR;

        cub::DeviceScan::InclusiveSum(tempstorage,
                    tempstoragesize,
                    d_blocksPerSubject,
                    d_blocksPerSubjectPrefixSum+1,
                    n_subjects,
                    stream); CUERR;

        cubCachingAllocator.DeviceFree(tempstorage);  CUERR;

        call_set_kernel_async(d_blocksPerSubjectPrefixSum,
                                0,
                                0,
                                stream);

    	dim3 block(blocksize, 1, 1);
        dim3 grid(std::min(n_subjects, max_blocks_per_device), 1, 1);

        msa_add_sequences_kernel_multiblock<memType>
                <<<grid, block, smem, stream>>>(
            d_coverage,
            d_msaColumnProperties,
            d_counts,
            d_weights,
            d_overlaps,
            d_shifts,
            d_nOps,
            d_bestAlignmentFlags,
            d_subjectSequencesData,
            d_candidateSequencesTransposedData,
            d_subjectSequencesLength,
            d_candidateSequencesLength,
            d_subjectQualities,
            d_candidateQualities,
            d_candidates_per_subject_prefixsum,
            d_indices,
            d_indices_per_subject,
            d_blocksPerSubjectPrefixSum,
            n_subjects,
            n_queries,
            canUseQualityScores,
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            msa_row_pitch,
            msa_weights_row_pitch_floats,
            d_canExecute
        ); CUERR;

        cubCachingAllocator.DeviceFree(d_blocksPerSubjectPrefixSum);
    }


    void call_msa_add_sequences_kernel_implicit_async(
                const MSAColumnProperties* d_msaColumnProperties,
                int* d_coverage,
                int* d_counts,
                float* d_weights,
                const int* d_overlaps,
                const int* d_shifts,
                const int* d_nOps,
                const BestAlignment_t* d_bestAlignmentFlags,
                const unsigned int* d_subjectSequencesData,
                const unsigned int* d_candidateSequencesTransposedData,
                const int* d_subjectSequencesLength,
                const int* d_candidateSequencesLength,
                const char* d_subjectQualities,
                const char* d_candidateQualities,
    			const int* d_candidates_per_subject_prefixsum,
    			const int* d_indices,
    			const int* d_indices_per_subject,
    			int n_subjects,
    			int n_queries,
    			bool canUseQualityScores,
                int encodedSequencePitchInInts,
    			size_t qualityPitchInBytes,
    			size_t msa_row_pitch,
                size_t msa_weights_row_pitch,
                const bool* d_canExecute,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle){

        //std::cout << n_subjects << " " << *h_num_indices << " " << n_queries << std::endl;

        constexpr MemoryType memType = MemoryType::Shared;

#if 0


        call_msaAddSequencesKernelMultiBlock_async<memType>(
            d_msaColumnProperties,
            d_coverage,
            d_counts,
            d_weights,
            d_overlaps,
            d_shifts,
            d_nOps,
            d_bestAlignmentFlags,
            d_subjectSequencesData,
            d_candidateSequencesTransposedData,
            d_subjectSequencesLength,
            d_candidateSequencesLength,
            d_subjectQualities,
            d_candidateQualities,
            d_candidates_per_subject_prefixsum,
            d_indices,
            d_indices_per_subject,
            n_subjects,
            n_queries,
            canUseQualityScores,
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            msa_row_pitch,
            msa_weights_row_pitch,
            d_canExecute,
            stream,
            handle
        );

        check_built_msa_kernel<<<n_subjects, 128, 0, stream>>>(
            d_msaColumnProperties,
            d_counts,
            d_weights,
            d_indices_per_subject,
            n_subjects,
            msa_weights_row_pitch,
            d_canExecute
        ); CUERR;

#else 

        call_msaAddSequencesKernelSingleBlock_async<memType>(
            d_msaColumnProperties,
            d_coverage,
            d_counts,
            d_weights,
            d_overlaps,
            d_shifts,
            d_nOps,
            d_bestAlignmentFlags,
            d_subjectSequencesData,
            d_candidateSequencesTransposedData,
            d_subjectSequencesLength,
            d_candidateSequencesLength,
            d_subjectQualities,
            d_candidateQualities,
            d_candidates_per_subject_prefixsum,
            d_indices,
            d_indices_per_subject,
            n_subjects,
            n_queries,
            canUseQualityScores,
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            msa_row_pitch,
            msa_weights_row_pitch,
            d_canExecute,
            stream,
            handle
        );


#endif

        

    }


    void call_msaFindConsensusKernel_async(
                            const MSAColumnProperties* __restrict__ d_msaColumnProperties,
                            const int* __restrict__ d_counts,
                            const float* __restrict__ d_weights,
                            float* __restrict__ d_support,
                            const int* __restrict__ d_coverage,
                            float* __restrict__ d_origWeights,
                            int* __restrict__ d_origCoverages,
                            char* __restrict__ d_consensus,
                            const unsigned int* __restrict__ subjectSequencesData,
                            const int* d_indices_per_subject,
                            int n_subjects,
                            int encodedSequencePitchInInts,
                            size_t msa_pitch,
                            size_t msa_weights_pitch,
                            const bool* d_canExecute,
                            cudaStream_t stream,
                            KernelLaunchHandle& handle){

        constexpr int blocksize = 128;
        constexpr int blocks_per_msa = 1;
        static_assert(blocks_per_msa == 1, "");

        const std::size_t smem = 0;

        int max_blocks_per_device = 1;

        KernelLaunchConfig kernelLaunchConfig;
        kernelLaunchConfig.threads_per_block = blocksize;
        kernelLaunchConfig.smem = smem;

        auto iter = handle.kernelPropertiesMap.find(KernelId::MSAFindConsensus);
        if(iter == handle.kernelPropertiesMap.end()){

            std::map<KernelLaunchConfig, KernelProperties> mymap;

            #define getProp(blocksize) { \
                KernelLaunchConfig kernelLaunchConfig; \
                kernelLaunchConfig.threads_per_block = (blocksize); \
                kernelLaunchConfig.smem = 0; \
                KernelProperties kernelProperties; \
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                                                                msaFindConsensusKernel<blocksize, blocks_per_msa>, \
                                                                kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
                mymap[kernelLaunchConfig] = kernelProperties; \
            }

            getProp(32);
            getProp(64);
            getProp(96);
            getProp(128);
            getProp(160);
            getProp(192);
            getProp(224);
            getProp(256);

            const auto& kernelProperties = mymap[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

            handle.kernelPropertiesMap[KernelId::MSAFindConsensus] = std::move(mymap);

            #undef getProp
        }else{
            std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
            const KernelProperties& kernelProperties = map[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

            // std::cerr << max_blocks_per_device 
            //         << " = " << handle.deviceProperties.multiProcessorCount << " * " << kernelProperties.max_blocks_per_SM << "\n";
        }
  #if 0      
        //std::cerr << grid.x << " " << max_blocks_per_device << " " << n_subjects << "\n";

        #define launch(blocksize) \
            dim3 block((blocksize), 1, 1); \
            dim3 grid(std::min(max_blocks_per_device, n_subjects), 1, 1); \
            msaFindConsensusKernel<(blocksize), blocks_per_msa><<<grid, block, 0, stream>>>( \
                                                                d_msapointers, \
                                                                d_sequencePointers, \
                                                                d_indices_per_subject, \
                                                                n_subjects, \
                                                                encodedSequencePitchInInts, \
                                                                msa_pitch, \
                                                                msa_weights_pitch, \
                                                                d_canExecute); CUERR;

        launch(128);

        #undef launch
  #endif 

        dim3 block(blocksize, 1, 1);
        dim3 grid(n_subjects, 1, 1);

        msaFindConsensusKernel<(blocksize), blocks_per_msa>
                <<<grid, block, 0, stream>>>(
            d_msaColumnProperties,
            d_counts,
            d_weights,
            d_support,
            d_coverage,
            d_origWeights,
            d_origCoverages,
            d_consensus,
            subjectSequencesData,
            d_indices_per_subject,
            n_subjects,
            encodedSequencePitchInInts,
            msa_pitch,
            msa_weights_pitch,
            d_canExecute
        ); CUERR;
    }




    void call_msa_findCandidatesOfDifferentRegion_kernel_async(
                int* d_newIndices,
                int* d_newIndicesPerSubject,
                int* d_newNumIndices,
                MSAPointers d_msapointers,
                AlignmentResultPointers d_alignmentresultpointers,
                ReadSequencesPointers d_sequencePointers,
                bool* d_shouldBeKept,
                const int* d_candidates_per_subject_prefixsum,
                int n_subjects,
                int n_candidates,
                int encodedSequencePitchInInts,
                size_t msa_pitch,
                size_t msa_weights_pitch,
                const int* d_indices,
                const int* d_indices_per_subject,
                int dataset_coverage,
                const bool* d_canExecute,
    			cudaStream_t stream,
    			KernelLaunchHandle& handle,
                const unsigned int* d_readids,
                bool debug){

        cudaMemsetAsync(d_newNumIndices, 0, sizeof(int), stream); CUERR;
        cudaMemsetAsync(d_newIndicesPerSubject, 0, sizeof(int) * n_subjects, stream); CUERR;


    	constexpr int max_block_size = 256;
    	const int blocksize = 256;
    	const std::size_t smem = 0;

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::MSAFindCandidatesOfDifferentRegion);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    	    #define getProp(blocksize) { \
            		KernelLaunchConfig kernelLaunchConfig; \
            		kernelLaunchConfig.threads_per_block = (blocksize); \
            		kernelLaunchConfig.smem = 0; \
            		KernelProperties kernelProperties; \
            		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
            					msa_findCandidatesOfDifferentRegion_kernel<(blocksize)>, \
            					kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem); CUERR; \
            		mymap[kernelLaunchConfig] = kernelProperties; \
            }

            getProp(1);
    		getProp(32);
    		getProp(64);
    		getProp(96);
    		getProp(128);
    		getProp(160);
    		getProp(192);
    		getProp(224);
    		getProp(256);

    		const auto& kernelProperties = mymap[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;

    		handle.kernelPropertiesMap[KernelId::MSAFindCandidatesOfDifferentRegion] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    	}

    	dim3 block(blocksize, 1, 1);
    	dim3 grid(std::min(max_blocks_per_device, n_subjects));

    		#define mycall(blocksize) msa_findCandidatesOfDifferentRegion_kernel<(blocksize)> \
                <<<grid, block, 0, stream>>>( \
                    d_newIndices, \
                    d_newIndicesPerSubject, \
                    d_newNumIndices, \
                    d_msapointers, \
                    d_alignmentresultpointers, \
                    d_sequencePointers, \
                    d_shouldBeKept, \
                    d_candidates_per_subject_prefixsum, \
                    n_subjects, \
                    n_candidates, \
                    encodedSequencePitchInInts, \
                    msa_pitch, \
                    msa_weights_pitch, \
                    d_indices, \
                    d_indices_per_subject, \
                    dataset_coverage, \
                    d_canExecute, \
                    d_readids, \
                    debug); CUERR;

    	assert(blocksize > 0 && blocksize <= max_block_size);

    	switch(blocksize) {
        case 1: mycall(1); break;
    	case 32: mycall(32); break;
    	case 64: mycall(64); break;
    	case 96: mycall(96); break;
    	case 128: mycall(128); break;
    	case 160: mycall(160); break;
    	case 192: mycall(192); break;
    	case 224: mycall(224); break;
    	case 256: mycall(256); break;
    	default: mycall(256); break;
    	}

    		#undef mycall
    }








    void callBuildMSAKernel_async(
            MSAColumnProperties* d_msaColumnProperties,
            int* d_counts,
            float* d_weights,
            int* d_coverage,
            float* d_origWeights,
            int* d_origCoverages,
            float* d_support,
            char* d_consensus,
            const int* d_overlaps,
            const int* d_shifts,
            const int* d_nOps,
            const BestAlignment_t* d_bestAlignmentFlags,
            const unsigned int* d_subjectSequencesData,
            const int* d_subjectSequencesLength,
            const unsigned int* d_candidateSequencesTransposedData,
            const int* d_candidateSequencesLength,
            const char* d_subjectQualities,
            const char* d_candidateQualities,
            bool canUseQualityScores,
            int encodedSequencePitchInInts,
            size_t qualityPitchInBytes,
            size_t msa_row_pitch,
            size_t msa_weights_row_pitch_floats,
            const int* d_indices,
            const int* d_indices_per_subject,
            const int* d_candidatesPerSubjectPrefixSum,
            int n_subjects,
            int n_candidates,
            const bool* d_canExecute,
            cudaStream_t stream,
            KernelLaunchHandle& kernelLaunchHandle){
#if 1

        call_msa_init_kernel_async_exp(
            d_msaColumnProperties,
            d_shifts,
            d_bestAlignmentFlags,
            d_subjectSequencesLength,
            d_candidateSequencesLength,
            d_indices,
            d_indices_per_subject,
            d_candidatesPerSubjectPrefixSum,
            n_subjects,
            n_candidates,
            d_canExecute,
            stream,
            kernelLaunchHandle
        );

        call_msa_add_sequences_kernel_implicit_async(
            d_msaColumnProperties,
            d_coverage,
            d_counts,
            d_weights,
            d_overlaps,
            d_shifts,
            d_nOps,
            d_bestAlignmentFlags,
            d_subjectSequencesData,
            d_candidateSequencesTransposedData,
            d_subjectSequencesLength,
            d_candidateSequencesLength,
            d_subjectQualities,
            d_candidateQualities,
            d_candidatesPerSubjectPrefixSum,
            d_indices,
            d_indices_per_subject,
            n_subjects,
            n_candidates,
            canUseQualityScores,
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            msa_row_pitch,
            msa_weights_row_pitch_floats * sizeof(float),
            d_canExecute,
            stream,
            kernelLaunchHandle
        );

        call_msaFindConsensusKernel_async(
            d_msaColumnProperties,
            d_counts,
            d_weights,
            d_support,
            d_coverage,
            d_origWeights,
            d_origCoverages,
            d_consensus,
            d_subjectSequencesData,
            d_indices_per_subject,
            n_subjects,
            encodedSequencePitchInInts,
            msa_row_pitch,
            msa_weights_row_pitch_floats * sizeof(float),
            d_canExecute,
            stream,
            kernelLaunchHandle
        );

#else 
        
#endif
    }



}
}