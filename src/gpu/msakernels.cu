//#define NDEBUG

#include <gpu/kernels.hpp>
#include <hostdevicefunctions.cuh>

#include <bestalignment.hpp>

#include <sequencehelpers.hpp>

#include <gpu/gpumsa.cuh>


#include <hpc_helpers.cuh>
#include <config.hpp>

#include <cassert>


#include <cub/cub.cuh>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;


namespace care{
namespace gpu{

    enum class MemoryType{
        Global,
        Shared
    };

    #ifdef __CUDACC_DEBUG__

        #define constructMultipleSequenceAlignmentsKernel_MIN_BLOCKS   1
        
    #else

        #if __CUDA_ARCH__ >= 610
            #define constructMultipleSequenceAlignmentsKernel_MIN_BLOCKS   8
        #else
            #define constructMultipleSequenceAlignmentsKernel_MIN_BLOCKS   4
        #endif

    #endif



    template<int BLOCKSIZE, MemoryType memoryType>
    __launch_bounds__(BLOCKSIZE, constructMultipleSequenceAlignmentsKernel_MIN_BLOCKS)
    __global__
    void constructMultipleSequenceAlignmentsKernel(
        GPUMultiMSA multiMSA,
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
        float desiredAlignmentMaxErrorRate,
        bool canUseQualityScores,
        int encodedSequencePitchInInts,
        size_t qualityPitchInBytes
    ){

        constexpr bool useSmemMSA = (memoryType == MemoryType::Shared);

        extern __shared__ float sharedmem[];
        __shared__ MSAColumnProperties shared_columnProperties;

        using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;            

        auto tbGroup = cg::this_thread_block();

        const int n_subjects = *d_numAnchors;

        typename BlockReduceInt::TempStorage* const cubTempStorage = (typename BlockReduceInt::TempStorage*)sharedmem;

        float* const shared_weights = sharedmem;
        int* const shared_counts = (int*)(shared_weights + 4 * multiMSA.columnPitchInElements);
        int* const shared_coverages = (int*)(shared_counts + 4 * multiMSA.columnPitchInElements);


        for(int subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
            const int myNumGoodCandidates = indices_per_subject[subjectIndex];

            if(myNumGoodCandidates > 0){

                tbGroup.sync(); //wait for smem of previous iteration

                GpuSingleMSA msa = multiMSA.getSingleMSA(subjectIndex);

                if(useSmemMSA){
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

                assert(columnsToCheck <= msa.columnPitchInElements);

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

                if(useSmemMSA){
                    // copy from counts and weights and coverages from shared to global
                    int* const gmemCounts = multiMSA.getCountsOfMSA(subjectIndex);
                    float* const gmemWeights = multiMSA.getWeightsOfMSA(subjectIndex);
                    int* const gmemCoverages = multiMSA.getCoveragesOfMSA(subjectIndex);

                    for(int index = tbGroup.thread_rank(); index < columnsToCheck; index += tbGroup.size()){
                        for(int k = 0; k < 4; k++){
                            const int* const srcCounts = msa.counts + k * msa.columnPitchInElements + index;
                            int* const destCounts = gmemCounts + k * msa.columnPitchInElements + index;
        
                            const float* const srcWeights = msa.weights + k * msa.columnPitchInElements + index;
                            float* const destWeights = gmemWeights + k * msa.columnPitchInElements + index;
        
                            *destCounts = *srcCounts;
                            *destWeights = *srcWeights;
                        }
                        gmemCoverages[index] = msa.coverages[index];
                    }
                }
            } 
        }

    }



    template<int BLOCKSIZE, MemoryType memoryType>
    __launch_bounds__(BLOCKSIZE, constructMultipleSequenceAlignmentsKernel_MIN_BLOCKS)
    __global__
    void constructMultipleSequenceAlignmentsWithContiguousCandidatesKernel(
        GPUMultiMSA multiMSA,
        const int* __restrict__ overlaps,
        const int* __restrict__ shifts,
        const int* __restrict__ nOps,
        const BestAlignment_t* __restrict__ bestAlignmentFlags,
        const int* __restrict__ anchorLengths,
        const int* __restrict__ candidateLengths,
        const int* __restrict__ indices_per_subject,
        const int* __restrict__ candidatesPerSubjectPrefixSum,            
        const unsigned int* __restrict__ subjectSequencesData,
        const unsigned int* __restrict__ candidateSequencesData,
        const char* __restrict__ subjectQualities,
        const char* __restrict__ candidateQualities,
        const int* __restrict__ d_numAnchors,
        float desiredAlignmentMaxErrorRate,
        bool canUseQualityScores,
        int encodedSequencePitchInInts,
        size_t qualityPitchInBytes
    ){

        constexpr bool useSmemMSA = (memoryType == MemoryType::Shared);

        extern __shared__ float sharedmem[];
        __shared__ MSAColumnProperties shared_columnProperties;

        using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;            

        auto tbGroup = cg::this_thread_block();

        const int n_subjects = *d_numAnchors;

        typename BlockReduceInt::TempStorage* const cubTempStorage = (typename BlockReduceInt::TempStorage*)sharedmem;

        float* const shared_weights = sharedmem;
        int* const shared_counts = (int*)(shared_weights + 4 * multiMSA.columnPitchInElements);
        int* const shared_coverages = (int*)(shared_counts + 4 * multiMSA.columnPitchInElements);


        for(int subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
            const int myNumGoodCandidates = indices_per_subject[subjectIndex];

            if(myNumGoodCandidates > 0){

                tbGroup.sync(); //wait for smem of previous iteration

                GpuSingleMSA msa = multiMSA.getSingleMSA(subjectIndex);

                if(useSmemMSA){
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

                assert(columnsToCheck <= msa.columnPitchInElements);

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

                if(useSmemMSA){
                    // copy from counts and weights and coverages from shared to global
                    int* const gmemCounts = multiMSA.getCountsOfMSA(subjectIndex);
                    float* const gmemWeights = multiMSA.getWeightsOfMSA(subjectIndex);
                    int* const gmemCoverages = multiMSA.getCoveragesOfMSA(subjectIndex);

                    for(int index = tbGroup.thread_rank(); index < columnsToCheck; index += tbGroup.size()){
                        for(int k = 0; k < 4; k++){
                            const int* const srcCounts = msa.counts + k * msa.columnPitchInElements + index;
                            int* const destCounts = gmemCounts + k * msa.columnPitchInElements + index;
        
                            const float* const srcWeights = msa.weights + k * msa.columnPitchInElements + index;
                            float* const destWeights = gmemWeights + k * msa.columnPitchInElements + index;
        
                            *destCounts = *srcCounts;
                            *destWeights = *srcWeights;
                        }
                        gmemCoverages[index] = msa.coverages[index];
                    }
                }
            } 
        }

    }





    template<int BLOCKSIZE, MemoryType memoryType>
    __global__
    void msaCandidateRefinement_singleiter_kernel(
        int* __restrict__ d_newIndices,
        int* __restrict__ d_newNumIndicesPerSubject,
        int* __restrict__ d_newNumIndices,
        GPUMultiMSA multiMSA,
        const BestAlignment_t* __restrict__ bestAlignmentFlags,
        const int* __restrict__ shifts,
        const int* __restrict__ nOps,
        const int* __restrict__ overlaps,
        const unsigned int* __restrict__ subjectSequencesData,
        const unsigned int* __restrict__ candidateSequencesData,
        const int* __restrict__ subjectSequencesLength,
        const int* __restrict__ candidateSequencesLength,
        const char* __restrict__ subjectQualities,
        const char* __restrict__ candidateQualities,
        bool* __restrict__ d_shouldBeKept,
        const int* __restrict__ d_candidates_per_subject_prefixsum,
        const int* __restrict__ d_numAnchors,
        float desiredAlignmentMaxErrorRate,
        bool canUseQualityScores,
        size_t encodedSequencePitchInInts,
        size_t qualityPitchInBytes,
        const int* __restrict__ d_indices,
        const int* __restrict__ d_indices_per_subject,
        int dataset_coverage,
        int iteration,
        bool* __restrict__ d_anchorIsFinished
    ){

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

        constexpr bool useSmemMSA = (memoryType == MemoryType::Shared);

        float* const shared_weights = externsharedmem;
        int* const shared_counts = (int*)(shared_weights + 4 * multiMSA.columnPitchInElements);
        int* const shared_coverages = (int*)(shared_counts + 4 * multiMSA.columnPitchInElements);

        const int n_subjects = *d_numAnchors;

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
                        shared_columnProperties = *(multiMSA.getColumnPropertiesOfMSA(subjectIndex));
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

                        if(tbGroup.thread_rank() == 0){
                            *(multiMSA.getColumnPropertiesOfMSA(subjectIndex)) = shared_columnProperties;
                        }

                        tbGroup.sync();

                        const int columnsToCheck = shared_columnProperties.lastColumn_excl;

                        assert(columnsToCheck <= msa.columnPitchInElements);

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
                                    const int* const srcCounts = msa.counts 
                                        + k * msa.columnPitchInElements + index;
                                    int* const destCounts = gmemCounts 
                                        + k * msa.columnPitchInElements + index;
                
                                    const float* const srcWeights = msa.weights 
                                        + k * msa.columnPitchInElements + index;
                                    float* const destWeights = gmemWeights 
                                        + k * msa.columnPitchInElements + index;
                
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



    #ifdef __CUDACC_DEBUG__

        #define msaCandidateRefinement_multiiter_MIN_BLOCKS   1
        
    #else

        #if __CUDA_ARCH__ >= 610
            #define msaCandidateRefinement_multiiter_MIN_BLOCKS   4
        #else
            #define msaCandidateRefinement_multiiter_MIN_BLOCKS   4
        #endif

    #endif


    template<int BLOCKSIZE, MemoryType memoryType>
    //__launch_bounds__(BLOCKSIZE, msaCandidateRefinement_multiiter_MIN_BLOCKS)
    __global__
    void msaCandidateRefinement_multiiter_kernel(
        int* __restrict__ d_newIndices,
        int* __restrict__ d_newNumIndicesPerSubject,
        int* __restrict__ d_newNumIndices,
        GPUMultiMSA multiMSA,
        const BestAlignment_t* __restrict__ bestAlignmentFlags,
        const int* __restrict__ shifts,
        const int* __restrict__ nOps,
        const int* __restrict__ overlaps,
        const unsigned int* __restrict__ subjectSequencesData,
        const unsigned int* __restrict__ candidateSequencesData,
        const int* __restrict__ subjectSequencesLength,
        const int* __restrict__ candidateSequencesLength,
        const char* __restrict__ subjectQualities,
        const char* __restrict__ candidateQualities,
        bool* __restrict__ d_shouldBeKept,
        const int* __restrict__ d_candidates_per_subject_prefixsum,
        const int* __restrict__ d_numAnchors,
        float desiredAlignmentMaxErrorRate,
        bool canUseQualityScores,
        size_t encodedSequencePitchInInts,
        size_t qualityPitchInBytes,
        int* __restrict__ d_indices,
        int* __restrict__ d_indices_per_subject,
        int dataset_coverage,
        int numRefinementIterations
    ){

        constexpr bool useSmemMSA = (memoryType == MemoryType::Shared);

        using BlockReduceBool = cub::BlockReduce<bool, BLOCKSIZE>;
        using BlockReduceInt2 = cub::BlockReduce<int2, BLOCKSIZE>;

        extern __shared__ float externsharedmem[];
        __shared__ MSAColumnProperties shared_columnProperties;

        __shared__ union{
            typename BlockReduceBool::TempStorage boolreduce;
            typename BlockReduceInt2::TempStorage int2reduce;
        } temp_storage;      

        assert(numRefinementIterations > 0);

        auto tbGroup = cg::this_thread_block();

        float* const shared_weights = externsharedmem;
        int* const shared_counts = (int*)(shared_weights + 4 * multiMSA.columnPitchInElements);
        int* const shared_coverages = (int*)(shared_counts + 4 * multiMSA.columnPitchInElements);            

        const int n_subjects = *d_numAnchors;

        for(int subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
            int myNumIndices = d_indices_per_subject[subjectIndex];                            

            if(myNumIndices > 0){

                tbGroup.sync();

                if(threadIdx.x == 0){
                    shared_columnProperties = *(multiMSA.getColumnPropertiesOfMSA(subjectIndex));
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

                    for(int k = tbGroup.thread_rank(); k < msa.columnPitchInElements; k += tbGroup.size()){
                        for(int i = 0; i < 4; i++){
                            shared_counts[k + i * msa.columnPitchInElements] 
                                = gmemCounts[k + i * msa.columnPitchInElements];
                            shared_weights[k + i * msa.columnPitchInElements] 
                                = gmemWeights[k + i * msa.columnPitchInElements];
                        }
                        shared_coverages[k] = gmemCoverages[k];
                    }
                }

                auto storeSmemMSAToGmem = [&](){
                    int* const gmemCounts = multiMSA.getCountsOfMSA(subjectIndex);
                    float* const gmemWeights = multiMSA.getWeightsOfMSA(subjectIndex);
                    int* const gmemCoverages = multiMSA.getCoveragesOfMSA(subjectIndex);

                    for(int k = tbGroup.thread_rank(); k < msa.columnPitchInElements; k += tbGroup.size()){
                        for(int i = 0; i < 4; i++){
                            gmemCounts[k + i * msa.columnPitchInElements] 
                                = shared_counts[k + i * msa.columnPitchInElements];
                            gmemWeights[k + i * msa.columnPitchInElements] 
                                = shared_weights[k + i * msa.columnPitchInElements];
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
                    
                    long long int t1 = clock64();

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

                    long long int t2 = clock64();

                    if(subjectIndex == 0 && tbGroup.thread_rank() == 0){
                        //printf("duration flag: %lu\n", t2-t1);
                    }

                    const int myNewNumIndices = *destNumIndices;
                    
                    assert(myNewNumIndices <= myNumIndices);
                    if(myNewNumIndices > 0 && myNewNumIndices < myNumIndices){
                        auto selector = [&](int i){
                            return !myShouldBeKept[i];
                        };

                        long long int t3 = clock64();

                        msa.removeCandidates(
                        //msa.removeCandidates_verticalthreads(
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

                        long long int t4 = clock64();

                        if(subjectIndex == 0 && tbGroup.thread_rank() == 0){
                            //printf("duration removal: %lu. candidates before %d, after %d\n", t4-t3, myNumIndices, myNewNumIndices);
                        }

                        msa.updateColumnProperties(tbGroup);

                        tbGroup.sync();

                        //msa.checkAfterBuild(tbGroup, subjectIndex);

                        assert(shared_columnProperties.firstColumn_incl != -1);
                        assert(shared_columnProperties.lastColumn_excl != -1);

                        long long int t5 = clock64();

                        msa.findConsensus(
                            tbGroup,
                            myAnchorSequenceData, 
                            encodedSequencePitchInInts, 
                            subjectIndex
                        );

                        if(tbGroup.thread_rank() == 0){
                            *(multiMSA.getColumnPropertiesOfMSA(subjectIndex)) = shared_columnProperties;
                        }

                        tbGroup.sync();

                        long long int t6 = clock64();

                        if(subjectIndex == 0 && tbGroup.thread_rank() == 0){
                            //printf("duration consensus: %lu\n", t6-t5);
                        }

                        myNumIndices = myNewNumIndices;

                        if(refinementIteration == numRefinementIterations - 1){
                            //copy shared mem msa back to gmem

                            if(useSmemMSA){                                            
                                storeSmemMSAToGmem();
                            }

                            finalizeRefinement(myNewNumIndices);
                        }

                    }else{
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








    //####################   KERNEL DISPATCH   ####################
    
    void callMsaCandidateRefinementKernel_multiiter_async(
        int* d_newIndices,
        int* d_newNumIndicesPerSubject,
        int* d_newNumIndices,
        GPUMultiMSA multiMSA,
        const BestAlignment_t* d_bestAlignmentFlags,
        const int* d_shifts,
        const int* d_nOps,
        const int* d_overlaps,
        const unsigned int* d_subjectSequencesData,
        const unsigned int* d_candidateSequencesData,
        const int* d_subjectSequencesLength,
        const int* d_candidateSequencesLength,
        const char* d_subjectQualities,
        const char* d_candidateQualities,
        bool* d_shouldBeKept,
        const int* d_candidates_per_subject_prefixsum,
        const int* d_numAnchors,
        float desiredAlignmentMaxErrorRate,
        int maxNumAnchors,
        int maxNumCandidates,
        bool canUseQualityScores,
        size_t encodedSequencePitchInInts,
        size_t qualityPitchInBytes,
        int* d_indices,
        int* d_indices_per_subject,
        int dataset_coverage,
        int numIterations,
        cudaStream_t stream,
        KernelLaunchHandle& handle
    ){

        helpers::call_fill_kernel_async(
            d_newNumIndices,
            1,
            0,
            stream
        );

        constexpr int blocksize = 128;

        constexpr MemoryType memType = MemoryType::Shared;

        constexpr bool usesSmem = memType == MemoryType::Shared;

        const std::size_t smemAddSequences = (usesSmem ? 
                                                sizeof(float) * 4 * multiMSA.columnPitchInElements // weights
                                                    + sizeof(int) * 4 * multiMSA.columnPitchInElements // counts
                                                    + sizeof(int) * multiMSA.columnPitchInElements // coverages
                                                : 0);

        const std::size_t smem = smemAddSequences;

        constexpr auto kernelId = KernelId::MSACandidateRefinementMultiIter;

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
                msaCandidateRefinement_multiiter_kernel<blocksize, memType>,
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


        msaCandidateRefinement_multiiter_kernel<blocksize, memType>
                <<<grid, block, smem, stream>>>(
            d_newIndices,
            d_newNumIndicesPerSubject,
            d_newNumIndices,
            multiMSA,
            d_bestAlignmentFlags,
            d_shifts,
            d_nOps,
            d_overlaps,
            d_subjectSequencesData,
            d_candidateSequencesData,
            d_subjectSequencesLength,
            d_candidateSequencesLength,
            d_subjectQualities,
            d_candidateQualities,
            d_shouldBeKept,
            d_candidates_per_subject_prefixsum,
            d_numAnchors,
            desiredAlignmentMaxErrorRate,
            canUseQualityScores,
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            d_indices,
            d_indices_per_subject,
            dataset_coverage,
            numIterations
        );

        CUERR;
    }





    void callMsaCandidateRefinementKernel_singleiter_async(
        int* d_newIndices,
        int* d_newNumIndicesPerSubject,
        int* d_newNumIndices,
        GPUMultiMSA multiMSA,
        const BestAlignment_t* d_bestAlignmentFlags,
        const int* d_shifts,
        const int* d_nOps,
        const int* d_overlaps,
        const unsigned int* d_subjectSequencesData,
        const unsigned int* d_candidateSequencesData,
        const int* d_subjectSequencesLength,
        const int* d_candidateSequencesLength,
        const char* d_subjectQualities,
        const char* d_candidateQualities,
        bool* d_shouldBeKept,
        const int* d_candidates_per_subject_prefixsum,
        const int* d_numAnchors,
        float desiredAlignmentMaxErrorRate,
        int maxNumAnchors,
        int maxNumCandidates,
        bool canUseQualityScores,
        size_t encodedSequencePitchInInts,
        size_t qualityPitchInBytes,
        const int* d_indices,
        const int* d_indices_per_subject,
        int dataset_coverage,
        int iteration,
        bool* d_anchorIsFinished,
        cudaStream_t stream,
        KernelLaunchHandle& handle
    ){

        helpers::call_fill_kernel_async(
            d_newNumIndices,
            1,
            0,
            stream
        );

        constexpr int blocksize = 128;

        constexpr MemoryType memoryType = MemoryType::Shared;

        constexpr bool addSequencesUsesSmem = memoryType == MemoryType::Shared;

        const std::size_t smemAddSequences = (addSequencesUsesSmem ? 
                                                sizeof(float) * 4 * multiMSA.columnPitchInElements // weights
                                                    + sizeof(int) * 4 * multiMSA.columnPitchInElements // counts
                                                    + sizeof(int) * multiMSA.columnPitchInElements // coverages
                                                : 0);

        const std::size_t smem = smemAddSequences;

        constexpr auto kernelId = KernelId::MSACandidateRefinementSingleIter;

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
                msaCandidateRefinement_singleiter_kernel<blocksize, memoryType>,
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


        msaCandidateRefinement_singleiter_kernel<blocksize, memoryType><<<grid, block, smem, stream>>>(
            d_newIndices,
            d_newNumIndicesPerSubject,
            d_newNumIndices,
            multiMSA,
            d_bestAlignmentFlags,
            d_shifts,
            d_nOps,
            d_overlaps,
            d_subjectSequencesData,
            d_candidateSequencesData,
            d_subjectSequencesLength,
            d_candidateSequencesLength,
            d_subjectQualities,
            d_candidateQualities,
            d_shouldBeKept,
            d_candidates_per_subject_prefixsum,
            d_numAnchors,
            desiredAlignmentMaxErrorRate,
            canUseQualityScores,
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            d_indices,
            d_indices_per_subject,
            dataset_coverage,
            iteration,
            d_anchorIsFinished
        );

        CUERR;
    }





    void callConstructMultipleSequenceAlignmentsKernel_async(
        GPUMultiMSA multiMSA,
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
        const unsigned int* d_candidateSequencesData,
        const char* d_subjectQualities,
        const char* d_candidateQualities,
        const int* d_numAnchors,
        float desiredAlignmentMaxErrorRate,
        int maxNumAnchors,
        int maxNumCandidates,
        bool canUseQualityScores,
        int encodedSequencePitchInInts,
        size_t qualityPitchInBytes,
        cudaStream_t stream,
        KernelLaunchHandle& handle){
            

    constexpr MemoryType memoryType = MemoryType::Shared;
    constexpr bool addSequencesUsesSmem = memoryType == MemoryType::Shared;
    constexpr int BLOCKSIZE = 128;

    using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;
    using BlockReduceIntStorage = typename BlockReduceInt::TempStorage;

    const std::size_t smemCub = sizeof(BlockReduceIntStorage);
    const std::size_t smemAddSequences = (addSequencesUsesSmem ? 
                                            sizeof(float) * 4 * multiMSA.columnPitchInElements // weights
                                                + sizeof(int) * 4 * multiMSA.columnPitchInElements // counts
                                                + sizeof(int) * multiMSA.columnPitchInElements // coverages
                                            : 0);

    const std::size_t smem = std::max(smemCub, smemAddSequences);

    int max_blocks_per_device = 1;

    KernelLaunchConfig kernelLaunchConfig;
    kernelLaunchConfig.threads_per_block = BLOCKSIZE;
    kernelLaunchConfig.smem = smem;

    constexpr auto kernelId = KernelId::MSAConstruction;

    auto iter = handle.kernelPropertiesMap.find(kernelId);
    if(iter == handle.kernelPropertiesMap.end()) {

        std::map<KernelLaunchConfig, KernelProperties> mymap;

        KernelProperties kernelProperties;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &kernelProperties.max_blocks_per_SM,
            constructMultipleSequenceAlignmentsKernel<BLOCKSIZE, memoryType>,
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

    dim3 block(BLOCKSIZE, 1, 1);
    //dim3 grid(maxNumAnchors, 1, 1);
    dim3 grid(max_blocks_per_device, 1, 1);
    
    constructMultipleSequenceAlignmentsKernel<BLOCKSIZE, memoryType><<<grid, block, smem, stream>>>(
        multiMSA,         
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
        d_candidateSequencesData,
        d_subjectQualities,
        d_candidateQualities,
        d_numAnchors,
        desiredAlignmentMaxErrorRate,
        canUseQualityScores,
        encodedSequencePitchInInts,
        qualityPitchInBytes
    ); CUERR;



}





}
}
