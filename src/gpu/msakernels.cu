// #define NDEBUG

#include <gpu/kernels.hpp>
#include <hostdevicefunctions.cuh>
#include <gpu/cudaerrorcheck.cuh>
#include <gpu/groupmemcpy.cuh>
#include <gpu/gpumsa.cuh>
#include <alignmentorientation.hpp>
#include <sequencehelpers.hpp>



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


    template<int BLOCKSIZE>
    __global__
    void computeMaximumMsaWidthKernel(
        int* __restrict__ result, //initialized with 0
        const int* __restrict__ shifts,
        const int* __restrict__ anchorLengths,
        const int* __restrict__ candidateLengths,
        const int* __restrict__ indices,
        const int* __restrict__ indices_per_anchor,
        const int* __restrict__ candidatesPerAnchorPrefixSum,
        const int numAnchors
    ){

        using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;

        __shared__ typename BlockReduceInt::TempStorage cubTempStorage;

        auto tbGroup = cg::this_thread_block();

        int myMaxWidth = 0; //only valid for thread_rank 0

        for(int anchorIndex = blockIdx.x; anchorIndex < numAnchors; anchorIndex += gridDim.x){
            const int myNumGoodCandidates = indices_per_anchor[anchorIndex];
            tbGroup.sync();

            const int globalCandidateOffset = candidatesPerAnchorPrefixSum[anchorIndex];

            const int* const myShifts = shifts + globalCandidateOffset;
            const int anchorLength = anchorLengths[anchorIndex];
            const int* const myCandidateLengths = candidateLengths + globalCandidateOffset;
            const int* const myIndices = indices + globalCandidateOffset;

            auto groupReduceIntMin = [&](int data){
                data = BlockReduceInt(cubTempStorage).Reduce(data, cub::Min());
                tbGroup.sync();
                return data;
            };

            auto groupReduceIntMax = [&](int data){                        
                data = BlockReduceInt(cubTempStorage).Reduce(data, cub::Max());
                tbGroup.sync();
                return data;
            };

            MSAColumnProperties columnProperties = GpuSingleMSA::computeColumnProperties(
                tbGroup,
                groupReduceIntMin,
                groupReduceIntMax,
                myIndices,
                myNumGoodCandidates,
                myShifts,
                anchorLength,
                myCandidateLengths
            );

            //only thread 0 has valid column properties. 
            myMaxWidth = max(myMaxWidth, columnProperties.lastColumn_excl);
        }

        if(tbGroup.thread_rank() == 0){
            atomicMax(result, myMaxWidth);
        }
    }

    __global__
    void computeMsaConsensusQualityKernel(
        char* consensusQuality,
        int consensusQualityPitchInBytes,
        GPUMultiMSA multiMSA
    ){
        auto group = cg::this_thread_block();

        for(int t = blockIdx.x; t < multiMSA.numMSAs; t += gridDim.x){
            const gpu::GpuSingleMSA singleMSA = multiMSA.getSingleMSA(t);
            singleMSA.computeConsensusQuality(
                group, 
                consensusQuality + t * consensusQualityPitchInBytes, 
                consensusQualityPitchInBytes
            );
        }
    }

    __global__
    void computeDecodedMsaConsensusKernel(
        char* consensus,
        int consensusPitchInBytes,
        GPUMultiMSA multiMSA
    ){
        auto group = cg::this_thread_block();

        for(int t = blockIdx.x; t < multiMSA.numMSAs; t += gridDim.x){
            const gpu::GpuSingleMSA singleMSA = multiMSA.getSingleMSA(t);
            singleMSA.computeDecodedConsensus(
                group, 
                consensus + t * consensusPitchInBytes, 
                consensusPitchInBytes
            );
        }
    }

    __global__
    void computeMsaSizesKernel(
        int* sizes,
        GPUMultiMSA multiMSA
    ){

        for(int t = threadIdx.x + blockIdx.x * blockDim.x; t < multiMSA.numMSAs; t += gridDim.x * blockDim.x){
            const gpu::GpuSingleMSA singleMSA = multiMSA.getSingleMSA(t);

            sizes[t] = singleMSA.computeSize();
        }
    }



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
        const AlignmentOrientation* __restrict__ bestAlignmentFlags,
        const int* __restrict__ anchorLengths,
        const int* __restrict__ candidateLengths,
        const int* __restrict__ indices,
        const int* __restrict__ indices_per_anchor,
        const int* __restrict__ candidatesPerAnchorPrefixSum,            
        const unsigned int* __restrict__ anchorSequencesData,
        const unsigned int* __restrict__ candidateSequencesData,
        const bool* __restrict__ d_isPairedCandidate,
        const char* __restrict__ anchorQualities,
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

        const int n_anchors = *d_numAnchors;

        typename BlockReduceInt::TempStorage* const cubTempStorage = (typename BlockReduceInt::TempStorage*)sharedmem;

        float* const shared_weights = sharedmem;
        int* const shared_counts = (int*)(shared_weights + 4 * multiMSA.columnPitchInElements);
        int* const shared_coverages = (int*)(shared_counts + 4 * multiMSA.columnPitchInElements);


        for(int anchorIndex = blockIdx.x; anchorIndex < n_anchors; anchorIndex += gridDim.x){
            const int myNumGoodCandidates = indices_per_anchor[anchorIndex];

            //if(myNumGoodCandidates > 0){

                tbGroup.sync(); //wait for smem of previous iteration

                GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);

                if(useSmemMSA){
                    msa.counts = shared_counts;
                    msa.weights = shared_weights;
                    msa.coverages = shared_coverages;
                }

                const int globalCandidateOffset = candidatesPerAnchorPrefixSum[anchorIndex];

                const int* const myOverlaps = overlaps + globalCandidateOffset;
                const int* const myShifts = shifts + globalCandidateOffset;
                const int* const myNops = nOps + globalCandidateOffset;
                const AlignmentOrientation* const myAlignmentFlags = bestAlignmentFlags + globalCandidateOffset;
                const int anchorLength = anchorLengths[anchorIndex];
                const int* const myCandidateLengths = candidateLengths + globalCandidateOffset;
                const int* const myIndices = indices + globalCandidateOffset;

                const unsigned int* const myAnchorSequenceData = anchorSequencesData + std::size_t(anchorIndex) * encodedSequencePitchInInts;
                const unsigned int* const myCandidateSequencesData = candidateSequencesData + size_t(globalCandidateOffset) * encodedSequencePitchInInts;
                const char* const myAnchorQualityData = anchorQualities + std::size_t(anchorIndex) * qualityPitchInBytes;
                const char* const myCandidateQualities = candidateQualities + size_t(globalCandidateOffset) * qualityPitchInBytes;
                const bool* const myIsPairedCandidate = d_isPairedCandidate + globalCandidateOffset;

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
                    anchorLength,
                    myCandidateLengths
                );

                //only thread 0 has valid column properties. 
                //save to global memory and broadcast to all threads in block
                if(tbGroup.thread_rank() == 0){
                    auto* globalDest = multiMSA.getColumnPropertiesOfMSA(anchorIndex);
                    *globalDest = columnProperties;
                    shared_columnProperties = columnProperties;
                }

                tbGroup.sync();

                columnProperties = shared_columnProperties;

                const int columnsToCheck = columnProperties.lastColumn_excl;
                //printf("got %d, max %d\n", columnsToCheck, msa.columnPitchInElements);

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
                    myIsPairedCandidate,
                    myIndices,
                    myNumGoodCandidates,
                    canUseQualityScores, 
                    encodedSequencePitchInInts,
                    qualityPitchInBytes,
                    desiredAlignmentMaxErrorRate,
                    anchorIndex
                );

                tbGroup.sync();
        
                msa.checkAfterBuild(tbGroup, anchorIndex);

                // bool mychecksuccess = msa.checkCoverages(tbGroup);
                // int numchecksuccess = BlockReduceInt(*cubTempStorage).Sum(mychecksuccess ? 1 : 0);
                // tbGroup.sync();
                // bool checksuccess = numchecksuccess == tbGroup.size();
                // if(tbGroup.thread_rank() == 0){
                //     if(!checksuccess){
                //         printf("numchecksuccess %d\n ", numchecksuccess);

                //         const int firstColumnIncl = msa.columnProperties->firstColumn_incl;
                //         const int lastColumnExcl = msa.columnProperties->lastColumn_excl;

                //         printf("firstColumnIncl %d, lastColumnExcl %d\n", firstColumnIncl, lastColumnExcl);
                //         printf("coverages:\n");
                //         for(int x = 0; x < msa.columnPitchInElements; x++){
                //             printf("%d, ", msa.coverages[x]);
                //         }
                //         printf("\n");
                //     }
                //     assert(checksuccess);
                // }
                // tbGroup.sync();

                msa.findConsensus(
                    tbGroup,
                    myAnchorSequenceData, 
                    encodedSequencePitchInInts, 
                    anchorIndex
                );

                if(useSmemMSA){
                    // copy from counts and weights and coverages from shared to global
                    int* const gmemCounts = multiMSA.getCountsOfMSA(anchorIndex);
                    float* const gmemWeights = multiMSA.getWeightsOfMSA(anchorIndex);
                    int* const gmemCoverages = multiMSA.getCoveragesOfMSA(anchorIndex);

                    care::gpu::memcpy<int4>(tbGroup, gmemCoverages, msa.coverages, sizeof(int) * columnsToCheck);

                    for(int k = 0; k < 4; k++){
                        const int* const srcCounts = msa.counts + k * msa.columnPitchInElements;
                        int* const destCounts = gmemCounts + k * msa.columnPitchInElements;
                        care::gpu::memcpy<int4>(tbGroup, destCounts, srcCounts, sizeof(int) * columnsToCheck);
                    }

                    for(int k = 0; k < 4; k++){
                        const float* const srcWeights = msa.weights + k * msa.columnPitchInElements;
                        float* const destWeights = gmemWeights + k * msa.columnPitchInElements;
                        care::gpu::memcpy<int4>(tbGroup, destWeights, srcWeights, sizeof(float) * columnsToCheck);
                    }
                }
            //} 
        }

    }


    template<int BLOCKSIZE, MemoryType memoryType>
    __global__
    void msaCandidateRefinement_singleiter_kernel(
        int* __restrict__ d_newIndices,
        int* __restrict__ d_newNumIndicesPerAnchor,
        int* __restrict__ d_newNumIndices,
        GPUMultiMSA multiMSA,
        const AlignmentOrientation* __restrict__ bestAlignmentFlags,
        const int* __restrict__ shifts,
        const int* __restrict__ nOps,
        const int* __restrict__ overlaps,
        const unsigned int* __restrict__ anchorSequencesData,
        const unsigned int* __restrict__ candidateSequencesData,
        const bool* __restrict__ d_isPairedCandidate,
        const int* __restrict__ anchorSequencesLength,
        const int* __restrict__ candidateSequencesLength,
        const char* __restrict__ anchorQualities,
        const char* __restrict__ candidateQualities,
        bool* __restrict__ d_shouldBeKept,
        const int* __restrict__ d_candidates_per_anchor_prefixsum,
        const int* __restrict__ d_numAnchors,
        float desiredAlignmentMaxErrorRate,
        bool canUseQualityScores,
        size_t encodedSequencePitchInInts,
        size_t qualityPitchInBytes,
        const int* __restrict__ d_indices,
        const int* __restrict__ d_indices_per_anchor,
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

        const int n_anchors = *d_numAnchors;

        auto tbGroup = cg::this_thread_block();

        for(int anchorIndex = blockIdx.x; anchorIndex < n_anchors; anchorIndex += gridDim.x){
            const bool myAnchorIsFinished = d_anchorIsFinished[anchorIndex];
            const int myNumIndices = d_indices_per_anchor[anchorIndex];

            if(myAnchorIsFinished){
                if(threadIdx.x == 0){
                    atomicAdd(d_newNumIndices, myNumIndices);
                }
            }else{               

                if(myNumIndices > 0){
                    GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);
                    msa.columnProperties = &shared_columnProperties;

                    tbGroup.sync(); //wait for previous iteration

                    if(threadIdx.x == 0){
                        shared_columnProperties = *(multiMSA.getColumnPropertiesOfMSA(anchorIndex));
                    }
                    tbGroup.sync();

                    const int globalOffset = d_candidates_per_anchor_prefixsum[anchorIndex];
                    const int* const myIndices = d_indices + globalOffset;
                    const int* const myNumIndicesPerAnchorPtr = d_indices_per_anchor + anchorIndex;

                    int* const myNewIndicesPtr = d_newIndices + globalOffset;
                    int* const myNewNumIndicesPerAnchorPtr = d_newNumIndicesPerAnchor + anchorIndex;

                    bool* const myShouldBeKept = d_shouldBeKept + globalOffset;                    

                    const AlignmentOrientation* const myAlignmentFlags = bestAlignmentFlags + globalOffset;
                    const int* const myShifts = shifts + globalOffset;
                    const int* const myNops = nOps + globalOffset;
                    const int* const myOverlaps = overlaps + globalOffset;

                    const unsigned int* const myAnchorSequenceData = anchorSequencesData 
                        + std::size_t(anchorIndex) * encodedSequencePitchInInts;
                    const unsigned int* const myCandidateSequencesData = candidateSequencesData 
                        + std::size_t(globalOffset) * encodedSequencePitchInInts;

                    const char* const myAnchorQualityData = anchorQualities + std::size_t(anchorIndex) * qualityPitchInBytes;
                    const char* const myCandidateQualities = candidateQualities 
                        + size_t(globalOffset) * qualityPitchInBytes;

                    const int anchorLength = anchorSequencesLength[anchorIndex];
                    const int* const myCandidateLengths = candidateSequencesLength + globalOffset;

                    const bool* const myIsPairedCandidate = d_isPairedCandidate + globalOffset;

                    const int* const srcIndices = myIndices;
                    int* const destIndices = myNewIndicesPtr;

                    const int* const srcNumIndices = myNumIndicesPerAnchorPtr;
                    int* const destNumIndices = myNewNumIndicesPerAnchorPtr;

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
                        anchorLength,
                        myCandidateSequencesData,
                        myCandidateLengths,
                        myAlignmentFlags,
                        myShifts,
                        myNops,
                        myOverlaps,
                        myShouldBeKept,
                        desiredAlignmentMaxErrorRate,
                        anchorIndex,
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
                            anchorLength,
                            myCandidateLengths
                        );

                        if(tbGroup.thread_rank() == 0){
                            *(multiMSA.getColumnPropertiesOfMSA(anchorIndex)) = shared_columnProperties;
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
                            myIsPairedCandidate,
                            destIndices,
                            *destNumIndices,
                            canUseQualityScores, 
                            encodedSequencePitchInInts,
                            qualityPitchInBytes,
                            desiredAlignmentMaxErrorRate,
                            anchorIndex
                        );

                        tbGroup.sync();
                
                        msa.checkAfterBuild(tbGroup, anchorIndex);

                        msa.findConsensus(
                            tbGroup,
                            myAnchorSequenceData, 
                            encodedSequencePitchInInts, 
                            anchorIndex
                        );

                        tbGroup.sync();

                        if(useSmemMSA){
                            // copy from counts and weights and coverages from shared to global
                            int* const gmemCounts = multiMSA.getCountsOfMSA(anchorIndex);
                            float* const gmemWeights = multiMSA.getWeightsOfMSA(anchorIndex);
                            int* const gmemCoverages = multiMSA.getCoveragesOfMSA(anchorIndex);

                            care::gpu::memcpy<int4>(tbGroup, gmemCoverages, msa.coverages, sizeof(int) * columnsToCheck);

                            for(int k = 0; k < 4; k++){
                                const int* const srcCounts = msa.counts + k * msa.columnPitchInElements;
                                int* const destCounts = gmemCounts + k * msa.columnPitchInElements;
                                care::gpu::memcpy<int4>(tbGroup, destCounts, srcCounts, sizeof(int) * columnsToCheck);
                            }

                            for(int k = 0; k < 4; k++){
                                const float* const srcWeights = msa.weights + k * msa.columnPitchInElements;
                                float* const destWeights = gmemWeights + k * msa.columnPitchInElements;
                                care::gpu::memcpy<int4>(tbGroup, destWeights, srcWeights, sizeof(float) * columnsToCheck);
                            }

                            tbGroup.sync();
                        }

                        
                
                    }else{
                        if(threadIdx.x == 0){
                            d_anchorIsFinished[anchorIndex] = true;
                        }
                    }
                }else{
                    if(threadIdx.x == 0){
                        d_newNumIndicesPerAnchor[anchorIndex] = 0;
                        d_anchorIsFinished[anchorIndex] = true;
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
        int* __restrict__ d_newNumIndicesPerAnchor,
        int* __restrict__ d_newNumIndices,
        GPUMultiMSA multiMSA,
        const AlignmentOrientation* __restrict__ bestAlignmentFlags,
        const int* __restrict__ shifts,
        const int* __restrict__ nOps,
        const int* __restrict__ overlaps,
        const unsigned int* __restrict__ anchorSequencesData,
        const unsigned int* __restrict__ candidateSequencesData,
        const bool* __restrict__ d_isPairedCandidate,
        const int* __restrict__ anchorSequencesLength,
        const int* __restrict__ candidateSequencesLength,
        const char* __restrict__ anchorQualities,
        const char* __restrict__ candidateQualities,
        bool* __restrict__ d_shouldBeKept,
        const int* __restrict__ d_candidates_per_anchor_prefixsum,
        const int* __restrict__ d_numAnchors,
        float desiredAlignmentMaxErrorRate,
        bool canUseQualityScores,
        size_t encodedSequencePitchInInts,
        size_t qualityPitchInBytes,
        int* __restrict__ d_indices,
        int* __restrict__ d_indices_per_anchor,
        int dataset_coverage,
        int numRefinementIterations,
        const read_number* __restrict__ anchorReadIds = nullptr
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

        const int n_anchors = *d_numAnchors;

        for(int anchorIndex = blockIdx.x; anchorIndex < n_anchors; anchorIndex += gridDim.x){
            int myNumIndices = d_indices_per_anchor[anchorIndex];

            if(myNumIndices > 0){

                tbGroup.sync();

                if(threadIdx.x == 0){
                    shared_columnProperties = *(multiMSA.getColumnPropertiesOfMSA(anchorIndex));
                }
                tbGroup.sync();

                const read_number anchorReadId = anchorReadIds == nullptr ? 0 : anchorReadIds[anchorIndex];
                constexpr read_number debugreadid = 42;

                const int globalOffset = d_candidates_per_anchor_prefixsum[anchorIndex];
                int* const myIndices = d_indices + globalOffset;
                int* const myNumIndicesPerAnchorPtr = d_indices_per_anchor + anchorIndex;

                int* const myNewIndicesPtr = d_newIndices + globalOffset;
                int* const myNewNumIndicesPerAnchorPtr = d_newNumIndicesPerAnchor + anchorIndex;

                bool* const myShouldBeKept = d_shouldBeKept + globalOffset;
                const bool* const myIsPairedCandidate = d_isPairedCandidate + globalOffset;

                GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);
                msa.columnProperties = &shared_columnProperties;

                msa.debugflag = anchorReadId == debugreadid;

                if(useSmemMSA){
                    msa.counts = shared_counts;
                    msa.weights = shared_weights;
                    msa.coverages = shared_coverages;
                }

                //TODO find out why memcpy path is slower

                #if 0
                if(useSmemMSA){
                    //load counts weights and coverages from gmem to smem

                    const int* const gmemCounts = multiMSA.getCountsOfMSA(anchorIndex);
                    const float* const gmemWeights = multiMSA.getWeightsOfMSA(anchorIndex);
                    const int* const gmemCoverages = multiMSA.getCoveragesOfMSA(anchorIndex);
                    const int columns = msa.columnProperties->lastColumn_excl;

                    care::gpu::memcpy<int>(tbGroup, shared_coverages, gmemCoverages,  sizeof(int) * columns);

                    for(int k = 0; k < 4; k++){
                        int* const destCounts = shared_counts + k * msa.columnPitchInElements;
                        const int* const srcCounts = gmemCounts + k * msa.columnPitchInElements;
                        care::gpu::memcpy<int>(tbGroup, destCounts, srcCounts, sizeof(int) * columns);
                    }

                    for(int k = 0; k < 4; k++){
                        float* const destWeights = shared_weights + k * msa.columnPitchInElements;
                        const float* const srcWeights = gmemWeights + k * msa.columnPitchInElements;
                        care::gpu::memcpy<int>(tbGroup, destWeights, srcWeights, sizeof(float) * columns);
                    }
                }
                #else
                if(useSmemMSA){
                    //load counts weights and coverages from gmem to smem

                    const int* const gmemCounts = multiMSA.getCountsOfMSA(anchorIndex);
                    const float* const gmemWeights = multiMSA.getWeightsOfMSA(anchorIndex);
                    const int* const gmemCoverages = multiMSA.getCoveragesOfMSA(anchorIndex);
                    const int columns = msa.columnProperties->lastColumn_excl;

                    for(int k = tbGroup.thread_rank(); k < columns; k += tbGroup.size()){
                        for(int i = 0; i < 4; i++){
                            shared_counts[k + i * msa.columnPitchInElements] 
                                = gmemCounts[k + i * msa.columnPitchInElements];
                            shared_weights[k + i * msa.columnPitchInElements] 
                                = gmemWeights[k + i * msa.columnPitchInElements];
                        }
                        shared_coverages[k] = gmemCoverages[k];
                    }
                }

                #endif

                #if 0
                auto storeSmemMSAToGmem = [&](){
                    int* const gmemCounts = multiMSA.getCountsOfMSA(anchorIndex);
                    float* const gmemWeights = multiMSA.getWeightsOfMSA(anchorIndex);
                    int* const gmemCoverages = multiMSA.getCoveragesOfMSA(anchorIndex);
                    const int columns = msa.columnProperties->lastColumn_excl;

                    care::gpu::memcpy<int>(tbGroup, gmemCoverages, shared_coverages, sizeof(int) * columns);

                    for(int k = 0; k < 4; k++){
                        const int* const srcCounts = shared_counts + k * msa.columnPitchInElements;
                        int* const destCounts = gmemCounts + k * msa.columnPitchInElements;
                        care::gpu::memcpy<int>(tbGroup, destCounts, srcCounts, sizeof(int) * columns);
                    }

                    for(int k = 0; k < 4; k++){
                        const float* const srcWeights = shared_weights + k * msa.columnPitchInElements;
                        float* const destWeights = gmemWeights + k * msa.columnPitchInElements;
                        care::gpu::memcpy<int>(tbGroup, destWeights, srcWeights, sizeof(float) * columns);
                    }
                };
                #else
                auto storeSmemMSAToGmem = [&](){
                    int* const gmemCounts = multiMSA.getCountsOfMSA(anchorIndex);
                    float* const gmemWeights = multiMSA.getWeightsOfMSA(anchorIndex);
                    int* const gmemCoverages = multiMSA.getCoveragesOfMSA(anchorIndex);
                    const int columns = msa.columnProperties->lastColumn_excl;

                    for(int k = tbGroup.thread_rank(); k < columns; k += tbGroup.size()){
                        for(int i = 0; i < 4; i++){
                            gmemCounts[k + i * msa.columnPitchInElements] 
                                = shared_counts[k + i * msa.columnPitchInElements];
                            gmemWeights[k + i * msa.columnPitchInElements] 
                                = shared_weights[k + i * msa.columnPitchInElements];
                        }
                        gmemCoverages[k] = shared_coverages[k];
                    }
                };
                #endif

                const AlignmentOrientation* const myAlignmentFlags = bestAlignmentFlags + globalOffset;
                const int* const myShifts = shifts + globalOffset;
                const int* const myNops = nOps + globalOffset;
                const int* const myOverlaps = overlaps + globalOffset;

                const unsigned int* const myAnchorSequenceData = anchorSequencesData 
                    + std::size_t(anchorIndex) * encodedSequencePitchInInts;
                const unsigned int* const myCandidateSequencesData = candidateSequencesData 
                    + std::size_t(globalOffset) * encodedSequencePitchInInts;

                const char* const myCandidateQualities = candidateQualities 
                    + std::size_t(globalOffset) * qualityPitchInBytes;

                const int anchorLength = anchorSequencesLength[anchorIndex];
                const int* const myCandidateLengths = candidateSequencesLength + globalOffset;

                for(int refinementIteration = 0; 
                        refinementIteration < numRefinementIterations; 
                        refinementIteration++){

                    auto finalizeRefinement = [&](int newNumIndicesPerAnchor){
                        //copy indices to correct output array
                        if(refinementIteration % 2 == 1){
                            for(int i = tbGroup.thread_rank(); i < myNumIndices; i += tbGroup.size()){
                                myNewIndicesPtr[i] = myIndices[i];
                            }
                            if(tbGroup.thread_rank() == 0){
                                *myNewNumIndicesPerAnchorPtr = *myNumIndicesPerAnchorPtr;
                            }
                        }

                        if(tbGroup.thread_rank() == 0){
                            atomicAdd(d_newNumIndices, newNumIndicesPerAnchor);
                        }
                    };

                    int* const srcIndices = (refinementIteration % 2 == 0) ?
                            myIndices : myNewIndicesPtr;
                    int* const destIndices = (refinementIteration % 2 == 0) ?
                            myNewIndicesPtr : myIndices;

                    int* const srcNumIndices = (refinementIteration % 2 == 0) ?
                        myNumIndicesPerAnchorPtr : myNewNumIndicesPerAnchorPtr;
                    int* const destNumIndices = (refinementIteration % 2 == 0) ?
                        myNewNumIndicesPerAnchorPtr : myNumIndicesPerAnchorPtr;

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
                        anchorLength,
                        myCandidateSequencesData,
                        myCandidateLengths,
                        myAlignmentFlags,
                        myShifts,
                        myNops,
                        myOverlaps,
                        myShouldBeKept,
                        desiredAlignmentMaxErrorRate,
                        anchorIndex,
                        encodedSequencePitchInInts,
                        srcIndices,
                        *srcNumIndices,
                        dataset_coverage
                    );

                    tbGroup.sync();

                    long long int t2 = clock64();

                    if(anchorIndex == 0 && tbGroup.thread_rank() == 0){
                        //printf("duration flag: %lu\n", t2-t1);
                    }

                    const int myNewNumIndices = *destNumIndices;
                    
                    assert(myNewNumIndices <= myNumIndices);
                    if(myNewNumIndices > 0 && myNewNumIndices < myNumIndices){
                        auto selector = [&](int i){
                            return !myShouldBeKept[i];
                        };

                        long long int t3 = clock64();

                        // bool mychecksuccess = msa.checkCoverages(tbGroup);
                        // bool checksuccess = groupReduceBool(mychecksuccess, [](auto l, auto r){
                        //     return l && r;
                        // });
                        // tbGroup.sync();
                        // if(tbGroup.thread_rank() == 0){
                        //     if(!checksuccess){
                        //         const int firstColumnIncl = msa.columnProperties->firstColumn_incl;
                        //         const int lastColumnExcl = msa.columnProperties->lastColumn_excl;

                        //         printf("firstColumnIncl %d, lastColumnExcl %d\n", firstColumnIncl, lastColumnExcl);
                        //         printf("coverages:\n");
                        //         for(int x = 0; x < msa.columnPitchInElements; x++){
                        //             printf("%d, ", msa.coverages[x]);
                        //         }
                        //         printf("\n");
                        //         assert(checksuccess);
                        //     }
                        // }
                        // tbGroup.sync();


                        // if(tbGroup.thread_rank() == 0 && anchorReadId == debugreadid && refinementIteration >= 0){
                        //     printf("counts before remove in iteration %d\n", refinementIteration);
                        //     msa.printCounts(275,285);
                        //     printf("weights before remove in iteration %d\n", refinementIteration);
                        //     msa.printWeights(275,285);
                        // }
                        // tbGroup.sync();

                        msa.checkAfterBuild(tbGroup, anchorIndex, __LINE__, anchorReadId);
                        tbGroup.sync();

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
                            myIsPairedCandidate,
                            *srcNumIndices,
                            canUseQualityScores, 
                            encodedSequencePitchInInts,
                            qualityPitchInBytes,
                            desiredAlignmentMaxErrorRate
                        );

                        tbGroup.sync();

                        msa.setWeightsToZeroIfCountIsZero(tbGroup);
                        tbGroup.sync();

                        // mychecksuccess = msa.checkCoverages(tbGroup);
                        // checksuccess = groupReduceBool(mychecksuccess, [](auto l, auto r){
                        //     return l && r;
                        // });
                        // tbGroup.sync();
                        // if(tbGroup.thread_rank() == 0){
                        //     if(!checksuccess){
                        //         const int firstColumnIncl = msa.columnProperties->firstColumn_incl;
                        //         const int lastColumnExcl = msa.columnProperties->lastColumn_excl;

                        //         printf("firstColumnIncl %d, lastColumnExcl %d\n", firstColumnIncl, lastColumnExcl);
                        //         printf("coverages:\n");
                        //         for(int x = 0; x < msa.columnPitchInElements; x++){
                        //             printf("%d, ", msa.coverages[x]);
                        //         }
                        //         printf("\n");
                        //         assert(checksuccess);
                        //     }
                        // }
                        // tbGroup.sync();

                        long long int t4 = clock64();

                        if(anchorIndex == 0 && tbGroup.thread_rank() == 0){
                            //printf("duration removal: %lu. candidates before %d, after %d\n", t4-t3, myNumIndices, myNewNumIndices);
                        }

                        bool error = msa.updateColumnProperties(tbGroup);

                        tbGroup.sync();

                        // if(tbGroup.thread_rank() == 0 && anchorReadId == debugreadid && refinementIteration >= 0){
                        //     printf("counts after remove in iteration %d\n", refinementIteration);
                        //     msa.printCounts(275,285);
                        //     printf("weights after remove in iteration %d\n", refinementIteration);
                        //     msa.printWeights(275,285);
                        // }
                        // tbGroup.sync();


                        msa.checkAfterBuild(tbGroup, anchorIndex, __LINE__, anchorReadId);
                        tbGroup.sync();

                        // if(error){
                        //     if(tbGroup.thread_rank() == 0){
                        //         printf("error updateColumnProperties\n");
                        //         printf("shifts")
                        //     }
                        // }
                        // tbGroup.sync();
                        assert(!error);

                        //msa.checkAfterBuild(tbGroup, anchorIndex);

                        assert(shared_columnProperties.firstColumn_incl != -1);
                        assert(shared_columnProperties.lastColumn_excl != -1);

                        long long int t5 = clock64();

                        msa.findConsensus(
                            tbGroup,
                            myAnchorSequenceData, 
                            encodedSequencePitchInInts, 
                            anchorIndex
                        );

                        if(tbGroup.thread_rank() == 0){
                            *(multiMSA.getColumnPropertiesOfMSA(anchorIndex)) = shared_columnProperties;
                        }

                        tbGroup.sync();

                        long long int t6 = clock64();

                        if(anchorIndex == 0 && tbGroup.thread_rank() == 0){
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
                    d_newNumIndicesPerAnchor[anchorIndex] = 0;
                }
                ; //nothing else to do if there are no candidates in msa
            }                
        }
    }








    //####################   KERNEL DISPATCH   ####################

    void callComputeMaximumMsaWidthKernel(
        int* d_result,
        const int* d_shifts,
        const int* d_anchorLengths,
        const int* d_candidateLengths,
        const int* d_indices,
        const int* d_indices_per_anchor,
        const int* d_candidatesPerAnchorPrefixSum,
        const int numAnchors,
        cudaStream_t stream
    ){
        CUDACHECK(cudaMemsetAsync(d_result, 0, sizeof(int), stream));

        computeMaximumMsaWidthKernel<128><<<numAnchors, 128, 0, stream>>>(
            d_result,
            d_shifts,
            d_anchorLengths,
            d_candidateLengths,
            d_indices,
            d_indices_per_anchor,
            d_candidatesPerAnchorPrefixSum,
            numAnchors
        );
    }

    void callComputeMsaConsensusQualityKernel(
        char* d_consensusQuality,
        int consensusQualityPitchInBytes,
        GPUMultiMSA multiMSA,
        cudaStream_t stream
    ){
        dim3 block = 128;
        dim3 grid = multiMSA.numMSAs;
        computeMsaConsensusQualityKernel<<<grid, block, 0, stream>>>(
            d_consensusQuality,
            consensusQualityPitchInBytes,
            multiMSA
        ); CUDACHECKASYNC;
    }

    void callComputeDecodedMsaConsensusKernel(
        char* d_consensus,
        int consensusPitchInBytes,
        GPUMultiMSA multiMSA,
        cudaStream_t stream
    ){
        dim3 block = 128;
        dim3 grid = multiMSA.numMSAs;
        computeDecodedMsaConsensusKernel<<<grid, block, 0, stream>>>(
            d_consensus,
            consensusPitchInBytes,
            multiMSA
        ); CUDACHECKASYNC;
    }

    void callComputeMsaSizesKernel(
        int* d_sizes,
        GPUMultiMSA multiMSA,
        cudaStream_t stream
    ){
        dim3 block = 128;
        dim3 grid = SDIV(multiMSA.numMSAs, block.x);

        computeMsaSizesKernel<<<grid, block, 0, stream>>>(
            d_sizes,
            multiMSA
        ); CUDACHECKASYNC;
    }
    
    void callMsaCandidateRefinementKernel_multiiter_async(
        int* d_newIndices,
        int* d_newNumIndicesPerAnchor,
        int* d_newNumIndices,
        GPUMultiMSA multiMSA,
        const AlignmentOrientation* d_bestAlignmentFlags,
        const int* d_shifts,
        const int* d_nOps,
        const int* d_overlaps,
        const unsigned int* d_anchorSequencesData,
        const unsigned int* d_candidateSequencesData,
        const bool* d_isPairedCandidate,
        const int* d_anchorSequencesLength,
        const int* d_candidateSequencesLength,
        const char* d_anchorQualities,
        const char* d_candidateQualities,
        bool* d_shouldBeKept,
        const int* d_candidates_per_anchor_prefixsum,
        const int* d_numAnchors,
        float desiredAlignmentMaxErrorRate,
        int /*maxNumAnchors*/,
        int /*maxNumCandidates*/,
        bool canUseQualityScores,
        size_t encodedSequencePitchInInts,
        size_t qualityPitchInBytes,
        int* d_indices,
        int* d_indices_per_anchor,
        int dataset_coverage,
        int numIterations,
        cudaStream_t stream,
        const read_number* d_anchorReadIds
    ){

        helpers::call_fill_kernel_async(
            d_newNumIndices,
            1,
            0,
            stream
        );

        constexpr int blocksize = 128;

        constexpr MemoryType memoryType = MemoryType::Shared;

        constexpr bool usesSmem = memoryType == MemoryType::Shared;

        const std::size_t smemAddSequences = (usesSmem ? 
                                                sizeof(float) * 4 * multiMSA.columnPitchInElements // weights
                                                    + sizeof(int) * 4 * multiMSA.columnPitchInElements // counts
                                                    + sizeof(int) * multiMSA.columnPitchInElements // coverages
                                                : 0);

        const std::size_t smem = smemAddSequences;

        int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            msaCandidateRefinement_multiiter_kernel<blocksize, memoryType>,
            blocksize, 
            smem
        ));

        const int maxBlocks = maxBlocksPerSM * numSMs;

        dim3 block(blocksize, 1, 1);
        //dim3 grid(maxNumAnchors, 1, 1);
        dim3 grid(maxBlocks, 1, 1);


        msaCandidateRefinement_multiiter_kernel<blocksize, memoryType>
                <<<grid, block, smem, stream>>>(
            d_newIndices,
            d_newNumIndicesPerAnchor,
            d_newNumIndices,
            multiMSA,
            d_bestAlignmentFlags,
            d_shifts,
            d_nOps,
            d_overlaps,
            d_anchorSequencesData,
            d_candidateSequencesData,
            d_isPairedCandidate,
            d_anchorSequencesLength,
            d_candidateSequencesLength,
            d_anchorQualities,
            d_candidateQualities,
            d_shouldBeKept,
            d_candidates_per_anchor_prefixsum,
            d_numAnchors,
            desiredAlignmentMaxErrorRate,
            canUseQualityScores,
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            d_indices,
            d_indices_per_anchor,
            dataset_coverage,
            numIterations,
            d_anchorReadIds
        );

        CUDACHECKASYNC;
    }





    void callMsaCandidateRefinementKernel_singleiter_async(
        int* d_newIndices,
        int* d_newNumIndicesPerAnchor,
        int* d_newNumIndices,
        GPUMultiMSA multiMSA,
        const AlignmentOrientation* d_bestAlignmentFlags,
        const int* d_shifts,
        const int* d_nOps,
        const int* d_overlaps,
        const unsigned int* d_anchorSequencesData,
        const unsigned int* d_candidateSequencesData,
        const bool* d_isPairedCandidate,
        const int* d_anchorSequencesLength,
        const int* d_candidateSequencesLength,
        const char* d_anchorQualities,
        const char* d_candidateQualities,
        bool* d_shouldBeKept,
        const int* d_candidates_per_anchor_prefixsum,
        const int* d_numAnchors,
        float desiredAlignmentMaxErrorRate,
        int /*maxNumAnchors*/,
        int /*maxNumCandidates*/,
        bool canUseQualityScores,
        size_t encodedSequencePitchInInts,
        size_t qualityPitchInBytes,
        const int* d_indices,
        const int* d_indices_per_anchor,
        int dataset_coverage,
        int iteration,
        bool* d_anchorIsFinished,
        cudaStream_t stream
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

        int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            msaCandidateRefinement_singleiter_kernel<blocksize, memoryType>,
            blocksize, 
            smem
        ));

        const int maxBlocks = maxBlocksPerSM * numSMs;

        dim3 block(blocksize, 1, 1);
        //dim3 grid(maxNumAnchors, 1, 1);
        dim3 grid(maxBlocks, 1, 1);

        msaCandidateRefinement_singleiter_kernel<blocksize, memoryType><<<grid, block, smem, stream>>>(
            d_newIndices,
            d_newNumIndicesPerAnchor,
            d_newNumIndices,
            multiMSA,
            d_bestAlignmentFlags,
            d_shifts,
            d_nOps,
            d_overlaps,
            d_anchorSequencesData,
            d_candidateSequencesData,
            d_isPairedCandidate,
            d_anchorSequencesLength,
            d_candidateSequencesLength,
            d_anchorQualities,
            d_candidateQualities,
            d_shouldBeKept,
            d_candidates_per_anchor_prefixsum,
            d_numAnchors,
            desiredAlignmentMaxErrorRate,
            canUseQualityScores,
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            d_indices,
            d_indices_per_anchor,
            dataset_coverage,
            iteration,
            d_anchorIsFinished
        );

        CUDACHECKASYNC;
    }





    void callConstructMultipleSequenceAlignmentsKernel_async(
        GPUMultiMSA multiMSA,
        const int* d_overlaps,
        const int* d_shifts,
        const int* d_nOps,
        const AlignmentOrientation* d_bestAlignmentFlags,
        const int* d_anchorLengths,
        const int* d_candidateLengths,
        const int* d_indices,
        const int* d_indices_per_anchor,
        const int* d_candidatesPerAnchorPrefixSum,            
        const unsigned int* d_anchorSequencesData,
        const unsigned int* d_candidateSequencesData,
        const bool* d_isPairedCandidate,
        const char* d_anchorQualities,
        const char* d_candidateQualities,
        const int* d_numAnchors,
        float desiredAlignmentMaxErrorRate,
        int /*maxNumAnchors*/,
        int /*maxNumCandidates*/,
        bool canUseQualityScores,
        int encodedSequencePitchInInts,
        size_t qualityPitchInBytes,
        cudaStream_t stream){
            

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

    int deviceId = 0;
    int numSMs = 0;
    int maxBlocksPerSM = 0;
    CUDACHECK(cudaGetDevice(&deviceId));
    CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        constructMultipleSequenceAlignmentsKernel<BLOCKSIZE, memoryType>,
        BLOCKSIZE, 
        smem
    ));

    const int maxBlocks = maxBlocksPerSM * numSMs;

    dim3 block(BLOCKSIZE, 1, 1);
    //dim3 grid(maxNumAnchors, 1, 1);
    dim3 grid(maxBlocks, 1, 1);
    
    constructMultipleSequenceAlignmentsKernel<BLOCKSIZE, memoryType><<<grid, block, smem, stream>>>(
        multiMSA,         
        d_overlaps,
        d_shifts,
        d_nOps,
        d_bestAlignmentFlags,
        d_anchorLengths,
        d_candidateLengths,
        d_indices,
        d_indices_per_anchor,
        d_candidatesPerAnchorPrefixSum,            
        d_anchorSequencesData,
        d_candidateSequencesData,
        d_isPairedCandidate,
        d_anchorQualities,
        d_candidateQualities,
        d_numAnchors,
        desiredAlignmentMaxErrorRate,
        canUseQualityScores,
        encodedSequencePitchInInts,
        qualityPitchInBytes
    ); CUDACHECKASYNC;



}





}
}
