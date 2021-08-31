//#define NDEBUG

#include <gpu/kernels.hpp>
#include <gpu/kernellaunch.hpp>
#include <hostdevicefunctions.cuh>
#include <gpu/gpumsa.cuh>
#include <gpu/cudaerrorcheck.cuh>
#include <bestalignment.hpp>

#include <sequencehelpers.hpp>
#include <correctedsequence.hpp>

#include <hpc_helpers.cuh>
#include <config.hpp>
#include <cassert>

#include <gpu/forest_gpu.cuh>
#include <gpu/classification_gpu.cuh>

#include <cub/cub.cuh>

#include <cooperative_groups.h>

#include <thrust/functional.h>

namespace cg = cooperative_groups;

namespace care{
namespace gpu{


    template<int BLOCKSIZE>
    __global__
    void msaCorrectAnchorsKernel(
        char* __restrict__ correctedSubjects,
        bool* __restrict__ subjectIsCorrected,
        AnchorHighQualityFlag* __restrict__ isHighQualitySubject,
        GPUMultiMSA multiMSA,
        const unsigned int* __restrict__ subjectSequencesData,
        const int* __restrict__ d_indices_per_subject,
        const int* __restrict__ numAnchorsPtr,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        float estimatedErrorrate,
        float avg_support_threshold,
        float min_support_threshold,
        float min_coverage_threshold
    ){

        auto tbGroup = cg::this_thread_block();
        auto thread = cg::this_thread();

        const int n_subjects = *numAnchorsPtr;

        for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
            const int myNumIndices = d_indices_per_subject[subjectIndex];
            if(myNumIndices > 0){

                const GpuSingleMSA msa = multiMSA.getSingleMSA(subjectIndex);                

                const int subjectColumnsBegin_incl = msa.columnProperties->subjectColumnsBegin_incl;
                const int subjectColumnsEnd_excl = msa.columnProperties->subjectColumnsEnd_excl;

                auto i_f = thrust::identity<float>{};
                auto i_i = thrust::identity<int>{};

                GpuMSAProperties msaProperties = msa.getMSAProperties(
                    thread, i_f, i_f, i_i, i_i,
                    subjectColumnsBegin_incl,
                    subjectColumnsEnd_excl
                );

                AnchorCorrectionQuality correctionQuality(avg_support_threshold, min_support_threshold, min_coverage_threshold, estimatedErrorrate);

                const bool canBeCorrectedBySimpleConsensus = correctionQuality.canBeCorrectedBySimpleConsensus(msaProperties.avg_support, msaProperties.min_support, msaProperties.min_coverage);
                const bool isHQCorrection = correctionQuality.isHQCorrection(msaProperties.avg_support, msaProperties.min_support, msaProperties.min_coverage);

                if(tbGroup.thread_rank() == 0){
                    subjectIsCorrected[subjectIndex] = true;
                    isHighQualitySubject[subjectIndex].hq(isHQCorrection);
                }

                char* const my_corrected_subject = correctedSubjects + subjectIndex * decodedSequencePitchInBytes;

                if(canBeCorrectedBySimpleConsensus){
                    for(int i = subjectColumnsBegin_incl + tbGroup.thread_rank(); 
                            i < subjectColumnsEnd_excl; 
                            i += tbGroup.size()){

                        const std::uint8_t nuc = msa.consensus[i];
                        assert(nuc < 4);

                        my_corrected_subject[i - subjectColumnsBegin_incl] = SequenceHelpers::decodeBase(nuc);
                    }
                }else{
                    //correct only positions with high support.
                    for(int i = subjectColumnsBegin_incl + tbGroup.thread_rank(); 
                            i < subjectColumnsEnd_excl; 
                            i += tbGroup.size()){

                        
                        if(msa.support[i] > 0.90f && msa.origCoverages[i] <= 2){
                            my_corrected_subject[i - subjectColumnsBegin_incl] = SequenceHelpers::decodeBase(msa.consensus[i]);
                        }else{
                            const unsigned int* const subject = subjectSequencesData + std::size_t(subjectIndex) * encodedSequencePitchInInts;
                            const std::uint8_t encodedBase = SequenceHelpers::getEncodedNuc2Bit(subject, subjectColumnsEnd_excl- subjectColumnsBegin_incl, i - subjectColumnsBegin_incl);
                            const char base = SequenceHelpers::decodeBase(encodedBase);
                            assert(base == 'A' || base == 'C' || base == 'G' || base == 'T');
                            my_corrected_subject[i - subjectColumnsBegin_incl] = base;
                        }
                    }
                }
            }else{
                if(tbGroup.thread_rank() == 0){
                    isHighQualitySubject[subjectIndex].hq(false);
                    subjectIsCorrected[subjectIndex] = false;
                }
            }
        }
    }


    template<int BLOCKSIZE, class AnchorExtractor, class GpuClf>
    __global__
    void msaCorrectAnchorsWithForestKernel(
        char* __restrict__ correctedSubjects,
        bool* __restrict__ subjectIsCorrected,
        AnchorHighQualityFlag* __restrict__ isHighQualitySubject,
        GPUMultiMSA multiMSA,
        GpuClf gpuForest,
        float forestThreshold,
        const unsigned int* __restrict__ subjectSequencesData,
        const int* __restrict__ d_indices_per_subject,
        const int numAnchors,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        float estimatedErrorrate,
        float estimatedCoverage,
        float avg_support_threshold,
        float min_support_threshold,
        float min_coverage_threshold
    ){

        constexpr int subgroupsize = 32;
        constexpr int numSubGroupsInBlock = BLOCKSIZE / subgroupsize;

        static_assert(subgroupsize == 32);
        static_assert(BLOCKSIZE % subgroupsize == 0);

        auto tbGroup = cg::this_thread_block();
        auto subgroup = cg::tiled_partition<subgroupsize>(tbGroup);
        auto thread = cg::this_thread();

        const int subgroupIdInBlock = threadIdx.x / subgroupsize;

        using WarpReduceFloat = cub::WarpReduce<float>;    

        __shared__ typename WarpReduceFloat::TempStorage warpfloatreducetemp[numSubGroupsInBlock];       
        __shared__ float sharedFeatures[numSubGroupsInBlock][AnchorExtractor::numFeatures()];
        __shared__ ExtractAnchorInputData sharedExtractInput[numSubGroupsInBlock];

        extern __shared__ int externalsmem[];
        
        char* const sharedCorrectedAnchor = (char*)&externalsmem[0];

        auto subgroupReduceFloatSum = [&](float f){
            const float result = WarpReduceFloat(warpfloatreducetemp[subgroupIdInBlock]).Sum(f);
            subgroup.sync();
            return result;
        };

        for(unsigned subjectIndex = blockIdx.x; subjectIndex < numAnchors; subjectIndex += gridDim.x){
            const int myNumIndices = d_indices_per_subject[subjectIndex];
            if(myNumIndices > 0){

                const GpuSingleMSA msa = multiMSA.getSingleMSA(subjectIndex);                

                const int subjectColumnsBegin_incl = msa.columnProperties->subjectColumnsBegin_incl;
                const int subjectColumnsEnd_excl = msa.columnProperties->subjectColumnsEnd_excl;

                auto i_f = thrust::identity<float>{};
                auto i_i = thrust::identity<int>{};

                GpuMSAProperties msaProperties = msa.getMSAProperties(
                    thread, i_f, i_f, i_i, i_i,
                    subjectColumnsBegin_incl,
                    subjectColumnsEnd_excl
                );

                AnchorCorrectionQuality correctionQuality(avg_support_threshold, min_support_threshold, min_coverage_threshold, estimatedErrorrate);

                const bool canBeCorrectedBySimpleConsensus = correctionQuality.canBeCorrectedBySimpleConsensus(msaProperties.avg_support, msaProperties.min_support, msaProperties.min_coverage);
                const bool isHQCorrection = correctionQuality.isHQCorrection(msaProperties.avg_support, msaProperties.min_support, msaProperties.min_coverage);

                if(tbGroup.thread_rank() == 0){
                    subjectIsCorrected[subjectIndex] = true;
                    isHighQualitySubject[subjectIndex].hq(isHQCorrection);
                }

                const int anchorLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
                const unsigned int* const subject = subjectSequencesData + std::size_t(subjectIndex) * encodedSequencePitchInInts;
                char* const globalCorrectedAnchor = correctedSubjects + subjectIndex * decodedSequencePitchInBytes;

                if(isHQCorrection){

                    //set corrected anchor to consensus
                    for(int i = tbGroup.thread_rank(); i < anchorLength; i += tbGroup.size()){
                        const std::uint8_t nuc = msa.consensus[subjectColumnsBegin_incl + i];
                        assert(nuc < 4);
                        globalCorrectedAnchor[i] = SequenceHelpers::decodeBase(nuc);
                    }

                }else{

                    //set corrected anchor to consensus
                    for(int i = tbGroup.thread_rank(); i < anchorLength; i += tbGroup.size()){
                        const std::uint8_t nuc = msa.consensus[subjectColumnsBegin_incl + i];
                        assert(nuc < 4);
                        sharedCorrectedAnchor[i] = SequenceHelpers::decodeBase(nuc);
                    }
                    
                    tbGroup.sync();                                   
                    
                    //maybe revert some positions to original base
                    for (int i = subgroupIdInBlock; i < anchorLength; i += numSubGroupsInBlock){
                        const int msaPos = subjectColumnsBegin_incl + i;
                        const std::uint8_t origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(subject, anchorLength, i);
                        const std::uint8_t consensusEncodedBase = msa.consensus[msaPos];

                        if (origEncodedBase != consensusEncodedBase){                            

                            if(subgroup.thread_rank() == 0){

                                ExtractAnchorInputData& extractorInput = sharedExtractInput[subgroupIdInBlock];

                                extractorInput.origBase = SequenceHelpers::decodeBase(origEncodedBase);
                                extractorInput.consensusBase = SequenceHelpers::decodeBase(consensusEncodedBase);
                                extractorInput.estimatedCoverage = estimatedCoverage;
                                extractorInput.msaPos = msaPos;
                                extractorInput.subjectColumnsBegin_incl = subjectColumnsBegin_incl;
                                extractorInput.subjectColumnsEnd_excl = subjectColumnsEnd_excl;
                                extractorInput.msaProperties = msaProperties;
                                extractorInput.msa = msa;

                                AnchorExtractor extractFeatures{};
                                extractFeatures(&sharedFeatures[subgroupIdInBlock][0], extractorInput);
                            }

                            subgroup.sync();

                            //only thread 0 of group has valid result
                            const bool useConsensus = gpuForest.decide(subgroup, &sharedFeatures[subgroupIdInBlock][0], forestThreshold, subgroupReduceFloatSum);
                            if(subgroup.thread_rank() == 0){
                                if(!useConsensus){
                                    sharedCorrectedAnchor[i] = sharedExtractInput[subgroupIdInBlock].origBase;
                                }
                            }
                        }
                    }

                    tbGroup.sync();

                    //copy shared correction to gmem
                    const int fullInts1 = anchorLength / sizeof(int);

                    for(int i = tbGroup.thread_rank(); i < fullInts1; i += tbGroup.size()) {
                        ((int*)globalCorrectedAnchor)[i] = ((int*)sharedCorrectedAnchor)[i];
                    }

                    for(int i = tbGroup.thread_rank(); i < anchorLength - fullInts1 * sizeof(int); i += tbGroup.size()) {
                        globalCorrectedAnchor[fullInts1 * sizeof(int) + i] 
                            = sharedCorrectedAnchor[fullInts1 * sizeof(int) + i];
                    } 

                }
                
                tbGroup.sync();

            }else{
                if(tbGroup.thread_rank() == 0){
                    isHighQualitySubject[subjectIndex].hq(false);
                    subjectIsCorrected[subjectIndex] = false;
                }
            }            
        }
    }


    


    template<int BLOCKSIZE, int groupsize, class CandidateExtractor, class GpuClf>
    __global__
    void msaCorrectCandidatesWithForestKernel(
        char* __restrict__ correctedCandidates,
        EncodedCorrectionEdit* __restrict__ d_editsPerCorrectedCandidate,
        int* __restrict__ d_numEditsPerCorrectedCandidate,
        GPUMultiMSA multiMSA,
        GpuClf gpuForest,
        float forestThreshold,
        float estimatedCoverage,
        const int* __restrict__ shifts,
        const BestAlignment_t* __restrict__ bestAlignmentFlags,
        const unsigned int* __restrict__ candidateSequencesData,
        const int* __restrict__ candidateSequencesLengths,
        const bool* __restrict__ d_candidateContainsN,
        const int* __restrict__ candidateIndicesOfCandidatesToBeCorrected,
        const int* __restrict__ numCandidatesToBeCorrected,
        const int* __restrict__ anchorIndicesOfCandidates,
        int doNotUseEditsValue,
        int numEditsThreshold,            
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        size_t editsPitchInBytes,
        size_t dynamicsmemSequencePitchInInts,
        const read_number* candidateReadIds
    ){

        /*
            Use groupsize threads per candidate to perform correction
        */
        static_assert(BLOCKSIZE % groupsize == 0, "BLOCKSIZE % groupsize != 0");
        constexpr int groupsPerBlock = BLOCKSIZE / groupsize;
        static_assert(groupsize == 32);

        auto tgroup = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int numGroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int groupIdInBlock = threadIdx.x / groupsize;

        auto thread = cg::this_thread();

        using WarpReduceFloat = cub::WarpReduce<float>;

        __shared__ typename WarpReduceFloat::TempStorage floatreduce[groupsPerBlock];
        __shared__ float sharedFeatures[groupsPerBlock][CandidateExtractor::numFeatures()];
        __shared__ ExtractCandidateInputData sharedExtractInput[groupsPerBlock];
       
        extern __shared__ int dynamicsmem[]; // for sequences

        auto groupReduceFloatSum = [&](float f){
            const float result = WarpReduceFloat(floatreduce[groupIdInBlock]).Sum(f);
            tgroup.sync();
            return result;
        };

        const std::size_t treePointersPitchInInts = SDIV(sizeof(void*) * gpuForest.numTrees, sizeof(int));

        const typename GpuClf::NodeType** sharedForestNodes = (const typename GpuClf::NodeType**)dynamicsmem;

        char* const shared_correctedCandidate = (char*)(dynamicsmem + treePointersPitchInInts + dynamicsmemSequencePitchInInts * groupIdInBlock);

        const int loopEnd = *numCandidatesToBeCorrected;

        GpuClf localForest;
        localForest.numTrees = gpuForest.numTrees;
        localForest.data = sharedForestNodes;

        for(int i = threadIdx.x; i < localForest.numTrees; i += BLOCKSIZE){
            localForest.data[i] = gpuForest.data[i];
        }
        
        __syncthreads();

        for(int id = groupId; id < loopEnd; id += numGroups){

            const int candidateIndex = candidateIndicesOfCandidatesToBeCorrected[id];
            const int subjectIndex = anchorIndicesOfCandidates[candidateIndex];
            const int destinationIndex = id;

            const GpuSingleMSA msa = multiMSA.getSingleMSA(subjectIndex);

            const int candidate_length = candidateSequencesLengths[candidateIndex];

            const int shift = shifts[candidateIndex];
            const int subjectColumnsBegin_incl = msa.columnProperties->subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = msa.columnProperties->subjectColumnsEnd_excl;
            const int queryColumnsBegin_incl = subjectColumnsBegin_incl + shift;
            const int queryColumnsEnd_excl = subjectColumnsBegin_incl + shift + candidate_length;

            auto i_f = thrust::identity<float>{};
            auto i_i = thrust::identity<int>{};

            GpuMSAProperties msaProperties = msa.getMSAProperties(
                thread, i_f, i_f, i_i, i_i,
                queryColumnsBegin_incl,
                queryColumnsEnd_excl
            );

            for(int i = tgroup.thread_rank(); i < candidate_length; i += tgroup.size()) {
                shared_correctedCandidate[i] = SequenceHelpers::decodeBase(msa.consensus[queryColumnsBegin_incl + i]);
            }

            tgroup.sync(); 

            const BestAlignment_t bestAlignmentFlag = bestAlignmentFlags[candidateIndex];

            const unsigned int* const encUncorrectedCandidate = candidateSequencesData 
                        + std::size_t(candidateIndex) * encodedSequencePitchInInts;

            for(int i = 0; i < candidate_length; i += 1){
                std::uint8_t origEncodedBase = 0;

                if(bestAlignmentFlag == BestAlignment_t::ReverseComplement){
                    origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(
                        encUncorrectedCandidate,
                        candidate_length,
                        candidate_length - i - 1
                    );
                    origEncodedBase = SequenceHelpers::complementBase2Bit(origEncodedBase);
                }else{
                    origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(
                        encUncorrectedCandidate,
                        candidate_length,
                        i
                    );
                }

                const char origBase = SequenceHelpers::decodeBase(origEncodedBase);
                const char consensusBase = shared_correctedCandidate[i];
                if(origBase != consensusBase){

                    const int msaPos = queryColumnsBegin_incl + i;

                    if(tgroup.thread_rank() == 0){
                        ExtractCandidateInputData& extractorInput = sharedExtractInput[groupIdInBlock];

                        extractorInput.origBase = origBase;
                        extractorInput.consensusBase = consensusBase;
                        extractorInput.estimatedCoverage = estimatedCoverage;
                        extractorInput.msaPos = msaPos;
                        extractorInput.subjectColumnsBegin_incl = subjectColumnsBegin_incl;
                        extractorInput.subjectColumnsEnd_excl = subjectColumnsEnd_excl;
                        extractorInput.queryColumnsBegin_incl = queryColumnsBegin_incl;
                        extractorInput.queryColumnsEnd_excl = queryColumnsEnd_excl;
                        extractorInput.msaProperties = msaProperties;
                        extractorInput.msa = msa;

                        CandidateExtractor extractFeatures{};
                        extractFeatures(&sharedFeatures[groupIdInBlock][0], extractorInput);
                    }

                    tgroup.sync();

                    //only thread 0 has valid result
                    //localForest gpuForest
                    const bool useConsensus = localForest.decide(tgroup, &sharedFeatures[groupIdInBlock][0], forestThreshold, groupReduceFloatSum);

                    if(tgroup.thread_rank() == 0){
                        if(!useConsensus){
                            shared_correctedCandidate[i] = origBase;
                        }
                    }

                    tgroup.sync();
                }
            }            

            tgroup.sync();

            //the forward strand will be returned -> make reverse complement again
            if(bestAlignmentFlag == BestAlignment_t::ReverseComplement) {
                SequenceHelpers::reverseComplementDecodedSequence(tgroup, shared_correctedCandidate, candidate_length);
            }else{
                //orientation ok
            }

            char* const my_corrected_candidate = correctedCandidates + destinationIndex * decodedSequencePitchInBytes;
            
            //copy corrected sequence from smem to global output
            const int fullInts1 = candidate_length / sizeof(int);

            for(int i = tgroup.thread_rank(); i < fullInts1; i += tgroup.size()) {
                ((int*)my_corrected_candidate)[i] = ((int*)shared_correctedCandidate)[i];
            }

            for(int i = tgroup.thread_rank(); i < candidate_length - fullInts1 * sizeof(int); i += tgroup.size()) {
                my_corrected_candidate[fullInts1 * sizeof(int) + i] 
                    = shared_correctedCandidate[fullInts1 * sizeof(int) + i];
            }

            tgroup.sync(); //sync before handling next candidate                        
        }
    }



    __device__ __forceinline__
    bool checkIfCandidateShouldBeCorrectedGlobal(
        const GpuSingleMSA& msa,
        const int alignmentShift,
        const int candidateLength,
        float min_support_threshold,
        float min_coverage_threshold,
        int new_columns_to_correct
    ){

        const auto columnProperties = *msa.columnProperties;

        const int& subjectColumnsBegin_incl = columnProperties.subjectColumnsBegin_incl;
        const int& subjectColumnsEnd_excl = columnProperties.subjectColumnsEnd_excl;
        const int& lastColumn_excl = columnProperties.lastColumn_excl;

        const int shift = alignmentShift;
        const int candidate_length = candidateLength;
        const int queryColumnsBegin_incl = subjectColumnsBegin_incl + shift;
        const int queryColumnsEnd_excl = subjectColumnsBegin_incl + shift + candidate_length;

        if(subjectColumnsBegin_incl - new_columns_to_correct <= queryColumnsBegin_incl
           && queryColumnsBegin_incl <= subjectColumnsBegin_incl + new_columns_to_correct
           && queryColumnsEnd_excl <= subjectColumnsEnd_excl + new_columns_to_correct) {

            float newColMinSupport = 1.0f;
            int newColMinCov = std::numeric_limits<int>::max();
            //check new columns left of subject
            for(int columnindex = subjectColumnsBegin_incl - new_columns_to_correct;
                columnindex < subjectColumnsBegin_incl;
                columnindex++) {

                assert(columnindex < lastColumn_excl);
                if(queryColumnsBegin_incl <= columnindex) {
                    newColMinSupport = msa.support[columnindex] < newColMinSupport ? msa.support[columnindex] : newColMinSupport;
                    newColMinCov = msa.coverages[columnindex] < newColMinCov ? msa.coverages[columnindex] : newColMinCov;
                }
            }
            //check new columns right of subject
            for(int columnindex = subjectColumnsEnd_excl;
                    columnindex < subjectColumnsEnd_excl + new_columns_to_correct
                        && columnindex < lastColumn_excl;
                    columnindex++) {

                newColMinSupport = msa.support[columnindex] < newColMinSupport ? msa.support[columnindex] : newColMinSupport;
                newColMinCov = msa.coverages[columnindex] < newColMinCov ? msa.coverages[columnindex] : newColMinCov;
            }

            bool result = fgeq(newColMinSupport, min_support_threshold)
                            && fgeq(newColMinCov, min_coverage_threshold);

            //return result;
            return true;
        }else{
            return false;
        }

    }

    __global__ 
    void flagCandidatesToBeCorrectedKernel(
        bool* __restrict__ candidateCanBeCorrected,
        int* __restrict__ numCorrectedCandidatesPerAnchor,
        GPUMultiMSA multiMSA,
        const int* __restrict__ alignmentShifts,
        const int* __restrict__ candidateSequencesLengths,
        const int* __restrict__ anchorIndicesOfCandidates,
        const AnchorHighQualityFlag* __restrict__ hqflags,
        const int* __restrict__ numCandidatesPerSubjectPrefixsum,
        const int* __restrict__ localGoodCandidateIndices,
        const int* __restrict__ numLocalGoodCandidateIndicesPerSubject,
        const int* __restrict__ d_numAnchors,
        const int* __restrict__ d_numCandidates,
        float min_support_threshold,
        float min_coverage_threshold,
        int new_columns_to_correct
    ){

        __shared__ int numAgg;

        const int n_subjects = *d_numAnchors;

        for(int anchorIndex = blockIdx.x; anchorIndex < n_subjects; anchorIndex += gridDim.x){

            if(threadIdx.x == 0){
                numAgg = 0;
            }
            __syncthreads();

            const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);

            const bool isHighQualitySubject = hqflags[anchorIndex].hq();
            const int numGoodIndices = numLocalGoodCandidateIndicesPerSubject[anchorIndex];
            const int dataoffset = numCandidatesPerSubjectPrefixsum[anchorIndex];
            const int* myGoodIndices = localGoodCandidateIndices + dataoffset;

            if(isHighQualitySubject){

                for(int tid = threadIdx.x; tid < numGoodIndices; tid += blockDim.x){
                    const int localCandidateIndex = myGoodIndices[tid];
                    const int globalCandidateIndex = dataoffset + localCandidateIndex;

                    const bool canHandleCandidate =  checkIfCandidateShouldBeCorrectedGlobal(
                        msa,
                        alignmentShifts[globalCandidateIndex],
                        candidateSequencesLengths[globalCandidateIndex],
                        min_support_threshold,
                        min_coverage_threshold,
                        new_columns_to_correct
                    );

                    candidateCanBeCorrected[globalCandidateIndex] = canHandleCandidate;

                    if(canHandleCandidate){
                        atomicAdd(&numAgg, 1);
                        //atomicAdd(numCorrectedCandidatesPerAnchor + anchorIndex, 1);
                    }
                }

                __syncthreads();

                if(threadIdx.x == 0){
                    numCorrectedCandidatesPerAnchor[anchorIndex] = numAgg;
                }
                
            }
        }
    }


    __global__ 
    void flagCandidatesToBeCorrectedWithExcludeFlagsKernel(
        bool* __restrict__ candidateCanBeCorrected,
        int* __restrict__ numCorrectedCandidatesPerAnchor,
        GPUMultiMSA multiMSA,
        const bool* __restrict__ excludeFlags,
        const int* __restrict__ alignmentShifts,
        const int* __restrict__ candidateSequencesLengths,
        const int* __restrict__ anchorIndicesOfCandidates,
        const AnchorHighQualityFlag* __restrict__ hqflags,
        const int* __restrict__ numCandidatesPerSubjectPrefixsum,
        const int* __restrict__ localGoodCandidateIndices,
        const int* __restrict__ numLocalGoodCandidateIndicesPerSubject,
        const int* __restrict__ d_numAnchors,
        const int* __restrict__ d_numCandidates,
        float min_support_threshold,
        float min_coverage_threshold,
        int new_columns_to_correct
    ){

        __shared__ int numAgg;

        const int n_subjects = *d_numAnchors;

        for(int anchorIndex = blockIdx.x; 
                anchorIndex < n_subjects; 
                anchorIndex += gridDim.x){

            if(threadIdx.x == 0){
                numAgg = 0;
            }
            __syncthreads();

            const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);

            const bool isHighQualitySubject = hqflags[anchorIndex].hq();
            const int numGoodIndices = numLocalGoodCandidateIndicesPerSubject[anchorIndex];
            const int dataoffset = numCandidatesPerSubjectPrefixsum[anchorIndex];
            const int* myGoodIndices = localGoodCandidateIndices + dataoffset;

            if(isHighQualitySubject){

                for(int tid = threadIdx.x; tid < numGoodIndices; tid += blockDim.x){
                    const int localCandidateIndex = myGoodIndices[tid];
                    const int globalCandidateIndex = dataoffset + localCandidateIndex;

                    const bool excludeCandidate = excludeFlags[globalCandidateIndex];

                    const bool canHandleCandidate = !excludeCandidate && checkIfCandidateShouldBeCorrectedGlobal(
                        msa,
                        alignmentShifts[globalCandidateIndex],
                        candidateSequencesLengths[globalCandidateIndex],
                        min_support_threshold,
                        min_coverage_threshold,
                        new_columns_to_correct
                    );

                    candidateCanBeCorrected[globalCandidateIndex] = canHandleCandidate;

                    if(canHandleCandidate){
                        atomicAdd(&numAgg, 1);
                        //atomicAdd(numCorrectedCandidatesPerAnchor + anchorIndex, 1);
                    }
                }

                __syncthreads();

                if(threadIdx.x == 0){
                    numCorrectedCandidatesPerAnchor[anchorIndex] = numAgg;
                }
                
            }
        }
    }


    template<int BLOCKSIZE, int groupsize>
    __global__
    void msaCorrectCandidatesKernel(
        char* __restrict__ correctedCandidates,
        EncodedCorrectionEdit* __restrict__ d_editsPerCorrectedCandidate,
        int* __restrict__ d_numEditsPerCorrectedCandidate,
        GPUMultiMSA multiMSA,
        const int* __restrict__ shifts,
        const BestAlignment_t* __restrict__ bestAlignmentFlags,
        const unsigned int* __restrict__ candidateSequencesData,
        const int* __restrict__ candidateSequencesLengths,
        const bool* __restrict__ d_candidateContainsN,
        const int* __restrict__ candidateIndicesOfCandidatesToBeCorrected,
        const int* __restrict__ numCandidatesToBeCorrected,
        const int* __restrict__ anchorIndicesOfCandidates,
        const int* __restrict__ d_numAnchors,
        const int* __restrict__ d_numCandidates,
        int doNotUseEditsValue,
        int numEditsThreshold,            
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        size_t editsPitchInBytes,
        size_t dynamicsmemSequencePitchInInts
    ){

        /*
            Use groupsize threads per candidate to perform correction
        */
        static_assert(BLOCKSIZE % groupsize == 0, "BLOCKSIZE % groupsize != 0");
        constexpr int groupsPerBlock = BLOCKSIZE / groupsize;

        __shared__ int shared_numEditsOfCandidate[groupsPerBlock];

        extern __shared__ int dynamicsmem[]; // for sequences

        auto tgroup = cg::tiled_partition<groupsize>(cg::this_thread_block());

        const int numGroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int groupIdInBlock = threadIdx.x / groupsize;

        const std::size_t smemPitchEditsInInts = SDIV(editsPitchInBytes, sizeof(int));

        char* const shared_correctedCandidate = (char*)(dynamicsmem + dynamicsmemSequencePitchInInts * groupIdInBlock);

        EncodedCorrectionEdit* const shared_Edits 
            = (EncodedCorrectionEdit*)((dynamicsmem + dynamicsmemSequencePitchInInts * groupsPerBlock) 
                + smemPitchEditsInInts * groupIdInBlock);

        const int loopEnd = *numCandidatesToBeCorrected;

        for(int id = groupId; id < loopEnd; id += numGroups){

            const int candidateIndex = candidateIndicesOfCandidatesToBeCorrected[id];
            const int subjectIndex = anchorIndicesOfCandidates[candidateIndex];
            const int destinationIndex = id;

            const GpuSingleMSA msa = multiMSA.getSingleMSA(subjectIndex);

            char* const my_corrected_candidate = correctedCandidates + destinationIndex * decodedSequencePitchInBytes;
            const int candidate_length = candidateSequencesLengths[candidateIndex];

            const int shift = shifts[candidateIndex];
            const int subjectColumnsBegin_incl = msa.columnProperties->subjectColumnsBegin_incl;
            const int queryColumnsBegin_incl = subjectColumnsBegin_incl + shift;
            const int queryColumnsEnd_excl = subjectColumnsBegin_incl + shift + candidate_length;

            const BestAlignment_t bestAlignmentFlag = bestAlignmentFlags[candidateIndex];

            if(tgroup.thread_rank() == 0){                        
                shared_numEditsOfCandidate[groupIdInBlock] = 0;
            }
            tgroup.sync();          

            const int copyposbegin = queryColumnsBegin_incl;
            const int copyposend = queryColumnsEnd_excl;

            //the forward strand will be returned -> make reverse complement again
            if(bestAlignmentFlag == BestAlignment_t::ReverseComplement) {
                for(int i = copyposbegin + tgroup.thread_rank(); i < copyposend; i += tgroup.size()) {
                    shared_correctedCandidate[i - queryColumnsBegin_incl] = SequenceHelpers::decodeBase(SequenceHelpers::complementBase2Bit(msa.consensus[i]));
                }
                tgroup.sync(); // threads may access elements in shared memory which were written by another thread
                SequenceHelpers::reverseAlignedDecodedSequenceWithGroupShfl(tgroup, shared_correctedCandidate, candidate_length);
                tgroup.sync();
            }else{
                for(int i = copyposbegin + tgroup.thread_rank(); i < copyposend; i += tgroup.size()) {
                    shared_correctedCandidate[i - queryColumnsBegin_incl] = SequenceHelpers::decodeBase(msa.consensus[i]);
                }
                tgroup.sync();
            }
            
            //copy corrected sequence from smem to global output
            const int fullInts1 = candidate_length / sizeof(int);

            for(int i = tgroup.thread_rank(); i < fullInts1; i += tgroup.size()) {
                ((int*)my_corrected_candidate)[i] = ((int*)shared_correctedCandidate)[i];
            }

            for(int i = tgroup.thread_rank(); i < candidate_length - fullInts1 * sizeof(int); i += tgroup.size()) {
                my_corrected_candidate[fullInts1 * sizeof(int) + i] 
                    = shared_correctedCandidate[fullInts1 * sizeof(int) + i];
            }       

            //compare corrected candidate with uncorrected candidate, calculate edits   
            
            const unsigned int* const encUncorrectedCandidate = candidateSequencesData 
                        + std::size_t(candidateIndex) * encodedSequencePitchInInts;
            const bool thisSequenceContainsN = d_candidateContainsN[candidateIndex];            

            if(thisSequenceContainsN){
                if(tgroup.thread_rank() == 0){
                    d_numEditsPerCorrectedCandidate[destinationIndex] = doNotUseEditsValue;
                }
            }else{
                const int maxEdits = min(candidate_length / 7, numEditsThreshold);

                auto countAndSaveEditInSmem = [&](const int posInSequence, const char correctedNuc){
                    cg::coalesced_group g = cg::coalesced_threads();
                                
                    int currentNumEdits = 0;
                    if(g.thread_rank() == 0){
                        currentNumEdits = atomicAdd(&shared_numEditsOfCandidate[groupIdInBlock], g.size());
                    }
                    currentNumEdits = g.shfl(currentNumEdits, 0);
    
                    if(currentNumEdits + g.size() <= maxEdits){
                        const int myEditOutputPos = g.thread_rank() + currentNumEdits;
                        if(myEditOutputPos < maxEdits){
                            const auto theEdit = EncodedCorrectionEdit{posInSequence, correctedNuc};
                            //myEdits[myEditOutputPos] = theEdit;
                            //shared_Edits[groupIdInBlock][myEditOutputPos] = theEdit;
                            shared_Edits[myEditOutputPos] = theEdit;
                        }
                    }
                };

                auto countAndSaveEditInSmem2 = [&](const int posInSequence, const char correctedNuc){
                    const int groupsPerWarp = 32 / tgroup.size();
                    if(groupsPerWarp == 1){
                        countAndSaveEditInSmem(posInSequence, correctedNuc);
                    }else{
                        const int groupIdInWarp = (threadIdx.x % 32) / tgroup.size();
                        unsigned int subwarpmask = ((1u << (tgroup.size() - 1)) | ((1u << (tgroup.size() - 1)) - 1));
                        subwarpmask <<= (tgroup.size() * groupIdInWarp);

                        unsigned int lanemask_lt;
                        asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask_lt));
                        const unsigned int writemask = subwarpmask & __activemask();
                        const unsigned int total = __popc(writemask);
                        const unsigned int prefix = __popc(writemask & lanemask_lt);

                        const int elected_lane = __ffs(writemask) - 1;
                        int currentNumEdits = 0;
                        if (prefix == 0) {
                            currentNumEdits = atomicAdd(&shared_numEditsOfCandidate[groupIdInBlock], total);
                        }
                        currentNumEdits = __shfl_sync(writemask, currentNumEdits, elected_lane);

                        if(currentNumEdits + total <= maxEdits){
                            const int myEditOutputPos = prefix + currentNumEdits;
                            if(myEditOutputPos < maxEdits){
                                const auto theEdit = EncodedCorrectionEdit{posInSequence, correctedNuc};
                                //myEdits[myEditOutputPos] = theEdit;
                                //shared_Edits[groupIdInBlock][myEditOutputPos] = theEdit;
                                shared_Edits[myEditOutputPos] = theEdit;
                            }
                        }

                    }
                };

                constexpr int basesPerInt = SequenceHelpers::basesPerInt2Bit();
                const int fullInts = candidate_length / basesPerInt;   
                
                for(int i = 0; i < fullInts; i++){
                    const unsigned int encodedDataInt = encUncorrectedCandidate[i];

                    //compare with basesPerInt bases of corrected sequence

                    for(int k = tgroup.thread_rank(); k < basesPerInt; k += tgroup.size()){
                        const int posInInt = k;
                        const int posInSequence = i * basesPerInt + posInInt;
                        const std::uint8_t encodedUncorrectedNuc = SequenceHelpers::getEncodedNucFromInt2Bit(encodedDataInt, posInInt);
                        const char correctedNuc = shared_correctedCandidate[posInSequence];

                        if(correctedNuc != SequenceHelpers::decodeBase(encodedUncorrectedNuc)){
                            countAndSaveEditInSmem2(posInSequence, correctedNuc);
                        }
                    }

                    tgroup.sync();

                    if(shared_numEditsOfCandidate[groupIdInBlock] > maxEdits){
                        break;
                    }
                }

                //process remaining positions
                if(shared_numEditsOfCandidate[groupIdInBlock] <= maxEdits){
                    const int remainingPositions = candidate_length - basesPerInt * fullInts;

                    if(remainingPositions > 0){
                        const unsigned int encodedDataInt = encUncorrectedCandidate[fullInts];
                        for(int posInInt = tgroup.thread_rank(); posInInt < remainingPositions; posInInt += tgroup.size()){
                            const int posInSequence = fullInts * basesPerInt + posInInt;
                            const std::uint8_t encodedUncorrectedNuc = SequenceHelpers::getEncodedNucFromInt2Bit(encodedDataInt, posInInt);
                            const char correctedNuc = shared_correctedCandidate[posInSequence];

                            if(correctedNuc != SequenceHelpers::decodeBase(encodedUncorrectedNuc)){
                                countAndSaveEditInSmem2(posInSequence, correctedNuc);
                            }
                        }
                    }
                }

                tgroup.sync();

                int* const myNumEdits = d_numEditsPerCorrectedCandidate + destinationIndex;

                EncodedCorrectionEdit* const myEdits 
                    = (EncodedCorrectionEdit*)(((char*)d_editsPerCorrectedCandidate) + destinationIndex * editsPitchInBytes);

                if(shared_numEditsOfCandidate[groupIdInBlock] <= maxEdits){
                    const int numEdits = shared_numEditsOfCandidate[groupIdInBlock];

                    if(tgroup.thread_rank() == 0){ 
                        *myNumEdits = numEdits;
                    }

                    const int fullInts = (numEdits * sizeof(EncodedCorrectionEdit)) / sizeof(int);
                    static_assert(sizeof(EncodedCorrectionEdit) * 2 == sizeof(int), "");

                    for(int i = tgroup.thread_rank(); i < fullInts; i += tgroup.size()) {
                        ((int*)myEdits)[i] = ((int*)shared_Edits)[i];
                    }

                    for(int i = tgroup.thread_rank(); i < numEdits - fullInts * 2; i += tgroup.size()) {
                        myEdits[fullInts * 2 + i] = shared_Edits[fullInts * 2 + i];
                    } 
                }else{
                    if(tgroup.thread_rank() == 0){
                        *myNumEdits = doNotUseEditsValue;
                    }
                }

            }
            

            tgroup.sync(); //sync before handling next candidate
                        
        }
    }


    //if isCompactCorrection == true, compare originalsequence[d_indicesOfCorrectedSequences[i]] with correctedsequence[i] to compute edits
    //if isCompactCorrection == false, compare originalsequence[d_indicesOfCorrectedSequences[i]] with correctedsequence[d_indicesOfCorrectedSequences[i]] to compute edits
    //uses one group per sequence, 1 <= groupsize <= 32, groupsize is power of 2
    template<bool isCompactCorrection, int groupsize>
    __global__
    void constructSequenceCorrectionResultsKernel(
        EncodedCorrectionEdit* __restrict__ d_edits,
        int* __restrict__ d_numEditsPerCorrection,
        int doNotUseEditsValue,
        const int* __restrict__ d_indicesOfUncorrectedSequences,
        const int* __restrict__ d_numIndices,
        const bool* __restrict__ d_readContainsN,
        const unsigned int* __restrict__ d_uncorrectedEncodedSequences,
        const int* __restrict__ d_sequenceLengths,
        const char* __restrict__ d_correctedSequences,
        int numEditsThreshold,
        size_t encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        size_t editsPitchInBytes
    ){
        constexpr int maxblocksize = 256;
        assert(blockDim.x <= maxblocksize);

        __shared__ int sharedNumEdits[maxblocksize / groupsize];

        auto threadblock = cg::this_thread_block();
        auto group = cg::tiled_partition<groupsize>(threadblock);

        const int globalGroupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int numGroups = (blockDim.x * gridDim.x) / groupsize;
        const int localGroupId = threadIdx.x / groupsize;

        const int numIndicesToProcess = *d_numIndices;

        for(int indexOutput = globalGroupId; indexOutput < numIndicesToProcess; indexOutput += numGroups){
            const int indexOfUncorrected = d_indicesOfUncorrectedSequences[indexOutput];
            const int indexOfCorrected = isCompactCorrection ? indexOutput : indexOfUncorrected;

            const bool thisSequenceContainsN = d_readContainsN[indexOfUncorrected];            
            int* const myNumEditsGlobal = d_numEditsPerCorrection + indexOutput;
            int* const myNumEditsShared = sharedNumEdits + localGroupId;

            const unsigned int* const encodedUncorrectedSequence = d_uncorrectedEncodedSequences + encodedSequencePitchInInts * indexOfUncorrected;
            const char* const decodedCorrectedSequence = d_correctedSequences + decodedSequencePitchInBytes * indexOfCorrected;

            EncodedCorrectionEdit* const myEditsGlobal = (EncodedCorrectionEdit*)(((char*)d_edits) + editsPitchInBytes * indexOutput);

            if(thisSequenceContainsN){
                if(group.thread_rank() == 0){
                    *myNumEditsGlobal = doNotUseEditsValue;
                }
            }else{
                const int length = d_sequenceLengths[indexOfUncorrected];

                if(group.thread_rank() == 0){
                    *myNumEditsShared = 0;
                }
                group.sync();

                const int maxEdits = min(length / 7, numEditsThreshold);

                auto countAndSaveEditsWarp = [&](const int posInSequence, const char correctedNuc){
                    cg::coalesced_group g = cg::coalesced_threads();
                                
                    int currentNumEdits = 0;
                    if(g.thread_rank() == 0){
                        currentNumEdits = atomicAdd(myNumEditsShared, g.size());
                    }
                    currentNumEdits = g.shfl(currentNumEdits, 0);
    
                    if(currentNumEdits + g.size() <= maxEdits){
                        const int myEditOutputPos = g.thread_rank() + currentNumEdits;
                        if(myEditOutputPos < maxEdits){
                            const auto theEdit = EncodedCorrectionEdit{posInSequence, correctedNuc};
                            myEditsGlobal[myEditOutputPos] = theEdit;
                        }
                    }
                };

                auto countAndSaveEditsGroup = [&](const int posInSequence, const char correctedNuc){
                    const int groupsPerWarp = 32 / group.size();
                    if(groupsPerWarp == 1){
                        countAndSaveEditsWarp(posInSequence, correctedNuc);
                    }else{
                        const int groupIdInWarp = (threadIdx.x % 32) / group.size();
                        unsigned int subwarpmask = ((1u << (group.size() - 1)) | ((1u << (group.size() - 1)) - 1));
                        subwarpmask <<= (group.size() * groupIdInWarp);

                        unsigned int lanemask_lt;
                        asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask_lt));
                        const unsigned int writemask = subwarpmask & __activemask();
                        const unsigned int total = __popc(writemask);
                        const unsigned int prefix = __popc(writemask & lanemask_lt);

                        const int elected_lane = __ffs(writemask) - 1;
                        int currentNumEdits = 0;
                        if (prefix == 0) {
                            currentNumEdits = atomicAdd(myNumEditsShared, total);
                        }
                        currentNumEdits = __shfl_sync(writemask, currentNumEdits, elected_lane);

                        if(currentNumEdits + total <= maxEdits){
                            const int myEditOutputPos = prefix + currentNumEdits;
                            if(myEditOutputPos < maxEdits){
                                const auto theEdit = EncodedCorrectionEdit{posInSequence, correctedNuc};
                                myEditsGlobal[myEditOutputPos] = theEdit;
                            }
                        }

                    }
                };

                constexpr int basesPerInt = SequenceHelpers::basesPerInt2Bit();
                const int fullInts = length / basesPerInt;   
                
                for(int i = 0; i < fullInts; i++){
                    const unsigned int encodedDataInt = encodedUncorrectedSequence[i];

                    //compare with basesPerInt bases of corrected sequence
                    for(int k = group.thread_rank(); k < basesPerInt; k += group.size()){
                        const int posInInt = k;
                        const int posInSequence = i * basesPerInt + posInInt;
                        const std::uint8_t encodedUncorrectedNuc = SequenceHelpers::getEncodedNucFromInt2Bit(encodedDataInt, posInInt);
                        const char correctedNuc = decodedCorrectedSequence[posInSequence];

                        if(correctedNuc != SequenceHelpers::decodeBase(encodedUncorrectedNuc)){
                            countAndSaveEditsGroup(posInSequence, correctedNuc);
                        }
                    }

                    group.sync();

                    if(*myNumEditsShared > maxEdits){
                        break;
                    }
                }

                //process remaining positions
                if(*myNumEditsShared <= maxEdits){
                    const int remainingPositions = length - basesPerInt * fullInts;

                    if(remainingPositions > 0){
                        const unsigned int encodedDataInt = encodedUncorrectedSequence[fullInts];
                        for(int posInInt = group.thread_rank(); posInInt < remainingPositions; posInInt += group.size()){
                            const int posInSequence = fullInts * basesPerInt + posInInt;
                            const std::uint8_t encodedUncorrectedNuc = SequenceHelpers::getEncodedNucFromInt2Bit(encodedDataInt, posInInt);
                            const char correctedNuc = decodedCorrectedSequence[posInSequence];

                            if(correctedNuc != SequenceHelpers::decodeBase(encodedUncorrectedNuc)){
                                countAndSaveEditsGroup(posInSequence, correctedNuc);
                            }
                        }
                    }
                }

                group.sync();

                if(*myNumEditsShared <= maxEdits){
                    if(group.thread_rank() == 0){ 
                        *myNumEditsGlobal = *myNumEditsShared;
                    }

                    // const int fullInts = (numEdits * sizeof(EncodedCorrectionEdit)) / sizeof(int);
                    // static_assert(sizeof(EncodedCorrectionEdit) * 2 == sizeof(int), "");

                    // for(int i = tgroup.thread_rank(); i < fullInts; i += tgroup.size()) {
                    //     ((int*)myEdits)[i] = ((int*)shared_Edits)[i];
                    // }

                    // for(int i = tgroup.thread_rank(); i < numEdits - fullInts * 2; i += tgroup.size()) {
                    //     myEdits[fullInts * 2 + i] = shared_Edits[fullInts * 2 + i];
                    // } 
                }else{
                    if(group.thread_rank() == 0){
                        *myNumEditsGlobal = doNotUseEditsValue;
                    }
                }
            }

            group.sync(); //sync before handling next sequence
                    
        }
    }




    //####################   KERNEL DISPATCH   ####################


    void call_msaCorrectAnchorsKernel_async(
        char* d_correctedSubjects,
        bool* d_subjectIsCorrected,
        AnchorHighQualityFlag* d_isHighQualitySubject,
        GPUMultiMSA multiMSA,
        const unsigned int* d_subjectSequencesData,
        const int* d_indices_per_subject,
        const int* d_numAnchors,
        int /*maxNumAnchors*/,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        float estimatedErrorrate,
        float avg_support_threshold,
        float min_support_threshold,
        float min_coverage_threshold,
        int maximum_sequence_length,
        cudaStream_t stream,
        KernelLaunchHandle& handle
    ){

        const int max_block_size = 256;
        const int blocksize = std::min(max_block_size, SDIV(maximum_sequence_length, 32) * 32);
        const std::size_t smem = 0;

        int max_blocks_per_device = 1;

        KernelLaunchConfig kernelLaunchConfig;
        kernelLaunchConfig.threads_per_block = blocksize;
        kernelLaunchConfig.smem = smem;

        auto iter = handle.kernelPropertiesMap.find(KernelId::MSACorrectSubjectImplicit);
        if(iter == handle.kernelPropertiesMap.end()){

            std::map<KernelLaunchConfig, KernelProperties> mymap;

            #define getProp(blocksize) { \
                KernelLaunchConfig kernelLaunchConfig; \
                kernelLaunchConfig.threads_per_block = (blocksize); \
                kernelLaunchConfig.smem = 0; \
                KernelProperties kernelProperties; \
                CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                    msaCorrectAnchorsKernel<(blocksize)>, \
                    kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem)); \
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

            handle.kernelPropertiesMap[KernelId::MSACorrectSubjectImplicit] = std::move(mymap);

            #undef getProp
        }else{
            std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
            const KernelProperties& kernelProperties = map[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
        }

        dim3 block(blocksize, 1, 1);
        //dim3 grid(std::min(maxNumAnchors, max_blocks_per_device));
        dim3 grid(max_blocks_per_device);

        #define mycall(blocksize) msaCorrectAnchorsKernel<(blocksize)> \
                                <<<grid, block, 0, stream>>>( \
                                    d_correctedSubjects, \
                                    d_subjectIsCorrected, \
                                    d_isHighQualitySubject, \
                                    multiMSA, \
                                    d_subjectSequencesData, \
                                    d_indices_per_subject, \
                                    d_numAnchors, \
                                    encodedSequencePitchInInts, \
                                    decodedSequencePitchInBytes, \
                                    estimatedErrorrate, \
                                    avg_support_threshold, \
                                    min_support_threshold, \
                                    min_coverage_threshold \
                                ); CUDACHECKASYNC;

        assert(blocksize > 0 && blocksize <= max_block_size);

        switch(blocksize){
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

    void callMsaCorrectAnchorsWithForestKernel(
        char* d_correctedSubjects,
        bool* d_subjectIsCorrected,
        AnchorHighQualityFlag* d_isHighQualitySubject,
        GPUMultiMSA multiMSA,
        GpuForest::Clf gpuForest,
        float forestThreshold,
        const unsigned int* d_subjectSequencesData,
        const int* d_indices_per_subject,
        const int numAnchors,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        int maximumSequenceLength,
        float estimatedErrorrate,
        float estimatedCoverage,
        float avg_support_threshold,
        float min_support_threshold,
        float min_coverage_threshold,
        cudaStream_t stream,
        KernelLaunchHandle& /*handle*/
    ){
        if(numAnchors == 0) return;

        constexpr int blocksize = 128;
        const int numBlocks = numAnchors;

        const std::size_t smem = SDIV(sizeof(char) * maximumSequenceLength, sizeof(int)) * sizeof(int);

        msaCorrectAnchorsWithForestKernel<blocksize, anchor_extractor><<<numBlocks, blocksize, smem, stream>>>(
            d_correctedSubjects,
            d_subjectIsCorrected,
            d_isHighQualitySubject,
            multiMSA,
            gpuForest,
            forestThreshold,
            d_subjectSequencesData,
            d_indices_per_subject,
            numAnchors,
            encodedSequencePitchInInts,
            decodedSequencePitchInBytes,
            estimatedErrorrate,
            estimatedCoverage,
            avg_support_threshold,
            min_support_threshold,
            min_coverage_threshold
        );
    }

    void callMsaCorrectCandidatesWithForestKernel(
        char* d_correctedCandidates,
        EncodedCorrectionEdit* d_editsPerCorrectedCandidate,
        int* d_numEditsPerCorrectedCandidate,
        GPUMultiMSA multiMSA,
        GpuForest::Clf gpuForest,
        float forestThreshold,
        float estimatedCoverage,
        const int* d_shifts,
        const BestAlignment_t* d_bestAlignmentFlags,
        const unsigned int* d_candidateSequencesData,
        const int* d_candidateSequencesLengths,
        const bool* d_candidateContainsN,
        const int* d_candidateIndicesOfCandidatesToBeCorrected,
        const int* d_numCandidatesToBeCorrected,
        const int* d_anchorIndicesOfCandidates,
        const int /*numCandidates*/,
        int doNotUseEditsValue,
        int numEditsThreshold,            
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        size_t editsPitchInBytes,
        int maximum_sequence_length,
        cudaStream_t stream,
        KernelLaunchHandle& /*handle*/,
        const read_number* candidateReadIds
    ){

        constexpr int blocksize = 128;
        constexpr int groupsize = 32;

        const std::size_t dynamicsmemPitchInInts = SDIV(maximum_sequence_length, sizeof(int));
        const std::size_t treePointersPitchInInts = SDIV(sizeof(void*) * gpuForest.numTrees, sizeof(int));

        auto calculateSmemUsage = [&](int blockDim){
            const int numGroupsPerBlock = blockDim / groupsize;
            std::size_t smem = numGroupsPerBlock * (sizeof(int) * dynamicsmemPitchInInts)
                + treePointersPitchInInts * sizeof(int); 

            return smem;
        };

        const std::size_t smem = calculateSmemUsage(blocksize);

        dim3 block = blocksize;
        dim3 grid = 480;

        msaCorrectCandidatesWithForestKernel<blocksize, groupsize, cands_extractor><<<grid, block, smem, stream>>>(
            d_correctedCandidates,
            d_editsPerCorrectedCandidate,
            d_numEditsPerCorrectedCandidate,
            multiMSA,
            gpuForest,
            forestThreshold,
            estimatedCoverage,
            d_shifts,
            d_bestAlignmentFlags,
            d_candidateSequencesData,
            d_candidateSequencesLengths,
            d_candidateContainsN,
            d_candidateIndicesOfCandidatesToBeCorrected,
            d_numCandidatesToBeCorrected,
            d_anchorIndicesOfCandidates,            
            doNotUseEditsValue,
            numEditsThreshold,            
            encodedSequencePitchInInts,
            decodedSequencePitchInBytes,
            editsPitchInBytes,
            dynamicsmemPitchInInts,
            candidateReadIds
        );
    }

    void callFlagCandidatesToBeCorrectedKernel_async(
        bool* d_candidateCanBeCorrected,
        int* d_numCorrectedCandidatesPerAnchor,
        GPUMultiMSA multiMSA,
        const int* d_alignmentShifts,
        const int* d_candidateSequencesLengths,
        const int* d_anchorIndicesOfCandidates,
        const AnchorHighQualityFlag* d_hqflags,
        const int* d_candidatesPerSubjectPrefixsum,
        const int* d_localGoodCandidateIndices,
        const int* d_numLocalGoodCandidateIndicesPerSubject,
        const int* d_numAnchors,
        const int* d_numCandidates,
        float min_support_threshold,
        float min_coverage_threshold,
        int new_columns_to_correct,
        cudaStream_t stream,
        KernelLaunchHandle& handle
    ){

        constexpr int blocksize = 256;
        const std::size_t smem = 0;

        int max_blocks_per_device = 1;

        KernelLaunchConfig kernelLaunchConfig;
        kernelLaunchConfig.threads_per_block = blocksize;
        kernelLaunchConfig.smem = smem;

        auto iter = handle.kernelPropertiesMap.find(KernelId::FlagCandidatesToBeCorrected);
        if(iter == handle.kernelPropertiesMap.end()){

            std::map<KernelLaunchConfig, KernelProperties> mymap;

            KernelProperties kernelProperties;

            CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &kernelProperties.max_blocks_per_SM,
                flagCandidatesToBeCorrectedKernel,
                kernelLaunchConfig.threads_per_block, 
                kernelLaunchConfig.smem
            ));

            mymap[kernelLaunchConfig] = kernelProperties;

            max_blocks_per_device = handle.deviceProperties.multiProcessorCount 
                                        * kernelProperties.max_blocks_per_SM;

            handle.kernelPropertiesMap[KernelId::FlagCandidatesToBeCorrected] = std::move(mymap);

            #undef getProp
        }else{
            std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
            const KernelProperties& kernelProperties = map[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount 
                                        * kernelProperties.max_blocks_per_SM;
        }



        dim3 block(blocksize);
        dim3 grid(max_blocks_per_device);

        flagCandidatesToBeCorrectedKernel<<<grid, block, 0, stream>>>(
            d_candidateCanBeCorrected,
            d_numCorrectedCandidatesPerAnchor,
            multiMSA,
            d_alignmentShifts,
            d_candidateSequencesLengths,
            d_anchorIndicesOfCandidates,
            d_hqflags,
            d_candidatesPerSubjectPrefixsum,
            d_localGoodCandidateIndices,
            d_numLocalGoodCandidateIndicesPerSubject,
            d_numAnchors,
            d_numCandidates,
            min_support_threshold,
            min_coverage_threshold,
            new_columns_to_correct
        );

        CUDACHECKASYNC;

    }


    void callFlagCandidatesToBeCorrectedWithExcludeFlagsKernel(
        bool* d_candidateCanBeCorrected,
        int* d_numCorrectedCandidatesPerAnchor,
        GPUMultiMSA multiMSA,
        const bool* d_excludeFlags, //candidates with flag == true will not be considered
        const int* d_alignmentShifts,
        const int* d_candidateSequencesLengths,
        const int* d_anchorIndicesOfCandidates,
        const AnchorHighQualityFlag* d_hqflags,
        const int* d_candidatesPerSubjectPrefixsum,
        const int* d_localGoodCandidateIndices,
        const int* d_numLocalGoodCandidateIndicesPerSubject,
        const int* d_numAnchors,
        const int* d_numCandidates,
        float min_support_threshold,
        float min_coverage_threshold,
        int new_columns_to_correct,
        cudaStream_t stream,
        KernelLaunchHandle& handle
    ){

        constexpr int blocksize = 256;
        const std::size_t smem = 0;

        int max_blocks_per_device = 1;

        KernelLaunchConfig kernelLaunchConfig;
        kernelLaunchConfig.threads_per_block = blocksize;
        kernelLaunchConfig.smem = smem;

        const auto kernelId = KernelId::FlagCandidatesToBeCorrectedWithExcludeFlags;

        auto iter = handle.kernelPropertiesMap.find(kernelId);
        if(iter == handle.kernelPropertiesMap.end()){

            std::map<KernelLaunchConfig, KernelProperties> mymap;

            KernelProperties kernelProperties;

            CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &kernelProperties.max_blocks_per_SM,
                flagCandidatesToBeCorrectedWithExcludeFlagsKernel,
                kernelLaunchConfig.threads_per_block, 
                kernelLaunchConfig.smem
            ));

            mymap[kernelLaunchConfig] = kernelProperties;

            max_blocks_per_device = handle.deviceProperties.multiProcessorCount 
                                        * kernelProperties.max_blocks_per_SM;

            handle.kernelPropertiesMap[kernelId] = std::move(mymap);

            #undef getProp
        }else{
            std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
            const KernelProperties& kernelProperties = map[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount 
                                        * kernelProperties.max_blocks_per_SM;
        }



        dim3 block(blocksize);
        dim3 grid(max_blocks_per_device);

        flagCandidatesToBeCorrectedWithExcludeFlagsKernel<<<grid, block, 0, stream>>>(
            d_candidateCanBeCorrected,
            d_numCorrectedCandidatesPerAnchor,
            multiMSA,
            d_excludeFlags,
            d_alignmentShifts,
            d_candidateSequencesLengths,
            d_anchorIndicesOfCandidates,
            d_hqflags,
            d_candidatesPerSubjectPrefixsum,
            d_localGoodCandidateIndices,
            d_numLocalGoodCandidateIndicesPerSubject,
            d_numAnchors,
            d_numCandidates,
            min_support_threshold,
            min_coverage_threshold,
            new_columns_to_correct
        );

        CUDACHECKASYNC;

    }



    void callCorrectCandidatesKernel_async(
        char* __restrict__ correctedCandidates,
        EncodedCorrectionEdit* __restrict__ d_editsPerCorrectedCandidate,
        int* __restrict__ d_numEditsPerCorrectedCandidate,
        GPUMultiMSA multiMSA,
        const int* __restrict__ shifts,
        const BestAlignment_t* __restrict__ bestAlignmentFlags,
        const unsigned int* __restrict__ candidateSequencesData,
        const int* __restrict__ candidateSequencesLengths,
        const bool* __restrict__ d_candidateContainsN,
        const int* __restrict__ candidateIndicesOfCandidatesToBeCorrected,
        const int* __restrict__ numCandidatesToBeCorrected,
        const int* __restrict__ anchorIndicesOfCandidates,
        const int* d_numAnchors,
        const int* d_numCandidates,
        int doNotUseEditsValue,
        int numEditsThreshold,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        size_t editsPitchInBytes,
        int maximum_sequence_length,
        cudaStream_t stream,
        KernelLaunchHandle& handle
    ){

        constexpr int blocksize = 128;
        constexpr int groupsize = 32;

        const size_t dynamicsmemPitchInInts = SDIV(maximum_sequence_length, sizeof(int));
        const size_t smemPitchEditsInInts = SDIV(editsPitchInBytes, sizeof(int));

        auto calculateSmemUsage = [&](int blockDim){
            const int numGroupsPerBlock = blockDim / groupsize;
            std::size_t smem = numGroupsPerBlock * (sizeof(int) * dynamicsmemPitchInInts)
                + numGroupsPerBlock * (sizeof(int) * smemPitchEditsInInts);

            return smem;
        };

        const std::size_t smem = calculateSmemUsage(blocksize);

    	int max_blocks_per_device = 1;

    	KernelLaunchConfig kernelLaunchConfig;
    	kernelLaunchConfig.threads_per_block = blocksize;
    	kernelLaunchConfig.smem = smem;

    	auto iter = handle.kernelPropertiesMap.find(KernelId::MSACorrectCandidates);
    	if(iter == handle.kernelPropertiesMap.end()) {

    		std::map<KernelLaunchConfig, KernelProperties> mymap;

    	    #define getProp(blocksize) { \
                KernelLaunchConfig kernelLaunchConfig; \
                kernelLaunchConfig.threads_per_block = (blocksize); \
                kernelLaunchConfig.smem = calculateSmemUsage((blocksize)); \
                KernelProperties kernelProperties; \
                CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                    msaCorrectCandidatesKernel<(blocksize), groupsize>, \
                            kernelLaunchConfig.threads_per_block, kernelLaunchConfig.smem)); \
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
            
            // std::cerr << "msaCorrectCandidatesKernel "
            //     << "multiProcessorCount = " << handle.deviceProperties.multiProcessorCount
            //     << " max_blocks_per_SM = " << kernelProperties.max_blocks_per_SM << "\n"; 

    		handle.kernelPropertiesMap[KernelId::MSACorrectCandidates] = std::move(mymap);

    	    #undef getProp
    	}else{
    		std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
    		const KernelProperties& kernelProperties = map[kernelLaunchConfig];
    		max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
    	}

    	dim3 block(blocksize, 1, 1);
        //dim3 grid(std::min(max_blocks_per_device, n_candidates * numGroupsPerBlock));
        dim3 grid(max_blocks_per_device);
        
        assert(smem % sizeof(int) == 0);

    	#define mycall(blocksize) msaCorrectCandidatesKernel<(blocksize), groupsize> \
    	        <<<grid, block, smem, stream>>>( \
                    correctedCandidates, \
                    d_editsPerCorrectedCandidate, \
                    d_numEditsPerCorrectedCandidate, \
                    multiMSA, \
                    shifts, \
                    bestAlignmentFlags, \
                    candidateSequencesData, \
                    candidateSequencesLengths, \
                    d_candidateContainsN, \
                    candidateIndicesOfCandidatesToBeCorrected, \
                    numCandidatesToBeCorrected, \
                    anchorIndicesOfCandidates, \
                    d_numAnchors, \
                    d_numCandidates, \
                    doNotUseEditsValue, \
                    numEditsThreshold, \
                    encodedSequencePitchInInts, \
                    decodedSequencePitchInBytes, \
                    editsPitchInBytes, \
                    dynamicsmemPitchInInts \
                ); CUDACHECKASYNC;


    	switch(blocksize) {
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


    void callConstructSequenceCorrectionResultsKernel(
        EncodedCorrectionEdit* d_edits,
        int* d_numEditsPerCorrection,
        int doNotUseEditsValue,
        const int* d_indicesOfUncorrectedSequences,
        const int* d_numIndices,
        const bool* d_readContainsN,
        const unsigned int* d_uncorrectedEncodedSequences,
        const int* d_sequenceLengths,
        const char* d_correctedSequences,
        const int numCorrectedSequencesUpperBound, // >= *d_numIndices. d_edits must be large enought to store the edits of this many sequences
        bool isCompactCorrection,
        int numEditsThreshold,
        size_t encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        size_t editsPitchInBytes,        
        cudaStream_t stream,
        KernelLaunchHandle& /*handle*/
    ){
        if(numCorrectedSequencesUpperBound == 0) return;

        CUDACHECK(cudaMemsetAsync(
            d_edits, 
            0, 
            editsPitchInBytes * numCorrectedSequencesUpperBound, 
            stream
        ));

        constexpr int groupsize = 16;

        const int blocksize = 128;
        const std::size_t smem = 0;

        dim3 block = blocksize;
        dim3 grid = SDIV(numCorrectedSequencesUpperBound, blocksize / groupsize);

        if(isCompactCorrection){
            constructSequenceCorrectionResultsKernel<true, groupsize><<<grid, block, smem, stream>>>(
                d_edits,
                d_numEditsPerCorrection,
                doNotUseEditsValue,
                d_indicesOfUncorrectedSequences,
                d_numIndices,
                d_readContainsN,
                d_uncorrectedEncodedSequences,
                d_sequenceLengths,
                d_correctedSequences,
                numEditsThreshold,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                editsPitchInBytes
            ); CUDACHECKASYNC;
        }else{
            constructSequenceCorrectionResultsKernel<false, groupsize><<<grid, block, smem, stream>>>(
                d_edits,
                d_numEditsPerCorrection,
                doNotUseEditsValue,
                d_indicesOfUncorrectedSequences,
                d_numIndices,
                d_readContainsN,
                d_uncorrectedEncodedSequences,
                d_sequenceLengths,
                d_correctedSequences,
                numEditsThreshold,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                editsPitchInBytes
            ); CUDACHECKASYNC;
        }
    }







}
}
