//#define NDEBUG

#include <gpu/kernels.hpp>
#include <hostdevicefunctions.cuh>
#include <gpu/gpumsa.cuh>
#include <gpu/cudaerrorcheck.cuh>
#include <alignmentorientation.hpp>
#include <util_iterator.hpp>
#include <sequencehelpers.hpp>
#include <correctedsequence.hpp>

#include <hpc_helpers.cuh>
#include <config.hpp>
#include <cassert>

#include <gpu/forest_gpu.cuh>
#include <gpu/classification_gpu.cuh>

#include <cub/cub.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <thrust/functional.h>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>


namespace cg = cooperative_groups;

namespace care{
namespace gpu{


    __global__
    void msaCorrectAnchorsKernel(
        char* __restrict__ correctedAnchors,
        bool* __restrict__ anchorIsCorrected,
        AnchorHighQualityFlag* __restrict__ isHighQualityAnchor,
        GPUMultiMSA multiMSA,
        const unsigned int* __restrict__ anchorSequencesData,
        const int* __restrict__ d_indices_per_anchor,
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

        const int n_anchors = *numAnchorsPtr;

        for(unsigned anchorIndex = blockIdx.x; anchorIndex < n_anchors; anchorIndex += gridDim.x){
            const int myNumIndices = d_indices_per_anchor[anchorIndex];
            if(myNumIndices > 0){

                const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);                

                const int anchorColumnsBegin_incl = msa.columnProperties->anchorColumnsBegin_incl;
                const int anchorColumnsEnd_excl = msa.columnProperties->anchorColumnsEnd_excl;

                auto i_f = thrust::identity<float>{};
                auto i_i = thrust::identity<int>{};

                GpuMSAProperties msaProperties = msa.getMSAProperties(
                    thread, i_f, i_f, i_i, i_i,
                    anchorColumnsBegin_incl,
                    anchorColumnsEnd_excl
                );

                AnchorCorrectionQuality correctionQuality(avg_support_threshold, min_support_threshold, min_coverage_threshold, estimatedErrorrate);

                const bool canBeCorrectedBySimpleConsensus = correctionQuality.canBeCorrectedBySimpleConsensus(msaProperties.avg_support, msaProperties.min_support, msaProperties.min_coverage);
                const bool isHQCorrection = correctionQuality.isHQCorrection(msaProperties.avg_support, msaProperties.min_support, msaProperties.min_coverage);

                if(tbGroup.thread_rank() == 0){
                    anchorIsCorrected[anchorIndex] = true;
                    isHighQualityAnchor[anchorIndex].hq(isHQCorrection);
                }

                char* const my_corrected_anchor = correctedAnchors + anchorIndex * decodedSequencePitchInBytes;

                if(canBeCorrectedBySimpleConsensus){
                    for(int i = anchorColumnsBegin_incl + tbGroup.thread_rank(); 
                            i < anchorColumnsEnd_excl; 
                            i += tbGroup.size()){

                        const std::uint8_t nuc = msa.consensus[i];
                        assert(nuc < 4);

                        my_corrected_anchor[i - anchorColumnsBegin_incl] = SequenceHelpers::decodeBase(nuc);
                    }
                }else{
                    //correct only positions with high support.
                    for(int i = anchorColumnsBegin_incl + tbGroup.thread_rank(); 
                            i < anchorColumnsEnd_excl; 
                            i += tbGroup.size()){

                        
                        if(msa.support[i] > 0.90f && msa.origCoverages[i] <= 2){
                            my_corrected_anchor[i - anchorColumnsBegin_incl] = SequenceHelpers::decodeBase(msa.consensus[i]);
                        }else{
                            const unsigned int* const anchor = anchorSequencesData + std::size_t(anchorIndex) * encodedSequencePitchInInts;
                            const std::uint8_t encodedBase = SequenceHelpers::getEncodedNuc2Bit(anchor, anchorColumnsEnd_excl- anchorColumnsBegin_incl, i - anchorColumnsBegin_incl);
                            const char base = SequenceHelpers::decodeBase(encodedBase);
                            assert(base == 'A' || base == 'C' || base == 'G' || base == 'T');
                            my_corrected_anchor[i - anchorColumnsBegin_incl] = base;
                        }
                    }
                }
            }else{
                if(tbGroup.thread_rank() == 0){
                    isHighQualityAnchor[anchorIndex].hq(false);
                    anchorIsCorrected[anchorIndex] = false;
                }
            }
        }
    }


    template<int BLOCKSIZE, class AnchorExtractor, class GpuClf>
    __global__
    void msaCorrectAnchorsWithForestKernel(
        char* __restrict__ correctedAnchors,
        bool* __restrict__ anchorIsCorrected,
        AnchorHighQualityFlag* __restrict__ isHighQualityAnchor,
        GPUMultiMSA multiMSA,
        GpuClf gpuForest,
        float forestThreshold,
        const unsigned int* __restrict__ anchorSequencesData,
        const int* __restrict__ d_indices_per_anchor,
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

        for(unsigned anchorIndex = blockIdx.x; anchorIndex < numAnchors; anchorIndex += gridDim.x){
            const int myNumIndices = d_indices_per_anchor[anchorIndex];
            if(myNumIndices > 0){

                const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);                

                const int anchorColumnsBegin_incl = msa.columnProperties->anchorColumnsBegin_incl;
                const int anchorColumnsEnd_excl = msa.columnProperties->anchorColumnsEnd_excl;

                auto i_f = thrust::identity<float>{};
                auto i_i = thrust::identity<int>{};

                GpuMSAProperties msaProperties = msa.getMSAProperties(
                    thread, i_f, i_f, i_i, i_i,
                    anchorColumnsBegin_incl,
                    anchorColumnsEnd_excl
                );

                AnchorCorrectionQuality correctionQuality(avg_support_threshold, min_support_threshold, min_coverage_threshold, estimatedErrorrate);

                const bool canBeCorrectedBySimpleConsensus = correctionQuality.canBeCorrectedBySimpleConsensus(msaProperties.avg_support, msaProperties.min_support, msaProperties.min_coverage);
                const bool isHQCorrection = correctionQuality.isHQCorrection(msaProperties.avg_support, msaProperties.min_support, msaProperties.min_coverage);

                if(tbGroup.thread_rank() == 0){
                    anchorIsCorrected[anchorIndex] = true;
                    isHighQualityAnchor[anchorIndex].hq(isHQCorrection);
                }

                const int anchorLength = anchorColumnsEnd_excl - anchorColumnsBegin_incl;
                const unsigned int* const anchor = anchorSequencesData + std::size_t(anchorIndex) * encodedSequencePitchInInts;
                char* const globalCorrectedAnchor = correctedAnchors + anchorIndex * decodedSequencePitchInBytes;

                if(isHQCorrection){

                    //set corrected anchor to consensus
                    for(int i = tbGroup.thread_rank(); i < anchorLength; i += tbGroup.size()){
                        const std::uint8_t nuc = msa.consensus[anchorColumnsBegin_incl + i];
                        assert(nuc < 4);
                        globalCorrectedAnchor[i] = SequenceHelpers::decodeBase(nuc);
                    }

                }else{

                    //set corrected anchor to consensus
                    for(int i = tbGroup.thread_rank(); i < anchorLength; i += tbGroup.size()){
                        const std::uint8_t nuc = msa.consensus[anchorColumnsBegin_incl + i];
                        assert(nuc < 4);
                        sharedCorrectedAnchor[i] = SequenceHelpers::decodeBase(nuc);
                    }
                    
                    tbGroup.sync();                                   
                    
                    //maybe revert some positions to original base
                    for (int i = subgroupIdInBlock; i < anchorLength; i += numSubGroupsInBlock){
                        const int msaPos = anchorColumnsBegin_incl + i;
                        const std::uint8_t origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(anchor, anchorLength, i);
                        const std::uint8_t consensusEncodedBase = msa.consensus[msaPos];

                        if (origEncodedBase != consensusEncodedBase){                            
                            
                            if(subgroup.thread_rank() == 0){

                                // if(anchorIndex == 1){
                                //     printf("anchorIndex %d, position %d\n", anchorIndex, i);
                                // }

                                ExtractAnchorInputData& extractorInput = sharedExtractInput[subgroupIdInBlock];

                                extractorInput.origBase = SequenceHelpers::decodeBase(origEncodedBase);
                                extractorInput.consensusBase = SequenceHelpers::decodeBase(consensusEncodedBase);
                                extractorInput.estimatedCoverage = estimatedCoverage;
                                extractorInput.msaPos = msaPos;
                                extractorInput.anchorColumnsBegin_incl = anchorColumnsBegin_incl;
                                extractorInput.anchorColumnsEnd_excl = anchorColumnsEnd_excl;
                                extractorInput.msaProperties = msaProperties;
                                extractorInput.msa = msa;

                                AnchorExtractor extractFeatures{};
                                extractFeatures(&sharedFeatures[subgroupIdInBlock][0], extractorInput);
                            }

                            subgroup.sync();

                            //only thread 0 of group has valid result
                            const bool useConsensus = gpuForest.decide(subgroup, &sharedFeatures[subgroupIdInBlock][0], forestThreshold, subgroupReduceFloatSum);
                            if(subgroup.thread_rank() == 0){
                                // if(anchorIndex == 1){
                                //     if(i == 35){
                                //         for(int x = 0; x < AnchorExtractor::numFeatures(); x++){
                                //             printf("%f, ", sharedFeatures[subgroupIdInBlock][x]);
                                //         }
                                //         printf("\n");
                                //     }
                                //     printf("old anchorIndex %d, position %d, %c %c, useConsensus %d\n", anchorIndex, i, SequenceHelpers::decodeBase(origEncodedBase), SequenceHelpers::decodeBase(consensusEncodedBase), useConsensus);
                                // }
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
                    isHighQualityAnchor[anchorIndex].hq(false);
                    anchorIsCorrected[anchorIndex] = false;
                }
            }            
        }
    }

    template<int BLOCKSIZE, int groupsize, class AnchorExtractor, class GpuClf>
    __global__
    void msaCorrectAnchorsWithForestKernel2(
        char* __restrict__ correctedAnchors,
        bool* __restrict__ anchorIsCorrected,
        AnchorHighQualityFlag* __restrict__ isHighQualityAnchor,
        GPUMultiMSA multiMSA,
        GpuClf gpuForest,
        float forestThreshold,
        const unsigned int* __restrict__ anchorSequencesData,
        const int* __restrict__ d_indices_per_anchor,
        const int numAnchors,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        float estimatedErrorrate,
        float estimatedCoverage,
        float avg_support_threshold,
        float min_support_threshold,
        float min_coverage_threshold
    ){

        static_assert(BLOCKSIZE % groupsize == 0, "BLOCKSIZE % groupsize != 0");
        constexpr int groupsPerBlock = BLOCKSIZE / groupsize;
        static_assert(groupsize == 32);

        auto thread = cg::this_thread();
        auto tgroup = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int numGroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int groupIdInBlock = threadIdx.x / groupsize;

        auto minreduce = [&](auto val){
            using T = decltype(val);
            return cg::reduce(tgroup, val, cg::less<T>{});
        };

        auto maxreduce = [&](auto val){
            using T = decltype(val);
            return cg::reduce(tgroup, val, cg::greater<T>{});
        };

        auto sumreduce = [&](auto val){
            using T = decltype(val);
            return cg::reduce(tgroup, val, cg::plus<T>{});
        };
      
        __shared__ float sharedFeatures[BLOCKSIZE * AnchorExtractor::numFeatures()];

        float* myFeaturesTransposed = &sharedFeatures[threadIdx.x];

        extern __shared__ int externalsmem[];
        
        char* const sharedCorrectedAnchor = ((char*)&externalsmem[0]) + groupIdInBlock * decodedSequencePitchInBytes;

        for(unsigned anchorIndex = groupId; anchorIndex < numAnchors; anchorIndex += numGroups){
            const int myNumIndices = d_indices_per_anchor[anchorIndex];
            if(myNumIndices > 0){

                const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);                

                const int anchorColumnsBegin_incl = msa.columnProperties->anchorColumnsBegin_incl;
                const int anchorColumnsEnd_excl = msa.columnProperties->anchorColumnsEnd_excl;

                GpuMSAProperties msaProperties = msa.getMSAProperties(
                    tgroup, 
                    sumreduce,
                    minreduce, 
                    minreduce, 
                    maxreduce,
                    anchorColumnsBegin_incl,
                    anchorColumnsEnd_excl
                );

                AnchorCorrectionQuality correctionQuality(avg_support_threshold, min_support_threshold, min_coverage_threshold, estimatedErrorrate);

                const bool canBeCorrectedBySimpleConsensus = correctionQuality.canBeCorrectedBySimpleConsensus(msaProperties.avg_support, msaProperties.min_support, msaProperties.min_coverage);
                const bool isHQCorrection = correctionQuality.isHQCorrection(msaProperties.avg_support, msaProperties.min_support, msaProperties.min_coverage);

                if(tgroup.thread_rank() == 0){
                    anchorIsCorrected[anchorIndex] = true;
                    isHighQualityAnchor[anchorIndex].hq(isHQCorrection);
                }

                const int anchorLength = anchorColumnsEnd_excl - anchorColumnsBegin_incl;
                const unsigned int* const anchor = anchorSequencesData + std::size_t(anchorIndex) * encodedSequencePitchInInts;
                char* const globalCorrectedAnchor = correctedAnchors + anchorIndex * decodedSequencePitchInBytes;

                if(isHQCorrection){

                    //set corrected anchor to consensus
                    for(int i = tgroup.thread_rank(); i < anchorLength; i += tgroup.size()){
                        const std::uint8_t nuc = msa.consensus[anchorColumnsBegin_incl + i];
                        assert(nuc < 4);
                        globalCorrectedAnchor[i] = SequenceHelpers::decodeBase(nuc);
                    }

                }else{

                    //set corrected anchor to consensus
                    for(int i = tgroup.thread_rank(); i < anchorLength; i += tgroup.size()){
                        const std::uint8_t nuc = msa.consensus[anchorColumnsBegin_incl + i];
                        assert(nuc < 4);
                        sharedCorrectedAnchor[i] = SequenceHelpers::decodeBase(nuc);
                    }
                    
                    tgroup.sync();                                   
                    
                    //maybe revert some positions to original base
                    for (int i = tgroup.thread_rank(); i < anchorLength; i += tgroup.size()){
                        const int msaPos = anchorColumnsBegin_incl + i;
                        const std::uint8_t origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(anchor, anchorLength, i);
                        const std::uint8_t consensusEncodedBase = msa.consensus[msaPos];

                        if (origEncodedBase != consensusEncodedBase){                            

                            ExtractAnchorInputData extractorInput;

                            extractorInput.origBase = SequenceHelpers::decodeBase(origEncodedBase);
                            extractorInput.consensusBase = SequenceHelpers::decodeBase(consensusEncodedBase);
                            extractorInput.estimatedCoverage = estimatedCoverage;
                            extractorInput.msaPos = msaPos;
                            extractorInput.anchorColumnsBegin_incl = anchorColumnsBegin_incl;
                            extractorInput.anchorColumnsEnd_excl = anchorColumnsEnd_excl;
                            extractorInput.msaProperties = msaProperties;
                            extractorInput.msa = msa;

                            AnchorExtractor extractFeatures{};

                            StridedIterator myFeatures{myFeaturesTransposed, blockDim.x};

                            extractFeatures(myFeatures, extractorInput);

                            auto sumreducethread = thrust::identity<float>{};

                            const bool useConsensus = gpuForest.decide(thread, myFeatures, forestThreshold, sumreducethread);
                            if(!useConsensus){
                                sharedCorrectedAnchor[i] = extractorInput.origBase;
                            }
                        }
                    }

                    tgroup.sync();

                    //copy shared correction to gmem
                    const int fullInts1 = anchorLength / sizeof(int);

                    for(int i = tgroup.thread_rank(); i < fullInts1; i += tgroup.size()) {
                        ((int*)globalCorrectedAnchor)[i] = ((int*)sharedCorrectedAnchor)[i];
                    }

                    for(int i = tgroup.thread_rank(); i < anchorLength - fullInts1 * sizeof(int); i += tgroup.size()) {
                        globalCorrectedAnchor[fullInts1 * sizeof(int) + i] 
                            = sharedCorrectedAnchor[fullInts1 * sizeof(int) + i];
                    } 

                }
                
                tgroup.sync();

            }else{
                if(tgroup.thread_rank() == 0){
                    isHighQualityAnchor[anchorIndex].hq(false);
                    anchorIsCorrected[anchorIndex] = false;
                }
            }            
        }
    }


    template<int BLOCKSIZE, int groupsize>
    __global__
    void msaCorrectAnchorsWithForestKernel_multiphase_initkernel(
        char* __restrict__ correctedAnchors,
        bool* __restrict__ anchorIsCorrected,
        AnchorHighQualityFlag* __restrict__ isHighQualityAnchor,
        GPUMultiMSA multiMSA,
        const unsigned int* __restrict__ anchorSequencesData,
        const int* __restrict__ d_indices_per_anchor,
        const int numAnchors,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        float estimatedErrorrate,
        float estimatedCoverage,
        float avg_support_threshold,
        float min_support_threshold,
        float min_coverage_threshold,
        GpuMSAProperties* __restrict__ msaPropertiesPerAnchor,
        int* __restrict__ numMismatches
    ){

        static_assert(BLOCKSIZE % groupsize == 0, "BLOCKSIZE % groupsize != 0");
        constexpr int groupsPerBlock = BLOCKSIZE / groupsize;
        static_assert(groupsize == 32);

        auto thread = cg::this_thread();
        auto tgroup = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int numGroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int groupIdInBlock = threadIdx.x / groupsize;

        auto minreduce = [&](auto val){
            using T = decltype(val);
            return cg::reduce(tgroup, val, cg::less<T>{});
        };

        auto maxreduce = [&](auto val){
            using T = decltype(val);
            return cg::reduce(tgroup, val, cg::greater<T>{});
        };

        auto sumreduce = [&](auto val){
            using T = decltype(val);
            return cg::reduce(tgroup, val, cg::plus<T>{});
        };

        using BlockReduce = cub::BlockReduce<int, BLOCKSIZE>;
        __shared__ typename BlockReduce::TempStorage temp_reduce;
      
        int myNumMismatches = 0;

        for(unsigned anchorIndex = groupId; anchorIndex < numAnchors; anchorIndex += numGroups){
            const int myNumIndices = d_indices_per_anchor[anchorIndex];
            if(myNumIndices > 0){

                const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);                

                const int anchorColumnsBegin_incl = msa.columnProperties->anchorColumnsBegin_incl;
                const int anchorColumnsEnd_excl = msa.columnProperties->anchorColumnsEnd_excl;

                GpuMSAProperties msaProperties = msa.getMSAProperties(
                    tgroup, 
                    sumreduce,
                    minreduce, 
                    minreduce, 
                    maxreduce,
                    anchorColumnsBegin_incl,
                    anchorColumnsEnd_excl
                );

                AnchorCorrectionQuality correctionQuality(avg_support_threshold, min_support_threshold, min_coverage_threshold, estimatedErrorrate);

                const bool isHQCorrection = correctionQuality.isHQCorrection(msaProperties.avg_support, msaProperties.min_support, msaProperties.min_coverage);

                if(tgroup.thread_rank() == 0){
                    anchorIsCorrected[anchorIndex] = true;
                    isHighQualityAnchor[anchorIndex].hq(isHQCorrection);
                    msaPropertiesPerAnchor[anchorIndex] = msaProperties;
                }

                const int anchorLength = anchorColumnsEnd_excl - anchorColumnsBegin_incl;
                const unsigned int* const anchor = anchorSequencesData + std::size_t(anchorIndex) * encodedSequencePitchInInts;
                char* const globalCorrectedAnchor = correctedAnchors + anchorIndex * decodedSequencePitchInBytes;

                //set corrected anchor to consensus
                for(int i = tgroup.thread_rank(); i < anchorLength; i += tgroup.size()){
                    const std::uint8_t nuc = msa.consensus[anchorColumnsBegin_incl + i];
                    assert(nuc < 4);
                    globalCorrectedAnchor[i] = SequenceHelpers::decodeBase(nuc);
                }

                if(!isHQCorrection){
                                
                    for (int i = tgroup.thread_rank(); i < anchorLength; i += tgroup.size()){
                        const int msaPos = anchorColumnsBegin_incl + i;
                        const std::uint8_t origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(anchor, anchorLength, i);
                        const std::uint8_t consensusEncodedBase = msa.consensus[msaPos];

                        if (origEncodedBase != consensusEncodedBase){                            
                            myNumMismatches++;
                        }
                    }

                }
            }else{
                if(tgroup.thread_rank() == 0){
                    isHighQualityAnchor[anchorIndex].hq(false);
                    anchorIsCorrected[anchorIndex] = false;
                }
            }            
        }
    
        int blockNumMismatches = BlockReduce(temp_reduce).Sum(myNumMismatches); 
        __syncthreads();

        if(threadIdx.x == 0){
            atomicAdd(numMismatches, blockNumMismatches);
        }
    }

    template<int BLOCKSIZE, int groupsize>
    __global__
    void msaCorrectAnchorsWithForestKernel_multiphase_gathermismatcheskernel(
        const AnchorHighQualityFlag* __restrict__ isHighQualityAnchor,
        GPUMultiMSA multiMSA,
        const unsigned int* __restrict__ anchorSequencesData,
        const int numAnchors,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        float estimatedErrorrate,
        float estimatedCoverage,
        float avg_support_threshold,
        float min_support_threshold,
        float min_coverage_threshold,
        int* __restrict__ numMismatches,
        int* __restrict__ mismatchAnchorIndices,
        int* __restrict__ mismatchPositionsInAnchors
    ){

        static_assert(BLOCKSIZE % groupsize == 0, "BLOCKSIZE % groupsize != 0");
        constexpr int groupsPerBlock = BLOCKSIZE / groupsize;
        static_assert(groupsize == 32);

        auto thread = cg::this_thread();
        auto tgroup = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int numGroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int groupIdInBlock = threadIdx.x / groupsize;

        __shared__ int smem_mismatchAnchorIndices[groupsPerBlock][groupsize];
        __shared__ int smem_mismatchPositionsInAnchors[groupsPerBlock][groupsize];
        __shared__ int smem_numMismatches[groupsPerBlock];

        if(tgroup.thread_rank() == 0){
            smem_numMismatches[groupIdInBlock] = 0;
        }
        tgroup.sync();

        for(unsigned anchorIndex = groupId; anchorIndex < numAnchors; anchorIndex += numGroups){
            const bool isHQCorrection = isHighQualityAnchor[anchorIndex].hq();

            const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);                

            const int anchorColumnsBegin_incl = msa.columnProperties->anchorColumnsBegin_incl;
            const int anchorColumnsEnd_excl = msa.columnProperties->anchorColumnsEnd_excl;

            const int anchorLength = anchorColumnsEnd_excl - anchorColumnsBegin_incl;
            const unsigned int* const anchor = anchorSequencesData + std::size_t(anchorIndex) * encodedSequencePitchInInts;

            if(!isHQCorrection){
                const int loopend = SDIV(anchorLength, groupsize) * groupsize;

                for (int i = tgroup.thread_rank(); i < loopend; i += tgroup.size()){
                    if(i < anchorLength){
                        const int msaPos = anchorColumnsBegin_incl + i;
                        const std::uint8_t origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(anchor, anchorLength, i);
                        const std::uint8_t consensusEncodedBase = msa.consensus[msaPos];

                        if (origEncodedBase != consensusEncodedBase){
                            // if(anchorIndex == 1){
                            //     printf("anchorIndex %d, position %d\n", anchorIndex, i);
                            // }                    
                            auto selectedgroup = cg::coalesced_threads();

                            const int smemarraypos = selectedgroup.thread_rank();
                            smem_mismatchAnchorIndices[groupIdInBlock][smemarraypos] = anchorIndex;
                            smem_mismatchPositionsInAnchors[groupIdInBlock][smemarraypos] = i;

                            if(selectedgroup.thread_rank() == 0){
                                smem_numMismatches[groupIdInBlock] = selectedgroup.size();
                            }
                        }
                    }
                    tgroup.sync();
                    if(smem_numMismatches[groupIdInBlock] > 0){
                        int globalIndex;

                        if (tgroup.thread_rank() == 0) {
                            globalIndex = atomicAdd(numMismatches, smem_numMismatches[groupIdInBlock]);
                        }

                        globalIndex = tgroup.shfl(globalIndex, 0);
                        for(int k = tgroup.thread_rank(); k < smem_numMismatches[groupIdInBlock]; k += tgroup.size()){
                            mismatchAnchorIndices[globalIndex + k] = smem_mismatchAnchorIndices[groupIdInBlock][k];
                            mismatchPositionsInAnchors[globalIndex + k] = smem_mismatchPositionsInAnchors[groupIdInBlock][k];
                        }

                        tgroup.sync();
                        smem_numMismatches[groupIdInBlock] = 0;
                        tgroup.sync();
                    }
                }
            }            
        }
    }


    template<int BLOCKSIZE, class AnchorExtractor>
    __global__
    void msaCorrectAnchorsWithForestKernel_multiphase_extractkernel(
        GPUMultiMSA multiMSA,
        const unsigned int* __restrict__ anchorSequencesData,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        float estimatedErrorrate,
        float estimatedCoverage,
        float avg_support_threshold,
        float min_support_threshold,
        float min_coverage_threshold,
        float* __restrict__ gmemFeaturesTransposed,
        int numMismatches,
        const int* __restrict__ mismatchAnchorIndices,
        const int* __restrict__ mismatchPositionsInAnchors,
        const GpuMSAProperties* __restrict__ msaPropertiesPerAnchor
    ){

        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        for(int p = tid; p < numMismatches; p += stride){
            float* const myFeaturesTransposed = gmemFeaturesTransposed + p;
            int const myFeaturesStride = numMismatches;

            const int anchorIndex = mismatchAnchorIndices[p];
            const int positionInAnchor = mismatchPositionsInAnchors[p];

            const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);                

            const int anchorColumnsBegin_incl = msa.columnProperties->anchorColumnsBegin_incl;
            const int anchorColumnsEnd_excl = msa.columnProperties->anchorColumnsEnd_excl;
            const GpuMSAProperties& msaProperties = msaPropertiesPerAnchor[anchorIndex];

            const int anchorLength = anchorColumnsEnd_excl - anchorColumnsBegin_incl;
            const unsigned int* const anchor = anchorSequencesData + std::size_t(anchorIndex) * encodedSequencePitchInInts;

            const int msaPos = anchorColumnsBegin_incl + positionInAnchor;
            const std::uint8_t origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(anchor, anchorLength, positionInAnchor);
            const std::uint8_t consensusEncodedBase = msa.consensus[msaPos];                   

            ExtractAnchorInputData extractorInput;

            extractorInput.origBase = SequenceHelpers::decodeBase(origEncodedBase);
            extractorInput.consensusBase = SequenceHelpers::decodeBase(consensusEncodedBase);
            extractorInput.estimatedCoverage = estimatedCoverage;
            extractorInput.msaPos = msaPos;
            extractorInput.anchorColumnsBegin_incl = anchorColumnsBegin_incl;
            extractorInput.anchorColumnsEnd_excl = anchorColumnsEnd_excl;
            extractorInput.msaProperties = msaProperties;
            extractorInput.msa = msa;

            AnchorExtractor extractFeatures{};

            StridedIterator myFeatures{myFeaturesTransposed, myFeaturesStride};
            extractFeatures(myFeatures, extractorInput);     
        }
    }

    template<int BLOCKSIZE, class GpuClf>
    __global__
    void msaCorrectAnchorsWithForestKernel_multiphase_correctkernel_thread(
        char* __restrict__ correctedAnchors,
        GPUMultiMSA multiMSA,
        GpuClf gpuForest,
        float forestThreshold,
        const unsigned int* __restrict__ anchorSequencesData,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        float estimatedErrorrate,
        float estimatedCoverage,
        float avg_support_threshold,
        float min_support_threshold,
        float min_coverage_threshold,
        int numFeatures,
        bool useGmemFeatures,
        const float* __restrict__ gmemFeaturesTransposed,
        int numMismatches,
        const int* __restrict__ mismatchAnchorIndices,
        const int* __restrict__ mismatchPositionsInAnchors
    ){

        auto thread = cg::this_thread();
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;
      
        extern __shared__ float smem[];

        float* mySmemFeaturesTransposed = &smem[threadIdx.x];
        int mySmemFeaturesStride = blockDim.x;

        for(int p = tid; p < numMismatches; p += stride){
            const int anchorIndex = mismatchAnchorIndices[p];
            const int positionInAnchor = mismatchPositionsInAnchors[p];

            const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);                

            const int anchorColumnsBegin_incl = msa.columnProperties->anchorColumnsBegin_incl;
            const int anchorColumnsEnd_excl = msa.columnProperties->anchorColumnsEnd_excl;

            const int anchorLength = anchorColumnsEnd_excl - anchorColumnsBegin_incl;
            const unsigned int* const anchor = anchorSequencesData + std::size_t(anchorIndex) * encodedSequencePitchInInts;
            char* const globalCorrectedAnchor = correctedAnchors + anchorIndex * decodedSequencePitchInBytes;

            if(!useGmemFeatures){
                for(int k = 0; k < numFeatures; k++){
                    mySmemFeaturesTransposed[k * blockDim.x] = gmemFeaturesTransposed[p + numMismatches * k];
                }
            }

            const float* const myFeaturesTransposed = useGmemFeatures ? gmemFeaturesTransposed + p : mySmemFeaturesTransposed;
            const int myFeaturesStride = useGmemFeatures ? numMismatches : blockDim.x;

            StridedIterator myFeatures{myFeaturesTransposed, myFeaturesStride};

            auto sumreducethread = thrust::identity<float>{};

            const bool useConsensus = gpuForest.decide(thread, myFeatures, forestThreshold, sumreducethread);
            if(!useConsensus){
                globalCorrectedAnchor[positionInAnchor] = SequenceHelpers::decodeBase(SequenceHelpers::getEncodedNuc2Bit(anchor, anchorLength, positionInAnchor));
            }        
        }
    }

    template<int BLOCKSIZE, int groupsize, class GpuClf>
    __global__
    void msaCorrectAnchorsWithForestKernel_multiphase_correctkernel_group(
        char* __restrict__ correctedAnchors,
        GPUMultiMSA multiMSA,
        GpuClf gpuForest,
        float forestThreshold,
        const unsigned int* __restrict__ anchorSequencesData,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        float estimatedErrorrate,
        float estimatedCoverage,
        float avg_support_threshold,
        float min_support_threshold,
        float min_coverage_threshold,
        int numFeatures,
        bool useGmemFeatures,
        const float* __restrict__ gmemFeaturesTransposed,
        int numMismatches,
        const int* __restrict__ mismatchAnchorIndices,
        const int* __restrict__ mismatchPositionsInAnchors
    ){

        static_assert(BLOCKSIZE % groupsize == 0, "BLOCKSIZE % groupsize != 0");
        constexpr int groupsPerBlock = BLOCKSIZE / groupsize;
        //static_assert(groupsize == 32);

        auto thread = cg::this_thread();
        auto tgroup = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int numGroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int groupIdInBlock = threadIdx.x / groupsize;
      
        extern __shared__ float smem[];

        float* mySmemFeatures = &smem[groupIdInBlock * numFeatures];

        for(int p = groupId; p < numMismatches; p += numGroups){
            const int anchorIndex = mismatchAnchorIndices[p];
            const int positionInAnchor = mismatchPositionsInAnchors[p];

            const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);                

            const int anchorColumnsBegin_incl = msa.columnProperties->anchorColumnsBegin_incl;
            const int anchorColumnsEnd_excl = msa.columnProperties->anchorColumnsEnd_excl;

            const int anchorLength = anchorColumnsEnd_excl - anchorColumnsBegin_incl;
            const unsigned int* const anchor = anchorSequencesData + std::size_t(anchorIndex) * encodedSequencePitchInInts;
            char* const globalCorrectedAnchor = correctedAnchors + anchorIndex * decodedSequencePitchInBytes;

            bool useConsensus = false;

            if(!useGmemFeatures){
                //noncoalesced
                for(int k = tgroup.thread_rank(); k < numFeatures; k += tgroup.size()){
                    mySmemFeatures[k] = gmemFeaturesTransposed[p + numMismatches * k];
                }
                tgroup.sync();

                auto sumreduce = [&](auto val){
                    using T = decltype(val);
                    return cg::reduce(tgroup, val, cg::plus<T>{});
                };

                useConsensus = gpuForest.decide(tgroup, &mySmemFeatures[0], forestThreshold, sumreduce);
            }else{
                StridedIterator myFeatures{gmemFeaturesTransposed + p, numMismatches};

                auto sumreduce = [&](auto val){
                    using T = decltype(val);
                    return cg::reduce(tgroup, val, cg::plus<T>{});
                };

                useConsensus = gpuForest.decide(tgroup, &mySmemFeatures[0], forestThreshold, sumreduce);
            }

            if(!useConsensus){
                if(tgroup.thread_rank() == 0){
                    globalCorrectedAnchor[positionInAnchor] = SequenceHelpers::decodeBase(SequenceHelpers::getEncodedNuc2Bit(anchor, anchorLength, positionInAnchor));
                }
            }        
        }
    }

    template<int BLOCKSIZE, class GpuClf>
    __global__
    void msaCorrectAnchorsWithForestKernel_multiphase_correctkernel_block(
        char* __restrict__ correctedAnchors,
        GPUMultiMSA multiMSA,
        GpuClf gpuForest,
        float forestThreshold,
        const unsigned int* __restrict__ anchorSequencesData,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        float estimatedErrorrate,
        float estimatedCoverage,
        float avg_support_threshold,
        float min_support_threshold,
        float min_coverage_threshold,
        int numFeatures,
        bool useGmemFeatures,
        const float* __restrict__ gmemFeaturesTransposed,
        int numMismatches,
        const int* __restrict__ mismatchAnchorIndices,
        const int* __restrict__ mismatchPositionsInAnchors
    ){
        constexpr int numWarps = BLOCKSIZE / 32;

        auto tgroup = cg::this_thread_block();
        auto warp = cg::tiled_partition<32>(tgroup);

        const int numGroups = gridDim.x;
        const int groupId = blockIdx.x;
        const int groupIdInBlock = 0;
      
        extern __shared__ float smem[];
        __shared__ float temp_reduce[numWarps+1];

        float* mySmemFeatures = &smem[groupIdInBlock * numFeatures];

        //reduce over thread block tgroup
        auto sumreduce = [&](auto val){
            using T = decltype(val);

            tgroup.sync();
            const auto warpreduced = cg::reduce(warp, val, cg::plus<T>{});
            if(warp.thread_rank() == 0){
                temp_reduce[warp.meta_group_rank()] = warpreduced;
            }
            tgroup.sync();
            if(warp.meta_group_rank() == 0){
                auto warpval = 0.0f;
                if(warp.thread_rank() < numWarps){
                    warpval = temp_reduce[warp.thread_rank()];
                }
                const auto blockreduced = cg::reduce(warp, warpval, cg::plus<T>{});
                if(warp.thread_rank() == 0){
                    temp_reduce[numWarps] = blockreduced;
                }
            }
            tgroup.sync();
            return temp_reduce[numWarps];
        };

        for(int p = groupId; p < numMismatches; p += numGroups){
            const int anchorIndex = mismatchAnchorIndices[p];
            const int positionInAnchor = mismatchPositionsInAnchors[p];

            const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);                

            const int anchorColumnsBegin_incl = msa.columnProperties->anchorColumnsBegin_incl;
            const int anchorColumnsEnd_excl = msa.columnProperties->anchorColumnsEnd_excl;

            const int anchorLength = anchorColumnsEnd_excl - anchorColumnsBegin_incl;
            const unsigned int* const anchor = anchorSequencesData + std::size_t(anchorIndex) * encodedSequencePitchInInts;
            char* const globalCorrectedAnchor = correctedAnchors + anchorIndex * decodedSequencePitchInBytes;

            bool useConsensus = false;

            if(!useGmemFeatures){
                //noncoalesced
                for(int k = tgroup.thread_rank(); k < numFeatures; k += tgroup.size()){
                    mySmemFeatures[k] = gmemFeaturesTransposed[p + numMismatches * k];
                }
                tgroup.sync();                

                useConsensus = gpuForest.decide(tgroup, &mySmemFeatures[0], forestThreshold, sumreduce);
            }else{
                StridedIterator myFeatures{gmemFeaturesTransposed + p, numMismatches};

                useConsensus = gpuForest.decide(tgroup, &mySmemFeatures[0], forestThreshold, sumreduce);
            }

            if(!useConsensus){
                if(tgroup.thread_rank() == 0){
                    globalCorrectedAnchor[positionInAnchor] = SequenceHelpers::decodeBase(SequenceHelpers::getEncodedNuc2Bit(anchor, anchorLength, positionInAnchor));
                }
            }        
        }
    }


    template<int BLOCKSIZE, class AnchorExtractor, class GpuClf>
    __global__
    void msaCorrectAnchorsWithForestKernel_multiphase_extractcorrectkernel(
        char* __restrict__ correctedAnchors,
        GPUMultiMSA multiMSA,
        GpuClf gpuForest,
        float forestThreshold,
        const unsigned int* __restrict__ anchorSequencesData,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        float estimatedErrorrate,
        float estimatedCoverage,
        float avg_support_threshold,
        float min_support_threshold,
        float min_coverage_threshold,
        bool useGmemFeatures,
        float* __restrict__ gmemFeaturesTransposed,
        int numMismatches,
        const int* __restrict__ mismatchAnchorIndices,
        const int* __restrict__ mismatchPositionsInAnchors,
        const GpuMSAProperties* __restrict__ msaPropertiesPerAnchor
    ){

        auto thread = cg::this_thread();
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;
      
        extern __shared__ float smem[];

        float* myFeaturesTransposed = &smem[threadIdx.x];
        int myFeaturesStride = blockDim.x;

        if(useGmemFeatures){
            myFeaturesTransposed = gmemFeaturesTransposed + tid;
            myFeaturesStride = stride;
        }

        for(int p = tid; p < numMismatches; p += stride){
            const int anchorIndex = mismatchAnchorIndices[p];
            const int positionInAnchor = mismatchPositionsInAnchors[p];

            const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);                

            const int anchorColumnsBegin_incl = msa.columnProperties->anchorColumnsBegin_incl;
            const int anchorColumnsEnd_excl = msa.columnProperties->anchorColumnsEnd_excl;
            const GpuMSAProperties& msaProperties = msaPropertiesPerAnchor[anchorIndex];

            const int anchorLength = anchorColumnsEnd_excl - anchorColumnsBegin_incl;
            const unsigned int* const anchor = anchorSequencesData + std::size_t(anchorIndex) * encodedSequencePitchInInts;
            char* const globalCorrectedAnchor = correctedAnchors + anchorIndex * decodedSequencePitchInBytes;

            const int msaPos = anchorColumnsBegin_incl + positionInAnchor;
            const std::uint8_t origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(anchor, anchorLength, positionInAnchor);
            const std::uint8_t consensusEncodedBase = msa.consensus[msaPos];                   

            ExtractAnchorInputData extractorInput;

            extractorInput.origBase = SequenceHelpers::decodeBase(origEncodedBase);
            extractorInput.consensusBase = SequenceHelpers::decodeBase(consensusEncodedBase);
            extractorInput.estimatedCoverage = estimatedCoverage;
            extractorInput.msaPos = msaPos;
            extractorInput.anchorColumnsBegin_incl = anchorColumnsBegin_incl;
            extractorInput.anchorColumnsEnd_excl = anchorColumnsEnd_excl;
            extractorInput.msaProperties = msaProperties;
            extractorInput.msa = msa;

            AnchorExtractor extractFeatures{};

            StridedIterator myFeatures{myFeaturesTransposed, myFeaturesStride};
            extractFeatures(myFeatures, extractorInput);

            auto sumreducethread = thrust::identity<float>{};

            const bool useConsensus = gpuForest.decide(thread, myFeatures, forestThreshold, sumreducethread);
            if(anchorIndex == 1){
                // if(positionInAnchor == 35){
                //     for(int x = 0; x < AnchorExtractor::numFeatures(); x++){
                //         printf("%f, ", *(myFeatures + x));
                //     }
                //     printf("\n");
                // }
                // printf("new anchorIndex %d, position %d, %c %c, useConsensus %d\n", anchorIndex, positionInAnchor, extractorInput.origBase, extractorInput.consensusBase, useConsensus);
            }
            if(!useConsensus){
                globalCorrectedAnchor[positionInAnchor] = extractorInput.origBase;
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
        const AlignmentOrientation* __restrict__ bestAlignmentFlags,
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

        GpuClf localForest = gpuForest;
        // localForest.numTrees = gpuForest.numTrees;
        // localForest.data = sharedForestNodes;

        // for(int i = threadIdx.x; i < localForest.numTrees; i += BLOCKSIZE){
        //     localForest.data[i] = gpuForest.data[i];
        // }
        
        // __syncthreads();

        for(int id = groupId; id < loopEnd; id += numGroups){

            const int candidateIndex = candidateIndicesOfCandidatesToBeCorrected[id];
            const int anchorIndex = anchorIndicesOfCandidates[candidateIndex];
            const int destinationIndex = id;

            const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);

            const int candidate_length = candidateSequencesLengths[candidateIndex];

            const int shift = shifts[candidateIndex];
            const int anchorColumnsBegin_incl = msa.columnProperties->anchorColumnsBegin_incl;
            const int anchorColumnsEnd_excl = msa.columnProperties->anchorColumnsEnd_excl;
            const int queryColumnsBegin_incl = anchorColumnsBegin_incl + shift;
            const int queryColumnsEnd_excl = anchorColumnsBegin_incl + shift + candidate_length;

            auto i_f = thrust::identity<float>{};
            auto i_i = thrust::identity<int>{};

            GpuMSAProperties msaProperties = msa.getMSAProperties(
                thread, i_f, i_f, i_i, i_i,
                queryColumnsBegin_incl,
                queryColumnsEnd_excl
            );

            // if(id == 1 && tgroup.thread_rank() == 0){
            //     printf("is reverse complement? %d\n", (bestAlignmentFlags[candidateIndex] == AlignmentOrientation::ReverseComplement));
            //     printf("msa consensus beginning at %d\n", queryColumnsBegin_incl);
            //     for(int i =0; i < candidate_length; i +=1) {
            //         printf("%c", SequenceHelpers::decodeBase(msa.consensus[queryColumnsBegin_incl + i]));
            //     }
            //     printf("\n");
            // }

            for(int i = tgroup.thread_rank(); i < candidate_length; i += tgroup.size()) {
                shared_correctedCandidate[i] = SequenceHelpers::decodeBase(msa.consensus[queryColumnsBegin_incl + i]);
            }

            tgroup.sync(); 

            const AlignmentOrientation bestAlignmentFlag = bestAlignmentFlags[candidateIndex];

            const unsigned int* const encUncorrectedCandidate = candidateSequencesData 
                        + std::size_t(candidateIndex) * encodedSequencePitchInInts;

            
            // if(id == 1 && tgroup.thread_rank() == 0){
            //     printf("shared_correctedCandidate\n");
            //     for(int i =0; i < candidate_length; i +=1) {
            //         printf("%c", shared_correctedCandidate[i]);
            //     }
            //     printf("\n");

            //     printf("orig\n");
            //     for(int i =0; i < candidate_length; i +=1) {
            //         std::uint8_t origEncodedBase = 0;

            //         if(bestAlignmentFlag == AlignmentOrientation::ReverseComplement){
            //             origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(
            //                 encUncorrectedCandidate,
            //                 candidate_length,
            //                 candidate_length - i - 1
            //             );
            //             origEncodedBase = SequenceHelpers::complementBase2Bit(origEncodedBase);
            //         }else{
            //             origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(
            //                 encUncorrectedCandidate,
            //                 candidate_length,
            //                 i
            //             );
            //         }

            //         const char origBase = SequenceHelpers::decodeBase(origEncodedBase);
            //         printf("%c", origBase);
            //     }
            //     printf("\n");
            // }

            for(int i = 0; i < candidate_length; i += 1){
                std::uint8_t origEncodedBase = 0;

                if(bestAlignmentFlag == AlignmentOrientation::ReverseComplement){
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

                // if(id == 1){
                //     if(tgroup.thread_rank() == 0){
                //         for(int k = 0; k < candidate_length; k++){
                //             if(i == k){
                //                 printf("orig cons position %d, %c %c\n", i, origBase, consensusBase);
                //             }
                //             tgroup.sync();
                //         }
                //     }
                // }
                
                if(origBase != consensusBase){
                    const int msaPos = queryColumnsBegin_incl + i;

                    if(tgroup.thread_rank() == 0){
                        // if(id == 1){
                        //     printf("orig cons mismatch position %d, %c %c\n", i, origBase, consensusBase);
                        // }

                        ExtractCandidateInputData& extractorInput = sharedExtractInput[groupIdInBlock];

                        extractorInput.origBase = origBase;
                        extractorInput.consensusBase = consensusBase;
                        extractorInput.estimatedCoverage = estimatedCoverage;
                        extractorInput.msaPos = msaPos;
                        extractorInput.anchorColumnsBegin_incl = anchorColumnsBegin_incl;
                        extractorInput.anchorColumnsEnd_excl = anchorColumnsEnd_excl;
                        extractorInput.queryColumnsBegin_incl = queryColumnsBegin_incl;
                        extractorInput.queryColumnsEnd_excl = queryColumnsEnd_excl;
                        extractorInput.msaProperties = msaProperties;
                        extractorInput.msa = msa;

                        CandidateExtractor extractFeatures{};
                        extractFeatures(&sharedFeatures[groupIdInBlock][0], extractorInput);
                    }

                    tgroup.sync();

                    auto sumreduce = [&](auto val){
                        using T = decltype(val);
                        return cg::reduce(tgroup, val, cg::plus<T>{});
                    };

                    //localForest gpuForest
                    const bool useConsensus = localForest.decide(tgroup, &sharedFeatures[groupIdInBlock][0], forestThreshold, sumreduce);

                    if(tgroup.thread_rank() == 0){
                        if(!useConsensus){
                            shared_correctedCandidate[i] = origBase;
                        }
                    }

                    tgroup.sync();
                }
            }            

            //the forward strand will be returned -> make reverse complement again
            if(bestAlignmentFlag == AlignmentOrientation::ReverseComplement) {
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


    /*
        Pass 1 (this kernel): Set corrected candidate to msa consensus
        Pass 2: Use RF to change some positions of corrected candidates
    */

    template<int BLOCKSIZE, int groupsize>
    __global__
    void msaCorrectCandidatesWithForestKernel_multiphase_initCorrectedCandidatesKernel(
        char* __restrict__ correctedCandidates,
        GPUMultiMSA multiMSA,
        const int* __restrict__ shifts,
        const AlignmentOrientation* __restrict__ bestAlignmentFlags,
        const unsigned int* __restrict__ candidateSequencesData,
        const int* __restrict__ candidateSequencesLengths,
        const bool* __restrict__ d_candidateContainsN,
        const int* __restrict__ candidateIndicesOfCandidatesToBeCorrected,
        const int* __restrict__ numCandidatesToBeCorrected,
        const int* __restrict__ anchorIndicesOfCandidates,         
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        size_t dynamicsmemSequencePitchInInts
    ){

        static_assert(BLOCKSIZE % groupsize == 0, "BLOCKSIZE % groupsize != 0");
        //constexpr int groupsPerBlock = BLOCKSIZE / groupsize;
        static_assert(groupsize == 32);

        auto tgroup = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int numGroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int groupIdInBlock = threadIdx.x / groupsize;

        extern __shared__ int dynamicsmem[]; // for sequences

        char* const shared_correctedCandidate = (char*)(&dynamicsmem[0] + dynamicsmemSequencePitchInInts * groupIdInBlock);

        const int loopEnd = *numCandidatesToBeCorrected;

        for(int id = groupId; id < loopEnd; id += numGroups){

            const int candidateIndex = candidateIndicesOfCandidatesToBeCorrected[id];
            const int anchorIndex = anchorIndicesOfCandidates[candidateIndex];
            const int destinationIndex = id;

            const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);

            const int candidate_length = candidateSequencesLengths[candidateIndex];
            const int shift = shifts[candidateIndex];
            const int anchorColumnsBegin_incl = msa.columnProperties->anchorColumnsBegin_incl;
            const int queryColumnsBegin_incl = anchorColumnsBegin_incl + shift;

            // if(id == 1 && tgroup.thread_rank() == 0){
            //     printf("msa consensus beginning at %d\n", queryColumnsBegin_incl);
            //     for(int i =0; i < candidate_length; i +=1) {
            //         printf("%c", SequenceHelpers::decodeBase(msa.consensus[queryColumnsBegin_incl + i]));
            //     }
            //     printf("\n");
            // }


            for(int i = tgroup.thread_rank(); i < candidate_length; i += tgroup.size()) {
                shared_correctedCandidate[i] = SequenceHelpers::decodeBase(msa.consensus[queryColumnsBegin_incl + i]);
            }

            tgroup.sync(); 

            const AlignmentOrientation bestAlignmentFlag = bestAlignmentFlags[candidateIndex];

            //the forward strand will be returned -> make reverse complement again
            if(bestAlignmentFlag == AlignmentOrientation::ReverseComplement) {
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

    template<int BLOCKSIZE, int groupsize>
    __global__
    void msaCorrectCandidatesWithForestKernel_multiphase_countMismatchesKernel(
        const char* __restrict__ correctedCandidates,
        const AlignmentOrientation* __restrict__ bestAlignmentFlags,
        const unsigned int* __restrict__ candidateSequencesData,
        const int* __restrict__ candidateSequencesLengths,
        const int* __restrict__ candidateIndicesOfCandidatesToBeCorrected,
        const int* __restrict__ numCandidatesToBeCorrected,
        const int* __restrict__ anchorIndicesOfCandidates,         
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        int* __restrict__ numMismatches
    ){        
        static_assert(BLOCKSIZE % groupsize == 0, "BLOCKSIZE % groupsize != 0");
        constexpr int groupsPerBlock = BLOCKSIZE / groupsize;
        static_assert(groupsize == 32);

        auto tgroup = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int numGroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int groupIdInBlock = threadIdx.x / groupsize;

        using BlockReduce = cub::BlockReduce<int, BLOCKSIZE>;
        __shared__ typename BlockReduce::TempStorage temp_reduce;

       
        const int loopEnd = *numCandidatesToBeCorrected;

        int myNumMismatches = 0;

        for(int id = groupId; id < loopEnd; id += numGroups){

            const int candidateIndex = candidateIndicesOfCandidatesToBeCorrected[id];
            const int anchorIndex = anchorIndicesOfCandidates[candidateIndex];
            const int destinationIndex = id;

            const int candidate_length = candidateSequencesLengths[candidateIndex];
            const AlignmentOrientation bestAlignmentFlag = bestAlignmentFlags[candidateIndex];

            const unsigned int* const encUncorrectedCandidate = candidateSequencesData 
                        + std::size_t(candidateIndex) * encodedSequencePitchInInts;
            const char* const decCorrectedCandidate = correctedCandidates + destinationIndex * decodedSequencePitchInBytes;

            const int loopEnd = SDIV(candidate_length, tgroup.size()) * tgroup.size();

            for(int i = tgroup.thread_rank(); i < loopEnd; i += tgroup.size()){
                if(i < candidate_length){
                    std::uint8_t origEncodedBase = 0;
                    char consensusBase = 'F';

                    if(bestAlignmentFlag == AlignmentOrientation::ReverseComplement){
                        origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(
                            encUncorrectedCandidate,
                            candidate_length,
                            candidate_length - i - 1
                        );
                        origEncodedBase = SequenceHelpers::complementBase2Bit(origEncodedBase);
                        consensusBase = SequenceHelpers::complementBaseDecoded(decCorrectedCandidate[candidate_length - i - 1]);
                    }else{
                        origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(
                            encUncorrectedCandidate,
                            candidate_length,
                            i
                        );
                        consensusBase = decCorrectedCandidate[i];
                    }

                    const char origBase = SequenceHelpers::decodeBase(origEncodedBase);

                    if(origBase != consensusBase){
                        myNumMismatches++;
                    }
                }
            }
        }

        int blockNumMismatches = BlockReduce(temp_reduce).Sum(myNumMismatches); 
        __syncthreads();

        if(threadIdx.x == 0){
            atomicAdd(numMismatches, blockNumMismatches);
        }
    }

    struct MismatchPositionsRaw{
        int maxPositions;
        char* origBase;
        char* consensusBase;
        int* position;
        int* anchorIndex;
        int* candidateIndex;
        int* destinationIndex;
        int* numPositions;
    };

    struct MismatchPositions{
        int maxPositions;
        rmm::device_uvector<char> origBase;
        rmm::device_uvector<char> consensusBase;
        rmm::device_uvector<int> position;
        rmm::device_uvector<int> anchorIndex;
        rmm::device_uvector<int> candidateIndex;
        rmm::device_uvector<int> destinationIndex;
        rmm::device_scalar<int> numPositions;

        MismatchPositions(int maxPositions, cudaStream_t stream, rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
        : maxPositions(maxPositions),
            origBase(maxPositions, stream, mr),
            consensusBase(maxPositions, stream, mr),
            position(maxPositions, stream, mr),
            anchorIndex(maxPositions, stream, mr),
            candidateIndex(maxPositions, stream, mr),
            destinationIndex(maxPositions, stream, mr),
            numPositions(0, stream, mr)
        {}

        operator MismatchPositionsRaw(){
            return MismatchPositionsRaw{
                maxPositions, 
                origBase.data(), 
                consensusBase.data(),
                position.data(),
                anchorIndex.data(),
                candidateIndex.data(),
                destinationIndex.data(),
                numPositions.data(),
            };
        }
    };

    template<int BLOCKSIZE, int groupsize>
    __global__
    void msaCorrectCandidatesWithForestKernel_multiphase_findMismatchesKernel(
        const char* __restrict__ correctedCandidates,
        const AlignmentOrientation* __restrict__ bestAlignmentFlags,
        const unsigned int* __restrict__ candidateSequencesData,
        const int* __restrict__ candidateSequencesLengths,
        const int* __restrict__ candidateIndicesOfCandidatesToBeCorrected,
        const int* __restrict__ numCandidatesToBeCorrected,
        const int* __restrict__ anchorIndicesOfCandidates,         
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        MismatchPositionsRaw mismatchPositions
    ){
       
        static_assert(BLOCKSIZE % groupsize == 0, "BLOCKSIZE % groupsize != 0");
        constexpr int groupsPerBlock = BLOCKSIZE / groupsize;
        static_assert(groupsize == 32);

        auto tgroup = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int numGroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int groupIdInBlock = threadIdx.x / groupsize;

        constexpr int maxNumMismatchLocations = groupsize;

        __shared__ char smem_mismatchLocations_origBase[groupsPerBlock][maxNumMismatchLocations];
        __shared__ char smem_mismatchLocations_consensusBase[groupsPerBlock][maxNumMismatchLocations];
        __shared__ int smem_mismatchLocations_anchorIndex[groupsPerBlock][maxNumMismatchLocations];
        __shared__ int smem_mismatchLocations_candidateIndex[groupsPerBlock][maxNumMismatchLocations];
        __shared__ int smem_mismatchLocations_destinationIndex[groupsPerBlock][maxNumMismatchLocations];
        __shared__ int smem_mismatchLocations_position[groupsPerBlock][maxNumMismatchLocations];
        __shared__ int smem_numMismatchLocations[groupsPerBlock];

        if(tgroup.thread_rank() == 0){
            smem_numMismatchLocations[groupIdInBlock] = 0;
        }
        tgroup.sync();
      
        const int loopEnd = *numCandidatesToBeCorrected;

        for(int id = groupId; id < loopEnd; id += numGroups){

            const int candidateIndex = candidateIndicesOfCandidatesToBeCorrected[id];
            const int anchorIndex = anchorIndicesOfCandidates[candidateIndex];
            const int destinationIndex = id;

            const int candidate_length = candidateSequencesLengths[candidateIndex];
            const AlignmentOrientation bestAlignmentFlag = bestAlignmentFlags[candidateIndex];

            const unsigned int* const encUncorrectedCandidate = candidateSequencesData 
                        + std::size_t(candidateIndex) * encodedSequencePitchInInts;
            const char* const decCorrectedCandidate = correctedCandidates + destinationIndex * decodedSequencePitchInBytes;

            const int loopEnd = SDIV(candidate_length, tgroup.size()) * tgroup.size();

            for(int i = tgroup.thread_rank(); i < loopEnd; i += tgroup.size()){

                if(i < candidate_length){
                    std::uint8_t origEncodedBase = 0;
                    char consensusBase = 'F';

                    if(bestAlignmentFlag == AlignmentOrientation::ReverseComplement){
                        origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(
                            encUncorrectedCandidate,
                            candidate_length,
                            candidate_length - i - 1
                        );
                        origEncodedBase = SequenceHelpers::complementBase2Bit(origEncodedBase);
                        consensusBase = SequenceHelpers::complementBaseDecoded(decCorrectedCandidate[candidate_length - i - 1]);
                    }else{
                        origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(
                            encUncorrectedCandidate,
                            candidate_length,
                            i
                        );
                        consensusBase = decCorrectedCandidate[i];
                    }

                    const char origBase = SequenceHelpers::decodeBase(origEncodedBase);

                    if(origBase != consensusBase){
                        auto selectedgroup = cg::coalesced_threads();

                        const int smemarraypos = selectedgroup.thread_rank();

                        smem_mismatchLocations_anchorIndex[groupIdInBlock][smemarraypos] = anchorIndex;
                        smem_mismatchLocations_candidateIndex[groupIdInBlock][smemarraypos] = candidateIndex;
                        smem_mismatchLocations_destinationIndex[groupIdInBlock][smemarraypos] = destinationIndex;
                        smem_mismatchLocations_position[groupIdInBlock][smemarraypos] = i;
                        smem_mismatchLocations_origBase[groupIdInBlock][smemarraypos] = origBase;
                        smem_mismatchLocations_consensusBase[groupIdInBlock][smemarraypos] = consensusBase;

                        if(selectedgroup.thread_rank() == 0){
                            smem_numMismatchLocations[groupIdInBlock] = selectedgroup.size();
                        }
                    }
                }
                tgroup.sync();

                if(smem_numMismatchLocations[groupIdInBlock] > 0){
                    int globalIndex;

                    // elect the first active thread to perform atomic add
                    if (tgroup.thread_rank() == 0) {
                        globalIndex = atomicAdd(mismatchPositions.numPositions, smem_numMismatchLocations[groupIdInBlock]);
                    }

                    globalIndex = tgroup.shfl(globalIndex, 0);
                    for(int k = tgroup.thread_rank(); k < smem_numMismatchLocations[groupIdInBlock]; k += tgroup.size()){
                        mismatchPositions.origBase[globalIndex + k] = smem_mismatchLocations_origBase[groupIdInBlock][k];
                        mismatchPositions.consensusBase[globalIndex + k] = smem_mismatchLocations_consensusBase[groupIdInBlock][k];
                        mismatchPositions.position[globalIndex + k] = smem_mismatchLocations_position[groupIdInBlock][k];
                        mismatchPositions.anchorIndex[globalIndex + k] = smem_mismatchLocations_anchorIndex[groupIdInBlock][k];
                        mismatchPositions.candidateIndex[globalIndex + k] = smem_mismatchLocations_candidateIndex[groupIdInBlock][k];
                        mismatchPositions.destinationIndex[globalIndex + k] = smem_mismatchLocations_destinationIndex[groupIdInBlock][k];
                    }

                    tgroup.sync();
                    smem_numMismatchLocations[groupIdInBlock] = 0;
                    tgroup.sync();
                }

            }
        }

    }


    template<int BLOCKSIZE, int groupsize>
    __global__
    void msaCorrectCandidatesWithForestKernel_multiphase_msapropsKernel(
        GPUMultiMSA multiMSA,
        const int* __restrict__ shifts,
        const int* __restrict__ candidateSequencesLengths,       
        MismatchPositionsRaw mismatchPositions,
        GpuMSAProperties* __restrict__ msaPropertiesPerPosition
    ){

        /*
            Use groupsize threads per candidate to perform correction
        */
        static_assert(BLOCKSIZE % groupsize == 0, "BLOCKSIZE % groupsize != 0");
        constexpr int groupsPerBlock = BLOCKSIZE / groupsize;
        static_assert(groupsize == 32);

        auto thread = cg::this_thread();
        auto tgroup = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int numGroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;

        const int numPositions = *mismatchPositions.numPositions;

        for(int p = groupId; p < numPositions; p += numGroups){
            const int anchorIndex = mismatchPositions.anchorIndex[p];
            const int candidateIndex = mismatchPositions.candidateIndex[p];
            const int candidate_length = candidateSequencesLengths[candidateIndex];
            const int shift = shifts[candidateIndex];
            const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);
            const int anchorColumnsBegin_incl = msa.columnProperties->anchorColumnsBegin_incl;
            const int queryColumnsBegin_incl = anchorColumnsBegin_incl + shift;
            const int queryColumnsEnd_excl = anchorColumnsBegin_incl + shift + candidate_length;

            auto minreduce = [&](auto val){
                using T = decltype(val);
                return cg::reduce(tgroup, val, cg::less<T>{});
            };

            auto maxreduce = [&](auto val){
                using T = decltype(val);
                return cg::reduce(tgroup, val, cg::greater<T>{});
            };

            auto sumreduce = [&](auto val){
                using T = decltype(val);
                return cg::reduce(tgroup, val, cg::plus<T>{});
            };

            GpuMSAProperties msaProperties = msa.getMSAProperties(
                tgroup, sumreduce, minreduce, minreduce, maxreduce,
                queryColumnsBegin_incl,
                queryColumnsEnd_excl
            );

            if(tgroup.thread_rank() == 0){
                msaPropertiesPerPosition[p] = msaProperties;
            }
        }        
    }


    template<class CandidateExtractor>
    __global__
    void msaCorrectCandidatesWithForestKernel_multiphase_extractKernel(
        float* __restrict__ featuresTransposed,
        GPUMultiMSA multiMSA,
        float estimatedCoverage,
        const int* __restrict__ shifts,
        const int* __restrict__ candidateSequencesLengths,
        MismatchPositionsRaw mismatchPositions,
        const GpuMSAProperties* __restrict__ msaPropertiesPerPosition
    ){

        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        const int numPositions = *mismatchPositions.numPositions;

        for(int p = tid; p < numPositions; p += stride){
            const int anchorIndex = mismatchPositions.anchorIndex[p];
            const int candidateIndex = mismatchPositions.candidateIndex[p];
            const int candidate_length = candidateSequencesLengths[candidateIndex];
            const int shift = shifts[candidateIndex];
            const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);
            const int anchorColumnsBegin_incl = msa.columnProperties->anchorColumnsBegin_incl;
            const int anchorColumnsEnd_excl = anchorColumnsBegin_incl + candidate_length;
            const int queryColumnsBegin_incl = anchorColumnsBegin_incl + shift;
            const int queryColumnsEnd_excl = anchorColumnsBegin_incl + shift + candidate_length;

            const int msaPos = queryColumnsBegin_incl + mismatchPositions.position[p];
            ExtractCandidateInputData extractorInput;

            extractorInput.origBase = mismatchPositions.origBase[p];
            extractorInput.consensusBase = mismatchPositions.consensusBase[p];
            extractorInput.estimatedCoverage = estimatedCoverage;
            extractorInput.msaPos = msaPos;
            extractorInput.anchorColumnsBegin_incl = anchorColumnsBegin_incl;
            extractorInput.anchorColumnsEnd_excl = anchorColumnsEnd_excl;
            extractorInput.queryColumnsBegin_incl = queryColumnsBegin_incl;
            extractorInput.queryColumnsEnd_excl = queryColumnsEnd_excl;
            extractorInput.msaProperties = msaPropertiesPerPosition[p];
            extractorInput.msa = msa;

            StridedIterator myFeaturesBegin{featuresTransposed + p, numPositions};

            CandidateExtractor extractFeatures{};
            extractFeatures(myFeaturesBegin, extractorInput);
        }       
    }

    template<int BLOCKSIZE, int groupsize, class GpuClf>
    __global__
    void msaCorrectCandidatesWithForestKernel_multiphase_correctKernelGroup(
        char* __restrict__ correctedCandidates,
        GPUMultiMSA multiMSA,
        GpuClf gpuForest,
        float forestThreshold,
        const AlignmentOrientation* __restrict__ bestAlignmentFlags,
        const int* __restrict__ candidateSequencesLengths,     
        size_t decodedSequencePitchInBytes,
        MismatchPositionsRaw mismatchPositions,
        const float* __restrict__ featuresTransposed,
        bool useGlobalInsteadOfSmem,
        int numFeatures
    ){

        /*
            Use groupsize threads per candidate to perform correction
        */
        static_assert(BLOCKSIZE % groupsize == 0, "BLOCKSIZE % groupsize != 0");
        constexpr int groupsPerBlock = BLOCKSIZE / groupsize;
        static_assert(groupsize == 32);

        auto thread = cg::this_thread();
        auto tgroup = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int numGroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int groupIdInBlock = threadIdx.x / groupsize;

        extern __shared__ float smemFeatures[];

        float* myGroupFeatures = &smemFeatures[0] + groupIdInBlock * numFeatures;

        auto sumreduce = [&](auto val){
            using T = decltype(val);
            return cg::reduce(tgroup, val, cg::plus<T>{});
        };

        const int numPositions = *mismatchPositions.numPositions;

        for(int p = groupId; p < numPositions; p += numGroups){
            const int candidateIndex = mismatchPositions.candidateIndex[p];
            const int destinationIndex = mismatchPositions.destinationIndex[p];
            const int position = mismatchPositions.position[p];
            const int origBase = mismatchPositions.origBase[p];
            const int candidate_length = candidateSequencesLengths[candidateIndex];
            const AlignmentOrientation bestAlignmentFlag = bestAlignmentFlags[candidateIndex];

            bool useConsensus = false;
            StridedIterator myFeaturesBegin{featuresTransposed + p, numPositions};

            if(useGlobalInsteadOfSmem){
                useConsensus = gpuForest.decide(tgroup, myFeaturesBegin, forestThreshold, sumreduce);
            }else{
                //noncoalesced access
                for(int f = tgroup.thread_rank(); f < numFeatures; f += tgroup.size()){
                    myGroupFeatures[f] = *(myFeaturesBegin + f);
                }
                tgroup.sync();
                useConsensus = gpuForest.decide(tgroup, myGroupFeatures, forestThreshold, sumreduce);
            }

            if(!useConsensus){
                if(tgroup.thread_rank() == 0){
                    char* const myOutput = correctedCandidates + destinationIndex * decodedSequencePitchInBytes;
                    const int outputPos = (bestAlignmentFlag == AlignmentOrientation::Forward ? position : candidate_length - 1 - position);
                    const char outputBase = (bestAlignmentFlag == AlignmentOrientation::Forward ? origBase : SequenceHelpers::complementBaseDecoded(origBase));
                    myOutput[outputPos] = outputBase;
                }
            }
        }        
    }

    template<int BLOCKSIZE, class GpuClf>
    __global__
    void msaCorrectCandidatesWithForestKernel_multiphase_correctKernelThread(
        char* __restrict__ correctedCandidates,
        GPUMultiMSA multiMSA,
        GpuClf gpuForest,
        float forestThreshold,
        const AlignmentOrientation* __restrict__ bestAlignmentFlags,
        const int* __restrict__ candidateSequencesLengths,     
        size_t decodedSequencePitchInBytes,
        MismatchPositionsRaw mismatchPositions,
        const float* __restrict__ featuresTransposed,
        bool useGlobalInsteadOfSmem,
        int numFeatures
    ){

        auto tgroup = cg::this_thread();
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        extern __shared__ float smemFeatures[];

        StridedIterator mySmemFeaturesTransposed(&smemFeatures[0] + threadIdx.x, BLOCKSIZE);

        auto sumreduce = [&](auto val){
            using T = decltype(val);
            return cg::reduce(tgroup, val, cg::plus<T>{});
        };

        const int numPositions = *mismatchPositions.numPositions;

        for(int p = tid; p < numPositions; p += stride){
            const int candidateIndex = mismatchPositions.candidateIndex[p];
            const int destinationIndex = mismatchPositions.destinationIndex[p];
            const int position = mismatchPositions.position[p];
            const int origBase = mismatchPositions.origBase[p];
            const int candidate_length = candidateSequencesLengths[candidateIndex];
            const AlignmentOrientation bestAlignmentFlag = bestAlignmentFlags[candidateIndex];

            bool useConsensus = false;
            StridedIterator myFeaturesBegin{featuresTransposed + p, numPositions};

            if(useGlobalInsteadOfSmem){
                useConsensus = gpuForest.decide(tgroup, myFeaturesBegin, forestThreshold, sumreduce);
            }else{
                for(int f = 0; f < numFeatures; f += 1){
                    *(mySmemFeaturesTransposed + f) = *(myFeaturesBegin + f);
                }
                useConsensus = gpuForest.decide(tgroup, mySmemFeaturesTransposed, forestThreshold, sumreduce);
            }

            if(!useConsensus){
                char* const myOutput = correctedCandidates + destinationIndex * decodedSequencePitchInBytes;
                const int outputPos = (bestAlignmentFlag == AlignmentOrientation::Forward ? position : candidate_length - 1 - position);
                const char outputBase = (bestAlignmentFlag == AlignmentOrientation::Forward ? origBase : SequenceHelpers::complementBaseDecoded(origBase));
                myOutput[outputPos] = outputBase;
            }
        }   

        
    }



    /*
        Pass 1: Set corrected candidate to msa consensus
        Pass 2 (this kernel): Use RF to change some positions of corrected candidates
    */
    template<int BLOCKSIZE, int groupsize, class CandidateExtractor, class GpuClf>
    __global__
    void msaCorrectCandidatesWithForestKernel_multiphase_comparemsapropsextractcorrectKernel(
        char* __restrict__ correctedCandidates,
        GPUMultiMSA multiMSA,
        GpuClf gpuForest,
        float forestThreshold,
        float estimatedCoverage,
        const int* __restrict__ shifts,
        const AlignmentOrientation* __restrict__ bestAlignmentFlags,
        const unsigned int* __restrict__ candidateSequencesData,
        const int* __restrict__ candidateSequencesLengths,
        const bool* __restrict__ d_candidateContainsN,
        const int* __restrict__ candidateIndicesOfCandidatesToBeCorrected,
        const int* __restrict__ numCandidatesToBeCorrected,
        const int* __restrict__ anchorIndicesOfCandidates,         
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes
    ){

        /*
            Use groupsize threads per candidate to perform correction
        */
        static_assert(BLOCKSIZE % groupsize == 0, "BLOCKSIZE % groupsize != 0");
        constexpr int groupsPerBlock = BLOCKSIZE / groupsize;
        static_assert(groupsize == 32);

        auto thread = cg::this_thread();
        auto tgroup = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int numGroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int groupIdInBlock = threadIdx.x / groupsize;

        struct WorkPos{
            char origBase;
            char consensusBase;
            int anchorIndex;
            int candidateIndex;
            int destinationIndex;
            int position;            
        };

        constexpr int maxNumMismatchLocations = 128;

        //__shared__ WorkPos mismatchLocations[groupsPerBlock][maxNumMismatchLocations];
        __shared__ char mismatchLocations_origBase[groupsPerBlock][maxNumMismatchLocations];
        __shared__ char mismatchLocations_consensusBase[groupsPerBlock][maxNumMismatchLocations];
        __shared__ int mismatchLocations_anchorIndex[groupsPerBlock][maxNumMismatchLocations];
        __shared__ int mismatchLocations_candidateIndex[groupsPerBlock][maxNumMismatchLocations];
        __shared__ int mismatchLocations_destinationIndex[groupsPerBlock][maxNumMismatchLocations];
        __shared__ int mismatchLocations_position[groupsPerBlock][maxNumMismatchLocations];
        __shared__ int numMismatchLocations[groupsPerBlock];

        auto processMismatchLocations = [&](){
            const int numLocations = std::min(numMismatchLocations[groupIdInBlock], maxNumMismatchLocations);

            for(int l = tgroup.thread_rank(); l < numLocations; l += tgroup.size()){
                #if 0
                const WorkPos workPos = mismatchLocations[groupIdInBlock][l];

                const int candidateIndex = workPos.candidateIndex;
                const int anchorIndex = workPos.anchorIndex;
                const int destinationIndex = workPos.destinationIndex;
                const char origBase = workPos.origBase;
                const char consensusBase = workPos.consensusBase;
                const int position = workPos.position;
                #else
                const int candidateIndex = mismatchLocations_candidateIndex[groupIdInBlock][l];
                const int anchorIndex = mismatchLocations_anchorIndex[groupIdInBlock][l];
                const int destinationIndex = mismatchLocations_destinationIndex[groupIdInBlock][l];
                const char origBase = mismatchLocations_origBase[groupIdInBlock][l];
                const char consensusBase = mismatchLocations_consensusBase[groupIdInBlock][l];
                const int position = mismatchLocations_position[groupIdInBlock][l];
                #endif

                const int candidate_length = candidateSequencesLengths[candidateIndex];
                const AlignmentOrientation bestAlignmentFlag = bestAlignmentFlags[candidateIndex];

                const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);

                const int shift = shifts[candidateIndex];
                const int anchorColumnsBegin_incl = msa.columnProperties->anchorColumnsBegin_incl;
                const int anchorColumnsEnd_excl = msa.columnProperties->anchorColumnsEnd_excl;
                const int queryColumnsBegin_incl = anchorColumnsBegin_incl + shift;
                const int queryColumnsEnd_excl = anchorColumnsBegin_incl + shift + candidate_length;

                auto i_f = thrust::identity<float>{};
                auto i_i = thrust::identity<int>{};

                GpuMSAProperties msaProperties = msa.getMSAProperties(
                    thread, i_f, i_f, i_i, i_i,
                    queryColumnsBegin_incl,
                    queryColumnsEnd_excl
                );

                const int msaPos = queryColumnsBegin_incl + position;
                ExtractCandidateInputData extractorInput;

                extractorInput.origBase = origBase;
                extractorInput.consensusBase = consensusBase;
                extractorInput.estimatedCoverage = estimatedCoverage;
                extractorInput.msaPos = msaPos;
                extractorInput.anchorColumnsBegin_incl = anchorColumnsBegin_incl;
                extractorInput.anchorColumnsEnd_excl = anchorColumnsEnd_excl;
                extractorInput.queryColumnsBegin_incl = queryColumnsBegin_incl;
                extractorInput.queryColumnsEnd_excl = queryColumnsEnd_excl;
                extractorInput.msaProperties = msaProperties;
                extractorInput.msa = msa;

                float features[CandidateExtractor::numFeatures()];
                CandidateExtractor extractFeatures{};
                extractFeatures(&features[0], extractorInput);

                const bool useConsensus = gpuForest.decide(&features[0], forestThreshold);

                if(!useConsensus){
                    char* const myOutput = correctedCandidates + destinationIndex * decodedSequencePitchInBytes;
                    const int outputPos = (bestAlignmentFlag == AlignmentOrientation::Forward ? position : candidate_length - 1 - position);
                    const char outputBase = (bestAlignmentFlag == AlignmentOrientation::Forward ? origBase : SequenceHelpers::complementBaseDecoded(origBase));
                    myOutput[outputPos] = outputBase;
                }
            }
        };

        if(tgroup.thread_rank() == 0){
            numMismatchLocations[groupIdInBlock] = 0;
        }
        tgroup.sync();
       
        const int loopEnd = *numCandidatesToBeCorrected;

        const GpuClf localForest = gpuForest;

        int id = groupId;
        int startposition = 0;

        while(id < loopEnd){

            const int candidateIndex = candidateIndicesOfCandidatesToBeCorrected[id];
            const int anchorIndex = anchorIndicesOfCandidates[candidateIndex];
            const int destinationIndex = id;

            const int candidate_length = candidateSequencesLengths[candidateIndex];
            const AlignmentOrientation bestAlignmentFlag = bestAlignmentFlags[candidateIndex];

            const unsigned int* const encUncorrectedCandidate = candidateSequencesData 
                        + std::size_t(candidateIndex) * encodedSequencePitchInInts;
            char* decCorrectedCandidate = correctedCandidates + destinationIndex * decodedSequencePitchInBytes;

            bool exceededCapacity = false;

            const int loopEnd = SDIV(candidate_length, tgroup.size()) * tgroup.size();
            for(int i = tgroup.thread_rank(); i < loopEnd; i += tgroup.size()){
                if(i < candidate_length){
                    std::uint8_t origEncodedBase = 0;
                    char consensusBase = 'F';

                    if(bestAlignmentFlag == AlignmentOrientation::ReverseComplement){
                        origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(
                            encUncorrectedCandidate,
                            candidate_length,
                            candidate_length - i - 1
                        );
                        origEncodedBase = SequenceHelpers::complementBase2Bit(origEncodedBase);
                        consensusBase = SequenceHelpers::complementBaseDecoded(decCorrectedCandidate[candidate_length - i - 1]);
                    }else{
                        origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(
                            encUncorrectedCandidate,
                            candidate_length,
                            i
                        );
                        consensusBase = decCorrectedCandidate[i];
                    }

                    const char origBase = SequenceHelpers::decodeBase(origEncodedBase);
                    //const char consensusBase = decCorrectedCandidate[i];

                    // if(id == 1){
                    //     for(int k = 0; k < candidate_length; k++){
                    //         if(i == k){
                    //             printf("orig cons position %d, %c %c\n", i, origBase, consensusBase);
                    //         }
                    //         tgroup.sync();
                    //     }
                    // }

                    if(origBase != consensusBase){
                        // if(id == 1){
                        //     printf("orig cons mismatch position %d, %c %c\n", i, origBase, consensusBase);
                        // }

                        // WorkPos workPos;
                        // workPos.anchorIndex = anchorIndex;
                        // workPos.candidateIndex = candidateIndex;
                        // workPos.destinationIndex = destinationIndex;
                        // workPos.position = i;
                        // workPos.origBase = origBase;
                        // workPos.consensusBase = consensusBase;

                        int smemarraypos = atomicAdd(numMismatchLocations + groupIdInBlock, 1);
                        if(smemarraypos < maxNumMismatchLocations){
                            //printf("%d %d %d\n", blockIdx.x, groupId, smemarraypos);
                            //mismatchLocations[groupIdInBlock][smemarraypos] = workPos;
                            mismatchLocations_anchorIndex[groupIdInBlock][smemarraypos] = anchorIndex;
                            mismatchLocations_candidateIndex[groupIdInBlock][smemarraypos] = candidateIndex;
                            mismatchLocations_destinationIndex[groupIdInBlock][smemarraypos] = destinationIndex;
                            mismatchLocations_position[groupIdInBlock][smemarraypos] = i;
                            mismatchLocations_origBase[groupIdInBlock][smemarraypos] = origBase;
                            mismatchLocations_consensusBase[groupIdInBlock][smemarraypos] = consensusBase;
                        }
                    }
                }
                tgroup.sync();

                if(numMismatchLocations[groupIdInBlock] > maxNumMismatchLocations){
                    exceededCapacity = true;
                    break;
                }
            }
            
            //if work positions are full, or if last iteration, classify all work positions using RF
            if((numMismatchLocations[groupIdInBlock] >= maxNumMismatchLocations) || (id + numGroups >= loopEnd)){
                processMismatchLocations();
                tgroup.sync();

                if(tgroup.thread_rank() == 0){
                    numMismatchLocations[groupIdInBlock] = 0;
                }
                tgroup.sync();
            }

            //if array capacity was exceeded, at least one nucleotide of the current id loop iteration could not be processed
            //in that case, the loop variable is not incremented to repeat this id.
            if(!exceededCapacity){
                id += numGroups;
            }
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

        const int& anchorColumnsBegin_incl = columnProperties.anchorColumnsBegin_incl;
        const int& anchorColumnsEnd_excl = columnProperties.anchorColumnsEnd_excl;
        const int& lastColumn_excl = columnProperties.lastColumn_excl;

        const int shift = alignmentShift;
        const int candidate_length = candidateLength;
        const int queryColumnsBegin_incl = anchorColumnsBegin_incl + shift;
        const int queryColumnsEnd_excl = anchorColumnsBegin_incl + shift + candidate_length;

        if(anchorColumnsBegin_incl - new_columns_to_correct <= queryColumnsBegin_incl
           && queryColumnsBegin_incl <= anchorColumnsBegin_incl + new_columns_to_correct
           && queryColumnsEnd_excl <= anchorColumnsEnd_excl + new_columns_to_correct) {

            float newColMinSupport = 1.0f;
            int newColMinCov = std::numeric_limits<int>::max();
            //check new columns left of anchor
            for(int columnindex = anchorColumnsBegin_incl - new_columns_to_correct;
                columnindex < anchorColumnsBegin_incl;
                columnindex++) {

                assert(columnindex < lastColumn_excl);
                if(queryColumnsBegin_incl <= columnindex) {
                    newColMinSupport = msa.support[columnindex] < newColMinSupport ? msa.support[columnindex] : newColMinSupport;
                    newColMinCov = msa.coverages[columnindex] < newColMinCov ? msa.coverages[columnindex] : newColMinCov;
                }
            }
            //check new columns right of anchor
            for(int columnindex = anchorColumnsEnd_excl;
                    columnindex < anchorColumnsEnd_excl + new_columns_to_correct
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
        const int* __restrict__ numCandidatesPerAnchorPrefixsum,
        const int* __restrict__ localGoodCandidateIndices,
        const int* __restrict__ numLocalGoodCandidateIndicesPerAnchor,
        const int* __restrict__ d_numAnchors,
        const int* __restrict__ d_numCandidates,
        float min_support_threshold,
        float min_coverage_threshold,
        int new_columns_to_correct
    ){

        __shared__ int numAgg;

        const int n_anchors = *d_numAnchors;

        for(int anchorIndex = blockIdx.x; anchorIndex < n_anchors; anchorIndex += gridDim.x){

            if(threadIdx.x == 0){
                numAgg = 0;
            }
            __syncthreads();

            const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);

            const bool isHighQualityAnchor = hqflags[anchorIndex].hq();
            const int numGoodIndices = numLocalGoodCandidateIndicesPerAnchor[anchorIndex];
            const int dataoffset = numCandidatesPerAnchorPrefixsum[anchorIndex];
            const int* myGoodIndices = localGoodCandidateIndices + dataoffset;

            if(isHighQualityAnchor){

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
        const int* __restrict__ numCandidatesPerAnchorPrefixsum,
        const int* __restrict__ localGoodCandidateIndices,
        const int* __restrict__ numLocalGoodCandidateIndicesPerAnchor,
        const int* __restrict__ d_numAnchors,
        const int* __restrict__ d_numCandidates,
        float min_support_threshold,
        float min_coverage_threshold,
        int new_columns_to_correct
    ){

        __shared__ int numAgg;

        const int n_anchors = *d_numAnchors;

        for(int anchorIndex = blockIdx.x; 
                anchorIndex < n_anchors; 
                anchorIndex += gridDim.x){

            if(threadIdx.x == 0){
                numAgg = 0;
            }
            __syncthreads();

            const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);

            const bool isHighQualityAnchor = hqflags[anchorIndex].hq();
            const int numGoodIndices = numLocalGoodCandidateIndicesPerAnchor[anchorIndex];
            const int dataoffset = numCandidatesPerAnchorPrefixsum[anchorIndex];
            const int* myGoodIndices = localGoodCandidateIndices + dataoffset;

            if(isHighQualityAnchor){

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
    void msaCorrectCandidatesAndComputeEditsKernel(
        char* __restrict__ correctedCandidates,
        EncodedCorrectionEdit* __restrict__ d_editsPerCorrectedCandidate,
        int* __restrict__ d_numEditsPerCorrectedCandidate,
        GPUMultiMSA multiMSA,
        const int* __restrict__ shifts,
        const AlignmentOrientation* __restrict__ bestAlignmentFlags,
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
            const int anchorIndex = anchorIndicesOfCandidates[candidateIndex];
            const int destinationIndex = id;

            const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);

            char* const my_corrected_candidate = correctedCandidates + destinationIndex * decodedSequencePitchInBytes;
            const int candidate_length = candidateSequencesLengths[candidateIndex];

            const int shift = shifts[candidateIndex];
            const int anchorColumnsBegin_incl = msa.columnProperties->anchorColumnsBegin_incl;
            const int queryColumnsBegin_incl = anchorColumnsBegin_incl + shift;
            const int queryColumnsEnd_excl = anchorColumnsBegin_incl + shift + candidate_length;

            const AlignmentOrientation bestAlignmentFlag = bestAlignmentFlags[candidateIndex];

            if(tgroup.thread_rank() == 0){                        
                shared_numEditsOfCandidate[groupIdInBlock] = 0;
            }
            tgroup.sync();          

            const int copyposbegin = queryColumnsBegin_incl;
            const int copyposend = queryColumnsEnd_excl;

            //the forward strand will be returned -> make reverse complement again
            if(bestAlignmentFlag == AlignmentOrientation::ReverseComplement) {
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


    template<int BLOCKSIZE, int groupsize>
    __global__
    void msaCorrectCandidatesKernel(
        char* __restrict__ correctedCandidates,
        GPUMultiMSA multiMSA,
        const int* __restrict__ shifts,
        const AlignmentOrientation* __restrict__ bestAlignmentFlags,
        const unsigned int* __restrict__ candidateSequencesData,
        const int* __restrict__ candidateSequencesLengths,
        const bool* __restrict__ d_candidateContainsN,
        const int* __restrict__ candidateIndicesOfCandidatesToBeCorrected,
        const int* __restrict__ numCandidatesToBeCorrected,
        const int* __restrict__ anchorIndicesOfCandidates,
        const int* __restrict__ d_numAnchors,
        const int* __restrict__ d_numCandidates,         
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        size_t dynamicsmemSequencePitchInInts
    ){

        /*
            Use groupsize threads per candidate to perform correction
        */
        static_assert(BLOCKSIZE % groupsize == 0, "BLOCKSIZE % groupsize != 0");

        extern __shared__ int dynamicsmem[]; // for sequences

        auto tgroup = cg::tiled_partition<groupsize>(cg::this_thread_block());

        const int numGroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int groupIdInBlock = threadIdx.x / groupsize;

        char* const shared_correctedCandidate = (char*)(dynamicsmem + dynamicsmemSequencePitchInInts * groupIdInBlock);

        const int loopEnd = *numCandidatesToBeCorrected;

        for(int id = groupId; id < loopEnd; id += numGroups){

            const int candidateIndex = candidateIndicesOfCandidatesToBeCorrected[id];
            const int anchorIndex = anchorIndicesOfCandidates[candidateIndex];
            const int destinationIndex = id;

            const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);

            char* const my_corrected_candidate = correctedCandidates + destinationIndex * decodedSequencePitchInBytes;
            const int candidate_length = candidateSequencesLengths[candidateIndex];

            const int shift = shifts[candidateIndex];
            const int anchorColumnsBegin_incl = msa.columnProperties->anchorColumnsBegin_incl;
            const int queryColumnsBegin_incl = anchorColumnsBegin_incl + shift;
            const int queryColumnsEnd_excl = anchorColumnsBegin_incl + shift + candidate_length;

            const AlignmentOrientation bestAlignmentFlag = bestAlignmentFlags[candidateIndex];       

            const int copyposbegin = queryColumnsBegin_incl;
            const int copyposend = queryColumnsEnd_excl;

            //the forward strand will be returned -> make reverse complement again
            if(bestAlignmentFlag == AlignmentOrientation::ReverseComplement) {
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
        char* d_correctedAnchors,
        bool* d_anchorIsCorrected,
        AnchorHighQualityFlag* d_isHighQualityAnchor,
        GPUMultiMSA multiMSA,
        const unsigned int* d_anchorSequencesData,
        const int* d_indices_per_anchor,
        const int* d_numAnchors,
        int /*maxNumAnchors*/,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        float estimatedErrorrate,
        float avg_support_threshold,
        float min_support_threshold,
        float min_coverage_threshold,
        int maximum_sequence_length,
        cudaStream_t stream
    ){

        const int max_block_size = 256;
        const int blocksize = std::min(max_block_size, SDIV(maximum_sequence_length, 32) * 32);
        const std::size_t smem = 0;

        int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            msaCorrectAnchorsKernel,
            blocksize, 
            smem
        ));

        const int maxBlocks = maxBlocksPerSM * numSMs;

        dim3 block(blocksize);
        dim3 grid(maxBlocks);        

        msaCorrectAnchorsKernel<<<grid, block, 0, stream>>>( 
            d_correctedAnchors, 
            d_anchorIsCorrected, 
            d_isHighQualityAnchor, 
            multiMSA, 
            d_anchorSequencesData, 
            d_indices_per_anchor, 
            d_numAnchors, 
            encodedSequencePitchInInts, 
            decodedSequencePitchInBytes, 
            estimatedErrorrate, 
            avg_support_threshold, 
            min_support_threshold, 
            min_coverage_threshold 
        ); CUDACHECKASYNC;

    }


    void callMsaCorrectAnchorsWithForestKernel_multiphase(
        char* d_correctedAnchors,
        bool* d_anchorIsCorrected,
        AnchorHighQualityFlag* d_isHighQualityAnchor,
        GPUMultiMSA multiMSA,
        GpuForest::Clf gpuForest,
        float forestThreshold,
        const unsigned int* d_anchorSequencesData,
        const int* d_indices_per_anchor,
        const int numAnchors,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        int maximumSequenceLength,
        float estimatedErrorrate,
        float estimatedCoverage,
        float avg_support_threshold,
        float min_support_threshold,
        float min_coverage_threshold,
        cudaStream_t stream
    ){
        if(numAnchors == 0) return;

        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();

        int deviceId = 0;
        int numSMs = 0;

        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));


        constexpr int blocksizeinit = 128;
        constexpr int groupsizeinit = 32;

        const std::size_t smeminit = 0;        
        int maxBlocksPerSMinit = 0;

        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSMinit,
            msaCorrectAnchorsWithForestKernel_multiphase_initkernel<blocksizeinit, groupsizeinit>,
            blocksizeinit, 
            smeminit
        ));

        rmm::device_scalar<int> d_numMismatches(0, stream, mr);
        rmm::device_uvector<GpuMSAProperties> d_msaProperties(numAnchors, stream, mr);

        dim3 blockinit(blocksizeinit);
        dim3 gridinit(std::min(SDIV(numAnchors, blocksizeinit / groupsizeinit), maxBlocksPerSMinit * numSMs));

        //helpers::GpuTimer timerinit(stream, "initkernel");

        msaCorrectAnchorsWithForestKernel_multiphase_initkernel<blocksizeinit, groupsizeinit>
        <<<gridinit, blockinit, smeminit, stream>>>(
            d_correctedAnchors,
            d_anchorIsCorrected,
            d_isHighQualityAnchor,
            multiMSA,
            d_anchorSequencesData,
            d_indices_per_anchor,
            numAnchors,
            encodedSequencePitchInInts,
            decodedSequencePitchInBytes,
            estimatedErrorrate,
            estimatedCoverage,
            avg_support_threshold,
            min_support_threshold,
            min_coverage_threshold,
            d_msaProperties.data(),
            d_numMismatches.data()
        );
        CUDACHECKASYNC;
        //timerinit.print();

        const int numMismatches = d_numMismatches.value(stream);
        //std::cerr << "numMismatches: " << numMismatches  << "\n";

        constexpr int blocksizegather = 128;
        constexpr int groupsizegather = 32;

        const std::size_t smemgather = 0;        
        int maxBlocksPerSMgather = 0;

        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSMgather,
            msaCorrectAnchorsWithForestKernel_multiphase_initkernel<blocksizegather, groupsizegather>,
            blocksizegather, 
            smemgather
        ));

        d_numMismatches.set_value_to_zero_async(stream);
        rmm::device_uvector<int> d_mismatchAnchorIndices(numMismatches, stream, mr);
        rmm::device_uvector<int> d_mismatchPositionsInAnchors(numMismatches, stream, mr);

        dim3 blockgather(blocksizegather);
        dim3 gridgather(std::min(SDIV(numAnchors, blocksizegather / groupsizegather), maxBlocksPerSMgather * numSMs));

        //helpers::GpuTimer timergather(stream, "gatherkernel");        

        msaCorrectAnchorsWithForestKernel_multiphase_gathermismatcheskernel<blocksizegather, groupsizegather>
        <<<gridgather, blockgather, smemgather, stream>>>(
            d_isHighQualityAnchor,
            multiMSA,
            d_anchorSequencesData,
            numAnchors,
            encodedSequencePitchInInts,
            decodedSequencePitchInBytes,
            estimatedErrorrate,
            estimatedCoverage,
            avg_support_threshold,
            min_support_threshold,
            min_coverage_threshold,
            d_numMismatches.data(),
            d_mismatchAnchorIndices.data(),
            d_mismatchPositionsInAnchors.data()
        );
        CUDACHECKASYNC;
        //timergather.print();



        constexpr int blocksizeextract = 128;
        constexpr std::size_t maxSmemextract = 32 * 1024;
        std::size_t featuresizeextract = numMismatches * anchor_extractor::numFeatures() * sizeof(float);

        int maxBlocksPerSMextract = 0;
        std::size_t smemextract = 0;

        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSMextract,
            msaCorrectAnchorsWithForestKernel_multiphase_extractkernel<blocksizeextract, anchor_extractor>,
            blocksizeextract, 
            smemextract
        ));

        rmm::device_buffer gmemFeaturesTransposedExtract(featuresizeextract, stream, mr);

        dim3 blockextract(blocksizeextract);
        dim3 gridextract(std::min(SDIV(numMismatches, blocksizeextract), maxBlocksPerSMextract * numSMs));

        //helpers::GpuTimer timerextract(stream, "extractkernel");

        msaCorrectAnchorsWithForestKernel_multiphase_extractkernel<blocksizeextract, anchor_extractor>
        <<<gridextract, blockextract, smemextract, stream>>>(
            multiMSA,
            d_anchorSequencesData,
            encodedSequencePitchInInts,
            decodedSequencePitchInBytes,
            estimatedErrorrate,
            estimatedCoverage,
            avg_support_threshold,
            min_support_threshold,
            min_coverage_threshold,
            reinterpret_cast<float*>(gmemFeaturesTransposedExtract.data()),
            numMismatches,
            d_mismatchAnchorIndices.data(),
            d_mismatchPositionsInAnchors.data(),
            d_msaProperties.data()
        );
        CUDACHECKASYNC;
        //timerextract.print();


        // constexpr int blocksizecorrect = 128;
        // constexpr int maxSmemCorrect = 32 * 1024;
        // std::size_t smemcorrect = blocksizecorrect * anchor_extractor::numFeatures() * sizeof(float);
        // if(maxSmemCorrect < smemcorrect){
        //     smemcorrect = 0;
        // }
        // int maxBlocksPerSMcorrect = 0;

        // CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        //     &maxBlocksPerSMcorrect,
        //     msaCorrectAnchorsWithForestKernel_multiphase_correctkernel_thread<blocksizecorrect, GpuForest::Clf>,
        //     blocksizecorrect, 
        //     smemcorrect
        // ));

        // dim3 blockcorrect(blocksizecorrect);
        // dim3 gridcorrect(std::min(SDIV(numMismatches, blocksizecorrect), maxBlocksPerSMcorrect * numSMs));

        // //helpers::GpuTimer timercorrectthread(stream, "correctkernel");

        // msaCorrectAnchorsWithForestKernel_multiphase_correctkernel_thread<blocksizecorrect, GpuForest::Clf>
        // <<<gridcorrect, blockcorrect, smemcorrect, stream>>>(
        //     d_correctedAnchors,
        //     multiMSA,
        //     gpuForest,
        //     forestThreshold,
        //     d_anchorSequencesData,
        //     encodedSequencePitchInInts,
        //     decodedSequencePitchInBytes,
        //     estimatedErrorrate,
        //     estimatedCoverage,
        //     avg_support_threshold,
        //     min_support_threshold,
        //     min_coverage_threshold,
        //     anchor_extractor::numFeatures(),
        //     smemcorrect == 0,
        //     reinterpret_cast<float*>(gmemFeaturesTransposedExtract.data()),
        //     numMismatches,
        //     d_mismatchAnchorIndices.data(),
        //     d_mismatchPositionsInAnchors.data()
        // );
        // CUDACHECKASYNC;
        //timercorrectthread.print();

        #if 1
        constexpr int blocksizecorrectgroup = 128;
        constexpr int groupsizecorrectgroup = 32;
        constexpr int maxSmemCorrectgroup = 32 * 1024;
        std::size_t smemcorrectgroup = blocksizecorrectgroup * anchor_extractor::numFeatures() * sizeof(float);
        if(maxSmemCorrectgroup < smemcorrectgroup){
            smemcorrectgroup = 0;
        }
        int maxBlocksPerSMcorrectgroup = 0;

        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSMcorrectgroup,
            msaCorrectAnchorsWithForestKernel_multiphase_correctkernel_group<blocksizecorrectgroup, groupsizecorrectgroup, GpuForest::Clf>,
            blocksizecorrectgroup, 
            smemcorrectgroup
        ));

        dim3 blockcorrectgroup(blocksizecorrectgroup);
        dim3 gridcorrectgroup(std::min(SDIV(numMismatches, blocksizecorrectgroup / groupsizecorrectgroup), maxBlocksPerSMcorrectgroup * numSMs));

        //helpers::GpuTimer timercorrectgroup(stream, "correctgroupkernel");

        msaCorrectAnchorsWithForestKernel_multiphase_correctkernel_group<blocksizecorrectgroup, groupsizecorrectgroup, GpuForest::Clf>
        <<<gridcorrectgroup, blockcorrectgroup, smemcorrectgroup, stream>>>(
            d_correctedAnchors,
            multiMSA,
            gpuForest,
            forestThreshold,
            d_anchorSequencesData,
            encodedSequencePitchInInts,
            decodedSequencePitchInBytes,
            estimatedErrorrate,
            estimatedCoverage,
            avg_support_threshold,
            min_support_threshold,
            min_coverage_threshold,
            anchor_extractor::numFeatures(),
            smemcorrectgroup == 0,
            reinterpret_cast<float*>(gmemFeaturesTransposedExtract.data()),
            numMismatches,
            d_mismatchAnchorIndices.data(),
            d_mismatchPositionsInAnchors.data()
        );
        CUDACHECKASYNC;
        //timercorrectgroup.print();
        #endif

        #if 0
        constexpr int blocksizecorrectblock = 128;
        constexpr int maxSmemCorrectblock = 32 * 1024;
        std::size_t smemcorrectblock = anchor_extractor::numFeatures() * sizeof(float);
        if(maxSmemCorrectblock < smemcorrectblock){
            smemcorrectblock = 0;
        }
        int maxBlocksPerSMcorrectblock = 0;

        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSMcorrectblock,
            msaCorrectAnchorsWithForestKernel_multiphase_correctkernel_block<blocksizecorrectblock, GpuForest::Clf>,
            blocksizecorrectblock, 
            smemcorrectblock
        ));

        dim3 blockcorrectblock(blocksizecorrectblock);
        dim3 gridcorrectblock(std::min(numMismatches, maxBlocksPerSMcorrectblock * numSMs));

        //helpers::GpuTimer timercorrectblock(stream, "correctblockkernel");

        msaCorrectAnchorsWithForestKernel_multiphase_correctkernel_block<blocksizecorrectblock, GpuForest::Clf>
        <<<gridcorrectblock, blockcorrectblock, smemcorrectblock, stream>>>(
            d_correctedAnchors,
            multiMSA,
            gpuForest,
            forestThreshold,
            d_anchorSequencesData,
            encodedSequencePitchInInts,
            decodedSequencePitchInBytes,
            estimatedErrorrate,
            estimatedCoverage,
            avg_support_threshold,
            min_support_threshold,
            min_coverage_threshold,
            anchor_extractor::numFeatures(),
            smemcorrectblock == 0,
            reinterpret_cast<float*>(gmemFeaturesTransposedExtract.data()),
            numMismatches,
            d_mismatchAnchorIndices.data(),
            d_mismatchPositionsInAnchors.data()
        );
        CUDACHECKASYNC;
        //timercorrectblock.print();
        #endif




        // constexpr int blocksizeextractcorrect = 128;
        // constexpr std::size_t maxSmemextractcorrect = 32 * 1024;
        // std::size_t featuresizeextractcorrect = blocksizeextractcorrect * anchor_extractor::numFeatures() * sizeof(float);

        // std::size_t smemextractcorrect = featuresizeextractcorrect > maxSmemextractcorrect ? 0 : featuresizeextractcorrect;
        // bool useGmemFeatures = featuresizeextractcorrect > maxSmemextractcorrect;

        // int maxBlocksPerSMextractcorrect = 0;

        // CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        //     &maxBlocksPerSMextractcorrect,
        //     msaCorrectAnchorsWithForestKernel_multiphase_extractcorrectkernel<blocksizeextractcorrect, anchor_extractor, GpuForest::Clf>,
        //     blocksizeextractcorrect, 
        //     smemextractcorrect
        // ));

        // rmm::device_buffer gmemFeaturesTransposed(useGmemFeatures ? featuresizeextractcorrect : 0, stream, mr);

        // //helpers::GpuTimer timerextractcorrect(stream, "extractcorrectkernel");

        // dim3 blockextractcorrect(blocksizeextractcorrect);
        // dim3 gridextractcorrect(std::min(SDIV(numMismatches, blocksizeextractcorrect), maxBlocksPerSMextractcorrect * numSMs));

        // msaCorrectAnchorsWithForestKernel_multiphase_extractcorrectkernel<blocksizeextractcorrect, anchor_extractor, GpuForest::Clf>
        // <<<gridextractcorrect, blockextractcorrect, smemextractcorrect, stream>>>(
        //     d_correctedAnchors,
        //     multiMSA,
        //     gpuForest,
        //     forestThreshold,
        //     d_anchorSequencesData,
        //     encodedSequencePitchInInts,
        //     decodedSequencePitchInBytes,
        //     estimatedErrorrate,
        //     estimatedCoverage,
        //     avg_support_threshold,
        //     min_support_threshold,
        //     min_coverage_threshold,
        //     useGmemFeatures,
        //     reinterpret_cast<float*>(gmemFeaturesTransposed.data()),
        //     numMismatches,
        //     d_mismatchAnchorIndices.data(),
        //     d_mismatchPositionsInAnchors.data(),
        //     d_msaProperties.data()
        // );
        // CUDACHECKASYNC
        // //timerextractcorrect.print();

        // constexpr int blocksizefull = 128;
        // constexpr int groupsizefull = 32;
        // constexpr int groupsPerBlockfull = blocksizefull / groupsizefull;

        // const std::size_t smemfull = SDIV(sizeof(char) * decodedSequencePitchInBytes * groupsPerBlockfull, sizeof(int)) * sizeof(int);

        // int maxBlocksPerSMfull = 0;

        // CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        //     &maxBlocksPerSMfull,
        //     msaCorrectAnchorsWithForestKernel2<blocksizefull, groupsizefull, anchor_extractor, GpuForest::Clf>,
        //     blocksizefull, 
        //     smemfull
        // ));

        // dim3 blockfull(blocksizefull);
        // dim3 gridfull(std::min(numAnchors, maxBlocksPerSMfull * numSMs));

        // msaCorrectAnchorsWithForestKernel2<blocksizefull, groupsizefull, anchor_extractor><<<gridfull, blockfull, smemfull, stream>>>(
        //     d_correctedAnchors,
        //     d_anchorIsCorrected,
        //     d_isHighQualityAnchor,
        //     multiMSA,
        //     gpuForest,
        //     forestThreshold,
        //     d_anchorSequencesData,
        //     d_indices_per_anchor,
        //     numAnchors,
        //     encodedSequencePitchInInts,
        //     decodedSequencePitchInBytes,
        //     estimatedErrorrate,
        //     estimatedCoverage,
        //     avg_support_threshold,
        //     min_support_threshold,
        //     min_coverage_threshold
        // );
        // CUDACHECKASYNC;
    }

    void callMsaCorrectAnchorsWithForestKernel(
        char* d_correctedAnchors,
        bool* d_anchorIsCorrected,
        AnchorHighQualityFlag* d_isHighQualityAnchor,
        GPUMultiMSA multiMSA,
        GpuForest::Clf gpuForest,
        float forestThreshold,
        const unsigned int* d_anchorSequencesData,
        const int* d_indices_per_anchor,
        const int numAnchors,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        int maximumSequenceLength,
        float estimatedErrorrate,
        float estimatedCoverage,
        float avg_support_threshold,
        float min_support_threshold,
        float min_coverage_threshold,
        cudaStream_t stream
    ){
        if(numAnchors == 0) return;

        // rmm::device_uvector<char> tmpaaa(numAnchors * decodedSequencePitchInBytes, stream);

        // {
        //     //helpers::GpuTimer timermultiphase(stream, "multiphase");
        //     callMsaCorrectAnchorsWithForestKernel_multiphase(
        //         //tmpaaa.data(),
        //         d_correctedAnchors,
        //         d_anchorIsCorrected,
        //         d_isHighQualityAnchor,
        //         multiMSA,
        //         gpuForest,
        //         forestThreshold,
        //         d_anchorSequencesData,
        //         d_indices_per_anchor,
        //         numAnchors,
        //         encodedSequencePitchInInts,
        //         decodedSequencePitchInBytes,
        //         maximumSequenceLength,
        //         estimatedErrorrate,
        //         estimatedCoverage,
        //         avg_support_threshold,
        //         min_support_threshold,
        //         min_coverage_threshold,
        //         stream
        //     );
        //     //timermultiphase.print();
        // }

        if(1){

            constexpr int blocksize = 128;

            //helpers::GpuTimer timerold(stream, "old");

            const std::size_t smem = SDIV(sizeof(char) * maximumSequenceLength, sizeof(int)) * sizeof(int);

            int deviceId = 0;
            int numSMs = 0;
            int maxBlocksPerSM = 0;
            CUDACHECK(cudaGetDevice(&deviceId));
            CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
            CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxBlocksPerSM,
                msaCorrectAnchorsWithForestKernel<blocksize, anchor_extractor, GpuForest::Clf>,
                blocksize, 
                smem
            ));

            const int maxBlocks = maxBlocksPerSM * numSMs;

            dim3 block(blocksize);
            dim3 grid(std::min(numAnchors, maxBlocks));

            msaCorrectAnchorsWithForestKernel<blocksize, anchor_extractor><<<grid, block, smem, stream>>>(
                d_correctedAnchors,
                d_anchorIsCorrected,
                d_isHighQualityAnchor,
                multiMSA,
                gpuForest,
                forestThreshold,
                d_anchorSequencesData,
                d_indices_per_anchor,
                numAnchors,
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                estimatedErrorrate,
                estimatedCoverage,
                avg_support_threshold,
                min_support_threshold,
                min_coverage_threshold
            );

            //timerold.print();

        }

        // helpers::lambda_kernel<<<1, 1, 0, stream>>>(
        //     [
        //         d_correctedAnchors,
        //         tmpaaa = tmpaaa.data(),
        //         decodedSequencePitchInBytes,
        //         numAnchors,
        //         d_anchorIsCorrected
        //     ] __device__ (){
        //         for(int c = blockIdx.x; c < numAnchors; c += gridDim.x){
        //             if(d_anchorIsCorrected[c]){
        //                 const char* ptrold = d_correctedAnchors + c * decodedSequencePitchInBytes;
        //                 const char* ptrnew = tmpaaa + c * decodedSequencePitchInBytes;
        //                 // if(c == 1){
        //                 //     printf("old: ");
        //                 //     for(int i = threadIdx.x; i < 100; i += blockDim.x){
        //                 //         printf("%c", ptrold[i]);
        //                 //     }
        //                 //     printf("\n");
        //                 //     printf("new: ");
        //                 //     for(int i = threadIdx.x; i < 100; i += blockDim.x){
        //                 //         printf("%c", ptrnew[i]);
        //                 //     }
        //                 //     printf("\n");
        //                 //     __syncthreads();
        //                 //     assert(false);
        //                 // }
        //                 for(int i = threadIdx.x; i < 100; i += blockDim.x){
        //                     if(ptrold[i] != ptrnew[i]){
        //                         printf("%d %d %c %c\n", c, i, ptrold[i], ptrnew[i]);
        //                         assert(false);
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // );
    }

    void callMsaCorrectCandidatesWithForestKernelMultiPhase(
        char* d_correctedCandidates,
        EncodedCorrectionEdit* d_editsPerCorrectedCandidate,
        int* d_numEditsPerCorrectedCandidate,
        GPUMultiMSA multiMSA,
        GpuForest::Clf gpuForest,
        float forestThreshold,
        float estimatedCoverage,
        const int* d_shifts,
        const AlignmentOrientation* d_bestAlignmentFlags,
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
        const read_number* candidateReadIds
    ){
        constexpr int blocksize = 128;
        constexpr int groupsize = 32;
        constexpr int numGroupsPerBlock = blocksize / groupsize;

        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();

        int deviceId = 0;
        int numSMs = 0;

        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));

        int numCandidatesToProcess = 0;
        CUDACHECK(cudaMemcpyAsync(&numCandidatesToProcess, d_numCandidatesToBeCorrected, sizeof(int), D2H, stream));
        CUDACHECK(cudaStreamSynchronize(stream));


        
        int maxBlocksPerSMPass1 = 0;
        const std::size_t dynamicsmemPitchInInts = SDIV(maximum_sequence_length, sizeof(int));
        const std::size_t smemPass1 = numGroupsPerBlock * (sizeof(int) * dynamicsmemPitchInInts);

        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSMPass1,
            msaCorrectCandidatesWithForestKernel_multiphase_initCorrectedCandidatesKernel<blocksize, groupsize>,
            blocksize, 
            smemPass1
        ));

        dim3 block1 = blocksize;
        dim3 grid1 = maxBlocksPerSMPass1 * numSMs;

        helpers::GpuTimer timerinitCorrectedCandidatesKernel(stream, "initCorrectedCandidatesKernel");

        msaCorrectCandidatesWithForestKernel_multiphase_initCorrectedCandidatesKernel<blocksize, groupsize>
        <<<grid1, block1, smemPass1, stream>>>(
            d_correctedCandidates,
            multiMSA,
            d_shifts,
            d_bestAlignmentFlags,
            d_candidateSequencesData,
            d_candidateSequencesLengths,
            d_candidateContainsN,
            d_candidateIndicesOfCandidatesToBeCorrected,
            d_numCandidatesToBeCorrected,
            d_anchorIndicesOfCandidates,         
            encodedSequencePitchInInts,
            decodedSequencePitchInBytes,
            dynamicsmemPitchInInts
        );
        CUDACHECKASYNC;
        //timerinitCorrectedCandidatesKernel.print();


        int maxBlocksPerSMCountMismatches = 0;
        const std::size_t smemCountMismatches = 0;
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSMCountMismatches,
            msaCorrectCandidatesWithForestKernel_multiphase_countMismatchesKernel<blocksize, groupsize>,
            blocksize, 
            smemCountMismatches
        ));

        dim3 blockCountMismatches = blocksize;
        dim3 gridCountMismatches = std::min(maxBlocksPerSMCountMismatches * numSMs, SDIV(numCandidatesToProcess, (blocksize / groupsize)));

        rmm::device_scalar<int> d_numMismatches(0, stream, mr);

        helpers::GpuTimer timercountMismatchesKernel(stream, "countMismatchesKernel");

        msaCorrectCandidatesWithForestKernel_multiphase_countMismatchesKernel<blocksize, groupsize>
            <<<gridCountMismatches, blockCountMismatches, smemCountMismatches, stream>>>(
            d_correctedCandidates,
            d_bestAlignmentFlags,
            d_candidateSequencesData,
            d_candidateSequencesLengths,
            d_candidateIndicesOfCandidatesToBeCorrected,
            d_numCandidatesToBeCorrected,
            d_anchorIndicesOfCandidates,         
            encodedSequencePitchInInts,
            decodedSequencePitchInBytes,
            d_numMismatches.data()
        );
        CUDACHECKASYNC;
        //timercountMismatchesKernel.print();

        const int numMismatches = d_numMismatches.value(stream);
        //std::cerr << "numMismatches cands: " << numMismatches << "\n";

        const std::size_t smemFindMismatches = 0;
        int maxBlocksPerSMFindMismatches = 0;
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSMFindMismatches,
            msaCorrectCandidatesWithForestKernel_multiphase_findMismatchesKernel<blocksize, groupsize>,
            blocksize, 
            smemFindMismatches
        ));

        dim3 blockFindMismatches = blocksize;
        dim3 gridFindMismatches = std::min(maxBlocksPerSMFindMismatches * numSMs, SDIV(numCandidatesToProcess, (blocksize / groupsize)));

        MismatchPositions mismatchPositions(numMismatches, stream, mr);

        helpers::GpuTimer timerfindMismatchesKernel(stream, "findMismatchesKernel");

        msaCorrectCandidatesWithForestKernel_multiphase_findMismatchesKernel<blocksize, groupsize>
            <<<gridFindMismatches, blockFindMismatches, smemFindMismatches, stream>>>(
            d_correctedCandidates,
            d_bestAlignmentFlags,
            d_candidateSequencesData,
            d_candidateSequencesLengths,
            d_candidateIndicesOfCandidatesToBeCorrected,
            d_numCandidatesToBeCorrected,
            d_anchorIndicesOfCandidates,         
            encodedSequencePitchInInts,
            decodedSequencePitchInBytes,
            mismatchPositions
        );
        CUDACHECKASYNC;
        //timerfindMismatchesKernel.print();


        const std::size_t smemMsaProps = 0;
        int maxBlocksPerSMMsaProps = 0;
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSMMsaProps,
            msaCorrectCandidatesWithForestKernel_multiphase_msapropsKernel<blocksize, groupsize>,
            blocksize, 
            smemMsaProps
        ));

        dim3 blockMsaProps = blocksize;
        dim3 gridMsaProps = std::min(maxBlocksPerSMMsaProps * numSMs, SDIV(numMismatches, (blocksize / groupsize)));

        rmm::device_uvector<GpuMSAProperties> d_msaPropertiesPerPosition(numMismatches, stream, mr);

        //CUDACHECK(cudaStreamSynchronize(stream));
        helpers::GpuTimer timermsaprops(stream, "msapropsKernel");

        msaCorrectCandidatesWithForestKernel_multiphase_msapropsKernel<blocksize, groupsize>
            <<<gridMsaProps, blockMsaProps, smemMsaProps, stream>>>(
            multiMSA,
            d_shifts,
            d_candidateSequencesLengths,       
            mismatchPositions,
            d_msaPropertiesPerPosition.data()
        );
        CUDACHECKASYNC;
        //timermsaprops.print();



        const std::size_t smemExtract = 0;
        int maxBlocksPerSMExtract = 0;
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSMExtract,
            msaCorrectCandidatesWithForestKernel_multiphase_extractKernel<cands_extractor>,
            blocksize, 
            smemExtract
        ));

        dim3 blockExtract = blocksize;
        dim3 gridExtract = std::min(maxBlocksPerSMExtract * numSMs, SDIV(numMismatches, blocksize));

        rmm::device_uvector<float> d_featuresTransposed(numMismatches * cands_extractor::numFeatures(), stream, mr);

        helpers::GpuTimer timerextract(stream, "extractKernel");

        msaCorrectCandidatesWithForestKernel_multiphase_extractKernel<cands_extractor>
            <<<gridExtract, blockExtract, smemExtract, stream>>>(
            d_featuresTransposed.data(),
            multiMSA,
            estimatedCoverage,
            d_shifts,
            d_candidateSequencesLengths,  
            mismatchPositions,
            d_msaPropertiesPerPosition.data()
        );
        CUDACHECKASYNC;
        //timerextract.print();

        #if 0
        constexpr int maxSmemCorrect = 32 * 1024;
        const std::size_t blockFeaturesBytesCorrectGroup = sizeof(float) * cands_extractor::numFeatures() * (blocksize / groupsize);
        bool useGlobalInsteadOfSmemCorrectGroup = blockFeaturesBytesCorrectGroup > maxSmemCorrect;
        const std::size_t smemCorrectGroup = useGlobalInsteadOfSmemCorrectGroup ? 0 : blockFeaturesBytesCorrectGroup;

        int maxBlocksPerSMCorrectGroup = 0;
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSMCorrectGroup,
            msaCorrectCandidatesWithForestKernel_multiphase_correctKernelGroup<blocksize, groupsize, GpuForest::Clf>,
            blocksize, 
            smemCorrectGroup
        ));

        dim3 blockCorrectGroup = blocksize;
        dim3 gridCorrectGroup = std::min(maxBlocksPerSMCorrectGroup * numSMs, SDIV(numMismatches, (blocksize / groupsize)));

        helpers::GpuTimer timercorrectGroup(stream, "correctKernelGroup");
        msaCorrectCandidatesWithForestKernel_multiphase_correctKernelGroup<blocksize, groupsize, GpuForest::Clf>
            <<<gridCorrectGroup, blockCorrectGroup, smemCorrectGroup, stream>>>(
            d_correctedCandidates,
            multiMSA,
            gpuForest,
            forestThreshold,
            d_bestAlignmentFlags,
            d_candidateSequencesLengths,     
            decodedSequencePitchInBytes,
            mismatchPositions,
            d_featuresTransposed.data(),
            useGlobalInsteadOfSmemCorrectGroup,
            cands_extractor::numFeatures()
        );
        CUDACHECKASYNC;
        //timercorrectGroup.print();

        #else

        constexpr int maxSmemCorrect = 32 * 1024;

        const std::size_t blockFeaturesBytesCorrectThread = sizeof(float) * cands_extractor::numFeatures() * blocksize;
        bool useGlobalInsteadOfSmemCorrectThread = blockFeaturesBytesCorrectThread > maxSmemCorrect;
        const std::size_t smemCorrectThread = useGlobalInsteadOfSmemCorrectThread ? 0 : blockFeaturesBytesCorrectThread;
        assert(!useGlobalInsteadOfSmemCorrectThread);

        int maxBlocksPerSMCorrectThread = 0;
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSMCorrectThread,
            msaCorrectCandidatesWithForestKernel_multiphase_correctKernelThread<blocksize, GpuForest::Clf>,
            blocksize, 
            smemCorrectThread
        ));

        dim3 blockCorrectThread = blocksize;
        dim3 gridCorrectThread = std::min(maxBlocksPerSMCorrectThread * numSMs, SDIV(numMismatches, blocksize));

        helpers::GpuTimer timercorrectThread(stream, "correctKernelThread");
        msaCorrectCandidatesWithForestKernel_multiphase_correctKernelThread<blocksize, GpuForest::Clf>
            <<<gridCorrectThread, blockCorrectThread, smemCorrectThread, stream>>>(
            d_correctedCandidates,
            multiMSA,
            gpuForest,
            forestThreshold,
            d_bestAlignmentFlags,
            d_candidateSequencesLengths,     
            decodedSequencePitchInBytes,
            mismatchPositions,
            d_featuresTransposed.data(),
            useGlobalInsteadOfSmemCorrectThread,
            cands_extractor::numFeatures()
        );
        CUDACHECKASYNC;
        //timercorrectThread.print();

        #endif

        // const std::size_t smemPass2 = 0;
        // int maxBlocksPerSMPass2 = 0;
        // CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        //     &maxBlocksPerSMPass2,
        //     msaCorrectCandidatesWithForestKernel_multiphase_comparemsapropsextractcorrectKernel<blocksize, groupsize, cands_extractor, GpuForest::Clf>,
        //     blocksize, 
        //     smemPass2
        // ));

        // dim3 block2 = blocksize;
        // dim3 grid2 = maxBlocksPerSMPass2 * numSMs;

        // helpers::GpuTimer timercomparemsapropsextractcorrectKernel(stream, "comparemsapropsextractcorrectKernel");

        // msaCorrectCandidatesWithForestKernel_multiphase_comparemsapropsextractcorrectKernel<blocksize, groupsize, cands_extractor, GpuForest::Clf>
        // <<<grid2, block2, smemPass2, stream>>>(
        //     d_correctedCandidates,
        //     multiMSA,
        //     gpuForest,
        //     forestThreshold,
        //     estimatedCoverage,
        //     d_shifts,
        //     d_bestAlignmentFlags,
        //     d_candidateSequencesData,
        //     d_candidateSequencesLengths,
        //     d_candidateContainsN,
        //     d_candidateIndicesOfCandidatesToBeCorrected,
        //     d_numCandidatesToBeCorrected,
        //     d_anchorIndicesOfCandidates,         
        //     encodedSequencePitchInInts,
        //     decodedSequencePitchInBytes
        // );
        // CUDACHECKASYNC;
        //timercomparemsapropsextractcorrectKernel.print();
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
        const AlignmentOrientation* d_bestAlignmentFlags,
        const unsigned int* d_candidateSequencesData,
        const int* d_candidateSequencesLengths,
        const bool* d_candidateContainsN,
        const int* d_candidateIndicesOfCandidatesToBeCorrected,
        const int* d_numCandidatesToBeCorrected,
        const int* d_anchorIndicesOfCandidates,
        const int numCandidates,
        int doNotUseEditsValue,
        int numEditsThreshold,            
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        size_t editsPitchInBytes,
        int maximum_sequence_length,
        cudaStream_t stream,
        const read_number* candidateReadIds
    ){

        constexpr int blocksize = 128;
        constexpr int groupsize = 32;
        constexpr int numGroupsPerBlock = blocksize / groupsize;

        //CUDACHECK(cudaStreamSynchronize(stream));

        //rmm::device_uvector<char> tmpaaa(decodedSequencePitchInBytes * 250000, stream);

        #if 1
        {
            callMsaCorrectCandidatesWithForestKernelMultiPhase(
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
                numCandidates,
                doNotUseEditsValue,
                numEditsThreshold,            
                encodedSequencePitchInInts,
                decodedSequencePitchInBytes,
                editsPitchInBytes,
                maximum_sequence_length,
                stream,
                candidateReadIds
            );
        }

        #else
        {
        const std::size_t dynamicsmemPitchInInts = SDIV(maximum_sequence_length, sizeof(int));
        const std::size_t treePointersPitchInInts = SDIV(sizeof(void*) * gpuForest.numTrees, sizeof(int));

        auto calculateSmemUsage = [&](int blockDim){
            const int numGroupsPerBlock = blockDim / groupsize;
            std::size_t smem = numGroupsPerBlock * (sizeof(int) * dynamicsmemPitchInInts)
                + treePointersPitchInInts * sizeof(int); 

            return smem;
        };

        const std::size_t smem = calculateSmemUsage(blocksize);

        int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            msaCorrectCandidatesWithForestKernel<blocksize, groupsize, cands_extractor, GpuForest::Clf>,
            blocksize, 
            smem
        ));

        const int maxBlocks = maxBlocksPerSM * numSMs;

        dim3 block = blocksize;
        dim3 grid = maxBlocks;

        //helpers::GpuTimer timer(stream, "combined");

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
        CUDACHECKASYNC;

        //timer.print();
        }
        #endif


        // helpers::lambda_kernel<<<1, 1, 0, stream>>>(
        //     [
        //         d_correctedCandidates,
        //         tmpaaa = tmpaaa.data(),
        //         decodedSequencePitchInBytes,
        //         d_numCandidatesToBeCorrected
        //     ] __device__ (){
        //         for(int c = blockIdx.x; c < *d_numCandidatesToBeCorrected; c += gridDim.x){
        //             const char* ptrold = d_correctedCandidates + c * decodedSequencePitchInBytes;
        //             const char* ptrnew = tmpaaa + c * decodedSequencePitchInBytes;
        //             // if(c == 1){
        //             //     printf("old: ");
        //             //     for(int i = threadIdx.x; i < 100; i += blockDim.x){
        //             //         printf("%c", ptrold[i]);
        //             //     }
        //             //     printf("\n");
        //             //     printf("new: ");
        //             //     for(int i = threadIdx.x; i < 100; i += blockDim.x){
        //             //         printf("%c", ptrnew[i]);
        //             //     }
        //             //     printf("\n");
        //             //     __syncthreads();
        //             //     assert(false);
        //             // }
        //             for(int i = threadIdx.x; i < 100; i += blockDim.x){
        //                 if(ptrold[i] != ptrnew[i]){
        //                     printf("%d %d %c %c\n", c, i, ptrold[i], ptrnew[i]);
        //                     assert(false);
        //                 }
        //             }
        //         }
        //     }
        // );
    }

    void callFlagCandidatesToBeCorrectedKernel_async(
        bool* d_candidateCanBeCorrected,
        int* d_numCorrectedCandidatesPerAnchor,
        GPUMultiMSA multiMSA,
        const int* d_alignmentShifts,
        const int* d_candidateSequencesLengths,
        const int* d_anchorIndicesOfCandidates,
        const AnchorHighQualityFlag* d_hqflags,
        const int* d_candidatesPerAnchorPrefixsum,
        const int* d_localGoodCandidateIndices,
        const int* d_numLocalGoodCandidateIndicesPerAnchor,
        const int* d_numAnchors,
        const int* d_numCandidates,
        float min_support_threshold,
        float min_coverage_threshold,
        int new_columns_to_correct,
        cudaStream_t stream
    ){

        constexpr int blocksize = 256;
        const std::size_t smem = 0;

        int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            flagCandidatesToBeCorrectedKernel,
            blocksize, 
            smem
        ));

        const int maxBlocks = maxBlocksPerSM * numSMs;

        dim3 block(blocksize);
        dim3 grid(maxBlocks);

        flagCandidatesToBeCorrectedKernel<<<grid, block, 0, stream>>>(
            d_candidateCanBeCorrected,
            d_numCorrectedCandidatesPerAnchor,
            multiMSA,
            d_alignmentShifts,
            d_candidateSequencesLengths,
            d_anchorIndicesOfCandidates,
            d_hqflags,
            d_candidatesPerAnchorPrefixsum,
            d_localGoodCandidateIndices,
            d_numLocalGoodCandidateIndicesPerAnchor,
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
        const int* d_candidatesPerAnchorPrefixsum,
        const int* d_localGoodCandidateIndices,
        const int* d_numLocalGoodCandidateIndicesPerAnchor,
        const int* d_numAnchors,
        const int* d_numCandidates,
        float min_support_threshold,
        float min_coverage_threshold,
        int new_columns_to_correct,
        cudaStream_t stream
    ){

        constexpr int blocksize = 256;
        const std::size_t smem = 0;

        int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            flagCandidatesToBeCorrectedWithExcludeFlagsKernel,
            blocksize, 
            smem
        ));

        const int maxBlocks = maxBlocksPerSM * numSMs;

        dim3 block(blocksize);
        dim3 grid(maxBlocks);

        flagCandidatesToBeCorrectedWithExcludeFlagsKernel<<<grid, block, 0, stream>>>(
            d_candidateCanBeCorrected,
            d_numCorrectedCandidatesPerAnchor,
            multiMSA,
            d_excludeFlags,
            d_alignmentShifts,
            d_candidateSequencesLengths,
            d_anchorIndicesOfCandidates,
            d_hqflags,
            d_candidatesPerAnchorPrefixsum,
            d_localGoodCandidateIndices,
            d_numLocalGoodCandidateIndicesPerAnchor,
            d_numAnchors,
            d_numCandidates,
            min_support_threshold,
            min_coverage_threshold,
            new_columns_to_correct
        );

        CUDACHECKASYNC;

    }



    void callCorrectCandidatesAndComputeEditsKernel(
        char* __restrict__ correctedCandidates,
        EncodedCorrectionEdit* __restrict__ d_editsPerCorrectedCandidate,
        int* __restrict__ d_numEditsPerCorrectedCandidate,
        GPUMultiMSA multiMSA,
        const int* __restrict__ shifts,
        const AlignmentOrientation* __restrict__ bestAlignmentFlags,
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
        cudaStream_t stream
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
        assert(smem % sizeof(int) == 0);

    	int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            msaCorrectCandidatesAndComputeEditsKernel<blocksize, groupsize>,
            blocksize, 
            smem
        ));

        const int maxBlocks = maxBlocksPerSM * numSMs;

    	dim3 block(blocksize, 1, 1);
        //dim3 grid(std::min(maxBlocks, n_candidates * numGroupsPerBlock));
        dim3 grid(maxBlocks);        

    	msaCorrectCandidatesAndComputeEditsKernel<blocksize, groupsize><<<grid, block, smem, stream>>>( 
            correctedCandidates, 
            d_editsPerCorrectedCandidate, 
            d_numEditsPerCorrectedCandidate, 
            multiMSA, 
            shifts, 
            bestAlignmentFlags, 
            candidateSequencesData, 
            candidateSequencesLengths, 
            d_candidateContainsN, 
            candidateIndicesOfCandidatesToBeCorrected, 
            numCandidatesToBeCorrected, 
            anchorIndicesOfCandidates, 
            d_numAnchors, 
            d_numCandidates, 
            doNotUseEditsValue, 
            numEditsThreshold, 
            encodedSequencePitchInInts, 
            decodedSequencePitchInBytes, 
            editsPitchInBytes, 
            dynamicsmemPitchInInts 
        ); 
        CUDACHECKASYNC;

    }

    void callCorrectCandidatesKernel(
        char* __restrict__ correctedCandidates,
        GPUMultiMSA multiMSA,
        const int* __restrict__ shifts,
        const AlignmentOrientation* __restrict__ bestAlignmentFlags,
        const unsigned int* __restrict__ candidateSequencesData,
        const int* __restrict__ candidateSequencesLengths,
        const bool* __restrict__ d_candidateContainsN,
        const int* __restrict__ candidateIndicesOfCandidatesToBeCorrected,
        const int* __restrict__ numCandidatesToBeCorrected,
        const int* __restrict__ anchorIndicesOfCandidates,
        const int* d_numAnchors,
        const int* d_numCandidates,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        int maximum_sequence_length,
        cudaStream_t stream
    ){
        constexpr int blocksize = 128;
        constexpr int groupsize = 32;

        const size_t dynamicsmemPitchInInts = SDIV(maximum_sequence_length, sizeof(int));
        auto calculateSmemUsage = [&](int blockDim){
            const int numGroupsPerBlock = blockDim / groupsize;
            std::size_t smem = numGroupsPerBlock * (sizeof(int) * dynamicsmemPitchInInts);

            return smem;
        };

        const std::size_t smem = calculateSmemUsage(blocksize);

    	assert(smem % sizeof(int) == 0);

    	int deviceId = 0;
        int numSMs = 0;
        int maxBlocksPerSM = 0;
        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            msaCorrectCandidatesKernel<blocksize, groupsize>,
            blocksize, 
            smem
        ));

        const int maxBlocks = maxBlocksPerSM * numSMs;

    	dim3 block(blocksize, 1, 1);
        //dim3 grid(std::min(maxBlocks, n_candidates * numGroupsPerBlock));
        dim3 grid(maxBlocks); 

    	msaCorrectCandidatesKernel<blocksize, groupsize><<<grid, block, smem, stream>>>(
            correctedCandidates, 
            multiMSA, 
            shifts, 
            bestAlignmentFlags, 
            candidateSequencesData, 
            candidateSequencesLengths, 
            d_candidateContainsN, 
            candidateIndicesOfCandidatesToBeCorrected, 
            numCandidatesToBeCorrected, 
            anchorIndicesOfCandidates, 
            d_numAnchors, 
            d_numCandidates, 
            encodedSequencePitchInInts, 
            decodedSequencePitchInBytes, 
            dynamicsmemPitchInInts 
        ); 
        CUDACHECKASYNC;

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
        cudaStream_t stream
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

        const int neededBlocks = SDIV(numCorrectedSequencesUpperBound, blocksize / groupsize);

        dim3 block = blocksize;        

        if(isCompactCorrection){
            int deviceId = 0;
            int numSMs = 0;
            int maxBlocksPerSM = 0;
            CUDACHECK(cudaGetDevice(&deviceId));
            CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
            CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxBlocksPerSM,
                constructSequenceCorrectionResultsKernel<true, groupsize>,
                blocksize, 
                smem
            ));

            const int maxBlocks = maxBlocksPerSM * numSMs;

            dim3 grid = std::min(maxBlocks, neededBlocks);

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
            int deviceId = 0;
            int numSMs = 0;
            int maxBlocksPerSM = 0;
            CUDACHECK(cudaGetDevice(&deviceId));
            CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
            CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxBlocksPerSM,
                constructSequenceCorrectionResultsKernel<false, groupsize>,
                blocksize, 
                smem
            ));

            const int maxBlocks = maxBlocksPerSM * numSMs;

            dim3 grid = std::min(maxBlocks, neededBlocks);

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
