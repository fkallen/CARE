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

    #if 0
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

    #else

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

        constexpr int numWarps = BLOCKSIZE / 32;

        auto tgroup = cg::this_thread_block();
        auto warp = cg::tiled_partition<32>(tgroup);

        //const int numGroups = gridDim.x;
        //const int groupId = blockIdx.x;
        //const int groupIdInBlock = 0;
  
        __shared__ float sharedFeatures[AnchorExtractor::numFeatures()];

        extern __shared__ int externalsmem[];
        
        char* const sharedCorrectedAnchor = (char*)&externalsmem[0];

        auto blockreduce = [&](auto val, auto op, auto neutral){
            using T = decltype(val);
            __shared__ T temp_reduce[numWarps+1];

            tgroup.sync();
            const auto warpreduced = cg::reduce(warp, val, op);
            if(warp.thread_rank() == 0){
                temp_reduce[warp.meta_group_rank()] = warpreduced;
            }
            tgroup.sync();
            if(warp.meta_group_rank() == 0){
                auto warpval = neutral;
                if(warp.thread_rank() < numWarps){
                    warpval = temp_reduce[warp.thread_rank()];
                }
                const auto blockreduced = cg::reduce(warp, warpval, op);
                if(warp.thread_rank() == 0){
                    temp_reduce[numWarps] = blockreduced;
                }
            }
            tgroup.sync();
            return temp_reduce[numWarps];
        };

        auto blockreducesum = [&](auto val){
            using T = decltype(val);
            return blockreduce(val, cg::plus<T>{}, T(0));
        };

        auto blockreducemin = [&](auto val){
            using T = decltype(val);
            return blockreduce(val, cg::less<T>{}, std::numeric_limits<T>::max());
        };

        auto blockreducemax = [&](auto val){
            using T = decltype(val);
            return blockreduce(val, cg::greater<T>{}, std::numeric_limits<T>::min());
        };

        for(unsigned anchorIndex = blockIdx.x; anchorIndex < numAnchors; anchorIndex += gridDim.x){
            const int myNumIndices = d_indices_per_anchor[anchorIndex];
            if(myNumIndices > 0){

                const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);                

                const int anchorColumnsBegin_incl = msa.columnProperties->anchorColumnsBegin_incl;
                const int anchorColumnsEnd_excl = msa.columnProperties->anchorColumnsEnd_excl;

                GpuMSAProperties msaProperties = msa.getMSAProperties(
                    tgroup, 
                    blockreducesum,
                    blockreducemin,
                    blockreducemin,
                    blockreducemax,
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
                    for (int i = 0; i < anchorLength; i += 1){
                        const int msaPos = anchorColumnsBegin_incl + i;
                        const std::uint8_t origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(anchor, anchorLength, i);
                        const std::uint8_t consensusEncodedBase = msa.consensus[msaPos];

                        if (origEncodedBase != consensusEncodedBase){                            
                            
                            ExtractAnchorInputData extractorInput;;

                            extractorInput.origBase = SequenceHelpers::decodeBase(origEncodedBase);
                            extractorInput.consensusBase = SequenceHelpers::decodeBase(consensusEncodedBase);
                            extractorInput.estimatedCoverage = estimatedCoverage;
                            extractorInput.msaPos = msaPos;
                            extractorInput.anchorColumnsBegin_incl = anchorColumnsBegin_incl;
                            extractorInput.anchorColumnsEnd_excl = anchorColumnsEnd_excl;
                            extractorInput.msaProperties = msaProperties;
                            extractorInput.msa = msa;

                            AnchorExtractor extractFeatures{};
                            extractFeatures(&sharedFeatures[0], extractorInput);

                            const bool useConsensus = gpuForest.decide(tgroup, &sharedFeatures[0], forestThreshold, blockreducesum);
                            if(tgroup.thread_rank() == 0){
                                if(!useConsensus){
                                    sharedCorrectedAnchor[i] = extractorInput.origBase;
                                }
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

    #endif

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
        //int* __restrict__ numMismatchesPerAnchor = nullptr
    ){

        static_assert(BLOCKSIZE % groupsize == 0, "BLOCKSIZE % groupsize != 0");
        //constexpr int groupsPerBlock = BLOCKSIZE / groupsize;
        static_assert(groupsize == 32);

        auto thread = cg::this_thread();
        auto tgroup = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int numGroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        //const int groupIdInBlock = threadIdx.x / groupsize;

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
            //int anchorMismatches = 0;

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
                            //printf("init: anchorIndex %d, position %d, orig %d, cons %d\n", anchorIndex, i, int(origEncodedBase), int(consensusEncodedBase));          
                            myNumMismatches++;
                            //anchorMismatches++;
                        }
                    }

                }
            }else{
                if(tgroup.thread_rank() == 0){
                    isHighQualityAnchor[anchorIndex].hq(false);
                    anchorIsCorrected[anchorIndex] = false;
                }
            }

            // if(numMismatchesPerAnchor != nullptr){
            //     int totalAnchorMismatches = cg::reduce(tgroup, anchorMismatches, cg::plus<int>{});
            //     if(tgroup.thread_rank() == 0){
            //         numMismatchesPerAnchor[anchorIndex] = totalAnchorMismatches;
            //     }
            // }
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
        const int* __restrict__ d_indices_per_anchor,
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
        //const int* __restrict__ d_numMismatchesPerAnchor = nullptr
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
        //__shared__ int smem_totalNumMismatchesForAnchor[groupsPerBlock];

        if(tgroup.thread_rank() == 0){
            smem_numMismatches[groupIdInBlock] = 0;
        }
        tgroup.sync();

        for(unsigned anchorIndex = groupId; anchorIndex < numAnchors; anchorIndex += numGroups){
            const int myNumIndices = d_indices_per_anchor[anchorIndex];
            if(myNumIndices > 0){
                const bool isHQCorrection = isHighQualityAnchor[anchorIndex].hq();

                const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);                

                const int anchorColumnsBegin_incl = msa.columnProperties->anchorColumnsBegin_incl;
                const int anchorColumnsEnd_excl = msa.columnProperties->anchorColumnsEnd_excl;

                const int anchorLength = anchorColumnsEnd_excl - anchorColumnsBegin_incl;
                const unsigned int* const anchor = anchorSequencesData + std::size_t(anchorIndex) * encodedSequencePitchInInts;

                if(!isHQCorrection){
                    const int loopend = SDIV(anchorLength, groupsize) * groupsize;

                    // if(tgroup.thread_rank() == 0){
                    //     smem_totalNumMismatchesForAnchor[groupIdInBlock] = 0;
                    // }
                    // tgroup.sync();
                    

                    for (int i = tgroup.thread_rank(); i < loopend; i += tgroup.size()){
                        if(i < anchorLength){
                            const int msaPos = anchorColumnsBegin_incl + i;
                            const std::uint8_t origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(anchor, anchorLength, i);
                            const std::uint8_t consensusEncodedBase = msa.consensus[msaPos];

                            if (origEncodedBase != consensusEncodedBase){
                                // if(anchorIndex == 1){
                                //     printf("anchorIndex %d, position %d\n", anchorIndex, i);
                                // }
                                //printf("gath: anchorIndex %d, position %d, orig %d, cons %d\n", anchorIndex, i, int(origEncodedBase), int(consensusEncodedBase));
                                auto selectedgroup = cg::coalesced_threads();

                                const int smemarraypos = selectedgroup.thread_rank();
                                smem_mismatchAnchorIndices[groupIdInBlock][smemarraypos] = anchorIndex;
                                smem_mismatchPositionsInAnchors[groupIdInBlock][smemarraypos] = i;

                                if(selectedgroup.thread_rank() == 0){
                                    smem_numMismatches[groupIdInBlock] = selectedgroup.size();
                                    //smem_totalNumMismatchesForAnchor[groupIdInBlock] += selectedgroup.size();
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
                    // if(d_numMismatchesPerAnchor != nullptr){
                    //     if(d_numMismatchesPerAnchor[anchorIndex] != smem_totalNumMismatchesForAnchor[groupIdInBlock]){
                    //         if(tgroup.thread_rank() == 0){
                    //             printf("error anchor %d, expected %d mismatches, got %d mismatches\n", anchorIndex, d_numMismatchesPerAnchor[anchorIndex], smem_totalNumMismatchesForAnchor[groupIdInBlock]);
                    //             assert(false);
                    //         }       
                    //         tgroup.sync();    
                    //     }
                    // }
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
        //constexpr int groupsPerBlock = BLOCKSIZE / groupsize;
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
        int /*maximumSequenceLength*/,
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

        if(numMismatches == 0){
            return;
        }else{

            constexpr int blocksizegather = 128;
            constexpr int groupsizegather = 32;

            const std::size_t smemgather = 0;        
            int maxBlocksPerSMgather = 0;

            CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxBlocksPerSMgather,
                msaCorrectAnchorsWithForestKernel_multiphase_gathermismatcheskernel<blocksizegather, groupsizegather>,
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
                d_indices_per_anchor,
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
            //constexpr std::size_t maxSmemextract = 32 * 1024;
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


    void callMsaCorrectAnchorsWithForestKernel_multiphase(
        std::vector<char*>& vec_d_correctedAnchors,
        std::vector<bool*>& vec_d_anchorIsCorrected,
        std::vector<AnchorHighQualityFlag*>& vec_d_isHighQualityAnchor,
        const std::vector<GPUMultiMSA>& vec_multiMSA,
        const std::vector<GpuForest::Clf>& vec_gpuForest,
        float forestThreshold,
        const std::vector<unsigned int*>& vec_d_anchorSequencesData,
        const std::vector<int*>& vec_d_indices_per_anchor,
        const std::vector<int>& vec_numAnchors,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        int /*maximumSequenceLength*/,
        float estimatedErrorrate,
        float estimatedCoverage,
        float avg_support_threshold,
        float min_support_threshold,
        float min_coverage_threshold,
        const std::vector<cudaStream_t>& streams,
        const std::vector<int>& deviceIds,
        int* h_tempstorage // sizeof(int) * deviceIds.size()
    ){
        // std::cout << "Launch forests\n";
        // {
        //     cub::SwitchDevice sd(deviceIds[0]);
        //     helpers::lambda_kernel<<<1,1,0, streams[0]>>>(
        //         [
        //             multiMSA = vec_multiMSA[0],
        //             d_indices_per_anchor = vec_d_indices_per_anchor[0],
        //             anchorSequencesData = vec_d_anchorSequencesData[0],
        //             encodedSequencePitchInInts,
        //             decodedSequencePitchInBytes,
        //             estimatedErrorrate,
        //             estimatedCoverage,
        //             avg_support_threshold,
        //             min_support_threshold,
        //             min_coverage_threshold
        //         ] __device__ (){
        //             auto tgroup = cg::this_thread();
        //             auto minreduce = [&](auto val){
        //                 using T = decltype(val);
        //                 return cg::reduce(tgroup, val, cg::less<T>{});
        //             };
            
        //             auto maxreduce = [&](auto val){
        //                 using T = decltype(val);
        //                 return cg::reduce(tgroup, val, cg::greater<T>{});
        //             };
            
        //             auto sumreduce = [&](auto val){
        //                 using T = decltype(val);
        //                 return cg::reduce(tgroup, val, cg::plus<T>{});
        //             };

        //             const int anchorIndex = 1707;

                    
        //             const int myNumIndices = d_indices_per_anchor[anchorIndex];
        //             printf("myNumIndices %d .\n", myNumIndices);
        //             if(myNumIndices > 0){

        //                 const GpuSingleMSA msa = multiMSA.getSingleMSA(anchorIndex);                

        //                 const int anchorColumnsBegin_incl = msa.columnProperties->anchorColumnsBegin_incl;
        //                 const int anchorColumnsEnd_excl = msa.columnProperties->anchorColumnsEnd_excl;

        //                 GpuMSAProperties msaProperties = msa.getMSAProperties(
        //                     tgroup, 
        //                     sumreduce,
        //                     minreduce, 
        //                     minreduce, 
        //                     maxreduce,
        //                     anchorColumnsBegin_incl,
        //                     anchorColumnsEnd_excl
        //                 );

        //                 AnchorCorrectionQuality correctionQuality(avg_support_threshold, min_support_threshold, min_coverage_threshold, estimatedErrorrate);

        //                 const bool isHQCorrection = correctionQuality.isHQCorrection(msaProperties.avg_support, msaProperties.min_support, msaProperties.min_coverage);

        //                 printf("anchorColumnsBegin_incl %d, anchorColumnsEnd_excl %d isHQCorrection %d.\n", anchorColumnsBegin_incl, anchorColumnsEnd_excl, isHQCorrection);

        //                 const int anchorLength = anchorColumnsEnd_excl - anchorColumnsBegin_incl;
        //                 const unsigned int* const anchor = anchorSequencesData + std::size_t(anchorIndex) * encodedSequencePitchInInts;

        //                 if(!isHQCorrection){
        //                     printf("anchor seq: ");
        //                     for (int i = tgroup.thread_rank(); i < anchorLength; i += tgroup.size()){
        //                         const std::uint8_t origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(anchor, anchorLength, i);
        //                         printf("%d ", int(origEncodedBase));
        //                     }
        //                     printf("\n");

        //                     printf("conse seq: ");
        //                     for (int i = tgroup.thread_rank(); i < anchorLength; i += tgroup.size()){
        //                         const int msaPos = anchorColumnsBegin_incl + i;
        //                         const std::uint8_t consensusEncodedBase = msa.consensus[msaPos];
        //                         printf("%d ", int(consensusEncodedBase));
        //                     }
        //                     printf("\n");

        //                     for (int i = tgroup.thread_rank(); i < anchorLength; i += tgroup.size()){
        //                         const int msaPos = anchorColumnsBegin_incl + i;
        //                         const std::uint8_t origEncodedBase = SequenceHelpers::getEncodedNuc2Bit(anchor, anchorLength, i);
        //                         const std::uint8_t consensusEncodedBase = msa.consensus[msaPos];

        //                         if (origEncodedBase != consensusEncodedBase){                  
        //                             printf("check: anchorIndex %d, position %d, orig %d, cons %d\n", anchorIndex, i, int(origEncodedBase), int(consensusEncodedBase));          
        //                         }
        //                     }

        //                 }
        //             }
        //         }
        //     );
        //     CUDACHECK(cudaStreamSynchronize(streams[0]));
        // }

        const int numGpus = deviceIds.size();
        std::vector<int> vec_numSMs(numGpus);
        std::vector<rmm::device_uvector<int>> vec_d_numMismatches;
        std::vector<rmm::device_uvector<GpuMSAProperties>> vec_d_msaProperties;
        std::vector<rmm::device_uvector<int>> vec_d_mismatchAnchorIndices;
        std::vector<rmm::device_uvector<int>> vec_d_mismatchPositionsInAnchors;
        std::vector<rmm::device_buffer> vec_gmemFeaturesTransposedExtract;

        //std::vector<rmm::device_uvector<int>> vec_d_numMismatchesPerAnchor;

        int* const h_numMismatchesPerGpu = h_tempstorage;


        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd{deviceIds[g]};

            CUDACHECK(cudaDeviceGetAttribute(&vec_numSMs[g], cudaDevAttrMultiProcessorCount, deviceIds[g]));
            vec_d_numMismatches.emplace_back(1, streams[g]);
            vec_d_msaProperties.emplace_back(0, streams[g]);
            vec_d_mismatchAnchorIndices.emplace_back(0, streams[g]);
            vec_d_mismatchPositionsInAnchors.emplace_back(0, streams[g]);
            vec_gmemFeaturesTransposedExtract.emplace_back(0, streams[g]);

            //vec_d_numMismatchesPerAnchor.emplace_back(vec_numAnchors[g], streams[g]);

            CUDACHECK(cudaMemsetAsync(
                vec_d_numMismatches[g].data(),
                0,
                sizeof(int),
                streams[g]
            ));
        }

        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd{deviceIds[g]};
            const int numAnchors = vec_numAnchors[g];
            if(numAnchors > 0){
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

                vec_d_msaProperties[g].resize(numAnchors, streams[g]);

                dim3 blockinit(blocksizeinit);
                dim3 gridinit(std::min(SDIV(numAnchors, blocksizeinit / groupsizeinit), maxBlocksPerSMinit * vec_numSMs[g]));

                //helpers::GpuTimer timerinit(stream, "initkernel");

                // CUDACHECK(cudaMemsetAsync(
                //     vec_d_numMismatchesPerAnchor[g].data(),
                //     0,
                //     sizeof(int) * numAnchors,
                //     streams[g]
                // ));

                msaCorrectAnchorsWithForestKernel_multiphase_initkernel<blocksizeinit, groupsizeinit>
                <<<gridinit, blockinit, smeminit, streams[g]>>>(
                    vec_d_correctedAnchors[g],
                    vec_d_anchorIsCorrected[g],
                    vec_d_isHighQualityAnchor[g],
                    vec_multiMSA[g],
                    vec_d_anchorSequencesData[g],
                    vec_d_indices_per_anchor[g],
                    numAnchors,
                    encodedSequencePitchInInts,
                    decodedSequencePitchInBytes,
                    estimatedErrorrate,
                    estimatedCoverage,
                    avg_support_threshold,
                    min_support_threshold,
                    min_coverage_threshold,
                    vec_d_msaProperties[g].data(),
                    vec_d_numMismatches[g].data()
                    //vec_d_numMismatchesPerAnchor[g].data()
                );
                CUDACHECKASYNC;
                //timerinit.print();

                CUDACHECK(cudaMemcpyAsync(
                    h_numMismatchesPerGpu + g,
                    vec_d_numMismatches[g].data(),
                    sizeof(int),
                    D2H,
                    streams[g]
                ));
            }
        }

        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd{deviceIds[g]};
            const int numAnchors = vec_numAnchors[g];
            if(numAnchors > 0){
                CUDACHECK(cudaStreamSynchronize(streams[g]));

                const int numMismatches = h_numMismatchesPerGpu[g];
                if(numMismatches > 0){
                    constexpr int blocksizegather = 128;
                    constexpr int groupsizegather = 32;

                    const std::size_t smemgather = 0;        
                    int maxBlocksPerSMgather = 0;

                    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &maxBlocksPerSMgather,
                        msaCorrectAnchorsWithForestKernel_multiphase_gathermismatcheskernel<blocksizegather, groupsizegather>,
                        blocksizegather, 
                        smemgather
                    ));

                    CUDACHECK(cudaMemsetAsync(
                        vec_d_numMismatches[g].data(),
                        0,
                        sizeof(int),
                        streams[g]
                    ));
                    vec_d_mismatchAnchorIndices[g].resize(numMismatches, streams[g]);
                    vec_d_mismatchPositionsInAnchors[g].resize(numMismatches, streams[g]);

                    dim3 blockgather(blocksizegather);
                    dim3 gridgather(std::min(SDIV(numAnchors, blocksizegather / groupsizegather), maxBlocksPerSMgather * vec_numSMs[g]));

                    //helpers::GpuTimer timergather(stream, "gatherkernel");        

                    msaCorrectAnchorsWithForestKernel_multiphase_gathermismatcheskernel<blocksizegather, groupsizegather>
                    <<<gridgather, blockgather, smemgather, streams[g]>>>(
                        vec_d_isHighQualityAnchor[g],
                        vec_multiMSA[g],
                        vec_d_anchorSequencesData[g],
                        vec_d_indices_per_anchor[g],
                        numAnchors,
                        encodedSequencePitchInInts,
                        decodedSequencePitchInBytes,
                        estimatedErrorrate,
                        estimatedCoverage,
                        avg_support_threshold,
                        min_support_threshold,
                        min_coverage_threshold,
                        vec_d_numMismatches[g].data(),
                        vec_d_mismatchAnchorIndices[g].data(),
                        vec_d_mismatchPositionsInAnchors[g].data()
                        //vec_d_numMismatchesPerAnchor[g].data()
                    );
                    CUDACHECKASYNC;
                    //timergather.print();
                }
            }
        }


        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd{deviceIds[g]};
            const int numAnchors = vec_numAnchors[g];
            if(numAnchors > 0){
                const int numMismatches = h_numMismatchesPerGpu[g];
                if(numMismatches > 0){
                    constexpr int blocksizeextract = 128;
                    //constexpr std::size_t maxSmemextract = 32 * 1024;
                    std::size_t featuresizeextract = numMismatches * anchor_extractor::numFeatures() * sizeof(float);

                    int maxBlocksPerSMextract = 0;
                    std::size_t smemextract = 0;

                    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &maxBlocksPerSMextract,
                        msaCorrectAnchorsWithForestKernel_multiphase_extractkernel<blocksizeextract, anchor_extractor>,
                        blocksizeextract, 
                        smemextract
                    ));

                    vec_gmemFeaturesTransposedExtract[g].resize(featuresizeextract, streams[g]);

                    dim3 blockextract(blocksizeextract);
                    dim3 gridextract(std::min(SDIV(numMismatches, blocksizeextract), maxBlocksPerSMextract * vec_numSMs[g]));

                    //helpers::GpuTimer timerextract(stream, "extractkernel");

                    msaCorrectAnchorsWithForestKernel_multiphase_extractkernel<blocksizeextract, anchor_extractor>
                    <<<gridextract, blockextract, smemextract, streams[g]>>>(
                        vec_multiMSA[g],
                        vec_d_anchorSequencesData[g],
                        encodedSequencePitchInInts,
                        decodedSequencePitchInBytes,
                        estimatedErrorrate,
                        estimatedCoverage,
                        avg_support_threshold,
                        min_support_threshold,
                        min_coverage_threshold,
                        reinterpret_cast<float*>(vec_gmemFeaturesTransposedExtract[g].data()),
                        numMismatches,
                        vec_d_mismatchAnchorIndices[g].data(),
                        vec_d_mismatchPositionsInAnchors[g].data(),
                        vec_d_msaProperties[g].data()
                    );
                    CUDACHECKASYNC;
                    //timerextract.print();
                }
            }
        }


        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd{deviceIds[g]};
            const int numAnchors = vec_numAnchors[g];
            if(numAnchors > 0){
                const int numMismatches = h_numMismatchesPerGpu[g];
                if(numMismatches > 0){
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
                    dim3 gridcorrectgroup(std::min(SDIV(numMismatches, blocksizecorrectgroup / groupsizecorrectgroup), maxBlocksPerSMcorrectgroup * vec_numSMs[g]));

                    //helpers::GpuTimer timercorrectgroup(stream, "correctgroupkernel");

                    msaCorrectAnchorsWithForestKernel_multiphase_correctkernel_group<blocksizecorrectgroup, groupsizecorrectgroup, GpuForest::Clf>
                    <<<gridcorrectgroup, blockcorrectgroup, smemcorrectgroup, streams[g]>>>(
                        vec_d_correctedAnchors[g],
                        vec_multiMSA[g],
                        vec_gpuForest[g],
                        forestThreshold,
                        vec_d_anchorSequencesData[g],
                        encodedSequencePitchInInts,
                        decodedSequencePitchInBytes,
                        estimatedErrorrate,
                        estimatedCoverage,
                        avg_support_threshold,
                        min_support_threshold,
                        min_coverage_threshold,
                        anchor_extractor::numFeatures(),
                        smemcorrectgroup == 0,
                        reinterpret_cast<float*>(vec_gmemFeaturesTransposedExtract[g].data()),
                        numMismatches,
                        vec_d_mismatchAnchorIndices[g].data(),
                        vec_d_mismatchPositionsInAnchors[g].data()
                    );
                    CUDACHECKASYNC;
                    //timercorrectgroup.print();
                }
            }
        }

        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd{deviceIds[g]};

            //ensure release of memory on the correct device
            vec_d_numMismatches[g].release();
            vec_d_msaProperties[g].release();
            vec_d_mismatchAnchorIndices[g].release();
            vec_d_mismatchPositionsInAnchors[g].release();
            //vec_gmemFeaturesTransposedExtract[g].release();
            auto toDeallocate = std::move(vec_gmemFeaturesTransposedExtract[g]);
            
            
            //vec_d_numMismatchesPerAnchor[g].release();
        }
        
    }


}
}
