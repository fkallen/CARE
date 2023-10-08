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
#include <gpu/rmm_utilities.cuh>


namespace cg = cooperative_groups;

namespace care{
namespace gpu{


    template<int BLOCKSIZE, int groupsize, class CandidateExtractor, class GpuClf>
    __global__
    void msaCorrectCandidatesWithForestKernel(
        char* __restrict__ correctedCandidates,
        GPUMultiMSA multiMSA,
        GpuClf gpuForest,
        float forestThreshold,
        float estimatedCoverage,
        const int* __restrict__ shifts,
        const AlignmentOrientation* __restrict__ bestAlignmentFlags,
        const unsigned int* __restrict__ candidateSequencesData,
        const int* __restrict__ candidateSequencesLengths,
        const int* __restrict__ candidateIndicesOfCandidatesToBeCorrected,
        int numCandidatesToBeCorrected,
        const int* __restrict__ anchorIndicesOfCandidates,         
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        size_t dynamicsmemSequencePitchInInts
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

        char* const shared_correctedCandidate = (char*)(dynamicsmem + dynamicsmemSequencePitchInInts * groupIdInBlock);

        GpuClf localForest = gpuForest;

        for(int id = groupId; id < numCandidatesToBeCorrected; id += numGroups){

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

    template<int BLOCKSIZE, int groupsize>
    __global__
    void msaCorrectCandidatesWithForestKernel_multiphase_initCorrectedCandidatesKernel(
        char* __restrict__ correctedCandidates,
        GPUMultiMSA multiMSA,
        const int* __restrict__ shifts,
        const AlignmentOrientation* __restrict__ bestAlignmentFlags,
        const unsigned int* __restrict__ candidateSequencesData,
        const int* __restrict__ candidateSequencesLengths,
        const int* __restrict__ candidateIndicesOfCandidatesToBeCorrected,
        int numCandidatesToBeCorrected,
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

        for(int id = groupId; id < numCandidatesToBeCorrected; id += numGroups){

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
        int numCandidatesToBeCorrected,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        int* __restrict__ numMismatches
    ){        
        static_assert(BLOCKSIZE % groupsize == 0, "BLOCKSIZE % groupsize != 0");
        //constexpr int groupsPerBlock = BLOCKSIZE / groupsize;
        static_assert(groupsize == 32);

        auto tgroup = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int numGroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        //const int groupIdInBlock = threadIdx.x / groupsize;

        using BlockReduce = cub::BlockReduce<int, BLOCKSIZE>;
        __shared__ typename BlockReduce::TempStorage temp_reduce;

        int myNumMismatches = 0;

        for(int id = groupId; id < numCandidatesToBeCorrected; id += numGroups){

            const int candidateIndex = candidateIndicesOfCandidatesToBeCorrected[id];
            //const int anchorIndex = anchorIndicesOfCandidates[candidateIndex];
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
        {

        }

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
        int numCandidatesToBeCorrected,
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
      
        for(int id = groupId; id < numCandidatesToBeCorrected; id += numGroups){

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
        //constexpr int groupsPerBlock = BLOCKSIZE / groupsize;
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
        int numCandidates,         
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

        for(int id = groupId; id < numCandidates; id += numGroups){

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



    //####################   KERNEL DISPATCH   ####################

    void callMsaCorrectCandidatesWithForestKernelMultiPhase(
        char* d_correctedCandidates,
        GPUMultiMSA multiMSA,
        GpuForest::Clf gpuForest,
        float forestThreshold,
        float estimatedCoverage,
        const int* d_shifts,
        const AlignmentOrientation* d_bestAlignmentFlags,
        const unsigned int* d_candidateSequencesData,
        const int* d_candidateSequencesLengths,
        const int* d_candidateIndicesOfCandidatesToBeCorrected,
        const int* d_anchorIndicesOfCandidates,
        const int numCandidatesToProcess,          
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        int maximum_sequence_length,
        cudaStream_t stream
    ){
        if(numCandidatesToProcess == 0) return;

        constexpr int blocksize = 128;
        constexpr int groupsize = 32;
        constexpr int numGroupsPerBlock = blocksize / groupsize;

        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();

        int deviceId = 0;
        int numSMs = 0;

        CUDACHECK(cudaGetDevice(&deviceId));
        CUDACHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
        
        int maxBlocksPerSMinit = 0;
        const std::size_t dynamicsmemPitchInInts = SDIV(maximum_sequence_length, sizeof(int));
        const std::size_t smeminit = numGroupsPerBlock * (sizeof(int) * dynamicsmemPitchInInts);

        CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSMinit,
            msaCorrectCandidatesWithForestKernel_multiphase_initCorrectedCandidatesKernel<blocksize, groupsize>,
            blocksize, 
            smeminit
        ));

        dim3 block1 = blocksize;
        dim3 grid1 = maxBlocksPerSMinit * numSMs;

        //helpers::GpuTimer timerinitCorrectedCandidatesKernel(stream, "initCorrectedCandidatesKernel");

        msaCorrectCandidatesWithForestKernel_multiphase_initCorrectedCandidatesKernel<blocksize, groupsize>
        <<<grid1, block1, smeminit, stream>>>(
            d_correctedCandidates,
            multiMSA,
            d_shifts,
            d_bestAlignmentFlags,
            d_candidateSequencesData,
            d_candidateSequencesLengths,
            d_candidateIndicesOfCandidatesToBeCorrected,
            numCandidatesToProcess,
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

        //helpers::GpuTimer timercountMismatchesKernel(stream, "countMismatchesKernel");

        msaCorrectCandidatesWithForestKernel_multiphase_countMismatchesKernel<blocksize, groupsize>
            <<<gridCountMismatches, blockCountMismatches, smemCountMismatches, stream>>>(
            d_correctedCandidates,
            d_bestAlignmentFlags,
            d_candidateSequencesData,
            d_candidateSequencesLengths,
            d_candidateIndicesOfCandidatesToBeCorrected,
            numCandidatesToProcess,       
            encodedSequencePitchInInts,
            decodedSequencePitchInBytes,
            d_numMismatches.data()
        );
        CUDACHECKASYNC;
        //timercountMismatchesKernel.print();

        const int numMismatches = d_numMismatches.value(stream);
        //std::cerr << "numMismatches cands: " << numMismatches << "\n";

        if(numMismatches == 0){
            return;
        }else{
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

            //helpers::GpuTimer timerfindMismatchesKernel(stream, "findMismatchesKernel");

            msaCorrectCandidatesWithForestKernel_multiphase_findMismatchesKernel<blocksize, groupsize>
                <<<gridFindMismatches, blockFindMismatches, smemFindMismatches, stream>>>(
                d_correctedCandidates,
                d_bestAlignmentFlags,
                d_candidateSequencesData,
                d_candidateSequencesLengths,
                d_candidateIndicesOfCandidatesToBeCorrected,
                numCandidatesToProcess,
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
            //helpers::GpuTimer timermsaprops(stream, "msapropsKernel");

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

            //helpers::GpuTimer timerextract(stream, "extractKernel");

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

            //helpers::GpuTimer timercorrectGroup(stream, "correctKernelGroup");
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

            //helpers::GpuTimer timercorrectThread(stream, "correctKernelThread");
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
            //     d_candidateIndicesOfCandidatesToBeCorrected,
            //     d_numCandidatesToBeCorrected,
            //     d_anchorIndicesOfCandidates,         
            //     encodedSequencePitchInInts,
            //     decodedSequencePitchInBytes
            // );
            // CUDACHECKASYNC;
            //timercomparemsapropsextractcorrectKernel.print();
        }
    }

    void callMsaCorrectCandidatesWithForestKernelSinglePhaseOld(
        char* d_correctedCandidates,
        GPUMultiMSA multiMSA,
        GpuForest::Clf gpuForest,
        float forestThreshold,
        float estimatedCoverage,
        const int* d_shifts,
        const AlignmentOrientation* d_bestAlignmentFlags,
        const unsigned int* d_candidateSequencesData,
        const int* d_candidateSequencesLengths,
        const int* d_candidateIndicesOfCandidatesToBeCorrected,
        const int* d_anchorIndicesOfCandidates,
        const int numCandidates,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        int maximum_sequence_length,
        cudaStream_t stream
    ){
        if(numCandidates == 0) return;

        constexpr int blocksize = 128;
        constexpr int groupsize = 32;

        const std::size_t dynamicsmemPitchInInts = SDIV(maximum_sequence_length, sizeof(int));

        auto calculateSmemUsage = [&](int blockDim){
            const int numGroupsPerBlock = blockDim / groupsize;
            std::size_t smem = numGroupsPerBlock * (sizeof(int) * dynamicsmemPitchInInts);

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

        msaCorrectCandidatesWithForestKernel<blocksize, groupsize, cands_extractor><<<grid, block, smem, stream>>>(
            d_correctedCandidates,
            multiMSA,
            gpuForest,
            forestThreshold,
            estimatedCoverage,
            d_shifts,
            d_bestAlignmentFlags,
            d_candidateSequencesData,
            d_candidateSequencesLengths,
            d_candidateIndicesOfCandidatesToBeCorrected,
            numCandidates,
            d_anchorIndicesOfCandidates, 
            encodedSequencePitchInInts,
            decodedSequencePitchInBytes,
            dynamicsmemPitchInInts
        );
        CUDACHECKASYNC;
    }


    void callMsaCorrectCandidatesWithForestKernel(
        char* d_correctedCandidates,
        GPUMultiMSA multiMSA,
        GpuForest::Clf gpuForest,
        float forestThreshold,
        float estimatedCoverage,
        const int* d_shifts,
        const AlignmentOrientation* d_bestAlignmentFlags,
        const unsigned int* d_candidateSequencesData,
        const int* d_candidateSequencesLengths,
        const int* d_candidateIndicesOfCandidatesToBeCorrected,
        const int* d_anchorIndicesOfCandidates,
        const int numCandidates,      
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        int maximum_sequence_length,
        cudaStream_t stream
    ){
        #if 1

        callMsaCorrectCandidatesWithForestKernelMultiPhase(
            d_correctedCandidates,
            multiMSA,
            gpuForest,
            forestThreshold,
            estimatedCoverage,
            d_shifts,
            d_bestAlignmentFlags,
            d_candidateSequencesData,
            d_candidateSequencesLengths,
            d_candidateIndicesOfCandidatesToBeCorrected,
            d_anchorIndicesOfCandidates,
            numCandidates,       
            encodedSequencePitchInInts,
            decodedSequencePitchInBytes,
            maximum_sequence_length,
            stream
        );

        #else

        callMsaCorrectCandidatesWithForestKernelSinglePhaseOld(
            d_correctedCandidates,
            multiMSA,
            gpuForest,
            forestThreshold,
            estimatedCoverage,
            d_shifts,
            d_bestAlignmentFlags,
            d_candidateSequencesData,
            d_candidateSequencesLengths,
            d_candidateIndicesOfCandidatesToBeCorrected,
            d_anchorIndicesOfCandidates,
            numCandidates,       
            encodedSequencePitchInInts,
            decodedSequencePitchInBytes,
            maximum_sequence_length,
            stream
        );

        #endif
    }


    void callMsaCorrectCandidatesWithForestKernelMultiPhase(
        std::vector<CandidateForestCorrectionTempStorage>& vec_correctionTempStorage,
        std::vector<char*>& vec_d_correctedCandidates,
        const std::vector<GPUMultiMSA>& vec_multiMSA,
        const std::vector<GpuForest::Clf>& vec_gpuForest,
        float forestThreshold,
        float estimatedCoverage,
        const std::vector<int*>& vec_d_shifts,
        const std::vector<AlignmentOrientation*>& vec_d_bestAlignmentFlags,
        const std::vector<unsigned int*>& vec_d_candidateSequencesData,
        const std::vector<int*>& vec_d_candidateSequencesLengths,
        const std::vector<int*>& vec_d_candidateIndicesOfCandidatesToBeCorrected,
        const std::vector<int*>& vec_d_anchorIndicesOfCandidates,
        const std::vector<int>& vec_numCandidatesToProcess,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
        int maximum_sequence_length,
        const std::vector<cudaStream_t>& streams,
        const std::vector<int>& deviceIds,
        int* h_tempstorage // sizeof(int) * deviceIds.size()
    ){

        constexpr int blocksize = 128;
        constexpr int groupsize = 32;
        constexpr int numGroupsPerBlock = blocksize / groupsize;

        const int numGpus = deviceIds.size();
        std::vector<int> vec_numSMs(numGpus);

        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd{deviceIds[g]};
            CUDACHECK(cudaDeviceGetAttribute(&vec_numSMs[g], cudaDevAttrMultiProcessorCount, deviceIds[g]));
        }

        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd{deviceIds[g]};
            if(vec_numCandidatesToProcess[g] > 0){
                int maxBlocksPerSMinit = 0;
                const std::size_t dynamicsmemPitchInInts = SDIV(maximum_sequence_length, sizeof(int));
                const std::size_t smeminit = numGroupsPerBlock * (sizeof(int) * dynamicsmemPitchInInts);

                CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &maxBlocksPerSMinit,
                    msaCorrectCandidatesWithForestKernel_multiphase_initCorrectedCandidatesKernel<blocksize, groupsize>,
                    blocksize, 
                    smeminit
                ));

                dim3 block1 = blocksize;
                dim3 grid1 = maxBlocksPerSMinit * vec_numSMs[g];

                //helpers::GpuTimer timerinitCorrectedCandidatesKernel(stream, "initCorrectedCandidatesKernel");

                msaCorrectCandidatesWithForestKernel_multiphase_initCorrectedCandidatesKernel<blocksize, groupsize>
                <<<grid1, block1, smeminit, streams[g]>>>(
                    vec_d_correctedCandidates[g],
                    vec_multiMSA[g],
                    vec_d_shifts[g],
                    vec_d_bestAlignmentFlags[g],
                    vec_d_candidateSequencesData[g],
                    vec_d_candidateSequencesLengths[g],
                    vec_d_candidateIndicesOfCandidatesToBeCorrected[g],
                    vec_numCandidatesToProcess[g],
                    vec_d_anchorIndicesOfCandidates[g],         
                    encodedSequencePitchInInts,
                    decodedSequencePitchInBytes,
                    dynamicsmemPitchInInts
                );
                CUDACHECKASYNC;
                //timerinitCorrectedCandidatesKernel.print();
            }
        }

        int* const h_numMismatchesPerGpu = h_tempstorage;
        
        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd{deviceIds[g]};
            if(vec_numCandidatesToProcess[g] > 0){
                int maxBlocksPerSMCountMismatches = 0;
                const std::size_t smemCountMismatches = 0;
                CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &maxBlocksPerSMCountMismatches,
                    msaCorrectCandidatesWithForestKernel_multiphase_countMismatchesKernel<blocksize, groupsize>,
                    blocksize, 
                    smemCountMismatches
                ));
                CUDACHECK(cudaMemsetAsync(
                    vec_correctionTempStorage[g].d_numMismatches.data(),
                    0,
                    sizeof(int),
                    streams[g]
                ));

                dim3 blockCountMismatches = blocksize;
                dim3 gridCountMismatches = std::min(maxBlocksPerSMCountMismatches * vec_numSMs[g], SDIV(vec_numCandidatesToProcess[g], (blocksize / groupsize)));

                //helpers::GpuTimer timercountMismatchesKernel(stream, "countMismatchesKernel");

                msaCorrectCandidatesWithForestKernel_multiphase_countMismatchesKernel<blocksize, groupsize>
                    <<<gridCountMismatches, blockCountMismatches, smemCountMismatches, streams[g]>>>(
                    vec_d_correctedCandidates[g],
                    vec_d_bestAlignmentFlags[g],
                    vec_d_candidateSequencesData[g],
                    vec_d_candidateSequencesLengths[g],
                    vec_d_candidateIndicesOfCandidatesToBeCorrected[g],
                    vec_numCandidatesToProcess[g],
                    encodedSequencePitchInInts,
                    decodedSequencePitchInBytes,
                    vec_correctionTempStorage[g].d_numMismatches.data()
                );
                CUDACHECKASYNC;
                //timercountMismatchesKernel.print();

                CUDACHECK(cudaMemcpyAsync(
                    h_numMismatchesPerGpu + g,
                    vec_correctionTempStorage[g].d_numMismatches.data(),
                    sizeof(int),
                    D2H,
                    streams[g]
                ));
            }
        }
        
        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd{deviceIds[g]};
            if(vec_numCandidatesToProcess[g] > 0){
                CUDACHECK(cudaStreamSynchronize(streams[g]));

                const int numMismatches = h_numMismatchesPerGpu[g];
                if(numMismatches > 0){
                    const std::size_t smemFindMismatches = 0;
                    int maxBlocksPerSMFindMismatches = 0;
                    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &maxBlocksPerSMFindMismatches,
                        msaCorrectCandidatesWithForestKernel_multiphase_findMismatchesKernel<blocksize, groupsize>,
                        blocksize, 
                        smemFindMismatches
                    ));

                    dim3 blockFindMismatches = blocksize;
                    dim3 gridFindMismatches = std::min(maxBlocksPerSMFindMismatches * vec_numSMs[g], SDIV(vec_numCandidatesToProcess[g], (blocksize / groupsize)));

                    CUDACHECK(cudaMemsetAsync(
                        vec_correctionTempStorage[g].d_numMismatches.data(),
                        0,
                        sizeof(int),
                        streams[g]
                    ));

                    vec_correctionTempStorage[g].resize(numMismatches, streams[g]);

                    //helpers::GpuTimer timerfindMismatchesKernel(stream, "findMismatchesKernel");

                    msaCorrectCandidatesWithForestKernel_multiphase_findMismatchesKernel<blocksize, groupsize>
                        <<<gridFindMismatches, blockFindMismatches, smemFindMismatches, streams[g]>>>(
                        vec_d_correctedCandidates[g],
                        vec_d_bestAlignmentFlags[g],
                        vec_d_candidateSequencesData[g],
                        vec_d_candidateSequencesLengths[g],
                        vec_d_candidateIndicesOfCandidatesToBeCorrected[g],
                        vec_numCandidatesToProcess[g],
                        vec_d_anchorIndicesOfCandidates[g],
                        encodedSequencePitchInInts,
                        decodedSequencePitchInBytes,
                        vec_correctionTempStorage[g].rawMismatchPositions
                    );
                    CUDACHECKASYNC;
                    //timerfindMismatchesKernel.print();
                }
            }
        }

        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd{deviceIds[g]};
            if(vec_numCandidatesToProcess[g] > 0){

                const int numMismatches = h_numMismatchesPerGpu[g];
                if(numMismatches > 0){
                    const std::size_t smemMsaProps = 0;
                    int maxBlocksPerSMMsaProps = 0;
                    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &maxBlocksPerSMMsaProps,
                        msaCorrectCandidatesWithForestKernel_multiphase_msapropsKernel<blocksize, groupsize>,
                        blocksize, 
                        smemMsaProps
                    ));

                    dim3 blockMsaProps = blocksize;
                    dim3 gridMsaProps = std::min(maxBlocksPerSMMsaProps * vec_numSMs[g], SDIV(numMismatches, (blocksize / groupsize)));

                    //helpers::GpuTimer timermsaprops(stream, "msapropsKernel");

                    msaCorrectCandidatesWithForestKernel_multiphase_msapropsKernel<blocksize, groupsize>
                        <<<gridMsaProps, blockMsaProps, smemMsaProps, streams[g]>>>(
                        vec_multiMSA[g],
                        vec_d_shifts[g],
                        vec_d_candidateSequencesLengths[g],
                        vec_correctionTempStorage[g].rawMismatchPositions,
                        vec_correctionTempStorage[g].d_msaPropertiesPerPosition
                    );
                    CUDACHECKASYNC;
                    //timermsaprops.print();
                }
            }
        }


        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd{deviceIds[g]};
            if(vec_numCandidatesToProcess[g] > 0){

                const int numMismatches = h_numMismatchesPerGpu[g];
                if(numMismatches > 0){
                    const std::size_t smemExtract = 0;
                    int maxBlocksPerSMExtract = 0;
                    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &maxBlocksPerSMExtract,
                        msaCorrectCandidatesWithForestKernel_multiphase_extractKernel<cands_extractor>,
                        blocksize, 
                        smemExtract
                    ));

                    dim3 blockExtract = blocksize;
                    dim3 gridExtract = std::min(maxBlocksPerSMExtract * vec_numSMs[g], SDIV(numMismatches, blocksize));

                    //helpers::GpuTimer timerextract(stream, "extractKernel");

                    msaCorrectCandidatesWithForestKernel_multiphase_extractKernel<cands_extractor>
                        <<<gridExtract, blockExtract, smemExtract, streams[g]>>>(
                        vec_correctionTempStorage[g].d_featuresTransposed,
                        vec_multiMSA[g],
                        estimatedCoverage,
                        vec_d_shifts[g],
                        vec_d_candidateSequencesLengths[g],  
                        vec_correctionTempStorage[g].rawMismatchPositions,
                        vec_correctionTempStorage[g].d_msaPropertiesPerPosition
                    );
                    CUDACHECKASYNC;
                    //timerextract.print();
                }
            }
        }

        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd{deviceIds[g]};
            if(vec_numCandidatesToProcess[g] > 0){

                const int numMismatches = h_numMismatchesPerGpu[g];
                if(numMismatches > 0){
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
                    dim3 gridCorrectThread = std::min(maxBlocksPerSMCorrectThread * vec_numSMs[g], SDIV(numMismatches, blocksize));

                    //helpers::GpuTimer timercorrectThread(stream, "correctKernelThread");
                    msaCorrectCandidatesWithForestKernel_multiphase_correctKernelThread<blocksize, GpuForest::Clf>
                        <<<gridCorrectThread, blockCorrectThread, smemCorrectThread, streams[g]>>>(
                        vec_d_correctedCandidates[g],
                        vec_multiMSA[g],
                        vec_gpuForest[g],
                        forestThreshold,
                        vec_d_bestAlignmentFlags[g],
                        vec_d_candidateSequencesLengths[g],
                        decodedSequencePitchInBytes,
                        vec_correctionTempStorage[g].rawMismatchPositions,
                        vec_correctionTempStorage[g].d_featuresTransposed,
                        useGlobalInsteadOfSmemCorrectThread,
                        cands_extractor::numFeatures()
                    );
                    CUDACHECKASYNC;
                    //timercorrectThread.print();
                }
            }
        }
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
        int numCandidates,
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
        dim3 grid(std::min(maxBlocks, numCandidates * (blocksize / groupsize)));
        //dim3 grid(maxBlocks); 

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
            numCandidates, 
            encodedSequencePitchInInts, 
            decodedSequencePitchInBytes, 
            dynamicsmemPitchInInts 
        ); 
        CUDACHECKASYNC;

    }



}
}
