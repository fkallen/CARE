//#define NDEBUG

#include <gpu/kernels.hpp>
#include <hostdevicefunctions.cuh>
#include <gpu/cudaerrorcheck.cuh>
#include <sequencehelpers.hpp>
#include <correctedsequence.hpp>
#include <hpc_helpers.cuh>
#include <config.hpp>

#include <cassert>

#include <cub/cub.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>


namespace cg = cooperative_groups;

namespace care{
namespace gpu{



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
