#ifndef READ_EXTENDER_GPU_HPP
#define READ_EXTENDER_GPU_HPP

#include <config.hpp>
#include <hpc_helpers.cuh>

#include <gpu/gpumsa.cuh>
#include <gpu/gpumsamanaged.cuh>
#include <gpu/kernels.hpp>
#include <gpu/kernellaunch.hpp>
#include <gpu/gpuminhasher.cuh>
#include <gpu/segmented_set_operations.cuh>
#include <gpu/cachingallocator.cuh>
#include <sequencehelpers.hpp>
#include <hostdevicefunctions.cuh>
#include <util.hpp>
#include <gpu/gpucpureadstorageadapter.cuh>
#include <gpu/gpucpuminhasheradapter.cuh>
#include <readextender_cpu.hpp>
#include <util_iterator.hpp>
#include <readextender_common.hpp>
#include <gpu/cubvector.cuh>
#include <gpu/cuda_block_select.cuh>


#include <algorithm>
#include <vector>
#include <numeric>

#include <cub/cub.cuh>

#include <thrust/device_new_allocator.h>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/logical.h>
#include <thrust/equal.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>


#define DO_REMOVE_USED_IDS_AND_MATE_IDS_ON_GPU

#define DO_ONLY_REMOVE_MATE_IDS



#if 0
    #define DEBUGDEVICESYNC { \
        cudaDeviceSynchronize(); CUERR; \
    }

#else 
    #define DEBUGDEVICESYNC {}

#endif

namespace care{




    template<int blocksize>
    struct CheckAmbiguousColumns{

        using BlockReduce = cub::BlockReduce<int, blocksize>;
        //using BlockScan = cub::BlockScan<int, blocksize>;
        using BlockSort = cub::BlockRadixSort<std::uint64_t, blocksize, 1>;
        using BlockDiscontinuity = cub::BlockDiscontinuity<std::uint64_t, blocksize>;
        using MyBlockSelect = BlockSelect<std::uint64_t, blocksize>;

        const int* countsA;
        const int* countsC;
        const int* countsG;
        const int* countsT;
        const int* coverages;

        void* tempstorage;

        __host__ __device__
        CheckAmbiguousColumns(const int* cA, const int* cC, const int* cG, const int* cT, const int* cov) 
            : countsA(cA), countsC(cC), countsG(cG), countsT(cT), coverages(cov){}

        //thread 0 returns number of ambiguous columns in given range. Block-wide algorithm
        __device__
        int getAmbiguousColumnCount(int begin, int end, typename BlockReduce::TempStorage& temp) const{ 

            int myCount = 0;

            for(int col = threadIdx.x; col < end; col += blocksize){
                if(col >= begin){

                    int numNucs = 0;

                    auto checkNuc = [&](const auto& counts, const char nuc){
                        const float ratio = float(counts[col]) / float(coverages[col]);
                        if(counts[col] >= 2 && fgeq(ratio, 0.4f) && fleq(ratio, 0.6f)){
                            numNucs++;                                
                        }
                    };

                    checkNuc(countsA, 'A');
                    checkNuc(countsC, 'C');
                    checkNuc(countsG, 'G');
                    checkNuc(countsT, 'T');

                    if(numNucs > 0){
                        myCount++;
                    }
                }
            }

            myCount = BlockReduce(temp).Sum(myCount);

            return myCount;
        }

        struct SplitInfo{
            char nuc;
            int column;
            //float ratio;

            SplitInfo() = default;

            __host__ __device__
            SplitInfo(char c, int n) : nuc(c), column(n){}

            __host__ __device__
            SplitInfo(char c, int n, float f) : nuc(c), column(n)/*, ratio(f)*/{}
        };

        struct SplitInfos{       
            static constexpr int maxSplitInfos = 64;

            int numSplitInfos;
            SplitInfo splitInfos[maxSplitInfos];
        };

        struct TempStorage{
            int broadcastint;
            union{
                typename BlockReduce::TempStorage blockreducetemp;
                //typename BlockScan::TempStorage blockscantemp;
                typename BlockSort::TempStorage blocksorttemp;
                typename BlockDiscontinuity::TempStorage blockdiscontinuity;
                typename MyBlockSelect::TempStorage blockselect;
            } cub;

            static constexpr int maxNumEncodedRows = 128;
            int numEncodedRows;
            std::uint64_t encodedRows[maxNumEncodedRows];
            int flags[maxNumEncodedRows];
        };



        __device__
        void getSplitInfos(int begin, int end, float relativeCountLowerBound, float relativeCountUpperBound, SplitInfos& result) const{
            using PSC = MultipleSequenceAlignment::PossibleSplitColumn;

            if(threadIdx.x == 0){
                result.numSplitInfos = 0;
            }
            __syncthreads();

            //find columns where counts/coverage is in range [relativeCountLowerBound, relativeCountUpperBound]
            //if there are exactly 2 different bases for which this is true in a column, SplitInfos of both cases are saved in result.splitInfos
            for(int col = threadIdx.x; col < end; col += blocksize){
                if(col >= begin){

                    SplitInfo myResults[2];
                    int myNumResults = 0;

                    auto checkNuc = [&](const auto& counts, const char nuc){
                        const float ratio = float(counts[col]) / float(coverages[col]);
                        if(counts[col] >= 2 && fgeq(ratio, relativeCountLowerBound) && fleq(ratio, relativeCountUpperBound)){

                            #pragma unroll
                            for(int k = 0; k < 2; k++){
                                if(myNumResults == k){
                                    myResults[k] = {nuc, col, ratio};
                                    myNumResults++;
                                    break;
                                }
                            }
                              
                        }
                    };

                    checkNuc(countsA, 'A');
                    checkNuc(countsC, 'C');
                    checkNuc(countsG, 'G');
                    checkNuc(countsT, 'T');

                    if(myNumResults == 2){
                        if(result.numSplitInfos < SplitInfos::maxSplitInfos){
                            int firstPos = atomicAdd(&result.numSplitInfos, 2);
                            if(firstPos <= SplitInfos::maxSplitInfos - 2){
                                result.splitInfos[firstPos + 0] = myResults[0];
                                result.splitInfos[firstPos + 1] = myResults[1];
                            }
                        }
                    }
                }
            }

            __syncthreads();
            if(threadIdx.x == 0){
                if(result.numSplitInfos > SplitInfos::maxSplitInfos){
                    result.numSplitInfos = SplitInfos::maxSplitInfos;
                }
            }
            __syncthreads();
        }

        __device__
        int getNumberOfSplits(
            const SplitInfos& splitInfos, 
            const gpu::MSAColumnProperties& msaColumnProperties,
            int numCandidates, 
            const int* candidateShifts,
            const BestAlignment_t* bestAlignmentFlags,
            const int* candidateLengths,
            const unsigned int* encodedCandidates, 
            int encodedSequencePitchInInts, 
            TempStorage& temp
        ){
            assert(TempStorage::maxNumEncodedRows == blocksize);

            // 64-bit flags. 2 bit per column. 
            // 00 -> nuc does not match any of both splitInfos
            // 10 -> nuc  matches first splitInfos
            // 11 -> nuc  matches second splitInfos
            constexpr int numPossibleColumnsPerFlag = 32; 
            //

            if(splitInfos.numSplitInfos == 0) return 1;
            if(splitInfos.numSplitInfos == 2) return 2;

            if(threadIdx.x == 0){
                temp.numEncodedRows = 0;
            }
            __syncthreads();

            const int subjectColumnsBegin_incl = msaColumnProperties.subjectColumnsBegin_incl;
            const int numColumnsToCheck = std::min(numPossibleColumnsPerFlag, splitInfos.numSplitInfos / 2);
            const int maxCandidatesToCheck = std::min(blocksize, numCandidates);

            #if 0
            if(threadIdx.x == 0){                
                if(splitInfos.numSplitInfos > 0){
                    printf("numSplitInfos %d\n", splitInfos.numSplitInfos);
                    for(int i = 0; i < splitInfos.numSplitInfos; i++){
                        printf("(%c,%d, %f) ", 
                            splitInfos.splitInfos[i].nuc, 
                            splitInfos.splitInfos[i].column,
                            splitInfos.splitInfos[i].ratio);
                    }
                    printf("\n");

                    for(int c = 0; c < maxCandidatesToCheck; c++){
                        const unsigned int* const myCandidate = encodedCandidates + c * encodedSequencePitchInInts;
                        const int candidateShift = candidateShifts[c];
                        const int candidateLength = candidateLengths[c];
                        const BestAlignment_t alignmentFlag = bestAlignmentFlags[c];

                        for(int i = 0; i < candidateShift; i++){
                            printf("0");
                        }
                        for(int i = 0; i < candidateLength; i++){
                            char nuc = 'F';
                            if(alignmentFlag == BestAlignment_t::Forward){
                                const int positionInCandidate = i;
                                std::uint8_t encodedCandidateNuc = SequenceHelpers::getEncodedNuc2Bit(myCandidate, candidateLength, positionInCandidate);
                                nuc = SequenceHelpers::decodeBase(encodedCandidateNuc);
                            }else{
                                const int positionInCandidate = candidateLength - 1 - i;
                                std::uint8_t encodedCandidateNuc = SequenceHelpers::getEncodedNuc2Bit(myCandidate, candidateLength, positionInCandidate);
                                nuc = SequenceHelpers::complementBaseDecoded(SequenceHelpers::decodeBase(encodedCandidateNuc));
                            }
                            printf("%c", nuc);
                        }
                        printf("\n");
                    }
                }
            }

            __syncthreads(); //DEBUG
            #endif

            for(int c = threadIdx.x; c < maxCandidatesToCheck; c += blocksize){
                if(temp.numEncodedRows < TempStorage::maxNumEncodedRows){
                    std::uint64_t flags = 0;

                    const unsigned int* const myCandidate = encodedCandidates + c * encodedSequencePitchInInts;
                    const int candidateShift = candidateShifts[c];
                    const int candidateLength = candidateLengths[c];
                    const BestAlignment_t alignmentFlag = bestAlignmentFlags[c];

                    for(int k = 0; k < numColumnsToCheck; k++){
                        flags <<= 2;

                        const SplitInfo psc0 = splitInfos.splitInfos[2*k+0];
                        const SplitInfo psc1 = splitInfos.splitInfos[2*k+1];
                        assert(psc0.column == psc1.column);

                        const int candidateColumnsBegin_incl = candidateShift + subjectColumnsBegin_incl;
                        const int candidateColumnsEnd_excl = candidateLength + candidateColumnsBegin_incl;
                        
                        //column range check for row
                        if(candidateColumnsBegin_incl <= psc0.column && psc0.column < candidateColumnsEnd_excl){                        

                            char nuc = 'F';
                            if(alignmentFlag == BestAlignment_t::Forward){
                                const int positionInCandidate = psc0.column - candidateColumnsBegin_incl;
                                std::uint8_t encodedCandidateNuc = SequenceHelpers::getEncodedNuc2Bit(myCandidate, candidateLength, positionInCandidate);
                                nuc = SequenceHelpers::decodeBase(encodedCandidateNuc);
                            }else{
                                const int positionInCandidate = candidateLength - 1 - (psc0.column - candidateColumnsBegin_incl);
                                std::uint8_t encodedCandidateNuc = SequenceHelpers::getEncodedNuc2Bit(myCandidate, candidateLength, positionInCandidate);
                                nuc = SequenceHelpers::complementBaseDecoded(SequenceHelpers::decodeBase(encodedCandidateNuc));
                            }

                            //printf("cand %d col %d %c\n", c, psc0.column, nuc);

                            if(nuc == psc0.nuc){
                                flags = flags | 0b10;
                            }else if(nuc == psc1.nuc){
                                flags = flags | 0b11;
                            }else{
                                flags = flags | 0b00;
                            } 

                        }else{
                            flags = flags | 0b00;
                        } 

                    }

                    const int tempPos = atomicAdd(&temp.numEncodedRows, 1);
                    if(tempPos < TempStorage::maxNumEncodedRows){
                        temp.encodedRows[tempPos] = flags;
                    }
                }
            }
        
            __syncthreads();
            if(threadIdx.x == 0){
                if(temp.numEncodedRows > TempStorage::maxNumEncodedRows){
                    temp.numEncodedRows = TempStorage::maxNumEncodedRows;
                }
            }
            __syncthreads();

            {
                //sort the computed encoded rows, and make them unique

                std::uint64_t encodedRow[1];

                if(threadIdx.x < temp.numEncodedRows){
                    encodedRow[0] = temp.encodedRows[threadIdx.x];
                }else{
                    encodedRow[0] = std::numeric_limits<std::uint64_t>::max();
                }

                BlockSort(temp.cub.blocksorttemp).Sort(encodedRow);
                __syncthreads();

                if(threadIdx.x < temp.numEncodedRows){
                    temp.encodedRows[threadIdx.x] = encodedRow[0];
                    temp.flags[threadIdx.x] = 0;
                }

                __syncthreads();

                int headflag[1];

                if(threadIdx.x < temp.numEncodedRows){
                    encodedRow[0] = temp.encodedRows[threadIdx.x];
                }else{
                    encodedRow[0] = temp.encodedRows[temp.numEncodedRows-1];
                }

                BlockDiscontinuity(temp.cub.blockdiscontinuity).FlagHeads(headflag, encodedRow, cub::Inequality());

                __syncthreads();

                int numselected = MyBlockSelect(temp.cub.blockselect).Flagged(encodedRow, headflag, &temp.encodedRows[0], temp.numEncodedRows);

                __syncthreads();
                if(threadIdx.x == 0){
                    temp.numEncodedRows = numselected;
                }
                __syncthreads();
            }

            // if(threadIdx.x == 0){                
            //     if(splitInfos.numSplitInfos > 0){
            //         printf("numEncodedRows %d\n", temp.numEncodedRows);
            //         for(int i = 0; i < temp.numEncodedRows; i++){
            //             printf("%lu ", temp.encodedRows[i]);
            //         }
            //         printf("\n");
            //     }
            // }

            // __syncthreads(); //DEBUG

            for(int i = 0; i < temp.numEncodedRows - 1; i++){
                const std::uint64_t encodedRow = temp.encodedRows[i];

                std::uint64_t mask = 0;
                for(int s = 0; s < numColumnsToCheck; s++){
                    if(encodedRow >> (2*s+1) & 1){
                        mask = mask | (0x03 << (2*s));
                    }
                }
                
                if(i < threadIdx.x && threadIdx.x < temp.numEncodedRows){
                    //check if encodedRow is equal to another flag masked with mask. if yes, it can be removed
                    if((temp.encodedRows[threadIdx.x] & mask) == encodedRow){
                        // if(encodedRow == 172){
                        //     printf("i = %d, thread=%d temp.encodedRows[threadIdx.x] = %lu, mask = %lu", i, threadIdx.x, temp.encodedRows[threadIdx.x], mask)
                        // }
                        atomicAdd(&temp.flags[i], 1);
                    }
                }
            }

            __syncthreads();

            //count number of remaining row flags
            int count = 0;
            if(threadIdx.x < temp.numEncodedRows){
                count = temp.flags[threadIdx.x] == 0 ? 1 : 0;
            }
            count = BlockReduce(temp.cub.blockreducetemp).Sum(count);
            if(threadIdx.x == 0){
                temp.broadcastint = count;
                // if(splitInfos.numSplitInfos > 0){
                //     printf("count = %d\n", count);
                // }
            }
            __syncthreads();

            count = temp.broadcastint;

            return count;
        }

    };

namespace readextendergpukernels{

    template<int blocksize>
    __global__
    void computeExtensionStepFromMsaKernel(
        int insertSize,
        int insertSizeStddev,
        const gpu::GPUMultiMSA multiMSA,
        const int* d_numCandidatesPerAnchor,
        const int* d_numCandidatesPerAnchorPrefixSum,
        const int* d_anchorSequencesLength,
        const int* d_accumExtensionsLengths,
        const int* d_inputMateLengths,
        extension::AbortReason* d_abortReasons,
        int* d_accumExtensionsLengthsOUT,
        char* d_outputAnchors,
        int outputAnchorPitchInBytes,
        char* d_outputAnchorQualities,
        int outputAnchorQualityPitchInBytes,
        int* d_outputAnchorLengths,
        const bool* d_isPairedTask,
        const unsigned int* d_inputanchormatedata,
        int encodedSequencePitchInInts,
        int decodedMatesRevCPitchInBytes,
        bool* d_outputMateHasBeenFound,
        int* d_sizeOfGapToMate,
        int minCoverageForExtension,
        int fixedStepsize
    ){

        using BlockReduce = cub::BlockReduce<int, blocksize>;
        using BlockReduceFloat = cub::BlockReduce<float, blocksize>;

        __shared__ union{
            typename BlockReduce::TempStorage reduce;
            typename BlockReduceFloat::TempStorage reduceFloat;
        } temp;

        constexpr int smemEncodedMateInts = 32;
        __shared__ unsigned int smemEncodedMate[smemEncodedMateInts];

        __shared__ int broadcastsmem_int;

        for(int t = blockIdx.x; t < multiMSA.numMSAs; t += gridDim.x){
            const int numCandidates = d_numCandidatesPerAnchor[t];

            if(numCandidates > 0){
                const gpu::GpuSingleMSA msa = multiMSA.getSingleMSA(t);

                const int anchorLength = d_anchorSequencesLength[t];
                int accumExtensionsLength = d_accumExtensionsLengths[t];
                const int mateLength = d_inputMateLengths[t];
                const bool isPaired = d_isPairedTask[t];

                const int consensusLength = msa.computeSize();

                auto consensusDecoded = msa.getDecodedConsensusIterator();
                auto consensusQuality = msa.getConsensusQualityIterator();

                extension::AbortReason* const abortReasonPtr = d_abortReasons + t;
                char* const outputAnchor = d_outputAnchors + t * outputAnchorPitchInBytes;
                char* const outputAnchorQuality = d_outputAnchorQualities + t * outputAnchorQualityPitchInBytes;
                int* const outputAnchorLengthPtr = d_outputAnchorLengths + t;
                bool* const mateHasBeenFoundPtr = d_outputMateHasBeenFound + t;

                int extendBy = std::min(
                    consensusLength - anchorLength, 
                    std::max(0, fixedStepsize)
                );
                //cannot extend over fragment 
                extendBy = std::min(extendBy, (insertSize + insertSizeStddev - mateLength) - accumExtensionsLength);

                //auto firstLowCoverageIter = std::find_if(coverage + anchorLength, coverage + consensusLength, [&](int cov){ return cov < minCoverageForExtension; });
                //coverage is monotonically decreasing. convert coverages to 1 if >= minCoverageForExtension, else 0. Find position of first 0
                int myPos = consensusLength;
                for(int i = anchorLength + threadIdx.x; i < consensusLength; i += blockDim.x){
                    int flag = msa.coverages[i] < minCoverageForExtension ? 0 : 1;
                    if(flag == 0 && i < myPos){
                        myPos = i;
                    }
                }

                myPos = BlockReduce(temp.reduce).Reduce(myPos, cub::Min{});

                if(threadIdx.x == 0){
                    broadcastsmem_int = myPos;
                }
                __syncthreads();
                myPos = broadcastsmem_int;
                __syncthreads();

                if(fixedStepsize <= 0){
                    extendBy = myPos - anchorLength;
                    extendBy = std::min(extendBy, (insertSize + insertSizeStddev - mateLength) - accumExtensionsLength);
                }

                auto makeAnchorForNextIteration = [&](){
                    if(extendBy == 0){
                        if(threadIdx.x == 0){
                            *abortReasonPtr = extension::AbortReason::MsaNotExtended;
                        }
                    }else{
                        if(threadIdx.x == 0){
                            d_accumExtensionsLengthsOUT[t] = accumExtensionsLength + extendBy;
                            *outputAnchorLengthPtr = anchorLength;
                        }           

                        for(int i = threadIdx.x; i < anchorLength; i += blockDim.x){
                            outputAnchor[i] = consensusDecoded[extendBy + i];
                            outputAnchorQuality[i] = consensusQuality[extendBy + i];
                        }
                    }
                };

                constexpr int requiredOverlapMate = 70; //TODO relative overlap 
                constexpr float maxRelativeMismatchesInOverlap = 0.06f;
                constexpr int maxAbsoluteMismatchesInOverlap = 10;

                const int maxNumMismatches = std::min(int(mateLength * maxRelativeMismatchesInOverlap), maxAbsoluteMismatchesInOverlap);

                

                if(isPaired && accumExtensionsLength + consensusLength - requiredOverlapMate + mateLength >= insertSize - insertSizeStddev){
                    //for each possibility to overlap the mate and consensus such that the merged sequence would end in the desired range [insertSize - insertSizeStddev, insertSize + insertSizeStddev]

                    const int firstStartpos = std::max(0, insertSize - insertSizeStddev - accumExtensionsLength - mateLength);
                    const int lastStartposExcl = std::min(
                        std::max(0, insertSize + insertSizeStddev - accumExtensionsLength - mateLength) + 1,
                        consensusLength - requiredOverlapMate
                    );

                    int bestOverlapMismatches = std::numeric_limits<int>::max();
                    int bestOverlapStartpos = -1;

                    const unsigned int* encodedMate = nullptr;
                    {
                        const unsigned int* const gmemEncodedMate = d_inputanchormatedata + t * encodedSequencePitchInInts;
                        const int requirednumints = SequenceHelpers::getEncodedNumInts2Bit(mateLength);
                        if(smemEncodedMateInts >= requirednumints){
                            for(int i = threadIdx.x; i < requirednumints; i += blockDim.x){
                                smemEncodedMate[i] = gmemEncodedMate[i];
                            }
                            encodedMate = &smemEncodedMate[0];
                            __syncthreads();
                        }else{
                            encodedMate = &gmemEncodedMate[0];
                        }
                    }

                    for(int startpos = firstStartpos; startpos < lastStartposExcl; startpos++){
                        //compute metrics of overlap

                        //Hamming distance. positions which do not overlap are not accounted for
                        int ham = 0;
                        for(int i = threadIdx.x; i < min(consensusLength - startpos, mateLength); i += blockDim.x){
                            std::uint8_t encbasemate = SequenceHelpers::getEncodedNuc2Bit(encodedMate, mateLength, mateLength - 1 - i);
                            std::uint8_t encbasematecomp = SequenceHelpers::complementBase2Bit(encbasemate);
                            char decbasematecomp = SequenceHelpers::decodeBase(encbasematecomp);

                            //TODO store consensusDecoded in smem ?
                            ham += (consensusDecoded[startpos + i] != decbasematecomp) ? 1 : 0;
                        }

                        ham = BlockReduce(temp.reduce).Sum(ham);

                        if(threadIdx.x == 0){
                            broadcastsmem_int = ham;
                        }
                        __syncthreads();
                        ham = broadcastsmem_int;
                        __syncthreads();

                        if(bestOverlapMismatches > ham){
                            bestOverlapMismatches = ham;
                            bestOverlapStartpos = startpos;
                        }

                        if(bestOverlapMismatches == 0){
                            break;
                        }
                    }

                    // if(threadIdx.x == 0){
                    //     printf("gpu: bestOverlapMismatches %d,bestOverlapStartpos %d\n", bestOverlapMismatches, bestOverlapStartpos);
                    // }

                    if(bestOverlapMismatches <= maxNumMismatches){
                        const int mateStartposInConsensus = bestOverlapStartpos;
                        const int missingPositionsBetweenAnchorEndAndMateBegin = std::max(0, mateStartposInConsensus - anchorLength);
                        // if(threadIdx.x == 0){
                        //     printf("missingPositionsBetweenAnchorEndAndMateBegin %d\n", missingPositionsBetweenAnchorEndAndMateBegin);
                        // }

                        if(missingPositionsBetweenAnchorEndAndMateBegin > 0){
                            //bridge the gap between current anchor and mate

                            for(int i = threadIdx.x; i < missingPositionsBetweenAnchorEndAndMateBegin; i += blockDim.x){
                                outputAnchor[i] = consensusDecoded[anchorLength + i];
                                outputAnchorQuality[i] = consensusQuality[anchorLength + i];
                            }

                            if(threadIdx.x == 0){
                                d_accumExtensionsLengthsOUT[t] = accumExtensionsLength + anchorLength;
                                *outputAnchorLengthPtr = missingPositionsBetweenAnchorEndAndMateBegin;
                                *mateHasBeenFoundPtr = true;
                                d_sizeOfGapToMate[t] = missingPositionsBetweenAnchorEndAndMateBegin;
                            }
                        }else{

                            if(threadIdx.x == 0){
                                d_accumExtensionsLengthsOUT[t] = accumExtensionsLength + mateStartposInConsensus;
                                *outputAnchorLengthPtr = 0;
                                *mateHasBeenFoundPtr = true;
                                d_sizeOfGapToMate[t] = 0;
                            }
                        }

                        
                    }else{
                        makeAnchorForNextIteration();
                    }
                }else{
                    makeAnchorForNextIteration();
                }

            }else{ //numCandidates == 0
                if(threadIdx.x == 0){
                    d_abortReasons[t] = extension::AbortReason::NoPairedCandidatesAfterAlignment;
                }
            }
        }
    }

    template<int blocksize>
    __global__
    void computeExtensionStepQualityKernel(
        float* d_goodscores,
        const gpu::GPUMultiMSA multiMSA,
        const extension::AbortReason* d_abortReasons,
        const bool* d_mateHasBeenFound,
        const int* accumExtensionLengthsBefore,
        const int* accumExtensionLengthsAfter,
        const int* anchorLengths,
        const int* d_numCandidatesPerAnchor,
        const int* d_numCandidatesPerAnchorPrefixSum,
        const int* d_candidateSequencesLengths,
        const int* d_alignment_shifts,
        const BestAlignment_t* d_alignment_best_alignment_flags,
        const unsigned int* d_candidateSequencesData,
        const gpu::MSAColumnProperties* d_msa_column_properties,
        int encodedSequencePitchInInts
    ){
        using BlockReduce = cub::BlockReduce<int, blocksize>;
        using BlockReduceFloat = cub::BlockReduce<float, blocksize>;
        using AmbiguousColumnsChecker = CheckAmbiguousColumns<blocksize>;

        __shared__ union{
            typename BlockReduce::TempStorage reduce;
            typename BlockReduceFloat::TempStorage reduceFloat;
            typename AmbiguousColumnsChecker::TempStorage columnschecker;
        } temp;

        __shared__ typename AmbiguousColumnsChecker::SplitInfos smemSplitInfos;

        for(int t = blockIdx.x; t < multiMSA.numMSAs; t += gridDim.x){
            if(d_abortReasons[t] == extension::AbortReason::None && !d_mateHasBeenFound[t]){

                const gpu::GpuSingleMSA msa = multiMSA.getSingleMSA(t);

                const int extendedBy = accumExtensionLengthsAfter[t] - accumExtensionLengthsBefore[t];
                const int anchorLength = anchorLengths[t];

                const float* const mySupport = msa.support;
                const int* const myCounts = msa.counts;
                const int* const myCoverage = msa.coverages;

                const int* const myCandidateLengths = d_candidateSequencesLengths + d_numCandidatesPerAnchorPrefixSum[t];
                const int* const myCandidateShifts = d_alignment_shifts + d_numCandidatesPerAnchorPrefixSum[t];
                const BestAlignment_t* const myCandidateBestAlignmentFlags = d_alignment_best_alignment_flags + d_numCandidatesPerAnchorPrefixSum[t];
                const unsigned int* const myCandidateSequencesData = d_candidateSequencesData + encodedSequencePitchInInts * d_numCandidatesPerAnchorPrefixSum[t];

                // float supportSum = 0.0f;

                // for(int i = threadIdx.x; i < extendedBy; i += blockDim.x){
                //     supportSum += 1.0f - mySupport[anchorLength + i];
                // }

                // float reducedSupport = BlockReduceFloat(temp.reduceFloat).Sum(supportSum);

                // //printf("reducedSupport %f\n", reducedSupport);
                // if(threadIdx.x == 0){
                //     d_goodscores[t] = reducedSupport;
                // }

                AmbiguousColumnsChecker checker(
                    myCounts + 0 * msa.columnPitchInElements,
                    myCounts + 1 * msa.columnPitchInElements,
                    myCounts + 2 * msa.columnPitchInElements,
                    myCounts + 3 * msa.columnPitchInElements,
                    myCoverage
                );

                //int count = checker.getAmbiguousColumnCount(anchorLength, anchorLength + extendedBy, temp.reduce);

                //auto a = clock();

                checker.getSplitInfos(anchorLength, anchorLength + extendedBy, 0.4f, 0.6f, smemSplitInfos);

                //auto b = clock();

                int count = checker.getNumberOfSplits(
                    smemSplitInfos, 
                    d_msa_column_properties[t],
                    d_numCandidatesPerAnchor[t], 
                    myCandidateShifts,
                    myCandidateBestAlignmentFlags,
                    myCandidateLengths,
                    myCandidateSequencesData, 
                    encodedSequencePitchInInts, 
                    temp.columnschecker
                );

                //auto c = clock();

                if(threadIdx.x == 0){
                    d_goodscores[t] = count;
                    //printf("cand %d extendedBy %d, %lu %lu, infos %d, count %d\n", d_numCandidatesPerAnchor[t], extendedBy, b-a, c-b, smemSplitInfos.numSplitInfos, count);
                }
                
                __syncthreads();
            }
        }
    }

    template<int blocksize, int itemsPerThread, bool inclusive, class T>
    __global__
    void prefixSumSingleBlockKernel(
        const T* input,
        T* output,
        int N
    ){
        struct BlockPrefixCallbackOp{
            // Running prefix
            int running_total;

            __device__
            BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
            // Callback operator to be entered by the first warp of threads in the block.
            // Thread-0 is responsible for returning a value for seeding the block-wide scan.
            __device__
            int operator()(int block_aggregate){
                int old_prefix = running_total;
                running_total += block_aggregate;
                return old_prefix;
            }
        };

        assert(blocksize == blockDim.x);

        using BlockScan = cub::BlockScan<T, blocksize>;
        using BlockLoad = cub::BlockLoad<T, blocksize, itemsPerThread, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
        using BlockStore = cub::BlockStore<T, blocksize, itemsPerThread, cub::BLOCK_STORE_WARP_TRANSPOSE>;

        __shared__ typename BlockScan::TempStorage blockscantemp;
        __shared__ union{
            typename BlockLoad::TempStorage load;
            typename BlockStore::TempStorage store;
        } temp;

        T items[itemsPerThread];

        BlockPrefixCallbackOp prefix_op(0);

        const int iterations = SDIV(N, blocksize);

        int remaining = N;

        const T* currentInput = input;
        T* currentOutput = output;

        for(int iteration = 0; iteration < iterations; iteration++){
            const int valid_items = min(itemsPerThread * blocksize, remaining);

            BlockLoad(temp.load).Load(currentInput, items, valid_items, 0);

            if(inclusive){
                BlockScan(blockscantemp).InclusiveSum(
                    items, items, prefix_op
                );
            }else{
                BlockScan(blockscantemp).ExclusiveSum(
                    items, items, prefix_op
                );
            }
            __syncthreads();

            BlockStore(temp.store).Store(currentOutput, items, valid_items);
            __syncthreads();

            remaining -= valid_items;
            currentInput += valid_items;
            currentOutput += valid_items;
        }
    }

    //output[map[i]] = input[i];
    template<class T, class U>
    __global__ 
    void setFirstSegmentIdsKernel(
        const T* __restrict__ segmentSizes,
        int* __restrict__ segmentIds,
        const U* __restrict__ segmentOffsets,
        int N
    ){
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;
        const int stride = blockDim.x * gridDim.x;

        for(int i = tid; i < N; i += stride){
            if(segmentSizes[i] > 0){
                segmentIds[segmentOffsets[i]] = i;
            }
        }
    }

    __global__
    void setSegmentIndicesKernel(
        int* __restrict__ d_indices,
        int N,
        const int* __restrict__ d_segment_sizes,
        const int* __restrict__ d_segment_sizes_prefixsum
    ){
        for(int segmentIndex = blockIdx.x; segmentIndex < N; segmentIndex += gridDim.x){
            const int offset = d_segment_sizes_prefixsum[segmentIndex];
            const int size = d_segment_sizes[segmentIndex];
            int* const beginptr = &d_indices[offset];

            for(int localindex = threadIdx.x; localindex < size; localindex += blockDim.x){
                beginptr[localindex] = segmentIndex;
            }
        }
    }

    //flag candidates to remove because they are equal to anchor id or equal to mate id
    __global__
    void flagCandidateIdsWhichAreEqualToAnchorOrMateKernel(
        const read_number* __restrict__ candidateReadIds,
        const read_number* __restrict__ anchorReadIds,
        const read_number* __restrict__ mateReadIds,
        const int* __restrict__ numCandidatesPerAnchorPrefixSum,
        const int* __restrict__ numCandidatesPerAnchor,
        bool* __restrict__ keepflags, // size numCandidates
        bool* __restrict__ mateRemovedFlags, //size numTasks
        int* __restrict__ numCandidatesPerAnchorOut,
        int numTasks,
        bool isPairedEnd
    ){

        using BlockReduceInt = cub::BlockReduce<int, 128>;

        __shared__ typename BlockReduceInt::TempStorage intreduce1;
        __shared__ typename BlockReduceInt::TempStorage intreduce2;

        for(int a = blockIdx.x; a < numTasks; a += gridDim.x){
            const int size = numCandidatesPerAnchor[a];
            const int offset = numCandidatesPerAnchorPrefixSum[a];
            const read_number anchorId = anchorReadIds[a];
            read_number mateId = 0;
            if(isPairedEnd){
                mateId = mateReadIds[a];
            }

            int mateIsRemoved = 0;
            int numRemoved = 0;

            // if(threadIdx.x == 0){
            //     printf("looking for anchor %u, mate %u\n", anchorId, mateId);
            // }
            __syncthreads();

            for(int i = threadIdx.x; i < size; i+= blockDim.x){
                bool keep = true;

                const read_number candidateId = candidateReadIds[offset + i];
                //printf("tid %d, comp %u at position %d\n", threadIdx.x, candidateId, offset + i);

                if(candidateId == anchorId){
                    keep = false;
                    numRemoved++;
                }

                if(isPairedEnd && candidateId == mateId){
                    if(keep){
                        keep = false;
                        numRemoved++;
                    }
                    mateIsRemoved++;
                    //printf("mate removed. i = %d\n", i);
                }

                keepflags[offset + i] = keep;
            }
            //printf("tid = %d, mateIsRemoved = %d\n", threadIdx.x, mateIsRemoved);
            int numRemovedBlock = BlockReduceInt(intreduce1).Sum(numRemoved);
            int numMateRemovedBlock = BlockReduceInt(intreduce2).Sum(mateIsRemoved);
            if(threadIdx.x == 0){
                numCandidatesPerAnchorOut[a] = size - numRemovedBlock;
                //printf("numMateRemovedBlock %d\n", numMateRemovedBlock);
                if(numMateRemovedBlock > 0){
                    mateRemovedFlags[a] = true;
                }else{
                    mateRemovedFlags[a] = false;
                }
            }
        }
    }

    template<int dummy=0>
    __global__
    void setSegmentIdsOfCandidateskernel(
        int* __restrict__ d_segmentIdsOfCandidates,
        const int* __restrict__ numAnchorsPtr,
        const int* __restrict__ d_candidates_per_anchor,
        const int* __restrict__ d_candidates_per_anchor_prefixsum
    ){
        for(int anchorIndex = blockIdx.x; anchorIndex < *numAnchorsPtr; anchorIndex += gridDim.x){
            const int offset = d_candidates_per_anchor_prefixsum[anchorIndex];
            const int numCandidatesOfAnchor = d_candidates_per_anchor[anchorIndex];
            int* const beginptr = &d_segmentIdsOfCandidates[offset];

            for(int localindex = threadIdx.x; localindex < numCandidatesOfAnchor; localindex += blockDim.x){
                beginptr[localindex] = anchorIndex;
            }
        }
    }
    

    template<int blocksize>
    __global__
    void reverseComplement2bitKernel(
        const int* __restrict__ lengths,
        const unsigned int* __restrict__ forward,
        unsigned int* __restrict__ reverse,
        int num,
        int encodedSequencePitchInInts
    ){

        for(int s = threadIdx.x + blockIdx.x * blockDim.x; s < num; s += blockDim.x * gridDim.x){
            const unsigned int* input = forward + encodedSequencePitchInInts * s;
            unsigned int* output = reverse + encodedSequencePitchInInts * s;
            const int length = lengths[s];

            SequenceHelpers::reverseComplementSequence2Bit(
                output,
                input,
                length,
                [](auto i){return i;},
                [](auto i){return i;}
            );
        }

        // constexpr int smemsizeints = blocksize * 16;
        // __shared__ unsigned int sharedsequences[smemsizeints]; //sequences will be stored transposed

        // const int sequencesPerSmem = std::min(blocksize, smemsizeints / encodedSequencePitchInInts);
        // assert(sequencesPerSmem > 0);

        // const int smemiterations = SDIV(num, sequencesPerSmem);

        // for(int smemiteration = blockIdx.x; smemiteration < smemiterations; smemiteration += gridDim.x){

        //     const int idBegin = smemiteration * sequencesPerSmem;
        //     const int idEnd = std::min((smemiteration+1) * sequencesPerSmem, num);

        //     __syncthreads();

        //     for(int s = idBegin + threadIdx.x; s < idEnd; s += blockDim.x){
        //         for(int intindex = 0; intindex < encodedSequencePitchInInts; intindex++){ //load intindex-th element of sequence s
        //             sharedsequences[intindex * sequencesPerSmem + s] = forward[encodedSequencePitchInInts * s + intindex];
        //         }
        //     }

        //     __syncthreads();

        //     for(int s = idBegin + threadIdx.x; s < idEnd; s += blockDim.x){
        //         SequenceHelpers::reverseComplementSequenceInplace2Bit(&sharedsequences[s], lengths[s], [&](auto i){return i * sequencesPerSmem;});
        //     }

        //     __syncthreads();

        //     for(int s = idBegin + threadIdx.x; s < idEnd; s += blockDim.x){
        //         for(int intindex = 0; intindex < encodedSequencePitchInInts; intindex++){ //load intindex-th element of sequence s
        //             reverse[encodedSequencePitchInInts * s + intindex] = sharedsequences[intindex * sequencesPerSmem + s];
        //         }
        //     }
        // }
    }

    template<int groupsize>
    __global__
    void encodeSequencesTo2BitKernel(
        unsigned int* __restrict__ encodedSequences,
        const char* __restrict__ decodedSequences,
        const int* __restrict__ sequenceLengths,
        int decodedSequencePitchInBytes,
        int encodedSequencePitchInInts,
        int numSequences
    ){
        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());

        const int numGroups = (blockDim.x * gridDim.x) / group.size();
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / group.size();

        for(int a = groupId; a < numSequences; a += numGroups){
            unsigned int* const out = encodedSequences + a * encodedSequencePitchInInts;
            const char* const in = decodedSequences + a * decodedSequencePitchInBytes;
            const int length = sequenceLengths[a];

            const int nInts = SequenceHelpers::getEncodedNumInts2Bit(length);
            constexpr int basesPerInt = SequenceHelpers::basesPerInt2Bit();

            for(int i = group.thread_rank(); i < nInts; i += group.size()){
                unsigned int data = 0;

                const int loopend = min((i+1) * basesPerInt, length);
                
                for(int nucIndex = i * basesPerInt; nucIndex < loopend; nucIndex++){
                    switch(in[nucIndex]) {
                    case 'A':
                        data = (data << 2) | SequenceHelpers::encodedbaseA();
                        break;
                    case 'C':
                        data = (data << 2) | SequenceHelpers::encodedbaseC();
                        break;
                    case 'G':
                        data = (data << 2) | SequenceHelpers::encodedbaseG();
                        break;
                    case 'T':
                        data = (data << 2) | SequenceHelpers::encodedbaseT();
                        break;
                    default:
                        data = (data << 2) | SequenceHelpers::encodedbaseA();
                        break;
                    }
                }

                if(i == nInts-1){
                    //pack bits of last integer into higher order bits
                    int leftoverbits = 2 * (nInts * basesPerInt - length);
                    if(leftoverbits > 0){
                        data <<= leftoverbits;
                    }
                }

                out[i] = data;
            }
        }
    }

    template<int groupsize>
    __global__
    void decodeSequencesFrom2BitKernel(
        char* __restrict__ decodedSequences,
        const unsigned int* __restrict__ encodedSequences,
        const int* __restrict__ sequenceLengths,
        int decodedSequencePitchInBytes,
        int encodedSequencePitchInInts,
        int numSequences
    ){
        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());

        const int numGroups = (blockDim.x * gridDim.x) / group.size();
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / group.size();

        for(int a = groupId; a < numSequences; a += numGroups){
            char* const out = decodedSequences + a * decodedSequencePitchInBytes;
            const unsigned int* const in = encodedSequences + a * encodedSequencePitchInInts;
            const int length = sequenceLengths[a];

            const int nInts = SequenceHelpers::getEncodedNumInts2Bit(length);
            constexpr int basesPerInt = SequenceHelpers::basesPerInt2Bit();

            for(int i = group.thread_rank(); i < nInts; i += group.size()){
                unsigned int data = in[i];

                if(i < nInts-1){
                    //not last iteration. int encodes 16 chars
                    __align__(16) char nucs[16];

                    #pragma unroll
                    for(int p = 0; p < 16; p++){
                        const std::uint8_t encodedBase = SequenceHelpers::getEncodedNucFromInt2Bit(data, p);
                        nucs[p] = SequenceHelpers::decodeBase(encodedBase);
                    }
                    ((int4*)out)[i] = *((const int4*)&nucs[0]);
                }else{
                    const int remaining = length - i * basesPerInt;

                    for(int p = 0; p < remaining; p++){
                        const std::uint8_t encodedBase = SequenceHelpers::getEncodedNucFromInt2Bit(data, p);
                        out[i * basesPerInt + p] = SequenceHelpers::decodeBase(encodedBase);
                    }
                }
            }
        }
    }

    template<int blocksize, int groupsize>
    __global__
    void filtermatekernel(
        const unsigned int* __restrict__ anchormatedata,
        const unsigned int* __restrict__ candidatefwddata,
        //const unsigned int* __restrict__ candidatefwddata2,
        int encodedSequencePitchInInts,
        const int* __restrict__ numCandidatesPerAnchor,
        const int* __restrict__ numCandidatesPerAnchorPrefixSum,
        const int* __restrict__ activeTaskIndices,
        int numTasksWithRemovedMate,
        bool* __restrict__ output_keepflags
    ){

        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int groupindex = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int numgroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupindexinblock = threadIdx.x / groupsize;

        static_assert(blocksize % groupsize == 0);
        //constexpr int groupsperblock = blocksize / groupsize;

        extern __shared__ unsigned int smem[]; //sizeof(unsigned int) * groupsperblock * encodedSequencePitchInInts

        unsigned int* sharedMate = smem + groupindexinblock * encodedSequencePitchInInts;

        for(int task = groupindex; task < numTasksWithRemovedMate; task += numgroups){

            const int globalTaskIndex = activeTaskIndices[task];
            const int numCandidates = numCandidatesPerAnchor[globalTaskIndex];
            const int candidatesOffset = numCandidatesPerAnchorPrefixSum[globalTaskIndex];

            for(int p = group.thread_rank(); p < encodedSequencePitchInInts; p++){
                sharedMate[p] = anchormatedata[encodedSequencePitchInInts * task + p];
            }
            group.sync();

            //compare mate to candidates. 1 thread per candidate
            for(int c = group.thread_rank(); c < numCandidates; c += group.size()){
                bool doKeep = false;
                const unsigned int* const candidateptr = candidatefwddata + encodedSequencePitchInInts * (candidatesOffset + c);

                for(int p = 0; p < encodedSequencePitchInInts; p++){
                    const unsigned int aaa = sharedMate[p];
                    const unsigned int bbb = candidateptr[p];

                    if(aaa != bbb){
                        doKeep = true;
                        break;
                    }
                }

                output_keepflags[(candidatesOffset + c)] = doKeep;
            }
        }
    }

    template<int groupsize>
    __global__
    void compactUsedIdsOfSelectedTasks(
        const int* __restrict__ indices,
        int numIndices,
        const read_number* __restrict__ d_usedReadIdsIn,
        read_number* __restrict__ d_usedReadIdsOut,
        int* __restrict__ segmentIdsOut,
        const int* __restrict__ d_numUsedReadIdsPerAnchor,
        const int* __restrict__ inputSegmentOffsets,
        const int* __restrict__ outputSegmentOffsets
    ){
        const int warpid = (threadIdx.x + blockDim.x * blockIdx.x) / groupsize;
        const int numwarps = (blockDim.x * gridDim.x) / groupsize;
        const int lane = threadIdx.x % groupsize;

        for(int t = warpid; t < numIndices; t += numwarps){
            const int activeIndex = indices[t];
            const int num = d_numUsedReadIdsPerAnchor[t];
            const int inputOffset = inputSegmentOffsets[activeIndex];
            const int outputOffset = outputSegmentOffsets[t];

            for(int i = lane; i < num; i += groupsize){
                //copy read id
                d_usedReadIdsOut[outputOffset + i] = d_usedReadIdsIn[inputOffset + i];
                //set new segment id
                segmentIdsOut[outputOffset + i] = t;
            }
        }
    }

    template<int blocksize, int groupsize>
    __global__
    void createGpuTaskData(
        int numReadPairs,
        const read_number* __restrict__ d_readpair_readIds,
        const int* __restrict__ d_readpair_readLengths,
        const unsigned int* __restrict__ d_readpair_sequences,
        const char* __restrict__ d_readpair_qualities,
        unsigned int* __restrict__ d_subjectSequencesData,
        int* __restrict__ d_anchorSequencesLength,
        char* __restrict__ d_anchorQualityScores,
        unsigned int* __restrict__ d_inputanchormatedata,
        int* __restrict__ d_inputMateLengths,
        bool* __restrict__ d_isPairedTask,
        read_number* __restrict__ d_anchorReadIds,
        read_number* __restrict__ d_mateReadIds,
        int* __restrict__ d_accumExtensionsLengths,
        int decodedSequencePitchInBytes,
        int qualityPitchInBytes,
        int encodedSequencePitchInInts
    ){
        constexpr int numGroupsInBlock = blocksize / groupsize;

        __shared__ unsigned int sharedEncodedSequence[numGroupsInBlock][32];

        assert(encodedSequencePitchInInts <= 32);

        auto group = cg::tiled_partition<groupsize>(cg::this_thread_block());
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int groupIdInBlock = groupId % numGroupsInBlock;
        const int numGroups = (blockDim.x * gridDim.x) / groupsize;
        
        const int numTasks = numReadPairs * 4;

        //handle scalars
        #if 1
        for(int t = threadIdx.x + blockIdx.x * blockDim.x; t < numTasks; t += blockDim.x * gridDim.x){
            d_accumExtensionsLengths[t] = 0;

            const int pairId = t / 4;
            const int id = t % 4;

            if(id == 0){
                d_anchorReadIds[t] = d_readpair_readIds[2 * pairId + 0];
                d_mateReadIds[t] = d_readpair_readIds[2 * pairId + 1];
                d_anchorSequencesLength[t] = d_readpair_readLengths[2 * pairId + 0];
                d_inputMateLengths[t] = d_readpair_readLengths[2 * pairId + 1];
                d_isPairedTask[t] = true;
            }else if(id == 1){
                d_anchorReadIds[t] = d_readpair_readIds[2 * pairId + 1];
                d_mateReadIds[t] = std::numeric_limits<read_number>::max();
                d_anchorSequencesLength[t] = d_readpair_readLengths[2 * pairId + 1];
                d_inputMateLengths[t] = 0;
                d_isPairedTask[t] = false;
            }else if(id == 2){
                d_anchorReadIds[t] = d_readpair_readIds[2 * pairId + 1];
                d_mateReadIds[t] = d_readpair_readIds[2 * pairId + 0];
                d_anchorSequencesLength[t] = d_readpair_readLengths[2 * pairId + 1];
                d_inputMateLengths[t] = d_readpair_readLengths[2 * pairId + 0];
                d_isPairedTask[t] = true;
            }else{
                //id == 3
                d_anchorReadIds[t] = d_readpair_readIds[2 * pairId + 0];
                d_mateReadIds[t] = std::numeric_limits<read_number>::max();
                d_anchorSequencesLength[t] = d_readpair_readLengths[2 * pairId + 0];
                d_inputMateLengths[t] = 0;
                d_isPairedTask[t] = false;
            }
        }

        #endif

        #if 1
        //handle sequences
        for(int t = groupId; t < numTasks; t += numGroups){
            const int pairId = t / 4;
            const int id = t % 4;

            const unsigned int* const myReadpairSequences = d_readpair_sequences + 2 * pairId * encodedSequencePitchInInts;
            unsigned int* const myAnchorSequence = d_subjectSequencesData + t * encodedSequencePitchInInts;
            unsigned int* const myMateSequence = d_inputanchormatedata + t * encodedSequencePitchInInts;

            if(id == 0){
                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    myAnchorSequence[k] = myReadpairSequences[k];
                    myMateSequence[k] = myReadpairSequences[encodedSequencePitchInInts + k];
                }
            }else if(id == 1){
                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    sharedEncodedSequence[groupIdInBlock][k] = myReadpairSequences[encodedSequencePitchInInts + k];
                }
                group.sync();
                if(group.thread_rank() == 0){
                    SequenceHelpers::reverseComplementSequence2Bit(
                        myAnchorSequence,
                        &sharedEncodedSequence[groupIdInBlock][0],
                        d_readpair_readLengths[2 * pairId + 1],
                        [](auto i){return i;},
                        [](auto i){return i;}
                    );
                }
                group.sync();
            }else if(id == 2){
                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    myAnchorSequence[k] = myReadpairSequences[encodedSequencePitchInInts + k];
                    myMateSequence[k] = myReadpairSequences[k];
                }
            }else{
                //id == 3
                for(int k = group.thread_rank(); k < encodedSequencePitchInInts; k += group.size()){
                    sharedEncodedSequence[groupIdInBlock][k] = myReadpairSequences[k];
                }
                group.sync();
                if(group.thread_rank() == 0){
                    SequenceHelpers::reverseComplementSequence2Bit(
                        myAnchorSequence,
                        &sharedEncodedSequence[groupIdInBlock][0],
                        d_readpair_readLengths[2 * pairId + 0],
                        [](auto i){return i;},
                        [](auto i){return i;}
                    );
                }
                group.sync();
            }
        }
        #endif

        #if 1

        //handle qualities
        for(int t = blockIdx.x; t < numTasks; t += gridDim.x){
            const int pairId = t / 4;
            const int id = t % 4;

            const int* const myReadPairLengths = d_readpair_readLengths + 2 * pairId;
            const char* const myReadpairQualities = d_readpair_qualities + 2 * pairId * qualityPitchInBytes;
            char* const myAnchorQualities = d_anchorQualityScores + t * qualityPitchInBytes;

            //const int numInts = qualityPitchInBytes / sizeof(int);
            int l0 = myReadPairLengths[0];
            int l1 = myReadPairLengths[1];

            if(id == 0){
                for(int k = threadIdx.x; k < l0; k += blockDim.x){
                    myAnchorQualities[k] = myReadpairQualities[k];
                }
            }else if(id == 1){
                for(int k = threadIdx.x; k < l1; k += blockDim.x){
                    myAnchorQualities[k] = myReadpairQualities[qualityPitchInBytes + l1 - 1 - k];
                }
            }else if(id == 2){
                for(int k = threadIdx.x; k < l1; k += blockDim.x){
                    myAnchorQualities[k] = myReadpairQualities[qualityPitchInBytes + k];
                }
            }else{
                //id == 3
                for(int k = threadIdx.x; k < l0; k += blockDim.x){
                    myAnchorQualities[k] = myReadpairQualities[l0 - 1 - k];
                }
            }
        }

        #endif
    }
}


struct GpuReadExtender{
    template<class T>
    using DeviceBuffer = helpers::SimpleAllocationDevice<T>;
    //using DeviceBuffer = helpers::SimpleAllocationPinnedHost<T>;

    template<class T>
    using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T>;

    struct ExtensionTaskCpuData{
        bool pairedEnd = false;
        bool mateHasBeenFound = false;
        int id = 0;
        int pairId = 0;
        int myLength = 0;
        int mateLength = 0;
        int currentAnchorLength = 0;
        int accumExtensionLengths = 0;
        int iteration = 0;
        float goodscore = 0.0f;
        read_number myReadId = 0;
        read_number mateReadId = 0;
        extension::AbortReason abortReason = extension::AbortReason::None;
        care::extension::ExtensionDirection direction{};
        std::vector<char> decodedMateRevC{};
        std::vector<char> mateQualityScoresReversed{};
        std::vector<int> totalDecodedAnchorsLengths{};
        std::vector<char> totalDecodedAnchorsFlat{};
        std::vector<char> totalAnchorQualityScoresFlat{};
        std::vector<int> totalAnchorBeginInExtendedRead{};

        std::size_t sizeInBytes() const{
            std::size_t result = 0;
            result += 1;
            result += 1;
            result += 4;
            result += 4;
            result += 4;
            result += 4;
            result += 4;
            result += 4;
            result += 4;
            result += 4;
            result += 4;
            result += 4;
            result += 4;
            result += 4;
            result += 4 + sizeof(char) * decodedMateRevC.size();
            result += 4 + sizeof(char) * mateQualityScoresReversed.size();
            result += 4 + sizeof(int) * totalDecodedAnchorsLengths.size();
            result += 4 + sizeof(char) * totalDecodedAnchorsFlat.size();
            result += 4 + sizeof(char) * totalAnchorQualityScoresFlat.size();
            result += 4 + sizeof(int) * totalAnchorBeginInExtendedRead.size();
            return result;
        }

        bool isActive(int insertSize, int insertSizeStddev) const noexcept{
            return (iteration < insertSize 
                && accumExtensionLengths < insertSize - (mateLength) + insertSizeStddev
                && (abortReason == extension::AbortReason::None) 
                && !mateHasBeenFound);
        }

        bool operator==(const ExtensionTaskCpuData& rhs) const{
            if(pairedEnd != rhs.pairedEnd){
                return false;
            }
            if(mateHasBeenFound != rhs.mateHasBeenFound){
                return false;
            }
            if(id != rhs.id){
                return false;
            }
            if(pairId != rhs.pairId){
                return false;
            }
            if(myLength != rhs.myLength){
                return false;
            }
            if(mateLength != rhs.mateLength){
                return false;
            }
            if(currentAnchorLength != rhs.currentAnchorLength){
                return false;
            }
            if(accumExtensionLengths != rhs.accumExtensionLengths){
                return false;
            }
            if(iteration != rhs.iteration){
                return false;
            }
            if(goodscore != rhs.goodscore){
                return false;
            }
            if(myReadId != rhs.myReadId){
                return false;
            }
            if(mateReadId != rhs.mateReadId){
                return false;
            }
            if(abortReason != rhs.abortReason){
                return false;
            }
            if(direction != rhs.direction){
                return false;
            }
            if(decodedMateRevC != rhs.decodedMateRevC){
                return false;
            }
            if(mateQualityScoresReversed != rhs.mateQualityScoresReversed){
                return false;
            }
            if(totalDecodedAnchorsLengths != rhs.totalDecodedAnchorsLengths){
                return false;
            }
            if(totalDecodedAnchorsFlat != rhs.totalDecodedAnchorsFlat){
                return false;
            }
            if(totalAnchorBeginInExtendedRead != rhs.totalAnchorBeginInExtendedRead){
                return false;
            }

            return true;
        }
    };

    // struct ExtensionTaskCpuDataNonOwning{
    //     bool pairedEnd = false;
    //     bool mateHasBeenFound = false;
    //     int id = 0;
    //     int pairId = 0;
    //     int myLength = 0;
    //     int mateLength = 0;
    //     int currentAnchorLength = 0;
    //     int accumExtensionLengths = 0;
    //     int iteration = 0;
    //     float goodscore = 0.0f;
    //     read_number myReadId = 0;
    //     read_number mateReadId = 0;
    //     extension::AbortReason abortReason = extension::AbortReason::None;
    //     care::extension::ExtensionDirection direction{};
    //     std::vector<char>* decodedMateRevC{};
    //     std::vector<char>* mateQualityScoresReversed{};
    //     std::vector<int>* totalDecodedAnchorsLengths{};
    //     std::vector<char>* totalDecodedAnchorsFlat{};
    //     std::vector<char>* totalAnchorQualityScoresFlat{};
    //     std::vector<int>* totalAnchorBeginInExtendedRead{};

    //     std::size_t sizeInBytes() const{
    //         std::size_t result = 0;
    //         result += 1;
    //         result += 1;
    //         result += 4;
    //         result += 4;
    //         result += 4;
    //         result += 4;
    //         result += 4;
    //         result += 4;
    //         result += 4;
    //         result += 4;
    //         result += 4;
    //         result += 4;
    //         result += 4;
    //         result += 4;
    //         result += 4 + sizeof(char) * decodedMateRevC->size();
    //         result += 4 + sizeof(char) * mateQualityScoresReversed->size();
    //         result += 4 + sizeof(int) * totalDecodedAnchorsLengths->size();
    //         result += 4 + sizeof(char) * totalDecodedAnchorsFlat->size();
    //         result += 4 + sizeof(char) * totalAnchorQualityScoresFlat->size();
    //         result += 4 + sizeof(int) * totalAnchorBeginInExtendedRead->size();
    //         return result;
    //     }

    //     bool isActive(int insertSize, int insertSizeStddev) const noexcept{
    //         return (*iteration < insertSize 
    //             && *accumExtensionLengths < insertSize - (*mateLength) + insertSizeStddev
    //             && (*abortReason == extension::AbortReason::None) 
    //             && !*mateHasBeenFound);
    //     }
    // };

    struct SoAExtensionTaskCpuData{
        int entries = 0;
        std::vector<bool> pairedEnd{};
        std::vector<bool> mateHasBeenFound{};
        std::vector<int> id{};
        std::vector<int> pairId{};
        std::vector<int> myLength{};
        std::vector<int> mateLength{};
        std::vector<int> currentAnchorLength{};
        std::vector<int> accumExtensionLengths{};
        std::vector<int> iteration{};
        std::vector<float> goodscore{};
        std::vector<read_number> myReadId{};
        std::vector<read_number> mateReadId{};
        std::vector<extension::AbortReason> abortReason{};
        std::vector<extension::ExtensionDirection> direction{};
        std::vector<std::vector<char>> decodedMateRevC{};
        std::vector<std::vector<char>> mateQualityScoresReversed{};
        std::vector<std::vector<int>> totalDecodedAnchorsLengths{};
        std::vector<std::vector<char>> totalDecodedAnchorsFlat{};
        std::vector<std::vector<char>> totalAnchorQualityScoresFlat{};
        std::vector<std::vector<int>> totalAnchorBeginInExtendedRead{};

        SoAExtensionTaskCpuData() = default;
        SoAExtensionTaskCpuData(int size) {
            resize(size);
        }

        void resize(int newsize){
            pairedEnd.resize(newsize);
            mateHasBeenFound.resize(newsize);
            id.resize(newsize);
            pairId.resize(newsize);
            myLength.resize(newsize);
            mateLength.resize(newsize);
            currentAnchorLength.resize(newsize);
            accumExtensionLengths.resize(newsize);
            iteration.resize(newsize);
            goodscore.resize(newsize);
            myReadId.resize(newsize);
            mateReadId.resize(newsize);
            abortReason.resize(newsize);
            direction.resize(newsize);
            decodedMateRevC.resize(newsize);
            mateQualityScoresReversed.resize(newsize);
            totalDecodedAnchorsLengths.resize(newsize);
            totalDecodedAnchorsFlat.resize(newsize);
            totalAnchorQualityScoresFlat.resize(newsize);
            totalAnchorBeginInExtendedRead.resize(newsize);

            entries = newsize;
        }

        void append(const SoAExtensionTaskCpuData& rhs){
            pairedEnd.insert(pairedEnd.end(), rhs.pairedEnd.begin(), rhs.pairedEnd.end());
            mateHasBeenFound.insert(mateHasBeenFound.end(), rhs.mateHasBeenFound.begin(), rhs.mateHasBeenFound.end());
            id.insert(id.end(), rhs.id.begin(), rhs.id.end());
            pairId.insert(pairId.end(), rhs.pairId.begin(), rhs.pairId.end());
            myLength.insert(myLength.end(), rhs.myLength.begin(), rhs.myLength.end());
            mateLength.insert(mateLength.end(), rhs.mateLength.begin(), rhs.mateLength.end());
            currentAnchorLength.insert(currentAnchorLength.end(), rhs.currentAnchorLength.begin(), rhs.currentAnchorLength.end());
            accumExtensionLengths.insert(accumExtensionLengths.end(), rhs.accumExtensionLengths.begin(), rhs.accumExtensionLengths.end());
            iteration.insert(iteration.end(), rhs.iteration.begin(), rhs.iteration.end());
            goodscore.insert(goodscore.end(), rhs.goodscore.begin(), rhs.goodscore.end());
            myReadId.insert(myReadId.end(), rhs.myReadId.begin(), rhs.myReadId.end());
            mateReadId.insert(mateReadId.end(), rhs.mateReadId.begin(), rhs.mateReadId.end());
            abortReason.insert(abortReason.end(), rhs.abortReason.begin(), rhs.abortReason.end());
            direction.insert(direction.end(), rhs.direction.begin(), rhs.direction.end());
            decodedMateRevC.insert(decodedMateRevC.end(), rhs.decodedMateRevC.begin(), rhs.decodedMateRevC.end());
            mateQualityScoresReversed.insert(mateQualityScoresReversed.end(), rhs.mateQualityScoresReversed.begin(), rhs.mateQualityScoresReversed.end());
            totalDecodedAnchorsLengths.insert(totalDecodedAnchorsLengths.end(), rhs.totalDecodedAnchorsLengths.begin(), rhs.totalDecodedAnchorsLengths.end());
            totalDecodedAnchorsFlat.insert(totalDecodedAnchorsFlat.end(), rhs.totalDecodedAnchorsFlat.begin(), rhs.totalDecodedAnchorsFlat.end());
            totalAnchorQualityScoresFlat.insert(totalAnchorQualityScoresFlat.end(), rhs.totalAnchorQualityScoresFlat.begin(), rhs.totalAnchorQualityScoresFlat.end());
            totalAnchorBeginInExtendedRead.insert(totalAnchorBeginInExtendedRead.end(), rhs.totalAnchorBeginInExtendedRead.begin(), rhs.totalAnchorBeginInExtendedRead.end());

            entries = pairedEnd.size();
        }

        SoAExtensionTaskCpuData select(bool* selectionFlags) const{
            int numTrue = std::count_if(selectionFlags, selectionFlags + entries, thrust::identity<bool>{});

            SoAExtensionTaskCpuData selection(numTrue);

            auto inputScalars1Begin = thrust::make_zip_iterator(thrust::make_tuple(
                pairedEnd.begin(),
                mateHasBeenFound.begin(),
                id.begin(),
                pairId.begin(),
                myLength.begin(),
                mateLength.begin(),
                currentAnchorLength.begin(),
                accumExtensionLengths.begin()
            ));

            auto outputScalars1Begin = thrust::make_zip_iterator(thrust::make_tuple(
                selection.pairedEnd.begin(),
                selection.mateHasBeenFound.begin(),
                selection.id.begin(),
                selection.pairId.begin(),
                selection.myLength.begin(),
                selection.mateLength.begin(),
                selection.currentAnchorLength.begin(),
                selection.accumExtensionLengths.begin()
            ));            

            thrust::copy_if(
                inputScalars1Begin,
                inputScalars1Begin + entries,
                selectionFlags,
                outputScalars1Begin,
                thrust::identity<bool>{}
            );

            auto inputScalars2Begin = thrust::make_zip_iterator(thrust::make_tuple(
                iteration.begin(),
                goodscore.begin(),
                myReadId.begin(),
                mateReadId.begin(),
                abortReason.begin(),
                direction.begin()
            ));

            auto outputScalars2Begin = thrust::make_zip_iterator(thrust::make_tuple(
                selection.iteration.begin(),
                selection.goodscore.begin(),
                selection.myReadId.begin(),
                selection.mateReadId.begin(),
                selection.abortReason.begin(),
                selection.direction.begin()
            ));            

            thrust::copy_if(
                inputScalars2Begin,
                inputScalars2Begin + entries,
                selectionFlags,
                outputScalars2Begin,
                thrust::identity<bool>{}
            );

            auto inputVectors1Begin = thrust::make_zip_iterator(thrust::make_tuple(
                decodedMateRevC.begin(),
                mateQualityScoresReversed.begin(),
                totalDecodedAnchorsLengths.begin(),
                totalDecodedAnchorsFlat.begin(),
                totalAnchorQualityScoresFlat.begin(),
                totalAnchorBeginInExtendedRead.begin()
            ));

            auto outputVectors1Begin = thrust::make_zip_iterator(thrust::make_tuple(
                selection.decodedMateRevC.begin(),
                selection.mateQualityScoresReversed.begin(),
                selection.totalDecodedAnchorsLengths.begin(),
                selection.totalDecodedAnchorsFlat.begin(),
                selection.totalAnchorQualityScoresFlat.begin(),
                selection.totalAnchorBeginInExtendedRead.begin()
            ));

            thrust::copy_if(
                inputVectors1Begin,
                inputVectors1Begin + entries,
                selectionFlags,
                outputVectors1Begin,
                thrust::identity<bool>{}
            );

            return selection;
        }

        void sortByPairIdAndId(){
            std::vector<int> indices(entries);
            std::iota(indices.begin(), indices.end(), 0);

            thrust::sort(thrust::host, indices.data(), indices.data() + entries,
                [
                    pairIds = pairId.data(),
                    ids = id.data()
                ] (int l, int r){
                    if(pairIds[l] < pairIds[r]) return true;
                    if(pairIds[l] > pairIds[r]) return false;
                    return ids[l] < ids[r];
                }
            );

            if(!thrust::equal(indices.begin(), indices.end(), thrust::make_counting_iterator(0))){
                permute(indices.data());
            }
        }

        void permute(int* indices){
            SoAExtensionTaskCpuData selection(entries);

            auto inputScalars1Begin = thrust::make_zip_iterator(thrust::make_tuple(
                pairedEnd.begin(),
                mateHasBeenFound.begin(),
                id.begin(),
                pairId.begin(),
                myLength.begin(),
                mateLength.begin(),
                currentAnchorLength.begin(),
                accumExtensionLengths.begin()
            ));

            auto outputScalars1Begin = thrust::make_zip_iterator(thrust::make_tuple(
                selection.pairedEnd.begin(),
                selection.mateHasBeenFound.begin(),
                selection.id.begin(),
                selection.pairId.begin(),
                selection.myLength.begin(),
                selection.mateLength.begin(),
                selection.currentAnchorLength.begin(),
                selection.accumExtensionLengths.begin()
            ));            

            thrust::copy_n(
                thrust::make_permutation_iterator(inputScalars1Begin, indices),
                entries,
                outputScalars1Begin
            );

            auto inputScalars2Begin = thrust::make_zip_iterator(thrust::make_tuple(
                iteration.begin(),
                goodscore.begin(),
                myReadId.begin(),
                mateReadId.begin(),
                abortReason.begin(),
                direction.begin()
            ));

            auto outputScalars2Begin = thrust::make_zip_iterator(thrust::make_tuple(
                selection.iteration.begin(),
                selection.goodscore.begin(),
                selection.myReadId.begin(),
                selection.mateReadId.begin(),
                selection.abortReason.begin(),
                selection.direction.begin()
            ));            

            thrust::copy_n(
                thrust::make_permutation_iterator(inputScalars2Begin, indices),
                entries,
                outputScalars2Begin
            );

            auto inputVectors1Begin = thrust::make_zip_iterator(thrust::make_tuple(
                decodedMateRevC.begin(),
                mateQualityScoresReversed.begin(),
                totalDecodedAnchorsLengths.begin(),
                totalDecodedAnchorsFlat.begin(),
                totalAnchorQualityScoresFlat.begin(),
                totalAnchorBeginInExtendedRead.begin()
            ));

            auto outputVectors1Begin = thrust::make_zip_iterator(thrust::make_tuple(
                selection.decodedMateRevC.begin(),
                selection.mateQualityScoresReversed.begin(),
                selection.totalDecodedAnchorsLengths.begin(),
                selection.totalDecodedAnchorsFlat.begin(),
                selection.totalAnchorQualityScoresFlat.begin(),
                selection.totalAnchorBeginInExtendedRead.begin()
            ));

            thrust::copy_n(
                thrust::make_permutation_iterator(inputVectors1Begin, indices),
                entries,
                outputVectors1Begin
            );

            std::swap(selection, *this);
        }

        std::size_t sizeInBytes() const{
            std::size_t result = 0;
            for(int i = 0; i < entries; i++){
                result += 1;
                result += 1;
                result += 4;
                result += 4;
                result += 4;
                result += 4;
                result += 4;
                result += 4;
                result += 4;
                result += 4;
                result += 4;
                result += 4;
                result += 4;
                result += 4;
                result += 4 + sizeof(char) * decodedMateRevC[i].size();
                result += 4 + sizeof(char) * mateQualityScoresReversed[i].size();
                result += 4 + sizeof(int) * totalDecodedAnchorsLengths[i].size();
                result += 4 + sizeof(char) * totalDecodedAnchorsFlat[i].size();
                result += 4 + sizeof(char) * totalAnchorQualityScoresFlat[i].size();
                result += 4 + sizeof(int) * totalAnchorBeginInExtendedRead[i].size();
            }
            return result;
        }

        bool isActive(int which, int insertSize, int insertSizeStddev) const noexcept{
            return (iteration[which] < insertSize 
                && accumExtensionLengths[which] < insertSize - (mateLength[which]) + insertSizeStddev
                && (abortReason[which] == extension::AbortReason::None) 
                && !mateHasBeenFound[which]);
        }

        bool operator==(const SoAExtensionTaskCpuData& rhs) const{
            if(pairedEnd != rhs.pairedEnd){
                return false;
            }
            if(mateHasBeenFound != rhs.mateHasBeenFound){
                return false;
            }
            if(id != rhs.id){
                return false;
            }
            if(pairId != rhs.pairId){
                return false;
            }
            if(myLength != rhs.myLength){
                return false;
            }
            if(mateLength != rhs.mateLength){
                return false;
            }
            if(currentAnchorLength != rhs.currentAnchorLength){
                return false;
            }
            if(accumExtensionLengths != rhs.accumExtensionLengths){
                return false;
            }
            if(iteration != rhs.iteration){
                return false;
            }
            if(goodscore != rhs.goodscore){
                return false;
            }
            if(myReadId != rhs.myReadId){
                return false;
            }
            if(mateReadId != rhs.mateReadId){
                return false;
            }
            if(abortReason != rhs.abortReason){
                return false;
            }
            if(direction != rhs.direction){
                return false;
            }
            if(decodedMateRevC != rhs.decodedMateRevC){
                return false;
            }
            if(mateQualityScoresReversed != rhs.mateQualityScoresReversed){
                return false;
            }
            if(totalDecodedAnchorsLengths != rhs.totalDecodedAnchorsLengths){
                return false;
            }
            if(totalDecodedAnchorsFlat != rhs.totalDecodedAnchorsFlat){
                return false;
            }
            if(totalAnchorBeginInExtendedRead != rhs.totalAnchorBeginInExtendedRead){
                return false;
            }

            return true;
        }
    };

    enum class State{
        BeforeHash,
        BeforeRemoveIds,
        BeforeComputePairFlags,
        BeforeLoadCandidates,
        BeforeEraseData,
        BeforeAlignment,
        BeforeAlignmentFilter,
        BeforeMSA,
        BeforeExtend,
        BeforeUpdateUsedCandidateIds,
        BeforeUnpack,
        BeforePrepareNextIteration,
        Finished,
        None
    };

    static std::string to_string(State s){
        switch(s){
            case State::BeforeHash: return "BeforeHash";
            case State::BeforeRemoveIds: return "BeforeRemoveIds";
            case State::BeforeComputePairFlags: return "BeforeComputePairFlags";
            case State::BeforeLoadCandidates: return "BeforeLoadCandidates";
            case State::BeforeEraseData: return "BeforeEraseData";
            case State::BeforeAlignment: return "BeforeAlignment";
            case State::BeforeAlignmentFilter: return "BeforeAlignmentFilter";
            case State::BeforeMSA: return "BeforeMSA";
            case State::BeforeExtend: return "BeforeExtend";
            case State::BeforeUpdateUsedCandidateIds: return "BeforeUpdateUsedCandidateIds";
            case State::BeforeUnpack: return "BeforeUnpack";
            case State::BeforePrepareNextIteration: return "BeforePrepareNextIteration";
            case State::Finished: return "Finished";
            case State::None: return "None";
            default: return "Missing case GpuReadExtender::to_string(State)\n";
        };
    }

    bool isEmpty() const noexcept{
        return tasks.empty();
    }

    bool printTransitions = false;

    void setState(State newstate){      
        if(printTransitions){
            std::cerr << "batchdata " << someId << " statechange " << to_string(state) << " -> " << to_string(newstate);
            std::cerr << ", task: " << tasks.size() << ", finishedTasks: " << finishedTasks.size();
            std::cerr << "\n";
        }

        state = newstate;
    }



    GpuReadExtender(
        bool isPairedEnd_,
        const gpu::GpuReadStorage& rs, 
        const gpu::GpuMinhasher& gpuMinhasher_,
        const CorrectionOptions& coropts,
        const GoodAlignmentProperties& gap,
        const cpu::QualityScoreConversion& qualityConversion_,
        int insertSize_,
        int insertSizeStddev_,
        int maxextensionPerStep_,
        std::array<cudaStream_t, 4> streams_,
        cub::CachingDeviceAllocator& cubAllocator_
    ) : 
        pairedEnd(isPairedEnd_),
        insertSize(insertSize_),
        insertSizeStddev(insertSizeStddev_),
        maxextensionPerStep(maxextensionPerStep_),
        cubAllocator(&cubAllocator_),
        gpuReadStorage(&rs),
        gpuMinhasher(&gpuMinhasher_),
        minhashHandle(gpuMinhasher->makeMinhasherHandle()),
        correctionOptions(&coropts),
        goodAlignmentProperties(&gap),
        qualityConversion(&qualityConversion_),
        readStorageHandle(gpuReadStorage->makeHandle()),
        d_mateIdHasBeenRemoved(cubAllocator_),
        d_candidateSequencesData(cubAllocator_),
        d_candidateSequencesLength(cubAllocator_),    
        d_candidateReadIds(cubAllocator_),
        d_isPairedCandidate(cubAllocator_),
        d_segmentIdsOfCandidates(cubAllocator_),
        d_alignment_overlaps(cubAllocator_),
        d_alignment_shifts(cubAllocator_),
        d_alignment_nOps(cubAllocator_),
        d_alignment_best_alignment_flags(cubAllocator_),
        d_numCandidatesPerAnchor(cubAllocator_),
        d_numCandidatesPerAnchorPrefixSum(cubAllocator_),
        d_inputanchormatedata(cubAllocator_),
        d_subjectSequencesDataDecoded(cubAllocator_),
        d_anchorQualityScores(cubAllocator_),
        d_anchorSequencesLength(cubAllocator_),
        d_anchorReadIds(cubAllocator_),
        d_mateReadIds(cubAllocator_),
        d_inputMateLengths(cubAllocator_),
        d_isPairedTask(cubAllocator_),
        d_subjectSequencesData(cubAllocator_),
        d_accumExtensionsLengths(cubAllocator_),
        d_usedReadIds(cubAllocator_),
        d_numUsedReadIdsPerAnchor(cubAllocator_),
        d_numUsedReadIdsPerAnchorPrefixSum(cubAllocator_),
        d_segmentIdsOfUsedReadIds(cubAllocator_),
        d_fullyUsedReadIds(cubAllocator_),
        d_numFullyUsedReadIdsPerAnchor(cubAllocator_),
        d_numFullyUsedReadIdsPerAnchorPrefixSum(cubAllocator_),
        d_segmentIdsOfFullyUsedReadIds(cubAllocator_),
        multiMSA(cubAllocator_),
        d_consensusQuality(cubAllocator_),
        d_outputAnchors(cubAllocator_),
        d_outputAnchorQualities(cubAllocator_),
        d_outputMateHasBeenFound(cubAllocator_),
        d_abortReasons(cubAllocator_),
        d_outputAnchorLengths(cubAllocator_),
        d_isFullyUsedCandidate(cubAllocator_),
        streams(streams_)
    {

        cudaGetDevice(&deviceId); CUERR;

        kernelLaunchHandle = gpu::make_kernel_launch_handle(deviceId);

        h_numUsedReadIds.resize(1);
        h_numFullyUsedReadIds.resize(1);
        h_numAnchors.resize(1);
        h_numCandidates.resize(1);
        h_numAnchorsWithRemovedMates.resize(1);
        h_numFullyUsedReadIds2.resize(1);

        d_numAnchors.resize(1);
        d_numCandidates.resize(1);
        d_numCandidates2.resize(1);

        *h_numUsedReadIds = 0;
        *h_numFullyUsedReadIds = 0;
        *h_numAnchors = 0;
        *h_numCandidates = 0;
        *h_numAnchorsWithRemovedMates = 0;
        *h_numFullyUsedReadIds2 = 0;

        numTasks = 0;   
    }

    ~GpuReadExtender(){
        gpuMinhasher->destroyHandle(minhashHandle);
    }

    static constexpr int getNumRefinementIterations() noexcept{
        return 5;
    }

    template<class TaskIter>
    void addTasks(TaskIter extraTasksBegin, TaskIter extraTasksEnd){
        const int numAdditionalTasks = std::distance(extraTasksBegin, extraTasksEnd);
        assert(numAdditionalTasks % 4 == 0);
        if(numAdditionalTasks == 0) return;

        auto comp = [](const auto& l, const auto& r){
            return std::tie(l.pairId, l.id) < std::tie(r.pairId, r.id);
        };
        assert(std::is_sorted(extraTasksBegin, extraTasksEnd, comp));

        const int currentNumTasks = tasks.size();
        const int newNumTasks = currentNumTasks + numAdditionalTasks;

        h_anchorReadIds.resize(newNumTasks);
        h_mateReadIds.resize(newNumTasks);
        h_subjectSequencesDataDecoded.resize(newNumTasks * decodedSequencePitchInBytes);
        h_anchorSequencesLength.resize(newNumTasks);
        h_anchorQualityScores.resize(newNumTasks * qualityPitchInBytes);
        h_inputanchormatedata.resize(newNumTasks * encodedSequencePitchInInts);
        h_inputMateLengths.resize(newNumTasks);
        h_isPairedTask.resize(newNumTasks);

        d_numUsedReadIdsPerAnchor.resize(newNumTasks, streams[0]);
        d_numFullyUsedReadIdsPerAnchor.resize(newNumTasks, streams[0]);
        cudaMemsetAsync(d_numUsedReadIdsPerAnchor.data() + currentNumTasks, 0, numAdditionalTasks * sizeof(int), streams[0]); CUERR;
        cudaMemsetAsync(d_numFullyUsedReadIdsPerAnchor.data() + currentNumTasks, 0, numAdditionalTasks * sizeof(int), streams[0]); CUERR;


        d_numUsedReadIdsPerAnchorPrefixSum.resizeUninitialized(newNumTasks, streams[0]);
        d_numFullyUsedReadIdsPerAnchorPrefixSum.resizeUninitialized(newNumTasks, streams[0]);

        cubExclusiveSum(
            d_numUsedReadIdsPerAnchor.data(), 
            d_numUsedReadIdsPerAnchorPrefixSum.data(), 
            newNumTasks, 
            streams[0]
        );

        cubExclusiveSum(
            d_numFullyUsedReadIdsPerAnchor.data(), 
            d_numFullyUsedReadIdsPerAnchorPrefixSum.data(), 
            newNumTasks, 
            streams[0]
        );

        //copy input task data to pinned buffers

        for(int t = 0; t < numAdditionalTasks; t++){
            const auto& task = *(extraTasksBegin + t);

            h_anchorReadIds[t] = task.myReadId;
            h_mateReadIds[t] = task.mateReadId;

            std::copy(
                task.encodedMate.begin(),
                task.encodedMate.end(),
                h_inputanchormatedata.begin() + t * encodedSequencePitchInInts
            );

            h_anchorSequencesLength[t] = task.currentAnchorLength;

            std::copy(
                task.totalDecodedAnchors.back().begin(),
                task.totalDecodedAnchors.back().end(),
                h_subjectSequencesDataDecoded.begin() + t * decodedSequencePitchInBytes
            );

            assert(h_anchorQualityScores.size() >= (t+1) * qualityPitchInBytes);

            std::copy(
                task.totalAnchorQualityScores.back().begin(),
                task.totalAnchorQualityScores.back().end(),
                h_anchorQualityScores.begin() + t * qualityPitchInBytes
            );

            h_inputMateLengths[t] = task.mateLength;
            h_isPairedTask[t] = task.pairedEnd;
        }

        d_subjectSequencesData.resize(newNumTasks * encodedSequencePitchInInts, streams[0]);
        d_anchorSequencesLength.resize(newNumTasks, streams[0]);
        d_anchorQualityScores.resize(newNumTasks * qualityPitchInBytes, streams[0]);
        d_inputanchormatedata.resize(newNumTasks * encodedSequencePitchInInts, streams[0]);
        d_inputMateLengths.resize(newNumTasks, streams[0]);
        d_isPairedTask.resize(newNumTasks, streams[0]);
        d_anchorReadIds.resize(newNumTasks, streams[0]);
        d_mateReadIds.resize(newNumTasks, streams[0]);
        d_accumExtensionsLengths.resize(newNumTasks, streams[0]);

        cudaMemcpyAsync(
            d_inputanchormatedata.data() + currentNumTasks * encodedSequencePitchInInts,
            h_inputanchormatedata.data(),
            sizeof(unsigned int) * encodedSequencePitchInInts * numAdditionalTasks,
            H2D,
            streams[0]
        ); CUERR;

        cudaMemcpyAsync(
            d_anchorSequencesLength.data() + currentNumTasks,
            h_anchorSequencesLength.data(),
            sizeof(int) * numAdditionalTasks,
            H2D,
            streams[0]
        ); CUERR;

        cudaMemcpyAsync(
            d_anchorReadIds.data() + currentNumTasks,
            h_anchorReadIds.data(),
            sizeof(read_number) * numAdditionalTasks,
            H2D,
            streams[0]
        ); CUERR;

        cudaMemcpyAsync(
            d_mateReadIds.data() + currentNumTasks,
            h_mateReadIds.data(),
            sizeof(read_number) * numAdditionalTasks,
            H2D,
            streams[0]
        ); CUERR;

        cudaMemcpyAsync(
            d_inputMateLengths.data() + currentNumTasks,
            h_inputMateLengths.data(),
            sizeof(int) * numAdditionalTasks,
            H2D,
            streams[0]
        ); CUERR;

        cudaMemcpyAsync(
            d_isPairedTask.data() + currentNumTasks,
            h_isPairedTask.data(),
            sizeof(bool) * numAdditionalTasks,
            H2D,
            streams[0]
        ); CUERR;

        helpers::call_fill_kernel_async(
            d_accumExtensionsLengths.data() + currentNumTasks,
            numAdditionalTasks,
            0,
            streams[0]
        ); CUERR;

        d_subjectSequencesDataDecoded.resize(newNumTasks * decodedSequencePitchInBytes, streams[0]);
        cudaMemcpyAsync(
            d_subjectSequencesDataDecoded.data() + currentNumTasks * decodedSequencePitchInBytes,
            h_subjectSequencesDataDecoded.data(),
            sizeof(char) * numAdditionalTasks * decodedSequencePitchInBytes,
            H2D,
            streams[0]
        ); CUERR;

        cudaMemcpyAsync(
            d_anchorQualityScores.data() + currentNumTasks * qualityPitchInBytes,
            h_anchorQualityScores.data(),
            sizeof(char) * numAdditionalTasks * qualityPitchInBytes,
            H2D,
            streams[0]
        ); CUERR;

        readextendergpukernels::encodeSequencesTo2BitKernel<8>
        <<<SDIV(numAdditionalTasks, (128 / 8)), 128, 0, streams[0]>>>(
            d_subjectSequencesData.data() + currentNumTasks * encodedSequencePitchInInts,
            d_subjectSequencesDataDecoded.data() + currentNumTasks * decodedSequencePitchInBytes,
            d_anchorSequencesLength.data() + currentNumTasks,
            decodedSequencePitchInBytes,
            encodedSequencePitchInInts,
            numAdditionalTasks
        ); CUERR;


        //init flat arrays
        for(auto it = extraTasksBegin; it != extraTasksEnd; ++it){

            const int expectedNumIterations = 2 + (insertSize / maxextensionPerStep);
            
            it->totalDecodedAnchorsFlat.reserve(expectedNumIterations * decodedSequencePitchInBytes);
            it->totalDecodedAnchorsFlat.resize(decodedSequencePitchInBytes);
            std::copy(
                it->totalDecodedAnchors.back().begin(),
                it->totalDecodedAnchors.back().end(),
                it->totalDecodedAnchorsFlat.begin()
            );

            it->totalDecodedAnchorsLengths.reserve(expectedNumIterations);
            it->totalDecodedAnchorsLengths.emplace_back(it->totalDecodedAnchors.back().size());

            it->totalAnchorQualityScoresFlat.reserve(expectedNumIterations * qualityPitchInBytes);
            it->totalAnchorQualityScoresFlat.resize(qualityPitchInBytes);
            std::copy(
                it->totalAnchorQualityScores.back().begin(),
                it->totalAnchorQualityScores.back().end(),
                it->totalAnchorQualityScoresFlat.begin()
            );
        }

        //save tasks and update indices of active tasks

        tasks.reserve(tasks.size() + numAdditionalTasks);

        for(auto it = extraTasksBegin; it != extraTasksEnd; ++it){
            ExtensionTaskCpuData data;
            data.pairedEnd = it->pairedEnd;
            data.mateHasBeenFound = it->mateHasBeenFound;
            data.id = it->id;
            data.pairId = it->pairId;
            data.myLength = it->myLength;
            data.mateLength = it->mateLength;
            data.currentAnchorLength = it->currentAnchorLength;
            data.accumExtensionLengths = it->accumExtensionLengths;
            data.iteration = it->iteration;
            data.goodscore = it->goodscore;
            data.myReadId = it->myReadId;
            data.mateReadId = it->mateReadId;
            data.abortReason = it->abortReason;
            data.direction = it->direction;
            data.decodedMateRevC.insert(data.decodedMateRevC.begin(), it->decodedMateRevC.begin(), it->decodedMateRevC.end());
            data.mateQualityScoresReversed.insert(data.mateQualityScoresReversed.begin(), it->mateQualityScoresReversed.begin(), it->mateQualityScoresReversed.end());
            data.totalDecodedAnchorsLengths = std::move(it->totalDecodedAnchorsLengths);
            data.totalDecodedAnchorsFlat = std::move(it->totalDecodedAnchorsFlat);
            data.totalAnchorQualityScoresFlat = std::move(it->totalAnchorQualityScoresFlat);
            data.totalAnchorBeginInExtendedRead = std::move(it->totalAnchorBeginInExtendedRead);

            tasks.push_back(std::move(data));
        }

        //tasks.insert(tasks.end(), std::make_move_iterator(extraTasksBegin), std::make_move_iterator(extraTasksEnd));        

        numTasks = tasks.size();

        alltimeMaximumNumberOfTasks = std::max(alltimeMaximumNumberOfTasks, numTasks);

        state = State::BeforeHash;
    }

    void addTasks(
        int numReadPairs,
        // for the arrays, two consecutive numbers / sequences belong to same read pair
        const read_number* d_readpair_readIds,
        const int* d_readpair_readLengths,
        const unsigned int * d_readpair_sequences,
        const char* d_readpair_qualities,
        cudaStream_t stream
    ){
        if(numReadPairs == 0) return;

        cudaEventRecord(events[0], stream); CUERR;
        cudaStreamWaitEvent(streams[0], events[0], 0); CUERR;
        cudaStreamWaitEvent(streams[1], events[0], 0); CUERR;

        const int numAdditionalTasks = 4 * numReadPairs;
        const int currentNumTasks = tasks.size();
        const int newNumTasks = currentNumTasks + numAdditionalTasks;

        h_anchorReadIds.resize(2 * numReadPairs);
        h_anchorSequencesLength.resize(2 * numReadPairs);
        h_anchorQualityScores.resize(2 * numReadPairs * qualityPitchInBytes);

        cudaMemcpyAsync(
            h_anchorReadIds.data(),
            d_readpair_readIds,
            sizeof(read_number) * 2 * numReadPairs,
            D2H,
            streams[1]
        ); CUERR;

        cudaMemcpyAsync(
            h_anchorSequencesLength.data(),
            d_readpair_readLengths,
            sizeof(int) * 2 * numReadPairs,
            D2H,
            streams[1]
        ); CUERR;

        cudaMemcpyAsync(
            h_anchorQualityScores.data(),
            d_readpair_qualities,
            sizeof(char) * 2 * numReadPairs * qualityPitchInBytes,
            D2H,
            streams[1]
        ); CUERR;

        // set used ids to 0 for new tasks
        d_numUsedReadIdsPerAnchor.resize(newNumTasks, streams[0]);
        d_numFullyUsedReadIdsPerAnchor.resize(newNumTasks, streams[0]);
        cudaMemsetAsync(d_numUsedReadIdsPerAnchor.data() + currentNumTasks, 0, numAdditionalTasks * sizeof(int), streams[0]); CUERR;
        cudaMemsetAsync(d_numFullyUsedReadIdsPerAnchor.data() + currentNumTasks, 0, numAdditionalTasks * sizeof(int), streams[0]); CUERR;

        d_numUsedReadIdsPerAnchorPrefixSum.resizeUninitialized(newNumTasks, streams[0]);
        d_numFullyUsedReadIdsPerAnchorPrefixSum.resizeUninitialized(newNumTasks, streams[0]);

        cubExclusiveSum(
            d_numUsedReadIdsPerAnchor.data(), 
            d_numUsedReadIdsPerAnchorPrefixSum.data(), 
            newNumTasks, 
            streams[0]
        );

        cubExclusiveSum(
            d_numFullyUsedReadIdsPerAnchor.data(), 
            d_numFullyUsedReadIdsPerAnchorPrefixSum.data(), 
            newNumTasks, 
            streams[0]
        );

        d_subjectSequencesData.resize(newNumTasks * encodedSequencePitchInInts, streams[0]);
        d_anchorSequencesLength.resize(newNumTasks, streams[0]);
        d_anchorQualityScores.resize(newNumTasks * qualityPitchInBytes, streams[0]);
        d_inputanchormatedata.resize(newNumTasks * encodedSequencePitchInInts, streams[0]);
        d_inputMateLengths.resize(newNumTasks, streams[0]);
        d_isPairedTask.resize(newNumTasks, streams[0]);
        
        d_anchorReadIds.resize(newNumTasks, streams[0]);
        d_mateReadIds.resize(newNumTasks, streams[0]);
        d_accumExtensionsLengths.resize(newNumTasks, streams[0]);
        d_subjectSequencesDataDecoded.resize(newNumTasks * decodedSequencePitchInBytes, streams[0]);

        //CachedDeviceUVector<unsigned int> d_inputanchormatedataTmp(newNumTasks * encodedSequencePitchInInts, streams[0], *cubAllocator);

        readextendergpukernels::createGpuTaskData<128,8>
            <<<SDIV(numAdditionalTasks, (128 / 8)), 128, 0, streams[0]>>>(
            numReadPairs,
            d_readpair_readIds,
            d_readpair_readLengths,
            d_readpair_sequences,
            d_readpair_qualities,
            d_subjectSequencesData.data() + currentNumTasks * encodedSequencePitchInInts,
            d_anchorSequencesLength.data() + currentNumTasks,
            d_anchorQualityScores.data() + currentNumTasks * decodedSequencePitchInBytes,
            d_inputanchormatedata.data() + currentNumTasks * encodedSequencePitchInInts,
            d_inputMateLengths.data() + currentNumTasks,
            d_isPairedTask.data() + currentNumTasks,
            d_anchorReadIds.data() + currentNumTasks,
            d_mateReadIds.data() + currentNumTasks,
            d_accumExtensionsLengths.data() + currentNumTasks,
            decodedSequencePitchInBytes,
            qualityPitchInBytes,
            encodedSequencePitchInInts
        ); CUERR;

        readextendergpukernels::decodeSequencesFrom2BitKernel<8>
            <<<SDIV(numAdditionalTasks, (128 / 8)), 128, 0, streams[0]>>>(
            d_subjectSequencesDataDecoded.data() + currentNumTasks * decodedSequencePitchInBytes,
            d_subjectSequencesData.data() + currentNumTasks * encodedSequencePitchInInts,
            d_anchorSequencesLength.data() + currentNumTasks,
            decodedSequencePitchInBytes,
            encodedSequencePitchInInts,
            numAdditionalTasks
        ); CUERR;

        h_subjectSequencesDataDecoded.resize(numAdditionalTasks * decodedSequencePitchInBytes);

        cudaMemcpyAsync(
            h_subjectSequencesDataDecoded.data(),
            d_subjectSequencesDataDecoded.data() + currentNumTasks * decodedSequencePitchInBytes,
            sizeof(char) * numAdditionalTasks * decodedSequencePitchInBytes,
            D2H,
            streams[0]
        ); CUERR;

        std::vector<ExtensionTaskCpuData> newTaskData(numAdditionalTasks);
        tasks.reserve(tasks.size() + numAdditionalTasks);

        cudaStreamSynchronize(streams[1]); CUERR;

        for(int t = 0; t < numAdditionalTasks; t++){
            ExtensionTaskCpuData& data = newTaskData[t];
            const int groupId = t / 4;
            const int id = t % 4;

            if(id == 0){
                data.myReadId = h_anchorReadIds[2 * groupId + 0];
                data.mateReadId = h_anchorReadIds[2 * groupId + 1];
                data.myLength = h_anchorSequencesLength[2 * groupId + 0];
                data.mateLength = h_anchorSequencesLength[2 * groupId + 1];
                data.pairedEnd = true;
                data.direction = extension::ExtensionDirection::LR;

                data.totalAnchorQualityScoresFlat.resize(qualityPitchInBytes);
                std::copy(
                    h_anchorQualityScores + (2 * groupId) * qualityPitchInBytes,
                    h_anchorQualityScores + (2 * groupId) * qualityPitchInBytes + data.myLength,
                    data.totalAnchorQualityScoresFlat.begin()
                );

                data.mateQualityScoresReversed.resize(data.mateLength);
                std::reverse_copy(
                    h_anchorQualityScores + (2 * groupId + 1) * qualityPitchInBytes,
                    h_anchorQualityScores + (2 * groupId + 1) * qualityPitchInBytes + data.mateLength,
                    data.mateQualityScoresReversed.begin()
                );
            }else if(id == 1){
                data.myReadId = h_anchorReadIds[2 * groupId + 1];
                data.mateReadId = std::numeric_limits<read_number>::max();
                data.myLength = h_anchorSequencesLength[2 * groupId + 1];
                data.mateLength = 0;
                data.pairedEnd = false;
                data.direction = extension::ExtensionDirection::LR;

                data.totalAnchorQualityScoresFlat.resize(qualityPitchInBytes);
                std::reverse_copy(
                    h_anchorQualityScores + (2 * groupId + 1) * qualityPitchInBytes,
                    h_anchorQualityScores + (2 * groupId + 1) * qualityPitchInBytes + data.myLength,
                    data.totalAnchorQualityScoresFlat.begin()
                );

                data.mateQualityScoresReversed.clear();
            }else if(id == 2){
                data.myReadId = h_anchorReadIds[2 * groupId + 1];
                data.mateReadId = h_anchorReadIds[2 * groupId + 0];
                data.myLength = h_anchorSequencesLength[2 * groupId + 1];
                data.mateLength = h_anchorSequencesLength[2 * groupId + 0];
                data.pairedEnd = true;
                data.direction = extension::ExtensionDirection::RL;

                data.totalAnchorQualityScoresFlat.resize(qualityPitchInBytes);
                std::copy(
                    h_anchorQualityScores + (2 * groupId + 1) * qualityPitchInBytes,
                    h_anchorQualityScores + (2 * groupId + 1) * qualityPitchInBytes + data.myLength,
                    data.totalAnchorQualityScoresFlat.begin()
                );

                data.mateQualityScoresReversed.resize(data.mateLength);
                std::reverse_copy(
                    h_anchorQualityScores + (2 * groupId) * qualityPitchInBytes,
                    h_anchorQualityScores + (2 * groupId) * qualityPitchInBytes + data.mateLength,
                    data.mateQualityScoresReversed.begin()
                );
            }else{
                //id == 3
                data.myReadId = h_anchorReadIds[2 * groupId + 0];
                data.mateReadId = std::numeric_limits<read_number>::max();
                data.myLength = h_anchorSequencesLength[2 * groupId + 0];
                data.mateLength = 0;
                data.pairedEnd =  false;
                data.direction = extension::ExtensionDirection::RL;

                data.totalAnchorQualityScoresFlat.resize(qualityPitchInBytes);
                std::reverse_copy(
                    h_anchorQualityScores + (2 * groupId) * qualityPitchInBytes,
                    h_anchorQualityScores + (2 * groupId) * qualityPitchInBytes + data.myLength,
                    data.totalAnchorQualityScoresFlat.begin()
                );

                data.mateQualityScoresReversed.clear();
            }

            data.mateHasBeenFound = false;
            data.id = id;
            data.pairId = data.myReadId / 2;
            data.currentAnchorLength = data.myLength;
            data.accumExtensionLengths = 0;
            data.iteration = 0;
            data.goodscore = 0.0f;
            data.abortReason = extension::AbortReason::None;
            
            data.totalDecodedAnchorsLengths.push_back(data.myLength);
            data.totalAnchorBeginInExtendedRead.push_back(0);
        }

        cudaStreamSynchronize(streams[0]); CUERR; //wait for decoded anchor sequences transfer

        for(int t = 0; t < numAdditionalTasks; t++){
            ExtensionTaskCpuData& data = newTaskData[t];
            const int id = t % 4;

            if(id == 0){
                data.decodedMateRevC.resize(data.mateLength);
                std::copy(
                    h_subjectSequencesDataDecoded.data() + (t + 1) * decodedSequencePitchInBytes,
                    h_subjectSequencesDataDecoded.data() + (t + 1) * decodedSequencePitchInBytes + data.mateLength,
                    data.decodedMateRevC.begin()
                );

                data.totalDecodedAnchorsFlat.resize(decodedSequencePitchInBytes);
                std::copy(
                    h_subjectSequencesDataDecoded.data() + (t) * decodedSequencePitchInBytes,
                    h_subjectSequencesDataDecoded.data() + (t) * decodedSequencePitchInBytes + data.myLength,
                    data.totalDecodedAnchorsFlat.begin()
                );
            }else if(id == 1){
                data.decodedMateRevC.clear();

                data.totalDecodedAnchorsFlat.resize(decodedSequencePitchInBytes);
                std::copy(
                    h_subjectSequencesDataDecoded.data() + (t) * decodedSequencePitchInBytes,
                    h_subjectSequencesDataDecoded.data() + (t) * decodedSequencePitchInBytes + data.myLength,
                    data.totalDecodedAnchorsFlat.begin()
                );
            }else if(id == 2){
                data.decodedMateRevC.resize(data.mateLength);
                std::copy(
                    h_subjectSequencesDataDecoded.data() + (t + 1) * decodedSequencePitchInBytes,
                    h_subjectSequencesDataDecoded.data() + (t + 1) * decodedSequencePitchInBytes + data.mateLength,
                    data.decodedMateRevC.begin()
                );

                data.totalDecodedAnchorsFlat.resize(decodedSequencePitchInBytes);
                std::copy(
                    h_subjectSequencesDataDecoded.data() + (t) * decodedSequencePitchInBytes,
                    h_subjectSequencesDataDecoded.data() + (t) * decodedSequencePitchInBytes + data.myLength,
                    data.totalDecodedAnchorsFlat.begin()
                );
            }else{
                data.decodedMateRevC.clear();

                data.totalDecodedAnchorsFlat.resize(decodedSequencePitchInBytes);
                std::copy(
                    h_subjectSequencesDataDecoded.data() + (t) * decodedSequencePitchInBytes,
                    h_subjectSequencesDataDecoded.data() + (t) * decodedSequencePitchInBytes + data.myLength,
                    data.totalDecodedAnchorsFlat.begin()
                );
            }
        }

        tasks.insert(tasks.end(), std::make_move_iterator(newTaskData.begin()), std::make_move_iterator(newTaskData.end()));

        numTasks = tasks.size();
        alltimeMaximumNumberOfTasks = std::max(alltimeMaximumNumberOfTasks, numTasks);

        state = State::BeforeHash;
    }

    void setMaxExtensionPerStep(int e) noexcept{
        maxextensionPerStep = e;
    }

    void setMinCoverageForExtension(int c) noexcept{
        minCoverageForExtension = c;
    }

    void process(){
        assert(state == GpuReadExtender::State::BeforeHash);

        while(state != GpuReadExtender::State::Finished){
            performNextStep();
        }
    }

    void processOneIteration(){
        assert(state == GpuReadExtender::State::BeforeHash || state == GpuReadExtender::State::Finished);

        while(state != GpuReadExtender::State::Finished){
            performNextStep();

            if(state == GpuReadExtender::State::BeforeHash){
                break;
            }
        }
    }

    void performNextStep(){
        const auto name = GpuReadExtender::to_string(state);

        nvtx::push_range(name, static_cast<int>(state));

        switch(state){
            case GpuReadExtender::State::BeforeHash: getCandidateReadIds(); break;
            case GpuReadExtender::State::BeforeRemoveIds: removeUsedIdsAndMateIds(); break;
            case GpuReadExtender::State::BeforeComputePairFlags: computePairFlagsGpu(); break;
            case GpuReadExtender::State::BeforeLoadCandidates: loadCandidateSequenceData(); break;
            case GpuReadExtender::State::BeforeEraseData: eraseDataOfRemovedMates(); break;
            case GpuReadExtender::State::BeforeAlignment: calculateAlignments(); break;
            case GpuReadExtender::State::BeforeAlignmentFilter: filterAlignments(); break;
            case GpuReadExtender::State::BeforeMSA: computeMSAs(); break;
            case GpuReadExtender::State::BeforeExtend: computeExtendedSequencesFromMSAs(); break;
            case GpuReadExtender::State::BeforeUpdateUsedCandidateIds: updateUsedCandidateIds(); break;
            case GpuReadExtender::State::BeforeUnpack: unpackResultsIntoTasks(); break;
            case GpuReadExtender::State::BeforePrepareNextIteration: prepareNextIteration(); break;
            case GpuReadExtender::State::Finished: break;
            case GpuReadExtender::State::None: break;
            default: break;
        };

        if(state == GpuReadExtender::State::Finished){
            assert(tasks.size() == 0);
            assert(finishedTasks.size() % 4 == 0);
        }

        nvtx::pop_range();
    }

    void getCandidateReadIds(){
        assert(state == GpuReadExtender::State::BeforeHash);

        cudaStream_t stream = streams[0];

        d_numCandidatesPerAnchor.resizeUninitialized(numTasks, stream);
        d_numCandidatesPerAnchorPrefixSum.resizeUninitialized(numTasks + 1, stream);

        int totalNumValues = 0;

        gpuMinhasher->determineNumValues(
            minhashHandle,
            d_subjectSequencesData.data(),
            encodedSequencePitchInInts,
            d_anchorSequencesLength.data(),
            numTasks,
            d_numCandidatesPerAnchor.data(),
            totalNumValues,
            stream
        );

        cudaStreamSynchronize(stream); CUERR;

        d_candidateReadIds.resizeUninitialized(totalNumValues, stream);    

        if(totalNumValues == 0){
            cudaMemsetAsync(d_numCandidatesPerAnchor.data(), 0, sizeof(int) * numTasks , stream); CUERR;
            cudaMemsetAsync(d_numCandidatesPerAnchorPrefixSum.data(), 0, sizeof(int) * (1 + numTasks), stream); CUERR;
            totalNumCandidates = 0;

            setStateToFinished();
            return;
        }

        gpuMinhasher->retrieveValues(
            minhashHandle,
            nullptr,
            numTasks,              
            totalNumValues,
            d_candidateReadIds.data(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            stream
        );

        cudaMemcpyAsync(
            h_numCandidates.data(),
            d_numCandidatesPerAnchorPrefixSum.data() + numTasks,
            sizeof(int),
            D2H,
            stream
        ); CUERR;

        cudaStreamSynchronize(stream); CUERR;

        totalNumCandidates = *h_numCandidates;

        setState(GpuReadExtender::State::BeforeRemoveIds);
    }

    void removeUsedIdsAndMateIds(){
        assert(state == GpuReadExtender::State::BeforeRemoveIds);

        cudaStream_t firstStream = streams[0];

        CachedDeviceUVector<read_number> d_candidateReadIds2(totalNumCandidates, firstStream, *cubAllocator);

        h_numCandidatesPerAnchor.resize(numTasks);

        //determine required temp bytes for following cub calls, and allocate temp storage
               
        CachedDeviceUVector<bool> d_shouldBeKept(totalNumCandidates, firstStream, *cubAllocator);
        CachedDeviceUVector<int> d_numCandidatesPerAnchor2(numTasks, firstStream, *cubAllocator);        

        d_mateIdHasBeenRemoved.resizeUninitialized(numTasks, firstStream);

        helpers::call_fill_kernel_async(d_shouldBeKept.data(), totalNumCandidates, false, firstStream);

        
        //flag candidates ids to remove because they are equal to anchor id or equal to mate id
        readextendergpukernels::flagCandidateIdsWhichAreEqualToAnchorOrMateKernel<<<numTasks, 128, 0, firstStream>>>(
            d_candidateReadIds.data(),
            d_anchorReadIds.data(),
            d_mateReadIds.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_numCandidatesPerAnchor.data(),
            d_shouldBeKept.data(),
            d_mateIdHasBeenRemoved.data(),
            d_numCandidatesPerAnchor2.data(),
            numTasks,
            pairedEnd
        );
        CUERR;

        cudaMemcpyAsync(
            h_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchor2.data(),
            sizeof(int) * numTasks,
            D2H,
            firstStream
        ); CUERR;       

        //copy selected candidate ids

        assert(d_candidateReadIds2.data() != nullptr);
        assert(h_numCandidates.data() != nullptr);

        assert(d_candidateReadIds.size() >= totalNumCandidates);

        cubSelectFlagged(
            d_candidateReadIds.data(),
            d_shouldBeKept.data(),
            d_candidateReadIds2.data(),
            h_numCandidates.data(),
            totalNumCandidates,
            firstStream
        );

        d_shouldBeKept.destroy();

        cudaStreamSynchronize(firstStream); CUERR; //wait for h_numCandidates
        totalNumCandidates = *h_numCandidates;

        d_candidateReadIds2.erase(d_candidateReadIds2.begin() + *h_numCandidates, d_candidateReadIds2.end(), firstStream);

        CachedDeviceUVector<int> d_numCandidatesPerAnchorPrefixSum2(numTasks + 1, firstStream, *cubAllocator);

        //compute prefix sum of number of candidates per anchor
        cudaMemsetAsync(d_numCandidatesPerAnchorPrefixSum2.data(), 0, sizeof(int), firstStream); CUERR;

        cubInclusiveSum(
            d_numCandidatesPerAnchor2.data(), 
            d_numCandidatesPerAnchorPrefixSum2.data() + 1, 
            numTasks,
            firstStream
        );

        CachedDeviceUVector<int> d_segmentIdsOfCandidates2(totalNumCandidates, firstStream, *cubAllocator);

        helpers::call_fill_kernel_async(d_segmentIdsOfCandidates2.data(), totalNumCandidates, 0, firstStream);

        setGpuSegmentIds(
            d_segmentIdsOfCandidates2.data(),
            numTasks,
            totalNumCandidates,
            d_numCandidatesPerAnchor2.data(),
            d_numCandidatesPerAnchorPrefixSum2.data(),
            firstStream
        );
   

        ThrustCachingAllocator<char> thrustCachingAllocator1(deviceId, cubAllocator, firstStream);

        assert(d_candidateReadIds.size() >= totalNumCandidates);

        #ifdef DO_ONLY_REMOVE_MATE_IDS
            std::swap(d_candidateReadIds, d_candidateReadIds2);
            std::swap(d_segmentIdsOfCandidates, d_segmentIdsOfCandidates2);
            std::swap(d_numCandidatesPerAnchor, d_numCandidatesPerAnchor2);

            auto d_candidateReadIds_end = d_candidateReadIds.data() + totalNumCandidates;
        #else

            d_segmentIdsOfCandidates.resizeUninitialized(totalNumCandidates, firstStream);
        
            //compute segmented set difference.  d_candidateReadIds = d_candidateReadIds2 \ d_usedReadIds
            auto d_candidateReadIds_end = GpuSegmentedSetOperation::set_difference(
                thrustCachingAllocator1,
                d_candidateReadIds2.data(),
                d_numCandidatesPerAnchor2.data(),
                d_numCandidatesPerAnchorPrefixSum2.data(),
                d_segmentIdsOfCandidates2.data(),
                totalNumCandidates,
                numTasks,
                d_fullyUsedReadIds.data(),
                d_numFullyUsedReadIdsPerAnchor.data(),
                d_numFullyUsedReadIdsPerAnchorPrefixSum.data(),
                d_segmentIdsOfFullyUsedReadIds.data(),
                *h_numFullyUsedReadIds,
                numTasks,        
                d_candidateReadIds.data(),
                d_numCandidatesPerAnchor.data(),
                d_segmentIdsOfCandidates.data(),
                numTasks,
                firstStream
            );

            totalNumCandidates = std::distance(d_candidateReadIds.data(), d_candidateReadIds_end);

            d_candidateReadIds.erase(d_candidateReadIds.begin() + totalNumCandidates, d_candidateReadIds.end(), firstStream);
            d_segmentIdsOfCandidates.erase(d_segmentIdsOfCandidates.begin() + totalNumCandidates, d_segmentIdsOfCandidates.end(), firstStream);

        #endif

        d_numCandidatesPerAnchor2.destroy();

        
        
        cudaMemsetAsync(d_numCandidatesPerAnchorPrefixSum.data(), 0, sizeof(int), firstStream); CUERR;
        //compute prefix sum of new segment sizes
        cubInclusiveSum(
            d_numCandidatesPerAnchor.data(), 
            d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            numTasks,
            firstStream
        );

        //removeUsedIdsAndMateIds is a compaction step. check early exit.
        if(totalNumCandidates == 0){
            setStateToFinished();
        }else{
            setState(GpuReadExtender::State::BeforeComputePairFlags);
        }
    }

    void computePairFlagsGpu() {
        assert(state == GpuReadExtender::State::BeforeComputePairFlags);

        cudaStream_t stream = streams[0];
        DEBUGDEVICESYNC

        d_isPairedCandidate.resizeUninitialized(totalNumCandidates, stream);

        helpers::call_fill_kernel_async(d_isPairedCandidate.data(), totalNumCandidates, false, stream);

        DEBUGDEVICESYNC

        h_firstTasksOfPairsToCheck.resize(numTasks);
        int numChecks = 0;

        for(int first = 0, second = 1; second < numTasks; ){
            const int taskindex1 = first;
            const int taskindex2 = second;

            const bool areConsecutiveTasks = tasks[taskindex1].id + 1 == tasks[taskindex2].id;
            const bool arePairedTasks = (tasks[taskindex1].id % 2) + 1 == (tasks[taskindex2].id % 2);

            if(areConsecutiveTasks && arePairedTasks){
                h_firstTasksOfPairsToCheck[numChecks] = first;
                numChecks++;
                
                first += 2; second += 2;
            }else{
                first += 1; second += 1;
            }
        }

        if(numChecks > 0){

            CachedDeviceUVector<int> d_firstTasksOfPairsToCheck(numTasks, stream, *cubAllocator);

            cudaMemcpyAsync(
                d_firstTasksOfPairsToCheck.data(),
                h_firstTasksOfPairsToCheck.data(),
                sizeof(int) * numChecks,
                H2D,
                stream
            ); CUERR;

            dim3 block = 128;
            dim3 grid = numChecks;

            helpers::lambda_kernel<<<grid, block, 0, stream>>>(
                [
                    numChecks,
                    d_firstTasksOfPairsToCheck = d_firstTasksOfPairsToCheck.data(),
                    d_numCandidatesPerAnchor = d_numCandidatesPerAnchor.data(),
                    d_numCandidatesPerAnchorPrefixSum = d_numCandidatesPerAnchorPrefixSum.data(), // numTasks + 1
                    d_numCandidatesPerAnchorPrefixSumsize = d_numCandidatesPerAnchorPrefixSum.size(),
                    d_candidateReadIds = d_candidateReadIds.data(),
                    d_candidateReadIdssize = d_candidateReadIds.size(),
                    d_numUsedReadIdsPerAnchor = d_numUsedReadIdsPerAnchor.data(),
                    d_numUsedReadIdsPerAnchorsize = d_numUsedReadIdsPerAnchor.size(),
                    d_numUsedReadIdsPerAnchorPrefixSum = d_numUsedReadIdsPerAnchorPrefixSum.data(), // numTasks
                    d_numUsedReadIdsPerAnchorPrefixSumsize = d_numUsedReadIdsPerAnchorPrefixSum.size(), // numTasks
                    d_usedReadIds = d_usedReadIds.data(),
                    d_usedReadIdssize = d_usedReadIds.size(),

                    d_isPairedCandidate = d_isPairedCandidate.data()
                ] __device__ (){

                    constexpr int numSharedElements = 1024;

                    __shared__ read_number sharedElements[numSharedElements];

                    //search elements of array1 in array2. if found, set output element to true
                    //array1 and array2 must be sorted
                    auto process = [&](
                        const read_number* array1,
                        int numElements1,
                        const read_number* array2,
                        int numElements2,
                        bool* output,
                        const read_number* boundary1,
                        const read_number* boundary2,
                        const bool* boundaryoutput
                    ){
                        const int numIterations = SDIV(numElements2, numSharedElements);

                        for(int iteration = 0; iteration < numIterations; iteration++){

                            const int begin = iteration * numSharedElements;
                            const int end = min((iteration+1) * numSharedElements, numElements2);
                            const int num = end - begin;

                            for(int i = threadIdx.x; i < num; i += blockDim.x){
                                assert(array2 + begin + i < boundary2);
                                sharedElements[i] = array2[begin + i];
                            }

                            __syncthreads();

                            //TODO in iteration > 0, we may skip elements at the beginning of first range

                            for(int i = threadIdx.x; i < numElements1; i += blockDim.x){
                                assert(output + i < boundaryoutput);
                                if(!output[i]){
                                    assert(array1 + i < boundary1);
                                    const read_number readId = array1[i];
                                    const read_number readIdToFind = readId % 2 == 0 ? readId + 1 : readId - 1;

                                    const bool found = thrust::binary_search(thrust::seq, sharedElements, sharedElements + num, readIdToFind);
                                    if(found){
                                        output[i] = true;
                                    }
                                }
                            }

                            __syncthreads();
                        }
                    };

                    auto process2 = [&](
                        const read_number* array1,
                        int numElements1,
                        const read_number* array2,
                        int numElements2,
                        bool* output,
                        const read_number* boundary1,
                        const read_number* boundary2,
                        const bool* boundaryoutput
                    ){
                        const int numIterations = SDIV(numElements2, numSharedElements);

                        for(int iteration = 0; iteration < numIterations; iteration++){

                            const int begin = iteration * numSharedElements;
                            const int end = min((iteration+1) * numSharedElements, numElements2);
                            const int num = end - begin;

                            for(int i = threadIdx.x; i < num; i += blockDim.x){
                                assert(array2 + begin + i < boundary2);
                                sharedElements[i] = array2[begin + i];
                            }

                            __syncthreads();

                            //TODO in iteration > 0, we may skip elements at the beginning of first range

                            for(int i = threadIdx.x; i < numElements1; i += blockDim.x){
                                assert(output + i < boundaryoutput);
                                if(!output[i]){
                                    assert(array1 + i < boundary1);
                                    const read_number readId = array1[i];
                                    const read_number readIdToFind = readId % 2 == 0 ? readId + 1 : readId - 1;

                                    const bool found = thrust::binary_search(thrust::seq, sharedElements, sharedElements + num, readIdToFind);
                                    if(found){
                                        output[i] = true;
                                    }
                                }
                            }

                            __syncthreads();
                        }
                    };

                    for(int a = blockIdx.x; a < numChecks; a += gridDim.x){
                        const int firstTask = d_firstTasksOfPairsToCheck[a];
                        //const int secondTask = firstTask + 1;

                        //check for pairs in current candidates
                        assert(firstTask < d_numCandidatesPerAnchorPrefixSumsize);
                        assert(firstTask+2 < d_numCandidatesPerAnchorPrefixSumsize);
                        assert(firstTask+2 < d_numCandidatesPerAnchorPrefixSumsize);
                        const int rangeBegin = d_numCandidatesPerAnchorPrefixSum[firstTask];                        
                        const int rangeMid = d_numCandidatesPerAnchorPrefixSum[firstTask + 1];
                        const int rangeEnd = d_numCandidatesPerAnchorPrefixSum[firstTask + 2];

                        process(
                            d_candidateReadIds + rangeBegin,
                            rangeMid - rangeBegin,
                            d_candidateReadIds + rangeMid,
                            rangeEnd - rangeMid,
                            d_isPairedCandidate + rangeBegin,
                            d_candidateReadIds + d_candidateReadIdssize,
                            d_candidateReadIds + d_candidateReadIdssize,
                            d_isPairedCandidate + d_candidateReadIdssize
                        );

                        process(
                            d_candidateReadIds + rangeMid,
                            rangeEnd - rangeMid,
                            d_candidateReadIds + rangeBegin,
                            rangeMid - rangeBegin,
                            d_isPairedCandidate + rangeMid,
                            d_candidateReadIds + d_candidateReadIdssize,
                            d_candidateReadIds + d_candidateReadIdssize,
                            d_isPairedCandidate + d_candidateReadIdssize
                        );

                        //check for pairs in candidates of previous extension iterations

                        assert(firstTask < d_numUsedReadIdsPerAnchorPrefixSumsize);
                        assert(firstTask+1 < d_numUsedReadIdsPerAnchorPrefixSumsize);
                        assert(firstTask+1 < d_numUsedReadIdsPerAnchorsize);

                        const int usedRangeBegin = d_numUsedReadIdsPerAnchorPrefixSum[firstTask];                        
                        const int usedRangeMid = d_numUsedReadIdsPerAnchorPrefixSum[firstTask + 1];
                        const int usedRangeEnd = usedRangeMid + d_numUsedReadIdsPerAnchor[firstTask + 1];

                        process2(
                            d_candidateReadIds + rangeBegin,
                            rangeMid - rangeBegin,
                            d_usedReadIds + usedRangeMid,
                            usedRangeEnd - usedRangeMid,
                            d_isPairedCandidate + rangeBegin,
                            d_candidateReadIds + d_candidateReadIdssize,
                            d_usedReadIds + d_usedReadIdssize,
                            d_isPairedCandidate + d_candidateReadIdssize
                        );

                        process2(
                            d_candidateReadIds + rangeMid,
                            rangeEnd - rangeMid,
                            d_usedReadIds + usedRangeBegin,
                            usedRangeMid - usedRangeBegin,
                            d_isPairedCandidate + rangeMid,
                            d_candidateReadIds + d_candidateReadIdssize,
                            d_usedReadIds + d_usedReadIdssize,
                            d_isPairedCandidate + d_candidateReadIdssize
                        );
                    }
                }
            ); CUERR;

        }

        setState(GpuReadExtender::State::BeforeLoadCandidates);

    }

    void loadCandidateSequenceData() {
        assert(state == GpuReadExtender::State::BeforeLoadCandidates);

        cudaStream_t stream = streams[0];

        d_candidateSequencesLength.resizeUninitialized(totalNumCandidates, stream);
        d_candidateSequencesData.resizeUninitialized(encodedSequencePitchInInts * totalNumCandidates, stream);

        assert(d_candidateReadIds.size() >= totalNumCandidates);

        gpuReadStorage->gatherSequences(
            readStorageHandle,
            d_candidateSequencesData.data(),
            encodedSequencePitchInInts,
            h_candidateReadIds.data(),
            d_candidateReadIds.data(), //device accessible
            totalNumCandidates,
            stream
        );

        gpuReadStorage->gatherSequenceLengths(
            readStorageHandle,
            d_candidateSequencesLength.data(),
            d_candidateReadIds.data(),
            totalNumCandidates,
            stream
        );

        setState(GpuReadExtender::State::BeforeEraseData);
    }

    void eraseDataOfRemovedMates(){
        assert(state == GpuReadExtender::State::BeforeEraseData);

        cudaStream_t stream = streams[0];

        CachedDeviceUVector<int> d_positionsOfAnchorsToRemoveMateSequence(numTasks, stream, *cubAllocator);

        //determine task ids with removed mates
        cubSelectFlagged(
            thrust::make_counting_iterator(0),
            d_mateIdHasBeenRemoved.data(),
            d_positionsOfAnchorsToRemoveMateSequence.data(),
            h_numAnchorsWithRemovedMates.data(),
            numTasks,
            stream
        );
        cudaStreamSynchronize(stream); CUERR; //wait for h_numAnchorsWithRemovedMates
        const int numTasksWithMateRemoved = *h_numAnchorsWithRemovedMates;

        if(numTasksWithMateRemoved > 0){

            CachedDeviceUVector<bool> d_keepflags(totalNumCandidates, stream, *cubAllocator);

            CachedDeviceUVector<unsigned int> d_sequencesOfMatesWhichShouldBeRemoved(numTasksWithMateRemoved * encodedSequencePitchInInts, stream, *cubAllocator);

            //Gather mate sequence data of tasks which removed mate read id from candidate list

            cubSelectFlagged(
                d_inputanchormatedata.data(),
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator(0),
                    make_iterator_multiplier(d_mateIdHasBeenRemoved.data(), encodedSequencePitchInInts)
                ),
                d_sequencesOfMatesWhichShouldBeRemoved.data(),
                thrust::make_discard_iterator(),
                numTasks * encodedSequencePitchInInts,
                stream
            );

            constexpr int groupsize = 32;
            constexpr int blocksize = 128;
            constexpr int groupsperblock = blocksize / groupsize;
            dim3 block(blocksize,1,1);
            dim3 grid(SDIV(numTasksWithMateRemoved * groupsize, blocksize), 1, 1);
            const std::size_t smembytes = sizeof(unsigned int) * groupsperblock * encodedSequencePitchInInts;

            //compute flags of candidates which should not be removed. Candidates which should be removed are identical to mate sequence
            helpers::call_fill_kernel_async(d_keepflags.data(), totalNumCandidates, true, stream);

            readextendergpukernels::filtermatekernel<blocksize,groupsize><<<grid, block, smembytes, stream>>>(
                d_sequencesOfMatesWhichShouldBeRemoved.data(),
                d_candidateSequencesData.data(),
                encodedSequencePitchInInts,
                d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchorPrefixSum.data(),
                d_positionsOfAnchorsToRemoveMateSequence.data(),
                numTasksWithMateRemoved,
                d_keepflags.data()
            ); CUERR;

            compactCandidateDataByFlagsExcludingAlignments(
                d_keepflags.data(),
                stream
            );
        }

        setState(GpuReadExtender::State::BeforeAlignment);
    }

    void calculateAlignments(){
        assert(state == GpuReadExtender::State::BeforeAlignment);

        cudaStream_t stream = streams[0];


        d_alignment_overlaps.resizeUninitialized(totalNumCandidates, stream);
        d_alignment_shifts.resizeUninitialized(totalNumCandidates, stream);
        d_alignment_nOps.resizeUninitialized(totalNumCandidates, stream);
        d_alignment_best_alignment_flags.resizeUninitialized(totalNumCandidates, stream);

        CachedDeviceUVector<bool> d_alignment_isValid(totalNumCandidates, stream, *cubAllocator);

        h_numAnchors[0] = numTasks;

        const bool* const d_anchorContainsN = nullptr;
        const bool* const d_candidateContainsN = nullptr;
        const bool removeAmbiguousAnchors = false;
        const bool removeAmbiguousCandidates = false;
        const int maxNumAnchors = numTasks;
        const int maxNumCandidates = totalNumCandidates; //this does not need to be exact, but it must be >= d_numCandidatesPerAnchorPrefixSum[numTasks]
        const int maximumSequenceLength = encodedSequencePitchInInts * 16;
        const int encodedSequencePitchInInts2Bit = encodedSequencePitchInInts;
        const int min_overlap = goodAlignmentProperties->min_overlap;
        const float maxErrorRate = goodAlignmentProperties->maxErrorRate;
        const float min_overlap_ratio = goodAlignmentProperties->min_overlap_ratio;
        const float estimatedNucleotideErrorRate = correctionOptions->estimatedErrorrate;

        auto callAlignmentKernel = [&](void* d_tempstorage, size_t& tempstoragebytes){

            call_popcount_rightshifted_hamming_distance_kernel_async(
                d_tempstorage,
                tempstoragebytes,
                d_alignment_overlaps.data(),
                d_alignment_shifts.data(),
                d_alignment_nOps.data(),
                d_alignment_isValid.data(),
                d_alignment_best_alignment_flags.data(),
                d_subjectSequencesData.data(),
                d_candidateSequencesData.data(),
                d_anchorSequencesLength.data(),
                d_candidateSequencesLength.data(),
                d_numCandidatesPerAnchorPrefixSum.data(),
                d_numCandidatesPerAnchor.data(),
                d_segmentIdsOfCandidates.data(),
                h_numAnchors.data(),
                &d_numCandidatesPerAnchorPrefixSum[numTasks],
                d_anchorContainsN,
                removeAmbiguousAnchors,
                d_candidateContainsN,
                removeAmbiguousCandidates,
                maxNumAnchors,
                maxNumCandidates,
                maximumSequenceLength,
                encodedSequencePitchInInts2Bit,
                min_overlap,
                maxErrorRate,
                min_overlap_ratio,
                estimatedNucleotideErrorRate,
                stream,
                kernelLaunchHandle
            );
        };

        size_t tempstoragebytes = 0;
        callAlignmentKernel(nullptr, tempstoragebytes);

        CachedDeviceUVector<char> d_tempstorage(tempstoragebytes, stream, *cubAllocator);

        callAlignmentKernel(d_tempstorage.data(), tempstoragebytes);

        setState(GpuReadExtender::State::BeforeAlignmentFilter);
    }

    void filterAlignments(){
        assert(state == GpuReadExtender::State::BeforeAlignmentFilter);

        cudaStream_t stream = streams[0];


        DEBUGDEVICESYNC

        const int numAnchors = numTasks;

        CachedDeviceUVector<int> d_numCandidatesPerAnchor2(numTasks, stream, *cubAllocator);

        h_numCandidates.resize(1);

        CachedDeviceUVector<unsigned int> d_candidateSequencesData2(encodedSequencePitchInInts * totalNumCandidates, stream, *cubAllocator);

        CachedDeviceUVector<bool> d_keepflags(totalNumCandidates, stream, *cubAllocator);
        helpers::call_fill_kernel_async(d_keepflags.data(), totalNumCandidates, true, stream);

        dim3 block(128,1,1);
        dim3 grid(numAnchors, 1, 1);

        #if 0
            //filter alignments of candidates. d_keepflags[i] will be set to false if candidate[i] should be removed
        //d_numCandidatesPerAnchor2[i] contains new number of candidates for anchor i
        helpers::lambda_kernel<<<grid, block, 0, stream>>>(
            [
                d_alignment_best_alignment_flags = d_alignment_best_alignment_flags.data(),
                d_alignment_shifts = d_alignment_shifts.data(),
                d_alignment_overlaps = d_alignment_overlaps.data(),
                d_alignment_nOps = d_alignment_nOps.data(),
                d_anchorSequencesLength = d_anchorSequencesLength.data(),
                d_numCandidatesPerAnchor = d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchor2 = d_numCandidatesPerAnchor2.data(),
                d_numCandidatesPerAnchorPrefixSum = d_numCandidatesPerAnchorPrefixSum.data(),
                d_isPairedCandidate = d_isPairedCandidate.data(),
                d_keepflags = d_keepflags.data(),
                min_overlap_ratio = goodAlignmentProperties->min_overlap_ratio,
                numAnchors
            ] __device__ (){

                using BlockReduceFloat = cub::BlockReduce<float, 128>;
                using BlockReduceInt = cub::BlockReduce<int, 128>;

                __shared__ union {
                    typename BlockReduceFloat::TempStorage floatreduce;
                    typename BlockReduceInt::TempStorage intreduce;
                } cubtemp;

                __shared__ int intbroadcast;
                __shared__ float floatbroadcast;

                for(int a = blockIdx.x; a < numAnchors; a += gridDim.x){
                    const int num = d_numCandidatesPerAnchor[a];
                    const int offset = d_numCandidatesPerAnchorPrefixSum[a];
                    const float anchorLength = d_anchorSequencesLength[a];
                    int removed = 0;

                    //loop over candidates to compute relative overlap threshold

                    for(int c = threadIdx.x; c < num; c += blockDim.x){
                        
                        const auto alignmentflag = d_alignment_best_alignment_flags[offset + c];
                        const int shift = d_alignment_shifts[offset + c];

                        if(alignmentflag != BestAlignment_t::None && shift >= 0){
                            if(d_isPairedCandidate[offset+c]){
                                d_keepflags[offset + c] = true; //paired candidates always pass
                            }else{
                                const float overlap = d_alignment_overlaps[offset + c];
                                const float numMismatches = d_alignment_nOps[offset + c];                          
                                const float relativeOverlap = overlap / anchorLength;
                                const float errorrate = numMismatches / overlap;

                                if(fleq(errorrate, 0.03f)){
                                    d_keepflags[offset + c] = true;
                                }else{
                                    d_keepflags[offset + c] = false;
                                    removed++;
                                }
                            }
                        }else{
                            //remove alignment with negative shift or bad alignments
                            d_keepflags[offset + c] = false;
                            removed++;
                        }                  
                    }

                    removed = BlockReduceInt(cubtemp.intreduce).Sum(removed);

                    if(threadIdx.x == 0){
                        d_numCandidatesPerAnchor2[a] = num - removed;
                    }
                    __syncthreads();
                }
            }
        ); CUERR;
        #else

        //filter alignments of candidates. d_keepflags[i] will be set to false if candidate[i] should be removed
        //d_numCandidatesPerAnchor2[i] contains new number of candidates for anchor i
        helpers::lambda_kernel<<<grid, block, 0, stream>>>(
            [
                d_alignment_best_alignment_flags = d_alignment_best_alignment_flags.data(),
                d_alignment_shifts = d_alignment_shifts.data(),
                d_alignment_overlaps = d_alignment_overlaps.data(),
                d_anchorSequencesLength = d_anchorSequencesLength.data(),
                d_numCandidatesPerAnchor = d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchor2 = d_numCandidatesPerAnchor2.data(),
                d_numCandidatesPerAnchorPrefixSum = d_numCandidatesPerAnchorPrefixSum.data(),
                d_isPairedCandidate = d_isPairedCandidate.data(),
                d_keepflags = d_keepflags.data(),
                min_overlap_ratio = goodAlignmentProperties->min_overlap_ratio,
                numAnchors
            ] __device__ (){

                using BlockReduceFloat = cub::BlockReduce<float, 128>;
                using BlockReduceInt = cub::BlockReduce<int, 128>;

                __shared__ union {
                    typename BlockReduceFloat::TempStorage floatreduce;
                    typename BlockReduceInt::TempStorage intreduce;
                } cubtemp;

                __shared__ int intbroadcast;
                __shared__ float floatbroadcast;

                for(int a = blockIdx.x; a < numAnchors; a += gridDim.x){
                    const int num = d_numCandidatesPerAnchor[a];
                    const int offset = d_numCandidatesPerAnchorPrefixSum[a];
                    const float anchorLength = d_anchorSequencesLength[a];
                    int removed = 0;

                    int threadReducedGoodAlignmentExists = 0;
                    float threadReducedRelativeOverlapThreshold = 0.0f;

                    //loop over candidates to compute relative overlap threshold

                    for(int c = threadIdx.x; c < num; c += blockDim.x){
                        const auto alignmentflag = d_alignment_best_alignment_flags[offset + c];
                        const int shift = d_alignment_shifts[offset + c];

                        if(alignmentflag != BestAlignment_t::None && shift >= 0){
                            if(!d_isPairedCandidate[offset+c]){
                                const float overlap = d_alignment_overlaps[offset + c];                            
                                const float relativeOverlap = overlap / anchorLength;
                                
                                if(relativeOverlap < 1.0f && fgeq(relativeOverlap, min_overlap_ratio)){
                                    threadReducedGoodAlignmentExists = 1;
                                    const float tmp = floorf(relativeOverlap * 10.0f) / 10.0f;
                                    threadReducedRelativeOverlapThreshold = fmaxf(threadReducedRelativeOverlapThreshold, tmp);
                                }
                            }
                        }else{
                            //remove alignment with negative shift or bad alignments
                            d_keepflags[offset + c] = false;
                            removed++;
                        }                       
                    }

                    int blockreducedGoodAlignmentExists = BlockReduceInt(cubtemp.intreduce)
                        .Sum(threadReducedGoodAlignmentExists);
                    if(threadIdx.x == 0){
                        intbroadcast = blockreducedGoodAlignmentExists;
                        //printf("task %d good: %d\n", a, blockreducedGoodAlignmentExists);
                    }
                    __syncthreads();

                    blockreducedGoodAlignmentExists = intbroadcast;

                    if(blockreducedGoodAlignmentExists > 0){
                        float blockreducedRelativeOverlapThreshold = BlockReduceFloat(cubtemp.floatreduce)
                            .Reduce(threadReducedRelativeOverlapThreshold, cub::Max());
                        if(threadIdx.x == 0){
                            floatbroadcast = blockreducedRelativeOverlapThreshold;
                            //printf("task %d thresh: %f\n", a, blockreducedRelativeOverlapThreshold);
                        }
                        __syncthreads();

                        blockreducedRelativeOverlapThreshold = floatbroadcast;

                        // loop over candidates and remove those with an alignment overlap threshold smaller than the computed threshold
                        for(int c = threadIdx.x; c < num; c += blockDim.x){
                            if(!d_isPairedCandidate[offset+c]){
                                if(d_keepflags[offset + c]){
                                    const float overlap = d_alignment_overlaps[offset + c];                            
                                    const float relativeOverlap = overlap / anchorLength;                 
        
                                    if(!fgeq(relativeOverlap, blockreducedRelativeOverlapThreshold)){
                                        d_keepflags[offset + c] = false;
                                        removed++;
                                    }
                                }
                            }
                        }
                    }else{
                        //NOOP.
                        //if no good alignment exists, no candidate is removed. we will try to work with the not-so-good alignments
                        // if(threadIdx.x == 0){
                        //     printf("no good alignment,nc %d\n", num);
                        // }
                    }

                    removed = BlockReduceInt(cubtemp.intreduce).Sum(removed);

                    if(threadIdx.x == 0){
                        d_numCandidatesPerAnchor2[a] = num - removed;
                    }
                    __syncthreads();
                }
            }
        ); CUERR;
        #endif

        compactCandidateDataByFlags(
            d_keepflags.data(),
            stream
        );

        //filterAlignments is a compaction step. check early exit.
        if(totalNumCandidates == 0){
            setStateToFinished();
        }else{
            setState(GpuReadExtender::State::BeforeMSA);
        }
    }

    void computeMSAs(){
        assert(state == GpuReadExtender::State::BeforeMSA);

        cudaStream_t firstStream = streams[0];
        //cudaStream_t secondStream = firstStream;

        CachedDeviceUVector<char> d_candidateQualityScores(qualityPitchInBytes * totalNumCandidates, firstStream, *cubAllocator);

        loadCandidateQualityScores(firstStream, d_candidateQualityScores.data());

        CachedDeviceUVector<int> d_numCandidatesPerAnchor2(numTasks, firstStream, *cubAllocator);

        CachedDeviceUVector<int> indices1(totalNumCandidates, firstStream, *cubAllocator);
        CachedDeviceUVector<int> indices2(totalNumCandidates, firstStream, *cubAllocator);       

        helpers::lambda_kernel<<<numTasks, 128, 0, firstStream>>>(
            [
                indices1 = indices1.data(),
                d_numCandidatesPerAnchor = d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchorPrefixSum = d_numCandidatesPerAnchorPrefixSum.data()
            ] __device__ (){
                const int num = d_numCandidatesPerAnchor[blockIdx.x];
                const int offset = d_numCandidatesPerAnchorPrefixSum[blockIdx.x];
                
                for(int i = threadIdx.x; i < num; i += blockDim.x){
                    indices1[offset + i] = i;
                }
            }
        );

        *h_numAnchors = numTasks;

        const bool useQualityScoresForMSA = true;

        multiMSA.construct(
            d_alignment_overlaps.data(),
            d_alignment_shifts.data(),
            d_alignment_nOps.data(),
            d_alignment_best_alignment_flags.data(),
            indices1.data(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_anchorSequencesLength.data(),
            d_subjectSequencesData.data(),
            d_anchorQualityScores.data(),
            numTasks,
            d_candidateSequencesLength.data(),
            d_candidateSequencesData.data(),
            d_candidateQualityScores.data(),
            d_isPairedCandidate.data(),
            totalNumCandidates,
            h_numAnchors.data(), //d_numAnchors
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            useQualityScoresForMSA,
            goodAlignmentProperties->maxErrorRate,
            gpu::MSAColumnCount(msaColumnPitchInElements),
            firstStream
        );

        multiMSA.refine(
            indices2.data(),
            d_numCandidatesPerAnchor2.data(),
            d_numCandidates2.data(),
            d_alignment_overlaps.data(),
            d_alignment_shifts.data(),
            d_alignment_nOps.data(),
            d_alignment_best_alignment_flags.data(),
            indices1.data(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_anchorSequencesLength.data(),
            d_subjectSequencesData.data(),
            d_anchorQualityScores.data(),
            numTasks,
            d_candidateSequencesLength.data(),
            d_candidateSequencesData.data(),
            d_candidateQualityScores.data(),
            d_isPairedCandidate.data(),
            totalNumCandidates,
            h_numAnchors.data(), //d_numAnchors
            encodedSequencePitchInInts,
            qualityPitchInBytes,
            useQualityScoresForMSA,
            goodAlignmentProperties->maxErrorRate,
            correctionOptions->estimatedCoverage,
            getNumRefinementIterations(),
            firstStream
        );
  
        CachedDeviceUVector<bool> d_shouldBeKept(totalNumCandidates, firstStream, *cubAllocator);

        helpers::call_fill_kernel_async(d_shouldBeKept.data(), totalNumCandidates, false, firstStream); CUERR;

        //convert output indices from task-local indices to global flags
        helpers::lambda_kernel<<<numTasks, 128, 0, firstStream>>>(
            [
                d_flagscandidates = d_shouldBeKept.data(),
                indices2 = indices2.data(),
                d_numCandidatesPerAnchor2 = d_numCandidatesPerAnchor2.data(),
                d_numCandidatesPerAnchorPrefixSum = d_numCandidatesPerAnchorPrefixSum.data()
            ] __device__ (){
                /*
                    Input:
                    indices2: 0,1,2,0,0,0,0,3,5,0
                    d_numCandidatesPerAnchorPrefixSum: 0,6,10

                    Output:
                    d_flagscandidates: 1,1,1,0,0,0,1,0,0,1,0,1
                */
                const int num = d_numCandidatesPerAnchor2[blockIdx.x];
                const int offset = d_numCandidatesPerAnchorPrefixSum[blockIdx.x];
                
                for(int i = threadIdx.x; i < num; i += blockDim.x){
                    const int globalIndex = indices2[offset + i] + offset;
                    d_flagscandidates[globalIndex] = true;
                }
            }
        ); CUERR;

        indices1.destroy();
        indices2.destroy();
        d_numCandidatesPerAnchor2.destroy();

        d_candidateQualityScores.destroy();

        compactCandidateDataByFlags(
            d_shouldBeKept.data(),
            firstStream
        );
        
        setState(GpuReadExtender::State::BeforeExtend);
    }


    void computeExtendedSequencesFromMSAs(){
        assert(state == GpuReadExtender::State::BeforeExtend);

        cudaStream_t stream = streams[0];

        outputAnchorPitchInBytes = SDIV(decodedSequencePitchInBytes, 128) * 128;
        outputAnchorQualityPitchInBytes = SDIV(qualityPitchInBytes, 128) * 128;
        decodedMatesRevCPitchInBytes = SDIV(decodedSequencePitchInBytes, 128) * 128;

        CachedDeviceUVector<int> d_accumExtensionsLengthsOUT(numTasks, stream, *cubAllocator);
        CachedDeviceUVector<int> d_sizeOfGapToMate(numTasks, stream, *cubAllocator);
        CachedDeviceUVector<float> d_goodscores(numTasks, stream, *cubAllocator);
        
        d_isFullyUsedCandidate.resizeUninitialized(totalNumCandidates, stream);
        d_outputAnchors.resizeUninitialized(numTasks * outputAnchorPitchInBytes, stream);
        d_outputAnchorQualities.resizeUninitialized(numTasks * outputAnchorQualityPitchInBytes, stream);
        d_outputMateHasBeenFound.resizeUninitialized(numTasks, stream);
        d_abortReasons.resizeUninitialized(numTasks, stream);
        d_outputAnchorLengths.resizeUninitialized(numTasks, stream);      

        helpers::call_fill_kernel_async(d_outputMateHasBeenFound.data(), numTasks, false, stream); CUERR;
        helpers::call_fill_kernel_async(d_abortReasons.data(), numTasks, extension::AbortReason::None, stream); CUERR;
        helpers::call_fill_kernel_async(d_isFullyUsedCandidate.data(), totalNumCandidates, false, stream); CUERR;
        helpers::call_fill_kernel_async(d_goodscores.data(), numTasks, 0.0f, stream); CUERR;
      
        //compute extensions

        readextendergpukernels::computeExtensionStepFromMsaKernel<128><<<numTasks, 128, 0, stream>>>(
            insertSize,
            insertSizeStddev,
            multiMSA.multiMSAView(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_anchorSequencesLength.data(),
            d_accumExtensionsLengths.data(),
            d_inputMateLengths.data(),
            d_abortReasons.data(),
            d_accumExtensionsLengthsOUT.data(),
            d_outputAnchors.data(),
            outputAnchorPitchInBytes,
            d_outputAnchorQualities.data(),
            outputAnchorQualityPitchInBytes,
            d_outputAnchorLengths.data(),
            d_isPairedTask.data(),
            d_inputanchormatedata.data(),
            encodedSequencePitchInInts,
            decodedMatesRevCPitchInBytes,
            d_outputMateHasBeenFound.data(),
            d_sizeOfGapToMate.data(),
            minCoverageForExtension,
            maxextensionPerStep
        );

        readextendergpukernels::computeExtensionStepQualityKernel<128><<<numTasks, 128, 0, stream>>>(
            d_goodscores.data(),
            multiMSA.multiMSAView(),
            d_abortReasons.data(),
            d_outputMateHasBeenFound.data(),
            d_accumExtensionsLengths.data(),
            d_accumExtensionsLengthsOUT.data(),
            d_anchorSequencesLength.data(),
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_candidateSequencesLength.data(),
            d_alignment_shifts.data(),
            d_alignment_best_alignment_flags.data(),
            d_candidateSequencesData.data(),
            multiMSA.d_columnProperties.data(),
            encodedSequencePitchInInts
        );

        //check which candidates are fully used in the extension
        helpers::lambda_kernel<<<numTasks, 128, 0, stream>>>(
            [
                numTasks = numTasks,
                d_numCandidatesPerAnchor = d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchorPrefixSum = d_numCandidatesPerAnchorPrefixSum.data(),
                d_candidateSequencesLengths = d_candidateSequencesLength.data(),
                d_alignment_shifts = d_alignment_shifts.data(),
                d_anchorSequencesLength = d_anchorSequencesLength.data(),
                d_oldaccumExtensionsLengths = d_accumExtensionsLengths.data(),
                d_newaccumExtensionsLengths = d_accumExtensionsLengthsOUT.data(),
                d_abortReasons = d_abortReasons.data(),
                d_outputMateHasBeenFound = d_outputMateHasBeenFound.data(),
                d_isFullyUsedCandidate = d_isFullyUsedCandidate.data()
            ] __device__ (){


                for(int task = blockIdx.x; task < numTasks; task += gridDim.x){
                    const int numCandidates = d_numCandidatesPerAnchor[task];
                    const auto abortReason = d_abortReasons[task];

                    if(numCandidates > 0 && abortReason == extension::AbortReason::None){
                        const int anchorLength = d_anchorSequencesLength[task];
                        const int offset = d_numCandidatesPerAnchorPrefixSum[task];
                        const int oldAccumExtensionsLength = d_oldaccumExtensionsLengths[task];
                        const int newAccumExtensionsLength = d_newaccumExtensionsLengths[task];
                        const int lengthOfExtension = newAccumExtensionsLength - oldAccumExtensionsLength;

                        for(int c = threadIdx.x; c < numCandidates; c += blockDim.x){
                            const int candidateLength = d_candidateSequencesLengths[c];
                            const int shift = d_alignment_shifts[c];

                            if(candidateLength + shift <= anchorLength + lengthOfExtension){
                                d_isFullyUsedCandidate[offset + c] = true;
                            }
                        }
                    }
                }
            }
        );

        cudaEventRecord(events[0], stream); CUERR;
        cudaStreamWaitEvent(hostOutputStream, events[0], 0); CUERR;

        h_accumExtensionsLengths.resize(numTasks);
        h_abortReasons.resize(numTasks);
        h_outputAnchors.resize(numTasks * outputAnchorPitchInBytes);
        h_outputAnchorQualities.resize(numTasks * outputAnchorQualityPitchInBytes);
        h_outputAnchorLengths.resize(numTasks);
        h_outputMateHasBeenFound.resize(numTasks);
        h_sizeOfGapToMate.resize(numTasks);
        h_isFullyUsedCandidate.resize(totalNumCandidates);
        h_goodscores.resize(numTasks);

        helpers::call_copy_n_kernel(
            thrust::make_zip_iterator(thrust::make_tuple(
                d_accumExtensionsLengthsOUT.data(),
                d_abortReasons.data(),
                d_outputMateHasBeenFound.data(),
                d_sizeOfGapToMate.data(),
                d_outputAnchorLengths.data(),
                d_goodscores.data()
            )),
            numTasks,
            thrust::make_zip_iterator(thrust::make_tuple(
                h_accumExtensionsLengths.data(),
                h_abortReasons.data(),
                h_outputMateHasBeenFound.data(),
                h_sizeOfGapToMate.data(),
                h_outputAnchorLengths.data(),
                h_goodscores.data()
            )),
            hostOutputStream
        );

        cudaMemcpyAsync(
            h_outputAnchors.data(),
            d_outputAnchors.data(),
            sizeof(char) * outputAnchorPitchInBytes * numTasks,
            D2H,
            hostOutputStream
        ); CUERR;

        cudaMemcpyAsync(
            h_outputAnchorQualities.data(),
            d_outputAnchorQualities.data(),
            sizeof(char) * outputAnchorQualityPitchInBytes * numTasks,
            D2H,
            hostOutputStream
        ); CUERR;

        cudaMemcpyAsync(
            h_isFullyUsedCandidate.data(),
            d_isFullyUsedCandidate.data(),
            sizeof(bool) * totalNumCandidates,
            D2H,
            hostOutputStream
        ); CUERR;

        std::swap(d_accumExtensionsLengths, d_accumExtensionsLengthsOUT);

        setState(GpuReadExtender::State::BeforeUpdateUsedCandidateIds);
    }


    void updateUsedCandidateIds(){
        assert(state == GpuReadExtender::State::BeforeUpdateUsedCandidateIds);

        cudaStream_t stream = streams[0];

        setGpuSegmentIds(
            d_segmentIdsOfCandidates.data(),
            numTasks,
            totalNumCandidates,
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            stream
        );

        {

            const int maxoutputsize = totalNumCandidates + *h_numUsedReadIds;

            CachedDeviceUVector<read_number> d_newUsedReadIds(maxoutputsize, stream, *cubAllocator);
            CachedDeviceUVector<int> d_newNumUsedreadIdsPerAnchor(numTasks, stream, *cubAllocator);
            CachedDeviceUVector<int> d_newSegmentIdsOfUsedReadIds(maxoutputsize, stream, *cubAllocator);

            ThrustCachingAllocator<char> thrustCachingAllocator1(deviceId, cubAllocator, stream);

            auto d_newUsedReadIds_end = GpuSegmentedSetOperation::set_union(
                thrustCachingAllocator1,
                d_candidateReadIds.data(),
                d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchorPrefixSum.data(),
                d_segmentIdsOfCandidates.data(),
                totalNumCandidates,
                numTasks,
                d_usedReadIds.data(),
                d_numUsedReadIdsPerAnchor.data(),
                d_numUsedReadIdsPerAnchorPrefixSum.data(),
                d_segmentIdsOfUsedReadIds.data(),
                *h_numUsedReadIds,
                numTasks,        
                d_newUsedReadIds.data(),
                d_newNumUsedreadIdsPerAnchor.data(),
                d_newSegmentIdsOfUsedReadIds.data(),
                numTasks,
                stream
            );

            int newsize = std::distance(d_newUsedReadIds.data(), d_newUsedReadIds_end);

            d_newUsedReadIds.erase(d_newUsedReadIds.begin() + newsize, d_newUsedReadIds.end(), stream);
            d_newSegmentIdsOfUsedReadIds.erase(d_newSegmentIdsOfUsedReadIds.begin() + newsize, d_newSegmentIdsOfUsedReadIds.end(), stream);

            std::swap(d_usedReadIds, d_newUsedReadIds);
            std::swap(d_segmentIdsOfUsedReadIds, d_newSegmentIdsOfUsedReadIds);
            std::swap(d_numUsedReadIdsPerAnchor, d_newNumUsedreadIdsPerAnchor);

            cubExclusiveSum(
                d_numUsedReadIdsPerAnchor.data(), 
                d_numUsedReadIdsPerAnchorPrefixSum.data(), 
                numTasks,
                stream
            );

            *h_numUsedReadIds = newsize;

        }

        {

            CachedDeviceUVector<read_number> d_currentFullyUsedReadIds(totalNumCandidates, stream, *cubAllocator);
            CachedDeviceUVector<int> d_currentNumFullyUsedreadIdsPerAnchor(numTasks, stream, *cubAllocator);
            CachedDeviceUVector<int> d_currentNumFullyUsedreadIdsPerAnchorPS(numTasks, stream, *cubAllocator);
            CachedDeviceUVector<int> d_currentSegmentIdsOfFullyUsedReadIds(totalNumCandidates, stream, *cubAllocator);
            
            auto candidatesAndSegmentIdsIn = thrust::make_zip_iterator(
                thrust::make_tuple(
                    d_candidateReadIds.data(),
                    d_segmentIdsOfCandidates.data()
                )
            );

            auto candidatesAndSegmentIdsOut = thrust::make_zip_iterator(
                thrust::make_tuple(
                    d_currentFullyUsedReadIds.data(),
                    d_currentSegmentIdsOfFullyUsedReadIds.data()
                )
            );

            //make compact list of current fully used candidates
            cubSelectFlagged(
                candidatesAndSegmentIdsIn,
                d_isFullyUsedCandidate.data(),
                candidatesAndSegmentIdsOut,
                h_numFullyUsedReadIds2.data(),
                totalNumCandidates,
                stream
            );

            cudaEventRecord(h_numFullyUsedReadIds2Event, stream); CUERR;

            //compute current number of fully used candidates per segment
            cubSegmentedReduceSum(
                d_isFullyUsedCandidate.data(),
                d_currentNumFullyUsedreadIdsPerAnchor.data(),
                numTasks,
                d_numCandidatesPerAnchorPrefixSum.data(),
                d_numCandidatesPerAnchorPrefixSum.data() + 1,
                stream
            );

            //compute prefix sum of current number of fully used candidates per segment

            cubExclusiveSum(
                d_currentNumFullyUsedreadIdsPerAnchor.data(), 
                d_currentNumFullyUsedreadIdsPerAnchorPS.data(), 
                numTasks,
                stream
            );

            cudaEventSynchronize(h_numFullyUsedReadIds2Event); CUERR;

            d_currentFullyUsedReadIds.erase(d_currentFullyUsedReadIds.begin() + *h_numFullyUsedReadIds2, d_currentFullyUsedReadIds.end(), stream);
            d_currentSegmentIdsOfFullyUsedReadIds.erase(d_currentSegmentIdsOfFullyUsedReadIds.begin() + *h_numFullyUsedReadIds2, d_currentSegmentIdsOfFullyUsedReadIds.end(), stream);

            const int maxoutputsize = *h_numFullyUsedReadIds2 + *h_numFullyUsedReadIds;

            CachedDeviceUVector<read_number> d_newFullyUsedReadIds(maxoutputsize, stream, *cubAllocator);
            CachedDeviceUVector<int> d_newNumFullyUsedreadIdsPerAnchor(numTasks, stream, *cubAllocator);
            CachedDeviceUVector<int> d_newSegmentIdsOfFullyUsedReadIds(maxoutputsize, stream, *cubAllocator);

            ThrustCachingAllocator<char> thrustCachingAllocator1(deviceId, cubAllocator, stream);

            auto d_newFullyUsedReadIds_end = GpuSegmentedSetOperation::set_union(
                thrustCachingAllocator1,
                d_currentFullyUsedReadIds.data(),
                d_currentNumFullyUsedreadIdsPerAnchor.data(),
                d_currentNumFullyUsedreadIdsPerAnchorPS.data(),
                d_currentSegmentIdsOfFullyUsedReadIds.data(),
                *h_numFullyUsedReadIds2,
                numTasks,
                d_fullyUsedReadIds.data(),
                d_numFullyUsedReadIdsPerAnchor.data(),
                d_numFullyUsedReadIdsPerAnchorPrefixSum.data(),
                d_segmentIdsOfFullyUsedReadIds.data(),
                *h_numFullyUsedReadIds,
                numTasks,        
                d_newFullyUsedReadIds.data(),
                d_newNumFullyUsedreadIdsPerAnchor.data(),
                d_newSegmentIdsOfFullyUsedReadIds.data(),
                numTasks,
                stream
            );

            int newsize = std::distance(d_newFullyUsedReadIds.data(), d_newFullyUsedReadIds_end);
            *h_numFullyUsedReadIds = newsize;

            d_newFullyUsedReadIds.erase(d_newFullyUsedReadIds.begin() + newsize, d_newFullyUsedReadIds.end(), stream);
            d_newSegmentIdsOfFullyUsedReadIds.erase(d_newSegmentIdsOfFullyUsedReadIds.begin() + newsize, d_newSegmentIdsOfFullyUsedReadIds.end(), stream);

            std::swap(d_fullyUsedReadIds, d_newFullyUsedReadIds);
            std::swap(d_segmentIdsOfFullyUsedReadIds, d_newSegmentIdsOfFullyUsedReadIds);
            std::swap(d_numFullyUsedReadIdsPerAnchor, d_newNumFullyUsedreadIdsPerAnchor);

            cubExclusiveSum(
                d_numFullyUsedReadIdsPerAnchor.data(), 
                d_numFullyUsedReadIdsPerAnchorPrefixSum.data(), 
                numTasks,
                stream
            );
        
        }

        setState(GpuReadExtender::State::BeforeUnpack);
    }
    
    void unpackResultsIntoTasks(){
        assert(state == GpuReadExtender::State::BeforeUnpack);

        cudaStreamSynchronize(hostOutputStream); CUERR;

        for(int i = 0; i < numTasks; i++){ 
            auto& task = tasks[i];

            task.goodscore += h_goodscores[i];

            task.abortReason = h_abortReasons[i];
            if(task.abortReason == extension::AbortReason::None){
                task.mateHasBeenFound = h_outputMateHasBeenFound[i];

                const int myNumDecodedAnchors = task.totalDecodedAnchorsLengths.size();

                if(!task.mateHasBeenFound){
                    const int newlength = h_outputAnchorLengths[i];
                    
                    task.currentAnchorLength = newlength;
                    task.accumExtensionLengths = h_accumExtensionsLengths[i];

                    task.totalDecodedAnchorsFlat.resize((myNumDecodedAnchors+1) * decodedSequencePitchInBytes);
                    assert(newlength <= decodedSequencePitchInBytes);
                    std::copy_n(
                        h_outputAnchors.data() + i * outputAnchorPitchInBytes,
                        newlength,
                        task.totalDecodedAnchorsFlat.begin()
                            + myNumDecodedAnchors * decodedSequencePitchInBytes
                    );
                    task.totalDecodedAnchorsLengths.emplace_back(newlength);

                    task.totalAnchorQualityScoresFlat.resize((myNumDecodedAnchors+1) * qualityPitchInBytes);
                    assert(newlength <= qualityPitchInBytes);
                    std::copy_n(
                        h_outputAnchorQualities.data() + i * outputAnchorQualityPitchInBytes,
                        newlength,
                        task.totalAnchorQualityScoresFlat.begin()
                            + myNumDecodedAnchors * qualityPitchInBytes
                    );


                    task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);
                    
                }else{
                    const int sizeofGap = h_sizeOfGapToMate[i];
                    if(sizeofGap == 0){
                        task.accumExtensionLengths = h_accumExtensionsLengths[i];
                        task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);
  
                        task.totalDecodedAnchorsFlat.resize((myNumDecodedAnchors+1) * decodedSequencePitchInBytes);
                        assert(task.mateLength <= decodedSequencePitchInBytes);
                        std::copy(
                            task.decodedMateRevC.begin(),
                            task.decodedMateRevC.end(),
                            task.totalDecodedAnchorsFlat.begin()
                                + myNumDecodedAnchors * decodedSequencePitchInBytes
                        );
                        task.totalDecodedAnchorsLengths.emplace_back(task.mateLength);

                        task.totalAnchorQualityScoresFlat.resize((myNumDecodedAnchors + 1) * qualityPitchInBytes);
                        assert(task.mateLength <= qualityPitchInBytes);
                        std::copy(
                            task.mateQualityScoresReversed.begin(),
                            task.mateQualityScoresReversed.end(),
                            task.totalAnchorQualityScoresFlat.begin()
                                + myNumDecodedAnchors * qualityPitchInBytes
                        );

                    }else{
                        const int newlength = h_outputAnchorLengths[i];

                        task.totalDecodedAnchorsFlat.resize((myNumDecodedAnchors+2) * decodedSequencePitchInBytes);
                        task.totalAnchorQualityScoresFlat.resize((myNumDecodedAnchors + 2) * qualityPitchInBytes);

                        std::string newq(h_outputAnchorQualities.data() + i * outputAnchorQualityPitchInBytes, newlength);

                        task.accumExtensionLengths = h_accumExtensionsLengths[i];
                        assert(newlength <= decodedSequencePitchInBytes);
                        std::copy_n(
                            h_outputAnchors.data() + i * outputAnchorPitchInBytes,
                            newlength,
                            task.totalDecodedAnchorsFlat.begin()
                                + myNumDecodedAnchors * decodedSequencePitchInBytes
                        );
                        task.totalDecodedAnchorsLengths.emplace_back(newlength);

                        assert(newlength <= qualityPitchInBytes);
                        std::copy_n(
                            h_outputAnchorQualities.data() + i * outputAnchorQualityPitchInBytes,
                            newlength,
                            task.totalAnchorQualityScoresFlat.begin()
                                + myNumDecodedAnchors * qualityPitchInBytes
                        );

                        task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);

                        task.accumExtensionLengths += newlength;
                        task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);
                        //task.totalDecodedAnchors.emplace_back(task.decodedMateRevC);
                        assert(task.mateLength <= decodedSequencePitchInBytes);
                        std::copy(
                            task.decodedMateRevC.begin(),
                            task.decodedMateRevC.end(),
                            task.totalDecodedAnchorsFlat.begin()
                                + (myNumDecodedAnchors + 1) * decodedSequencePitchInBytes
                        );
                        task.totalDecodedAnchorsLengths.emplace_back(task.mateLength);
                        
                        assert(task.mateLength <= qualityPitchInBytes);
                        std::copy(
                            task.mateQualityScoresReversed.begin(),
                            task.mateQualityScoresReversed.end(),
                            task.totalAnchorQualityScoresFlat.begin()
                                + (myNumDecodedAnchors + 1) * qualityPitchInBytes
                        );
                    }
                }
            }
        }

        handleEarlyExitOfTasks4();

        for(int i = 0; i < numTasks; i++){
            auto& task = tasks[i];

            task.iteration++;
        }

        setState(GpuReadExtender::State::BeforePrepareNextIteration);
    }

    void prepareNextIteration(){
        assert(state == GpuReadExtender::State::BeforePrepareNextIteration);

        //update list of active task indices
        h_newPositionsOfActiveTasks.resize(numTasks);
        int newPosSize = 0;

        assert(numTasks == int(tasks.size()));
        const int totalTasksBefore = tasks.size() + finishedTasks.size();

        std::vector<ExtensionTaskCpuData> newActiveTasks;
        std::vector<ExtensionTaskCpuData> newlyFinishedTasks;
        newActiveTasks.reserve(numTasks);
        newlyFinishedTasks.reserve(numTasks);

        for(int i = 0; i < numTasks; i++){
            if(tasks[i].isActive(insertSize, insertSizeStddev)){
                h_newPositionsOfActiveTasks[newPosSize] = i;

                newActiveTasks.emplace_back(std::move(tasks[i]));

                newPosSize++;
            }else{
                newlyFinishedTasks.emplace_back(std::move(tasks[i]));
            }
        }

        h_newPositionsOfActiveTasks.resize(newPosSize);
        std::swap(tasks, newActiveTasks);
        nvtx::push_range("addSortedFinishedTasks", 5);
        addSortedFinishedTasks(newlyFinishedTasks);
        nvtx::pop_range();

        std::size_t bytesFinishedTasks = 0;
        for(const auto& task : finishedTasks){
            bytesFinishedTasks += task.sizeInBytes();
        }
        std::size_t bytesTasks = 0;
        for(const auto& task : tasks){
            bytesTasks += task.sizeInBytes();
        }
        //std::cerr << "bytesTasks = " << bytesTasks << "\n";
        //std::cerr << "bytesFinishedTasks = " << bytesFinishedTasks << "\n";

        alltimetotalTaskBytes = std::max(alltimetotalTaskBytes, bytesTasks + bytesFinishedTasks);
        //std::cerr << "alltimetotalTaskBytes = " << alltimetotalTaskBytes << "\n";

        const int totalTasksAfter = tasks.size() + finishedTasks.size();
        assert(totalTasksAfter == totalTasksBefore);

        if(!isEmpty()){

            CachedDeviceUVector<int> d_newPositionsOfActiveTasks(h_newPositionsOfActiveTasks.size(), streams[0], *cubAllocator);

            cudaMemcpyAsync(
                d_newPositionsOfActiveTasks.data(),
                h_newPositionsOfActiveTasks.data(),
                sizeof(int) * h_newPositionsOfActiveTasks.size(),
                H2D,
                streams[0]
            ); CUERR;

            nvtx::push_range("updateBuffersForNextIteration", 6);

            updateBuffersForNextIteration(d_newPositionsOfActiveTasks.data(), d_newPositionsOfActiveTasks.size());

            nvtx::pop_range();
        }

        numTasks = tasks.size();

        if(!isEmpty()){
            setState(GpuReadExtender::State::BeforeHash);
        }else{
            setStateToFinished();
        }
        
    }


    void updateBuffersForNextIteration(int* d_newPositionsOfActiveTasks, int newNumTasks){
        nvtx::push_range("removeUsedIdsOfFinishedTasks", 6);

        removeUsedIdsOfFinishedTasks(d_newPositionsOfActiveTasks, newNumTasks);

        nvtx::pop_range();

        //compute selection flags of remaining tasks

        CachedDeviceUVector<bool> d_isActive(numTasks, streams[0], *cubAllocator);
        cudaMemsetAsync(d_isActive.data(), 0, numTasks, streams[0]); CUERR;

        helpers::lambda_kernel<<<SDIV(newNumTasks, 128), 128, 0, streams[0]>>>(
            [
                d_isActive = d_isActive.data(),
                d_newPositionsOfActiveTasks = d_newPositionsOfActiveTasks,
                newNumTasks
            ] __device__ (){
                const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                const int stride = blockDim.x * gridDim.x;

                for(int i = tid; i < newNumTasks; i += stride){
                    d_isActive[d_newPositionsOfActiveTasks[i]] = true;
                }
            }
        ); CUERR;

        //set new decoded anchors
        d_subjectSequencesDataDecoded.resizeUninitialized(newNumTasks * decodedSequencePitchInBytes, streams[0]);

        cubSelectFlagged(
            d_outputAnchors.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_isActive.data(), outputAnchorPitchInBytes)
            ),
            d_subjectSequencesDataDecoded.data(),
            thrust::make_discard_iterator(),
            numTasks * outputAnchorPitchInBytes,
            streams[0]
        );
        
        // set new anchor quality scores
        d_anchorQualityScores.resizeUninitialized(newNumTasks * qualityPitchInBytes, streams[0]);

        cubSelectFlagged(
            d_outputAnchorQualities.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_isActive.data(), outputAnchorQualityPitchInBytes)
            ),
            d_anchorQualityScores.data(),
            thrust::make_discard_iterator(),
            numTasks * outputAnchorQualityPitchInBytes,
            streams[0]
        );

        //set new anchorReadIds, mateReadIds, and anchor lengths

        CachedDeviceUVector<read_number> d_anchorReadIds2(alltimeMaximumNumberOfTasks, streams[0], *cubAllocator);
        CachedDeviceUVector<read_number> d_mateReadIds2(alltimeMaximumNumberOfTasks, streams[0], *cubAllocator);
        CachedDeviceUVector<int> d_inputMateLengths2(alltimeMaximumNumberOfTasks, streams[0], *cubAllocator);
        CachedDeviceUVector<bool> d_isPairedTask2(alltimeMaximumNumberOfTasks, streams[0], *cubAllocator);
        CachedDeviceUVector<int> d_accumExtensionsLengths2(alltimeMaximumNumberOfTasks, streams[0], *cubAllocator);

        d_anchorSequencesLength.resizeUninitialized(newNumTasks, streams[0]);

        cubSelectFlagged(
            thrust::make_zip_iterator(thrust::make_tuple(
                d_anchorReadIds.data(),
                d_mateReadIds.data(),
                d_outputAnchorLengths.data(),
                d_inputMateLengths.data(),
                d_isPairedTask.data(),
                d_accumExtensionsLengths.data()
            )),
            d_isActive.data(),
            thrust::make_zip_iterator(thrust::make_tuple(
                d_anchorReadIds2.data(),
                d_mateReadIds2.data(),
                d_anchorSequencesLength.data(),
                d_inputMateLengths2.data(),
                d_isPairedTask2.data(),
                d_accumExtensionsLengths2.data()
            )),
            thrust::make_discard_iterator(),
            numTasks,
            streams[0]
        );

        d_anchorReadIds2.erase(d_anchorReadIds2.begin() + newNumTasks, d_anchorReadIds2.end(), streams[0]);
        d_mateReadIds2.erase(d_mateReadIds2.begin() + newNumTasks, d_mateReadIds2.end(), streams[0]);
        d_inputMateLengths2.erase(d_inputMateLengths2.begin() + newNumTasks, d_inputMateLengths2.end(), streams[0]);
        d_isPairedTask2.erase(d_isPairedTask2.begin() + newNumTasks, d_isPairedTask2.end(), streams[0]);
        d_accumExtensionsLengths2.erase(d_accumExtensionsLengths2.begin() + newNumTasks, d_accumExtensionsLengths2.end(), streams[0]);

        std::swap(d_anchorReadIds, d_anchorReadIds2);
        std::swap(d_mateReadIds, d_mateReadIds2);
        std::swap(d_inputMateLengths, d_inputMateLengths2);
        std::swap(d_isPairedTask, d_isPairedTask2);
        std::swap(d_accumExtensionsLengths, d_accumExtensionsLengths2);


        //set new encoded mate data

        CachedDeviceUVector<unsigned int> d_inputanchormatedata2(alltimeMaximumNumberOfTasks * encodedSequencePitchInInts, streams[0], *cubAllocator);

        cubSelectFlagged(
            d_inputanchormatedata.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_isActive.data(), encodedSequencePitchInInts)
            ),
            d_inputanchormatedata2.data(),
            thrust::make_discard_iterator(),
            numTasks * encodedSequencePitchInInts,
            streams[0]
        );

        d_inputanchormatedata2.erase(d_inputanchormatedata2.begin() + newNumTasks * encodedSequencePitchInInts, d_inputanchormatedata2.end(), streams[0]);

        std::swap(d_inputanchormatedata, d_inputanchormatedata2);
        
        //convert new anchors to 2bit representation

        d_subjectSequencesData.resizeUninitialized(newNumTasks * encodedSequencePitchInInts, streams[0]);

        readextendergpukernels::encodeSequencesTo2BitKernel<8>
        <<<SDIV(newNumTasks, (128 / 8)), 128, 0, streams[0]>>>(
            d_subjectSequencesData.data(),
            d_subjectSequencesDataDecoded.data(),
            d_anchorSequencesLength.data(),
            decodedSequencePitchInBytes,
            encodedSequencePitchInInts,
            newNumTasks
        ); CUERR;


        //shrink remaining buffers
        d_numCandidatesPerAnchor.erase(d_numCandidatesPerAnchor.begin() + newNumTasks, d_numCandidatesPerAnchor.end(), streams[0]);
        d_numCandidatesPerAnchorPrefixSum.erase(d_numCandidatesPerAnchorPrefixSum.begin() + (newNumTasks + 1), d_numCandidatesPerAnchorPrefixSum.end(), streams[0]);
    }

    void removeUsedIdsOfFinishedTasks(int* d_newPositionsOfActiveTasks, int newNumTasks){

        if(newNumTasks == 0) return;

        assert(newNumTasks <= numTasks);

        // {
        //     std::size_t free, total;
        //     cudaMemGetInfo(&free, &total);
        //     std::cerr << "before removeUsedIdsOfFinishedTasks " << free << "\n";
        // }


        //update used ids

        {
            CachedDeviceUVector<int> d_numUsedReadIdsPerAnchor2(newNumTasks, streams[0], *cubAllocator);
            CachedDeviceUVector<int> d_numUsedReadIdsPerAnchorPrefixSum2(newNumTasks, streams[0], *cubAllocator);      

            helpers::lambda_kernel<<<SDIV(newNumTasks,256), 256, 0, streams[0]>>>(
                [
                    indicesOfActiveTasks = d_newPositionsOfActiveTasks,
                    newNumTasks,
                    d_numUsedReadIdsPerAnchorOut = d_numUsedReadIdsPerAnchor2.data(),
                    d_numUsedReadIdsPerAnchorIn = d_numUsedReadIdsPerAnchor.data()
                ] __device__ (){
                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    const int stride = blockDim.x * gridDim.x;

                    for(int t = tid; t < newNumTasks; t += stride){
                        d_numUsedReadIdsPerAnchorOut[t] = d_numUsedReadIdsPerAnchorIn[indicesOfActiveTasks[t]];
                    }
                }
            ); CUERR;

            cubReduceSum(
                d_numUsedReadIdsPerAnchor2.data(), 
                h_numUsedReadIds.data(),
                newNumTasks,
                streams[0]
            );

            cudaEventRecord(h_numUsedReadIdsEvent, streams[0]);

            cubExclusiveSum(
                d_numUsedReadIdsPerAnchor2.data(), 
                d_numUsedReadIdsPerAnchorPrefixSum2.data(),  
                newNumTasks,
                streams[0]
            );

            cudaEventSynchronize(h_numUsedReadIdsEvent); CUERR; //wait until h_numUsedReadIds is ready

            CachedDeviceUVector<read_number> d_usedReadIds2(*h_numUsedReadIds, streams[0], *cubAllocator);
            CachedDeviceUVector<int> d_segmentIdsOfUsedReadIds2(*h_numUsedReadIds, streams[0], *cubAllocator);        

            const int possibleNumWarps = newNumTasks;
            const int possibleNumBlocks = SDIV(possibleNumWarps, 128 / 32);
            const int numBlocks = std::min(256, possibleNumBlocks);

            readextendergpukernels::compactUsedIdsOfSelectedTasks<32><<<numBlocks, 128, 0, streams[0]>>>(
                d_newPositionsOfActiveTasks,
                newNumTasks,
                d_usedReadIds.data(),
                d_usedReadIds2.data(),
                d_segmentIdsOfUsedReadIds2.data(),
                d_numUsedReadIdsPerAnchor2.data(),
                d_numUsedReadIdsPerAnchorPrefixSum.data(), 
                d_numUsedReadIdsPerAnchorPrefixSum2.data()
            ); CUERR;

            std::swap(d_usedReadIds, d_usedReadIds2);
            std::swap(d_numUsedReadIdsPerAnchor, d_numUsedReadIdsPerAnchor2);
            std::swap(d_numUsedReadIdsPerAnchorPrefixSum, d_numUsedReadIdsPerAnchorPrefixSum2);
            std::swap(d_segmentIdsOfUsedReadIds, d_segmentIdsOfUsedReadIds2);
        }

        //update fully used ids
        
        {
            CachedDeviceUVector<int> d_numFullyUsedReadIdsPerAnchor2(newNumTasks, streams[0], *cubAllocator);
            CachedDeviceUVector<int> d_numFullyUsedReadIdsPerAnchorPrefixSum2(newNumTasks, streams[0], *cubAllocator);  

            helpers::lambda_kernel<<<SDIV(newNumTasks,256), 256, 0, streams[0]>>>(
                [
                    indicesOfActiveTasks = d_newPositionsOfActiveTasks,
                    newNumTasks,
                    d_numFullyUsedReadIdsPerAnchorOut = d_numFullyUsedReadIdsPerAnchor2.data(),
                    d_numFullyUsedReadIdsPerAnchorIn = d_numFullyUsedReadIdsPerAnchor.data()
                ] __device__ (){
                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    const int stride = blockDim.x * gridDim.x;

                    for(int t = tid; t < newNumTasks; t += stride){
                        d_numFullyUsedReadIdsPerAnchorOut[t] = d_numFullyUsedReadIdsPerAnchorIn[indicesOfActiveTasks[t]];
                    }
                }
            ); CUERR;
            
            cubReduceSum(
                d_numFullyUsedReadIdsPerAnchor2.data(), 
                h_numFullyUsedReadIds.data(),
                newNumTasks,
                streams[0]
            );

            cudaEventRecord(h_numFullyUsedReadIdsEvent, streams[0]);

            cubExclusiveSum(
                d_numFullyUsedReadIdsPerAnchor2.data(), 
                d_numFullyUsedReadIdsPerAnchorPrefixSum2.data(),  
                newNumTasks,
                streams[0]
            );

            cudaEventSynchronize(h_numFullyUsedReadIdsEvent); CUERR; //wait until h_numFullyUsedReadIds is ready

            CachedDeviceUVector<read_number> d_fullyUsedReadIds2(*h_numFullyUsedReadIds, streams[0], *cubAllocator);
            CachedDeviceUVector<int> d_segmentIdsOfFullyUsedReadIds2(*h_numFullyUsedReadIds, streams[0], *cubAllocator); 

            const int possibleNumWarps = newNumTasks;
            const int possibleNumBlocks = SDIV(possibleNumWarps, 128 / 32);
            const int numBlocks = std::min(256, possibleNumBlocks);

            readextendergpukernels::compactUsedIdsOfSelectedTasks<32><<<numBlocks, 128, 0, streams[0]>>>(
                d_newPositionsOfActiveTasks,
                newNumTasks,
                d_fullyUsedReadIds.data(),
                d_fullyUsedReadIds2.data(),
                d_segmentIdsOfFullyUsedReadIds2.data(),
                d_numFullyUsedReadIdsPerAnchor2.data(),
                d_numFullyUsedReadIdsPerAnchorPrefixSum.data(), 
                d_numFullyUsedReadIdsPerAnchorPrefixSum2.data()
            ); CUERR;

            std::swap(d_fullyUsedReadIds, d_fullyUsedReadIds2);
            std::swap(d_numFullyUsedReadIdsPerAnchor, d_numFullyUsedReadIdsPerAnchor2);
            std::swap(d_numFullyUsedReadIdsPerAnchorPrefixSum, d_numFullyUsedReadIdsPerAnchorPrefixSum2);
            std::swap(d_segmentIdsOfFullyUsedReadIds, d_segmentIdsOfFullyUsedReadIds2);

        }
    }


    //construct results for each group of 4 tasks belonging to the same read pair
    std::vector<extension::ExtendResult> constructResults4(){

        //determine tasks in groups of 4
        std::vector<ExtensionTaskCpuData> finishedTasks4{};
        std::vector<ExtensionTaskCpuData> finishedTasksNot4{};

        {
            auto l = finishedTasks.begin();
            auto r = finishedTasks.begin();

            while(r != finishedTasks.end()){

                while(r != finishedTasks.end() && l->pairId == r->pairId){
                    ++r;
                }

                if(std::distance(l,r) == 4){
                    finishedTasks4.insert(
                        finishedTasks4.end(), 
                        std::make_move_iterator(l), 
                        std::make_move_iterator(r)
                    );
                }else{
                    assert(std::distance(l,r) < 4);
                    finishedTasksNot4.insert(
                        finishedTasksNot4.end(), 
                        std::make_move_iterator(l), 
                        std::make_move_iterator(r)
                    );
                }

                l = r;
            }
        }

        //std::cerr << "finishedTasks: " << finishedTasks.size() << ", finishedTasks4: " << finishedTasks4.size() << ", finishedTasksNot4: " << finishedTasksNot4.size() << "\n";

        //update remaining finished tasks
        std::swap(finishedTasks, finishedTasksNot4);

        nvtx::push_range("clear finishedTasksNot4", 0);
        finishedTasksNot4.clear();
        nvtx::pop_range();

        int resultMSAColumnPitchInElements = 1024;

        nvtx::push_range("constructResults4", 2);
        {
            const int numFinishedTasks = finishedTasks4.size();
            if(numFinishedTasks == 0){
                return std::vector<extension::ExtendResult>{};
            }
            cudaStream_t stream = streams[0];

            h_numCandidatesPerAnchor.resize(numFinishedTasks);
            h_numCandidatesPerAnchorPrefixSum.resize(numFinishedTasks + 1);

            for(int i = 0; i < numFinishedTasks; i++){
                const auto& task = finishedTasks4[i];

                h_numCandidatesPerAnchor[i] = task.totalDecodedAnchorsLengths.size() - 1;
            }

            h_numCandidatesPerAnchorPrefixSum[0] = 0;
            std::inclusive_scan(
                h_numCandidatesPerAnchor.begin(),
                h_numCandidatesPerAnchor.end(),
                h_numCandidatesPerAnchorPrefixSum.begin() + 1
            );
            const int numCandidates = h_numCandidatesPerAnchorPrefixSum[numFinishedTasks];
            assert(numCandidates >= 0);

            //if there are no candidates, the resulting sequences will be identical to the input anchors. no computing required
            if(numCandidates == 0){

                resultMSAColumnPitchInElements = decodedSequencePitchInBytes;

                h_outputAnchors.resize(numFinishedTasks * resultMSAColumnPitchInElements);
                h_outputAnchorQualities.resize(numFinishedTasks * resultMSAColumnPitchInElements);
                h_anchorSequencesLength.resize(numFinishedTasks);

                for(int i = 0; i < numFinishedTasks; i++){
                    const auto& task = finishedTasks4[i];

                    const int num = h_numCandidatesPerAnchor[i];

                    std::copy(
                        task.totalDecodedAnchorsFlat.begin(),
                        task.totalDecodedAnchorsFlat.begin() + task.totalDecodedAnchorsLengths[0],
                        h_outputAnchors + i * resultMSAColumnPitchInElements
                    );

                    std::fill(
                        h_outputAnchorQualities + i * resultMSAColumnPitchInElements,
                        h_outputAnchorQualities + i * resultMSAColumnPitchInElements + task.totalDecodedAnchorsLengths[0],
                        'I'
                    );

                    h_anchorSequencesLength[i] = task.totalDecodedAnchorsLengths[0];
                }

            }else{

                d_numCandidatesPerAnchor.resizeUninitialized(numFinishedTasks, stream);
                d_numCandidatesPerAnchorPrefixSum.resizeUninitialized(numFinishedTasks + 1, stream);

                cudaMemcpyAsync(
                    d_numCandidatesPerAnchor.data(),
                    h_numCandidatesPerAnchor.data(),
                    sizeof(int) * numFinishedTasks,
                    H2D,
                    stream
                ); CUERR;

                cudaMemcpyAsync(
                    d_numCandidatesPerAnchorPrefixSum.data(),
                    h_numCandidatesPerAnchorPrefixSum.data(),
                    sizeof(int) * (numFinishedTasks + 1),
                    H2D,
                    stream
                ); CUERR;

                //copy anchor lengths, anchor sequences and mateHasBeenFound flag

                h_outputAnchors.resize(numFinishedTasks * decodedSequencePitchInBytes);
                h_anchorSequencesLength.resize(numFinishedTasks);
                h_outputMateHasBeenFound.resize(numFinishedTasks);

                CachedDeviceUVector<int> d_anchorSequencesLength2(numFinishedTasks, stream, *cubAllocator);
                CachedDeviceUVector<unsigned int> d_inputAnchors(numFinishedTasks * encodedSequencePitchInInts, stream, *cubAllocator);
                CachedDeviceUVector<char> d_subjectSequencesDataDecoded2(numFinishedTasks * decodedSequencePitchInBytes, stream, *cubAllocator);
                CachedDeviceUVector<bool> d_mateHasBeenFound(numFinishedTasks, stream, *cubAllocator);

                for(int i = 0; i < numFinishedTasks; i++){
                    const auto& task = finishedTasks4[i];

                    const int num = h_numCandidatesPerAnchor[i];

                    std::copy(
                        task.totalDecodedAnchorsFlat.begin(),
                        task.totalDecodedAnchorsFlat.begin() + decodedSequencePitchInBytes,
                        h_outputAnchors + i * decodedSequencePitchInBytes
                    );

                    h_anchorSequencesLength[i] = task.totalDecodedAnchorsLengths[0];
                    h_outputMateHasBeenFound[i] = task.mateHasBeenFound;
                }

                cudaMemcpyAsync(
                    d_anchorSequencesLength2.data(),
                    h_anchorSequencesLength.data(),
                    sizeof(int) * numFinishedTasks,
                    H2D,
                    stream
                ); CUERR;

                cudaMemcpyAsync(
                    d_mateHasBeenFound.data(),
                    h_outputMateHasBeenFound.data(),
                    sizeof(bool) * numFinishedTasks,
                    H2D,
                    stream
                ); CUERR;

                cudaMemcpyAsync(
                    d_subjectSequencesDataDecoded2.data(),
                    h_outputAnchors.data(),
                    sizeof(char) * decodedSequencePitchInBytes * numFinishedTasks,
                    H2D,
                    stream
                ); CUERR;

                readextendergpukernels::encodeSequencesTo2BitKernel<8>
                <<<SDIV(numFinishedTasks, (128 / 8)), 128, 0, streams[0]>>>(
                    d_inputAnchors.data(),
                    d_subjectSequencesDataDecoded2.data(),
                    d_anchorSequencesLength2.data(),
                    decodedSequencePitchInBytes,
                    encodedSequencePitchInInts,
                    numFinishedTasks
                ); CUERR;

                CachedDeviceUVector<char> d_inputAnchorQualities(numFinishedTasks * qualityPitchInBytes, stream, *cubAllocator);

                //copy anchor qualities
                h_outputAnchorQualities.resize(numFinishedTasks * qualityPitchInBytes);

                for(int i = 0; i < numFinishedTasks; i++){
                    const auto& task = finishedTasks4[i];

                    const int num = h_numCandidatesPerAnchor[i];

                    std::copy(
                        task.totalAnchorQualityScoresFlat.begin(),
                        task.totalAnchorQualityScoresFlat.begin() + qualityPitchInBytes,
                        h_outputAnchorQualities + i * qualityPitchInBytes
                    );
                }

                cudaMemcpyAsync(
                    d_inputAnchorQualities.data(),
                    h_outputAnchorQualities.data(),
                    sizeof(char) * qualityPitchInBytes * numFinishedTasks,
                    H2D,
                    stream
                ); CUERR;

                //synchronize before reusing pinned buffers
                cudaStreamSynchronize(stream); CUERR;

                //copy "candidate" sequences

                h_outputAnchors.resize(numCandidates * decodedSequencePitchInBytes);
                h_anchorSequencesLength.resize(numCandidates);

                d_candidateSequencesLength.resizeUninitialized(numCandidates, stream);
                d_candidateSequencesData.resizeUninitialized(numCandidates * encodedSequencePitchInInts, stream);
                CachedDeviceUVector<char> d_candidateSequencesDataDecoded(decodedSequencePitchInBytes * numCandidates, stream, *cubAllocator);

                for(int i = 0; i < numFinishedTasks; i++){
                    const auto& task = finishedTasks4[i];

                    const int num = h_numCandidatesPerAnchor[i];
                    const int offset = h_numCandidatesPerAnchorPrefixSum[i];

                    std::copy(
                        task.totalDecodedAnchorsFlat.begin() + decodedSequencePitchInBytes,
                        task.totalDecodedAnchorsFlat.end(),
                        h_outputAnchors + offset * decodedSequencePitchInBytes
                    );

                    std::copy(
                        task.totalDecodedAnchorsLengths.begin() + 1,
                        task.totalDecodedAnchorsLengths.end(),
                        h_anchorSequencesLength + offset
                    );
                }

                cudaMemcpyAsync(
                    d_candidateSequencesLength.data(),
                    h_anchorSequencesLength.data(),
                    sizeof(int) * numCandidates,
                    H2D,
                    stream
                ); CUERR;

                cudaMemcpyAsync(
                    d_candidateSequencesDataDecoded.data(),
                    h_outputAnchors.data(),
                    sizeof(char) * decodedSequencePitchInBytes * numCandidates,
                    H2D,
                    stream
                ); CUERR;

                readextendergpukernels::encodeSequencesTo2BitKernel<8>
                <<<SDIV(numCandidates, (128 / 8)), 128, 0, streams[0]>>>(
                    d_candidateSequencesData.data(),
                    d_candidateSequencesDataDecoded.data(),
                    d_candidateSequencesLength.data(),
                    decodedSequencePitchInBytes,
                    encodedSequencePitchInInts,
                    numCandidates
                ); CUERR;

                //copy "candidate" qualities
                // h_outputAnchorQualities.resize(numCandidates * qualityPitchInBytes);
                // CachedDeviceUVector<char> d_candidateQualityScores(qualityPitchInBytes * numCandidates, stream, *cubAllocator);

                // for(int i = 0; i < numFinishedTasks; i++){
                //     const auto& task = finishedTasks4[i];

                //     const int num = h_numCandidatesPerAnchor[i];
                //     const int offset = h_numCandidatesPerAnchorPrefixSum[i];

                //     std::copy(
                //         task.totalDecodedAnchorsFlat.begin() + qualityPitchInBytes,
                //         task.totalDecodedAnchorsFlat.end(),
                //         h_outputAnchorQualities + offset * qualityPitchInBytes
                //     );
                // }

                // cudaMemcpyAsync(
                //     d_candidateQualityScores,
                //     h_outputAnchorQualities.data(),
                //     sizeof(char) * qualityPitchInBytes * numCandidates,
                //     H2D,
                //     stream
                // ); CUERR;

                //synchronize before reusing pinned buffers
                cudaStreamSynchronize(stream); CUERR;

                //sequence data has been transfered to gpu. now set up remaining msa input data

                d_alignment_overlaps.resizeUninitialized(numCandidates, stream);
                d_alignment_shifts.resizeUninitialized(numCandidates, stream);
                d_alignment_nOps.resizeUninitialized(numCandidates, stream);
                d_alignment_best_alignment_flags.resizeUninitialized(numCandidates, stream);
                d_isPairedCandidate.resizeUninitialized(numCandidates, stream);
                
                helpers::call_fill_kernel_async(d_alignment_overlaps.begin(), numCandidates, 100, stream);
                helpers::call_fill_kernel_async(d_alignment_nOps.begin(), numCandidates, 0, stream);
                helpers::call_fill_kernel_async(d_alignment_best_alignment_flags.begin(), numCandidates, BestAlignment_t::Forward, stream);
                helpers::call_fill_kernel_async(d_isPairedCandidate.begin(), numCandidates, false, stream);

                h_sizeOfGapToMate.resize(numCandidates);
                for(int i = 0; i < numFinishedTasks; i++){
                    const auto& task = finishedTasks4[i];

                    const int offset = h_numCandidatesPerAnchorPrefixSum[i];

                    std::copy(
                        task.totalAnchorBeginInExtendedRead.begin() + 1,
                        task.totalAnchorBeginInExtendedRead.end(),
                        h_sizeOfGapToMate + offset
                    );

                    //assert(task.totalAnchorBeginInExtendedRead.back() + task.totalDecodedAnchorsLengths.back() <= insertSize + insertSizeStddev);
                }

                cudaMemcpyAsync(
                    d_alignment_shifts.data(),
                    h_sizeOfGapToMate.data(),
                    sizeof(int) * numCandidates,
                    H2D,
                    stream
                ); CUERR;

                //all input data ready. now set up msa

                CachedDeviceUVector<int> indices1(numCandidates, stream, *cubAllocator);

                helpers::lambda_kernel<<<numFinishedTasks, 128, 0, stream>>>(
                    [
                        indices1 = indices1.data(),
                        d_numCandidatesPerAnchor = d_numCandidatesPerAnchor.data(),
                        d_numCandidatesPerAnchorPrefixSum = d_numCandidatesPerAnchorPrefixSum.data()
                    ] __device__ (){
                        const int num = d_numCandidatesPerAnchor[blockIdx.x];
                        const int offset = d_numCandidatesPerAnchorPrefixSum[blockIdx.x];
                        
                        for(int i = threadIdx.x; i < num; i += blockDim.x){
                            indices1[offset + i] = i;
                        }
                    }
                );

                *h_numAnchors = numFinishedTasks;

                multiMSA.construct(
                    d_alignment_overlaps.data(),
                    d_alignment_shifts.data(),
                    d_alignment_nOps.data(),
                    d_alignment_best_alignment_flags.data(),
                    indices1.data(),
                    d_numCandidatesPerAnchor.data(),
                    d_numCandidatesPerAnchorPrefixSum.data(),
                    d_anchorSequencesLength2.data(),
                    d_inputAnchors.data(),
                    nullptr, //anchor qualities
                    numFinishedTasks,
                    d_candidateSequencesLength.data(),
                    d_candidateSequencesData.data(),
                    nullptr, //candidate qualities
                    d_isPairedCandidate.data(),
                    numCandidates,
                    h_numAnchors.data(), //d_numAnchors
                    encodedSequencePitchInInts,
                    qualityPitchInBytes,
                    false, //useQualityScores
                    goodAlignmentProperties->maxErrorRate,
                    gpu::MSAColumnCount::unknown(),
                    stream
                );

                assert(multiMSA.numMSAs == numFinishedTasks);

                indices1.destroy();

                resultMSAColumnPitchInElements = multiMSA.getMaximumMsaWidth();

                //compute quality of consensus
                d_consensusQuality.resizeUninitialized(numFinishedTasks * resultMSAColumnPitchInElements, stream);

                CachedDeviceUVector<char> d_decodedConsensus(numFinishedTasks * resultMSAColumnPitchInElements, stream, *cubAllocator);
                CachedDeviceUVector<int> d_resultLengths(numFinishedTasks, stream, *cubAllocator);
                
                h_outputAnchorQualities.resize(numFinishedTasks * resultMSAColumnPitchInElements);
                h_outputAnchors.resize(numFinishedTasks * resultMSAColumnPitchInElements);
                h_anchorSequencesLength.resize(numFinishedTasks);

                multiMSA.computeConsensusQuality(
                    d_consensusQuality.data(),
                    resultMSAColumnPitchInElements,
                    stream
                );

                multiMSA.computeConsensus(
                    d_decodedConsensus.data(),
                    resultMSAColumnPitchInElements,
                    stream
                );

                multiMSA.computeMsaSizes(
                    d_resultLengths.data(),
                    stream
                );

                helpers::lambda_kernel<<<SDIV(numFinishedTasks,(128 / 32)), 128,0, stream>>>(
                    [
                        resultMSAColumnPitchInElements = resultMSAColumnPitchInElements,
                        numFinishedTasks = numFinishedTasks,
                        d_decodedConsensus = d_decodedConsensus.data(),
                        d_consensusQuality = d_consensusQuality.data(),
                        d_resultLengths = d_resultLengths.data(),
                        d_inputAnchors = d_inputAnchors.data(),
                        d_inputAnchorLengths = d_anchorSequencesLength2.data(),
                        d_inputAnchorQualities = d_inputAnchorQualities.data(),
                        d_mateHasBeenFound = d_mateHasBeenFound.data(),
                        encodedSequencePitchInInts = encodedSequencePitchInInts,
                        qualityPitchInBytes = qualityPitchInBytes
                    ] __device__ (){
                        // __shared__ unsigned int sharedEncodedSequence[4][32];
                        // assert(encodedSequencePitchInInts <= 32);

                        const int numPairs = numFinishedTasks / 4;

                        auto group = cg::tiled_partition<32>(cg::this_thread_block());
                        const int groupIdInBlock = threadIdx.x / 32;

                        for(int pair = blockIdx.x; pair < numPairs; pair += gridDim.x){
                            const int resultLength = d_resultLengths[4 * pair + groupIdInBlock];
                            const int anchorLength = d_inputAnchorLengths[4 * pair + groupIdInBlock];
                            const unsigned int* const inputAnchor = &d_inputAnchors[(4 * pair + groupIdInBlock) * encodedSequencePitchInInts];
                            char* const resultSequence = &d_decodedConsensus[(4 * pair + groupIdInBlock) * resultMSAColumnPitchInElements];
                            const char* const inputQuality = &d_inputAnchorQualities[(4 * pair + groupIdInBlock) * qualityPitchInBytes];
                            char* const resultQuality = &d_consensusQuality[(4 * pair + groupIdInBlock) * resultMSAColumnPitchInElements];

                            SequenceHelpers::decodeSequence2Bit<int4>(group, inputAnchor, anchorLength, resultSequence);

                            //copy anchor quality
                            {
                                const int numIters = anchorLength / sizeof(int);
                                for(int i = group.thread_rank(); i < numIters; i += group.size()){
                                    ((int*)resultQuality)[i] = ((const int*)inputQuality)[i];
                                }
                                const int remaining = anchorLength - sizeof(int) * numIters;
                                if(remaining > 0){
                                    for(int i = group.thread_rank(); i < remaining; i += group.size()){
                                        resultQuality[sizeof(int) * numIters + i] = inputQuality[sizeof(int) * numIters + i];
                                    }
                                }
                            }

                            if(d_mateHasBeenFound[4 * pair + groupIdInBlock]){
                                const int mateLength = d_inputAnchorLengths[4 * pair + groupIdInBlock + 1];
                                const unsigned int* const anchorMate = &d_inputAnchors[(4 * pair + groupIdInBlock + 1) * encodedSequencePitchInInts];
                                const char* const anchorMateQuality = &d_inputAnchorQualities[(4 * pair + groupIdInBlock + 1) * qualityPitchInBytes];
                                SequenceHelpers::decodeSequence2Bit<char>(group, anchorMate, mateLength, resultSequence + resultLength - mateLength);

                                for(int i = group.thread_rank(); i < mateLength; i += group.size()){
                                    resultQuality[resultLength - mateLength + i] = anchorMateQuality[i];
                                }
                            }
                        }
                    }
                ); CUERR;

                cudaMemcpyAsync(
                    h_outputAnchors.data(),
                    d_decodedConsensus.data(),
                    sizeof(char) * numFinishedTasks * resultMSAColumnPitchInElements,
                    D2H,
                    stream
                ); CUERR;

                cudaMemcpyAsync(
                    h_outputAnchorQualities.data(),
                    d_consensusQuality.data(),
                    sizeof(char) * numFinishedTasks * resultMSAColumnPitchInElements,
                    D2H,
                    stream
                ); CUERR;

                cudaMemcpyAsync(
                    h_anchorSequencesLength.data(),
                    d_resultLengths.data(),
                    sizeof(int) * numFinishedTasks,
                    D2H,
                    stream
                ); CUERR;

                d_consensusQuality.destroy();

                cudaStreamSynchronize(stream); CUERR;
            }

        }
        nvtx::pop_range();

        nvtx::push_range("newresultcombination", 6);

        std::vector<int> dataRead2Begins(finishedTasks4.size());
        for(std::size_t t = 0; t < finishedTasks4.size(); t++){
            const auto& task = finishedTasks4[t];
            if(task.mateHasBeenFound){
                dataRead2Begins[t] = h_anchorSequencesLength[t] - task.decodedMateRevC.size();
            }else{
                dataRead2Begins[t] = -1;
            }
        }

        std::vector<extension::ExtendResult> extendResultsTmp = makePairResultsFromFinishedTasks(
            finishedTasks4.size() / 4,
            finishedTasks4.size(),
            finishedTasks4.data(),
            dataRead2Begins.data(),
            h_anchorSequencesLength.data(),
            h_outputAnchors.data(),
            h_outputAnchorQualities.data(),
            resultMSAColumnPitchInElements,
            resultMSAColumnPitchInElements
        );

        nvtx::pop_range();

        nvtx::push_range("updateResults", 3);

        std::vector<extension::ExtendResult> extendResults;
        extendResults.reserve(finishedTasks4.size());

        for(std::size_t t = 0; t < finishedTasks4.size(); t++){
            const auto& task = finishedTasks4[t];

            //std::cerr << task.allFullyUsedCandidateReadIdPairs.size() << " / " << task.allUsedCandidateReadIdPairs.size() << "\n";

            extension::ExtendResult extendResult;
            extendResult.direction = task.direction;
            extendResult.numIterations = task.iteration;
            extendResult.aborted = task.abortReason != extension::AbortReason::None;
            extendResult.abortReason = task.abortReason;
            extendResult.readId1 = task.myReadId;
            extendResult.readId2 = task.mateReadId;
            extendResult.originalLength = task.myLength;
            extendResult.originalMateLength = task.mateLength;
            extendResult.read1begin = 0;
            extendResult.goodscore = task.goodscore;

            // std::cerr << "task " << x << ". iteration = " << task.iteration << ", abort = " << task.abort << ", abortReasond = " << extension::to_string(task.abortReason)
            //     << ", matefound = " << task.mateHasBeenFound << ", id = " << task.id << ", myReadid = " << task.myReadId << "\n";

            // x++;

            //construct extended read

            const int numsteps = task.totalDecodedAnchorsLengths.size();

            const int gpuLength = h_anchorSequencesLength[t];
            std::string extendedRead(h_outputAnchors.data() + t * resultMSAColumnPitchInElements, gpuLength);
            std::string extendedReadQuality(h_outputAnchorQualities.data() + t * resultMSAColumnPitchInElements, gpuLength);

            if(task.mateHasBeenFound){
                extendResult.read2begin = extendedRead.length() - task.decodedMateRevC.size();
            }else{
                extendResult.read2begin = -1;
            }

            extendResult.extendedRead = std::move(extendedRead);
            extendResult.qualityScores = std::move(extendedReadQuality);

            extendResult.mateHasBeenFound = task.mateHasBeenFound;

            extendResults.emplace_back(std::move(extendResult));
        }

        nvtx::pop_range();

        nvtx::push_range("combinePairedEndDirectionResults4", 4);

        // std::vector<extension::ExtendResult> extendResultsCombined = extension::combinePairedEndDirectionResults4(
        //     extendResults,
        //     insertSize,
        //     insertSizeStddev
        // );

        extension::combinePairedEndDirectionResults4Inplace(
            extendResults,
            insertSize,
            insertSizeStddev
        );

        nvtx::pop_range();

        if(extendResultsTmp != extendResults){
            //assert(extendResultsTmp == extendResults);
            assert(false);
        }

        return extendResults;
    }

    //helpers

    void setGpuSegmentIds(
        int* d_segmentIds,
        int numSegments,
        int numElements,
        const int* d_numElementsPerSegment,
        const int* d_numElementsPerSegmentPrefixSum,
        cudaStream_t stream
    ) const {
        cudaMemsetAsync(d_segmentIds, 0, sizeof(int) * numElements, stream); CUERR;
        
        readextendergpukernels::setFirstSegmentIdsKernel<<<SDIV(numSegments, 256), 256, 0, stream>>>(
            d_numElementsPerSegment,
            d_segmentIds,
            d_numElementsPerSegmentPrefixSum,
            numSegments
        );

        cubInclusiveScan(
            d_segmentIds, 
            d_segmentIds, 
            cub::Max{},
            numElements,
            stream
        );
    }

    void loadCandidateQualityScores(cudaStream_t stream, char* d_qualityscores){
        char* outputQualityScores = d_qualityscores;

        if(correctionOptions->useQualityScores){
            h_candidateReadIds.resize(totalNumCandidates);
            

            cudaMemcpyAsync(
                h_candidateReadIds.data(),
                d_candidateReadIds.data(),
                sizeof(read_number) * totalNumCandidates,
                D2H,
                stream
            ); CUERR;

            cudaStreamSynchronize(stream); CUERR;


            gpuReadStorage->gatherQualities(
                readStorageHandle,
                outputQualityScores,
                qualityPitchInBytes,
                h_candidateReadIds.data(),
                d_candidateReadIds.data(),
                totalNumCandidates,
                stream
            );

        }else{
            helpers::call_fill_kernel_async(
                outputQualityScores,
                qualityPitchInBytes * totalNumCandidates,
                'I',
                stream
            ); CUERR;
        }        
    }

    void compactCandidateDataByFlagsExcludingAlignments(
        const bool* d_keepFlags,
        cudaStream_t stream
    ){
        CachedDeviceUVector<int> d_numCandidatesPerAnchor2(numTasks, stream, *cubAllocator);

        cubSegmentedReduceSum(
            d_keepFlags,
            d_numCandidatesPerAnchor2.data(),
            numTasks,
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_numCandidatesPerAnchorPrefixSum.data() + 1,
            stream
        );

        auto d_zip_data = thrust::make_zip_iterator(
            thrust::make_tuple(
                d_candidateReadIds.data(),
                d_candidateSequencesLength.data(),
                d_isPairedCandidate.data()
            )
        );

        CachedDeviceUVector<int> d_candidateSequencesLength2(totalNumCandidates, stream, *cubAllocator);
        CachedDeviceUVector<read_number> d_candidateReadIds2(totalNumCandidates, stream, *cubAllocator);
        CachedDeviceUVector<bool> d_isPairedCandidate2(totalNumCandidates, stream, *cubAllocator);
  
        auto d_zip_data_tmp = thrust::make_zip_iterator(
            thrust::make_tuple(
                d_candidateReadIds2.data(),
                d_candidateSequencesLength2.data(),
                d_isPairedCandidate2.data()
            )
        );

        //compact 1d arrays

        cubSelectFlagged(
            d_zip_data, 
            d_keepFlags, 
            d_zip_data_tmp, 
            h_numCandidates.data(), 
            totalNumCandidates, 
            stream
        );

        cudaEventRecord(h_numCandidatesEvent, stream); CUERR;


        cudaMemsetAsync(d_numCandidatesPerAnchorPrefixSum.data(), 0, sizeof(int), stream); CUERR;
        cubInclusiveSum(
            d_numCandidatesPerAnchor2.data(), 
            d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            numTasks, 
            stream
        );
        std::swap(d_numCandidatesPerAnchor, d_numCandidatesPerAnchor2); 

        d_numCandidatesPerAnchor2.destroy();

        cudaEventSynchronize(h_numCandidatesEvent); CUERR; //wait for h_numCandidates
        const int oldTotalNumCandidates = totalNumCandidates;

        totalNumCandidates = *h_numCandidates; 

        d_candidateReadIds2.erase(d_candidateReadIds2.begin() + totalNumCandidates, d_candidateReadIds2.end(), stream);
        d_candidateSequencesLength2.erase(d_candidateSequencesLength2.begin() + totalNumCandidates, d_candidateSequencesLength2.end(), stream);
        d_isPairedCandidate2.erase(d_isPairedCandidate2.begin() + totalNumCandidates, d_isPairedCandidate2.end(), stream);

        std::swap(d_candidateReadIds, d_candidateReadIds2);
        std::swap(d_candidateSequencesLength, d_candidateSequencesLength2);
        std::swap(d_isPairedCandidate, d_isPairedCandidate2);

        d_candidateSequencesLength2.destroy();
        d_candidateReadIds2.destroy();
        d_isPairedCandidate2.destroy();
        
        //update candidate sequences data
        CachedDeviceUVector<unsigned int> d_candidateSequencesData2(encodedSequencePitchInInts * totalNumCandidates, stream, *cubAllocator);

        cubSelectFlagged(
            d_candidateSequencesData.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_keepFlags, encodedSequencePitchInInts)
            ),
            d_candidateSequencesData2.data(),
            thrust::make_discard_iterator(),
            oldTotalNumCandidates * encodedSequencePitchInInts,
            stream
        );

        std::swap(d_candidateSequencesData, d_candidateSequencesData2);
        d_candidateSequencesData2.destroy();

        //update candidate quality scores
        // assert(qualityPitchInBytes % sizeof(int) == 0);
        // CachedDeviceUVector<char> d_candidateQualities2(qualityPitchInBytes * totalNumCandidates, stream, *cubAllocator);

        // cubSelectFlagged(
        //     (const int*)d_candidateQualityScores.data(),
        //     thrust::make_transform_iterator(
        //         thrust::make_counting_iterator(0),
        //         make_iterator_multiplier(d_keepFlags, qualityPitchInBytes / sizeof(int))
        //     ),
        //     (int*)d_candidateQualities2.data(),
        //     thrust::make_discard_iterator(),
        //     oldTotalNumCandidates * qualityPitchInBytes / sizeof(int),
        //     firstStream
        // );

        // std::swap(d_candidateQualityScores, d_candidateQualities2);

        setGpuSegmentIds(
            d_segmentIdsOfCandidates.data(),
            numTasks,
            totalNumCandidates,
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            stream
        );
    }


    void compactCandidateDataByFlags(
        const bool* d_keepFlags,
        cudaStream_t stream
    ){
        CachedDeviceUVector<int> d_numCandidatesPerAnchor2(numTasks, stream, *cubAllocator);

        cubSegmentedReduceSum(
            d_keepFlags,
            d_numCandidatesPerAnchor2.data(),
            numTasks,
            d_numCandidatesPerAnchorPrefixSum.data(),
            d_numCandidatesPerAnchorPrefixSum.data() + 1,
            stream
        );

        auto d_zip_data = thrust::make_zip_iterator(
            thrust::make_tuple(
                d_alignment_nOps.data(),
                d_alignment_overlaps.data(),
                d_alignment_shifts.data(),
                d_alignment_best_alignment_flags.data(),
                d_candidateReadIds.data(),
                d_candidateSequencesLength.data(),
                d_isPairedCandidate.data()
            )
        );

        CachedDeviceUVector<int> d_alignment_overlaps2(totalNumCandidates, stream, *cubAllocator);
        CachedDeviceUVector<int> d_alignment_shifts2(totalNumCandidates, stream, *cubAllocator);
        CachedDeviceUVector<int> d_alignment_nOps2(totalNumCandidates, stream, *cubAllocator);
        CachedDeviceUVector<BestAlignment_t> d_alignment_best_alignment_flags2(totalNumCandidates, stream, *cubAllocator);
        CachedDeviceUVector<int> d_candidateSequencesLength2(totalNumCandidates, stream, *cubAllocator);
        CachedDeviceUVector<read_number> d_candidateReadIds2(totalNumCandidates, stream, *cubAllocator);
        CachedDeviceUVector<bool> d_isPairedCandidate2(totalNumCandidates, stream, *cubAllocator);
  
        auto d_zip_data_tmp = thrust::make_zip_iterator(
            thrust::make_tuple(
                d_alignment_nOps2.data(),
                d_alignment_overlaps2.data(),
                d_alignment_shifts2.data(),
                d_alignment_best_alignment_flags2.data(),
                d_candidateReadIds2.data(),
                d_candidateSequencesLength2.data(),
                d_isPairedCandidate2.data()
            )
        );

        //compact 1d arrays

        cubSelectFlagged(
            d_zip_data, 
            d_keepFlags, 
            d_zip_data_tmp, 
            h_numCandidates.data(), 
            totalNumCandidates, 
            stream
        );

        cudaEventRecord(h_numCandidatesEvent, stream); CUERR;


        cudaMemsetAsync(d_numCandidatesPerAnchorPrefixSum.data(), 0, sizeof(int), stream); CUERR;
        cubInclusiveSum(
            d_numCandidatesPerAnchor2.data(), 
            d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            numTasks, 
            stream
        );
        std::swap(d_numCandidatesPerAnchor, d_numCandidatesPerAnchor2); 

        d_numCandidatesPerAnchor2.destroy();

        cudaEventSynchronize(h_numCandidatesEvent); CUERR; //wait for h_numCandidates
        const int oldTotalNumCandidates = totalNumCandidates;

        totalNumCandidates = *h_numCandidates; 

        d_alignment_nOps2.erase(d_alignment_nOps2.begin() + totalNumCandidates, d_alignment_nOps2.end(), stream);
        d_alignment_overlaps2.erase(d_alignment_overlaps2.begin() + totalNumCandidates, d_alignment_overlaps2.end(), stream);
        d_alignment_shifts2.erase(d_alignment_shifts2.begin() + totalNumCandidates, d_alignment_shifts2.end(), stream);
        d_alignment_best_alignment_flags2.erase(d_alignment_best_alignment_flags2.begin() + totalNumCandidates, d_alignment_best_alignment_flags2.end(), stream);
        d_candidateReadIds2.erase(d_candidateReadIds2.begin() + totalNumCandidates, d_candidateReadIds2.end(), stream);
        d_candidateSequencesLength2.erase(d_candidateSequencesLength2.begin() + totalNumCandidates, d_candidateSequencesLength2.end(), stream);
        d_isPairedCandidate2.erase(d_isPairedCandidate2.begin() + totalNumCandidates, d_isPairedCandidate2.end(), stream);

        std::swap(d_alignment_nOps, d_alignment_nOps2);
        std::swap(d_alignment_overlaps, d_alignment_overlaps2);
        std::swap(d_alignment_shifts, d_alignment_shifts2);
        std::swap(d_alignment_best_alignment_flags, d_alignment_best_alignment_flags2);
        std::swap(d_candidateReadIds, d_candidateReadIds2);
        std::swap(d_candidateSequencesLength, d_candidateSequencesLength2);
        std::swap(d_isPairedCandidate, d_isPairedCandidate2);

        d_alignment_overlaps2.destroy();
        d_alignment_shifts2.destroy();
        d_alignment_nOps2.destroy();
        d_alignment_best_alignment_flags2.destroy();
        d_candidateSequencesLength2.destroy();
        d_candidateReadIds2.destroy();
        d_isPairedCandidate2.destroy();
        
        //update candidate sequences data
        CachedDeviceUVector<unsigned int> d_candidateSequencesData2(encodedSequencePitchInInts * totalNumCandidates, stream, *cubAllocator);

        cubSelectFlagged(
            d_candidateSequencesData.data(),
            thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                make_iterator_multiplier(d_keepFlags, encodedSequencePitchInInts)
            ),
            d_candidateSequencesData2.data(),
            thrust::make_discard_iterator(),
            oldTotalNumCandidates * encodedSequencePitchInInts,
            stream
        );

        std::swap(d_candidateSequencesData, d_candidateSequencesData2);
        d_candidateSequencesData2.destroy();

        //update candidate quality scores
        // assert(qualityPitchInBytes % sizeof(int) == 0);
        // CachedDeviceUVector<char> d_candidateQualities2(qualityPitchInBytes * totalNumCandidates, stream, *cubAllocator);

        // cubSelectFlagged(
        //     (const int*)d_candidateQualityScores.data(),
        //     thrust::make_transform_iterator(
        //         thrust::make_counting_iterator(0),
        //         make_iterator_multiplier(d_keepFlags, qualityPitchInBytes / sizeof(int))
        //     ),
        //     (int*)d_candidateQualities2.data(),
        //     thrust::make_discard_iterator(),
        //     oldTotalNumCandidates * qualityPitchInBytes / sizeof(int),
        //     firstStream
        // );

        // std::swap(d_candidateQualityScores, d_candidateQualities2);

        setGpuSegmentIds(
            d_segmentIdsOfCandidates.data(),
            numTasks,
            totalNumCandidates,
            d_numCandidatesPerAnchor.data(),
            d_numCandidatesPerAnchorPrefixSum.data(),
            stream
        );
    }

    void setStateToFinished(){
        // for(auto&& task : tasks){
        //     addFinishedTask(std::move(task));
        // }
        addSortedFinishedTasks(tasks);
        tasks.clear();
        numTasks = 0;

        setState(GpuReadExtender::State::Finished);
    }
    
    void addFinishedTask(ExtensionTaskCpuData&& task){
        //finished tasks must be stored sorted by pairId. Tasks with same pairId are sorted by id
        auto comp = [](const auto& l, const auto& r){
            return std::tie(l.pairId, l.id) < std::tie(r.pairId, r.id);
        };
        auto where = std::upper_bound(finishedTasks.begin(), finishedTasks.end(), task, comp);
        finishedTasks.insert(where, std::move(task));
    }

    void addSortedFinishedTasks(std::vector<ExtensionTaskCpuData>& tasksToAdd){
        //finished tasks must be stored sorted by pairId. Tasks with same pairId are sorted by id

        auto comp = [](const auto& l, const auto& r){
            return std::tie(l.pairId, l.id) < std::tie(r.pairId, r.id);
        };

        std::vector<ExtensionTaskCpuData> newFinishedTasks(finishedTasks.size() + tasksToAdd.size());

        newFinishedTasks.erase(
            std::merge(
                std::make_move_iterator(tasksToAdd.begin()), 
                std::make_move_iterator(tasksToAdd.end()), 
                std::make_move_iterator(finishedTasks.begin()), 
                std::make_move_iterator(finishedTasks.end()), 
                newFinishedTasks.begin(),
                comp
            ),
            newFinishedTasks.end()
        );

        std::swap(newFinishedTasks, finishedTasks);
    }

    void handleEarlyExitOfTasks4(){

        constexpr bool disableOtherStrand = false;

        for(int i = 0; i < numTasks; i++){ 
            const auto& task = tasks[i];
            const int whichtype = task.id % 4;

            //whichtype 0: LR, strand1 searching mate to the right.
            //whichtype 1: LR, strand1 just extend to the right.
            //whichtype 2: RL, strand2 searching mate to the right.
            //whichtype 3: RL, strand2 just extend to the right.

            if(whichtype == 0){
                assert(task.direction == extension::ExtensionDirection::LR);
                assert(task.pairedEnd == true);

                if(task.mateHasBeenFound){
                    for(int k = 1; k <= 4; k++){
                        if(tasks[i + k].pairId == task.pairId){
                            if(tasks[i+k].id == task.id + 1){
                                //disable LR partner task
                                tasks[i + k].abortReason = extension::AbortReason::PairedAnchorFinished;
                            }else if(tasks[i+k].id == task.id + 2){
                                //disable RL search task
                                if(disableOtherStrand){
                                    tasks[i + k].abortReason = extension::AbortReason::OtherStrandFoundMate;
                                }
                            }
                        }else{
                            break;
                        }
                    }
                }else if(task.abortReason != extension::AbortReason::None){
                    for(int k = 1; k <= 4; k++){
                        if(tasks[i + k].pairId == task.pairId){
                            if(tasks[i+k].id == task.id + 1){
                                //disable LR partner task  
                                tasks[i + k].abortReason = extension::AbortReason::PairedAnchorFinished;
                                break;
                            }
                        }else{
                            break;
                        }
                    }
                }
            }else if(whichtype == 2){
                assert(task.direction == extension::ExtensionDirection::RL);
                assert(task.pairedEnd == true);

                if(task.mateHasBeenFound){
                    if(tasks[i + 1].pairId == task.pairId){
                        if(tasks[i + 1].id == task.id + 1){
                            //disable RL partner task
                            tasks[i + 1].abortReason = extension::AbortReason::PairedAnchorFinished;
                        }
                    }

                    for(int k = 1; k <= 2; k++){
                        if(tasks[i - k].pairId == task.pairId){
                            if(tasks[i - k].id == task.id - 2){
                                //disable LR search task
                                if(disableOtherStrand){
                                    tasks[i - k].abortReason = extension::AbortReason::OtherStrandFoundMate;
                                }
                            }
                        }else{
                            break;
                        }
                    }
                    
                }else if(task.abortReason != extension::AbortReason::None){
                    if(tasks[i + 1].pairId == task.pairId){
                        if(tasks[i + 1].id == task.id + 1){
                            //disable RL partner task
                            tasks[i + 1].abortReason = extension::AbortReason::PairedAnchorFinished;
                        }
                    }
                }
            }
        }
    }

    std::vector<extension::ExtendResult> makePairResultsFromFinishedTasks(
        int numPairs,
        int numData,
        const ExtensionTaskCpuData* finishedData,
        const int* dataRead2Begins,
        const int* dataExtendedReadLengths,
        const char* dataExtendedReadSequences,
        const char* dataExtendedReadQualities,
        std::size_t extendedReadSequencesPitch,
        std::size_t extendedReadQualitiesPitch
    ) const{
        auto idcomp = [](const auto& l, const auto& r){ return l.pairId < r.pairId;};
        //auto lengthcomp = [](const auto& l, const auto& r){ return l.extendedRead.length() < r.extendedRead.length();};

        const bool isSorted = std::is_sorted(
            finishedData, 
            finishedData + numData,
            idcomp
        );

        if(!isSorted){
            throw std::runtime_error("Error not sorted");
        }

        //merge extensions of the same pair and same strand
        auto merge = [&](int l, int r, extension::ExtendResult& result){
            assert(l+1 == r);
            assert(l % 2 == 0);

            const int lengthL = dataExtendedReadLengths[l];
            const int lengthR = dataExtendedReadLengths[r];
   
            auto overlapstart = dataRead2Begins[l];
            const int resultsize = overlapstart + lengthR;
            
            result.extendedRead.resize(resultsize);
            result.qualityScores.resize(resultsize);

            auto sIt = std::copy_n(
                dataExtendedReadSequences + l * extendedReadSequencesPitch, 
                lengthL,
                result.extendedRead.begin()
            );

            std::copy_n(
                dataExtendedReadSequences + r * extendedReadSequencesPitch + finishedData[r].myLength, 
                lengthR - finishedData[r].myLength, 
                sIt
            );

            auto qIt = std::copy_n(
                dataExtendedReadQualities + l * extendedReadQualitiesPitch, 
                lengthL,
                result.qualityScores.begin()
            );

            std::copy_n(
                dataExtendedReadQualities + r * extendedReadQualitiesPitch + finishedData[r].myLength, 
                lengthR - finishedData[r].myLength, 
                qIt
            );
        };

        std::vector<extension::ExtendResult> results(numPairs);

        for(int p = 0; p < numPairs; p++){
            const int i0 = 4 * p + 0;
            const int i1 = 4 * p + 1;
            const int i2 = 4 * p + 2;
            const int i3 = 4 * p + 3;

            const auto& d0 = finishedData[i0]; //LR search
            const auto& d1 = finishedData[i1]; //LR partner
            const auto& d2 = finishedData[i2]; //RL search
            const auto& d3 = finishedData[i3]; //RL partner

            auto LRmatefoundfunc = [&](){
                auto& myResult = results[p];
                merge(i0,i1,myResult);

                myResult.direction = d0.direction;
                myResult.numIterations = d0.iteration;
                myResult.aborted = d0.abortReason != extension::AbortReason::None;
                myResult.abortReason = d0.abortReason;
                myResult.readId1 = d0.myReadId;
                myResult.readId2 = d0.mateReadId;
                myResult.originalLength = d0.myLength;
                myResult.originalMateLength = d0.mateLength;
                myResult.read1begin = 0;
                myResult.goodscore = d0.goodscore;
                myResult.read2begin = dataRead2Begins[i0];
                myResult.mateHasBeenFound = d0.mateHasBeenFound;

                if(dataExtendedReadLengths[i3] > d3.myLength){
                    //insert extensions of reverse complement of d3 at beginning of d0
                    std::string r4revcNewPositions = SequenceHelpers::reverseComplementSequenceDecoded(
                        dataExtendedReadSequences + i3 * extendedReadSequencesPitch + d3.myLength, 
                        dataExtendedReadLengths[i3] - d3.myLength
                    );
                    std::string r4revNewQualities(
                        dataExtendedReadQualities + i3 * extendedReadQualitiesPitch + d3.myLength, 
                        dataExtendedReadLengths[i3] - d3.myLength
                    );
                    std::reverse(r4revNewQualities.begin(), r4revNewQualities.end());

                    myResult.extendedRead.insert(myResult.extendedRead.begin(), r4revcNewPositions.begin(), r4revcNewPositions.end());
                    myResult.qualityScores.insert(myResult.qualityScores.begin(), r4revNewQualities.begin(), r4revNewQualities.end());

                    myResult.read1begin += r4revcNewPositions.size();
                    myResult.read2begin += r4revcNewPositions.size();
                }

                myResult.mergedFromReadsWithoutMate = false;
            };

            auto RLmatefoundfunc = [&](){
                auto& myResult = results[p];
                merge(i2,i3,myResult);

                myResult.direction = d2.direction;
                myResult.numIterations = d2.iteration;
                myResult.aborted = d2.abortReason != extension::AbortReason::None;
                myResult.abortReason = d2.abortReason;
                myResult.readId1 = d2.myReadId;
                myResult.readId2 = d2.mateReadId;
                myResult.originalLength = d2.myLength;
                myResult.originalMateLength = d2.mateLength;
                myResult.read1begin = 0;
                myResult.goodscore = d2.goodscore;
                myResult.read2begin = dataRead2Begins[i2];
                myResult.mateHasBeenFound = d2.mateHasBeenFound;

                int extlength = myResult.extendedRead.size();

                SequenceHelpers::reverseComplementSequenceDecodedInplace(myResult.extendedRead.data(), extlength);
                std::reverse(myResult.qualityScores.begin(), myResult.qualityScores.end());

                const int sizeOfRightExtension = extlength - (myResult.read2begin + myResult.originalMateLength);

                int newread2begin = extlength - (myResult.read1begin + myResult.originalLength);
                int newread2length = myResult.originalLength;
                int newread1begin = sizeOfRightExtension;
                int newread1length = myResult.originalMateLength;

                assert(newread1begin >= 0);
                assert(newread2begin >= 0);
                assert(newread1begin + newread1length <= extlength);
                assert(newread2begin + newread2length <= extlength);

                myResult.read1begin = newread1begin;
                myResult.read2begin = newread2begin;
                myResult.originalLength = newread1length;
                myResult.originalMateLength = newread2length;

                if(dataExtendedReadLengths[i1] > d1.myLength){
                    //insert extensions of d1 at end of d2

                    myResult.extendedRead.insert(
                        myResult.extendedRead.end(), 
                        dataExtendedReadSequences + i1 * extendedReadSequencesPitch + d1.myLength, 
                        dataExtendedReadSequences + i1 * extendedReadSequencesPitch + dataExtendedReadLengths[i1]
                    );
                    myResult.qualityScores.insert(
                        myResult.qualityScores.end(), 
                        dataExtendedReadQualities + i1 * extendedReadQualitiesPitch + d1.myLength, 
                        dataExtendedReadQualities + i1 * extendedReadQualitiesPitch + dataExtendedReadLengths[i1]
                    );
                }
                
                myResult.mergedFromReadsWithoutMate = false;
            };
        
            if(d0.mateHasBeenFound && d2.mateHasBeenFound){
                if(d0.goodscore < d2.goodscore){
                    LRmatefoundfunc();
                }else{
                    RLmatefoundfunc();
                }
            }else 
            if(d0.mateHasBeenFound){
                LRmatefoundfunc();
            }else if(d2.mateHasBeenFound){
                RLmatefoundfunc();                
            }else{
                //try to find an overlap between d0 and d2 to create an extended read with proper length which reaches the mate
                auto& myResult = results[p];
                myResult.direction = d0.direction;
                myResult.numIterations = d0.iteration;
                myResult.aborted = d0.abortReason != extension::AbortReason::None;
                myResult.abortReason = d0.abortReason;
                myResult.readId1 = d0.myReadId;
                myResult.readId2 = d0.mateReadId;
                myResult.originalLength = d0.myLength;
                myResult.originalMateLength = d0.mateLength;
                myResult.read1begin = 0;
                myResult.goodscore = d0.goodscore;
                myResult.read2begin = dataRead2Begins[i0];
                myResult.mateHasBeenFound = d0.mateHasBeenFound;

                const int r1l = dataExtendedReadLengths[i0];
                const int r3l = dataExtendedReadLengths[i2];

                constexpr int minimumOverlap = 40;
                constexpr float maxRelativeErrorInOverlap = 0.05;

                bool didMergeDifferentStrands = false;
                if(r1l + r3l >= insertSize - insertSizeStddev + minimumOverlap){
                    std::string r3revc = SequenceHelpers::reverseComplementSequenceDecoded(
                        dataExtendedReadSequences + i2 * extendedReadSequencesPitch, 
                        dataExtendedReadLengths[i2]
                    );

                    MismatchRatioGlueDecider decider(minimumOverlap, maxRelativeErrorInOverlap);
                    //WeightedGapGluer gluer(d0.myLength);
                    QualityWeightedGapGluer gluer(d0.myLength, d2.myLength);

                    std::vector<std::pair<std::string, std::string>> possibleResults;

                    const int maxNumberOfPossibilities = 2*insertSizeStddev + 1;

                    for(int p = 0; p < maxNumberOfPossibilities; p++){
                        auto decision = decider(
                            std::string_view(dataExtendedReadSequences + i0 * extendedReadSequencesPitch, dataExtendedReadLengths[i0]), 
                            r3revc, 
                            insertSize - insertSizeStddev + p,
                            std::string_view(dataExtendedReadQualities + i0 * extendedReadQualitiesPitch, dataExtendedReadLengths[i0]),
                            std::string_view(dataExtendedReadQualities + i2 * extendedReadQualitiesPitch, dataExtendedReadLengths[i2])
                        );

                        if(decision.has_value()){
                            possibleResults.emplace_back(gluer(*decision));
                            break;
                        }
                    }

                    if(possibleResults.size() > 0){
                        didMergeDifferentStrands = true;

                        auto& mergeresult = possibleResults.front();

                        myResult.extendedRead = std::move(mergeresult.first);
                        myResult.qualityScores = std::move(mergeresult.second);
                        myResult.read2begin = myResult.extendedRead.size() - d2.myLength;
                        myResult.originalMateLength = d2.myLength;
                        myResult.mateHasBeenFound = true;
                        myResult.aborted = false;
                    }
                }

                if(!didMergeDifferentStrands){
                    //initialize result with d0
                    myResult.extendedRead.insert(
                        myResult.extendedRead.end(), 
                        dataExtendedReadSequences + i0 * extendedReadSequencesPitch,
                        dataExtendedReadSequences + i0 * extendedReadSequencesPitch + dataExtendedReadLengths[i0]
                    );
                    myResult.qualityScores.insert(
                        myResult.qualityScores.end(), 
                        dataExtendedReadQualities + i0 * extendedReadQualitiesPitch,
                        dataExtendedReadQualities + i0 * extendedReadQualitiesPitch + dataExtendedReadLengths[i0]
                    );
                }

                if(didMergeDifferentStrands && dataExtendedReadLengths[i1] > d1.myLength){
                    //insert extensions of d1 at end
                    myResult.extendedRead.insert(
                        myResult.extendedRead.end(),
                        dataExtendedReadSequences + i1 * extendedReadSequencesPitch + d1.myLength,
                        dataExtendedReadSequences + i1 * extendedReadSequencesPitch + dataExtendedReadLengths[i1]
                    );
                    myResult.qualityScores.insert(
                        myResult.qualityScores.end(),
                        dataExtendedReadQualities + i1 * extendedReadQualitiesPitch + d1.myLength,
                        dataExtendedReadQualities + i1 * extendedReadQualitiesPitch + dataExtendedReadLengths[i1]
                    );
                }

                if(dataExtendedReadLengths[i3] > d3.myLength){
                    //insert extensions of reverse complement of d3 at beginning

                    std::string r4revcNewPositions = SequenceHelpers::reverseComplementSequenceDecoded(
                        dataExtendedReadSequences + i3 * extendedReadSequencesPitch + d3.myLength,
                        dataExtendedReadLengths[i3] - d3.myLength
                    );
                    
                    assert(d3.myLength > 0);
                    std::string r4revNewQualities(
                        dataExtendedReadQualities + i3 * extendedReadQualitiesPitch + d3.myLength,
                        dataExtendedReadLengths[i3] - d3.myLength
                    );
                    std::reverse(r4revNewQualities.begin(), r4revNewQualities.end());

                    myResult.extendedRead.insert(myResult.extendedRead.begin(), r4revcNewPositions.begin(), r4revcNewPositions.end());
                    myResult.qualityScores.insert(myResult.qualityScores.begin(), r4revNewQualities.begin(), r4revNewQualities.end());

                    myResult.read1begin += r4revcNewPositions.size();
                    if(myResult.mateHasBeenFound){
                        myResult.read2begin += r4revcNewPositions.size();
                    }
                }

                myResult.mergedFromReadsWithoutMate = didMergeDifferentStrands;
            }
        }


        return results;
    }

    template<typename InputIteratorT , typename OutputIteratorT >
    void cubExclusiveSum(
        InputIteratorT d_in,
        OutputIteratorT d_out,
        int num_items,
        cudaStream_t stream = 0,
        bool debug_synchronous = false
    ) const {
        std::size_t bytes = 0;
        cudaError_t status = cudaSuccess;

        status = cub::DeviceScan::ExclusiveSum(
            nullptr,
            bytes,
            d_in, 
            d_out, 
            num_items, 
            stream,
            debug_synchronous
        );
        assert(status == cudaSuccess);

        CachedDeviceUVector<char> temp(bytes, stream, *cubAllocator);

        status = cub::DeviceScan::ExclusiveSum(
            temp.data(),
            bytes,
            d_in, 
            d_out, 
            num_items, 
            stream,
            debug_synchronous
        );
        assert(status == cudaSuccess);
    }

    template<typename InputIteratorT , typename OutputIteratorT >
    void cubInclusiveSum(
        InputIteratorT d_in,
        OutputIteratorT d_out,
        int num_items,
        cudaStream_t stream = 0,
        bool debug_synchronous = false
    ) const {
        std::size_t bytes = 0;
        cudaError_t status = cudaSuccess;

        status = cub::DeviceScan::InclusiveSum(
            nullptr,
            bytes,
            d_in, 
            d_out, 
            num_items, 
            stream,
            debug_synchronous
        );
        assert(status == cudaSuccess);

        CachedDeviceUVector<char> temp(bytes, stream, *cubAllocator);

        status = cub::DeviceScan::InclusiveSum(
            temp.data(),
            bytes,
            d_in, 
            d_out, 
            num_items, 
            stream,
            debug_synchronous
        );
        assert(status == cudaSuccess);
    }

    template<typename InputIteratorT , typename OutputIteratorT , typename ScanOpT >
    void cubInclusiveScan(
        InputIteratorT d_in,
        OutputIteratorT d_out,
        ScanOpT scan_op,
        int num_items,
        cudaStream_t stream = 0,
        bool debug_synchronous = false 
    ) const {
        std::size_t bytes = 0;
        cudaError_t status = cudaSuccess;

        status = cub::DeviceScan::InclusiveScan(
            nullptr,
            bytes,
            d_in, 
            d_out, 
            scan_op, 
            num_items, 
            stream,
            debug_synchronous
        );
        assert(status == cudaSuccess);

        CachedDeviceUVector<char> temp(bytes, stream, *cubAllocator);

        status = cub::DeviceScan::InclusiveScan(
            temp.data(),
            bytes,
            d_in, 
            d_out, 
            scan_op, 
            num_items, 
            stream,
            debug_synchronous
        );
        assert(status == cudaSuccess);
    }

    template<typename InputIteratorT , typename OutputIteratorT >
    void cubReduceSum(
        InputIteratorT d_in,
        OutputIteratorT d_out,
        int num_items,
        cudaStream_t stream = 0,
        bool debug_synchronous = false 
    ) const {
        std::size_t bytes = 0;
        cudaError_t status = cudaSuccess;

        status = cub::DeviceReduce::Sum(
            nullptr,
            bytes,
            d_in, 
            d_out, 
            num_items, 
            stream,
            debug_synchronous
        );
        assert(status == cudaSuccess);

        CachedDeviceUVector<char> temp(bytes, stream, *cubAllocator);

        status = cub::DeviceReduce::Sum(
            temp.data(),
            bytes,
            d_in, 
            d_out, 
            num_items, 
            stream,
            debug_synchronous
        );
        assert(status == cudaSuccess);
    }

    template<typename InputIteratorT , typename FlagIterator , typename OutputIteratorT , typename NumSelectedIteratorT >
    void cubSelectFlagged(
        InputIteratorT d_in,
        FlagIterator d_flags,
        OutputIteratorT d_out,
        NumSelectedIteratorT d_num_selected_out,
        int num_items,
        cudaStream_t stream = 0,
        bool debug_synchronous = false 
    ) const {
        std::size_t bytes = 0;
        cudaError_t status = cudaSuccess;

        status = cub::DeviceSelect::Flagged(
            nullptr, 
            bytes, 
            d_in, 
            d_flags, 
            d_out, 
            d_num_selected_out, 
            num_items, 
            stream,
            debug_synchronous
        );
        assert(status == cudaSuccess);

        CachedDeviceUVector<char> temp(bytes, stream, *cubAllocator);

        status = cub::DeviceSelect::Flagged(
            temp.data(), 
            bytes, 
            d_in, 
            d_flags, 
            d_out, 
            d_num_selected_out, 
            num_items, 
            stream,
            debug_synchronous
        );
        assert(status == cudaSuccess);
    }

    template<typename InputIteratorT , typename OutputIteratorT , typename OffsetIteratorT >
    void cubSegmentedReduceSum(
        InputIteratorT d_in,
        OutputIteratorT d_out,
        int num_segments,
        OffsetIteratorT	d_begin_offsets,
        OffsetIteratorT d_end_offsets,
        cudaStream_t stream = 0,
        bool debug_synchronous = false 
    ) const {
        std::size_t bytes = 0;
        cudaError_t status = cudaSuccess;

        status = cub::DeviceSegmentedReduce::Sum(
            nullptr, 
            bytes, 
            d_in, 
            d_out, 
            num_segments, 
            d_begin_offsets, 
            d_end_offsets,
            stream,
            debug_synchronous
        );
        assert(status == cudaSuccess);

        CachedDeviceUVector<char> temp(bytes, stream, *cubAllocator);

        status = cub::DeviceSegmentedReduce::Sum(
            temp.data(), 
            bytes, 
            d_in, 
            d_out, 
            num_segments, 
            d_begin_offsets, 
            d_end_offsets,
            stream,
            debug_synchronous
        );
        assert(status == cudaSuccess);
    }



    bool pairedEnd = false;
    State state = State::None;
    int numTasks = 0;
    int someId = 0;
    int alltimeMaximumNumberOfTasks = 0;
    std::size_t alltimetotalTaskBytes = 0;

    int totalNumCandidates = 0;

    int deviceId{};
    int insertSize{};
    int insertSizeStddev{};
    int maxextensionPerStep{1};
    int minCoverageForExtension{1};
    cub::CachingDeviceAllocator* cubAllocator{};
    const gpu::GpuReadStorage* gpuReadStorage{};
    const gpu::GpuMinhasher* gpuMinhasher{};
    mutable MinhasherHandle minhashHandle{};
    const CorrectionOptions* correctionOptions{};
    const GoodAlignmentProperties* goodAlignmentProperties{};
    const cpu::QualityScoreConversion* qualityConversion{};
    mutable ReadStorageHandle readStorageHandle{};
    mutable gpu::KernelLaunchHandle kernelLaunchHandle{};

    std::size_t encodedSequencePitchInInts = 0;
    std::size_t decodedSequencePitchInBytes = 0;
    std::size_t msaColumnPitchInElements = 0;
    std::size_t qualityPitchInBytes = 0;

    std::size_t outputAnchorPitchInBytes = 0;
    std::size_t outputAnchorQualityPitchInBytes = 0;
    std::size_t decodedMatesRevCPitchInBytes = 0;

    
    PinnedBuffer<read_number> h_candidateReadIds{};

    CachedDeviceUVector<bool> d_mateIdHasBeenRemoved{};

    PinnedBuffer<int> h_numCandidatesPerAnchor{};
    PinnedBuffer<int> h_numCandidatesPerAnchorPrefixSum{};



    PinnedBuffer<int> h_numAnchors{};
    PinnedBuffer<int> h_numCandidates{};
    DeviceBuffer<int> d_numAnchors{};
    DeviceBuffer<int> d_numCandidates{};
    DeviceBuffer<int> d_numCandidates2{};
    PinnedBuffer<int> h_numAnchorsWithRemovedMates{};

    // ----- candidate data
    CachedDeviceUVector<unsigned int> d_candidateSequencesData{};
    CachedDeviceUVector<int> d_candidateSequencesLength{};    
    CachedDeviceUVector<read_number> d_candidateReadIds{};
    CachedDeviceUVector<bool> d_isPairedCandidate{};
    CachedDeviceUVector<int> d_segmentIdsOfCandidates{};
    CachedDeviceUVector<int> d_alignment_overlaps{};
    CachedDeviceUVector<int> d_alignment_shifts{};
    CachedDeviceUVector<int> d_alignment_nOps{};
    CachedDeviceUVector<BestAlignment_t> d_alignment_best_alignment_flags{};

    CachedDeviceUVector<int> d_numCandidatesPerAnchor{};
    CachedDeviceUVector<int> d_numCandidatesPerAnchorPrefixSum{};
    // ----- 
    

    // ----- staging buffers for input
    PinnedBuffer<char> h_anchorQualityScores{};
    PinnedBuffer<char> h_mateQualityScoresReversed{};
    PinnedBuffer<char> h_subjectSequencesDataDecoded{};
    PinnedBuffer<read_number> h_anchorReadIds{};
    PinnedBuffer<read_number> h_mateReadIds{};
    PinnedBuffer<int> h_anchorSequencesLength{};
    PinnedBuffer<unsigned int> h_inputanchormatedata{};
    PinnedBuffer<int> h_inputMateLengths;
    PinnedBuffer<bool> h_isPairedTask;
    // ----- 

    // ----- input data

    CachedDeviceUVector<unsigned int> d_inputanchormatedata{};
    CachedDeviceUVector<char> d_subjectSequencesDataDecoded{};
    CachedDeviceUVector<char> d_anchorQualityScores{};
    CachedDeviceUVector<int> d_anchorSequencesLength{};
    CachedDeviceUVector<read_number> d_anchorReadIds{};
    CachedDeviceUVector<read_number> d_mateReadIds{};
    CachedDeviceUVector<int> d_inputMateLengths{};
    CachedDeviceUVector<bool> d_isPairedTask{};
    CachedDeviceUVector<unsigned int> d_subjectSequencesData{};
    CachedDeviceUVector<int> d_accumExtensionsLengths{};

    // -----

    // ----- tracking used ids
    CachedDeviceUVector<read_number> d_usedReadIds{};
    CachedDeviceUVector<int> d_numUsedReadIdsPerAnchor{};
    CachedDeviceUVector<int> d_numUsedReadIdsPerAnchorPrefixSum{};
    CachedDeviceUVector<int> d_segmentIdsOfUsedReadIds{};

    PinnedBuffer<int> h_numUsedReadIds{};

    CachedDeviceUVector<read_number> d_fullyUsedReadIds{};
    CachedDeviceUVector<int> d_numFullyUsedReadIdsPerAnchor{};
    CachedDeviceUVector<int> d_numFullyUsedReadIdsPerAnchorPrefixSum{};
    CachedDeviceUVector<int> d_segmentIdsOfFullyUsedReadIds{};

    PinnedBuffer<int> h_numFullyUsedReadIds{};
    PinnedBuffer<int> h_numFullyUsedReadIds2{};
    // -----
    
    // ----- MSA data
    gpu::ManagedGPUMultiMSA multiMSA;
    CachedDeviceUVector<char> d_consensusQuality{};
    // -----

    // ----- Extension output of a single iteration
    CachedDeviceUVector<char> d_outputAnchors;
    CachedDeviceUVector<char> d_outputAnchorQualities;
    CachedDeviceUVector<bool> d_outputMateHasBeenFound;
    CachedDeviceUVector<extension::AbortReason> d_abortReasons;
    CachedDeviceUVector<int> d_outputAnchorLengths{};
    CachedDeviceUVector<bool> d_isFullyUsedCandidate{};
    // -----

    PinnedBuffer<int> h_firstTasksOfPairsToCheck;
    PinnedBuffer<int> h_newPositionsOfActiveTasks{};

    PinnedBuffer<int> h_accumExtensionsLengths;
    PinnedBuffer<extension::AbortReason> h_abortReasons;
    PinnedBuffer<char> h_outputAnchors;
    PinnedBuffer<char> h_outputAnchorQualities;
    PinnedBuffer<int> h_outputAnchorLengths;
    PinnedBuffer<bool> h_outputMateHasBeenFound;
    PinnedBuffer<int> h_sizeOfGapToMate;
    PinnedBuffer<bool> h_isFullyUsedCandidate{};
    PinnedBuffer<float> h_goodscores{};

    // ----- Ready-events for pinned outputs
    CudaEvent h_numAnchorsEvent{};
    CudaEvent h_numCandidatesEvent{};
    CudaEvent h_numAnchorsWithRemovedMatesEvent{};
    CudaEvent h_numUsedReadIdsEvent{};
    CudaEvent h_numFullyUsedReadIdsEvent{};
    CudaEvent h_numFullyUsedReadIds2Event{};

    // -----



    CudaStream hostOutputStream{};
    
    std::array<CudaEvent, 1> events{};
    std::array<cudaStream_t, 4> streams{};
    std::vector<ExtensionTaskCpuData> tasks{};
    std::vector<ExtensionTaskCpuData> finishedTasks{};

};


}


#endif