//#define NDEBUG

#include <gpu/kernels.hpp>
#include <gpu/kernellaunch.hpp>
#include <hostdevicefunctions.cuh>

// //#include <gpu/bestalignment.hpp>
// #include <bestalignment.hpp>
#include <gpu/utility_kernels.cuh>

// #include <msa.hpp>
#include <sequence.hpp>
// #include <correctionresultprocessing.hpp>

// #include <shiftedhammingdistance_common.hpp>

// #include <hpc_helpers.cuh>
// #include <config.hpp>

// #include <cassert>


#include <cub/cub.cuh>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// #include <thrust/binary_search.h>


namespace care{
namespace gpu{




    template<int BLOCKSIZE>
    __global__
    void msa_correct_subject_kernel_old(
                char* __restrict__ correctedSubjects,
                bool* __restrict__ subjectIsCorrected,
                AnchorHighQualityFlag* __restrict__ isHighQualitySubject,
                int* __restrict__ uncorrected_positions_per_subject,
                int* __restrict__ num_uncorrected_positions_per_subject,
                const MSAColumnProperties* __restrict__ msaColumnProperties,
                const float* __restrict__ support,
                const int* __restrict__ coverage,
                const int* __restrict__ origCoverages,
                const char* __restrict__ consensus,
                const int* __restrict__ shifts,
                const BestAlignment_t* __restrict__ bestAlignmentFlags,
                const int* __restrict__ nOps,
                const int* __restrict__ overlaps,
                const unsigned int* __restrict__ subjectSequencesData,
                const unsigned int* __restrict__ candidateSequencesData,
                const int* __restrict__ candidateSequencesLength,
                const int* __restrict__ d_indices,
                const int* __restrict__ d_indices_per_subject,
                const int* __restrict__ d_candidates_per_subject_prefixsum,
                int n_subjects,
                int encodedSequencePitchInInts,
                size_t sequence_pitch,
                size_t msa_pitch,
                size_t msa_weights_pitch,
                int maximumSequenceLength,
                float estimatedErrorrate,
                float desiredAlignmentMaxErrorRate,
                float avg_support_threshold,
                float min_support_threshold,
                float min_coverage_threshold,
                float max_coverage_threshold,
                int k_region){

        using BlockReduceBool = cub::BlockReduce<bool, BLOCKSIZE>;
        using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;
        using BlockReduceFloat = cub::BlockReduce<float, BLOCKSIZE>;

        __shared__ union {
            typename BlockReduceBool::TempStorage boolreduce;
            typename BlockReduceInt::TempStorage intreduce;
            typename BlockReduceFloat::TempStorage floatreduce;
        } temp_storage;

        __shared__ bool broadcastbuffer;

        __shared__ int numUncorrectedPositions;
        __shared__ int uncorrectedPositions[BLOCKSIZE];
        __shared__ float avgCountPerWeight[4];

        auto get = [] (const char* data, int length, int index){
            //return Sequence_t::get_as_nucleotide(data, length, index);
            return getEncodedNuc2Bit((const unsigned int*)data, length, index, [](auto i){return i;});
        };

        auto getSubjectPtr = [&] (int subjectIndex){
            const unsigned int* result = subjectSequencesData + std::size_t(subjectIndex) * encodedSequencePitchInInts;
            return result;
        };

        auto getCandidatePtr = [&] (int candidateIndex){
            const unsigned int* result = candidateSequencesData + std::size_t(candidateIndex) * encodedSequencePitchInInts;
            return result;
        };

        auto getCandidateLength = [&](int candidateIndex){
            return candidateSequencesLength[candidateIndex];
        };

        auto isGoodAvgSupport = [&](float avgsupport){
            return fgeq(avgsupport, avg_support_threshold);
        };
        auto isGoodMinSupport = [&](float minsupport){
            return fgeq(minsupport, min_support_threshold);
        };
        auto isGoodMinCoverage = [&](float mincoverage){
            return fgeq(mincoverage, min_coverage_threshold);
        };

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

        auto saveUncorrectedPositionInSmem = [&](int pos){
            const int smemindex = atomicAdd(&numUncorrectedPositions, 1);
            uncorrectedPositions[smemindex] = pos;
        };

        const size_t msa_weights_pitch_floats = msa_weights_pitch / sizeof(float);

        for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
            const int myNumIndices = d_indices_per_subject[subjectIndex];
            if(myNumIndices > 0){

                const float* const my_support = support + msa_weights_pitch_floats * subjectIndex;
                const int* const my_coverage = coverage + msa_weights_pitch_floats * subjectIndex;
                const int* const my_orig_coverage = origCoverages + msa_weights_pitch_floats * subjectIndex;
                const char* const my_consensus = consensus + msa_pitch  * subjectIndex;
                char* const my_corrected_subject = correctedSubjects + subjectIndex * sequence_pitch;

                const int subjectColumnsBegin_incl = msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
                const int subjectColumnsEnd_excl = msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;
                const int lastColumn_excl = msaColumnProperties[subjectIndex].lastColumn_excl;

                float avg_support = 0;
                float min_support = 1.0f;
                //int max_coverage = 0;
                int min_coverage = std::numeric_limits<int>::max();

                for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                    assert(i < lastColumn_excl);

                    avg_support += my_support[i];
                    min_support = min(my_support[i], min_support);
                    //max_coverage = max(my_coverage[i], max_coverage);
                    min_coverage = min(my_coverage[i], min_coverage);
                }

                avg_support = BlockReduceFloat(temp_storage.floatreduce).Sum(avg_support);
                __syncthreads();

                min_support = BlockReduceFloat(temp_storage.floatreduce).Reduce(min_support, cub::Min());
                __syncthreads();

                //max_coverage = BlockReduceInt(temp_storage.intreduce).Reduce(max_coverage, cub::Max());

                min_coverage = BlockReduceInt(temp_storage.intreduce).Reduce(min_coverage, cub::Min());
                __syncthreads();

                avg_support /= (subjectColumnsEnd_excl - subjectColumnsBegin_incl);

                bool isHQ = isGoodAvgSupport(avg_support) && isGoodMinSupport(min_support) && isGoodMinCoverage(min_coverage);
                //bool isHQ = true;

                if(threadIdx.x == 0){
                    broadcastbuffer = isHQ;
                    isHighQualitySubject[subjectIndex].hq(isHQ);
                    //printf("%f %f %d %d\n", avg_support, min_support, min_coverage, isHQ);
                }
                __syncthreads();

                isHQ = broadcastbuffer;

                if(isHQ){
                    for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                        //assert(my_consensus[i] == 'A' || my_consensus[i] == 'C' || my_consensus[i] == 'G' || my_consensus[i] == 'T');
                        my_corrected_subject[i - subjectColumnsBegin_incl] = to_nuc(my_consensus[i]);
                    }
                    if(threadIdx.x == 0){
                        subjectIsCorrected[subjectIndex] = true;
                    }
                }else{

                    //decode orignal sequence and copy to corrected sequence
                    const int subjectLength = subjectColumnsEnd_excl - subjectColumnsBegin_incl;
                    const unsigned int* const subject = getSubjectPtr(subjectIndex);
                    for(int i = threadIdx.x; i < subjectLength; i += BLOCKSIZE){
                        my_corrected_subject[i] = to_nuc(get((const char*)subject, subjectLength, i));
                    }

                    bool foundAColumn = false;
                    int* globalUncorrectedPostitionsPtr = uncorrected_positions_per_subject + subjectIndex * maximumSequenceLength;
                    int* const globalNumUncorrectedPositionsPtr = num_uncorrected_positions_per_subject + subjectIndex;

                    //round up to next multiple of BLOCKSIZE;
                    const int loopIters = SDIV(subjectLength, BLOCKSIZE) * BLOCKSIZE;
                    for(int loopIter = 0; loopIter < loopIters; loopIter++){
                        if(threadIdx.x == 0){
                            numUncorrectedPositions = 0;
                        }
                        __syncthreads();

                        const int i = threadIdx.x + loopIter * BLOCKSIZE;

                        if(i < subjectLength){
                            const int globalIndex = subjectColumnsBegin_incl + i;

                            const int origCoverage = my_orig_coverage[globalIndex];
                            const char origBase = my_corrected_subject[i];
                            const char consensusBase = my_consensus[globalIndex];

                            float maxOverlapWeightOrigBase = 0.0f;
                            float maxOverlapWeightConsensusBase = 0.0f;
                            int origBaseCount = 1;
                            int consensusBaseCount = 0;

                            bool goodOrigOverlapExists = false;

                            const int globalOffset = d_candidates_per_subject_prefixsum[subjectIndex];

                            const int* myIndices = d_indices + globalOffset;

                            for(int candidatenr = 0; candidatenr < myNumIndices; candidatenr++){
                                const int arrayindex = myIndices[candidatenr] + globalOffset;

                                const unsigned int* candidateptr = getCandidatePtr(arrayindex);
                                const int candidateLength = getCandidateLength(arrayindex);
                                const int candidateShift = shifts[arrayindex];
                                const int candidateBasePosition = globalIndex - (subjectColumnsBegin_incl + candidateShift);
                                if(candidateBasePosition >= 0 && candidateBasePosition < candidateLength){
                                    char candidateBaseEnc = 0xFF;
                                    if(bestAlignmentFlags[arrayindex] == BestAlignment_t::ReverseComplement){
                                        candidateBaseEnc = get((const char*)candidateptr, candidateLength, candidateLength - candidateBasePosition-1);
                                        candidateBaseEnc = (~candidateBaseEnc) & 0x03;
                                    }else{
                                        candidateBaseEnc = get((const char*)candidateptr, candidateLength, candidateBasePosition);
                                    }
                                    const char candidateBase = candidateBaseEnc;

                                    const int mynOps = nOps[arrayindex];
                                    const int overlapsize = overlaps[arrayindex];
                                    const float overlapweight = calculateOverlapWeight(subjectLength, mynOps, overlapsize, desiredAlignmentMaxErrorRate);
                                    assert(overlapweight <= 1.0f);
                                    assert(overlapweight >= 0.0f);

                                    constexpr float goodOverlapThreshold = 0.90f;

                                    if(origBase == candidateBase){
                                        maxOverlapWeightOrigBase = max(maxOverlapWeightOrigBase, overlapweight);
                                        origBaseCount++;

                                        if(fgeq(overlapweight, goodOverlapThreshold)){
                                            goodOrigOverlapExists = true;
                                        }
                                    }else{
                                        if(consensusBase == candidateBase){
                                            maxOverlapWeightConsensusBase = max(maxOverlapWeightConsensusBase, overlapweight);
                                            consensusBaseCount++;
                                        }
                                    }
                                }
                            }

                            if(my_support[globalIndex] > 0.5f){

                                constexpr float maxOverlapWeightLowerBound = 0.15f;

                                bool allowCorrectionToConsensus = false;

                                //if(maxOverlapWeightOrigBase < maxOverlapWeightConsensusBase){
                                    allowCorrectionToConsensus = true;
                                //}

                                // if(maxOverlapWeightOrigBase == 0 && maxOverlapWeightConsensusBase == 0){
                                //     //correct to orig;
                                //     allowCorrectionToConsensus = false;
                                // }else if(maxOverlapWeightConsensusBase < maxOverlapWeightLowerBound){
                                //     //correct to orig
                                //     allowCorrectionToConsensus = false;
                                // }else if(maxOverlapWeightOrigBase < maxOverlapWeightLowerBound){
                                //     //correct to consensus
                                //     allowCorrectionToConsensus = true;
                                //     if(origBaseCount < 4){
                                //         allowCorrectionToConsensus = true;
                                //     }
                                // }else if(maxOverlapWeightConsensusBase < maxOverlapWeightOrigBase - 0.2f){
                                //     //maybe correct to orig
                                //     allowCorrectionToConsensus = false;
                                // }else if(maxOverlapWeightConsensusBase  - 0.2f > maxOverlapWeightOrigBase){
                                //     //maybe correct to consensus
                                //     if(origBaseCount < 4){
                                //         allowCorrectionToConsensus = true;
                                //     }
                                // }

                                if(!goodOrigOverlapExists && allowCorrectionToConsensus){

                                    float avgsupportkregion = 0;
                                    int c = 0;
                                    bool kregioncoverageisgood = true;


                                    for(int j = i - k_region/2; j <= i + k_region/2 && kregioncoverageisgood; j++){
                                        if(j != i && j >= 0 && j < subjectLength){
                                            avgsupportkregion += my_support[subjectColumnsBegin_incl + j];
                                            kregioncoverageisgood &= fgeq(my_coverage[subjectColumnsBegin_incl + j], min_coverage_threshold);
                                            //kregioncoverageisgood &= (my_coverage[subjectColumnsBegin_incl + j] >= 1);
                                            c++;
                                        }
                                    }
                                    avgsupportkregion /= c;

                                    if(kregioncoverageisgood && fgeq(avgsupportkregion, 1.0f-4*estimatedErrorrate / 2.0f)){


                                        // constexpr float maxOverlapWeightLowerBound = 0.25f;
                                        //
                                        // bool correctToConsensus = false;//maxOverlapWeightOrigBase < maxOverlapWeightLowerBound;
                                        // // correctToConsensus |= maxOverlapWeightConsensusBase >= maxOverlapWeightOrigBase;
                                        // // correctToConsensus &= !goodOrigOverlapExists;
                                        // if(!goodOrigOverlapExists && (origBase != consensusBase && my_support[globalIndex] > 0.5f)){
                                        //     correctToConsensus = true;
                                        // }

                                        // if(maxOverlapWeightOrigBase == 0 && maxOverlapWeightConsensusBase == 0){
                                        //     //correct to orig;
                                        // }else if(maxOverlapWeightConsensusBase < maxOverlapWeightLowerBound){
                                        //     //correct to orig
                                        // }else if(maxOverlapWeightOrigBase < maxOverlapWeightLowerBound){
                                        //     //correct to consensus
                                        //     my_corrected_subject[i] = consensusBase;
                                        // }else if(maxOverlapWeightConsensusBase < maxOverlapWeightOrigBase){
                                        //     //maybe correct to orig
                                        // }else if(maxOverlapWeightConsensusBase >= maxOverlapWeightOrigBase){
                                        //     //maybe correct to consensus
                                        //     my_corrected_subject[i] = consensusBase;
                                        // }

                                        //if(correctToConsensus){
                                            my_corrected_subject[i] = to_nuc(consensusBase);
                                            foundAColumn = true;
                                        // }else{
                                        //     saveUncorrectedPositionInSmem(i);
                                        // }
                                    }else{
                                        saveUncorrectedPositionInSmem(i);
                                    }
                                }
                            }else{
                                saveUncorrectedPositionInSmem(i);
                            }
                        }

                        __syncthreads();

                        if(threadIdx.x == 0){
                            *globalNumUncorrectedPositionsPtr += numUncorrectedPositions;
                        }

                        for(int k = threadIdx.x; k < numUncorrectedPositions; k += BLOCKSIZE){
                            globalUncorrectedPostitionsPtr[k] = uncorrectedPositions[k];
                        }
                        globalUncorrectedPostitionsPtr += numUncorrectedPositions;

                        if(loopIter < loopIters - 1){
                            __syncthreads();
                        }
                    }

                    //perform block wide or-reduction on foundAColumn
                    foundAColumn = BlockReduceBool(temp_storage.boolreduce).Reduce(foundAColumn, [](bool a, bool b){return a || b;});
                    __syncthreads();

                    if(threadIdx.x == 0){
                        subjectIsCorrected[subjectIndex] = true;//foundAColumn;
                    }
                }
            }
        }
    }





    template<int BLOCKSIZE>
    __global__
    void msaCorrectAnchorsKernel(
            char* __restrict__ correctedSubjects,
            bool* __restrict__ subjectIsCorrected,
            AnchorHighQualityFlag* __restrict__ isHighQualitySubject,
            const MSAColumnProperties* __restrict__ msaColumnProperties,
            const float* __restrict__ support,
            const int* __restrict__ coverage,
            const int* __restrict__ origCoverages,
            const char* __restrict__ consensus,
            const unsigned int* __restrict__ subjectSequencesData,
            const unsigned int* __restrict__ candidateSequencesData,
            const int* __restrict__ candidateSequencesLength,
            const int* __restrict__ d_indices_per_subject,
            const int* __restrict__ numAnchorsPtr,
            int encodedSequencePitchInInts,
            size_t sequence_pitch,
            size_t msaColumnPitchInElements,
            int maximumSequenceLength,
            float estimatedErrorrate,
            float desiredAlignmentMaxErrorRate,
            float avg_support_threshold,
            float min_support_threshold,
            float min_coverage_threshold,
            float max_coverage_threshold,
            int k_region){

        using BlockReduceBool = cub::BlockReduce<bool, BLOCKSIZE>;
        using BlockReduceInt = cub::BlockReduce<int, BLOCKSIZE>;
        using BlockReduceFloat = cub::BlockReduce<float, BLOCKSIZE>;

        __shared__ union {
            typename BlockReduceBool::TempStorage boolreduce;
            typename BlockReduceInt::TempStorage intreduce;
            typename BlockReduceFloat::TempStorage floatreduce;
        } temp_storage;

        __shared__ int broadcastbuffer;

        //__shared__ int numUncorrectedPositions;
        //__shared__ int uncorrectedPositions[BLOCKSIZE];
        //__shared__ float avgCountPerWeight[4];

        const int n_subjects = *numAnchorsPtr;

        auto get = [] (const char* data, int length, int index){
            //return Sequence_t::get_as_nucleotide(data, length, index);
            return getEncodedNuc2Bit((const unsigned int*)data, length, index, [](auto i){return i;});
        };

        auto isGoodAvgSupport = [&](float avgsupport){
            return fgeq(avgsupport, avg_support_threshold);
        };
        auto isGoodMinSupport = [&](float minsupport){
            return fgeq(minsupport, min_support_threshold);
        };
        auto isGoodMinCoverage = [&](float mincoverage){
            return fgeq(mincoverage, min_coverage_threshold);
        };

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

        for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
            const int myNumIndices = d_indices_per_subject[subjectIndex];
            if(myNumIndices > 0){

                const float* const my_support = support + msaColumnPitchInElements * subjectIndex;
                const int* const my_coverage = coverage + msaColumnPitchInElements * subjectIndex;
                const int* const my_orig_coverage = origCoverages + msaColumnPitchInElements * subjectIndex;
                const char* const my_consensus = consensus + msaColumnPitchInElements  * subjectIndex;
                char* const my_corrected_subject = correctedSubjects + subjectIndex * sequence_pitch;

                const int subjectColumnsBegin_incl = msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
                const int subjectColumnsEnd_excl = msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;
                const int lastColumn_excl = msaColumnProperties[subjectIndex].lastColumn_excl;

                float avg_support = 0;
                float min_support = 1.0f;
                //int max_coverage = 0;
                int min_coverage = std::numeric_limits<int>::max();

                for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                    assert(i < lastColumn_excl);

                    avg_support += my_support[i];
                    min_support = min(my_support[i], min_support);
                    //max_coverage = max(my_coverage[i], max_coverage);
                    min_coverage = min(my_coverage[i], min_coverage);
                }

                avg_support = BlockReduceFloat(temp_storage.floatreduce).Sum(avg_support);
                __syncthreads();

                min_support = BlockReduceFloat(temp_storage.floatreduce).Reduce(min_support, cub::Min());
                __syncthreads();

                //max_coverage = BlockReduceInt(temp_storage.intreduce).Reduce(max_coverage, cub::Max());

                min_coverage = BlockReduceInt(temp_storage.intreduce).Reduce(min_coverage, cub::Min());
                __syncthreads();

                avg_support /= (subjectColumnsEnd_excl - subjectColumnsBegin_incl);


                //const float avg_support_threshold = 1.0f-1.0f*estimatedErrorrate;
        		//const float min_support_threshold = 1.0f-3.0f*estimatedErrorrate;

                if(threadIdx.x == 0){
                    subjectIsCorrected[subjectIndex] = true; //canBeCorrected;

                    const bool canBeCorrectedByConsensus = isGoodAvgSupport(avg_support) && isGoodMinSupport(min_support) && isGoodMinCoverage(min_coverage);
                    int flag = 0;

                    if(canBeCorrectedByConsensus){
                        int smallestErrorrateThatWouldMakeHQ = 100;

                        const int estimatedErrorratePercent = ceil(estimatedErrorrate * 100.0f);
                        for(int percent = estimatedErrorratePercent; percent >= 0; percent--){
                            const float factor = percent / 100.0f;
                            const float avg_threshold = 1.0f - 1.0f * factor;
                            const float min_threshold = 1.0f - 3.0f * factor;
                            if(fgeq(avg_support, avg_threshold) && fgeq(min_support, min_threshold)){
                                smallestErrorrateThatWouldMakeHQ = percent;
                            }
                            // if(readId == 134){
                            //     printf("avg_support %f, avg_threshold %f, min_support %f, min_threshold %f\n", 
                            //     avg_support, avg_threshold, min_support, min_threshold);
                            // }
                        }

                        const bool isHQ = isGoodMinCoverage(min_coverage)
                                            && fleq(smallestErrorrateThatWouldMakeHQ, estimatedErrorratePercent * 0.5f);

                        //broadcastbuffer = isHQ;
                        isHighQualitySubject[subjectIndex].hq(isHQ);

                        flag = isHQ ? 2 : 1;

                        // if(readId == 134){
                        //     printf("read 134 isHQ %d, min_coverage %d, avg_support %f, min_support %f, smallestErrorrateThatWouldMakeHQ %d, min_coverage_threshold %f\n", 
                        //         isHQ, min_coverage, avg_support, min_support, smallestErrorrateThatWouldMakeHQ, min_coverage_threshold);
                        // }
                    }else{
                        isHighQualitySubject[subjectIndex].hq(false);
                    }

                    broadcastbuffer = flag;
                }
                __syncthreads();

                // for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                //     //assert(my_consensus[i] == 'A' || my_consensus[i] == 'C' || my_consensus[i] == 'G' || my_consensus[i] == 'T');
                //     if(my_support[i] > 0.90f && my_orig_coverage[i] <= 2){
                //         my_corrected_subject[i - subjectColumnsBegin_incl] = my_consensus[i];
                //     }else{
                //         const char* subject = getSubjectPtr(subjectIndex);
                //         const char encodedBase = get(subject, subjectColumnsEnd_excl- subjectColumnsBegin_incl, i - subjectColumnsBegin_incl);
                //         const char base = to_nuc(encodedBase);
                //         my_corrected_subject[i - subjectColumnsBegin_incl] = base;
                //     }
                // }

                const int flag = broadcastbuffer;

                if(flag > 0){
                    for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                        const char nuc = my_consensus[i];
                        //assert(nuc == 'A' || nuc == 'C' || nuc == 'G' || nuc == 'T');
                        assert(0 <= nuc && nuc < 4);

                        my_corrected_subject[i - subjectColumnsBegin_incl] = to_nuc(nuc);
                    }
                }else{
                    //correct only positions with high support.
                    for(int i = subjectColumnsBegin_incl + threadIdx.x; i < subjectColumnsEnd_excl; i += BLOCKSIZE){
                        //assert(my_consensus[i] == 'A' || my_consensus[i] == 'C' || my_consensus[i] == 'G' || my_consensus[i] == 'T');
                        if(my_support[i] > 0.90f && my_orig_coverage[i] <= 2){
                            my_corrected_subject[i - subjectColumnsBegin_incl] = to_nuc(my_consensus[i]);
                        }else{
                            const unsigned int* const subject = subjectSequencesData + std::size_t(subjectIndex) * encodedSequencePitchInInts;
                            const char encodedBase = get((const char*)subject, subjectColumnsEnd_excl- subjectColumnsBegin_incl, i - subjectColumnsBegin_incl);
                            const char base = to_nuc(encodedBase);
                            assert(base == 'A' || base == 'C' || base == 'G' || base == 'T');
                            my_corrected_subject[i - subjectColumnsBegin_incl] = base;
                        }
                    }
                }
            }else{
                if(threadIdx.x == 0){
                    isHighQualitySubject[subjectIndex].hq(false);
                    subjectIsCorrected[subjectIndex] = false;
                }
            }
        }
    }



    __device__ __forceinline__
    bool checkIfCandidateShouldBeCorrectedGlobal(
            const float* __restrict__ support,
            const int* __restrict__ coverages,
            const MSAColumnProperties* __restrict__ msaColumnProperties,
            const int* __restrict__ alignmentShifts,
            const int* __restrict__ candidateSequencesLengths,
            size_t msa_weights_pitch_floats,
            float min_support_threshold,
            float min_coverage_threshold,
            int new_columns_to_correct,
            int subjectIndex,
            int global_candidate_index){

        const float* const my_support = support + msa_weights_pitch_floats * subjectIndex;
        const int* const my_coverage = coverages + msa_weights_pitch_floats * subjectIndex;

        const int subjectColumnsBegin_incl = msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
        const int subjectColumnsEnd_excl = msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;
        const int lastColumn_excl = msaColumnProperties[subjectIndex].lastColumn_excl;

        const int shift = alignmentShifts[global_candidate_index];
        const int candidate_length = candidateSequencesLengths[global_candidate_index];
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
                    newColMinSupport = my_support[columnindex] < newColMinSupport ? my_support[columnindex] : newColMinSupport;
                    newColMinCov = my_coverage[columnindex] < newColMinCov ? my_coverage[columnindex] : newColMinCov;
                }
            }
            //check new columns right of subject
            for(int columnindex = subjectColumnsEnd_excl;
                    columnindex < subjectColumnsEnd_excl + new_columns_to_correct
                        && columnindex < lastColumn_excl;
                    columnindex++) {

                newColMinSupport = my_support[columnindex] < newColMinSupport ? my_support[columnindex] : newColMinSupport;
                newColMinCov = my_coverage[columnindex] < newColMinCov ? my_coverage[columnindex] : newColMinCov;
            }

            bool result = fgeq(newColMinSupport, min_support_threshold)
                            && fgeq(newColMinCov, min_coverage_threshold);

            //return result;
            return true;
        }else{
            return false;
        }

    }


    template<int blocksize, int tilesize>
    __global__
    void getNumCorrectedCandidatesPerAnchorKernel(
            int* __restrict__ d_numIndicesPerAnchor,
            const bool* __restrict__ d_isCorrectedCandidate,
            const int* __restrict__ d_numGoodIndicesPerSubject,
            const int* __restrict__ d_candidates_per_subject_prefixsum,
            const int* __restrict__ d_anchorIndicesOfCandidates,
            int numAnchors,
            int numCandidates){

        static_assert(blocksize % tilesize == 0);
        static_assert(tilesize == 32);

        constexpr int numTilesPerBlock = blocksize / tilesize;

        const int numTiles = (gridDim.x * blocksize) / tilesize;
        const int tileId = (threadIdx.x + blockIdx.x * blocksize) / tilesize;
        const int tileIdInBlock = threadIdx.x / tilesize;

        __shared__ int counts[numTilesPerBlock];


        auto tile = cg::tiled_partition<tilesize>(cg::this_thread_block());

        for(int anchorIndex = tileId; anchorIndex < numAnchors; anchorIndex += numTiles){

            const int offset = d_candidates_per_subject_prefixsum[anchorIndex];
            int* const numIndicesPtr = d_numIndicesPerAnchor + anchorIndex;

            const int numCandidatesForAnchor = d_numGoodIndicesPerSubject[anchorIndex];

            if(tile.thread_rank() == 0){
                counts[tileIdInBlock] = 0;
            }
            tile.sync();

            for(int localCandidateIndex = tile.thread_rank(); 
                    localCandidateIndex < numCandidatesForAnchor; 
                    localCandidateIndex += tile.size()){
                
                const int globalCandidateIndex = localCandidateIndex + offset;
                const bool isCorrected = d_isCorrectedCandidate[globalCandidateIndex];

                if(isCorrected){
                    cg::coalesced_group g = cg::coalesced_threads();
                    if (g.thread_rank() == 0) {
                        atomicAdd(&counts[tileIdInBlock], g.size());
                    }
                }
            }

            tile.sync();
            if(tile.thread_rank() == 0){
                atomicAdd(numIndicesPtr, counts[tileIdInBlock]);
            }

        }

    }

    template<int BLOCKSIZE>
    __global__
    void compactCandidateCorrectionResultsKernel(
            char* __restrict__ compactedCorrectedCandidates,
            TempCorrectedSequence::Edit* __restrict__ compactedEditsPerCorrectedCandidate,
            const int* __restrict__ numCorrectedCandidatesPerAnchor,
            const int* __restrict__ numCorrectedCandidatesPerAnchorPrefixsum, //exclusive
            const int* __restrict__ high_quality_subject_indices,
            const int* __restrict__ num_high_quality_subject_indices,
            const int* __restrict__ candidates_per_subject_prefixsum,
            const char* __restrict__ correctedCandidates,
            const int* __restrict__ correctedCandidateLengths,
            const TempCorrectedSequence::Edit* __restrict__ editsPerCorrectedCandidate,
            size_t decodedSequencePitch,
            int numEditsThreshold,
            int n_subjects){

        constexpr int groupsize = 32;
        static_assert(groupsize <= 32);
        static_assert(BLOCKSIZE % groupsize == 0);

        const int numHqSubjects = *num_high_quality_subject_indices;

        auto tgroup = cg::tiled_partition<groupsize>(cg::this_thread_block());

        const int numGroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        //const int groupIdInBlock = threadIdx.x / groupsize;

        for(int hqsubjectIndex = groupId;
                hqsubjectIndex < numHqSubjects;
                hqsubjectIndex += numGroups){

            const int subjectIndex = high_quality_subject_indices[hqsubjectIndex];
            const int myNumCorrectedCandidates = numCorrectedCandidatesPerAnchor[subjectIndex];

            if(myNumCorrectedCandidates > 0){
                const int inputbaseoffset = candidates_per_subject_prefixsum[subjectIndex];
                const int outputbaseoffset = numCorrectedCandidatesPerAnchorPrefixsum[subjectIndex];

                for(int cIndex = tgroup.thread_rank(); cIndex < myNumCorrectedCandidates; cIndex += tgroup.size()){

                    for(int i = 0; i < decodedSequencePitch; i++){
                        compactedCorrectedCandidates[(outputbaseoffset+cIndex) * decodedSequencePitch + i] 
                            = correctedCandidates[(inputbaseoffset+cIndex) * decodedSequencePitch + i];
                    }

                    for(int i = 0; i < numEditsThreshold; i++){
                        compactedEditsPerCorrectedCandidate[(outputbaseoffset+cIndex) * numEditsThreshold + i] 
                            = editsPerCorrectedCandidate[(inputbaseoffset+cIndex) * numEditsThreshold + i];
                    }
                }
            }
        }
    }


    __global__ 
    void flagCandidatesToBeCorrectedKernel(
            bool* __restrict__ candidateCanBeCorrected,
            int* __restrict__ numCorrectedCandidatesPerAnchor,
            const float* __restrict__ support,
            const int* __restrict__ coverages,
            const MSAColumnProperties* __restrict__ msaColumnProperties,
            const int* __restrict__ alignmentShifts,
            const int* __restrict__ candidateSequencesLengths,
            const int* __restrict__ anchorIndicesOfCandidates,
            const AnchorHighQualityFlag* __restrict__ hqflags,
            const int* __restrict__ numCandidatesPerSubjectPrefixsum,
            const int* __restrict__ localGoodCandidateIndices,
            const int* __restrict__ numLocalGoodCandidateIndicesPerSubject,
            const int* __restrict__ d_numAnchors,
            const int* __restrict__ d_numCandidates,
            size_t msa_weights_pitch_floats,
            float min_support_threshold,
            float min_coverage_threshold,
            int new_columns_to_correct){

        const int n_subjects = *d_numAnchors;

        for(int anchorIndex = blockIdx.x; 
                anchorIndex < n_subjects; 
                anchorIndex += gridDim.x){

            const bool isHighQualitySubject = hqflags[anchorIndex].hq();
            const int numGoodIndices = numLocalGoodCandidateIndicesPerSubject[anchorIndex];
            const int dataoffset = numCandidatesPerSubjectPrefixsum[anchorIndex];
            const int* myGoodIndices = localGoodCandidateIndices + dataoffset;

            if(isHighQualitySubject){

                for(int tid = threadIdx.x; tid < numGoodIndices; tid += blockDim.x){
                    const int localCandidateIndex = myGoodIndices[tid];
                    const int globalCandidateIndex = dataoffset + localCandidateIndex;

                    const bool canHandleCandidate = checkIfCandidateShouldBeCorrectedGlobal(
                        support,
                        coverages,
                        msaColumnProperties,
                        alignmentShifts,
                        candidateSequencesLengths,
                        msa_weights_pitch_floats,
                        min_support_threshold,
                        min_coverage_threshold,
                        new_columns_to_correct,
                        anchorIndex,
                        globalCandidateIndex
                    );

                    candidateCanBeCorrected[globalCandidateIndex] = canHandleCandidate;

                    if(canHandleCandidate){
                        atomicAdd(numCorrectedCandidatesPerAnchor + anchorIndex, 1);
                    }
                }
                
            }
        }
    }


    template<int BLOCKSIZE, int groupsize>
    __global__
    void msa_correct_candidates_with_group_kernel2(
            char* __restrict__ correctedCandidates,
            TempCorrectedSequence::Edit* __restrict__ d_editsPerCorrectedCandidate,
            int* __restrict__ d_numEditsPerCorrectedCandidate,
            const MSAColumnProperties* __restrict__ msaColumnProperties,
            const char* __restrict__ consensus,
            const float* __restrict__ support,
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
            size_t sequence_pitch,
            size_t msaColumnPitchInElements,
            size_t dynamicsmemPitchInInts){

        /*
            Use groupsize threads per candidate to perform correction
        */
        static_assert(BLOCKSIZE % groupsize == 0, "BLOCKSIZE % groupsize != 0");
        constexpr int groupsPerBlock = BLOCKSIZE / groupsize;

        auto decodedReverseComplementInplaceGroup = [](auto group, char* sequence, int sequencelength){
            //char4* c4sequence = (char4*)sequence;

            auto make_reverse_complement_nuc = [](char in){
                switch(in){
                    case 'A': return 'T';
                    case 'C': return 'G';
                    case 'G': return 'C';
                    case 'T': return 'A';
                    default :return 'F';
                }
            };

            for(int i = group.thread_rank(); i < sequencelength/2; i += group.size()){
                const std::uint8_t front = make_reverse_complement_nuc(sequence[i]);
                const std::uint8_t back = make_reverse_complement_nuc(sequence[sequencelength - 1 - i]);
                sequence[i] = back;
                sequence[sequencelength - 1 - i] = front;
            }

            if(sequencelength % 2 == 1 && group.thread_rank() == 0){
                const int middleindex = sequencelength/2;
                sequence[middleindex] = make_reverse_complement_nuc(sequence[middleindex]);
            }
        };

        auto getEncodedNucFromInt2Bit = [](unsigned int data, int pos){
            return ((data >> (30 - 2*pos)) & 0x00000003);
        };

        auto to_nuc = [](char c){
            constexpr char A_enc = 0x00;
            constexpr char C_enc = 0x01;
            constexpr char G_enc = 0x02;
            constexpr char T_enc = 0x03;

            switch(c){
            case A_enc: return 'A';
            case C_enc: return 'C';
            case G_enc: return 'G';
            case T_enc: return 'T';
            default: return 'F';
            }
        };

        __shared__ int shared_numEditsOfCandidate[groupsPerBlock];

        extern __shared__ int dynamicsmem[]; // for sequences



        auto tgroup = cg::tiled_partition<groupsize>(cg::this_thread_block());

        const int numGroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int groupIdInBlock = threadIdx.x / groupsize;

        char* const shared_correctedCandidate = (char*)(dynamicsmem + dynamicsmemPitchInInts * groupIdInBlock);
        // char4* const shared_consensusstage = (char4*)(dynamicsmem + dynamicsmemPitchInInts * numGroups
        //     + (msaColumnPitchInElements / 4) * groupIdInBlock);

        //char4 shared_consensusstage[256/4];
        //__shared__ char shared_consensusstage[256];

        const int loopEnd = *numCandidatesToBeCorrected;

        for(int id = groupId;
                id < loopEnd;
                id += numGroups){

            const int candidateIndex = candidateIndicesOfCandidatesToBeCorrected[id];
            const int subjectIndex = anchorIndicesOfCandidates[candidateIndex];
            const int destinationIndex = id;

            char* const my_corrected_candidate = correctedCandidates + destinationIndex * sequence_pitch;
            const int candidate_length = candidateSequencesLengths[candidateIndex];

            const int shift = shifts[candidateIndex];
            const int subjectColumnsBegin_incl = msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
            const int queryColumnsBegin_incl = subjectColumnsBegin_incl + shift;
            const int queryColumnsEnd_excl = subjectColumnsBegin_incl + shift + candidate_length;

            const BestAlignment_t bestAlignmentFlag = bestAlignmentFlags[candidateIndex];

            const char* const my_consensus = consensus + msaColumnPitchInElements * subjectIndex;

            if(tgroup.thread_rank() == 0){                        
                shared_numEditsOfCandidate[groupIdInBlock] = 0;
            }
            tgroup.sync();          

            
            

            const int copyposbegin = queryColumnsBegin_incl; //max(queryColumnsBegin_incl, subjectColumnsBegin_incl);
            const int copyposend = queryColumnsEnd_excl; //min(queryColumnsEnd_excl, subjectColumnsEnd_excl);
            assert(copyposend - copyposbegin == candidate_length);

            // const int consChar4Begin = copyposbegin / 4;
            // const int consChar4End = (copyposend-1) / 4;

            // for(int i = consChar4Begin + tgroup.thread_rank(); i <= consChar4End; i += tgroup.size()){
            //     char data[4];
            //     memcpy(data, my_consensus + 4 * i, sizeof(char4));

            //     #pragma unroll
            //     for(int k = 0; k < 4; k++){
            //         const int column = i * 4 + k;
            //         if(copyposbegin <= column && column < copyposend){
            //             shared_correctedCandidate[column - copyposbegin] = to_nuc(data[k]);
            //         }
            //     }
            // }

            for(int i = copyposbegin + tgroup.thread_rank(); i < copyposend; i += tgroup.size()) {
                shared_correctedCandidate[i - queryColumnsBegin_incl] = to_nuc(my_consensus[i]);
            }



            //const float* const my_support = support + msaColumnPitchInElements * subjectIndex;
            

            // for(int i = copyposbegin; i < copyposend; i += 1) {
            //     //assert(my_consensus[i] == 'A' || my_consensus[i] == 'C' || my_consensus[i] == 'G' || my_consensus[i] == 'T');
            //     if(my_support[i] > 0.90f){
            //         my_corrected_candidates[destinationindex * sequence_pitch + (i - queryColumnsBegin_incl)] = my_consensus[i];
            //     }else{
            //         const char encodedBase = get(candidate, queryColumnsEnd_excl- queryColumnsBegin_incl, i - queryColumnsBegin_incl);
            //         const char base = to_nuc(encodedBase);
            //         my_corrected_candidates[destinationindex * sequence_pitch + (i - queryColumnsBegin_incl)] = base;
            //     }
            // }

            

            //the forward strand will be returned -> make reverse complement again
            if(bestAlignmentFlag == BestAlignment_t::ReverseComplement) {
                tgroup.sync(); // threads may access elements in shared memory which were written by another thread
                decodedReverseComplementInplaceGroup(tgroup, shared_correctedCandidate, candidate_length);
                tgroup.sync();
            }

            // { 
            //     // leave positions untouched which did not overlap anchor

            //     auto get = [] (const unsigned int* data, int length, int index){
            //         return getEncodedNuc2Bit(data, length, index);
            //     };

            //     const unsigned int* const encUncorrectedCandidate = candidateSequencesData 
            //             + std::size_t(candidateIndex) * encodedSequencePitchInInts;

            //     for(int i = queryColumnsBegin_incl + tgroup.thread_rank(); i < subjectColumnsBegin_incl; i += tgroup.size()) {
            //         const char encodedBase = get(encUncorrectedCandidate, queryColumnsEnd_excl - queryColumnsBegin_incl, i - queryColumnsBegin_incl);
            //         const char base = to_nuc(encodedBase);
            //         shared_correctedCandidate[i - queryColumnsBegin_incl] = base;
            //     }

            //     const int subjectColumnsEnd_excl = msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;

            //     for(int i = subjectColumnsEnd_excl + tgroup.thread_rank(); i < queryColumnsEnd_excl; i += tgroup.size()) {
            //         const char encodedBase = get(encUncorrectedCandidate, queryColumnsEnd_excl - queryColumnsBegin_incl, i - queryColumnsBegin_incl);
            //         const char base = to_nuc(encodedBase);
            //         shared_correctedCandidate[i - queryColumnsBegin_incl] = base;
            //     }

            // }

            
            //copy from smem to global output
            const int fullInts1 = candidate_length / sizeof(int);

            for(int i = tgroup.thread_rank(); i < fullInts1; i += tgroup.size()) {
                ((int*)my_corrected_candidate)[i] = ((int*)shared_correctedCandidate)[i];
            }

            for(int i = tgroup.thread_rank(); i < candidate_length - fullInts1 * sizeof(int); i += tgroup.size()) {
                my_corrected_candidate[fullInts1 * sizeof(int) + i] 
                    = shared_correctedCandidate[fullInts1 * sizeof(int) + i];
            } 

            // for(int i = tgroup.thread_rank(); i < candidate_length; i += tgroup.size()) {
            //     my_corrected_candidate[i] = shared_correctedCandidate[i];
            // }            

            //compare corrected candidate with uncorrected candidate, calculate edits   
            
            const unsigned int* const encUncorrectedCandidate = candidateSequencesData 
                        + std::size_t(candidateIndex) * encodedSequencePitchInInts;
            const bool thisSequenceContainsN = d_candidateContainsN[candidateIndex];            
            int* const myNumEdits = d_numEditsPerCorrectedCandidate + candidateIndex;
            TempCorrectedSequence::Edit* const myEdits = d_editsPerCorrectedCandidate + destinationIndex * numEditsThreshold;

            if(thisSequenceContainsN){
                if(tgroup.thread_rank() == 0){
                    *myNumEdits = doNotUseEditsValue;
                }
            }else{
                const int maxEdits = min(candidate_length / 7, numEditsThreshold);

                const int fullInts = candidate_length / 16;
                

                for(int i = 0; i < fullInts; i++){
                    const unsigned int encodedDataInt = encUncorrectedCandidate[i];

                    //compare with 16 bases of corrected sequence

                    for(int k = tgroup.thread_rank(); k < 16; k += tgroup.size()){
                        const int posInInt = k;
                        const int posInSequence = i * 16 + posInInt;
                        const char encodedUncorrectedNuc = getEncodedNucFromInt2Bit(encodedDataInt, posInInt);
                        const char correctedNuc = shared_correctedCandidate[posInSequence];

                        if(correctedNuc != to_nuc(encodedUncorrectedNuc)){
                            cg::coalesced_group g = cg::coalesced_threads();

                            int currentNumEdits = 0;
                            if(g.thread_rank() == 0){
                                currentNumEdits = atomicAdd(&shared_numEditsOfCandidate[groupIdInBlock], g.size());
                            }
                            currentNumEdits = g.shfl(currentNumEdits, 0);

                            if(currentNumEdits + g.size() <= maxEdits){
                                const int myEditOutputPos = g.thread_rank() + currentNumEdits;
                                myEdits[myEditOutputPos] = TempCorrectedSequence::Edit{posInSequence, correctedNuc};
                            }
                        }
                    }

                    tgroup.sync();

                    if(shared_numEditsOfCandidate[groupIdInBlock] > maxEdits){
                        break;
                    }
                }

                //process remaining positions
                if(shared_numEditsOfCandidate[groupIdInBlock] <= maxEdits){
                    const int remainingPositions = candidate_length - 16 * fullInts;
                    if(remainingPositions > 0){
                        const unsigned int encodedDataInt = encUncorrectedCandidate[fullInts];
                        for(int posInInt = tgroup.thread_rank(); posInInt < remainingPositions; posInInt += tgroup.size()){
                            const int posInSequence = fullInts * 16 + posInInt;
                            const char encodedUncorrectedNuc = getEncodedNucFromInt2Bit(encodedDataInt, posInInt);
                            const char correctedNuc = shared_correctedCandidate[posInSequence];

                            if(correctedNuc != to_nuc(encodedUncorrectedNuc)){
                                cg::coalesced_group g = cg::coalesced_threads();
                                
                                int currentNumEdits = 0;
                                if(g.thread_rank() == 0){
                                    currentNumEdits = atomicAdd(&shared_numEditsOfCandidate[groupIdInBlock], g.size());
                                }
                                currentNumEdits = g.shfl(currentNumEdits, 0);

                                if(currentNumEdits + g.size() <= maxEdits){
                                    const int myEditOutputPos = g.thread_rank() + currentNumEdits;
                                    if(myEditOutputPos < maxEdits){
                                        myEdits[myEditOutputPos] = TempCorrectedSequence::Edit{posInSequence, correctedNuc};
                                    }
                                }
                            }
                        }
                    }
                }

                tgroup.sync();

                if(tgroup.thread_rank() == 0){                            
                    if(shared_numEditsOfCandidate[groupIdInBlock] <= maxEdits){
                        *myNumEdits = shared_numEditsOfCandidate[groupIdInBlock];
                    }else{
                        *myNumEdits = doNotUseEditsValue;
                    }
                }
            }
            

            tgroup.sync(); //sync before handling next candidate
            
            //printf("subjectIndex %d global_candidate_index %d\n", subjectIndex, global_candidate_index);


            
        }
    }







    template<int BLOCKSIZE, int groupsize>
    __global__
    void msa_correct_candidates_with_group_kernel2_new(
            char* __restrict__ correctedCandidates,
            TempCorrectedSequence::Edit* __restrict__ d_editsPerCorrectedCandidate,
            int* __restrict__ d_numEditsPerCorrectedCandidate,
            const MSAColumnProperties* __restrict__ msaColumnProperties,
            const char* __restrict__ consensus,
            const float* __restrict__ support,
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
            size_t sequence_pitch,
            size_t msaColumnPitchInElements,
            size_t dynamicsmemPitchInInts){

        /*
            Use groupsize threads per candidate to perform correction
        */
        static_assert(BLOCKSIZE % groupsize == 0, "BLOCKSIZE % groupsize != 0");
        constexpr int groupsPerBlock = BLOCKSIZE / groupsize;

        auto decodedReverseComplementInplaceGroup = [](auto group, char* sequence, int sequencelength){
            //char4* c4sequence = (char4*)sequence;

            auto make_reverse_complement_nuc = [](char in){
                return in;
                // switch(in){
                //     case 'A': return 'T';
                //     case 'C': return 'G';
                //     case 'G': return 'C';
                //     case 'T': return 'A';
                //     default :return 'F';
                // }
            };

            for(int i = group.thread_rank(); i < sequencelength/2; i += group.size()){
                const std::uint8_t front = make_reverse_complement_nuc(sequence[i]);
                const std::uint8_t back = make_reverse_complement_nuc(sequence[sequencelength - 1 - i]);
                sequence[i] = back;
                sequence[sequencelength - 1 - i] = front;
            }

            if(sequencelength % 2 == 1 && group.thread_rank() == 0){
                const int middleindex = sequencelength/2;
                sequence[middleindex] = make_reverse_complement_nuc(sequence[middleindex]);
            }
        };

        auto reverseWithGroup = [](auto& group, char* sequence, int sequenceLength){
            // assume 4 chars per int, 0 = A, 1 = C, 2 = G, 3 = T

            auto reverse = [](char4 data){
                char4 s;
                s.x = data.w;
                s.y = data.z;
                s.z = data.y;
                s.w = data.x;
                return s;
            };

            const int arrayLength = SDIV(sequenceLength, 4); // 4 bases per int
            const int unusedPositions = arrayLength * 4 - sequenceLength;
            char4* sequenceAsInts = (char4*)sequence;

            for(int i = group.thread_rank(); i < arrayLength/2; i += group.size()){
                const char4 fdata = ((char4*)sequence)[i];
                const char4 bdata = ((char4*)sequence)[arrayLength - 1 - i];

                const char4 front = reverse(fdata);
                const char4 back = reverse(bdata);
                sequenceAsInts[i] = back;
                sequenceAsInts[arrayLength - 1 - i] = front;
            }

            if(arrayLength % 2 == 1 && group.thread_rank() == 0){
                const int middleindex = arrayLength/2;
                const char4 mdata = ((char4*)sequence)[middleindex];
                sequenceAsInts[middleindex] = reverse(mdata);
            }

            unsigned int* sequenceAsInts2 = (unsigned int*)sequence;
    
            if(unusedPositions > 0){
                for(int i = 0; i < arrayLength-1; i++){
                    sequenceAsInts2[i] = (sequenceAsInts2[i] << (8*unusedPositions))
                                                   | (sequenceAsInts2[i+1] >> (8 * (4-unusedPositions)));
    
                }
                sequenceAsInts2[arrayLength-1] <<= (8*unusedPositions);
            }
        };

        auto getEncodedNucFromInt2Bit = [](unsigned int data, int pos){
            return ((data >> (30 - 2*pos)) & 0x00000003);
        };

        auto to_nuc = [](char c){
            // constexpr char A_enc = 0x00;
            // constexpr char C_enc = 0x01;
            // constexpr char G_enc = 0x02;
            // constexpr char T_enc = 0x03;

            // switch(c){
            // case A_enc: return 'A';
            // case C_enc: return 'C';
            // case G_enc: return 'G';
            // case T_enc: return 'T';
            // default: return 'F';
            // }

            return convertIntToDNACharNoIf(c);
        };

        __shared__ int shared_numEditsOfCandidate[groupsPerBlock];
        __shared__ TempCorrectedSequence::Edit shared_Edits[groupsPerBlock][15];

        extern __shared__ int dynamicsmem[]; // for sequences



        auto tgroup = cg::tiled_partition<groupsize>(cg::this_thread_block());

        const int numGroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int groupIdInBlock = threadIdx.x / groupsize;

        char* const shared_correctedCandidate = (char*)(dynamicsmem + dynamicsmemPitchInInts * groupIdInBlock);
        // char4* const shared_consensusstage = (char4*)(dynamicsmem + dynamicsmemPitchInInts * numGroups
        //     + (msaColumnPitchInElements / 4) * groupIdInBlock);

        //char4 shared_consensusstage[256/4];
        //__shared__ char shared_consensusstage[256];

        const int loopEnd = *numCandidatesToBeCorrected;

        for(int id = groupId;
                id < loopEnd;
                id += numGroups){

            const int candidateIndex = candidateIndicesOfCandidatesToBeCorrected[id];
            const int subjectIndex = anchorIndicesOfCandidates[candidateIndex];
            const int destinationIndex = id;

            char* const my_corrected_candidate = correctedCandidates + destinationIndex * sequence_pitch;
            const int candidate_length = candidateSequencesLengths[candidateIndex];

            const int shift = shifts[candidateIndex];
            const int subjectColumnsBegin_incl = msaColumnProperties[subjectIndex].subjectColumnsBegin_incl;
            const int queryColumnsBegin_incl = subjectColumnsBegin_incl + shift;
            const int queryColumnsEnd_excl = subjectColumnsBegin_incl + shift + candidate_length;

            const BestAlignment_t bestAlignmentFlag = bestAlignmentFlags[candidateIndex];

            const char* const my_consensus = consensus + msaColumnPitchInElements * subjectIndex;

            if(tgroup.thread_rank() == 0){                        
                shared_numEditsOfCandidate[groupIdInBlock] = 0;
            }
            tgroup.sync();          

            
            

            const int copyposbegin = queryColumnsBegin_incl; //max(queryColumnsBegin_incl, subjectColumnsBegin_incl);
            const int copyposend = queryColumnsEnd_excl; //min(queryColumnsEnd_excl, subjectColumnsEnd_excl);
            assert(copyposend - copyposbegin == candidate_length);

            // const int consChar4Begin = copyposbegin / 4;
            // const int consChar4End = (copyposend-1) / 4;

            // for(int i = consChar4Begin + tgroup.thread_rank(); i <= consChar4End; i += tgroup.size()){
            //     char data[4];
            //     memcpy(data, my_consensus + 4 * i, sizeof(char4));

            //     #pragma unroll
            //     for(int k = 0; k < 4; k++){
            //         const int column = i * 4 + k;
            //         if(copyposbegin <= column && column < copyposend){
            //             shared_correctedCandidate[column - copyposbegin] = to_nuc(data[k]);
            //         }
            //     }
            // }

            for(int i = copyposbegin + tgroup.thread_rank(); i < copyposend; i += tgroup.size()) {
                shared_correctedCandidate[i - queryColumnsBegin_incl] = to_nuc(my_consensus[i]);
            }



            //const float* const my_support = support + msaColumnPitchInElements * subjectIndex;
            

            // for(int i = copyposbegin; i < copyposend; i += 1) {
            //     //assert(my_consensus[i] == 'A' || my_consensus[i] == 'C' || my_consensus[i] == 'G' || my_consensus[i] == 'T');
            //     if(my_support[i] > 0.90f){
            //         my_corrected_candidates[destinationindex * sequence_pitch + (i - queryColumnsBegin_incl)] = my_consensus[i];
            //     }else{
            //         const char encodedBase = get(candidate, queryColumnsEnd_excl- queryColumnsBegin_incl, i - queryColumnsBegin_incl);
            //         const char base = to_nuc(encodedBase);
            //         my_corrected_candidates[destinationindex * sequence_pitch + (i - queryColumnsBegin_incl)] = base;
            //     }
            // }

            

            //the forward strand will be returned -> make reverse complement again
            if(bestAlignmentFlag == BestAlignment_t::ReverseComplement) {
                for(int i = copyposbegin + tgroup.thread_rank(); i < copyposend; i += tgroup.size()) {
                    shared_correctedCandidate[i - queryColumnsBegin_incl] = to_nuc(3-my_consensus[i]);
                }
                tgroup.sync(); // threads may access elements in shared memory which were written by another thread
                //decodedReverseComplementInplaceGroup(tgroup, shared_correctedCandidate, candidate_length);
                reverseWithGroup(tgroup, shared_correctedCandidate, candidate_length);
                tgroup.sync();
            }else{
                for(int i = copyposbegin + tgroup.thread_rank(); i < copyposend; i += tgroup.size()) {
                    shared_correctedCandidate[i - queryColumnsBegin_incl] = to_nuc(my_consensus[i]);
                }
            }

            // { 
            //     // leave positions untouched which did not overlap anchor

            //     auto get = [] (const unsigned int* data, int length, int index){
            //         return getEncodedNuc2Bit(data, length, index);
            //     };

            //     const unsigned int* const encUncorrectedCandidate = candidateSequencesData 
            //             + std::size_t(candidateIndex) * encodedSequencePitchInInts;

            //     for(int i = queryColumnsBegin_incl + tgroup.thread_rank(); i < subjectColumnsBegin_incl; i += tgroup.size()) {
            //         const char encodedBase = get(encUncorrectedCandidate, queryColumnsEnd_excl - queryColumnsBegin_incl, i - queryColumnsBegin_incl);
            //         const char base = to_nuc(encodedBase);
            //         shared_correctedCandidate[i - queryColumnsBegin_incl] = base;
            //     }

            //     const int subjectColumnsEnd_excl = msaColumnProperties[subjectIndex].subjectColumnsEnd_excl;

            //     for(int i = subjectColumnsEnd_excl + tgroup.thread_rank(); i < queryColumnsEnd_excl; i += tgroup.size()) {
            //         const char encodedBase = get(encUncorrectedCandidate, queryColumnsEnd_excl - queryColumnsBegin_incl, i - queryColumnsBegin_incl);
            //         const char base = to_nuc(encodedBase);
            //         shared_correctedCandidate[i - queryColumnsBegin_incl] = base;
            //     }

            // }

            
            //copy from smem to global output
            const int fullInts1 = candidate_length / sizeof(int);

            for(int i = tgroup.thread_rank(); i < fullInts1; i += tgroup.size()) {
                ((int*)my_corrected_candidate)[i] = ((int*)shared_correctedCandidate)[i];
            }

            for(int i = tgroup.thread_rank(); i < candidate_length - fullInts1 * sizeof(int); i += tgroup.size()) {
                my_corrected_candidate[fullInts1 * sizeof(int) + i] 
                    = shared_correctedCandidate[fullInts1 * sizeof(int) + i];
            } 

            // for(int i = tgroup.thread_rank(); i < candidate_length; i += tgroup.size()) {
            //     my_corrected_candidate[i] = shared_correctedCandidate[i];
            // }            

            //compare corrected candidate with uncorrected candidate, calculate edits   
            
            const unsigned int* const encUncorrectedCandidate = candidateSequencesData 
                        + std::size_t(candidateIndex) * encodedSequencePitchInInts;
            const bool thisSequenceContainsN = d_candidateContainsN[candidateIndex];            
            int* const myNumEdits = d_numEditsPerCorrectedCandidate + candidateIndex;
            TempCorrectedSequence::Edit* const myEdits = d_editsPerCorrectedCandidate + destinationIndex * numEditsThreshold;

            if(thisSequenceContainsN){
                if(tgroup.thread_rank() == 0){
                    *myNumEdits = doNotUseEditsValue;
                }
            }else{
                const int maxEdits = min(candidate_length / 7, numEditsThreshold);

                const int fullInts = candidate_length / 16;
                

                for(int i = 0; i < fullInts; i++){
                    const unsigned int encodedDataInt = encUncorrectedCandidate[i];

                    //compare with 16 bases of corrected sequence

                    for(int k = tgroup.thread_rank(); k < 16; k += tgroup.size()){
                        const int posInInt = k;
                        const int posInSequence = i * 16 + posInInt;
                        const char encodedUncorrectedNuc = getEncodedNucFromInt2Bit(encodedDataInt, posInInt);
                        const char correctedNuc = shared_correctedCandidate[posInSequence];

                        if(correctedNuc != to_nuc(encodedUncorrectedNuc)){
                            cg::coalesced_group g = cg::coalesced_threads();

                            int currentNumEdits = 0;
                            if(g.thread_rank() == 0){
                                currentNumEdits = atomicAdd(&shared_numEditsOfCandidate[groupIdInBlock], g.size());
                            }
                            currentNumEdits = g.shfl(currentNumEdits, 0);

                            if(currentNumEdits + g.size() <= maxEdits){
                                const int myEditOutputPos = g.thread_rank() + currentNumEdits;
                                const auto theEdit = TempCorrectedSequence::Edit{posInSequence, correctedNuc};
                                //myEdits[myEditOutputPos] = theEdit;
                                shared_Edits[groupIdInBlock][myEditOutputPos] = theEdit;
                            }
                        }
                    }

                    tgroup.sync();

                    if(shared_numEditsOfCandidate[groupIdInBlock] > maxEdits){
                        break;
                    }
                }

                //process remaining positions
                if(shared_numEditsOfCandidate[groupIdInBlock] <= maxEdits){
                    const int remainingPositions = candidate_length - 16 * fullInts;
                    if(remainingPositions > 0){
                        const unsigned int encodedDataInt = encUncorrectedCandidate[fullInts];
                        for(int posInInt = tgroup.thread_rank(); posInInt < remainingPositions; posInInt += tgroup.size()){
                            const int posInSequence = fullInts * 16 + posInInt;
                            const char encodedUncorrectedNuc = getEncodedNucFromInt2Bit(encodedDataInt, posInInt);
                            const char correctedNuc = shared_correctedCandidate[posInSequence];

                            if(correctedNuc != to_nuc(encodedUncorrectedNuc)){
                                cg::coalesced_group g = cg::coalesced_threads();
                                
                                int currentNumEdits = 0;
                                if(g.thread_rank() == 0){
                                    currentNumEdits = atomicAdd(&shared_numEditsOfCandidate[groupIdInBlock], g.size());
                                }
                                currentNumEdits = g.shfl(currentNumEdits, 0);

                                if(currentNumEdits + g.size() <= maxEdits){
                                    const int myEditOutputPos = g.thread_rank() + currentNumEdits;
                                    if(myEditOutputPos < maxEdits){
                                        const auto theEdit = TempCorrectedSequence::Edit{posInSequence, correctedNuc};
                                        //myEdits[myEditOutputPos] = theEdit;
                                        shared_Edits[groupIdInBlock][myEditOutputPos] = theEdit;
                                    }
                                }
                            }
                        }
                    }
                }

                tgroup.sync();

                if(shared_numEditsOfCandidate[groupIdInBlock] <= maxEdits){
                    if(tgroup.thread_rank() == 0){ 
                        *myNumEdits = shared_numEditsOfCandidate[groupIdInBlock];
                    }

                    for(int i = tgroup.thread_rank(); i < shared_numEditsOfCandidate[groupIdInBlock]; i += tgroup.size()){
                        myEdits[i] = shared_Edits[groupIdInBlock][i];
                    }
                }else{
                    if(tgroup.thread_rank() == 0){
                        *myNumEdits = doNotUseEditsValue;
                    }
                }

                // if(tgroup.thread_rank() == 0){                            
                //     if(shared_numEditsOfCandidate[groupIdInBlock] <= maxEdits){
                //         *myNumEdits = shared_numEditsOfCandidate[groupIdInBlock];
                //     }else{
                //         *myNumEdits = doNotUseEditsValue;
                //     }
                // }
            }
            

            tgroup.sync(); //sync before handling next candidate
            
            //printf("subjectIndex %d global_candidate_index %d\n", subjectIndex, global_candidate_index);


            
        }
    }





    __global__
    void constructAnchorResultsKernel(
            TempCorrectedSequence::Edit* __restrict__ d_editsPerCorrectedSubject,
            int* __restrict__ d_numEditsPerCorrectedSubject,
            int doNotUseEditsValue,
            const int* __restrict__ d_indicesOfCorrectedSubjects,
            const int* __restrict__ d_numIndicesOfCorrectedSubjects,
            const bool* __restrict__ d_readContainsN,
            const unsigned int* __restrict__ d_uncorrectedSubjects,
            const int* __restrict__ d_subjectLengths,
            const char* __restrict__ d_correctedSubjects,
            int numEditsThreshold,
            size_t encodedSequencePitchInInts,
            size_t decodedSequencePitchInBytes){

        auto get = [] (const unsigned int* data, int length, int index, auto trafo){
            return getEncodedNuc2Bit(data, length, index, trafo);
        };
        
        // auto getEncodedNucFromInt2Bit = [](unsigned int data, int pos){
        //     return ((data >> (30 - 2*pos)) & 0x00000003);
        // };

        auto to_nuc = [](char c){
            constexpr char A_enc = 0x00;
            constexpr char C_enc = 0x01;
            constexpr char G_enc = 0x02;
            constexpr char T_enc = 0x03;
            
            switch(c){
            case A_enc: return 'A';
            case C_enc: return 'C';
            case G_enc: return 'G';
            case T_enc: return 'T';
            default: return 'F';
            }
        };

        const int numIndicesToProcess = *d_numIndicesOfCorrectedSubjects;

        for(int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < numIndicesToProcess; tid += blockDim.x * gridDim.x){
            const int indexOfCorrectedSubject = d_indicesOfCorrectedSubjects[tid];

            const bool thisSequenceContainsN = d_readContainsN[indexOfCorrectedSubject];            
            int* const myNumEdits = d_numEditsPerCorrectedSubject + tid;

            if(thisSequenceContainsN){
                *myNumEdits = doNotUseEditsValue;
            }else{
                const int length = d_subjectLengths[indexOfCorrectedSubject];

                //find correct pointers
                const unsigned int* const encodedUncorrectedSequence = d_uncorrectedSubjects + encodedSequencePitchInInts * indexOfCorrectedSubject;
                const char* const decodedCorrectedSequence = d_correctedSubjects + decodedSequencePitchInBytes * indexOfCorrectedSubject;
    
                TempCorrectedSequence::Edit* const myEdits = d_editsPerCorrectedSubject + numEditsThreshold * tid;

                const int maxEdits = min(length / 7, numEditsThreshold);
                int edits = 0;
                
                for(int i = 0; i < length && edits <= maxEdits; i++){
                    const char correctedNuc = decodedCorrectedSequence[i];
                    const char uncorrectedNuc = to_nuc(get(encodedUncorrectedSequence, length, i, [](auto i){return i;}));

                    if(correctedNuc != uncorrectedNuc){
                        if(edits < maxEdits){
                            myEdits[edits] = TempCorrectedSequence::Edit{i, correctedNuc};
                        }
                        edits++;
                    }
                }
                if(edits <= maxEdits){
                    *myNumEdits = edits;
                }else{
                    *myNumEdits = doNotUseEditsValue;
                }
            }
        }
    }




    //####################   KERNEL DISPATCH   ####################

    

    void callGetNumCorrectedCandidatesPerAnchorKernel(
            int* d_numIndicesPerAnchor,
            const bool* d_isCorrectedCandidate,
            const int* d_numGoodIndicesPerSubject,
            const int* d_candidates_per_subject_prefixsum,
            const int* d_anchorIndicesOfCandidates,
            int numAnchors,
            int numCandidates,
            cudaStream_t stream,
            KernelLaunchHandle& handle){

        constexpr int blocksize = 128;
        constexpr int tilesize = 32;

        const std::size_t smem = 0;

        int max_blocks_per_device = 1;

        KernelLaunchConfig kernelLaunchConfig;
        kernelLaunchConfig.threads_per_block = blocksize;
        kernelLaunchConfig.smem = smem;

        auto iter = handle.kernelPropertiesMap.find(KernelId::GetNumCorrectedCandidatesPerAnchor);
        if(iter == handle.kernelPropertiesMap.end()){

            std::map<KernelLaunchConfig, KernelProperties> mymap;

            #define getProp(blocksize) { \
                KernelLaunchConfig kernelLaunchConfig; \
                kernelLaunchConfig.threads_per_block = (blocksize); \
                kernelLaunchConfig.smem = 0; \
                KernelProperties kernelProperties; \
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                    getNumCorrectedCandidatesPerAnchorKernel<(blocksize), tilesize>, \
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

            handle.kernelPropertiesMap[KernelId::GetNumCorrectedCandidatesPerAnchor] = std::move(mymap);

            #undef getProp
        }else{
            std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
            const KernelProperties& kernelProperties = map[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
        }

        cudaMemsetAsync(d_numIndicesPerAnchor, 0, numAnchors * sizeof(int), stream); CUERR;

        dim3 block(blocksize, 1, 1);
        dim3 grid(std::min(SDIV(numCandidates, blocksize), max_blocks_per_device));

        getNumCorrectedCandidatesPerAnchorKernel<blocksize, tilesize><<<grid, block, 0, stream>>>(
            d_numIndicesPerAnchor,
            d_isCorrectedCandidate,
            d_numGoodIndicesPerSubject,
            d_candidates_per_subject_prefixsum,
            d_anchorIndicesOfCandidates,
            numAnchors,
            numCandidates
        );
    }




    void call_msaCorrectAnchorsKernel_async(
                            char* d_correctedSubjects,
                            bool* d_subjectIsCorrected,
                            AnchorHighQualityFlag* d_isHighQualitySubject,
                            const MSAColumnProperties* d_msaColumnProperties,
                            const float* d_support,
                            const int* d_coverage,
                            const int* d_origCoverages,
                            const char* d_consensus,
                            const unsigned int* d_subjectSequencesData,
                            const unsigned int* d_candidateSequencesData,
                            const int* d_candidateSequencesLength,
                            const int* d_indices_per_subject,
                            const int* d_numAnchors,
                            int maxNumAnchors,
                            int encodedSequencePitchInInts,
                            size_t sequence_pitch,
                            size_t msaColumnPitchElements,
                            int maximumSequenceLength,
                            float estimatedErrorrate,
                            float desiredAlignmentMaxErrorRate,
                            float avg_support_threshold,
                            float min_support_threshold,
                            float min_coverage_threshold,
                            float max_coverage_threshold,
                            int k_region,
                            int maximum_sequence_length,
                            cudaStream_t stream,
                            KernelLaunchHandle& handle){

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
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                    msaCorrectAnchorsKernel<(blocksize)>, \
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

            handle.kernelPropertiesMap[KernelId::MSACorrectSubjectImplicit] = std::move(mymap);

            #undef getProp
        }else{
            std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
            const KernelProperties& kernelProperties = map[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
        }

        //cudaMemsetAsync(d_correctionResultPointers.isHighQualitySubject, 0, n_subjects * sizeof(AnchorHighQualityFlag), stream); CUERR;

        dim3 block(blocksize, 1, 1);
        //dim3 grid(std::min(maxNumAnchors, max_blocks_per_device));
        dim3 grid(max_blocks_per_device);

        #define mycall(blocksize) msaCorrectAnchorsKernel<(blocksize)> \
                                <<<grid, block, 0, stream>>>( \
                                    d_correctedSubjects, \
                                    d_subjectIsCorrected, \
                                    d_isHighQualitySubject, \
                                    d_msaColumnProperties, \
                                    d_support, \
                                    d_coverage, \
                                    d_origCoverages, \
                                    d_consensus, \
                                    d_subjectSequencesData, \
                                    d_candidateSequencesData, \
                                    d_candidateSequencesLength, \
                                    d_indices_per_subject, \
                                    d_numAnchors, \
                                    encodedSequencePitchInInts, \
                                    sequence_pitch, \
                                    msaColumnPitchElements, \
                                    maximumSequenceLength, \
                                    estimatedErrorrate, \
                                    desiredAlignmentMaxErrorRate, \
                                    avg_support_threshold, \
                                    min_support_threshold, \
                                    min_coverage_threshold, \
                                    max_coverage_threshold, \
                                    k_region \
                                ); CUERR;

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



    void callFlagCandidatesToBeCorrectedKernel_async(
            bool* d_candidateCanBeCorrected,
            int* d_numCorrectedCandidatesPerAnchor,
            const float* d_support,
            const int* d_coverages,
            const MSAColumnProperties* d_msaColumnProperties,
            const int* d_alignmentShifts,
            const int* d_candidateSequencesLengths,
            const int* d_anchorIndicesOfCandidates,
            const AnchorHighQualityFlag* d_hqflags,
            const int* d_candidatesPerSubjectPrefixsum,
            const int* d_localGoodCandidateIndices,
            const int* d_numLocalGoodCandidateIndicesPerSubject,
            const int* d_numAnchors,
            const int* d_numCandidates,
            size_t msaColumnPitchInElements,
            float min_support_threshold,
            float min_coverage_threshold,
            int new_columns_to_correct,
            cudaStream_t stream,
            KernelLaunchHandle& handle){

        // cudaMemsetAsync(
        //     d_numCorrectedCandidatesPerAnchor, 
        //     0, 
        //     sizeof(int) * n_subjects, 
        //     stream
        // ); CUERR;

        // cudaMemsetAsync(
        //     d_candidateCanBeCorrected, 
        //     0, 
        //     sizeof(bool) * n_candidates, 
        //     stream
        // ); CUERR;

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

            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &kernelProperties.max_blocks_per_SM,
                flagCandidatesToBeCorrectedKernel,
                kernelLaunchConfig.threads_per_block, 
                kernelLaunchConfig.smem
            ); CUERR;

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
            d_support,
            d_coverages,
            d_msaColumnProperties,
            d_alignmentShifts,
            d_candidateSequencesLengths,
            d_anchorIndicesOfCandidates,
            d_hqflags,
            d_candidatesPerSubjectPrefixsum,
            d_localGoodCandidateIndices,
            d_numLocalGoodCandidateIndicesPerSubject,
            d_numAnchors,
            d_numCandidates,
            msaColumnPitchInElements,
            min_support_threshold,
            min_coverage_threshold,
            new_columns_to_correct
        );

    }



    void callCorrectCandidatesWithGroupKernel2_async(
            char* __restrict__ correctedCandidates,
            TempCorrectedSequence::Edit* __restrict__ d_editsPerCorrectedCandidate,
            int* __restrict__ d_numEditsPerCorrectedCandidate,
            const MSAColumnProperties* __restrict__ msaColumnProperties,
            const char* __restrict__ consensus,
            const float* __restrict__ support,
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
            size_t sequence_pitch,
            size_t msaColumnPitchInElements,
            int maximum_sequence_length,
            cudaStream_t stream,
            KernelLaunchHandle& handle){


        constexpr int blocksize = 128;
        constexpr int groupsize = 32;
        constexpr int numGroupsPerBlock = blocksize / groupsize;

        const size_t dynamicsmemPitchInInts = SDIV(maximum_sequence_length, sizeof(int));
        const std::size_t smem = numGroupsPerBlock * (sizeof(int) * (dynamicsmemPitchInInts));

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
                kernelLaunchConfig.smem = numGroupsPerBlock * sizeof(char) * (SDIV(maximum_sequence_length, 4) * 4); \
                KernelProperties kernelProperties; \
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                    msa_correct_candidates_with_group_kernel2_new<(blocksize), groupsize>, \
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

    	#define mycall(blocksize) msa_correct_candidates_with_group_kernel2_new<(blocksize), groupsize> \
    	        <<<grid, block, smem, stream>>>( \
                    correctedCandidates, \
                    d_editsPerCorrectedCandidate, \
                    d_numEditsPerCorrectedCandidate, \
                    msaColumnProperties, \
                    consensus, \
                    support, \
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
                    sequence_pitch, \
                    msaColumnPitchInElements, \
                    dynamicsmemPitchInInts \
                ); CUERR;


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





    void callCompactCandidateCorrectionResultsKernel_async(
            char* __restrict__ d_compactedCorrectedCandidates, 
            TempCorrectedSequence::Edit* __restrict__ d_compactedEditsPerCorrectedCandidate,
            const int* __restrict__ d_numCorrectedCandidatesPerAnchor,
            const int* __restrict__ d_numCorrectedCandidatesPerAnchorPrefixsum, //exclusive
            const int* __restrict__ d_high_quality_subject_indices,
            const int* __restrict__ d_num_high_quality_subject_indices,
            const int* __restrict__ d_candidates_per_subject_prefixsum,
            const char* __restrict__ d_correctedCandidates,
            const int* __restrict__ d_correctedCandidateLengths,
            const TempCorrectedSequence::Edit* __restrict__ d_editsPerCorrectedCandidate,
            size_t decodedSequencePitch,
            int numEditsThreshold,
            int n_subjects,
            cudaStream_t stream,
            KernelLaunchHandle& /*handle*/){

        constexpr int blocksize = 256;

        dim3 block(blocksize);
        dim3 grid(SDIV(n_subjects, blocksize / 32));

        compactCandidateCorrectionResultsKernel<blocksize><<<grid, block, 0, stream>>>(
            d_compactedCorrectedCandidates,
            d_compactedEditsPerCorrectedCandidate,
            d_numCorrectedCandidatesPerAnchor,
            d_numCorrectedCandidatesPerAnchorPrefixsum,
            d_high_quality_subject_indices,
            d_num_high_quality_subject_indices,
            d_candidates_per_subject_prefixsum,
            d_correctedCandidates,
            d_correctedCandidateLengths,
            d_editsPerCorrectedCandidate,
            decodedSequencePitch,
            numEditsThreshold,
            n_subjects
        );
    }






    void callConstructAnchorResultsKernelAsync(
            TempCorrectedSequence::Edit* __restrict__ d_editsPerCorrectedSubject,
            int* __restrict__ d_numEditsPerCorrectedSubject,
            int doNotUseEditsValue,
            const int* __restrict__ d_indicesOfCorrectedSubjects,
            const int* __restrict__ d_numIndicesOfCorrectedSubjects,
            const bool* __restrict__ d_readContainsN,
            const unsigned int* __restrict__ d_uncorrectedSubjects,
            const int* __restrict__ d_subjectLengths,
            const char* __restrict__ d_correctedSubjects,
            int numEditsThreshold,
            size_t encodedSequencePitchInInts,
            size_t decodedSequencePitchInBytes,
            const int* d_numAnchors,
            int maxNumAnchors,
            cudaStream_t stream,
            KernelLaunchHandle& handle){

        generic_kernel<<<1,256,0,stream>>>(
            [=]__device__(){
                for(int i = threadIdx.x; i < *d_numAnchors; i += 256){
                    memset(&d_editsPerCorrectedSubject[i], 0, sizeof(TempCorrectedSequence::Edit));
                }
            }
        ); CUERR;
        //cudaMemsetAsync(d_editsPerCorrectedSubject, 0, sizeof(TempCorrectedSequence::Edit) * maxNumAnchors, stream);

        const int blocksize = 128;
        const std::size_t smem = 0;

        int max_blocks_per_device = 1;

        KernelLaunchConfig kernelLaunchConfig;
        kernelLaunchConfig.threads_per_block = blocksize;
        kernelLaunchConfig.smem = smem;

        auto iter = handle.kernelPropertiesMap.find(KernelId::ConstructAnchorResults);
        if(iter == handle.kernelPropertiesMap.end()){

            std::map<KernelLaunchConfig, KernelProperties> mymap;

            #define getProp(blocksize) { \
                KernelLaunchConfig kernelLaunchConfig; \
                kernelLaunchConfig.threads_per_block = (blocksize); \
                kernelLaunchConfig.smem = 0; \
                KernelProperties kernelProperties; \
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                    constructAnchorResultsKernel, \
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

            handle.kernelPropertiesMap[KernelId::ConstructAnchorResults] = std::move(mymap);

            #undef getProp
        }else{
            std::map<KernelLaunchConfig, KernelProperties>& map = iter->second;
            const KernelProperties& kernelProperties = map[kernelLaunchConfig];
            max_blocks_per_device = handle.deviceProperties.multiProcessorCount * kernelProperties.max_blocks_per_SM;
        }

        dim3 block(blocksize, 1, 1);
        dim3 grid(std::min(SDIV(maxNumAnchors, blocksize), max_blocks_per_device));

        #define mycall(blocksize) constructAnchorResultsKernel \
                                <<<grid, block, 0, stream>>>( \
                                        d_editsPerCorrectedSubject, \
                                        d_numEditsPerCorrectedSubject, \
                                        doNotUseEditsValue, \
                                        d_indicesOfCorrectedSubjects, \
                                        d_numIndicesOfCorrectedSubjects, \
                                        d_readContainsN, \
                                        d_uncorrectedSubjects, \
                                        d_subjectLengths, \
                                        d_correctedSubjects, \
                                        numEditsThreshold, \
                                        encodedSequencePitchInInts, \
                                        decodedSequencePitchInBytes); CUERR;

        mycall();

        // switch(blocksize){
        //     case 32: mycall(32); break;
        //     case 64: mycall(64); break;
        //     case 96: mycall(96); break;
        //     case 128: mycall(128); break;
        //     case 160: mycall(160); break;
        //     case 192: mycall(192); break;
        //     case 224: mycall(224); break;
        //     case 256: mycall(256); break;
        //     default: mycall(256); break;
        // }
         #undef mycall
    }











}
}
