//#define NDEBUG

#include <gpu/kernels.hpp>
#include <gpu/kernellaunch.hpp>
#include <hostdevicefunctions.cuh>
#include <gpu/gpumsa.cuh>

#include <bestalignment.hpp>

#include <sequencehelpers.hpp>
#include <correctionresultprocessing.hpp>

#include <hpc_helpers.cuh>
#include <config.hpp>
#include <cassert>


#include <cub/cub.cuh>

#include <cooperative_groups.h>

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
            const unsigned int* __restrict__ candidateSequencesData,
            const int* __restrict__ candidateSequencesLength,
            const int* __restrict__ d_indices_per_subject,
            const int* __restrict__ numAnchorsPtr,
            int encodedSequencePitchInInts,
            size_t decodedSequencePitchInBytes,
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

        auto tbGroup = cg::this_thread_block();

        const int n_subjects = *numAnchorsPtr;

        auto isGoodAvgSupport = [&](float avgsupport){
            return fgeq(avgsupport, avg_support_threshold);
        };
        auto isGoodMinSupport = [&](float minsupport){
            return fgeq(minsupport, min_support_threshold);
        };
        auto isGoodMinCoverage = [&](float mincoverage){
            return fgeq(mincoverage, min_coverage_threshold);
        };

        auto to_nuc = [](std::uint8_t c){
            return SequenceHelpers::decodeBase(c);
        };

        for(unsigned subjectIndex = blockIdx.x; subjectIndex < n_subjects; subjectIndex += gridDim.x){
            const int myNumIndices = d_indices_per_subject[subjectIndex];
            if(myNumIndices > 0){

                const GpuSingleMSA msa = multiMSA.getSingleMSA(subjectIndex);

                char* const my_corrected_subject = correctedSubjects + subjectIndex * decodedSequencePitchInBytes;

                const int subjectColumnsBegin_incl = msa.columnProperties->subjectColumnsBegin_incl;
                const int subjectColumnsEnd_excl = msa.columnProperties->subjectColumnsEnd_excl;
                const int lastColumn_excl = msa.columnProperties->lastColumn_excl;

                float avg_support = 0;
                float min_support = 1.0f;
                //int max_coverage = 0;
                int min_coverage = std::numeric_limits<int>::max();

                for(int i = subjectColumnsBegin_incl + tbGroup.thread_rank(); 
                        i < subjectColumnsEnd_excl; 
                        i += tbGroup.size()){

                    assert(i < lastColumn_excl);

                    avg_support += msa.support[i];
                    min_support = min(msa.support[i], min_support);
                    //max_coverage = max(my_coverage[i], max_coverage);
                    min_coverage = min(msa.coverages[i], min_coverage);
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

                if(tbGroup.thread_rank() == 0){
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

                tbGroup.sync();

                const int flag = broadcastbuffer;

                if(flag > 0){
                    for(int i = subjectColumnsBegin_incl + tbGroup.thread_rank(); 
                            i < subjectColumnsEnd_excl; 
                            i += tbGroup.size()){

                        const std::uint8_t nuc = msa.consensus[i];
                        //assert(nuc == 'A' || nuc == 'C' || nuc == 'G' || nuc == 'T');
                        assert(0 == nuc || nuc < 4);

                        my_corrected_subject[i - subjectColumnsBegin_incl] = to_nuc(nuc);
                    }
                }else{
                    //correct only positions with high support.
                    for(int i = subjectColumnsBegin_incl + tbGroup.thread_rank(); 
                            i < subjectColumnsEnd_excl; 
                            i += tbGroup.size()){

                        
                        if(msa.support[i] > 0.90f && msa.origCoverages[i] <= 2){
                            my_corrected_subject[i - subjectColumnsBegin_incl] = to_nuc(msa.consensus[i]);
                        }else{
                            const unsigned int* const subject = subjectSequencesData + std::size_t(subjectIndex) * encodedSequencePitchInInts;
                            const std::uint8_t encodedBase = SequenceHelpers::getEncodedNuc2Bit(subject, subjectColumnsEnd_excl- subjectColumnsBegin_incl, i - subjectColumnsBegin_incl);
                            const char base = to_nuc(encodedBase);
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



    __device__ __forceinline__
    bool checkIfCandidateShouldBeCorrectedGlobal(
        const GpuSingleMSA msa,
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
            int new_columns_to_correct){

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


    template<int BLOCKSIZE, int groupsize>
    __global__
    void msa_correct_candidates_with_group_kernel(
            char* __restrict__ correctedCandidates,
            TempCorrectedSequence::EncodedEdit* __restrict__ d_editsPerCorrectedCandidate,
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
            size_t dynamicsmemSequencePitchInInts){

        /*
            Use groupsize threads per candidate to perform correction
        */
        static_assert(BLOCKSIZE % groupsize == 0, "BLOCKSIZE % groupsize != 0");
        constexpr int groupsPerBlock = BLOCKSIZE / groupsize;

        auto reverseWithGroupShfl = [](auto& group, char* sequence, int sequenceLength){

            auto reverse = [](char4 data){
                char4 s;
                s.x = data.w;
                s.y = data.z;
                s.z = data.y;
                s.w = data.x;
                return s;
            };
        
            auto shiftLeft1 = [](char4 data){
                char4 s;
                s.x = data.y;
                s.y = data.z;
                s.z = data.w;
                s.w = '\0';
                return s;
            };
        
            auto shiftLeft2 = [](char4 data){
                char4 s;
                s.x = data.z;
                s.y = data.w;
                s.z = '\0';
                s.w = '\0';
                return s;
            };
        
            auto shiftLeft3 = [](char4 data){
                char4 s;
                s.x = data.w;
                s.y = '\0';
                s.z = '\0';
                s.w = '\0';
                return s;
            };
        
            //treat [left,right] as "char8", shift to the left by one char. return leftmost 4 chars
            auto handleUnusedPositions1 = [](char4 left, char4 right){
                char4 s;
                s.x = left.y;
                s.y = left.z;
                s.z = left.w;
                s.w = right.x;
                return s;
            };
        
            //treat [left,right] as "char8", shift to the left by two chars. return leftmost 4 chars
            auto handleUnusedPositions2 = [](char4 left, char4 right){
                char4 s;
                s.x = left.z;
                s.y = left.w;
                s.z = right.x;
                s.w = right.y;
                return s;
            };
        
            //treat [left,right] as "char8", shift to the left by three chars. return leftmost 4 chars
            auto handleUnusedPositions3 = [](char4 left, char4 right){
                char4 s;
                s.x = left.w;
                s.y = right.x;
                s.z = right.y;
                s.w = right.z;
                return s;
            };
        
            if(sequenceLength <= 1) return;
        
            const int arrayLength = SDIV(sequenceLength, 4); // 4 bases per int
            const int unusedPositions = arrayLength * 4 - sequenceLength;
            char4* sequenceAsChar4 = (char4*)sequence;
        
            for(int i = group.thread_rank(); i < arrayLength/2; i += group.size()){
                const char4 fdata = ((char4*)sequence)[i];
                const char4 bdata = ((char4*)sequence)[arrayLength - 1 - i];
        
                const char4 front = reverse(fdata);
                const char4 back = reverse(bdata);
                sequenceAsChar4[i] = back;
                sequenceAsChar4[arrayLength - 1 - i] = front;
            }
        
            if(arrayLength % 2 == 1 && group.thread_rank() == 0){
                const int middleindex = arrayLength/2;
                const char4 mdata = ((char4*)sequence)[middleindex];
                sequenceAsChar4[middleindex] = reverse(mdata);
            }
        
            group.sync();
        
            if(unusedPositions > 0){
        
                char4 left;
                char4 right;
                char4 tmp;
        
                const int numIterations = SDIV(arrayLength-1, group.size());
        
                for(int iteration = 0; iteration < numIterations; iteration++){
                    const int index = iteration * group.size() + group.thread_rank();
                    if(index < arrayLength){
                        left = sequenceAsChar4[index];
                    }
                    const int index2 = (iteration+1) * group.size() + group.thread_rank();
                    if(index2 < arrayLength && group.thread_rank() == 0){
                        tmp = sequenceAsChar4[index2];
                    }
                    #if __CUDACC_VER_MAJOR__ < 11
                    //CUDA < 11 does not have shuffle api for char4
                    *((int*)(&right)) = group.shfl_down(*((const int*)(&left)), 1);
                    *((int*)(&tmp)) = group.shfl(*((const int*)(&tmp)), 0);
                    #else
                    right = group.shfl_down(left, 1);
                    tmp = group.shfl(tmp, 0);
                    #endif
                    if(group.thread_rank() == group.size() - 1){
                        right = tmp;
                    }
        
                    if(unusedPositions == 1){
                        char4 result = handleUnusedPositions1(left, right);
                        if(index < arrayLength - 1){
                            sequenceAsChar4[index] = result;
                        }
                    }else if(unusedPositions == 2){
                        char4 result = handleUnusedPositions2(left, right);
                        if(index < arrayLength - 1){
                            sequenceAsChar4[index] = result;
                        }
                    }else{
                        char4 result = handleUnusedPositions3(left, right);
                        if(index < arrayLength - 1){
                            sequenceAsChar4[index] = result;
                        }
                    }
                }
        
                group.sync();
        
                if(group.thread_rank() == 0){
                    if(unusedPositions == 1){
                        sequenceAsChar4[arrayLength-1] = shiftLeft1(sequenceAsChar4[arrayLength-1]);
                    }else if(unusedPositions == 2){
                        sequenceAsChar4[arrayLength-1] = shiftLeft2(sequenceAsChar4[arrayLength-1]);
                    }else{
                        assert(unusedPositions == 3);
                        sequenceAsChar4[arrayLength-1] = shiftLeft3(sequenceAsChar4[arrayLength-1]);
                    }
                }
            }
        };

        auto to_nuc = [](std::uint8_t c){
            return SequenceHelpers::convertIntToDNACharNoIf(c);
        };

        __shared__ int shared_numEditsOfCandidate[groupsPerBlock];

        extern __shared__ int dynamicsmem[]; // for sequences

        auto tgroup = cg::tiled_partition<groupsize>(cg::this_thread_block());

        const int numGroups = (gridDim.x * blockDim.x) / groupsize;
        const int groupId = (threadIdx.x + blockIdx.x * blockDim.x) / groupsize;
        const int groupIdInBlock = threadIdx.x / groupsize;

        const std::size_t smemPitchEditsInInts = SDIV(editsPitchInBytes, sizeof(int));

        char* const shared_correctedCandidate = (char*)(dynamicsmem + dynamicsmemSequencePitchInInts * groupIdInBlock);



        TempCorrectedSequence::EncodedEdit* const shared_Edits 
            = (TempCorrectedSequence::EncodedEdit*)((dynamicsmem + dynamicsmemSequencePitchInInts * groupsPerBlock) 
                + smemPitchEditsInInts * groupIdInBlock);

        const int loopEnd = *numCandidatesToBeCorrected;

        for(int id = groupId;
                id < loopEnd;
                id += numGroups){

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
            assert(copyposend - copyposbegin == candidate_length);

            //the forward strand will be returned -> make reverse complement again
            if(bestAlignmentFlag == BestAlignment_t::ReverseComplement) {
                for(int i = copyposbegin + tgroup.thread_rank(); i < copyposend; i += tgroup.size()) {
                    shared_correctedCandidate[i - queryColumnsBegin_incl] = to_nuc(SequenceHelpers::complementBase2Bit(msa.consensus[i]));
                }
                tgroup.sync(); // threads may access elements in shared memory which were written by another thread
                reverseWithGroupShfl(tgroup, shared_correctedCandidate, candidate_length);
                tgroup.sync();
            }else{
                for(int i = copyposbegin + tgroup.thread_rank(); i < copyposend; i += tgroup.size()) {
                    shared_correctedCandidate[i - queryColumnsBegin_incl] = to_nuc(msa.consensus[i]);
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
                    d_numEditsPerCorrectedCandidate[candidateIndex] = doNotUseEditsValue;
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
                            const auto theEdit = TempCorrectedSequence::EncodedEdit{posInSequence, correctedNuc};
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
                                const auto theEdit = TempCorrectedSequence::EncodedEdit{posInSequence, correctedNuc};
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

                        if(correctedNuc != to_nuc(encodedUncorrectedNuc)){
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

                            if(correctedNuc != to_nuc(encodedUncorrectedNuc)){
                                countAndSaveEditInSmem2(posInSequence, correctedNuc);
                            }
                        }
                    }
                }

                tgroup.sync();

                //int* const myNumEdits = d_numEditsPerCorrectedCandidate + candidateIndex;
                int* const myNumEdits = d_numEditsPerCorrectedCandidate + destinationIndex;

                TempCorrectedSequence::EncodedEdit* const myEdits 
                    = (TempCorrectedSequence::EncodedEdit*)(((char*)d_editsPerCorrectedCandidate) + destinationIndex * editsPitchInBytes);

                if(shared_numEditsOfCandidate[groupIdInBlock] <= maxEdits){
                    const int numEdits = shared_numEditsOfCandidate[groupIdInBlock];

                    if(tgroup.thread_rank() == 0){ 
                        *myNumEdits = numEdits;
                    }

                    const int fullInts = (numEdits * sizeof(TempCorrectedSequence::EncodedEdit)) / sizeof(int);
                    static_assert(sizeof(TempCorrectedSequence::EncodedEdit) * 2 == sizeof(int), "");

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





    __global__
    void constructAnchorResultsKernel(
        TempCorrectedSequence::EncodedEdit* __restrict__ d_editsPerCorrectedSubject,
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
        size_t editsPitchInBytes
    ){

        auto to_nuc = [](std::uint8_t enc){
            return SequenceHelpers::decodeBase(enc);
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
    
                TempCorrectedSequence::EncodedEdit* const myEdits = (TempCorrectedSequence::EncodedEdit*)(((char*)d_editsPerCorrectedSubject) + editsPitchInBytes * tid);

                const int maxEdits = min(length / 7, numEditsThreshold);
                int edits = 0;
                
                for(int i = 0; i < length && edits <= maxEdits; i++){
                    const char correctedNuc = decodedCorrectedSequence[i];
                    const char uncorrectedNuc = to_nuc(SequenceHelpers::getEncodedNuc2Bit(encodedUncorrectedSequence, length, i));

                    if(correctedNuc != uncorrectedNuc){
                        if(edits < maxEdits){
                            myEdits[edits] = TempCorrectedSequence::EncodedEdit{i, correctedNuc};
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


    void call_msaCorrectAnchorsKernel_async(
        char* d_correctedSubjects,
        bool* d_subjectIsCorrected,
        AnchorHighQualityFlag* d_isHighQualitySubject,
        GPUMultiMSA multiMSA,
        const unsigned int* d_subjectSequencesData,
        const unsigned int* d_candidateSequencesData,
        const int* d_candidateSequencesLength,
        const int* d_indices_per_subject,
        const int* d_numAnchors,
        int maxNumAnchors,
        int encodedSequencePitchInInts,
        size_t decodedSequencePitchInBytes,
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
                                    multiMSA, \
                                    d_subjectSequencesData, \
                                    d_candidateSequencesData, \
                                    d_candidateSequencesLength, \
                                    d_indices_per_subject, \
                                    d_numAnchors, \
                                    encodedSequencePitchInInts, \
                                    decodedSequencePitchInBytes, \
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

        CUERR;

    }



    void callCorrectCandidatesKernel_async(
        char* __restrict__ correctedCandidates,
        TempCorrectedSequence::EncodedEdit* __restrict__ d_editsPerCorrectedCandidate,
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
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kernelProperties.max_blocks_per_SM, \
                    msa_correct_candidates_with_group_kernel<(blocksize), groupsize>, \
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
            
            // std::cerr << "msa_correct_candidates_with_group_kernel "
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

    	#define mycall(blocksize) msa_correct_candidates_with_group_kernel<(blocksize), groupsize> \
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




    void callConstructAnchorResultsKernelAsync(
        TempCorrectedSequence::EncodedEdit* __restrict__ d_editsPerCorrectedSubject,
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
        size_t editsPitchInBytes,
        const int* d_numAnchors,
        int maxNumAnchors,
        cudaStream_t stream,
        KernelLaunchHandle& handle
    ){

        cudaMemsetAsync(
            d_editsPerCorrectedSubject, 
            0, 
            editsPitchInBytes * maxNumAnchors, 
            stream
        ); CUERR;

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
                                        decodedSequencePitchInBytes, \
                                        editsPitchInBytes); CUERR;

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
