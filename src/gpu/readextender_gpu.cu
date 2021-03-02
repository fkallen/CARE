#include <gpu/readextender_gpu.hpp>
#include <readextenderbase.hpp>

#include <vector>
#include <algorithm>
#include <sequencehelpers.hpp>
#include <string>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <gpu/segmented_set_operations.cuh>
#include <gpu/cachingallocator.cuh>
#include <hostdevicefunctions.cuh>

//#define checkdebugtasks

namespace care{

    struct SequenceFlagMultiplier{
        int pitch{};
        const bool* flags{};

        __host__ __device__
        SequenceFlagMultiplier(const bool* flags_, int pitch_)
            :pitch(pitch_), flags(flags_){

        }

        __host__ __device__
        bool operator()(int i) const{
            return flags[i / pitch];
        }
    };

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













    std::vector<ReadExtenderBase::ExtendResult> ReadExtenderGpu::processPairedEndTasks(
        std::vector<ReadExtenderBase::Task>& tasks
    ) {
 
        std::vector<ExtendResult> extendResults;

        std::vector<int> indicesOfActiveTasks(tasks.size());
        std::vector<int> indicesOfActiveTasksTmp(tasks.size());
        std::iota(indicesOfActiveTasks.begin(), indicesOfActiveTasks.end(), 0);

        std::map<read_number, int> splitTracker; //counts number of tasks per read id, which can change by splitting a task
        for(const auto& t : tasks){
            splitTracker[t.myReadId] = 1;
        }

        //set input string as current anchor
        for(auto& task : tasks){
            std::string decodedAnchor(task.currentAnchorLength, '\0');

            SequenceHelpers::decode2BitSequence(
                &decodedAnchor[0],
                task.currentAnchor.data(),
                task.currentAnchorLength
            );

            task.totalDecodedAnchors.emplace_back(std::move(decodedAnchor));
            task.totalAnchorBeginInExtendedRead.emplace_back(0);
        }


#if 1
        //undo: replace vecAccess\(([a-zA-z]+), ([a-zA-z]+)\) by $1[$2]
        auto vecAccess = [](auto& vec, auto index) -> decltype(vec[index]){
            return vec[index];
        };
#else 
        auto vecAccess = [](auto& vec, auto index) -> decltype(vec.at(index)){
            return vec.at(index);
        };
#endif 

        int deviceId = 0;
        cudaGetDevice(&deviceId); CUERR;

        

        cudaStream_t firstStream = streams[0];
        cudaStream_t secondStream = streams[1];

        ThrustCachingAllocator<char> thrustCachingAllocator1(deviceId, cubAllocator, firstStream);
        auto thrustPolicy1 = thrust::cuda::par(thrustCachingAllocator1).on(firstStream);

        const int numTasks = tasks.size();

        batchData.h_numAnchors.resize(1);
        batchData.h_numCandidates.resize(1);
        batchData.d_numAnchors.resize(1);
        batchData.d_numCandidates.resize(1);

        auto processTasksOldStyle = [this, vecAccess, &indicesOfActiveTasks](auto& tasks){

            getCandidateReadIds(tasks, indicesOfActiveTasks);

            for(int indexOfActiveTask : indicesOfActiveTasks){
                auto& task = vecAccess(tasks, indexOfActiveTask);

                // remove self from candidate list
                auto readIdPos = std::lower_bound(
                    task.candidateReadIds.begin(),                                            
                    task.candidateReadIds.end(),
                    task.myReadId
                );

                if(readIdPos != task.candidateReadIds.end() && *readIdPos == task.myReadId){
                    task.candidateReadIds.erase(readIdPos);
                }

                if(task.pairedEnd){

                    //remove mate of input from candidate list
                    auto mateReadIdPos = std::lower_bound(
                        task.candidateReadIds.begin(),                                            
                        task.candidateReadIds.end(),
                        task.mateReadId
                    );

                    if(mateReadIdPos != task.candidateReadIds.end() && *mateReadIdPos == task.mateReadId){
                        task.candidateReadIds.erase(mateReadIdPos);
                        task.mateRemovedFromCandidates = true;
                    }
                }
            }

            for(int indexOfActiveTask : indicesOfActiveTasks){
                auto& task = vecAccess(tasks, indexOfActiveTask);

                std::vector<read_number> tmp(task.candidateReadIds.size());

                auto end = std::set_difference(
                    task.candidateReadIds.begin(),
                    task.candidateReadIds.end(),
                    task.allUsedCandidateReadIdPairs.begin(),
                    task.allUsedCandidateReadIdPairs.end(),
                    tmp.begin()
                );

                tmp.erase(end, tmp.end());

                std::swap(task.candidateReadIds, tmp);
            }

            loadCandidateSequenceData(tasks, indicesOfActiveTasks);

            eraseDataOfRemovedMates(tasks, indicesOfActiveTasks);

            calculateAlignments(tasks, indicesOfActiveTasks);

            #if 1

            for(int indexOfActiveTask : indicesOfActiveTasks){
                auto& task = vecAccess(tasks, indexOfActiveTask);

                /*
                    Remove bad alignments
                */        

                const int size = task.alignments.size();

                std::vector<int> positionsOfCandidatesToKeep(size);
                std::vector<int> tmpPositionsOfCandidatesToKeep(size);

                task.numRemainingCandidates = 0;

                //select candidates with good alignment and positive shift
                for(int c = 0; c < size; c++){
                    const BestAlignment_t alignmentFlag0 = vecAccess(task.alignmentFlags, c);
                    
                    if(alignmentFlag0 != BestAlignment_t::None && vecAccess(task.alignments, c).shift >= 0){
                        vecAccess(positionsOfCandidatesToKeep, task.numRemainingCandidates) = c;
                        task.numRemainingCandidates++;
                    }else{
                        ; // remove alignment
                    }
                }

                positionsOfCandidatesToKeep.erase(
                    positionsOfCandidatesToKeep.begin() + task.numRemainingCandidates, 
                    positionsOfCandidatesToKeep.end()
                );

                if(task.numRemainingCandidates == 0){
                    task.abort = true;
                    task.abortReason = AbortReason::NoPairedCandidatesAfterAlignment;

                    task.candidateReadIds.erase(task.candidateReadIds.begin(), task.candidateReadIds.end());
                    task.candidateSequenceLengths.erase(task.candidateSequenceLengths.begin(), task.candidateSequenceLengths.end());
                    task.candidateSequenceData.erase(task.candidateSequenceData.begin(), task.candidateSequenceData.end());
                    task.alignments.erase(task.alignments.begin(), task.alignments.end());
                    task.alignmentFlags.erase(task.alignmentFlags.begin(), task.alignmentFlags.end());

                    continue; //stop processing task
                }

                float relativeOverlapThreshold = 0.9f;
                bool goodAlignmentExists = false;

                while(!goodAlignmentExists && fgeq(relativeOverlapThreshold, goodAlignmentProperties.min_overlap_ratio)){                    

                    goodAlignmentExists = std::any_of(
                        positionsOfCandidatesToKeep.begin(), 
                        positionsOfCandidatesToKeep.end(),
                        [&](const auto& position){
                            const auto& alignment = vecAccess(task.alignments, position);
                            const float relativeOverlap = float(alignment.overlap) / float(task.currentAnchorLength);
                            return fgeq(relativeOverlap, relativeOverlapThreshold) && relativeOverlap < 1.0f;
                        }
                    );

                    if(!goodAlignmentExists){
                        relativeOverlapThreshold -= 0.1f;
                    }
                }

                // std::cerr << "task " << indexOfActiveTask
                //      << ", goodAlignmentExists " << goodAlignmentExists
                //      <<", relativeOverlapThreshold " << relativeOverlapThreshold;
                

                if(goodAlignmentExists){
                    positionsOfCandidatesToKeep.erase(
                        std::remove_if(
                            positionsOfCandidatesToKeep.begin(), 
                            positionsOfCandidatesToKeep.end(),
                            [&](const auto& position){
                                const auto& alignment = vecAccess(task.alignments, position);
                                const float relativeOverlap = float(alignment.overlap) / float(task.currentAnchorLength);
                                return !fgeq(relativeOverlap, relativeOverlapThreshold);
                            }
                        ),
                        positionsOfCandidatesToKeep.end()
                    );
                    task.numRemainingCandidates = positionsOfCandidatesToKeep.size();
                }

                // std::cerr << ", numRemainingCandidates = " << task.numRemainingCandidates << "\n";

                // std::cerr << "positionsOfCandidatesToKeep: ";

                // for(int x : positionsOfCandidatesToKeep){
                //     std::cerr << x << " ";
                // }
                // std::cerr << "\n";

                //std::cerr << ", remaining candidates " << task.numRemainingCandidates << "\n";


                //compact selected candidates inplace

                

                {
                    task.candidateSequenceData.resize(task.numRemainingCandidates * encodedSequencePitchInInts);

                    for(int c = 0; c < task.numRemainingCandidates; c++){
                        const int index = vecAccess(positionsOfCandidatesToKeep, c);

                        vecAccess(task.alignments, c) = vecAccess(task.alignments, index);
                        vecAccess(task.alignmentFlags, c) = vecAccess(task.alignmentFlags, index);
                        vecAccess(task.candidateReadIds, c) = vecAccess(task.candidateReadIds, index);
                        vecAccess(task.candidateSequenceLengths, c) = vecAccess(task.candidateSequenceLengths, index);
                        
                        assert(vecAccess(task.alignmentFlags, index) != BestAlignment_t::None);

                        // std::cerr << "cand " << index << " dir " 
                        //     << ((vecAccess(task.alignmentFlags, index) == BestAlignment_t::Forward) ? 'f' : 'r') << "\n";

                        if(vecAccess(task.alignmentFlags, index) == BestAlignment_t::Forward){
                            std::copy_n(
                                task.candidateSequencesFwdData.data() + index * encodedSequencePitchInInts,
                                encodedSequencePitchInInts,
                                task.candidateSequenceData.data() + c * encodedSequencePitchInInts
                            );
                        }else{
                            //BestAlignment_t::ReverseComplement

                            std::copy_n(
                                task.candidateSequencesRevcData.data() + index * encodedSequencePitchInInts,
                                encodedSequencePitchInInts,
                                task.candidateSequenceData.data() + c * encodedSequencePitchInInts
                            );
                        }

                        // //not sure if these 2 arrays will be required further on
                        // std::copy_n(
                        //     candidateSequencesFwdData.data() + index * encodedSequencePitchInInts,
                        //     encodedSequencePitchInInts,
                        //     candidateSequencesFwdData.data() + c * encodedSequencePitchInInts
                        // );

                        // std::copy_n(
                        //     candidateSequencesRevcData.data() + index * encodedSequencePitchInInts,
                        //     encodedSequencePitchInInts,
                        //     candidateSequencesRevcData.data() + c * encodedSequencePitchInInts
                        // );
                        
                    }

                    //erase past-end elements
                    task.alignments.erase(
                        task.alignments.begin() + task.numRemainingCandidates, 
                        task.alignments.end()
                    );
                    task.alignmentFlags.erase(
                        task.alignmentFlags.begin() + task.numRemainingCandidates, 
                        task.alignmentFlags.end()
                    );
                    task.candidateReadIds.erase(
                        task.candidateReadIds.begin() + task.numRemainingCandidates, 
                        task.candidateReadIds.end()
                    );
                    task.candidateSequenceLengths.erase(
                        task.candidateSequenceLengths.begin() + task.numRemainingCandidates, 
                        task.candidateSequenceLengths.end()
                    );
                    // //not sure if these 2 arrays will be required further on
                    // candidateSequencesFwdData.erase(
                    //     candidateSequencesFwdData.begin() + task.numRemainingCandidates * encodedSequencePitchInInts, 
                    //     candidateSequencesFwdData.end()
                    // );
                    // candidateSequencesRevcData.erase(
                    //     candidateSequencesRevcData.begin() + task.numRemainingCandidates * encodedSequencePitchInInts, 
                    //     candidateSequencesRevcData.end()
                    // );
                    
                }

            }
            #endif
        };
        

        while(indicesOfActiveTasks.size() > 0){

            #ifdef checkdebugtasks
            auto debugtasks = tasks;
            processTasksOldStyle(debugtasks);
            #endif

            //perform one extension iteration for active tasks

            //setup batchdata for active tasks
            const int numActiveTasks = indicesOfActiveTasks.size();
            batchData.numTasks = numActiveTasks;

            batchData.h_numAnchors.resize(1);
            batchData.d_numAnchors.resize(1);
            batchData.h_numCandidates.resize(1);
            batchData.d_numCandidates.resize(1);
            batchData.d_numCandidates2.resize(1);

            batchData.h_anchorReadIds.resize(numActiveTasks);
            batchData.d_anchorReadIds.resize(numActiveTasks);
            batchData.h_mateReadIds.resize(numActiveTasks);
            batchData.d_mateReadIds.resize(numActiveTasks);
            
            batchData.h_subjectSequencesData.resize(encodedSequencePitchInInts * numActiveTasks);
            batchData.d_subjectSequencesData.resize(encodedSequencePitchInInts * numActiveTasks);
            batchData.h_anchorSequencesLength.resize(numActiveTasks);
            batchData.d_anchorSequencesLength.resize(numActiveTasks);

            batchData.h_anchormatedata.resize(numActiveTasks * encodedSequencePitchInInts);
            batchData.d_anchormatedata.resize(numActiveTasks * encodedSequencePitchInInts);

            batchData.h_inputanchormatedata.resize(numActiveTasks * encodedSequencePitchInInts);
            batchData.d_inputanchormatedata.resize(numActiveTasks * encodedSequencePitchInInts);

            batchData.h_numCandidatesPerAnchor.resize(numActiveTasks);
            batchData.d_numCandidatesPerAnchor.resize(numActiveTasks);
            batchData.h_numCandidatesPerAnchor2.resize(numActiveTasks);
            batchData.d_numCandidatesPerAnchor2.resize(numActiveTasks);
            batchData.h_numCandidatesPerAnchor3.resize(numActiveTasks);
            batchData.d_numCandidatesPerAnchor3.resize(numActiveTasks);
            batchData.h_numCandidatesPerAnchorPrefixSum.resize(numActiveTasks+1);
            batchData.h_numCandidatesPerAnchorPrefixSum2.resize(numActiveTasks+1);
            batchData.h_numCandidatesPerAnchorPrefixSum3.resize(numActiveTasks+1);
            batchData.d_numCandidatesPerAnchorPrefixSum.resize(numActiveTasks+1);
            batchData.d_numCandidatesPerAnchorPrefixSum2.resize(numActiveTasks+1);
            batchData.d_numCandidatesPerAnchorPrefixSum3.resize(numActiveTasks+1);

            batchData.h_numCandidatesPerAnchorPrefixSum[0] = 0;
            batchData.h_numCandidatesPerAnchorPrefixSum2[0] = 0;
            batchData.h_numCandidatesPerAnchorPrefixSum3[0] = 0;

            batchData.h_indexlist1.resize(numActiveTasks);
            batchData.d_indexlist1.resize(numActiveTasks);

            batchData.h_indexlist2.resize(numActiveTasks);
            batchData.d_indexlist2.resize(numActiveTasks);

            batchData.totalNumberOfUsedIds = 0;

            for(int t = 0; t < numActiveTasks; t++){
                auto& task = tasks[indicesOfActiveTasks[t]];
                task.dataIsAvailable = false;

                batchData.h_anchorReadIds[t] = task.myReadId;
                batchData.h_mateReadIds[t] = task.mateReadId;
                batchData.totalNumberOfUsedIds += task.allUsedCandidateReadIdPairs.size();

                std::copy(
                    task.encodedMate.begin(),
                    task.encodedMate.end(),
                    batchData.h_inputanchormatedata.begin() + t * encodedSequencePitchInInts
                );
    
                if(task.iteration >= 0){
    
                    batchData.h_anchorSequencesLength[t] = task.currentAnchorLength;
    
                    std::copy(
                        task.currentAnchor.begin(),
                        task.currentAnchor.end(),
                        batchData.h_subjectSequencesData.begin() + t * encodedSequencePitchInInts
                    );
                }else{
                    //only hash kmers which include extended positions
    
                    const int extendedPositionsPreviousIteration 
                        = task.totalAnchorBeginInExtendedRead.at(task.iteration) - task.totalAnchorBeginInExtendedRead.at(task.iteration-1);
    
                    const int lengthToHash = std::min(task.currentAnchorLength, kmerLength + extendedPositionsPreviousIteration - 1);
                    batchData.h_anchorSequencesLength[t] = lengthToHash;
    
                    //std::cerr << "lengthToHash = " << lengthToHash << "\n";
    
                    std::vector<char> buf(task.currentAnchorLength);
                    SequenceHelpers::decode2BitSequence(buf.data(), task.currentAnchor.data(), task.currentAnchorLength);
                    SequenceHelpers::encodeSequence2Bit(
                        batchData.h_subjectSequencesData.get() + t * encodedSequencePitchInInts, 
                        buf.data() + task.currentAnchorLength - lengthToHash, 
                        lengthToHash
                    );    

                    if(task.myReadId == 33){
                        std::cerr << "Anchor for iteration " << task.iteration << "\n";
                        assert(lengthToHash == task.currentAnchorLength);
                        for(auto c : buf){
                            std::cerr << c;
                        }
                        std::cerr << "\n";
                    }
                }
            }

            batchData.h_usedReadIds.resize(batchData.totalNumberOfUsedIds);
            batchData.h_numUsedReadIdsPerAnchor.resize(batchData.numTasks);
            batchData.h_numUsedReadIdsPerAnchorPrefixSum.resize(batchData.numTasks);
            batchData.d_segmentIds2.resize(batchData.totalNumberOfUsedIds);

            cudaMemcpyAsync(
                batchData.d_inputanchormatedata.get(),
                batchData.h_inputanchormatedata.get(),
                sizeof(unsigned int) * numActiveTasks * encodedSequencePitchInInts,
                H2D,
                firstStream
            ); CUERR;            

            cudaMemcpyAsync(
                batchData.d_subjectSequencesData.get(),
                batchData.h_subjectSequencesData.get(),
                sizeof(unsigned int) * numActiveTasks * encodedSequencePitchInInts,
                H2D,
                firstStream
            ); CUERR;
    
            cudaMemcpyAsync(
                batchData.d_anchorSequencesLength.get(),
                batchData.h_anchorSequencesLength.get(),
                sizeof(int) * numActiveTasks,
                H2D,
                firstStream
            ); CUERR;

            cudaMemcpyAsync(
                batchData.d_anchorReadIds.get(),
                batchData.h_anchorReadIds.get(),
                sizeof(read_number) * batchData.numTasks,
                H2D,
                secondStream
            ); CUERR;

            cudaMemcpyAsync(
                batchData.d_mateReadIds.get(),
                batchData.h_mateReadIds.get(),
                sizeof(read_number) * batchData.numTasks,
                H2D,
                secondStream
            ); CUERR;

            batchData.h_numUsedReadIdsPerAnchorPrefixSum[0] = 0;
            {
                auto h_usedReadIdsIter = batchData.h_usedReadIds.begin();
                for(int i = 0; i < batchData.numTasks; i++){
                    auto& task = vecAccess(tasks, indicesOfActiveTasks[i]);
                    h_usedReadIdsIter = std::copy(
                        task.allUsedCandidateReadIdPairs.begin(),
                        task.allUsedCandidateReadIdPairs.end(),
                        h_usedReadIdsIter
                    );
                    batchData.h_numUsedReadIdsPerAnchor[i] = task.allUsedCandidateReadIdPairs.size();

                    if(i < batchData.numTasks - 1){
                        batchData.h_numUsedReadIdsPerAnchorPrefixSum[i+1] 
                            = batchData.h_numUsedReadIdsPerAnchorPrefixSum[i] + batchData.h_numUsedReadIdsPerAnchor[i];
                    }
                }
            }

            hashTimer.start();

            getCandidateReadIds(batchData, firstStream);

            cudaMemcpyAsync(
                &batchData.totalNumCandidates,
                batchData.d_numCandidatesPerAnchorPrefixSum.get() + batchData.numTasks,
                sizeof(int),
                D2H,
                firstStream
            ); CUERR;
    
            cudaStreamSynchronize(firstStream); CUERR;
            cudaEventRecord(events[0], secondStream); CUERR;
            cudaStreamWaitEvent(firstStream, events[0], 0); CUERR;

            batchData.h_candidateReadIds.resize(batchData.totalNumCandidates);

            batchData.d_anchorIndicesOfCandidates.resize(batchData.totalNumCandidates);
            batchData.d_anchorIndicesOfCandidates2.resize(batchData.totalNumCandidates);
            batchData.h_flagscandidates.resize(batchData.totalNumCandidates);
            batchData.d_flagscandidates.resize(batchData.totalNumCandidates);
            batchData.h_flagsanchors.resize(batchData.numTasks);
            batchData.d_flagsanchors.resize(batchData.numTasks);
            batchData.h_candidateReadIds2.resize(batchData.totalNumCandidates);
            batchData.d_candidateReadIds2.resize(batchData.totalNumCandidates);
            batchData.h_candidateReadIds3.resize(batchData.totalNumCandidates);

            //determine required temp bytes for following cub calls, and allocate temp storage

            std::size_t cubBytes = 0;
            std::size_t cubBytes2 = 0;

            cub::DeviceSelect::Flagged(
                nullptr,
                cubBytes2,
                batchData.d_candidateReadIds.data(),
                batchData.d_flagscandidates.data(),
                batchData.d_candidateReadIds2.data(),
                batchData.h_numCandidates.data(),
                batchData.totalNumCandidates,
                firstStream
            );
            
            cubBytes = std::max(cubBytes, cubBytes2);
            
            cub::DeviceScan::InclusiveSum(
                nullptr, 
                cubBytes2, 
                batchData.d_numCandidatesPerAnchor2.data(), 
                batchData.d_numCandidatesPerAnchorPrefixSum2.data() + 1, 
                batchData.numTasks,
                firstStream
            );
            
            cubBytes = std::max(cubBytes, cubBytes2);
            
            cub::DeviceScan::InclusiveScan(
                nullptr, 
                cubBytes2, 
                batchData.d_anchorIndicesOfCandidates.data(), 
                batchData.d_anchorIndicesOfCandidates.data(), 
                cub::Max{},
                batchData.totalNumCandidates,
                firstStream
            );
            
            cubBytes = std::max(cubBytes, cubBytes2);
            
            cub::DeviceScan::InclusiveScan(
                nullptr, 
                cubBytes2, 
                batchData.d_segmentIds2.data(), 
                batchData.d_segmentIds2.data(), 
                cub::Max{},
                batchData.totalNumberOfUsedIds,
                firstStream
            );
            
            cubBytes = std::max(cubBytes, cubBytes2);
            
            cub::DeviceScan::InclusiveSum(
                nullptr, 
                cubBytes2, 
                batchData.d_numCandidatesPerAnchor.data(), 
                batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1, 
                batchData.numTasks,
                firstStream
            );
            
            cubBytes = std::max(cubBytes, cubBytes2);
            
            cub::DeviceSelect::Flagged(
                nullptr,
                cubBytes2,
                thrust::make_counting_iterator(0),
                batchData.d_flagsanchors.data(),
                batchData.d_indexlist1.data(),
                batchData.h_numAnchors.data(),
                batchData.numTasks,
                firstStream
            );
            
            cubBytes = std::max(cubBytes, cubBytes2);
            
            cub::DeviceSelect::Flagged(
                nullptr,
                cubBytes2,
                batchData.d_inputanchormatedata.data(),
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator(0),
                    SequenceFlagMultiplier{batchData.d_flagsanchors.data(), int(encodedSequencePitchInInts)}
                ),
                batchData.d_anchormatedata.data(),
                thrust::make_discard_iterator(),
                batchData.numTasks * encodedSequencePitchInInts,
                firstStream
            );
            
            cubBytes = std::max(cubBytes, cubBytes2);

            //auto cubtempstorage = thrustCachingAllocator1.allocate(cubBytes);

            void* cubtempstorage; cubAllocator->DeviceAllocate((void**)&cubtempstorage, cubBytes, firstStream);

            //cub temp allocated          
            
            helpers::call_fill_kernel_async(batchData.d_flagscandidates.data(), batchData.d_flagscandidates.size(), false, firstStream);
            
            //flag candidates to remove because they are equal to anchor id or equal to mate id
            flagCandidateIdsWhichAreEqualToAnchorOrMateKernel<<<4096, 128, 0, firstStream>>>(
                batchData.d_candidateReadIds.data(),
                batchData.d_anchorReadIds.data(),
                batchData.d_mateReadIds.data(),
                batchData.d_numCandidatesPerAnchorPrefixSum.data(),
                batchData.d_numCandidatesPerAnchor.data(),
                batchData.d_flagscandidates.data(),
                batchData.d_flagsanchors.data(),
                batchData.d_numCandidatesPerAnchor2.data(),
                batchData.numTasks,
                tasks[0].pairedEnd
            );
            CUERR;

            //copy selected candidate ids

            cub::DeviceSelect::Flagged(
                cubtempstorage,
                cubBytes,
                batchData.d_candidateReadIds.data(),
                batchData.d_flagscandidates.data(),
                batchData.d_candidateReadIds2.data(),
                batchData.h_numCandidates.data(),
                batchData.totalNumCandidates,
                firstStream
            );

            cudaStreamSynchronize(firstStream); CUERR;

            batchData.totalNumCandidates = *batchData.h_numCandidates;

            //compute prefix sum of number of candidates per anchor
            helpers::call_set_kernel_async(batchData.d_numCandidatesPerAnchorPrefixSum2.data(), 0, 0, firstStream);

            cub::DeviceScan::InclusiveSum(
                cubtempstorage, 
                cubBytes, 
                batchData.d_numCandidatesPerAnchor2.data(), 
                batchData.d_numCandidatesPerAnchorPrefixSum2.data() + 1, 
                batchData.numTasks,
                firstStream
            );

            //compute segment ids for candidate read ids
            helpers::call_fill_kernel_async(batchData.d_anchorIndicesOfCandidates.data(), batchData.totalNumCandidates, 0, firstStream);

            setFirstSegmentIdsKernel<<<SDIV(batchData.numTasks, 256), 256, 0, firstStream>>>(
                batchData.d_numCandidatesPerAnchor2.data(),
                batchData.d_anchorIndicesOfCandidates.data(),
                batchData.d_numCandidatesPerAnchorPrefixSum2.data(),
                batchData.numTasks
            );

            cub::DeviceScan::InclusiveScan(
                cubtempstorage, 
                cubBytes, 
                batchData.d_anchorIndicesOfCandidates.data(), 
                batchData.d_anchorIndicesOfCandidates.data(), 
                cub::Max{},
                batchData.totalNumCandidates,
                firstStream
            );

            //compute segment ids for used candidate read ids
            helpers::call_fill_kernel_async(batchData.d_segmentIds2.data(), batchData.totalNumberOfUsedIds, 0, firstStream);

            setFirstSegmentIdsKernel<<<SDIV(batchData.numTasks, 256), 256, 0, firstStream>>>(
                batchData.h_numUsedReadIdsPerAnchor.data(),
                batchData.d_segmentIds2.data(),
                batchData.h_numUsedReadIdsPerAnchorPrefixSum.data(),
                batchData.numTasks
            );

            cub::DeviceScan::InclusiveScan(
                cubtempstorage, 
                cubBytes, 
                batchData.d_segmentIds2.data(), 
                batchData.d_segmentIds2.data(), 
                cub::Max{},
                batchData.totalNumberOfUsedIds,
                firstStream
            );
            
            //compute segmented set difference between candidate read ids and used candidate read ids
            auto d_candidateReadIds_end = GpuSegmentedSetOperation{}.difference(
                thrustCachingAllocator1,
                batchData.d_candidateReadIds2.data(),
                batchData.d_numCandidatesPerAnchor2.data(),
                batchData.d_numCandidatesPerAnchorPrefixSum2.data(),
                batchData.d_anchorIndicesOfCandidates.data(),
                batchData.totalNumCandidates,
                batchData.h_usedReadIds.data(),
                batchData.h_numUsedReadIdsPerAnchor.data(),
                batchData.h_numUsedReadIdsPerAnchorPrefixSum.data(),
                batchData.d_segmentIds2.data(),
                batchData.totalNumberOfUsedIds,
                batchData.numTasks,        
                batchData.d_candidateReadIds.data(),
                batchData.d_numCandidatesPerAnchor.data(),
                batchData.d_anchorIndicesOfCandidates2.data(),
                firstStream
            );

            std::swap(batchData.d_anchorIndicesOfCandidates, batchData.d_anchorIndicesOfCandidates2);

            batchData.totalNumCandidates = std::distance(batchData.d_candidateReadIds.data(), d_candidateReadIds_end);

            //compute prefix sum of new segment sizes
            cub::DeviceScan::InclusiveSum(
                cubtempstorage, 
                cubBytes, 
                batchData.d_numCandidatesPerAnchor.data(), 
                batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1, 
                batchData.numTasks,
                firstStream
            );

            //determine task ids with removed mates

            cub::DeviceSelect::Flagged(
                cubtempstorage,
                cubBytes,
                thrust::make_counting_iterator(0),
                batchData.d_flagsanchors.data(),
                batchData.d_indexlist1.data(),
                batchData.h_numAnchors.data(),
                batchData.numTasks,
                firstStream
            );

            cudaStreamSynchronize(firstStream); CUERR;

            batchData.numTasksWithMateRemoved = *batchData.h_numAnchors;

            if(batchData.numTasksWithMateRemoved > 0){

                //copy mate sequence data of removed mates
                    
                cub::DeviceSelect::Flagged(
                    cubtempstorage,
                    cubBytes,
                    batchData.d_inputanchormatedata.data(),
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator(0),
                        SequenceFlagMultiplier{batchData.d_flagsanchors.data(), int(encodedSequencePitchInInts)}
                    ),
                    batchData.d_anchormatedata.data(),
                    thrust::make_discard_iterator(),
                    batchData.numTasks * encodedSequencePitchInInts,
                    firstStream
                );
            }

            cubAllocator->DeviceFree(cubtempstorage);

            hashTimer.stop();
   
            //allocate data for candidate sequences 
            batchData.h_candidateSequencesLength.resize(batchData.totalNumCandidates);
            batchData.h_candidateSequencesData.resize(encodedSequencePitchInInts * batchData.totalNumCandidates);
            batchData.h_candidateSequencesRevcData.resize(encodedSequencePitchInInts * batchData.totalNumCandidates);

            batchData.d_candidateSequencesLength.resize(batchData.totalNumCandidates);
            batchData.d_candidateSequencesData.resize(encodedSequencePitchInInts * batchData.totalNumCandidates);
            batchData.d_candidateSequencesRevcData.resize(encodedSequencePitchInInts * batchData.totalNumCandidates);

            batchData.d_candidateSequencesLength2.resize(batchData.totalNumCandidates);
            batchData.d_candidateSequencesData2.resize(encodedSequencePitchInInts * batchData.totalNumCandidates);
            batchData.d_candidateSequencesRevcData2.resize(encodedSequencePitchInInts * batchData.totalNumCandidates);
            batchData.d_candidateReadIds2.resize(batchData.totalNumCandidates);

            batchData.h_intbuffercandidates.resize(batchData.totalNumCandidates);
            batchData.d_intbuffercandidates.resize(batchData.totalNumCandidates);
            batchData.h_flagscandidates.resize(batchData.totalNumCandidates);
            batchData.d_flagscandidates.resize(batchData.totalNumCandidates);

            batchData.h_alignment_overlaps.resize(batchData.totalNumCandidates);
            batchData.h_alignment_shifts.resize(batchData.totalNumCandidates);
            batchData.h_alignment_nOps.resize(batchData.totalNumCandidates);
            batchData.h_alignment_isValid.resize(batchData.totalNumCandidates);
            batchData.h_alignment_best_alignment_flags.resize(batchData.totalNumCandidates);

            batchData.d_alignment_overlaps.resize(batchData.totalNumCandidates);
            batchData.d_alignment_shifts.resize(batchData.totalNumCandidates);
            batchData.d_alignment_nOps.resize(batchData.totalNumCandidates);
            batchData.d_alignment_isValid.resize(batchData.totalNumCandidates);
            batchData.d_alignment_best_alignment_flags.resize(batchData.totalNumCandidates);

            batchData.d_alignment_overlaps2.resize(batchData.totalNumCandidates);
            batchData.d_alignment_shifts2.resize(batchData.totalNumCandidates);
            batchData.d_alignment_nOps2.resize(batchData.totalNumCandidates);
            batchData.d_alignment_isValid2.resize(batchData.totalNumCandidates);
            batchData.d_alignment_best_alignment_flags2.resize(batchData.totalNumCandidates);

            collectTimer.start();

            loadCandidateSequenceData(batchData, firstStream);      

            eraseDataOfRemovedMates(batchData, firstStream);

            thrust::copy_n(
                thrustPolicy1,
                thrust::make_zip_iterator(thrust::make_tuple(
                    batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1,
                    batchData.d_numCandidatesPerAnchor.data()
                )),
                batchData.numTasks,
                thrust::make_zip_iterator(thrust::make_tuple(
                    batchData.h_numCandidatesPerAnchorPrefixSum.data() + 1,
                    batchData.h_numCandidatesPerAnchor.data()
                ))
            );

            // cudaMemcpyAsync(
            //     batchData.h_numCandidatesPerAnchorPrefixSum.get(),
            //     batchData.d_numCandidatesPerAnchorPrefixSum.get(),
            //     sizeof(int) * (batchData.numTasks+1),
            //     D2H,
            //     firstStream
            // ); CUERR;

            // cudaMemcpyAsync(
            //     batchData.h_numCandidatesPerAnchor.get(),
            //     batchData.d_numCandidatesPerAnchor.get(),
            //     sizeof(int) * batchData.numTasks,
            //     D2H,
            //     firstStream
            // ); CUERR;
    
            cudaStreamSynchronize(firstStream); CUERR;

            batchData.totalNumCandidates = batchData.h_numCandidatesPerAnchorPrefixSum[batchData.numTasks];
    
            #ifdef checkdebugtasks
            cudaMemcpyAsync(
                batchData.h_candidateSequencesData.get(),
                batchData.d_candidateSequencesData.get(),
                sizeof(unsigned int) * batchData.totalNumCandidates * encodedSequencePitchInInts,
                H2D,
                secondStream
            ); CUERR;
    
            cudaMemcpyAsync(
                batchData.h_candidateSequencesRevcData.get(),
                batchData.d_candidateSequencesRevcData.get(),
                sizeof(unsigned int) * batchData.totalNumCandidates * encodedSequencePitchInInts,
                H2D,
                secondStream
            ); CUERR;
            #endif
            


            collectTimer.stop();

            /*
                Compute alignments
            */

            alignmentTimer.start();

            calculateAlignments(batchData, firstStream);

            alignmentTimer.stop();

            

            #ifdef checkdebugtasks
            cudaStreamSynchronize(secondStream); CUERR;
            // unpack batchData sequences into tasks
            for(int i = 0; i < numActiveTasks; i++){
                auto& task = vecAccess(tasks, indicesOfActiveTasks[i]);

                const int numCandidates = batchData.h_numCandidatesPerAnchor[i];
                const int offset = batchData.h_numCandidatesPerAnchorPrefixSum[i];

                task.candidateSequencesFwdData.resize(numCandidates * encodedSequencePitchInInts);
                std::copy_n(batchData.h_candidateSequencesData.data() + offset * encodedSequencePitchInInts, numCandidates * encodedSequencePitchInInts, task.candidateSequencesFwdData.begin());

                task.candidateSequencesRevcData.resize(numCandidates * encodedSequencePitchInInts);
                std::copy_n(batchData.h_candidateSequencesRevcData.data() + offset * encodedSequencePitchInInts, numCandidates * encodedSequencePitchInInts, task.candidateSequencesRevcData.begin());
            }
            #endif
           
       
            filterAlignments(batchData, firstStream);

            thrust::copy_n(
                thrustPolicy1,
                thrust::make_zip_iterator(thrust::make_tuple(
                    batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1,
                    batchData.d_numCandidatesPerAnchor.data()
                )),
                batchData.numTasks,
                thrust::make_zip_iterator(thrust::make_tuple(
                    batchData.h_numCandidatesPerAnchorPrefixSum.data() + 1,
                    batchData.h_numCandidatesPerAnchor.data()
                ))
            );
    
            cudaStreamSynchronize(firstStream); CUERR;

            batchData.totalNumCandidates = batchData.h_numCandidatesPerAnchorPrefixSum[batchData.numTasks];

            cudaMemcpyAsync(
                batchData.h_candidateSequencesData.get(),
                batchData.d_candidateSequencesData.get(),
                sizeof(unsigned int) * batchData.totalNumCandidates * encodedSequencePitchInInts,
                H2D,
                firstStream
            ); CUERR;

            auto d_zipped_begin = thrust::make_zip_iterator(
                thrust::make_tuple(
                    batchData.d_candidateReadIds.data(),
                    batchData.d_candidateSequencesLength.data(),
                    batchData.d_alignment_overlaps.data(),
                    batchData.d_alignment_isValid.data(),
                    batchData.d_alignment_shifts.data(),
                    batchData.d_alignment_nOps.data(),
                    batchData.d_alignment_best_alignment_flags.data()
                )
            );

            auto h_zipped_begin = thrust::make_zip_iterator(
                thrust::make_tuple(
                    batchData.h_candidateReadIds.data(),
                    batchData.h_candidateSequencesLength.data(),
                    batchData.h_alignment_overlaps.data(),
                    batchData.h_alignment_isValid.data(),
                    batchData.h_alignment_shifts.data(),
                    batchData.h_alignment_nOps.data(),
                    batchData.h_alignment_best_alignment_flags.data()
                )
            );

            thrust::copy_n(
                thrustPolicy1,
                d_zipped_begin,
                batchData.totalNumCandidates,
                h_zipped_begin
            );

            cudaStreamSynchronize(firstStream); CUERR;

            // unpack batchData into tasks

            auto copyBatchDataIntoTask = [&](auto& task, int taskindex){
                const int numCandidates = batchData.h_numCandidatesPerAnchor[taskindex];
                const int offset = batchData.h_numCandidatesPerAnchorPrefixSum[taskindex];

                task.candidateReadIds.resize(numCandidates);
                std::copy_n(batchData.h_candidateReadIds.data() + offset, numCandidates, task.candidateReadIds.begin());

                task.candidateSequenceLengths.resize(numCandidates);
                std::copy_n(batchData.h_candidateSequencesLength.data() + offset, numCandidates, task.candidateSequenceLengths.begin());

                task.candidateSequenceData.resize(numCandidates * encodedSequencePitchInInts);
                std::copy_n(
                    batchData.h_candidateSequencesData.data() + offset * encodedSequencePitchInInts, 
                    numCandidates * encodedSequencePitchInInts, 
                    task.candidateSequenceData.begin()
                );

                task.alignmentFlags.resize(numCandidates);
                task.alignments.resize(numCandidates);

                for(int c = 0; c < numCandidates; c++){
                    task.alignments[c].shift = batchData.h_alignment_shifts[offset + c];
                    task.alignments[c].overlap = batchData.h_alignment_overlaps[offset + c];
                    task.alignments[c].nOps = batchData.h_alignment_nOps[offset + c];
                    #ifdef checkdebugtasks
                    task.alignments[c].isValid = batchData.h_alignment_isValid[offset + c];
                    #endif
                    task.alignmentFlags[c] = batchData.h_alignment_best_alignment_flags[offset + c];
                }

                task.mateRemovedFromCandidates = false; //debug. not required
                task.numRemainingCandidates = numCandidates;

                if(task.numRemainingCandidates == 0){
                    task.abort = true;
                    task.abortReason = AbortReason::NoPairedCandidatesAfterAlignment;
                }
            };


            for(int i = 0; i < numActiveTasks; i++){
                auto& task = vecAccess(tasks, indicesOfActiveTasks[i]);

                #ifdef checkdebugtasks
                copyBatchDataIntoTask(task, i);
                task.dataIsAvailable = true;
                #else
                task.mateRemovedFromCandidates = false; //debug. not required
                task.numRemainingCandidates = batchData.h_numCandidatesPerAnchor[i];

                if(task.numRemainingCandidates == 0){
                    task.abort = true;
                    task.abortReason = AbortReason::NoPairedCandidatesAfterAlignment;
                }
                #endif
            }

            #ifdef checkdebugtasks
            for(int i = 0; i < numActiveTasks; i++){
                auto& newtask = tasks[indicesOfActiveTasks[i]];
                auto& oldtask = debugtasks[indicesOfActiveTasks[i]];

                if(newtask != oldtask){
                    std::cerr << "old task and new task differ. i=" 
                        << i << ", indicesOfActiveTasks[i] " << indicesOfActiveTasks[i] << "\n";
                    assert(false);
                }
            }
            #endif

            if(true){
                //construct gpu msa
                
                batchData.h_consensus.resize(numActiveTasks * msaColumnPitchInElements);
                batchData.h_msa_column_properties.resize(numActiveTasks);

                batchData.d_consensus.resize(numActiveTasks * msaColumnPitchInElements);
                batchData.d_support.resize(numActiveTasks * msaColumnPitchInElements);
                batchData.d_coverage.resize(numActiveTasks * msaColumnPitchInElements);
                batchData.d_origWeights.resize(numActiveTasks * msaColumnPitchInElements);
                batchData.d_origCoverages.resize(numActiveTasks * msaColumnPitchInElements);
                batchData.d_msa_column_properties.resize(numActiveTasks);
                batchData.d_counts.resize(numActiveTasks * 4 * msaColumnPitchInElements);
                batchData.d_weights.resize(numActiveTasks * 4 * msaColumnPitchInElements);

                batchData.d_intbuffercandidates.resize(batchData.totalNumCandidates);
                batchData.d_indexlist1.resize(batchData.totalNumCandidates);
                batchData.d_numCandidatesPerAnchor2.resize(batchData.numTasks);
                batchData.d_flagscandidates.resize(batchData.totalNumCandidates);

                helpers::call_fill_kernel_async(batchData.d_msa_column_properties.data(), numActiveTasks, gpu::MSAColumnProperties{-1,-1,-1,-1}, firstStream); CUERR;                

                int* const indices1 = batchData.d_intbuffercandidates.data();
                int* const indices2 = batchData.d_indexlist1.data();

                helpers::lambda_kernel<<<batchData.numTasks, 128, 0, firstStream>>>(
                    [
                        indices1,
                        d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.get(),
                        d_numCandidatesPerAnchorPrefixSum = batchData.d_numCandidatesPerAnchorPrefixSum.get()
                    ] __device__ (){
                        const int num = d_numCandidatesPerAnchor[blockIdx.x];
                        const int offset = d_numCandidatesPerAnchorPrefixSum[blockIdx.x];
                        
                        for(int i = threadIdx.x; i < num; i += blockDim.x){
                            indices1[offset + i] = i;
                        }
                    }
                );

                gpu::GPUMultiMSA multiMSA;

                *batchData.h_numAnchors = numActiveTasks;

                multiMSA.numMSAs = numActiveTasks;
                multiMSA.columnPitchInElements = msaColumnPitchInElements;
                multiMSA.counts = batchData.d_counts.get();
                multiMSA.weights = batchData.d_weights.get();
                multiMSA.coverages = batchData.d_coverage.get();
                multiMSA.consensus = batchData.d_consensus.get();
                multiMSA.support = batchData.d_support.get();
                multiMSA.origWeights = batchData.d_origWeights.get();
                multiMSA.origCoverages = batchData.d_origCoverages.get();
                multiMSA.columnProperties = batchData.d_msa_column_properties.get();

                //sequences which have alignment flag BestAlignment_t::ReverseComplement are already present as reverse complement in d_candidateSequencesData.
                //To not reverse complement them again during msa construction, treat all stored sequences are forward
                helpers::call_fill_kernel_async(batchData.d_alignment_best_alignment_flags2.data(), batchData.totalNumCandidates, BestAlignment_t::Forward, firstStream); CUERR;

                callConstructMultipleSequenceAlignmentsKernel_async(
                    multiMSA,
                    batchData.d_alignment_overlaps.get(),
                    batchData.d_alignment_shifts.get(),
                    batchData.d_alignment_nOps.get(),
                    //batchData.d_alignment_best_alignment_flags.get(),
                    batchData.d_alignment_best_alignment_flags2.get(),
                    batchData.d_anchorSequencesLength.get(),
                    batchData.d_candidateSequencesLength.get(),
                    indices1, //d_indices,
                    batchData.d_numCandidatesPerAnchor.get(),
                    batchData.d_numCandidatesPerAnchorPrefixSum.get(),
                    batchData.d_subjectSequencesData.get(),
                    batchData.d_candidateSequencesData.get(),
                    nullptr, //d_anchor_qualities.get(),
                    nullptr, //d_candidate_qualities.get(),
                    batchData.h_numAnchors.get(), //d_numAnchors
                    goodAlignmentProperties.maxErrorRate,
                    batchData.numTasks,
                    batchData.totalNumCandidates,
                    false, //correctionOptions->useQualityScores,
                    encodedSequencePitchInInts,
                    qualityPitchInBytes,
                    firstStream,
                    kernelLaunchHandle
                );

                //refine msa
                bool* d_shouldBeKept = (bool*)batchData.d_flagscandidates.get();
                helpers::call_fill_kernel_async(indices2, batchData.totalNumCandidates, -42, firstStream); CUERR;
                helpers::call_fill_kernel_async(batchData.d_numCandidatesPerAnchor2.data(), batchData.numTasks, -13, firstStream); CUERR;
                helpers::call_fill_kernel_async(batchData.d_numCandidates2.data(), 1, -9, firstStream); CUERR;
                callMsaCandidateRefinementKernel_multiiter_async(
                    indices2,
                    batchData.d_numCandidatesPerAnchor2.data(),
                    batchData.d_numCandidates2.get(),
                    multiMSA,
                    //batchData.d_alignment_best_alignment_flags.get(),
                    batchData.d_alignment_best_alignment_flags2.get(),
                    batchData.d_alignment_shifts.get(),
                    batchData.d_alignment_nOps.get(),
                    batchData.d_alignment_overlaps.get(),
                    batchData.d_subjectSequencesData.get(),
                    batchData.d_candidateSequencesData.get(),
                    batchData.d_anchorSequencesLength.get(),
                    batchData.d_candidateSequencesLength.get(),
                    nullptr, //d_anchor_qualities.get(),
                    nullptr, //d_candidate_qualities.get(),
                    d_shouldBeKept,
                    batchData.d_numCandidatesPerAnchorPrefixSum.get(),
                    batchData.h_numAnchors.get(),
                    goodAlignmentProperties.maxErrorRate,
                    batchData.numTasks,
                    batchData.totalNumCandidates,
                    false, //correctionOptions->useQualityScores,
                    encodedSequencePitchInInts,
                    qualityPitchInBytes,
                    indices1, //d_indices,
                    batchData.d_numCandidatesPerAnchor.get(),
                    correctionOptions.estimatedCoverage,
                    getNumRefinementIterations(),
                    firstStream,
                    kernelLaunchHandle
                );

                if(tasks[1].myReadId == 3 && tasks[1].iteration == 0){
                    helpers::lambda_kernel<<<1, 1, 0, firstStream>>>(
                        [
                            shifts = batchData.d_alignment_shifts.data(),
                            d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.data(),
                            d_counts = batchData.d_counts.data(),
                            d_coverage = batchData.d_coverage.data(),
                            msaColumnPitchInElements = this->msaColumnPitchInElements,
                            d_msa_column_properties = batchData.d_msa_column_properties.data(),
                            numTasks = batchData.numTasks,
                            splitcolumnsPitchElements = 32,
                            d_possibleSplitColumns = batchData.d_possibleSplitColumns.data(),
                            d_numPossibleSplitColumnsPerTask = batchData.d_numPossibleSplitColumnsPerAnchor.data(),
                            d_candidateSequencesLength = batchData.d_candidateSequencesLength.data(),
                            d_candidateSequencesData = batchData.d_candidateSequencesData.data(),
                            encodedSequencePitchInInts = this->encodedSequencePitchInInts,
                            d_subjectSequencesData = batchData.d_subjectSequencesData.data(),
                            d_numCandidatesPerAnchorPrefixSum = batchData.d_numCandidatesPerAnchorPrefixSum.data()
                        ] __device__ (){
        
                            const int task = 1;

                            const int offset = d_numCandidatesPerAnchorPrefixSum[task];
    
                                const int firstColumn = d_msa_column_properties[task].firstColumn_incl;
                                const int lastColumnExcl = d_msa_column_properties[task].lastColumn_excl;                                

                                int* myCountsPtr[4];
                                myCountsPtr[0] = d_counts + 4 * msaColumnPitchInElements * task + 0 * msaColumnPitchInElements;
                                myCountsPtr[1] = d_counts + 4 * msaColumnPitchInElements * task + 1 * msaColumnPitchInElements;
                                myCountsPtr[2] = d_counts + 4 * msaColumnPitchInElements * task + 2 * msaColumnPitchInElements;
                                myCountsPtr[3] = d_counts + 4 * msaColumnPitchInElements * task + 3 * msaColumnPitchInElements;

                                printf("anchor\n");
                                for (int k = 0; k < 100; k++){
                                    const std::uint8_t base = SequenceHelpers::getEncodedNuc2Bit(d_subjectSequencesData + task * encodedSequencePitchInInts, 100, k);
                                    const char aaaa = SequenceHelpers::decodeBase(base);
                                    printf("%c", aaaa);
                                }
                                printf("\n");

                                printf("candidates:\n");
                                for(int i = 0; i < d_numCandidatesPerAnchor[task]; i++){
                                    printf("len %d\n", d_candidateSequencesLength[i]);
                                    for (int k = 0; k < d_candidateSequencesLength[i]; k++){
                                        const std::uint8_t base = SequenceHelpers::getEncodedNuc2Bit(
                                            d_candidateSequencesData + (offset + i) * encodedSequencePitchInInts, 
                                            d_candidateSequencesLength[i], 
                                            k
                                        );
                                        const char aaaa = SequenceHelpers::decodeBase(base);
                                        printf("%c", aaaa);
                                    }
                                    printf("\n");
                                }
                                printf("\n");

                                printf("shifts:\n");
                                for(int i = 0; i < d_numCandidatesPerAnchor[task]; i++){
                                    printf("%d ", shifts[offset + i]);
                                }
                                printf("\n");

                                printf("A:\n");
                                for(int i = firstColumn; i < lastColumnExcl; i++){
                                    printf("%d ", myCountsPtr[0][i]);
                                }
                                printf("\n");

                                printf("C:\n");
                                for(int i = firstColumn; i < lastColumnExcl; i++){
                                    printf("%d ", myCountsPtr[1][i]);
                                }
                                printf("\n");

                                printf("G:\n");
                                for(int i = firstColumn; i < lastColumnExcl; i++){
                                    printf("%d ", myCountsPtr[2][i]);
                                }
                                printf("\n");

                                printf("T:\n");
                                for(int i = firstColumn; i < lastColumnExcl; i++){
                                    printf("%d ", myCountsPtr[3][i]);
                                }
                                printf("\n");


                                printf("cov:\n");
                                for(int i = firstColumn; i < lastColumnExcl; i++){
                                    printf("%d ", d_coverage[i]);
                                }
                                printf("\n");
                        }
                    ); CUERR;

                    cudaDeviceSynchronize(); CUERR; //DEBUG
                }



                cudaEventRecord(events[0], firstStream); CUERR;
                cudaStreamWaitEvent(secondStream, events[0], 0); CUERR;

                cudaMemcpyAsync(
                    batchData.h_numCandidates.data(),
                    batchData.d_numCandidates.data(),
                    sizeof(int),
                    D2H,
                    secondStream
                ); CUERR;

                cudaEventRecord(events[0], secondStream); CUERR;

                cudaMemcpyAsync(
                    batchData.h_msa_column_properties.data(),
                    batchData.d_msa_column_properties.data(),
                    sizeof(gpu::MSAColumnProperties) * batchData.numTasks,
                    D2H,
                    secondStream
                ); CUERR;

                //convert encoded consensus to characters and copy to host
                helpers::lambda_kernel<<<batchData.numTasks, 128, 0, secondStream>>>(
                    [
                        msaColumnPitchInElements = this->msaColumnPitchInElements,
                        d_consensus = batchData.d_consensus.data(),
                        h_consensus = batchData.h_consensus.data()
                    ] __device__ (){
                        
                        for(int i = threadIdx.x; i < msaColumnPitchInElements; i += blockDim.x){
                            std::uint8_t encoded = d_consensus[blockIdx.x * msaColumnPitchInElements + i];

                            char decoded = 'F';
                            if(encoded == std::uint8_t{0}){
                                decoded = 'A';
                            }else if(encoded == std::uint8_t{1}){
                                decoded = 'C';
                            }else if(encoded == std::uint8_t{2}){
                                decoded = 'G';
                            }else if(encoded == std::uint8_t{3}){
                                decoded = 'T';
                            }

                            h_consensus[blockIdx.x * msaColumnPitchInElements + i] = decoded;
                        }
                    }
                );



                bool cubdebugsync = false;

                //allocate cub storage
                std::size_t cubBytes = 0;
                std::size_t cubBytes2 = 0;
                cudaError_t cubstatus = cub::DeviceScan::InclusiveSum(
                    nullptr,
                    cubBytes,
                    batchData.d_numCandidatesPerAnchor2.get(), 
                    batchData.d_numCandidatesPerAnchorPrefixSum2.get() + 1, 
                    batchData.numTasks, 
                    firstStream,
                    cubdebugsync
                );
                assert(cubstatus == cudaSuccess);

                auto in_zipped_begin = thrust::make_zip_iterator(
                    thrust::make_tuple(
                        batchData.d_candidateReadIds.data(),
                        batchData.d_candidateSequencesLength.data(),
                        batchData.d_alignment_overlaps.data(),
                        batchData.d_alignment_isValid.data(),
                        batchData.d_alignment_shifts.data(),
                        batchData.d_alignment_nOps.data(),
                        batchData.d_alignment_best_alignment_flags.data()
                    )
                );
    
                auto out_zipped_begin = thrust::make_zip_iterator(
                    thrust::make_tuple(
                        batchData.d_candidateReadIds2.data(),
                        batchData.d_candidateSequencesLength2.data(),
                        batchData.d_alignment_overlaps2.data(),
                        batchData.d_alignment_isValid2.data(),
                        batchData.d_alignment_shifts2.data(),
                        batchData.d_alignment_nOps2.data(),
                        batchData.d_alignment_best_alignment_flags2.data()
                    )
                );

                cubstatus = cub::DeviceSelect::Flagged(
                    nullptr,
                    cubBytes2,
                    in_zipped_begin,
                    batchData.d_flagscandidates.data(),
                    out_zipped_begin,
                    thrust::make_discard_iterator(),
                    batchData.totalNumCandidates,
                    firstStream,
                    cubdebugsync
                );

                assert(cubstatus == cudaSuccess);
                cubBytes = std::max(cubBytes, cubBytes2);

                cubstatus = cub::DeviceSelect::Flagged(
                    nullptr,
                    cubBytes2,
                    batchData.d_candidateSequencesData.data(),
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator(0),
                        SequenceFlagMultiplier{batchData.d_flagscandidates.data(), int(encodedSequencePitchInInts)}
                    ),
                    batchData.d_candidateSequencesData2.data(),
                    thrust::make_discard_iterator(),
                    batchData.totalNumCandidates * encodedSequencePitchInInts,
                    firstStream,
                    cubdebugsync
                );

                assert(cubstatus == cudaSuccess);
                cubBytes = std::max(cubBytes, cubBytes2);

                void* cubtemp; cubAllocator->DeviceAllocate((void**)&cubtemp, cubBytes, firstStream);

                cubstatus = cub::DeviceScan::InclusiveSum(
                    cubtemp,
                    cubBytes,
                    batchData.d_numCandidatesPerAnchor2.get(), 
                    batchData.d_numCandidatesPerAnchorPrefixSum2.get() + 1, 
                    batchData.numTasks, 
                    firstStream,
                    cubdebugsync
                );
                assert(cubstatus == cudaSuccess);

                helpers::call_fill_kernel_async(batchData.d_flagscandidates.data(), batchData.totalNumCandidates, false, firstStream); CUERR;

                //convert output indices from task-local indices to global indices
                helpers::lambda_kernel<<<batchData.numTasks, 128, 0, firstStream>>>(
                    [
                        d_flagscandidates = batchData.d_flagscandidates.data(),
                        indices2,
                        d_numCandidatesPerAnchor2 = batchData.d_numCandidatesPerAnchor2.get(),
                        d_numCandidatesPerAnchorPrefixSum = batchData.d_numCandidatesPerAnchorPrefixSum.get()
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

                //compact candidate sequences according to flags                

                cubstatus = cub::DeviceSelect::Flagged(
                    cubtemp,
                    cubBytes,
                    batchData.d_candidateSequencesData.data(),
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator(0),
                        SequenceFlagMultiplier{batchData.d_flagscandidates.data(), int(encodedSequencePitchInInts)}
                    ),
                    batchData.d_candidateSequencesData2.data(),
                    thrust::make_discard_iterator(),
                    batchData.totalNumCandidates * encodedSequencePitchInInts,
                    firstStream,
                    cubdebugsync
                );

                assert(cubstatus == cudaSuccess);

                //compact other candidate buffers according to flags

                cubstatus = cub::DeviceSelect::Flagged(
                    cubtemp,
                    cubBytes,
                    in_zipped_begin,
                    batchData.d_flagscandidates.data(),
                    out_zipped_begin,
                    thrust::make_discard_iterator(),
                    batchData.totalNumCandidates,
                    firstStream,
                    cubdebugsync
                );

                assert(cubstatus == cudaSuccess);

                std::swap(batchData.d_numCandidatesPerAnchor, batchData.d_numCandidatesPerAnchor2);
                std::swap(batchData.d_numCandidatesPerAnchorPrefixSum, batchData.d_numCandidatesPerAnchorPrefixSum2);                
                std::swap(batchData.d_candidateSequencesData, batchData.d_candidateSequencesData2);
                std::swap(batchData.d_candidateReadIds, batchData.d_candidateReadIds2);
                std::swap(batchData.d_candidateSequencesLength, batchData.d_candidateSequencesLength2);
                std::swap(batchData.d_alignment_overlaps, batchData.d_alignment_overlaps2);
                std::swap(batchData.d_alignment_isValid, batchData.d_alignment_isValid2);
                std::swap(batchData.d_alignment_shifts, batchData.d_alignment_shifts2);
                std::swap(batchData.d_alignment_nOps, batchData.d_alignment_nOps2);
                std::swap(batchData.d_alignment_best_alignment_flags, batchData.d_alignment_best_alignment_flags2);


                //compute possible msa splits

                batchData.h_possibleSplitColumns.resize(32 * batchData.numTasks);
                batchData.h_numPossibleSplitColumnsPerAnchor.resize(batchData.numTasks);

                batchData.d_possibleSplitColumns.resize(32 * batchData.numTasks);
                batchData.d_numPossibleSplitColumnsPerAnchor.resize(batchData.numTasks);

                helpers::lambda_kernel<<<batchData.numTasks, 128, 0, firstStream>>>(
                    [
                        d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.data(),
                        d_counts = batchData.d_counts.data(),
                        d_coverage = batchData.d_coverage.data(),
                        msaColumnPitchInElements = this->msaColumnPitchInElements,
                        d_msa_column_properties = batchData.d_msa_column_properties.data(),
                        numTasks = batchData.numTasks,
                        splitcolumnsPitchElements = 32,
                        d_possibleSplitColumns = batchData.d_possibleSplitColumns.data(),
                        d_numPossibleSplitColumnsPerTask = batchData.d_numPossibleSplitColumnsPerAnchor.data(),
                        size1 = batchData.d_msa_column_properties.size(),
                        size2 = batchData.d_counts.size(),
                        size3 = batchData.d_coverage.size()
                    ] __device__ (){

                        using PSC = MultipleSequenceAlignment::PossibleSplitColumn;
                        constexpr int maxColumnsPerTask = 32;

                        __shared__ PSC sharedPSC[maxColumnsPerTask];
                        __shared__ int broadcastint;

                        using BlockReduce = cub::BlockReduce<int, 128>;
                        using BlockScan = cub::BlockScan<int, 128>;
                        __shared__ typename BlockReduce::TempStorage blockreducetemp;
                        __shared__ typename BlockScan::TempStorage blockscantemp;

                        for(int task = blockIdx.x; task < numTasks; task += gridDim.x){

                            assert(task < size1);

                            int* const numSplitColumnsPtr = d_numPossibleSplitColumnsPerTask + task;
                            PSC* const splitColumnsPtr = d_possibleSplitColumns + splitcolumnsPitchElements * task;

                            if(d_numCandidatesPerAnchor[task] > 0){

                                //only check columns to the right of anchor
                                const int firstColumn = d_msa_column_properties[task].subjectColumnsEnd_excl;
                                const int lastColumnExcl = d_msa_column_properties[task].lastColumn_excl;                                

                                int* myCountsPtr[4];
                                myCountsPtr[0] = d_counts + 4 * msaColumnPitchInElements * task + 0 * msaColumnPitchInElements;
                                myCountsPtr[1] = d_counts + 4 * msaColumnPitchInElements * task + 1 * msaColumnPitchInElements;
                                myCountsPtr[2] = d_counts + 4 * msaColumnPitchInElements * task + 2 * msaColumnPitchInElements;
                                myCountsPtr[3] = d_counts + 4 * msaColumnPitchInElements * task + 3 * msaColumnPitchInElements;   
                                
                                int* myCoveragesPtr = d_coverage + msaColumnPitchInElements * task;

                                int totalNumResults = 0;

                                const int numIterations = SDIV(lastColumnExcl - firstColumn, blockDim.x);

                                for(int iteration = 0; iteration < numIterations; iteration++){
                                    const int col = firstColumn + iteration * blockDim.x + threadIdx.x;

                                    PSC myresults[3];
                                    int myNumResults = 0;

                                    if(col < lastColumnExcl){                                   
        
                                        auto checkNuc = [&](const auto& counts, const char nuc){
                                            if(myNumResults < 3){


                                                const float ratio = float(counts[col]) / float(myCoveragesPtr[col]);
                                                //if((counts[col] == 2 && fgeq(ratio, 0.4f) && fleq(ratio, 0.6f)) || counts[col] > 2){
                                                if(counts[col] >= 2 && fgeq(ratio, 0.4f) && fleq(ratio, 0.6f)){
        
                                                    #pragma unroll
                                                    for(int k = 0; k < 3; k++){
                                                        if(myNumResults == k){
                                                            myresults[k] = {nuc, col, ratio};
                                                            myNumResults++;
                                                            break;
                                                        }
                                                    }
                                                    
                                                }
                                            }
                                        };
        
                                        checkNuc(myCountsPtr[0], 'A');
                                        checkNuc(myCountsPtr[1], 'C');
                                        checkNuc(myCountsPtr[2], 'G');
                                        checkNuc(myCountsPtr[3], 'T');
        
                                        if(myNumResults != 2){
                                            myNumResults = 0;
                                        }    
                                    }

                                    int totalNumResultsIteration = BlockReduce(blockreducetemp).Sum(myNumResults);
                                    if(threadIdx.x == 0){
                                        broadcastint = totalNumResultsIteration;
                                    }
                                    __syncthreads();
                                    totalNumResultsIteration = broadcastint;

                                    if(totalNumResultsIteration + totalNumResults > maxColumnsPerTask){
                                        totalNumResults = 0;
                                        break;
                                    }else{
                                        int outputoffset = 0;

                                        BlockScan(blockscantemp).ExclusiveSum(myNumResults, outputoffset);

                                        if(myNumResults == 2){
                                            sharedPSC[totalNumResults + outputoffset + 0] = myresults[0];
                                            sharedPSC[totalNumResults + outputoffset + 1] = myresults[1];
                                        }

                                        totalNumResults += totalNumResultsIteration;
                                    }

                                    __syncthreads();
                                }

                                for(int i = threadIdx.x; i < totalNumResults; i += blockDim.x){
                                    splitColumnsPtr[i] = sharedPSC[i];
                                }

                                if(threadIdx.x == 0){
                                    *numSplitColumnsPtr = totalNumResults;
                                }
                            }else{
                                if(threadIdx.x == 0){
                                    *numSplitColumnsPtr = 0;
                                }
                            }
                        }
                    }
                ); CUERR;

                cudaMemcpyAsync(
                    batchData.h_possibleSplitColumns.get(),
                    batchData.d_possibleSplitColumns.get(),
                    sizeof(MultipleSequenceAlignment::PossibleSplitColumn) * 32 * batchData.numTasks,
                    D2H,
                    firstStream
                ); CUERR;

                cudaMemcpyAsync(
                    batchData.h_numPossibleSplitColumnsPerAnchor.get(),
                    batchData.d_numPossibleSplitColumnsPerAnchor.get(),
                    sizeof(int) * batchData.numTasks,
                    D2H,
                    firstStream
                ); CUERR;

                cudaEventSynchronize(events[0]); CUERR; //wait for h_numCandidates

                #if 0

                batchData.totalNumCandidates = *batchData.h_numCandidates;

                thrust::copy_n(
                    thrustPolicy1,
                    thrust::make_zip_iterator(thrust::make_tuple(
                        batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1,
                        batchData.d_numCandidatesPerAnchor.data()
                    )),
                    batchData.numTasks,
                    thrust::make_zip_iterator(thrust::make_tuple(
                        batchData.h_numCandidatesPerAnchorPrefixSum.data() + 1,
                        batchData.h_numCandidatesPerAnchor.data()
                    ))
                );

                //copy data to host
                cudaMemcpyAsync(
                    batchData.h_candidateSequencesData.get(),
                    batchData.d_candidateSequencesData.get(),
                    sizeof(unsigned int) * batchData.totalNumCandidates * encodedSequencePitchInInts,
                    D2H,
                    firstStream
                ); CUERR;
    
                auto d_zipped_begin = thrust::make_zip_iterator(
                    thrust::make_tuple(
                        batchData.d_candidateReadIds.data(),
                        batchData.d_candidateSequencesLength.data(),
                        batchData.d_alignment_overlaps.data(),
                        batchData.d_alignment_isValid.data(),
                        batchData.d_alignment_shifts.data(),
                        batchData.d_alignment_nOps.data(),
                        batchData.d_alignment_best_alignment_flags.data()
                    )
                );
    
                auto h_zipped_begin = thrust::make_zip_iterator(
                    thrust::make_tuple(
                        batchData.h_candidateReadIds.data(),
                        batchData.h_candidateSequencesLength.data(),
                        batchData.h_alignment_overlaps.data(),
                        batchData.h_alignment_isValid.data(),
                        batchData.h_alignment_shifts.data(),
                        batchData.h_alignment_nOps.data(),
                        batchData.h_alignment_best_alignment_flags.data()
                    )
                );
    
                thrust::copy_n(
                    thrustPolicy1,
                    d_zipped_begin,
                    batchData.totalNumCandidates,
                    h_zipped_begin
                );
                
    
                cudaStreamSynchronize(firstStream); CUERR;

                for(int i = 0; i < numActiveTasks; i++){
                    auto& task = vecAccess(tasks, indicesOfActiveTasks[i]);
    
                    #ifdef checkdebugtasks
                    copyBatchDataIntoTask(task, i);
                    task.dataIsAvailable = true;
                    #else
                    task.mateRemovedFromCandidates = false; //debug. not required
                    task.numRemainingCandidates = batchData.h_numCandidatesPerAnchor[i];
    
                    if(task.numRemainingCandidates == 0){
                        task.abort = true;
                        task.abortReason = AbortReason::NoPairedCandidatesAfterAlignment;
                    }
                    #endif
                }

                #endif

                cudaStreamSynchronize(firstStream); CUERR;
                cudaStreamSynchronize(secondStream); CUERR;

                cubAllocator->DeviceFree(cubtemp);

            }
    

            std::vector<Task> newTasksFromSplit;
            std::vector<int> newTaskIndices;


            auto constructMsaWithDataFromTask = [&](auto& task, int taskindex){
                const std::string& decodedAnchor = task.totalDecodedAnchors.back();

                MultipleSequenceAlignment msa;

                auto build = [&](){

                    task.candidateShifts.resize(task.numRemainingCandidates);
                    task.candidateOverlapWeights.resize(task.numRemainingCandidates);

                    //gather data required for msa
                    for(int c = 0; c < task.numRemainingCandidates; c++){
                        vecAccess(task.candidateShifts, c) = vecAccess(task.alignments, c).shift;

                        vecAccess(task.candidateOverlapWeights, c) = calculateOverlapWeight(
                            task.currentAnchorLength, 
                            vecAccess(task.alignments, c).nOps,
                            vecAccess(task.alignments, c).overlap,
                            goodAlignmentProperties.maxErrorRate
                        );
                    }

                    task.candidateStrings.resize(decodedSequencePitchInBytes * task.numRemainingCandidates, '\0');

                    //decode the candidates for msa
                    for(int c = 0; c < task.numRemainingCandidates; c++){
                        SequenceHelpers::decode2BitSequence(
                            task.candidateStrings.data() + c * decodedSequencePitchInBytes,
                            task.candidateSequenceData.data() + c * encodedSequencePitchInInts,
                            vecAccess(task.candidateSequenceLengths, c)
                        );
                    }

                    MultipleSequenceAlignment::InputData msaInput;
                    msaInput.useQualityScores = false;
                    msaInput.subjectLength = task.currentAnchorLength;
                    msaInput.nCandidates = task.numRemainingCandidates;
                    msaInput.candidatesPitch = decodedSequencePitchInBytes;
                    msaInput.candidateQualitiesPitch = 0;
                    msaInput.subject = decodedAnchor.c_str();
                    msaInput.candidates = task.candidateStrings.data();
                    msaInput.subjectQualities = nullptr;
                    msaInput.candidateQualities = nullptr;
                    msaInput.candidateLengths = task.candidateSequenceLengths.data();
                    msaInput.candidateShifts = task.candidateShifts.data();
                    msaInput.candidateDefaultWeightFactors = task.candidateOverlapWeights.data();                    

                    msa.build(msaInput);
                };

                build();

                #if 1

                constexpr int max_num_minimizations = 5;

                auto removeCandidatesOfDifferentRegion = [&](const auto& minimizationResult){
                    const int numCandidates = task.candidateReadIds.size();

                    int insertpos = 0;
                    for(int i = 0; i < numCandidates; i++){
                        if(!minimizationResult.differentRegionCandidate[i]){               
                            //keep candidate

                            task.candidateReadIds[insertpos] = task.candidateReadIds[i];

                            std::copy_n(
                                task.candidateSequenceData.data() + i * size_t(encodedSequencePitchInInts),
                                encodedSequencePitchInInts,
                                task.candidateSequenceData.data() + insertpos * size_t(encodedSequencePitchInInts)
                            );

                            task.candidateSequenceLengths[insertpos] = task.candidateSequenceLengths[i];
                            task.alignmentFlags[insertpos] = task.alignmentFlags[i];
                            task.alignments[insertpos] = task.alignments[i];
                            task.candidateOverlapWeights[insertpos] = task.candidateOverlapWeights[i];
                            task.candidateShifts[insertpos] = task.candidateShifts[i];

                            std::copy_n(
                                task.candidateStrings.data() + i * size_t(decodedSequencePitchInBytes),
                                decodedSequencePitchInBytes,
                                task.candidateStrings.data() + insertpos * size_t(decodedSequencePitchInBytes)
                            );

                            insertpos++;
                        }
                    }

                    task.numRemainingCandidates = insertpos;

                    task.candidateReadIds.erase(
                        task.candidateReadIds.begin() + insertpos, 
                        task.candidateReadIds.end()
                    );
                    task.candidateSequenceData.erase(
                        task.candidateSequenceData.begin() + encodedSequencePitchInInts * insertpos, 
                        task.candidateSequenceData.end()
                    );
                    task.candidateSequenceLengths.erase(
                        task.candidateSequenceLengths.begin() + insertpos, 
                        task.candidateSequenceLengths.end()
                    );
                    task.alignmentFlags.erase(
                        task.alignmentFlags.begin() + insertpos, 
                        task.alignmentFlags.end()
                    );
                    task.alignments.erase(
                        task.alignments.begin() + insertpos, 
                        task.alignments.end()
                    );

                    task.candidateStrings.erase(
                        task.candidateStrings.begin() + decodedSequencePitchInBytes * insertpos, 
                        task.candidateStrings.end()
                    );
                    task.candidateOverlapWeights.erase(
                        task.candidateOverlapWeights.begin() + insertpos, 
                        task.candidateOverlapWeights.end()
                    );
                    task.candidateShifts.erase(
                        task.candidateShifts.begin() + insertpos, 
                        task.candidateShifts.end()
                    );
                    
                };

                if(max_num_minimizations > 0){                

                    for(int numIterations = 0; numIterations < max_num_minimizations; numIterations++){
                        const auto minimizationResult = msa.findCandidatesOfDifferentRegion(
                            correctionOptions.estimatedCoverage
                        );

                        if(minimizationResult.performedMinimization){
                            removeCandidatesOfDifferentRegion(minimizationResult);

                            //build minimized multiple sequence alignment
                            build();
                        }else{
                            break;
                        }               
                        
                    }
                }   

                #endif

                return msa;
            };

            auto constructMsaWithDataFromBatchData = [&](auto& task, int taskindex){
                const std::string& decodedAnchor = task.totalDecodedAnchors.back();

                const int offset = batchData.h_numCandidatesPerAnchorPrefixSum[taskindex];

                MultipleSequenceAlignment msa;

                int* const myShifts = &batchData.h_alignment_shifts[offset];
                int* const myNops = &batchData.h_alignment_nOps[offset];
                int* const myOverlaps = &batchData.h_alignment_overlaps[offset];
                auto* const myAlignmentFlags = &batchData.h_alignment_best_alignment_flags[offset];
                unsigned int* const myCandidateSequencesData = &batchData.h_candidateSequencesData[offset * encodedSequencePitchInInts];
                int* const myCandidateSequencesLengths = &batchData.h_candidateSequencesLength[offset];
                read_number* const myCandidateReadIds = &batchData.h_candidateReadIds[offset];


                auto build = [&](){

                    task.candidateOverlapWeights.resize(task.numRemainingCandidates);

                    //gather data required for msa
                    for(int c = 0; c < task.numRemainingCandidates; c++){

                        vecAccess(task.candidateOverlapWeights, c) = calculateOverlapWeight(
                            task.currentAnchorLength, 
                            myNops[c],
                            myOverlaps[c],
                            goodAlignmentProperties.maxErrorRate
                        );
                    }

                    task.candidateStrings.resize(decodedSequencePitchInBytes * task.numRemainingCandidates, '\0');

                    //decode the candidates for msa
                    for(int c = 0; c < task.numRemainingCandidates; c++){
                        SequenceHelpers::decode2BitSequence(
                            task.candidateStrings.data() + c * decodedSequencePitchInBytes,
                            myCandidateSequencesData + c * encodedSequencePitchInInts,
                            myCandidateSequencesLengths[c]
                        );
                    }

                    MultipleSequenceAlignment::InputData msaInput;
                    msaInput.useQualityScores = false;
                    msaInput.subjectLength = task.currentAnchorLength;
                    msaInput.nCandidates = task.numRemainingCandidates;
                    msaInput.candidatesPitch = decodedSequencePitchInBytes;
                    msaInput.candidateQualitiesPitch = 0;
                    msaInput.subject = decodedAnchor.c_str();
                    msaInput.candidates = task.candidateStrings.data();
                    msaInput.subjectQualities = nullptr;
                    msaInput.candidateQualities = nullptr;
                    msaInput.candidateLengths = myCandidateSequencesLengths;
                    msaInput.candidateShifts = myShifts;
                    msaInput.candidateDefaultWeightFactors = task.candidateOverlapWeights.data();                    

                    msa.build(msaInput);
                };

                build();

                #if 1

                constexpr int max_num_minimizations = 5;

                auto removeCandidatesOfDifferentRegion = [&](const auto& minimizationResult){
                    const int numCandidates = batchData.h_numCandidatesPerAnchor[taskindex];

                    int insertpos = 0;
                    for(int i = 0; i < numCandidates; i++){
                        if(!minimizationResult.differentRegionCandidate[i]){               
                            //keep candidate

                            myCandidateReadIds[insertpos] = myCandidateReadIds[i];

                            std::copy_n(
                                myCandidateSequencesData + i * size_t(encodedSequencePitchInInts),
                                encodedSequencePitchInInts,
                                myCandidateSequencesData + insertpos * size_t(encodedSequencePitchInInts)
                            );

                            myCandidateSequencesLengths[insertpos] = myCandidateSequencesLengths[i];
                            myAlignmentFlags[insertpos] = myAlignmentFlags[i];
                            myShifts[insertpos] = myShifts[i];
                            myNops[insertpos] = myNops[i];
                            myOverlaps[insertpos] = myOverlaps[i];
                            task.candidateOverlapWeights[insertpos] = task.candidateOverlapWeights[i];
                            myShifts[insertpos] = myShifts[i];

                            std::copy_n(
                                task.candidateStrings.data() + i * size_t(decodedSequencePitchInBytes),
                                decodedSequencePitchInBytes,
                                task.candidateStrings.data() + insertpos * size_t(decodedSequencePitchInBytes)
                            );

                            insertpos++;
                        }
                    }

                    batchData.h_numCandidatesPerAnchor[taskindex] = insertpos;
                    task.numRemainingCandidates = insertpos;

                    task.candidateStrings.erase(
                        task.candidateStrings.begin() + decodedSequencePitchInBytes * insertpos, 
                        task.candidateStrings.end()
                    );
                    task.candidateOverlapWeights.erase(
                        task.candidateOverlapWeights.begin() + insertpos, 
                        task.candidateOverlapWeights.end()
                    );                    
                };

                if(max_num_minimizations > 0){                

                    for(int numIterations = 0; numIterations < max_num_minimizations; numIterations++){
                        const auto minimizationResult = msa.findCandidatesOfDifferentRegion(
                            correctionOptions.estimatedCoverage
                        );

                        if(minimizationResult.performedMinimization){
                            removeCandidatesOfDifferentRegion(minimizationResult);

                            //build minimized multiple sequence alignment
                            build();
                        }else{
                            break;
                        }               
                        
                    }
                }   

                #endif

                return msa;
            };

            auto constructMsa = [&](auto& task, int taskindex){
                if(task.dataIsAvailable){
                    return constructMsaWithDataFromTask(task, taskindex);
                }else{
                    return constructMsaWithDataFromBatchData(task, taskindex);
                }
            };

            auto extendWithMsa = [&](auto& task, const auto& msa, int taskIndex){

                int consensusLength = msa.consensus.size();
                //can extend by at most maxextensionPerStep bps
                int extendBy = std::min(
                    consensusLength - task.currentAnchorLength, 
                    maxextensionPerStep
                );
                //cannot extend over fragment 
                extendBy = std::min(extendBy, (insertSize + insertSizeStddev - task.mateLength) - task.accumExtensionLengths);

                auto makeAnchorForNextIteration = [&](){
                    if(extendBy == 0){
                        task.abort = true;
                        task.abortReason = AbortReason::MsaNotExtended;
                    }else{
                        task.accumExtensionLengths += extendBy;

                        //update data for next iteration of outer while loop                           

                        std::string decodedAnchor(msa.consensus.data() + extendBy, task.currentAnchorLength);

                        const int numInts = SequenceHelpers::getEncodedNumInts2Bit(task.currentAnchorLength);

                        task.currentAnchor.resize(numInts);

                        SequenceHelpers::encodeSequence2Bit(
                            task.currentAnchor.data(), 
                            decodedAnchor.data(), 
                            task.currentAnchorLength
                        );

                        task.totalDecodedAnchors.emplace_back(std::move(decodedAnchor));
                        task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);

                        // task.resultsequence.insert(
                        //     task.resultsequence.end(), 
                        //     msa.consensus.data() + task.currentAnchorLength, 
                        //     msa.consensus.data() + task.currentAnchorLength + extendBy
                        // );


                        // std::string tmp(task.currentAnchorLength, '\0');

                        // decode2BitSequence(
                        //     &tmp[0],
                        //     task.currentAnchor.data(),
                        //     task.currentAnchorLength
                        // );

                        // auto sub = task.resultsequence.substr(task.resultsequence.length() - task.currentAnchorLength);

                        // assert(sub == tmp);
                    }
                };

                constexpr int requiredOverlapMate = 70; //TODO relative overlap 
                constexpr int numMismatchesUpperBound = 2;

                if(task.pairedEnd && task.accumExtensionLengths + consensusLength - requiredOverlapMate + task.mateLength >= insertSize - insertSizeStddev){
                    //check if mate can be overlapped with consensus 

                    //hamMap[i] stores possible starting positions of overlaps which would have hamming distance i
                    std::map<int, std::vector<int>> hamMap;

                    //longmatchMap[i] stores possible starting positions of overlaps which would have a longest match of length i between mate and msa consensus
                    //std::map<int, std::vector<int>> longmatchMap; //map length of longest match to list start positions

                    //for each possibility to overlap the mate and consensus such that the merged sequence would end in the desired range [insertSize - insertSizeStddev, insertSize + insertSizeStddev]

                    const int firstStartpos = std::max(0, insertSize - insertSizeStddev - task.accumExtensionLengths - task.mateLength);
                    const int lastStartposExcl = std::min(
                        std::max(0, insertSize + insertSizeStddev - task.accumExtensionLengths - task.mateLength) + 1,
                        consensusLength - requiredOverlapMate
                    );

                    for(int startpos = firstStartpos; startpos < lastStartposExcl; startpos++){
                        //compute metrics of overlap
                            
                        const int ham = cpu::hammingDistanceOverlap(
                            msa.consensus.begin() + startpos, msa.consensus.end(), 
                            task.decodedMateRevC.begin(), task.decodedMateRevC.end()
                        );

                        hamMap[ham].emplace_back(startpos);

                        // const int longest = cpu::longestMatch(
                        //     msa.consensus.begin() + startpos, msa.consensus.end(), 
                        //     task.decodedMateRevC.begin(), task.decodedMateRevC.end()
                        // );

                        // longmatchMap[longest].emplace_back(startpos);
                    }
                    
                    std::vector<std::pair<int, std::vector<int>>> flatMap(hamMap.begin(), hamMap.end());
                    //sort by hamming distance, ascending
                    std::sort(flatMap.begin(), flatMap.end(), [](const auto& p1, const auto& p2){return p1.first < p2.first;});

                    //std::vector<std::pair<int, std::vector<int>>> flatMap2(longmatchMap.begin(), longmatchMap.end());
                    //sort by length of longest match, descending
                    //std::sort(flatMap2.begin(), flatMap2.end(), [](const auto& p1, const auto& p2){return p2.first < p1.first;});

                    //if there exists an overlap between msa consensus and mate which would end the merge, use the best one
                    if(flatMap.size() > 0 && flatMap[0].first <= numMismatchesUpperBound){
                    //if(flatMap2.size() > 0 && flatMap2[0].first >= 40){
                        const int mateStartposInConsensus = flatMap[0].second.front();
                        const int missingPositionsBetweenAnchorEndAndMateBegin = std::max(0, mateStartposInConsensus - task.currentAnchorLength);

                        if(missingPositionsBetweenAnchorEndAndMateBegin > 0){
                            //bridge the gap between current anchor and mate
                            task.totalDecodedAnchors.emplace_back(
                                msa.consensus.data() + missingPositionsBetweenAnchorEndAndMateBegin,
                                msa.consensus.data() + missingPositionsBetweenAnchorEndAndMateBegin + mateStartposInConsensus
                            );
                            task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths + missingPositionsBetweenAnchorEndAndMateBegin);
                        }


                        task.mateHasBeenFound = true;

                        //const int currentAccumExtensionLengths = task.accumExtensionLengths;
                        
                        task.accumExtensionLengths += mateStartposInConsensus;
                        std::string decodedAnchor(task.decodedMateRevC);

                        task.totalDecodedAnchors.emplace_back(std::move(decodedAnchor));
                        task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);

                        // const int startpos = mateStartposInConsensus;
                        // task.resultsequence.resize(currentAccumExtensionLengths + startpos + task.decodedMateRevC.length());
                        // const auto replaceBegin = task.resultsequence.begin() + currentAccumExtensionLengths + startpos;
                        // task.resultsequence.replace(
                        //     replaceBegin, 
                        //     replaceBegin + task.decodedMateRevC.length(), 
                        //     task.decodedMateRevC.begin(), 
                        //     task.decodedMateRevC.end()
                        // );

                    }else{
                        makeAnchorForNextIteration();
                    }
                }else{
                    makeAnchorForNextIteration();
                }
            };

            auto keepSelectedCandidates = [&](auto& task, const auto& selectedCandidateIndices, int taskIndex){
                assert(task.dataIsAvailable);

                const int numCandidateIndices = selectedCandidateIndices.size();
                assert(numCandidateIndices <= task.numRemainingCandidates);

                for(int i = 0; i < numCandidateIndices; i++){
                    const int c = vecAccess(selectedCandidateIndices, i);
                    // if(!(0 <= c && c < task.candidateReadIds.size())){
                    //     std::cerr << "c = " << c << ", candidateReadIds.size() = " << task.candidateReadIds.size() << "\n";
                    // }

                    // assert(0 <= c && c < task.candidateReadIds.size());
                    // assert(0 <= c && c < task.candidateSequenceLengths.size());
                    // assert(0 <= c && c < task.alignments.size());
                    // assert(0 <= c && c < task.alignmentFlags.size());

                    // assert(0 <= c && c*encodedSequencePitchInInts < task.candidateSequencesFwdData.size());
                    // assert(0 <= c && c*encodedSequencePitchInInts < task.candidateSequencesRevcData.size());
                    // assert(0 <= c && c*encodedSequencePitchInInts < task.candidateSequenceData.size());

                    vecAccess(task.candidateReadIds, i) = vecAccess(task.candidateReadIds, c);
                    vecAccess(task.candidateSequenceLengths , i) = vecAccess(task.candidateSequenceLengths, c);
                    vecAccess(task.alignments, i) = vecAccess(task.alignments, c);
                    vecAccess(task.alignmentFlags, i) = vecAccess(task.alignmentFlags, c);
                    vecAccess(task.candidateShifts, i) = vecAccess(task.candidateShifts, c);
                    vecAccess(task.candidateOverlapWeights, i) = vecAccess(task.candidateOverlapWeights, c);

                    // std::copy_n(
                    //     task.candidateSequencesFwdData.begin() + c * encodedSequencePitchInInts,
                    //     encodedSequencePitchInInts,
                    //     task.candidateSequencesFwdData.begin() + i * encodedSequencePitchInInts
                    // );

                    // std::copy_n(
                    //     task.candidateSequencesRevcData.begin() + c * encodedSequencePitchInInts,
                    //     encodedSequencePitchInInts,
                    //     task.candidateSequencesRevcData.begin() + i * encodedSequencePitchInInts
                    // );

                    std::copy_n(
                        task.candidateSequenceData.begin() + c * encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        task.candidateSequenceData.begin() + i * encodedSequencePitchInInts
                    );

                    std::copy_n(
                        task.candidateStrings.begin() + c * decodedSequencePitchInBytes,
                        decodedSequencePitchInBytes,
                        task.candidateStrings.begin() + i * decodedSequencePitchInBytes
                    );
                }

                task.candidateReadIds.erase(
                    task.candidateReadIds.begin() + numCandidateIndices,
                    task.candidateReadIds.end()
                );
                task.candidateSequenceLengths.erase(
                    task.candidateSequenceLengths.begin() + numCandidateIndices,
                    task.candidateSequenceLengths.end()
                );
                task.alignments.erase(
                    task.alignments.begin() + numCandidateIndices,
                    task.alignments.end()
                );
                task.alignmentFlags.erase(
                    task.alignmentFlags.begin() + numCandidateIndices,
                    task.alignmentFlags.end()
                );
                // task.candidateSequencesFwdData.erase(
                //     task.candidateSequencesFwdData.begin() + numCandidateIndices * encodedSequencePitchInInts,
                //     task.candidateSequencesFwdData.end()
                // );
                // task.candidateSequencesRevcData.erase(
                //     task.candidateSequencesRevcData.begin() + numCandidateIndices * encodedSequencePitchInInts,
                //     task.candidateSequencesRevcData.end()
                // );
                task.candidateSequenceData.erase(
                    task.candidateSequenceData.begin() + numCandidateIndices * encodedSequencePitchInInts,
                    task.candidateSequenceData.end()
                );
                if(task.pairedEnd){
                    task.mateIdLocationIter = std::lower_bound(
                        task.candidateReadIds.begin(),
                        task.candidateReadIds.end(),
                        task.mateReadId
                    );

                    task.mateHasBeenFound = (task.mateIdLocationIter != task.candidateReadIds.end() 
                        && *task.mateIdLocationIter == task.mateReadId);
                }
                task.numRemainingCandidates = numCandidateIndices;
            };

            nvtx::push_range("MSA", 6);
            msaTimer.start();

            for(int i = 0; i < numActiveTasks; i++){
                const int indexOfActiveTask = indicesOfActiveTasks[i];
                auto& task = vecAccess(tasks, indexOfActiveTask);

                if(task.numRemainingCandidates == 0){
                    continue;
                }
                assert(task.numRemainingCandidates > 0);

                //nvtx::push_range("constructMsa", 7);
                const MultipleSequenceAlignment msa = constructMsa(task, i);
                //nvtx::pop_range();

                // std::cerr << "original msa\n";
                // msa.print(std::cerr);
                // std::cerr << "\n";

                
#if 1
                //if(task.splitDepth == 0){
                if(splitTracker[task.myReadId] <= 4){
                    std::vector<MultipleSequenceAlignment::PossibleSplitColumn> possibleSplitColumns = computePossibleSplitColumns(
                        task.currentAnchorLength, 
                        msa.nColumns,
                        msa.countsA.data(),
                        msa.countsC.data(),
                        msa.countsG.data(),
                        msa.countsT.data(),
                        msa.coverage.data()
                    );

                    if(batchData.h_numPossibleSplitColumnsPerAnchor[i] == 0){
                        if(!(possibleSplitColumns.size() == 0 || possibleSplitColumns.size() > 32)){
                            std::cerr << "possibleSplitColumns.size() = " << possibleSplitColumns.size() << "\n";
                        }
                        assert(possibleSplitColumns.size() == 0 || possibleSplitColumns.size() > 32);
                    }else{
                        if(!(possibleSplitColumns.size() == batchData.h_numPossibleSplitColumnsPerAnchor[i])){
                            std::cerr << "possibleSplitColumns.size() = " << possibleSplitColumns.size() 
                                << "i = " << i << ", batchData.h_numPossibleSplitColumnsPerAnchor[i] = " << batchData.h_numPossibleSplitColumnsPerAnchor[i] << "\n";
                        }
                        assert(possibleSplitColumns.size() == batchData.h_numPossibleSplitColumnsPerAnchor[i]);

                        auto* ptr = batchData.h_possibleSplitColumns.data() + 32 * i;
                        for(int k = 0; k < batchData.h_numPossibleSplitColumnsPerAnchor[i]; k++){
                            assert(possibleSplitColumns[k] == ptr[k]);
                        }
                    }

                    assert(batchData.h_msa_column_properties[i].lastColumn_excl - batchData.h_msa_column_properties[i].firstColumn_incl == msa.consensus.size());

                    // for(int k = 0; k < int(msa.consensus.size()); k++){
                    //     assert(batchData.h_consensus[i * msaColumnPitchInElements + k] == msa.consensus[k]);
                    // }

                    auto possibleSplitsNew = inspectColumnsRegionSplit(
                        possibleSplitColumns.data(),
                        possibleSplitColumns.size(),
                        task.currentAnchorLength, 
                        msa.nColumns,
                        msa.subjectColumnsBegin_incl,
                        msa.nCandidates,
                        task.candidateStrings.data(),
                        decodedSequencePitchInBytes,
                        msa.inputData.candidateShifts,
                        msa.inputData.candidateLengths
                    );

                    //auto possibleSplitsOld = msa.inspectColumnsRegionSplit(task.currentAnchorLength);
                    //assert(possibleSplitsOld == possibleSplitsNew);

                    auto& possibleSplits = possibleSplitsNew;

                    if(possibleSplits.splits.size() > 1){
                        //nvtx::push_range("split msa", 8);
                        //auto& task = tasks[indexOfActiveTask];
                        #if 1
                        std::sort(
                            possibleSplits.splits.begin(), 
                            possibleSplits.splits.end(),
                            [](const auto& split1, const auto& split2){
                                //sort by size, descending
                                return split2.listOfCandidates.size() < split1.listOfCandidates.size();
                            }
                        );
                        #else
                        std::nth_element(
                            possibleSplits.splits.begin(), 
                            possibleSplits.splits.begin() + 1, 
                            possibleSplits.splits.end(),
                            [](const auto& split1, const auto& split2){
                                //sort by size, descending
                                return split2.listOfCandidates.size() < split1.listOfCandidates.size();
                            }
                        );
                        #endif

                        // std::cerr << "split[0] = ";
                        // for(auto x : possibleSplits.splits[0].listOfCandidates) std::cerr << x << " ";
                        // std::cerr << "\nsplit[1] = ";
                        // for(auto x : possibleSplits.splits[1].listOfCandidates) std::cerr << x << " ";
                        // std::cerr << "\n";

                        // auto printColumnInfo = [](const auto& x){
                        //     std::cerr << "(" << x.column << ", " << x.letter << ", " << x.ratio << ") ";
                        // };

                        // std::cerr << "columns[0] = ";
                        // for(auto x : possibleSplits.splits[0].columnInfo) printColumnInfo(x);
                        // std::cerr << "\ncolumns[1] = ";
                        // for(auto x : possibleSplits.splits[1].columnInfo) printColumnInfo(x);
                        // std::cerr << "\n";


                        //copy task's data from batchData into task

                        copyBatchDataIntoTask(task, i);
                        //create the separate shifts array
                        task.candidateShifts.resize(task.alignments.size());
                        for(int c = 0; c < task.numRemainingCandidates; c++){
                            task.candidateShifts[c] = task.alignments[c].shift;
                        }
                        task.dataIsAvailable = true;
                        //create a copy of task, and only keep candidates of first split
                        
                        Task taskCopy = task;
                        taskCopy.splitDepth++;

                        // std::cerr << "split\n";
                        // msa.print(std::cerr); 
                        // std::cerr << "\n into \n";

                        keepSelectedCandidates(taskCopy, possibleSplits.splits[0].listOfCandidates, -1);
                        const MultipleSequenceAlignment msaOfCopy = constructMsa(taskCopy, -1);

                        // msaOfCopy.print(std::cerr); 
                        // std::cerr << "\n and \n";

                        extendWithMsa(taskCopy, msaOfCopy, -1);

                        //only keep canddiates of second split
                        keepSelectedCandidates(task, possibleSplits.splits[1].listOfCandidates, -1);
                        const MultipleSequenceAlignment newMsa = constructMsa(task, -1);

                        // newMsa.print(std::cerr); 
                        // std::cerr << "\n";

                        extendWithMsa(task, newMsa, -1);

                        //if extension was not possible in task, replace task by task copy
                        if(task.abort && task.abortReason == AbortReason::MsaNotExtended){
                            //replace task by taskCopy
                            task = std::move(taskCopy);
                        }else if(!taskCopy.abort){
                            //if extension was possible in both task and taskCopy, taskCopy will be added to tasks and list of active tasks
                            newTaskIndices.emplace_back(tasks.size() + newTasksFromSplit.size());
                            newTasksFromSplit.emplace_back(std::move(taskCopy));

                            splitTracker[task.myReadId]++;


                        }
                        //nvtx::pop_range();                     
                    }else{
                        extendWithMsa(task, msa, indexOfActiveTask);
                    }
                }else{
                    extendWithMsa(task, msa, indexOfActiveTask);
                }
#else 
                extendWithMsa(task, msa, indexOfActiveTask);
#endif

            }

            msaTimer.stop();

            nvtx::pop_range();

            if(newTasksFromSplit.size() > 0){
                //std::cerr << "Added " << newTasksFromSplit.size() << " tasks\n";
                tasks.insert(tasks.end(), std::make_move_iterator(newTasksFromSplit.begin()), std::make_move_iterator(newTasksFromSplit.end()));
                indicesOfActiveTasks.insert(indicesOfActiveTasks.end(), newTaskIndices.begin(), newTaskIndices.end());

                indicesOfActiveTasksTmp.resize(indicesOfActiveTasks.size());
            }           

            /*
                update book-keeping of used candidates
            */  

            for(int i = 0; i < numActiveTasks; i++){
                auto& task = vecAccess(tasks, indicesOfActiveTasks[i]);

                                      
                {
                    if(task.dataIsAvailable){
                        std::vector<read_number> tmp(task.allUsedCandidateReadIdPairs.size() + task.candidateReadIds.size());
                        auto tmp_end = std::merge(
                            task.allUsedCandidateReadIdPairs.begin(),
                            task.allUsedCandidateReadIdPairs.end(),
                            task.candidateReadIds.begin(),
                            task.candidateReadIds.end(),
                            tmp.begin()
                        );

                        tmp.erase(tmp_end, tmp.end());

                        std::swap(task.allUsedCandidateReadIdPairs, tmp);
                    }else{
                        const int numCandidates = batchData.h_numCandidatesPerAnchor[i];
                        const int offset = batchData.h_numCandidatesPerAnchorPrefixSum[i];
                        const read_number* ids = &batchData.h_candidateReadIds[offset];

                        std::vector<read_number> tmp(task.allUsedCandidateReadIdPairs.size() + numCandidates);
                        auto tmp_end = std::merge(
                            task.allUsedCandidateReadIdPairs.begin(),
                            task.allUsedCandidateReadIdPairs.end(),
                            ids,
                            ids + numCandidates,
                            tmp.begin()
                        );

                        tmp.erase(tmp_end, tmp.end());

                        std::swap(task.allUsedCandidateReadIdPairs, tmp);
                    }
                }

                // task.usedCandidateReadIdsPerIteration.emplace_back(std::move(task.candidateReadIds));
                // task.usedAlignmentsPerIteration.emplace_back(std::move(task.alignments));
                // task.usedAlignmentFlagsPerIteration.emplace_back(std::move(task.alignmentFlags));

                task.iteration++;
            }
            
            //update list of active task indices

            indicesOfActiveTasks.erase(
                std::remove_if(
                    indicesOfActiveTasks.begin(), 
                    indicesOfActiveTasks.end(),
                    [&](int index){
                        return !tasks[index].isActive(insertSize, insertSizeStddev);
                    }
                ),
                indicesOfActiveTasks.end()
            );
        }

        //construct results

        for(const auto& task : tasks){

            ExtendResult extendResult;
            extendResult.direction = task.direction;
            extendResult.numIterations = task.iteration;
            extendResult.aborted = task.abort;
            extendResult.abortReason = task.abortReason;
            extendResult.readId1 = task.myReadId;
            extendResult.readId2 = task.mateReadId;
            extendResult.originalLength = task.myLength;

#if 0
            //extendResult.extendedRead = std::move(task.resultsequence);
            extendResult.success = true;
            extendResult.mateHasBeenFound = task.mateHasBeenFound;

#else
            // if(abort){
            //     ; //no read extension possible
            // }else
            {
                //if(mateHasBeenFound){
                {
                    //construct extended read
                    //build msa of all saved totalDecodedAnchors[0]

                    const int numsteps = task.totalDecodedAnchors.size();

                    // if(task.myReadId == 90 || task.mateReadId == 90){
                    //     std::cerr << "task.totalDecodedAnchors\n";
                    // }

                    int maxlen = 0;
                    for(const auto& s: task.totalDecodedAnchors){
                        const int len = s.length();
                        if(len > maxlen){
                            maxlen = len;
                        }

                        // if(task.myReadId == 90 || task.mateReadId == 90){
                        //     std::cerr << s << "\n";
                        // }
                    }

                    // if(task.myReadId == 90 || task.mateReadId == 90){
                    //     std::cerr << "\n";
                    // }

                    const std::string& decodedAnchor = vecAccess(task.totalDecodedAnchors, 0);

                    const std::vector<int> shifts(task.totalAnchorBeginInExtendedRead.begin() + 1, task.totalAnchorBeginInExtendedRead.end());
                    std::vector<float> initialWeights(numsteps-1, 1.0f);


                    std::vector<char> stepstrings(maxlen * (numsteps-1), '\0');
                    std::vector<int> stepstringlengths(numsteps-1);
                    for(int c = 1; c < numsteps; c++){
                        std::copy(
                            vecAccess(task.totalDecodedAnchors, c).begin(),
                            vecAccess(task.totalDecodedAnchors, c).end(),
                            stepstrings.begin() + (c-1) * maxlen
                        );
                        vecAccess(stepstringlengths, c-1) = vecAccess(task.totalDecodedAnchors, c).size();
                    }

                    MultipleSequenceAlignment::InputData msaInput;
                    msaInput.useQualityScores = false;
                    msaInput.subjectLength = decodedAnchor.length();
                    msaInput.nCandidates = numsteps-1;
                    msaInput.candidatesPitch = maxlen;
                    msaInput.candidateQualitiesPitch = 0;
                    msaInput.subject = decodedAnchor.c_str();
                    msaInput.candidates = stepstrings.data();
                    msaInput.subjectQualities = nullptr;
                    msaInput.candidateQualities = nullptr;
                    msaInput.candidateLengths = stepstringlengths.data();
                    msaInput.candidateShifts = shifts.data();
                    msaInput.candidateDefaultWeightFactors = initialWeights.data();

                    MultipleSequenceAlignment msa;

                    msa.build(msaInput);

                    // if(task.myReadId == 90 || task.mateReadId == 90){
                    //     std::cerr << "Id " << task.myReadId << ", Final\n";
                    //     msa.print(std::cerr);
                    //     std::cerr << "\n";
                    // }

                    extendResult.success = true;

                    std::string extendedRead(msa.consensus.begin(), msa.consensus.end());
                    //std::cerr << "before: " << extendedRead << "\n";
                    std::copy(decodedAnchor.begin(), decodedAnchor.end(), extendedRead.begin());
                    if(task.mateHasBeenFound){
                        std::copy(
                            task.decodedMateRevC.begin(),
                            task.decodedMateRevC.end(),
                            extendedRead.begin() + extendedRead.length() - task.decodedMateRevC.length()
                        );
                    }
                    // extendedRead.replace(extendedRead.begin(), extendedRead.begin() + decodedAnchor, decodedAnchor.begin(), decodedAnchor.end());
                    // std::cerr << "after : " << extendedRead << "\n";
                    
                    // msa.print(std::cerr);
                    // std::cerr << "msa cons:\n";
                    // std::cerr << extendedRead << "\n";
                    // std::cerr << "new cons:\n";
                    // std::cerr << task.resultsequence << "\n";


                    extendResult.extendedRead = std::move(extendedRead);

                    extendResult.mateHasBeenFound = task.mateHasBeenFound;
                }
                // else{
                //     ; //no read extension possible
                // }
            }

            // if(extendResult.extendedRead.length() != task.resultsequence.length()){
            //     std::cerr << task.myReadId << "\n";
            //     std::cerr << extendResult.extendedRead << "\n";
            //     std::cerr << task.resultsequence << "\n";
            //     std::exit(0);
            // }
#endif
            extendResults.emplace_back(std::move(extendResult));

        }

        return extendResults;
    }


    std::vector<ReadExtenderBase::ExtendResult> ReadExtenderGpu::processSingleEndTasks(
        std::vector<ReadExtenderBase::Task>& tasks
    ){
        return processPairedEndTasks(tasks);
    }


    void ReadExtenderGpu::getCandidateReadIds(BatchData& batchData, cudaStream_t stream) const{
        nvtx::push_range("gpu_hashing", 2);

        int totalNumValues = 0;

        gpuMinhasher->determineNumValues(
            minhashHandle,
            batchData.d_subjectSequencesData.get(),
            encodedSequencePitchInInts,
            batchData.d_anchorSequencesLength.get(),
            batchData.numTasks,
            batchData.d_numCandidatesPerAnchor.get(),
            totalNumValues,
            stream
        );

        cudaStreamSynchronize(stream); CUERR;

        batchData.d_candidateReadIds.resize(totalNumValues);        

        if(totalNumValues == 0){
            cudaMemsetAsync(batchData.d_numCandidatesPerAnchor.get(), 0, sizeof(int) * batchData.numTasks , stream); CUERR;
            cudaMemsetAsync(batchData.d_numCandidatesPerAnchorPrefixSum.get(), 0, sizeof(int) * (1 + batchData.numTasks), stream); CUERR;
            return;
        }

        gpuMinhasher->retrieveValues(
            minhashHandle,
            nullptr,
            batchData.numTasks,              
            totalNumValues,
            batchData.d_candidateReadIds.get(),
            batchData.d_numCandidatesPerAnchor.get(),
            batchData.d_numCandidatesPerAnchorPrefixSum.get(),
            stream
        );

        nvtx::pop_range();
    }


    void ReadExtenderGpu::loadCandidateSequenceData(BatchData& batchData, cudaStream_t stream) const{

        nvtx::push_range("gpu_loadCandidates", 2);

        const int totalNumCandidates = batchData.totalNumCandidates;

        gpuReadStorage->gatherSequences(
            readStorageHandle,
            batchData.d_candidateSequencesData.get(),
            encodedSequencePitchInInts,
            batchData.h_candidateReadIds.get(),
            batchData.d_candidateReadIds.get(), //device accessible
            totalNumCandidates,
            stream
        );

        gpuReadStorage->gatherSequenceLengths(
            readStorageHandle,
            batchData.d_candidateSequencesLength.get(),
            batchData.d_candidateReadIds.get(),
            totalNumCandidates,
            stream
        );

        readextendergpukernels::reverseComplement2bitKernel<128><<<320,128,0,stream>>>(
            batchData.d_candidateSequencesLength.get(),
            batchData.d_candidateSequencesData.get(),
            batchData.d_candidateSequencesRevcData.get(),
            totalNumCandidates,
            encodedSequencePitchInInts
        ); CUERR;

        nvtx::pop_range();
    }


    void ReadExtenderGpu::eraseDataOfRemovedMates(BatchData& batchData, cudaStream_t stream) const{
        nvtx::push_range("gpu_eraseDataOfRemovedMates", 3);

        auto vecAccess = [](auto& vec, auto index) -> decltype(vec.at(index)){
            return vec.at(index);
        };

        if(batchData.numTasksWithMateRemoved > 0){
            const int totalNumCandidates = batchData.totalNumCandidates;

            constexpr int groupsize = 32;
            constexpr int blocksize = 128;
            constexpr int groupsperblock = blocksize / groupsize;
            dim3 block(blocksize,1,1);
            dim3 grid(SDIV(batchData.numTasksWithMateRemoved * groupsize, blocksize), 1, 1);
            const std::size_t smembytes = sizeof(unsigned int) * groupsperblock * encodedSequencePitchInInts;

            helpers::call_fill_kernel_async(batchData.d_flagscandidates.data(), batchData.d_flagscandidates.size(), false, stream);

            readextendergpukernels::filtermatekernel<blocksize,groupsize><<<grid, block, smembytes, stream>>>(
                batchData.d_anchormatedata.data(),
                batchData.d_candidateSequencesData.data(),
                encodedSequencePitchInInts,
                batchData.d_numCandidatesPerAnchor.data(),
                batchData.d_numCandidatesPerAnchorPrefixSum.data(),
                batchData.d_indexlist1.data(),
                batchData.numTasksWithMateRemoved,
                batchData.d_flagscandidates.data()
            ); CUERR;

            auto negate = [] __device__ (bool b){
                return !b;
            };

            cub::TransformInputIterator<bool, decltype(negate), bool*> d_keepflags(batchData.d_flagscandidates.data(), negate);

            std::size_t requiredCubSize = 0;
            std::size_t requiredCubSize1 = 0;
            std::size_t requiredCubSize2 = 0;
            cudaError_t cubstatus = cub::DeviceScan::ExclusiveSum(
                nullptr,
                requiredCubSize1,
                d_keepflags, 
                batchData.d_intbuffercandidates.data(), 
                totalNumCandidates, 
                stream
            );
            assert(cudaSuccess == cubstatus);

            cubstatus = cub::DeviceScan::InclusiveSum(
                nullptr,
                requiredCubSize2,
                batchData.d_numCandidatesPerAnchor.data(), 
                batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1, 
                batchData.numTasks, 
                stream
            );
            assert(cudaSuccess == cubstatus);

            requiredCubSize = std::max(requiredCubSize1, requiredCubSize2);

            void* cubtemp; cubAllocator->DeviceAllocate((void**)&cubtemp, requiredCubSize, stream);

            cubstatus = cub::DeviceScan::ExclusiveSum(
                cubtemp,
                requiredCubSize,
                d_keepflags, 
                batchData.d_intbuffercandidates.data(), 
                totalNumCandidates, 
                stream
            );
            assert(cudaSuccess == cubstatus);

            helpers::lambda_kernel<<<4096, 128, 0, stream>>>(
                [
                    numTasks = batchData.numTasks,
                    encodedSequencePitchInInts = encodedSequencePitchInInts,
                    d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.data(),
                    d_numCandidatesPerAnchorPrefixSum = batchData.d_numCandidatesPerAnchorPrefixSum.data(),
                    d_removalflags = batchData.d_flagscandidates.data(),
                    d_outputpositions = batchData.d_intbuffercandidates.data(),
                    d_candidateReadIds = batchData.d_candidateReadIds.data(),
                    d_candidateSequencesLength = batchData.d_candidateSequencesLength.data(),
                    d_candidateSequencesData = batchData.d_candidateSequencesData.data(),
                    d_candidateSequencesRevcData = batchData.d_candidateSequencesRevcData.data(),
                    d_anchorIndicesOfCandidates = batchData.d_anchorIndicesOfCandidates.data(),
                    d_candidateReadIdsOut = batchData.d_candidateReadIds2.data(),
                    d_candidateSequencesLengthOut = batchData.d_candidateSequencesLength2.data(),
                    d_candidateSequencesDataOut = batchData.d_candidateSequencesData2.data(),
                    d_candidateSequencesRevcDataOut = batchData.d_candidateSequencesRevcData2.data(),
                    d_anchorIndicesOfCandidatesOut = batchData.d_anchorIndicesOfCandidates2.data()
                ] __device__ (){

                    constexpr int elementsPerIteration = 128;
                    __shared__ bool smem_removalflags[elementsPerIteration];
                    __shared__ int smem_outputpositions[elementsPerIteration];

                    using BlockReduce = cub::BlockReduce<int, elementsPerIteration>;
                    __shared__ typename BlockReduce::TempStorage temp_storage;

                    auto group = cg::tiled_partition<8>(cg::this_thread_block());
                    const int numGroupsInBlock = blockDim.x / 8;
                    const int groupInBlock = threadIdx.x / 8;

                    for(int t = blockIdx.x; t < numTasks; t += gridDim.x){
                        const int numCandidates = d_numCandidatesPerAnchor[t];
                        const int inputOffset = d_numCandidatesPerAnchorPrefixSum[t];

                        int numSelected = 0;

                        for(int i = threadIdx.x; i < numCandidates; i += blockDim.x){
                            if(!d_removalflags[inputOffset + i]){
                                const int outputLocation = d_outputpositions[inputOffset + i];

                                d_candidateReadIdsOut[outputLocation] = d_candidateReadIds[inputOffset + i];
                                d_candidateSequencesLengthOut[outputLocation] = d_candidateSequencesLength[inputOffset + i];
                                d_anchorIndicesOfCandidatesOut[outputLocation] = d_anchorIndicesOfCandidates[inputOffset + i];

                                numSelected++;
                            }
                        }

                        for(int i = threadIdx.x; i < numCandidates * encodedSequencePitchInInts; i += blockDim.x){
                            const int which = i / encodedSequencePitchInInts;
                            const int what = i % encodedSequencePitchInInts;

                            if(!d_removalflags[inputOffset + which]){
                                d_candidateSequencesDataOut[d_outputpositions[inputOffset + which] * encodedSequencePitchInInts + what] = d_candidateSequencesData[(inputOffset + which) * encodedSequencePitchInInts + what];
                                d_candidateSequencesRevcDataOut[d_outputpositions[inputOffset + which] * encodedSequencePitchInInts + what] = d_candidateSequencesRevcData[(inputOffset + which) * encodedSequencePitchInInts + what];
                            }
                        }

                        numSelected = BlockReduce(temp_storage).Sum(numSelected);
                        __syncthreads();
                        
                        if(threadIdx.x == 0){
                            if(numSelected != numCandidates){
                                assert(numSelected < numCandidates);
                                d_numCandidatesPerAnchor[t] = numSelected;
                                //printf("task %d, removed %d\n", t, numCandidates - numSelected);
                            }
                        }

                    }
                }
            ); CUERR;

            //update prefix sum

            cubstatus = cub::DeviceScan::InclusiveSum(
                cubtemp,
                requiredCubSize,
                batchData.d_numCandidatesPerAnchor.data(), 
                batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1, 
                batchData.numTasks, 
                stream
            );
            assert(cudaSuccess == cubstatus);

            cubAllocator->DeviceFree(cubtemp);

            std::swap(batchData.d_candidateReadIds2, batchData.d_candidateReadIds);
            std::swap(batchData.d_candidateSequencesLength2, batchData.d_candidateSequencesLength);
            std::swap(batchData.d_candidateSequencesData2, batchData.d_candidateSequencesData);
            std::swap(batchData.d_candidateSequencesRevcData2, batchData.d_candidateSequencesRevcData);
            std::swap(batchData.d_anchorIndicesOfCandidates2, batchData.d_anchorIndicesOfCandidates);

        }

        nvtx::pop_range();
       
    }


    void ReadExtenderGpu::calculateAlignments(BatchData& batchData, cudaStream_t stream) const{
        nvtx::push_range("gpu_alignment", 4);

        
        batchData.h_numAnchors[0] = batchData.numTasks;

        const bool* const d_anchorContainsN = nullptr;
        const bool* const d_candidateContainsN = nullptr;
        const bool removeAmbiguousAnchors = false;
        const bool removeAmbiguousCandidates = false;
        const int maxNumAnchors = batchData.numTasks;
        const int maxNumCandidates = batchData.totalNumCandidates;
        const int maximumSequenceLength = 100; //encodedSequencePitchInInts * 16;
        const int encodedSequencePitchInInts2Bit = encodedSequencePitchInInts;
        const int min_overlap = goodAlignmentProperties.min_overlap;
        const float maxErrorRate = goodAlignmentProperties.maxErrorRate;
        const float min_overlap_ratio = goodAlignmentProperties.min_overlap_ratio;
        const float estimatedNucleotideErrorRate = correctionOptions.estimatedErrorrate;

        auto callAlignmentKernel = [&](void* d_tempstorage, size_t& tempstoragebytes){

            call_popcount_rightshifted_hamming_distance_kernel_async(
                d_tempstorage,
                tempstoragebytes,
                batchData.d_alignment_overlaps.get(),
                batchData.d_alignment_shifts.get(),
                batchData.d_alignment_nOps.get(),
                batchData.d_alignment_isValid.get(),
                batchData.d_alignment_best_alignment_flags.get(),
                batchData.d_subjectSequencesData.get(),
                batchData.d_candidateSequencesData.get(),
                batchData.d_anchorSequencesLength.get(),
                batchData.d_candidateSequencesLength.get(),
                batchData.d_numCandidatesPerAnchorPrefixSum.get(),
                batchData.d_numCandidatesPerAnchor.get(),
                batchData.d_anchorIndicesOfCandidates.get(),
                //batchData.d_intbuffercandidates.get(),
                batchData.h_numAnchors.get(),
                &batchData.d_numCandidatesPerAnchorPrefixSum[batchData.numTasks],
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

        const int num = batchData.totalNumCandidates;

        // assert(batchData.d_intbuffercandidates.size() >= num);
        // assert(batchData.h_numAnchors[0] == batchData.numTasks);
        // assert(batchData.d_numCandidatesPerAnchor.size() >= batchData.numTasks);
        // assert(batchData.d_numCandidatesPerAnchorPrefixSum.size() >= batchData.numTasks+1);

        // readextendergpukernels::setAnchorIndicesOfCandidateskernel<<<1024, 128, 0, stream>>>(
        //     batchData.d_intbuffercandidates.data(),
        //     batchData.h_numAnchors.data(),
        //     batchData.d_numCandidatesPerAnchor.get(),
        //     batchData.d_numCandidatesPerAnchorPrefixSum.get()
        // );

        size_t tempstoragebytes = 0;
        callAlignmentKernel(nullptr, tempstoragebytes);

        batchData.d_tempstorage.resize(tempstoragebytes);

        callAlignmentKernel(batchData.d_tempstorage.get(), tempstoragebytes);

        nvtx::pop_range();
    }



    void ReadExtenderGpu::filterAlignments(BatchData& batchData, cudaStream_t stream) const{
        nvtx::push_range("gpu_filterAlignments", 5);

        const int totalNumCandidates = batchData.totalNumCandidates;
        const int numAnchors = batchData.numTasks;

        helpers::call_fill_kernel_async(batchData.d_flagscandidates.data(), batchData.d_flagscandidates.size(), true, stream);

        bool* const d_keepflags = batchData.d_flagscandidates.data();

        dim3 block(128,1,1);
        dim3 grid(numAnchors, 1, 1);

        //filter alignments of candidates. d_keepflags[i] will be set to false if candidate[i] should be removed
        //batchData.d_numCandidatesPerAnchor2[i] contains new number of candidates for anchor i
        helpers::lambda_kernel<<<grid, block, 0, stream>>>(
            [
                d_alignment_best_alignment_flags = batchData.d_alignment_best_alignment_flags.data(),
                d_alignment_shifts = batchData.d_alignment_shifts.data(),
                d_alignment_overlaps = batchData.d_alignment_overlaps.data(),
                d_anchorSequencesLength = batchData.d_anchorSequencesLength.data(),
                d_numCandidatesPerAnchor = batchData.d_numCandidatesPerAnchor.data(),
                d_numCandidatesPerAnchor2 = batchData.d_numCandidatesPerAnchor2.data(),
                d_numCandidatesPerAnchorPrefixSum = batchData.d_numCandidatesPerAnchorPrefixSum.data(),
                d_keepflags,
                min_overlap_ratio = goodAlignmentProperties.min_overlap_ratio,
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
                            bool goodAlignmentExists = false;
                            const float overlap = d_alignment_overlaps[offset + c];                            
                            const float relativeOverlap = overlap / anchorLength;
                            
                            if(relativeOverlap < 1.0f && fgeq(relativeOverlap, min_overlap_ratio)){
                                threadReducedGoodAlignmentExists = 1;
                                const float tmp = floorf(relativeOverlap * 10.0f) / 10.0f;
                                threadReducedRelativeOverlapThreshold = fmaxf(threadReducedRelativeOverlapThreshold, tmp);
                            }

                            // while(!goodAlignmentExists && fgeq(relativeOverlapThreshold, min_overlap_ratio)){

                            //     goodAlignmentExists = fgeq(relativeOverlap, relativeOverlapThreshold) && relativeOverlap < 1.0f;

                            //     if(!goodAlignmentExists){
                            //         relativeOverlapThreshold -= 0.1f;
                            //     }
                            // }

                            // if(goodAlignmentExists){
                            //     threadReducedGoodAlignmentExists = 1;
                            //     threadReducedRelativeOverlapThreshold = max(threadReducedRelativeOverlapThreshold, relativeOverlapThreshold);
                            // }

                            // if(a == 1){
                            //     printf("a %d c %d relativeOverlap %f, thread good %d, thresh %f\n", 
                            //         a, c, relativeOverlap, goodAlignmentExists, threadReducedRelativeOverlapThreshold);
                            // }
                        }else{
                            //remove alignment with negative shift
                            d_keepflags[offset + c] = false;
                            removed++;
                        }

                        
                    }
                    // __syncthreads(); //debug
                    // if(threadIdx.x < num){
                    //     printf("a %d thread good %d, thresh %f\n", a, threadReducedGoodAlignmentExists, threadReducedRelativeOverlapThreshold);
                    // }
                    // __syncthreads(); //debug

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
    
                            if(d_keepflags[offset + c]){
                                const float overlap = d_alignment_overlaps[offset + c];                            
                                const float relativeOverlap = overlap / anchorLength;                 
    
                                if(!fgeq(relativeOverlap, blockreducedRelativeOverlapThreshold)){
                                    d_keepflags[offset + c] = false;
                                    removed++;
                                }
                            }
                        }
                    }else{
                        //NOOP.
                        //if no good alignment exists, no candidate is removed. we will try to work with the not-so-good alignments
                    }

                    removed = BlockReduceInt(cubtemp.intreduce).Sum(removed);

                    if(threadIdx.x == 0){
                        d_numCandidatesPerAnchor2[a] = num - removed;
                        //printf("task %d remaining: %d - %d = %d\n", a, num, removed, num - removed);
                    }
                    __syncthreads();
                }
            }
        ); CUERR;

        //setup cub 
        auto d_zip_input = thrust::make_zip_iterator(
            thrust::make_tuple(
                batchData.d_alignment_nOps.data(),
                batchData.d_alignment_overlaps.data(),
                batchData.d_alignment_shifts.data(),
                batchData.d_alignment_isValid.data(),
                batchData.d_alignment_best_alignment_flags.data(),
                batchData.d_candidateReadIds.data(),
                batchData.d_candidateSequencesLength.data()
            )
        );

        auto d_zip_output = thrust::make_zip_iterator(
            thrust::make_tuple(
                batchData.d_alignment_nOps2.data(),
                batchData.d_alignment_overlaps2.data(),
                batchData.d_alignment_shifts2.data(),
                batchData.d_alignment_isValid2.data(),
                batchData.d_alignment_best_alignment_flags2.data(),
                batchData.d_candidateReadIds2.data(),
                batchData.d_candidateSequencesLength2.data()
            )
        );

        std::size_t requiredCubSize1 = 0;
        cudaError_t cubstatus = cub::DeviceSelect::Flagged(
            nullptr, 
            requiredCubSize1, 
            d_zip_input, 
            d_keepflags, 
            d_zip_output, 
            batchData.d_numCandidates.data(), 
            totalNumCandidates, 
            stream
        );
        assert(cubstatus == cudaSuccess);

        std::size_t requiredCubSize2 = 0;
        cubstatus = cub::DeviceScan::ExclusiveSum(
            nullptr,
            requiredCubSize2,
            d_keepflags, 
            batchData.d_intbuffercandidates.data(), 
            totalNumCandidates, 
            stream
        );
        assert(cubstatus == cudaSuccess);

        std::size_t requiredCubSize3 = 0;
        cubstatus = cub::DeviceScan::InclusiveSum(
            nullptr,
            requiredCubSize3,
            batchData.d_numCandidatesPerAnchor2.data(), 
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            batchData.numTasks, 
            stream
        );
        assert(cudaSuccess == cubstatus);

        std::size_t requiredCubSize = std::max(std::max(requiredCubSize1, requiredCubSize2), requiredCubSize3);
        void* cubtemp; cubAllocator->DeviceAllocate((void**)&cubtemp, requiredCubSize, stream);
        //batchData.d_tempstorage.resize(requiredCubSize);

        //compute output positions for selected candidates

        cubstatus = cub::DeviceScan::ExclusiveSum(
            cubtemp, 
            requiredCubSize,
            d_keepflags, 
            batchData.d_intbuffercandidates.data(), 
            totalNumCandidates, 
            stream
        );
        assert(cubstatus == cudaSuccess);

        //compact zip data
        cubstatus = cub::DeviceSelect::Flagged(
            cubtemp, 
            requiredCubSize, 
            d_zip_input, 
            d_keepflags, 
            d_zip_output, 
            batchData.d_numCandidates.data(), 
            totalNumCandidates, 
            stream
        );
        assert(cubstatus == cudaSuccess);

        //compact sequence data. if alignmentflag is forward, forward sequence data will be copied, 
        //else reverse complement will be copied
        helpers::lambda_kernel<<<4096, 128, 0, stream>>>(
            [
                encodedSequencePitchInInts = encodedSequencePitchInInts,
                d_keepflags,
                totalNumCandidates,
                d_outputpositions = batchData.d_intbuffercandidates.data(),
                d_alignment_best_alignment_flags = batchData.d_alignment_best_alignment_flags.data(),
                d_candidateSequencesData = batchData.d_candidateSequencesData.data(),
                d_candidateSequencesRevcData = batchData.d_candidateSequencesRevcData.data(),
                d_candidateSequencesDataOut = batchData.d_candidateSequencesData2.data()
            ] __device__ (){

                const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                const int stride = blockDim.x * gridDim.x;
                const int elements = totalNumCandidates * encodedSequencePitchInInts;

                for(int i = tid; i < elements; i += stride){
                    const int which = i / encodedSequencePitchInInts;
                    const int what = i % encodedSequencePitchInInts;

                    if(d_keepflags[which]){

                        const int outputindex = d_outputpositions[which] * encodedSequencePitchInInts + what;
                        const int inputindex = which * encodedSequencePitchInInts + what;

                        const auto alignmentflag = d_alignment_best_alignment_flags[which];
                        
                        if(alignmentflag == BestAlignment_t::Forward){                             
                            d_candidateSequencesDataOut[outputindex] = d_candidateSequencesData[inputindex];
                        }else{
                            d_candidateSequencesDataOut[outputindex] = d_candidateSequencesRevcData[inputindex];
                        }
                    }
                }
            }
        ); CUERR;


        //update prefix sum
        cubstatus = cub::DeviceScan::InclusiveSum(
            cubtemp, 
            requiredCubSize, 
            batchData.d_numCandidatesPerAnchor2.data(), 
            batchData.d_numCandidatesPerAnchorPrefixSum.data() + 1, 
            batchData.numTasks, 
            stream
        );
        assert(cudaSuccess == cubstatus);

        cubAllocator->DeviceFree(cubtemp);

        std::swap(batchData.d_alignment_nOps2, batchData.d_alignment_nOps);
        std::swap(batchData.d_alignment_overlaps2, batchData.d_alignment_overlaps);
        std::swap(batchData.d_alignment_shifts2, batchData.d_alignment_shifts);
        std::swap(batchData.d_alignment_isValid2, batchData.d_alignment_isValid);
        std::swap(batchData.d_alignment_best_alignment_flags2, batchData.d_alignment_best_alignment_flags);
        std::swap(batchData.d_candidateReadIds2, batchData.d_candidateReadIds);
        std::swap(batchData.d_candidateSequencesLength2, batchData.d_candidateSequencesLength);
        std::swap(batchData.d_numCandidatesPerAnchor2, batchData.d_numCandidatesPerAnchor);
        std::swap(batchData.d_candidateSequencesData2, batchData.d_candidateSequencesData);

        nvtx::pop_range();
    }




}