#include <readextender.hpp>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>
#include <cassert>
#include <iterator>

namespace care{

    std::vector<ReadExtenderBase::ExtendResult> ReadExtenderBase::processPairedEndTasks(
        std::vector<ReadExtenderBase::Task>& tasks
    ){
 
        std::vector<ExtendResult> extendResults;

        std::vector<int> indicesOfActiveTasks(tasks.size());
        std::vector<int> indicesOfActiveTasksTmp(tasks.size());
        std::iota(indicesOfActiveTasks.begin(), indicesOfActiveTasks.end(), 0);

        while(indicesOfActiveTasks.size() > 0){
            //perform one extension iteration for active tasks

            for(int indexOfActiveTask : indicesOfActiveTasks){
                auto& task = tasks[indexOfActiveTask];

                //update "total" arrays
                {
                    std::string decodedAnchor(task.currentAnchorLength, '\0');

                    decode2BitSequence(
                        &decodedAnchor[0],
                        task.currentAnchor.data(),
                        task.currentAnchorLength
                    );

                    // if(task.myReadId == 90 || task.mateReadId == 90){
                    //     std::cerr << "Id " << task.myReadId << ", Iteration: " << task.iteration << "\n";
                    //     std::cerr << "task.totalDecodedAnchors.emplace_back " << decodedAnchor << "\n";
                    // }

                    task.totalDecodedAnchors.emplace_back(std::move(decodedAnchor));
                    task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);
                } 
            }



            hashTimer.start();

            getCandidateReadIds(tasks, indicesOfActiveTasks);

            for(int indexOfActiveTask : indicesOfActiveTasks){
                auto& task = tasks[indexOfActiveTask];

                // remove self from candidate list
                auto readIdPos = std::lower_bound(
                    task.candidateReadIds.begin(),                                            
                    task.candidateReadIds.end(),
                    task.myReadId
                );

                if(readIdPos != task.candidateReadIds.end() && *readIdPos == task.myReadId){
                    task.candidateReadIds.erase(readIdPos);
                }

                //remove mate of input from candidate list if it is not possible that mate could be reached at the current iteration
                if(task.mateLength + task.accumExtensionLengths < insertSize - insertSizeStddev){
                    auto readIdPos = std::lower_bound(
                        task.candidateReadIds.begin(),                                            
                        task.candidateReadIds.end(),
                        task.mateReadId
                    );

                    if(readIdPos != task.candidateReadIds.end() && *readIdPos == task.mateReadId){
                        task.candidateReadIds.erase(readIdPos);
                    }
                }
            }

            hashTimer.stop();
                

            collectTimer.start();

            /*
                Remove candidate pairs which have already been used for extension
            */

            for(int indexOfActiveTask : indicesOfActiveTasks){
                auto& task = tasks[indexOfActiveTask];

                
                {

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
            }

            loadCandidateSequenceData(tasks, indicesOfActiveTasks);

            /*
                Compute reverse complement of candidates
            */

            for(int indexOfActiveTask : indicesOfActiveTasks){
                auto& task = tasks[indexOfActiveTask];

                

                const int numCandidates = task.candidateReadIds.size();

                task.candidateSequencesRevcData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);

                for(int c = 0; c < numCandidates; c++){
                    const unsigned int* const seqPtr = task.candidateSequencesFwdData.data() 
                                                        + std::size_t(encodedSequencePitchInInts) * c;
                    unsigned int* const seqrevcPtr = task.candidateSequencesRevcData.data() 
                                                        + std::size_t(encodedSequencePitchInInts) * c;

                    reverseComplement2Bit(
                        seqrevcPtr,  
                        seqPtr,
                        task.candidateSequenceLengths[c]
                    );
                }
            }

            collectTimer.stop();

            /*
                Compute alignments
            */

            alignmentTimer.start();

            calculateAlignments(tasks, indicesOfActiveTasks);

            alignmentTimer.stop();

            alignmentFilterTimer.start();

            for(int indexOfActiveTask : indicesOfActiveTasks){
                auto& task = tasks[indexOfActiveTask];

                /*
                    Remove bad alignments and the corresponding alignments of their mate
                */        

                const int size = task.alignments.size();

                std::vector<int> positionsOfCandidatesToKeep(size);
                std::vector<int> tmpPositionsOfCandidatesToKeep(size);

                task.numRemainingCandidates = 0;

                //select candidates with good alignment and positive shift
                for(int c = 0; c < size; c++){
                    const BestAlignment_t alignmentFlag0 = task.alignmentFlags[c];
                    
                    if(alignmentFlag0 != BestAlignment_t::None && task.alignments[c].shift >= 0){
                        positionsOfCandidatesToKeep[task.numRemainingCandidates] = c;
                        task.numRemainingCandidates++;
                    }else{
                        ; //if any of the mates aligns badly, remove both of them
                    }
                }

                positionsOfCandidatesToKeep.erase(
                    positionsOfCandidatesToKeep.begin() + task.numRemainingCandidates, 
                    positionsOfCandidatesToKeep.end()
                );

                if(task.numRemainingCandidates == 0){
                    task.abort = true;
                    task.abortReason = AbortReason::NoPairedCandidatesAfterAlignment;

                    continue; //stop processing task
                }

                float relativeOverlapThreshold = 0.9f;
                bool goodAlignmentExists = false;

                while(!goodAlignmentExists && fgeq(relativeOverlapThreshold, goodAlignmentProperties.min_overlap_ratio)){                    

                    goodAlignmentExists = std::any_of(
                        positionsOfCandidatesToKeep.begin(), 
                        positionsOfCandidatesToKeep.end(),
                        [&](const auto& position){
                            const auto& alignment = task.alignments[position];
                            const float relativeOverlap = float(alignment.overlap) / float(task.currentAnchorLength);
                            return fgeq(relativeOverlap, relativeOverlapThreshold) && relativeOverlap < 1.0f;
                        }
                    );

                    if(!goodAlignmentExists){
                        relativeOverlapThreshold -= 0.1f;
                    }
                }
                

                // const bool goodAlignmentExists = std::any_of(
                //     positionsOfCandidatesToKeep.begin(), 
                //     positionsOfCandidatesToKeep.end(),
                //     [&](const auto& position){
                //         const auto& alignment = task.alignments[position];
                //         const float relativeOverlap = float(alignment.overlap) / float(task.currentAnchorLength);
                //         return fgeq(relativeOverlap, 0.7f) && relativeOverlap < 1.0f; //fleq(relativeOverlap, 1.0f);
                //     }
                // );

                if(goodAlignmentExists){
                    tmpPositionsOfCandidatesToKeep.resize(positionsOfCandidatesToKeep.size());

                    auto end = std::copy_if(
                        positionsOfCandidatesToKeep.begin(), 
                        positionsOfCandidatesToKeep.end(),
                        tmpPositionsOfCandidatesToKeep.begin(),
                        [&](const auto& position){
                            const auto& alignment = task.alignments[position];
                            const float relativeOverlap = float(alignment.overlap) / float(task.currentAnchorLength);
                            return fgeq(relativeOverlap, relativeOverlapThreshold);
                        }
                    );

                    tmpPositionsOfCandidatesToKeep.erase(
                        end,
                        tmpPositionsOfCandidatesToKeep.end()
                    );

                    int numRemainingCandidatesTmp = tmpPositionsOfCandidatesToKeep.size();

                    std::swap(tmpPositionsOfCandidatesToKeep, positionsOfCandidatesToKeep);
                    std::swap(numRemainingCandidatesTmp, task.numRemainingCandidates);
                }


                //compact selected candidates inplace

                

                {
                    task.candidateSequenceData.resize(task.numRemainingCandidates * encodedSequencePitchInInts);

                    for(int c = 0; c < task.numRemainingCandidates; c++){
                        const int index = positionsOfCandidatesToKeep[c];

                        task.alignments[c] = task.alignments[index];
                        task.alignmentFlags[c] = task.alignmentFlags[index];
                        task.candidateReadIds[c] = task.candidateReadIds[index];
                        task.candidateSequenceLengths[c] = task.candidateSequenceLengths[index];
                        
                        assert(task.alignmentFlags[index] != BestAlignment_t::None);

                        if(task.alignmentFlags[index] == BestAlignment_t::Forward){
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

            
                //check if mate has been reached
                task.mateIdLocationIter = std::lower_bound(
                    task.candidateReadIds.begin(),
                    task.candidateReadIds.end(),
                    task.mateReadId
                );

                task.mateHasBeenFound = (task.mateIdLocationIter != task.candidateReadIds.end() && *task.mateIdLocationIter == task.mateReadId);

                if(task.mateHasBeenFound){



                    const int mateIndex = std::distance(task.candidateReadIds.begin(), task.mateIdLocationIter);
                    const auto& mateAlignment = task.alignments[mateIndex];

                    bool discardMateId = false;
                    //check that mate alignes as reverse complement (to be on the same strand is current read)
                    if(task.alignmentFlags[mateIndex] != BestAlignment_t::ReverseComplement){
                        discardMateId = true;                    
                    }
                    
                    //check that extending to mate does not leave fragment. If it does, remove mate from candidate list
                    if(task.accumExtensionLengths + task.mateLength + mateAlignment.shift > insertSize + insertSizeStddev){
                        discardMateId = true;
                    }

                    if(discardMateId){
                        task.mateHasBeenFound = false;

                        task.alignments.erase(task.alignments.begin() + mateIndex);
                        task.alignmentFlags.erase(task.alignmentFlags.begin() + mateIndex);
                        task.candidateReadIds.erase(task.candidateReadIds.begin() + mateIndex);
                        task.candidateSequenceLengths.erase(task.candidateSequenceLengths.begin() + mateIndex);

                        task.candidateSequenceData.erase(
                            task.candidateSequenceData.begin() + mateIndex * encodedSequencePitchInInts,
                            task.candidateSequenceData.begin() + (mateIndex + 1) * encodedSequencePitchInInts
                        );

                        task.numRemainingCandidates--;
                    }
                }

            }

            alignmentFilterTimer.stop();

            std::vector<Task> newTasksFromSplit;
            std::vector<int> newTaskIndices;


            auto constructMsa = [&](const auto& task){
                const std::string& decodedAnchor = task.totalDecodedAnchors.back();

                auto calculateOverlapWeight = [](int anchorlength, int nOps, int overlapsize){
                    constexpr float maxErrorPercentInOverlap = 0.2f;

                    return 1.0f - sqrtf(nOps / (overlapsize * maxErrorPercentInOverlap));
                };

                std::vector<int> candidateShifts(task.numRemainingCandidates);
                std::vector<float> candidateOverlapWeights(task.numRemainingCandidates);

                for(int c = 0; c < task.numRemainingCandidates; c++){
                    candidateShifts[c] = task.alignments[c].shift;

                    candidateOverlapWeights[c] = calculateOverlapWeight(
                        task.currentAnchorLength, 
                        task.alignments[c].nOps,
                        task.alignments[c].overlap
                    );
                }

                std::vector<char> candidateStrings(decodedSequencePitchInBytes * task.numRemainingCandidates, '\0');

                for(int c = 0; c < task.numRemainingCandidates; c++){
                    decode2BitSequence(
                        candidateStrings.data() + c * decodedSequencePitchInBytes,
                        task.candidateSequenceData.data() + c * encodedSequencePitchInInts,
                        task.candidateSequenceLengths[c]
                    );
                }

                MultipleSequenceAlignment::InputData msaInput;
                msaInput.useQualityScores = false;
                msaInput.subjectLength = task.currentAnchorLength;
                msaInput.nCandidates = task.numRemainingCandidates;
                msaInput.candidatesPitch = decodedSequencePitchInBytes;
                msaInput.candidateQualitiesPitch = 0;
                msaInput.subject = decodedAnchor.c_str();
                msaInput.candidates = candidateStrings.data();
                msaInput.subjectQualities = nullptr;
                msaInput.candidateQualities = nullptr;
                msaInput.candidateLengths = task.candidateSequenceLengths.data();
                msaInput.candidateShifts = candidateShifts.data();
                msaInput.candidateDefaultWeightFactors = candidateOverlapWeights.data();

                MultipleSequenceAlignment msa;

                msa.build(msaInput);

                return msa;
            };

            auto extendWithMsa = [&](auto& task, const auto& msa){
                if(!task.mateHasBeenFound){
                    //mate not found. prepare next while-loop iteration
                    int consensusLength = msa.consensus.size();

                    //scanning from right to left, find first column with coverage >= 3
                    // int lastGoodColumn = 0;
                    // for(int col = consensusLength - 1; col >= 0; col--){
                    //     if(msa.coverage[col] >= 3){
                    //         lastGoodColumn = col;
                    //         break;
                    //     }
                    // }

                    // const int maxextensionPerStepByGoodColumn = std::max(0, (lastGoodColumn+1) - task.currentAnchorLength);

                    //the first currentAnchorLength columns are occupied by anchor. try to extend read 
                    //by at most maxextensionPerStep bp.

                    //can extend by at most maxextensionPerStep bps
                    int extendBy = std::min(
                        consensusLength - task.currentAnchorLength, 
                        maxextensionPerStep
                        // std::min(
                        //     maxextensionPerStepByGoodColumn, 
                        //     maxextensionPerStep
                        // )
                    );
                    //cannot extend over fragment 
                    extendBy = std::min(extendBy, (insertSize + insertSizeStddev - task.mateLength) - task.accumExtensionLengths);

                    if(extendBy == 0){
                        task.abort = true;
                        task.abortReason = AbortReason::MsaNotExtended;
                    }else{
                        task.accumExtensionLengths += extendBy;

                        //update data for next iteration of outer while loop
                        const int numInts = getEncodedNumInts2Bit(task.currentAnchorLength);

                        task.currentAnchor.resize(numInts);

                        encodeSequence2Bit(
                            task.currentAnchor.data(), 
                            msa.consensus.data() + extendBy, 
                            task.currentAnchorLength
                        );
                    }
                }else{
                    //find end of mate in msa
                    const int index = std::distance(task.candidateReadIds.begin(), task.mateIdLocationIter);
                    const int shift = task.alignments[index].shift;
                    const int clength = task.candidateSequenceLengths[index];
                    assert(shift >= 0);
                    const int endcolumn = shift + clength;

                    const int extendby = shift;
                    assert(extendby >= 0);
                    task.accumExtensionLengths += extendby;

                    std::string decodedAnchor(msa.consensus.data() + extendby, endcolumn - extendby);

                    task.totalDecodedAnchors.emplace_back(std::move(decodedAnchor));
                    task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);
                }
            };

            auto keepSelectedCandidates = [&](auto& task, const auto& selectedCandidateIndices){
                const int numCandidateIndices = selectedCandidateIndices.size();
                assert(numCandidateIndices <= task.numRemainingCandidates);

                for(int i = 0; i < numCandidateIndices; i++){
                    const int c = selectedCandidateIndices[i];
                    if(!(0 <= c && c < task.candidateReadIds.size())){
                        std::cerr << "c = " << c << ", candidateReadIds.size() = " << task.candidateReadIds.size() << "\n";
                    }

                    assert(0 <= c && c < task.candidateReadIds.size());
                    assert(0 <= c && c < task.candidateSequenceLengths.size());
                    assert(0 <= c && c < task.alignments.size());
                    assert(0 <= c && c < task.alignmentFlags.size());

                    assert(0 <= c && c*encodedSequencePitchInInts < task.candidateSequencesFwdData.size());
                    assert(0 <= c && c*encodedSequencePitchInInts < task.candidateSequencesRevcData.size());
                    assert(0 <= c && c*encodedSequencePitchInInts < task.candidateSequenceData.size());

                    task.candidateReadIds[i] = task.candidateReadIds[c];
                    task.candidateSequenceLengths[i] = task.candidateSequenceLengths[c];
                    task.alignments[i] = task.alignments[c];
                    task.alignmentFlags[i] = task.alignmentFlags[c];

                    std::copy_n(
                        task.candidateSequencesFwdData.begin() + c * encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        task.candidateSequencesFwdData.begin() + i * encodedSequencePitchInInts
                    );

                    std::copy_n(
                        task.candidateSequencesRevcData.begin() + c * encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        task.candidateSequencesRevcData.begin() + i * encodedSequencePitchInInts
                    );

                    std::copy_n(
                        task.candidateSequenceData.begin() + c * encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        task.candidateSequenceData.begin() + i * encodedSequencePitchInInts
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
                task.candidateSequencesFwdData.erase(
                    task.candidateSequencesFwdData.begin() + numCandidateIndices * encodedSequencePitchInInts,
                    task.candidateSequencesFwdData.end()
                );
                task.candidateSequencesRevcData.erase(
                    task.candidateSequencesRevcData.begin() + numCandidateIndices * encodedSequencePitchInInts,
                    task.candidateSequencesRevcData.end()
                );
                task.candidateSequenceData.erase(
                    task.candidateSequenceData.begin() + numCandidateIndices * encodedSequencePitchInInts,
                    task.candidateSequenceData.end()
                );
                task.mateIdLocationIter = std::lower_bound(
                    task.candidateReadIds.begin(),
                    task.candidateReadIds.end(),
                    task.mateReadId
                );

                task.mateHasBeenFound = (task.mateIdLocationIter != task.candidateReadIds.end() 
                    && *task.mateIdLocationIter == task.mateReadId);
                task.numRemainingCandidates = numCandidateIndices;
            };

            msaTimer.start();

            for(int indexOfActiveTask : indicesOfActiveTasks){
                auto& task = tasks[indexOfActiveTask];

                const MultipleSequenceAlignment msa = constructMsa(task);

                
#if 1
                if(task.splitDepth < 1){
                    auto possibleSplits = msa.inspectColumnsRegionSplit(task.currentAnchorLength);

                    if(possibleSplits.splits.size() > 1){
                        //auto& task = tasks[indexOfActiveTask];
                        
                        std::sort(
                            possibleSplits.splits.begin(), 
                            possibleSplits.splits.end(),
                            [](const auto& vec1, const auto& vec2){
                                //sort by size, descending
                                return vec2.size() < vec1.size();
                            }
                        );

                        //create a copy of task, and only keep candidates of first split
                        Task taskCopy = task;
                        taskCopy.splitDepth++;

                        keepSelectedCandidates(taskCopy, possibleSplits.splits[0]);
                        const MultipleSequenceAlignment msaOfCopy = constructMsa(taskCopy);
                        extendWithMsa(taskCopy, msaOfCopy);

                        //only keep canddiates of second split
                        keepSelectedCandidates(task, possibleSplits.splits[1]);
                        const MultipleSequenceAlignment newMsa = constructMsa(task);
                        extendWithMsa(task, newMsa);

                        //if extension was not possible in task, replace task by task copy
                        if(task.abort && task.abortReason == AbortReason::MsaNotExtended){
                            //replace task by taskCopy
                            task = std::move(taskCopy);
                        }else{
                            //if extension was possible possible in both task and taskCopy, taskCopy will be added to tasks and list of active tasks
                            newTaskIndices.emplace_back(tasks.size() + newTasksFromSplit.size());
                            newTasksFromSplit.emplace_back(std::move(taskCopy));
                        }

                        // std::cerr << "msa before split:\n";
                        // msa.print(std::cerr);
                        // std::cerr << "\n";

                        // int numsplit = 0;

                        // for(const auto& split : possibleSplits.splits){
                        //     const int numCandidates = split.size();

                        //     std::vector<char> newCandidateStrings(decodedSequencePitchInBytes * numCandidates);
                        //     std::vector<int> newCandidateLengths(numCandidates);
                        //     std::vector<int> newCandidateShifts(numCandidates);
                        //     std::vector<float> newCandidateOverlapWeights(numCandidates);

                        //     for(int i = 0; i < numCandidates; i++){
                        //         const int c = split[i];
                        //         std::copy_n(
                        //             candidateStrings.begin() + c * decodedSequencePitchInBytes,
                        //             decodedSequencePitchInBytes,
                        //             newCandidateStrings.begin() + i * decodedSequencePitchInBytes
                        //         );

                        //         newCandidateLengths[i] = task.candidateSequenceLengths[c];
                        //         newCandidateShifts[i] = candidateShifts[c];
                        //         newCandidateOverlapWeights[i] = candidateOverlapWeights[c];
                        //     }

                        //     MultipleSequenceAlignment::InputData newMsaInput;
                        //     newMsaInput.useQualityScores = false;
                        //     newMsaInput.subjectLength = task.currentAnchorLength;
                        //     newMsaInput.nCandidates = numCandidates;
                        //     newMsaInput.candidatesPitch = decodedSequencePitchInBytes;
                        //     newMsaInput.candidateQualitiesPitch = 0;
                        //     newMsaInput.subject = decodedAnchor.c_str();
                        //     newMsaInput.candidates = newCandidateStrings.data();
                        //     newMsaInput.subjectQualities = nullptr;
                        //     newMsaInput.candidateQualities = nullptr;
                        //     newMsaInput.candidateLengths = newCandidateLengths.data();
                        //     newMsaInput.candidateShifts = newCandidateShifts.data();
                        //     newMsaInput.candidateDefaultWeightFactors = newCandidateOverlapWeights.data();

                        //     msa.build(newMsaInput);

                        //     std::cerr << "msa after split " << numsplit << ":\n";
                        //     msa.print(std::cerr);
                        //     std::cerr << "\n";

                        //     numsplit++;
                        // }

                        
                    }else{
                        extendWithMsa(task, msa);
                    }
                }else{
                    extendWithMsa(task, msa);
                }
#else 
                extendWithMsa(task, msa);
#endif

            }

            msaTimer.stop();

            if(newTasksFromSplit.size() > 0){
                //std::cerr << "Added " << newTasksFromSplit.size() << " tasks\n";
                tasks.insert(tasks.end(), std::make_move_iterator(newTasksFromSplit.begin()), std::make_move_iterator(newTasksFromSplit.end()));
                indicesOfActiveTasks.insert(indicesOfActiveTasks.end(), newTaskIndices.begin(), newTaskIndices.end());

                indicesOfActiveTasksTmp.resize(indicesOfActiveTasks.size());
            }           

            /*
                update book-keeping of used candidates
            */  

            for(int indexOfActiveTask : indicesOfActiveTasks){
                auto& task = tasks[indexOfActiveTask];

                                      
                {
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
                }

                task.usedCandidateReadIdsPerIteration.emplace_back(std::move(task.candidateReadIds));
                task.usedAlignmentsPerIteration.emplace_back(std::move(task.alignments));
                task.usedAlignmentFlagsPerIteration.emplace_back(std::move(task.alignmentFlags));

                task.iteration++;
            }
            
            //update list of active task indices
            indicesOfActiveTasksTmp.erase(
                std::copy_if(
                    indicesOfActiveTasks.begin(), 
                    indicesOfActiveTasks.end(), 
                    indicesOfActiveTasksTmp.begin(),
                    [&](int index){
                        return tasks[index].isActive(insertSize, insertSizeStddev);
                    }
                ),
                indicesOfActiveTasksTmp.end()
            );

            std::swap(indicesOfActiveTasks, indicesOfActiveTasksTmp);
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

                    const std::string& decodedAnchor = task.totalDecodedAnchors[0];

                    const std::vector<int> shifts(task.totalAnchorBeginInExtendedRead.begin() + 1, task.totalAnchorBeginInExtendedRead.end());
                    std::vector<float> initialWeights(numsteps-1, 1.0f);


                    std::vector<char> stepstrings(maxlen * (numsteps-1), '\0');
                    std::vector<int> stepstringlengths(numsteps-1);
                    for(int c = 1; c < numsteps; c++){
                        std::copy(
                            task.totalDecodedAnchors[c].begin(),
                            task.totalDecodedAnchors[c].end(),
                            stepstrings.begin() + (c-1) * maxlen
                        );
                        stepstringlengths[c-1] = task.totalDecodedAnchors[c].size();
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

                    extendResult.extendedRead = std::move(extendedRead);

                    extendResult.mateHasBeenFound = task.mateHasBeenFound;
                }
                // else{
                //     ; //no read extension possible
                // }
            }

            extendResults.emplace_back(std::move(extendResult));

        }

        return extendResults;
    }

    std::vector<ReadExtenderBase::ExtendResult> ReadExtenderBase::combinePairedEndDirectionResults(
        std::vector<ReadExtenderBase::ExtendResult>& resultsLR,
        std::vector<ReadExtenderBase::ExtendResult>& resultsRL
    ){
        auto idcomp = [](const auto& l, const auto& r){ return l.getReadPairId() < r.getReadPairId();};
        auto lengthcomp = [](const auto& l, const auto& r){ return l.extendedRead.length() < r.extendedRead.length();};

        std::sort(resultsLR.begin(), resultsLR.end(), idcomp);

        std::sort(resultsRL.begin(), resultsRL.end(), idcomp);

        std::vector<ReadExtenderBase::ExtendResult> combinedResults(resultsLR.size() +  resultsRL.size());

        std::merge(
            resultsLR.begin(), resultsLR.end(), 
            resultsRL.begin(), resultsRL.end(), 
            combinedResults.begin(),
            idcomp
        );

        auto combineWithSameId = [&](auto begin, auto end){
            auto partitionPoint = std::partition(begin, end, [](const auto& x){ return x.mateHasBeenFound;});

            //if there are results which found mate, choose longest
            if(std::distance(begin, partitionPoint) > 0){
                return *std::max_element(begin, partitionPoint, lengthcomp);
            }else{
                //from results which did not find mate, choose longest
                 return *std::max_element(partitionPoint, end, lengthcomp);
            }
        };

        auto iter1 = combinedResults.begin();
        auto iter2 = combinedResults.begin();
        auto dest = combinedResults.begin();

        while(iter1 != combinedResults.end()){
            while(iter2 != combinedResults.end() && iter1->getReadPairId() == iter2->getReadPairId()){
                ++iter2;
            }

            //range [iter1, iter2) has same read pair id
            *dest = combineWithSameId(iter1, iter2);

            ++dest;
            iter1 = iter2;
        }

        combinedResults.erase(dest, combinedResults.end());

        return combinedResults;
    }

    int batchId = 0;

    std::vector<ReadExtenderBase::ExtendResult> ReadExtenderBase::extendPairedReadBatch(
        const std::vector<ExtendInput>& inputs
    ){

        std::vector<Task> tasks(inputs.size());

        //std::cerr << "Transform LR " << batchId << "\n";
        std::transform(inputs.begin(), inputs.end(), tasks.begin(), 
            [this](const auto& i){return makePairedEndTask(i, ExtensionDirection::LR);});

        //std::cerr << "Process LR " << batchId << "\n";
        std::vector<ExtendResult> extendResultsLR = processPairedEndTasks(tasks);

        std::vector<Task> tasks2(inputs.size());

        //std::cerr << "Transform RL " << batchId << "\n";
        std::transform(inputs.begin(), inputs.end(), tasks2.begin(), 
            [this](const auto& i){return makePairedEndTask(i, ExtensionDirection::RL);});

        //std::cerr << "Process RL " << batchId << "\n";
        std::vector<ExtendResult> extendResultsRL = processPairedEndTasks(tasks2);

        //std::cerr << "Combine " << batchId << "\n";
        std::vector<ExtendResult> extendResultsCombined = combinePairedEndDirectionResults(
            extendResultsLR,
            extendResultsRL
        );

        //std::cerr << "replace " << batchId << "\n";
        //replace original positions in extend read by original sequences
        for(std::size_t i = 0; i < inputs.size(); i++){
            auto& comb = extendResultsCombined[i];
            const auto& input = inputs[i];

            if(comb.direction == ExtensionDirection::LR){
                decode2BitSequence(
                    comb.extendedRead.data(),
                    input.encodedRead1,
                    input.readLength1
                );

                if(comb.mateHasBeenFound){
                    std::vector<char> buf(input.readLength2);
                    decode2BitSequence(
                        buf.data(),
                        input.encodedRead2,
                        input.readLength2
                    );
                    reverseComplementStringInplace(buf.data(), buf.size());
                    std::copy(
                        buf.begin(),
                        buf.end(),
                        comb.extendedRead.begin() + comb.extendedRead.length() - input.readLength2
                    );
                }
            }else{
                decode2BitSequence(
                    comb.extendedRead.data(),
                    input.encodedRead2,
                    input.readLength2
                );

                if(comb.mateHasBeenFound){
                    std::vector<char> buf(input.readLength1);
                    decode2BitSequence(
                        buf.data(),
                        input.encodedRead1,
                        input.readLength1
                    );
                    reverseComplementStringInplace(buf.data(), buf.size());
                    std::copy(
                        buf.begin(),
                        buf.end(),
                        comb.extendedRead.begin() + comb.extendedRead.length() - input.readLength1
                    );
                }
            }
        }

        //std::cerr << "done " << batchId << "\n";

        batchId++;

        return extendResultsCombined;
    }


    /*
        SINGLE END
    */


    std::vector<ReadExtenderBase::ExtendResult> ReadExtenderBase::processSingleEndTasks(
        std::vector<Task>& tasks
    ){
        return processPairedEndTasks(tasks);
    }

    std::vector<ReadExtenderBase::ExtendResult> ReadExtenderBase::combineSingleEndDirectionResults(
        std::vector<ReadExtenderBase::ExtendResult>& resultsLR,
        std::vector<ReadExtenderBase::ExtendResult>& resultsRL,
        const std::vector<ReadExtenderBase::Task>& tasks
    ){
        auto idcomp = [](const auto& l, const auto& r){ return l.readId1 < r.readId1;};
        auto lengthcomp = [](const auto& l, const auto& r){ return l.extendedRead.length() < r.extendedRead.length();};

        //for each consecutive range with same readId, keep the longest sequence
        auto keepLongest = [&](auto& vec){
            auto iter1 = vec.begin();
            auto iter2 = vec.begin();
            auto dest = vec.begin();

            while(iter1 != vec.end()){
                while(iter2 != vec.end() && iter1->readId1 == iter2->readId1){
                    ++iter2;
                }

                //range [iter1, iter2) has same read id
                *dest =  *std::max_element(iter1, iter2, lengthcomp);

                ++dest;
                iter1 = iter2;
            }

            return dest;
        };

        std::sort(resultsLR.begin(), resultsLR.end(), idcomp);
        
        auto resultsLR_end = keepLongest(resultsLR);

        std::sort(resultsRL.begin(), resultsRL.end(), idcomp);

        auto resultsRL_end = keepLongest(resultsRL);

        const int remainingLR = std::distance(resultsLR.begin(), resultsLR_end);
        const int remainingRL = std::distance(resultsRL.begin(), resultsRL_end);

        assert(remainingLR == remainingRL);

        std::vector<ReadExtenderBase::ExtendResult> combinedResults(remainingRL);

        for(int i = 0; i < remainingRL; i++){
            auto& comb = combinedResults[i];
            auto& res1 = resultsLR[i];
            auto& res2 = resultsRL[i];
            const auto& task = tasks[i];

            assert(res1.readId1 == res2.readId1);
            assert(task.myReadId == res1.readId1);

            comb.success = true;
            comb.numIterations = res1.numIterations + res2.numIterations;
            comb.readId1 = res1.readId1;
            comb.readId2 = res1.readId2;

            //get reverse complement of RL extension. overlap it with LR extension
            const int newbasesRL = res2.extendedRead.length() - task.myLength;
            if(newbasesRL > 0){
                reverseComplementStringInplace(res2.extendedRead.data() + task.myLength, newbasesRL);
                comb.extendedRead.append(res2.extendedRead.data() + task.myLength, newbasesRL);
            }

            comb.extendedRead.append(res1.extendedRead);

            
        }

        return combinedResults;
    }

    std::vector<ReadExtenderBase::ExtendResult> ReadExtenderBase::extendSingleEndReadBatch(
        const std::vector<ExtendInput>& inputs
    ){

        std::vector<Task> tasks(inputs.size());

        std::transform(inputs.begin(), inputs.end(), tasks.begin(), 
            [this](const auto& i){return makeSingleEndTask(i, ExtensionDirection::LR);});

        std::vector<ExtendResult> extendResultsLR = processSingleEndTasks(tasks);

        std::vector<Task> tasks2(inputs.size());
        std::transform(inputs.begin(), inputs.end(), tasks2.begin(), 
            [this](const auto& i){return makeSingleEndTask(i, ExtensionDirection::RL);});

        //make sure candidates which were used in LR direction cannot be used again in RL direction

        for(std::size_t i = 0; i < inputs.size(); i++){
            tasks2[i].allUsedCandidateReadIdPairs = std::move(tasks[i].allUsedCandidateReadIdPairs);
        }

        std::vector<ExtendResult> extendResultsRL = processSingleEndTasks(tasks2);

        std::vector<ExtendResult> extendResultsCombined = combineSingleEndDirectionResults(
            extendResultsLR,
            extendResultsRL,
            tasks
        );

        return extendResultsCombined;
    }

}