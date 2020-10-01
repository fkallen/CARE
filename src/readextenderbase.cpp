#include <readextender.hpp>
#include <cpu_alignment.hpp>

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

        std::map<read_number, int> splitTracker; //counts number of tasks per read id, which can change by splitting a task
        for(const auto& t : tasks){
            splitTracker[t.myReadId] = 1;
        }

        //set input string as current anchor
        for(auto& task : tasks){
            std::string decodedAnchor(task.currentAnchorLength, '\0');

            decode2BitSequence(
                &decodedAnchor[0],
                task.currentAnchor.data(),
                task.currentAnchorLength
            );

            task.totalDecodedAnchors.emplace_back(std::move(decodedAnchor));
            task.totalAnchorBeginInExtendedRead.emplace_back(0);
        }

#if 1
        auto vecAccess = [](auto& vec, auto index) -> decltype(vec[index]){
            return vec[index];
        };
#else 
        auto vecAccess = [](auto& vec, auto index) -> decltype(vec.at(index)){
            return vec.at(index);
        };
#endif 

        while(indicesOfActiveTasks.size() > 0){
            //perform one extension iteration for active tasks

            hashTimer.start();

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

                //remove mate of input from candidate list
                auto mateReadIdPos = std::lower_bound(
                    task.candidateReadIds.begin(),                                            
                    task.candidateReadIds.end(),
                    task.mateReadId
                );

                if(mateReadIdPos != task.candidateReadIds.end() && *mateReadIdPos == task.mateReadId){
                    task.candidateReadIds.erase(mateReadIdPos);
                }
            }

            hashTimer.stop();
                

            collectTimer.start();

            /*
                Remove candidate pairs which have already been used for extension
            */

            for(int indexOfActiveTask : indicesOfActiveTasks){
                auto& task = vecAccess(tasks, indexOfActiveTask);

                
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
                auto& task = vecAccess(tasks, indexOfActiveTask);

                

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
                auto& task = vecAccess(tasks, indexOfActiveTask);

                /*
                    Remove bad alignments and the corresponding alignments of their mate
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
                            const auto& alignment = vecAccess(task.alignments, position);
                            const float relativeOverlap = float(alignment.overlap) / float(task.currentAnchorLength);
                            return fgeq(relativeOverlap, relativeOverlapThreshold) && relativeOverlap < 1.0f;
                        }
                    );

                    if(!goodAlignmentExists){
                        relativeOverlapThreshold -= 0.1f;
                    }
                }
                

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

            alignmentFilterTimer.stop();

            std::vector<Task> newTasksFromSplit;
            std::vector<int> newTaskIndices;


            auto constructMsa = [&](auto& task){
                const std::string& decodedAnchor = task.totalDecodedAnchors.back();

                auto calculateOverlapWeight = [](int anchorlength, int nOps, int overlapsize){
                    constexpr float maxErrorPercentInOverlap = 0.2f;

                    return 1.0f - sqrtf(nOps / (overlapsize * maxErrorPercentInOverlap));
                };

                task.candidateShifts.resize(task.numRemainingCandidates);
                task.candidateOverlapWeights.resize(task.numRemainingCandidates);

                for(int c = 0; c < task.numRemainingCandidates; c++){
                    vecAccess(task.candidateShifts, c) = vecAccess(task.alignments, c).shift;

                    vecAccess(task.candidateOverlapWeights, c) = calculateOverlapWeight(
                        task.currentAnchorLength, 
                        vecAccess(task.alignments, c).nOps,
                        vecAccess(task.alignments, c).overlap
                    );
                }

                task.candidateStrings.resize(decodedSequencePitchInBytes * task.numRemainingCandidates, '\0');

                for(int c = 0; c < task.numRemainingCandidates; c++){
                    decode2BitSequence(
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

                MultipleSequenceAlignment msa;

                msa.build(msaInput);

                return msa;
            };

            auto extendWithMsa = [&](auto& task, const auto& msa){

                int consensusLength = msa.consensus.size();
                //can extend by at most maxextensionPerStep bps
                int extendBy = std::min(
                    consensusLength - task.currentAnchorLength, 
                    maxextensionPerStep
                );
                //cannot extend over fragment 
                extendBy = std::min(extendBy, (insertSize + insertSizeStddev - task.mateLength) - task.accumExtensionLengths);

                constexpr int requiredOverlapMate = 70; //TODO relative overlap 
                constexpr int numMismatchesUpperBound = 2;

                if(task.pairedEnd && task.accumExtensionLengths + consensusLength - requiredOverlapMate + task.mateLength >= insertSize - insertSizeStddev){
                    //check if mate can be overlapped with consensus 
                    std::map<int, std::vector<int>> hamMap; //map hamming distance to list start positions
                    for(int startpos = 0; startpos < consensusLength - requiredOverlapMate; startpos++){
                        if(task.accumExtensionLengths + startpos + task.mateLength >= insertSize - insertSizeStddev 
                                && task.accumExtensionLengths + startpos + task.mateLength <= insertSize + insertSizeStddev){
                            
                            const int ham = cpu::hammingDistanceOverlap(
                                msa.consensus.begin() + startpos, msa.consensus.end(), 
                                task.decodedMateRevC.begin(), task.decodedMateRevC.end()
                            );

                            hamMap[ham].emplace_back(startpos);
                        }
                    }

                    std::vector<std::pair<int, std::vector<int>>> flatMap(hamMap.begin(), hamMap.end());
                    std::sort(flatMap.begin(), flatMap.end(), [](const auto& p1, const auto& p2){return p1.first < p2.first;});


                    if(flatMap.size() > 0 && flatMap[0].first <= numMismatchesUpperBound){
                        task.mateHasBeenFound = true;

                        //const int currentAccumExtensionLengths = task.accumExtensionLengths;
                        
                        task.accumExtensionLengths += flatMap[0].second.front();
                        std::string decodedAnchor(task.decodedMateRevC);

                        task.totalDecodedAnchors.emplace_back(std::move(decodedAnchor));
                        task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);

                        // const int startpos = flatMap[0].second.front();
                        // task.resultsequence.resize(currentAccumExtensionLengths + startpos + task.decodedMateRevC.length());
                        // const auto replaceBegin = task.resultsequence.begin() + currentAccumExtensionLengths + startpos;
                        // task.resultsequence.replace(
                        //     replaceBegin, 
                        //     replaceBegin + task.decodedMateRevC.length(), 
                        //     task.decodedMateRevC.begin(), 
                        //     task.decodedMateRevC.end()
                        // );

                    }else{
                        if(extendBy == 0){
                            task.abort = true;
                            task.abortReason = AbortReason::MsaNotExtended;
                        }else{
                            task.accumExtensionLengths += extendBy;

                            //update data for next iteration of outer while loop                           

                            std::string decodedAnchor(msa.consensus.data() + extendBy, task.currentAnchorLength);

                            const int numInts = getEncodedNumInts2Bit(task.currentAnchorLength);

                            task.currentAnchor.resize(numInts);

                            encodeSequence2Bit(
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
                    }
                }else{
                    if(extendBy == 0){
                        task.abort = true;
                        task.abortReason = AbortReason::MsaNotExtended;
                    }else{
                        task.accumExtensionLengths += extendBy;

                        //update data for next iteration of outer while loop
                        std::string decodedAnchor(msa.consensus.data() + extendBy, task.currentAnchorLength);

                        const int numInts = getEncodedNumInts2Bit(task.currentAnchorLength);

                        task.currentAnchor.resize(numInts);

                        encodeSequence2Bit(
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
                }
            };

            auto keepSelectedCandidates = [&](auto& task, const auto& selectedCandidateIndices){
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
                auto& task = vecAccess(tasks, indexOfActiveTask);

                const MultipleSequenceAlignment msa = constructMsa(task);

                // std::cerr << "original msa\n";
                // msa.print(std::cerr);
                // std::cerr << "\n";

                
#if 1
                //if(task.splitDepth == 0){
                if(splitTracker[task.myReadId] <= 4){
                    auto possibleSplits = msa.inspectColumnsRegionSplit(task.currentAnchorLength);

                    if(possibleSplits.splits.size() > 1){
                        //auto& task = tasks[indexOfActiveTask];
                        
                        std::sort(
                            possibleSplits.splits.begin(), 
                            possibleSplits.splits.end(),
                            [](const auto& split1, const auto& split2){
                                //sort by size, descending
                                return split2.listOfCandidates.size() < split1.listOfCandidates.size();
                            }
                        );

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


                        //create a copy of task, and only keep candidates of first split
                        Task taskCopy = task;
                        taskCopy.splitDepth++;

                        // std::cerr << "split\n";
                        // msa.print(std::cerr); 
                        // std::cerr << "\n into \n";

                        keepSelectedCandidates(taskCopy, possibleSplits.splits[0].listOfCandidates);
                        const MultipleSequenceAlignment msaOfCopy = constructMsa(taskCopy);

                        // msaOfCopy.print(std::cerr); 
                        // std::cerr << "\n and \n";

                        extendWithMsa(taskCopy, msaOfCopy);

                        //only keep canddiates of second split
                        keepSelectedCandidates(task, possibleSplits.splits[1].listOfCandidates);
                        const MultipleSequenceAlignment newMsa = constructMsa(task);

                        // newMsa.print(std::cerr); 
                        // std::cerr << "\n";

                        extendWithMsa(task, newMsa);

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

        auto combineWithSameIdFoundMate = [&](auto begin, auto end){
            assert(std::distance(begin, end) > 0);

            //return longest read
            return *std::max_element(begin, end, lengthcomp);
        };

        auto combineWithSameIdNoMate = [&](auto begin, auto end){
            assert(std::distance(begin, end) > 0);

            //TODO optimization: store pairs of indices to results
            std::vector<std::pair<ReadExtenderBase::ExtendResult, ReadExtenderBase::ExtendResult>> pairsToCheck;

            constexpr int minimumOverlap = 40;

            //try to find a pair of extensions with opposite directions which could be overlapped to produce an extension which reached the mate
            for(auto x = begin; x != end; ++x){
                for(auto y = std::next(x); y != end; ++y){
                    const int xl = x->extendedRead.length();
                    const int yl = y->extendedRead.length();

                    if((x->direction == ExtensionDirection::LR && y->direction == ExtensionDirection::RL)
                            || (x->direction == ExtensionDirection::RL && y->direction == ExtensionDirection::LR)){
                        if(xl + yl >= insertSize - insertSizeStddev + minimumOverlap){

                            //put direction LR first
                            if(x->direction == ExtensionDirection::LR){
                                pairsToCheck.emplace_back(*x, *y);
                            }else{
                                pairsToCheck.emplace_back(*y, *x);
                            }
                        }
                    }
                }
            }

            auto glue = [&](const std::string& lrString, const std::string& rlString){
                std::vector<std::string> possibleResults;

                const int lrLength = lrString.length();
                const int rlLength = rlString.length();

                const int maxNumberOfPossibilities = 2*insertSizeStddev + 1;

                for(int p = 0; p < maxNumberOfPossibilities; p++){
                    //the last position of rlString should be at position x in the combined string
                    const int x = (insertSize-1) - insertSizeStddev + p;

                    const int rlBeginInCombined = x - rlLength + 1; //theoretical position of first character of rl in the combined string
                    const int lrEndInCombined = std::min(x, lrLength - 1);
                    const int overlapSize = std::min(
                        std::max(0, lrEndInCombined - rlBeginInCombined + 1),
                        std::min(lrLength, rlLength)
                    );

                    if(overlapSize >= minimumOverlap){
                        const int rlStart = std::max(0, -rlBeginInCombined);
                        const int ham = cpu::hammingDistanceOverlap(
                            lrString.begin() + (lrEndInCombined+1) - overlapSize, lrString.end(), 
                            rlString.begin() + rlStart, rlString.end()
                        );
                        const float mismatchRatio = float(ham) / float(overlapSize);

                        if(fleq(mismatchRatio, 0.05f)){
                            const int newLength = x+1;
                            const int lr1remaining = newLength - (rlLength - rlStart);
                            std::string sequence(newLength, 'F');
                            //assert(newLength == lr1remaining + rlLength - rlStart);
                            const auto it = std::copy_n(lrString.begin(), lr1remaining, sequence.begin());
                            std::copy(rlString.begin() + rlStart, rlString.end(), it);

                            // std::cerr << "alignment:\n";
                            // std::cerr << "ham = " << ham << ", overlap = " << overlapSize << "\n";
                            // std::cerr << lrString << "\n\n" << rlString << "\n";

                            possibleResults.emplace_back(std::move(sequence));
                        }
                    }
                }

                return possibleResults;
            };

            std::vector<std::string> possibleResults;

            for(const auto& pair : pairsToCheck){
                const auto& lr = pair.first;
                const auto& rl = pair.second;
                assert(lr.direction == ExtensionDirection::LR);
                assert(rl.direction == ExtensionDirection::RL);

                std::string revcRLSeq(rl.extendedRead.begin(), rl.extendedRead.end());
                reverseComplementStringInplace(revcRLSeq.data(), revcRLSeq.size());

                //  std::stringstream sstream;

                //  sstream << to_string(lr.abortReason) << " " << to_string(rl.abortReason) << " - " << lr.readId1 << "\n";
                //  sstream << lr.extendedRead << "\n";
                //  sstream << revcRLSeq << "\n\n";

                // std::cerr << sstream.rdbuf();

                auto strings = glue(lr.extendedRead, revcRLSeq);
                possibleResults.insert(possibleResults.end(), std::make_move_iterator(strings.begin()), std::make_move_iterator(strings.end()));
            }

            if(possibleResults.size() > 0){
                
                std::map<std::string, int> histogram;
                for(const auto& r : possibleResults){
                    histogram[r]++;
                }

                //find sequence with highest frequency and return it;
                auto maxIter = std::max_element(
                    histogram.begin(), histogram.end(),
                    [](const auto& p1, const auto& p2){
                        return p1.second < p2.second;
                    }
                );

                // if(histogram.size() >= 1){
                //     std::cerr << "Possible results:\n";

                //     for(const auto& pair : histogram){
                //         std::cerr << pair.second << " : " << pair.first << "\n";
                //     }
                // }

                ExtendResult er;
                er.mateHasBeenFound = true;
                er.success = true;
                er.aborted = false;
                er.numIterations = -1;

                er.direction = ExtensionDirection::LR;
                er.abortReason = AbortReason::None;
                
                er.readId1 = pairsToCheck[0].first.readId1;
                er.readId2 = pairsToCheck[0].first.readId2;
                er.extendedRead = std::move(maxIter->first);

                return er;
            }else{
                //from results which did not find mate, choose longest
                // std::cerr << "Could not merge the following extensions:\n";

                // for(auto it = begin; it != end; ++it){
                //     std::cerr << "id: " << it->readId1;
                //     std::cerr << ", aborted: " << it->aborted;
                //     std::cerr << ", reason: " << to_string(it->abortReason);
                //     std::cerr << ", direction: " << to_string(it->direction) << "\n";
                //     std::cerr << it->extendedRead << "\n";
                // }
                // std::cerr << "\n";
                return *std::max_element(begin, end, lengthcomp);    
            }
        };

        auto combineWithSameId = [&](auto begin, auto end){
            assert(std::distance(begin, end) > 0);

            auto partitionPoint = std::partition(begin, end, [](const auto& x){ return x.mateHasBeenFound;});

            //if there are results which found mate, choose longest
            if(std::distance(begin, partitionPoint) > 0){
                return combineWithSameIdFoundMate(begin, partitionPoint);
            }else{
#if 1                
                return combineWithSameIdNoMate(partitionPoint, end);
#else
                //from results which did not find mate, choose longest
                return *std::max_element(partitionPoint, end, lengthcomp);
#endif                
            }
        };

        auto iter1 = combinedResults.begin();
        auto iter2 = combinedResults.begin();
        auto dest = combinedResults.begin();

        while(iter1 != combinedResults.end()){
            while(iter2 != combinedResults.end() && iter1->getReadPairId() == iter2->getReadPairId()){
                ++iter2;
            }

            //elements in range [iter1, iter2) have same read pair id
            *dest = combineWithSameId(iter1, iter2);

            ++dest;
            iter1 = iter2;
        }

        combinedResults.erase(dest, combinedResults.end());

        return combinedResults;
    }

    //int batchId = 0;

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

        //batchId++;

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