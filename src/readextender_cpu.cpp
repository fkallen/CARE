#include <readextender_cpu.hpp>
#include <readextenderbase.hpp>

#include <vector>
#include <algorithm>
#include <sequencehelpers.hpp>
#include <string>

namespace care{

        std::vector<ReadExtenderBase::ExtendResult> ReadExtenderCpu::processPairedEndTasks(
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

            hashTimer.stop();
                

            collectTimer.start();

            /*
                Remove candidate pairs which have already been used for extension
            */

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

            /*
                Compute reverse complement of candidates
            */

            // for(int indexOfActiveTask : indicesOfActiveTasks){
            //     auto& task = vecAccess(tasks, indexOfActiveTask);               

            //     const int numCandidates = task.candidateReadIds.size();

            //     task.candidateSequencesRevcData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);

            //     for(int c = 0; c < numCandidates; c++){
            //         const unsigned int* const seqPtr = task.candidateSequencesFwdData.data() 
            //                                             + std::size_t(encodedSequencePitchInInts) * c;
            //         unsigned int* const seqrevcPtr = task.candidateSequencesRevcData.data() 
            //                                             + std::size_t(encodedSequencePitchInInts) * c;

            //         SequenceHelpers::reverseComplementSequence2Bit(
            //             seqrevcPtr,  
            //             seqPtr,
            //             task.candidateSequenceLengths[c]
            //         );
            //     }
            // }

            /*
                If mate has been removed from candidate list, remove all candidates which are equivalent to mate
            */

           eraseDataOfRemovedMates(tasks, indicesOfActiveTasks);

           

            // /*
            //     Compute reverse complement of candidates
            // */

            // for(int indexOfActiveTask : indicesOfActiveTasks){
            //     auto& task = vecAccess(tasks, indexOfActiveTask);               

            //     const int numCandidates = task.candidateReadIds.size();

            //     task.candidateSequencesRevcData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);

            //     for(int c = 0; c < numCandidates; c++){
            //         const unsigned int* const seqPtr = task.candidateSequencesFwdData.data() 
            //                                             + std::size_t(encodedSequencePitchInInts) * c;
            //         unsigned int* const seqrevcPtr = task.candidateSequencesRevcData.data() 
            //                                             + std::size_t(encodedSequencePitchInInts) * c;

            //         SequenceHelpers::reverseComplementSequence2Bit(
            //             seqrevcPtr,  
            //             seqPtr,
            //             task.candidateSequenceLengths[c]
            //         );
            //     }
            // }

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

                //gather data required for msa
                for(int c = 0; c < task.numRemainingCandidates; c++){
                    vecAccess(task.candidateShifts, c) = vecAccess(task.alignments, c).shift;

                    vecAccess(task.candidateOverlapWeights, c) = calculateOverlapWeight(
                        task.currentAnchorLength, 
                        vecAccess(task.alignments, c).nOps,
                        vecAccess(task.alignments, c).overlap
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


    std::vector<ReadExtenderBase::ExtendResult> ReadExtenderCpu::processSingleEndTasks(
        std::vector<ReadExtenderBase::Task>& tasks
    ){
        return processPairedEndTasks(tasks);
    }




}