#include <readextender.hpp>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

namespace care{

    std::vector<ReadExtenderBase::ExtendResult> ReadExtenderBase::extendPairedReadBatch(
        const std::vector<ExtendInput>& inputs
    ){

        std::vector<Task> tasks(inputs.size());

        std::transform(inputs.begin(), inputs.end(), tasks.begin(), [this](const auto& i){return makeTask(i);});

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

            getCandidates(tasks, indicesOfActiveTasks);

            for(int indexOfActiveTask : indicesOfActiveTasks){
                auto& task = tasks[indexOfActiveTask];

                // remove self from candidate list
                if(task.iteration == 0){
                    auto readIdPos = std::lower_bound(
                        task.candidateReadIds.begin(),                                            
                        task.candidateReadIds.end(),
                        task.currentAnchorReadId
                    );

                    if(readIdPos != task.candidateReadIds.end() && *readIdPos == task.currentAnchorReadId){
                        task.candidateReadIds.erase(readIdPos);
                    }
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

            for(int indexOfActiveTask : indicesOfActiveTasks){
                auto& task = tasks[indexOfActiveTask];

                /*
                    Remove candidate pairs which have already been used for extension
                */
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

                /*
                    Load candidate sequences and compute reverse complements
                */

                {
                    const int numCandidates = task.candidateReadIds.size();

                    task.candidateSequenceLengths.resize(numCandidates);
                    task.candidateSequencesFwdData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);
                    task.candidateSequencesRevcData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);

                    readStorage->gatherSequenceLengths(
                        readStorageGatherHandle,
                        task.candidateReadIds.data(),
                        task.candidateReadIds.size(),
                        task.candidateSequenceLengths.data()
                    );

                    readStorage->gatherSequenceData(
                        readStorageGatherHandle,
                        task.candidateReadIds.data(),
                        task.candidateReadIds.size(),
                        task.candidateSequencesFwdData.data(),
                        encodedSequencePitchInInts
                    );

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

            }

            /*
                Compute alignments
            */

            alignmentTimer.start();

            calculateAlignments(tasks, indicesOfActiveTasks);

            alignmentTimer.stop();

            for(int indexOfActiveTask : indicesOfActiveTasks){
                auto& task = tasks[indexOfActiveTask];

                alignmentFilterTimer.start();

                /*
                    Remove bad alignments and the corresponding alignments of their mate
                */        

                const int size = task.alignments.size();

                std::vector<int> positionsOfCandidatesToKeep(size);
                std::vector<int> tmpPositionsOfCandidatesToKeep(size);

                int numRemainingCandidates = 0;

                //select candidates with good alignment and positive shift
                for(int c = 0; c < size; c++){
                    const BestAlignment_t alignmentFlag0 = task.alignmentFlags[c];
                    
                    if(alignmentFlag0 != BestAlignment_t::None && task.alignments[c].shift >= 0){
                        positionsOfCandidatesToKeep[numRemainingCandidates] = c;
                        numRemainingCandidates++;
                    }else{
                        ; //if any of the mates aligns badly, remove both of them
                    }
                }

                positionsOfCandidatesToKeep.erase(
                    positionsOfCandidatesToKeep.begin() + numRemainingCandidates, 
                    positionsOfCandidatesToKeep.end()
                );

                if(numRemainingCandidates == 0){
                    task.abort = true;
                    task.abortReason = AbortReason::NoPairedCandidatesAfterAlignment;

                    continue; //stop processing task
                }

                // //stable_partition is required to make sure candidate read ids remaing sorted
                // auto partitionPointIter = std::stable_partition(
                //     positionsOfCandidatesToKeep.begin(), 
                //     positionsOfCandidatesToKeep.end(),
                //     [&](const auto& position){
                //         const auto& alignment = alignments[positions];
                //         const float relativeOverlap = float(alignment.overlap) / float(input.readLength1);
                //         return fgeq(relativeOverlap, 0.7f) && fleq();
                //     }
                // );



                const bool goodAlignmentExists = std::any_of(
                    positionsOfCandidatesToKeep.begin(), 
                    positionsOfCandidatesToKeep.end(),
                    [&](const auto& position){
                        const auto& alignment = task.alignments[position];
                        const float relativeOverlap = float(alignment.overlap) / float(task.currentAnchorLength);
                        return fgeq(relativeOverlap, 0.7f) && relativeOverlap < 1.0f; //fleq(relativeOverlap, 1.0f);
                    }
                );

                if(goodAlignmentExists){
                    tmpPositionsOfCandidatesToKeep.resize(positionsOfCandidatesToKeep.size());

                    auto end = std::copy_if(
                        positionsOfCandidatesToKeep.begin(), 
                        positionsOfCandidatesToKeep.end(),
                        tmpPositionsOfCandidatesToKeep.begin(),
                        [&](const auto& position){
                            const auto& alignment = task.alignments[position];
                            const float relativeOverlap = float(alignment.overlap) / float(task.currentAnchorLength);
                            return fgeq(relativeOverlap, 0.7f);
                        }
                    );

                    tmpPositionsOfCandidatesToKeep.erase(
                        end,
                        tmpPositionsOfCandidatesToKeep.end()
                    );

                    int numRemainingCandidatesTmp = tmpPositionsOfCandidatesToKeep.size();

                    std::swap(tmpPositionsOfCandidatesToKeep, positionsOfCandidatesToKeep);
                    std::swap(numRemainingCandidatesTmp, numRemainingCandidates);
                }

    #if 0
                //perform binning of candidates. keep candidates in best bins 
                //such that total number of kept candidates passes a threshold

                {
                    constexpr int numBins = 5;
                    std::array<int, numBins> bins{}; //relative overlap
                    std::array<float, numBins> boundaries{1.0f, 0.7f, 0.6f, 0.5f, 0.0f};

                    /*
                        100%,
                        70% - 100%
                        60% - 70%
                        50% - 60%
                        < 50%
                    */
                    for(int i = 0; i < numRemainingCandidates; i++){
                        const int candidateIndex = positionsOfCandidatesToKeep[i];
                        const auto& alignment = alignments[candidateIndex];

                        const float relativeOverlap = float(alignment.overlap) / float(input.readLength1);
                        if(fgeq(relativeOverlap, boundaries[0])){
                            bins[0]++;
                        }else if(fgeq(relativeOverlap, boundaries[1])){
                            bins[1]++;
                        }else if(fgeq(relativeOverlap, boundaries[2])){
                            bins[2]++;
                        }else if(fgeq(relativeOverlap, boundaries[3])){
                            bins[3]++;
                        }else{
                            bins[4]++;
                        }
                    }

                    const int threshold = 5;

                    //select bins. bin[0] does not count towards threshold, because reads with overlap 100% cannot be used for extension,
                    //but only for calculating consensus in MSA
                    int selectedBin = 1;
                    int numSelectedInBins = 0;
                    for(int i = 1; i < numBins; i++){
                        numSelectedInBins += bins[i];
                        selectedBin = i;
                        if(numSelectedInBins >= threshold){
                            break;
                        }
                    }

                    // for(int i = 0; i <numBins; i++){
                    //     std::cerr << bins[i] << " ";
                    // }
                    // std::cerr << "\n";

                    int numRemainingCandidatesTmp = 0;

                    for(int i = 0; i < numRemainingCandidates; i++){
                        const int candidateIndex = positionsOfCandidatesToKeep[i];
                        const auto& alignment = alignments[candidateIndex];

                        const float relativeOverlap = float(alignment.overlap) / float(input.readLength1);

                        if(fgeq(relativeOverlap, boundaries[selectedBin])){
                            tmpPositionsOfCandidatesToKeep[numRemainingCandidatesTmp++] = candidateIndex;
                        }
                    }

                    std::swap(tmpPositionsOfCandidatesToKeep, positionsOfCandidatesToKeep);
                    std::swap(numRemainingCandidatesTmp, numRemainingCandidates);
                }
    #endif


                //compact selected candidates inplace

                std::vector<unsigned int> candidateSequenceData;

                {
                    candidateSequenceData.resize(numRemainingCandidates * encodedSequencePitchInInts);

                    for(int c = 0; c < numRemainingCandidates; c++){
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
                                candidateSequenceData.data() + c * encodedSequencePitchInInts
                            );
                        }else{
                            //BestAlignment_t::ReverseComplement

                            std::copy_n(
                                task.candidateSequencesRevcData.data() + index * encodedSequencePitchInInts,
                                encodedSequencePitchInInts,
                                candidateSequenceData.data() + c * encodedSequencePitchInInts
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
                        task.alignments.begin() + numRemainingCandidates, 
                        task.alignments.end()
                    );
                    task.alignmentFlags.erase(
                        task.alignmentFlags.begin() + numRemainingCandidates, 
                        task.alignmentFlags.end()
                    );
                    task.candidateReadIds.erase(
                        task.candidateReadIds.begin() + numRemainingCandidates, 
                        task.candidateReadIds.end()
                    );
                    task.candidateSequenceLengths.erase(
                        task.candidateSequenceLengths.begin() + numRemainingCandidates, 
                        task.candidateSequenceLengths.end()
                    );
                    // //not sure if these 2 arrays will be required further on
                    // candidateSequencesFwdData.erase(
                    //     candidateSequencesFwdData.begin() + numRemainingCandidates * encodedSequencePitchInInts, 
                    //     candidateSequencesFwdData.end()
                    // );
                    // candidateSequencesRevcData.erase(
                    //     candidateSequencesRevcData.begin() + numRemainingCandidates * encodedSequencePitchInInts, 
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

                //check that extending to mate does not leave fragment
                if(task.mateHasBeenFound){
                    const int mateIndex = std::distance(task.candidateReadIds.begin(), task.mateIdLocationIter);
                    const auto& mateAlignment = task.alignments[mateIndex];

                    if(task.accumExtensionLengths + task.mateLength + mateAlignment.shift > insertSize + insertSizeStddev){
                        task.mateHasBeenFound = false;

                        task.alignments.erase(task.alignments.begin() + mateIndex);
                        task.alignmentFlags.erase(task.alignmentFlags.begin() + mateIndex);
                        task.candidateReadIds.erase(task.candidateReadIds.begin() + mateIndex);
                        task.candidateSequenceLengths.erase(task.candidateSequenceLengths.begin() + mateIndex);

                        candidateSequenceData.erase(
                            candidateSequenceData.begin() + mateIndex * encodedSequencePitchInInts,
                            candidateSequenceData.begin() + (mateIndex + 1) * encodedSequencePitchInInts
                        );
                    }
                }

                alignmentFilterTimer.stop();

                msaTimer.start();

                

                /*
                    Construct MSAs
                */

                {
                    const std::string& decodedAnchor = task.totalDecodedAnchors.back();

                    auto calculateOverlapWeight = [](int anchorlength, int nOps, int overlapsize){
                        constexpr float maxErrorPercentInOverlap = 0.2f;

                        return 1.0f - sqrtf(nOps / (overlapsize * maxErrorPercentInOverlap));
                    };

                    std::vector<int> candidateShifts(numRemainingCandidates);
                    std::vector<float> candidateOverlapWeights(numRemainingCandidates);

                    for(int c = 0; c < numRemainingCandidates; c++){
                        candidateShifts[c] = task.alignments[c].shift;

                        candidateOverlapWeights[c] = calculateOverlapWeight(
                            task.currentAnchorLength, 
                            task.alignments[c].nOps,
                            task.alignments[c].overlap
                        );
                    }

                    std::vector<char> candidateStrings(decodedSequencePitchInBytes * numRemainingCandidates, '\0');

                    for(int c = 0; c < numRemainingCandidates; c++){
                        decode2BitSequence(
                            candidateStrings.data() + c * decodedSequencePitchInBytes,
                            candidateSequenceData.data() + c * encodedSequencePitchInInts,
                            task.candidateSequenceLengths[c]
                        );
                    }

                    MultipleSequenceAlignment::InputData msaInput;
                    msaInput.useQualityScores = false;
                    msaInput.subjectLength = task.currentAnchorLength;
                    msaInput.nCandidates = numRemainingCandidates;
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

                    // if(task.myReadId == 90 || task.mateReadId == 90){
                    //     std::cerr << "Id " << task.myReadId << ", Iteration: " << task.iteration << "\n";
                    //     msa.print(std::cerr);
                    //     std::cerr << "\n";
                    // }

                    if(!task.mateHasBeenFound){
                        //mate not found. prepare next while-loop iteration

                        {
                            int consensusLength = msa.consensus.size();

                            //scanning from right to left, find first column with coverage >= 3
                            // int lastGoodColumn = 0;
                            // for(int col = consensusLength - 1; col >= 0; col--){
                            //     if(msa.coverage[col] >= 3){
                            //         lastGoodColumn = col;
                            //         break;
                            //     }
                            // }

                            // const int maxExtensionByGoodColumn = std::max(0, (lastGoodColumn+1) - currentAnchorLength);

                            //the first currentAnchorLength columns are occupied by anchor. try to extend read 
                            //by at most maxextension bp.

                            //can extend by at most maxextension bps
                            int extendBy = std::min(
                                consensusLength - task.currentAnchorLength, 
                                maxextension
                                // std::min(
                                //     maxExtensionByGoodColumn, 
                                //     maxextension
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
                                const std::string nextDecodedAnchor(msa.consensus.data() + extendBy, task.currentAnchorLength);
                                const int numInts = getEncodedNumInts2Bit(nextDecodedAnchor.size());

                                task.currentAnchor.resize(numInts);
                                encodeSequence2Bit(
                                    task.currentAnchor.data(), 
                                    nextDecodedAnchor.c_str(), 
                                    nextDecodedAnchor.size()
                                );
                                task.currentAnchorLength = nextDecodedAnchor.size();
                            }
                        
                        }
                    }else{
                        {
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
                    }

                }

                msaTimer.stop();

                /*
                    update book-keeping of used candidates
                */                        
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
        std::vector<ExtendResult> extendResults;

        for(const auto& task : tasks){

            ExtendResult extendResult;
            extendResult.numIterations = task.iteration;
            extendResult.aborted = task.abort;
            extendResult.abortReason = task.abortReason;
            extendResult.extensionLengths.emplace_back(task.totalAnchorBeginInExtendedRead.back());

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

                    extendResult.extendedReads.emplace_back(task.myReadId, std::move(extendedRead));

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

}