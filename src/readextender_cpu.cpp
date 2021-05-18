#include <readextender_cpu.hpp>
#include <readextenderbase.hpp>
#include <msa.hpp>
#include <sequencehelpers.hpp>
#include <hostdevicefunctions.cuh>

#include <vector>
#include <algorithm>
#include <string>

#define DO_ONLY_REMOVE_MATE_IDS


namespace care{

    std::vector<extension::Task>& ReadExtenderCpu::processPairedEndTasks(
        std::vector<extension::Task>& tasks
    ){
 
        std::vector<int> indicesOfActiveTasks(tasks.size());
        std::vector<int> indicesOfActiveTasksTmp(tasks.size());
        std::iota(indicesOfActiveTasks.begin(), indicesOfActiveTasks.end(), 0);


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

            for(int indexOfActiveTask : indicesOfActiveTasks){
                auto& task = vecAccess(tasks, indexOfActiveTask);

                task.currentAnchor.resize(SequenceHelpers::getEncodedNumInts2Bit(task.currentAnchorLength));

                SequenceHelpers::encodeSequence2Bit(
                    task.currentAnchor.data(), 
                    task.totalDecodedAnchors.back().data(), 
                    task.currentAnchorLength
                );
            }

            hashTimer.start();

            getCandidateReadIds(tasks, indicesOfActiveTasks);

            /*
                Remove anchor ids and mate ids from candidates
            */

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

            computePairFlags(tasks, indicesOfActiveTasks);                

            collectTimer.start();

            /*
                Remove candidate pairs which have already been used for extension
            */
            #ifndef DO_ONLY_REMOVE_MATE_IDS

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

            #endif

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

            filterAlignments(tasks, indicesOfActiveTasks);

            alignmentFilterTimer.stop();

            msaTimer.start();

            computeMSAsAndExtendTasks(tasks, indicesOfActiveTasks);

            msaTimer.stop();

            for(int i = 0; i < int(tasks.size()); i++){
                const auto& task = tasks[i];

                std::cerr << "i = " <<i << "\n";
                std::cerr << "id " << task.id << "\n";
                std::cerr << "numRemainingCandidates " << task.numRemainingCandidates << "\n";
                std::cerr << "iteration " << task.iteration << "\n";
                std::cerr << "mateHasBeenFound " << task.mateHasBeenFound << "\n";
                std::cerr << "abort " << task.abort << "\n";
                std::cerr << "abortReason " << to_string(task.abortReason) << "\n";
            }

            

            //check early exit for tasks

            for(int i = 0; i < int(indicesOfActiveTasks.size()); i++){ 
                const int indexOfActiveTask = indicesOfActiveTasks[i];
                const auto& task = tasks[indexOfActiveTask];

                const int whichtype = task.id % 4;

                assert(indexOfActiveTask % 4 == whichtype);

                if(whichtype == 0){
                    assert(task.direction == extension::ExtensionDirection::LR);
                    assert(task.pairedEnd == true);

                    if(task.mateHasBeenFound){                    
                        tasks[indexOfActiveTask + 1].abort = true;
                        tasks[indexOfActiveTask + 1].abortReason = extension::AbortReason::PairedAnchorFinished;
                        tasks[indexOfActiveTask + 3].abort = true;
                        tasks[indexOfActiveTask + 3].abortReason = extension::AbortReason::OtherStrandFoundMate;
                    }else if(task.abort){
                        tasks[indexOfActiveTask + 1].abort = true;
                        tasks[indexOfActiveTask + 1].abortReason = extension::AbortReason::PairedAnchorFinished;
                    }
                }else if(whichtype == 2){
                    assert(task.direction == extension::ExtensionDirection::RL);
                    assert(task.pairedEnd == true);

                    if(task.mateHasBeenFound){                    
                        tasks[indexOfActiveTask - 1].abort = true;
                        tasks[indexOfActiveTask - 1].abortReason = extension::AbortReason::OtherStrandFoundMate;
                        tasks[indexOfActiveTask + 1].abort = true;
                        tasks[indexOfActiveTask + 1].abortReason = extension::AbortReason::PairedAnchorFinished;
                    }else if(task.abort){
                        tasks[indexOfActiveTask + 1].abort = true;
                        tasks[indexOfActiveTask + 1].abortReason = extension::AbortReason::PairedAnchorFinished;
                    }
                }
            }    

            /*
                update book-keeping of used candidates
            */  

            for(int indexOfActiveTask : indicesOfActiveTasks){
                auto& task = tasks[indexOfActiveTask];

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

        return tasks;
    }







}