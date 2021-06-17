#include <config.hpp>

#include <cpu_alignment.hpp>
#include <cpuminhasher.hpp>
#include <cpureadstorage.hpp>
#include <options.hpp>
#include <bestalignment.hpp>
#include <sequencehelpers.hpp>
#include <hpc_helpers.cuh>
#include <msa.hpp>
#include <readextender_common.hpp>
#include <qualityscoreweights.hpp>
#include <hostdevicefunctions.cuh>

#include <vector>
#include <algorithm>
#include <string>

namespace care{

#define DO_ONLY_REMOVE_MATE_IDS

//forward declaration
struct GpuExtensionStepper;

struct ReadExtenderCpu{
    friend struct GpuExtensionStepper;
public:

    ReadExtenderCpu() = default;

    ReadExtenderCpu(
        int insertSize_,
        int insertSizeStddev_,
        int maxextensionPerStep_,
        int maximumSequenceLength_,
        const CpuReadStorage& rs, 
        const CpuMinhasher& mh,
        const CorrectionOptions& coropts,
        const GoodAlignmentProperties& gap,
        const cpu::QualityScoreConversion* qualityConversion_
    ) : 
        readStorage(&rs), minhasher(&mh), 
        qualityConversion(qualityConversion_),
        insertSize(insertSize_), 
        insertSizeStddev(insertSizeStddev_),
        maxextensionPerStep(maxextensionPerStep_),
        maximumSequenceLength(maximumSequenceLength_),
        encodedSequencePitchInInts(SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength_)),
        decodedSequencePitchInBytes(maximumSequenceLength_),
        qualityPitchInBytes(maximumSequenceLength_),
        correctionOptions(coropts),
        goodAlignmentProperties(gap),
        minhashHandle{mh.makeMinhasherHandle()}{

        setActiveReadStorage(readStorage);
        setActiveMinhasher(minhasher);
    }

    ~ReadExtenderCpu(){
        if(minhasher!=nullptr){
            minhasher->destroyHandle(minhashHandle);
        }
    }

    void printTimers(){
        hashTimer.print();
        collectTimer.print();
        alignmentTimer.print();
        alignmentFilterTimer.print();
        msaTimer.print();
    }

    std::vector<extension::ExtendResult> extend(std::vector<extension::ExtendInput> inputs){
        auto tasks = makePairedEndTasksFromInput4(inputs.begin(), inputs.end());
        
        auto extendedTasks = processPairedEndTasks(tasks);

        auto extendResults = constructResults(
            extendedTasks
        );

        return extendResults;
    }

    void setMaxExtensionPerStep(int e) noexcept{
        maxextensionPerStep = e;
    }

    void setMinCoverageForExtension(int c) noexcept{
        minCoverageForExtension = c;
    }

    void setActiveMinhasher(const CpuMinhasher* active){
        if(active == nullptr){
            //reset to default
            activeMinhasher = minhasher;
        }else{
            activeMinhasher = active;
        }
    }

    void setActiveReadStorage(const CpuReadStorage* active){
        if(active == nullptr){
            //reset to default
            activeReadStorage = readStorage;
        }else{
            activeReadStorage = active;
        }
    }
     
private:

    struct ExtendWithMsaResult{
        bool mateHasBeenFound = false;
        extension::AbortReason abortReason = extension::AbortReason::None;
        int newLength = 0;
        int newAccumExtensionLength = 0;
        int sizeOfGapToMate = 0;
        std::string newAnchor = "";
        std::string newQuality = "";
    };    

    std::vector<extension::Task>& processPairedEndTasks(
        std::vector<extension::Task>& tasks
    ) const{
 
        std::vector<int> indicesOfActiveTasks(tasks.size());
        std::iota(indicesOfActiveTasks.begin(), indicesOfActiveTasks.end(), 0);

        while(indicesOfActiveTasks.size() > 0){
            //perform one extension iteration for active tasks

            #if 1

            doOneExtensionIteration(tasks, indicesOfActiveTasks);

            #else

            for(int indexOfActiveTask : indicesOfActiveTasks){
                auto& task = tasks[indexOfActiveTask];

                task.currentAnchor.resize(SequenceHelpers::getEncodedNumInts2Bit(task.currentAnchorLength));

                SequenceHelpers::encodeSequence2Bit(
                    task.currentAnchor.data(), 
                    task.totalDecodedAnchors.back().data(), 
                    task.currentAnchorLength
                );
            }

            hashTimer.start();

            getCandidateReadIds(tasks, indicesOfActiveTasks);

            removeUsedIdsAndMateIds(tasks, indicesOfActiveTasks);
            

            hashTimer.stop();

            computePairFlags(tasks, indicesOfActiveTasks);                

            collectTimer.start();        

            loadCandidateSequenceData(tasks, indicesOfActiveTasks);

            eraseDataOfRemovedMates(tasks, indicesOfActiveTasks);


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

            #endif
            
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

    void doOneExtensionIteration(std::vector<extension::Task>& tasks, const std::vector<int>& indicesOfActiveTasks) const{
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            task.currentAnchor.resize(SequenceHelpers::getEncodedNumInts2Bit(task.currentAnchorLength));

            SequenceHelpers::encodeSequence2Bit(
                task.currentAnchor.data(), 
                task.totalDecodedAnchors.back().data(), 
                task.currentAnchorLength
            );
        }

        hashTimer.start();

        getCandidateReadIds(tasks, indicesOfActiveTasks);

        // for(auto indexOfActiveTask : indicesOfActiveTasks){
        //     const auto& task = tasks[indexOfActiveTask];

        //     if(task.myReadId == 0 && task.id == 3 && maxextensionPerStep == 6){
        //         std::cerr << "Anchor: " << task.totalDecodedAnchors.back() << "\n";
        //         std::cerr << "iteration " << task.iteration << ", candidates raw\n";
        //         for(auto x : task.candidateReadIds){
        //             std::cerr << x << " ";
        //         }
        //         std::cerr << "\n";
        //     }
        // }

        removeUsedIdsAndMateIds(tasks, indicesOfActiveTasks);

        // for(auto indexOfActiveTask : indicesOfActiveTasks){
        //     const auto& task = tasks[indexOfActiveTask];

        //     if(task.myReadId == 0 && task.id == 3 && maxextensionPerStep == 6){
        //         std::cerr << "iteration " << task.iteration << ", candidates after remove\n";
        //         for(auto x : task.candidateReadIds){
        //             std::cerr << x << " ";
        //         }
        //         std::cerr << "\n";
        //     }
        // }
        

        hashTimer.stop();

        computePairFlags(tasks, indicesOfActiveTasks);                

        collectTimer.start();        

        loadCandidateSequenceData(tasks, indicesOfActiveTasks);

        eraseDataOfRemovedMates(tasks, indicesOfActiveTasks);


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

        // for(auto indexOfActiveTask : indicesOfActiveTasks){
        //     const auto& task = tasks[indexOfActiveTask];

        //     if(task.myReadId == 0 && task.id == 3 && maxextensionPerStep == 6){
        //         std::cerr << "iteration " << task.iteration << ", candidates after alignment filter\n";
        //         for(auto x : task.candidateReadIds){
        //             std::cerr << x << " ";
        //         }
        //         std::cerr << "\n";
        //     }
        // }

        msaTimer.start();

        computeMSAsAndExtendTasks(tasks, indicesOfActiveTasks);

        msaTimer.stop();       

        handleEarlyExitOfTasks4(tasks, indicesOfActiveTasks);

        /*
            update book-keeping of used candidates
        */  

        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            std::vector<read_number> tmp(task.allUsedCandidateReadIdPairs.size() + task.candidateReadIds.size());
            auto tmp_end = std::set_union(
                task.allUsedCandidateReadIdPairs.begin(),
                task.allUsedCandidateReadIdPairs.end(),
                task.candidateReadIds.begin(),
                task.candidateReadIds.end(),
                tmp.begin()
            );

            tmp.erase(tmp_end, tmp.end());

            std::swap(task.allUsedCandidateReadIdPairs, tmp);

            const int numCandidates = task.candidateReadIds.size();

            if(numCandidates > 0 && task.abortReason == extension::AbortReason::None){
                assert(task.totalAnchorBeginInExtendedRead.size() >= 2);
                const int oldAccumExtensionsLength 
                    = task.totalAnchorBeginInExtendedRead[task.totalAnchorBeginInExtendedRead.size() - 2];
                const int newAccumExtensionsLength = task.totalAnchorBeginInExtendedRead.back();
                const int lengthOfExtension = newAccumExtensionsLength - oldAccumExtensionsLength;

                std::vector<read_number> fullyUsedIds;

                for(int c = 0; c < numCandidates; c += 1){
                    const int candidateLength = task.candidateSequenceLengths[c];
                    const int shift = task.alignments[c].shift;

                    if(candidateLength + shift <= task.currentAnchorLength + lengthOfExtension){
                        fullyUsedIds.emplace_back(task.candidateReadIds[c]);
                    }
                }

                std::vector<read_number> tmp2(task.allFullyUsedCandidateReadIdPairs.size() + fullyUsedIds.size());
                auto tmp2_end = std::set_union(
                    task.allFullyUsedCandidateReadIdPairs.begin(),
                    task.allFullyUsedCandidateReadIdPairs.end(),
                    fullyUsedIds.begin(),
                    fullyUsedIds.end(),
                    tmp2.begin()
                );

                tmp2.erase(tmp2_end, tmp2.end());
                std::swap(task.allFullyUsedCandidateReadIdPairs, tmp2);

                assert(task.allFullyUsedCandidateReadIdPairs.size() <= task.allUsedCandidateReadIdPairs.size());
            }

            // std::cerr << "task readid " << task.myReadId << "iteration " << task.iteration << " fullyused\n";
            // std::copy(task.allFullyUsedCandidateReadIdPairs.begin(), task.allFullyUsedCandidateReadIdPairs.end(), std::ostream_iterator<read_number>(std::cerr, " "));
            // std::cerr << "\n";

            task.iteration++;
        }
    }

    void getCandidateReadIdsSingle(
        std::vector<read_number>& result, 
        const unsigned int* encodedRead, 
        int readLength, 
        read_number readId,
        int beginPos = 0 // only positions [beginPos, readLength] are hashed
    ) const{

        result.clear();

        bool containsN = false;
        activeReadStorage->areSequencesAmbiguous(&containsN, &readId, 1);

        //exclude anchors with ambiguous bases
        if(!(correctionOptions.excludeAmbiguousReads && containsN)){

            int numValuesPerSequence = 0;
            int totalNumValues = 0;

            activeMinhasher->determineNumValues(
                minhashHandle,encodedRead,
                encodedSequencePitchInInts,
                &readLength,
                1,
                &numValuesPerSequence,
                totalNumValues
            );

            result.resize(totalNumValues);
            std::array<int, 2> offsets{};

            activeMinhasher->retrieveValues(
                minhashHandle,
                nullptr, //do not remove selfid
                1,
                totalNumValues,
                result.data(),
                &numValuesPerSequence,
                offsets.data()
            );

            result.erase(result.begin() + numValuesPerSequence, result.end());

            //exclude candidates with ambiguous bases

            if(correctionOptions.excludeAmbiguousReads){
                auto minhashResultsEnd = std::remove_if(
                    result.begin(),
                    result.end(),
                    [&](read_number readId){
                        bool containsN = false;
                        activeReadStorage->areSequencesAmbiguous(&containsN, &readId, 1);
                        return containsN;
                    }
                );

                result.erase(minhashResultsEnd, result.end());
            }            

        }else{
            ; // no candidates
        }
    }

    void getCandidateReadIds(std::vector<extension::Task>& tasks, const std::vector<int>& indicesOfActiveTasks) const{
        #if 0
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            getCandidateReadIdsSingle(
                task.candidateReadIds, 
                task.currentAnchor.data(), 
                task.currentAnchorLength,
                task.currentAnchorReadId
            );

        }
        #else
            const int numSequences = indicesOfActiveTasks.size();

            int totalNumValues = 0;
            std::vector<int> numValuesPerSequence(numSequences);

            {
                std::vector<unsigned int> sequences(encodedSequencePitchInInts * numSequences);
                std::vector<int> lengths(numSequences);

                for(int i = 0; i < numSequences; i++){
                    const auto& task = tasks[indicesOfActiveTasks[i]];
                    std::copy(task.currentAnchor.begin(), task.currentAnchor.end(), sequences.begin() + i * encodedSequencePitchInInts);
                    lengths[i] = task.currentAnchorLength;
                }


                activeMinhasher->determineNumValues(
                    minhashHandle,
                    sequences.data(),
                    encodedSequencePitchInInts,
                    lengths.data(),
                    numSequences,
                    numValuesPerSequence.data(),
                    totalNumValues
                );
            }

            std::vector<read_number> allCandidates(totalNumValues);
            std::vector<int> offsets(numSequences + 1);

            activeMinhasher->retrieveValues(
                minhashHandle,
                nullptr, //do not remove selfid
                numSequences,
                totalNumValues,
                allCandidates.data(),
                numValuesPerSequence.data(),
                offsets.data()
            );

            for(int i = 0; i < numSequences; i++){
                auto& task = tasks[indicesOfActiveTasks[i]];

                task.candidateReadIds.resize(numValuesPerSequence[i]);
                std::copy_n(allCandidates.begin() + offsets[i], numValuesPerSequence[i], task.candidateReadIds.begin());
            }

        #endif
    }

    void removeUsedIdsAndMateIds(std::vector<extension::Task>& tasks, const std::vector<int>& indicesOfActiveTasks) const{
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

            /*
                Remove candidate pairs which have already been used for extension
            */
            #ifndef DO_ONLY_REMOVE_MATE_IDS

            for(int indexOfActiveTask : indicesOfActiveTasks){
                auto& task = tasks[indexOfActiveTask];

                std::vector<read_number> tmp(task.candidateReadIds.size());

                auto end = std::set_difference(
                    task.candidateReadIds.begin(),
                    task.candidateReadIds.end(),
                    task.allFullyUsedCandidateReadIdPairs.begin(),
                    task.allFullyUsedCandidateReadIdPairs.end(),
                    tmp.begin()
                );

                tmp.erase(end, tmp.end());

                std::swap(task.candidateReadIds, tmp);
            }

            #endif
        }
    }

    void loadCandidateSequenceData(std::vector<extension::Task>& tasks, const std::vector<int>& indicesOfActiveTasks) const{
        #if 0
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            const int numCandidates = task.candidateReadIds.size();

            task.candidateSequenceLengths.resize(numCandidates);
            task.candidateSequencesFwdData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);
            task.candidateSequencesRevcData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);

            activeReadStorage->gatherSequenceLengths(
                task.candidateSequenceLengths.data(),
                task.candidateReadIds.data(),
                task.candidateReadIds.size()
            );

            activeReadStorage->gatherSequences(
                task.candidateSequencesFwdData.data(),
                encodedSequencePitchInInts,
                task.candidateReadIds.data(),
                task.candidateReadIds.size()
            );

            for(int c = 0; c < numCandidates; c++){
                const unsigned int* const seqPtr = task.candidateSequencesFwdData.data() 
                                                    + std::size_t(encodedSequencePitchInInts) * c;
                unsigned int* const seqrevcPtr = task.candidateSequencesRevcData.data() 
                                                    + std::size_t(encodedSequencePitchInInts) * c;

                SequenceHelpers::reverseComplementSequence2Bit(
                    seqrevcPtr,  
                    seqPtr,
                    task.candidateSequenceLengths[c]
                );
            }
        }
        #else
            const int numSequences = indicesOfActiveTasks.size();

            std::vector<int> offsets(numSequences);
            offsets[0] = 0;

            int totalNumberOfCandidates = 0;
            for(int i = 0; i < numSequences; i++){
                const auto& task = tasks[indicesOfActiveTasks[i]];
                totalNumberOfCandidates += task.candidateReadIds.size();
                if(i < numSequences - 1){
                    offsets[i+1] = totalNumberOfCandidates;
                }
            }

            std::vector<read_number> readIds(totalNumberOfCandidates);
            std::vector<int> lengths(totalNumberOfCandidates);
            std::vector<unsigned int> forwarddata(totalNumberOfCandidates * encodedSequencePitchInInts);

            for(int i = 0; i < numSequences; i++){
                const auto& task = tasks[indicesOfActiveTasks[i]];
                
                std::copy(task.candidateReadIds.begin(), task.candidateReadIds.end(), readIds.begin() + offsets[i]);
            }

            activeReadStorage->gatherSequenceLengths(
                lengths.data(),
                readIds.data(),
                totalNumberOfCandidates
            );

            activeReadStorage->gatherSequences(
                forwarddata.data(),
                encodedSequencePitchInInts,
                readIds.data(),
                totalNumberOfCandidates
            );

            for(int i = 0; i < numSequences; i++){
                auto& task = tasks[indicesOfActiveTasks[i]];
                const int numCandidates = task.candidateReadIds.size();
                const int offset = offsets[i];

                task.candidateSequenceLengths.resize(numCandidates);
                task.candidateSequencesFwdData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);
                task.candidateSequencesRevcData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);
                
                std::copy_n(lengths.begin() + offset, numCandidates, task.candidateSequenceLengths.begin());
                std::copy_n(
                    forwarddata.begin() + offset * encodedSequencePitchInInts, 
                    numCandidates * encodedSequencePitchInInts, 
                    task.candidateSequencesFwdData.begin()
                );

                for(int c = 0; c < numCandidates; c++){
                    const unsigned int* const seqPtr = task.candidateSequencesFwdData.data() 
                                                        + std::size_t(encodedSequencePitchInInts) * c;
                    unsigned int* const seqrevcPtr = task.candidateSequencesRevcData.data() 
                                                        + std::size_t(encodedSequencePitchInInts) * c;

                    SequenceHelpers::reverseComplementSequence2Bit(
                        seqrevcPtr,  
                        seqPtr,
                        task.candidateSequenceLengths[c]
                    );
                }
            }

        #endif
    }

    void computePairFlags(std::vector<extension::Task>& tasks, const std::vector<int>& indicesOfActiveTasks) const{
        const int numTasks = indicesOfActiveTasks.size();

        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            task.isPairedCandidate.resize(task.candidateReadIds.size());
            std::fill(task.isPairedCandidate.begin(), task.isPairedCandidate.end(), false);
        }

        for(int first = 0, second = 1; second < numTasks; ){
            const int taskindex1 = indicesOfActiveTasks[first];
            const int taskindex2 = indicesOfActiveTasks[second];

            const bool areConsecutiveTasks = tasks[taskindex1].id + 1 == tasks[taskindex2].id;
            const bool arePairedTasks = (tasks[taskindex1].id % 2) + 1 == (tasks[taskindex2].id % 2);

            assert(tasks[taskindex1].isPairedCandidate.size() ==  tasks[taskindex1].candidateReadIds.size());
            assert(tasks[taskindex2].isPairedCandidate.size() ==  tasks[taskindex2].candidateReadIds.size());

            if(areConsecutiveTasks && arePairedTasks){
                const int begin1 = 0;
                const int end1 = tasks[taskindex1].candidateReadIds.size();
                const int begin2 = 0;
                const int end2 = tasks[taskindex2].candidateReadIds.size();

                // assert(std::is_sorted(pairIds + begin1, pairIds + end1));
                // assert(std::is_sorted(pairIds + begin2, pairIds + end2));

                std::vector<int> pairedPositions(std::min(end1-begin1, end2-begin2));
                std::vector<int> pairedPositions2(std::min(end1-begin1, end2-begin2));

                auto endIters = findPositionsOfPairedReadIds(
                    tasks[taskindex1].candidateReadIds.begin() + begin1,
                    tasks[taskindex1].candidateReadIds.begin() + end1,
                    tasks[taskindex2].candidateReadIds.begin() + begin2,
                    tasks[taskindex2].candidateReadIds.begin() + end2,
                    pairedPositions.begin(),
                    pairedPositions2.begin()
                );

                pairedPositions.erase(endIters.first, pairedPositions.end());
                pairedPositions2.erase(endIters.second, pairedPositions2.end());
                
                for(auto i : pairedPositions){
                    tasks[taskindex1].isPairedCandidate[begin1 + i] = true;
                }
                for(auto i : pairedPositions2){
                    tasks[taskindex2].isPairedCandidate[begin2 + i] = true;
                }
                
                first += 2; second += 2;
            }else{
                first += 1; second += 1;
            }

            
        }
    }

    void eraseDataOfRemovedMates(std::vector<extension::Task>& tasks, const std::vector<int>& indicesOfActiveTasks) const{
        
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            if(task.mateRemovedFromCandidates){
                const int numCandidates = task.candidateReadIds.size();

                std::vector<int> positionsOfCandidatesToKeep;
                positionsOfCandidatesToKeep.reserve(numCandidates);

                for(int c = 0; c < numCandidates; c++){
                    const unsigned int* const seqPtr = task.candidateSequencesFwdData.data() 
                                                    + std::size_t(encodedSequencePitchInInts) * c;

                    auto mismatchIters = std::mismatch(
                        task.encodedMate.begin(), task.encodedMate.end(),
                        seqPtr, seqPtr + encodedSequencePitchInInts
                    );

                    //candidate differs from mate
                    if(mismatchIters.first != task.encodedMate.end()){                            
                        positionsOfCandidatesToKeep.emplace_back(c);
                    }else{
                        ;//std::cerr << "";
                    }
                }

                //compact
                const int toKeep = positionsOfCandidatesToKeep.size();
                for(int c = 0; c < toKeep; c++){
                    const int index = positionsOfCandidatesToKeep[c];

                    task.candidateReadIds[c] = task.candidateReadIds[index];
                    task.candidateSequenceLengths[c] = task.candidateSequenceLengths[index];

                    std::copy_n(
                        task.candidateSequencesFwdData.data() + index * encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        task.candidateSequencesFwdData.data() + c * encodedSequencePitchInInts
                    );

                    std::copy_n(
                        task.candidateSequencesRevcData.data() + index * encodedSequencePitchInInts,
                        encodedSequencePitchInInts,
                        task.candidateSequencesRevcData.data() + c * encodedSequencePitchInInts
                    );

                    task.isPairedCandidate[c] = task.isPairedCandidate[index];

                    
                }

                //erase
                task.candidateReadIds.erase(
                    task.candidateReadIds.begin() + toKeep, 
                    task.candidateReadIds.end()
                );
                task.candidateSequenceLengths.erase(
                    task.candidateSequenceLengths.begin() + toKeep, 
                    task.candidateSequenceLengths.end()
                );
                task.candidateSequencesFwdData.erase(
                    task.candidateSequencesFwdData.begin() + toKeep * encodedSequencePitchInInts, 
                    task.candidateSequencesFwdData.end()
                );
                task.candidateSequencesRevcData.erase(
                    task.candidateSequencesRevcData.begin() + toKeep * encodedSequencePitchInInts, 
                    task.candidateSequencesRevcData.end()
                );
                task.isPairedCandidate.erase(
                    task.isPairedCandidate.begin() + toKeep,
                    task.isPairedCandidate.end()
                );

                task.mateRemovedFromCandidates = false;
            }

        }
    }

    void calculateAlignments(std::vector<extension::Task>& tasks, const std::vector<int>& indicesOfActiveTasks) const{
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            const int numCandidates = task.candidateReadIds.size();

            std::vector<care::cpu::SHDResult> forwardAlignments;
            std::vector<care::cpu::SHDResult> revcAlignments;

            forwardAlignments.resize(numCandidates);
            revcAlignments.resize(numCandidates);
            task.alignmentFlags.resize(numCandidates);
            task.alignments.resize(numCandidates);

            care::cpu::shd::cpuShiftedHammingDistancePopcount2BitWithDirection<care::cpu::shd::ShiftDirection::Right>(
                alignmentHandle,
                forwardAlignments.data(),
                task.currentAnchor.data(),
                task.currentAnchorLength,
                task.candidateSequencesFwdData.data(),
                encodedSequencePitchInInts,
                task.candidateSequenceLengths.data(),
                numCandidates,
                goodAlignmentProperties.min_overlap,
                goodAlignmentProperties.maxErrorRate,
                goodAlignmentProperties.min_overlap_ratio
            );

            care::cpu::shd::cpuShiftedHammingDistancePopcount2BitWithDirection<care::cpu::shd::ShiftDirection::Right>(
                alignmentHandle,
                revcAlignments.data(),
                task.currentAnchor.data(),
                task.currentAnchorLength,
                task.candidateSequencesRevcData.data(),
                encodedSequencePitchInInts,
                task.candidateSequenceLengths.data(),
                numCandidates,
                goodAlignmentProperties.min_overlap,
                goodAlignmentProperties.maxErrorRate,
                goodAlignmentProperties.min_overlap_ratio
            );

            //decide whether to keep forward or reverse complement, and keep it

            for(int c = 0; c < numCandidates; c++){
                const auto& forwardAlignment = forwardAlignments[c];
                const auto& revcAlignment = revcAlignments[c];
                const int candidateLength = task.candidateSequenceLengths[c];

                task.alignmentFlags[c] = care::choose_best_alignment(
                    forwardAlignment,
                    revcAlignment,
                    task.currentAnchorLength,
                    candidateLength,
                    goodAlignmentProperties.min_overlap_ratio,
                    goodAlignmentProperties.min_overlap,
                    correctionOptions.estimatedErrorrate
                );

                if(task.alignmentFlags[c] == BestAlignment_t::Forward){
                    task.alignments[c] = forwardAlignment;
                }else{
                    task.alignments[c] = revcAlignment;
                }
            }
        }
    }

    void filterAlignments(std::vector<extension::Task>& tasks, const std::vector<int>& indicesOfActiveTasks) const{
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            /*
                Remove bad alignments
            */        

            const int size = task.alignments.size();

            std::vector<bool> keepflags(size, true);
            int removed = 0;
            bool goodAlignmentExists = false;
            float relativeOverlapThreshold = 0.0f;

            for(int c = 0; c < size; c++){
                const BestAlignment_t alignmentFlag0 = task.alignmentFlags[c];
                const int shift = task.alignments[c].shift;
                
                if(alignmentFlag0 != BestAlignment_t::None && shift >= 0){
                    if(!task.isPairedCandidate[c]){
                        const float overlap = task.alignments[c].overlap;
                        const float relativeOverlap = overlap / float(task.currentAnchorLength);

                        if(relativeOverlap < 1.0f && fgeq(relativeOverlap, goodAlignmentProperties.min_overlap_ratio)){
                            goodAlignmentExists = true;
                            const float tmp = floorf(relativeOverlap * 10.0f) / 10.0f;
                            relativeOverlapThreshold = fmaxf(relativeOverlapThreshold, tmp);
                        }
                    }
                }else{
                    keepflags[c] = false;
                    removed++;
                }
            }

            if(goodAlignmentExists){

                for(int c = 0; c < size; c++){
                    if(!task.isPairedCandidate[c]){
                        if(keepflags[c]){
                            const float overlap = task.alignments[c].overlap;
                            const float relativeOverlap = overlap / float(task.currentAnchorLength);                

                            if(!fgeq(relativeOverlap, relativeOverlapThreshold)){
                                keepflags[c] = false;
                                removed++;
                            }
                        }
                    }
                }
            }else{
                //NOOP.
                //if no good alignment exists, no other candidate is removed. we will try to work with the not-so-good alignments
            }


            task.numRemainingCandidates = 0;

            //compact inplace
            task.candidateSequenceData.resize((size - removed) * encodedSequencePitchInInts);

            for(int c = 0; c < size; c++){
                if(keepflags[c]){
                    task.alignments[task.numRemainingCandidates] = task.alignments[c];
                    task.alignmentFlags[task.numRemainingCandidates] = task.alignmentFlags[c];
                    task.candidateReadIds[task.numRemainingCandidates] = task.candidateReadIds[c];
                    task.candidateSequenceLengths[task.numRemainingCandidates] = task.candidateSequenceLengths[c];
                    task.isPairedCandidate[task.numRemainingCandidates] = task.isPairedCandidate[c];

                    assert(task.alignmentFlags[c] != BestAlignment_t::None);

                    if(task.alignmentFlags[c] == BestAlignment_t::Forward){
                        std::copy_n(
                            task.candidateSequencesFwdData.data() + c * encodedSequencePitchInInts,
                            encodedSequencePitchInInts,
                            task.candidateSequenceData.data() + task.numRemainingCandidates * encodedSequencePitchInInts
                        );
                    }else{
                        //BestAlignment_t::ReverseComplement

                        std::copy_n(
                            task.candidateSequencesRevcData.data() + c * encodedSequencePitchInInts,
                            encodedSequencePitchInInts,
                            task.candidateSequenceData.data() + task.numRemainingCandidates * encodedSequencePitchInInts
                        );
                    }

                    task.numRemainingCandidates++;
                }                
            }

            assert(task.numRemainingCandidates + removed == size);

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
            task.isPairedCandidate.erase(
                task.isPairedCandidate.begin() + task.numRemainingCandidates, 
                task.isPairedCandidate.end()
            );

            task.candidateSequencesFwdData.clear();
            task.candidateSequencesRevcData.clear();

            if(task.numRemainingCandidates == 0){
                task.abort = true;
                task.abortReason = extension::AbortReason::NoPairedCandidatesAfterAlignment;
            }

            // std::cerr << "candidates of task " << task.id << " after filter in iteration "<< task.iteration << ":\n";
            // for(int i = 0; i < int(task.candidateReadIds.size()); i++){
            //     std::cerr << task.candidateReadIds[i] << " ";
            // }
            // std::cerr << "\n";

        }
    }

    MultipleSequenceAlignment constructMSA(extension::Task& task, char* candidateQualities) const{
        const std::string& decodedAnchor = task.totalDecodedAnchors.back();

        MultipleSequenceAlignment msa(qualityConversion);

        const bool useQualityScoresForMSA = true;

        // std::vector<char> candidateQualities(task.numRemainingCandidates * qualityPitchInBytes);

        // if(correctionOptions.useQualityScores){

        //     activeReadStorage->gatherQualities(
        //         candidateQualities.data(),
        //         qualityPitchInBytes,
        //         task.candidateReadIds.data(),
        //         task.numRemainingCandidates
        //     );

        //     for(int c = 0; c < task.numRemainingCandidates; c++){
        //         if(task.alignmentFlags[c] == BestAlignment_t::ReverseComplement){
        //             std::reverse(
        //                 candidateQualities.data() + c * qualityPitchInBytes,
        //                 candidateQualities.data() + c * qualityPitchInBytes + task.candidateSequenceLengths[c]
        //             );
        //         }
        //     }

        // }else{
        //     std::fill(candidateQualities.begin(), candidateQualities.end(), 'I');
        // }

        auto build = [&](){

            task.candidateShifts.resize(task.numRemainingCandidates);
            task.candidateOverlapWeights.resize(task.numRemainingCandidates);

            //gather data required for msa
            for(int c = 0; c < task.numRemainingCandidates; c++){
                task.candidateShifts[c] = task.alignments[c].shift;

                task.candidateOverlapWeights[c] = calculateOverlapWeight(
                    task.currentAnchorLength, 
                    task.alignments[c].nOps,
                    task.alignments[c].overlap,
                    goodAlignmentProperties.maxErrorRate
                );
            }

            task.candidateStrings.resize(decodedSequencePitchInBytes * task.numRemainingCandidates, '\0');

            //decode the candidates for msa
            for(int c = 0; c < task.numRemainingCandidates; c++){
                SequenceHelpers::decode2BitSequence(
                    task.candidateStrings.data() + c * decodedSequencePitchInBytes,
                    task.candidateSequenceData.data() + c * encodedSequencePitchInInts,
                    task.candidateSequenceLengths[c]
                );
            }

            MultipleSequenceAlignment::InputData msaInput;
            msaInput.useQualityScores = useQualityScoresForMSA;
            msaInput.subjectLength = task.currentAnchorLength;
            msaInput.nCandidates = task.numRemainingCandidates;
            msaInput.candidatesPitch = decodedSequencePitchInBytes;
            msaInput.candidateQualitiesPitch = qualityPitchInBytes;
            msaInput.subject = decodedAnchor.c_str();
            msaInput.candidates = task.candidateStrings.data();
            msaInput.subjectQualities = task.currentQualityScores.c_str();
            //msaInput.candidateQualities = candidateQualities.data();
            msaInput.candidateQualities = candidateQualities;
            msaInput.candidateLengths = task.candidateSequenceLengths.data();
            msaInput.candidateShifts = task.candidateShifts.data();
            msaInput.candidateDefaultWeightFactors = task.candidateOverlapWeights.data();                    

            msa.build(msaInput);
        };

        build();

        #if 1

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
                    task.isPairedCandidate[insertpos] = task.isPairedCandidate[i];

                    std::copy_n(
                        task.candidateStrings.data() + i * size_t(decodedSequencePitchInBytes),
                        decodedSequencePitchInBytes,
                        task.candidateStrings.data() + insertpos * size_t(decodedSequencePitchInBytes)
                    );

                    std::copy_n(                        
                        candidateQualities + i * size_t(qualityPitchInBytes),
                        qualityPitchInBytes,
                        candidateQualities + insertpos * size_t(qualityPitchInBytes)
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

            task.isPairedCandidate.erase(
                task.isPairedCandidate.begin() + insertpos, 
                task.isPairedCandidate.end()
            );

            // candidateQualities.erase(
            //     candidateQualities.begin() + qualityPitchInBytes * insertpos, 
            //     candidateQualities.end()
            // );

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

        if(getNumRefinementIterations() > 0){                

            for(int numIterations = 0; numIterations < getNumRefinementIterations(); numIterations++){
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
    }

    MultipleSequenceAlignment constructMSA(extension::Task& task) const{
        const std::string& decodedAnchor = task.totalDecodedAnchors.back();

        MultipleSequenceAlignment msa(qualityConversion);

        std::vector<char> candidateQualities(task.numRemainingCandidates * qualityPitchInBytes);

        if(correctionOptions.useQualityScores){

            activeReadStorage->gatherQualities(
                candidateQualities.data(),
                qualityPitchInBytes,
                task.candidateReadIds.data(),
                task.numRemainingCandidates
            );

            for(int c = 0; c < task.numRemainingCandidates; c++){
                if(task.alignmentFlags[c] == BestAlignment_t::ReverseComplement){
                    std::reverse(
                        candidateQualities.data() + c * qualityPitchInBytes,
                        candidateQualities.data() + c * qualityPitchInBytes + task.candidateSequenceLengths[c]
                    );
                }
            }

        }else{
            std::fill(candidateQualities.begin(), candidateQualities.end(), 'I');
        }

        auto build = [&](){

            task.candidateShifts.resize(task.numRemainingCandidates);
            task.candidateOverlapWeights.resize(task.numRemainingCandidates);

            //gather data required for msa
            for(int c = 0; c < task.numRemainingCandidates; c++){
                task.candidateShifts[c] = task.alignments[c].shift;

                task.candidateOverlapWeights[c] = calculateOverlapWeight(
                    task.currentAnchorLength, 
                    task.alignments[c].nOps,
                    task.alignments[c].overlap,
                    goodAlignmentProperties.maxErrorRate
                );
            }

            task.candidateStrings.resize(decodedSequencePitchInBytes * task.numRemainingCandidates, '\0');

            //decode the candidates for msa
            for(int c = 0; c < task.numRemainingCandidates; c++){
                SequenceHelpers::decode2BitSequence(
                    task.candidateStrings.data() + c * decodedSequencePitchInBytes,
                    task.candidateSequenceData.data() + c * encodedSequencePitchInInts,
                    task.candidateSequenceLengths[c]
                );
            }

            MultipleSequenceAlignment::InputData msaInput;
            msaInput.useQualityScores = true;
            msaInput.subjectLength = task.currentAnchorLength;
            msaInput.nCandidates = task.numRemainingCandidates;
            msaInput.candidatesPitch = decodedSequencePitchInBytes;
            msaInput.candidateQualitiesPitch = qualityPitchInBytes;
            msaInput.subject = decodedAnchor.c_str();
            msaInput.candidates = task.candidateStrings.data();
            msaInput.subjectQualities = task.currentQualityScores.c_str();
            msaInput.candidateQualities = candidateQualities.data();
            msaInput.candidateLengths = task.candidateSequenceLengths.data();
            msaInput.candidateShifts = task.candidateShifts.data();
            msaInput.candidateDefaultWeightFactors = task.candidateOverlapWeights.data();                    

            msa.build(msaInput);
        };

        build();

        #if 1

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
                    task.isPairedCandidate[insertpos] = task.isPairedCandidate[i];

                    std::copy_n(
                        task.candidateStrings.data() + i * size_t(decodedSequencePitchInBytes),
                        decodedSequencePitchInBytes,
                        task.candidateStrings.data() + insertpos * size_t(decodedSequencePitchInBytes)
                    );

                    std::copy_n(                        
                        candidateQualities.data() + i * size_t(qualityPitchInBytes),
                        qualityPitchInBytes,
                        candidateQualities.data() + insertpos * size_t(qualityPitchInBytes)
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

            task.isPairedCandidate.erase(
                task.isPairedCandidate.begin() + insertpos, 
                task.isPairedCandidate.end()
            );

            candidateQualities.erase(
                candidateQualities.begin() + qualityPitchInBytes * insertpos, 
                candidateQualities.end()
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

        if(getNumRefinementIterations() > 0){                

            for(int numIterations = 0; numIterations < getNumRefinementIterations(); numIterations++){
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
    }

    ExtendWithMsaResult extendWithMsa(extension::Task& task, const MultipleSequenceAlignment& msa) const{
        
        // if(task.myReadId == 0 && task.id == 3 && maxextensionPerStep == 6){
        //     std::cerr << "task id " << task.id << " myReadId " << task.myReadId << "\n";
        //     std::cerr << "candidates\n";
        //     for(auto x : task.candidateReadIds){
        //         std::cerr << x << " ";
        //     }
        //     std::cerr << "\n";

        //     std::cerr << "consensus\n";
        //     for(auto x : msa.consensus){
        //         std::cerr << x;
        //     }
        //     std::cerr << "\n";

        //     if(task.iteration == 3){
        //         const int num = task.numRemainingCandidates;
        //         std::cerr << "cand strings\n";
        //         for(int k = 0; k < num; k++){
        //             for(int c = 0; c < task.candidateSequenceLengths[k]; c++){
        //                 std::cerr << task.candidateStrings[k * decodedSequencePitchInBytes + c];
        //             }
        //             std::cerr << " " << task.alignments[k].shift;
        //             std::cerr << "\n";
        //         }
        //     }
        // }
        
        const int consensusLength = msa.consensus.size();
        const int anchorLength = task.currentAnchorLength;
        const int mateLength = task.mateLength;

        //can extend by at most maxextensionPerStep bps
        int extendBy = std::min(
            consensusLength - anchorLength, 
            std::max(0, maxextensionPerStep)
        );
        //cannot extend over fragment 
        extendBy = std::min(extendBy, (insertSize + insertSizeStddev - mateLength) - task.accumExtensionLengths);

        if(maxextensionPerStep <= 0){

            auto iter = std::find_if(
                msa.coverage.begin() + anchorLength,
                msa.coverage.end(),
                [&](int cov){
                    return cov < minCoverageForExtension;
                }
            );

            extendBy = std::distance(msa.coverage.begin() + anchorLength, iter);
            extendBy = std::min(extendBy, (insertSize + insertSizeStddev - mateLength) - task.accumExtensionLengths);
        }

        auto makeAnchorForNextIteration = [&](){
            ExtendWithMsaResult result;
            
            if(extendBy == 0){
                result.abortReason = extension::AbortReason::MsaNotExtended;
            }else{
                result.newAccumExtensionLength = task.accumExtensionLengths + extendBy;
                result.newLength = anchorLength;
                result.newAnchor = std::string(msa.consensus.data() + extendBy, anchorLength);
                result.newQuality.resize(anchorLength);
                std::transform(msa.support.begin(), msa.support.begin() + anchorLength, result.newQuality.begin(),
                    [](const float f){
                        return getQualityChar(f);
                    }
                );
            }

            return result;
        };

        constexpr int requiredOverlapMate = 70; //TODO relative overlap 
        constexpr float maxRelativeMismatchesInOverlap = 0.06f;
        constexpr int maxAbsoluteMismatchesInOverlap = 10;

        const int maxNumMismatches = std::min(int(mateLength * maxRelativeMismatchesInOverlap), maxAbsoluteMismatchesInOverlap);


        if(task.pairedEnd && task.accumExtensionLengths + consensusLength - requiredOverlapMate + mateLength >= insertSize - insertSizeStddev){
            //check if mate can be overlapped with consensus 
            //for each possibility to overlap the mate and consensus such that the merged sequence would end in the desired range [insertSize - insertSizeStddev, insertSize + insertSizeStddev]

            const int firstStartpos = std::max(0, insertSize - insertSizeStddev - task.accumExtensionLengths - mateLength);
            const int lastStartposExcl = std::min(
                std::max(0, insertSize + insertSizeStddev - task.accumExtensionLengths - mateLength) + 1,
                consensusLength - requiredOverlapMate
            );

            int bestOverlapMismatches = std::numeric_limits<int>::max();
            int bestOverlapStartpos = -1;

            for(int startpos = firstStartpos; startpos < lastStartposExcl; startpos++){
                //compute metrics of overlap
                    
                const int ham = cpu::hammingDistanceOverlap(
                    msa.consensus.begin() + startpos, msa.consensus.end(), 
                    task.decodedMateRevC.begin(), task.decodedMateRevC.end()
                );

                if(bestOverlapMismatches > ham){
                    bestOverlapMismatches = ham;
                    bestOverlapStartpos = startpos;
                }

                if(bestOverlapMismatches == 0){
                    break;
                }
            }
            
            if(bestOverlapMismatches <= maxNumMismatches){
                const int mateStartposInConsensus = bestOverlapStartpos;
                const int missingPositionsBetweenAnchorEndAndMateBegin = std::max(0, mateStartposInConsensus - task.currentAnchorLength);


                ExtendWithMsaResult result;

                if(missingPositionsBetweenAnchorEndAndMateBegin > 0){
                    //bridge the gap between current anchor and mate

                    result.newAnchor = std::string(msa.consensus.data() + anchorLength, missingPositionsBetweenAnchorEndAndMateBegin);
                    result.newQuality.resize(missingPositionsBetweenAnchorEndAndMateBegin);
                    std::transform(
                        msa.support.begin() + anchorLength, 
                        msa.support.begin() + anchorLength + missingPositionsBetweenAnchorEndAndMateBegin, 
                        result.newQuality.begin(),
                        [](const float f){
                            return getQualityChar(f);
                        }
                    );

                    result.newAccumExtensionLength = task.accumExtensionLengths + task.currentAnchorLength;
                    result.newLength = missingPositionsBetweenAnchorEndAndMateBegin;
                    result.mateHasBeenFound = true;
                    result.sizeOfGapToMate = missingPositionsBetweenAnchorEndAndMateBegin;
                }else{
                    result.newAccumExtensionLength = task.accumExtensionLengths + mateStartposInConsensus;
                    result.newLength = 0;
                    result.mateHasBeenFound = true;
                    result.sizeOfGapToMate = 0;
                }

                return result;
            }else{
                return makeAnchorForNextIteration();
            }
        }else{
            return makeAnchorForNextIteration();
        }
    }

    void computeMSAsAndExtendTasks(std::vector<extension::Task>& tasks, const std::vector<int>& indicesOfActiveTasks) const{
        const int numSequences = indicesOfActiveTasks.size();

        std::vector<int> offsets(numSequences);
        offsets[0] = 0;

        int totalNumberOfCandidates = 0;
        for(int i = 0; i < numSequences; i++){
            const auto& task = tasks[indicesOfActiveTasks[i]];
            totalNumberOfCandidates += task.candidateReadIds.size();
            if(i < numSequences - 1){
                offsets[i+1] = totalNumberOfCandidates;
            }
        }

        std::vector<read_number> readIds(totalNumberOfCandidates);

        for(int i = 0; i < numSequences; i++){
            const auto& task = tasks[indicesOfActiveTasks[i]];
            assert(task.numRemainingCandidates == int(task.candidateReadIds.size()));
            std::copy(task.candidateReadIds.begin(), task.candidateReadIds.end(), readIds.begin() + offsets[i]);
        }

        std::vector<char> candidateQualities(totalNumberOfCandidates * qualityPitchInBytes);

        if(correctionOptions.useQualityScores){

            activeReadStorage->gatherQualities(
                candidateQualities.data(),
                qualityPitchInBytes,
                readIds.data(),
                totalNumberOfCandidates
            );

            for(int i = 0; i < numSequences; i++){
                const auto& task = tasks[indicesOfActiveTasks[i]];

                for(int c = 0; c < task.numRemainingCandidates; c++){
                    if(task.alignmentFlags[c] == BestAlignment_t::ReverseComplement){
                        std::reverse(
                            candidateQualities.data() + (offsets[i] + c) * qualityPitchInBytes,
                            candidateQualities.data() + (offsets[i] + c) * qualityPitchInBytes + task.candidateSequenceLengths[c]
                        );
                    }
                }
            }

        }else{
            std::fill(candidateQualities.begin(), candidateQualities.end(), 'I');
        }

        // const int numTasks = indicesOfActiveTasks.size();
        // for(int i = 0; i < numTasks; i++){
        //     auto& task = tasks[indicesOfActiveTasks[i]];

        // for(int i = 0; i < int(indicesOfActiveTasks.size()); i++){
        //     auto& task = tasks[indicesOfActiveTasks[i]];
        for(const auto& indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            if(task.numRemainingCandidates > 0){
                // std::vector<char> candidateQualitiesSingle(task.numRemainingCandidates * qualityPitchInBytes);

                // if(correctionOptions.useQualityScores){

                //     activeReadStorage->gatherQualities(
                //         candidateQualitiesSingle.data(),
                //         qualityPitchInBytes,
                //         task.candidateReadIds.data(),
                //         task.numRemainingCandidates
                //     );

                //     for(int c = 0; c < task.numRemainingCandidates; c++){
                //         if(task.alignmentFlags[c] == BestAlignment_t::ReverseComplement){
                //             std::reverse(
                //                 candidateQualitiesSingle.data() + c * qualityPitchInBytes,
                //                 candidateQualitiesSingle.data() + c * qualityPitchInBytes + task.candidateSequenceLengths[c]
                //             );
                //         }
                //     }

                // }else{
                //     std::fill(candidateQualitiesSingle.begin(), candidateQualitiesSingle.end(), 'I');
                // }

                // for(int x = 0; x < qualityPitchInBytes * task.numRemainingCandidates){
                //     assert(candidateQualitiesSingle[x] == )
                // }

                //const auto msa = constructMSA(task, candidateQualitiesSingle.data());
                //const auto msa = constructMSA(task, candidateQualities.data() + offsets[i] * qualityPitchInBytes);
                const auto msa = constructMSA(task, candidateQualities.data() + offsets[&indexOfActiveTask - indicesOfActiveTasks.data()] * qualityPitchInBytes);
                //const auto msa = constructMSA(task);
                const auto result = extendWithMsa(task, msa);

                task.abortReason = result.abortReason;
                if(task.abortReason == extension::AbortReason::None){
                    task.mateHasBeenFound = result.mateHasBeenFound;

                    if(!task.mateHasBeenFound){
                        task.currentAnchorLength = result.newLength;
                        task.accumExtensionLengths = result.newAccumExtensionLength;
                        task.totalDecodedAnchors.emplace_back(std::move(result.newAnchor));
                        task.totalAnchorQualityScores.emplace_back(std::move(result.newQuality));
                        task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);

                        task.currentQualityScores = task.totalAnchorQualityScores.back(); 
                        
                    }else{
                        const int sizeofGap = result.sizeOfGapToMate;
                        if(sizeofGap == 0){
                            task.accumExtensionLengths = result.newAccumExtensionLength;
                            task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);
                            task.totalDecodedAnchors.emplace_back(task.decodedMateRevC);
                            task.totalAnchorQualityScores.emplace_back(task.mateQualityScoresReversed);
                        }else{

                            task.accumExtensionLengths = result.newAccumExtensionLength;
                            task.totalDecodedAnchors.emplace_back(std::move(result.newAnchor));
                            task.totalAnchorQualityScores.emplace_back(std::move(result.newQuality));
                            task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);

                            task.accumExtensionLengths += result.newLength;
                            task.totalAnchorBeginInExtendedRead.emplace_back(task.accumExtensionLengths);
                            task.totalDecodedAnchors.emplace_back(task.decodedMateRevC);
                            task.totalAnchorQualityScores.emplace_back(task.mateQualityScoresReversed);
                        }
                    }
                }

                task.abort = task.abortReason != extension::AbortReason::None;
            }else{
                //std::cerr << "did not extend task id " << task.id << " readid " << task.myReadId << " iteration " << task.iteration << " because no candidates.\n";
            }
        }
    }

    std::vector<extension::ExtendResult> constructResults(const std::vector<extension::Task>& tasks) const{
        std::vector<extension::ExtendResult> extendResults;
        extendResults.reserve(tasks.size());

        for(const auto& task : tasks){

            extension::ExtendResult extendResult;
            extendResult.direction = task.direction;
            extendResult.numIterations = task.iteration;
            extendResult.aborted = task.abort;
            extendResult.abortReason = task.abortReason;
            extendResult.readId1 = task.myReadId;
            extendResult.readId2 = task.mateReadId;
            extendResult.originalLength = task.myLength;
            extendResult.originalMateLength = task.mateLength;
            extendResult.read1begin = 0;

            //construct extended read
            //build msa of all saved totalDecodedAnchors[0]

            const int numsteps = task.totalDecodedAnchors.size();

            int maxlen = 0;
            for(const auto& s: task.totalDecodedAnchors){
                const int len = s.length();
                if(len > maxlen){
                    maxlen = len;
                }
            }

            const std::string& decodedAnchor = task.totalDecodedAnchors[0];
            const std::string& anchorQuality = task.totalAnchorQualityScores[0];

            const std::vector<int> shifts(task.totalAnchorBeginInExtendedRead.begin() + 1, task.totalAnchorBeginInExtendedRead.end());
            std::vector<float> initialWeights(numsteps-1, 1.0f);


            std::vector<char> stepstrings(maxlen * (numsteps-1), '\0');
            std::vector<char> stepqualities(maxlen * (numsteps-1), '\0');
            std::vector<int> stepstringlengths(numsteps-1);
            for(int c = 1; c < numsteps; c++){
                std::copy(
                    task.totalDecodedAnchors[c].begin(),
                    task.totalDecodedAnchors[c].end(),
                    stepstrings.begin() + (c-1) * maxlen
                );
                assert(int(task.totalAnchorQualityScores[c].size()) <= maxlen);
                std::copy(
                    task.totalAnchorQualityScores[c].begin(),
                    task.totalAnchorQualityScores[c].end(),
                    stepqualities.begin() + (c-1) * maxlen
                );
                stepstringlengths[c-1] = task.totalDecodedAnchors[c].size();
            }

            MultipleSequenceAlignment::InputData msaInput;
            msaInput.useQualityScores = false;
            msaInput.subjectLength = decodedAnchor.length();
            msaInput.nCandidates = numsteps-1;
            msaInput.candidatesPitch = maxlen;
            msaInput.candidateQualitiesPitch = maxlen;
            msaInput.subject = decodedAnchor.c_str();
            msaInput.candidates = stepstrings.data();
            msaInput.subjectQualities = anchorQuality.data();
            msaInput.candidateQualities = stepqualities.data();
            msaInput.candidateLengths = stepstringlengths.data();
            msaInput.candidateShifts = shifts.data();
            msaInput.candidateDefaultWeightFactors = initialWeights.data();

            MultipleSequenceAlignment msa(qualityConversion);

            msa.build(msaInput);

            //msa.print(std::cerr);

            std::string extendedRead(msa.consensus.begin(), msa.consensus.end());
            std::string extendedReadQuality(msa.consensus.size(), '\0');
            std::transform(msa.support.begin(), msa.support.end(), extendedReadQuality.begin(),
                [](const float f){
                    return getQualityChar(f);
                }
            );

            std::copy(decodedAnchor.begin(), decodedAnchor.end(), extendedRead.begin());
            std::copy(anchorQuality.begin(), anchorQuality.end(), extendedReadQuality.begin());


            //alternative extendedRead. no msa + consensus, just concat

            // std::string extendedReadTmp;

            // if(numsteps > 1){
            //     extendedReadTmp.resize(shifts.back() + stepstringlengths.back(), '\0');

            //     auto b = std::copy(decodedAnchor.begin(), decodedAnchor.end(), extendedReadTmp.begin());
            //     for(int i = 0; i < numsteps - 1; i++){
            //         const int currentEnd = std::distance(extendedReadTmp.begin(), b);

            //         const int nextLength = stepstringlengths[i];
            //         const int nextBegin = shifts[i];

            //         if(nextBegin + nextLength > currentEnd){
            //             const int copybegin = currentEnd - nextBegin;
            //             b = std::copy(
            //                 task.totalDecodedAnchors[i+1].begin() + copybegin,
            //                 task.totalDecodedAnchors[i+1].end(),
            //                 b
            //             );
            //         }
            //     }

            //     assert(b == extendedReadTmp.end());

            //     // if(extendedReadTmp != extendedRead){
            //     //     std::cerr << "old: " << extendedRead << "\n";
            //     //     std::cerr << "new: " << extendedReadTmp << "\n";
            //     // }
            // }else{
            //     extendedReadTmp = decodedAnchor;
            // }

            
            //std::swap(extendedReadTmp, extendedRead);




            

            if(task.mateHasBeenFound){
                //std::cerr << "copy " << task.decodedMateRevC << " to end of consensus " << task.myReadId << "\n";
                std::copy(
                    task.decodedMateRevC.begin(),
                    task.decodedMateRevC.end(),
                    extendedRead.begin() + extendedRead.length() - task.decodedMateRevC.length()
                );

                std::copy(
                    task.mateQualityScoresReversed.begin(),
                    task.mateQualityScoresReversed.end(),
                    extendedReadQuality.begin() + extendedReadQuality.length() - task.decodedMateRevC.length()
                );

                extendResult.read2begin = extendedRead.length() - task.decodedMateRevC.length();
            }else{
                extendResult.read2begin = -1;
            }

            extendResult.extendedRead = std::move(extendedRead);
            extendResult.qualityScores = std::move(extendedReadQuality);

            extendResult.mateHasBeenFound = task.mateHasBeenFound;

            extendResults.emplace_back(std::move(extendResult));
        }

        std::vector<extension::ExtendResult> extendResultsCombined = extension::combinePairedEndDirectionResults4(
            extendResults,
            insertSize,
            insertSizeStddev
        );

        return extendResultsCombined;
    }

    static constexpr int getNumRefinementIterations() noexcept{
        return 5;
    }


    const CpuReadStorage* readStorage{};
    const CpuMinhasher* minhasher{};
    const cpu::QualityScoreConversion* qualityConversion{};

    const CpuReadStorage* activeReadStorage{};
    const CpuMinhasher* activeMinhasher{};

    int insertSize{};
    int insertSizeStddev{};
    int maxextensionPerStep{1};
    int minCoverageForExtension{1};
    int maximumSequenceLength{};
    std::size_t encodedSequencePitchInInts{};
    std::size_t decodedSequencePitchInBytes{};
    std::size_t qualityPitchInBytes{};

    CorrectionOptions correctionOptions{};
    GoodAlignmentProperties goodAlignmentProperties{};

    mutable MinhasherHandle minhashHandle;
    mutable cpu::shd::CpuAlignmentHandle alignmentHandle;

    mutable helpers::CpuTimer hashTimer{"hashtimer"};
    mutable helpers::CpuTimer collectTimer{"gathertimer"};
    mutable helpers::CpuTimer alignmentTimer{"alignmenttimer"};
    mutable helpers::CpuTimer alignmentFilterTimer{"filtertimer"};
    mutable helpers::CpuTimer msaTimer{"msatimer"};

};


#ifdef DO_ONLY_REMOVE_MATE_IDS
#undef DO_ONLY_REMOVE_MATE_IDS
#endif

}

