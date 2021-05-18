#include <readextenderbase.hpp>
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

namespace care{

struct ReadExtenderCpu{
public:


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
        readStorageHandle{rs.makeHandle()}, minhashHandle{mh.makeQueryHandle()}{

    }

    ~ReadExtenderCpu(){
        readStorage->destroyHandle(readStorageHandle);
        //minhasher->destroyHandle(minhashHandle);
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
     
private:

    

    std::vector<extension::Task>& processPairedEndTasks(std::vector<extension::Task>& tasks);

    void getCandidateReadIdsSingle(
        std::vector<read_number>& result, 
        const unsigned int* encodedRead, 
        int readLength, 
        read_number readId,
        int beginPos = 0 // only positions [beginPos, readLength] are hashed
    ){

        result.clear();

        bool containsN = false;
        readStorage->areSequencesAmbiguous(readStorageHandle, &containsN, &readId, 1);

        //exclude anchors with ambiguous bases
        if(!(correctionOptions.excludeAmbiguousReads && containsN)){

            int numValuesPerSequence = 0;
            int totalNumValues = 0;

            minhasher->determineNumValues(
                minhashHandle,encodedRead,
                encodedSequencePitchInInts,
                &readLength,
                1,
                &numValuesPerSequence,
                totalNumValues
            );

            result.resize(totalNumValues);
            std::array<int, 2> offsets{};

            minhasher->retrieveValues(
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
                        readStorage->areSequencesAmbiguous(readStorageHandle, &containsN, &readId, 1);
                        return containsN;
                    }
                );

                result.erase(minhashResultsEnd, result.end());
            }            

        }else{
            ; // no candidates
        }
    }

    void getCandidateReadIds(std::vector<extension::Task>& tasks, const std::vector<int>& indicesOfActiveTasks){
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            getCandidateReadIdsSingle(
                task.candidateReadIds, 
                task.currentAnchor.data(), 
                task.currentAnchorLength,
                task.currentAnchorReadId
            );

        }
    }


    void loadCandidateSequenceData(std::vector<extension::Task>& tasks, const std::vector<int>& indicesOfActiveTasks){
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            const int numCandidates = task.candidateReadIds.size();

            task.candidateSequenceLengths.resize(numCandidates);
            task.candidateSequencesFwdData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);
            task.candidateSequencesRevcData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);

            readStorage->gatherSequenceLengths(
                readStorageHandle,
                task.candidateSequenceLengths.data(),
                task.candidateReadIds.data(),
                task.candidateReadIds.size()
            );

            readStorage->gatherSequences(
                readStorageHandle,
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
    }

    void computePairFlags(std::vector<extension::Task>& tasks, const std::vector<int>& indicesOfActiveTasks){
        const int numTasks = indicesOfActiveTasks.size();

        for(int first = 0, second = 1; second < numTasks; ){
            const int taskindex1 = indicesOfActiveTasks[first];
            const int taskindex2 = indicesOfActiveTasks[second];

            const bool areConsecutiveTasks = tasks[taskindex1].id + 1 == tasks[taskindex2].id;
            const bool arePairedTasks = (tasks[taskindex1].id % 2) + 1 == (tasks[taskindex2].id % 2);

            tasks[taskindex1].isPairedCandidate.resize(tasks[taskindex1].candidateReadIds.size());

            std::fill(tasks[taskindex1].isPairedCandidate.begin(), tasks[taskindex1].isPairedCandidate.end(), false);

            tasks[taskindex2].isPairedCandidate.resize(tasks[taskindex2].candidateReadIds.size());

            std::fill(tasks[taskindex2].isPairedCandidate.begin(), tasks[taskindex2].isPairedCandidate.end(), false);

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

    void eraseDataOfRemovedMates(std::vector<extension::Task>& tasks, const std::vector<int>& indicesOfActiveTasks){
        
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

                task.mateRemovedFromCandidates = false;
            }

        }
    }

    void calculateAlignments(std::vector<extension::Task>& tasks, const std::vector<int>& indicesOfActiveTasks){
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

    void filterAlignments(std::vector<extension::Task>& tasks, const std::vector<int>& indicesOfActiveTasks){
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
                    // if(task.candidateReadIds[c] == 22182866){
                    //     std::cerr << "removed 22182866 in task id" << task.id << "\n";
                    // }
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
                                // if(task.candidateReadIds[c] == 22182866){
                                //     std::cerr << "removed 22182866 in task id" << task.id << ". relativeOverlap = " << relativeOverlap << ", relativeOverlapThreshold = " << relativeOverlapThreshold << "\n";
                                // }
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

            // std::cerr << "candidates of task " << task.id << " before filter in iteration "<< task.iteration << ":\n";
            // for(int i = 0; i < int(task.candidateReadIds.size()); i++){
            //     std::cerr << task.candidateReadIds[i] << " ";
            // }
            // std::cerr << "\n";

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

    MultipleSequenceAlignment constructMSA(extension::Task& task){
        const std::string& decodedAnchor = task.totalDecodedAnchors.back();

        MultipleSequenceAlignment msa(qualityConversion);

        std::vector<char> candidateQualities(task.numRemainingCandidates * qualityPitchInBytes);

        if(correctionOptions.useQualityScores){

            readStorage->gatherQualities(
                readStorageHandle,
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

    struct ExtendWithMsaResult{
        bool mateHasBeenFound = false;
        extension::AbortReason abortReason = extension::AbortReason::None;
        int newLength = 0;
        int newAccumExtensionLength = 0;
        int sizeOfGapToMate = 0;
        std::string newAnchor = "";
        std::string newQuality = "";

    };

    ExtendWithMsaResult extendWithMsa(extension::Task& task, const MultipleSequenceAlignment& msa){

        
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

    void computeMSAsAndExtendTasks(std::vector<extension::Task>& tasks, const std::vector<int>& indicesOfActiveTasks){
        for(int indexOfActiveTask : indicesOfActiveTasks){
            auto& task = tasks[indexOfActiveTask];

            if(task.numRemainingCandidates > 0){
                const auto msa = constructMSA(task);
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

            extendResult.success = true;

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

    int insertSize{};
    int insertSizeStddev{};
    int maxextensionPerStep{};
    int minCoverageForExtension{1};
    int maximumSequenceLength{};
    std::size_t encodedSequencePitchInInts{};
    std::size_t decodedSequencePitchInBytes{};
    std::size_t qualityPitchInBytes{};

    CorrectionOptions correctionOptions{};
    GoodAlignmentProperties goodAlignmentProperties{};

    ReadStorageHandle readStorageHandle;
    CpuMinhasher::QueryHandle minhashHandle;
    cpu::shd::CpuAlignmentHandle alignmentHandle;

    helpers::CpuTimer hashTimer{};
    helpers::CpuTimer collectTimer{};
    helpers::CpuTimer alignmentTimer{};
    helpers::CpuTimer alignmentFilterTimer{};
    helpers::CpuTimer msaTimer{};

};

}

