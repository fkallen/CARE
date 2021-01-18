#ifndef CARE_CORRECTOR_HPP
#define CARE_CORRECTOR_HPP

#include <config.hpp>

#include <options.hpp>

#include <cpuminhasher.hpp>

#include <cpureadstorage.hpp>
#include <cpu_alignment.hpp>
#include <bestalignment.hpp>
#include <msa.hpp>
#include <qualityscoreweights.hpp>
#include <correctionresultprocessing.hpp>
#include <hostdevicefunctions.cuh>
#include <corrector_common.hpp>


#include <cstddef>
#include <memory>
#include <sstream>
#include <vector>
#include <algorithm>
#include <chrono>

namespace care{

    


class CpuErrorCorrector{
public:
    struct CorrectionInput{
        int anchorLength{};
        read_number anchorReadId{};
        const unsigned int* encodedAnchor{};
        const char* anchorQualityscores{};
    };

    struct MultiCorrectionInput{
        std::vector<int> anchorLengths;
        std::vector<read_number> anchorReadIds;
        std::vector<const unsigned int*> encodedAnchors;
        std::vector<const char*> anchorQualityscores;
    };

    struct MultiCandidateIds{
        std::vector<read_number> candidateReadIds;
        std::vector<int> numCandidatesPerAnchor;
        std::vector<int> numCandidatesPerAnchorPS;
    };

    struct MultiCandidateData{
        std::vector<int> candidateLengths;
        std::vector<unsigned int> encodedCandidates;
        std::vector<char> candidateQualities;
    };

    struct CorrectionOutput{
        bool hasAnchorCorrection{};
        TempCorrectedSequence anchorCorrection{};
        std::vector<TempCorrectedSequence> candidateCorrections{};
    };

    struct TimeMeasurements{
        std::chrono::duration<double> getSubjectSequenceDataTimeTotal{0};
        std::chrono::duration<double> getCandidatesTimeTotal{0};
        std::chrono::duration<double> copyCandidateDataToBufferTimeTotal{0};
        std::chrono::duration<double> getAlignmentsTimeTotal{0};
        std::chrono::duration<double> findBestAlignmentDirectionTimeTotal{0};
        std::chrono::duration<double> gatherBestAlignmentDataTimeTotal{0};
        std::chrono::duration<double> mismatchRatioFilteringTimeTotal{0};
        std::chrono::duration<double> compactBestAlignmentDataTimeTotal{0};
        std::chrono::duration<double> fetchQualitiesTimeTotal{0};
        std::chrono::duration<double> makeCandidateStringsTimeTotal{0};
        std::chrono::duration<double> msaAddSequencesTimeTotal{0};
        std::chrono::duration<double> msaFindConsensusTimeTotal{0};
        std::chrono::duration<double> msaMinimizationTimeTotal{0};
        std::chrono::duration<double> msaCorrectSubjectTimeTotal{0};
        std::chrono::duration<double> msaCorrectCandidatesTimeTotal{0};

        TimeMeasurements& operator+=(const TimeMeasurements& rhs) noexcept{
            getSubjectSequenceDataTimeTotal += rhs.getSubjectSequenceDataTimeTotal;
            getCandidatesTimeTotal += rhs.getCandidatesTimeTotal;
            copyCandidateDataToBufferTimeTotal += rhs.copyCandidateDataToBufferTimeTotal;
            getAlignmentsTimeTotal += rhs.getAlignmentsTimeTotal;
            findBestAlignmentDirectionTimeTotal += rhs.findBestAlignmentDirectionTimeTotal;
            gatherBestAlignmentDataTimeTotal += rhs.gatherBestAlignmentDataTimeTotal;
            mismatchRatioFilteringTimeTotal += rhs.mismatchRatioFilteringTimeTotal;
            compactBestAlignmentDataTimeTotal += rhs.compactBestAlignmentDataTimeTotal;
            fetchQualitiesTimeTotal += rhs.fetchQualitiesTimeTotal;
            makeCandidateStringsTimeTotal += rhs.makeCandidateStringsTimeTotal;
            msaAddSequencesTimeTotal += rhs.msaAddSequencesTimeTotal;
            msaFindConsensusTimeTotal += rhs.msaFindConsensusTimeTotal;
            msaMinimizationTimeTotal += rhs.msaMinimizationTimeTotal;
            msaCorrectSubjectTimeTotal += rhs.msaCorrectSubjectTimeTotal;
            msaCorrectCandidatesTimeTotal += rhs.msaCorrectCandidatesTimeTotal;

            return *this;
        }

        std::chrono::duration<double> getSumOfDurations() const noexcept{
            std::chrono::duration<double> sum = getSubjectSequenceDataTimeTotal
                                            + getCandidatesTimeTotal
                                            + copyCandidateDataToBufferTimeTotal
                                            + getAlignmentsTimeTotal
                                            + findBestAlignmentDirectionTimeTotal
                                            + gatherBestAlignmentDataTimeTotal
                                            + mismatchRatioFilteringTimeTotal
                                            + compactBestAlignmentDataTimeTotal
                                            + fetchQualitiesTimeTotal
                                            + makeCandidateStringsTimeTotal
                                            + msaAddSequencesTimeTotal
                                            + msaFindConsensusTimeTotal
                                            + msaMinimizationTimeTotal
                                            + msaCorrectSubjectTimeTotal
                                            + msaCorrectCandidatesTimeTotal;
            return sum;
        }
    };   



    CpuErrorCorrector() = default;
    CpuErrorCorrector(
        std::size_t encodedSequencePitchInInts_,
        std::size_t decodedSequencePitchInBytes_,
        std::size_t qualityPitchInBytes_,
        const CorrectionOptions& correctionOptions_,
        const GoodAlignmentProperties& goodAlignmentProperties_,
        const CpuMinhasher& minhasher_,
        const CpuReadStorage& readStorage_,
        ReadCorrectionFlags& correctionFlags_
    ) : encodedSequencePitchInInts(encodedSequencePitchInInts_),
        decodedSequencePitchInBytes(decodedSequencePitchInBytes_),
        qualityPitchInBytes(qualityPitchInBytes_),
        correctionOptions(&correctionOptions_),
        goodAlignmentProperties(&goodAlignmentProperties_),
        minhasher{&minhasher_},
        minhashHandle{minhasher->makeQueryHandle()},
        readStorage{&readStorage_},
        readStorageHandle{readStorage->makeHandle()},
        correctionFlags(&correctionFlags_),
        qualityCoversion(std::make_unique<cpu::QualityScoreConversion>())
    {

    }

    ~CpuErrorCorrector(){
        readStorage->destroyHandle(readStorageHandle);
    }

    CorrectionOutput process(const CorrectionInput input){
        Task task = makeTask(input);

        TimeMeasurements timings;

        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        auto tpa = std::chrono::system_clock::now();
        #endif

        determineCandidateReadIds(task);

        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        timings.getCandidatesTimeTotal += std::chrono::system_clock::now() - tpa;
        #endif


        if(task.candidateReadIds.size() == 0){
            //return uncorrected anchor
            return CorrectionOutput{};
        }

        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        tpa = std::chrono::system_clock::now();
        #endif

        getCandidateSequenceData(task);

        computeReverseComplementCandidates(task);

        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        timings.copyCandidateDataToBufferTimeTotal += std::chrono::system_clock::now() - tpa;
        #endif


        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        tpa = std::chrono::system_clock::now();
        #endif

        getCandidateAlignments(task);

        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        timings.getAlignmentsTimeTotal += std::chrono::system_clock::now() - tpa;
        #endif


        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        tpa = std::chrono::system_clock::now();
        #endif

        filterCandidatesByAlignmentFlag(task);

        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        timings.gatherBestAlignmentDataTimeTotal += std::chrono::system_clock::now() - tpa;
        #endif


        if(task.candidateReadIds.size() == 0){
            //return uncorrected anchor
            return CorrectionOutput{};
        }

        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        tpa = std::chrono::system_clock::now();
        #endif

        filterCandidatesByAlignmentMismatchRatio(task);

        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        timings.mismatchRatioFilteringTimeTotal += std::chrono::system_clock::now() - tpa;
        #endif


        if(task.candidateReadIds.size() == 0){
            //return uncorrected anchor
            return CorrectionOutput{};
        }

        if(correctionOptions->useQualityScores){

            #ifdef ENABLE_CPU_CORRECTOR_TIMING
            tpa = std::chrono::system_clock::now();
            #endif

            getCandidateQualities(task);
            reverseQualitiesOfRCAlignments(task);

            #ifdef ENABLE_CPU_CORRECTOR_TIMING
            timings.fetchQualitiesTimeTotal += std::chrono::system_clock::now() - tpa;
            #endif

        }

        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        tpa = std::chrono::system_clock::now();
        #endif

        makeCandidateStrings(task);

        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        timings.makeCandidateStringsTimeTotal += std::chrono::system_clock::now() - tpa;
        #endif


        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        tpa = std::chrono::system_clock::now();
        #endif

        alignmentsComputeWeightsAndAoStoSoA(task);

        buildMultipleSequenceAlignment(task);

        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        timings.msaFindConsensusTimeTotal += std::chrono::system_clock::now() - tpa;
        #endif


        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        tpa = std::chrono::system_clock::now();
        #endif

        refineMSA(task);

        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        timings.msaMinimizationTimeTotal += std::chrono::system_clock::now() - tpa;
        #endif


        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        tpa = std::chrono::system_clock::now();
        #endif

        correctAnchor(task);

        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        timings.msaCorrectSubjectTimeTotal += std::chrono::system_clock::now() - tpa;
        #endif

        if(task.subjectCorrection.isCorrected){
            if(task.msaProperties.isHQ){
                correctionFlags->setCorrectedAsHqAnchor(task.input.anchorReadId);
            }
        }else{
            correctionFlags->setCouldNotBeCorrectedAsAnchor(task.input.anchorReadId);
        }


        if(task.msaProperties.isHQ && correctionOptions->correctCandidates){

            #ifdef ENABLE_CPU_CORRECTOR_TIMING
            tpa = std::chrono::system_clock::now();
            #endif

            correctCandidates(task);

            #ifdef ENABLE_CPU_CORRECTOR_TIMING
            timings.msaCorrectCandidatesTimeTotal += std::chrono::system_clock::now() - tpa;
            #endif

        }

        CorrectionOutput correctionOutput = makeOutputOfTask(task);

        totalTime += timings;

        return correctionOutput;
    }

    std::vector<CorrectionOutput> processMulti(const MultiCorrectionInput input){
        const int numAnchors = input.anchorReadIds.size();
        if(numAnchors == 0){
            return {};
        }

        std::vector<CorrectionOutput> resultVector;
        resultVector.reserve(numAnchors);

        TimeMeasurements timings;

        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        auto tpa = std::chrono::system_clock::now();
        #endif

        MultiCandidateIds multiIds = determineCandidateReadIds(input);

        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        timings.getCandidatesTimeTotal += std::chrono::system_clock::now() - tpa;
        #endif

        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        tpa = std::chrono::system_clock::now();
        #endif
        MultiCandidateData multiCandidates = getCandidateSequencesData(multiIds);

        #ifdef ENABLE_CPU_CORRECTOR_TIMING
        timings.copyCandidateDataToBufferTimeTotal += std::chrono::system_clock::now() - tpa;
        #endif

        for(int anchorIndex = 0; anchorIndex < numAnchors; anchorIndex++){

            Task task = makeTask(input, multiIds, multiCandidates, anchorIndex);

            if(task.candidateReadIds.size() == 0){
                //return uncorrected anchor
                resultVector.emplace_back();
                continue;
            }

            computeReverseComplementCandidates(task);

            #ifdef ENABLE_CPU_CORRECTOR_TIMING
            tpa = std::chrono::system_clock::now();
            #endif

            getCandidateAlignments(task);

            #ifdef ENABLE_CPU_CORRECTOR_TIMING
            timings.getAlignmentsTimeTotal += std::chrono::system_clock::now() - tpa;
            #endif


            #ifdef ENABLE_CPU_CORRECTOR_TIMING
            tpa = std::chrono::system_clock::now();
            #endif

            filterCandidatesByAlignmentFlag(task);

            #ifdef ENABLE_CPU_CORRECTOR_TIMING
            timings.gatherBestAlignmentDataTimeTotal += std::chrono::system_clock::now() - tpa;
            #endif


            if(task.candidateReadIds.size() == 0){
                //return uncorrected anchor
                resultVector.emplace_back();
                continue;
            }

            #ifdef ENABLE_CPU_CORRECTOR_TIMING
            tpa = std::chrono::system_clock::now();
            #endif

            filterCandidatesByAlignmentMismatchRatio(task);

            #ifdef ENABLE_CPU_CORRECTOR_TIMING
            timings.mismatchRatioFilteringTimeTotal += std::chrono::system_clock::now() - tpa;
            #endif


            if(task.candidateReadIds.size() == 0){
                //return uncorrected anchor
                resultVector.emplace_back();
                continue;
            }

            if(correctionOptions->useQualityScores){

                #ifdef ENABLE_CPU_CORRECTOR_TIMING
                tpa = std::chrono::system_clock::now();
                #endif

                getQualitiesFromMultiCandidates(task, multiIds, multiCandidates, anchorIndex);

                reverseQualitiesOfRCAlignments(task);

                #ifdef ENABLE_CPU_CORRECTOR_TIMING
                timings.fetchQualitiesTimeTotal += std::chrono::system_clock::now() - tpa;
                #endif

            }

            #ifdef ENABLE_CPU_CORRECTOR_TIMING
            tpa = std::chrono::system_clock::now();
            #endif

            makeCandidateStrings(task);

            #ifdef ENABLE_CPU_CORRECTOR_TIMING
            timings.makeCandidateStringsTimeTotal += std::chrono::system_clock::now() - tpa;
            #endif


            #ifdef ENABLE_CPU_CORRECTOR_TIMING
            tpa = std::chrono::system_clock::now();
            #endif

            alignmentsComputeWeightsAndAoStoSoA(task);

            buildMultipleSequenceAlignment(task);

            #ifdef ENABLE_CPU_CORRECTOR_TIMING
            timings.msaFindConsensusTimeTotal += std::chrono::system_clock::now() - tpa;
            #endif


            #ifdef ENABLE_CPU_CORRECTOR_TIMING
            tpa = std::chrono::system_clock::now();
            #endif

            refineMSA(task);

            #ifdef ENABLE_CPU_CORRECTOR_TIMING
            timings.msaMinimizationTimeTotal += std::chrono::system_clock::now() - tpa;
            #endif


            #ifdef ENABLE_CPU_CORRECTOR_TIMING
            tpa = std::chrono::system_clock::now();
            #endif

            correctAnchor(task);

            #ifdef ENABLE_CPU_CORRECTOR_TIMING
            timings.msaCorrectSubjectTimeTotal += std::chrono::system_clock::now() - tpa;
            #endif

            if(task.subjectCorrection.isCorrected){
                if(task.msaProperties.isHQ){
                    correctionFlags->setCorrectedAsHqAnchor(task.input.anchorReadId);
                }
            }else{
                correctionFlags->setCouldNotBeCorrectedAsAnchor(task.input.anchorReadId);
            }


            if(task.msaProperties.isHQ && correctionOptions->correctCandidates){

                #ifdef ENABLE_CPU_CORRECTOR_TIMING
                tpa = std::chrono::system_clock::now();
                #endif

                correctCandidates(task);

                #ifdef ENABLE_CPU_CORRECTOR_TIMING
                timings.msaCorrectCandidatesTimeTotal += std::chrono::system_clock::now() - tpa;
                #endif

            }

            resultVector.emplace_back(makeOutputOfTask(task));

        }

        totalTime += timings;

        return resultVector;
    }

    const TimeMeasurements& getTimings() const noexcept{
        return totalTime;
    }

private:

    struct Task{
        bool active{};

        std::vector<read_number> candidateReadIds{};
        std::vector<read_number> filteredReadIds{};
        std::vector<unsigned int> candidateSequencesData{};
        std::vector<unsigned int> candidateSequencesRevcData{};
        std::vector<int> candidateSequencesLengths{};
        std::vector<int> alignmentShifts{};
        std::vector<int> alignmentOps{};
        std::vector<int> alignmentOverlaps{};
        std::vector<float> alignmentWeights{};
        std::vector<char> candidateQualities{};
        std::vector<char> decodedAnchor{};
        std::vector<char> decodedCandidateSequences{};
        std::vector<cpu::SHDResult> alignments{};
        std::vector<cpu::SHDResult> revcAlignments{};
        std::vector<BestAlignment_t> alignmentFlags{};

        CorrectionInput input{};

        CorrectionResult subjectCorrection;
        std::vector<CorrectedCandidate> candidateCorrections;
        MSAProperties msaProperties;
        MultipleSequenceAlignment multipleSequenceAlignment;
    };

    Task makeTask(const CorrectionInput& input){
        Task task;
        task.active = true;
        task.input = input;
        task.multipleSequenceAlignment.setQualityConversion(qualityCoversion.get());

        const int length = input.anchorLength;

        //decode anchor
        task.decodedAnchor.resize(length);
        SequenceHelpers::decode2BitSequence(task.decodedAnchor.data(), input.encodedAnchor, length);

        return task;
    }

    Task makeTask(const MultiCorrectionInput& multiinput, const MultiCandidateIds& multiids, const MultiCandidateData& multicandidateData, int index){
        CorrectionInput input;
        input.anchorLength = multiinput.anchorLengths[index];
        input.anchorReadId = multiinput.anchorReadIds[index];
        input.encodedAnchor = multiinput.encodedAnchors[index];
        input.anchorQualityscores = multiinput.anchorQualityscores[index];

        Task task = makeTask(input);

        const int offsetBegin = multiids.numCandidatesPerAnchorPS[index];
        const int offsetEnd = multiids.numCandidatesPerAnchorPS[index + 1];

        task.candidateReadIds.insert(
            task.candidateReadIds.end(),
            multiids.candidateReadIds.begin() + offsetBegin,
            multiids.candidateReadIds.begin() + offsetEnd
        );
        task.candidateSequencesLengths.insert(
            task.candidateSequencesLengths.end(),
            multicandidateData.candidateLengths.begin() + offsetBegin,
            multicandidateData.candidateLengths.begin() + offsetEnd
        );
        task.candidateSequencesData.insert(
            task.candidateSequencesData.end(),
            multicandidateData.encodedCandidates.begin() + encodedSequencePitchInInts * offsetBegin,
            multicandidateData.encodedCandidates.begin() + encodedSequencePitchInInts * offsetEnd
        );
        
        return task;
    }


    void determineCandidateReadIds(Task& task) const{

        task.candidateReadIds.clear();

        const read_number readId = task.input.anchorReadId;

        //const bool containsN = readProvider->readContainsN(readId);
        bool containsN = false;
        readStorage->areSequencesAmbiguous(readStorageHandle, &containsN, &readId, 1);

        //exclude anchors with ambiguous bases
        if(!(correctionOptions->excludeAmbiguousReads && containsN)){

            assert(task.input.anchorLength == int(task.decodedAnchor.size()));

            // candidateIdsProvider->getCandidates(
            //     task.candidateReadIds,
            //     task.decodedAnchor.data(),
            //     task.decodedAnchor.size()
            // );
            int numPerSequence = 0;
            int maxnumcandidates = 0;
            std::array<int, 2> offsets;

            minhasher->determineNumValues(
                minhashHandle,
                task.input.encodedAnchor,
                encodedSequencePitchInInts,
                &task.input.anchorLength,
                1,
                &numPerSequence,
                maxnumcandidates
            );

            task.candidateReadIds.resize(maxnumcandidates);

            minhasher->retrieveValues(
                minhashHandle,
                nullptr,
                1,
                maxnumcandidates,
                task.candidateReadIds.data(),
                &numPerSequence,
                offsets.data()
            );

            task.candidateReadIds.erase(task.candidateReadIds.begin() + numPerSequence, task.candidateReadIds.end());

            //remove self
            auto readIdPos = std::lower_bound(task.candidateReadIds.begin(),
                                            task.candidateReadIds.end(),
                                            readId);

            if(readIdPos != task.candidateReadIds.end() && *readIdPos == readId){
                task.candidateReadIds.erase(readIdPos);
            }

            auto resultsEnd = task.candidateReadIds.end();
            //exclude candidates with ambiguous bases

            if(correctionOptions->excludeAmbiguousReads){
                resultsEnd = std::remove_if(
                    task.candidateReadIds.begin(),
                    task.candidateReadIds.end(),
                    [&](read_number readId){
                        bool isAmbig = false;
                        readStorage->areSequencesAmbiguous(readStorageHandle, &isAmbig, &readId, 1);
                        return isAmbig;
                    }
                );
            }

            task.candidateReadIds.erase(resultsEnd, task.candidateReadIds.end());
        }
    }



    MultiCandidateIds determineCandidateReadIds(const MultiCorrectionInput& multiInput) const{
        const int numAnchors = multiInput.anchorReadIds.size();

        MultiCandidateIds multiCandidateIds;
        multiCandidateIds.numCandidatesPerAnchor.resize(numAnchors, 0);
        multiCandidateIds.numCandidatesPerAnchorPS.resize(numAnchors + 1, 0);

        for(int i = 0; i < numAnchors; i++){
            const read_number readId = multiInput.anchorReadIds[i];
            const int readlength = multiInput.anchorLengths[i];

            std::vector<char> decodedAnchor(readlength);
            SequenceHelpers::decode2BitSequence(decodedAnchor.data(), multiInput.encodedAnchors[i], readlength);

            bool containsN = false;
            readStorage->areSequencesAmbiguous(readStorageHandle, &containsN, &readId, 1);

            std::vector<read_number> candidateIds;

            //exclude anchors with ambiguous bases
            if(!(correctionOptions->excludeAmbiguousReads && containsN)){

                // candidateIdsProvider->getCandidates(
                //     candidateIds,
                //     decodedAnchor.data(),
                //     decodedAnchor.size()
                // );

                int numPerSequence = 0;
                int maxnumcandidates = 0;
                std::array<int, 2> offsets;

                minhasher->determineNumValues(
                    minhashHandle,
                    multiInput.encodedAnchors[i],
                    encodedSequencePitchInInts,
                    &readlength,
                    1,
                    &numPerSequence,
                    maxnumcandidates
                );

                candidateIds.resize(maxnumcandidates);

                minhasher->retrieveValues(
                    minhashHandle,
                    nullptr,
                    1,
                    maxnumcandidates,
                    candidateIds.data(),
                    &numPerSequence,
                    offsets.data()
                );

                candidateIds.erase(candidateIds.begin() + numPerSequence, candidateIds.end());

                //remove self
                auto readIdPos = std::lower_bound(candidateIds.begin(),
                                                candidateIds.end(),
                                                readId);

                if(readIdPos != candidateIds.end() && *readIdPos == readId){
                    candidateIds.erase(readIdPos);
                }

                auto resultsEnd = candidateIds.end();
                //exclude candidates with ambiguous bases

                if(correctionOptions->excludeAmbiguousReads){
                    resultsEnd = std::remove_if(
                        candidateIds.begin(),
                        candidateIds.end(),
                        [&](read_number readId){
                            bool isAmbig = false;
                            readStorage->areSequencesAmbiguous(readStorageHandle, &isAmbig, &readId, 1);
                            return isAmbig;
                        }
                    );
                }

                candidateIds.erase(resultsEnd, candidateIds.end());
            }
        
            multiCandidateIds.numCandidatesPerAnchor[i] = candidateIds.size();
            multiCandidateIds.numCandidatesPerAnchorPS[i + 1] = 
                multiCandidateIds.numCandidatesPerAnchorPS[i] + candidateIds.size();
            multiCandidateIds.candidateReadIds.insert(multiCandidateIds.candidateReadIds.end(), candidateIds.begin(), candidateIds.end());
        }

        return multiCandidateIds;
    }



    MultiCandidateData getCandidateSequencesData(const MultiCandidateIds& multiIds) const{
        const int numIds = multiIds.candidateReadIds.size();
        if(numIds == 0){
            return MultiCandidateData{};
        }

        MultiCandidateData multiData;
        multiData.candidateLengths.resize(numIds);
        multiData.encodedCandidates.resize(numIds * encodedSequencePitchInInts);
        multiData.candidateQualities.resize(numIds * qualityPitchInBytes);

        readStorage->gatherSequenceLengths(
            readStorageHandle,
            multiData.candidateLengths.data(),
            multiIds.candidateReadIds.data(),
            numIds
        );

        readStorage->gatherSequences(
            readStorageHandle,
            multiData.encodedCandidates.data(),
            encodedSequencePitchInInts,
            multiIds.candidateReadIds.data(),
            numIds
        ); 

        if(correctionOptions->useQualityScores){

            readStorage->gatherQualities(
                readStorageHandle,
                multiData.candidateQualities.data(),
                qualityPitchInBytes,
                multiIds.candidateReadIds.data(),
                numIds
            ); 
        }

        return multiData;
    }


    void getQualitiesFromMultiCandidates(Task& task, const MultiCandidateIds& multiids, const MultiCandidateData& multicandidateData, int index) const{
        const int offsetBegin = multiids.numCandidatesPerAnchorPS[index];
        const int offsetEnd = multiids.numCandidatesPerAnchorPS[index + 1];

        const int numCandidates = task.candidateReadIds.size();
        task.candidateQualities.resize(qualityPitchInBytes * numCandidates);

        auto first1 = task.candidateReadIds.cbegin();
        auto last1 = task.candidateReadIds.cend();
        auto first2 = multiids.candidateReadIds.cbegin() + offsetBegin;
        auto last2 = multiids.candidateReadIds.cbegin() + offsetEnd;

        auto qualOutputIter = task.candidateQualities.begin();
        auto qualInputIter = multicandidateData.candidateQualities.cbegin() + offsetBegin * qualityPitchInBytes;

        //copy quality scores of candidates which have not been removed (set_intersection, range 1 is a subset of range2)
        while (first1 != last1 && first2 != last2) {
            if (*first1 < *first2) {
                ++first1;
            } else  {
                if (!(*first2 < *first1)) {
                    qualOutputIter = std::copy_n(qualInputIter, qualityPitchInBytes, qualOutputIter);
                }
                ++first2;
                qualInputIter += qualityPitchInBytes;
            }
        }
    }


    //Gets forward sequence and reverse complement sequence of each candidate
    void getCandidateSequenceData(Task& task) const{

        const int numCandidates = task.candidateReadIds.size();
        
        if(numCandidates == 0) return;

        task.candidateSequencesLengths.resize(numCandidates);
        task.candidateSequencesData.clear();
        task.candidateSequencesData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);
        task.candidateSequencesRevcData.clear();
        task.candidateSequencesRevcData.resize(size_t(encodedSequencePitchInInts) * numCandidates, 0);

        readStorage->gatherSequenceLengths(
            readStorageHandle,
            task.candidateSequencesLengths.data(),
            task.candidateReadIds.data(),
            numCandidates
        );

        readStorage->gatherSequences(
            readStorageHandle,
            task.candidateSequencesData.data(),
            encodedSequencePitchInInts,
            task.candidateReadIds.data(),
            numCandidates
        );        
    }

    void computeReverseComplementCandidates(Task& task){
        const int numCandidates = task.candidateReadIds.size();
        task.candidateSequencesRevcData.resize(task.candidateSequencesData.size());

        for(int i = 0; i < numCandidates; i++){
            const unsigned int* const seqPtr = task.candidateSequencesData.data() 
                                                + std::size_t(encodedSequencePitchInInts) * i;
            unsigned int* const seqrevcPtr = task.candidateSequencesRevcData.data() 
                                                + std::size_t(encodedSequencePitchInInts) * i;

            SequenceHelpers::reverseComplementSequence2Bit(
                seqrevcPtr,  
                seqPtr,
                task.candidateSequencesLengths[i]
            );
        }
    } 

    //compute alignments between anchor sequence and candidate sequences
    void getCandidateAlignments(Task& task) const{
        const int numCandidates = task.candidateReadIds.size();

        task.alignments.resize(numCandidates);
        task.revcAlignments.resize(numCandidates);
        task.alignmentFlags.resize(numCandidates);

        cpu::shd::cpuShiftedHammingDistancePopcount2Bit(
            alignmentHandle,
            task.alignments.begin(),
            task.input.encodedAnchor,
            task.input.anchorLength,
            task.candidateSequencesData.data(),
            encodedSequencePitchInInts,
            task.candidateSequencesLengths.data(),
            numCandidates,
            goodAlignmentProperties->min_overlap,
            goodAlignmentProperties->maxErrorRate,
            goodAlignmentProperties->min_overlap_ratio
        );

        cpu::shd::cpuShiftedHammingDistancePopcount2Bit(
            alignmentHandle,
            task.revcAlignments.begin(),
            task.input.encodedAnchor,
            task.input.anchorLength,
            task.candidateSequencesRevcData.data(),
            encodedSequencePitchInInts,
            task.candidateSequencesLengths.data(),
            numCandidates,
            goodAlignmentProperties->min_overlap,
            goodAlignmentProperties->maxErrorRate,
            goodAlignmentProperties->min_overlap_ratio
        );

        //decide whether to keep forward or reverse complement

        for(int i = 0; i < numCandidates; i++){
            const auto& forwardAlignment = task.alignments[i];
            const auto& revcAlignment = task.revcAlignments[i];
            const int candidateLength = task.candidateSequencesLengths[i];

            BestAlignment_t bestAlignmentFlag = care::choose_best_alignment(
                forwardAlignment,
                revcAlignment,
                task.input.anchorLength,
                candidateLength,
                goodAlignmentProperties->min_overlap_ratio,
                goodAlignmentProperties->min_overlap,
                correctionOptions->estimatedErrorrate
            );

            task.alignmentFlags[i] = bestAlignmentFlag;
        }
    }

    //remove candidates with alignment flag None
    void filterCandidatesByAlignmentFlag(Task& task) const{

        const int numCandidates = task.candidateReadIds.size();

        int insertpos = 0;

        for(int i = 0; i < numCandidates; i++){

            const BestAlignment_t flag = task.alignmentFlags[i];

            if(flag == BestAlignment_t::Forward){
                task.candidateReadIds[insertpos] = task.candidateReadIds[i];
                std::copy_n(
                    task.candidateSequencesData.data() + i * size_t(encodedSequencePitchInInts),
                    encodedSequencePitchInInts,
                    task.candidateSequencesData.data() + insertpos * size_t(encodedSequencePitchInInts)
                );
                task.candidateSequencesLengths[insertpos] = task.candidateSequencesLengths[i];
                task.alignmentFlags[insertpos] = task.alignmentFlags[i];
                task.alignments[insertpos] = task.alignments[i];                

                insertpos++;
            }else if(flag == BestAlignment_t::ReverseComplement){
                task.candidateReadIds[insertpos] = task.candidateReadIds[i];
                std::copy_n(
                    task.candidateSequencesRevcData.data() + i * size_t(encodedSequencePitchInInts),
                    encodedSequencePitchInInts,
                    task.candidateSequencesData.data() + insertpos * size_t(encodedSequencePitchInInts)
                );
                task.candidateSequencesLengths[insertpos] = task.candidateSequencesLengths[i];
                task.alignmentFlags[insertpos] = task.alignmentFlags[i];
                task.alignments[insertpos] = task.revcAlignments[i];                

                insertpos++;
            }else{
                ;//BestAlignment_t::None discard alignment
            }
        }

        task.candidateReadIds.erase(
            task.candidateReadIds.begin() + insertpos, 
            task.candidateReadIds.end()
        );
        task.candidateSequencesData.erase(
            task.candidateSequencesData.begin() + encodedSequencePitchInInts * insertpos, 
            task.candidateSequencesData.end()
        );
        task.candidateSequencesLengths.erase(
            task.candidateSequencesLengths.begin() + insertpos, 
            task.candidateSequencesLengths.end()
        );
        task.alignmentFlags.erase(
            task.alignmentFlags.begin() + insertpos, 
            task.alignmentFlags.end()
        );
        task.alignments.erase(
            task.alignments.begin() + insertpos, 
            task.alignments.end()
        );

        task.revcAlignments.clear();
        task.candidateSequencesRevcData.clear();
    }

    //remove candidates with bad alignment mismatch ratio
    void filterCandidatesByAlignmentMismatchRatio(Task& task) const{
        
        auto lastResortFunc = [](){
            return false;
        };

        const float mismatchratioBaseFactor = correctionOptions->estimatedErrorrate * 1.0f;
        const float goodAlignmentsCountThreshold = correctionOptions->estimatedCoverage * correctionOptions->m_coverage;
        const int numCandidates = task.candidateReadIds.size();

        std::array<int, 3> counts({0,0,0});

        for(int i = 0; i < numCandidates; i++){
            const auto& alignment = task.alignments[i];
            const float mismatchratio = float(alignment.nOps) / float(alignment.overlap);

            if (mismatchratio < 2 * mismatchratioBaseFactor) {
                counts[0] += 1;
            }
            if (mismatchratio < 3 * mismatchratioBaseFactor) {
                counts[1] += 1;
            }
            if (mismatchratio < 4 * mismatchratioBaseFactor) {
                counts[2] += 1;
            }
        }

        float mismatchratioThreshold = std::numeric_limits<float>::lowest();
        if( std::any_of(counts.begin(), counts.end(), [](auto c){return c > 0;}) ){
            
            if (counts[0] >= goodAlignmentsCountThreshold) {
                mismatchratioThreshold = 2 * mismatchratioBaseFactor;
            } else if (counts[1] >= goodAlignmentsCountThreshold) {
                mismatchratioThreshold = 3 * mismatchratioBaseFactor;
            } else if (counts[2] >= goodAlignmentsCountThreshold) {
                mismatchratioThreshold = 4 * mismatchratioBaseFactor;
            } else {
                if(lastResortFunc()){
                    mismatchratioThreshold = 4 * mismatchratioBaseFactor;
                }else{
                    ; //mismatchratioThreshold = std::numeric_limits<float>::lowest();
                }
            }
        }

        int insertpos = 0;
        for(int i = 0; i < numCandidates; i++){
            const auto& alignment = task.alignments[i];
            const float mismatchratio = float(alignment.nOps) / float(alignment.overlap);
            const bool notremoved = mismatchratio < mismatchratioThreshold;

            if(notremoved){
                task.candidateReadIds[insertpos] = task.candidateReadIds[i];
                std::copy_n(
                    task.candidateSequencesData.data() + i * size_t(encodedSequencePitchInInts),
                    encodedSequencePitchInInts,
                    task.candidateSequencesData.data() + insertpos * size_t(encodedSequencePitchInInts)
                );
                task.candidateSequencesLengths[insertpos] = task.candidateSequencesLengths[i];
                task.alignmentFlags[insertpos] = task.alignmentFlags[i];
                task.alignments[insertpos] = task.alignments[i]; 

                insertpos++;
            }
        }

        task.candidateReadIds.erase(
            task.candidateReadIds.begin() + insertpos, 
            task.candidateReadIds.end()
        );
        task.candidateSequencesData.erase(
            task.candidateSequencesData.begin() + encodedSequencePitchInInts * insertpos, 
            task.candidateSequencesData.end()
        );
        task.candidateSequencesLengths.erase(
            task.candidateSequencesLengths.begin() + insertpos, 
            task.candidateSequencesLengths.end()
        );
        task.alignmentFlags.erase(
            task.alignmentFlags.begin() + insertpos, 
            task.alignmentFlags.end()
        );
        task.alignments.erase(
            task.alignments.begin() + insertpos, 
            task.alignments.end()
        );
    }

    //get quality scores of candidates with respect to alignment direction
    void getCandidateQualities(Task& task) const{
        const int numCandidates = task.candidateReadIds.size();

        task.candidateQualities.resize(qualityPitchInBytes * numCandidates);

        readStorage->gatherQualities(
            readStorageHandle,
            task.candidateQualities.data(),
            qualityPitchInBytes,
            task.candidateReadIds.data(),
            numCandidates
        );         
    }

    void reverseQualitiesOfRCAlignments(Task& task){
        const int numCandidates = task.candidateReadIds.size();

        //reverse quality scores of candidates with reverse complement alignment
        for(int c = 0; c < numCandidates; c++){
            if(task.alignmentFlags[c] == BestAlignment_t::ReverseComplement){
                std::reverse(
                    task.candidateQualities.data() + c * size_t(qualityPitchInBytes),
                    task.candidateQualities.data() + (c+1) * size_t(qualityPitchInBytes)
                );
            }
        } 
    }

    //compute decoded candidate strings with respect to alignment direction
    void makeCandidateStrings(Task& task) const{
        const int numCandidates = task.candidateReadIds.size();

        task.decodedCandidateSequences.resize(decodedSequencePitchInBytes * numCandidates);
        
        for(int i = 0; i < numCandidates; i++){
            const unsigned int* const srcptr = task.candidateSequencesData.data() + i * encodedSequencePitchInInts;
            char* const destptr = task.decodedCandidateSequences.data() + i * decodedSequencePitchInBytes;
            const int length = task.candidateSequencesLengths[i];

            SequenceHelpers::decode2BitSequence(
                destptr,
                srcptr,
                length
            );
        }
    }

    void alignmentsComputeWeightsAndAoStoSoA(Task& task) const{
        const int numCandidates = task.candidateReadIds.size();

        task.alignmentShifts.resize(numCandidates);
        task.alignmentOps.resize(numCandidates);
        task.alignmentOverlaps.resize(numCandidates);
        task.alignmentWeights.resize(numCandidates);

        for(int i = 0; i < numCandidates; i++){
            task.alignmentShifts[i] = task.alignments[i].shift;
            task.alignmentOps[i] = task.alignments[i].nOps;
            task.alignmentOverlaps[i] = task.alignments[i].overlap;

            task.alignmentWeights[i] = calculateOverlapWeight(
                task.input.anchorLength,
                task.alignments[i].nOps, 
                task.alignments[i].overlap,
                goodAlignmentProperties->maxErrorRate
            );
        }
    }

    void buildMultipleSequenceAlignment(Task& task) const{

        const int numCandidates = task.candidateReadIds.size();

        const char* const candidateQualityPtr = correctionOptions->useQualityScores ?
                                                task.candidateQualities.data()
                                                : nullptr;

        MultipleSequenceAlignment::InputData buildArgs;
        buildArgs.useQualityScores = correctionOptions->useQualityScores;
        buildArgs.subjectLength = task.input.anchorLength;
        buildArgs.nCandidates = numCandidates;
        buildArgs.candidatesPitch = decodedSequencePitchInBytes;
        buildArgs.candidateQualitiesPitch = qualityPitchInBytes;
        buildArgs.subject = task.decodedAnchor.data();
        buildArgs.candidates = task.decodedCandidateSequences.data();
        buildArgs.subjectQualities = task.input.anchorQualityscores;
        buildArgs.candidateQualities = candidateQualityPtr;
        buildArgs.candidateLengths = task.candidateSequencesLengths.data();
        buildArgs.candidateShifts = task.alignmentShifts.data();
        buildArgs.candidateDefaultWeightFactors = task.alignmentWeights.data();
    
        task.multipleSequenceAlignment.build(buildArgs);
    }

    void refineMSA(Task& task) const{

        constexpr int max_num_minimizations = 5;

        auto removeCandidatesOfDifferentRegion = [&](const auto& minimizationResult){
            const int numCandidates = task.candidateReadIds.size();

            int insertpos = 0;
            for(int i = 0; i < numCandidates; i++){
                if(!minimizationResult.differentRegionCandidate[i]){                        
                    //keep candidate

                    task.candidateReadIds[insertpos] = task.candidateReadIds[i];

                    std::copy_n(
                        task.candidateSequencesData.data() + i * size_t(encodedSequencePitchInInts),
                        encodedSequencePitchInInts,
                        task.candidateSequencesData.data() + insertpos * size_t(encodedSequencePitchInInts)
                    );

                    task.candidateSequencesLengths[insertpos] = task.candidateSequencesLengths[i];
                    task.alignmentFlags[insertpos] = task.alignmentFlags[i];
                    task.alignments[insertpos] = task.alignments[i]; 
                    task.alignmentOps[insertpos] = task.alignmentOps[i];
                    task.alignmentShifts[insertpos] = task.alignmentShifts[i];
                    task.alignmentOverlaps[insertpos] = task.alignmentOverlaps[i];
                    task.alignmentWeights[insertpos] = task.alignmentWeights[i];

                    if(correctionOptions->useQualityScores){
                        std::copy_n(
                            task.candidateQualities.data() + i * size_t(qualityPitchInBytes),
                            qualityPitchInBytes,
                            task.candidateQualities.data() + insertpos * size_t(qualityPitchInBytes)
                        );
                    }

                    std::copy_n(
                        task.decodedCandidateSequences.data() + i * size_t(decodedSequencePitchInBytes),
                        decodedSequencePitchInBytes,
                        task.decodedCandidateSequences.data() + insertpos * size_t(decodedSequencePitchInBytes)
                    );

                    insertpos++;
                }
            }

            task.candidateReadIds.erase(
                task.candidateReadIds.begin() + insertpos, 
                task.candidateReadIds.end()
            );
            task.candidateSequencesData.erase(
                task.candidateSequencesData.begin() + encodedSequencePitchInInts * insertpos, 
                task.candidateSequencesData.end()
            );
            task.candidateSequencesLengths.erase(
                task.candidateSequencesLengths.begin() + insertpos, 
                task.candidateSequencesLengths.end()
            );
            task.alignmentFlags.erase(
                task.alignmentFlags.begin() + insertpos, 
                task.alignmentFlags.end()
            );
            task.alignments.erase(
                task.alignments.begin() + insertpos, 
                task.alignments.end()
            );

            if(correctionOptions->useQualityScores){
                task.candidateQualities.erase(
                    task.candidateQualities.begin() + qualityPitchInBytes * insertpos, 
                    task.candidateQualities.end()
                );
            }
            
            task.decodedCandidateSequences.erase(
                task.decodedCandidateSequences.begin() + decodedSequencePitchInBytes * insertpos, 
                task.decodedCandidateSequences.end()
            );
            task.alignmentOps.erase(
                task.alignmentOps.begin() + insertpos, 
                task.alignmentOps.end()
            );
            task.alignmentShifts.erase(
                task.alignmentShifts.begin() + insertpos, 
                task.alignmentShifts.end()
            );
            task.alignmentOverlaps.erase(
                task.alignmentOverlaps.begin() + insertpos, 
                task.alignmentOverlaps.end()
            );
            task.alignmentWeights.erase(
                task.alignmentWeights.begin() + insertpos, 
                task.alignmentWeights.end()
            );
        };

        if(max_num_minimizations > 0){                

            for(int numIterations = 0; numIterations < max_num_minimizations; numIterations++){
                const auto minimizationResult = task.multipleSequenceAlignment.findCandidatesOfDifferentRegion(
                    correctionOptions->estimatedCoverage
                );

                if(minimizationResult.performedMinimization){
                    removeCandidatesOfDifferentRegion(minimizationResult);

                    //build minimized multiple sequence alignment
                    buildMultipleSequenceAlignment(task);
                }else{
                    break;
                }               
                
            }
        }      
    }

    void correctAnchorClassic(Task& task) const{

        const int subjectColumnsBegin_incl = task.multipleSequenceAlignment.subjectColumnsBegin_incl;
        const int subjectColumnsEnd_excl = task.multipleSequenceAlignment.subjectColumnsEnd_excl;

        task.msaProperties = task.multipleSequenceAlignment.getMSAProperties(
            subjectColumnsBegin_incl,
            subjectColumnsEnd_excl,
            correctionOptions->estimatedErrorrate,
            correctionOptions->estimatedCoverage,
            correctionOptions->m_coverage
        );

        task.subjectCorrection = task.multipleSequenceAlignment.getCorrectedSubject(
            task.msaProperties,
            correctionOptions->estimatedErrorrate,
            correctionOptions->estimatedCoverage,
            correctionOptions->m_coverage,
            correctionOptions->kmerlength,
            task.input.anchorReadId
        );        
    }       

    void correctAnchor(Task& task) const{
        correctAnchorClassic(task);
    }

    void correctCandidatesClassic(Task& task) const{

        task.candidateCorrections = task.multipleSequenceAlignment.getCorrectedCandidates(
            correctionOptions->estimatedErrorrate,
            correctionOptions->estimatedCoverage,
            correctionOptions->m_coverage,
            correctionOptions->new_columns_to_correct
        );
    }

    void correctCandidates(Task& task) const{
        correctCandidatesClassic(task);
    }

    CorrectionOutput makeOutputOfTask(Task& task) const{
        CorrectionOutput result;

        result.hasAnchorCorrection = task.subjectCorrection.isCorrected;

        if(result.hasAnchorCorrection){
            auto& correctedSequenceString = task.subjectCorrection.correctedSequence;
            const int correctedlength = correctedSequenceString.length();
            bool originalReadContainsN = false;
            readStorage->areSequencesAmbiguous(readStorageHandle, &originalReadContainsN, &task.input.anchorReadId, 1);
            
            TempCorrectedSequence tmp;
            
            if(!originalReadContainsN){
                const int maxEdits = correctedlength / 7;
                int edits = 0;
                for(int i = 0; i < correctedlength && edits <= maxEdits; i++){
                    if(correctedSequenceString[i] != task.decodedAnchor[i]){
                        tmp.edits.emplace_back(i, correctedSequenceString[i]);
                        edits++;
                    }
                }
                tmp.useEdits = edits <= maxEdits;
            }else{
                tmp.useEdits = false;
            }
            
            tmp.hq = task.msaProperties.isHQ;
            tmp.type = TempCorrectedSequence::Type::Anchor;
            tmp.readId = task.input.anchorReadId;
            tmp.sequence = std::move(correctedSequenceString); 
            
            result.anchorCorrection = std::move(tmp);
        }
        
        
        
        for(const auto& correctedCandidate : task.candidateCorrections){
            const read_number candidateId = task.candidateReadIds[correctedCandidate.index];
            
            //don't save candidate if hq anchor correction exists
            bool savingIsOk = !correctionFlags->isCorrectedAsHQAnchor(candidateId);
                        
            if (savingIsOk) {                            
                
                TempCorrectedSequence tmp;
                
                tmp.type = TempCorrectedSequence::Type::Candidate;
                tmp.readId = candidateId;
                tmp.shift = correctedCandidate.shift;

                const bool candidateIsForward = task.alignmentFlags[correctedCandidate.index] == BestAlignment_t::Forward;

                if(candidateIsForward){
                    tmp.sequence = std::move(correctedCandidate.sequence);
                }else{
                    //if input candidate for correction is reverse complement, corrected candidate is also reverse complement
                    //get forward sequence
                    std::string fwd;
                    fwd.resize(correctedCandidate.sequence.length());
                    SequenceHelpers::reverseComplementSequenceDecoded(
                        &fwd[0], 
                        correctedCandidate.sequence.c_str(), 
                                            correctedCandidate.sequence.length()
                    );
                    tmp.sequence = std::move(fwd);
                }
                
                bool originalCandidateReadContainsN = false;
                readStorage->areSequencesAmbiguous(readStorageHandle, &originalCandidateReadContainsN, &candidateId, 1);
                
                if(!originalCandidateReadContainsN){
                    const std::size_t offset = correctedCandidate.index * decodedSequencePitchInBytes;
                    const char* const uncorrectedCandidate = &task.decodedCandidateSequences[offset];
                    const int uncorrectedCandidateLength = task.candidateSequencesLengths[correctedCandidate.index];
                    const int correctedCandidateLength = tmp.sequence.length();
                    
                    assert(uncorrectedCandidateLength == correctedCandidateLength);
                    
                    const int maxEdits = correctedCandidateLength / 7;
                    int edits = 0;
                    if(candidateIsForward){
                        for(int pos = 0; pos < correctedCandidateLength && edits <= maxEdits; pos++){
                            if(tmp.sequence[pos] != uncorrectedCandidate[pos]){
                                tmp.edits.emplace_back(pos, tmp.sequence[pos]);
                                edits++;
                            }
                        }
                    }else{
                        //tmp.sequence is forward sequence, but uncorrectedCandidate is reverse complement
                        std::string fwduncorrected;
                        fwduncorrected.resize(uncorrectedCandidateLength);
                        SequenceHelpers::reverseComplementSequenceDecoded(
                            &fwduncorrected[0], 
                            uncorrectedCandidate, 
                            uncorrectedCandidateLength
                        );

                        for(int pos = 0; pos < correctedCandidateLength && edits <= maxEdits; pos++){
                            if(tmp.sequence[pos] != fwduncorrected[pos]){
                                tmp.edits.emplace_back(pos, tmp.sequence[pos]);
                                edits++;
                            }
                        }
                    }
                    

                    tmp.useEdits = edits <= maxEdits;
                }else{
                    tmp.useEdits = false;
                }
                
                result.candidateCorrections.emplace_back(std::move(tmp));
            }
        }

        return result;
    }


private:

    std::size_t encodedSequencePitchInInts{};
    std::size_t decodedSequencePitchInBytes{};
    std::size_t qualityPitchInBytes{};

    const CorrectionOptions* correctionOptions{};
    const GoodAlignmentProperties* goodAlignmentProperties{};
    const CpuMinhasher* minhasher{};
    mutable CpuMinhasher::QueryHandle minhashHandle;
    const CpuReadStorage* readStorage{};
    mutable ReadStorageHandle readStorageHandle;

    ReadCorrectionFlags* correctionFlags{};
  
    mutable cpu::shd::CpuAlignmentHandle alignmentHandle{};

    std::unique_ptr<cpu::QualityScoreConversion> qualityCoversion;

    TimeMeasurements totalTime{};
};



}



#endif