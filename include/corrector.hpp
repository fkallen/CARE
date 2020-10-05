#ifndef CARE_CORRECTOR_HPP
#define CARE_CORRECTOR_HPP

#include <config.hpp>

#include <options.hpp>

#include <minhasher.hpp>
#include <readstorage.hpp>
#include <cpu_alignment.hpp>
#include <bestalignment.hpp>
#include <msa.hpp>
#include <qualityscoreweights.hpp>
#include <forest.hpp>
#include <correctionresultprocessing.hpp>
#include <hostdevicefunctions.cuh>

#include <cstddef>
#include <memory>
#include <sstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>

namespace care{

class CpuErrorCorrector{
public:
    struct CorrectionInput{
        int anchorLength{};
        read_number anchorReadId{};
        const unsigned int* encodedAnchor{};
        const char* anchorQualityscores{};
    };

    struct CorrectionOutput{
        bool hasAnchorCorrection{};
        TempCorrectedSequence anchorCorrection{};
        std::vector<TempCorrectedSequence> candidateCorrections{};
    };

    struct ReadCorrectionFlags{
        friend class CpuErrorCorrector;
    public:
        ReadCorrectionFlags() = default;

        ReadCorrectionFlags(std::size_t numReads)
            : size(numReads), flags(std::make_unique<std::uint8_t[]>(numReads)){
            std::fill(flags.get(), flags.get() + size, 0);
        }

        std::size_t sizeInBytes() const noexcept{
            return size * sizeof(std::uint8_t);
        }

    private:
        static constexpr std::uint8_t readCorrectedAsHQAnchor() noexcept{ return 1; };
        static constexpr std::uint8_t readCouldNotBeCorrectedAsAnchor() noexcept{ return 2; };

        void setCorrectedAsHqAnchor(std::int64_t position) const noexcept{
            flags[position] = readCorrectedAsHQAnchor();
        }

        void setCouldNotBeCorrectedAsAnchor(std::int64_t position) const noexcept{
            flags[position] = readCouldNotBeCorrectedAsAnchor();
        }

        std::uint8_t getFlag(std::int64_t position) const noexcept{
            return flags[position];
        }

        std::size_t size;
        std::unique_ptr<std::uint8_t[]> flags{};
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

    struct MLSettings{
        std::shared_ptr<ForestClf> classifier_anchor, classifier_cands;
        std::mt19937 rndGenerator;
    };

    CpuErrorCorrector() = default;
    CpuErrorCorrector(
        std::size_t encodedSequencePitchInInts_,
        std::size_t decodedSequencePitchInBytes_,
        std::size_t qualityPitchInBytes_,
        const CorrectionOptions& correctionOptions_,
        const GoodAlignmentProperties& goodAlignmentProperties_,
        const Minhasher& minhasher_,
        const cpu::ContiguousReadStorage& readStorage_,
        ReadCorrectionFlags& correctionFlags_,
        MLSettings* mlSettings_
    ) : encodedSequencePitchInInts(encodedSequencePitchInInts_),
        decodedSequencePitchInBytes(decodedSequencePitchInBytes_),
        qualityPitchInBytes(qualityPitchInBytes_),
        correctionOptions(&correctionOptions_),
        goodAlignmentProperties(&goodAlignmentProperties_),
        minhasher(&minhasher_),
        readStorage(&readStorage_),
        correctionFlags(&correctionFlags_),
        mlSettings(mlSettings_)
    {

    }

    CorrectionOutput process(const CorrectionInput input){
        Task task = makeTask(input);

        TimeMeasurements timings;

        #ifdef ENABLE_TIMING
        auto tpa = std::chrono::system_clock::now();
        #endif

        determineCandidateReadIds(task);

        #ifdef ENABLE_TIMING
        timings.getCandidatesTimeTotal += std::chrono::system_clock::now() - tpa;
        #endif


        if(task.candidateReadIds.size() == 0){
            //TODO return uncorrected anchor
        }

        #ifdef ENABLE_TIMING
        tpa = std::chrono::system_clock::now();
        #endif

        getCandidateSequenceData(task);

        #ifdef ENABLE_TIMING
        timings.copyCandidateDataToBufferTimeTotal += std::chrono::system_clock::now() - tpa;
        #endif


        #ifdef ENABLE_TIMING
        tpa = std::chrono::system_clock::now();
        #endif

        getCandidateAlignments(task);

        #ifdef ENABLE_TIMING
        timings.getAlignmentsTimeTotal += std::chrono::system_clock::now() - tpa;
        #endif


        #ifdef ENABLE_TIMING
        tpa = std::chrono::system_clock::now();
        #endif

        filterCandidatesByAlignmentFlag(task);

        #ifdef ENABLE_TIMING
        timings.gatherBestAlignmentDataTimeTotal += std::chrono::system_clock::now() - tpa;
        #endif


        if(task.candidateReadIds.size() == 0){
            //TODO return uncorrected anchor
        }

        #ifdef ENABLE_TIMING
        tpa = std::chrono::system_clock::now();
        #endif

        filterCandidatesByAlignmentMismatchRatio(task);

        #ifdef ENABLE_TIMING
        timings.mismatchRatioFilteringTimeTotal += std::chrono::system_clock::now() - tpa;
        #endif


        if(task.candidateReadIds.size() == 0){
            //TODO return uncorrected anchor
        }

        if(correctionOptions->useQualityScores){

            #ifdef ENABLE_TIMING
            tpa = std::chrono::system_clock::now();
            #endif

            getCandidateQualities(task);

            #ifdef ENABLE_TIMING
            timings.fetchQualitiesTimeTotal += std::chrono::system_clock::now() - tpa;
            #endif

        }

        #ifdef ENABLE_TIMING
        tpa = std::chrono::system_clock::now();
        #endif

        makeCandidateStrings(task);

        #ifdef ENABLE_TIMING
        timings.makeCandidateStringsTimeTotal += std::chrono::system_clock::now() - tpa;
        #endif


        #ifdef ENABLE_TIMING
        tpa = std::chrono::system_clock::now();
        #endif

        alignmentsComputeWeightsAndAoStoSoA(task);

        buildMultipleSequenceAlignment(task);

        #ifdef ENABLE_TIMING
        timings.msaFindConsensusTimeTotal += std::chrono::system_clock::now() - tpa;
        #endif


        #ifdef ENABLE_TIMING
        tpa = std::chrono::system_clock::now();
        #endif

        removeCandidatesOfDifferentRegionByMSA(task);

        #ifdef ENABLE_TIMING
        timings.msaMinimizationTimeTotal += std::chrono::system_clock::now() - tpa;
        #endif


        #ifdef ENABLE_TIMING
        tpa = std::chrono::system_clock::now();
        #endif

        correctAnchor(task);

        #ifdef ENABLE_TIMING
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

            #ifdef ENABLE_TIMING
            tpa = std::chrono::system_clock::now();
            #endif

            correctCandidates(task);

            #ifdef ENABLE_TIMING
            batchData.timings.msaCorrectCandidatesTimeTotal += std::chrono::system_clock::now() - tpa;
            #endif

        }

        CorrectionOutput correctionOutput = makeOutputOfTask(task);

        totalTime += timings;

        return correctionOutput;
    }

    const TimeMeasurements& getTimings() const noexcept{
        return totalTime;
    }

    std::stringstream& getMlStreamAnchor(){
        return ml_stream_anchor;
    }

    std::stringstream& getMlStreamCandidates(){
        return ml_stream_cands;
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
        return task;
    }

    void determineCandidateReadIds(Task& task) const{

        task.candidateReadIds.clear();

        const read_number readId = task.input.anchorReadId;

        const bool containsN = readStorage->readContainsN(readId);

        //exclude anchors with ambiguous bases
        if(!(correctionOptions->excludeAmbiguousReads && containsN)){

            const int length = task.input.anchorLength;
            char* const decodedBegin = task.decodedAnchor.data();

            decode2BitSequence(decodedBegin, task.input.encodedAnchor, length);

            //TODO modify minhasher to work with char ptr + size instead of string
            std::string sequence(decodedBegin, length);

            minhasher->getCandidates_any_map(
                minhashHandle,
                sequence,
                0
            );

            auto readIdPos = std::lower_bound(minhashHandle.result().begin(),
                                            minhashHandle.result().end(),
                                            readId);

            if(readIdPos != minhashHandle.result().end() && *readIdPos == readId){
                minhashHandle.result().erase(readIdPos);
            }

            auto minhashResultsEnd = minhashHandle.result().end();
            //exclude candidates with ambiguous bases

            if(correctionOptions->excludeAmbiguousReads){
                minhashResultsEnd = std::remove_if(
                    minhashHandle.result().begin(),
                    minhashHandle.result().end(),
                    [&](read_number readId){
                        return readStorage->readContainsN(readId);
                    }
                );
            }

            task.candidateReadIds.insert(
                task.candidateReadIds.end(),
                minhashHandle.result().begin(),
                minhashResultsEnd
            );

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
            readStorageGatherHandle,
            task.candidateReadIds.data(),
            numCandidates,
            task.candidateSequencesLengths.data()
        );

        readStorage->gatherSequenceData(
            readStorageGatherHandle,
            task.candidateReadIds.data(),
            numCandidates,
            task.candidateSequencesData.data(),
            encodedSequencePitchInInts
        );

        for(int i = 0; i < numCandidates; i++){
            const unsigned int* const seqPtr = task.candidateSequencesData.data() 
                                                + std::size_t(encodedSequencePitchInInts) * i;
            unsigned int* const seqrevcPtr = task.candidateSequencesRevcData.data() 
                                                + std::size_t(encodedSequencePitchInInts) * i;

            reverseComplement2Bit(
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

        float mismatchratioThreshold = std::numeric_limits<float>::min();
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
                    ; //mismatchratioThreshold = std::numeric_limits<float>::min();
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

        readStorage->gatherSequenceQualities(
            readStorageGatherHandle,
            task.candidateReadIds.data(),
            task.candidateReadIds.size(),
            task.candidateQualities.data(),
            qualityPitchInBytes
        );
        
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

        for(int i = 0; i < numCandidates; i++){
            const unsigned int* const srcptr = task.candidateSequencesData.data() + i * encodedSequencePitchInInts;
            char* const destptr = task.decodedCandidateSequences.data() + i * decodedSequencePitchInBytes;
            const int length = task.candidateSequencesLengths[i];

            decode2BitSequence(
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

    void removeCandidatesOfDifferentRegionByMSA(Task& task) const{

        constexpr int max_num_minimizations = 5;

        auto findCandidatesLambda = [&](){
            return findCandidatesOfDifferentRegion(
                task.decodedAnchor.data(),
                task.input.anchorLength,
                task.decodedCandidateSequences.data(),
                task.candidateSequencesLengths.data(),
                task.candidateReadIds.size(),
                decodedSequencePitchInBytes,
                task.multipleSequenceAlignment.consensus.data(),
                task.multipleSequenceAlignment.countsA.data(),
                task.multipleSequenceAlignment.countsC.data(),
                task.multipleSequenceAlignment.countsG.data(),
                task.multipleSequenceAlignment.countsT.data(),
                task.multipleSequenceAlignment.weightsA.data(),
                task.multipleSequenceAlignment.weightsC.data(),
                task.multipleSequenceAlignment.weightsG.data(),
                task.multipleSequenceAlignment.weightsT.data(),
                task.alignmentOps.data(), 
                task.alignmentOverlaps.data(),
                task.multipleSequenceAlignment.subjectColumnsBegin_incl,
                task.multipleSequenceAlignment.subjectColumnsEnd_excl,
                task.alignmentShifts.data(),
                correctionOptions->estimatedCoverage,
                goodAlignmentProperties->maxErrorRate
            );
        };

        auto removeCandidatesOfDifferentRegion = [&](const auto& minimizationResult){

            if(minimizationResult.performedMinimization){
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

                        std::copy_n(
                            task.candidateQualities.data() + i * size_t(qualityPitchInBytes),
                            qualityPitchInBytes,
                            task.candidateQualities.data() + insertpos * size_t(qualityPitchInBytes)
                        );
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
                task.candidateQualities.erase(
                    task.candidateQualities.begin() + qualityPitchInBytes * insertpos, 
                    task.candidateQualities.end()
                );
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

                //build minimized multiple sequence alignment
                buildMultipleSequenceAlignment(task);
            }
        };

        if(max_num_minimizations > 0){                

            for(int numIterations = 0; numIterations < max_num_minimizations; numIterations++){
                const auto minimizationResult = findCandidatesLambda();
                removeCandidatesOfDifferentRegion(minimizationResult);
                if(!minimizationResult.performedMinimization){
                    break;
                }
            }
        }
#if 0
        if(task.subjectReadId == 10307280){
            std::cerr << "subjectColumnsBegin_incl = " << task.multipleSequenceAlignment.subjectColumnsBegin_incl << "\n";
            std::cerr << "subjectColumnsEnd_excl = " << task.multipleSequenceAlignment.subjectColumnsEnd_excl << "\n";
            //std::cerr << "lastColumn_excl = " << dataArrays.h_msa_column_properties[i].lastColumn_excl << "\n";
            std::cerr << "counts: \n";
            for(int k = 0; k < task.multipleSequenceAlignment.countsA.size(); k++){
                std::cerr << task.multipleSequenceAlignment.countsA[k] << ' ';
            }
            std::cerr << "\n";
            for(int k = 0; k < task.multipleSequenceAlignment.countsC.size(); k++){
                std::cerr << task.multipleSequenceAlignment.countsC[k] << ' ';
            }
            std::cerr << "\n";
            for(int k = 0; k < task.multipleSequenceAlignment.countsG.size(); k++){
                std::cerr << task.multipleSequenceAlignment.countsG[k] << ' ';
            }
            std::cerr << "\n";
            for(int k = 0; k < task.multipleSequenceAlignment.countsT.size(); k++){
                std::cerr << task.multipleSequenceAlignment.countsT[k] << ' ';
            }
            std::cerr << "\n";

            std::cerr << "weights: \n";
            for(int k = 0; k < task.multipleSequenceAlignment.weightsA.size(); k++){
                std::cerr << task.multipleSequenceAlignment.weightsA[k] << ' ';
            }
            std::cerr << "\n";
            for(int k = 0; k < task.multipleSequenceAlignment.weightsC.size(); k++){
                std::cerr << task.multipleSequenceAlignment.weightsC[k] << ' ';
            }
            std::cerr << "\n";
            for(int k = 0; k < task.multipleSequenceAlignment.weightsG.size(); k++){
                std::cerr << task.multipleSequenceAlignment.weightsG[k] << ' ';
            }
            std::cerr << "\n";
            for(int k = 0; k < task.multipleSequenceAlignment.weightsT.size(); k++){
                std::cerr << task.multipleSequenceAlignment.weightsT[k] << ' ';
            }
            std::cerr << "\n";

            std::cerr << "coverage: \n";
            for(int k = 0; k < task.multipleSequenceAlignment.coverage.size(); k++){
                std::cerr << task.multipleSequenceAlignment.coverage[k] << ' ';
            }
            std::cerr << "\n";

            std::cerr << "support: \n";
            for(int k = 0; k < task.multipleSequenceAlignment.support.size(); k++){
                std::cerr << task.multipleSequenceAlignment.support[k] << ' ';
            }
            std::cerr << "\n";

            std::cerr << "consensus: \n";
            for(int k = 0; k < task.multipleSequenceAlignment.consensus.size(); k++){
                std::cerr << task.multipleSequenceAlignment.consensus[k] << ' ';
            }
            std::cerr << "\n";
        }
#endif            
    }

    void correctAnchorClassic(Task& task) const{

        assert(correctionOptions->correctionType == CorrectionType::Classic);

        const int subjectColumnsBegin_incl = task.multipleSequenceAlignment.subjectColumnsBegin_incl;
        const int subjectColumnsEnd_excl = task.multipleSequenceAlignment.subjectColumnsEnd_excl;

        task.msaProperties = getMSAProperties2(
            task.multipleSequenceAlignment.support.data(),
            task.multipleSequenceAlignment.coverage.data(),
            subjectColumnsBegin_incl,
            subjectColumnsEnd_excl,
            correctionOptions->estimatedErrorrate,
            correctionOptions->estimatedCoverage,
            correctionOptions->m_coverage
        );

        task.subjectCorrection = getCorrectedSubjectNew(
            task.multipleSequenceAlignment.consensus.data() + subjectColumnsBegin_incl,
            task.multipleSequenceAlignment.support.data() + subjectColumnsBegin_incl,
            task.multipleSequenceAlignment.coverage.data() + subjectColumnsBegin_incl,
            task.multipleSequenceAlignment.origCoverages.data() + subjectColumnsBegin_incl,
            task.input.anchorLength,
            task.decodedAnchor.data(),
            subjectColumnsBegin_incl,
            task.decodedCandidateSequences.data(),
            task.candidateReadIds.size(),
            task.alignmentWeights.data(),
            task.candidateSequencesLengths.data(),
            task.alignmentShifts.data(),
            decodedSequencePitchInBytes,
            task.msaProperties,
            correctionOptions->estimatedErrorrate,
            correctionOptions->estimatedCoverage,
            correctionOptions->m_coverage,
            correctionOptions->kmerlength,
            task.input.anchorReadId
        );

        task.msaProperties.isHQ = task.subjectCorrection.isHQ;
        
    }       

    void correctAnchorClf(Task& task) const
    {
        const int subject_b = task.multipleSequenceAlignment.subjectColumnsBegin_incl;
        const int subject_e = task.multipleSequenceAlignment.subjectColumnsEnd_excl;
        auto& cons = task.multipleSequenceAlignment.consensus;
        auto& orig = task.decodedAnchor;
        auto& corr = task.subjectCorrection.correctedSequence;
        
        task.msaProperties = getMSAProperties2(
            task.multipleSequenceAlignment.support.data(),
            task.multipleSequenceAlignment.coverage.data(),
            subject_b,
            subject_e,
            correctionOptions->estimatedErrorrate,
            correctionOptions->estimatedCoverage,
            correctionOptions->m_coverage
        );

        corr.insert(0, cons.data()+subject_b, task.input.anchorLength);
        if (!task.msaProperties.isHQ) {
            constexpr float THRESHOLD = 0.73f;
            for (int i = 0; i < task.input.anchorLength; ++i) {
                if (orig[i] != cons[subject_b+i] &&
                    mlSettings->classifier_anchor->decide(
                        make_sample(
                            task.multipleSequenceAlignment,
                            task.msaProperties,
                            orig[i],
                            subject_b+i,
                            correctionOptions->estimatedCoverage
                        )
                    ) < THRESHOLD)
                {
                    corr[i] = orig[i];
                }
            }
        }

        task.subjectCorrection.isCorrected = true;
    }

    void correctAnchorPrint(Task& task) const{
        const int subject_b = task.multipleSequenceAlignment.subjectColumnsBegin_incl;
        const int subject_e = task.multipleSequenceAlignment.subjectColumnsEnd_excl;
        const auto& cons = task.multipleSequenceAlignment.consensus;
        const auto& orig = task.decodedAnchor;

        task.msaProperties = getMSAProperties2(
            task.multipleSequenceAlignment.support.data(),
            task.multipleSequenceAlignment.coverage.data(),
            subject_b,
            subject_e,
            correctionOptions->estimatedErrorrate,
            correctionOptions->estimatedCoverage,
            correctionOptions->m_coverage
        );

        if (!task.msaProperties.isHQ) {
            for (int i = 0; i < task.input.anchorLength; ++i) {
                if (orig[i] != cons[subject_b+i]) {
                    ml_sample_t sample = make_sample(task.multipleSequenceAlignment,
                                                        task.msaProperties,
                                                        orig[i],
                                                        subject_b+i,
                                                        correctionOptions->estimatedCoverage);
                    ml_stream_anchor << task.input.anchorReadId << ' ' << i << " ";
                    for (float j: sample) ml_stream_anchor << j << ' ';
                    ml_stream_anchor << '\n';
                }
            }
        }
        task.subjectCorrection.isCorrected = false;
    }

    void correctAnchor(Task& task) const{
        if(correctionOptions->correctionType == CorrectionType::Classic)
            correctAnchorClassic(task);
        else if(correctionOptions->correctionType == CorrectionType::Forest)
            correctAnchorClf(task);
        else if (correctionOptions->correctionType == CorrectionType::Print)
            correctAnchorPrint(task);
    }

    void correctCandidatesClassic(Task& task) const{

        task.candidateCorrections = getCorrectedCandidatesNew(
            task.multipleSequenceAlignment.consensus.data(),
            task.multipleSequenceAlignment.support.data(),
            task.multipleSequenceAlignment.coverage.data(),
            task.multipleSequenceAlignment.nColumns,
            task.multipleSequenceAlignment.subjectColumnsBegin_incl,
            task.multipleSequenceAlignment.subjectColumnsEnd_excl,
            task.alignmentShifts.data(),
            task.candidateSequencesLengths.data(),
            task.multipleSequenceAlignment.nCandidates,
            correctionOptions->estimatedErrorrate,
            correctionOptions->estimatedCoverage,
            correctionOptions->m_coverage,
            correctionOptions->new_columns_to_correct
        );
        
        if(0) /*if(task.subjectReadId == 1)*/{
            for(const auto& correctedCandidate : task.candidateCorrections){
                const read_number candidateId = task.candidateReadIds[correctedCandidate.index];
                
                if(task.alignmentFlags[correctedCandidate.index] == BestAlignment_t::Forward){
                    std::cerr << candidateId << " " << correctedCandidate.sequence << "\n";
                }else{
                    std::string fwd;
                    fwd.resize(correctedCandidate.sequence.length());
                    reverseComplementString(
                        &fwd[0], 
                        correctedCandidate.sequence.c_str(), 
                                            correctedCandidate.sequence.length()
                    );
                    std::cerr << "revc " << candidateId << " " << fwd << "\n";
                }
            }
        }
    }

    void correctCandidatesPrint(Task& task) const{

        const auto& msa = task.multipleSequenceAlignment;
        const int subject_begin = msa.subjectColumnsBegin_incl;
        const int subject_end = msa.subjectColumnsEnd_excl;

        std::bernoulli_distribution coinflip(0.01);

        for(int cand = 0; cand < msa.nCandidates; ++cand) {
            const int cand_begin = msa.subjectColumnsBegin_incl + task.alignmentShifts[cand];
            const int cand_length = task.candidateSequencesLengths[cand];
            const int cand_end = cand_begin + cand_length;
            const int offset = cand * decodedSequencePitchInBytes;
            
            MSAProperties props = getMSAProperties2(
                msa.support.data(),
                msa.coverage.data(),
                cand_begin,
                cand_end,
                correctionOptions->estimatedErrorrate,
                correctionOptions->estimatedCoverage,
                correctionOptions->m_coverage);
            
            if(cand_begin >= subject_begin - correctionOptions->new_columns_to_correct
                && cand_end <= subject_end + correctionOptions->new_columns_to_correct)
            {
                for (int i = 0; i < cand_length; ++i) {
                    if (task.decodedCandidateSequences[offset+i] != msa.consensus[cand_begin+i] && coinflip(mlSettings->rndGenerator)) {
                        auto sample = make_sample(msa, props, task.decodedCandidateSequences[offset+i], cand_begin+i, correctionOptions->estimatedCoverage);
                        ml_stream_cands << task.candidateReadIds[cand] << ' ' 
                            << (task.alignmentFlags[cand]==BestAlignment_t::ReverseComplement?-i-1:i) << ' ';
                        for (float j: sample) ml_stream_cands << j << ' ';
                        ml_stream_cands << '\n';
                    }
                }
            }
        }
        task.candidateCorrections = std::vector<CorrectedCandidate>{};
    }

    void correctCandidatesClf(Task& task) const 
    {
        const auto& msa = task.multipleSequenceAlignment;

        task.candidateCorrections = std::vector<CorrectedCandidate>{};

        const int subject_begin = msa.subjectColumnsBegin_incl;
        const int subject_end = msa.subjectColumnsEnd_excl;
        for(int cand = 0; cand < msa.nCandidates; ++cand) {
            const int cand_begin = msa.subjectColumnsBegin_incl + task.alignmentShifts[cand];
            const int cand_length = task.candidateSequencesLengths[cand];
            const int cand_end = cand_begin + cand_length;
            const int offset = cand * decodedSequencePitchInBytes;
            
            MSAProperties props = getMSAProperties2(
                msa.support.data(),
                msa.coverage.data(),
                cand_begin,
                cand_end,
                correctionOptions->estimatedErrorrate,
                correctionOptions->estimatedCoverage,
                correctionOptions->m_coverage);
            
            if(cand_begin >= subject_begin - correctionOptions->new_columns_to_correct
                && cand_end <= subject_end + correctionOptions->new_columns_to_correct)
            {

                task.candidateCorrections.emplace_back(cand, task.alignmentShifts[cand],
                    std::string(&msa.consensus[cand_begin], cand_length));


                for (int i = 0; i < cand_length; ++i) {
                    constexpr float THRESHOLD = 0.73f;
                    if (task.decodedCandidateSequences[offset+i] != msa.consensus[cand_begin+i]
                        && mlSettings->classifier_cands->decide(
                                make_sample(msa, props, task.decodedCandidateSequences[offset+i], cand_begin+i, correctionOptions->estimatedCoverage)
                            ) < THRESHOLD)
                    {
                        task.candidateCorrections.back().sequence[i] = task.decodedCandidateSequences[offset+i];
                    }
                }
            }
        }
    }

    void correctCandidates(Task& task) const{
        switch (correctionOptions->correctionTypeCands) {
            case CorrectionType::Print:
                correctCandidatesPrint(task);
                break;
            case CorrectionType::Forest:
                correctCandidatesClf(task);
                break;
            default:
                correctCandidatesClassic(task);
        }
    }

    CorrectionOutput makeOutputOfTask(Task& task) const{
        CorrectionOutput result;

        result.hasAnchorCorrection = task.subjectCorrection.isCorrected;

        if(result.hasAnchorCorrection){
            auto& correctedSequenceString = task.subjectCorrection.correctedSequence;
            const int correctedlength = correctedSequenceString.length();
            const bool originalReadContainsN = readStorage->readContainsN(task.input.anchorReadId);
            
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
            tmp.uncorrectedPositionsNoConsensus = std::move(task.subjectCorrection.uncorrectedPositionsNoConsensus);
            tmp.readId = task.input.anchorReadId;
            tmp.sequence = std::move(correctedSequenceString); 
            
            result.anchorCorrection = std::move(tmp);
        }
        
        
        
        for(const auto& correctedCandidate : task.candidateCorrections){
            const read_number candidateId = task.candidateReadIds[correctedCandidate.index];
            
            bool savingIsOk = false;
            
            if(!(correctionFlags->getFlag(candidateId) & ReadCorrectionFlags::readCorrectedAsHQAnchor())){
                savingIsOk = true;
            }
            
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
                    reverseComplementString(
                        &fwd[0], 
                        correctedCandidate.sequence.c_str(), 
                                            correctedCandidate.sequence.length()
                    );
                    tmp.sequence = std::move(fwd);
                }
                
                const bool originalCandidateReadContainsN = readStorage->readContainsN(candidateId);
                
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
                        reverseComplementString(
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

    ml_sample_t make_sample(const MultipleSequenceAlignment& msa, const MSAProperties& props, char orig, size_t pos, float norm) const noexcept
    {   
        // std::cerr << "making sample" << std::endl;
        float countsACGT = msa.countsA[pos] + msa.countsC[pos] + msa.countsG[pos] + msa.countsT[pos];
        return {
            float(orig == 'A'),
            float(orig == 'C'),
            float(orig == 'G'),
            float(orig == 'T'),
            float(msa.consensus[pos] == 'A'),
            float(msa.consensus[pos] == 'C'),
            float(msa.consensus[pos] == 'G'),
            float(msa.consensus[pos] == 'T'),
            orig == 'A'?msa.countsA[pos]/countsACGT:0,
            orig == 'C'?msa.countsC[pos]/countsACGT:0,
            orig == 'G'?msa.countsG[pos]/countsACGT:0,
            orig == 'T'?msa.countsT[pos]/countsACGT:0,
            orig == 'A'?msa.weightsA[pos]:0,
            orig == 'C'?msa.weightsC[pos]:0,
            orig == 'G'?msa.weightsG[pos]:0,
            orig == 'T'?msa.weightsT[pos]:0,
            msa.consensus[pos] == 'A'?msa.countsA[pos]/countsACGT:0,
            msa.consensus[pos] == 'C'?msa.countsC[pos]/countsACGT:0,
            msa.consensus[pos] == 'G'?msa.countsG[pos]/countsACGT:0,
            msa.consensus[pos] == 'T'?msa.countsT[pos]/countsACGT:0,
            msa.consensus[pos] == 'A'?msa.weightsA[pos]:0,
            msa.consensus[pos] == 'C'?msa.weightsC[pos]:0,
            msa.consensus[pos] == 'G'?msa.weightsG[pos]:0,
            msa.consensus[pos] == 'T'?msa.weightsT[pos]:0,
            msa.weightsA[pos],
            msa.weightsC[pos],
            msa.weightsG[pos],
            msa.weightsT[pos],
            msa.countsA[pos]/countsACGT,
            msa.countsC[pos]/countsACGT,
            msa.countsG[pos]/countsACGT,
            msa.countsT[pos]/countsACGT,
            props.avg_support,
            props.min_support,
            float(props.max_coverage)/norm,
            float(props.min_coverage)/norm
        };
    }

private:

    std::size_t encodedSequencePitchInInts{};
    std::size_t decodedSequencePitchInBytes{};
    std::size_t qualityPitchInBytes{};

    const CorrectionOptions* correctionOptions{};
    const GoodAlignmentProperties* goodAlignmentProperties{};
    const Minhasher* minhasher{};
    const cpu::ContiguousReadStorage* readStorage{};

    ReadCorrectionFlags* correctionFlags{};
    MLSettings* mlSettings{};

    mutable cpu::ContiguousReadStorage::GatherHandle readStorageGatherHandle{};    
    mutable Minhasher::Handle minhashHandle{};
    mutable cpu::shd::CpuAlignmentHandle alignmentHandle{};

    mutable std::stringstream ml_stream_anchor;
    mutable std::stringstream ml_stream_cands;

    TimeMeasurements totalTime{};
};



}



#endif