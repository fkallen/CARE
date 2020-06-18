//#include <cpu_correction_thread.hpp>

#include <correctionresultprocessing.hpp>

#include <config.hpp>

#include "options.hpp"

#include <minhasher.hpp>
#include <readstorage.hpp>
#include "cpu_alignment.hpp"
#include "bestalignment.hpp"
#include <msa.hpp>
#include "qualityscoreweights.hpp"
#include "rangegenerator.hpp"

#include <threadpool.hpp>
#include <memoryfile.hpp>
#include <util.hpp>
#include <filehelpers.hpp>
#include <forestclassifier.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include <vector>

#include <omp.h>

#define USE_MSA_MINIMIZATION


#define ENABLE_TIMING

//#define DO_PROFILE

#ifdef DO_PROFILE
constexpr std::int64_t num_reads_to_profile = 100000;
#endif


//#define PRINT_MSA

namespace care{
namespace cpu{

        //read status bitmask
        constexpr std::uint8_t readCorrectedAsHQAnchor = 1;
        constexpr std::uint8_t readCouldNotBeCorrectedAsAnchor = 2;

        constexpr bool useSortedIdsForGather = false;


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

            TimeMeasurements& operator+=(TimeMeasurements& rhs) noexcept{
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


        //BatchData = Working set of a thread

        struct BatchData{
            struct OutputData{
                std::vector<TempCorrectedSequence> anchorCorrections;
                std::vector<EncodedTempCorrectedSequence> encodedAnchorCorrections;
                std::vector<TempCorrectedSequence> candidateCorrections;
                std::vector<EncodedTempCorrectedSequence> encodedCandidateCorrections;
            };

            //Task = Working set to correct a single read.
            struct Task{
                bool active;
                int numCandidates;
                int numGoodAlignmentFlags;
                int numFilteredCandidates;
                int subjectSequenceLength;
                read_number subjectReadId;
                read_number* candidateReadIds;
                int* candidateSequencesLengths; 
                unsigned int* subjectSequenceData;
                unsigned int* candidateSequencesData;
                unsigned int* candidateSequencesRevcData;
                char* subjectQualities;
                char* decodedSubjectSequence;
                

                SHDResult* bestAlignments;
                BestAlignment_t* bestAlignmentFlags;
                int*bestAlignmentShifts;
                float* bestAlignmentWeights;
                read_number* bestCandidateReadIds;
                int* bestCandidateLengths;
                unsigned int* bestCandidateData;
                char* bestCandidateQualities;

                CorrectionResult subjectCorrection;
                std::vector<CorrectedCandidate> candidateCorrections;
                MSAProperties msaProperties;

                void reset(){
                    active = false;
                    subjectReadId = std::numeric_limits<read_number>::max();
                    subjectCorrection.reset();
                    candidateCorrections.clear();
                    msaProperties = MSAProperties{};
                }            
            };

            // data for all tasks within batch
            std::vector<read_number> subjectReadIds;
            std::vector<read_number> candidateReadIds;
            std::vector<unsigned int> subjectSequencesData;
            std::vector<unsigned int> candidateSequencesData;
            std::vector<unsigned int> candidateSequencesRevcData;
            std::vector<int> subjectSequencesLengths;
            std::vector<int> candidateSequencesLengths;
            std::vector<char> subjectQualities;
            std::vector<char> candidateQualities;

            std::vector<char> decodedSubjectSequences;

            std::vector<SHDResult> bestAlignments;
            std::vector<BestAlignment_t> bestAlignmentFlags;
            std::vector<int> bestAlignmentShifts;
            std::vector<float> bestAlignmentWeights;

            std::vector<int> candidatesPerSubject;
            std::vector<int> candidatesPerSubjectPrefixSum;
            std::vector<read_number> filteredReadIds;
            // data used by a single task. is shared by all tasks within batch -> no interleaved access
            std::vector<SHDResult> forwardAlignments;
            std::vector<SHDResult> revcAlignments;
            std::vector<BestAlignment_t> alignmentFlags;
            std::vector<int> filterIndices;
            std::vector<char> decodedCandidateSequences;

            std::vector<int> tmpnOps;
            std::vector<int> tmpoverlaps;

            OutputData outputData;

            std::vector<int> indicesOfCandidatesEqualToSubject;


            // ------------------------------------------------
            std::vector<Task> batchTasks;

            ContiguousReadStorage::GatherHandle readStorageGatherHandle;
            Minhasher::Handle minhashHandle;
            shd::CpuAlignmentHandle alignmentHandle;

            MultipleSequenceAlignment multipleSequenceAlignment;

            TimeMeasurements timings;

            int encodedSequencePitchInInts = 0;
            int decodedSequencePitchInBytes = 0;
            int qualityPitchInBytes = 0;

            std::shared_ptr<ForestClassifier> forestClassifier1;
        };

        void makeBatchTasks(BatchData& data){
            const int numSubjects = data.subjectReadIds.size();

            data.batchTasks.resize(numSubjects);

            for(int i = 0; i < numSubjects; i++){
                auto& task = data.batchTasks[i];
                const int offset = data.candidatesPerSubjectPrefixSum[i];

                task.reset();

                task.active = true;
                task.numCandidates = data.candidatesPerSubject[i];
                task.subjectSequenceLength = data.subjectSequencesLengths[i];
                task.subjectReadId = data.subjectReadIds[i];
                task.candidateReadIds = data.candidateReadIds.data() + offset;
                task.candidateSequencesLengths = data.candidateSequencesLengths.data() + offset; 
                task.subjectSequenceData = data.subjectSequencesData.data() + size_t(i) * data.encodedSequencePitchInInts;
                task.candidateSequencesData = data.candidateSequencesData.data() + size_t(offset) * data.encodedSequencePitchInInts;
                task.candidateSequencesRevcData = data.candidateSequencesRevcData.data() + size_t(offset) * data.encodedSequencePitchInInts;

                task.decodedSubjectSequence = data.decodedSubjectSequences.data() + size_t(i) * data.decodedSequencePitchInBytes;

                task.bestAlignments = data.bestAlignments.data() + offset;
                task.bestAlignmentFlags = data.bestAlignmentFlags.data() + offset;
                task.bestAlignmentShifts = data.bestAlignmentShifts.data() + offset;
                task.bestAlignmentWeights = data.bestAlignmentWeights.data() + offset;
                task.bestCandidateReadIds = task.candidateReadIds;
                task.bestCandidateLengths = task.candidateSequencesLengths;
                task.bestCandidateData = task.candidateSequencesData;  

                if(task.numCandidates == 0){
                    task.active = false;
                }                             
            }
        }

        // struct InterestingStruct{
        //     read_number readId;
        //     std::vector<int> positions;
        // };

        // std::vector<read_number> interestingReadIds;
        // std::mutex interestingMutex;

        template<class Iter>
        Iter findBestAlignmentDirection(
                Iter result,
                const SHDResult* forwardAlignments,
                const SHDResult* revcAlignments,
                int numCandidates,
                const int subjectLength,
                const int* candidateLengths,
                const int min_overlap,
                const float estimatedErrorrate,
                const float min_overlap_ratio){


            for(int i = 0; i < numCandidates; i++, ++result){
                const SHDResult& forwardAlignment = forwardAlignments[i];
                const SHDResult& revcAlignment = revcAlignments[i];
                const int candidateLength = candidateLengths[i];

                BestAlignment_t bestAlignmentFlag = care::choose_best_alignment(forwardAlignment,
                                                                                revcAlignment,
                                                                                subjectLength,
                                                                                candidateLength,
                                                                                min_overlap_ratio,
                                                                                min_overlap,
                                                                                estimatedErrorrate);

                *result = bestAlignmentFlag;
            }

            return result;
        }

        /*
            Filters alignments by good mismatch ratio.

            Returns an sorted index list to alignments which pass the filter.
        */

        template<class Iter, class Func>
        Iter
        filterAlignmentsByMismatchRatio(Iter result,
                                        const SHDResult* alignments,
                                        int numAlignments,
                                        const float estimatedErrorrate,
                                        const int estimatedCoverage,
                                        const float m_coverage,
                                        Func lastResortFunc){

            const float mismatchratioBaseFactor = estimatedErrorrate * 1.0f;
            const float goodAlignmentsCountThreshold = estimatedCoverage * m_coverage;

            std::array<int, 3> counts({0,0,0});

            for(int i = 0; i < numAlignments; i++){
                const auto& alignment = alignments[i];
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

            //no correction possible without enough candidates
            if(std::none_of(counts.begin(), counts.end(), [](auto c){return c > 0;})){
                return result;
            }

            //std::cerr << "Read " << task.readId << ", good alignments after bining: " << std::accumulate(counts.begin(), counts.end(), int(0)) << '\n';
            //std::cerr << "Read " << task.readId << ", bins: " << counts[0] << " " << counts[1] << " " << counts[2] << '\n';


            float mismatchratioThreshold = 0;
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
                    return result; //no correction possible without good candidates
                }
            }

            for(int i = 0; i < numAlignments; i++){
                const auto& alignment = alignments[i];
                const float mismatchratio = float(alignment.nOps) / float(alignment.overlap);
                const bool notremoved = mismatchratio < mismatchratioThreshold;

                if(notremoved){
                    *result = i;
                    ++result;
                }
            }

            return result;
        }


        void getSubjectSequenceData(BatchData& data,
                                    const cpu::ContiguousReadStorage& readStorage){

            const int numSubjects = data.subjectReadIds.size();

            data.subjectSequencesLengths.clear();
            data.subjectSequencesLengths.resize(numSubjects);
            data.subjectSequencesData.clear();
            data.subjectSequencesData.resize(data.encodedSequencePitchInInts * numSubjects, 0);

            readStorage.gatherSequenceLengths(
                data.readStorageGatherHandle,
                data.subjectReadIds.data(),
                numSubjects,
                data.subjectSequencesLengths.data()
            );

            readStorage.gatherSequenceData(
                data.readStorageGatherHandle,
                data.subjectReadIds.data(),
                numSubjects,
                data.subjectSequencesData.data(),
                data.encodedSequencePitchInInts
            );

            data.subjectQualities.resize(size_t(data.qualityPitchInBytes) * numSubjects);
            
            data.outputData.anchorCorrections.reserve(numSubjects);            
        }

        void determineCandidateReadIds(BatchData& data,
                                        const Minhasher& minhasher,
                                        const cpu::ContiguousReadStorage& readStorage,
                                        const CorrectionOptions& correctionOptions){

            const int numSubjects = data.subjectReadIds.size();

            data.decodedSubjectSequences.clear();
            data.decodedSubjectSequences.resize(numSubjects * data.decodedSequencePitchInBytes);

            data.candidatesPerSubject.resize(numSubjects);
            data.candidatesPerSubjectPrefixSum.resize(numSubjects+1);

            data.candidateReadIds.clear();

            int maxCandidatesPerSubject = 0;

            for(int i = 0; i < numSubjects; i++){
                const read_number readId = data.subjectReadIds[i];

                const bool containsN = readStorage.readContainsN(readId);

                //exclude anchors with ambiguous bases
                if(!(correctionOptions.excludeAmbiguousReads && containsN)){

                    const int length = data.subjectSequencesLengths[i];
                    char* const decodedBegin = &data.decodedSubjectSequences[i * data.decodedSequencePitchInBytes];

                    decode2BitSequence(decodedBegin,
                                        &data.subjectSequencesData[i * data.encodedSequencePitchInInts],
                                        length);
                    //TODO modify minhasher to work with char ptr + size instead of string
                    std::string sequence(decodedBegin, length);

                    minhasher.getCandidates_any_map(
                        data.minhashHandle,
                        sequence,
                        0
                    );

                    auto readIdPos = std::lower_bound(data.minhashHandle.result().begin(),
                                                    data.minhashHandle.result().end(),
                                                    readId);

                    if(readIdPos != data.minhashHandle.result().end() && *readIdPos == readId){
                        data.minhashHandle.result().erase(readIdPos);
                    }

                    auto minhashResultsEnd = data.minhashHandle.result().end();
                    //exclude candidates with ambiguous bases

                    if(correctionOptions.excludeAmbiguousReads){
                        minhashResultsEnd = std::remove_if(
                            data.minhashHandle.result().begin(),
                            data.minhashHandle.result().end(),
                            [&](read_number readId){
                                return readStorage.readContainsN(readId);
                            }
                        );
                    }

                    //auto debugit = std::find(data.minhashHandle.result().begin(), data.minhashHandle.result().end(), 32141191);
                    // if(readId != 32141191 && debugit == data.minhashHandle.result().end()){
                    //     const int candidatesPerSubject = 0;
                    //     maxCandidatesPerSubject = std::max(maxCandidatesPerSubject, candidatesPerSubject);
                    //     data.candidatesPerSubject[i] = candidatesPerSubject;
                    // }else{
                        //std::cerr << "found id 32141191 as candidate of read " << readId << "\n";
                        data.candidateReadIds.insert(
                            data.candidateReadIds.end(),
                            data.minhashHandle.result().begin(),
                            minhashResultsEnd
                        );

                        const int candidatesPerSubject = std::distance(data.minhashHandle.result().begin(), minhashResultsEnd);
                        maxCandidatesPerSubject = std::max(maxCandidatesPerSubject, candidatesPerSubject);
                        data.candidatesPerSubject[i] = candidatesPerSubject;

                    // }
                }else{
                    const int candidatesPerSubject = 0;
                    maxCandidatesPerSubject = std::max(maxCandidatesPerSubject, candidatesPerSubject);
                    data.candidatesPerSubject[i] = candidatesPerSubject;
                }
            }

            data.forwardAlignments.resize(maxCandidatesPerSubject);
            data.revcAlignments.resize(maxCandidatesPerSubject);
            data.alignmentFlags.resize(maxCandidatesPerSubject);
            data.decodedCandidateSequences.resize(size_t(data.decodedSequencePitchInBytes) * maxCandidatesPerSubject);
            data.filterIndices.resize(maxCandidatesPerSubject);

            data.tmpnOps.resize(maxCandidatesPerSubject);
            data.tmpoverlaps.resize(maxCandidatesPerSubject);

            std::partial_sum(
                data.candidatesPerSubject.begin(),
                data.candidatesPerSubject.end(),
                data.candidatesPerSubjectPrefixSum.begin() + 1
            );

            data.candidatesPerSubjectPrefixSum[0] = 0;

            const int totalNumCandidates = data.candidatesPerSubjectPrefixSum.back();

            
            data.candidateQualities.resize(size_t(data.qualityPitchInBytes) * totalNumCandidates);
            data.bestAlignments.resize(totalNumCandidates);
            data.bestAlignmentFlags.resize(totalNumCandidates);
            data.bestAlignmentShifts.resize(totalNumCandidates);
            data.bestAlignmentWeights.resize(totalNumCandidates);

            data.filteredReadIds.resize(totalNumCandidates);
            
            data.outputData.candidateCorrections.reserve(numSubjects * 5);

        }

        void getCandidateSequenceData(BatchData& data,
                                    const cpu::ContiguousReadStorage& readStorage){

            const int numCandidates = data.candidatesPerSubjectPrefixSum.back();

            data.candidateSequencesLengths.resize(numCandidates);

            data.candidateSequencesData.clear();
            data.candidateSequencesData.resize(size_t(data.encodedSequencePitchInInts) * numCandidates, 0);
            data.candidateSequencesRevcData.clear();
            data.candidateSequencesRevcData.resize(size_t(data.encodedSequencePitchInInts) * numCandidates, 0);

            readStorage.gatherSequenceLengths(
                data.readStorageGatherHandle,
                data.candidateReadIds.data(),
                numCandidates,
                data.candidateSequencesLengths.data()
            );

            if(useSortedIdsForGather){
                readStorage.gatherSequenceDataSpecial(
                    data.readStorageGatherHandle,
                    data.candidateReadIds.data(),
                    numCandidates,
                    data.candidateSequencesData.data(),
                    data.encodedSequencePitchInInts
                );
            }else{
                readStorage.gatherSequenceData(
                    data.readStorageGatherHandle,
                    data.candidateReadIds.data(),
                    numCandidates,
                    data.candidateSequencesData.data(),
                    data.encodedSequencePitchInInts
                );
            }

            for(int i = 0; i < numCandidates; i++){
                const unsigned int* const seqPtr = data.candidateSequencesData.data() 
                                                    + std::size_t(data.encodedSequencePitchInInts) * i;
                unsigned int* const seqrevcPtr = data.candidateSequencesRevcData.data() 
                                                    + std::size_t(data.encodedSequencePitchInInts) * i;

                reverseComplement2Bit(
                    seqrevcPtr,  
                    seqPtr,
                    data.candidateSequencesLengths[i]
                );
            }
        }

        void getCandidateAlignments(BatchData& data,
                                    BatchData::Task& task,
                                    const GoodAlignmentProperties& alignmentProps,
                                    const CorrectionOptions& correctionOptions){

            shd::cpuShiftedHammingDistancePopcount2Bit(
                data.alignmentHandle,
                data.forwardAlignments.begin(),
                task.subjectSequenceData,
                task.subjectSequenceLength,
                task.candidateSequencesData,
                data.encodedSequencePitchInInts,
                task.candidateSequencesLengths,
                task.numCandidates,
                alignmentProps.min_overlap,
                alignmentProps.maxErrorRate,
                alignmentProps.min_overlap_ratio
            );

            shd::cpuShiftedHammingDistancePopcount2Bit(
                data.alignmentHandle,
                data.revcAlignments.begin(),
                task.subjectSequenceData,
                task.subjectSequenceLength,
                task.candidateSequencesRevcData,
                data.encodedSequencePitchInInts,
                task.candidateSequencesLengths,
                task.numCandidates,
                alignmentProps.min_overlap,
                alignmentProps.maxErrorRate,
                alignmentProps.min_overlap_ratio
            );

            //decide whether to keep forward or reverse complement

            findBestAlignmentDirection(
                data.alignmentFlags.begin(),
                data.forwardAlignments.data(),
                data.revcAlignments.data(),
                task.numCandidates,
                task.subjectSequenceLength,
                task.candidateSequencesLengths,
                alignmentProps.min_overlap,
                correctionOptions.estimatedErrorrate,
                alignmentProps.min_overlap_ratio
            );

            task.numGoodAlignmentFlags = std::count_if(
                data.alignmentFlags.begin(),
                data.alignmentFlags.begin() + task.numCandidates,
                [](const auto flag){
                    return flag != BestAlignment_t::None;
                }
            );

            if(task.numGoodAlignmentFlags == 0){
                task.active = false;
            }
        }

        void gatherBestAlignmentData(BatchData& data,
                                  BatchData::Task& task){

            task.numFilteredCandidates = 0;

            for(int i = 0, insertpos = 0; i < task.numCandidates; i++){

                const BestAlignment_t flag = data.alignmentFlags[i];
                const auto& fwdAlignment = data.forwardAlignments[i];
                const auto& revcAlignment = data.revcAlignments[i];
                const read_number candidateId = task.candidateReadIds[i];
                const int candidateLength = task.candidateSequencesLengths[i];

                if(flag == BestAlignment_t::Forward){
                    task.bestAlignmentFlags[insertpos] = flag;
                    task.bestCandidateReadIds[insertpos] = candidateId;
                    task.bestCandidateLengths[insertpos] = candidateLength;

                    task.bestAlignments[insertpos] = fwdAlignment;
                    std::copy_n(
                        task.candidateSequencesData + i * size_t(data.encodedSequencePitchInInts),
                        data.encodedSequencePitchInInts,
                        task.bestCandidateData + insertpos * size_t(data.encodedSequencePitchInInts)
                    );

                    insertpos++;
                    task.numFilteredCandidates++;

                }else if(flag == BestAlignment_t::ReverseComplement){
                    task.bestAlignmentFlags[insertpos] = flag;
                    task.bestCandidateReadIds[insertpos] = candidateId;
                    task.bestCandidateLengths[insertpos] = candidateLength;

                    task.bestAlignments[insertpos] = revcAlignment;
                    std::copy_n(
                        task.candidateSequencesRevcData + i * size_t(data.encodedSequencePitchInInts),
                        data.encodedSequencePitchInInts,
                        task.bestCandidateData + insertpos * size_t(data.encodedSequencePitchInInts)
                    );

                    insertpos++;
                    task.numFilteredCandidates++;

                }else{
                    ;//BestAlignment_t::None discard alignment
                }
            }
        }

        void filterBestAlignmentsByMismatchRatio(BatchData& data,
                  BatchData::Task& task,
                  const CorrectionOptions& correctionOptions,
                  const GoodAlignmentProperties& alignmentProps){
            //get indices of alignments which have a good mismatch ratio

            auto filterIndicesEnd = filterAlignmentsByMismatchRatio(
                data.filterIndices.begin(),
                task.bestAlignments,
                task.numFilteredCandidates,
                correctionOptions.estimatedErrorrate,
                correctionOptions.estimatedCoverage,
                correctionOptions.m_coverage,
                [](){
                    return false;
                }
            );

            task.numFilteredCandidates = std::distance(data.filterIndices.begin(), filterIndicesEnd);

            if(task.numFilteredCandidates == 0){
                task.active = false; //no good mismatch ratio
            }else{
                //compaction. keep only data at positions given by filterIndices

                for(int i = 0; i < task.numFilteredCandidates; i++){
                    const int fromIndex = data.filterIndices[i];
                    const int toIndex = i;
                    
                    //std::cerr << "goodIndices[" << i << "]=" << fromIndex << "\n";

                    task.bestAlignments[toIndex] = task.bestAlignments[fromIndex];
                    task.bestAlignmentFlags[toIndex] = task.bestAlignmentFlags[fromIndex];
                    task.bestCandidateReadIds[toIndex] = task.bestCandidateReadIds[fromIndex];
                    task.bestCandidateLengths[toIndex] = task.bestCandidateLengths[fromIndex];

                    std::copy_n(
                        task.bestCandidateData + fromIndex * size_t(data.encodedSequencePitchInInts),
                        data.encodedSequencePitchInInts,
                        task.bestCandidateData + toIndex * size_t(data.encodedSequencePitchInInts)
                    );
                }

                auto calculateOverlapWeight = [](int anchorlength, int nOps, int overlapsize){
                    constexpr float maxErrorPercentInOverlap = 0.2f;

                    return 1.0f - sqrtf(nOps / (overlapsize * maxErrorPercentInOverlap));
                };

                for(int i = 0; i < task.numFilteredCandidates; i++){
                    task.bestAlignmentShifts[i] = task.bestAlignments[i].shift;

                    task.bestAlignmentWeights[i] = calculateOverlapWeight(
                        task.subjectSequenceLength, 
                        task.bestAlignments[i].nOps, 
                        task.bestAlignments[i].overlap
                    );
                }
            }
            
            
        }

        void removeInactiveTasks(BatchData& data){

            const int numSubjects = data.batchTasks.size();

            int numRemainingSubjects = 0;
            for(int i = 0; i < numSubjects; i++){
                const auto& task = data.batchTasks[i];
                if(task.active){
                    data.batchTasks[numRemainingSubjects] = task;
                    data.candidatesPerSubject[numRemainingSubjects] = task.numFilteredCandidates;
                    numRemainingSubjects++;
                }
            }

            data.batchTasks.erase(
                data.batchTasks.begin() + numRemainingSubjects,
                data.batchTasks.end()
            );

            std::partial_sum(
                data.candidatesPerSubject.begin(),
                data.candidatesPerSubject.begin() + numRemainingSubjects,
                data.candidatesPerSubjectPrefixSum.begin() + 1
            );

            for(int i = 0; i < numRemainingSubjects; i++){
                auto& task = data.batchTasks[i];
                const size_t offset = data.candidatesPerSubjectPrefixSum[i];
                task.subjectQualities = data.subjectQualities.data() + size_t(i) * data.qualityPitchInBytes;
                task.bestCandidateQualities = data.candidateQualities.data() + offset * data.qualityPitchInBytes;
            }            
        }


        void getQualities(BatchData& data,
                          const cpu::ContiguousReadStorage& readStorage){

            const int numSubjects = data.batchTasks.size();

            // get qualities subjects

            for(int i = 0; i < numSubjects; i++){
                data.filteredReadIds[i] = data.batchTasks[i].subjectReadId;
            }

            readStorage.gatherSequenceQualities(
                data.readStorageGatherHandle,
                data.filteredReadIds.data(),
                numSubjects,
                data.subjectQualities.data(),
                data.qualityPitchInBytes
            );

            // get qualities of candidates

            const int numCandidates = data.candidatesPerSubjectPrefixSum.back();
            for(int i = 0; i < numSubjects; i++){
                std::copy_n(
                    data.batchTasks[i].candidateReadIds,
                    data.batchTasks[i].numFilteredCandidates,
                    data.filteredReadIds.begin() + data.candidatesPerSubjectPrefixSum[i]
                );                
            }

            if(useSortedIdsForGather){
                readStorage.gatherSequenceQualitiesSpecial(
                    data.readStorageGatherHandle,
                    data.filteredReadIds.data(),
                    numCandidates,
                    data.candidateQualities.data(),
                    data.qualityPitchInBytes
                );
            }else{
                readStorage.gatherSequenceQualities(
                    data.readStorageGatherHandle,
                    data.filteredReadIds.data(),
                    numCandidates,
                    data.candidateQualities.data(),
                    data.qualityPitchInBytes
                );
            }
            
            //reverse quality scores
            for(int i = 0; i < numSubjects; i++){
                auto& task = data.batchTasks[i];
                for(int c = 0; c < task.numFilteredCandidates; c++){
                    if(task.bestAlignmentFlags[c] == BestAlignment_t::ReverseComplement){
                        std::reverse(
                            task.bestCandidateQualities + c * size_t(data.qualityPitchInBytes),
                            task.bestCandidateQualities + (c+1) * size_t(data.qualityPitchInBytes)
                        );
                    }
                }             
            }
        }

        void makeCandidateStrings(BatchData& data,
                  BatchData::Task& task){

            const size_t decodedpitch = data.decodedSequencePitchInBytes;
            const size_t encodedpitch = data.encodedSequencePitchInInts;

            for(int i = 0; i < task.numFilteredCandidates; i++){
                const unsigned int* const srcptr = task.bestCandidateData + i * encodedpitch;
                char* const destptr = data.decodedCandidateSequences.data() + i * decodedpitch;
                const int length = task.bestCandidateLengths[i];

                decode2BitSequence(
                    destptr,
                    srcptr,
                    length
                );
            }

            
            
            if(0) /*if(task.subjectReadId == 1)*/{
                for(int i = 0; i < task.numFilteredCandidates; i++){
                    std::cerr << task.bestCandidateReadIds[i] << " : ";
                    for(int k = 0; k < task.bestCandidateLengths[i]; k++){
                        std::cerr << data.decodedCandidateSequences[i * decodedpitch + k];
                    }
                    std::cerr << "\n";
                }                
            }
        }

        void buildMultipleSequenceAlignment(
                BatchData& data,
                BatchData::Task& task,
                const CorrectionOptions& correctionOptions){


            const char* const candidateQualityPtr = correctionOptions.useQualityScores ?
                                                    task.bestCandidateQualities
                                                    : nullptr;

            data.multipleSequenceAlignment.build(task.decodedSubjectSequence,
                                            task.subjectSequenceLength,
                                            data.decodedCandidateSequences.data(),
                                            task.bestCandidateLengths,
                                            task.numFilteredCandidates,
                                            task.bestAlignmentShifts,
                                            task.bestAlignmentWeights,
                                            task.subjectQualities,
                                            candidateQualityPtr,
                                            data.decodedSequencePitchInBytes,
                                            data.qualityPitchInBytes,
                                            correctionOptions.useQualityScores);
        }



        void removeCandidatesOfDifferentRegionFromMSA(
                BatchData& data,
                BatchData::Task& task,
                const CorrectionOptions& correctionOptions){

            constexpr int max_num_minimizations = 5;

            auto findCandidatesLambda = [&](){
                return findCandidatesOfDifferentRegion(task.decodedSubjectSequence,
                                                        task.subjectSequenceLength,
                                                        data.decodedCandidateSequences.data(),
                                                        task.bestCandidateLengths,
                                                        task.numFilteredCandidates,
                                                        data.decodedSequencePitchInBytes,
                                                        data.multipleSequenceAlignment.consensus.data(),
                                                        data.multipleSequenceAlignment.countsA.data(),
                                                        data.multipleSequenceAlignment.countsC.data(),
                                                        data.multipleSequenceAlignment.countsG.data(),
                                                        data.multipleSequenceAlignment.countsT.data(),
                                                        data.multipleSequenceAlignment.weightsA.data(),
                                                        data.multipleSequenceAlignment.weightsC.data(),
                                                        data.multipleSequenceAlignment.weightsG.data(),
                                                        data.multipleSequenceAlignment.weightsT.data(),
                                                        data.tmpnOps.data(), 
                                                        data.tmpoverlaps.data(),
                                                        data.multipleSequenceAlignment.subjectColumnsBegin_incl,
                                                        data.multipleSequenceAlignment.subjectColumnsEnd_excl,
                                                        task.bestAlignmentShifts,
                                                        correctionOptions.estimatedCoverage);
            };

            auto removeCandidatesOfDifferentRegion = [&](const auto& minimizationResult){

                if(minimizationResult.performedMinimization){
                    const int numDifferentRegionCandidates = minimizationResult.differentRegionCandidate.size();
                    assert(numDifferentRegionCandidates == task.numFilteredCandidates);
                    
                    if(0) /*if(task.subjectReadId == 1)*/{
                        std::cerr << "------\n";
                    }

                    //bool anyRemoved = false;
                    int cur = 0;
                    for(int i = 0; i < numDifferentRegionCandidates; i++){
                        if(!minimizationResult.differentRegionCandidate[i]){
                            
                            if(0) /*if(task.subjectReadId == 1)*/{
                                std::cerr << "keep " << i << "\n";                
                            }

                            task.bestAlignments[cur] = task.bestAlignments[i];
                            task.bestAlignmentShifts[cur] = task.bestAlignmentShifts[i];
                            task.bestAlignmentWeights[cur] = task.bestAlignmentWeights[i];
                            task.bestAlignmentFlags[cur] = task.bestAlignmentFlags[i];
                            task.bestCandidateReadIds[cur] = task.bestCandidateReadIds[i];
                            task.bestCandidateLengths[cur] = task.bestCandidateLengths[i];

                            std::copy_n(
                                task.bestCandidateData + i * data.encodedSequencePitchInInts,
                                data.encodedSequencePitchInInts,
                                task.bestCandidateData + cur * data.encodedSequencePitchInInts
                            );
                            std::copy_n(
                                task.bestCandidateQualities + i * data.qualityPitchInBytes,
                                data.qualityPitchInBytes,
                                task.bestCandidateQualities + cur * data.qualityPitchInBytes
                            );
                            std::copy_n(
                                data.decodedCandidateSequences.begin() + i * data.decodedSequencePitchInBytes,
                                data.decodedSequencePitchInBytes,
                                data.decodedCandidateSequences.begin() + cur * data.decodedSequencePitchInBytes
                            );

                            data.tmpnOps[cur] = data.tmpnOps[i];
                            data.tmpoverlaps[cur] = data.tmpoverlaps[i];

                            cur++;

                        }else{
                            //anyRemoved = true;
                        }
                    }
                    
                    if(0) /*if(task.subjectReadId == 1)*/{
                        std::cerr << "------\n";
                    }

                    task.numFilteredCandidates = cur;

                    //assert(anyRemoved);

                    //build minimized multiple sequence alignment

                    buildMultipleSequenceAlignment(
                        data,
                        task,
                        correctionOptions
                    );
                }
            };

            if(max_num_minimizations > 0){                

                for(int i = 0; i < task.numFilteredCandidates; i++){
                    data.tmpnOps[i] = task.bestAlignments[i].nOps;
                    data.tmpoverlaps[i] = task.bestAlignments[i].overlap;
                }

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
                std::cerr << "subjectColumnsBegin_incl = " << data.multipleSequenceAlignment.subjectColumnsBegin_incl << "\n";
                std::cerr << "subjectColumnsEnd_excl = " << data.multipleSequenceAlignment.subjectColumnsEnd_excl << "\n";
                //std::cerr << "lastColumn_excl = " << dataArrays.h_msa_column_properties[i].lastColumn_excl << "\n";
                std::cerr << "counts: \n";
                for(int k = 0; k < data.multipleSequenceAlignment.countsA.size(); k++){
                    std::cerr << data.multipleSequenceAlignment.countsA[k] << ' ';
                }
                std::cerr << "\n";
                for(int k = 0; k < data.multipleSequenceAlignment.countsC.size(); k++){
                    std::cerr << data.multipleSequenceAlignment.countsC[k] << ' ';
                }
                std::cerr << "\n";
                for(int k = 0; k < data.multipleSequenceAlignment.countsG.size(); k++){
                    std::cerr << data.multipleSequenceAlignment.countsG[k] << ' ';
                }
                std::cerr << "\n";
                for(int k = 0; k < data.multipleSequenceAlignment.countsT.size(); k++){
                    std::cerr << data.multipleSequenceAlignment.countsT[k] << ' ';
                }
                std::cerr << "\n";

                std::cerr << "weights: \n";
                for(int k = 0; k < data.multipleSequenceAlignment.weightsA.size(); k++){
                    std::cerr << data.multipleSequenceAlignment.weightsA[k] << ' ';
                }
                std::cerr << "\n";
                for(int k = 0; k < data.multipleSequenceAlignment.weightsC.size(); k++){
                    std::cerr << data.multipleSequenceAlignment.weightsC[k] << ' ';
                }
                std::cerr << "\n";
                for(int k = 0; k < data.multipleSequenceAlignment.weightsG.size(); k++){
                    std::cerr << data.multipleSequenceAlignment.weightsG[k] << ' ';
                }
                std::cerr << "\n";
                for(int k = 0; k < data.multipleSequenceAlignment.weightsT.size(); k++){
                    std::cerr << data.multipleSequenceAlignment.weightsT[k] << ' ';
                }
                std::cerr << "\n";

                std::cerr << "coverage: \n";
                for(int k = 0; k < data.multipleSequenceAlignment.coverage.size(); k++){
                    std::cerr << data.multipleSequenceAlignment.coverage[k] << ' ';
                }
                std::cerr << "\n";

                std::cerr << "support: \n";
                for(int k = 0; k < data.multipleSequenceAlignment.support.size(); k++){
                    std::cerr << data.multipleSequenceAlignment.support[k] << ' ';
                }
                std::cerr << "\n";

                std::cerr << "consensus: \n";
                for(int k = 0; k < data.multipleSequenceAlignment.consensus.size(); k++){
                    std::cerr << data.multipleSequenceAlignment.consensus[k] << ' ';
                }
                std::cerr << "\n";
            }
#endif            
        }



        void correctSubjectClassic(
                BatchData& data,
                BatchData::Task& task,
                const CorrectionOptions& correctionOptions){

            assert(correctionOptions.correctionType == CorrectionType::Classic);

            const int subjectColumnsBegin_incl = data.multipleSequenceAlignment.subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = data.multipleSequenceAlignment.subjectColumnsEnd_excl;

            task.msaProperties = getMSAProperties2(
                data.multipleSequenceAlignment.support.data(),
                data.multipleSequenceAlignment.coverage.data(),
                subjectColumnsBegin_incl,
                subjectColumnsEnd_excl,
                correctionOptions.estimatedErrorrate,
                correctionOptions.estimatedCoverage,
                correctionOptions.m_coverage
            );

            task.subjectCorrection = getCorrectedSubjectNew(
                data.multipleSequenceAlignment.consensus.data() + subjectColumnsBegin_incl,
                data.multipleSequenceAlignment.support.data() + subjectColumnsBegin_incl,
                data.multipleSequenceAlignment.coverage.data() + subjectColumnsBegin_incl,
                data.multipleSequenceAlignment.origCoverages.data() + subjectColumnsBegin_incl,
                task.subjectSequenceLength,
                task.decodedSubjectSequence,
                subjectColumnsBegin_incl,
                data.decodedCandidateSequences.data(),
                task.numFilteredCandidates,
                task.bestAlignmentWeights,
                task.bestCandidateLengths,
                task.bestAlignmentShifts,
                data.decodedSequencePitchInBytes,
                task.msaProperties,
                correctionOptions.estimatedErrorrate,
                correctionOptions.estimatedCoverage,
                correctionOptions.m_coverage,
                correctionOptions.kmerlength,
                task.subjectReadId
            );

            task.msaProperties.isHQ = task.subjectCorrection.isHQ;
            
        }

        void correctSubjectForest(
                BatchData& data,
                BatchData::Task& task,
                const CorrectionOptions& correctionOptions){

            assert(correctionOptions.correctionType == CorrectionType::Forest);

            task.msaProperties = getMSAProperties2(
                data.multipleSequenceAlignment.support.data(),
                data.multipleSequenceAlignment.coverage.data(),
                subjectColumnsBegin_incl,
                subjectColumnsEnd_excl,
                correctionOptions.estimatedErrorrate,
                correctionOptions.estimatedCoverage,
                correctionOptions.m_coverage
            );

            
            /*
                auto asdf = data.forestClassifier1->shouldCorrect_forest(...)
            */

            //this must be computed using the results of forest classifier
            /*            
                struct CorrectionResult{
                    bool isCorrected;
                    bool isHQ;
                    std::string correctedSequence;
                };
            */
            task.subjectCorrection = CorrectionResult{};


            task.msaProperties.isHQ = task.subjectCorrection.isHQ;

        }

        void correctSubject(
                BatchData& data,
                BatchData::Task& task,
                const CorrectionOptions& correctionOptions){

            if(correctionOptions.correctionType == CorrectionType::Classic){
                correctSubjectClassic(data, task, correctionOptions);
            }else if(correctionOptions.correctionType == CorrectionType::Forest){
                correctSubjectForest(data, task, correctionOptions);
            }else{
                assert(false);
            }
            
        }

        void correctCandidates(
                BatchData& data,
                BatchData::Task& task,
                const CorrectionOptions& correctionOptions){

            task.candidateCorrections = getCorrectedCandidatesNew(
                data.multipleSequenceAlignment.consensus.data(),
                data.multipleSequenceAlignment.support.data(),
                data.multipleSequenceAlignment.coverage.data(),
                data.multipleSequenceAlignment.nColumns,
                data.multipleSequenceAlignment.subjectColumnsBegin_incl,
                data.multipleSequenceAlignment.subjectColumnsEnd_excl,
                task.bestAlignmentShifts,
                task.bestCandidateLengths,
                data.multipleSequenceAlignment.nCandidates,
                correctionOptions.estimatedErrorrate,
                correctionOptions.estimatedCoverage,
                correctionOptions.m_coverage,
                correctionOptions.new_columns_to_correct
            );
            
            if(0) /*if(task.subjectReadId == 1)*/{
                for(const auto& correctedCandidate : task.candidateCorrections){
                    const read_number candidateId = task.bestCandidateReadIds[correctedCandidate.index];
                    
                    if(task.bestAlignmentFlags[correctedCandidate.index] == BestAlignment_t::Forward){
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

        void setCorrectionStatusFlags( 
                    BatchData& data,
                    BatchData::Task& task,
                    std::uint8_t* correctionStatusFlagsPerRead){
            if(task.active){
                if(task.subjectCorrection.isCorrected){
                    if(task.msaProperties.isHQ){
                        correctionStatusFlagsPerRead[task.subjectReadId] |= readCorrectedAsHQAnchor;
                    }
                }else{
                    correctionStatusFlagsPerRead[task.subjectReadId] |= readCouldNotBeCorrectedAsAnchor;
                }
            }
        }
        
        void makeOutputDataOfTask(
                BatchData& data,
                BatchData::Task& task,
                const cpu::ContiguousReadStorage& readStorage,
                const std::uint8_t* correctionStatusFlagsPerRead){            
               
            if(task.active){
                
                if(task.subjectCorrection.isCorrected){
                    auto& correctedSequenceString = task.subjectCorrection.correctedSequence;
                    const int correctedlength = correctedSequenceString.length();
                    const bool originalReadContainsN = readStorage.readContainsN(task.subjectReadId);
                    
                    TempCorrectedSequence tmp;
                    
                    if(!originalReadContainsN){
                        const int maxEdits = correctedlength / 7;
                        int edits = 0;
                        for(int i = 0; i < correctedlength && edits <= maxEdits; i++){
                            if(correctedSequenceString[i] != task.decodedSubjectSequence[i]){
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
                    tmp.readId = task.subjectReadId;
                    tmp.sequence = std::move(correctedSequenceString); 

                    // if(tmp.readId == 32141191 /* || tmp.readId == 10307280 || tmp.readId == 42537816*/){
                    //     std::cerr << "readid = " << tmp.readId << ", anchor\n";
                    //     std::cerr << "hq = " << tmp.hq;
                    //     if(!tmp.useEdits){
                    //         std::cerr << ", sequence = " << tmp.sequence << "\n";
                    //     }else{
                    //         std::cerr << "numEdits = " << tmp.edits.size();
                    //         std::cerr << "\nedits: \n";
                    //         for(int i = 0; i < int(tmp.edits.size()); i++){
                    //             std::cerr << tmp.edits[i].base << ' ' << tmp.edits[i].pos << "\n";
                    //         }
                    //     }                           
                    // }
                    
                    data.outputData.anchorCorrections.emplace_back(std::move(tmp));
                }
                
                
                
                for(const auto& correctedCandidate : task.candidateCorrections){
                    const read_number candidateId = task.bestCandidateReadIds[correctedCandidate.index];
                    
                    bool savingIsOk = false;
                    
                    const std::uint8_t mask = correctionStatusFlagsPerRead[candidateId];
                    if(!(mask & readCorrectedAsHQAnchor)) {
                        savingIsOk = true;
                    }
                    
                    if (savingIsOk) {                            
                        
                        TempCorrectedSequence tmp;
                        
                        tmp.type = TempCorrectedSequence::Type::Candidate;
                        tmp.readId = candidateId;
                        tmp.shift = correctedCandidate.shift;

                        const bool candidateIsForward = task.bestAlignmentFlags[correctedCandidate.index] == BestAlignment_t::Forward;

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
                        
                        const bool originalCandidateReadContainsN = readStorage.readContainsN(candidateId);
                        
                        if(!originalCandidateReadContainsN){
                            const std::size_t offset = correctedCandidate.index * data.decodedSequencePitchInBytes;
                            const char* const uncorrectedCandidate = &data.decodedCandidateSequences[offset];
                            const int uncorrectedCandidateLength = task.bestCandidateLengths[correctedCandidate.index];
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

                        // if(tmp.readId == 32141191){
                        //     std::cerr << "readid = " << tmp.readId << ", as candidate of anchor with id " << task.subjectReadId << "\n";
                        //     std::cerr << "hq = " << tmp.hq;
                        //     if(!tmp.useEdits){
                        //         std::cerr << ", sequence = " << tmp.sequence << "\n";
                        //     }else{
                        //         std::cerr << "numEdits = " << tmp.edits.size();
                        //         std::cerr << "\nedits: \n";
                        //         for(int i = 0; i < int(tmp.edits.size()); i++){
                        //             std::cerr << tmp.edits[i].base << ' ' << tmp.edits[i].pos << "\n";
                        //         }
                        //     }                            
                        // }
                        
                        data.outputData.candidateCorrections.emplace_back(std::move(tmp));
                    }
                }
            }
        }


        void encodeOutputData(BatchData& data){

            data.outputData.encodedAnchorCorrections.reserve(data.outputData.anchorCorrections.size());
            data.outputData.encodedCandidateCorrections.reserve(data.outputData.candidateCorrections.size());

            for(const auto& tmp : data.outputData.anchorCorrections){
                data.outputData.encodedAnchorCorrections.emplace_back(tmp.encode());
            }

            for(const auto& tmp : data.outputData.candidateCorrections){
                data.outputData.encodedCandidateCorrections.emplace_back(tmp.encode());
            }
        }


MemoryFileFixedSize<EncodedTempCorrectedSequence>
correct_cpu(
    const GoodAlignmentProperties& goodAlignmentProperties,
    const CorrectionOptions& correctionOptions,
    const RuntimeOptions& runtimeOptions,
    const FileOptions& fileOptions,
    const MemoryOptions& memoryOptions,
    const SequenceFileProperties& sequenceFileProperties,
    Minhasher& minhasher,
    cpu::ContiguousReadStorage& readStorage
){

    omp_set_num_threads(runtimeOptions.threads);


    // std::ofstream outputstream;

    // outputstream = std::move(std::ofstream(tmpfiles[0]));
    // if(!outputstream){
    //     throw std::runtime_error("Could not open output file " + tmpfiles[0]);
    // }

    const auto rsMemInfo = readStorage.getMemoryInfo();
    const auto mhMemInfo = minhasher.getMemoryInfo();

    std::size_t memoryAvailableBytesHost = memoryOptions.memoryTotalLimit;
    if(memoryAvailableBytesHost > rsMemInfo.host){
        memoryAvailableBytesHost -= rsMemInfo.host;
    }else{
        memoryAvailableBytesHost = 0;
    }
    if(memoryAvailableBytesHost > mhMemInfo.host){
        memoryAvailableBytesHost -= mhMemInfo.host;
    }else{
        memoryAvailableBytesHost = 0;
    }

    std::unique_ptr<std::uint8_t[]> correctionStatusFlagsPerRead = std::make_unique<std::uint8_t[]>(sequenceFileProperties.nReads);

    #pragma omp parallel for
    for(read_number i = 0; i < sequenceFileProperties.nReads; i++){
        correctionStatusFlagsPerRead[i] = 0;
    }

    std::cerr << "correctionStatusFlagsPerRead bytes: " << sizeof(std::uint8_t) * sequenceFileProperties.nReads / 1024. / 1024. << " MB\n";

    if(memoryAvailableBytesHost > sizeof(std::uint8_t) * sequenceFileProperties.nReads){
        memoryAvailableBytesHost -= sizeof(std::uint8_t) * sequenceFileProperties.nReads;
    }else{
        memoryAvailableBytesHost = 0;
    }

    const std::size_t availableMemoryInBytes = memoryAvailableBytesHost; //getAvailableMemoryInKB() * 1024;
    std::size_t memoryForPartialResultsInBytes = 0;

    if(availableMemoryInBytes > 2*(std::size_t(1) << 30)){
        memoryForPartialResultsInBytes = availableMemoryInBytes - 2*(std::size_t(1) << 30);
    }

    const std::string tmpfilename{fileOptions.tempdirectory + "/" + "MemoryFileFixedSizetmp"};
    MemoryFileFixedSize<EncodedTempCorrectedSequence> partialResults(memoryForPartialResultsInBytes, tmpfilename);


#ifndef DO_PROFILE
    cpu::RangeGenerator<read_number> readIdGenerator(sequenceFileProperties.nReads);
#else
    cpu::RangeGenerator<read_number> readIdGenerator(num_reads_to_profile);
#endif


    auto saveCorrectedSequence = [&](TempCorrectedSequence tmp, EncodedTempCorrectedSequence encoded){
          //std::unique_lock<std::mutex> l(outputstreammutex);
          //std::cerr << tmp.readId  << " hq " << tmp.hq << " " << "useedits " << tmp.useEdits << " emptyedits " << tmp.edits.empty() << "\n";
          if(!(tmp.hq && tmp.useEdits && tmp.edits.empty())){
              //std::cerr << tmp.readId << " " << tmp << '\n';
              partialResults.storeElement(std::move(encoded));
          }
      };

    // std::size_t nLocksForProcessedFlags = runtimeOptions.threads * 1000;
    // std::unique_ptr<std::mutex[]> locksForProcessedFlags(new std::mutex[nLocksForProcessedFlags]);


    // auto lock = [&](read_number readId){
    //     read_number index = readId % nLocksForProcessedFlags;
    //     locksForProcessedFlags[index].lock();
    // };

    // auto unlock = [&](read_number readId){
    //     read_number index = readId % nLocksForProcessedFlags;
    //     locksForProcessedFlags[index].unlock();
    // };


    BackgroundThread outputThread(true);

    TimeMeasurements timingsOfAllThreads;

    std::shared_ptr<ForestClassifier> forestClassifier1;
    if(correctionOptions.correctionType == CorrectionType::Forest){
        forestClassifier1 = std::make_shared<ForestClassifier>(fileOptions.mlForestfile);
    }


    // std::ifstream interestingstream("interestingIds.txt");
    // if(interestingstream){
    //     std::string line;
    //     while(std::getline(interestingstream, line)){
    //         auto tokens = split(line, ' ');
    //         if(!tokens.empty()){
    //             read_number n = std::stoull(tokens[0]);
    //             interestingReadIds.emplace_back(n);
    //         }
    //     }
    //     auto it = std::unique(interestingReadIds.begin(), interestingReadIds.end());
    //     interestingReadIds.erase(it, interestingReadIds.end());

    //     std::cerr << "Looking for " << interestingReadIds.size() << " interesting read ids\n";
    // }else{
    //     std::cerr << "Looking for no interesting read id\n";
    // }

    auto showProgress = [&](auto totalCount, auto seconds){
        if(runtimeOptions.showProgress){

            printf("Processed %10u of %10lu reads (Runtime: %03d:%02d:%02d)\r",
                    totalCount, sequenceFileProperties.nReads,
                    int(seconds / 3600),
                    int(seconds / 60) % 60,
                    int(seconds) % 60);
            std::cout.flush();
        }

        if(totalCount == sequenceFileProperties.nReads){
            std::cerr << '\n';
        }
    };

    auto updateShowProgressInterval = [](auto duration){
        return duration;
    };

    ProgressThread<read_number> progressThread(sequenceFileProperties.nReads, showProgress, updateShowProgressInterval);

    const int numThreads = runtimeOptions.threads;

    #pragma omp parallel
    {
        //const int threadId = omp_get_thread_num();

        BatchData batchData;
        batchData.subjectReadIds.resize(correctionOptions.batchsize);
        batchData.encodedSequencePitchInInts = getEncodedNumInts2Bit(sequenceFileProperties.maxSequenceLength);
        batchData.decodedSequencePitchInBytes = sequenceFileProperties.maxSequenceLength;
        batchData.qualityPitchInBytes = sequenceFileProperties.maxSequenceLength;
        batchData.forestClassifier1 = forestClassifier1;

        while(!(readIdGenerator.empty())){

            batchData.subjectReadIds.resize(correctionOptions.batchsize);

            auto readIdsEnd = readIdGenerator.next_n_into_buffer(
                correctionOptions.batchsize, 
                batchData.subjectReadIds.begin()
            );
            
            batchData.subjectReadIds.erase(readIdsEnd, batchData.subjectReadIds.end());

            if(batchData.subjectReadIds.empty()){
                continue;
            }

            #ifdef ENABLE_TIMING
            auto tpa = std::chrono::system_clock::now();
            #endif

            getSubjectSequenceData(batchData, readStorage);

            #ifdef ENABLE_TIMING
            batchData.timings.getSubjectSequenceDataTimeTotal += std::chrono::system_clock::now() - tpa;
            #endif

            #ifdef ENABLE_TIMING
            tpa = std::chrono::system_clock::now();
            #endif

            determineCandidateReadIds(batchData, minhasher, readStorage, correctionOptions);

            #ifdef ENABLE_TIMING
            batchData.timings.getCandidatesTimeTotal += std::chrono::system_clock::now() - tpa;
            #endif

            #ifdef ENABLE_TIMING
            tpa = std::chrono::system_clock::now();
            #endif

            getCandidateSequenceData(batchData, readStorage);

            #ifdef ENABLE_TIMING
            batchData.timings.copyCandidateDataToBufferTimeTotal += std::chrono::system_clock::now() - tpa;
            #endif

            makeBatchTasks(batchData);

            for(auto& batchTask : batchData.batchTasks){
                if(batchTask.active){

                    #ifdef ENABLE_TIMING
                    tpa = std::chrono::system_clock::now();
                    #endif

                    getCandidateAlignments(
                        batchData,
                        batchTask,
                        goodAlignmentProperties,
                        correctionOptions
                    );

                    #ifdef ENABLE_TIMING
                    batchData.timings.getAlignmentsTimeTotal += std::chrono::system_clock::now() - tpa;
                    #endif

                    #ifdef ENABLE_TIMING
                    tpa = std::chrono::system_clock::now();
                    #endif

                    gatherBestAlignmentData(batchData, batchTask);

                    #ifdef ENABLE_TIMING
                    batchData.timings.gatherBestAlignmentDataTimeTotal += std::chrono::system_clock::now() - tpa;
                    #endif

                    #ifdef ENABLE_TIMING
                    tpa = std::chrono::system_clock::now();
                    #endif

                    filterBestAlignmentsByMismatchRatio(
                        batchData,
                        batchTask,
                        correctionOptions,
                        goodAlignmentProperties
                    );

                    #ifdef ENABLE_TIMING
                    batchData.timings.mismatchRatioFilteringTimeTotal += std::chrono::system_clock::now() - tpa;
                    #endif

                }
            }

            removeInactiveTasks(batchData);

            if(correctionOptions.useQualityScores){

                #ifdef ENABLE_TIMING
                tpa = std::chrono::system_clock::now();
                #endif

                getQualities(batchData, readStorage);

                #ifdef ENABLE_TIMING
                batchData.timings.fetchQualitiesTimeTotal += std::chrono::system_clock::now() - tpa;
                #endif

            }

            for(auto& batchTask : batchData.batchTasks){

                #ifdef ENABLE_TIMING
                tpa = std::chrono::system_clock::now();
                #endif

                makeCandidateStrings(batchData, batchTask);

                #ifdef ENABLE_TIMING
                batchData.timings.makeCandidateStringsTimeTotal += std::chrono::system_clock::now() - tpa;
                #endif

                #ifdef ENABLE_TIMING
                tpa = std::chrono::system_clock::now();
                #endif

                buildMultipleSequenceAlignment(batchData, batchTask, correctionOptions);

                #ifdef ENABLE_TIMING
                batchData.timings.msaFindConsensusTimeTotal += std::chrono::system_clock::now() - tpa;
                #endif

                #ifdef ENABLE_TIMING
                tpa = std::chrono::system_clock::now();
                #endif

                removeCandidatesOfDifferentRegionFromMSA(batchData, batchTask, correctionOptions);

                #ifdef ENABLE_TIMING
                batchData.timings.msaMinimizationTimeTotal += std::chrono::system_clock::now() - tpa;
                #endif

                #ifdef ENABLE_TIMING
                tpa = std::chrono::system_clock::now();
                #endif

                correctSubject(batchData, batchTask, correctionOptions);

                #ifdef ENABLE_TIMING
                batchData.timings.msaCorrectSubjectTimeTotal += std::chrono::system_clock::now() - tpa;
                #endif

                setCorrectionStatusFlags(batchData, batchTask, correctionStatusFlagsPerRead.get());

                if(batchTask.msaProperties.isHQ && correctionOptions.correctCandidates){

                    #ifdef ENABLE_TIMING
                    tpa = std::chrono::system_clock::now();
                    #endif

                    correctCandidates(batchData, batchTask, correctionOptions);

                    #ifdef ENABLE_TIMING
                    batchData.timings.msaCorrectCandidatesTimeTotal += std::chrono::system_clock::now() - tpa;
                    #endif

                }
                
                makeOutputDataOfTask(batchData, batchTask, readStorage, correctionStatusFlagsPerRead.get());
            }

            //makeOutputData(batchData, readStorage, correctionStatusFlagsPerRead.data());

            encodeOutputData(batchData);

            auto outputfunction = [&, outputData = std::move(batchData.outputData)](){
                for(int i = 0; i < int(outputData.anchorCorrections.size()); i++){
                    saveCorrectedSequence(
                        std::move(outputData.anchorCorrections[i]), 
                        std::move(outputData.encodedAnchorCorrections[i])
                    );
                }

                for(int i = 0; i < int(outputData.candidateCorrections.size()); i++){
                    saveCorrectedSequence(
                        std::move(outputData.candidateCorrections[i]), 
                        std::move(outputData.encodedCandidateCorrections[i])
                    );
                }
            };

            outputThread.enqueue(std::move(outputfunction));

            progressThread.addProgress(batchData.subjectReadIds.size()); 
        } //while unprocessed reads exist loop end   

        #pragma omp critical
        {
            timingsOfAllThreads += batchData.timings;
        }


    } // parallel end

    progressThread.finished();

    outputThread.stopThread(BackgroundThread::StopType::FinishAndStop);

    //outputstream.flush();
    partialResults.flush();

    #ifdef ENABLE_TIMING

    auto totalDurationOfThreads = timingsOfAllThreads.getSumOfDurations();

    auto printDuration = [&](const auto& name, const auto& duration){
        std::cout << "# average time per thread ("<< name << "): "
                  << duration.count() / numThreads  << " s. "
                  << (100.0 * duration.count() / totalDurationOfThreads.count()) << " %."<< std::endl;
    };

    #define printme(x) printDuration((#x), timingsOfAllThreads.x);

    printme(getSubjectSequenceDataTimeTotal);
    printme(getCandidatesTimeTotal);
    printme(copyCandidateDataToBufferTimeTotal);
    printme(getAlignmentsTimeTotal);
    printme(findBestAlignmentDirectionTimeTotal);
    printme(gatherBestAlignmentDataTimeTotal);
    printme(mismatchRatioFilteringTimeTotal);
    printme(compactBestAlignmentDataTimeTotal);
    printme(fetchQualitiesTimeTotal);
    printme(makeCandidateStringsTimeTotal);
    printme(msaAddSequencesTimeTotal);
    printme(msaFindConsensusTimeTotal);
    printme(msaMinimizationTimeTotal);
    printme(msaCorrectSubjectTimeTotal);
    printme(msaCorrectCandidatesTimeTotal);

    #undef printme

    #endif

#ifdef DO_PROFILE

    return;

#endif

    return partialResults;

}






#if 0
        void correctSubjectWithForest(...){

            auto MSAFeatures = extractFeatures(data.multipleSequenceAlignment.consensus.data(),
                                            data.multipleSequenceAlignment.support.data(),
                                            data.multipleSequenceAlignment.coverage.data(),
                                            data.multipleSequenceAlignment.origCoverages.data(),
                                            data.multipleSequenceAlignment.nColumns,
                                            data.multipleSequenceAlignment.subjectColumnsBegin_incl,
                                            data.multipleSequenceAlignment.subjectColumnsEnd_excl,
                                            task.original_subject_string,
                                            correctionOptions.kmerlength, 0.5f,
                                            correctionOptions.estimatedCoverage);

            task.corrected_subject = task.original_subject_string;

            for(const auto& msafeature : MSAFeatures){
                constexpr float maxgini = 0.05f;
                constexpr float forest_correction_fraction = 0.5f;

                const bool doCorrect = forestClassifier.shouldCorrect(
                                                msafeature.position_support,
                                                msafeature.position_coverage,
                                                msafeature.alignment_coverage,
                                                msafeature.dataset_coverage,
                                                msafeature.min_support,
                                                msafeature.min_coverage,
                                                msafeature.max_support,
                                                msafeature.max_coverage,
                                                msafeature.mean_support,
                                                msafeature.mean_coverage,
                                                msafeature.median_support,
                                                msafeature.median_coverage,
                                                maxgini,
                                                forest_correction_fraction);

                if(doCorrect){
                    task.corrected = true;

                    const int globalIndex = data.multipleSequenceAlignment.subjectColumnsBegin_incl + msafeature.position;
                    task.corrected_subject[msafeature.position] = data.multipleSequenceAlignment.consensus[globalIndex];
                }
            }
        }
#endif

#if 0
        void correctSubjectWithNeuralNetwork(...){
            assert(false);
            /*auto MSAFeatures3 = extractFeatures3_2(
                                    data.multipleSequenceAlignment.countsA.data(),
                                    data.multipleSequenceAlignment.countsC.data(),
                                    data.multipleSequenceAlignment.countsG.data(),
                                    data.multipleSequenceAlignment.countsT.data(),
                                    data.multipleSequenceAlignment.weightsA.data(),
                                    data.multipleSequenceAlignment.weightsC.data(),
                                    data.multipleSequenceAlignment.weightsG.data(),
                                    data.multipleSequenceAlignment.weightsT.data(),
                                    data.multipleSequenceAlignment.nRows,
                                    data.multipleSequenceAlignment.columnProperties.columnsToCheck,
                                    data.multipleSequenceAlignment.consensus.data(),
                                    data.multipleSequenceAlignment.support.data(),
                                    data.multipleSequenceAlignment.coverage.data(),
                                    data.multipleSequenceAlignment.origCoverages.data(),
                                    data.multipleSequenceAlignment.columnProperties.subjectColumnsBegin_incl,
                                    data.multipleSequenceAlignment.columnProperties.subjectColumnsEnd_excl,
                                    task.original_subject_string,
                                    correctionOptions.estimatedCoverage);

                std::vector<float> predictions = nnClassifier.infer(MSAFeatures3);
                assert(predictions.size() == MSAFeatures3.size());

                task.corrected_subject = task.original_subject_string;

                for(size_t index = 0; index < predictions.size(); index++){
                    constexpr float threshold = 0.8;
                    const auto& msafeature = MSAFeatures3[index];

                    if(predictions[index] >= threshold){
                        task.corrected = true;

                        const int globalIndex = data.multipleSequenceAlignment.columnProperties.subjectColumnsBegin_incl + msafeature.position;
                        task.corrected_subject[msafeature.position] = data.multipleSequenceAlignment.consensus[globalIndex];
                    }
                }*/
        }
#endif 


    #if 0
                std::cout << correctionTasks[0].readId << " MSA: rows = " << (int(bestAlignments.size()) + 1) << " columns = " << multipleSequenceAlignment.nColumns << "\n";
                std::cout << "Consensus:\n   ";
                for(int i = 0; i < multipleSequenceAlignment.nColumns; i++){
                    std::cout << multipleSequenceAlignment.consensus[i];
                }
                std::cout << '\n';

                /*printSequencesInMSA(std::cout,
                                    correctionTasks[0].original_subject_string.c_str(), 
                                    subjectLength,
                                    bestCandidateStrings.data(),
                                    bestCandidateLengths.data(),
                                    int(bestAlignments.size()),
                                    bestAlignmentShifts.data(),
                                    multipleSequenceAlignment.subjectColumnsBegin_incl,
                                    multipleSequenceAlignment.subjectColumnsEnd_excl,
                                    multipleSequenceAlignment.nColumns,
                                    sequenceFileProperties.maxSequenceLength);*/

                printSequencesInMSAConsEq(std::cout,
                                    correctionTasks[0].original_subject_string.c_str(),
                                    subjectLength,
                                    bestCandidateStrings.data(),
                                    bestCandidateLengths.data(),
                                    int(bestAlignments.size()),
                                    bestAlignmentShifts.data(),
                                    multipleSequenceAlignment.consensus.data(),
                                    multipleSequenceAlignment.subjectColumnsBegin_incl,
                                    multipleSequenceAlignment.subjectColumnsEnd_excl,
                                    multipleSequenceAlignment.nColumns,
                                    sequenceFileProperties.maxSequenceLength);
    #endif


}
}


#ifdef MSA_IMPLICIT
#undef MSA_IMPLICIT
#endif

#ifdef ENABLE_TIMING
#undef ENABLE_TIMING
#endif
