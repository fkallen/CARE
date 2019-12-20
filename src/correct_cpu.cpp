//#include <cpu_correction_thread.hpp>

#include <config.hpp>

#include "options.hpp"

#include <minhasher.hpp>
#include <readstorage.hpp>
#include "cpu_alignment.hpp"
#include "bestalignment.hpp"
#include <msa.hpp>
#include "qualityscoreweights.hpp"
#include "rangegenerator.hpp"
#include "featureextractor.hpp"
#include "forestclassifier.hpp"
//#include "nn_classifier.hpp"
#include <sequencefileio.hpp>
#include "cpu_correction_core.hpp"
#include <threadpool.hpp>
#include <memoryfile.hpp>
#include <util.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
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

        struct TaskData;

        struct CorrectionTask{
            CorrectionTask(){}

            CorrectionTask(read_number readId)
                :   active(true),
                    corrected(false),
                    readId(readId)
                    {}

            CorrectionTask(const CorrectionTask& other)
                : active(other.active),
                corrected(other.corrected),
                readId(other.readId),
                encodedSubjectPtr(other.encodedSubjectPtr),
                subjectQualityPtr(other.subjectQualityPtr),
                taskDataPtr(other.taskDataPtr),
                original_subject_string(other.original_subject_string),
                corrected_subject(other.corrected_subject),
                correctedCandidates(other.correctedCandidates),
                candidate_read_ids(other.candidate_read_ids),
                corrected_candidates_shifts(other.corrected_candidates_shifts),
                uncorrectedPositionsNoConsensus(other.uncorrectedPositionsNoConsensus),
                anchoroutput(other.anchoroutput),
                candidatesoutput(other.candidatesoutput){

            }

            CorrectionTask(CorrectionTask&& other){
                operator=(other);
            }

            CorrectionTask& operator=(const CorrectionTask& other){
                CorrectionTask tmp(other);
                swap(*this, tmp);
                return *this;
            }

            CorrectionTask& operator=(CorrectionTask&& other){
                swap(*this, other);
                return *this;
            }

            friend void swap(CorrectionTask& l, CorrectionTask& r) noexcept{
                using std::swap;

                swap(l.active, r.active);
                swap(l.corrected, r.corrected);
                swap(l.readId, r.readId);
                swap(l.encodedSubjectPtr, r.encodedSubjectPtr);
                swap(l.subjectQualityPtr, r.subjectQualityPtr);
                swap(l.taskDataPtr, r.taskDataPtr);
                swap(l.original_subject_string, r.original_subject_string);
                swap(l.corrected_subject, r.corrected_subject);
                swap(l.correctedCandidates, r.correctedCandidates);
                swap(l.candidate_read_ids, r.candidate_read_ids);
                swap(l.corrected_candidates_shifts, r.corrected_candidates_shifts);
                swap(l.uncorrectedPositionsNoConsensus, r.uncorrectedPositionsNoConsensus);
                swap(l.anchoroutput, r.anchoroutput);
                swap(l.candidatesoutput, r.candidatesoutput);
            }

            bool active;
            bool corrected;
            read_number readId;
            const unsigned int* encodedSubjectPtr;
            const char* subjectQualityPtr;
            TaskData* taskDataPtr;
            std::string original_subject_string;
            std::string corrected_subject;
            std::vector<CorrectedCandidate> correctedCandidates;
            std::vector<read_number> candidate_read_ids;
            std::vector<int> corrected_candidates_shifts;
            std::vector<int> uncorrectedPositionsNoConsensus;
            TempCorrectedSequence anchoroutput;
            std::vector<TempCorrectedSequence> candidatesoutput;
        };

        struct TaskData{
            MultipleSequenceAlignment multipleSequenceAlignment;
            MSAProperties msaProperties;

            shd::CpuAlignmentHandle alignmentHandle;
            Minhasher::Handle minhashHandle;
            ContiguousReadStorage::GatherHandle readStorageGatherHandle;

            std::vector<unsigned int> subjectsequence;
            std::vector<unsigned int> candidateData;
            std::vector<unsigned int> candidateRevcData;
            std::vector<int> candidateLengths;
            int max_candidate_length = 0;
            std::vector<unsigned int*> candidateDataPtrs;
            std::vector<unsigned int*> candidateRevcDataPtrs;

            // std::vector<unsigned int> subjectsequenceHiLo;
            // std::vector<unsigned int> candidateDataHiLo;
            // std::vector<unsigned int> candidateRevcDataHiLo;

            std::vector<SHDResult> forwardAlignments;
            std::vector<SHDResult> revcAlignments;
            std::vector<BestAlignment_t> alignmentFlags;
            size_t numGoodAlignmentFlags;

            std::vector<SHDResult> bestAlignments;
            std::vector<BestAlignment_t> bestAlignmentFlags;
            std::vector<int> bestAlignmentShifts;
            std::vector<float> bestAlignmentWeights;
            std::vector<read_number> bestCandidateReadIds;
            std::vector<int> bestCandidateLengths;
            std::vector<unsigned int> bestCandidateData;
            std::vector<unsigned int*> bestCandidatePtrs;

            std::vector<char> bestCandidateQualityData;
            std::vector<char*> bestCandidateQualityPtrs;
            //std::vector<std::string> bestCandidateStrings;
            std::vector<char> bestCandidateStrings;

            std::vector<MSAFeature> msaforestfeatures;

            std::vector<int> indicesOfCandidatesEqualToSubject;
        };



        struct ThreadData{
            std::vector<read_number> subjectReadIds;
            std::vector<read_number> candidateReadIds;
            std::vector<unsigned int> subjectSequencesData;
            std::vector<unsigned int> candidateSequencesData;
            std::vector<unsigned int> candidateSequencesRevcData;
            std::vector<int> subjectSequencesLengths;
            std::vector<int> candidateSequencesLengths;
            std::vector<char> subjectQualities;
            std::vector<char> candidateQualities;

            std::vector<read_number> tempReadIds;
            std::vector<int> permutationIndices;
        };

        void getSubjectSequencesData(
                ThreadData& threadData,
                const cpu::ContiguousReadStorage& readStorage,
                size_t encodedSequencePitchInInts){

            const int numSubjects = threadData.subjectReadIds.size();

            threadData.subjectSequencesData.resize(encodedSequencePitchInInts * numSubjects);
            threadData.subjectSequencesLengths.resize(numSubjects);

            for(int i = 0; i < numSubjects; i++){
                const read_number subjectId = threadData.subjectReadIds[i];
                const int length = readStorage.fetchSequenceLength(subjectId);
                threadData.subjectSequencesLengths[i] = length;
            }


            constexpr size_t prefetch_distance_sequences = 4;

            for(size_t i = 0; i < numSubjects && i < prefetch_distance_sequences; ++i) {
                const read_number nextSubjectReadId = threadData.subjectReadIds[i];
                const unsigned int* nextsequenceptr = (const unsigned int*)readStorage.fetchSequenceData_ptr(nextSubjectReadId);
                __builtin_prefetch(nextsequenceptr, 0, 0);
            }

            for(size_t i = 0; i < numSubjects; i++){
                if(i + prefetch_distance_sequences < numSubjects) {
                    const read_number nextSubjectReadId = threadData.subjectReadIds[i];
                    const unsigned int* nextsequenceptr = (const unsigned int*)readStorage.fetchSequenceData_ptr(nextSubjectReadId);
                    __builtin_prefetch(nextsequenceptr, 0, 0);
                }

                const read_number subjectReadId = threadData.subjectReadIds[i];
                const unsigned int* subjectPtr = (const unsigned int*)readStorage.fetchSequenceData_ptr(subjectReadId);
                const int subjectLength = threadData.subjectSequencesLengths[i];
                const int subjectNumInts = getEncodedNumInts2Bit(subjectLength);

                unsigned int* const subjectDataBegin = threadData.candidateSequencesData.data() + i * encodedSequencePitchInInts;

                std::copy_n(subjectPtr, subjectNumInts, subjectDataBegin);
            }


        }

        void getCandidateSequencesData(
                ThreadData& threadData,
                const cpu::ContiguousReadStorage& readStorage,
                size_t encodedSequencePitchInInts){

            const int numCandidates = threadData.candidateReadIds.size();

            threadData.candidateSequencesData.resize(encodedSequencePitchInInts * numCandidates);
            threadData.candidateSequencesLengths.resize(numCandidates);
            threadData.tempReadIds.resize(numCandidates);
            threadData.permutationIndices.resize(numCandidates);

            std::iota(
                threadData.permutationIndices.begin(), 
                threadData.permutationIndices.end(), 
                0
            );
            std::sort(
                threadData.permutationIndices.begin(), 
                threadData.permutationIndices.end(),
                [&](const auto& l, const auto& r){
                    return threadData.candidateReadIds[l] < threadData.candidateReadIds[r];
                }
            );

            for(int i = 0; i < numCandidates; i++){
                const int index = threadData.permutationIndices[i];
                const read_number candidateId = threadData.candidateReadIds[index];
                const int candidateLength = readStorage.fetchSequenceLength(candidateId);
                threadData.candidateSequencesLengths[i] = candidateLength;
            }


            constexpr size_t prefetch_distance_sequences = 4;

            for(size_t i = 0; i < numCandidates && i < prefetch_distance_sequences; ++i) {
                const read_number next_candidate_read_id = threadData.candidateReadIds[threadData.permutationIndices[i]];
                const unsigned int* nextsequenceptr = (const unsigned int*)readStorage.fetchSequenceData_ptr(next_candidate_read_id);
                __builtin_prefetch(nextsequenceptr, 0, 0);
            }

            for(size_t i = 0; i < numCandidates; i++){
                if(i + prefetch_distance_sequences < numCandidates) {
                    const read_number next_candidate_read_id = threadData.candidateReadIds[threadData.permutationIndices[i+prefetch_distance_sequences]];
                    const unsigned int* nextsequenceptr = (const unsigned int*)readStorage.fetchSequenceData_ptr(next_candidate_read_id);
                    __builtin_prefetch(nextsequenceptr, 0, 0);
                }

                const int index = threadData.permutationIndices[i+prefetch_distance_sequences];
                const read_number candidateId = threadData.candidateReadIds[index];
                const unsigned int* candidateptr = (const unsigned int*)readStorage.fetchSequenceData_ptr(candidateId);
                const int candidateLength = threadData.candidateSequencesLengths[index];
                const int candidateNumInts = getEncodedNumInts2Bit(candidateLength);

                unsigned int* const candidateDataBegin = threadData.candidateSequencesData.data() + index * encodedSequencePitchInInts;
                unsigned int* const candidateRevcDataBegin = threadData.candidateSequencesRevcData.data() + index * encodedSequencePitchInInts;

                std::copy_n(candidateptr, candidateNumInts, candidateDataBegin);
                reverseComplement2Bit(candidateRevcDataBegin,
                                       candidateDataBegin,
                                        candidateLength);

                //data.candidateDataPtrs[i] = candidateDataBegin;
                //data.candidateRevcDataPtrs[i] = candidateRevcDataBegin;
            }


        }



        struct InterestingStruct{
            read_number readId;
            std::vector<int> positions;
        };

        std::vector<read_number> interestingReadIds;
        std::mutex interestingMutex;


        void getCandidates(TaskData& data,
                            CorrectionTask& task,
                            const Minhasher& minhasher,
                            int maxNumberOfCandidates,
                            int requiredHitsPerCandidate){

            minhasher.getCandidates_any_map(
                data.minhashHandle,
                task.original_subject_string,
                maxNumberOfCandidates
            );

            std::swap(task.candidate_read_ids, data.minhashHandle.result());

            //remove our own read id from candidate list. candidate_read_ids is sorted.
            auto readIdPos = std::lower_bound(task.candidate_read_ids.begin(),
                                                task.candidate_read_ids.end(),
                                                task.readId);

            if(readIdPos != task.candidate_read_ids.end() && *readIdPos == task.readId){
                task.candidate_read_ids.erase(readIdPos);
            }
        }


        void getCandidateSequenceData(TaskData& data,
                                    const cpu::ContiguousReadStorage& readStorage,
                                    CorrectionTask& task,
                                    size_t encodedSequencePitchInInts){

            const int numCandidates = task.candidate_read_ids.size();

            data.candidateLengths.clear();
            data.candidateLengths.resize(numCandidates);

            // for(const read_number candidateId : task.candidate_read_ids){
            //     const int candidateLength = readStorage.fetchSequenceLength(candidateId);
            //     data.candidateLengths.emplace_back(candidateLength);
            // }

            data.candidateData.clear();
            data.candidateData.resize(encodedSequencePitchInInts * numCandidates, 0);
            data.candidateRevcData.clear();
            data.candidateRevcData.resize(encodedSequencePitchInInts * numCandidates, 0);
            data.candidateDataPtrs.clear();
            data.candidateDataPtrs.resize(numCandidates, nullptr);
            data.candidateRevcDataPtrs.clear();
            data.candidateRevcDataPtrs.resize(numCandidates, nullptr);

            //copy candidate data and reverse complements into buffer

            // constexpr size_t prefetch_distance_sequences = 4;

            // for(size_t i = 0; i < numCandidates && i < prefetch_distance_sequences; ++i) {
            //     const read_number next_candidate_read_id = task.candidate_read_ids[i];
            //     const unsigned int* nextsequenceptr = (const unsigned int*)readStorage.fetchSequenceData_ptr(next_candidate_read_id);
            //     __builtin_prefetch(nextsequenceptr, 0, 0);
            // }

            // for(size_t i = 0; i < numCandidates; i++){
            //     if(i + prefetch_distance_sequences < numCandidates) {
            //         const read_number next_candidate_read_id = task.candidate_read_ids[i + prefetch_distance_sequences];
            //         const unsigned int* nextsequenceptr = (const unsigned int*)readStorage.fetchSequenceData_ptr(next_candidate_read_id);
            //         __builtin_prefetch(nextsequenceptr, 0, 0);
            //     }

            //     const read_number candidateId = task.candidate_read_ids[i];
            //     const unsigned int* candidateptr = (const unsigned int*)readStorage.fetchSequenceData_ptr(candidateId);
            //     const int candidateLength = data.candidateLengths[i];
            //     const int candidateNumInts = getEncodedNumInts2Bit(candidateLength);

            //     unsigned int* const candidateDataBegin = data.candidateData.data() + i * encodedSequencePitchInInts;
            //     unsigned int* const candidateRevcDataBegin = data.candidateRevcData.data() + i * encodedSequencePitchInInts;

            //     std::copy_n(candidateptr, candidateNumInts, candidateDataBegin);
            //     reverseComplement2Bit((unsigned int*)(candidateRevcDataBegin),
            //                               (const unsigned int*)(candidateptr),
            //                               candidateLength);

            //     data.candidateDataPtrs[i] = candidateDataBegin;
            //     data.candidateRevcDataPtrs[i] = candidateRevcDataBegin;
            // }

            readStorage.gatherSequenceLengths(
                data.readStorageGatherHandle,
                task.candidate_read_ids.data(),
                numCandidates,
                data.candidateLengths.data()
            );

            readStorage.gatherSequenceData(
                data.readStorageGatherHandle,
                task.candidate_read_ids.data(),
                numCandidates,
                data.candidateData.data(),
                encodedSequencePitchInInts
            );

            for(int i = 0; i < numCandidates; i++){
                unsigned int* const seqPtr = data.candidateData.data() + std::size_t(encodedSequencePitchInInts) * i;
                unsigned int* const seqrevcPtr = data.candidateRevcData.data() + std::size_t(encodedSequencePitchInInts) * i;

                reverseComplement2Bit(
                    seqrevcPtr,  
                    seqPtr,
                    data.candidateLengths[i]
                );

                data.candidateDataPtrs[i] = seqPtr;
                data.candidateRevcDataPtrs[i] = seqrevcPtr;
            }
        }


        void getCandidateAlignments(TaskData& data,
                                    CorrectionTask& task,
                                    size_t candidatePitchInInts,
                                    const GoodAlignmentProperties& alignmentProps,
                                    const CorrectionOptions& correctionOptions){

            const std::size_t numCandidates = task.candidate_read_ids.size();

            data.forwardAlignments.resize(numCandidates);
            data.revcAlignments.resize(numCandidates);

            shd::cpuShiftedHammingDistancePopcount2Bit(
                data.alignmentHandle,
                data.forwardAlignments.begin(),
                task.encodedSubjectPtr,
                task.original_subject_string.size(),
                data.candidateData.data(),
                candidatePitchInInts,
                data.candidateLengths.data(),
                data.candidateLengths.size(),
                alignmentProps.min_overlap,
                alignmentProps.maxErrorRate,
                alignmentProps.min_overlap_ratio
            );

            shd::cpuShiftedHammingDistancePopcount2Bit(
                data.alignmentHandle,
                data.revcAlignments.begin(),
                task.encodedSubjectPtr,
                task.original_subject_string.size(),
                data.candidateRevcData.data(),
                candidatePitchInInts,
                data.candidateLengths.data(),
                data.candidateLengths.size(),
                alignmentProps.min_overlap,
                alignmentProps.maxErrorRate,
                alignmentProps.min_overlap_ratio
            );



            // std::vector<SHDResult> newforwardalignments(numCandidates);
            // std::vector<SHDResult> newrevcalignments(numCandidates);

            // shd::cpu_multi_shifted_hamming_distance_popcount_new(newforwardalignments.begin(),
            //                                                 (const char*)task.encodedSubjectPtr,
            //                                                 task.original_subject_string.size(),
            //                                                 data.candidateData,
            //                                                 data.candidateLengths,
            //                                                 encodedSequencePitch,
            //                                                 alignmentProps.min_overlap,
            //                                                 alignmentProps.maxErrorRate,
            //                                                 alignmentProps.min_overlap_ratio);

            // shd::cpu_multi_shifted_hamming_distance_popcount_new(newrevcalignments.begin(),
            //                                                 (const char*)task.encodedSubjectPtr,
            //                                                 task.original_subject_string.size(),
            //                                                 data.candidateRevcData,
            //                                                 data.candidateLengths,
            //                                                 encodedSequencePitch,
            //                                                 alignmentProps.min_overlap,
            //                                                 alignmentProps.maxErrorRate,
            //                                                 alignmentProps.min_overlap_ratio);

            // for(int i = 0; i < numCandidates; i++){
            //     if(data.forwardAlignments[i] != newforwardalignments[i]){
            //         std::cerr << "error task " << task.readId << " forward candidate " << i << '\n';
            //         std::cerr << "old\n";
            //         std::cerr << data.forwardAlignments[i].score << "\n";
            //         std::cerr << data.forwardAlignments[i].overlap << "\n";
            //         std::cerr << data.forwardAlignments[i].shift << "\n";
            //         std::cerr << data.forwardAlignments[i].nOps << "\n";
            //         std::cerr << data.forwardAlignments[i].isValid << "\n";
            //         std::cerr << "new\n";
            //         std::cerr << newforwardalignments[i].score << "\n";
            //         std::cerr << newforwardalignments[i].overlap << "\n";
            //         std::cerr << newforwardalignments[i].shift << "\n";
            //         std::cerr << newforwardalignments[i].nOps << "\n";
            //         std::cerr << newforwardalignments[i].isValid << "\n";
            //         std::exit(0);
            //     }
            // }

            // for(int i = 0; i < numCandidates; i++){
            //     if(data.revcAlignments[i] != newrevcalignments[i]){
            //         std::cerr << "error task " << task.readId << " revc candidate " << i << '\n';
            //         std::cerr << "old\n";
            //         std::cerr << data.revcAlignments[i].score << "\n";
            //         std::cerr << data.revcAlignments[i].overlap << "\n";
            //         std::cerr << data.revcAlignments[i].shift << "\n";
            //         std::cerr << data.revcAlignments[i].nOps << "\n";
            //         std::cerr << data.revcAlignments[i].isValid << "\n";
            //         std::cerr << "new\n";
            //         std::cerr << newrevcalignments[i].score << "\n";
            //         std::cerr << newrevcalignments[i].overlap << "\n";
            //         std::cerr << newrevcalignments[i].shift << "\n";
            //         std::cerr << newrevcalignments[i].nOps << "\n";
            //         std::cerr << newrevcalignments[i].isValid << "\n";
            //         std::exit(0);
            //     }
            // }

            //decide whether to keep forward or reverse complement
            data.alignmentFlags = findBestAlignmentDirection(data.forwardAlignments,
                                                            data.revcAlignments,
                                                            task.original_subject_string.size(),
                                                            data.candidateLengths,
                                                            alignmentProps.min_overlap,
                                                            correctionOptions.estimatedErrorrate,
                                                            alignmentProps.min_overlap_ratio);

        }

        void gatherBestAlignmentData(TaskData& data,
                                  CorrectionTask& task,
                                  size_t candidatePitchInInts){

              data.bestAlignments.clear();
              data.bestAlignmentFlags.clear();
              data.bestCandidateReadIds.clear();
              data.bestCandidateData.clear();
              data.bestCandidateLengths.clear();
              data.bestCandidatePtrs.clear();

              data.bestAlignments.resize(data.numGoodAlignmentFlags);
              data.bestAlignmentFlags.resize(data.numGoodAlignmentFlags);
              data.bestCandidateReadIds.resize(data.numGoodAlignmentFlags);
              data.bestCandidateData.resize(data.numGoodAlignmentFlags * candidatePitchInInts);
              data.bestCandidateLengths.resize(data.numGoodAlignmentFlags);
              data.bestCandidatePtrs.resize(data.numGoodAlignmentFlags);

              for(size_t i = 0; i < data.numGoodAlignmentFlags; i++){
                  data.bestCandidatePtrs[i] = data.bestCandidateData.data() + i * candidatePitchInInts;
              }

              for(size_t i = 0, insertpos = 0; i < data.alignmentFlags.size(); i++){

                  const BestAlignment_t flag = data.alignmentFlags[i];
                  const auto& fwdAlignment = data.forwardAlignments[i];
                  const auto& revcAlignment = data.revcAlignments[i];
                  const read_number candidateId = task.candidate_read_ids[i];
                  const int candidateLength = data.candidateLengths[i];

                  if(flag == BestAlignment_t::Forward){
                      data.bestAlignmentFlags[insertpos] = flag;
                      data.bestCandidateReadIds[insertpos] = candidateId;
                      data.bestCandidateLengths[insertpos] = candidateLength;

                      data.bestAlignments[insertpos] = fwdAlignment;
                      std::copy_n(data.candidateDataPtrs[i],
                                candidatePitchInInts,
                                data.bestCandidatePtrs[insertpos]);
                      insertpos++;
                  }else if(flag == BestAlignment_t::ReverseComplement){
                      data.bestAlignmentFlags[insertpos] = flag;
                      data.bestCandidateReadIds[insertpos] = candidateId;
                      data.bestCandidateLengths[insertpos] = candidateLength;

                      data.bestAlignments[insertpos] = revcAlignment;
                      std::copy_n(data.candidateRevcDataPtrs[i],
                                candidatePitchInInts,
                                data.bestCandidatePtrs[insertpos]);
                      insertpos++;
                  }else{
                      ;//BestAlignment_t::None discard alignment
                  }
              }
        }

        void filterBestAlignmentsByMismatchRatio(TaskData& data,
                  CorrectionTask& task,
                  size_t candidatePitchInInts,
                  const CorrectionOptions& correctionOptions,
                  const GoodAlignmentProperties& alignmentProps){
            //get indices of alignments which have a good mismatch ratio

            auto goodIndices = filterAlignmentsByMismatchRatio(data.bestAlignments,
                                                               correctionOptions.estimatedErrorrate,
                                                               correctionOptions.estimatedCoverage,
                                                               correctionOptions.m_coverage,
                                                               [hpc = correctionOptions.hits_per_candidate](){
                                                                   return hpc > 1;
                                                               });

            if(goodIndices.size() == 0){
                task.active = false; //no good mismatch ratio
            }else{
                //stream compaction. keep only data at positions given by goodIndices

                for(size_t i = 0; i < goodIndices.size(); i++){
                    const int fromIndex = goodIndices[i];
                    const int toIndex = i;

                    data.bestAlignments[toIndex] = data.bestAlignments[fromIndex];
                    data.bestAlignmentFlags[toIndex] = data.bestAlignmentFlags[fromIndex];
                    data.bestCandidateReadIds[toIndex] = data.bestCandidateReadIds[fromIndex];
                    data.bestCandidateLengths[toIndex] = data.bestCandidateLengths[fromIndex];

                    std::copy_n(data.bestCandidatePtrs[fromIndex],
                              candidatePitchInInts,
                              data.bestCandidatePtrs[toIndex]);
                }

                data.bestAlignments.erase(data.bestAlignments.begin() + goodIndices.size(),
                                     data.bestAlignments.end());
                data.bestAlignmentFlags.erase(data.bestAlignmentFlags.begin() + goodIndices.size(),
                                         data.bestAlignmentFlags.end());
                data.bestCandidateReadIds.erase(data.bestCandidateReadIds.begin() + goodIndices.size(),
                                           data.bestCandidateReadIds.end());
                data.bestCandidateLengths.erase(data.bestCandidateLengths.begin() + goodIndices.size(),
                                           data.bestCandidateLengths.end());
                data.bestCandidatePtrs.erase(data.bestCandidatePtrs.begin() + goodIndices.size(),
                                        data.bestCandidatePtrs.end());
                data.bestCandidateData.erase(data.bestCandidateData.begin() + goodIndices.size() * candidatePitchInInts,
                                        data.bestCandidateData.end());

                data.bestAlignmentShifts.resize(data.bestAlignments.size());
                data.bestAlignmentWeights.resize(data.bestAlignments.size());


                auto calculateOverlapWeight = [](int anchorlength, int nOps, int overlapsize){
                    constexpr float maxErrorPercentInOverlap = 0.2f;

                    return 1.0f - sqrtf(nOps / (overlapsize * maxErrorPercentInOverlap));
                };

                for(size_t i = 0; i < data.bestAlignments.size(); i++){
                    data.bestAlignmentShifts[i] = data.bestAlignments[i].shift;
                    // data.bestAlignmentWeights[i] = 1.0f - std::sqrt(data.bestAlignments[i].nOps
                    //                                                     / (data.bestAlignments[i].overlap
                    //                                                         * alignmentProps.maxErrorRate));

                    data.bestAlignmentWeights[i] = calculateOverlapWeight(task.original_subject_string.length(), 
                                                                        data.bestAlignments[i].nOps, 
                                                                        data.bestAlignments[i].overlap);
                }
            }
        }



        void getCandidateQualities(TaskData& data,
                                    const cpu::ContiguousReadStorage& readStorage,
                                    CorrectionTask& task,
                                    int maximumSequenceLength){

            task.subjectQualityPtr = readStorage.fetchQuality_ptr(task.readId);

            data.bestCandidateQualityData.clear();
            data.bestCandidateQualityData.resize(maximumSequenceLength * data.bestAlignments.size());

            data.bestCandidateQualityPtrs.clear();
            data.bestCandidateQualityPtrs.resize(data.bestAlignments.size());

            constexpr size_t prefetch_distance_qualities = 4;

            for(size_t i = 0; i < data.bestAlignments.size() && i < prefetch_distance_qualities; ++i) {
                const read_number next_candidate_read_id = data.bestCandidateReadIds[i];
                const char* nextqualityptr = readStorage.fetchQuality_ptr(next_candidate_read_id);
                __builtin_prefetch(nextqualityptr, 0, 0);
            }

            for(size_t i = 0; i < data.bestAlignments.size(); i++){
                if(i + prefetch_distance_qualities < data.bestAlignments.size()) {
                    const read_number next_candidate_read_id = data.bestCandidateReadIds[i + prefetch_distance_qualities];
                    const char* nextqualityptr = readStorage.fetchQuality_ptr(next_candidate_read_id);
                    __builtin_prefetch(nextqualityptr, 0, 0);
                }
                const char* qualityptr = readStorage.fetchQuality_ptr(data.bestCandidateReadIds[i]);
                const int length = data.bestCandidateLengths[i];
                const BestAlignment_t flag = data.bestAlignmentFlags[i];

                if(flag == BestAlignment_t::Forward){
                    std::copy(qualityptr, qualityptr + length, &data.bestCandidateQualityData[maximumSequenceLength * i]);
                }else{
                    std::reverse_copy(qualityptr, qualityptr + length, &data.bestCandidateQualityData[maximumSequenceLength * i]);
                }
            }

            for(size_t i = 0; i < data.bestAlignments.size(); i++){
                data.bestCandidateQualityPtrs[i] = data.bestCandidateQualityData.data() + i * maximumSequenceLength;
            }
        }


        void makeCandidateStrings(TaskData& data,
                                    CorrectionTask& task,
                                    int maximumSequenceLength){
            data.bestCandidateStrings.clear();
            data.bestCandidateStrings.resize(data.bestAlignments.size() * maximumSequenceLength);

            for(size_t i = 0; i < data.bestAlignments.size(); i++){
                const unsigned int* const ptr = data.bestCandidatePtrs[i];
                const int length = data.bestCandidateLengths[i];
                decode2BitSequence(&data.bestCandidateStrings[i * maximumSequenceLength],
                                        ptr,
                                        length);
            }
        }

        void getIndicesOfCandidatesEqualToSubject(TaskData& data,
                                                    const CorrectionTask& task,
                                                    int maximumSequenceLength){

            data.indicesOfCandidatesEqualToSubject.clear();
            data.indicesOfCandidatesEqualToSubject.reserve(data.bestAlignmentShifts.size());

            for(std::size_t i = 0; i < data.bestAlignmentShifts.size(); i++){
                if(data.bestAlignmentShifts[i] == 0){
                    std::size_t length = task.original_subject_string.length();
                    int cmpresult = std::memcmp(task.original_subject_string.c_str(),
                                            &data.bestCandidateStrings[i * maximumSequenceLength],
                                            length);
                    if(cmpresult == 0){
                        data.indicesOfCandidatesEqualToSubject.emplace_back(i);
                    }
                }
            }
        }

        void buildMultipleSequenceAlignment(TaskData& data,
                                            CorrectionTask& task,
                                            const CorrectionOptions& correctionOptions,
                                            int maximumSequenceLength){


            const char* const candidateQualityPtr = correctionOptions.useQualityScores ?
                                                    data.bestCandidateQualityData.data()
                                                    : nullptr;

            data.multipleSequenceAlignment.build(task.original_subject_string.c_str(),
                                            task.original_subject_string.length(),
                                            data.bestCandidateStrings.data(),
                                            data.bestCandidateLengths.data(),
                                            int(data.bestAlignments.size()),
                                            data.bestAlignmentShifts.data(),
                                            data.bestAlignmentWeights.data(),
                                            task.subjectQualityPtr,
                                            candidateQualityPtr,
                                            maximumSequenceLength,
                                            maximumSequenceLength,
                                            correctionOptions.useQualityScores);
        }

        void removeCandidatesOfDifferentRegionFromMSA(TaskData& data,
                                                        CorrectionTask& task,
                                                        const CorrectionOptions& correctionOptions,
                                                        size_t candidatePitchInInts,
                                                        int maximumSequenceLength){
            constexpr int max_num_minimizations = 5;

            if(max_num_minimizations > 0){
                int num_minimizations = 0;

                std::vector<int> nOps(data.bestAlignments.size());
                std::vector<int> overlaps(data.bestAlignments.size());
                for(int i = 0; i < int(data.bestAlignments.size()); i++){
                    nOps[i] = data.bestAlignments[i].nOps;
                    overlaps[i] = data.bestAlignments[i].overlap;
                }

                auto minimizationResult = findCandidatesOfDifferentRegion(task.original_subject_string.c_str(),
                                                                    task.original_subject_string.length(),
                                                                    data.bestCandidateStrings.data(),
                                                                    data.bestCandidateLengths.data(),
                                                                    int(data.bestAlignments.size()),
                                                                    maximumSequenceLength,
                                                                    data.multipleSequenceAlignment.consensus.data(),
                                                                    data.multipleSequenceAlignment.countsA.data(),
                                                                    data.multipleSequenceAlignment.countsC.data(),
                                                                    data.multipleSequenceAlignment.countsG.data(),
                                                                    data.multipleSequenceAlignment.countsT.data(),
                                                                    data.multipleSequenceAlignment.weightsA.data(),
                                                                    data.multipleSequenceAlignment.weightsC.data(),
                                                                    data.multipleSequenceAlignment.weightsG.data(),
                                                                    data.multipleSequenceAlignment.weightsT.data(),
                                                                    nOps.data(), 
                                                                    overlaps.data(),
                                                                    data.multipleSequenceAlignment.subjectColumnsBegin_incl,
                                                                    data.multipleSequenceAlignment.subjectColumnsEnd_excl,
                                                                    data.bestAlignmentShifts.data(),
                                                                    correctionOptions.estimatedCoverage);

                auto update_after_successfull_minimization = [&](){

                    if(minimizationResult.performedMinimization){
                        assert(minimizationResult.differentRegionCandidate.size() == data.bestAlignments.size());

                        bool anyRemoved = false;
                        size_t cur = 0;
                        for(size_t i = 0; i < minimizationResult.differentRegionCandidate.size(); i++){
                            if(!minimizationResult.differentRegionCandidate[i]){

                                data.bestAlignments[cur] = data.bestAlignments[i];
                                data.bestAlignmentShifts[cur] = data.bestAlignmentShifts[i];
                                data.bestAlignmentWeights[cur] = data.bestAlignmentWeights[i];
                                data.bestAlignmentFlags[cur] = data.bestAlignmentFlags[i];
                                data.bestCandidateReadIds[cur] = data.bestCandidateReadIds[i];
                                data.bestCandidateLengths[cur] = data.bestCandidateLengths[i];

                                std::copy_n(data.bestCandidateData.begin() + i * candidatePitchInInts,
                                        candidatePitchInInts,
                                        data.bestCandidateData.begin() + cur * candidatePitchInInts);
                                std::copy(data.bestCandidateQualityData.begin() + i * maximumSequenceLength,
                                        data.bestCandidateQualityData.begin() + (i+1) * maximumSequenceLength,
                                        data.bestCandidateQualityData.begin() + cur * maximumSequenceLength);
                                std::copy(data.bestCandidateStrings.begin() + i * maximumSequenceLength,
                                        data.bestCandidateStrings.begin() + (i+1) * maximumSequenceLength,
                                        data.bestCandidateStrings.begin() + cur * maximumSequenceLength);

                                cur++;

                            }else{
                                anyRemoved = true;
                            }
                        }

                        //assert(anyRemoved);

                        data.bestAlignments.erase(data.bestAlignments.begin() + cur, data.bestAlignments.end());
                        data.bestAlignmentShifts.erase(data.bestAlignmentShifts.begin() + cur, data.bestAlignmentShifts.end());
                        data.bestAlignmentWeights.erase(data.bestAlignmentWeights.begin() + cur, data.bestAlignmentWeights.end());
                        data.bestAlignmentFlags.erase(data.bestAlignmentFlags.begin() + cur, data.bestAlignmentFlags.end());
                        data.bestCandidateReadIds.erase(data.bestCandidateReadIds.begin() + cur, data.bestCandidateReadIds.end());
                        data.bestCandidateLengths.erase(data.bestCandidateLengths.begin() + cur, data.bestCandidateLengths.end());

                        data.bestCandidateData.erase(data.bestCandidateData.begin() + cur * candidatePitchInInts, data.bestCandidateData.end());
                        data.bestCandidateQualityData.erase(data.bestCandidateQualityData.begin() + cur * maximumSequenceLength, data.bestCandidateQualityData.end());
                        data.bestCandidateStrings.erase(data.bestCandidateStrings.begin() + cur * maximumSequenceLength, data.bestCandidateStrings.end());

                        //build minimized multiple sequence alignment

                        buildMultipleSequenceAlignment(data,
                                                        task,
                                                        correctionOptions,
                                                        maximumSequenceLength);
                    }
                };

                update_after_successfull_minimization();


                num_minimizations++;

                while(num_minimizations < max_num_minimizations
                        && minimizationResult.performedMinimization){

                    nOps.resize(data.bestAlignments.size());
                    overlaps.resize(data.bestAlignments.size());
                    for(int i = 0; i < int(data.bestAlignments.size()); i++){
                        nOps[i] = data.bestAlignments[i].nOps;
                        overlaps[i] = data.bestAlignments[i].overlap;
                    }

                    minimizationResult = findCandidatesOfDifferentRegion(task.original_subject_string.c_str(),
                                                                        task.original_subject_string.length(),
                                                                        data.bestCandidateStrings.data(),
                                                                        data.bestCandidateLengths.data(),
                                                                        int(data.bestAlignments.size()),
                                                                        maximumSequenceLength,
                                                                        data.multipleSequenceAlignment.consensus.data(),
                                                                        data.multipleSequenceAlignment.countsA.data(),
                                                                        data.multipleSequenceAlignment.countsC.data(),
                                                                        data.multipleSequenceAlignment.countsG.data(),
                                                                        data.multipleSequenceAlignment.countsT.data(),
                                                                        data.multipleSequenceAlignment.weightsA.data(),
                                                                        data.multipleSequenceAlignment.weightsC.data(),
                                                                        data.multipleSequenceAlignment.weightsG.data(),
                                                                        data.multipleSequenceAlignment.weightsT.data(),
                                                                        nOps.data(), 
                                                                        overlaps.data(),
                                                                        data.multipleSequenceAlignment.subjectColumnsBegin_incl,
                                                                        data.multipleSequenceAlignment.subjectColumnsEnd_excl,
                                                                        data.bestAlignmentShifts.data(),
                                                                        correctionOptions.estimatedCoverage);

                    update_after_successfull_minimization();

                    num_minimizations++;
                }
            }
        }




        void correctSubject(TaskData& data,
                            CorrectionTask& task,
                            const CorrectionOptions& correctionOptions,
                            int maximumSequenceLength){

            const int subjectColumnsBegin_incl = data.multipleSequenceAlignment.subjectColumnsBegin_incl;
            const int subjectColumnsEnd_excl = data.multipleSequenceAlignment.subjectColumnsEnd_excl;

            data.msaProperties = getMSAProperties2(data.multipleSequenceAlignment.support.data(),
                                                            data.multipleSequenceAlignment.coverage.data(),
                                                            subjectColumnsBegin_incl,
                                                            subjectColumnsEnd_excl,
                                                            correctionOptions.estimatedErrorrate,
                                                            correctionOptions.estimatedCoverage,
                                                            correctionOptions.m_coverage);

            // auto correctionResult = getCorrectedSubject(data.multipleSequenceAlignment.consensus.data() + subjectColumnsBegin_incl,
            //                                             data.multipleSequenceAlignment.support.data() + subjectColumnsBegin_incl,
            //                                             data.multipleSequenceAlignment.coverage.data() + subjectColumnsBegin_incl,
            //                                             data.multipleSequenceAlignment.origCoverages.data() + subjectColumnsBegin_incl,
            //                                             int(task.original_subject_string.size()),
            //                                             task.original_subject_string.c_str(),
            //                                             subjectColumnsBegin_incl,
            //                                             data.bestCandidateStrings.data(),
            //                                             int(data.bestAlignmentWeights.size()),
            //                                             data.bestAlignmentWeights.data(),
            //                                             data.bestCandidateLengths.data(),
            //                                             data.bestAlignmentShifts.data(),
            //                                             maximumSequenceLength,
            //                                             data.msaProperties.isHQ,
            //                                             correctionOptions.estimatedErrorrate,
            //                                             correctionOptions.estimatedCoverage,
            //                                             correctionOptions.m_coverage,
            //                                             correctionOptions.kmerlength);

            auto correctionResult = getCorrectedSubjectNew(data.multipleSequenceAlignment.consensus.data() + subjectColumnsBegin_incl,
                                                        data.multipleSequenceAlignment.support.data() + subjectColumnsBegin_incl,
                                                        data.multipleSequenceAlignment.coverage.data() + subjectColumnsBegin_incl,
                                                        data.multipleSequenceAlignment.origCoverages.data() + subjectColumnsBegin_incl,
                                                        int(task.original_subject_string.size()),
                                                        task.original_subject_string.c_str(),
                                                        subjectColumnsBegin_incl,
                                                        data.bestCandidateStrings.data(),
                                                        int(data.bestAlignmentWeights.size()),
                                                        data.bestAlignmentWeights.data(),
                                                        data.bestCandidateLengths.data(),
                                                        data.bestAlignmentShifts.data(),
                                                        maximumSequenceLength,
                                                        data.msaProperties,
                                                        correctionOptions.estimatedErrorrate,
                                                        correctionOptions.estimatedCoverage,
                                                        correctionOptions.m_coverage,
                                                        correctionOptions.kmerlength);

            auto it = std::lower_bound(interestingReadIds.begin(), interestingReadIds.end(), task.readId);
            if(it != interestingReadIds.end() && *it == task.readId){
                std::lock_guard<std::mutex> lg(interestingMutex);

                std::cerr << "read id " << task.readId << " HQ: " << data.msaProperties.isHQ << "\n";
                if(!data.msaProperties.isHQ){
                    for(int i = 0; i < int(correctionResult.bestAlignmentWeightOfConsensusBase.size()); i++){
                        if(correctionResult.bestAlignmentWeightOfConsensusBase[i] != 0 || correctionResult.bestAlignmentWeightOfAnchorBase[i]){
                            std::cerr << "position " << i
                                        << " " << correctionResult.bestAlignmentWeightOfConsensusBase[i]
                                        << " " << correctionResult.bestAlignmentWeightOfAnchorBase[i] << "\n";
                        }
                    }
                }
            }

            data.msaProperties.isHQ = correctionResult.isHQ;

            if(correctionResult.isCorrected){
                task.corrected_subject = std::move(correctionResult.correctedSequence);
                task.uncorrectedPositionsNoConsensus = std::move(correctionResult.uncorrectedPositionsNoConsensus);
                task.corrected = true;                
            }
        }

        void correctCandidates(TaskData& data,
                                CorrectionTask& task,
                                const CorrectionOptions& correctionOptions){

            task.correctedCandidates = getCorrectedCandidatesNew(data.multipleSequenceAlignment.consensus.data(),
                                                            data.multipleSequenceAlignment.support.data(),
                                                            data.multipleSequenceAlignment.coverage.data(),
                                                            data.multipleSequenceAlignment.nColumns,
                                                            data.multipleSequenceAlignment.subjectColumnsBegin_incl,
                                                            data.multipleSequenceAlignment.subjectColumnsEnd_excl,
                                                            data.bestAlignmentShifts.data(),
                                                            data.bestCandidateLengths.data(),
                                                            data.multipleSequenceAlignment.nCandidates,
                                                            correctionOptions.estimatedErrorrate,
                                                            correctionOptions.estimatedCoverage,
                                                            correctionOptions.m_coverage,
                                                            correctionOptions.new_columns_to_correct);
        }

        void correctSubjectWithForest(TaskData& data,
                                CorrectionTask& task,
                                const ForestClassifier& forestClassifier,
                                const CorrectionOptions& correctionOptions){

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

#if 0
        void correctSubjectWithNeuralNetwork(TaskData& data,
                                            CorrectionTask& task,
                                            const NN_Correction_Classifier& nnClassifier,
                                            const CorrectionOptions& correctionOptions){
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

        void setCorrectionStatusFlags(const TaskData& taskdata, 
                                        CorrectionTask& task,
                                        std::uint8_t* correctionStatusFlagsPerRead){
            if(task.active){
                if(task.corrected){
                    if(taskdata.msaProperties.isHQ){
                        correctionStatusFlagsPerRead[task.readId] |= readCorrectedAsHQAnchor;
                    }
                }else{
                    correctionStatusFlagsPerRead[task.readId] |= readCouldNotBeCorrectedAsAnchor;
                }
            }
        }

        void createTemporaryResultsForOutput(const TaskData& taskdata, 
                                             CorrectionTask& task,
                                             const ContiguousReadStorage& readStorage,
                                             const std::uint8_t* correctionStatusFlagsPerRead,
                                             const SequenceFileProperties& sequenceFileProperties){
            if(task.active){
                if(task.corrected){
                    const int correctedlength = task.corrected_subject.length();

                    const bool originalReadContainsN = readStorage.readContainsN(task.readId);

                    auto& tmp = task.anchoroutput;
                    tmp = TempCorrectedSequence{};

                    if(!originalReadContainsN){
                        const int maxEdits = correctedlength / 7;
                        int edits = 0;
                        for(int i = 0; i < correctedlength && edits <= maxEdits; i++){
                            if(task.corrected_subject[i] != task.original_subject_string[i]){
                                tmp.edits.emplace_back(i, task.corrected_subject[i]);
                                edits++;
                            }
                        }
                        tmp.useEdits = edits <= maxEdits;
                    }else{
                        tmp.useEdits = false;
                    }

                    tmp.hq = taskdata.msaProperties.isHQ;
                    tmp.type = TempCorrectedSequence::Type::Anchor;
                    tmp.readId = task.readId;
                    tmp.sequence = std::move(task.corrected_subject);
                    tmp.uncorrectedPositionsNoConsensus = task.uncorrectedPositionsNoConsensus;
                }

                task.candidatesoutput.reserve(task.correctedCandidates.size());

                for(const auto& correctedCandidate : task.correctedCandidates){
                    const read_number candidateId = taskdata.bestCandidateReadIds[correctedCandidate.index];

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
                        if(taskdata.bestAlignmentFlags[correctedCandidate.index] == BestAlignment_t::Forward){
                            //write_candidate(candidateId, correctedCandidate.sequence);
                            tmp.sequence = std::move(correctedCandidate.sequence);
                        }else{
                                std::string fwd;
                                fwd.resize(correctedCandidate.sequence.length());
                                reverseComplementString(&fwd[0], correctedCandidate.sequence.c_str(), correctedCandidate.sequence.length());
                                //write_candidate(candidateId, fwd);
                                tmp.sequence = std::move(fwd);
                        }

                        const bool originalCandidateReadContainsN = readStorage.readContainsN(candidateId);

                        if(!originalCandidateReadContainsN){
                            const std::size_t offset = correctedCandidate.index * sequenceFileProperties.maxSequenceLength;
                            const char* const uncorrectedCandidate = &taskdata.bestCandidateStrings[offset];
                            const int uncorrectedCandidateLength = taskdata.bestCandidateLengths[correctedCandidate.index];
                            const int correctedCandidateLength = tmp.sequence.length();

                            assert(uncorrectedCandidateLength == correctedCandidateLength);


                            const int maxEdits = correctedCandidateLength / 7;
                            int edits = 0;
                            for(int pos = 0; pos < correctedCandidateLength && edits <= maxEdits; pos++){
                                if(tmp.sequence[pos] != uncorrectedCandidate[pos]){
                                    tmp.edits.emplace_back(pos, tmp.sequence[pos]);
                                    edits++;
                                }
                            }

                            tmp.useEdits = edits <= maxEdits;
                        }else{
                            tmp.useEdits = false;
                        }

                        task.candidatesoutput.emplace_back(std::move(tmp));
                    }
                }
            }
        }


void correct_cpu(const MinhashOptions& minhashOptions,
                  const AlignmentOptions& alignmentOptions,
                  const GoodAlignmentProperties& goodAlignmentProperties,
                  const CorrectionOptions& correctionOptions,
                  const RuntimeOptions& runtimeOptions,
                  const FileOptions& fileOptions,
                  const SequenceFileProperties& sequenceFileProperties,
                  Minhasher& minhasher,
                  cpu::ContiguousReadStorage& readStorage,
                  std::uint64_t maxCandidatesPerRead){

    int oldNumOMPThreads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        oldNumOMPThreads = omp_get_num_threads();
    }

    omp_set_num_threads(runtimeOptions.nCorrectorThreads);

    std::vector<std::string> tmpfiles{fileOptions.tempdirectory + "/" + fileOptions.outputfilename + "_tmp"};
    std::vector<std::string> featureTmpFiles{fileOptions.tempdirectory + "/" + fileOptions.outputfilename + "_features"};

    // std::ofstream outputstream;

    // outputstream = std::move(std::ofstream(tmpfiles[0]));
    // if(!outputstream){
    //     throw std::runtime_error("Could not open output file " + tmpfiles[0]);
    // }

    const std::size_t availableMemory = getAvailableMemoryInKB();
    const std::size_t memoryForPartialResults = availableMemory - (std::size_t(1) << 30);

    auto heapusageOfTCS = [](const auto& x){
        return x.data.capacity();
    };

    MemoryFile<EncodedTempCorrectedSequence> partialResults(memoryForPartialResults, tmpfiles[0], heapusageOfTCS);

    std::ofstream featurestream;
      //if(correctionOptions.extractFeatures){
          featurestream = std::move(std::ofstream(featureTmpFiles[0]));
          if(!featurestream && correctionOptions.extractFeatures){
              throw std::runtime_error("Could not open output feature file");
          }
      //}

#ifndef DO_PROFILE
    cpu::RangeGenerator<read_number> readIdGenerator(sequenceFileProperties.nReads);
#else
    cpu::RangeGenerator<read_number> readIdGenerator(num_reads_to_profile);
#endif

#if 0
    NN_Correction_Classifier_Base nnClassifierBase;
    NN_Correction_Classifier nnClassifier;
    if(correctionOptions.correctionType == CorrectionType::Convnet){
        nnClassifierBase = std::move(NN_Correction_Classifier_Base{"./nn_sources", fileOptions.nnmodelfilename});
        nnClassifier = std::move(NN_Correction_Classifier{&nnClassifierBase});
    }
#endif 

    ForestClassifier forestClassifier;
    if(correctionOptions.correctionType == CorrectionType::Forest){
        forestClassifier = std::move(ForestClassifier{fileOptions.forestfilename});
    }

    auto saveCorrectedSequence = [&](const TempCorrectedSequence& tmp, const EncodedTempCorrectedSequence& encoded){
          //std::unique_lock<std::mutex> l(outputstreammutex);
          if(!(tmp.hq && tmp.useEdits && tmp.edits.empty())){
              //outputstream << tmp << '\n';
              partialResults.storeElement(std::move(encoded));
          }
      };

    std::vector<std::uint8_t> correctionStatusFlagsPerRead;
    std::size_t nLocksForProcessedFlags = runtimeOptions.nCorrectorThreads * 1000;
    std::unique_ptr<std::mutex[]> locksForProcessedFlags(new std::mutex[nLocksForProcessedFlags]);

    correctionStatusFlagsPerRead.resize(sequenceFileProperties.nReads, 0);

    std::cerr << "correctionStatusFlagsPerRead bytes: " << correctionStatusFlagsPerRead.size() / 1024. / 1024. << " MB\n";

    auto lock = [&](read_number readId){
        read_number index = readId % nLocksForProcessedFlags;
        locksForProcessedFlags[index].lock();
    };

    auto unlock = [&](read_number readId){
        read_number index = readId % nLocksForProcessedFlags;
        locksForProcessedFlags[index].unlock();
    };

    std::chrono::time_point<std::chrono::system_clock> timepoint_begin = std::chrono::system_clock::now();
    std::chrono::duration<double> runtime = std::chrono::seconds(0);
    std::chrono::duration<double> previousProgressTime = std::chrono::seconds(0);
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
    std::chrono::duration<double> correctWithFeaturesTimeTotal{0};

    BackgroundThread outputThread(true);


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

    //const int encodedSequencePitch = sizeof(unsigned int) * getEncodedNumInts2Bit(sequenceFileProperties.maxSequenceLength);
    const int encodedSequencePitchInInts = getEncodedNumInts2Bit(sequenceFileProperties.maxSequenceLength);

    //std::chrono::time_point<std::chrono::system_clock> tpa, tpb, tpc, tpd;

    const int numThreads = runtimeOptions.nCorrectorThreads;


    //std::vector<TaskData> dataPerTask(correctionOptions.batchsize);
    //std::vector<CorrectionTask> correctionTasks;

    std::vector<std::vector<CorrectionTask>> correctionTasksPerThread(numThreads);
    std::vector<std::vector<TaskData>> dataPerTaskPerThread(numThreads);

    //std::cerr << "correctionOptions.hits_per_candidate " <<  correctionOptions.hits_per_candidate << ", max_candidates " << max_candidates << '\n';

    #pragma omp parallel
    {
        const int threadId = omp_get_thread_num();
        auto& correctionTasks = correctionTasksPerThread[threadId];
        auto& dataPerTask = dataPerTaskPerThread[threadId];

        while(!(readIdGenerator.empty())){

            std::vector<read_number> readIds = readIdGenerator.next_n(correctionOptions.batchsize);
            if(readIds.empty()){
                continue;
            }
            
            correctionTasks.resize(readIds.size());
            dataPerTask.resize(readIds.size());
            
            for(int i = 0; i < int(readIds.size());i++){

                const read_number readId = readIds[i];

                auto& task = correctionTasks[i];
                task = CorrectionTask(readId);

                auto& taskdata = dataPerTask[i];
                task.taskDataPtr = &taskdata;

                //bool ok = false;
                // lock(readId);
                // if (readIsCorrectedVector[readId] == 0) {
                //     readIsCorrectedVector[readId] = 1;
                //     ok = true;
                // }else{
                // }
                // unlock(readId);
                bool ok = true;
                if(ok){
                    const unsigned int* originalsubjectptr = readStorage.fetchSequenceData_ptr(readId);
                    const int originalsubjectLength = readStorage.fetchSequenceLength(readId);

                    task.original_subject_string = get2BitString(originalsubjectptr, originalsubjectLength);

                    task.encodedSubjectPtr = originalsubjectptr;
                }else{
                    task.active = false;
                }

                if(!task.active){
                    continue; //discard
                }

                #ifdef ENABLE_TIMING
                auto tpa = std::chrono::system_clock::now();
                #endif

                getCandidates(
                    taskdata,
                    task,
                    minhasher,
                    maxCandidatesPerRead,
                    correctionOptions.hits_per_candidate
                );

                #ifdef ENABLE_TIMING
                getCandidatesTimeTotal += std::chrono::system_clock::now() - tpa;
                #endif

                if(task.candidate_read_ids.empty()){
                    task.active = false;
                    continue; //discard
                }

                #ifdef ENABLE_TIMING
                tpa = std::chrono::system_clock::now();
                #endif

                getCandidateSequenceData(taskdata, readStorage, task, encodedSequencePitchInInts);

                #ifdef ENABLE_TIMING
                copyCandidateDataToBufferTimeTotal += std::chrono::system_clock::now() - tpa;
                #endif

                #ifdef ENABLE_TIMING
                tpa = std::chrono::system_clock::now();
                #endif

                getCandidateAlignments(taskdata, task, encodedSequencePitchInInts, goodAlignmentProperties, correctionOptions);

                taskdata.numGoodAlignmentFlags = std::count_if(taskdata.alignmentFlags.begin(),
                                                            taskdata.alignmentFlags.end(),
                                                            [](const auto flag){
                                                                return flag != BestAlignment_t::None;
                                                            });

                // std::cerr << "taskdata.numGoodAlignmentFlags : " << taskdata.numGoodAlignmentFlags << "\n";
                // for(const auto& f : taskdata.alignmentFlags){
                //     std::cerr << int(f) << " ";
                // }
                // std::cerr << "\n";

                #ifdef ENABLE_TIMING
                getAlignmentsTimeTotal += std::chrono::system_clock::now() - tpa;
                #endif

                if(taskdata.numGoodAlignmentFlags == 0){
                    task.active = false; //no good alignments
                    continue;
                }

                #ifdef ENABLE_TIMING
                tpa = std::chrono::system_clock::now();
                #endif

                gatherBestAlignmentData(taskdata, task, encodedSequencePitchInInts);

                #ifdef ENABLE_TIMING
                gatherBestAlignmentDataTimeTotal += std::chrono::system_clock::now() - tpa;
                #endif

                #ifdef ENABLE_TIMING
                tpa = std::chrono::system_clock::now();
                #endif

                filterBestAlignmentsByMismatchRatio(taskdata,
                                                task,
                                                encodedSequencePitchInInts,
                                                correctionOptions,
                                                goodAlignmentProperties);

                #ifdef ENABLE_TIMING
                mismatchRatioFilteringTimeTotal += std::chrono::system_clock::now() - tpa;
                #endif

                if(!task.active){
                    continue;
                }

                #ifdef ENABLE_TIMING
                tpa = std::chrono::system_clock::now();
                #endif

                if(correctionOptions.useQualityScores){
                    //gather quality scores of best alignments
                    getCandidateQualities(taskdata,
                                        readStorage,
                                        task,
                                        sequenceFileProperties.maxSequenceLength);
                }

                #ifdef ENABLE_TIMING
                fetchQualitiesTimeTotal += std::chrono::system_clock::now() - tpa;
                #endif

                #ifdef ENABLE_TIMING
                tpa = std::chrono::system_clock::now();
                #endif

                makeCandidateStrings(taskdata,
                                        task,
                                        sequenceFileProperties.maxSequenceLength);

                #ifdef ENABLE_TIMING
                makeCandidateStringsTimeTotal += std::chrono::system_clock::now() - tpa;
                #endif

                #ifdef ENABLE_TIMING
                tpa = std::chrono::system_clock::now();
                #endif

                buildMultipleSequenceAlignment(taskdata,
                                            task,
                                            correctionOptions,
                                            sequenceFileProperties.maxSequenceLength);

                #ifdef ENABLE_TIMING
                msaFindConsensusTimeTotal += std::chrono::system_clock::now() - tpa;
                #endif

    #ifdef USE_MSA_MINIMIZATION

    #ifdef PRINT_MSA
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

                #ifdef ENABLE_TIMING
                tpa = std::chrono::system_clock::now();
                #endif

                removeCandidatesOfDifferentRegionFromMSA(taskdata,
                                                        task,
                                                        correctionOptions,
                                                        encodedSequencePitchInInts,
                                                        sequenceFileProperties.maxSequenceLength);

                #ifdef ENABLE_TIMING
                msaMinimizationTimeTotal += std::chrono::system_clock::now() - tpa;
                #endif

    #endif // USE_MSA_MINIMIZATION

                if(correctionOptions.extractFeatures){

                    #if 1
                    taskdata.msaforestfeatures = extractFeatures(taskdata.multipleSequenceAlignment.consensus.data(),
                                                                taskdata.multipleSequenceAlignment.support.data(),
                                                                taskdata.multipleSequenceAlignment.coverage.data(),
                                                                taskdata.multipleSequenceAlignment.origCoverages.data(),
                                                                taskdata.multipleSequenceAlignment.nColumns,
                                                                taskdata.multipleSequenceAlignment.subjectColumnsBegin_incl,
                                                                taskdata.multipleSequenceAlignment.subjectColumnsEnd_excl,
                                                                task.original_subject_string,
                                                                correctionOptions.kmerlength, 0.5f,
                                                                correctionOptions.estimatedCoverage);
                    #else

                    auto MSAFeatures = extractFeatures3_2(
                                                multipleSequenceAlignment.countsA.data(),
                                                multipleSequenceAlignment.countsC.data(),
                                                multipleSequenceAlignment.countsG.data(),
                                                multipleSequenceAlignment.countsT.data(),
                                                multipleSequenceAlignment.weightsA.data(),
                                                multipleSequenceAlignment.weightsC.data(),
                                                multipleSequenceAlignment.weightsG.data(),
                                                multipleSequenceAlignment.weightsT.data(),
                                                multipleSequenceAlignment.nCandidates+1,
                                                multipleSequenceAlignment.nColumns,
                                                multipleSequenceAlignment.consensus.data(),
                                                multipleSequenceAlignment.support.data(),
                                                multipleSequenceAlignment.coverage.data(),
                                                multipleSequenceAlignment.origCoverages.data(),
                                                multipleSequenceAlignment.subjectColumnsBegin_incl,
                                                multipleSequenceAlignment.subjectColumnsEnd_excl,
                                                task.original_subject_string,
                                                correctionOptions.estimatedCoverage);
                    #endif

                }else{ //correction is not performed when extracting features
                    if(correctionOptions.correctionType == CorrectionType::Classic){

                        #ifdef ENABLE_TIMING
                        auto tpa = std::chrono::system_clock::now();
                        #endif

                        correctSubject(taskdata,
                                        task,
                                        correctionOptions,
                                        sequenceFileProperties.maxSequenceLength);

                        #ifdef ENABLE_TIMING
                        msaCorrectSubjectTimeTotal += std::chrono::system_clock::now() - tpa;
                        #endif

                        //get corrected candidates and write them to file
                        if(correctionOptions.correctCandidates){
                            #ifdef ENABLE_TIMING
                            tpa = std::chrono::system_clock::now();
                            #endif

                            //getIndicesOfCandidatesEqualToSubject(taskdata, task, sequenceFileProperties.maxSequenceLength);

                            if(taskdata.msaProperties.isHQ){
                                correctCandidates(taskdata, task, correctionOptions);
                            }

                            #ifdef ENABLE_TIMING
                            msaCorrectCandidatesTimeTotal += std::chrono::system_clock::now() - tpa;
                            #endif
                        }

                        // std::cerr << task.corrected_subject << "\n";
                        // std::cerr << task.correctedCandidates.size() << "\n";
                        // for(const auto& c : task.correctedCandidates){
                        //     std::cerr << c.sequence << "\n";
                        // }

                    }else{
                        #ifdef ENABLE_TIMING
                        auto tpa = std::chrono::system_clock::now();
                        #endif

                        #if 1
                        correctSubjectWithForest(taskdata, task, forestClassifier, correctionOptions);
                        #else
                        correctSubjectWithNeuralNetwork(taskdata, task, nnClassifier, correctionOptions);
                        #endif

                        #ifdef ENABLE_TIMING
                        correctWithFeaturesTimeTotal += std::chrono::system_clock::now() - tpa;
                        #endif
                    }
                }

                //create outputdata for temporary results
                setCorrectionStatusFlags(taskdata, 
                                        task,
                                        correctionStatusFlagsPerRead.data());

                createTemporaryResultsForOutput(taskdata, 
                                                task,
                                                readStorage,
                                                correctionStatusFlagsPerRead.data(),
                                                sequenceFileProperties);

                
            } //reads in batch loop end 


            if(correctionOptions.extractFeatures){
                std::cerr << "extractFeatures not implemented\n";
                /*for(const auto& correctionTasks : correctionTasksPerThread){
                        for(size_t i = 0; i < correctionTasks.size(); i++){
                            auto& task = correctionTasks[i];
                            auto& taskdata = *task.taskDataPtr;

                            if(task.active){
                                for(const auto& msafeature : taskdata.msaforestfeatures){
                                    featurestream << task.readId << '\t' << msafeature.position << '\t' << msafeature.consensus << '\n';
                                    featurestream << msafeature << '\n';
                                }
                            }
                        }
                }*/
            }else{
                //collect results of batch into a single buffer, then write results to file in another thread
                std::vector<TempCorrectedSequence> anchorcorrections;
                std::vector<EncodedTempCorrectedSequence> encodedanchorcorrections;
                anchorcorrections.reserve(readIds.size());
                encodedanchorcorrections.reserve(readIds.size());

                std::vector<std::vector<TempCorrectedSequence>> candidatecorrectionsPerTask;
                std::vector<std::vector<EncodedTempCorrectedSequence>> encodedcandidatecorrectionsPerTask;
                candidatecorrectionsPerTask.reserve(readIds.size());
                encodedcandidatecorrectionsPerTask.reserve(readIds.size());

                for(size_t i = 0; i < correctionTasks.size(); i++){
                    auto& task = correctionTasks[i];
                    auto& taskdata = *task.taskDataPtr;

                    if(task.active){
                        if(task.corrected){                        
                            anchorcorrections.emplace_back(std::move(task.anchoroutput));
                            encodedanchorcorrections.emplace_back(anchorcorrections.back().encode());
                        }
                        if(task.candidatesoutput.size() > 0){
                            candidatecorrectionsPerTask.emplace_back(std::move(task.candidatesoutput));

                            std::vector<EncodedTempCorrectedSequence> encodedvec;
                            for(const auto& tcs : candidatecorrectionsPerTask.back()){
                                encodedvec.emplace_back(tcs.encode());
                            }

                            encodedcandidatecorrectionsPerTask.emplace_back(std::move(encodedvec));
                        } 
                    }
                }

                auto outputfunction = [&,
                                    anchorcorrections = std::move(anchorcorrections),
                                    candidatecorrectionsPerTask = std::move(candidatecorrectionsPerTask),
                                    encodedanchorcorrections = std::move(encodedanchorcorrections),
                                    encodedcandidatecorrectionsPerTask = std::move(encodedcandidatecorrectionsPerTask)
                                    ](){
                    for(int i = 0; i < int(anchorcorrections.size()); i++){
                        saveCorrectedSequence(anchorcorrections[i], encodedanchorcorrections[i]);
                    }

                    for(int i = 0; i < (candidatecorrectionsPerTask.size()); i++){
                        for(int j = 0; j < (candidatecorrectionsPerTask[i].size()); j++){
                            saveCorrectedSequence(candidatecorrectionsPerTask[i][j], encodedcandidatecorrectionsPerTask[i][j]);
                        }
                    }
                };

                outputThread.enqueue(std::move(outputfunction));
            }

            progressThread.addProgress(readIds.size()); 
        } //while unprocessed reads exist loop end   


    } // parallel end

    progressThread.finished();

    // if(runtimeOptions.showProgress/* && readIdGenerator.getCurrentUnsafe() - previousprocessedreads > 100000*/){
    //     const auto now = std::chrono::system_clock::now();
    //     runtime = now - timepoint_begin;

    //     printf("Processed %10u of %10lu reads (Runtime: %03d:%02d:%02d)\n",
    //             readIdGenerator.getCurrentUnsafe() - readIdGenerator.getBegin(), sequenceFileProperties.nReads,
    //             int(std::chrono::duration_cast<std::chrono::hours>(runtime).count()),
    //             int(std::chrono::duration_cast<std::chrono::minutes>(runtime).count()) % 60,
    //             int(runtime.count()) % 60);
    //     //previousprocessedreads = readIdGenerator.getCurrentUnsafe();
    // }

    outputThread.stopThread(BackgroundThread::StopType::FinishAndStop);

    featurestream.flush();
    //outputstream.flush();
    partialResults.flush();

    minhasher.destroy();
    readStorage.destroy();


    std::chrono::duration<double> totalDuration = getCandidatesTimeTotal
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
                                                + msaCorrectCandidatesTimeTotal
                                                + correctWithFeaturesTimeTotal;

    auto printDuration = [&](const auto& name, const auto& duration){
        std::cout << "# elapsed time ("<< name << "): "
                  << duration.count()  << " s. "
                  << (100.0 * duration / totalDuration) << " %."<< std::endl;
    };

    #define printme(x) printDuration((#x),(x));

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
    printme(correctWithFeaturesTimeTotal);

    #undef printme

#ifdef DO_PROFILE

    return;

#endif

    std::cout << "Correction finished. Constructing result file." << std::endl;

    if(!correctionOptions.extractFeatures){

        std::cout << "begin merging reads" << std::endl;

        TIMERSTARTCPU(merge);

        mergeResultFiles(
                        fileOptions.tempdirectory,
                        sequenceFileProperties.nReads, 
                        fileOptions.inputfile, 
                        fileOptions.format, 
                        partialResults, 
                        fileOptions.outputfile, 
                        false);

        TIMERSTOPCPU(merge);

        std::cout << "end merging reads" << std::endl;

    }

    deleteFiles(tmpfiles);

    std::vector<std::string> featureFiles(tmpfiles);
    for(auto& s : featureFiles)
        s = s + "_features";

      //concatenate feature files one file

      if(correctionOptions.extractFeatures){
          std::cout << "begin merging features" << std::endl;

          std::stringstream commandbuilder;

          commandbuilder << "cat";

          for(const auto& featureFile : featureTmpFiles){
              commandbuilder << " \"" << featureFile << "\"";
          }

          commandbuilder << " > \"" << fileOptions.outputfile << "_features\"";

          const std::string command = commandbuilder.str();
          TIMERSTARTCPU(concat_feature_files);
          int r1 = std::system(command.c_str());
          TIMERSTOPCPU(concat_feature_files);

          if(r1 != 0){
              std::cerr << "Warning. Feature files could not be concatenated!\n";
              std::cerr << "This command returned a non-zero error value: \n";
              std::cerr << command +  '\n';
              std::cerr << "Please concatenate the following files manually\n";
              for(const auto& s : featureTmpFiles)
                  std::cerr << s << '\n';
          }else{
              deleteFiles(featureTmpFiles);
          }

          std::cout << "end merging features" << std::endl;
      }else{
          deleteFiles(featureTmpFiles);
      }

    std::cout << "end merge" << std::endl;

    omp_set_num_threads(oldNumOMPThreads);
}



}
}


#ifdef MSA_IMPLICIT
#undef MSA_IMPLICIT
#endif

#ifdef ENABLE_TIMING
#undef ENABLE_TIMING
#endif
