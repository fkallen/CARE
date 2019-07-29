#include <cpu_correction_thread.hpp>

#include <config.hpp>

#include "options.hpp"

#include "cpu_alignment.hpp"
#include "bestalignment.hpp"
#include <msa.hpp>
#include "qualityscoreweights.hpp"
#include "rangegenerator.hpp"
#include "featureextractor.hpp"
#include "forestclassifier.hpp"
#include "nn_classifier.hpp"

#include "cpu_correction_core.hpp"

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
//#define USE_SUBJECT_CLIPPING

#define ENABLE_TIMING

//#define PRINT_MSA

namespace care{
namespace cpu{

        template<class Iter>
        void initializeTasks(std::vector<CPUCorrectionThread::CorrectionTask>& tasks,
                             const cpu::ContiguousReadStorage& readStorage,
                             const std::vector<read_number>& readIds,
                             int numThreads = 1){

            auto identity = [](auto i){return i;};

            omp_set_num_threads(numThreads);

            #pragma omp parallel for schedule(dynamic, 5)
            for(size_t i = 0; i < readIds.size(); i++){
                auto& task = tasks[i];
                const read_number readId = readIds[i];

                const char* const originalsubjectptr = readStorage.fetchSequenceData_ptr(readId);
                const int originalsubjectLength = readStorage.fetchSequenceLength(readId);

                task = std::move(CPUCorrectionThread::CorrectionTask{readId});

                task.original_subject_string.resize(originalsubjectLength);
                decode2BitHiLoSequence(&task.original_subject_string[0],
                                        (const unsigned int*) originalsubjectptr, originalsubjectLength, identity);
            }
        }

        void getCandidates(std::vector<CPUCorrectionThread::CorrectionTask>& tasks,
                            const Minhasher& minhasher,
                            int maxNumberOfCandidates,
                            int requiredHitsPerCandidate,
                            size_t maxNumResultsPerMapQuery,
                            int numThreads = 1){

            omp_set_num_threads(numThreads);

            #pragma omp parallel for schedule(dynamic, 5)
            for(size_t i = 0; i < tasks.size(); i++){
                auto& task = tasks[i];
                task.candidate_read_ids = minhasher.getCandidates(task.subject_string,
                                                                   requiredHitsPerCandidate,
                                                                   maxNumberOfCandidates,
                                                                   maxNumResultsPerMapQuery);

                //remove our own read id from candidate list. candidate_read_ids is sorted.
                auto readIdPos = std::lower_bound(task.candidate_read_ids.begin(),
                                                    task.candidate_read_ids.end(),
                                                    task.readId);

                if(readIdPos != task.candidate_read_ids.end() && *readIdPos == task.readId){
                    task.candidate_read_ids.erase(readIdPos);
                }
            }
        }


        void CPUCorrectionThread::run(){
            if(isRunning) throw std::runtime_error("CPUCorrectionThread::run: Is already running.");
            isRunning = true;
            thread = std::move(std::thread(&CPUCorrectionThread::execute, this));
        }

        void CPUCorrectionThread::join(){
            thread.join();
            isRunning = false;
        }

    	void CPUCorrectionThread::execute() {

            using MSA_t = care::MultipleSequenceAlignment;

    		isRunning = true;

            const int max_sequence_bytes = sizeof(unsigned int) * getEncodedNumInts2BitHiLo(fileProperties.maxSequenceLength);

    		//std::chrono::time_point<std::chrono::system_clock> tpa, tpb, tpc, tpd;

    		std::ofstream outputstream(threadOpts.outputfile);

            std::ofstream featurestream(threadOpts.outputfile + "_features");
    		auto write_read = [&](const read_number readId, const auto& sequence){
                //std::cout << readId << " " << sequence << std::endl;
    			auto& stream = outputstream;
			assert(sequence.size() > 0);

			for(const auto& c : sequence){
				assert(c == 'A' || c == 'C' || c == 'G' || c == 'T' || c =='N');
			}

    			stream << readId << ' ' << sequence << '\n';
    		};

    		auto lock = [&](read_number readId){
    			read_number index = readId % threadOpts.nLocksForProcessedFlags;
    			threadOpts.locksForProcessedFlags[index].lock();
    		};

    		auto unlock = [&](read_number readId){
    			read_number index = readId % threadOpts.nLocksForProcessedFlags;
    			threadOpts.locksForProcessedFlags[index].unlock();
    		};

            auto identity = [](auto i){return i;};

            MSA_t multipleSequenceAlignment;

            std::vector<read_number> readIds;

            ForestClassifier forestClassifier;
            if(correctionOptions.correctionType == CorrectionType::Forest){
                forestClassifier = std::move(ForestClassifier{fileOptions.forestfilename});
            }

            std::vector<CorrectionTask_t> correctionTasks(1);

            //std::cerr << "correctionOptions.hits_per_candidate " <<  correctionOptions.hits_per_candidate << ", max_candidates " << max_candidates << '\n';

            std::map<int, int> totalnumcandidatesmap;

    		while(!stopAndAbort && !(threadOpts.readIdGenerator->empty() && readIds.empty())){

                if(readIds.empty())
                    readIds = threadOpts.readIdGenerator->next_n(100);

                if(readIds.empty())
                    continue;

                correctionTasks[0] = std::move(CorrectionTask_t(readIds.back()));
                readIds.pop_back();

                bool ok = false;
                lock(correctionTasks[0].readId);
                if ((*threadOpts.readIsCorrectedVector)[correctionTasks[0].readId] == 0) {
                    (*threadOpts.readIsCorrectedVector)[correctionTasks[0].readId] = 1;
                    ok = true;
                }else{
                }
                unlock(correctionTasks[0].readId);

                if(!ok)
                    continue; //already corrected

                const char* originalsubjectptr = threadOpts.readStorage->fetchSequenceData_ptr(correctionTasks[0].readId);
                const int originalsubjectLength = threadOpts.readStorage->fetchSequenceLength(correctionTasks[0].readId);

                correctionTasks[0].original_subject_string.resize(originalsubjectLength);
                decode2BitHiLoSequence(&correctionTasks[0].original_subject_string[0],
                                        (const unsigned int*) originalsubjectptr, originalsubjectLength, identity);
                correctionTasks[0].subject_string = correctionTasks[0].original_subject_string;
                correctionTasks[0].clipping_begin = 0;
                correctionTasks[0].clipping_end = originalsubjectLength;



                // -----

                bool needsSecondPassAfterClipping = false;
                bool discardThisTask = false;
                std::vector<unsigned int> subjectsequence;


                std::vector<char> candidateData;
                std::vector<char> candidateRevcData;
                std::vector<int> candidateLengths;
                int max_candidate_length = 0;
                std::vector<char*> candidateDataPtrs;
                std::vector<char*> candidateRevcDataPtrs;

                std::vector<AlignmentResult_t> forwardAlignments;
                std::vector<AlignmentResult_t> revcAlignments;

                std::vector<AlignmentResult_t> bestAlignments;
                std::vector<BestAlignment_t> bestAlignmentFlags;
                std::vector<int> bestAlignmentShifts;
                std::vector<float> bestAlignmentWeights;
                std::vector<read_number> bestCandidateReadIds;
                std::vector<int> bestCandidateLengths;
                std::vector<char> bestCandidateData;
                std::vector<char*> bestCandidatePtrs;

                std::vector<char> bestCandidateQualityData;
                std::vector<char*> bestCandidateQualityPtrs;
                //std::vector<std::string> bestCandidateStrings;
                std::vector<char> bestCandidateStrings;

                //this loop allows a second pass after subject has been clipped
                int clippingIters = 0;

                do{

                    clippingIters++;

                    const char* subjectptr = originalsubjectptr;
                    int subjectLength = originalsubjectLength;
                    if(needsSecondPassAfterClipping){
                        const int length = correctionTasks[0].subject_string.length();
                        subjectsequence.resize(getEncodedNumInts2BitHiLo(length));
                        encodeSequence2BitHiLo(subjectsequence.data(), correctionTasks[0].subject_string.c_str(), length, identity);

                        subjectptr = (const char*)subjectsequence.data();
                        subjectLength = length;
                    }

                    if(clippingIters > 1){
                        //std::cout << "before: " << correctionTasks[0].candidate_read_ids.size() << " candidates\n";
                    }

#ifdef ENABLE_TIMING
                    auto tpa = std::chrono::system_clock::now();
#endif
                    const int candidate_limit = clippingIters > 1 ? std::numeric_limits<int>::max() : max_candidates;

                    const size_t maxNumResultsPerMapQuery = correctionOptions.estimatedCoverage * 2.5;

                    getCandidates(correctionTasks,
                                    *threadOpts.minhasher,
                                    candidate_limit,
                                    correctionOptions.hits_per_candidate,
                                    maxNumResultsPerMapQuery,
                                    1);


#ifdef ENABLE_TIMING
                    getCandidatesTimeTotal += std::chrono::system_clock::now() - tpa;
#endif

                    if(clippingIters > 1){
                        //std::cout << "after: " << correctionTasks[0].candidate_read_ids.size() << " candidates\n";
                    }

                    int myNumCandidates = int(correctionTasks[0].candidate_read_ids.size());

                    if(myNumCandidates == 0){
                        discardThisTask = true; //no candidates to use for correction
                        break;
                    }

#ifdef ENABLE_TIMING
                    tpa = std::chrono::system_clock::now();
#endif

                    //copy candidates lengths into buffer
                    candidateLengths.clear();
                    candidateLengths.reserve(myNumCandidates);

                    max_candidate_length = 0;
                    for(const read_number candidateId: correctionTasks[0].candidate_read_ids){
                        const int candidateLength = threadOpts.readStorage->fetchSequenceLength(candidateId);
                        candidateLengths.emplace_back(candidateLength);
                        max_candidate_length = std::max(max_candidate_length, candidateLength);
                    }

                    assert(max_candidate_length > 0);

                    //max_candidate_bytes = Sequence_t::getNumBytes(max_candidate_length);

                    candidateData.clear();
                    candidateData.resize(max_sequence_bytes * myNumCandidates, 0);
                    candidateRevcData.clear();
                    candidateRevcData.resize(max_sequence_bytes * myNumCandidates, 0);
                    candidateDataPtrs.clear();
                    candidateDataPtrs.resize(myNumCandidates, nullptr);
                    candidateRevcDataPtrs.clear();
                    candidateRevcDataPtrs.resize(myNumCandidates, nullptr);

                    //copy candidate data and reverse complements into buffer

                    constexpr int prefetch_distance_sequences = 4;

                    for(int i = 0; i < myNumCandidates && i < prefetch_distance_sequences; ++i) {
                        const read_number next_candidate_read_id = correctionTasks[0].candidate_read_ids[i];
                        const char* nextsequenceptr = threadOpts.readStorage->fetchSequenceData_ptr(next_candidate_read_id);
                        __builtin_prefetch(nextsequenceptr, 0, 0);
                    }

                    for(int i = 0; i < myNumCandidates; i++){
                        if(i + prefetch_distance_sequences < myNumCandidates) {
                            const read_number next_candidate_read_id = correctionTasks[0].candidate_read_ids[i + prefetch_distance_sequences];
                            const char* nextsequenceptr = threadOpts.readStorage->fetchSequenceData_ptr(next_candidate_read_id);
                            __builtin_prefetch(nextsequenceptr, 0, 0);
                        }

                        const read_number candidateId = correctionTasks[0].candidate_read_ids[i];
                        const char* candidateptr = threadOpts.readStorage->fetchSequenceData_ptr(candidateId);
                        const int candidateLength = candidateLengths[i];
                        const int bytes = getEncodedNumInts2BitHiLo(candidateLength) * sizeof(unsigned int);

                        char* const candidateDataBegin = candidateData.data() + i * max_sequence_bytes;
                        char* const candidateRevcDataBegin = candidateRevcData.data() + i * max_sequence_bytes;

                        std::copy(candidateptr, candidateptr + bytes, candidateDataBegin);
                        reverseComplement2BitHiLo((unsigned int*)(candidateRevcDataBegin),
                                                  (const unsigned int*)(candidateptr),
                                                  candidateLength,
                                                  identity,
                                                  identity);

                        candidateDataPtrs[i] = candidateDataBegin;
                        candidateRevcDataPtrs[i] = candidateRevcDataBegin;
                    }

#ifdef ENABLE_TIMING
                    copyCandidateDataToBufferTimeTotal += std::chrono::system_clock::now() - tpa;
#endif

#ifdef ENABLE_TIMING
                    tpa = std::chrono::system_clock::now();
#endif

                    //calculate alignments
                    forwardAlignments.resize(myNumCandidates);
                    revcAlignments.resize(myNumCandidates);

                    /*auto forwardAlignments = shd::cpu_multi_shifted_hamming_distance_popcount(subjectptr,
                                                                subjectLength,
                                                                candidateData,
                                                                candidateLengths,
                                                                max_sequence_bytes,
                                                                goodAlignmentProperties.min_overlap,
                                                                goodAlignmentProperties.maxErrorRate,
                                                                goodAlignmentProperties.min_overlap_ratio);
                    auto revcAlignments = shd::cpu_multi_shifted_hamming_distance_popcount(subjectptr,
                                                                subjectLength,
                                                                candidateRevcData,
                                                                candidateLengths,
                                                                max_sequence_bytes,
                                                                goodAlignmentProperties.min_overlap,
                                                                goodAlignmentProperties.maxErrorRate,
                                                                goodAlignmentProperties.min_overlap_ratio);*/

                    shd::cpu_multi_shifted_hamming_distance_popcount(forwardAlignments.begin(),
                                                                    subjectptr,
                                                                    subjectLength,
                                                                    candidateData,
                                                                    candidateLengths,
                                                                    max_sequence_bytes,
                                                                    goodAlignmentProperties.min_overlap,
                                                                    goodAlignmentProperties.maxErrorRate,
                                                                    goodAlignmentProperties.min_overlap_ratio);

                    shd::cpu_multi_shifted_hamming_distance_popcount(revcAlignments.begin(),
                                                                    subjectptr,
                                                                    subjectLength,
                                                                    candidateRevcData,
                                                                    candidateLengths,
                                                                    max_sequence_bytes,
                                                                    goodAlignmentProperties.min_overlap,
                                                                    goodAlignmentProperties.maxErrorRate,
                                                                    goodAlignmentProperties.min_overlap_ratio);

#ifdef ENABLE_TIMING
                    getAlignmentsTimeTotal += std::chrono::system_clock::now() - tpa;
#endif

#ifdef ENABLE_TIMING
                    tpa = std::chrono::system_clock::now();
#endif

                    //decide whether to keep forward or reverse complement
                    auto alignmentFlags = findBestAlignmentDirection(forwardAlignments,
                                                                    revcAlignments,
                                                                    subjectLength,
                                                                    candidateLengths,
                                                                    goodAlignmentProperties.min_overlap,
                                                                    goodAlignmentProperties.maxErrorRate,
                                                                    goodAlignmentProperties.min_overlap_ratio);

#ifdef ENABLE_TIMING
                    findBestAlignmentDirectionTimeTotal += std::chrono::system_clock::now() - tpa;
#endif

                    int numGoodDirection = std::count_if(alignmentFlags.begin(),
                                                         alignmentFlags.end(),
                                                         [](const auto flag){
                                                            return flag != BestAlignment_t::None;
                                                         });

                    if(numGoodDirection == 0){
                        discardThisTask = true; //no good alignments
                        break;
                    }

#ifdef ENABLE_TIMING
                    tpa = std::chrono::system_clock::now();
#endif

                    //gather data for candidates with good alignment direction

                    bestAlignments.clear();
                    bestAlignmentFlags.clear();
                    bestCandidateReadIds.clear();
                    bestCandidateData.clear();
                    bestCandidateLengths.clear();
                    bestCandidatePtrs.clear();

                    bestAlignments.resize(numGoodDirection);
                    bestAlignmentFlags.resize(numGoodDirection);
                    bestCandidateReadIds.resize(numGoodDirection);
                    bestCandidateData.resize(numGoodDirection * max_sequence_bytes);
                    bestCandidateLengths.resize(numGoodDirection);
                    bestCandidatePtrs.resize(numGoodDirection);

                    for(int i = 0; i < numGoodDirection; i++){
                        bestCandidatePtrs[i] = bestCandidateData.data() + i * max_sequence_bytes;
                    }

                    for(int i = 0, insertpos = 0; i < int(alignmentFlags.size()); i++){

                        const BestAlignment_t flag = alignmentFlags[i];
                        const auto& fwdAlignment = forwardAlignments[i];
                        const auto& revcAlignment = revcAlignments[i];
                        const read_number candidateId = correctionTasks[0].candidate_read_ids[i];
                        const int candidateLength = candidateLengths[i];

                        if(flag == BestAlignment_t::Forward){
                            bestAlignmentFlags[insertpos] = flag;
                            bestCandidateReadIds[insertpos] = candidateId;
                            bestCandidateLengths[insertpos] = candidateLength;

                            bestAlignments[insertpos] = fwdAlignment;
                            std::copy(candidateDataPtrs[i],
                                      candidateDataPtrs[i] + max_sequence_bytes,
                                      bestCandidatePtrs[insertpos]);
                            insertpos++;
                        }else if(flag == BestAlignment_t::ReverseComplement){
                            bestAlignmentFlags[insertpos] = flag;
                            bestCandidateReadIds[insertpos] = candidateId;
                            bestCandidateLengths[insertpos] = candidateLength;

                            bestAlignments[insertpos] = revcAlignment;
                            std::copy(candidateRevcDataPtrs[i],
                                      candidateRevcDataPtrs[i] + max_sequence_bytes,
                                      bestCandidatePtrs[insertpos]);
                            insertpos++;
                        }else{
                            ;// discard
                        }
                    }

#ifdef ENABLE_TIMING
                    gatherBestAlignmentDataTimeTotal += std::chrono::system_clock::now() - tpa;
#endif

#ifdef ENABLE_TIMING
                    tpa = std::chrono::system_clock::now();
#endif

                    //get indices of alignments which have a good mismatch ratio
                    auto goodIndices = filterAlignmentsByMismatchRatio(bestAlignments,
                                                                       correctionOptions.estimatedErrorrate,
                                                                       correctionOptions.estimatedCoverage,
                                                                       correctionOptions.m_coverage,
                                                                       [hpc = correctionOptions.hits_per_candidate](){
                                                                           return hpc > 1;
                                                                       });

#ifdef ENABLE_TIMING
                   mismatchRatioFilteringTimeTotal += std::chrono::system_clock::now() - tpa;
#endif

                    if(goodIndices.size() == 0){
                        discardThisTask = true; //no good mismatch ratio
                        break;

                        /*if(!needsSecondPassAfterClipping){
                            //std::cout << "A" << correctionTasks[0].readId << std::endl;
                            needsSecondPassAfterClipping = true;

                            correctionTasks[0].clipping_begin = 0;
                            correctionTasks[0].clipping_end = std::max(0, int(correctionTasks[0].original_subject_string.length()-10));
                            const int clipsize = correctionTasks[0].clipping_end - correctionTasks[0].clipping_begin;
                            correctionTasks[0].subject_string = correctionTasks[0].original_subject_string.substr(correctionTasks[0].clipping_begin, clipsize);

                            continue;
                        }else{
                            //std::cout << "B" << correctionTasks[0].readId << std::endl;
                            needsSecondPassAfterClipping = false;
                            discardThisTask = true;
                            break;
                        }*/
                    }else{
                        //needsSecondPassAfterClipping = false;
                    }

                    //std::cout << correctionTasks[0].readId << std::endl;


#ifdef ENABLE_TIMING
                    tpa = std::chrono::system_clock::now();
#endif

                    //stream compaction. keep only data at positions given by goodIndices

                    for(int i = 0; i < int(goodIndices.size()); i++){
                        const int fromIndex = goodIndices[i];
                        const int toIndex = i;

                        bestAlignments[toIndex] = bestAlignments[fromIndex];
                        bestAlignmentFlags[toIndex] = bestAlignmentFlags[fromIndex];
                        bestCandidateReadIds[toIndex] = bestCandidateReadIds[fromIndex];
                        bestCandidateLengths[toIndex] = bestCandidateLengths[fromIndex];

                        std::copy(bestCandidatePtrs[fromIndex],
                                  bestCandidatePtrs[fromIndex] + max_sequence_bytes,
                                  bestCandidatePtrs[toIndex]);
                    }

                    bestAlignments.erase(bestAlignments.begin() + goodIndices.size(),
                                         bestAlignments.end());
                    bestAlignmentFlags.erase(bestAlignmentFlags.begin() + goodIndices.size(),
                                             bestAlignmentFlags.end());
                    bestCandidateReadIds.erase(bestCandidateReadIds.begin() + goodIndices.size(),
                                               bestCandidateReadIds.end());
                    bestCandidateLengths.erase(bestCandidateLengths.begin() + goodIndices.size(),
                                               bestCandidateLengths.end());
                    bestCandidatePtrs.erase(bestCandidatePtrs.begin() + goodIndices.size(),
                                            bestCandidatePtrs.end());
                    bestCandidateData.erase(bestCandidateData.begin() + goodIndices.size() * max_sequence_bytes,
                                            bestCandidateData.end());

                    bestAlignmentShifts.resize(bestAlignments.size());
                    bestAlignmentWeights.resize(bestAlignments.size());

                    for(int i = 0; i < int(bestAlignments.size()); i++){
                        bestAlignmentShifts[i] = bestAlignments[i].shift;
                        bestAlignmentWeights[i] = 1.0f - std::sqrt(bestAlignments[i].nOps
                                                                    / (bestAlignments[i].overlap
                                                                        * goodAlignmentProperties.maxErrorRate));
                    }


#ifdef ENABLE_TIMING
                   compactBestAlignmentDataTimeTotal += std::chrono::system_clock::now() - tpa;
#endif

#ifdef ENABLE_TIMING
                    tpa = std::chrono::system_clock::now();
#endif

                    //gather quality scores of best alignments
                    if(correctionOptions.useQualityScores){
                        bestCandidateQualityData.clear();
                        bestCandidateQualityData.resize(max_candidate_length * bestAlignments.size());

                        bestCandidateQualityPtrs.clear();
                        bestCandidateQualityPtrs.resize(bestAlignments.size());

                        constexpr int prefetch_distance_qualities = 4;

                        for(int i = 0; i < int(bestAlignments.size()) && i < prefetch_distance_qualities; ++i) {
                            const read_number next_candidate_read_id = bestCandidateReadIds[i];
                            const char* nextqualityptr = threadOpts.readStorage->fetchQuality_ptr(next_candidate_read_id);
                            __builtin_prefetch(nextqualityptr, 0, 0);
                        }

                        for(int i = 0; i < int(bestAlignments.size()); i++){
                            if(i + prefetch_distance_qualities < int(bestAlignments.size())) {
                                const read_number next_candidate_read_id = bestCandidateReadIds[i + prefetch_distance_qualities];
                                const char* nextqualityptr = threadOpts.readStorage->fetchQuality_ptr(next_candidate_read_id);
                                __builtin_prefetch(nextqualityptr, 0, 0);
                            }
                            const char* qualityptr = threadOpts.readStorage->fetchQuality_ptr(bestCandidateReadIds[i]);
                            const int length = bestCandidateLengths[i];
                            const BestAlignment_t flag = bestAlignmentFlags[i];

                            if(flag == BestAlignment_t::Forward){
                                std::copy(qualityptr, qualityptr + length, &bestCandidateQualityData[max_candidate_length * i]);
                            }else{
                                std::reverse_copy(qualityptr, qualityptr + length, &bestCandidateQualityData[max_candidate_length * i]);
                            }
                        }

                        for(int i = 0; i < int(bestAlignments.size()); i++){
                            bestCandidateQualityPtrs[i] = bestCandidateQualityData.data() + i * max_candidate_length;
                        }
                    }
#ifdef ENABLE_TIMING
                    fetchQualitiesTimeTotal += std::chrono::system_clock::now() - tpa;
#endif

#ifdef ENABLE_TIMING
                    tpa = std::chrono::system_clock::now();
#endif


                    bestCandidateStrings.clear();
                    //bestCandidateStrings.reserve(bestAlignments.size());
                    bestCandidateStrings.resize(bestAlignments.size() * fileProperties.maxSequenceLength);

                    for(int i = 0; i < int(bestAlignments.size()); i++){
                        const char* ptr = bestCandidatePtrs[i];
                        const int length = bestCandidateLengths[i];
                        decode2BitHiLoSequence(&bestCandidateStrings[i * fileProperties.maxSequenceLength],
                                                (const unsigned int*)ptr,
                                                length,
                                                identity);
                    }

#ifdef ENABLE_TIMING
                    makeCandidateStringsTimeTotal += std::chrono::system_clock::now() - tpa;
#endif



#ifdef ENABLE_TIMING
                    tpa = std::chrono::system_clock::now();
#endif

                    const char* subjectQualityPtr = correctionOptions.useQualityScores ?
                                                    threadOpts.readStorage->fetchQuality_ptr(correctionTasks[0].readId)
                                                    : nullptr;
                    const char* candidateQualityPtr = correctionOptions.useQualityScores ?
                                                            bestCandidateQualityData.data()
                                                            : nullptr;
                    //build multiple sequence alignment
                    multipleSequenceAlignment.build(correctionTasks[0].subject_string.c_str(),
                                                subjectLength,
                                                bestCandidateStrings.data(),
                                                bestCandidateLengths.data(),
                                                int(bestAlignments.size()),
                                                bestAlignmentShifts.data(),
                                                bestAlignmentWeights.data(),
                                                subjectQualityPtr,
                                                candidateQualityPtr,
                                                fileProperties.maxSequenceLength,
                                                max_candidate_length,
                                                correctionOptions.useQualityScores);


#ifdef ENABLE_TIMING
                    msaFindConsensusTimeTotal += std::chrono::system_clock::now() - tpa;
#endif



    #if 0
                    auto print_multiple_sequence_alignment = [&](const auto& msa, const auto& alignments){
                        auto get_shift_of_row = [&](int row){
                            if(row == 0) return 0;
                            return alignments[row-1].shift;
                        };

                        const char* const my_multiple_sequence_alignment = msa.multiple_sequence_alignment.data();
                        const char* const my_consensus = msa.consensus.data();
                        const int ncolumns = msa.nColumns;
                        const int msa_rows = msa.nRows;

                        auto msaproperties = msa.getMSAProperties();
                        const bool isHQ = msaproperties.isHQ;

                        std::cout << "ReadId " << correctionTasks[0].readId << ": msa rows = " << msa_rows << ", columns = " << ncolumns << ", HQ-MSA: " << (isHQ ? "True" : "False")
                                    << '\n';

                        print_multiple_sequence_alignment_sorted_by_shift(std::cout, my_multiple_sequence_alignment, msa_rows, ncolumns, ncolumns, get_shift_of_row);
                        std::cout << '\n';
                        print_multiple_sequence_alignment_consensusdiff_sorted_by_shift(std::cout, my_multiple_sequence_alignment, my_consensus,
                                                                                        msa_rows, ncolumns, ncolumns, get_shift_of_row);
                        std::cout << '\n';
                    };

                    auto msa2 = multipleSequenceAlignment;
                    auto minimizationResult = msa2.minimize(correctionOptions.estimatedCoverage);

                    if(minimizationResult.performedMinimization && minimizationResult.num_discarded_candidates > 0){

                        msa2.findConsensus();
                        std::vector<AlignmentResult_t> remaining_alignments(minimizationResult.remaining_candidates.size());
                        for(int i = 0; i < int(minimizationResult.remaining_candidates.size()); i++){
                            remaining_alignments[i] = bestAlignments[minimizationResult.remaining_candidates[i]];
                        }
    /*
                        for(auto i : minimizationResult.remaining_candidates){
                            std::cout << i << ", ";
                        }
                        std::cout << '\n';*/

                        std::cout << ", num_discarded_candidates: " << minimizationResult.num_discarded_candidates;
                        std::cout << ", column: " << minimizationResult.column;
                        std::cout << ", significantBase: " << minimizationResult.significantBase;
                        std::cout << ", consensusBase: " << minimizationResult.consensusBase;
                        std::cout << ", originalBase: " << minimizationResult.originalBase;
                        std::cout << ", significantCount: " << minimizationResult.significantCount;
                        std::cout << ", consensuscount: " << minimizationResult.consensuscount;
                        std::cout << '\n';

                        std::cout << "Before minimization\n";
                        print_multiple_sequence_alignment(multipleSequenceAlignment, bestAlignments);
                        std::cout << "After minimization: discarded " << minimizationResult.num_discarded_candidates << " candidates\n";
                        print_multiple_sequence_alignment(msa2, remaining_alignments);


                        for(int i = 0; i < 5 && !msa2.getMSAProperties().isHQ; i++){
                            auto msa3 = msa2;
                            minimizationResult = msa3.minimize(correctionOptions.estimatedCoverage);

                            if(minimizationResult.performedMinimization && minimizationResult.num_discarded_candidates > 0){

                                msa3.findConsensus();
                                std::vector<AlignmentResult_t> remaining_alignments3(minimizationResult.remaining_candidates.size());
                                for(int i = 0; i < int(minimizationResult.remaining_candidates.size()); i++){
                                    remaining_alignments3[i] = remaining_alignments[minimizationResult.remaining_candidates[i]];
                                }

                                /*for(auto i : minimizationResult.remaining_candidates){
                                    std::cout << i << ", ";
                                }
                                std::cout << '\n';*/

                                std::cout << ", num_discarded_candidates: " << minimizationResult.num_discarded_candidates;
                                std::cout << ", column: " << minimizationResult.column;
                                std::cout << ", significantBase: " << minimizationResult.significantBase;
                                std::cout << ", consensusBase: " << minimizationResult.consensusBase;
                                std::cout << ", originalBase: " << minimizationResult.originalBase;
                                std::cout << ", significantCount: " << minimizationResult.significantCount;
                                std::cout << ", consensuscount: " << minimizationResult.consensuscount;
                                std::cout << '\n';

                                std::cout << "After minimization " << (i+2) << ": discarded " << minimizationResult.num_discarded_candidates << " candidates\n";
                                print_multiple_sequence_alignment(msa3, remaining_alignments3);

                                std::swap(msa2, msa3);
                                std::swap(remaining_alignments, remaining_alignments3);
                            }
                        }
                    }

    #endif

#ifdef USE_MSA_MINIMIZATION

#ifdef ENABLE_TIMING
                    tpa = std::chrono::system_clock::now();
#endif


#ifdef PRINT_MSA
                    std::cout << correctionTasks[0].readId << " MSA: rows = " << (int(bestAlignments.size()) + 1) << " columns = " << multipleSequenceAlignment.nColumns << "\n";
                    std::cout << "Consensus:\n   ";
                    for(int i = 0; i < multipleSequenceAlignment.nColumns; i++){
                        std::cout << multipleSequenceAlignment.consensus[i];
                    }
                    std::cout << '\n';

                    /*printSequencesInMSA(std::cout,
                                        correctionTasks[0].subject_string.c_str(),
                                        subjectLength,
                                        bestCandidateStrings.data(),
                                        bestCandidateLengths.data(),
                                        int(bestAlignments.size()),
                                        bestAlignmentShifts.data(),
                                        multipleSequenceAlignment.subjectColumnsBegin_incl,
                                        multipleSequenceAlignment.subjectColumnsEnd_excl,
                                        multipleSequenceAlignment.nColumns,
                                        fileProperties.maxSequenceLength);*/

                    printSequencesInMSAConsEq(std::cout,
                                        correctionTasks[0].subject_string.c_str(),
                                        subjectLength,
                                        bestCandidateStrings.data(),
                                        bestCandidateLengths.data(),
                                        int(bestAlignments.size()),
                                        bestAlignmentShifts.data(),
                                        multipleSequenceAlignment.consensus.data(),
                                        multipleSequenceAlignment.subjectColumnsBegin_incl,
                                        multipleSequenceAlignment.subjectColumnsEnd_excl,
                                        multipleSequenceAlignment.nColumns,
                                        fileProperties.maxSequenceLength);
#endif

                    constexpr int max_num_minimizations = 5;

                    if(max_num_minimizations > 0){
                        int num_minimizations = 0;

                        auto minimizationResult = findCandidatesOfDifferentRegion(correctionTasks[0].subject_string.c_str(),
                                                                            subjectLength,
                                                                            bestCandidateStrings.data(),
                                                                            bestCandidateLengths.data(),
                                                                            int(bestAlignments.size()),
                                                                            fileProperties.maxSequenceLength,
                                                                            multipleSequenceAlignment.consensus.data(),
                                                                            multipleSequenceAlignment.countsA.data(),
                                                                            multipleSequenceAlignment.countsC.data(),
                                                                            multipleSequenceAlignment.countsG.data(),
                                                                            multipleSequenceAlignment.countsT.data(),
                                                                            multipleSequenceAlignment.weightsA.data(),
                                                                            multipleSequenceAlignment.weightsC.data(),
                                                                            multipleSequenceAlignment.weightsG.data(),
                                                                            multipleSequenceAlignment.weightsT.data(),
                                                                            multipleSequenceAlignment.subjectColumnsBegin_incl,
                                                                            multipleSequenceAlignment.subjectColumnsEnd_excl,
                                                                            bestAlignmentShifts.data(),
                                                                            correctionOptions.estimatedCoverage);

                        auto update_after_successfull_minimization = [&](){
#if 0
                            if(correctionTasks[0].readId == 207){
                            int toKeep = std::count_if(minimizationResult.differentRegionCandidate.begin(),
                                                    minimizationResult.differentRegionCandidate.end(), [](auto b){return !b;});
                            int toRemove = std::count_if(minimizationResult.differentRegionCandidate.begin(),
                                                    minimizationResult.differentRegionCandidate.end(), [](auto b){return b;});
                            std::cerr << "numindices: " << minimizationResult.differentRegionCandidate.size() << ", to keep: " << toKeep << ", toRemove: " << toRemove << '\n';

                            std::cerr << "subjectColumnsBegin_incl: " << multipleSequenceAlignment.subjectColumnsBegin_incl
                                    << ", subjectColumnsEnd_excl: " << multipleSequenceAlignment.subjectColumnsEnd_excl << '\n';

                            /*std::cerr << "shifts:\n";
                            for(auto shift : bestAlignmentShifts){
                                std::cerr << shift << ", ";
                            }
                            std::cerr << '\n';

                            std::cerr << "consensus:\n";
                            for(int i = 0; i < multipleSequenceAlignment.consensus.size(); i++){
                                std::cerr << multipleSequenceAlignment.consensus[i] << ", ";
                            }
                            std::cerr << '\n';

                            std::cerr << "countsA:\n";
                            for(int i = 0; i < multipleSequenceAlignment.countsA.size(); i++){
                                std::cerr << multipleSequenceAlignment.countsA[i] << ", ";
                            }
                            std::cerr << '\n';

                            std::cerr << "countsC:\n";
                            for(int i = 0; i < multipleSequenceAlignment.countsC.size(); i++){
                                std::cerr << multipleSequenceAlignment.countsC[i] << ", ";
                            }
                            std::cerr << '\n';

                            std::cerr << "countsG:\n";
                            for(int i = 0; i < multipleSequenceAlignment.countsG.size(); i++){
                                std::cerr << multipleSequenceAlignment.countsG[i] << ", ";
                            }
                            std::cerr << '\n';

                            std::cerr << "countsT:\n";
                            for(int i = 0; i < multipleSequenceAlignment.countsT.size(); i++){
                                std::cerr << multipleSequenceAlignment.countsT[i] << ", ";
                            }
                            std::cerr << '\n';

                            std::cerr << "weightsA:\n";
                            for(int i = 0; i < multipleSequenceAlignment.weightsA.size(); i++){
                                std::cerr << multipleSequenceAlignment.weightsA[i] << ", ";
                            }
                            std::cerr << '\n';

                            std::cerr << "weightsC:\n";
                            for(int i = 0; i < multipleSequenceAlignment.weightsC.size(); i++){
                                std::cerr << multipleSequenceAlignment.weightsC[i] << ", ";
                            }
                            std::cerr << '\n';

                            std::cerr << "weightsG:\n";
                            for(int i = 0; i < multipleSequenceAlignment.weightsG.size(); i++){
                                std::cerr << multipleSequenceAlignment.weightsG[i] << ", ";
                            }
                            std::cerr << '\n';

                            std::cerr << "weightsT:\n";
                            for(int i = 0; i < multipleSequenceAlignment.weightsT.size(); i++){
                                std::cerr << multipleSequenceAlignment.weightsT[i] << ", ";
                            }
                            std::cerr << '\n';*/

                            //std::exit(0);
                        }
#endif
                            if(minimizationResult.performedMinimization){
                                assert(minimizationResult.differentRegionCandidate.size() == bestAlignments.size());

                                bool anyRemoved = false;
                                size_t cur = 0;
                                for(size_t i = 0; i < minimizationResult.differentRegionCandidate.size(); i++){
                                    if(!minimizationResult.differentRegionCandidate[i]){
                                        //std::cout << i << " ";

                                        bestAlignments[cur] = bestAlignments[i];
                                        bestAlignmentShifts[cur] = bestAlignmentShifts[i];
                                        bestAlignmentWeights[cur] = bestAlignmentWeights[i];
                                        bestAlignmentFlags[cur] = bestAlignmentFlags[i];
                                        bestCandidateReadIds[cur] = bestCandidateReadIds[i];
                                        bestCandidateLengths[cur] = bestCandidateLengths[i];

                                        std::copy(bestCandidateData.begin() + i * max_sequence_bytes,
                                                bestCandidateData.begin() + (i+1) * max_sequence_bytes,
                                                bestCandidateData.begin() + cur * max_sequence_bytes);
                                        std::copy(bestCandidateQualityData.begin() + i * max_candidate_length,
                                                bestCandidateQualityData.begin() + (i+1) * max_candidate_length,
                                                bestCandidateQualityData.begin() + cur * max_candidate_length);
                                        std::copy(bestCandidateStrings.begin() + i * max_candidate_length,
                                                bestCandidateStrings.begin() + (i+1) * max_candidate_length,
                                                bestCandidateStrings.begin() + cur * max_candidate_length);

                                        cur++;

                                    }else{
                                        /*multipleSequenceAlignment.removeSequence(correctionOptions.useQualityScores,
                                                                                bestCandidateStrings.data() + i * max_candidate_length,
                                                                                bestCandidateQualityData.data() + i * max_candidate_length,
                                                                                bestCandidateLengths[i],
                                                                                bestAlignmentShifts[i], bestAlignmentWeights[i]);*/
                                        anyRemoved = true;
                                    }
                                }


                                /*if(!anyRemoved){
                                    std::cout << correctionTasks[0].readId << std::endl;
                                    std::cout << num_minimizations << std::endl;
                                    for(const auto b : minimizationResult.differentRegionCandidate){
                                        std::cout << b << " ";
                                    }
                                    std::cout << std::endl;
                                    std::cout << minimizationResult.performedMinimization << std::endl;
                                    std::cout << minimizationResult.column << std::endl;
                                    std::cout << minimizationResult.significantBase << std::endl;
                                    std::cout << minimizationResult.consensusBase << std::endl;
                                    std::cout << minimizationResult.originalBase << std::endl;
                                    std::cout << minimizationResult.significantCount << std::endl;
                                    std::cout << minimizationResult.consensuscount << std::endl;

                                }*/
                                assert(anyRemoved);

                                bestAlignments.erase(bestAlignments.begin() + cur, bestAlignments.end());
                                bestAlignmentShifts.erase(bestAlignmentShifts.begin() + cur, bestAlignmentShifts.end());
                                bestAlignmentWeights.erase(bestAlignmentWeights.begin() + cur, bestAlignmentWeights.end());
                                bestAlignmentFlags.erase(bestAlignmentFlags.begin() + cur, bestAlignmentFlags.end());
                                bestCandidateReadIds.erase(bestCandidateReadIds.begin() + cur, bestCandidateReadIds.end());
                                bestCandidateLengths.erase(bestCandidateLengths.begin() + cur, bestCandidateLengths.end());

                                bestCandidateData.erase(bestCandidateData.begin() + cur * max_sequence_bytes, bestCandidateData.end());
                                bestCandidateQualityData.erase(bestCandidateQualityData.begin() + cur * max_candidate_length, bestCandidateQualityData.end());
                                bestCandidateStrings.erase(bestCandidateStrings.begin() + cur * max_candidate_length, bestCandidateStrings.end());

                                /*if(anyRemoved){
                                    multipleSequenceAlignment.findConsensus();
                                    multipleSequenceAlignment.findOrigWeightAndCoverage(correctionTasks[0].subject_string.c_str());
                                }*/

                                //build minimized multiple sequence alignment
                                multipleSequenceAlignment.build(correctionTasks[0].subject_string.c_str(),
                                                                subjectLength,
                                                                bestCandidateStrings.data(),
                                                                bestCandidateLengths.data(),
                                                                int(bestAlignments.size()),
                                                                bestAlignmentShifts.data(),
                                                                bestAlignmentWeights.data(),
                                                                subjectQualityPtr,
                                                                candidateQualityPtr,
                                                                fileProperties.maxSequenceLength,
                                                                max_candidate_length,
                                                                correctionOptions.useQualityScores);

#ifdef PRINT_MSA
                                std::cout << correctionTasks[0].readId << " MSA after minimization " << (num_minimizations+1) << ": rows = " << (int(bestAlignments.size()) + 1) << " columns = " << multipleSequenceAlignment.nColumns << "\n";
                                std::cout << "Consensus:\n   ";
                                for(int i = 0; i < multipleSequenceAlignment.nColumns; i++){
                                    std::cout << multipleSequenceAlignment.consensus[i];
                                }
                                std::cout << '\n';

                                /*printSequencesInMSA(std::cout,
                                                    correctionTasks[0].subject_string.c_str(),
                                                    subjectLength,
                                                    bestCandidateStrings.data(),
                                                    bestCandidateLengths.data(),
                                                    int(bestAlignments.size()),
                                                    bestAlignmentShifts.data(),
                                                    multipleSequenceAlignment.subjectColumnsBegin_incl,
                                                    multipleSequenceAlignment.subjectColumnsEnd_excl,
                                                    multipleSequenceAlignment.nColumns,
                                                    fileProperties.maxSequenceLength);*/



                                printSequencesInMSAConsEq(std::cout,
                                                    correctionTasks[0].subject_string.c_str(),
                                                    subjectLength,
                                                    bestCandidateStrings.data(),
                                                    bestCandidateLengths.data(),
                                                    int(bestAlignments.size()),
                                                    bestAlignmentShifts.data(),
                                                    multipleSequenceAlignment.consensus.data(),
                                                    multipleSequenceAlignment.subjectColumnsBegin_incl,
                                                    multipleSequenceAlignment.subjectColumnsEnd_excl,
                                                    multipleSequenceAlignment.nColumns,
                                                    fileProperties.maxSequenceLength);
#endif
                            }
                        };

                        update_after_successfull_minimization();


                        num_minimizations++;

                        while(num_minimizations <= max_num_minimizations
                                && minimizationResult.performedMinimization){

                            minimizationResult = findCandidatesOfDifferentRegion(correctionTasks[0].subject_string.c_str(),
                                                                                subjectLength,
                                                                                bestCandidateStrings.data(),
                                                                                bestCandidateLengths.data(),
                                                                                int(bestAlignments.size()),
                                                                                fileProperties.maxSequenceLength,
                                                                                multipleSequenceAlignment.consensus.data(),
                                                                                multipleSequenceAlignment.countsA.data(),
                                                                                multipleSequenceAlignment.countsC.data(),
                                                                                multipleSequenceAlignment.countsG.data(),
                                                                                multipleSequenceAlignment.countsT.data(),
                                                                                multipleSequenceAlignment.weightsA.data(),
                                                                                multipleSequenceAlignment.weightsC.data(),
                                                                                multipleSequenceAlignment.weightsG.data(),
                                                                                multipleSequenceAlignment.weightsT.data(),
                                                                                multipleSequenceAlignment.subjectColumnsBegin_incl,
                                                                                multipleSequenceAlignment.subjectColumnsEnd_excl,
                                                                                bestAlignmentShifts.data(),
                                                                                correctionOptions.estimatedCoverage);

                            update_after_successfull_minimization();

                            num_minimizations++;
                        }
                    }

#ifdef ENABLE_TIMING
                    msaMinimizationTimeTotal += std::chrono::system_clock::now() - tpa;
#endif

#endif // USE_MSA_MINIMIZATION



                    //minimization is finished here
#ifdef USE_SUBJECT_CLIPPING
                    if(!needsSecondPassAfterClipping){
                        auto goodregion = findGoodConsensusRegionOfSubject2(correctionTasks[0].subject_string.c_str(),
                                                                            subjectLength,
                                                                            multipleSequenceAlignment.coverage.data(),
                                                                            multipleSequenceAlignment.nColumns,
                                                                            multipleSequenceAlignment.subjectColumnsEnd_excl);

                        if(goodregion.first > 0 || goodregion.second < int(correctionTasks[0].subject_string.size())){
                            /*const int negativeShifts = std::count_if(multipleSequenceAlignment.shifts.begin(),
                                                                    multipleSequenceAlignment.shifts.end(),
                                                                    [](int s){return s < 0;});
                            const int positiveShifts = std::count_if(multipleSequenceAlignment.shifts.begin(),
                                                                    multipleSequenceAlignment.shifts.end(),
                                                                    [](int s){return s > 0;});

                            std::cout << "ReadId " << correctionTasks[0].readId << " : [" << goodregion.first << ", "
                                        << goodregion.second << "] negativeShifts " << negativeShifts
                                        << ", positiveShifts " << positiveShifts
                                        << ". Subject starts at column "
                                        << multipleSequenceAlignment.columnProperties.subjectColumnsBegin_incl
                                        << ". Subject ends at column "
                                        << multipleSequenceAlignment.columnProperties.subjectColumnsEnd_excl
                                        << " / " << multipleSequenceAlignment.nColumns << "\n";*/

                            needsSecondPassAfterClipping = true;

                            correctionTasks[0].clipping_begin = goodregion.first;
                            correctionTasks[0].clipping_end = goodregion.second;
                            const int clipsize = correctionTasks[0].clipping_end - correctionTasks[0].clipping_begin;
                            correctionTasks[0].subject_string = correctionTasks[0].original_subject_string.substr(correctionTasks[0].clipping_begin, clipsize);
                        }
                    }else{
                        //subject has already been clipped in previous iteration, do not clip again
                        needsSecondPassAfterClipping = false;
                    }
#endif

                }while(needsSecondPassAfterClipping && !discardThisTask);
                //}while(false);


                if(discardThisTask){
                    continue;
                }



                if(correctionOptions.extractFeatures){
#if 1
                    auto MSAFeatures = extractFeatures(multipleSequenceAlignment.consensus.data(),
                                                    multipleSequenceAlignment.support.data(),
                                                    multipleSequenceAlignment.coverage.data(),
                                                    multipleSequenceAlignment.origCoverages.data(),
                                                    multipleSequenceAlignment.nColumns,
                                                    multipleSequenceAlignment.subjectColumnsBegin_incl,
                                                    multipleSequenceAlignment.subjectColumnsEnd_excl,
                                                    correctionTasks[0].subject_string,
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
                                            correctionTasks[0].subject_string,
                                            correctionOptions.estimatedCoverage);


#endif

                    for(const auto& msafeature : MSAFeatures){
                        featurestream << correctionTasks[0].readId << '\t' << msafeature.position << '\t' << msafeature.consensus << '\n';
                        featurestream << msafeature << '\n';
                    }
                }else{ //correction is not performed when extracting features

                    if(correctionOptions.correctionType == CorrectionType::Classic){

    #ifdef ENABLE_TIMING
                        auto tpa = std::chrono::system_clock::now();
    #endif
                        //get corrected subject and write it to file

                        const int subjectColumnsBegin_incl = multipleSequenceAlignment.subjectColumnsBegin_incl;
                        const int subjectColumnsEnd_excl = multipleSequenceAlignment.subjectColumnsEnd_excl;

                        //if(correctionTasks[0].readId == 207){
                        //    std::cerr << "debug\n";
                        //}

                        MSAProperties msaProperties = getMSAProperties2(multipleSequenceAlignment.support.data(),
                                                                        multipleSequenceAlignment.coverage.data(),
                                                                        //int(correctionTasks[0].subject_string.size()),
                                                                        subjectColumnsBegin_incl,
                                                                        subjectColumnsEnd_excl,
                                                                        correctionOptions.estimatedErrorrate,
                                                                        correctionOptions.estimatedCoverage,
                                                                        correctionOptions.m_coverage);

                        // auto correctionResult = getCorrectedSubject(multipleSequenceAlignment.consensus.data() + subjectColumnsBegin_incl,
                        //                                             multipleSequenceAlignment.support.data() + subjectColumnsBegin_incl,
                        //                                             multipleSequenceAlignment.coverage.data() + subjectColumnsBegin_incl,
                        //                                             multipleSequenceAlignment.origCoverages.data() + subjectColumnsBegin_incl,
                        //                                             int(correctionTasks[0].subject_string.size()),
                        //                                             correctionTasks[0].subject_string.c_str(),
                        //                                             msaProperties.isHQ,
                        //                                             correctionOptions.estimatedErrorrate,
                        //                                             correctionOptions.estimatedCoverage,
                        //                                             correctionOptions.m_coverage,
                        //                                             correctionOptions.kmerlength);

                        auto correctionResult = getCorrectedSubject(multipleSequenceAlignment.consensus.data() + subjectColumnsBegin_incl,
                                                                    multipleSequenceAlignment.support.data() + subjectColumnsBegin_incl,
                                                                    multipleSequenceAlignment.coverage.data() + subjectColumnsBegin_incl,
                                                                    multipleSequenceAlignment.origCoverages.data() + subjectColumnsBegin_incl,
                                                                    int(correctionTasks[0].subject_string.size()),
                                                                    correctionTasks[0].subject_string.c_str(),
                                                                    subjectColumnsBegin_incl,
                                                                    bestCandidateStrings.data(),
                                                                    int(bestAlignmentWeights.size()),
                                                                    bestAlignmentWeights.data(),
                                                                    bestCandidateLengths.data(),
                                                                    bestAlignmentShifts.data(),
                                                                    fileProperties.maxSequenceLength,
                                                                    msaProperties.isHQ,
                                                                    correctionOptions.estimatedErrorrate,
                                                                    correctionOptions.estimatedCoverage,
                                                                    correctionOptions.m_coverage,
                                                                    correctionOptions.kmerlength);


    #ifdef ENABLE_TIMING
                        msaCorrectSubjectTimeTotal += std::chrono::system_clock::now() - tpa;
    #endif


                        if(correctionResult.isCorrected){
                            //need to replace the bases in the good region by the corrected bases of the clipped read
                            if(clippingIters == 1){
                                correctionTasks[0].corrected_subject = correctionResult.correctedSequence;
                            }else{
                                correctionTasks[0].corrected_subject = correctionTasks[0].original_subject_string;
                                std::copy(correctionResult.correctedSequence.begin(),
                                          correctionResult.correctedSequence.end(),
                                          correctionTasks[0].corrected_subject.begin() + correctionTasks[0].clipping_begin);
                            }
                        }




                        /*if(!correctionResult.isCorrected || correctionResult.correctedSequence == correctionTasks[0].original_subject_string){
                            const std::size_t numCandidates = correctionTasks[0].candidate_read_ids.size();
                            numCandidatesOfUncorrectedSubjects[numCandidates]++;
                        }*/

                        if(correctionResult.isCorrected){

                            write_read(correctionTasks[0].readId, correctionTasks[0].corrected_subject);
                            lock(correctionTasks[0].readId);
                            (*threadOpts.readIsCorrectedVector)[correctionTasks[0].readId] = 1;
                            unlock(correctionTasks[0].readId);
                        }else{

                            //make subject available for correction as a candidate
                            if((*threadOpts.readIsCorrectedVector)[correctionTasks[0].readId] == 1){
                                lock(correctionTasks[0].readId);
                                if((*threadOpts.readIsCorrectedVector)[correctionTasks[0].readId] == 1){
                                    (*threadOpts.readIsCorrectedVector)[correctionTasks[0].readId] = 0;
                                }
                                unlock(correctionTasks[0].readId);
                            }
                        }

                        //get corrected candidates and write them to file
                        if(correctionOptions.correctCandidates && msaProperties.isHQ){
    #ifdef ENABLE_TIMING
                            tpa = std::chrono::system_clock::now();
    #endif

                            auto correctedCandidates = getCorrectedCandidates(multipleSequenceAlignment.consensus.data(),
                                                                            multipleSequenceAlignment.support.data(),
                                                                            multipleSequenceAlignment.coverage.data(),
                                                                            multipleSequenceAlignment.nColumns,
                                                                            multipleSequenceAlignment.subjectColumnsBegin_incl,
                                                                            multipleSequenceAlignment.subjectColumnsEnd_excl,
                                                                            bestAlignmentShifts.data(),
                                                                            bestCandidateLengths.data(),
                                                                            multipleSequenceAlignment.nCandidates,
                                                                            correctionOptions.estimatedErrorrate,
                                                                            correctionOptions.estimatedCoverage,
                                                                            correctionOptions.m_coverage,
                                                                            correctionOptions.new_columns_to_correct);

    #ifdef ENABLE_TIMING
                            msaCorrectCandidatesTimeTotal += std::chrono::system_clock::now() - tpa;
    #endif

                            for(const auto& correctedCandidate : correctedCandidates){
                                const read_number candidateId = bestCandidateReadIds[correctedCandidate.index];
                                bool savingIsOk = false;
                                if((*threadOpts.readIsCorrectedVector)[candidateId] == 0){
                                    lock(candidateId);
                                    if((*threadOpts.readIsCorrectedVector)[candidateId]== 0) {
                                        (*threadOpts.readIsCorrectedVector)[candidateId] = 1; // we will process this read
                                        savingIsOk = true;
                                    }
                                    unlock(candidateId);
                                }
                                unlock(candidateId);

                                if (savingIsOk) {
                                    if(bestAlignmentFlags[correctedCandidate.index] == BestAlignment_t::Forward){
                                        write_read(candidateId, correctedCandidate.sequence);
                                    }else{
                                        std::string fwd;
                                        fwd.resize(correctedCandidate.sequence.length());
                                        reverseComplementString(&fwd[0], correctedCandidate.sequence.c_str(), correctedCandidate.sequence.length());
                                        write_read(candidateId, fwd);
                                    }
                                }
                            }
                        }

                    }else{

    #ifdef ENABLE_TIMING
                        auto tpa = std::chrono::system_clock::now();
    #endif
                        auto MSAFeatures = extractFeatures(multipleSequenceAlignment.consensus.data(),
                                                        multipleSequenceAlignment.support.data(),
                                                        multipleSequenceAlignment.coverage.data(),
                                                        multipleSequenceAlignment.origCoverages.data(),
                                                        multipleSequenceAlignment.nColumns,
                                                        multipleSequenceAlignment.subjectColumnsBegin_incl,
                                                        multipleSequenceAlignment.subjectColumnsEnd_excl,
                                                        correctionTasks[0].subject_string,
                                                        correctionOptions.kmerlength, 0.5f,
                                                        correctionOptions.estimatedCoverage);

                        correctionTasks[0].corrected_subject = correctionTasks[0].subject_string;
                        bool isCorrected = false;

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
                                isCorrected = true;

                                const int globalIndex = multipleSequenceAlignment.subjectColumnsBegin_incl + msafeature.position;
                                correctionTasks[0].corrected_subject[msafeature.position] = multipleSequenceAlignment.consensus[globalIndex];
                            }
                        }

    #ifdef ENABLE_TIMING
                        correctWithFeaturesTimeTotal += std::chrono::system_clock::now() - tpa;
    #endif

    #if 0
    MSAFeatures3 = extractFeatures3_2(
                                multipleSequenceAlignment.countsA.data(),
                                multipleSequenceAlignment.countsC.data(),
                                multipleSequenceAlignment.countsG.data(),
                                multipleSequenceAlignment.countsT.data(),
                                multipleSequenceAlignment.weightsA.data(),
                                multipleSequenceAlignment.weightsC.data(),
                                multipleSequenceAlignment.weightsG.data(),
                                multipleSequenceAlignment.weightsT.data(),
                                multipleSequenceAlignment.nRows,
                                multipleSequenceAlignment.columnProperties.columnsToCheck,
                                multipleSequenceAlignment.consensus.data(),
                                multipleSequenceAlignment.support.data(),
                                multipleSequenceAlignment.coverage.data(),
                                multipleSequenceAlignment.origCoverages.data(),
                                multipleSequenceAlignment.columnProperties.subjectColumnsBegin_incl,
                                multipleSequenceAlignment.columnProperties.subjectColumnsEnd_excl,
                                task.subject_string,
                                correctionOptions.estimatedCoverage);

                        std::vector<float> predictions = nnClassifier.infer(MSAFeatures3);
                        assert(predictions.size() == MSAFeatures3.size());

                        for(size_t index = 0; index < predictions.size(); index++){
                            constexpr float threshold = 0.8;
                            const auto& msafeature = MSAFeatures3[index];

                            if(predictions[index] >= threshold){
                                isCorrected = true;

                                const int globalIndex = multipleSequenceAlignment.columnProperties.subjectColumnsBegin_incl + msafeature.position;
                                task.corrected_subject[msafeature.position] = multipleSequenceAlignment.consensus[globalIndex];
                            }
                        }

    #endif

                        if(isCorrected){
                            write_read(correctionTasks[0].readId, correctionTasks[0].corrected_subject);
                            lock(correctionTasks[0].readId);
                            (*threadOpts.readIsCorrectedVector)[correctionTasks[0].readId] = 1;
                            unlock(correctionTasks[0].readId);
                        }else{
                            //make subject available for correction as a candidate
                            if((*threadOpts.readIsCorrectedVector)[correctionTasks[0].readId] == 1){
                                lock(correctionTasks[0].readId);
                                if((*threadOpts.readIsCorrectedVector)[correctionTasks[0].readId] == 1){
                                    (*threadOpts.readIsCorrectedVector)[correctionTasks[0].readId] = 0;
                                }
                                unlock(correctionTasks[0].readId);
                            }
                        }

                    }
                }
    		} // end batch processing

            featurestream.flush();
            outputstream.flush();

            lock(0);
            std::cout << "CPU worker finished" << std::endl;


                /*for(const auto& pair : numCandidatesOfUncorrectedSubjects){
                    std::cerr << pair.first << '\t' << pair.second << '\n';
                }*/
            unlock(0);


    	}

}
}


#ifdef MSA_IMPLICIT
#undef MSA_IMPLICIT
#endif

#ifdef ENABLE_TIMING
#undef ENABLE_TIMING
#endif
