#ifndef CARE_GPU_CPU_CORRECTION_THREAD_HPP
#define CARE_GPU_CPU_CORRECTION_THREAD_HPP

#include "options.hpp"
#include "tasktiming.hpp"
#include "cpu_alignment.hpp"
#include "bestalignment.hpp"
#include "msa.hpp"
#include "qualityscoreweights.hpp"
#include "rangegenerator.hpp"
#include "featureextractor.hpp"
#include "forestclassifier.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

#include <vector>

namespace care{
namespace cpu{

    /*template<bool indels>
    struct alignment_result_type;

    template<>
    struct alignment_result_type<true>{using type = SGAResult;};

    template<>
    struct alignment_result_type<false>{using type = SHDResult;};*/

    template<class minhasher_t,
    		 class readStorage_t,
    		 bool indels = false>
    struct CPUCorrectionThread{
        static_assert(indels == false, "indels != false");

        template<class Sequence_t, class ReadId_t>
        struct CorrectionTask{
            CorrectionTask(){}

            CorrectionTask(ReadId_t readId)
                :   active(true),
                    corrected(false),
                    readId(readId)
                    {}

            CorrectionTask(const CorrectionTask& other)
                : active(other.active),
                corrected(other.corrected),
                readId(other.readId),
                subject_string(other.subject_string),
                candidate_read_ids(other.candidate_read_ids),
                candidate_read_ids_begin(other.candidate_read_ids_begin),
                candidate_read_ids_end(other.candidate_read_ids_end),
                corrected_subject(other.corrected_subject),
                corrected_candidates(other.corrected_candidates),
                corrected_candidates_read_ids(other.corrected_candidates_read_ids){

                candidate_read_ids_begin = &(candidate_read_ids[0]);
                candidate_read_ids_end = &(candidate_read_ids[candidate_read_ids.size()]);

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
                swap(l.subject_string, r.subject_string);
                swap(l.candidate_read_ids, r.candidate_read_ids);
                swap(l.candidate_read_ids_begin, r.candidate_read_ids_begin);
                swap(l.candidate_read_ids_end, r.candidate_read_ids_end);
                swap(l.corrected_subject, r.corrected_subject);
                swap(l.corrected_candidates_read_ids, r.corrected_candidates_read_ids);
            }

            bool active;
            bool corrected;
            ReadId_t readId;

            std::vector<ReadId_t> candidate_read_ids;
            ReadId_t* candidate_read_ids_begin;
            ReadId_t* candidate_read_ids_end; // exclusive

            std::string subject_string;

            std::string corrected_subject;
            std::vector<std::string> corrected_candidates;
            std::vector<ReadId_t> corrected_candidates_read_ids;
        };

        using Minhasher_t = minhasher_t;
    	using ReadStorage_t = readStorage_t;
    	using Sequence_t = typename ReadStorage_t::Sequence_t;
    	using ReadId_t = typename ReadStorage_t::ReadId_t;
        using AlignmentResult_t = SHDResult;//typename alignment_result_type<indels>::type;

        using CorrectionTask_t = CorrectionTask<Sequence_t, ReadId_t>;
        using RangeGenerator_t = RangeGenerator<ReadId_t>;

    	struct CorrectionThreadOptions{
    		int threadId;

    		std::string outputfile;
    		RangeGenerator_t* readIdGenerator;
    		const Minhasher_t* minhasher;
    		const ReadStorage_t* readStorage;
    		std::mutex* coutLock;
    		std::vector<char>* readIsProcessedVector;
    		std::vector<char>* readIsCorrectedVector;
    		std::mutex* locksForProcessedFlags;
    		std::size_t nLocksForProcessedFlags;
    	};

        AlignmentOptions alignmentOptions;
        GoodAlignmentProperties goodAlignmentProperties;
        CorrectionOptions correctionOptions;
        CorrectionThreadOptions threadOpts;
        FileOptions fileOptions;

        SequenceFileProperties fileProperties;

        std::uint64_t max_candidates;


        std::uint64_t nProcessedReads = 0;

        std::uint64_t minhashcandidates = 0;
        std::uint64_t duplicates = 0;
    	int nProcessedQueries = 0;
    	int nCorrectedCandidates = 0; // candidates which were corrected in addition to query correction.

        int avgsupportfail = 0;
    	int minsupportfail = 0;
    	int mincoveragefail = 0;
    	int sobadcouldnotcorrect = 0;
    	int verygoodalignment = 0;

        std::map<int, int> numCandidatesOfUncorrectedSubjects;

        std::chrono::duration<double> getCandidatesTimeTotal;
    	std::chrono::duration<double> mapMinhashResultsToSequencesTimeTotal;
    	std::chrono::duration<double> getAlignmentsTimeTotal;
    	std::chrono::duration<double> determinegoodalignmentsTime;
    	std::chrono::duration<double> fetchgoodcandidatesTime;
    	std::chrono::duration<double> majorityvotetime;
    	std::chrono::duration<double> basecorrectiontime;
    	std::chrono::duration<double> readcorrectionTimeTotal;
    	std::chrono::duration<double> mapminhashresultsdedup;
    	std::chrono::duration<double> mapminhashresultsfetch;
    	std::chrono::duration<double> graphbuildtime;
    	std::chrono::duration<double> graphcorrectiontime;


        std::chrono::duration<double> initIdTimeTotal;

        std::chrono::duration<double> da, db, dc;

        TaskTimings detailedCorrectionTimings;

        std::thread thread;
        bool isRunning = false;
        volatile bool stopAndAbort = false;

        void run(){
            if(isRunning) throw std::runtime_error("CPUCorrectionThread::run: Is already running.");
            isRunning = true;
            thread = std::move(std::thread(&CPUCorrectionThread::execute, this));
        }

        void join(){
            thread.join();
            isRunning = false;
        }

    private:

    	void execute() {
    		isRunning = true;

    		//std::chrono::time_point<std::chrono::system_clock> tpa, tpb, tpc, tpd;

    		std::ofstream outputstream(threadOpts.outputfile);

            std::ofstream featurestream(threadOpts.outputfile + "_features");
    		auto write_read = [&](const ReadId_t readId, const auto& sequence){
                //std::cout << readId << " " << sequence << std::endl;
    			auto& stream = outputstream;
    #if 1
    			stream << readId << ' ' << sequence << '\n';
    #else
    			stream << readId << '\n';
    			stream << sequence << '\n';
    #endif
    		};

    		auto lock = [&](ReadId_t readId){
    			ReadId_t index = readId % threadOpts.nLocksForProcessedFlags;
    			threadOpts.locksForProcessedFlags[index].lock();
    		};

    		auto unlock = [&](ReadId_t readId){
    			ReadId_t index = readId % threadOpts.nLocksForProcessedFlags;
    			threadOpts.locksForProcessedFlags[index].unlock();
    		};

            care::cpu::MultipleSequenceAlignment multipleSequenceAlignment(correctionOptions.useQualityScores,
                                                                            correctionOptions.m_coverage,
                                                                            correctionOptions.kmerlength,
                                                                            correctionOptions.estimatedCoverage,
                                                                            correctionOptions.estimatedErrorrate);

            cpu::QualityScoreConversion qualityConversion;

            std::vector<ReadId_t> readIds;

            ForestClassifier forestClassifier;
            if(!correctionOptions.classicMode){
                forestClassifier = std::move(ForestClassifier{fileOptions.forestfilename});
            }

            //std::cerr << "correctionOptions.hits_per_candidate " <<  correctionOptions.hits_per_candidate << ", max_candidates " << max_candidates << '\n';

            std::map<int, int> totalnumcandidatesmap;
            int iterasdf = 0;

    		while(!stopAndAbort && !(threadOpts.readIdGenerator->empty() && readIds.empty())){
iterasdf++;
                if(readIds.empty())
                    readIds = threadOpts.readIdGenerator->next_n(100);

                if(readIds.empty())
                    continue;

                CorrectionTask_t task(readIds.back());
                readIds.pop_back();

                bool ok = false;
                lock(task.readId);
                if ((*threadOpts.readIsCorrectedVector)[task.readId] == 0) {
                    (*threadOpts.readIsCorrectedVector)[task.readId] = 1;
                    ok = true;
                }else{
                }
                unlock(task.readId);

                if(!ok)
                    continue; //already corrected

                const char* subjectptr = threadOpts.readStorage->fetchSequenceData_ptr(task.readId);
                const int subjectLength = threadOpts.readStorage->fetchSequenceLength(task.readId);

                task.subject_string = Sequence_t::Impl_t::toString((const std::uint8_t*)subjectptr, subjectLength);
                task.candidate_read_ids = threadOpts.minhasher->getCandidates(task.subject_string, correctionOptions.hits_per_candidate, max_candidates);

                //remove our own read id from candidate list. candidate_read_ids is sorted.
                auto readIdPos = std::lower_bound(task.candidate_read_ids.begin(), task.candidate_read_ids.end(), task.readId);
                if(readIdPos != task.candidate_read_ids.end() && *readIdPos == task.readId)
                    task.candidate_read_ids.erase(readIdPos);

                std::size_t myNumCandidates = task.candidate_read_ids.size();

                //totalnumcandidatesmap[myNumCandidates]++;
                /*if(iterasdf == 1000000){
                    for(auto p : totalnumcandidatesmap){
                        std::cerr << p.first << " " << p.second << '\n';
                    }
                    break;
                }*/
                if(myNumCandidates == 0){
                    continue; //no candidates to use for correction
                }

                //std::cerr << "Read " << task.readId << ", candidates: " << myNumCandidates << '\n';

                std::vector<AlignmentResult_t> bestAlignments;
                std::vector<BestAlignment_t> bestAlignmentFlags;
                std::vector<ReadId_t> bestCandidateReadIds;
                std::vector<std::unique_ptr<std::uint8_t[]>> bestReverseComplements;

                bestAlignments.reserve(task.candidate_read_ids.size());
                bestAlignmentFlags.reserve(task.candidate_read_ids.size());
                bestCandidateReadIds.reserve(task.candidate_read_ids.size());
                bestReverseComplements.reserve(task.candidate_read_ids.size());

                std::vector<ReadId_t> discardedAlignmentsCandidateReadIds;
                discardedAlignmentsCandidateReadIds.reserve(task.candidate_read_ids.size());

                //calculate alignments
                for(const ReadId_t candidateId: task.candidate_read_ids){
                    const char* candidateptr = threadOpts.readStorage->fetchSequenceData_ptr(candidateId);
                    const int candidateLength = threadOpts.readStorage->fetchSequenceLength(candidateId);

                    std::unique_ptr<std::uint8_t[]> reverse_complement_candidate = std::make_unique<std::uint8_t[]>(Sequence_t::getNumBytes(candidateLength));

                    Sequence_t::make_reverse_complement(reverse_complement_candidate.get(), (const std::uint8_t*)candidateptr, candidateLength);

                    AlignmentResult_t forwardAlignment =
                        CPUShiftedHammingDistanceChooser<Sequence_t>::cpu_shifted_hamming_distance(subjectptr,
                                                            subjectLength,
                    										candidateptr,
                    										candidateLength,
                                                            goodAlignmentProperties.min_overlap,
                                                            goodAlignmentProperties.maxErrorRate,
                                                            goodAlignmentProperties.min_overlap_ratio);

                    AlignmentResult_t reverseComplementAlignment =
                        CPUShiftedHammingDistanceChooser<Sequence_t>::cpu_shifted_hamming_distance(subjectptr,
                                                            subjectLength,
                    										(const char*)reverse_complement_candidate.get(),
                    										candidateLength,
                                                            goodAlignmentProperties.min_overlap,
                                                            goodAlignmentProperties.maxErrorRate,
                                                            goodAlignmentProperties.min_overlap_ratio);

                    BestAlignment_t bestAlignmentFlag = care::cpu::choose_best_alignment(forwardAlignment,
                                                                              reverseComplementAlignment,
                                                                              subjectLength,
                                                                              candidateLength,
                                                                              goodAlignmentProperties.min_overlap_ratio,
                                                                              goodAlignmentProperties.min_overlap,
                                                                              goodAlignmentProperties.maxErrorRate);

                    if(bestAlignmentFlag == BestAlignment_t::Forward){
                        bestAlignments.emplace_back(forwardAlignment);
                        bestAlignmentFlags.emplace_back(bestAlignmentFlag);
                        bestCandidateReadIds.emplace_back(candidateId);
                        bestReverseComplements.emplace_back(nullptr);
                    }else if(bestAlignmentFlag == BestAlignment_t::ReverseComplement){
                        bestAlignments.emplace_back(reverseComplementAlignment);
                        bestAlignmentFlags.emplace_back(bestAlignmentFlag);
                        bestCandidateReadIds.emplace_back(candidateId);
                        bestReverseComplements.emplace_back(std::move(reverse_complement_candidate));
                    }else{
                        // discard
                        discardedAlignmentsCandidateReadIds.emplace_back(candidateId);
                    }
                }

                //bin alignments
                std::array<int, 3> counts({0,0,0});
                const float mismatchratioBaseFactor = correctionOptions.estimatedErrorrate * 1.0f;

                for(const auto& alignment : bestAlignments){
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

                if(!std::any_of(counts.begin(), counts.end(), [](auto c){return c > 0;}))
                    continue; //no good alignments

                //std::cerr << "Read " << task.readId << ", good alignments after bining: " << std::accumulate(counts.begin(), counts.end(), int(0)) << '\n';
                //std::cerr << "Read " << task.readId << ", bins: " << counts[0] << " " << counts[1] << " " << counts[2] << '\n';

                const float goodAlignmentsCountThreshold = correctionOptions.estimatedCoverage * correctionOptions.m_coverage;
                float mismatchratioThreshold = 0;
                if (counts[0] >= goodAlignmentsCountThreshold) {
                    mismatchratioThreshold = 2 * mismatchratioBaseFactor;
                } else if (counts[1] >= goodAlignmentsCountThreshold) {
                    mismatchratioThreshold = 3 * mismatchratioBaseFactor;
                } else if (counts[2] >= goodAlignmentsCountThreshold) {
                    mismatchratioThreshold = 4 * mismatchratioBaseFactor;
                } else {
                    if(correctionOptions.hits_per_candidate > 1){
                        mismatchratioThreshold = 4 * mismatchratioBaseFactor;
                    }else
                        continue;  //no correction possible
                }

                std::vector<AlignmentResult_t> removedFilteredAlignments;
                std::vector<BestAlignment_t> removedFilteredAlignmentFlags;
                std::vector<ReadId_t> removedFilteredCandidateReadIds;
                std::vector<std::unique_ptr<std::uint8_t[]>> removedFilteredReverseComplements;

                removedFilteredAlignments.reserve(bestAlignments.size());
                removedFilteredAlignmentFlags.reserve(bestAlignments.size());
                removedFilteredCandidateReadIds.reserve(bestAlignments.size());
                removedFilteredReverseComplements.reserve(bestAlignments.size());

                //filter alignments
                std::size_t newsize = 0;
                for(std::size_t i = 0; i < bestAlignments.size(); i++){
                    auto& alignment = bestAlignments[i];
                    const float mismatchratio = float(alignment.nOps) / float(alignment.overlap);
                    const bool notremoved = mismatchratio < mismatchratioThreshold;

                    if(notremoved){
                        bestAlignments[newsize] = bestAlignments[i];
                        bestAlignmentFlags[newsize] = bestAlignmentFlags[i];
                        bestCandidateReadIds[newsize] = bestCandidateReadIds[i];
                        if(newsize != i)
                            bestReverseComplements[newsize] = std::move(bestReverseComplements[i]);

                        ++newsize;
                    }else{
                        removedFilteredAlignments.emplace_back(bestAlignments[i]);
                        removedFilteredAlignmentFlags.emplace_back(bestAlignmentFlags[i]);
                        removedFilteredCandidateReadIds.emplace_back(bestCandidateReadIds[i]);
                        removedFilteredReverseComplements.emplace_back(std::move(bestReverseComplements[i]));
                    }
                }

                //std::cerr << "Read " << task.readId << ", good alignments after bin filtering: " << newsize << '\n';


                bestAlignments.resize(newsize);
                bestAlignmentFlags.resize(newsize);
                bestCandidateReadIds.resize(newsize);
                bestReverseComplements.resize(newsize);

                std::vector<int> bestCandidateLengths;
                bestCandidateLengths.reserve(newsize);
                for(const ReadId_t readId : bestCandidateReadIds)
                    bestCandidateLengths.emplace_back(threadOpts.readStorage->fetchSequenceLength(readId));

                //build multiple sequence alignment

                multipleSequenceAlignment.init(subjectLength,
                                                bestCandidateLengths,
                                                bestAlignmentFlags,
                                                bestAlignments);

                const char* subjectQualityPtr = correctionOptions.useQualityScores ? threadOpts.readStorage->fetchQuality_ptr(task.readId) : nullptr;

                multipleSequenceAlignment.insertSubject(task.subject_string, [&](int i){
                    //return qscore_to_weight[(unsigned char)(subjectQualityPtr)[i]];
                    return qualityConversion.getWeight((subjectQualityPtr)[i]);
                });

                const float desiredAlignmentMaxErrorRate = goodAlignmentProperties.maxErrorRate;

                //add candidates to multiple sequence alignment

                for(std::size_t i = 0; i < bestAlignments.size(); i++){

                    const char* candidateSequencePtr;

                    if(bestAlignmentFlags[i] == BestAlignment_t::ReverseComplement){
                        candidateSequencePtr = (const char*)bestReverseComplements[i].get();
                    }else if(bestAlignmentFlags[i] == BestAlignment_t::Forward){
                        candidateSequencePtr = threadOpts.readStorage->fetchSequenceData_ptr(bestCandidateReadIds[i]);
                    }else{
                        assert(false);
                    }

                    const int length = bestCandidateLengths[i];
                    const std::string candidateSequence = Sequence_t::Impl_t::toString((const std::uint8_t*)candidateSequencePtr, length);
                    const char* candidateQualityPtr = correctionOptions.useQualityScores ?
                                                            threadOpts.readStorage->fetchQuality_ptr(bestCandidateReadIds[i])
                                                            : nullptr;

                    const int shift = bestAlignments[i].shift;
                    const float defaultweight = 1.0f - std::sqrt(bestAlignments[i].nOps
                                                                / (bestAlignments[i].overlap
                                                                    * desiredAlignmentMaxErrorRate));

                    if(bestAlignmentFlags[i] == BestAlignment_t::ReverseComplement){
                        multipleSequenceAlignment.insertCandidate(candidateSequence, shift, [&](int i){
                            //return (float)qscore_to_weight[(unsigned char)(candidateQualityPtr)[length - 1 - i]] * defaultweight;
                            return qualityConversion.getWeight((candidateQualityPtr)[length - 1 - i]) * defaultweight;
                        });
                    }else if(bestAlignmentFlags[i] == BestAlignment_t::Forward){
                        multipleSequenceAlignment.insertCandidate(candidateSequence, shift, [&](int i){
                            //return (float)qscore_to_weight[(unsigned char)(candidateQualityPtr)[i]] * defaultweight;
                            return qualityConversion.getWeight((candidateQualityPtr)[i]) * defaultweight;
                        });
                    }else{
                        assert(false);
                    }
                }

                multipleSequenceAlignment.find_consensus();

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

                    std::cout << "ReadId " << task.readId << ": msa rows = " << msa_rows << ", columns = " << ncolumns << ", HQ-MSA: " << (isHQ ? "True" : "False")
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

                    msa2.find_consensus();
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

                            msa3.find_consensus();
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

                constexpr int max_num_minimizations = 10;

                if(max_num_minimizations > 0){
                    int num_minimizations = 1;
                    auto minimizationResult = multipleSequenceAlignment.minimize(correctionOptions.estimatedCoverage);

                    auto update_after_successfull_minimization = [&](){
                        if(minimizationResult.performedMinimization && minimizationResult.num_discarded_candidates > 0){
                            std::vector<AlignmentResult_t> bestAlignments2(minimizationResult.remaining_candidates.size());
                            std::vector<BestAlignment_t> bestAlignmentFlags2(minimizationResult.remaining_candidates.size());
                            std::vector<ReadId_t> bestCandidateReadIds2(minimizationResult.remaining_candidates.size());
                            std::vector<std::unique_ptr<std::uint8_t[]>> bestReverseComplements2(minimizationResult.remaining_candidates.size());

                            for(int i = 0; i < int(minimizationResult.remaining_candidates.size()); i++){
                                bestAlignments2[i] = bestAlignments[minimizationResult.remaining_candidates[i]];
                                bestAlignmentFlags2[i] = bestAlignmentFlags[minimizationResult.remaining_candidates[i]];
                                bestCandidateReadIds2[i] = bestCandidateReadIds[minimizationResult.remaining_candidates[i]];
                                bestReverseComplements2[i] = std::move(bestReverseComplements[minimizationResult.remaining_candidates[i]]);
                            }

                            std::swap(bestAlignments2, bestAlignments);
                            std::swap(bestAlignmentFlags2, bestAlignmentFlags);
                            std::swap(bestCandidateReadIds2, bestCandidateReadIds);
                            std::swap(bestReverseComplements2, bestReverseComplements);

                            //multipleSequenceAlignment.find_consensus();

                            //std::cout << "Minimization " << num_minimizations << ", removed " << minimizationResult.num_discarded_candidates << std::endl;
                        }
                    };

                    update_after_successfull_minimization();

                    while(num_minimizations <= max_num_minimizations
                            && minimizationResult.performedMinimization && minimizationResult.num_discarded_candidates > 0){

                        minimizationResult = multipleSequenceAlignment.minimize(correctionOptions.estimatedCoverage);
                        num_minimizations++;

                        update_after_successfull_minimization();
                    }
                }

                std::vector<MSAFeature> MSAFeatures;

                if(correctionOptions.extractFeatures || !correctionOptions.classicMode){
                    MSAFeatures = extractFeatures(multipleSequenceAlignment.consensus.data(),
                                                    multipleSequenceAlignment.support.data(),
                                                    multipleSequenceAlignment.coverage.data(),
                                                    multipleSequenceAlignment.origCoverages.data(),
                                                    multipleSequenceAlignment.columnProperties.columnsToCheck,
                                                    multipleSequenceAlignment.columnProperties.subjectColumnsBegin_incl,
                                                    multipleSequenceAlignment.columnProperties.subjectColumnsEnd_excl,
                                                    task.subject_string,
                                                    multipleSequenceAlignment.kmerlength, 0.0f,
                                                    correctionOptions.estimatedCoverage);


                }

                if(correctionOptions.extractFeatures){
                    for(const auto& msafeature : MSAFeatures){
                        featurestream << task.readId << '\t' << msafeature.position << '\n';
                        featurestream << msafeature << '\n';
                    }
                }

                if(correctionOptions.classicMode){

                    //get corrected subject and write it to file
                    auto correctionResult = multipleSequenceAlignment.getCorrectedSubject();

                    /*if(!correctionResult.isCorrected || correctionResult.correctedSequence == task.subject_string){
                        const std::size_t numCandidates = task.candidate_read_ids.size();
                        numCandidatesOfUncorrectedSubjects[numCandidates]++;
                    }*/

                    if(correctionResult.isCorrected){
                        write_read(task.readId, correctionResult.correctedSequence);
                        lock(task.readId);
                        (*threadOpts.readIsCorrectedVector)[task.readId] = 1;
                        unlock(task.readId);
                    }else{

                        //make subject available for correction as a candidate
                        if((*threadOpts.readIsCorrectedVector)[task.readId] == 1){
                            lock(task.readId);
                            if((*threadOpts.readIsCorrectedVector)[task.readId] == 1){
                                (*threadOpts.readIsCorrectedVector)[task.readId] = 0;
                            }
                            unlock(task.readId);
                        }
                    }

                    //get corrected candidates and write them to file
                    if(correctionOptions.correctCandidates && correctionResult.msaProperties.isHQ){
                        auto correctedCandidates = multipleSequenceAlignment.getCorrectedCandidates(bestCandidateLengths,
                                                                            bestAlignments,
                                                                            correctionOptions.new_columns_to_correct);

                        for(const auto& correctedCandidate : correctedCandidates){
                            const ReadId_t candidateId = bestCandidateReadIds[correctedCandidate.index];
                            bool savingIsOk = false;
                            if((*threadOpts.readIsCorrectedVector)[candidateId] == 0){
                                lock(candidateId);
                                if((*threadOpts.readIsCorrectedVector)[candidateId]== 0) {
                                    (*threadOpts.readIsCorrectedVector)[candidateId] = 1; // we will process this read
                                    savingIsOk = true;
                                }
                                unlock(candidateId);
                            }
                            if (savingIsOk) {
                                if(bestAlignmentFlags[correctedCandidate.index] == BestAlignment_t::Forward){
                                    write_read(candidateId, correctedCandidate.sequence);
                                }else{
                                    const std::string fwd = SequenceString(correctedCandidate.sequence).reverseComplement().toString();
                                    write_read(candidateId, fwd);
                                }
                            }
                        }
                    }

                }else{

                    std::string corrected_subject = task.subject_string;
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

                            const int globalIndex = multipleSequenceAlignment.columnProperties.subjectColumnsBegin_incl + msafeature.position;
                            corrected_subject[msafeature.position] = multipleSequenceAlignment.consensus[globalIndex];
                        }
                    }

                    if(isCorrected){
                        write_read(task.readId, corrected_subject);
                        lock(task.readId);
                        (*threadOpts.readIsCorrectedVector)[task.readId] = 1;
                        unlock(task.readId);
                    }else{
                        //make subject available for correction as a candidate
                        if((*threadOpts.readIsCorrectedVector)[task.readId] == 1){
                            lock(task.readId);
                            if((*threadOpts.readIsCorrectedVector)[task.readId] == 1){
                                (*threadOpts.readIsCorrectedVector)[task.readId] = 0;
                            }
                            unlock(task.readId);
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
    };

}
}


#endif
