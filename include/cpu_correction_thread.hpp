#ifndef CARE_GPU_CPU_CORRECTION_THREAD_HPP
#define CARE_GPU_CPU_CORRECTION_THREAD_HPP

#include <config.hpp>

#include "options.hpp"
#include "tasktiming.hpp"
#include "cpu_alignment.hpp"
#include "bestalignment.hpp"
#include "msa.hpp"
#include "qualityscoreweights.hpp"
#include "rangegenerator.hpp"
#include "featureextractor.hpp"
#include "forestclassifier.hpp"

#include "cpu_correction_core.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

#include <vector>

#define MSA_IMPLICIT

#define ENABLE_TIMING

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

        template<class Sequence_t>
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
                original_subject_string(other.original_subject_string),
                subject_string(other.subject_string),
                candidate_read_ids(other.candidate_read_ids),
                candidate_read_ids_begin(other.candidate_read_ids_begin),
                candidate_read_ids_end(other.candidate_read_ids_end),
                clipping_begin(other.clipping_begin),
                clipping_end(other.clipping_end),
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
                swap(l.original_subject_string, r.original_subject_string);
                swap(l.subject_string, r.subject_string);
                swap(l.candidate_read_ids, r.candidate_read_ids);
                swap(l.candidate_read_ids_begin, r.candidate_read_ids_begin);
                swap(l.candidate_read_ids_end, r.candidate_read_ids_end);
                swap(l.clipping_begin, r.clipping_begin);
                swap(l.clipping_end, r.clipping_end);
                swap(l.corrected_subject, r.corrected_subject);
                swap(l.corrected_candidates_read_ids, r.corrected_candidates_read_ids);
            }

            bool active;
            bool corrected;
            read_number readId;

            std::vector<read_number> candidate_read_ids;
            read_number* candidate_read_ids_begin;
            read_number* candidate_read_ids_end; // exclusive

            std::string original_subject_string;
            std::string subject_string;

            int clipping_begin;
            int clipping_end;

            std::string corrected_subject;
            std::vector<std::string> corrected_candidates;
            std::vector<read_number> corrected_candidates_read_ids;
        };

        using Minhasher_t = minhasher_t;
        using ReadStorage_t = cpu::ContiguousReadStorage;
    	using Sequence_t = typename ReadStorage_t::Sequence_t;
        using AlignmentResult_t = SHDResult;//typename alignment_result_type<indels>::type;

        using CorrectionTask_t = CorrectionTask<Sequence_t>;
        using RangeGenerator_t = RangeGenerator<read_number>;

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
        std::chrono::duration<double> copyCandidateDataToBufferTimeTotal;
    	std::chrono::duration<double> getAlignmentsTimeTotal;
        std::chrono::duration<double> findBestAlignmentDirectionTimeTotal;
        std::chrono::duration<double> gatherBestAlignmentDataTimeTotal;
        std::chrono::duration<double> mismatchRatioFilteringTimeTotal;
        std::chrono::duration<double> compactBestAlignmentDataTimeTotal;
        std::chrono::duration<double> fetchCandidateStringsAndQualitiesTimeTotal;
        std::chrono::duration<double> msaAddSequencesTimeTotal;
        std::chrono::duration<double> msaFindConsensusTimeTotal;
        std::chrono::duration<double> msaMinimizationTimeTimeTotal;
        std::chrono::duration<double> msaCorrectSubjectTimeTimeTotal;
        std::chrono::duration<double> msaCorrectCandidatesTimeTimeTotal;




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
#ifndef MSA_IMPLICIT
            using MSA_t = care::cpu::MultipleSequenceAlignment;
#else
            using MSA_t = care::cpu::MultipleSequenceAlignmentImplicit;
#endif
    		isRunning = true;

            const int max_sequence_bytes = Sequence_t::getNumBytes(fileOptions.maximum_sequence_length);

    		//std::chrono::time_point<std::chrono::system_clock> tpa, tpb, tpc, tpd;

    		std::ofstream outputstream(threadOpts.outputfile);

            std::ofstream featurestream(threadOpts.outputfile + "_features");
    		auto write_read = [&](const read_number readId, const auto& sequence){
                //std::cout << readId << " " << sequence << std::endl;
    			auto& stream = outputstream;

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

            MSA_t multipleSequenceAlignment(correctionOptions.useQualityScores,
                                            correctionOptions.m_coverage,
                                            correctionOptions.kmerlength,
                                            correctionOptions.estimatedCoverage,
                                            correctionOptions.estimatedErrorrate);

            cpu::QualityScoreConversion qualityConversion;

            std::vector<read_number> readIds;

            ForestClassifier forestClassifier;
            if(!correctionOptions.classicMode){
                forestClassifier = std::move(ForestClassifier{fileOptions.forestfilename});
            }

            //std::cerr << "correctionOptions.hits_per_candidate " <<  correctionOptions.hits_per_candidate << ", max_candidates " << max_candidates << '\n';

            std::map<int, int> totalnumcandidatesmap;

    		while(!stopAndAbort && !(threadOpts.readIdGenerator->empty() && readIds.empty())){

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

                const char* originalsubjectptr = threadOpts.readStorage->fetchSequenceData_ptr(task.readId);
                const int originalsubjectLength = threadOpts.readStorage->fetchSequenceLength(task.readId);

                task.original_subject_string = Sequence_t::Impl_t::toString((const std::uint8_t*)originalsubjectptr, originalsubjectLength);
                task.subject_string = task.original_subject_string;
                task.clipping_begin = 0;
                task.clipping_end = originalsubjectLength;



                // -----

                bool needsSecondPassAfterClipping = false;
                bool discardThisTask = false;
                Sequence_t subjectsequence;


                std::vector<char> candidateData;
                std::vector<char> candidateRevcData;
                std::vector<int> candidateLengths;
                int max_candidate_length = 0;
                std::vector<char*> candidateDataPtrs;
                std::vector<char*> candidateRevcDataPtrs;

                std::vector<AlignmentResult_t> bestAlignments;
                std::vector<BestAlignment_t> bestAlignmentFlags;
                std::vector<read_number> bestCandidateReadIds;
                std::vector<int> bestCandidateLengths;
                std::vector<char> bestCandidateData;
                std::vector<char*> bestCandidatePtrs;

                std::vector<char> bestCandidateQualityData;
                std::vector<char*> bestCandidateQualityPtrs;
                std::vector<std::string> bestCandidateStrings;

                //this loop allows a second pass after subject has been clipped
                int clippingIters = 0;

                do{

                    clippingIters++;

                    const char* subjectptr = originalsubjectptr;
                    int subjectLength = originalsubjectLength;
                    if(needsSecondPassAfterClipping){
                        subjectsequence = std::move(Sequence_t{task.subject_string});
                        subjectptr = (const char*)subjectsequence.begin();
                        subjectLength = subjectsequence.length();
                    }

                    if(clippingIters > 1){
                        //std::cout << "before: " << task.candidate_read_ids.size() << " candidates\n";
                    }

#ifdef ENABLE_TIMING
                    auto tpa = std::chrono::system_clock::now();
#endif

                    const int candidate_limit = clippingIters > 1 ? std::numeric_limits<int>::max() : max_candidates;
                    task.candidate_read_ids = threadOpts.minhasher->getCandidates(task.subject_string, correctionOptions.hits_per_candidate, candidate_limit);

#ifdef ENABLE_TIMING
                    getCandidatesTimeTotal += std::chrono::system_clock::now() - tpa;
#endif

                    if(clippingIters > 1){
                        //std::cout << "after: " << task.candidate_read_ids.size() << " candidates\n";
                    }

                    //remove our own read id from candidate list. candidate_read_ids is sorted.
                    auto readIdPos = std::lower_bound(task.candidate_read_ids.begin(), task.candidate_read_ids.end(), task.readId);
                    if(readIdPos != task.candidate_read_ids.end() && *readIdPos == task.readId)
                        task.candidate_read_ids.erase(readIdPos);

                    int myNumCandidates = int(task.candidate_read_ids.size());

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
                    for(const read_number candidateId: task.candidate_read_ids){
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

                    for(int i = 0; i < myNumCandidates; i++){
                        const read_number candidateId = task.candidate_read_ids[i];
                        const char* candidateptr = threadOpts.readStorage->fetchSequenceData_ptr(candidateId);
                        const int candidateLength = candidateLengths[i];
                        const int bytes = Sequence_t::getNumBytes(candidateLength);

                        char* const candidateDataBegin = candidateData.data() + i * max_sequence_bytes;
                        char* const candidateRevcDataBegin = candidateRevcData.data() + i * max_sequence_bytes;

                        std::copy(candidateptr, candidateptr + bytes, candidateDataBegin);
                        Sequence_t::make_reverse_complement(reinterpret_cast<std::uint8_t*>(candidateRevcDataBegin),
                                                            reinterpret_cast<const std::uint8_t*>(candidateptr),
                                                            candidateLength);

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
#if 0
                    auto forwardAlignments = calculate_shd_alignments<Sequence_t>(subjectptr,
                                                                subjectLength,
                                                                candidateDataPtrs,
                                                                candidateLengths,
                                                                goodAlignmentProperties.min_overlap,
                                                                goodAlignmentProperties.maxErrorRate,
                                                                goodAlignmentProperties.min_overlap_ratio);
                    auto revcAlignments = calculate_shd_alignments<Sequence_t>(subjectptr,
                                                                subjectLength,
                                                                candidateRevcDataPtrs,
                                                                candidateLengths,
                                                                goodAlignmentProperties.min_overlap,
                                                                goodAlignmentProperties.maxErrorRate,
                                                                goodAlignmentProperties.min_overlap_ratio);

#else

                    auto forwardAlignments = calculate_shd_alignments<Sequence_t>(subjectptr,
                                                                subjectLength,
                                                                candidateData,
                                                                candidateLengths,
                                                                max_sequence_bytes,
                                                                goodAlignmentProperties.min_overlap,
                                                                goodAlignmentProperties.maxErrorRate,
                                                                goodAlignmentProperties.min_overlap_ratio);
                    auto revcAlignments = calculate_shd_alignments<Sequence_t>(subjectptr,
                                                                subjectLength,
                                                                candidateRevcData,
                                                                candidateLengths,
                                                                max_sequence_bytes,
                                                                goodAlignmentProperties.min_overlap,
                                                                goodAlignmentProperties.maxErrorRate,
                                                                goodAlignmentProperties.min_overlap_ratio);
#endif

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
                        const read_number candidateId = task.candidate_read_ids[i];
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
                    }


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

                        for(int i = 0; i < int(bestAlignments.size()); i++){
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

                    //decode sequences of best alignments
                    bestCandidateStrings.clear();
                    bestCandidateStrings.reserve(bestAlignments.size());

                    for(int i = 0; i < int(bestAlignments.size()); i++){
                        const char* ptr = bestCandidatePtrs[i];
                        const int length = bestCandidateLengths[i];
                        bestCandidateStrings.emplace_back(Sequence_t::Impl_t::toString((const std::uint8_t*)ptr, length));
                    }

#ifdef ENABLE_TIMING
                   fetchCandidateStringsAndQualitiesTimeTotal += std::chrono::system_clock::now() - tpa;
#endif



#ifdef ENABLE_TIMING
                    tpa = std::chrono::system_clock::now();
#endif

                    //build multiple sequence alignment

                    multipleSequenceAlignment.init(subjectLength,
                                                    bestCandidateLengths,
                                                    bestAlignmentFlags,
                                                    bestAlignments);

                    const char* subjectQualityPtr = correctionOptions.useQualityScores ? threadOpts.readStorage->fetchQuality_ptr(task.readId) : nullptr;

                    multipleSequenceAlignment.insertSubject(task.subject_string, [&](int i){
                        return qualityConversion.getWeight((subjectQualityPtr)[i]);
                    });

                    const float desiredAlignmentMaxErrorRate = goodAlignmentProperties.maxErrorRate;

                    //add candidates to multiple sequence alignment

                    std::vector<std::function<float(int)>> candidateQualityConversionFunctions;
                    candidateQualityConversionFunctions.reserve(bestAlignments.size());


                    for(std::size_t i = 0; i < bestAlignments.size(); i++){

                        const int length = bestCandidateLengths[i];
                        const std::string& candidateSequence = bestCandidateStrings[i];
                        const char* candidateQualityPtr = correctionOptions.useQualityScores ?
                                                                bestCandidateQualityPtrs[i]
                                                                : nullptr;

                        const int shift = bestAlignments[i].shift;
                        const float defaultweight = 1.0f - std::sqrt(bestAlignments[i].nOps
                                                                    / (bestAlignments[i].overlap
                                                                        * desiredAlignmentMaxErrorRate));

                        if(bestAlignmentFlags[i] == BestAlignment_t::ReverseComplement){
                            auto conversionFunction = [&, candidateQualityPtr, defaultweight, length](int i){
                                return qualityConversion.getWeight((candidateQualityPtr)[length - 1 - i]) * defaultweight;
                            };

                            multipleSequenceAlignment.insertCandidate(candidateSequence, shift, conversionFunction);

                            candidateQualityConversionFunctions.emplace_back(std::move(conversionFunction));
                        }else if(bestAlignmentFlags[i] == BestAlignment_t::Forward){
                            auto conversionFunction = [&, candidateQualityPtr, defaultweight](int i){
                                return qualityConversion.getWeight((candidateQualityPtr)[i]) * defaultweight;
                            };
                            multipleSequenceAlignment.insertCandidate(candidateSequence, shift, conversionFunction);

                            candidateQualityConversionFunctions.emplace_back(std::move(conversionFunction));
                        }else{
                            assert(false);
                        }

                    }

#ifdef ENABLE_TIMING
                    msaAddSequencesTimeTotal += std::chrono::system_clock::now() - tpa;
#endif

#ifdef ENABLE_TIMING
                    tpa = std::chrono::system_clock::now();
#endif

                    multipleSequenceAlignment.find_consensus();

#ifdef ENABLE_TIMING
                    msaFindConsensusTimeTotal += std::chrono::system_clock::now() - tpa;
#endif

    /*
                    auto goodregion = multipleSequenceAlignment.findGoodConsensusRegionOfSubject();

                    if(goodregion.first > 0 || goodregion.second < int(task.subject_string.size())){
                        const int negativeShifts = std::count_if(multipleSequenceAlignment.shifts.begin(),
                                                                multipleSequenceAlignment.shifts.end(),
                                                                [](int s){return s < 0;});
                        const int positiveShifts = std::count_if(multipleSequenceAlignment.shifts.begin(),
                                                                multipleSequenceAlignment.shifts.end(),
                                                                [](int s){return s > 0;});

                        std::cout << "ReadId " << task.readId << " : [" << goodregion.first << ", "
                                    << goodregion.second << "] negativeShifts " << negativeShifts
                                    << ", positiveShifts " << positiveShifts
                                    << ". Subject starts at column "
                                    << multipleSequenceAlignment.columnProperties.subjectColumnsBegin_incl
                                    << ". Subject ends at column "
                                    << multipleSequenceAlignment.columnProperties.subjectColumnsEnd_excl
                                    << " / " << multipleSequenceAlignment.nColumns << "\n";
                        for(int k = 0; k < goodregion.first; k++){
                            std::cout << task.subject_string[k];
                        }
                        std::cout << "  ";
                        for(int k = goodregion.first; k < goodregion.second; k++){
                            std::cout << task.subject_string[k];
                        }
                        std::cout << "  ";
                        for(int k = goodregion.second; k < int(task.subject_string.size()); k++){
                            std::cout << task.subject_string[k];
                        }
                        std::cout << '\n';

                        for(int k = 0; k < goodregion.first; k++){
                            std::cout << multipleSequenceAlignment.consensus[k + multipleSequenceAlignment.columnProperties.subjectColumnsBegin_incl];
                        }
                        std::cout << "  ";
                        for(int k = goodregion.first; k < goodregion.second; k++){
                            std::cout << multipleSequenceAlignment.consensus[k + multipleSequenceAlignment.columnProperties.subjectColumnsBegin_incl];
                        }
                        std::cout << "  ";
                        for(int k = goodregion.second; k < int(task.subject_string.size()); k++){
                            std::cout << multipleSequenceAlignment.consensus[k + multipleSequenceAlignment.columnProperties.subjectColumnsBegin_incl];
                        }
                        std::cout << '\n';
                    }
    */

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
    #if 0
#ifdef ENABLE_TIMING
                    tpa = std::chrono::system_clock::now();
#endif

                    constexpr int max_num_minimizations = 5;

                    if(max_num_minimizations > 0){
                        int num_minimizations = 1;
    #ifndef MSA_IMPLICIT
                        auto minimizationResult = multipleSequenceAlignment.minimize(correctionOptions.estimatedCoverage);
    #else


                        auto minimizationResult = multipleSequenceAlignment.minimize_implicit(
                                                            bestCandidateStrings,
                                                            correctionOptions.estimatedCoverage,
                                                            candidateQualityConversionFunctions);
    #endif
                        auto update_after_successfull_minimization = [&](){
                            if(minimizationResult.performedMinimization && minimizationResult.num_discarded_candidates > 0){
                                std::vector<AlignmentResult_t> bestAlignments2(minimizationResult.remaining_candidates.size());
                                std::vector<BestAlignment_t> bestAlignmentFlags2(minimizationResult.remaining_candidates.size());
                                std::vector<read_number> bestCandidateReadIds2(minimizationResult.remaining_candidates.size());
                                std::vector<std::unique_ptr<std::uint8_t[]>> bestReverseComplements2(minimizationResult.remaining_candidates.size());

                                std::vector<std::string> candidateStrings2(minimizationResult.remaining_candidates.size());
                                std::vector<std::function<float(int)>> candidateQualityConversionFunctions2(minimizationResult.remaining_candidates.size());

                                for(int i = 0; i < int(minimizationResult.remaining_candidates.size()); i++){
                                    const int remaining_index = minimizationResult.remaining_candidates[i];
                                    bestAlignments2[i] = bestAlignments[remaining_index];
                                    bestAlignmentFlags2[i] = bestAlignmentFlags[remaining_index];
                                    bestCandidateReadIds2[i] = bestCandidateReadIds[remaining_index];
                                    bestReverseComplements2[i] = std::move(bestReverseComplements[remaining_index]);

                                    candidateStrings2[i] = std::move(candidateStrings[remaining_index]);
                                    candidateQualityConversionFunctions2[i] = std::move(candidateQualityConversionFunctions[remaining_index]);
                                }

                                std::swap(bestAlignments2, bestAlignments);
                                std::swap(bestAlignmentFlags2, bestAlignmentFlags);
                                std::swap(bestCandidateReadIds2, bestCandidateReadIds);
                                std::swap(bestReverseComplements2, bestReverseComplements);
    #ifdef MSA_IMPLICIT
                                std::swap(candidateStrings2, candidateStrings);
                                std::swap(candidateQualityConversionFunctions2, candidateQualityConversionFunctions);
    #endif

                                //multipleSequenceAlignment.find_consensus();

                                //std::cout << "Minimization " << num_minimizations << ", removed " << minimizationResult.num_discarded_candidates << std::endl;
                            }
                        };

                        update_after_successfull_minimization();

                        while(num_minimizations <= max_num_minimizations
                                && minimizationResult.performedMinimization && minimizationResult.num_discarded_candidates > 0){

    #ifndef MSA_IMPLICIT
                            minimizationResult = multipleSequenceAlignment.minimize(correctionOptions.estimatedCoverage);
    #else


                            minimizationResult = multipleSequenceAlignment.minimize_implicit(
                                                                candidateStrings,
                                                                correctionOptions.estimatedCoverage,
                                                                candidateQualityConversionFunctions);
    #endif
                            num_minimizations++;

                            update_after_successfull_minimization();
                        }
                    }

#ifdef ENABLE_TIMING
                    msaMinimizationTimeTimeTotal += std::chrono::system_clock::now() - tpa;
#endif
    #endif




                    //minimization is finished here
#if 1
                    if(!needsSecondPassAfterClipping){
                        auto goodregion = multipleSequenceAlignment.findGoodConsensusRegionOfSubject();

                        if(goodregion.first > 0 || goodregion.second < int(task.subject_string.size())){
                            /*const int negativeShifts = std::count_if(multipleSequenceAlignment.shifts.begin(),
                                                                    multipleSequenceAlignment.shifts.end(),
                                                                    [](int s){return s < 0;});
                            const int positiveShifts = std::count_if(multipleSequenceAlignment.shifts.begin(),
                                                                    multipleSequenceAlignment.shifts.end(),
                                                                    [](int s){return s > 0;});

                            std::cout << "ReadId " << task.readId << " : [" << goodregion.first << ", "
                                        << goodregion.second << "] negativeShifts " << negativeShifts
                                        << ", positiveShifts " << positiveShifts
                                        << ". Subject starts at column "
                                        << multipleSequenceAlignment.columnProperties.subjectColumnsBegin_incl
                                        << ". Subject ends at column "
                                        << multipleSequenceAlignment.columnProperties.subjectColumnsEnd_excl
                                        << " / " << multipleSequenceAlignment.nColumns << "\n";*/

                            needsSecondPassAfterClipping = true;

                            task.clipping_begin = goodregion.first;
                            task.clipping_end = goodregion.second;
                            const int clipsize = task.clipping_end - task.clipping_begin;
                            task.subject_string = task.original_subject_string.substr(task.clipping_begin, clipsize);
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











                std::vector<MSAFeature> MSAFeatures;

                if(correctionOptions.extractFeatures || !correctionOptions.classicMode){
#if 0
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
#else

#if 0
                std::vector<MSAFeature3> MSAFeatures3 = extractFeatures3(
                                            multipleSequenceAlignment.multiple_sequence_alignment.data(),
                                            multipleSequenceAlignment.multiple_sequence_alignment_weights.data(),
                                            multipleSequenceAlignment.nRows,
                                            multipleSequenceAlignment.columnProperties.columnsToCheck,
                                            correctionOptions.useQualityScores,
                                            multipleSequenceAlignment.consensus.data(),
                                            multipleSequenceAlignment.support.data(),
                                            multipleSequenceAlignment.coverage.data(),
                                            multipleSequenceAlignment.origCoverages.data(),
                                            multipleSequenceAlignment.columnProperties.subjectColumnsBegin_incl,
                                            multipleSequenceAlignment.columnProperties.subjectColumnsEnd_excl,
                                            task.subject_string,
                                            correctionOptions.estimatedCoverage,
                                            false);
#else
                std::vector<MSAFeature3> MSAFeatures3 = extractFeatures3_2(
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
#endif

                    if(correctionOptions.extractFeatures){
                        for(const auto& msafeature : MSAFeatures3){
                            featurestream << task.readId << '\t' << msafeature.position << '\n';
                            featurestream << msafeature << '\n';
                        }
                    }
#endif
                }

                if(correctionOptions.extractFeatures){
                    for(const auto& msafeature : MSAFeatures){
                        featurestream << task.readId << '\t' << msafeature.position << '\n';
                        featurestream << msafeature << '\n';
                    }
                }

                if(correctionOptions.classicMode){

#ifdef ENABLE_TIMING
                    auto tpa = std::chrono::system_clock::now();
#endif
                    //get corrected subject and write it to file

                    auto correctionResult = multipleSequenceAlignment.getCorrectedSubject();

#ifdef ENABLE_TIMING
                    msaCorrectSubjectTimeTimeTotal += std::chrono::system_clock::now() - tpa;
#endif

                    if(correctionResult.isCorrected){
                        //need to replace the bases in the good region by the corrected bases of the clipped read
                        if(clippingIters == 1){
                            task.corrected_subject = correctionResult.correctedSequence;
                        }else{
                            task.corrected_subject = task.original_subject_string;
                            std::copy(correctionResult.correctedSequence.begin(),
                                      correctionResult.correctedSequence.end(),
                                      task.corrected_subject.begin() + task.clipping_begin);
                        }
                    }




                    /*if(!correctionResult.isCorrected || correctionResult.correctedSequence == task.original_subject_string){
                        const std::size_t numCandidates = task.candidate_read_ids.size();
                        numCandidatesOfUncorrectedSubjects[numCandidates]++;
                    }*/

                    if(correctionResult.isCorrected){

                        write_read(task.readId, task.corrected_subject);
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
#ifdef ENABLE_TIMING
                        tpa = std::chrono::system_clock::now();
#endif

                        auto correctedCandidates = multipleSequenceAlignment.getCorrectedCandidates(bestCandidateLengths,
                                                                            bestAlignments,
                                                                            correctionOptions.new_columns_to_correct);

#ifdef ENABLE_TIMING
                        msaCorrectCandidatesTimeTimeTotal += std::chrono::system_clock::now() - tpa;
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

                    task.corrected_subject = task.subject_string;
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
                            task.corrected_subject[msafeature.position] = multipleSequenceAlignment.consensus[globalIndex];
                        }
                    }

                    if(isCorrected){
                        write_read(task.readId, task.corrected_subject);
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


#ifdef MSA_IMPLICIT
#undef MSA_IMPLICIT
#endif

#ifdef ENABLE_TIMING
#undef ENABLE_TIMING
#endif


#endif
