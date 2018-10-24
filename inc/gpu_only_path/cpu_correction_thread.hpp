#ifndef CARE_GPU_CPU_CORRECTION_THREAD_HPP
#define CARE_GPU_CPU_CORRECTION_THREAD_HPP

#include "../options.hpp"
#include "../batchelem.hpp"
#include "../tasktiming.hpp"
#include "../cpu_alignment.hpp"
#include "../multiple_sequence_alignment.hpp"
#include "../bestalignment.hpp"
#include "../msa.hpp"
#include "../qualityscoreweights.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

#include <vector>

namespace care{
namespace gpu{

    template<bool indels>
    struct alignment_result_type;

    template<>
    struct alignment_result_type<true>{using type = SGAResult;};

    template<>
    struct alignment_result_type<false>{using type = SHDResult;};

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
        using AlignmentResult_t = typename alignment_result_type<indels>::type;
        using BatchElem_t = BatchElem<ReadStorage_t, AlignmentResult_t>;
        using CorrectionTask_t = CorrectionTask<Sequence_t, ReadId_t>;

    	struct CorrectionThreadOptions{
    		int threadId;
    		int deviceId;
    		int gpuThresholdSHD;
    		int gpuThresholdSGA;
            bool canUseGpu = false;

    		std::string outputfile;
    		BatchGenerator<ReadId_t>* batchGen;
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

        SequenceFileProperties fileProperties;

        std::uint64_t max_candidates;


        std::uint64_t nProcessedReads = 0;

        DetermineGoodAlignmentStats goodAlignmentStats;
        std::uint64_t minhashcandidates = 0;
        std::uint64_t duplicates = 0;
    	int nProcessedQueries = 0;
    	int nCorrectedCandidates = 0; // candidates which were corrected in addition to query correction.

        int avgsupportfail = 0;
    	int minsupportfail = 0;
    	int mincoveragefail = 0;
    	int sobadcouldnotcorrect = 0;
    	int verygoodalignment = 0;

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

    		std::chrono::time_point<std::chrono::system_clock> tpa, tpb, tpc, tpd;

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

            std::uint64_t cpuAlignments = 0;
            std::uint64_t gpuAlignments = 0;
            //std::uint64_t savedAlignments = 0;
            //std::uint64_t performedAlignments = 0;

    	//	constexpr int nStreams = 2;
        //    const bool canUseGpu = threadOpts.canUseGpu;

/*    		std::vector<SHDhandle> shdhandles(nStreams);
    		std::vector<SGAhandle> sgahandles(nStreams);

			for(auto& handle : shdhandles){
				init_SHDhandle(handle,
						threadOpts.deviceId,
						fileProperties.maxSequenceLength,
						Sequence_t::getNumBytes(fileProperties.maxSequenceLength),
						threadOpts.gpuThresholdSHD);
			}*/

            care::cpu::MultipleSequenceAlignment multipleSequenceAlignment(correctionOptions.useQualityScores,
                                                                            correctionOptions.m_coverage,
                                                                            correctionOptions.kmerlength,
                                                                            correctionOptions.estimatedCoverage);

            std::vector<ReadId_t> readIds;

    		while(!stopAndAbort && !(mybatchgen->empty() && readIds.empty())){

                if(readIds.empty())
                    readIds = mybatchgen->getNextReadIds(1000);

                if(readIds.empty())
                    continue;

                CorrectionTask_t task(readIds.back());
                readIds.pop_back();

                bool ok = false;
                lock(task.readId);
                if ((*transFuncData.readIsCorrectedVector)[task.readId] == 0) {
                    (*transFuncData.readIsCorrectedVector)[task.readId] = 1;
                    ok = true;
                }else{
                }
                unlock(task.readId);

                if(!ok)
                    continue; //already corrected

                const char* subjectptr = gpuReadStorage->fetchSequence_ptr(id);
                const int subjectLength = gpuReadStorage->fetchSequenceLength(id);

                task.subject_string = Sequence_t::Impl_t::toString((const std::uint8_t*)subjectptr, subjectLength);
                task.candidate_read_ids = *threadOpts.minhasher->getCandidates(task.subject_string, max_candidates);

                //remove our own read id from candidate list. candidate_read_ids is sorted.
                auto readIdPos = std::lower_bound(task.candidate_read_ids.begin(), task.candidate_read_ids.end(), task.readId);
                if(readIdPos != task.candidate_read_ids.end() && *readIdPos == task.readId)
                    task.candidate_read_ids.erase(readIdPos);

                std::size_t myNumCandidates = task.candidate_read_ids.size();

                if(myNumCandidates == 0){
                    continue; //no candidates to use for correction
                }


                std::vector<AlignmentResult_t> bestAlignments;
                std::vector<BestAlignment_t> bestAlignmentFlags;
                std::vector<ReadId_t> bestCandidateReadIds;
                std::vector<std::unique_ptr<std::uint8_t[]>> bestReverseComplements;

                bestAlignments.reserve(task.candidate_read_ids.size());
                bestAlignmentFlags.reserve(task.candidate_read_ids.size());
                bestCandidateReadIds.reserve(task.candidate_read_ids.size());
                bestReverseComplements.reserve(task.candidate_read_ids.size());

                //calculate alignments
                for(const ReadId_t candidateId: task.candidate_read_ids){
                    const char* candidateptr = gpuReadStorage->fetchSequence_ptr(candidateId);
                    const int candidateLength = gpuReadStorage->fetchSequenceLength(candidateId);

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

                    BestAlignment_t bestAlignmentFlag = choose_best_alignment(forwardAlignment,
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
                        ; // discard
                    }
                }

                //bin alignments
                std::array<int, 3> counts({0,0,0});
                const double mismatchratioBaseFactor = correctionOptions.estimatedErrorrate * 1.0;

                for(const auto& alignment : bestAlignments){
                    const double mismatchratio = double(alignment.nOps) / double(alignment.overlap);

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

                const double goodAlignmentsCountThreshold = correctionOptions.estimatedCoverage * correctionOptions.m_coverage;
                double mismatchratioThreshold = 0;
                if (counts[0] >= goodAlignmentsCountThreshold) {
                    mismatchratioThreshold = 2 * mismatchratioBaseFactor;
                } else if (counts[1] >= goodAlignmentsCountThreshold) {
                    mismatchratioThreshold = 3 * mismatchratioBaseFactor;
                } else if (counts[2] >= goodAlignmentsCountThreshold) {
                    mismatchratioThreshold = 4 * mismatchratioBaseFactor;
                } else {
                    continue;  //no correction possible
                }

                //filter alignments
                std::size_t newsize = 0;
                for(std::size_t i = 0; i < bestAlignments.size(); i++){
                    auto& alignment = bestAlignments[i];
                    const double mismatchratio = double(alignment.nOps) / double(alignment.overlap);
                    const bool notremoved = mismatchratio < mismatchratioThreshold;

                    if(notremoved){
                        bestAlignments[newsize] = bestAlignments[i];
                        bestAlignmentFlags[newsize] = bestAlignmentFlags[i];
                        bestCandidateReadIds[newsize] = bestCandidateReadIds[i];

                        ++newsize;
                    }
                }

                bestAlignments.resize(newsize);
                bestAlignmentFlags.resize(newsize);
                bestCandidateReadIds.resize(newsize);

                std::vector<int> bestCandidateLengths();
                bestCandidateLengths.reserve(newsize);
                for(const ReadId_t readId : bestCandidateReadIds)
                    bestCandidateLengths.emplace_back(gpuReadStorage->fetchSequenceLength(readId));

                //build multiple sequence alignment

                multipleSequenceAlignment.init(subjectLength,
                                                bestCandidateLengths,
                                                bestAlignmentFlags,
                                                bestAlignments);

                const std::string* subjectQualityPtr = useQualityScores ? gpuReadStorage->fetchQuality_ptr(task.readId) : nullptr;

                multipleSequenceAlignment.insertSubject(task.subject_string, [&](int i){
                    return qscore_to_weight[(unsigned char)(*subjectQualityPtr)[i]];
                });

                const float desiredAlignmentMaxErrorRate = goodAlignmentProperties.maxErrorRate;

                for(std::size_t i = 0; i < bestAlignments.size(); i++){

                    const char* candidateSequencePtr;

                    if(bestAlignmentFlags[i] == BestAlignment_t::ReverseComplement){
                        candidateSequencePtr = (const char*)bestReverseComplements[i].get();
                    }else if(bestAlignmentFlags[i] == BestAlignment_t::Forward){
                        candidateSequencePtr = gpuReadStorage->fetchSequence_ptr(bestCandidateReadIds[i]);
                    }else{
                        assert(false);
                    }

                    const int length = bestCandidateLengths[i];
                    const std::string candidateSequence = Sequence_t::Impl_t::toString((const std::uint8_t*)candidateSequencePtr, length);
                    const std::string* candidateQualityPtr = useQualityScores ? gpuReadStorage->fetchQuality_ptr(bestCandidateReadIds[i]) : nullptr;

                    const float defaultweight = 1.0 - std::sqrtf(bestAlignments[i].nOps
                                                                / (bestAlignments[i].overlap
                                                                    * desiredAlignmentMaxErrorRate));

                    if(bestAlignmentFlags[i] == BestAlignment_t::ReverseComplement){
                        multipleSequenceAlignment.insertCandidate(candidateSequence, [&](int i){
                            return qscore_to_weight[(unsigned char)(*candidateQualityPtr)[length - 1 - i]] * defaultweight;
                        });
                    }else if(bestAlignmentFlags[i] == BestAlignment_t::Forward){
                        multipleSequenceAlignment.insertCandidate(candidateSequence, [&](int i){
                            return qscore_to_weight[(unsigned char)(*candidateQualityPtr)[i]] * defaultweight;
                        });
                    }else{
                        assert(false);
                    }

                }







                    PUSH_RANGE("correct_batch" , 5);

    				for(auto it = batchElems[streamIndex].begin(); it != activeBatchElementsEnd; ++it){
    					auto& b = *it;
    					if(b.active){

							tpc = std::chrono::system_clock::now();
							std::pair<PileupCorrectionResult, TaskTimings> res =
														correct(pileupImage,
															b,
															goodAlignmentProperties.maxErrorRate,
															correctionOptions.estimatedErrorrate,
															correctionOptions.estimatedCoverage,
															correctionOptions.correctCandidates,
															correctionOptions.new_columns_to_correct,
                                                            correctionOptions.classicMode);

							tpd = std::chrono::system_clock::now();
							readcorrectionTimeTotal += tpd - tpc;

                            detailedCorrectionTimings += res.second;

							/*
								features
							*/

							if(correctionOptions.extractFeatures){
                                std::vector<float> tmp(pileupImage.h_support.size());
                                std::copy(pileupImage.h_support.begin(), pileupImage.h_support.end(), tmp.begin());

                                std::vector<MSAFeature> MSAFeatures = extractFeatures(pileupImage.h_consensus.data(),
                                        tmp.data(),
                                        pileupImage.h_coverage.data(),
                                        pileupImage.h_origCoverage.data(),
                                        pileupImage.columnProperties.columnsToCheck,
                                        pileupImage.columnProperties.subjectColumnsBegin_incl,
                                        pileupImage.columnProperties.subjectColumnsEnd_excl,
                                        b.fwdSequenceString,
                                        threadOpts.minhasher->minparams.k, 0.0,
                                        correctionOptions.estimatedCoverage);

								if(MSAFeatures.size() > 0){
									for(const auto& msafeature : MSAFeatures){
										featurestream << b.readId << '\t' << msafeature.position << '\n';
										featurestream << msafeature << '\n';
									}

								}
							}
							auto& correctionResult = res.first;

							avgsupportfail += correctionResult.stats.failedAvgSupport;
							minsupportfail += correctionResult.stats.failedMinSupport;
							mincoveragefail += correctionResult.stats.failedMinCoverage;
							verygoodalignment += correctionResult.stats.isHQ;

							if(correctionResult.isCorrected){
								write_read(b.readId, correctionResult.correctedSequence);
								lock(b.readId);
								(*threadOpts.readIsCorrectedVector)[b.readId] = 1;
								unlock(b.readId);
							}

							for(const auto& correctedCandidate : correctionResult.correctedCandidates){
								const int count = 1;//b.candidateCounts[correctedCandidate.index];
								for(int f = 0; f < count; f++){
									//ReadId_t candidateId = b.candidateIds[b.candidateCountsPrefixSum[correctedCandidate.index] + f];
                                    ReadId_t candidateId = b.candidateIds[correctedCandidate.index];
									bool savingIsOk = false;
									if((*threadOpts.readIsCorrectedVector)[candidateId] == 0){
										lock(candidateId);
										if((*threadOpts.readIsCorrectedVector)[candidateId]== 0) {
											(*threadOpts.readIsCorrectedVector)[candidateId] = 1; // we will process this read
											savingIsOk = true;
											nCorrectedCandidates++;
										}
										unlock(candidateId);
									}
                                    if (savingIsOk) {
                                        //if (b.bestIsForward[correctedCandidate.index])
                                        if(b.bestAlignmentFlags[correctedCandidate.index] == BestAlignment_t::Forward)
                                            write_read(candidateId, correctedCandidate.sequence);
                                        else {
                                            //correctedCandidate.sequence is reverse complement, make reverse complement again
                                            //const std::string fwd = SequenceGeneral(correctedCandidate.sequence, false).reverseComplement().toString();
                                            const std::string fwd = SequenceString(correctedCandidate.sequence).reverseComplement().toString();
                                            write_read(candidateId, fwd);
                                        }
                                    }
								}
							}
    					}
    				}

                    POP_RANGE;

    #endif
    			}
    #endif
    			// update local progress
    			//nProcessedReads += readIds.size();

    			//readIds = threadOpts.batchGen->getNextReadIds();

    		} // end batch processing

            featurestream.flush();

    	#if 1
    		{
    			std::lock_guard < std::mutex > lg(*threadOpts.coutLock);

                std::cout << "thread " << threadOpts.threadId
                        << " : preparation timings detail "
                        << da.count() << " " << db.count() << " " << dc.count()<< '\n';


                std::cout << "thread " << threadOpts.threadId
                        << " : init batch elems "
                        << initIdTimeTotal.count() << '\n';
    			std::cout << "thread " << threadOpts.threadId
    					<< " : find candidates time "
    					<< mapMinhashResultsToSequencesTimeTotal.count() << '\n';
    			std::cout << "thread " << threadOpts.threadId << " : alignment time "
    					<< getAlignmentsTimeTotal.count() << '\n';
    			std::cout << "thread " << threadOpts.threadId
    					<< " : determine good alignments time "
    					<< determinegoodalignmentsTime.count() << '\n';
                std::cout << "thread " << threadOpts.threadId
    					<< " : fetch good candidates time "
    					<< fetchgoodcandidatesTime.count() << '\n';
    			std::cout << "thread " << threadOpts.threadId << " : correction time "
    					<< readcorrectionTimeTotal.count() << '\n';

                std::cout << "thread " << threadOpts.threadId << " : detailed correction time " << '\n'
    					<< detailedCorrectionTimings << '\n';

                std::cout << "thread " << threadOpts.threadId << " : detailed alignment time " << '\n'
                        << shdhandles[0].timings << '\n';


    		}
    	#endif
    	}
    };

}
}


#endif
