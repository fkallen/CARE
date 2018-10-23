#ifndef CARE_GPU_CPU_CORRECTION_THREAD_HPP
#define CARE_GPU_CPU_CORRECTION_THREAD_HPP

#include "../options.hpp"
#include "../batchelem.hpp"
#include "../tasktiming.hpp"
#include "../alignmentwrapper.hpp"
#include "../multiple_sequence_alignment.hpp"
#include "../bestalignment.hpp"

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

        using Minhasher_t = minhasher_t;
    	using ReadStorage_t = readStorage_t;
    	using Sequence_t = typename ReadStorage_t::Sequence_t;
    	using ReadId_t = typename ReadStorage_t::ReadId_t;
        using AlignmentResult_t = typename alignment_result_type<indels>::type;
        using BatchElem_t = BatchElem<ReadStorage_t, AlignmentResult_t>;

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

    		constexpr int nStreams = 2;
            const bool canUseGpu = threadOpts.canUseGpu;

    		std::vector<SHDhandle> shdhandles(nStreams);
    		std::vector<SGAhandle> sgahandles(nStreams);

			for(auto& handle : shdhandles){
				init_SHDhandle(handle,
						threadOpts.deviceId,
						fileProperties.maxSequenceLength,
						Sequence_t::getNumBytes(fileProperties.maxSequenceLength),
						threadOpts.gpuThresholdSHD);
			}

    		pileup::PileupImage pileupImage(correctionOptions.m_coverage,
                                            correctionOptions.kmerlength,
                                            correctionOptions.estimatedCoverage);

    		std::array<std::vector<BatchElem_t>, nStreams> batchElems;
    		std::vector<ReadId_t> readIds = threadOpts.batchGen->getNextReadIds();

    		while(!stopAndAbort && !readIds.empty()){

    			for(int streamIndex = 0; streamIndex < nStreams; ++streamIndex){
                    PUSH_RANGE("init_batch", 0);

                    tpc = std::chrono::system_clock::now();
    				//fit vector size to actual batch size
    				if (batchElems[streamIndex].size() != readIds.size()) {
    					batchElems[streamIndex].resize(readIds.size(),
    									BatchElem_t(*threadOpts.readStorage,
    												correctionOptions, max_candidates));
    				}

    				for(std::size_t i = 0; i < readIds.size(); i++){
    					set_read_id(batchElems[streamIndex][i], readIds[i]);
    					nProcessedQueries++;
    				}

    				for(auto& b : batchElems[streamIndex]){
    					lock(b.readId);
    					if ((*threadOpts.readIsCorrectedVector)[b.readId] == 0) {
    						(*threadOpts.readIsCorrectedVector)[b.readId] = 1;
    					}else{
    						b.active = false;
    						nProcessedQueries--;
    					}
    					unlock(b.readId);
    				}

                    tpd = std::chrono::system_clock::now();
                    initIdTimeTotal += tpd - tpc;

                    POP_RANGE;


                    PUSH_RANGE("find_candidates", 1);


    				tpa = std::chrono::system_clock::now();

    				for(auto& b : batchElems[streamIndex]){


    					// get query data, determine candidates via minhashing, get candidate data
    					if(b.active){
    						findCandidates(b, [&, this](const std::string& sequencestring){
    							return threadOpts.minhasher->getCandidates(sequencestring, max_candidates);
    						});

    						//don't correct candidates with more than estimatedAlignmentCountThreshold alignments
    						if(b.n_candidates > max_candidates)
    							b.active = false;
    					}
    				}

    				tpb = std::chrono::system_clock::now();
    				mapMinhashResultsToSequencesTimeTotal += tpb - tpa;

                    POP_RANGE;

                    PUSH_RANGE("perform_alignments", 3);

    				tpa = std::chrono::system_clock::now();

    				auto activeBatchElementsEnd = std::stable_partition(batchElems[streamIndex].begin(), batchElems[streamIndex].end(), [](const auto& b){return b.active;});

    				if(std::distance(batchElems[streamIndex].begin(), activeBatchElementsEnd) > 0){

    					std::vector<const Sequence_t**> subjectsbegin;
    					std::vector<const Sequence_t**> subjectsend;
    					std::vector<typename std::vector<const Sequence_t*>::iterator> queriesbegin;
    					std::vector<typename std::vector<const Sequence_t*>::iterator> queriesend;
                        std::vector<typename std::vector<BestAlignment_t>::iterator> flagsbegin;
                        std::vector<typename std::vector<BestAlignment_t>::iterator> flagsend;

                        std::vector<typename std::vector<AlignmentResult_t>::iterator> alignmentsbegin;
                        std::vector<typename std::vector<AlignmentResult_t>::iterator> alignmentsend;
                        std::vector<typename std::vector<std::string>::iterator> bestSequenceStringsbegin;
                        std::vector<typename std::vector<std::string>::iterator> bestSequenceStringsend;


    					std::vector<int> queriesPerSubject;

    					subjectsbegin.reserve(batchElems[streamIndex].size());
    					subjectsend.reserve(batchElems[streamIndex].size());
    					queriesbegin.reserve(batchElems[streamIndex].size());
    					queriesend.reserve(batchElems[streamIndex].size());
    					alignmentsbegin.reserve(batchElems[streamIndex].size());
    					alignmentsend.reserve(batchElems[streamIndex].size());
                        flagsbegin.reserve(batchElems[streamIndex].size());
    					flagsend.reserve(batchElems[streamIndex].size());
                        bestSequenceStringsbegin.reserve(batchElems[streamIndex].size());
                        bestSequenceStringsend.reserve(batchElems[streamIndex].size());
    					queriesPerSubject.reserve(batchElems[streamIndex].size());

    					//for(auto& b : batchElems[streamIndex]){
    					for(auto it = batchElems[streamIndex].begin(); it != activeBatchElementsEnd; ++it){
    						auto& b = *it;
    				        auto& flags = b.bestAlignmentFlags;
    				        auto& alignments = b.alignments;
                            auto& strings = b.bestSequenceStrings;

    						subjectsbegin.emplace_back(&b.fwdSequence);
    						subjectsend.emplace_back(&b.fwdSequence + 1);
    						queriesbegin.emplace_back(b.fwdSequences.begin());
    						queriesend.emplace_back(b.fwdSequences.end());
    						alignmentsbegin.emplace_back(alignments.begin());
    						alignmentsend.emplace_back(alignments.end());
    		                flagsbegin.emplace_back(flags.begin());
    						flagsend.emplace_back(flags.end());
                            bestSequenceStringsbegin.emplace_back(strings.begin());
        					bestSequenceStringsend.emplace_back(strings.end());
    						queriesPerSubject.emplace_back(b.fwdSequences.size());
    					}

    				    AlignmentDevice device = AlignmentDevice::None;
    					if(indels){
    						device = semi_global_alignment_canonical_batched_async<Sequence_t>(sgahandles[streamIndex],
    														subjectsbegin,
    														subjectsend,
    														queriesbegin,
    														queriesend,
    														alignmentsbegin,
    														alignmentsend,
    														flagsbegin,
    														flagsend,
                                                            bestSequenceStringsbegin,
                                                            bestSequenceStringsend,
    														queriesPerSubject,
    														goodAlignmentProperties.min_overlap,
    														goodAlignmentProperties.maxErrorRate,
    														goodAlignmentProperties.min_overlap_ratio,
    														alignmentOptions.alignmentscore_match,
    														alignmentOptions.alignmentscore_sub,
    														alignmentOptions.alignmentscore_ins,
    														alignmentOptions.alignmentscore_del,
    														canUseGpu);
    					}else{
    						device = shifted_hamming_distance_canonical_batched_async<Sequence_t>(shdhandles[streamIndex],
    														subjectsbegin,
    														subjectsend,
    														queriesbegin,
    														queriesend,
    														alignmentsbegin,
    														alignmentsend,
    														flagsbegin,
    														flagsend,
                                                            bestSequenceStringsbegin,
                                                            bestSequenceStringsend,
    														queriesPerSubject,
    														goodAlignmentProperties.min_overlap,
    														goodAlignmentProperties.maxErrorRate,
    														goodAlignmentProperties.min_overlap_ratio,
    														canUseGpu);
    					}

    					if(device == AlignmentDevice::CPU)
    						cpuAlignments++;
    					else if (device == AlignmentDevice::GPU)
    						gpuAlignments++;

    				}

    				nProcessedReads += readIds.size();
    				readIds = threadOpts.batchGen->getNextReadIds();

    				tpb = std::chrono::system_clock::now();
    				getAlignmentsTimeTotal += tpb - tpa;

                    POP_RANGE;
    			}


    #if 1

    			for(int streamIndex = 0; streamIndex < nStreams; ++streamIndex){


                    PUSH_RANGE("wait_for_alignments" , 4);

    				tpa = std::chrono::system_clock::now();

    				std::vector<typename std::vector<BestAlignment_t>::iterator> flagsbegin;
    				std::vector<typename std::vector<BestAlignment_t>::iterator> flagsend;

    				std::vector<typename std::vector<AlignmentResult_t>::iterator> alignmentsbegin;
    				std::vector<typename std::vector<AlignmentResult_t>::iterator> alignmentsend;

                    std::vector<typename std::vector<std::string>::iterator> bestSequenceStringsbegin;
    				std::vector<typename std::vector<std::string>::iterator> bestSequenceStringsend;

    				alignmentsbegin.reserve(batchElems[streamIndex].size());
    				alignmentsend.reserve(batchElems[streamIndex].size());
    				flagsbegin.reserve(batchElems[streamIndex].size());
    				flagsend.reserve(batchElems[streamIndex].size());
                    bestSequenceStringsbegin.reserve(batchElems[streamIndex].size());
    				bestSequenceStringsend.reserve(batchElems[streamIndex].size());

    				auto activeBatchElementsEnd = std::stable_partition(batchElems[streamIndex].begin(), batchElems[streamIndex].end(), [](const auto& b){return b.active;});

    				//for(auto& b : batchElems[streamIndex]){
    				for(auto it = batchElems[streamIndex].begin(); it != activeBatchElementsEnd; ++it){
    					auto& b = *it;
    					auto& flags = b.bestAlignmentFlags;

    					auto& alignments = b.alignments;
                        auto& strings = b.bestSequenceStrings;

    					alignmentsbegin.emplace_back(alignments.begin());
    					alignmentsend.emplace_back(alignments.end());
    					flagsbegin.emplace_back(flags.begin());
    					flagsend.emplace_back(flags.end());
                        bestSequenceStringsbegin.emplace_back(strings.begin());
    					bestSequenceStringsend.emplace_back(strings.end());
    				}

					shifted_hamming_distance_canonical_get_results_batched(shdhandles[streamIndex],
											alignmentsbegin,
											alignmentsend,
											flagsbegin,
											flagsend,
                                            bestSequenceStringsbegin,
                                            bestSequenceStringsend,
											canUseGpu);

    				tpb = std::chrono::system_clock::now();
    				getAlignmentsTimeTotal += tpb - tpa;

                    POP_RANGE;
    	#if 1
    				//check quality of alignments
    				tpc = std::chrono::system_clock::now();
    				//for(auto& b : batchElems[streamIndex]){
    				for(auto it = batchElems[streamIndex].begin(); it != activeBatchElementsEnd; ++it){
    					auto& b = *it;
    					if(b.active){
    						determine_good_alignments(b);
    					}
    				}

    				tpd = std::chrono::system_clock::now();
    				determinegoodalignmentsTime += tpd - tpc;

                    tpc = std::chrono::system_clock::now();

    				for(auto it = batchElems[streamIndex].begin(); it != activeBatchElementsEnd; ++it){
    					auto& b = *it;
    					if(b.active && hasEnoughGoodCandidates(b)){
    						if(b.active){
    							//move candidates which are used for correction to the front
    							auto tup = prepare_good_candidates(b);
                                da += std::get<0>(tup);
                                db += std::get<1>(tup);
                                dc += std::get<2>(tup);
    						}
    					}else{
    						//not enough good candidates. cannot correct this read.
    						b.active = false;
    					}
    				}

                    tpd = std::chrono::system_clock::now();
                    fetchgoodcandidatesTime += tpd - tpc;

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
