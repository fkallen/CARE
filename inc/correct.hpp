#ifndef CARE_CORRECT_HPP
#define CARE_CORRECT_HPP

#include "options.hpp"

#include "alignmentwrapper.hpp"
#include "correctionwrapper.hpp"

#include "batchelem.hpp"
#include "graph.hpp"
#include "multiple_sequence_alignment.hpp"
#include "sequence.hpp"
#include "sequencefileio.hpp"
#include "qualityscoreweights.hpp"
#include "tasktiming.hpp"
#include "concatcontainer.hpp"

#include "featureextractor.hpp"
#include "bestalignment.hpp"


#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <thread>
#include <future>



#ifdef __NVCC__
#include <cuda_profiler_api.h>
#endif

namespace care{

    constexpr int maxCPUThreadsPerGPU = 64;

    namespace correctiondetail{

        template<class T, class Count>
		struct Dist{
			T max;
			T average;
			T stddev;
			Count maxCount;
			Count averageCount;
		};

		template<class T, class Count>
		Dist<T,Count> estimateDist(const std::map<T,Count>& map){
			Dist<T, Count> distribution;

			Count sum = 0;
			std::vector<std::pair<T, Count>> vec(map.begin(), map.end());
			std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b){
				return a.second < b.second;
			});

		// AVG
			sum = 0;
			for(const auto& pair : map){
				sum += pair.second;
			}
			distribution.averageCount = sum / vec.size();

			auto it = std::lower_bound(vec.begin(),
										vec.end(),
										std::make_pair(T{}, distribution.averageCount),
										[](const auto& a, const auto& b){
											return a.second < b.second;
										});
			if(it == vec.end())
				it = vec.end() - 1;

			distribution.average = it->first;
		// MAX
			it = std::max_element(vec.begin(), vec.end(), [](const auto& a, const auto& b){
				return a.second < b.second;
			});

			distribution.max = it->first;
			distribution.maxCount = it->second;
		// STDDEV
			T sum2 = 0;
			distribution.stddev = 0;
			for(const auto& pair : map){
				sum2 += pair.first - distribution.average;
			}

			distribution.stddev = std::sqrt(1.0/vec.size() * sum2);

			return distribution;
		}

        template<class minhasher_t, class readStorage_t>
        std::map<std::int64_t, std::int64_t> getCandidateCountHistogram(const minhasher_t& minhasher,
                                                                        const readStorage_t& readStorage,
                                                                        std::uint64_t candidatesToCheck,
                                                                        int threads){

        	using Minhasher_t = minhasher_t;
        	using ReadStorage_t = readStorage_t;
        	using Sequence_t = typename ReadStorage_t::Sequence_t;
        	using ReadId_t = typename ReadStorage_t::ReadId_t;

            std::vector<std::future<std::map<std::int64_t, std::int64_t>>> candidateCounterFutures;
            const ReadId_t sampleCount = candidatesToCheck;
            for(int i = 0; i < threads; i++){
                candidateCounterFutures.push_back(std::async(std::launch::async, [&,i]{
                    std::map<std::int64_t, std::int64_t> candidateMap;
                    std::vector<std::pair<ReadId_t, const Sequence_t*>> numseqpairs;

                    for(ReadId_t readId = i; readId < sampleCount; readId += threads){
                        std::string sequencestring = readStorage.fetchSequence_ptr(readId)->toString();
                        auto candidateList = minhasher.getCandidates(sequencestring, std::numeric_limits<std::uint64_t>::max());
                        std::int64_t count = std::int64_t(candidateList.size()) - 1;
                        candidateMap[count]++;
                    }

                    return candidateMap;
                }));
            }

            std::map<std::int64_t, std::int64_t> allncandidates;

            for(auto& future : candidateCounterFutures){
                const auto& tmpresult = future.get();
                for(const auto& pair : tmpresult){
                    allncandidates[pair.first] += pair.second;
                }
            }

            return allncandidates;
        }

    }

/*
    Block distribution
*/
template<class ReadId_t>
struct BatchGenerator{
    BatchGenerator(){}
    BatchGenerator(std::uint64_t firstId, std::uint64_t lastIdExcl, std::uint64_t batchsize)
            : batchsize(batchsize), firstId(firstId), lastIdExcl(lastIdExcl), currentId(firstId){
                if(batchsize == 0) throw std::runtime_error("BatchGenerator: invalid batch size");
                if(firstId >= lastIdExcl) throw std::runtime_error("BatchGenerator: firstId >= lastIdExcl");
            }
    BatchGenerator(std::uint64_t totalNumberOfReads, std::uint64_t batchsize_, int threadId, int nThreads){
        if(threadId < 0) throw std::runtime_error("BatchGenerator: invalid threadId");
        if(nThreads < 0) throw std::runtime_error("BatchGenerator: invalid nThreads");

    	std::uint64_t chunksize = totalNumberOfReads / nThreads;
    	int leftover = totalNumberOfReads % nThreads;

    	if(threadId < leftover){
    		chunksize++;
    		firstId = threadId == 0 ? 0 : threadId * chunksize;
    		lastIdExcl = firstId + chunksize;
    	}else{
    		firstId = leftover * (chunksize+1) + (threadId - leftover) * chunksize;;
    		lastIdExcl = firstId + chunksize;
    	}


        currentId = firstId;
        batchsize = batchsize_;
        //std::cout << "thread " << threadId << " firstId " << firstId << " lastIdExcl " << lastIdExcl << " batchsize " << batchsize << std::endl;
    };

    std::vector<ReadId_t> getNextReadIds(){
        std::vector<ReadId_t> result;
    	while(result.size() < batchsize && currentId < lastIdExcl){
    		result.push_back(currentId);
    		currentId++;
    	}
        return result;
    }

    std::uint64_t batchsize;
    std::uint64_t firstId;
    std::uint64_t lastIdExcl;
    std::uint64_t currentId;
};

/*
    Find alignment with lowest mismatch ratio.
    If both have same ratio choose alignment with longer overlap.
*/
template<class Alignment>
BestAlignment_t choose_best_alignment(const Alignment& fwdAlignment,
                                    const Alignment& revcmplAlignment,
                                    int querylength,
                                    int candidatelength,
                                    double min_overlap_ratio,
                                    int min_overlap,
                                    double maxErrorRate){
    const int overlap = fwdAlignment.get_overlap();
    const int revcomploverlap = revcmplAlignment.get_overlap();
    const int fwdMismatches = fwdAlignment.get_nOps();
    const int revcmplMismatches = revcmplAlignment.get_nOps();

    BestAlignment_t retval = BestAlignment_t::None;

    const int minimumOverlap = int(querylength * min_overlap_ratio) > min_overlap
                    ? int(querylength * min_overlap_ratio) : min_overlap;

    if(fwdAlignment.get_isValid() && overlap >= minimumOverlap){
        if(revcmplAlignment.get_isValid() && revcomploverlap >= minimumOverlap){
            const double ratio = (double)fwdMismatches / overlap;
            const double revcomplratio = (double)revcmplMismatches / revcomploverlap;

            if(ratio < revcomplratio){
                if(ratio < maxErrorRate){
                    retval = BestAlignment_t::Forward;
                }
            }else if(revcomplratio < ratio){
                if(revcomplratio < maxErrorRate){
                    retval = BestAlignment_t::ReverseComplement;
                }
            }else{
                if(ratio < maxErrorRate){
                    // both have same mismatch ratio, choose longest overlap
                    if(overlap > revcomploverlap){
                        retval = BestAlignment_t::Forward;
                    }else{
                        retval = BestAlignment_t::ReverseComplement;
                    }
                }
            }
        }else{
            if((double)fwdMismatches / overlap < maxErrorRate){
                retval = BestAlignment_t::Forward;
            }
        }
    }else{
        if(revcmplAlignment.get_isValid() && revcomploverlap >= minimumOverlap){
            if((double)revcmplMismatches / revcomploverlap < maxErrorRate){
                retval = BestAlignment_t::ReverseComplement;
            }
        }
    }

    return retval;
}

template<bool indels>
struct alignment_result_type;

template<>
struct alignment_result_type<true>{using type = SGAResult;};

template<>
struct alignment_result_type<false>{using type = SHDResult;};

#if 0
template<class minhasher_t,
		 class readStorage_t,
		 bool indels>
struct ErrorCorrectionThread;

//partial specialization for alignment without indels
template<class minhasher_t,
		 class readStorage_t>
struct ErrorCorrectionThread<minhasher_t, readStorage_t, false>{

	using Minhasher_t = minhasher_t;
	using ReadStorage_t = readStorage_t;
	using Sequence_t = typename ReadStorage_t::Sequence_t;
	using ReadId_t = typename ReadStorage_t::ReadId_t;
    using AlignmentResult_t = typename alignment_result_type<false>::type;
    using BatchElem_t = BatchElem<ReadStorage_t, AlignmentResult_t>;

	struct CorrectionThreadOptions{
		int threadId;
		int deviceId;
		int gpuThresholdSHD;
		int gpuThresholdSGA;
        bool canUseGpu;

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

    correctiondetail::Dist<std::int64_t, std::int64_t> candidateDistribution;

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

    std::thread thread;
    bool isRunning = false;
    volatile bool stopAndAbort = false;

    void run(){
        if(isRunning) throw std::runtime_error("ErrorCorrectionThread::run: Is already running.");
        isRunning = true;
        thread = std::move(std::thread(&ErrorCorrectionThread::execute, this));
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
			auto& stream = outputstream;
			stream << readId << '\n';
			stream << sequence << '\n';
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

        const std::uint64_t estimatedMeanAlignedCandidates = candidateDistribution.max;
        const std::uint64_t estimatedDeviationAlignedCandidates = candidateDistribution.stddev;
        const std::uint64_t estimatedAlignmentCountThreshold = estimatedMeanAlignedCandidates
                                                        + 2.5 * estimatedDeviationAlignedCandidates;

        const std::uint64_t max_candidates = estimatedAlignmentCountThreshold;// * correctionOptions.estimatedCoverage;
        //const std::uint64_t max_candidates = std::numeric_limits<std::uint64_t>::max();

        if(threadOpts.threadId == 0)
            std::cout << "max_candidates " << max_candidates << std::endl;

		constexpr int nStreams = 2;

		std::vector<SHDhandle> shdhandles(nStreams);

		for(auto& handle : shdhandles){
			init_SHDhandle(handle,
							threadOpts.deviceId,
							fileProperties.maxSequenceLength,
							Sequence_t::getNumBytes(fileProperties.maxSequenceLength),
							threadOpts.gpuThresholdSHD);

            handle.buffers.resize(correctionOptions.batchsize,
                correctionOptions.batchsize * max_candidates,
                2 * correctionOptions.batchsize * max_candidates,
                1.0);
		}

		pileup::PileupImage pileupImage(correctionOptions.m_coverage,
                                        correctionOptions.kmerlength,
                                        correctionOptions.estimatedCoverage);

		std::array<std::vector<BatchElem_t>, nStreams> batchElems;
		std::vector<ReadId_t> readIds = threadOpts.batchGen->getNextReadIds();

        const bool canUseGpu = threadOpts.canUseGpu;

		while(!stopAndAbort && !readIds.empty()){

            tpc = std::chrono::system_clock::now();

			for(int streamIndex = 0; streamIndex < nStreams; ++streamIndex){

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

				tpa = std::chrono::system_clock::now();

				for(auto& b : batchElems[streamIndex]){


					// get query data, determine candidates via minhashing, get candidate data
					if(b.active){
						findCandidates(b, [&, this](const std::string& sequencestring){
							return threadOpts.minhasher->getCandidates(sequencestring, max_candidates);
						});

						//don't correct candidates with more than estimatedAlignmentCountThreshold alignments
						if(b.n_candidates > estimatedAlignmentCountThreshold)
							b.active = false;
					}
				}

				tpb = std::chrono::system_clock::now();
				mapMinhashResultsToSequencesTimeTotal += tpb - tpa;

				tpa = std::chrono::system_clock::now();

				auto activeBatchElementsEnd = std::partition(batchElems[streamIndex].begin(), batchElems[streamIndex].end(), [](const auto& b){return b.active;});

				if(std::distance(batchElems[streamIndex].begin(), activeBatchElementsEnd) > 0){

					std::vector<const Sequence_t**> subjectsbegin;
					std::vector<const Sequence_t**> subjectsend;
					std::vector<typename std::vector<const Sequence_t*>::iterator> queriesbegin;
					std::vector<typename std::vector<const Sequence_t*>::iterator> queriesend;
                    std::vector<typename std::vector<BestAlignment_t>::iterator> flagsbegin;
                    std::vector<typename std::vector<BestAlignment_t>::iterator> flagsend;

                    std::vector<typename std::vector<AlignmentResult_t>::iterator> alignmentsbegin;
                    std::vector<typename std::vector<AlignmentResult_t>::iterator> alignmentsend;

					std::vector<int> queriesPerSubject;

					subjectsbegin.reserve(batchElems[streamIndex].size());
					subjectsend.reserve(batchElems[streamIndex].size());
					queriesbegin.reserve(batchElems[streamIndex].size());
					queriesend.reserve(batchElems[streamIndex].size());
					alignmentsbegin.reserve(batchElems[streamIndex].size());
					alignmentsend.reserve(batchElems[streamIndex].size());
                    flagsbegin.reserve(batchElems[streamIndex].size());
					flagsend.reserve(batchElems[streamIndex].size());
					queriesPerSubject.reserve(batchElems[streamIndex].size());

					//for(auto& b : batchElems[streamIndex]){
					for(auto it = batchElems[streamIndex].begin(); it != activeBatchElementsEnd; ++it){
						auto& b = *it;
                        auto& flags = b.bestAlignmentFlags;

                        auto& alignments = b.alignments;

						subjectsbegin.emplace_back(&b.fwdSequence);
						subjectsend.emplace_back(&b.fwdSequence + 1);
						queriesbegin.emplace_back(b.fwdSequences.begin());
						queriesend.emplace_back(b.fwdSequences.end());
						alignmentsbegin.emplace_back(alignments.begin());
						alignmentsend.emplace_back(alignments.end());
                        flagsbegin.emplace_back(flags.begin());
						flagsend.emplace_back(flags.end());
						queriesPerSubject.emplace_back(b.fwdSequences.size());
					}

                    AlignmentDevice device = shifted_hamming_distance_canonical_batched_async<Sequence_t>(shdhandles[streamIndex],
                                                subjectsbegin,
                                                subjectsend,
                                                queriesbegin,
                                                queriesend,
                                                alignmentsbegin,
                                                alignmentsend,
                                                flagsbegin,
                                                flagsend,
                                                queriesPerSubject,
                                                goodAlignmentProperties.min_overlap,
                                                goodAlignmentProperties.maxErrorRate,
                                                goodAlignmentProperties.min_overlap_ratio,
                                                canUseGpu);

					if(device == AlignmentDevice::CPU)
						cpuAlignments++;
					else if (device == AlignmentDevice::GPU)
						gpuAlignments++;


					//shifted_hamming_distance_with_revcompl_get_results_batched(shdhandles[streamIndex],
					//														alignmentsbegin,
					//														alignmentsend,
					//														canUseGpu);
				}

				nProcessedReads += readIds.size();
				readIds = threadOpts.batchGen->getNextReadIds();

				tpb = std::chrono::system_clock::now();
				getAlignmentsTimeTotal += tpb - tpa;
			}


#if 1

			for(int streamIndex = 0; streamIndex < nStreams; ++streamIndex){

                tpa = std::chrono::system_clock::now();

                std::vector<typename std::vector<BestAlignment_t>::iterator> flagsbegin;
                std::vector<typename std::vector<BestAlignment_t>::iterator> flagsend;

                std::vector<typename std::vector<AlignmentResult_t>::iterator> alignmentsbegin;
                std::vector<typename std::vector<AlignmentResult_t>::iterator> alignmentsend;

				alignmentsbegin.reserve(batchElems[streamIndex].size());
				alignmentsend.reserve(batchElems[streamIndex].size());
                flagsbegin.reserve(batchElems[streamIndex].size());
                flagsend.reserve(batchElems[streamIndex].size());

				auto activeBatchElementsEnd = std::partition(batchElems[streamIndex].begin(), batchElems[streamIndex].end(), [](const auto& b){return b.active;});

				//for(auto& b : batchElems[streamIndex]){
				for(auto it = batchElems[streamIndex].begin(); it != activeBatchElementsEnd; ++it){
					auto& b = *it;
                    auto& flags = b.bestAlignmentFlags;

                    auto& alignments = b.alignments;

                    alignmentsbegin.emplace_back(alignments.begin());
                    alignmentsend.emplace_back(alignments.end());
                    flagsbegin.emplace_back(flags.begin());
                    flagsend.emplace_back(flags.end());

				}

				shifted_hamming_distance_canonical_get_results_batched(shdhandles[streamIndex],
																			alignmentsbegin,
																			alignmentsend,
                                                                            flagsbegin,
                                                                            flagsend,
																			canUseGpu);

                tpb = std::chrono::system_clock::now();
                getAlignmentsTimeTotal += tpb - tpa;
	#if 1
				//check quality of alignments
				tpc = std::chrono::system_clock::now();
				//for(auto& b : batchElems[streamIndex]){
				for(auto it = batchElems[streamIndex].begin(); it != activeBatchElementsEnd; ++it){
					auto& b = *it;
					if(b.active){
						determine_good_alignments(b, [&](const AlignmentResult_t& fwdAlignment,
													const AlignmentResult_t& revcmplAlignment,
													int querylength,
													int candidatelength){
							return choose_best_alignment(fwdAlignment,
														revcmplAlignment,
														querylength,
														candidatelength,
														goodAlignmentProperties.min_overlap_ratio,
														goodAlignmentProperties.min_overlap,
														goodAlignmentProperties.maxErrorRate);
						});
					}
				}

				tpd = std::chrono::system_clock::now();
				determinegoodalignmentsTime += tpd - tpc;


				//for(auto& b : batchElems[streamIndex]){
				for(auto it = batchElems[streamIndex].begin(); it != activeBatchElementsEnd; ++it){
					auto& b = *it;
					if(b.active && hasEnoughGoodCandidates(b)){
						tpc = std::chrono::system_clock::now();
						if(b.active){
							//move candidates which are used for correction to the front
							prepare_good_candidates(b);
						}
						tpd = std::chrono::system_clock::now();
						fetchgoodcandidatesTime += tpd - tpc;
					}else{
						//not enough good candidates. cannot correct this read.
						b.active = false;
					}
				}

				//for(auto& b : batchElems[streamIndex]){
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
																				correctionOptions.new_columns_to_correct);

						tpd = std::chrono::system_clock::now();
						readcorrectionTimeTotal += tpd - tpc;

						/*
							features
						*/

						if(correctionOptions.extractFeatures){
							std::vector<MSAFeature> MSAFeatures =  extractFeatures(pileupImage, b.fwdSequenceString,
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

			std::cout << "thread " << threadOpts.threadId << " processed " << nProcessedQueries
					<< " queries" << std::endl;
			std::cout << "thread " << threadOpts.threadId << " corrected "
					<< nCorrectedCandidates << " candidates" << std::endl;
			std::cout << "thread " << threadOpts.threadId << " avgsupportfail "
					<< avgsupportfail << std::endl;
			std::cout << "thread " << threadOpts.threadId << " minsupportfail "
					<< minsupportfail << std::endl;
			std::cout << "thread " << threadOpts.threadId << " mincoveragefail "
					<< mincoveragefail << std::endl;
			std::cout << "thread " << threadOpts.threadId << " sobadcouldnotcorrect "
					<< sobadcouldnotcorrect << std::endl;
			std::cout << "thread " << threadOpts.threadId << " verygoodalignment "
					<< verygoodalignment << std::endl;
			/*std::cout << "thread " << threadOpts.threadId
					<< " CPU alignments " << cpuAlignments
					<< " GPU alignments " << gpuAlignments << std::endl;*/

		//   std::cout << "thread " << threadOpts.threadId << " savedAlignments "
		//           << savedAlignments << " performedAlignments " << performedAlignments << std::endl;
		}
	#endif

	#if 0
		{
			TaskTimings tmp;
			for(std::uint64_t i = 0; i < threadOpts.batchGen->batchsize; i++)
				tmp += batchElems[i].findCandidatesTiming;

			std::lock_guard < std::mutex > lg(*threadOpts.coutLock);
			std::cout << "thread " << threadOpts.threadId << " findCandidatesTiming:\n";
			std::cout << tmp << std::endl;
		}
	#endif

	#if 1
		{
			std::lock_guard < std::mutex > lg(*threadOpts.coutLock);

			std::cout << "thread " << threadOpts.threadId
					<< " : find candidates time "
					<< mapMinhashResultsToSequencesTimeTotal.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment time "
					<< getAlignmentsTimeTotal.count() << '\n';
			std::cout << "thread " << threadOpts.threadId
					<< " : determine good alignments time "
					<< determinegoodalignmentsTime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : correction time "
					<< readcorrectionTimeTotal.count() << '\n';
	#if 0
			if (correctionOptions.correctionMode == CorrectionMode::Hamming) {
				std::cout << "thread " << threadOpts.threadId << " : pileup vote "
						<< pileupImage.timings.findconsensustime.count() << '\n';
				std::cout << "thread " << threadOpts.threadId << " : pileup correct "
						<< pileupImage.timings.correctiontime.count() << '\n';

			} else if (correctionOptions.correctionMode == CorrectionMode::Graph) {
				std::cout << "thread " << threadOpts.threadId << " : graph build "
						<< graphbuildtime.count() << '\n';
				std::cout << "thread " << threadOpts.threadId << " : graph correct "
						<< graphcorrectiontime.count() << '\n';
			}
	#endif
		}
	#endif

		for(auto& handle : shdhandles)
		      destroy_SHDhandle(handle);
	}
};




//partial specialization for alignment with indels
template<class minhasher_t,
		 class readStorage_t>
struct ErrorCorrectionThread<minhasher_t, readStorage_t, true>{

	using Minhasher_t = minhasher_t;
	using ReadStorage_t = readStorage_t;
	using Sequence_t = typename ReadStorage_t::Sequence_t;
	using ReadId_t = typename ReadStorage_t::ReadId_t;
    using AlignmentResult_t = typename alignment_result_type<true>::type;
	using BatchElem_t = BatchElem<ReadStorage_t, AlignmentResult_t>;

	struct CorrectionThreadOptions{
		int threadId;
		int deviceId;
		int gpuThresholdSHD;
		int gpuThresholdSGA;
        bool canUseGpu;

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

    correctiondetail::Dist<std::int64_t, std::int64_t> candidateDistribution;

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

    std::thread thread;
    bool isRunning = false;
    volatile bool stopAndAbort = false;

    void run(){
        if(isRunning) throw std::runtime_error("ErrorCorrectionThread::run: Is already running.");
        isRunning = true;
        thread = std::move(std::thread(&ErrorCorrectionThread::execute, this));
    }

    void join(){
        thread.join();
        isRunning = false;
    }

private:

	void execute() {

		isRunning = true;
#if 1
		std::chrono::time_point<std::chrono::system_clock> tpa, tpb, tpc, tpd;

		std::ofstream outputstream(threadOpts.outputfile);

		auto write_read = [&](const ReadId_t readId, const auto& sequence){
			auto& stream = outputstream;
			stream << readId << '\n';
			stream << sequence << '\n';
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

        const std::uint64_t estimatedMeanAlignedCandidates = candidateDistribution.max;
        const std::uint64_t estimatedDeviationAlignedCandidates = candidateDistribution.stddev;
        const std::uint64_t estimatedAlignmentCountThreshold = estimatedMeanAlignedCandidates
                                                        + 2.5 * estimatedDeviationAlignedCandidates;

        const std::uint64_t max_candidates = estimatedAlignmentCountThreshold;// * correctionOptions.estimatedCoverage;
        //const std::uint64_t max_candidates = std::numeric_limits<std::uint64_t>::max();

        if(threadOpts.threadId == 0)
            std::cout << "max_candidates " << max_candidates << std::endl;

		constexpr int nStreams = 2;

		std::vector<SGAhandle> sgahandles(nStreams);

		for(auto& handle : sgahandles){
			init_SGAhandle(handle,
							threadOpts.deviceId,
							fileProperties.maxSequenceLength,
							Sequence_t::getNumBytes(fileProperties.maxSequenceLength),
							threadOpts.gpuThresholdSGA);

            handle.buffers.resize(correctionOptions.batchsize,
                correctionOptions.batchsize * max_candidates,
                2 * correctionOptions.batchsize * max_candidates,
                1.0);
		}

		errorgraph::ErrorGraph errorgraph;

        std::array<std::vector<BatchElem_t>, nStreams> batchElems;
		std::vector<ReadId_t> readIds = threadOpts.batchGen->getNextReadIds();

        const bool canUseGpu = threadOpts.canUseGpu;

		while(!stopAndAbort &&!readIds.empty()){

            tpc = std::chrono::system_clock::now();

            for(int streamIndex = 0; streamIndex < nStreams; ++streamIndex){

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

                tpa = std::chrono::system_clock::now();

                for(auto& b : batchElems[streamIndex]){


                    // get query data, determine candidates via minhashing, get candidate data
                    if(b.active){
                        findCandidates(b, [&, this](const std::string& sequencestring){
                            return threadOpts.minhasher->getCandidates(sequencestring, max_candidates);
                        });

                        //don't correct candidates with more than estimatedAlignmentCountThreshold alignments
                        if(b.n_candidates > estimatedAlignmentCountThreshold)
                            b.active = false;
                    }
                }

                tpb = std::chrono::system_clock::now();
                mapMinhashResultsToSequencesTimeTotal += tpb - tpa;

			    tpa = std::chrono::system_clock::now();

                auto activeBatchElementsEnd = std::partition(batchElems[streamIndex].begin(), batchElems[streamIndex].end(), [](const auto& b){return b.active;});

                if(std::distance(batchElems[streamIndex].begin(), activeBatchElementsEnd) > 0){

                    std::vector<const Sequence_t**> subjectsbegin;
                    std::vector<const Sequence_t**> subjectsend;
                    std::vector<typename std::vector<const Sequence_t*>::iterator> queriesbegin;
                    std::vector<typename std::vector<const Sequence_t*>::iterator> queriesend;
                    std::vector<typename std::vector<BestAlignment_t>::iterator> flagsbegin;
                    std::vector<typename std::vector<BestAlignment_t>::iterator> flagsend;

                    std::vector<typename std::vector<AlignmentResult_t>::iterator> alignmentsbegin;
                    std::vector<typename std::vector<AlignmentResult_t>::iterator> alignmentsend;

                    std::vector<int> queriesPerSubject;

                    subjectsbegin.reserve(batchElems[streamIndex].size());
                    subjectsend.reserve(batchElems[streamIndex].size());
                    queriesbegin.reserve(batchElems[streamIndex].size());
                    queriesend.reserve(batchElems[streamIndex].size());
                    alignmentsbegin.reserve(batchElems[streamIndex].size());
                    alignmentsend.reserve(batchElems[streamIndex].size());
                    flagsbegin.reserve(batchElems[streamIndex].size());
                    flagsend.reserve(batchElems[streamIndex].size());
                    queriesPerSubject.reserve(batchElems[streamIndex].size());

                    //for(auto& b : batchElems[streamIndex]){
                    for(auto it = batchElems[streamIndex].begin(); it != activeBatchElementsEnd; ++it){
                        auto& b = *it;
                        auto& flags = b.bestAlignmentFlags;

                        auto& alignments = b.alignments;

                        subjectsbegin.emplace_back(&b.fwdSequence);
                        subjectsend.emplace_back(&b.fwdSequence + 1);
                        queriesbegin.emplace_back(b.fwdSequences.begin());
                        queriesend.emplace_back(b.fwdSequences.end());
                        alignmentsbegin.emplace_back(alignments.begin());
                        alignmentsend.emplace_back(alignments.end());
                        flagsbegin.emplace_back(flags.begin());
                        flagsend.emplace_back(flags.end());
                        queriesPerSubject.emplace_back(b.fwdSequences.size());
                    }

                    AlignmentDevice device = semi_global_alignment_canonical_batched_async<Sequence_t>(sgahandles[streamIndex],
                                                subjectsbegin,
                                                subjectsend,
                                                queriesbegin,
                                                queriesend,
                                                alignmentsbegin,
                                                alignmentsend,
                                                flagsbegin,
                                                flagsend,
                                                queriesPerSubject,
                                                goodAlignmentProperties.min_overlap,
                                                goodAlignmentProperties.maxErrorRate,
                                                goodAlignmentProperties.min_overlap_ratio,
                                                alignmentOptions.alignmentscore_match,
                                                alignmentOptions.alignmentscore_sub,
                                                alignmentOptions.alignmentscore_ins,
                                                alignmentOptions.alignmentscore_del,
                                                canUseGpu);

					if(device == AlignmentDevice::CPU)
						cpuAlignments++;
					else if (device == AlignmentDevice::GPU)
						gpuAlignments++;
				}

                nProcessedReads += readIds.size();
                readIds = threadOpts.batchGen->getNextReadIds();

                tpb = std::chrono::system_clock::now();
                getAlignmentsTimeTotal += tpb - tpa;
			}

            for(int streamIndex = 0; streamIndex < nStreams; ++streamIndex){

                tpa = std::chrono::system_clock::now();

                std::vector<typename std::vector<BestAlignment_t>::iterator> flagsbegin;
                std::vector<typename std::vector<BestAlignment_t>::iterator> flagsend;

                std::vector<typename std::vector<AlignmentResult_t>::iterator> alignmentsbegin;
                std::vector<typename std::vector<AlignmentResult_t>::iterator> alignmentsend;

				alignmentsbegin.reserve(batchElems[streamIndex].size());
				alignmentsend.reserve(batchElems[streamIndex].size());
                flagsbegin.reserve(batchElems[streamIndex].size());
                flagsend.reserve(batchElems[streamIndex].size());

				auto activeBatchElementsEnd = std::partition(batchElems[streamIndex].begin(), batchElems[streamIndex].end(), [](const auto& b){return b.active;});

				//for(auto& b : batchElems[streamIndex]){
				for(auto it = batchElems[streamIndex].begin(); it != activeBatchElementsEnd; ++it){
					auto& b = *it;
                    auto& flags = b.bestAlignmentFlags;

                    auto& alignments = b.alignments;

                    alignmentsbegin.emplace_back(alignments.begin());
                    alignmentsend.emplace_back(alignments.end());
                    flagsbegin.emplace_back(flags.begin());
                    flagsend.emplace_back(flags.end());
				}

                semi_global_alignment_canonical_get_results_batched(sgahandles[streamIndex],
                                                            alignmentsbegin,
                                                            alignmentsend,
                                                            flagsbegin,
                                                            flagsend,
                                                            canUseGpu);

    			tpb = std::chrono::system_clock::now();
    			getAlignmentsTimeTotal += tpb - tpa;

    			//check quality of alignments
    			tpc = std::chrono::system_clock::now();
                for(auto it = batchElems[streamIndex].begin(); it != activeBatchElementsEnd; ++it){
                    auto& b = *it;
                    if(b.active){
                        determine_good_alignments(b, [&](const AlignmentResult_t& fwdAlignment,
                                                    const AlignmentResult_t& revcmplAlignment,
                                                    int querylength,
                                                    int candidatelength){
                            return choose_best_alignment(fwdAlignment,
                                                        revcmplAlignment,
                                                        querylength,
                                                        candidatelength,
                                                        goodAlignmentProperties.min_overlap_ratio,
                                                        goodAlignmentProperties.min_overlap,
                                                        goodAlignmentProperties.maxErrorRate);
                        });
                    }
                }

    			tpd = std::chrono::system_clock::now();
    			determinegoodalignmentsTime += tpd - tpc;

                for(auto it = batchElems[streamIndex].begin(); it != activeBatchElementsEnd; ++it){
                    auto& b = *it;
                    if(b.active && hasEnoughGoodCandidates(b)){
                        tpc = std::chrono::system_clock::now();
                        if(b.active){
                            //move candidates which are used for correction to the front
                            prepare_good_candidates(b);
                        }
                        tpd = std::chrono::system_clock::now();
                        fetchgoodcandidatesTime += tpd - tpc;
                    }else{
                        //not enough good candidates. cannot correct this read.
                        b.active = false;
                    }
                }

    			for(auto it = batchElems[streamIndex].begin(); it != activeBatchElementsEnd; ++it){
                    auto& b = *it;
    				if(b.active){
    					tpc = std::chrono::system_clock::now();

                        std::pair<GraphCorrectionResult, TaskTimings> res = correct(errorgraph,
                                                                            b,
                                                                            goodAlignmentProperties.maxErrorRate,
                                                                            correctionOptions.graphalpha,
                                                                            correctionOptions.graphx);

                        auto& correctionResult = res.first;

    					tpd = std::chrono::system_clock::now();
    					readcorrectionTimeTotal += tpd - tpc;

    					write_read(b.readId, correctionResult.correctedSequence);
    					lock(b.readId);
    					(*threadOpts.readIsCorrectedVector)[b.readId] = 1;
    					unlock(b.readId);
    				}
    			}

            }
		} // end batch processing

	#if 1
		{
			std::lock_guard < std::mutex > lg(*threadOpts.coutLock);

			std::cout << "thread " << threadOpts.threadId << " processed " << nProcessedQueries
					<< " queries" << std::endl;
			std::cout << "thread " << threadOpts.threadId << " corrected "
					<< nCorrectedCandidates << " candidates" << std::endl;
			std::cout << "thread " << threadOpts.threadId << " avgsupportfail "
					<< avgsupportfail << std::endl;
			std::cout << "thread " << threadOpts.threadId << " minsupportfail "
					<< minsupportfail << std::endl;
			std::cout << "thread " << threadOpts.threadId << " mincoveragefail "
					<< mincoveragefail << std::endl;
			std::cout << "thread " << threadOpts.threadId << " sobadcouldnotcorrect "
					<< sobadcouldnotcorrect << std::endl;
			std::cout << "thread " << threadOpts.threadId << " verygoodalignment "
					<< verygoodalignment << std::endl;
			/*std::cout << "thread " << threadOpts.threadId
					<< " CPU alignments " << cpuAlignments
					<< " GPU alignments " << gpuAlignments << std::endl;*/

		//   std::cout << "thread " << threadOpts.threadId << " savedAlignments "
		//           << savedAlignments << " performedAlignments " << performedAlignments << std::endl;
		}
	#endif

	#if 0
		{
			TaskTimings tmp;
			for(std::uint64_t i = 0; i < threadOpts.batchGen->batchsize; i++)
				tmp += batchElems[i].findCandidatesTiming;

			std::lock_guard < std::mutex > lg(*threadOpts.coutLock);
			std::cout << "thread " << threadOpts.threadId << " findCandidatesTiming:\n";
			std::cout << tmp << std::endl;
		}
	#endif

	#if 1
		{
			std::lock_guard < std::mutex > lg(*threadOpts.coutLock);

			std::cout << "thread " << threadOpts.threadId
					<< " : find candidates time "
					<< mapMinhashResultsToSequencesTimeTotal.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment time "
					<< getAlignmentsTimeTotal.count() << '\n';
			std::cout << "thread " << threadOpts.threadId
					<< " : determine good alignments time "
					<< determinegoodalignmentsTime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : correction time "
					<< readcorrectionTimeTotal.count() << '\n';
	#if 0
			if (correctionOptions.correctionMode == CorrectionMode::Hamming) {
				std::cout << "thread " << threadOpts.threadId << " : pileup vote "
						<< pileupImage.timings.findconsensustime.count() << '\n';
				std::cout << "thread " << threadOpts.threadId << " : pileup correct "
						<< pileupImage.timings.correctiontime.count() << '\n';

			} else if (correctionOptions.correctionMode == CorrectionMode::Graph) {
				std::cout << "thread " << threadOpts.threadId << " : graph build "
						<< graphbuildtime.count() << '\n';
				std::cout << "thread " << threadOpts.threadId << " : graph correct "
						<< graphcorrectiontime.count() << '\n';
			}
	#endif
		}
	#endif

		for(auto& handle : sgahandles)
		      destroy_SGAhandle(handle);
#endif
	}
};


#endif







template<class minhasher_t,
		 class readStorage_t,
		 bool indels>
struct ErrorCorrectionThreadCombined{
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
        bool canUseGpu;

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

    correctiondetail::Dist<std::int64_t, std::int64_t> candidateDistribution;

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

    std::thread thread;
    bool isRunning = false;
    volatile bool stopAndAbort = false;

    void run(){
        if(isRunning) throw std::runtime_error("ErrorCorrectionThreadCombined::run: Is already running.");
        isRunning = true;
        thread = std::move(std::thread(&ErrorCorrectionThreadCombined::execute, this));
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
			auto& stream = outputstream;
			stream << readId << '\n';
			stream << sequence << '\n';
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

        const std::uint64_t estimatedMeanAlignedCandidates = candidateDistribution.max;
        const std::uint64_t estimatedDeviationAlignedCandidates = candidateDistribution.stddev;
        const std::uint64_t estimatedAlignmentCountThreshold = estimatedMeanAlignedCandidates
                                                        + 2.5 * estimatedDeviationAlignedCandidates;

        const std::uint64_t max_candidates = estimatedAlignmentCountThreshold;// * correctionOptions.estimatedCoverage;
        //const std::uint64_t max_candidates = std::numeric_limits<std::uint64_t>::max();

        if(threadOpts.threadId == 0)
            std::cout << "max_candidates " << max_candidates << std::endl;

		constexpr int nStreams = 2;
        const bool canUseGpu = threadOpts.canUseGpu;

		std::vector<SHDhandle> shdhandles(nStreams);
		std::vector<SGAhandle> sgahandles(nStreams);

		if(indels){
			for(auto& handle : sgahandles){
				init_SGAhandle(handle,
						threadOpts.deviceId,
						fileProperties.maxSequenceLength,
						Sequence_t::getNumBytes(fileProperties.maxSequenceLength),
						threadOpts.gpuThresholdSGA);
                if(canUseGpu){
				    handle.buffers.resize(correctionOptions.batchsize,
                        					correctionOptions.batchsize * max_candidates,
                        					2 * correctionOptions.batchsize * max_candidates,
                        					1.0);
                }
			}
		}else{

			for(auto& handle : shdhandles){
				init_SHDhandle(handle,
						threadOpts.deviceId,
						fileProperties.maxSequenceLength,
						Sequence_t::getNumBytes(fileProperties.maxSequenceLength),
						threadOpts.gpuThresholdSHD);
                if(canUseGpu){
				    handle.buffers.resize(correctionOptions.batchsize,
                    					correctionOptions.batchsize * max_candidates,
                    					2 * correctionOptions.batchsize * max_candidates,
                    					1.0);
                }
			}
		}

		errorgraph::ErrorGraph errorgraph;
		pileup::PileupImage pileupImage(correctionOptions.m_coverage,
                                        correctionOptions.kmerlength,
                                        correctionOptions.estimatedCoverage);

		std::array<std::vector<BatchElem_t>, nStreams> batchElems;
		std::vector<ReadId_t> readIds = threadOpts.batchGen->getNextReadIds();



		while(!stopAndAbort && !readIds.empty()){

            tpc = std::chrono::system_clock::now();

			for(int streamIndex = 0; streamIndex < nStreams; ++streamIndex){

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

				tpa = std::chrono::system_clock::now();

				for(auto& b : batchElems[streamIndex]){


					// get query data, determine candidates via minhashing, get candidate data
					if(b.active){
						findCandidates(b, [&, this](const std::string& sequencestring){
							return threadOpts.minhasher->getCandidates(sequencestring, max_candidates);
						});

						//don't correct candidates with more than estimatedAlignmentCountThreshold alignments
						if(b.n_candidates > estimatedAlignmentCountThreshold)
							b.active = false;
					}
				}

				tpb = std::chrono::system_clock::now();
				mapMinhashResultsToSequencesTimeTotal += tpb - tpa;

				tpa = std::chrono::system_clock::now();

				auto activeBatchElemensEnd = std::partition(batchElems[streamIndex].begin(), batchElems[streamIndex].end(), [](const auto& b){return b.active;});

				if(std::distance(batchElems[streamIndex].begin(), activeBatchElemensEnd) > 0){

					std::vector<const Sequence_t**> subjectsbegin;
					std::vector<const Sequence_t**> subjectsend;
					std::vector<typename std::vector<const Sequence_t*>::iterator> queriesbegin;
					std::vector<typename std::vector<const Sequence_t*>::iterator> queriesend;
                    std::vector<typename std::vector<BestAlignment_t>::iterator> flagsbegin;
                    std::vector<typename std::vector<BestAlignment_t>::iterator> flagsend;

                    std::vector<typename std::vector<AlignmentResult_t>::iterator> alignmentsbegin;
                    std::vector<typename std::vector<AlignmentResult_t>::iterator> alignmentsend;

					std::vector<int> queriesPerSubject;

					subjectsbegin.reserve(batchElems[streamIndex].size());
					subjectsend.reserve(batchElems[streamIndex].size());
					queriesbegin.reserve(batchElems[streamIndex].size());
					queriesend.reserve(batchElems[streamIndex].size());
					alignmentsbegin.reserve(batchElems[streamIndex].size());
					alignmentsend.reserve(batchElems[streamIndex].size());
                    flagsbegin.reserve(batchElems[streamIndex].size());
					flagsend.reserve(batchElems[streamIndex].size());
					queriesPerSubject.reserve(batchElems[streamIndex].size());

					//for(auto& b : batchElems[streamIndex]){
					for(auto it = batchElems[streamIndex].begin(); it != activeBatchElemensEnd; ++it){
						auto& b = *it;
				        auto& flags = b.bestAlignmentFlags;

				        auto& alignments = b.alignments;

								subjectsbegin.emplace_back(&b.fwdSequence);
								subjectsend.emplace_back(&b.fwdSequence + 1);
								queriesbegin.emplace_back(b.fwdSequences.begin());
								queriesend.emplace_back(b.fwdSequences.end());
								alignmentsbegin.emplace_back(alignments.begin());
								alignmentsend.emplace_back(alignments.end());
				                flagsbegin.emplace_back(flags.begin());
								flagsend.emplace_back(flags.end());
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
			}


#if 1

			for(int streamIndex = 0; streamIndex < nStreams; ++streamIndex){

				tpa = std::chrono::system_clock::now();

				std::vector<typename std::vector<BestAlignment_t>::iterator> flagsbegin;
				std::vector<typename std::vector<BestAlignment_t>::iterator> flagsend;

				std::vector<typename std::vector<AlignmentResult_t>::iterator> alignmentsbegin;
				std::vector<typename std::vector<AlignmentResult_t>::iterator> alignmentsend;

				alignmentsbegin.reserve(batchElems[streamIndex].size());
				alignmentsend.reserve(batchElems[streamIndex].size());
				flagsbegin.reserve(batchElems[streamIndex].size());
				flagsend.reserve(batchElems[streamIndex].size());

				auto activeBatchElemensEnd = std::partition(batchElems[streamIndex].begin(), batchElems[streamIndex].end(), [](const auto& b){return b.active;});

				//for(auto& b : batchElems[streamIndex]){
				for(auto it = batchElems[streamIndex].begin(); it != activeBatchElemensEnd; ++it){
					auto& b = *it;
					auto& flags = b.bestAlignmentFlags;

					auto& alignments = b.alignments;

					alignmentsbegin.emplace_back(alignments.begin());
					alignmentsend.emplace_back(alignments.end());
					flagsbegin.emplace_back(flags.begin());
					flagsend.emplace_back(flags.end());

				}

				if(indels){
					semi_global_alignment_canonical_get_results_batched(sgahandles[streamIndex],
										                    alignmentsbegin,
										                    alignmentsend,
										                    flagsbegin,
										                    flagsend,
										                    canUseGpu);
				}else{
					shifted_hamming_distance_canonical_get_results_batched(shdhandles[streamIndex],
											alignmentsbegin,
											alignmentsend,
											flagsbegin,
											flagsend,
											canUseGpu);
				}


				tpb = std::chrono::system_clock::now();
				getAlignmentsTimeTotal += tpb - tpa;
	#if 1
				//check quality of alignments
				tpc = std::chrono::system_clock::now();
				//for(auto& b : batchElems[streamIndex]){
				for(auto it = batchElems[streamIndex].begin(); it != activeBatchElemensEnd; ++it){
					auto& b = *it;
					if(b.active){
						determine_good_alignments(b, [&](const AlignmentResult_t& fwdAlignment,
													const AlignmentResult_t& revcmplAlignment,
													int querylength,
													int candidatelength){
							return choose_best_alignment(fwdAlignment,
														revcmplAlignment,
														querylength,
														candidatelength,
														goodAlignmentProperties.min_overlap_ratio,
														goodAlignmentProperties.min_overlap,
														goodAlignmentProperties.maxErrorRate);
						});
					}
				}

				tpd = std::chrono::system_clock::now();
				determinegoodalignmentsTime += tpd - tpc;


				//for(auto& b : batchElems[streamIndex]){
				for(auto it = batchElems[streamIndex].begin(); it != activeBatchElemensEnd; ++it){
					auto& b = *it;
					if(b.active && hasEnoughGoodCandidates(b)){
						tpc = std::chrono::system_clock::now();
						if(b.active){
							//move candidates which are used for correction to the front
							prepare_good_candidates(b);
						}
						tpd = std::chrono::system_clock::now();
						fetchgoodcandidatesTime += tpd - tpc;
					}else{
						//not enough good candidates. cannot correct this read.
						b.active = false;
					}
				}

				//for(auto& b : batchElems[streamIndex]){
				for(auto it = batchElems[streamIndex].begin(); it != activeBatchElemensEnd; ++it){
					auto& b = *it;
					if(b.active){
						if(indels){
							tpc = std::chrono::system_clock::now();

							std::pair<GraphCorrectionResult, TaskTimings> res = correct(errorgraph,
														b,
														goodAlignmentProperties.maxErrorRate,
														correctionOptions.graphalpha,
														correctionOptions.graphx);

							auto& correctionResult = res.first;

							tpd = std::chrono::system_clock::now();
							readcorrectionTimeTotal += tpd - tpc;

							write_read(b.readId, correctionResult.correctedSequence);
							lock(b.readId);
							(*threadOpts.readIsCorrectedVector)[b.readId] = 1;
							unlock(b.readId);
						}else{
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

							/*
								features
							*/

							if(correctionOptions.extractFeatures){
								std::vector<MSAFeature> MSAFeatures =  extractFeatures(pileupImage, b.fwdSequenceString,
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
				}

#endif
			}
#endif
			// update local progress
			//nProcessedReads += readIds.size();

			//readIds = threadOpts.batchGen->getNextReadIds();

		} // end batch processing

        featurestream.flush();

	#if 0
		{
			std::lock_guard < std::mutex > lg(*threadOpts.coutLock);

			std::cout << "thread " << threadOpts.threadId << " processed " << nProcessedQueries
					<< " queries" << std::endl;
			std::cout << "thread " << threadOpts.threadId << " corrected "
					<< nCorrectedCandidates << " candidates" << std::endl;
			std::cout << "thread " << threadOpts.threadId << " avgsupportfail "
					<< avgsupportfail << std::endl;
			std::cout << "thread " << threadOpts.threadId << " minsupportfail "
					<< minsupportfail << std::endl;
			std::cout << "thread " << threadOpts.threadId << " mincoveragefail "
					<< mincoveragefail << std::endl;
			std::cout << "thread " << threadOpts.threadId << " sobadcouldnotcorrect "
					<< sobadcouldnotcorrect << std::endl;
			std::cout << "thread " << threadOpts.threadId << " verygoodalignment "
					<< verygoodalignment << std::endl;
			/*std::cout << "thread " << threadOpts.threadId
					<< " CPU alignments " << cpuAlignments
					<< " GPU alignments " << gpuAlignments << std::endl;*/

		//   std::cout << "thread " << threadOpts.threadId << " savedAlignments "
		//           << savedAlignments << " performedAlignments " << performedAlignments << std::endl;
		}
	#endif

	#if 0
		{
			TaskTimings tmp;
			for(std::uint64_t i = 0; i < threadOpts.batchGen->batchsize; i++)
				tmp += batchElems[i].findCandidatesTiming;

			std::lock_guard < std::mutex > lg(*threadOpts.coutLock);
			std::cout << "thread " << threadOpts.threadId << " findCandidatesTiming:\n";
			std::cout << tmp << std::endl;
		}
	#endif

	#if 0
		{
			std::lock_guard < std::mutex > lg(*threadOpts.coutLock);

			std::cout << "thread " << threadOpts.threadId
					<< " : find candidates time "
					<< mapMinhashResultsToSequencesTimeTotal.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment time "
					<< getAlignmentsTimeTotal.count() << '\n';
			std::cout << "thread " << threadOpts.threadId
					<< " : determine good alignments time "
					<< determinegoodalignmentsTime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : correction time "
					<< readcorrectionTimeTotal.count() << '\n';
	#if 0
			if (correctionOptions.correctionMode == CorrectionMode::Hamming) {
				std::cout << "thread " << threadOpts.threadId << " : pileup vote "
						<< pileupImage.timings.findconsensustime.count() << '\n';
				std::cout << "thread " << threadOpts.threadId << " : pileup correct "
						<< pileupImage.timings.correctiontime.count() << '\n';

			} else if (correctionOptions.correctionMode == CorrectionMode::Graph) {
				std::cout << "thread " << threadOpts.threadId << " : graph build "
						<< graphbuildtime.count() << '\n';
				std::cout << "thread " << threadOpts.threadId << " : graph correct "
						<< graphcorrectiontime.count() << '\n';
			}
	#endif
		}
	#endif

        if(canUseGpu){
            if(indels){
                for(auto& handle : sgahandles)
                    destroy_SGAhandle(handle);
            }else{
                for(auto& handle : shdhandles)
                    destroy_SHDhandle(handle);
            }
        }
	}
};






















template<class minhasher_t,
		 class readStorage_t,
		 bool indels>
void correct(const MinhashOptions& minhashOptions,
				  const AlignmentOptions& alignmentOptions,
				  const GoodAlignmentProperties& goodAlignmentProperties,
				  const CorrectionOptions& correctionOptions,
				  const RuntimeOptions& runtimeOptions,
				  const FileOptions& fileOptions,
                  minhasher_t& minhasher,
                  readStorage_t& readStorage,
				  std::vector<char>& readIsCorrectedVector,
				  std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
				  std::size_t nLocksForProcessedFlags,
				  const std::vector<int>& deviceIds){

	using Minhasher_t = minhasher_t;
	using ReadStorage_t = readStorage_t;
	using Sequence_t = typename ReadStorage_t::Sequence_t;
	using ReadId_t = typename ReadStorage_t::ReadId_t;

	using ErrorCorrectionThread_t = ErrorCorrectionThreadCombined<Minhasher_t, ReadStorage_t, indels>;

      // initialize qscore-to-weight lookup table
  	init_weights();

    SequenceFileProperties props = getSequenceFileProperties(fileOptions.inputfile, fileOptions.format);

    /*
        Make candidate statistics
    */

    std::cout << "estimating candidate cutoff" << std::endl;

    correctiondetail::Dist<std::int64_t, std::int64_t> candidateDistribution;

    {
        TIMERSTARTCPU(candidateestimation);
        std::map<std::int64_t, std::int64_t> candidateHistogram
                = correctiondetail::getCandidateCountHistogram(minhasher,
                                            readStorage,
                                            props.nReads / 10,
                                            runtimeOptions.threads);

        TIMERSTOPCPU(candidateestimation);

        candidateDistribution = correctiondetail::estimateDist(candidateHistogram);

        std::vector<std::pair<std::int64_t, std::int64_t>> vec(candidateHistogram.begin(), candidateHistogram.end());
        std::sort(vec.begin(), vec.end(), [](auto p1, auto p2){ return p1.second < p2.second;});

        std::ofstream of("ncandidates.txt");
        for(const auto& p : vec)
            of << p.first << " " << p.second << '\n';
        of.flush();
    }

    std::cout << "candidates.max " << candidateDistribution.max << std::endl;
    std::cout << "candidates.average " << candidateDistribution.average << std::endl;
    std::cout << "candidates.stddev " << candidateDistribution.stddev << std::endl;

    /*
        Spawn correction threads
    */

//#define DO_PROFILE

#if 1
    const int nCorrectorThreads = deviceIds.size() == 0 ? runtimeOptions.nCorrectorThreads
                        : std::min(runtimeOptions.nCorrectorThreads, maxCPUThreadsPerGPU * int(deviceIds.size()));
#else
	const int nCorrectorThreads = 1;
#endif

	std::cout << "Using " << nCorrectorThreads << " corrector threads" << std::endl;

    std::vector<std::string> tmpfiles;
    for(int i = 0; i < nCorrectorThreads; i++){
        tmpfiles.emplace_back(fileOptions.outputfile + "_tmp_" + std::to_string(1000 + i));
    }

    std::vector<BatchGenerator<ReadId_t>> generators(nCorrectorThreads);
    std::vector<ErrorCorrectionThread_t> ecthreads(nCorrectorThreads);
    std::vector<char> readIsProcessedVector(readIsCorrectedVector);
    std::mutex writelock;

    std::vector<int> gpuThresholdsSHD(deviceIds.size(), 0);
    for(std::size_t i = 0; i < deviceIds.size(); i++){
        /*int threshold = find_shifted_hamming_distance_gpu_threshold(deviceIds[i],
                                                                       props.minSequenceLength,
                                                                       SDIV(props.minSequenceLength, 4));*/
        gpuThresholdsSHD[i] = std::min(0, 0);
    }

    std::vector<int> gpuThresholdsSGA(deviceIds.size(), 0);
    for(std::size_t i = 0; i < deviceIds.size(); i++){
        /*int threshold = find_semi_global_alignment_gpu_threshold(deviceIds[i],
                                                                       props.minSequenceLength,
                                                                       SDIV(props.minSequenceLength, 4));*/
        gpuThresholdsSGA[i] = std::min(0, 0);
    }

    /*for(std::size_t i = 0; i < gpuThresholdsSHD.size(); i++)
        std::cout << "GPU " << i
                  << ": gpuThresholdSHD " << gpuThresholdsSHD[i]
                  << " gpuThresholdSGA " << gpuThresholdsSGA[i] << std::endl;*/

    for(int threadId = 0; threadId < nCorrectorThreads; threadId++){

        generators[threadId] = BatchGenerator<ReadId_t>(props.nReads, correctionOptions.batchsize, threadId, nCorrectorThreads);
        typename ErrorCorrectionThread_t::CorrectionThreadOptions threadOpts;
        threadOpts.threadId = threadId;
        threadOpts.deviceId = deviceIds.size() == 0 ? -1 : deviceIds[threadId % deviceIds.size()];
        threadOpts.gpuThresholdSHD = deviceIds.size() == 0 ? 0 : gpuThresholdsSHD[threadId % deviceIds.size()];
        threadOpts.gpuThresholdSGA = deviceIds.size() == 0 ? 0 : gpuThresholdsSGA[threadId % deviceIds.size()];
        threadOpts.canUseGpu = runtimeOptions.canUseGpu;
        threadOpts.outputfile = tmpfiles[threadId];
        threadOpts.batchGen = &generators[threadId];
        threadOpts.minhasher = &minhasher;
        threadOpts.readStorage = &readStorage;
        threadOpts.coutLock = &writelock;
        threadOpts.readIsProcessedVector = &readIsProcessedVector;
        threadOpts.readIsCorrectedVector = &readIsCorrectedVector;
        threadOpts.locksForProcessedFlags = locksForProcessedFlags.get();
        threadOpts.nLocksForProcessedFlags = nLocksForProcessedFlags;

        ecthreads[threadId].alignmentOptions = alignmentOptions;
        ecthreads[threadId].goodAlignmentProperties = goodAlignmentProperties;
        ecthreads[threadId].correctionOptions = correctionOptions;
        ecthreads[threadId].threadOpts = threadOpts;
        ecthreads[threadId].fileProperties = props;
        ecthreads[threadId].candidateDistribution = candidateDistribution;

        ecthreads[threadId].run();
    }

    std::cout << "Correcting..." << std::endl;


#ifndef DO_PROFILE

    bool showProgress = runtimeOptions.showProgress;

    std::thread progressThread = std::thread([&]() -> void{
        if(!showProgress)
            return;

        std::chrono::time_point<std::chrono::system_clock> timepoint_begin = std::chrono::system_clock::now();
        std::chrono::duration<double> runtime = std::chrono::seconds(0);
        std::chrono::duration<int> sleepinterval = std::chrono::seconds(1);

        while(showProgress){
            ReadId_t progress = 0;
            ReadId_t correctorProgress = 0;

            for(int i = 0; i < nCorrectorThreads; i++){
                correctorProgress += ecthreads[i].nProcessedReads;
            }

            progress = correctorProgress;

            printf("Progress: %3.2f %% %10u %10lu (Runtime: %03d:%02d:%02d)\r",
                    ((progress * 1.0 / props.nReads) * 100.0),
                    correctorProgress, props.nReads,
                    int(std::chrono::duration_cast<std::chrono::hours>(runtime).count()),
                    int(std::chrono::duration_cast<std::chrono::minutes>(runtime).count()) % 60,
                    int(runtime.count()) % 60);
            std::cout << std::flush;

            if(progress < props.nReads){
                  std::this_thread::sleep_for(sleepinterval);
                  runtime = std::chrono::system_clock::now() - timepoint_begin;
            }
        }
    });

#else

    constexpr int sleepiterbegin = 1;
    constexpr int sleepiterend = 2;

    int sleepiter = 0;

    std::chrono::duration<double> runtime = std::chrono::seconds(0);
    std::chrono::duration<int> sleepinterval = std::chrono::seconds(3);

    while(true){

        std::this_thread::sleep_for(sleepinterval);

        sleepiter++;

        #ifdef __NVCC__
            if(sleepiter == sleepiterbegin)
                cudaProfilerStart(); CUERR;
        #endif


        #ifdef __NVCC__
        if(sleepiter == sleepiterend){
            cudaProfilerStop(); CUERR;

            for(int threadId = 0; threadId < nCorrectorThreads; threadId++){
                ecthreads[threadId].stopAndAbort = true;
                ecthreads[threadId].join();
            }

            std::exit(0);
        }
        #endif

    }
#endif

TIMERSTARTCPU(correction);

    for (auto& thread : ecthreads)
        thread.join();

#ifndef DO_PROFILE
    showProgress = false;
    progressThread.join();
    if(runtimeOptions.showProgress)
        printf("Progress: %3.2f %%\n", 100.00);
#endif

TIMERSTOPCPU(correction);

    //std::cout << "threads done" << std::endl;



    minhasher.destroy();
	readStorage.destroy();

    generators.clear();
    ecthreads.clear();
    readIsProcessedVector.clear();
    readIsProcessedVector.shrink_to_fit();

    std::cout << "begin merge" << std::endl;
    TIMERSTARTCPU(merge);

    mergeResultFiles(props.nReads, fileOptions.inputfile, fileOptions.format, tmpfiles, fileOptions.outputfile);

    TIMERSTOPCPU(merge);

    deleteFiles(tmpfiles);

    if(!correctionOptions.extractFeatures){
        std::vector<std::string> featureFiles(tmpfiles);
        for(auto& s : featureFiles)
            s = s + "_features";
        deleteFiles(featureFiles);
    }

    std::cout << "end merge" << std::endl;
}


}

#endif
