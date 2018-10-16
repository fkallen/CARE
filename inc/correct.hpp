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


//EXPERIMENTAL
#include "gpu_only_path/correct_only_gpu.hpp"
#include "gpu_only_path/readstorage_gpu.hpp"



#ifdef __NVCC__
#include <cuda_profiler_api.h>
#endif

#define USE_NVTX


#if defined USE_NVTX && defined __NVCC__
#include <nvToolsExt.h>

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff, 0xdeadbeef };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
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


template<bool indels>
struct alignment_result_type;

template<>
struct alignment_result_type<true>{using type = SGAResult;};

template<>
struct alignment_result_type<false>{using type = SHDResult;};

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

    //correctiondetail::Dist<std::int64_t, std::int64_t> candidateDistribution;
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

        //int currentStreamIndex = 0;
        //int nextStreamIndex = 1;

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

				if(indels){
					semi_global_alignment_canonical_get_results_batched(sgahandles[streamIndex],
										                    alignmentsbegin,
										                    alignmentsend,
										                    flagsbegin,
										                    flagsend,
                                                            bestSequenceStringsbegin,
                                                            bestSequenceStringsend,
										                    canUseGpu);
				}else{
					shifted_hamming_distance_canonical_get_results_batched(shdhandles[streamIndex],
											alignmentsbegin,
											alignmentsend,
											flagsbegin,
											flagsend,
                                            bestSequenceStringsbegin,
                                            bestSequenceStringsend,
											canUseGpu);
				}


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
						if(indels){
							tpc = std::chrono::system_clock::now();

							std::pair<GraphCorrectionResult, TaskTimings> res = correct(errorgraph,
														b,
														goodAlignmentProperties.maxErrorRate,
														correctionOptions.graphalpha,
														correctionOptions.graphx);

							tpd = std::chrono::system_clock::now();
							readcorrectionTimeTotal += tpd - tpc;

                            detailedCorrectionTimings += res.second;

                            auto& correctionResult = res.first;

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

                            detailedCorrectionTimings += res.second;

							/*
								features
							*/

							if(correctionOptions.extractFeatures){
                                #if 0
								std::vector<MSAFeature> MSAFeatures =  extractFeatures(pileupImage, b.fwdSequenceString,
																threadOpts.minhasher->minparams.k, 0.0,
																correctionOptions.estimatedCoverage);
#else
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

#endif
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

                POP_RANGE;

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
struct ErrorCorrectionThreadCombinedThisThat{
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

    //correctiondetail::Dist<std::int64_t, std::int64_t> candidateDistribution;
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
        if(isRunning) throw std::runtime_error("ErrorCorrectionThreadCombinedThisThat::run: Is already running.");
        isRunning = true;
        thread = std::move(std::thread(&ErrorCorrectionThreadCombinedThisThat::execute, this));
    }

    void join(){
        thread.join();
        isRunning = false;
    }

private:

	void execute() {

        struct Batch{
            std::vector<BatchElem_t> batchElems;

#ifdef __NVCC__
            cudaStream_t stream;
#endif

        };

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


        const bool canUseGpu = threadOpts.canUseGpu;
        constexpr int nStreams = 2;

		errorgraph::ErrorGraph errorgraph;
		pileup::PileupImage pileupImage(correctionOptions.m_coverage,
                                        correctionOptions.kmerlength,
                                        correctionOptions.estimatedCoverage);

		std::array<std::vector<BatchElem_t>, nStreams> batchElems;
        std::vector<SHDhandle> shdhandles(nStreams);
        std::vector<SGAhandle> sgahandles(nStreams);
		std::vector<ReadId_t> readIds = threadOpts.batchGen->getNextReadIds();

        //Task 1. Init batch
        auto init_batch = [&](int activeStreamIndex){
            tpc = std::chrono::system_clock::now();
            //fit vector size to actual batch size
            if (batchElems[activeStreamIndex].size() != readIds.size()) {
                batchElems[activeStreamIndex].resize(readIds.size(),
                                BatchElem_t(*threadOpts.readStorage,
                                            correctionOptions, max_candidates));
            }

            for(std::size_t i = 0; i < readIds.size(); i++){
                set_read_id(batchElems[activeStreamIndex][i], readIds[i]);
                nProcessedQueries++;
            }

            for(auto& b : batchElems[activeStreamIndex]){
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
        };

        //Task 2. determine candidate reads
        auto find_candidates = [&](int activeStreamIndex){

                tpa = std::chrono::system_clock::now();

                for(auto& b : batchElems[activeStreamIndex]){
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

                nProcessedReads += readIds.size();
                readIds = threadOpts.batchGen->getNextReadIds();
        };

        //Task 3. Copy batch to gpu
        auto copy_batch_to_gpu = [&](int activeStreamIndex){

            tpa = std::chrono::system_clock::now();

            auto activeBatchElementsEnd = std::partition(batchElems[activeStreamIndex].begin(),
                                                            batchElems[activeStreamIndex].end(),
                                                            [](const auto& b){return b.active;});

            if(std::distance(batchElems[activeStreamIndex].begin(), activeBatchElementsEnd) > 0){

                std::vector<const Sequence_t**> subjectsbegin;
                std::vector<const Sequence_t**> subjectsend;
                std::vector<typename std::vector<const Sequence_t*>::iterator> queriesbegin;
                std::vector<typename std::vector<const Sequence_t*>::iterator> queriesend;

                std::vector<int> queriesPerSubject;

                subjectsbegin.reserve(batchElems[activeStreamIndex].size());
                subjectsend.reserve(batchElems[activeStreamIndex].size());
                queriesbegin.reserve(batchElems[activeStreamIndex].size());
                queriesend.reserve(batchElems[activeStreamIndex].size());
                queriesPerSubject.reserve(batchElems[activeStreamIndex].size());

                //for(auto& b : batchElems[streamIndex]){
                for(auto it = batchElems[activeStreamIndex].begin(); it != activeBatchElementsEnd; ++it){
                    auto& b = *it;
                    subjectsbegin.emplace_back(&b.fwdSequence);
                    subjectsend.emplace_back(&b.fwdSequence + 1);
                    queriesbegin.emplace_back(b.fwdSequences.begin());
                    queriesend.emplace_back(b.fwdSequences.end());
                    queriesPerSubject.emplace_back(b.fwdSequences.size());
                }

                if(indels){
                    alignment_copy_to_device_async(sgahandles[activeStreamIndex],
                                                    subjectsbegin,
                                                    subjectsend,
                                                    queriesbegin,
                                                    queriesend,
                                                    queriesPerSubject,
                                                    canUseGpu);
                }else{
                    alignment_copy_to_device_async(shdhandles[activeStreamIndex],
                                                    subjectsbegin,
                                                    subjectsend,
                                                    queriesbegin,
                                                    queriesend,
                                                    queriesPerSubject,
                                                    canUseGpu);
                }
            }

            tpb = std::chrono::system_clock::now();
            getAlignmentsTimeTotal += tpb - tpa;
        };

        //Task 4. Perform alignments of batch. Async, if alignment on gpu
        auto perform_alignments = [&](int activeStreamIndex){
            tpa = std::chrono::system_clock::now();

            auto activeBatchElementsEnd = std::partition(batchElems[activeStreamIndex].begin(),
                                                        batchElems[activeStreamIndex].end(),
                                                        [](const auto& b){return b.active;});

            if(std::distance(batchElems[activeStreamIndex].begin(), activeBatchElementsEnd) > 0){

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

                subjectsbegin.reserve(batchElems[activeStreamIndex].size());
                subjectsend.reserve(batchElems[activeStreamIndex].size());
                queriesbegin.reserve(batchElems[activeStreamIndex].size());
                queriesend.reserve(batchElems[activeStreamIndex].size());
                alignmentsbegin.reserve(batchElems[activeStreamIndex].size());
                alignmentsend.reserve(batchElems[activeStreamIndex].size());
                flagsbegin.reserve(batchElems[activeStreamIndex].size());
                flagsend.reserve(batchElems[activeStreamIndex].size());
                bestSequenceStringsbegin.reserve(batchElems[activeStreamIndex].size());
                bestSequenceStringsend.reserve(batchElems[activeStreamIndex].size());
                queriesPerSubject.reserve(batchElems[activeStreamIndex].size());

                //for(auto& b : batchElems[streamIndex]){
                for(auto it = batchElems[activeStreamIndex].begin(); it != activeBatchElementsEnd; ++it){
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
                    device = semi_global_alignment_canonical_batched_async<Sequence_t>(sgahandles[activeStreamIndex],
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
                    device = shifted_hamming_distance_canonical_batched_async_withoutH2D<Sequence_t>(shdhandles[activeStreamIndex],
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

            tpb = std::chrono::system_clock::now();
            getAlignmentsTimeTotal += tpb - tpa;
        };

        //Task 6. Wait for (gpu-)alignment calculation to finish
        auto wait_for_alignments = [&](int activeStreamIndex){
            tpa = std::chrono::system_clock::now();

            std::vector<typename std::vector<BestAlignment_t>::iterator> flagsbegin;
            std::vector<typename std::vector<BestAlignment_t>::iterator> flagsend;

            std::vector<typename std::vector<AlignmentResult_t>::iterator> alignmentsbegin;
            std::vector<typename std::vector<AlignmentResult_t>::iterator> alignmentsend;

            std::vector<typename std::vector<std::string>::iterator> bestSequenceStringsbegin;
            std::vector<typename std::vector<std::string>::iterator> bestSequenceStringsend;

            alignmentsbegin.reserve(batchElems[activeStreamIndex].size());
            alignmentsend.reserve(batchElems[activeStreamIndex].size());
            flagsbegin.reserve(batchElems[activeStreamIndex].size());
            flagsend.reserve(batchElems[activeStreamIndex].size());
            bestSequenceStringsbegin.reserve(batchElems[activeStreamIndex].size());
            bestSequenceStringsend.reserve(batchElems[activeStreamIndex].size());

            auto activeBatchElementsEnd = std::partition(batchElems[activeStreamIndex].begin(),
                                                        batchElems[activeStreamIndex].end(),
                                                        [](const auto& b){return b.active;});

            for(auto it = batchElems[activeStreamIndex].begin(); it != activeBatchElementsEnd; ++it){
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

            if(indels){
                semi_global_alignment_canonical_get_results_batched(sgahandles[activeStreamIndex],
                                                        alignmentsbegin,
                                                        alignmentsend,
                                                        flagsbegin,
                                                        flagsend,
                                                        bestSequenceStringsbegin,
                                                        bestSequenceStringsend,
                                                        canUseGpu);
            }else{
                shifted_hamming_distance_canonical_get_results_batched(shdhandles[activeStreamIndex],
                                        alignmentsbegin,
                                        alignmentsend,
                                        flagsbegin,
                                        flagsend,
                                        bestSequenceStringsbegin,
                                        bestSequenceStringsend,
                                        canUseGpu);
            }


            tpb = std::chrono::system_clock::now();
            getAlignmentsTimeTotal += tpb - tpa;
        };

        auto correct_batch = [&](int activeStreamIndex){

            auto activeBatchElementsEnd = std::partition(batchElems[activeStreamIndex].begin(),
                                                        batchElems[activeStreamIndex].end(),
                                                        [](const auto& b){return b.active;});

            //check quality of alignments
            tpc = std::chrono::system_clock::now();
            //for(auto& b : batchElems[streamIndex]){
            for(auto it = batchElems[activeStreamIndex].begin(); it != activeBatchElementsEnd; ++it){
                auto& b = *it;
                if(b.active){
                    determine_good_alignments(b);
                }
            }

            tpd = std::chrono::system_clock::now();
            determinegoodalignmentsTime += tpd - tpc;

            tpc = std::chrono::system_clock::now();

            for(auto it = batchElems[activeStreamIndex].begin(); it != activeBatchElementsEnd; ++it){
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

            for(auto it = batchElems[activeStreamIndex].begin(); it != activeBatchElementsEnd; ++it){
                auto& b = *it;
                if(b.active){
                    if(indels){
                        tpc = std::chrono::system_clock::now();

                        std::pair<GraphCorrectionResult, TaskTimings> res = correct(errorgraph,
                                                    b,
                                                    goodAlignmentProperties.maxErrorRate,
                                                    correctionOptions.graphalpha,
                                                    correctionOptions.graphx);

                        tpd = std::chrono::system_clock::now();
                        readcorrectionTimeTotal += tpd - tpc;

                        detailedCorrectionTimings += res.second;

                        auto& correctionResult = res.first;

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

                        detailedCorrectionTimings += res.second;

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
        };


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

        int currentStreamIndex = 0;
        int nextStreamIndex = 1;
        bool firstIter = true;

		while(!stopAndAbort && !readIds.empty()){
            //In first iteration currentStreamIndex's candidates are not calculated yet and not transfered to gpu
            //-> Get candidates of batchElems[currentStreamIndex] and copy them to gpu
#if 1
            PUSH_RANGE("init_batch(currentStreamIndex)", 0);
            init_batch(currentStreamIndex);
            POP_RANGE;
            //cudaEventRecord(event[7], mybuffers.streams[currentStreamIndex]);
            PUSH_RANGE("find_candidates(currentStreamIndex)", 1);
            find_candidates(currentStreamIndex);
            POP_RANGE;
            //cudaEventRecord(event[8], mybuffers.streams[currentStreamIndex]);
            PUSH_RANGE("copy_batch_to_gpu(currentStreamIndex)", 2);
            copy_batch_to_gpu(currentStreamIndex);
            POP_RANGE;
            //cudaEventRecord(event[9], mybuffers.streams[currentStreamIndex]);
            PUSH_RANGE("perform_alignments(currentStreamIndex)", 3);
            perform_alignments(currentStreamIndex);
            POP_RANGE;
            PUSH_RANGE("wait_for_alignments(currentStreamIndex)" , 4);
            wait_for_alignments(currentStreamIndex);
            POP_RANGE;
            //cudaEventRecord(event[5], mybuffers.streams[currentStreamIndex]);
            PUSH_RANGE("correct_batch(currentStreamIndex)" , 5);
            correct_batch(currentStreamIndex);
            POP_RANGE;
#else
    //cudaEvent_t events[10];
    //for(int i = 0; i < 10; i++){
    //    cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
    //}
            if(firstIter){
                //cudaEventRecord(event[6], mybuffers.streams[currentStreamIndex]);
                PUSH_RANGE("init_batch(currentStreamIndex)", 0);
                init_batch(currentStreamIndex);
                POP_RANGE;
                //cudaEventRecord(event[7], mybuffers.streams[currentStreamIndex]);
                PUSH_RANGE("find_candidates(currentStreamIndex)", 1);
                find_candidates(currentStreamIndex);
                POP_RANGE;
                //cudaEventRecord(event[8], mybuffers.streams[currentStreamIndex]);
                PUSH_RANGE("copy_batch_to_gpu(currentStreamIndex)", 2);
                copy_batch_to_gpu(currentStreamIndex);
                POP_RANGE;
                //cudaEventRecord(event[9], mybuffers.streams[currentStreamIndex]);
                PUSH_RANGE("perform_alignments(currentStreamIndex)", 3);
                perform_alignments(currentStreamIndex);
                POP_RANGE;
            }

            //cudaEventRecord(event[0], mybuffers.streams[nextStreamIndex]);
            PUSH_RANGE("init_batch(nextStreamIndex)" , 0);
            init_batch(nextStreamIndex);
            POP_RANGE;
            //cudaEventRecord(event[1], mybuffers.streams[nextStreamIndex]);
            PUSH_RANGE("find_candidates(nextStreamIndex)" , 1);
            find_candidates(nextStreamIndex);
            POP_RANGE;
            //cudaEventRecord(event[2], mybuffers.streams[nextStreamIndex]);
            PUSH_RANGE("copy_batch_to_gpu(nextStreamIndex)" , 2);
            copy_batch_to_gpu(nextStreamIndex);
            POP_RANGE;
            //cudaEventRecord(event[3], mybuffers.streams[nextStreamIndex]);
            PUSH_RANGE("perform_alignments(nextStreamIndex)" , 3);
            perform_alignments(nextStreamIndex);
            POP_RANGE;

            //cudaEventRecord(event[4], mybuffers.streams[currentStreamIndex]);
            PUSH_RANGE("wait_for_alignments(currentStreamIndex)" , 4);
            wait_for_alignments(currentStreamIndex);
            POP_RANGE;
            //cudaEventRecord(event[5], mybuffers.streams[currentStreamIndex]);
            PUSH_RANGE("correct_batch(currentStreamIndex)" , 5);
            correct_batch(currentStreamIndex);
            POP_RANGE;

            std::swap(currentStreamIndex, nextStreamIndex);
#endif

            firstIter = false;

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
struct ErrorCorrectionThreadCombinedSplitTasks{
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

    struct Batch{
        std::vector<BatchElem_t> batchElems;
        SHDhandle* shdhandle;
        SGAhandle* sgahandle;
    };

    AlignmentOptions alignmentOptions;
    GoodAlignmentProperties goodAlignmentProperties;
    CorrectionOptions correctionOptions;
    CorrectionThreadOptions threadOpts;

    SequenceFileProperties fileProperties;

    //correctiondetail::Dist<std::int64_t, std::int64_t> candidateDistribution;
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
        if(isRunning) throw std::runtime_error("ErrorCorrectionThreadCombinedSplitTasks::run: Is already running.");
        isRunning = true;
        thread = std::move(std::thread(&ErrorCorrectionThreadCombinedSplitTasks::execute, this));
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

        std::mutex outfilelock;
        std::mutex featurefilelock;

		auto write_read = [&](const ReadId_t readId, const auto& sequence){
            std::lock_guard<std::mutex> m(outfilelock);

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


        const bool canUseGpu = threadOpts.canUseGpu;





        //Task 1. Init batch
        auto init_batch = [&](Batch& batch, const std::vector<ReadId_t>& readIds){
            std::chrono::time_point<std::chrono::system_clock> tpa = std::chrono::system_clock::now();

            auto& batchElems = batch.batchElems;

            //fit vector size to actual batch size
            //PUSH_RANGE("init_batch_resize", 7);
            if (batchElems.size() != readIds.size()) {
                std::cout << batchElems.size() << " " << readIds.size() << std::endl;
                batchElems.resize(readIds.size(),
                                BatchElem_t(*threadOpts.readStorage,
                                            correctionOptions, max_candidates));
            }
            //POP_RANGE;

            //PUSH_RANGE("init_batch_set_read_id", 6);
            for(std::size_t i = 0; i < readIds.size(); i++){
                set_read_id(batchElems[i], readIds[i]);
                nProcessedQueries++;
            }
            //POP_RANGE;

            //PUSH_RANGE("init_batch_set_flags_locked", 7);
            for(auto& b : batchElems){
                lock(b.readId);
                if ((*threadOpts.readIsCorrectedVector)[b.readId] == 0) {
                    (*threadOpts.readIsCorrectedVector)[b.readId] = 1;
                }else{
                    b.active = false;
                    nProcessedQueries--;
                }
                unlock(b.readId);
            }
            //POP_RANGE;

            std::chrono::time_point<std::chrono::system_clock> tpb = std::chrono::system_clock::now();
            //initIdTimeTotal += tpb - tpa;
            return tpb - tpa;
        };

        //Task 2. determine candidate reads
        auto find_candidates = [&](Batch& batch){
            auto& batchElems = batch.batchElems;

            std::chrono::time_point<std::chrono::system_clock> tpa = std::chrono::system_clock::now();

            for(auto& b : batchElems){
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

            std::chrono::time_point<std::chrono::system_clock> tpb = std::chrono::system_clock::now();
            //mapMinhashResultsToSequencesTimeTotal += tpb - tpa;
            return tpb - tpa;
        };

        //Task 3. Copy batch to gpu
        auto copy_batch_to_gpu = [&](Batch& batch){

            auto& shdhandle = *batch.shdhandle;
            auto& sgahandle = *batch.sgahandle;

            auto& batchElems = batch.batchElems;

            std::chrono::time_point<std::chrono::system_clock> tpa = std::chrono::system_clock::now();

            auto activeBatchElementsEnd = std::partition(batchElems.begin(),
                                                            batchElems.end(),
                                                            [](const auto& b){return b.active;});

            if(std::distance(batchElems.begin(), activeBatchElementsEnd) > 0){

                std::vector<const Sequence_t**> subjectsbegin;
                std::vector<const Sequence_t**> subjectsend;
                std::vector<typename std::vector<const Sequence_t*>::iterator> queriesbegin;
                std::vector<typename std::vector<const Sequence_t*>::iterator> queriesend;

                std::vector<int> queriesPerSubject;

                subjectsbegin.reserve(batchElems.size());
                subjectsend.reserve(batchElems.size());
                queriesbegin.reserve(batchElems.size());
                queriesend.reserve(batchElems.size());
                queriesPerSubject.reserve(batchElems.size());

                //for(auto& b : batchElems[streamIndex]){
                for(auto it = batchElems.begin(); it != activeBatchElementsEnd; ++it){
                    auto& b = *it;
                    subjectsbegin.emplace_back(&b.fwdSequence);
                    subjectsend.emplace_back(&b.fwdSequence + 1);
                    queriesbegin.emplace_back(b.fwdSequences.begin());
                    queriesend.emplace_back(b.fwdSequences.end());
                    queriesPerSubject.emplace_back(b.fwdSequences.size());
                }

                if(indels){
                    alignment_copy_to_device_async(sgahandle,
                                                    subjectsbegin,
                                                    subjectsend,
                                                    queriesbegin,
                                                    queriesend,
                                                    queriesPerSubject,
                                                    canUseGpu);
                }else{
                    alignment_copy_to_device_async(shdhandle,
                                                    subjectsbegin,
                                                    subjectsend,
                                                    queriesbegin,
                                                    queriesend,
                                                    queriesPerSubject,
                                                    canUseGpu);
                }
            }

            std::chrono::time_point<std::chrono::system_clock> tpb = std::chrono::system_clock::now();
            //getAlignmentsTimeTotal += tpb - tpa;
            return tpb - tpa;
        };

        //Task 4. Perform alignments of batch. Async, if alignment on gpu
        auto perform_alignments = [&](Batch& batch){
            auto& shdhandle = *batch.shdhandle;
            auto& sgahandle = *batch.sgahandle;

            auto& batchElems = batch.batchElems;

            std::chrono::time_point<std::chrono::system_clock> tpa = std::chrono::system_clock::now();

            auto activeBatchElementsEnd = std::partition(batchElems.begin(),
                                                        batchElems.end(),
                                                        [](const auto& b){return b.active;});

            if(std::distance(batchElems.begin(), activeBatchElementsEnd) > 0){

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

                subjectsbegin.reserve(batchElems.size());
                subjectsend.reserve(batchElems.size());
                queriesbegin.reserve(batchElems.size());
                queriesend.reserve(batchElems.size());
                alignmentsbegin.reserve(batchElems.size());
                alignmentsend.reserve(batchElems.size());
                flagsbegin.reserve(batchElems.size());
                flagsend.reserve(batchElems.size());
                bestSequenceStringsbegin.reserve(batchElems.size());
                bestSequenceStringsend.reserve(batchElems.size());
                queriesPerSubject.reserve(batchElems.size());

                for(auto it = batchElems.begin(); it != activeBatchElementsEnd; ++it){
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

                //AlignmentDevice device = AlignmentDevice::None;
                if(indels){
                    /*device = */semi_global_alignment_canonical_batched_async<Sequence_t>(sgahandle,
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
                    /*device = */shifted_hamming_distance_canonical_batched_async_withoutH2D<Sequence_t>(shdhandle,
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

                /*if(device == AlignmentDevice::CPU)
                    cpuAlignments++;
                else if (device == AlignmentDevice::GPU)
                    gpuAlignments++;*/

            }

            std::chrono::time_point<std::chrono::system_clock> tpb = std::chrono::system_clock::now();
            //getAlignmentsTimeTotal += tpb - tpa;
            return tpb - tpa;
        };

        //Task 6. Wait for (gpu-)alignment calculation to finish
        auto wait_for_alignments = [&](Batch& batch){
            auto& shdhandle = *batch.shdhandle;
            auto& sgahandle = *batch.sgahandle;

            auto& batchElems = batch.batchElems;
            std::chrono::time_point<std::chrono::system_clock> tpc = std::chrono::system_clock::now();

            std::vector<typename std::vector<BestAlignment_t>::iterator> flagsbegin;
            std::vector<typename std::vector<BestAlignment_t>::iterator> flagsend;

            std::vector<typename std::vector<AlignmentResult_t>::iterator> alignmentsbegin;
            std::vector<typename std::vector<AlignmentResult_t>::iterator> alignmentsend;

            std::vector<typename std::vector<std::string>::iterator> bestSequenceStringsbegin;
            std::vector<typename std::vector<std::string>::iterator> bestSequenceStringsend;

            alignmentsbegin.reserve(batchElems.size());
            alignmentsend.reserve(batchElems.size());
            flagsbegin.reserve(batchElems.size());
            flagsend.reserve(batchElems.size());
            bestSequenceStringsbegin.reserve(batchElems.size());
            bestSequenceStringsend.reserve(batchElems.size());

            auto activeBatchElementsEnd = std::partition(batchElems.begin(),
                                                        batchElems.end(),
                                                        [](const auto& b){return b.active;});

            for(auto it = batchElems.begin(); it != activeBatchElementsEnd; ++it){
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

            if(indels){
                semi_global_alignment_canonical_get_results_batched(sgahandle,
                                                        alignmentsbegin,
                                                        alignmentsend,
                                                        flagsbegin,
                                                        flagsend,
                                                        bestSequenceStringsbegin,
                                                        bestSequenceStringsend,
                                                        canUseGpu);
            }else{
                shifted_hamming_distance_canonical_get_results_batched(shdhandle,
                                        alignmentsbegin,
                                        alignmentsend,
                                        flagsbegin,
                                        flagsend,
                                        bestSequenceStringsbegin,
                                        bestSequenceStringsend,
                                        canUseGpu);
            }


            std::chrono::time_point<std::chrono::system_clock> tpd = std::chrono::system_clock::now();
            //getAlignmentsTimeTotal += tpd - tpc;
            return tpd - tpc;
        };

        auto correct_batch = [&](Batch& batch, errorgraph::ErrorGraph& errorgraph, pileup::PileupImage& pileupImage){

            auto& batchElems = batch.batchElems;

            auto activeBatchElementsEnd = std::partition(batchElems.begin(),
                                                        batchElems.end(),
                                                        [](const auto& b){return b.active;});

            //check quality of alignments
            std::chrono::time_point<std::chrono::system_clock> tpc = std::chrono::system_clock::now();
            //for(auto& b : batchElems[streamIndex]){
            for(auto it = batchElems.begin(); it != activeBatchElementsEnd; ++it){
                auto& b = *it;
                if(b.active){
                    determine_good_alignments(b);
                }
            }

            std::chrono::time_point<std::chrono::system_clock> tpd = std::chrono::system_clock::now();
            std::chrono::duration<double> determinegoodalignmentsTime = tpd - tpc;

            /*int* d_indices = nullptr;
            int* d_indices_per_subject = nullptr;
            int* d_indices_per_subject_prefixsum = nullptr;
            int* d_num_indices = nullptr;

            cudaMalloc(&d_indices, sizeof(int) * shdhandle.buffers.n_queries); CUERR;
            cudaMalloc(&d_indices_per_subject, sizeof(int) * shdhandle.buffers.n_subjects); CUERR;
            cudaMalloc(&d_indices_per_subject_prefixsum, sizeof(int) * shdhandle.buffers.n_subjects); CUERR;
            cudaMalloc(&d_num_indices, sizeof(int) * 1); CUERR;

            auto& shdhandle = batch.shdhandle;

            auto select_alignments_by_flag = [&](){

                auto select_alignment_op = [] __device__ (const BestAlignment_t& flag){
                    return flag != BestAlignment_t::None;
                };

                cub::TransformInputIterator<bool,decltype(select_alignment_op), BestAlignment_t*>
                            d_isGoodAlignment(shdhandle.buffers.d_bestAlignmentFlags,
                                              select_alignment_op);

                std::size_t temp_storage_bytes = 0;
                cub::DeviceSelect::Flagged(nullptr,
                                            temp_storage_bytes,
                                            cub::CountingInputIterator<int>(0),
                                            d_isGoodAlignment,
                                            d_indices,
                                            d_num_indices,
                                            shdhandle.buffers.n_queries,
                                            shdhandle.buffers.streams[0]); CUERR;


                void* d_temp_storage = nullptr;
                cudaMalloc(&d_temp_storage, temp_storage_bytes);

                cub::DeviceSelect::Flagged(dataArrays[streamIndex].d_temp_storage,
                    temp_storage_bytes,
                    cub::CountingInputIterator<int>(0),
                    d_isGoodAlignment,
                    d_indices,
                    d_num_indices,
                    shdhandle.buffers.n_queries,
                    shdhandle.buffers.streams[0]); CUERR;

                cudaFree(d_temp_storage); CUERR;
            };

            //Determine indices i < M where d_alignment_best_alignment_flags[i] != BestAlignment_t::None. this selects all good alignments
            select_alignments_by_flag();

            call_cuda_filter_alignments_by_mismatchratio_kernel_async(
                                    shdhandle.buffers.d_bestAlignmentFlags,
                                    shdhandle.buffers.d_results,
                                    d_indices,
                                    d_indices_per_subject,
                                    d_indices_per_subject_prefixsum,
                                    shdhandle.buffers.n_subjects,
                                    shdhandle.buffers.n_queries,
                                    d_num_indices,
                                    correctionOptions.estimatedErrorrate,
                                    correctionOptions.estimatedCoverage * correctionOptions.m_coverage,
                                    shdhandle.buffers.streams[0]);

            //determine indices of remaining alignments
            select_alignments_by_flag();

            cudaFree(d_indices); CUERR;
            cudaFree(d_indices_per_subject); CUERR;
            cudaFree(d_indices_per_subject_prefixsum); CUERR;
            cudaFree(d_num_indices); CUERR;*/

            tpc = std::chrono::system_clock::now();

            for(auto it = batchElems.begin(); it != activeBatchElementsEnd; ++it){
                auto& b = *it;
                if(b.active && hasEnoughGoodCandidates(b)){
                    if(b.active){
                        //move candidates which are used for correction to the front
                        prepare_good_candidates(b);
                        //auto tup = prepare_good_candidates(b);
                        //da += std::get<0>(tup);
                        //db += std::get<1>(tup);
                        //dc += std::get<2>(tup);
                    }
                }else{
                    //not enough good candidates. cannot correct this read.
                    b.active = false;
                }
            }

            tpd = std::chrono::system_clock::now();
            std::chrono::duration<double> fetchgoodcandidatesTime = tpd - tpc;
            std::chrono::duration<double> readcorrectionTimeTotal{0};
            TaskTimings detailedCorrectionTimings;

            for(auto it = batchElems.begin(); it != activeBatchElementsEnd; ++it){
                auto& b = *it;
                if(b.active){
                    if(indels){
                        tpc = std::chrono::system_clock::now();

                        std::pair<GraphCorrectionResult, TaskTimings> res = correct(errorgraph,
                                                    b,
                                                    goodAlignmentProperties.maxErrorRate,
                                                    correctionOptions.graphalpha,
                                                    correctionOptions.graphx);

                        tpd = std::chrono::system_clock::now();
                        readcorrectionTimeTotal += tpd - tpc;

                        detailedCorrectionTimings += res.second;

                        auto& correctionResult = res.first;

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

                        detailedCorrectionTimings += res.second;

                        /*
                            features
                        */

                        if(correctionOptions.extractFeatures){
                            std::vector<MSAFeature> MSAFeatures =  extractFeatures(pileupImage, b.fwdSequenceString,
                                                            threadOpts.minhasher->minparams.k, 0.0,
                                                            correctionOptions.estimatedCoverage);

                            if(MSAFeatures.size() > 0){
                                std::lock_guard<std::mutex> m(featurefilelock);

                                for(const auto& msafeature : MSAFeatures){
                                    featurestream << b.readId << '\t' << msafeature.position << '\n';
                                    featurestream << msafeature << '\n';
                                }

                            }
                        }
                        auto& correctionResult = res.first;

                        //avgsupportfail += correctionResult.stats.failedAvgSupport;
                        //minsupportfail += correctionResult.stats.failedMinSupport;
                        //mincoveragefail += correctionResult.stats.failedMinCoverage;
                        //verygoodalignment += correctionResult.stats.isHQ;

                        if(correctionResult.isCorrected){
                            write_read(b.readId, correctionResult.correctedSequence);
                            lock(b.readId);
                            (*threadOpts.readIsCorrectedVector)[b.readId] = 1;
                            unlock(b.readId);
                        }

                        for(const auto& correctedCandidate : correctionResult.correctedCandidates){
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

            return std::tuple<std::chrono::duration<double>,
                    std::chrono::duration<double>,
                    std::chrono::duration<double>,
                    TaskTimings>{determinegoodalignmentsTime,
                                fetchgoodcandidatesTime,
                                readcorrectionTimeTotal,
                                detailedCorrectionTimings};
        };

        constexpr int max_batches_in_flight = 20;
        //constexpr int nStreams = 2;

        ThreadsafeBuffer<Batch*, max_batches_in_flight> freeBatchQueue;
        ThreadsafeBuffer<Batch*, max_batches_in_flight> alignerBatchQueue;

        std::vector<SHDhandle> shdhandles(max_batches_in_flight);
        std::vector<SGAhandle> sgahandles(max_batches_in_flight);


        std::vector<Batch> batches(max_batches_in_flight);
        for(int i = 0; i < max_batches_in_flight; i++){
            batches[i].shdhandle = &shdhandles[i];
            batches[i].sgahandle = &sgahandles[i];

            freeBatchQueue.add(&batches[i]);
        }



        std::vector<ReadId_t> readIds = threadOpts.batchGen->getNextReadIds();

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



        //this thread finds candidates and launches alignments
        auto producerfunction = [&](){

            std::chrono::duration<double> initIdTimeTotal{0};
            std::chrono::duration<double> mapMinhashResultsToSequencesTimeTotal{0};
            std::chrono::duration<double> getAlignmentsTimeTotal{0};
            std::chrono::duration<double> initIdTimeTotal3{0};


            while(!stopAndAbort && !readIds.empty()){

                PUSH_RANGE("freeBatchQueue.getNew()" , 7);
                auto popresult = freeBatchQueue.getNew();
                POP_RANGE;

                if(popresult.foreverEmpty)
                    break;
                Batch* batch = popresult.value;

                PUSH_RANGE("init_batch", 0);
                initIdTimeTotal += init_batch(*batch, readIds);
                POP_RANGE;

                PUSH_RANGE("find_candidates", 1);
                mapMinhashResultsToSequencesTimeTotal += find_candidates(*batch);
                POP_RANGE;

                PUSH_RANGE("copy_batch_to_gpu", 2);
                getAlignmentsTimeTotal += copy_batch_to_gpu(*batch);
                POP_RANGE;

                PUSH_RANGE("perform_alignments", 3);
                getAlignmentsTimeTotal += perform_alignments(*batch);
                POP_RANGE;

                PUSH_RANGE("alignerBatchQueue.add(batch)" , 6);
                alignerBatchQueue.add(batch);
                POP_RANGE;

                nProcessedReads += readIds.size();
                readIds = threadOpts.batchGen->getNextReadIds();
            }

            alignerBatchQueue.done();
        };

        //the current thread waits for alignment results and corrects batches
        auto correctionfunction = [&](){
            std::chrono::duration<double> getAlignmentsTimeTotal{0};
            std::chrono::duration<double> determinegoodalignmentsTime{0};
            std::chrono::duration<double> fetchgoodcandidatesTime{0};
            std::chrono::duration<double> readcorrectionTimeTotal{0};
            TaskTimings detailedCorrectionTimings;

            errorgraph::ErrorGraph errorgraph;
            pileup::PileupImage pileupImage(correctionOptions.m_coverage,
                                            correctionOptions.kmerlength,
                                            correctionOptions.estimatedCoverage);

            auto popresult = alignerBatchQueue.getNew();


            while(!stopAndAbort && !popresult.foreverEmpty){
                    Batch* batch = popresult.value;

                    PUSH_RANGE("wait_for_alignments" , 4);
                    getAlignmentsTimeTotal += wait_for_alignments(*batch);
                    POP_RANGE;

                    PUSH_RANGE("correct_batch" , 5);
                    auto timingstuple = correct_batch(*batch, errorgraph, pileupImage);

                    determinegoodalignmentsTime += std::get<0>(timingstuple);
                    fetchgoodcandidatesTime += std::get<1>(timingstuple);
                    readcorrectionTimeTotal += std::get<2>(timingstuple);
                    detailedCorrectionTimings += std::get<3>(timingstuple);
                    POP_RANGE;

                    PUSH_RANGE("freeBatchQueue.add(batch)" , 7);
                    freeBatchQueue.add(batch);
                    POP_RANGE;

                    PUSH_RANGE("alignerBatchQueue.getNew()" , 6);
                    popresult = alignerBatchQueue.getNew();
                    POP_RANGE;
            }
        };

        std::thread producer1(producerfunction);
        //std::thread producer2(producerfunction);
        //std::thread correctionworker1(correctionfunction);
        //std::thread correctionworker2(correctionfunction);
        //std::thread correctionworker3(correctionfunction);

        correctionfunction();

        producer1.join();
        //producer2.join();
        //correctionworker1.join();
        //correctionworker2.join();
        //correctionworker3.join();

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















#if 0

template<class minhasher_t,
		 class readStorage_t,
		 bool indels>
void correct(const MinhashOptions& minhashOptions,
				  const AlignmentOptions& alignmentOptions,
				  const GoodAlignmentProperties& goodAlignmentProperties,
				  const CorrectionOptions& correctionOptions,
				  const RuntimeOptions& runtimeOptions,
				  const FileOptions& fileOptions,
                  const SequenceFileProperties& props,
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

#if 0
#if 1
	using ErrorCorrectionThread_t = ErrorCorrectionThreadCombined<Minhasher_t, ReadStorage_t, indels>;
#else

    //using ErrorCorrectionThread_t = ErrorCorrectionThreadCombinedSplitTasks<Minhasher_t, ReadStorage_t, indels>;
    using ErrorCorrectionThread_t = ErrorCorrectionThreadCombinedThisThat<Minhasher_t, ReadStorage_t, indels>;
#endif
#else
	using ErrorCorrectionThread_t = gpu::ErrorCorrectionThreadOnlyGPU<Minhasher_t, ReadStorage_t, BatchGenerator<ReadId_t>>;
#endif

//#define DO_PROFILE

#if 0
    const int nCorrectorThreads = deviceIds.size() == 0 ? runtimeOptions.nCorrectorThreads
                        : std::min(runtimeOptions.nCorrectorThreads, maxCPUThreadsPerGPU * int(deviceIds.size()));
#else
	const int nCorrectorThreads = 1;
#endif

	std::cout << "Using " << nCorrectorThreads << " corrector threads" << std::endl;



      // initialize qscore-to-weight lookup table
  	init_weights();

    //SequenceFileProperties props = getSequenceFileProperties(fileOptions.inputfile, fileOptions.format);

    /*
        Make candidate statistics
    */

    std::uint64_t max_candidates = runtimeOptions.max_candidates;
    //std::uint64_t max_candidates = std::numeric_limits<std::uint64_t>::max();

    if(max_candidates == 0){
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

        const std::uint64_t estimatedMeanAlignedCandidates = candidateDistribution.max;
        const std::uint64_t estimatedDeviationAlignedCandidates = candidateDistribution.stddev;
        const std::uint64_t estimatedAlignmentCountThreshold = estimatedMeanAlignedCandidates
                                                        + 2.5 * estimatedDeviationAlignedCandidates;

        max_candidates = estimatedAlignmentCountThreshold;
    }

    std::cout << "Using candidate cutoff: " << max_candidates << std::endl;

    /*
        Spawn correction threads
    */

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
        //threadOpts.gpuThresholdSHD = deviceIds.size() == 0 ? 0 : gpuThresholdsSHD[threadId % deviceIds.size()];
        //threadOpts.gpuThresholdSGA = deviceIds.size() == 0 ? 0 : gpuThresholdsSGA[threadId % deviceIds.size()];
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
        ecthreads[threadId].max_candidates = max_candidates;

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
    constexpr int sleepiterend = 3;

    int sleepiter = 0;

    std::chrono::duration<int> sleepinterval = std::chrono::seconds(2);

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

            for(int deviceId : deviceIds){
				cudaSetDevice(deviceId); CUERR;
				cudaDeviceReset(); CUERR;
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



#else


template<class minhasher_t,
		 class readStorage_t,
		 bool indels>
void correct(const MinhashOptions& minhashOptions,
				  const AlignmentOptions& alignmentOptions,
				  const GoodAlignmentProperties& goodAlignmentProperties,
				  const CorrectionOptions& correctionOptions,
				  const RuntimeOptions& runtimeOptions,
				  const FileOptions& fileOptions,
                  const SequenceFileProperties& sequenceFileProperties,
                  minhasher_t& minhasher,
                  readStorage_t& readStorage,
				  std::vector<char>& readIsCorrectedVector,
				  std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
				  std::size_t nLocksForProcessedFlags,
				  const std::vector<int>& deviceIds){

	assert(indels == false);

	using Minhasher_t = minhasher_t;
	using ReadStorage_t = readStorage_t;
	using Sequence_t = typename ReadStorage_t::Sequence_t;
	using ReadId_t = typename ReadStorage_t::ReadId_t;
    using GPUReadStorage_t = GPUReadStorage<ReadStorage_t>;

	using CPUErrorCorrectionThread_t = ErrorCorrectionThreadCombined<Minhasher_t, ReadStorage_t, indels>;

	using GPUErrorCorrectionThread_t = gpu::ErrorCorrectionThreadOnlyGPU<Minhasher_t, ReadStorage_t, GPUReadStorage_t, care::gpu::BatchGenerator<ReadId_t>>;


//#define DO_PROFILE

#if 1
    const int nCorrectorThreads = deviceIds.size() == 0 ? runtimeOptions.nCorrectorThreads
                        : std::min(runtimeOptions.nCorrectorThreads, maxCPUThreadsPerGPU * int(deviceIds.size()));
#else
	const int nCorrectorThreads = 1;
#endif

	std::cout << "Using " << nCorrectorThreads << " corrector threads" << std::endl;



      // initialize qscore-to-weight lookup table
  	init_weights();

    //SequenceFileProperties sequenceFileProperties = getSequenceFileProperties(fileOptions.inputfile, fileOptions.format);

    /*
        Make candidate statistics
    */

    std::uint64_t max_candidates = runtimeOptions.max_candidates;
    //std::uint64_t max_candidates = std::numeric_limits<std::uint64_t>::max();

    if(max_candidates == 0){
        std::cout << "estimating candidate cutoff" << std::endl;

        correctiondetail::Dist<std::int64_t, std::int64_t> candidateDistribution;

        {
            TIMERSTARTCPU(candidateestimation);
            std::map<std::int64_t, std::int64_t> candidateHistogram
                    = correctiondetail::getCandidateCountHistogram(minhasher,
                                                readStorage,
                                                sequenceFileProperties.nReads / 10,
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

        const std::uint64_t estimatedMeanAlignedCandidates = candidateDistribution.max;
        const std::uint64_t estimatedDeviationAlignedCandidates = candidateDistribution.stddev;
        const std::uint64_t estimatedAlignmentCountThreshold = estimatedMeanAlignedCandidates
                                                        + 2.5 * estimatedDeviationAlignedCandidates;

        max_candidates = estimatedAlignmentCountThreshold;
    }

    std::cout << "Using candidate cutoff: " << max_candidates << std::endl;

    /*
        Spawn correction threads
    */

    std::vector<std::string> tmpfiles;
    for(int i = 0; i < nCorrectorThreads; i++){
        tmpfiles.emplace_back(fileOptions.outputfile + "_tmp_" + std::to_string(1000 + i));
    }

    int nGpuThreads = std::min(nCorrectorThreads, 1 * int(deviceIds.size()));
	int nCpuThreads = nCorrectorThreads - nGpuThreads;

    std::vector<BatchGenerator<ReadId_t>> cpubatchgenerators(nCpuThreads);
	std::vector<care::gpu::BatchGenerator<ReadId_t>> gpubatchgenerators(nGpuThreads);

    std::vector<CPUErrorCorrectionThread_t> cpucorrectorThreads(nCpuThreads);
	std::vector<GPUErrorCorrectionThread_t> gpucorrectorThreads(nGpuThreads);
    std::vector<char> readIsProcessedVector(readIsCorrectedVector);
    std::mutex writelock;

	std::uint64_t ncpuReads = nCpuThreads > 0 ? std::uint64_t(sequenceFileProperties.nReads / 7.0) : 0;
	std::uint64_t ngpuReads = sequenceFileProperties.nReads - ncpuReads;
	std::uint64_t nReadsPerGPU = SDIV(ngpuReads, nGpuThreads);

	std::cout << "nCpuThreads: " << nCpuThreads << ", nGpuThreads: " << nGpuThreads << std::endl;
	std::cout << "ncpuReads: " << ncpuReads << ", ngpuReads: " << ngpuReads << std::endl;

	for(int threadId = 0; threadId < nCpuThreads; threadId++){

        cpubatchgenerators[threadId] = BatchGenerator<ReadId_t>(ncpuReads, correctionOptions.batchsize, threadId, nCpuThreads);
        typename CPUErrorCorrectionThread_t::CorrectionThreadOptions threadOpts;
        threadOpts.threadId = threadId;
        threadOpts.deviceId = 0;
        threadOpts.canUseGpu = false;
        threadOpts.outputfile = tmpfiles[threadId];
        threadOpts.batchGen = &cpubatchgenerators[threadId];
        threadOpts.minhasher = &minhasher;
        threadOpts.readStorage = &readStorage;
        threadOpts.coutLock = &writelock;
        threadOpts.readIsProcessedVector = &readIsProcessedVector;
        threadOpts.readIsCorrectedVector = &readIsCorrectedVector;
        threadOpts.locksForProcessedFlags = locksForProcessedFlags.get();
        threadOpts.nLocksForProcessedFlags = nLocksForProcessedFlags;

        cpucorrectorThreads[threadId].alignmentOptions = alignmentOptions;
        cpucorrectorThreads[threadId].goodAlignmentProperties = goodAlignmentProperties;
        cpucorrectorThreads[threadId].correctionOptions = correctionOptions;
        cpucorrectorThreads[threadId].threadOpts = threadOpts;
        cpucorrectorThreads[threadId].fileProperties = sequenceFileProperties;
        cpucorrectorThreads[threadId].max_candidates = max_candidates;

        cpucorrectorThreads[threadId].run();
    }

    GPUReadStorage_t gpuReadStorage;
    bool canUseGPUReadStorage = true;
    /*GPUReadStorageType bestGPUReadStorageType = GPUReadStorage_t::getBestPossibleType(readStorage,
                                                                            Sequence_t::getNumBytes(fileProperties.maxSequenceLength),
                                                                            fileProperties.maxSequenceLength,
                                                                            0.8f,
                                                                            threadOpts.deviceId);


    if(bestGPUReadStorageType != GPUReadStorageType::None){
        //bestGPUReadStorageType = GPUReadStorageType::Sequences;

        gpuReadStorage = GPUReadStorage_t::createFrom(readStorage,
                                                        bestGPUReadStorageType,
                                                        Sequence_t::getNumBytes(fileProperties.maxSequenceLength),
                                                        fileProperties.maxSequenceLength,
                                                        threadOpts.deviceId);

        canUseGPUReadStorage = true;
        std::cout << "Using gpu read storage, type " << GPUReadStorage_t::nameOf(bestGPUReadStorageType) << std::endl;
    }*/
    std::cout << "External gpu read storage" << std::endl;
    gpuReadStorage = GPUReadStorage_t::createFrom(readStorage,
                                                Sequence_t::getNumBytes(sequenceFileProperties.maxSequenceLength),
                                                sequenceFileProperties.maxSequenceLength,
                                                0.8f,
                                                true,
                                                deviceIds.size() == 0 ? -1 : deviceIds[0]);

    std::cout << "Sequence Type: " << gpuReadStorage.getNameOfSequenceType() << std::endl;
    //std::cout << "Sequence length Type: " << gpuReadStorage.getNameOfSequenceLengthType() << std::endl;
    std::cout << "Quality Type: " << gpuReadStorage.getNameOfQualityType() << std::endl;


    for(int threadId = 0; threadId < nGpuThreads; threadId++){

        gpubatchgenerators[threadId] = care::gpu::BatchGenerator<ReadId_t>(ncpuReads + threadId * nReadsPerGPU,
                                                                            std::min(sequenceFileProperties.nReads,
                                                                            ncpuReads + (threadId+1) * nReadsPerGPU));
        typename GPUErrorCorrectionThread_t::CorrectionThreadOptions threadOpts;
        threadOpts.threadId = threadId;
        threadOpts.deviceId = deviceIds.size() == 0 ? -1 : deviceIds[threadId % deviceIds.size()];
        threadOpts.canUseGpu = runtimeOptions.canUseGpu;
        threadOpts.outputfile = tmpfiles[nCpuThreads + threadId];
        threadOpts.batchGen = &gpubatchgenerators[threadId];
        threadOpts.minhasher = &minhasher;
        threadOpts.readStorage = &readStorage;
        threadOpts.gpuReadStorage = &gpuReadStorage;
        threadOpts.canUseGPUReadStorage = canUseGPUReadStorage;
        threadOpts.coutLock = &writelock;
        threadOpts.readIsProcessedVector = &readIsProcessedVector;
        threadOpts.readIsCorrectedVector = &readIsCorrectedVector;
        threadOpts.locksForProcessedFlags = locksForProcessedFlags.get();
        threadOpts.nLocksForProcessedFlags = nLocksForProcessedFlags;

        gpucorrectorThreads[threadId].alignmentOptions = alignmentOptions;
        gpucorrectorThreads[threadId].goodAlignmentProperties = goodAlignmentProperties;
        gpucorrectorThreads[threadId].correctionOptions = correctionOptions;
        gpucorrectorThreads[threadId].threadOpts = threadOpts;
        gpucorrectorThreads[threadId].fileProperties = sequenceFileProperties;
        gpucorrectorThreads[threadId].max_candidates = max_candidates;

        gpucorrectorThreads[threadId].run();
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

            for(int i = 0; i < nCpuThreads; i++){
                correctorProgress += cpucorrectorThreads[i].nProcessedReads;
            }

            for(int i = 0; i < nGpuThreads; i++){
                correctorProgress += gpucorrectorThreads[i].nProcessedReads;
            }

            progress = correctorProgress;

            printf("Progress: %3.2f %% %10u %10lu (Runtime: %03d:%02d:%02d)\r",
                    ((progress * 1.0 / sequenceFileProperties.nReads) * 100.0),
                    correctorProgress, sequenceFileProperties.nReads,
                    int(std::chrono::duration_cast<std::chrono::hours>(runtime).count()),
                    int(std::chrono::duration_cast<std::chrono::minutes>(runtime).count()) % 60,
                    int(runtime.count()) % 60);
            std::cout << std::flush;

            if(progress < sequenceFileProperties.nReads){
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
    std::chrono::duration<int> sleepinterval = std::chrono::seconds(1);

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

			for(int i = 0; i < nCpuThreads; i++){
                cpucorrectorThreads[i].stopAndAbort = true;
				cpucorrectorThreads[i].join();
            }

            for(int i = 0; i < nGpuThreads; i++){
                gpucorrectorThreads[i].stopAndAbort = true;
				gpucorrectorThreads[i].join();
            }

            std::exit(0);
        }
        #endif

    }
#endif

TIMERSTARTCPU(correction);

    for (auto& thread : cpucorrectorThreads)
        thread.join();

	for (auto& thread : gpucorrectorThreads)
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

    if(canUseGPUReadStorage){
        GPUReadStorage_t::destroy(gpuReadStorage);
    }

   // generators.clear();
   // ecthreads.clear();
    readIsProcessedVector.clear();
    readIsProcessedVector.shrink_to_fit();

    std::cout << "begin merge" << std::endl;
    TIMERSTARTCPU(merge);

    mergeResultFiles(sequenceFileProperties.nReads, fileOptions.inputfile, fileOptions.format, tmpfiles, fileOptions.outputfile);

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




#endif


}

#endif
