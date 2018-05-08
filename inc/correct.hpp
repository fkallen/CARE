#ifndef CARE_CORRECT_HPP
#define CARE_CORRECT_HPP

#include "options.hpp"

#include "alignmentwrapper.hpp"
#include "correctionwrapper.hpp"

#include "batchelem.hpp"
#include "graph.hpp"
#include "pileup.hpp"
#include "sequence.hpp"
#include "sequencefileio.hpp"
#include "qualityscoreweights.hpp"
#include "tasktiming.hpp"
#include "concatcontainer.hpp"



#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <thread>
#include <future>

//#define DO_PROFILE

#ifdef __NVCC__
#include <cuda_profiler_api.h>
#endif

namespace care{

    constexpr int maxCPUThreadsPerGPU = 8;

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

		std::vector<SHDhandle> shdhandles(correctionOptions.batchsize);
		for(auto& handle : shdhandles){
			init_SHDhandle(handle,
							threadOpts.deviceId,
							fileProperties.maxSequenceLength,
							SDIV(fileProperties.maxSequenceLength, 4),
							threadOpts.gpuThresholdSHD);
		}

		pileup::PileupImage pileupImage(correctionOptions.m_coverage, correctionOptions.kmerlength);

		std::vector<BatchElem_t> batchElems;
		std::vector<ReadId_t> readIds = threadOpts.batchGen->getNextReadIds();

		std::uint64_t cpuAlignments = 0;
		std::uint64_t gpuAlignments = 0;
		//std::uint64_t savedAlignments = 0;
		//std::uint64_t performedAlignments = 0;

		const std::uint64_t estimatedMeanAlignedCandidates = candidateDistribution.max;
		const std::uint64_t estimatedDeviationAlignedCandidates = candidateDistribution.stddev;
		const std::uint64_t estimatedAlignmentCountThreshold = estimatedMeanAlignedCandidates
														+ 2.5 * estimatedDeviationAlignedCandidates;

        const std::uint64_t max_candidates = estimatedAlignmentCountThreshold * correctionOptions.estimatedCoverage;

        constexpr bool canUseGpu = true;

		while(!stopAndAbort &&!readIds.empty()){

			//fit vector size to actual batch size
			if (batchElems.size() != readIds.size()) {
				batchElems.resize(readIds.size(),
								BatchElem_t(*threadOpts.readStorage,
											correctionOptions));
			}

			for(std::size_t i = 0; i < readIds.size(); i++){
				set_read_id(batchElems[i], readIds[i]);
				nProcessedQueries++;
			}

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

			std::partition(batchElems.begin(), batchElems.end(), [](const auto& b){return b.active;});


            //pipelining to overlap gpu alignment computation with retrieval of canidates
            for(std::size_t batchindex = 0; batchindex < batchElems.size(); batchindex++){
                auto& b = batchElems[batchindex];

                tpa = std::chrono::system_clock::now();

	             // get query data, determine candidates via minhashing, get candidate data
				if(b.active){
                    findCandidates(b, [&, this](const std::string& sequencestring){
                        return threadOpts.minhasher->getCandidates(sequencestring, max_candidates);
                    });

                    //don't correct candidates with more than estimatedAlignmentCountThreshold alignments
                    if(b.n_unique_candidates > estimatedAlignmentCountThreshold)
    					b.active = false;
				}

                tpb = std::chrono::system_clock::now();
                mapMinhashResultsToSequencesTimeTotal += tpb - tpa;

			    tpa = std::chrono::system_clock::now();

                //start alignment
				if(b.active && !hasEnoughGoodCandidates(b)){

                    auto subjectsBegin = &b.fwdSequence;
                    auto subjectsEnd = subjectsBegin + 1;
                    auto queries = make_concat_container(b.fwdSequences.cbegin(), b.fwdSequences.cend(),
                                                      b.revcomplSequences.cbegin(), b.revcomplSequences.cend());

                    auto alignments = make_concat_container(b.fwdAlignments.begin(), b.fwdAlignments.end(),
                                                        b.revcomplAlignments.begin(), b.revcomplAlignments.end());

                    AlignmentDevice device = shifted_hamming_distance_async<Sequence_t>(shdhandles[batchindex],
                                                                    subjectsBegin,
                                                                    subjectsEnd,
                                                                    queries.begin(),
                                                                    queries.end(),
                                                                    alignments.begin(),
                                                                    alignments.end(),
                                                                    {int(std::distance(queries.begin(), queries.end()))},
                                                                    goodAlignmentProperties.min_overlap,
                                                                    goodAlignmentProperties.maxErrorRate,
                                                                    goodAlignmentProperties.min_overlap_ratio,
                                                                    canUseGpu);

					if(device == AlignmentDevice::CPU)
						cpuAlignments++;
					else if (device == AlignmentDevice::GPU)
						gpuAlignments++;
				}
            }

			//get results
            for(std::size_t batchindex = 0; batchindex < batchElems.size(); batchindex++){
                auto& b = batchElems[batchindex];
				if(b.active){
                    auto alignments = make_concat_container(b.fwdAlignments.begin(), b.fwdAlignments.end(),
                                                        b.revcomplAlignments.begin(), b.revcomplAlignments.end());
                    shifted_hamming_distance_get_results(shdhandles[batchindex],
                                                    alignments.begin(),
                                                    alignments.end(),
                                                    canUseGpu);
				}
			}
			tpb = std::chrono::system_clock::now();
			getAlignmentsTimeTotal += tpb - tpa;

			//check quality of alignments
			tpc = std::chrono::system_clock::now();
			for(auto& b : batchElems){
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


			for(auto& b : batchElems){
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

			for(auto& b : batchElems){
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
						const int count = b.candidateCounts[correctedCandidate.index];
						for(int f = 0; f < count; f++){
							ReadId_t candidateId = b.candidateIds[b.candidateCountsPrefixSum[correctedCandidate.index] + f];
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
								if (b.bestIsForward[correctedCandidate.index])
									write_read(candidateId, correctedCandidate.sequence);
								else {
									//correctedCandidate.sequence is reverse complement, make reverse complement again
									const std::string fwd = SequenceGeneral(correctedCandidate.sequence, false).reverseComplement().toString();
									write_read(candidateId, fwd);
								}
							}
						}
					}
				}
			}

			// update local progress
			nProcessedReads += readIds.size();

			readIds = threadOpts.batchGen->getNextReadIds();

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

	#if 1
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

		std::vector<SGAhandle> sgahandles(correctionOptions.batchsize);
		for(auto& handle : sgahandles){
			init_SGAhandle(handle,
							threadOpts.deviceId,
							fileProperties.maxSequenceLength,
							SDIV(fileProperties.maxSequenceLength, 4),
							threadOpts.gpuThresholdSGA);
		}

		errorgraph::ErrorGraph errorgraph;

		std::vector<BatchElem_t> batchElems;
		std::vector<ReadId_t> readIds = threadOpts.batchGen->getNextReadIds();

		std::uint64_t cpuAlignments = 0;
		std::uint64_t gpuAlignments = 0;
		//std::uint64_t savedAlignments = 0;
		//std::uint64_t performedAlignments = 0;

		const std::uint64_t estimatedMeanAlignedCandidates = candidateDistribution.max;
		const std::uint64_t estimatedDeviationAlignedCandidates = candidateDistribution.stddev;
		const std::uint64_t estimatedAlignmentCountThreshold = estimatedMeanAlignedCandidates
														+ 2.5 * estimatedDeviationAlignedCandidates;
        const std::uint64_t max_candidates = estimatedAlignmentCountThreshold * correctionOptions.estimatedCoverage;

        constexpr bool canUseGpu = true;

		while(!stopAndAbort &&!readIds.empty()){

			//fit vector size to actual batch size
			if (batchElems.size() != readIds.size()) {
				batchElems.resize(readIds.size(),
								BatchElem_t(*threadOpts.readStorage,
											correctionOptions));
			}

			for(std::size_t i = 0; i < readIds.size(); i++){
				set_read_id(batchElems[i], readIds[i]);
				nProcessedQueries++;
			}

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

			std::partition(batchElems.begin(), batchElems.end(), [](const auto& b){return b.active;});

            //pipelining to overlap gpu alignment computation with retrieval of canidates
            for(std::size_t batchindex = 0; batchindex < batchElems.size(); batchindex++){
                auto& b = batchElems[batchindex];

                tpa = std::chrono::system_clock::now();

	             // get query data, determine candidates via minhashing, get candidate data
				if(b.active){
                    findCandidates(b, [&, this](const std::string& sequencestring){
                        return threadOpts.minhasher->getCandidates(sequencestring, max_candidates);
                    });

                    //don't correct candidates with more than estimatedAlignmentCountThreshold alignments
                    if(b.n_unique_candidates > estimatedAlignmentCountThreshold)
    					b.active = false;
				}

                tpb = std::chrono::system_clock::now();
                mapMinhashResultsToSequencesTimeTotal += tpb - tpa;

			    tpa = std::chrono::system_clock::now();

                //start alignment
				if(b.active && !hasEnoughGoodCandidates(b)){

                    auto subjectsBegin = &b.fwdSequence;
                    auto subjectsEnd = subjectsBegin + 1;
                    auto queries = make_concat_container(b.fwdSequences.cbegin(), b.fwdSequences.cend(),
                                                      b.revcomplSequences.cbegin(), b.revcomplSequences.cend());

                    auto alignments = make_concat_container(b.fwdAlignments.begin(), b.fwdAlignments.end(),
                                                        b.revcomplAlignments.begin(), b.revcomplAlignments.end());


                    AlignmentDevice device = semi_global_alignment_async<Sequence_t>(sgahandles[batchindex],
                                                                    subjectsBegin,
                                                                    subjectsEnd,
                                                                    queries.begin(),
                                                                    queries.end(),
                                                                    alignments.begin(),
                                                                    alignments.end(),
                                                                    {int(std::distance(queries.begin(), queries.end()))},
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
			}

			//get results
            for(std::size_t batchindex = 0; batchindex < batchElems.size(); batchindex++){
                auto& b = batchElems[batchindex];
				if(b.active){
                    auto alignments = make_concat_container(b.fwdAlignments.begin(), b.fwdAlignments.end(),
                                                        b.revcomplAlignments.begin(), b.revcomplAlignments.end());
                    semi_global_alignment_get_results(sgahandles[batchindex],
                                                    alignments.begin(),
                                                    alignments.end(),
                                                    canUseGpu);
				}
			}
			tpb = std::chrono::system_clock::now();
			getAlignmentsTimeTotal += tpb - tpa;

			//check quality of alignments
			tpc = std::chrono::system_clock::now();
			for(auto& b : batchElems){
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

			for(auto& b : batchElems){
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

			for(auto& b : batchElems){
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

			// update local progress
			nProcessedReads += readIds.size();

			readIds = threadOpts.batchGen->getNextReadIds();

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

	#if 1
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

	using ErrorCorrectionThread_t = ErrorCorrectionThread<Minhasher_t, ReadStorage_t, indels>;

      // initialize qscore-to-weight lookup table
  	init_weights();

    SequenceFileProperties props = getSequenceFileProperties(fileOptions.inputfile, fileOptions.format);

    /*
        Make candidate statistics
    */

    TIMERSTARTCPU(candidateestimation);
    std::vector<std::future<std::map<std::int64_t, std::int64_t>>> candidateCounterFutures;
    const ReadId_t sampleCount = props.nReads / 10;
    for(int i = 0; i < runtimeOptions.threads; i++){
        candidateCounterFutures.push_back(std::async(std::launch::async, [&,i]{
            std::map<std::int64_t, std::int64_t> candidateMap;
            std::vector<std::pair<ReadId_t, const Sequence_t*>> numseqpairs;
#if 0
            for(ReadId_t readId = i; readId < sampleCount; readId += runtimeOptions.threads){
                std::string sequencestring = readStorage.fetchSequence_ptr(readId)->toString();
                auto candidateList = minhasher.getCandidates(sequencestring, std::numeric_limits<std::uint64_t>::max());
                candidateList.erase(std::find(candidateList.begin(), candidateList.end(), readId));

                numseqpairs.clear();
                numseqpairs.reserve(candidateList.size());

                for(const auto id : candidateList){
                    numseqpairs.emplace_back(id, readStorage.fetchSequence_ptr(id));
                }

                std::sort(numseqpairs.begin(), numseqpairs.end(), [](auto l, auto r){return l.second < r.second;});
                auto uniqueend = std::unique(numseqpairs.begin(), numseqpairs.end(), [](auto l, auto r){return l.second == r.second;});
                std::size_t numunique = std::distance(numseqpairs.begin(), uniqueend);
                candidateMap[numunique]++;
            }
#else
            for(ReadId_t readId = i; readId < sampleCount; readId += runtimeOptions.threads){
                std::string sequencestring = readStorage.fetchSequence_ptr(readId)->toString();
                auto candidateList = minhasher.getCandidates(sequencestring, std::numeric_limits<std::uint64_t>::max());
                std::int64_t count = std::int64_t(candidateList.size()) - 1;
                candidateMap[count]++;
            }
#endif
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

    auto candidateDistribution = correctiondetail::estimateDist(allncandidates);

    TIMERSTOPCPU(candidateestimation);

    std::cout << "candidates.max " << candidateDistribution.max << std::endl;
	std::cout << "candidates.average " << candidateDistribution.average << std::endl;
	std::cout << "candidates.stddev " << candidateDistribution.stddev << std::endl;

    std::vector<std::pair<std::int64_t, std::int64_t>> vec(allncandidates.begin(), allncandidates.end());
    std::sort(vec.begin(), vec.end(), [](auto p1, auto p2){ return p1.second < p2.second;});

    std::ofstream of("ncandidates.txt");
    for(const auto& p : vec)
        of << p.first << " " << p.second << '\n';
    of.flush();

    /*
        Spawn correction threads
    */

    const int nCorrectorThreads = deviceIds.size() == 0 ? runtimeOptions.nCorrectorThreads
                        : std::min(runtimeOptions.nCorrectorThreads, maxCPUThreadsPerGPU * int(deviceIds.size()));

    std::vector<std::string> tmpfiles;
    for(int i = 0; i < nCorrectorThreads; i++){
        tmpfiles.emplace_back(fileOptions.outputfile + "_tmp_" + std::to_string(i));
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

#ifdef DO_PROFILE
    int sleepiter = 0;
#endif

    std::chrono::time_point<std::chrono::system_clock> timepoint_begin = std::chrono::system_clock::now();
    std::chrono::duration<double> runtime = std::chrono::seconds(0);
    std::chrono::duration<int> sleepinterval = std::chrono::seconds(3);
    ReadId_t progress = 0;
    while(progress < props.nReads){
        progress = 0;

        for (const auto& thread : ecthreads)
            progress += thread.nProcessedReads;

        if(runtimeOptions.showProgress){
            printf("Progress: %3.2f %% (Runtime: %03d:%02d:%02d)\r",
                    ((progress * 1.0 / props.nReads) * 100.0),
                    int(std::chrono::duration_cast<std::chrono::hours>(runtime).count()),
                    int(std::chrono::duration_cast<std::chrono::minutes>(runtime).count()) % 60,
                    int(runtime.count()) % 60);
            std::cout << std::flush;
        }
        if(progress < props.nReads){
              std::this_thread::sleep_for(sleepinterval);
              runtime = std::chrono::system_clock::now() - timepoint_begin;
        }
#ifdef DO_PROFILE
        sleepiter++;

        #ifdef __NVCC__
            if(sleepiter == 5)
                cudaProfilerStart(); CUERR;
        #endif


        #ifdef __NVCC__
        if(sleepiter == 6){
            cudaProfilerStop(); CUERR;
            for(auto& t : ecthreads){
                t.stopAndAbort = true;
                t.join();
            }
            std::exit(0);
        }
        #endif
#endif
    }

    for (auto& thread : ecthreads)
        thread.join();

    if(runtimeOptions.showProgress)
        printf("Progress: %3.2f %%\n", 100.00);

    minhasher.clear();
	readStorage.destroy();

    std::cout << "begin merge" << std::endl;
    mergeResultFiles(props.nReads, fileOptions.inputfile, fileOptions.format, tmpfiles, fileOptions.outputfile);
    deleteFiles(tmpfiles);

    std::cout << "end merge" << std::endl;

}


}

#endif
