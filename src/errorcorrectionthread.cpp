#include "../inc/errorcorrectionthread.hpp"


#include "../inc/batchelem.hpp"
#include "../inc/graph.hpp"
#include "../inc/graphtools.hpp"
#include "../inc/hammingtools.hpp"
#include "../inc/pileup.hpp"
#include "../inc/read.hpp"


#include <fstream>
#include <iostream>

namespace care{

/// BatchGenerator

BatchGenerator::BatchGenerator(std::uint32_t firstBatch, std::uint32_t lastBatch, std::uint32_t batchsize)
        : batchsize(batchsize), firstBatch(firstBatch), lastBatch(lastBatch), currentBatch(firstBatch){
            if(batchsize == 0) throw std::runtime_error("BatchGenerator: invalid batch size");
            if(firstBatch > lastBatch) throw std::runtime_error("BatchGenerator: firstBatch > lastBatch");
        }

BatchGenerator::BatchGenerator(std::uint32_t totalNumberOfReads, std::uint32_t batchsize_, int threadId, int nThreads){
    if(threadId < 0) throw std::runtime_error("BatchGenerator: invalid threadId");
    if(nThreads < 0) throw std::runtime_error("BatchGenerator: invalid nThreads");

	std::uint32_t totalNumberOfBatches = (totalNumberOfReads + batchsize_- 1)
			/ batchsize_;
	std::uint32_t minBatchesPerThread = totalNumberOfBatches / nThreads;

	// the last thread is responsible for leftover batches. set chunk size accordingly.
	std::uint32_t chunkSize =
			(threadId == nThreads - 1 && threadId > 0) ?
					minBatchesPerThread + totalNumberOfBatches % nThreads :
					minBatchesPerThread;

    firstBatch = threadId * minBatchesPerThread;
    lastBatch = firstBatch + chunkSize - 1;
    currentBatch = firstBatch;
    batchsize = batchsize_;
std::cout << "thread " << threadId << " first " << firstBatch << " last " << lastBatch << " batchsize " << batchsize << std::endl;
}

BatchGenerator::BatchGenerator(const BatchGenerator& rhs){
    *this = rhs;
}
BatchGenerator::BatchGenerator(BatchGenerator&& rhs){
    *this = std::move(rhs);
}

BatchGenerator& BatchGenerator::operator=(const BatchGenerator& rhs){
    batchsize = rhs.batchsize;
    firstBatch = rhs.firstBatch;
    lastBatch = rhs.lastBatch;
    currentBatch = rhs.currentBatch;
    return *this;
}

BatchGenerator& BatchGenerator::operator=(BatchGenerator&& rhs){
    batchsize = rhs.batchsize;
    firstBatch = rhs.firstBatch;
    lastBatch = rhs.lastBatch;
    currentBatch = rhs.currentBatch;
    return *this;
}

/*
    Returns vector of read ids from the next batch to be processed.
*/
std::vector<std::uint32_t> BatchGenerator::getNextReadIds(){
    std::vector<std::uint32_t> result;
    if(currentBatch <= lastBatch){
        for(std::uint32_t i = 0; i < batchsize; i++){
            result.push_back(currentBatch * batchsize + i);
        }
        currentBatch++;
    }
    return result;
}


/// ErrorCorrectionThread

void ErrorCorrectionThread::run(){
    if(isRunning) throw std::runtime_error("ErrorCorrectionThread::run: Is already running.");
    thread = std::move(std::thread(&ErrorCorrectionThread::execute, this));
}

void ErrorCorrectionThread::join(){
    thread.join();
    isRunning = false;
}

void ErrorCorrectionThread::execute() {
    isRunning = true;

	std::chrono::time_point<std::chrono::system_clock> tpa, tpb, tpc, tpd;

	std::ofstream outputstream(threadOpts.outputfile);

	auto write_read = [&](const auto readId, const auto& sequence){
		auto& stream = outputstream;
		stream << readId << '\n';
		stream << sequence << '\n';
	};

    MinhasherBuffers minhasherbuffers(threadOpts.deviceId);

	hammingtools::SHDdata shddata(threadOpts.deviceId, 1, opts.maximum_sequence_length);

	graphtools::AlignerDataArrays sgadata(threadOpts.deviceId, opts.maximum_sequence_length, opts.alignmentscore_match,
			opts.alignmentscore_sub, opts.alignmentscore_ins, opts.alignmentscore_del);


    hammingtools::correction::PileupImage pileupImage(opts.useQualityScores, opts.correctCandidates, opts.estimatedCoverage,
                                                        opts.max_mismatch_ratio, opts.errorrate, opts.m_coverage, opts.kmerlength);
    graphtools::correction::ErrorGraph errorgraph(opts.useQualityScores, opts.max_mismatch_ratio, opts.graphalpha, opts.graphx);

    std::vector<BatchElem> batchElems;
    std::vector<std::uint32_t> readIds = threadOpts.batchGen->getNextReadIds();

	while(!readIds.empty()){

		//fit vector size to actual batch size
		if (batchElems.size() != readIds.size()) {
            batchElems.resize(readIds.size(),
                              BatchElem(threadOpts.readStorage, opts.errorrate, opts.estimatedCoverage, opts.m_coverage,
                                        opts.max_mismatch_ratio, opts.min_overlap, opts.min_overlap_ratio));
        }

        for(size_t i = 0; i < readIds.size(); i++){
            batchElems[i].set_read_id(readIds[i]);
            nProcessedQueries++;
        }

		if (opts.correctCandidates){
            for(auto& b : batchElems){
			    int batchlockindex = (b.readId + threadOpts.nLocksForProcessedFlags - 1 / threadOpts.nLocksForProcessedFlags);
			    std::unique_lock<std::mutex> lock(threadOpts.locksForProcessedFlags[batchlockindex]);
                if ((*threadOpts.readIsProcessedVector)[b.readId] == 0) {
					(*threadOpts.readIsProcessedVector)[b.readId] = 1;
				}else{
                    b.active = false;
                    nProcessedQueries--;
                }
            }
		}

        std::partition(batchElems.begin(), batchElems.end(), [](const auto& b){return b.active;});

        tpa = std::chrono::system_clock::now();

        for(auto& b : batchElems){
            if(b.active){
                b.fetch_query_data_from_readstorage();

                tpc = std::chrono::system_clock::now();
                b.set_candidate_ids(threadOpts.minhasher->getCandidates(minhasherbuffers, b.fwdSequenceString));
                tpd = std::chrono::system_clock::now();
        		getCandidatesTimeTotal += tpd - tpc;

				if(b.candidateIds.size() == 0){
					//no need for further processing
					b.active = false;
					write_read(b.readId, b.fwdSequenceString);
				}else{
                    b.make_unique_sequences();
                    duplicates += b.get_number_of_duplicate_sequences();
                    b.fetch_revcompl_sequences_from_readstorage();
                }
            }
        }

		tpb = std::chrono::system_clock::now();
		mapMinhashResultsToSequencesTimeTotal += tpb - tpa;

        tpa = std::chrono::system_clock::now();
        if (opts.correctionMode == CorrectionMode::Hamming) {
            hammingtools::getMultipleAlignments(shddata, batchElems, true);
        }else if (opts.correctionMode == CorrectionMode::Graph){
            graphtools::getMultipleAlignments(sgadata, batchElems, true);
        }else{
            throw std::runtime_error("Alignment: invalid correction mode.");
        }

        tpb = std::chrono::system_clock::now();
        getAlignmentsTimeTotal += tpb - tpa;

        //select candidates from alignments
        for(auto& b : batchElems){
            if(b.active){
                tpc = std::chrono::system_clock::now();

                DetermineGoodAlignmentStats Astats = b.determine_good_alignments();

                if(Astats.correctionCases[3] > 0){
                    //no correction because not enough good alignments. write original sequence to output
                    write_read(b.readId, b.fwdSequenceString);
                }

                tpd = std::chrono::system_clock::now();
                determinegoodalignmentsTime += tpd - tpc;

                goodAlignmentStats.correctionCases[0] += Astats.correctionCases[0];
                goodAlignmentStats.correctionCases[1] += Astats.correctionCases[1];
                goodAlignmentStats.correctionCases[2] += Astats.correctionCases[2];
                goodAlignmentStats.correctionCases[3] += Astats.correctionCases[3];

                tpc = std::chrono::system_clock::now();

                if(b.active){
                    //move candidates which are used for correction to the front
                    b.prepare_good_candidates();
                }

                tpd = std::chrono::system_clock::now();
                fetchgoodcandidatesTime += tpd - tpc;
            }
        }

		if (opts.correctionMode == CorrectionMode::Hamming) {
            for(auto& b : batchElems){
                if(b.active){
                    tpc = std::chrono::system_clock::now();

                    pileupImage.correct_batch_elem(b);

                    tpd = std::chrono::system_clock::now();
                    readcorrectionTimeTotal += tpd - tpc;

                    majorityvotetime += pileupImage.timings.findconsensustime;
                    basecorrectiontime += pileupImage.timings.correctiontime;

                    avgsupportfail += pileupImage.properties.failedAvgSupport;
                    minsupportfail += pileupImage.properties.failedMinSupport;
                    mincoveragefail += pileupImage.properties.failedMinCoverage;
                    verygoodalignment += pileupImage.properties.isHQ;

                    if(b.corrected){
						write_read(b.readId, b.correctedSequence);
                    }else{
						write_read(b.readId, b.fwdSequenceString);
                    }

                    if (opts.correctCandidates) {
                        for(const auto& correctedCandidate : b.correctedCandidates){
                            const int count = b.candidateCountsPrefixSum[correctedCandidate.index+1]
                            - b.candidateCountsPrefixSum[correctedCandidate.index];
                            for(int f = 0; f < count; f++){
                                const int candidateId = b.candidateIds[count + f];
                                int batchlockindex = (candidateId + threadOpts.nLocksForProcessedFlags - 1 / threadOpts.nLocksForProcessedFlags);
                                bool savingIsOk = false;
                                if((*threadOpts.readIsProcessedVector)[candidateId] == 0){
                                    std::unique_lock <std::mutex> lock(
                                                    threadOpts.locksForProcessedFlags[batchlockindex]);
                                    if((*threadOpts.readIsProcessedVector)[candidateId]== 0) {
                                        (*threadOpts.readIsProcessedVector)[candidateId] = 1; // we will process this read
                                        lock.unlock();
                                        savingIsOk = true;
                                        nCorrectedCandidates++;
                                    }
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
            }
        }else if (opts.correctionMode == CorrectionMode::Graph){

            for(auto& b : batchElems){
                if(b.active){
                    tpc = std::chrono::system_clock::now();

                    errorgraph.correct_batch_elem(b);

                    tpd = std::chrono::system_clock::now();
                    readcorrectionTimeTotal += tpd - tpc;

                    if(b.corrected){
						write_read(b.readId, b.correctedSequence);
                    }else{
						write_read(b.readId, b.fwdSequenceString);
                    }
                }
            }
        }else{
            throw std::runtime_error("Correction: invalid correction mode.");
        }

		// update local progress
		nProcessedReads += readIds.size();

        readIds = threadOpts.batchGen->getNextReadIds();

	} // end batch processing

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
		std::cout << "thread " << threadOpts.threadId << " correctionCases "
				<< goodAlignmentStats.correctionCases[0] << " " << goodAlignmentStats.correctionCases[1] << " "
				<< goodAlignmentStats.correctionCases[2] << " " << goodAlignmentStats.correctionCases[3] << " "
				<< std::endl;

	}

#if 1
	{
        std::lock_guard < std::mutex > lg(*threadOpts.coutLock);
		if (opts.correctionMode == CorrectionMode::Hamming) {
			std::cout << "thread " << threadOpts.threadId << " : getCandidatesTimeTotal "
					<< getCandidatesTimeTotal.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : mapminhashresultsdedup "
					<< mapminhashresultsdedup.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : mapminhashresultsfetch "
					<< mapminhashresultsfetch.count() << '\n';
			std::cout << "thread " << threadOpts.threadId
					<< " : mapMinhashResultsToSequencesTimeTotal "
					<< mapMinhashResultsToSequencesTimeTotal.count() << '\n';
            std::cout << "thread " << threadOpts.threadId
                    << " : duplicates "
                    << duplicates << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment resize buffer "
					<< shddata.resizetime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment preprocessing "
					<< shddata.preprocessingtime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment H2D "
					<< shddata.h2dtime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment calculation "
					<< shddata.alignmenttime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment D2H "
					<< shddata.d2htime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment postprocessing "
					<< shddata.postprocessingtime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment total "
					<< getAlignmentsTimeTotal.count() << '\n';
			std::cout << "thread " << threadOpts.threadId
					<< " : correction find good alignments "
					<< determinegoodalignmentsTime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId
					<< " : correction fetch good data "
					<< fetchgoodcandidatesTime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : pileup vote "
					<< pileupImage.timings.findconsensustime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : pileup correct "
					<< pileupImage.timings.correctiontime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : correction calculation "
					<< readcorrectionTimeTotal.count() << '\n';
			// std::cout << "thread " << threadOpts.threadId << " : pileup resize buffer "
			// 		<< hcorrectionbuffers.resizetime.count() << '\n';
			// std::cout << "thread " << threadOpts.threadId << " : pileup preprocessing "
			// 		<< hcorrectionbuffers.preprocessingtime.count() << '\n';
			// std::cout << "thread " << threadOpts.threadId << " : pileup H2D "
			// 		<< hcorrectionbuffers.h2dtime.count() << '\n';
			// std::cout << "thread " << threadOpts.threadId << " : pileup calculation "
			// 		<< hcorrectionbuffers.correctiontime.count() << '\n';
			// std::cout << "thread " << threadOpts.threadId << " : pileup D2H "
			// 		<< hcorrectionbuffers.d2htime.count() << '\n';
			// std::cout << "thread " << threadOpts.threadId << " : pileup postprocessing "
			// 		<< hcorrectionbuffers.postprocessingtime.count() << '\n';
		} else if (opts.correctionMode == CorrectionMode::Graph) {
			std::cout << "thread " << threadOpts.threadId << " : getCandidatesTimeTotal "
					<< getCandidatesTimeTotal.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : mapminhashresultsdedup "
					<< mapminhashresultsdedup.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : mapminhashresultsfetch "
					<< mapminhashresultsfetch.count() << '\n';
			std::cout << "thread " << threadOpts.threadId
					<< " : mapMinhashResultsToSequencesTimeTotal "
					<< mapMinhashResultsToSequencesTimeTotal.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment total "
					<< getAlignmentsTimeTotal.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment resize buffer " << sgadata.resizetime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment preprocessing " << sgadata.preprocessingtime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment H2D " << sgadata.h2dtime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment calculation " << sgadata.alignmenttime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment D2H " << sgadata.d2htime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment postprocessing " << sgadata.postprocessingtime.count() << '\n';
            std::cout << "thread " << threadOpts.threadId
					<< " : correction find good alignments "
					<< determinegoodalignmentsTime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId
					<< " : correction fetch good data "
					<< fetchgoodcandidatesTime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : graph build "
					<< graphbuildtime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : graph correct "
					<< graphcorrectiontime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : correction calculation "
					<< readcorrectionTimeTotal.count() << '\n';
		}
	}
#endif

	hammingtools::cuda_cleanup_SHDdata(shddata);
	graphtools::cuda_cleanup_AlignerDataArrays(sgadata);
	cuda_cleanup_MinhasherBuffers(minhasherbuffers);
}


}
