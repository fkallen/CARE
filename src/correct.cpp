#include "../inc/correct.hpp"

#include "../inc/batchelem.hpp"
#include "../inc/graph.hpp"
#include "../inc/shifted_hamming_distance.hpp"
#include "../inc/semi_global_alignment.hpp"
#include "../inc/pileup.hpp"
#include "../inc/sequence.hpp"
#include "../inc/sequencefileio.hpp"
#include "../inc/qualityscoreweights.hpp"
#include "../inc/tasktiming.hpp"

#include <cstdint>
#include <thread>

//#define DO_PROFILE

#ifdef __NVCC__
#include <cuda_profiler_api.h>
#endif

namespace care{

/*
    Block distribution
*/
struct BatchGenerator{
    BatchGenerator(){}
    BatchGenerator(std::uint32_t firstId, std::uint32_t lastIdExcl, std::uint32_t batchsize)
            : batchsize(batchsize), firstId(firstId), lastIdExcl(lastIdExcl), currentId(firstId){
                if(batchsize == 0) throw std::runtime_error("BatchGenerator: invalid batch size");
                if(firstId >= lastIdExcl) throw std::runtime_error("BatchGenerator: firstId >= lastIdExcl");
            }
    BatchGenerator(std::uint32_t totalNumberOfReads, std::uint32_t batchsize_, int threadId, int nThreads){
        if(threadId < 0) throw std::runtime_error("BatchGenerator: invalid threadId");
        if(nThreads < 0) throw std::runtime_error("BatchGenerator: invalid nThreads");

    	std::uint32_t chunksize = totalNumberOfReads / nThreads;
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
    BatchGenerator(const BatchGenerator& rhs) = default;
    BatchGenerator(BatchGenerator&& rhs) = default;
    BatchGenerator& operator=(const BatchGenerator& rhs) = default;
    BatchGenerator& operator=(BatchGenerator&& rhs) = default;

    std::vector<std::uint32_t> getNextReadIds(){
        std::vector<std::uint32_t> result;
    	while(result.size() < batchsize && currentId < lastIdExcl){
    		result.push_back(currentId);
    		currentId++;
    	}
        return result;
    }
private:
    std::uint32_t batchsize;
    std::uint32_t firstId;
    std::uint32_t lastIdExcl;
    std::uint32_t currentId;
};



struct CorrectionThreadOptions{
    int threadId;
    int deviceId;
    int gpuThresholdSHD;
    int gpuThresholdSGA;

    std::string outputfile;
    BatchGenerator* batchGen;
    const Minhasher* minhasher;
    const ReadStorage* readStorage;
    std::mutex* coutLock;
    std::vector<char>* readIsProcessedVector;
    std::vector<char>* readIsCorrectedVector;
    std::mutex* locksForProcessedFlags;
    std::size_t nLocksForProcessedFlags;
};

struct ErrorCorrectionThread{
    AlignmentOptions alignmentOptions;
    GoodAlignmentProperties goodAlignmentProperties;
    CorrectionOptions correctionOptions;
    CorrectionThreadOptions threadOpts;
    SequenceFileProperties fileProperties;

    std::uint32_t nProcessedReads = 0;

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

    std::map<std::uint64_t, std::uint64_t> ncandidates;

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
    void execute();
};

void ErrorCorrectionThread::execute() {
    isRunning = true;

	std::chrono::time_point<std::chrono::system_clock> tpa, tpb, tpc, tpd;

	std::ofstream outputstream(threadOpts.outputfile);

	auto write_read = [&](const auto readId, const auto& sequence){
		auto& stream = outputstream;
		stream << readId << '\n';
		stream << sequence << '\n';
	};

    auto lock = [&](std::uint64_t readId){
        std::uint64_t index = readId % threadOpts.nLocksForProcessedFlags;
        threadOpts.locksForProcessedFlags[index].lock();
    };

    auto unlock = [&](std::uint64_t readId){
        std::uint64_t index = readId % threadOpts.nLocksForProcessedFlags;
        threadOpts.locksForProcessedFlags[index].unlock();
    };

	std::vector<SHDdata> shdbuffers(correctionOptions.batchsize);
    for(auto& buffer : shdbuffers){
        cuda_init_SHDdata(buffer,
                          threadOpts.deviceId,
                          fileProperties.maxSequenceLength,
                          SDIV(fileProperties.maxSequenceLength, 4),
                          threadOpts.gpuThresholdSHD);
    }

    std::vector<SGAdata> sgabuffers(correctionOptions.batchsize);
    for(auto& buffer : sgabuffers){
        cuda_init_SGAdata(buffer,
                          threadOpts.deviceId,
                          fileProperties.maxSequenceLength,
                          SDIV(fileProperties.maxSequenceLength, 4),
                          threadOpts.gpuThresholdSGA);
    }


    PileupImage pileupImage(correctionOptions.useQualityScores, correctionOptions.correctCandidates,
                                                        correctionOptions.estimatedCoverage, goodAlignmentProperties.max_mismatch_ratio,
                                                        correctionOptions.estimatedErrorrate, correctionOptions.m_coverage, correctionOptions.kmerlength);
    ErrorGraph errorgraph(correctionOptions.useQualityScores, goodAlignmentProperties.max_mismatch_ratio,
                                                  correctionOptions.graphalpha, correctionOptions.graphx);

    std::vector<BatchElem> batchElems;
    std::vector<std::uint32_t> readIds = threadOpts.batchGen->getNextReadIds();

    std::uint64_t cpuAlignments = 0;
    std::uint64_t gpuAlignments = 0;
    std::uint64_t savedAlignments = 0;
    std::uint64_t performedAlignments = 0;

	while(!stopAndAbort &&!readIds.empty()){

		//fit vector size to actual batch size
		if (batchElems.size() != readIds.size()) {
            batchElems.resize(readIds.size(),
                              BatchElem(threadOpts.readStorage,
                                        threadOpts.minhasher,
                                        correctionOptions.estimatedErrorrate,
                                        correctionOptions.estimatedCoverage, correctionOptions.m_coverage,
                                        goodAlignmentProperties.max_mismatch_ratio, goodAlignmentProperties.min_overlap,
                                        goodAlignmentProperties.min_overlap_ratio));
        }

        for(size_t i = 0; i < readIds.size(); i++){
            batchElems[i].set_read_id(readIds[i]);
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

        tpa = std::chrono::system_clock::now();

        // get query data, determine candidates via minhashing, get candidate data
        for(auto& b : batchElems){
            if(b.active){
                b.findCandidates();
                ncandidates[b.n_unique_candidates]++;
            }
        }

		tpb = std::chrono::system_clock::now();
		mapMinhashResultsToSequencesTimeTotal += tpb - tpa;


        int finalIters[16]{0};
        int maxcandidates = 0;

        for(const auto& b : batchElems){
            if(b.active)
                maxcandidates = int(b.n_unique_candidates) > maxcandidates ? int(b.n_unique_candidates) : maxcandidates;
        }
        const int alignmentbatchsize = 1*int(correctionOptions.estimatedCoverage * correctionOptions.m_coverage);

        const int maxiters = alignmentbatchsize == 0 ? 0 : SDIV(maxcandidates, alignmentbatchsize);

        for(int iter = 0; iter < maxiters; iter++){
            int batchindex = 0;
            int begin = iter * alignmentbatchsize;
            //start async alignments
            tpa = std::chrono::system_clock::now();
            for(auto& b : batchElems){
                if(b.active && !b.hasEnoughGoodCandidates()){

                    AlignmentDevice device = AlignmentDevice::None;
                    if (correctionOptions.correctionMode == CorrectionMode::Hamming) {
                        device = shifted_hamming_distance_async(shdbuffers[batchindex],
                                                                b,
                                                                begin,
                                                                alignmentbatchsize,
                                                                goodAlignmentProperties,
                                                                true);
                    }else if (correctionOptions.correctionMode == CorrectionMode::Graph){
                        device = semi_global_alignment_async(sgabuffers[batchindex],
                                                             b,
                                                             begin,
                                                             alignmentbatchsize,
                                                             alignmentOptions,
                                                             true);
                    }else{
                        throw std::runtime_error("Alignment: invalid correction mode.");
                    }

                    if(device == AlignmentDevice::CPU)
                        cpuAlignments++;
                    else if (device == AlignmentDevice::GPU)
                        gpuAlignments++;

                    if(device != AlignmentDevice::None)
                        finalIters[batchindex] = iter;
                }
                batchindex++;
            }
            //get results
            batchindex = 0;
            for(auto& b : batchElems){
                if(b.active && !b.hasEnoughGoodCandidates()){
                    if (correctionOptions.correctionMode == CorrectionMode::Hamming) {
                        get_shifted_hamming_distance_results(shdbuffers[batchindex],
                                                                b,
                                                                begin,
                                                                alignmentbatchsize,
                                                                goodAlignmentProperties,
                                                                true);
                    }else if (correctionOptions.correctionMode == CorrectionMode::Graph){
                        get_semi_global_alignment_results(sgabuffers[batchindex],
                                                             b,
                                                             begin,
                                                             alignmentbatchsize,
                                                             alignmentOptions,
                                                             true);
                    }else{
                        throw std::runtime_error("Alignment: invalid correction mode.");
                    }
                }
                batchindex++;
            }
            tpb = std::chrono::system_clock::now();
            getAlignmentsTimeTotal += tpb - tpa;

            //check quality of alignments
            batchindex = 0;
            tpc = std::chrono::system_clock::now();
            for(auto& b : batchElems){
                if(b.active && !b.hasEnoughGoodCandidates()){
                    b.determine_good_alignments(begin, alignmentbatchsize);
                }
            }
            tpd = std::chrono::system_clock::now();
            determinegoodalignmentsTime += tpd - tpc;
        }

        int batchindex = 0;
        for(auto& b : batchElems){
            if(b.active){
                std::uint64_t alignments = 2 * std::min(size_t(finalIters[batchindex] + 1) * size_t(alignmentbatchsize),
                                                         b.fwdSequences.size());
                savedAlignments += 2 * b.fwdSequences.size() - alignments;
                performedAlignments += alignments;
            }
        }

        for(auto& b : batchElems){
            if(b.active && b.hasEnoughGoodCandidates()){
                tpc = std::chrono::system_clock::now();
                if(b.active){
                    //move candidates which are used for correction to the front
                    b.prepare_good_candidates();
                }
                tpd = std::chrono::system_clock::now();
                fetchgoodcandidatesTime += tpd - tpc;
            }else{
                //not enough good candidates. cannot correct this read.
                b.active = false;
            }
        }

		if (correctionOptions.correctionMode == CorrectionMode::Hamming) {
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
                        lock(b.readId);
                        (*threadOpts.readIsCorrectedVector)[b.readId] = 1;
                        unlock(b.readId);
                    }

                    if (correctionOptions.correctCandidates) {
                        for(const auto& correctedCandidate : b.correctedCandidates){
                            const int count = b.candidateCountsPrefixSum[correctedCandidate.index+1]
                            - b.candidateCountsPrefixSum[correctedCandidate.index];
                            for(int f = 0; f < count; f++){
                                std::uint64_t candidateId = b.candidateIds[b.candidateCountsPrefixSum[correctedCandidate.index] + f];
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
            }
        }else if (correctionOptions.correctionMode == CorrectionMode::Graph){

            for(auto& b : batchElems){
                if(b.active){
                    tpc = std::chrono::system_clock::now();

                    errorgraph.correct_batch_elem(b);

                    tpd = std::chrono::system_clock::now();
                    readcorrectionTimeTotal += tpd - tpc;

                    if(b.corrected){
						write_read(b.readId, b.correctedSequence);
                        lock(b.readId);
                        (*threadOpts.readIsCorrectedVector)[b.readId] = 1;
                        unlock(b.readId);
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
		/*std::cout << "thread " << threadOpts.threadId
                  << " CPU alignments " << cpuAlignments
                  << " GPU alignments " << gpuAlignments << std::endl;*/

        std::cout << "thread " << threadOpts.threadId << " savedAlignments "
                << savedAlignments << " performedAlignments " << performedAlignments << std::endl;
	}

#if 1
	{
        std::lock_guard < std::mutex > lg(*threadOpts.coutLock);
		if (correctionOptions.correctionMode == CorrectionMode::Hamming) {
			std::cout << "thread " << threadOpts.threadId
					<< " : mapMinhashResultsToSequencesTimeTotal "
					<< mapMinhashResultsToSequencesTimeTotal.count() << '\n';
			/*std::cout << "thread " << threadOpts.threadId << " : alignment resize buffer "
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
					<< shddata.postprocessingtime.count() << '\n';*/
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
		} else if (correctionOptions.correctionMode == CorrectionMode::Graph) {
			std::cout << "thread " << threadOpts.threadId
					<< " : mapMinhashResultsToSequencesTimeTotal "
					<< mapMinhashResultsToSequencesTimeTotal.count() << '\n';
			/*std::cout << "thread " << threadOpts.threadId << " : alignment resize buffer " << sgadata.resizetime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment preprocessing " << sgadata.preprocessingtime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment H2D " << sgadata.h2dtime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment calculation " << sgadata.alignmenttime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment D2H " << sgadata.d2htime.count() << '\n';
			std::cout << "thread " << threadOpts.threadId << " : alignment postprocessing " << sgadata.postprocessingtime.count() << '\n';*/
            std::cout << "thread " << threadOpts.threadId << " : alignment total "
					<< getAlignmentsTimeTotal.count() << '\n';
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

    for(auto& shdbuffer : shdbuffers)
	   cuda_cleanup_SHDdata(shdbuffer);

    for(auto& shdbuffer : sgabuffers)
   	   cuda_cleanup_SGAdata(shdbuffer);
}



void correct(const MinhashOptions& minhashOptions,
				  const AlignmentOptions& alignmentOptions,
				  const GoodAlignmentProperties& goodAlignmentProperties,
				  const CorrectionOptions& correctionOptions,
				  const RuntimeOptions& runtimeOptions,
				  const FileOptions& fileOptions,
                  Minhasher& minhasher,
                  ReadStorage& readStorage,
				  std::vector<char>& readIsCorrectedVector,
				  std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
				  size_t nLocksForProcessedFlags,
				  const std::vector<int>& deviceIds){

      // initialize qscore-to-weight lookup table
  	init_weights();

    SequenceFileProperties props = getSequenceFileProperties(fileOptions.inputfile, fileOptions.format);

    std::vector<std::string> tmpfiles;
    for(int i = 0; i < runtimeOptions.nCorrectorThreads; i++){
        tmpfiles.emplace_back(fileOptions.outputfile + "_tmp_" + std::to_string(i));
    }

    std::vector<BatchGenerator> generators(runtimeOptions.nCorrectorThreads);
    std::vector<ErrorCorrectionThread> ecthreads(runtimeOptions.nCorrectorThreads);
    std::vector<char> readIsProcessedVector(readIsCorrectedVector);
    std::mutex writelock;

    std::vector<int> gpuThresholdsSHD(deviceIds.size());
    for(size_t i = 0; i < deviceIds.size(); i++){
        int threshold = find_shifted_hamming_distance_gpu_threshold(deviceIds[i],
                                                                       props.minSequenceLength,
                                                                       SDIV(props.minSequenceLength, 4));
        gpuThresholdsSHD[i] = std::min(threshold, 0);
    }

    std::vector<int> gpuThresholdsSGA(deviceIds.size());
    for(size_t i = 0; i < deviceIds.size(); i++){
        int threshold = find_semi_global_alignment_gpu_threshold(deviceIds[i],
                                                                       props.minSequenceLength,
                                                                       SDIV(props.minSequenceLength, 4));
        gpuThresholdsSGA[i] = std::min(threshold, 0);
    }

    for(size_t i = 0; i < gpuThresholdsSHD.size(); i++)
        std::cout << "GPU " << i
                  << ": gpuThresholdSHD " << gpuThresholdsSHD[i]
                  << " gpuThresholdSGA " << gpuThresholdsSGA[i] << std::endl;

#ifdef DO_PROFILE
#ifdef __NVCC__
    cudaProfilerStart(); CUERR;
#endif
#endif

    for(int threadId = 0; threadId < runtimeOptions.nCorrectorThreads; threadId++){

        generators[threadId] = BatchGenerator(props.nReads, correctionOptions.batchsize, threadId, runtimeOptions.nCorrectorThreads);
        CorrectionThreadOptions threadOpts;
        threadOpts.threadId = threadId;
        threadOpts.deviceId = deviceIds.size() == 0 ? -1 : deviceIds[threadId % deviceIds.size()];
        threadOpts.gpuThresholdSHD = gpuThresholdsSHD[threadId % deviceIds.size()];
        threadOpts.gpuThresholdSGA = gpuThresholdsSGA[threadId % deviceIds.size()];
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

        ecthreads[threadId].run();
    }

#ifdef DO_PROFILE
    int sleepiter = 0;
#endif
    if(runtimeOptions.showProgress){
        std::chrono::duration<int> runtime = std::chrono::seconds(0);
        std::chrono::duration<int> sleepinterval = std::chrono::seconds(3);
        std::uint64_t progress = 0;
        while(progress < props.nReads){
            progress = 0;
            for(int threadId = 0; threadId < runtimeOptions.nCorrectorThreads; threadId++){
                progress += ecthreads[threadId].nProcessedReads;
            }
            printf("Progress: %3.2f %% (Runtime: %03d:%02d:%02d)\r",
                    ((progress * 1.0 / props.nReads) * 100.0),
                    int(std::chrono::duration_cast<std::chrono::hours>(runtime).count()),
                    int(std::chrono::duration_cast<std::chrono::minutes>(runtime).count() % 60),
                    int(runtime.count() % 60));
            std::cout << std::flush;
            if(progress < props.nReads){
                  std::this_thread::sleep_for(sleepinterval);
                  runtime += sleepinterval;
            }
#ifdef DO_PROFILE
            sleepiter++;

            #ifdef __NVCC__
            if(sleepiter > 0){
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
    }

    for (auto& thread : ecthreads)
        thread.join();

    if(runtimeOptions.showProgress)
        printf("Progress: %3.2f %%\n", 100.00);

    std::map<std::uint64_t, std::uint64_t> ncandidates;
    for(const auto& thread : ecthreads){
        for(const auto& pair : thread.ncandidates){
            ncandidates[pair.first] += pair.second;
        }
    }

    std::vector<std::pair<std::uint64_t, std::uint64_t>> vec(ncandidates.begin(), ncandidates.end());
    std::sort(vec.begin(), vec.end(), [](auto p1, auto p2){ return p1.second < p2.second;});

    std::ofstream of("ncandidates.txt");
    for(const auto& p : vec)
        of << p.first << " " << p.second << '\n';
    of.flush();

    minhasher.init(0);
	readStorage.destroy();

    std::cout << "begin merge" << std::endl;
    mergeResultFiles(props.nReads, fileOptions.inputfile, fileOptions.format, tmpfiles, fileOptions.outputfile);
    deleteFiles(tmpfiles);

    std::cout << "end merge" << std::endl;

}


}
