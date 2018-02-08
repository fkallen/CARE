#include "../inc/errorcorrector.hpp"
#include "../inc/read.hpp"
#include "../inc/fastareader.hpp"
#include "../inc/fastqreader.hpp"
#include "../inc/binarysequencehelpers.hpp"

#include "../inc/ganja/hpc_helpers.cuh"

#include "../inc/hammingtools.hpp"
#include "../inc/graphtools.hpp"

#include <cstdint>
#include <thread>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <map>
#include <set>
#include <functional>
#include <mutex>
#include <chrono>
#include <iterator>
#include <future>

#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __NVCC__
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/iterator/constant_iterator.h>

#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/sort.h>
#endif

#include <experimental/filesystem> // create_directories

#define ERRORCORRECTION_TIMING
//#define USE_REVCOMPL_FLAG

//constexpr int MAX_THREADS_PER_GPU = 30;

// how many correction results should be buffered before writing the correction results to file
constexpr int bufferedResultsThreshold = 1000;

// update global progress after correcting a multiple of this many reads
constexpr std::uint64_t progressThreshold = 5000;

//the most probable path in the errorgraph must have a probability of at least MINIMUM_CORRECTION_PROBABILITY
//constexpr double MINIMUM_CORRECTION_PROBABILITY = 0.0;

constexpr double HASHMAP_LOAD_FACTOR = 0.8;

constexpr bool CORRECT_CANDIDATE_READS_TOO = true;
//constexpr double CANDIDATE_CORRECTION_MIN_OVERLAP_FACTOR = 1.00;
//constexpr double CANDIDATE_CORRECTION_MAX_MISMATCH_RATIO = 0.00;

struct MinhashResultsDedupBuffers {
	const Sequence** d_candidateReads = nullptr;
	int* d_indexlist = nullptr;
	const Sequence** d_uniqueSequences = nullptr;
	int* d_frequencies = nullptr;
	int* h_indexlist = nullptr;

	size_t n_initial_results = 0;
	size_t n_unique_results = 0;
	size_t max_n_initial_results = 0;
	size_t max_n_unique_results = 0;
	int deviceId = -1;

#ifdef __NVCC__
	cudaStream_t stream;
#endif

	MinhashResultsDedupBuffers(int id) {
		deviceId = id;
#ifdef __NVCC__
		cudaSetDevice(deviceId); CUERR;
		cudaStreamCreate(&stream);
#endif
	}

	void resize_initial_results(size_t n_results) {
#ifdef __NVCC__
		cudaSetDevice(deviceId); CUERR;
		if(n_results > max_n_initial_results) {
			cudaFree(d_candidateReads); CUERR;
			cudaFree(d_indexlist); CUERR;
			cudaMalloc(&d_candidateReads, sizeof(const Sequence*) * n_results); CUERR;
			cudaMalloc(&d_indexlist, sizeof(int) * n_results); CUERR;

			cudaFreeHost(h_indexlist); CUERR
			cudaMallocHost(&h_indexlist, sizeof(int*) * n_results); CUERR;

			max_n_initial_results = n_results;
		}
#endif
		n_initial_results = n_results;
	}
	void resize_unique_results(size_t n_results) {
#ifdef __NVCC__
		cudaSetDevice(deviceId); CUERR;
		if(n_results > max_n_unique_results) {
			cudaFree(d_uniqueSequences); CUERR;
			cudaFree(d_frequencies); CUERR;
			cudaMalloc(&d_uniqueSequences, sizeof(const Sequence*) * n_results); CUERR;
			cudaMalloc(&d_frequencies, sizeof(int) * n_results); CUERR;

			max_n_unique_results = n_results;
		}
#endif
		n_unique_results = n_results;
	}
};

void cuda_cleanup_MinhashResultsDedupBuffers(
		MinhashResultsDedupBuffers buffer) {
#ifdef __NVCC__
	cudaSetDevice(buffer.deviceId); CUERR;
	cudaFree(buffer.d_candidateReads); CUERR;
	cudaFree(buffer.d_indexlist); CUERR;
	cudaFree(buffer.d_uniqueSequences); CUERR;
	cudaFree(buffer.d_frequencies); CUERR;
	cudaFreeHost(buffer.h_indexlist); CUERR;
	cudaStreamDestroy(buffer.stream); CUERR;
#endif
}

ErrorCorrector::ErrorCorrector() :
		ErrorCorrector(MinhashParameters(), 1, 1) {
}

ErrorCorrector::ErrorCorrector(const MinhashParameters& minhashparameters,
		int nInserterThreads_, int nCorrectorThreads_) :
		minhashparams(minhashparameters), nInserterThreads(nInserterThreads_), nCorrectorThreads(
				nCorrectorThreads_), outputPath("") {
	//cudaDeviceSetLimit(cudaLimitPrintfFifoSize,1 << 20); CUERR;
	correctionmode = CorrectionMode::Hamming;
	//correctionmode = CorrectionMode::Graph;

	minhasher.minparams = minhashparameters;

	buffers.resize(nInserterThreads);

	hammingtools::init_once();
	graphtools::init_once();

#ifdef __CUDACC__

	int nGpus;
	cudaGetDeviceCount(&nGpus); CUERR;
	if(nGpus == 0) throw std::runtime_error("No CUDA capable device found!");
	for(int i = 0; i < nGpus; i++)
	deviceIds.push_back(i);

#endif

}

void ErrorCorrector::mergeUnorderedThreadResults(
		const std::string& filename) const {

	std::string name = filename;
	std::string fileEnding = ".fq";

	size_t lastdotpos = filename.find_last_of(".");
	if (lastdotpos != std::string::npos) {
		name = name.substr(0, lastdotpos);
		fileEnding = filename.substr(lastdotpos);
	}

	size_t lastslashpos = filename.find_last_of("/");
	if (lastslashpos != std::string::npos)
		name = name.substr(lastslashpos + 1);

	std::string currentOutputFilename;

	if (outputFilename != "")
		currentOutputFilename = outputPath + "/" + outputFilename;
	else
		currentOutputFilename = outputPath + "/" + name + "_"
				+ std::to_string(minhashparams.k) + "_"
				+ std::to_string(minhashparams.maps) + "_1" + "_alpha_"
				+ std::to_string(graphalpha) + "_x_" + std::to_string(graphx)
				+ "_corrected" + fileEnding;

	std::cout << "merging into " << currentOutputFilename << std::endl;

	const std::uint32_t totalNumberOfReads = readsPerFile.at(filename);
	std::uint32_t nreads = 0;

	std::vector<Read> reads(totalNumberOfReads);
	for (int i = 0; i < nCorrectorThreads; i++) {

		std::unique_ptr<ReadReader> reader;
		switch (inputfileformat) {
		case Fileformat::FASTQ:
			reader.reset(new FastqReader(outputPath + "/" + std::to_string(i)));
			break;
		case Fileformat::FASTA:
			reader.reset(new FastaReader(outputPath + "/" + std::to_string(i)));
			break;

		default:
			assert(false && "merge inputfileformat");
			break;
		}

		Read read;

		while (reader->getNextRead(&read, nullptr)) {
			nreads++;
			//std::cout << nreads << std::endl;
			auto spacepos = read.header.find(" ");
			auto readnum = std::stoull(read.header.substr(0, spacepos));
			read.header.erase(0, spacepos + 1);
			reads[readnum] = std::move(read);
		}
	}

	if (nreads != totalNumberOfReads) {
		Read tmp;
		int asd = std::count_if(reads.begin(), reads.end(), [&](const auto& a) {
			return a == tmp;
		});

		std::cout << "totalNumberOfReads " << totalNumberOfReads << '\n'
				<< "nreads " << nreads << '\n' << "asd " << asd << '\n';
		assert(nreads == totalNumberOfReads);
	}
	assert(nreads == totalNumberOfReads);

	Read tmp;
	if (std::find(reads.begin(), reads.end(), tmp) != reads.end())
		std::cout << "error" << std::endl;

	std::cout << "done in a moment" << std::endl;
	std::ofstream outputfile(currentOutputFilename);

	for (const auto& read : reads) {
		outputfile << read.header << '\n' << read.sequence << '\n';

		if (inputfileformat == Fileformat::FASTQ)
			outputfile << '+' << '\n' << read.quality << '\n';
	}

	outputfile.flush();
	outputfile.close();

	for (int i = 0; i < nCorrectorThreads; i++) {
		std::string s = outputPath + "/" + std::to_string(i);
		int ret = std::remove(s.c_str());
		if (ret != 0)
			std::cout << "could not remove file " << s << std::endl;
	}
}

void ErrorCorrector::correct(const std::string& filename) {
	if (inputfileformat == Fileformat::FASTA)
		setUseQualityScores(false);

	if (outputPath == "")
		throw std::runtime_error("no output path specified");

	std::ifstream f(filename);
	if (!f)
		throw std::runtime_error("cannot read input file");

	std::uint64_t nLines = 0;
	std::string line;
	for (; std::getline(f, line); nLines++)
		;

	std::uint64_t linesPerRead = inputfileformat == Fileformat::FASTQ ? 4 : 2;
	std::uint64_t nReads = nLines / linesPerRead;

	if (inputfileformat == Fileformat::FASTA) {
		if (nLines % linesPerRead != 0)
			throw std::runtime_error(
					"input file has invalid fasta format. number of lines mod 2 != 0");
		std::cout << "Reads: " << nReads << std::endl;
	}

	if (inputfileformat == Fileformat::FASTQ) {
		if (nLines % linesPerRead != 0)
			throw std::runtime_error(
					"input file has invalid fastq format. number of lines mod 4 != 0");
		std::cout << "Reads: " << nReads << std::endl;
	}

	readsPerFile.insert( { filename, nReads });
#if 1
	minhasher.init(nReads, HASHMAP_LOAD_FACTOR);

	readStorage.init(nReads);

	if (CORRECT_CANDIDATE_READS_TOO) {
		readIsProcessedVector.resize(nReads, 0);
		nLocksForProcessedFlags = batchsize * nCorrectorThreads * 1000;
		locksForProcessedFlags.reset(new std::mutex[nLocksForProcessedFlags]);
	}

	std::cout << "begin insert" << std::endl;

	TIMERSTARTCPU(INSERT);
#if 0
	std::string mapfilename = filename;
	size_t lastslashpos = mapfilename.find_last_of("/");
	if(lastslashpos != std::string::npos)
	mapfilename = mapfilename.substr(lastslashpos + 1);
	if(!minhasher.loadTablesFromFile(outputPath + "/" + mapfilename+"_"+std::to_string(minhashparams.k)+"_"+std::to_string(minhashparams.maps)+"_map")) {
		insertFile(filename, true);
#if 1
		TIMERSTARTCPU(MAP_TRANSFORM);
		minhasher.transform();
		TIMERSTOPCPU(MAP_TRANSFORM);
#endif
		minhasher.saveTablesToFile(outputPath + "/" + mapfilename+"_"+std::to_string(minhashparams.k)+"_map");
		std::cout << "saved map to file " << (outputPath + "/" + mapfilename+"_"+std::to_string(minhashparams.k)+"_map") << std::endl;
	} else {
		insertFile(filename, false);
		std::cout << "loaded map from file " << (outputPath + "/" + mapfilename+"_"+std::to_string(minhashparams.k)+"_map") << std::endl;
	}
#else
	insertFile(filename, true);
#endif
	TIMERSTOPCPU(INSERT);

	std::cout << "end insert" << std::endl;

	TIMERSTARTCPU(MAP_TRANSFORM);
	minhasher.transform();
	TIMERSTOPCPU(MAP_TRANSFORM);

	TIMERSTARTCPU(readstorage_transform);
	readStorage.noMoreInserts();
	TIMERSTOPCPU(readstorage_transform);

	std::cout << "begin correct" << std::endl;

	TIMERSTARTCPU(CORRECT);
	errorcorrectFile(filename);
	printf("\n");
	TIMERSTOPCPU(CORRECT);

	std::cout << "end correct" << std::endl;

	if (CORRECT_CANDIDATE_READS_TOO) {
		int asd = std::count_if(readIsProcessedVector.begin(),
				readIsProcessedVector.end(), [](auto b) {return b;});
		std::cout << "total corrected reads: " << asd << std::endl;
	}

	minhasher.init(1, HASHMAP_LOAD_FACTOR);
	readStorage.clear();
	readIsProcessedVector.clear();

#endif
	std::cout << "begin merge" << std::endl;

	mergeUnorderedThreadResults(filename);

	std::cout << "end merge" << std::endl;
}

void ErrorCorrector::insertFile(const std::string& filename,
		bool buildHashmap) {

#if 0
	std::unique_ptr<ReadReader> reader;

	switch(inputfileformat) {
		case Fileformat::FASTQ: reader.reset(new FastqReader(filename)); break;
		case Fileformat::FASTA: reader.reset(new FastaReader(filename)); break;

		default: assert(false && "inputfileformat"); break;
	}

	Read read;
	std::uint32_t readnum = 0;
	std::uint64_t totalNumberOfReads = readsPerFile.at(filename);
	std::uint64_t progressprocessedReads = 0;

	while (reader->getNextRead(&read, &readnum)) {

		//replace 'N' with 'A'
		for(auto& c : read.sequence)
		if(c == 'N')
		c = 'A';

		if(buildHashmap) minhasher.insertSequence(read.sequence, readnum);

		readStorage.insertRead(readnum, read);

		progressprocessedReads++;

		// update global progress
		if(readnum > 3*progressThreshold) {
			updateGlobalProgress(progressprocessedReads, totalNumberOfReads);
			progressprocessedReads = 0;
		}
	}

	progress = 0;

#else

	std::vector<std::future<int>> inserterThreads;
	progress = 0;

	for (int threadId = 0; threadId < nInserterThreads; ++threadId) {

		inserterThreads.emplace_back(
				std::async(std::launch::async,
						[&, threadId]()->int {

							std::uint64_t progressprocessedReads = 0;
							std::uint64_t totalNumberOfReads = readsPerFile.at(filename);

							int maxlength = 0;

							std::pair<Read, std::uint32_t> pair = buffers[threadId].get();

							while (pair != buffers[threadId].defaultValue) {
								Read& read = pair.first;
								const std::uint32_t& readnum = pair.second;

								//replace 'N' with 'A'
								for(auto& c : read.sequence)
								if(c == 'N')
								c = 'A';
								if(int(read.sequence.length()) > maxlength)
								maxlength = read.sequence.length();

								if(buildHashmap) minhasher.insertSequence(read.sequence, readnum);

								readStorage.insertRead(readnum, read);

								pair = buffers[threadId].get();

								progressprocessedReads += 1;

								// update global progress
								if(progressprocessedReads > 3*progressThreshold) {
									updateGlobalProgress(progressprocessedReads, totalNumberOfReads);
									progressprocessedReads = 0;
								}

							}

							return maxlength;
						}));

	}

	std::unique_ptr<ReadReader> reader;

	switch (inputfileformat) {
	case Fileformat::FASTQ:
		reader.reset(new FastqReader(filename));
		break;
	case Fileformat::FASTA:
		reader.reset(new FastaReader(filename));
		break;

	default:
		assert(false && "inputfileformat");
		break;
	}

	Read read;
	std::uint32_t readnum = 0;
	int target = 0;

	while (reader->getNextRead(&read, &readnum)) {
		target = readnum % nInserterThreads;
		buffers[target].add( { read, readnum });
		readnum++;
	}
	//std::cout << "read distribution done" << std::endl;
	for (int i = 0; i < nInserterThreads; i++) {
		buffers[i].done();
	}
// producer done

	/*for (size_t i = 0; i < inserterThreads.size(); ++i ) {
	 inserterThreads[i].join();
	 //printf("buffer %d: addWait: %lu, addNoWait: %lu, getWait: %lu, getNoWait: %lu\n",
	 //	i, buffers[i].addWait, buffers[i].addNoWait, buffers[i].getWait, buffers[i].getNoWait);
	 buffers[i].reset();
	 }*/

	int maxlen = 0;

	for (size_t i = 0; i < inserterThreads.size(); ++i) {
		int res = inserterThreads[i].get();
		if (res > maxlen)
			maxlen = res;
		buffers[i].reset();
	}

	std::cout << "max sequence length in file: " << maxlen << '\n';

	maximum_sequence_length = maxlen;

	progress = 0;

#endif
}

void ErrorCorrector::errorcorrectFile(const std::string& filename) {
#if 1

#if 0
	std::vector<std::thread> consumerthreads;

	// spawn work on other threads
	for (int threadId = 1; threadId < nCorrectorThreads; ++threadId) {
		consumerthreads.emplace_back(&ErrorCorrector::errorcorrectWork,
				this,
				threadId, nCorrectorThreads,
				filename);
	}

	// run work on this thread

	errorcorrectWork(0, nCorrectorThreads, filename);

	// wait for other threads
	for (int i = 0; i < consumerthreads.size(); ++i )
	consumerthreads[i].join();

#else

	std::vector < std::thread > consumerthreads;

	// spawn work on other threads
	for (int threadId = 0; threadId < nCorrectorThreads; ++threadId) {
		consumerthreads.emplace_back(&ErrorCorrector::errorcorrectWork, this,
				threadId, nCorrectorThreads, filename);
	}

	for (auto& thread : consumerthreads)
		thread.join();

#endif

#else

	std::uint32_t totalNumberOfReads = readsPerFile.at(filename);

	std::vector<EC_Thread_Data> threadData;

	for (int threadId = 0; threadId < nCorrectorThreads; threadId++) {
		// perform block distribution of reads to the threads. thread will process reads [firstRead, firstRead + chunkSize[
		std::uint32_t firstRead = 0;
		std::uint32_t chunkSize = totalNumberOfReads / nCorrectorThreads;
		std::uint32_t leftover = totalNumberOfReads % nCorrectorThreads;

		if(threadId < leftover)
		chunkSize++;

		if(threadId < leftover) {
			firstRead = threadId * chunkSize;
		} else {
			firstRead = leftover * (chunkSize + 1) + (threadId - leftover) * chunkSize;
		}

		bool useGPU = deviceIds.size() > 0;

		//chunkSize = 4;

		threadData.emplace_back(threadId, threadId % deviceIds.size(), useGPU, &readStorage, &minhasher,
				graphx, graphalpha,
				firstRead, firstRead + chunkSize, batchsize,
				outputPath, &writelock);
	}

	std::vector<std::thread> correctorThreads;

	for (int threadId = 0; threadId < nCorrectorThreads; threadId++) {

		correctorThreads.emplace_back(&EC_Thread_Data::correct, std::ref(threadData[threadId]));
		//correctorThreads.emplace_back([&](){threadData[threadId].correct();});
	}

	std::cout << std::endl;

	bool correctionDone = false;
	while(!correctionDone) {
		/*std::cout << "\rProgress per Thread: ";
		 for(int i = 0; i < nCorrectorThreads; i++){
		 std::cout << (i == 0 ? "t": ", t") << i << ": " << (threadData[i].progress * 100.0) << "%";
		 }
		 std::cout << std::flush;

		 std::this_thread::sleep_for(std::chrono::seconds(5));

		 bool newDone = true;
		 for(int i = 0; i < nCorrectorThreads; i++)
		 newDone &= threadData[i].done;

		 correctionDone = newDone;*/
		std::uint32_t progress = 0;
		bool newDone = true;
		for(int i = 0; i < nCorrectorThreads; i++) {
			newDone &= threadData[i].done;
			progress += threadData[i].nProcessedReads;
		}
		correctionDone = newDone;

		printf("Progress: %3.2f %%\r" , ((progress * 1.0 / totalNumberOfReads) * 100.0));
		std::cout << std::flush;

		std::this_thread::sleep_for(std::chrono::seconds(10));
	}

	std::cout << std::endl;

	std::cout << "done."<< std::endl;

	for (int i = 0; i < nCorrectorThreads; ++i )
	correctorThreads[i].join();

#endif
}

void ErrorCorrector::errorcorrectWork(int threadId, int nThreads,
		const std::string& fileToCorrect) {

	std::chrono::duration<double> getCandidatesTimeTotal(0);
	std::chrono::duration<double> mapMinhashResultsToSequencesTimeTotal(0);
	std::chrono::duration<double> getAlignmentsTimeTotal(0);
	std::chrono::duration<double> correctReadTimeTotal(0);
	std::chrono::duration<double> determinegoodalignmentsTime(0);
	std::chrono::duration<double> fetchgoodcandidatesTime(0);
	std::chrono::duration<double> majorityvotetime(0);
	std::chrono::duration<double> basecorrectiontime(0);
	std::chrono::duration<double> readcorrectionTimeTotal(0);
	std::chrono::duration<double> fileoutputTimeTotal(0);
	std::chrono::duration<double> mapminhashresultsdedup(0);
	std::chrono::duration<double> mapminhashresultsfetch(0);
	std::chrono::duration<double> graphbuildtime(0);
	std::chrono::duration<double> graphcorrectiontime(0);

	std::chrono::time_point<std::chrono::system_clock> tpa, tpb, tpc, tpd;

	// the output file of this thread
	std::ofstream outputfile(outputPath + "/" + std::to_string(threadId));

	// number of buffered correction results. nBufferedResults < bufferedResultsThreshold	
	int nBufferedResults = 0;

	// buffer of correction results
	std::stringstream resultstringstream;

	// number of processed reads after previous progress update
	// resets after each progress update
	std::uint64_t progressprocessedReads = 0;

	// perform block distribution of reads to the threads. thread will process reads [firstRead, firstRead + chunkSize[
	std::uint32_t totalNumberOfReads = readsPerFile.at(fileToCorrect);
	std::uint32_t totalNumberOfBatches = (totalNumberOfReads + batchsize - 1)
			/ batchsize;
	std::uint32_t minBatchesPerThread = totalNumberOfBatches / nThreads;

	std::uint32_t firstBatch = threadId * minBatchesPerThread;
	// the last thread is responsible for leftover batches. set chunk size accordingly.
	std::uint32_t chunkSize =
			(threadId == nThreads - 1 && threadId > 0) ?
					minBatchesPerThread + totalNumberOfBatches % nThreads :
					minBatchesPerThread;

	int avgsupportfail = 0;
	int minsupportfail = 0;
	int mincoveragefail = 0;
	int sobadcouldnotcorrect = 0;
	int verygoodalignment = 0;

	int correctionCases[4] { 0, 0, 0, 0 }; // <= 2e, <= 3e, <= 4e, no correction

#if 0
	{
		std::lock_guard<std::mutex> lg(writelock);
		std::cout << "thread " << threadId << ": batches [" << firstBatch << ", " << (firstBatch + chunkSize) << ")" << std::endl;
	}
#endif

	// total number of candidates returned from minhasher
	std::uint64_t processedCandidates = 0;

	// candidates which were corrected, too, during query correction
	int nCorrectedCandidates = 0;
	int nProcessedQueries = 0;

	//printf("max seq length = %d\n", maximum_sequence_length);
	int deviceId =
			deviceIds.size() == 0 ? -1 : deviceIds[threadId % deviceIds.size()];
	hammingtools::SHDdata shddata(deviceId, SDIV(nThreads, deviceIds.size()),
			maximum_sequence_length);
	hammingtools::CorrectionBuffers hcorrectionbuffers(deviceId,
			maximum_sequence_length);
	graphtools::AlignerDataArrays sgadata(deviceId, ALIGNMENTSCORE_MATCH,
			ALIGNMENTSCORE_SUB, ALIGNMENTSCORE_INS, ALIGNMENTSCORE_DEL);

	MinhasherBuffers minhasherbuffers(deviceId);
	MinhashResultsDedupBuffers minhashresultsdedupbuffers(deviceId);

	// the query sequences fetched from ReadStorage
	std::vector<const Sequence*> queries(batchsize);
	std::vector < std::string > queryStrings(batchsize);
	// the readnums of candidates for each query
	//std::vector<std::vector<std::pair<std::uint64_t, int>>> candidateIds(batchsize);
	std::vector < std::vector < std::uint64_t >> candidateIds(batchsize);

	// the candidate sequences from candidateReadsWithFrequency
	std::vector<std::vector<const Sequence*>> candidateReads(batchsize);
	std::vector<std::vector<const Sequence*>> revComplcandidateReads(batchsize);
	std::vector<std::vector<int>> frequencies(batchsize);
	std::vector < std::vector < AlignResult >> alignmentResults(batchsize);
	std::vector<std::vector<const Sequence*>> candidateReadsAndRevcompls(
			batchsize);
	std::vector<const std::string*> queryQualities(batchsize);
	std::vector<std::vector<const std::string*>> candidateQualities(batchsize);
	std::vector<std::vector<const std::string*>> revcomplcandidateQualities(
			batchsize);
	std::vector<std::map<const Sequence*, std::vector<int>, SequencePtrLess>> sequenceToIdsMaps(
			batchsize);
	std::vector<bool> activeBatches(batchsize);

	const int maxReadsPerLock =
			(!CORRECT_CANDIDATE_READS_TOO) ?
					1 :
					((totalNumberOfReads + nLocksForProcessedFlags - 1)
							/ nLocksForProcessedFlags) + batchsize
							- ((totalNumberOfReads + nLocksForProcessedFlags - 1)
									/ nLocksForProcessedFlags) % batchsize;

	// loop over the reads to process
	//for(std::uint32_t readnum = firstRead; readnum < firstRead + chunkSize; readnum += batchsize){
	for (std::uint32_t currentBatchNum = firstBatch;
			currentBatchNum < firstBatch + chunkSize; currentBatchNum++) {
		const std::uint32_t readnum = currentBatchNum * batchsize; // id of first read in batch

		assert(readnum < totalNumberOfReads);

		// boundary condition. cannot process more reads than the remaining reads
		//std::uint32_t actualBatchSize = std::min(batchsize, firstRead + chunkSize - readnum);
		std::uint32_t actualBatchSize = std::min(batchsize,
				totalNumberOfReads - readnum);

		//fit vector size to actual batch size
		if (actualBatchSize < batchsize) {
			queries.resize(actualBatchSize);
			queryStrings.resize(actualBatchSize);
			candidateIds.resize(actualBatchSize);
			candidateReads.resize(actualBatchSize);
			revComplcandidateReads.resize(actualBatchSize);
			frequencies.resize(actualBatchSize);
			alignmentResults.resize(actualBatchSize);
			candidateReadsAndRevcompls.resize(actualBatchSize);
			queryQualities.resize(actualBatchSize);
			candidateQualities.resize(actualBatchSize);
			revcomplcandidateQualities.resize(actualBatchSize);
			sequenceToIdsMaps.resize(actualBatchSize);
			activeBatches.resize(actualBatchSize);
		}

		if (CORRECT_CANDIDATE_READS_TOO) {
			int batchlockindex = readnum / maxReadsPerLock;
			//assert lock ids are good
			/*for(std::uint32_t i = 0; i < actualBatchSize; i++){
			 if(std::uint32_t(batchlockindex) != (readnum + i) / maxReadsPerLock){
			 std::lock_guard<std::mutex> lg(writelock);
			 std::cout << "threads : " << nThreads << '\n'
			 << "threadId : " << threadId << '\n'
			 << "totalNumberOfReads : " << totalNumberOfReads << '\n'
			 << "nLocksForProcessedFlags : " << nLocksForProcessedFlags << '\n'
			 << "batchsize : " << batchsize << '\n'
			 << "actualBatchSize : " << actualBatchSize << '\n'
			 << "maxReadsPerLock : " << maxReadsPerLock << '\n'
			 << "readnum : " << readnum << '\n'
			 << "i : " << i << std::endl;
			 assert(std::uint32_t(batchlockindex) == (readnum + i) / maxReadsPerLock);
			 }

			 }*/
			std::unique_lock < std::mutex
					> lock(locksForProcessedFlags[batchlockindex]);
			for (std::uint32_t i = 0; i < actualBatchSize; i++) {
				if (readIsProcessedVector[readnum + i] == 0) {
					readIsProcessedVector[readnum + i] = 1;
					activeBatches[i] = true;
					nProcessedQueries++;
				} else {
					activeBatches[i] = false;
				}
			}
		} else {
			for (std::uint32_t i = 0; i < actualBatchSize; i++) {
				activeBatches[i] = true;
				nProcessedQueries++;
			}
		}

		// fetch the reads of current batch from the readstorage
		for (std::uint32_t i = 0; i < actualBatchSize; i++) {
			if (activeBatches[i]) {
				queries[i] = readStorage.fetchSequence_ptr(readnum + i);
				queryQualities[i] = readStorage.fetchQuality_ptr(readnum + i);
			}
		}

#ifdef ERRORCORRECTION_TIMING		
		tpa = std::chrono::system_clock::now();
#endif

		// for each read of the current batch, find their correction candidates
		for (std::uint32_t i = 0; i < actualBatchSize; i++) {
			if (activeBatches[i]) {
				queryStrings[i] = queries[i]->toString();
//				std::cout << "get candidates of read id " << (readnum + i) <<'\n';
				candidateIds[i] = minhasher.getCandidates(minhasherbuffers,
						queryStrings[i]);
			}
		}

#ifdef ERRORCORRECTION_TIMING
		tpb = std::chrono::system_clock::now();
		getCandidatesTimeTotal += tpb - tpa;

		tpa = std::chrono::system_clock::now();
#endif

#if 1
#if 0
		// map minhash ids to sequences
		for(std::uint32_t i = 0; i < actualBatchSize; i++) {
			if(activeBatches[i]) {
				sequenceToIdsMaps[i] = mapMinhashResultsToSequences(
						candidateIds[i],
						candidateReads[i],
						revComplcandidateReads[i],
						candidateQualities[i],
						revcomplcandidateQualities[i],
						frequencies[i],
						mapminhashresultsdedup,
						mapminhashresultsfetch);
			}
		}
#else

		for (std::uint32_t i = 0; i < actualBatchSize; i++) {
			if (activeBatches[i]) {
				const int nCandidates = candidateIds[i].size();
				if (nCandidates > 0) {
					candidateReads[i].resize(nCandidates);

					std::chrono::time_point < std::chrono::system_clock > t1 =
							std::chrono::system_clock::now();

					for (int k = 0; k < nCandidates; k++) {
						candidateReads[i][k] = readStorage.fetchSequence_ptr(
								candidateIds[i][k]);
					}

					std::chrono::time_point < std::chrono::system_clock > t2 =
							std::chrono::system_clock::now();

					mapminhashresultsfetch += (t2 - t1);

					t1 = std::chrono::system_clock::now();

					int n_unique_elements = -1;

//#ifdef __NVCC__
#if 0
					cudaSetDevice(minhashresultsdedupbuffers.deviceId);
					minhashresultsdedupbuffers.resize_initial_results(nCandidates);

					thrust::device_ptr<const Sequence*> d_candidateReads_ptr = thrust::device_pointer_cast(minhashresultsdedupbuffers.d_candidateReads);
					thrust::device_ptr<int> d_indexlist_ptr = thrust::device_pointer_cast(minhashresultsdedupbuffers.d_indexlist);

					cudaMemcpyAsync(d_candidateReads_ptr.get(),
							candidateReads[i].data(),
							sizeof(const Sequence*) * nCandidates,
							cudaMemcpyHostToDevice,
							minhashresultsdedupbuffers.stream); CUERR;
					cudaStreamSynchronize(minhashresultsdedupbuffers.stream); CUERR;

//std::cout << "readnum " << readnum << ", i " << i << std::endl;

					//thrust::copy(candidateReads[i].begin(), candidateReads[i].end(), d_candidateReads_ptr);
					thrust::sequence(thrust::cuda::par.on(minhashresultsdedupbuffers.stream),
							d_indexlist_ptr,
							d_indexlist_ptr + nCandidates);
					thrust::sort_by_key(thrust::cuda::par.on(minhashresultsdedupbuffers.stream),
							d_candidateReads_ptr,
							d_candidateReads_ptr + nCandidates,
							d_indexlist_ptr);

					//sort candidateIds by indexlist
					//thrust::copy(d_indexlist_ptr, d_indexlist_ptr + nCandidates, minhashresultsdedupbuffers.h_indexlist);
					cudaMemcpyAsync(minhashresultsdedupbuffers.h_indexlist,
							d_indexlist_ptr.get(),
							sizeof(int) * nCandidates,
							cudaMemcpyDeviceToHost,
							minhashresultsdedupbuffers.stream); CUERR;
					cudaStreamSynchronize(minhashresultsdedupbuffers.stream); CUERR;

					std::vector<std::uint64_t> tmp(nCandidates);
					for(int k = 0; k < nCandidates; k++) {
						tmp[k] = candidateIds[i][minhashresultsdedupbuffers.h_indexlist[k]];
					}
					candidateIds[i] = std::move(tmp);

					// count unique sequences
					n_unique_elements = thrust::inner_product(thrust::cuda::par.on(minhashresultsdedupbuffers.stream),
							d_candidateReads_ptr,
							d_candidateReads_ptr + nCandidates - 1,
							d_candidateReads_ptr + 1,
							int(1),
							thrust::plus<int>(),
							thrust::not_equal_to<const Sequence*>());

					minhashresultsdedupbuffers.resize_unique_results(n_unique_elements);

					thrust::device_ptr<const Sequence*> d_uniqueSequences_ptr = thrust::device_pointer_cast(minhashresultsdedupbuffers.d_uniqueSequences);
					thrust::device_ptr<int> d_frequencies_ptr = thrust::device_pointer_cast(minhashresultsdedupbuffers.d_frequencies);

					// save unique sequences and their frequencies
					thrust::reduce_by_key(thrust::cuda::par.on(minhashresultsdedupbuffers.stream),
							d_candidateReads_ptr, d_candidateReads_ptr + nCandidates,
							thrust::constant_iterator<int>(1),
							d_uniqueSequences_ptr,
							d_frequencies_ptr);

					candidateReads[i].resize(n_unique_elements);
					//thrust::copy(d_uniqueSequences_ptr, d_uniqueSequences_ptr + n_unique_elements, candidateReads[i].begin());
					cudaMemcpyAsync(candidateReads[i].data(),
							d_uniqueSequences_ptr.get(),
							sizeof(const Sequence*) * n_unique_elements,
							cudaMemcpyDeviceToHost,
							minhashresultsdedupbuffers.stream); CUERR;
					cudaStreamSynchronize(minhashresultsdedupbuffers.stream); CUERR;

					frequencies[i].resize(n_unique_elements);
					//thrust::copy(d_frequencies_ptr, d_frequencies_ptr + n_unique_elements, frequencies[i].begin());
					cudaMemcpyAsync(frequencies[i].data(),
							d_frequencies_ptr.get(),
							sizeof(int) * n_unique_elements,
							cudaMemcpyDeviceToHost,
							minhashresultsdedupbuffers.stream); CUERR;
					cudaStreamSynchronize(minhashresultsdedupbuffers.stream); CUERR;
#else					

					std::vector<int> indexlist(nCandidates);
					std::iota(indexlist.begin(), indexlist.end(), 0);

					//sort indexlist in order of sequence strings
					std::sort(indexlist.begin(), indexlist.end(),
							[&](int l, int r) {
								return candidateReads[i][l] < candidateReads[i][r];
							});

					//sort sequences
					std::sort(candidateReads[i].begin(),
							candidateReads[i].end());

					//sort candidateIds by indexlist
					std::vector < std::uint64_t > tmp(nCandidates);
					for (int k = 0; k < nCandidates; k++) {
						tmp[k] = candidateIds[i][indexlist[k]];
					}
					candidateIds[i] = std::move(tmp);

					// kind of like std::unique(candidateReads[i].begin(), candidateReads[i].end()),
					// but also count number of duplicates for each unique entry(inclusive) in sortedFreqs
					std::vector<int> sortedFreqs(1, 1);
					const Sequence* prevSeq = candidateReads[i][0];
					n_unique_elements = 1;
					for (int k = 1; k < nCandidates; k++) {
						const Sequence* curSeq = candidateReads[i][k];
						if (prevSeq == curSeq) {
							assert(*prevSeq == *curSeq);
							sortedFreqs.back()++;}
						else {
							sortedFreqs.push_back(1);
							candidateReads[i][n_unique_elements] = curSeq;
							n_unique_elements++;
						}
						prevSeq = curSeq;
					}

					candidateReads[i].resize(n_unique_elements);
					frequencies[i] = std::move(sortedFreqs);
#endif

#if 0
					if(uniquecount != n_unique_elements) {
						std::cout << "sequence dedup #unique elements wrong " << i << ". normal " << uniquecount << ", thrust " << n_unique_elements << std::endl;
					}
					for(int k = 0; k< uniquecount; k++) {
						if(candidateReads[i][k] != copy[k]) {
							std::cout << "sequence dedup unique element wrong " << i << " " << k << ". normal " << candidateReads[i][k] << ", thrust " << copy[k] << std::endl;
						}
						if(frequencies[i][k] != freqstmp[k]) {
							std::cout << "sequence dedup freq wrong " << i << " " << k << ". normal " << frequencies[i][k] << ", thrust " << freqstmp[k] << std::endl;
						}
					}
#endif
					//check
					int freqsum = 0;
					for (const auto f : frequencies[i])
						freqsum += f;
					assert(freqsum == nCandidates);

					t2 = std::chrono::system_clock::now();

					mapminhashresultsdedup += (t2 - t1);

					revComplcandidateReads[i].resize(n_unique_elements);

					t1 = std::chrono::system_clock::now();

					int fc = 0;
					for (int k = 0; k < n_unique_elements; k++) {
						revComplcandidateReads[i][k] =
								readStorage.fetchReverseComplementSequence_ptr(
										candidateIds[i][fc]);
						fc += frequencies[i][k];
					}
					t2 = std::chrono::system_clock::now();

					mapminhashresultsfetch += (t2 - t1);
				}
			}
		}

#endif
#endif

		/*for(int i = 0; i < candidateReads[0].size(); i++){

		 std::cout << candidateReads[0][i]->toString() << " " << frequencies[0][i] << std::endl;
		 }*/

#ifdef ERRORCORRECTION_TIMING
		tpb = std::chrono::system_clock::now();
		mapMinhashResultsToSequencesTimeTotal += tpb - tpa;
#endif

#if 1

#ifdef ERRORCORRECTION_TIMING		
		tpa = std::chrono::system_clock::now();
#endif

		for (std::uint32_t i = 0; i < actualBatchSize; i++) {
			candidateReadsAndRevcompls[i].resize(candidateReads[i].size() * 2);

			std::copy(candidateReads[i].begin(), candidateReads[i].end(),
					candidateReadsAndRevcompls[i].begin());

			std::copy(revComplcandidateReads[i].begin(),
					revComplcandidateReads[i].end(),
					candidateReadsAndRevcompls[i].begin()
							+ candidateReads[i].size());

			/*std::vector<const Sequence*> tmp(candidateReadsAndRevcompls[i]);
			 auto u = std::unique(tmp.begin(), tmp.end());
			 auto d = std::distance(u, tmp.end());
			 if(d > 0) std::cout << "killed " << d << " sequences before alignment\n";*/
		}

#if 1

		if (correctionmode == CorrectionMode::Hamming) {
			auto alignments = hammingtools::getMultipleAlignments(shddata,
					queries, candidateReadsAndRevcompls, activeBatches, true);

#ifdef ERRORCORRECTION_TIMING
			tpb = std::chrono::system_clock::now();
			getAlignmentsTimeTotal += tpb - tpa;

			tpa = std::chrono::system_clock::now();
#endif

			for (std::uint32_t i = 0; i < actualBatchSize; i++) {
				if (activeBatches[i]) {
					if (alignments[i].size()
							!= candidateReadsAndRevcompls[i].size()) {
						std::cout << readnum << '\n';
						assert(
								alignments[i].size()
										== candidateReadsAndRevcompls[i].size());
					}

					// for each candidate, compare its alignment to the alignment of the reverse complement.
					// find the best of both, if any, and
					// save the best alignment + additional data in these vectors
					std::vector<AlignResultCompact> insertedAlignments;
					std::vector<const Sequence*> insertedSequences;
					std::vector < std::uint64_t > insertedCandidateIds;
					std::vector<int> insertedFreqs;
					std::vector<bool> forwardRead;

					const int querylength = queries[i]->getNbases();

					int counts[3] { 0, 0, 0 };
					int countsDedup[3] { 0, 0, 0 };

					int bad = 0;
					int fc = 0;

					tpc = std::chrono::system_clock::now();

					for (size_t j = 0; j < alignments[i].size() / 2; j++) {
						auto& res = alignments[i][j];
						auto& revcomplres =
								alignments[i][candidateReads[i].size() + j];

						int candidatelength = candidateReads[i][j]->getNbases();

						BestAlignment_t best = get_best_alignment(res,
								revcomplres, querylength, candidatelength,
								MAX_MISMATCH_RATIO, MIN_OVERLAP,
								MIN_OVERLAP_RATIO);

						const int f = frequencies[i][j];

						if (best == BestAlignment_t::Forward) {
							bool useIt = true;
							const double mismatchratio = double(res.nOps)
									/ double(res.overlap);

							if (mismatchratio < 2 * errorrate) {
								counts[0] += f;
								countsDedup[0]++;
							}
							if (mismatchratio < 3 * errorrate) {
								counts[1] += f;
								countsDedup[1]++;
							}
							if (mismatchratio < 4 * errorrate) {
								counts[2] += f;
								countsDedup[2]++;
							} else {
								useIt = false;
							}

							if (useIt) {
								const Sequence* seq = candidateReads[i][j];

								insertedAlignments.push_back(std::move(res));
								insertedSequences.push_back(seq);
								insertedCandidateIds.insert(
										insertedCandidateIds.cend(),
										candidateIds[i].cbegin() + fc,
										candidateIds[i].cbegin() + fc + f);
								insertedFreqs.push_back(f);
								forwardRead.push_back(true);
							}

						} else if (best == BestAlignment_t::ReverseComplement) {
							bool useIt = true;
							const double mismatchratio = double(
									revcomplres.nOps)
									/ double(revcomplres.overlap);

							if (mismatchratio < 2 * errorrate) {
								counts[0] += f;
								countsDedup[0]++;
							}
							if (mismatchratio < 3 * errorrate) {
								counts[1] += f;
								countsDedup[1]++;
							}
							if (mismatchratio < 4 * errorrate) {
								counts[2] += f;
								countsDedup[2]++;
							} else {
								useIt = false;
							}

							if (useIt) {
								const Sequence* revseq =
										revComplcandidateReads[i][j];

								insertedAlignments.push_back(
										std::move(revcomplres));
								insertedSequences.push_back(revseq);
								insertedCandidateIds.insert(
										insertedCandidateIds.cend(),
										candidateIds[i].cbegin() + fc,
										candidateIds[i].cbegin() + fc + f);
								insertedFreqs.push_back(f);
								forwardRead.push_back(false);
							}
						} else {
							bad++; //both alignments are bad	
						}
						fc += f;
					}

					// check errorrate of good alignments. we want at least m_coverage * estimatedCoverage alignments.
					// if possible, we want to use only alignments with a max mismatch ratio of 2*errorrate
					// if there are not enough alignments, use max mismatch ratio of 3*errorrate
					// if there are not enough alignments, use max mismatch ratio of 4*errorrate
					// if there are not enough alignments, do not correct

					const int countThreshold = estimatedCoverage * m_coverage;
					bool correctQuery = false;
					double mismatchratioThreshold = 0;
					int candidatecount = 0;
					if (counts[0] >= countThreshold) {

						correctQuery = true;
						mismatchratioThreshold = 2 * errorrate;
						candidatecount = countsDedup[0];
						correctionCases[0]++;

					} else if (counts[1] >= countThreshold) {

						correctQuery = true;
						mismatchratioThreshold = 3 * errorrate;
						candidatecount = countsDedup[1];
						correctionCases[1]++;

					} else if (counts[2] >= countThreshold) {

						correctQuery = true;
						mismatchratioThreshold = 4 * errorrate;
						candidatecount = countsDedup[2];
						correctionCases[2]++;

					} else {
						correctQuery = false; //no correction
						correctionCases[3]++;
					}

					tpd = std::chrono::system_clock::now();

					determinegoodalignmentsTime += tpd - tpc;

					if (!correctQuery) {
						std::string header = *readStorage.fetchHeader_ptr(
								readnum + i);

						if (CORRECT_CANDIDATE_READS_TOO)
							resultstringstream << (readnum + i) << ' ';

						resultstringstream << header << '\n' << queryStrings[i]
								<< '\n';

						if (inputfileformat == Fileformat::FASTQ)
							resultstringstream << '+' << '\n'
									<< *(queryQualities[i]) << '\n';

						nBufferedResults++;
					} else {
						tpc = std::chrono::system_clock::now();
						//collect selected alignments
						std::vector<int> frequenciesPrefixSum(
								candidatecount + 1, 0);

						int newindex = 0;
						auto it = insertedCandidateIds.begin();
						auto it2 = insertedCandidateIds.begin();
						for (size_t k = 0; k < insertedAlignments.size(); k++) {
							auto& a = insertedAlignments[k];
							const double mismatchratio = double(a.nOps)
									/ double(a.overlap);
							if (mismatchratio < mismatchratioThreshold) {
								insertedAlignments[newindex] = std::move(a);
								insertedSequences[newindex] =
										insertedSequences[k];
								insertedFreqs[newindex] = insertedFreqs[k];
								frequenciesPrefixSum[newindex + 1] =
										frequenciesPrefixSum[newindex]
												+ insertedFreqs[k];
								forwardRead[newindex] = forwardRead[k];
								it = std::copy(it2, it2 + insertedFreqs[k], it);
								newindex++;
							}
							it2 += insertedFreqs[k];
						}
						if (candidatecount != newindex) {
							std::cout << "candidatecount " << candidatecount
									<< ", newindex " << newindex << " "
									<< counts[0] << " " << counts[1] << " "
									<< counts[2] << " " << countsDedup[0] << " "
									<< countsDedup[1] << " " << countsDedup[2]
									<< '\n';
							assert(candidatecount == newindex);
						}

						//get sequence strings and quality strings for candidates
						std::vector < std::string > candidateStrings;
						std::vector<const std::string*> candidatequals;
						candidateStrings.reserve(candidatecount);
						candidatequals.reserve(frequenciesPrefixSum.back());

						int qualindex = 0;
						for (int j = 0; j < candidatecount; j++) {
							candidateStrings.push_back(
									insertedSequences[j]->toString());
							const int freq = insertedFreqs[j];
							if (forwardRead[j]) {
								for (int f = 0; f < freq; f++) {
									candidatequals.push_back(
											readStorage.fetchQuality_ptr(
													insertedCandidateIds[qualindex
															+ f]));
								}
							} else {
								for (int f = 0; f < freq; f++) {
									candidatequals.push_back(
											readStorage.fetchReverseComplementQuality_ptr(
													insertedCandidateIds[qualindex
															+ f]));
								}
							}
							qualindex += freq;
						}

						tpd = std::chrono::system_clock::now();

						fetchgoodcandidatesTime += tpd - tpc;

						std::vector<bool> saveThisCandidate(candidatecount,
								false);

						tpc = std::chrono::system_clock::now();

						int status;
						std::chrono::duration<double> foo1;
						std::chrono::duration<double> foo2;
						std::tie(status, foo1, foo2) =
								hammingtools::performCorrection(
										hcorrectionbuffers, queryStrings[i],
										candidatecount, candidateStrings,
										insertedAlignments, *queryQualities[i],
										candidatequals, frequenciesPrefixSum,
										MAX_MISMATCH_RATIO, useQualityScores,
										saveThisCandidate,
										CORRECT_CANDIDATE_READS_TOO,
										estimatedCoverage, errorrate,
										m_coverage, minhashparams.k, true);

						tpd = std::chrono::system_clock::now();

						readcorrectionTimeTotal += tpd - tpc;

						majorityvotetime += foo1;
						basecorrectiontime += foo2;

						avgsupportfail += (((status >> 0) & 1) == 1);
						minsupportfail += (((status >> 1) & 1) == 1);
						mincoveragefail += (((status >> 2) & 1) == 1);
						sobadcouldnotcorrect += (((status >> 3) & 1) == 1);
						verygoodalignment += (status == 0);

						//assert(correctedQuery.size() == queryStrings[i].size());

						//bool correctedAndChanged = (queries[i]->operator!=(correctedQuery));

						std::string header = *readStorage.fetchHeader_ptr(
								readnum + i);

						resultstringstream << (readnum + i) << ' ';

						resultstringstream << header << '\n' << queryStrings[i]
								<< '\n';

						if (inputfileformat == Fileformat::FASTQ)
							resultstringstream << '+' << '\n'
									<< *(queryQualities[i]) << '\n';

						nBufferedResults++;

						if (CORRECT_CANDIDATE_READS_TOO) {
							int candidateIdIndex = 0;
							for (int j = 0; j < candidatecount; j++) {
								for (int f = 0; f < insertedFreqs[j]; f++) {
									if (saveThisCandidate[j]) {
										//check that candidate has not been corrected yet
										const int candidateId =
												insertedCandidateIds[candidateIdIndex];
										const int batchlockindex = candidateId
												/ maxReadsPerLock;
										bool savingIsOk = false;
										if (readIsProcessedVector[candidateId]
												== 0) {
											std::unique_lock < std::mutex
													> lock(
															locksForProcessedFlags[batchlockindex]);
											if (readIsProcessedVector[candidateId]
													== 0) {
												readIsProcessedVector[candidateId] =
														1; // we will process this read
												lock.unlock();
												savingIsOk = true;
												nCorrectedCandidates++;
											}
										}
										if (savingIsOk) {
											auto& s = candidateStrings[j];
											const int candidateId =
													insertedCandidateIds[candidateIdIndex];

											std::string header =
													*readStorage.fetchHeader_ptr(
															candidateId);
											resultstringstream << (candidateId)
													<< ' ';

											resultstringstream << header
													<< '\n';

											if (forwardRead[j])
												resultstringstream << s << '\n';
											else { //s is reverse complement, make reverse complement again
												std::string fwd =
														SequenceGeneral(s,
																false).reverseComplement().toString();
												resultstringstream << fwd
														<< '\n';
											}

											if (inputfileformat
													== Fileformat::FASTQ) {
												if (forwardRead[j])
													resultstringstream << '+'
															<< '\n'
															<< candidatequals[candidateIdIndex]
															<< '\n';
												else {
													// candidatequals contains reverse complement scores. fetch fwd scores. 
													auto qualptr =
															readStorage.fetchQuality_ptr(
																	candidateId);
													resultstringstream << '+'
															<< '\n' << *qualptr
															<< '\n';
												}
											}

											nBufferedResults++;
										}
									}
									candidateIdIndex++;
								}
							}
						}
					}
				}
			}
#ifdef ERRORCORRECTION_TIMING
			tpb = std::chrono::system_clock::now();
			correctReadTimeTotal += tpb - tpa;
#endif
			//std::exit(-1);

		} else {

			auto alignments = graphtools::getMultipleAlignments(sgadata,
					queries, candidateReadsAndRevcompls, activeBatches, true);

#ifdef ERRORCORRECTION_TIMING
			tpb = std::chrono::system_clock::now();
			getAlignmentsTimeTotal += tpb - tpa;

			tpa = std::chrono::system_clock::now();
#endif

			for (std::uint32_t i = 0; i < actualBatchSize; i++) {
				if (activeBatches[i]) {
					if (alignments[i].size()
							!= candidateReadsAndRevcompls[i].size()) {
						std::cout << readnum << '\n';
						assert(
								alignments[i].size()
										== candidateReadsAndRevcompls[i].size());
					}
				}

				// for each candidate, compare its alignment to the alignment of the reverse complement. 
				// find the best of both, if any, and
				// save the best alignment + additional data in these vectors
				std::vector<AlignResult> insertedAlignments;
				std::vector<const Sequence*> insertedSequences;
				std::vector < std::uint64_t > insertedCandidateIds;
				std::vector<int> insertedFreqs;
				std::vector<bool> forwardRead;

				const int querylength = queries[i]->getNbases();

				int bad = 0;
				int fc = 0;
				for (size_t j = 0; j < alignments[i].size() / 2; j++) {
					auto& res = alignments[i][j];
					auto& revcomplres = alignments[i][candidateReads[i].size()
							+ j];

					int candidatelength = candidateReads[i][j]->getNbases();

					BestAlignment_t best = get_best_alignment(res.arc,
							revcomplres.arc, querylength, candidatelength,
							MAX_MISMATCH_RATIO, MIN_OVERLAP, MIN_OVERLAP_RATIO);

					if (best == BestAlignment_t::Forward) {
						const Sequence* seq = candidateReads[i][j];

						insertedAlignments.push_back(std::move(res));
						insertedSequences.push_back(seq);
						insertedCandidateIds.insert(insertedCandidateIds.cend(),
								candidateIds[i].cbegin() + fc,
								candidateIds[i].cbegin() + fc
										+ frequencies[i][j]);
						insertedFreqs.push_back(frequencies[i][j]);
						forwardRead.push_back(true);
					} else if (best == BestAlignment_t::ReverseComplement) {
						const Sequence* revseq = revComplcandidateReads[i][j];

						insertedAlignments.push_back(std::move(revcomplres));
						insertedSequences.push_back(revseq);
						insertedCandidateIds.insert(insertedCandidateIds.cend(),
								candidateIds[i].cbegin() + fc,
								candidateIds[i].cbegin() + fc
										+ frequencies[i][j]);
						insertedFreqs.push_back(frequencies[i][j]);
						forwardRead.push_back(false);
					} else {
						bad++; //both alignments are bad	
					}
					fc += frequencies[i][j];
				}

				bool correctQuery = true;

				//TODO don't correct if not enough good candidates

				if (!correctQuery) {
					std::string header = *readStorage.fetchHeader_ptr(
							readnum + i);

					resultstringstream << (readnum + i) << ' ';

					resultstringstream << header << '\n' << queryStrings[i]
							<< '\n';

					if (inputfileformat == Fileformat::FASTQ)
						resultstringstream << '+' << '\n'
								<< *(queryQualities[i]) << '\n';

					nBufferedResults++;
				} else {

					// Now, use the good alignments for error correction. use errorgraph for correction.

					std::vector<int> frequenciesPrefixSum(
							insertedAlignments.size() + 1, 0);
					std::vector<const std::string*> candidatequals;
					candidatequals.reserve(insertedAlignments.size());

					int qualindex = 0;
					for (size_t j = 0; j < insertedAlignments.size(); j++) {
						const int freq = insertedFreqs[j];
						frequenciesPrefixSum[j + 1] = frequenciesPrefixSum[j]
								+ freq;
						if (forwardRead[j]) {
							for (int f = 0; f < freq; f++) {
								candidatequals.push_back(
										readStorage.fetchQuality_ptr(
												insertedCandidateIds[qualindex
														+ f]));
							}
						} else {
							for (int f = 0; f < freq; f++) {
								candidatequals.push_back(
										readStorage.fetchReverseComplementQuality_ptr(
												insertedCandidateIds[qualindex
														+ f]));
							}
						}
						qualindex += freq;
					}

					std::string newcorrected = queryStrings[i];

					tpc = std::chrono::system_clock::now();

					std::chrono::duration<double> foo1;
					std::chrono::duration<double> foo2;
					std::tie(foo1, foo2) = graphtools::performCorrection(
							newcorrected, insertedAlignments,
							*queryQualities[i], candidatequals,
							frequenciesPrefixSum, useQualityScores,
							MAX_MISMATCH_RATIO, graphalpha, graphx);

					tpd = std::chrono::system_clock::now();

					readcorrectionTimeTotal += tpd - tpc;
					graphbuildtime += foo1;
					graphcorrectiontime += foo2;

					std::string header = *readStorage.fetchHeader_ptr(
							readnum + i);

					resultstringstream << (readnum + i) << ' ';

					resultstringstream << header << '\n' << newcorrected
							<< '\n';

					if (inputfileformat == Fileformat::FASTQ)
						resultstringstream << '+' << '\n'
								<< *(queryQualities[i]) << '\n';

					nBufferedResults++;
				}
			}

#ifdef ERRORCORRECTION_TIMING
			tpb = std::chrono::system_clock::now();
			correctReadTimeTotal += tpb - tpa;
#endif
		}
#endif

		//std::cout << "foo" << std::endl;
		//std::exit(-1);

#endif

#if 1		
		// write result to output file if output buffer is full
		if (nBufferedResults >= bufferedResultsThreshold) {
#ifdef ERRORCORRECTION_TIMING
			tpa = std::chrono::system_clock::now();
#endif

			std::lock_guard < std::mutex > lg(writelock);

			outputfile << resultstringstream.rdbuf();
			nBufferedResults = 0;
			resultstringstream.str(std::string());
			resultstringstream.clear();

#ifdef ERRORCORRECTION_TIMING
			tpb = std::chrono::system_clock::now();
			fileoutputTimeTotal += tpb - tpa;
#endif
		}
#endif

		// update local progress
		progressprocessedReads += actualBatchSize;

		// update global progress
		if (progressprocessedReads > progressThreshold) {
			updateGlobalProgress(progressprocessedReads, totalNumberOfReads);
			progressprocessedReads = 0;
		}

	}

#if 1	
	// write remaining buffered results
	if (nBufferedResults > 0) {

#ifdef ERRORCORRECTION_TIMING
		tpa = std::chrono::system_clock::now();
#endif

		std::lock_guard < std::mutex > lg(writelock);

		outputfile << resultstringstream.rdbuf();
		nBufferedResults = 0;
		resultstringstream.str(std::string());
		resultstringstream.clear();

#ifdef ERRORCORRECTION_TIMING
		tpb = std::chrono::system_clock::now();
		fileoutputTimeTotal += tpb - tpa;
#endif

	}
#endif	

	//final progress update
	updateGlobalProgress(progressprocessedReads, totalNumberOfReads);

	{
		std::lock_guard < std::mutex > lg(writelock);
		std::cout << "thread " << threadId << " processed candidates "
				<< processedCandidates << std::endl;
		std::cout << "thread " << threadId << " processed " << nProcessedQueries
				<< " queries" << std::endl;
		std::cout << "thread " << threadId << " corrected "
				<< nCorrectedCandidates << " candidates" << std::endl;
		std::cout << "thread " << threadId << " avgsupportfail "
				<< avgsupportfail << std::endl;
		std::cout << "thread " << threadId << " minsupportfail "
				<< minsupportfail << std::endl;
		std::cout << "thread " << threadId << " mincoveragefail "
				<< mincoveragefail << std::endl;
		std::cout << "thread " << threadId << " sobadcouldnotcorrect "
				<< sobadcouldnotcorrect << std::endl;
		std::cout << "thread " << threadId << " verygoodalignment "
				<< verygoodalignment << std::endl;
		std::cout << "thread " << threadId << " correctionCases "
				<< correctionCases[0] << " " << correctionCases[1] << " "
				<< correctionCases[2] << " " << correctionCases[3] << " "
				<< std::endl;

	}

#ifdef ERRORCORRECTION_TIMING
	{
		if (correctionmode == CorrectionMode::Hamming) {
			std::lock_guard < std::mutex > lg(writelock);
			std::cout << "thread " << threadId << " : getCandidatesTimeTotal "
					<< getCandidatesTimeTotal.count() << '\n';
			std::cout << "thread " << threadId << " : mapminhashresultsdedup "
					<< mapminhashresultsdedup.count() << '\n';
			std::cout << "thread " << threadId << " : mapminhashresultsfetch "
					<< mapminhashresultsfetch.count() << '\n';
			std::cout << "thread " << threadId
					<< " : mapMinhashResultsToSequencesTimeTotal "
					<< mapMinhashResultsToSequencesTimeTotal.count() << '\n';
			std::cout << "thread " << threadId << " : alignment resize buffer "
					<< shddata.resizetime.count() << '\n';
			std::cout << "thread " << threadId << " : alignment preprocessing "
					<< shddata.preprocessingtime.count() << '\n';
			std::cout << "thread " << threadId << " : alignment H2D "
					<< shddata.h2dtime.count() << '\n';
			std::cout << "thread " << threadId << " : alignment calculation "
					<< shddata.alignmenttime.count() << '\n';
			std::cout << "thread " << threadId << " : alignment D2H "
					<< shddata.d2htime.count() << '\n';
			std::cout << "thread " << threadId << " : alignment postprocessing "
					<< shddata.postprocessingtime.count() << '\n';
			std::cout << "thread " << threadId
					<< " : correction find good alignments "
					<< determinegoodalignmentsTime.count() << '\n';
			std::cout << "thread " << threadId
					<< " : correction fetch good data "
					<< fetchgoodcandidatesTime.count() << '\n';
			std::cout << "thread " << threadId << " : pileup vote "
					<< majorityvotetime.count() << '\n';
			std::cout << "thread " << threadId << " : pileup correct "
					<< basecorrectiontime.count() << '\n';
			std::cout << "thread " << threadId << " : correction calculation "
					<< readcorrectionTimeTotal.count() << '\n';
			std::cout << "thread " << threadId << " : correctReadTimeTotal "
					<< correctReadTimeTotal.count() << '\n';
			std::cout << "thread " << threadId << " : pileup resize buffer "
					<< hcorrectionbuffers.resizetime.count() << '\n';
			std::cout << "thread " << threadId << " : pileup preprocessing "
					<< hcorrectionbuffers.preprocessingtime.count() << '\n';
			std::cout << "thread " << threadId << " : pileup H2D "
					<< hcorrectionbuffers.h2dtime.count() << '\n';
			std::cout << "thread " << threadId << " : pileup calculation "
					<< hcorrectionbuffers.correctiontime.count() << '\n';
			std::cout << "thread " << threadId << " : pileup D2H "
					<< hcorrectionbuffers.d2htime.count() << '\n';
			std::cout << "thread " << threadId << " : pileup postprocessing "
					<< hcorrectionbuffers.postprocessingtime.count() << '\n';
		} else if (correctionmode == CorrectionMode::Graph) {
			std::cout << "thread " << threadId << " : getCandidatesTimeTotal "
					<< getCandidatesTimeTotal.count() << '\n';
			std::cout << "thread " << threadId << " : mapminhashresultsdedup "
					<< mapminhashresultsdedup.count() << '\n';
			std::cout << "thread " << threadId << " : mapminhashresultsfetch "
					<< mapminhashresultsfetch.count() << '\n';
			std::cout << "thread " << threadId
					<< " : mapMinhashResultsToSequencesTimeTotal "
					<< mapMinhashResultsToSequencesTimeTotal.count() << '\n';
			std::cout << "thread " << threadId << " : alignment total "
					<< getAlignmentsTimeTotal.count() << '\n';
			//std::cout << "thread " << threadId << " : alignment resize buffer " << shddata.resizetime.count() << '\n';
			//std::cout << "thread " << threadId << " : alignment preprocessing " << shddata.preprocessingtime.count() << '\n';
			//std::cout << "thread " << threadId << " : alignment H2D " << shddata.h2dtime.count() << '\n';
			//std::cout << "thread " << threadId << " : alignment calculation " << shddata.alignmenttime.count() << '\n';
			//std::cout << "thread " << threadId << " : alignment D2H " << shddata.d2htime.count() << '\n';
			//std::cout << "thread " << threadId << " : alignment postprocessing " << shddata.postprocessingtime.count() << '\n';
			//std::cout << "thread " << threadId << " : correction find good alignments " << determinegoodalignmentsTime.count() << '\n';
			//std::cout << "thread " << threadId << " : correction fetch good data " << fetchgoodcandidatesTime.count() << '\n';
			std::cout << "thread " << threadId << " : graph build "
					<< graphbuildtime.count() << '\n';
			std::cout << "thread " << threadId << " : graph correct "
					<< graphcorrectiontime.count() << '\n';
			std::cout << "thread " << threadId << " : correction calculation "
					<< readcorrectionTimeTotal.count() << '\n';
			std::cout << "thread " << threadId << " : correctReadTimeTotal "
					<< correctReadTimeTotal.count() << '\n';
		}
	}
#endif

	hammingtools::cuda_cleanup_SHDdata(shddata);
	graphtools::cuda_cleanup_AlignerDataArrays(sgadata);
	cuda_cleanup_MinhasherBuffers(minhasherbuffers);
	cuda_cleanup_MinhashResultsDedupBuffers(minhashresultsdedupbuffers);
	hammingtools::cuda_cleanup_CorrectionBuffers(hcorrectionbuffers);
}

std::uint64_t ErrorCorrector::getReadPos(const std::string& readheader) const {
	char dir = 'f';
	auto slash = readheader.find("/");
	if (slash != std::string::npos) {
		dir = readheader[slash + 1];
	}

	if (dir != '1' && dir != '2') {
		std::cout << "dir = " << dir << std::endl;
		std::cout << readheader << std::endl;
		throw std::runtime_error("getReadPos : unexpected header format");
	}

	/* the following code extracts the position of the read from the header
	 i.e. if the header is @gi|556503834|ref|NC_000913.3|_4064476_4064981_0:0:0_0:0:0_0/1 , we need to extract 4064476
	 */
	// don't know if / how many '_' are used in read header. need to find all '_' and then count backwards to get to wanted position
	std::vector<int> locations;
	for (size_t i = 0; i < readheader.size(); i++)
		if (readheader[i] == '_')
			locations.push_back(i);

	int offset_left;
	int offset_right;
	if (dir == '1') {
		offset_left = 5;
		offset_right = 4;
	} else {  //dir == '2'
		offset_left = 4;
		offset_right = 3;
	}

	int substringpos = locations[locations.size() - offset_left] + 1;
	int substringlength = locations[locations.size() - offset_right]
			- locations[locations.size() - offset_left];

	const std::uint64_t pos = std::stoll(
			readheader.substr(substringpos, substringlength));
	return pos;
}

void ErrorCorrector::setOutputPath(const std::string& path) {
	outputPath = path;

	std::experimental::filesystem::create_directories(path);
}

void ErrorCorrector::setGraphSettings(double alpha, double x) {
	graphalpha = alpha;
	graphx = x;
}

void ErrorCorrector::updateGlobalProgress(std::uint64_t increment,
		std::uint64_t maxglobalprogress) {
	std::lock_guard < std::mutex > lock(progresslock);
	progress += increment;

	printf("Progress: %3.2f %%\r",
			((progress * 1.0 / maxglobalprogress) * 100.0));
	std::cout << std::flush;
}

void ErrorCorrector::setOutputFilename(const std::string& filename) {
	outputFilename = filename;
}

void ErrorCorrector::setBatchsize(int n) {
	if (n < 1)
		throw std::runtime_error("batchsize must be > 0");

	batchsize = n;
}

void ErrorCorrector::setAlignmentScores(int matchscore, int subscore,
		int insertscore, int delscore) {
	ALIGNMENTSCORE_MATCH = matchscore;
	ALIGNMENTSCORE_SUB = subscore;
	ALIGNMENTSCORE_INS = insertscore;
	ALIGNMENTSCORE_DEL = delscore;
}

void ErrorCorrector::setMaxMismatchRatio(double ratio) {
	if (ratio < 0.0 || ratio > 1)
		throw std::runtime_error(
				"max mismatch ratio must be >= 0.0 and <= 1.0");

	MAX_MISMATCH_RATIO = ratio;
}

void ErrorCorrector::setMinimumAlignmentOverlap(int overlap) {
	if (overlap < 0)
		throw std::runtime_error("batchsize must be >= 0");

	MIN_OVERLAP = overlap;
}

void ErrorCorrector::setMinimumAlignmentOverlapRatio(double ratio) {
	if (ratio < 0.0 || ratio > 1)
		throw std::runtime_error(
				"min alignment overlap ratio must be >= 0.0 and <= 1.0");

	MIN_OVERLAP_RATIO = ratio;
}

void ErrorCorrector::setFileFormat(const std::string& format) {
	if (format == "fasta")
		inputfileformat = Fileformat::FASTA;
	else if (format == "fastq")
		inputfileformat = Fileformat::FASTQ;
	else
		throw std::runtime_error("Set invalid file format : " + format);

	std::cout << "Set file format to " << format << std::endl;
}

void ErrorCorrector::setUseQualityScores(bool val) {
	useQualityScores = val;
}

void ErrorCorrector::setEstimatedCoverage(int cov) {
	if (cov < 1)
		throw std::runtime_error("set invalid estimated coverage");

	estimatedCoverage = cov;
}

void ErrorCorrector::setEstimatedErrorRate(double rate) {
	if (rate < 0 || rate >= 1.0)
		throw std::runtime_error("set invalid estimated error rate");

	errorrate = rate;
}

void ErrorCorrector::setM(double m) {
	if (m < 0)
		throw std::runtime_error("set invalid m");

	m_coverage = m;
}

