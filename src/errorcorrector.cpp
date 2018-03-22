#include "../inc/errorcorrector.hpp"
#include "../inc/read.hpp"
#include "../inc/fastareader.hpp"
#include "../inc/fastqreader.hpp"
#include "../inc/binarysequencehelpers.hpp"

#include "../inc/ganja/hpc_helpers.cuh"

#include "../inc/hammingtools.hpp"
#include "../inc/graphtools.hpp"

#include "../inc/errorcorrectionthread.hpp"
#include "../inc/batchelem.hpp"
#include "../inc/pileup.hpp"
#include "../inc/graph.hpp"

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

#include <cuda_profiler_api.h>
#endif

#include <experimental/filesystem> // create_directories

#define ERRORCORRECTION_TIMING

namespace care{

//constexpr int MAX_THREADS_PER_GPU = 30;

// update global progress after correcting a multiple of this many reads
constexpr std::uint64_t progressThreshold = 5000;

//the most probable path in the errorgraph must have a probability of at least MINIMUM_CORRECTION_PROBABILITY
//constexpr double MINIMUM_CORRECTION_PROBABILITY = 0.0;

constexpr bool CORRECT_CANDIDATE_READS_TOO = false;


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

/*
    Deletes every file in vector filenames
*/
void deleteFiles(std::vector<std::string> filenames){
    for (const auto& filename : filenames) {
        int ret = std::remove(filename.c_str());
        if (ret != 0){
            const std::string errormessage = "Could not remove file " + filename;
            std::perror(errormessage.c_str());
        }
    }
}

/*
    Merges temporary results with unordered reads into single file outputfile with ordered reads.
    Temporary result files are expected to be in format:

    readnumber
    sequence
    readnumber
    sequence
    ...
*/
void mergeResultFiles(std::uint32_t expectedNumReads, const std::string& originalReadFile,
                      Fileformat originalFormat,
                      const std::vector<std::string>& filesToMerge, const std::string& outputfile){
    std::vector<Read> reads(expectedNumReads);
    std::uint32_t nreads = 0;

    for(const auto& filename : filesToMerge){
        std::ifstream is(filename);
        if(!is)
            throw std::runtime_error("could not open tmp file: " + filename);

        std::string num;
        std::string seq;

        while(true){
            std::getline(is, num);
    		if (!is.good())
    			break;
            std::getline(is, seq);
            if (!is.good())
                break;

            nreads++;

            auto readnum = std::stoull(num);
            reads[readnum].sequence = std::move(seq);
        }
    }

    if (nreads != expectedNumReads){
		std::cout << "WARNING. Expected " << expectedNumReads
                  << " reads in results, but found only "
                  << nreads << " reads. Results may not be correct!" << std::endl;
	}

    std::unique_ptr<ReadReader> reader;
	switch (originalFormat) {
	case Fileformat::FASTQ:
		reader.reset(new FastqReader(originalReadFile));
		break;
	case Fileformat::FASTA:
		throw std::runtime_error("Merging FASTA is currently not supported.");
	default:
		throw std::runtime_error("Merging: Invalid file format.");
	}

    Read read;
	std::uint32_t readnum = 0;
	while (reader->getNextRead(&read, &readnum)) {
        reads[readnum].header = std::move(read.header);
        reads[readnum].quality = std::move(read.quality);
		readnum++;
	}

	std::ofstream outputstream(outputfile);

	for (const auto& read : reads) {
		outputstream << read.header << '\n' << read.sequence << '\n';

		if (originalFormat == Fileformat::FASTQ)
			outputstream << '+' << '\n' << read.quality << '\n';
	}

	outputstream.flush();
	outputstream.close();
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
	minhasher.init(nReads);

	readStorage.init(nReads);
	readStorage.setUseQualityScores(useQualityScores);

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

    std::vector<std::string> tmpfiles;
    for(int i = 0; i < nCorrectorThreads; i++){
        //tmpfiles.emplace_back(outputPath + "/" + filename + "_tmp_" + std::to_string(i));
        tmpfiles.emplace_back(outputPath + "/" + std::to_string(i));
    }

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

	minhasher.init(0);
	readStorage.destroy();
	readIsProcessedVector.clear();

#endif
	std::cout << "begin merge" << std::endl;

    mergeResultFiles(nReads, filename, inputfileformat, tmpfiles, outputPath + "/" + outputFilename);
    deleteFiles(tmpfiles);

	std::cout << "end merge" << std::endl;
}

void ErrorCorrector::insertFile(const std::string& filename,
		bool buildHashmap) {

//single-threaded insertion
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
    int Ncount = 0;
    char bases[4]{'A', 'C', 'G', 'T'};
    int maxlength = 0;
    int minlength = std::numeric_limits<int>::max();

	while (reader->getNextRead(&read, &readnum)) {

		//replace 'N' with 'A'
        for(auto& c : read.sequence){
            if(c == 'N'){
                c = bases[Ncount];
                Ncount = (Ncount + 1) % 4;
            }
        }

        int len = int(read.sequence.length());
        if(len > maxlength)
            maxlength = len;
        if(len < minlength)
            minlength = len;

		if(buildHashmap) minhasher.insertSequence(read.sequence, readnum);

		readStorage.insertRead(readnum, read);

		progressprocessedReads++;

		// update global progress
		if(readnum > 3*progressThreshold) {
			updateGlobalProgress(progressprocessedReads, totalNumberOfReads);
			progressprocessedReads = 0;
		}
	}
    std::cout << "min sequence length " << minlength << ", max sequence length " << maxlength << '\n';

    maximum_sequence_length = maxlength;

	progress = 0;

//multi-threaded insertion
#else

	std::vector<std::future<std::pair<int,int>>> inserterThreads;
	progress = 0;

	for (int threadId = 0; threadId < nInserterThreads; ++threadId) {

		inserterThreads.emplace_back(
				std::async(std::launch::async,
						[&, threadId]()->std::pair<int,int> {

							std::uint64_t progressprocessedReads = 0;
							std::uint64_t totalNumberOfReads = readsPerFile.at(filename);

							int maxlength = 0;
							int minlength = std::numeric_limits<int>::max();

							std::pair<Read, std::uint32_t> pair = buffers[threadId].get();
							int Ncount = 0;
							char bases[4]{'A', 'C', 'G', 'T'};
							while (pair != buffers[threadId].defaultValue) {
								Read& read = pair.first;
								const std::uint32_t& readnum = pair.second;

								//replace 'N' with "random" base
								for(auto& c : read.sequence){
									if(c == 'N'){
										c = bases[Ncount];
										Ncount = (Ncount + 1) % 4;
									}
								}

								int len = int(read.sequence.length());
								if(len > maxlength)
									maxlength = len;
								if(len < minlength)
									minlength = len;

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

							return {minlength,maxlength};
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
	int minlen = std::numeric_limits<int>::max();
	for (size_t i = 0; i < inserterThreads.size(); ++i) {
		auto res = inserterThreads[i].get();
		if (res.second > maxlen)
			maxlen = res.second;
		if (res.first < minlen)
			minlen = res.first;
		buffers[i].reset();
	}

	std::cout << "min sequence length " << minlen << ", max sequence length " << maxlen << '\n';

	maximum_sequence_length = maxlen;

	progress = 0;

#endif
}

void ErrorCorrector::errorcorrectFile(const std::string& filename) {



    CorrectionOptions opts;
    opts.correctCandidates = false;
    opts.useQualityScores = useQualityScores;
    opts.alignmentscore_match = alignmentscore_match;
    opts.alignmentscore_sub = alignmentscore_sub;
    opts.alignmentscore_ins = alignmentscore_ins;
    opts.alignmentscore_del = alignmentscore_del;
    opts.min_overlap = min_overlap;
    opts.kmerlength = minhashparams.k;
    opts.max_mismatch_ratio = max_mismatch_ratio;
    opts.min_overlap_ratio = min_overlap_ratio;
    opts.estimatedCoverage = estimatedCoverage;
    opts.errorrate = errorrate;
    opts.m_coverage = m_coverage;
    opts.graphalpha = graphalpha;
    opts.graphx = graphx;
    opts.maximum_sequence_length = maximum_sequence_length;

    std::vector<BatchGenerator> generators(nCorrectorThreads);
    std::vector<ErrorCorrectionThread> ecthreads(nCorrectorThreads);
    std::vector<std::thread> consumerthreads;

    for(int threadId = 0; threadId < nCorrectorThreads; threadId++){

        generators[threadId] = BatchGenerator(readsPerFile.at(filename), batchsize, threadId, nCorrectorThreads);
        CorrectionThreadOptions threadOpts;
        threadOpts.threadId = threadId;
        threadOpts.deviceId = deviceIds.size() == 0 ? -1 : deviceIds[threadId % deviceIds.size()];
        threadOpts.outputfile = outputPath + "/" + std::to_string(threadId);
        threadOpts.batchGen = &generators[threadId];
        threadOpts.minhasher = &minhasher;
        threadOpts.readStorage = &readStorage;
        threadOpts.coutLock = &writelock;
        threadOpts.readIsProcessedVector = &readIsProcessedVector;
        threadOpts.locksForProcessedFlags = locksForProcessedFlags.get();
        threadOpts.nLocksForProcessedFlags = nLocksForProcessedFlags;

        ecthreads[threadId].opts = opts;
        ecthreads[threadId].threadOpts = threadOpts;

        ecthreads[threadId].run();
    }

    std::uint32_t maxprogress = readsPerFile.at(filename);
    std::uint32_t progress = 0;
    while(progress < maxprogress){
        progress = 0;
        for(int threadId = 0; threadId < nCorrectorThreads; threadId++){
            progress += ecthreads[threadId].nProcessedReads;
        }
        printf("Progress: %3.2f %%\r",
    			((progress * 1.0 / maxprogress) * 100.0));
    	std::cout << std::flush;
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }


	for (auto& thread : ecthreads)
		thread.join();

    printf("Progress: %3.2f %%\n", 100.00);
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
	alignmentscore_match= matchscore;
	alignmentscore_sub = subscore;
	alignmentscore_ins = insertscore;
	alignmentscore_del = delscore;
}

void ErrorCorrector::setMaxMismatchRatio(double ratio) {
	if (ratio < 0.0 || ratio > 1)
		throw std::runtime_error(
				"max mismatch ratio must be >= 0.0 and <= 1.0");

	max_mismatch_ratio = ratio;
}

void ErrorCorrector::setMinimumAlignmentOverlap(int overlap) {
	if (overlap < 0)
		throw std::runtime_error("batchsize must be >= 0");

	min_overlap = overlap;
}

void ErrorCorrector::setMinimumAlignmentOverlapRatio(double ratio) {
	if (ratio < 0.0 || ratio > 1)
		throw std::runtime_error(
				"min alignment overlap ratio must be >= 0.0 and <= 1.0");

	min_overlap_ratio = ratio;
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

}
