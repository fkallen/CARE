#include "../inc/errorcorrector.hpp"
#include "../inc/read.hpp"
#include "../inc/sequencefileio.hpp"
#include "../inc/binarysequencehelpers.hpp"
#include "../inc/build.hpp"

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

namespace care{

//constexpr int MAX_THREADS_PER_GPU = 30;

//the most probable path in the errorgraph must have a probability of at least MINIMUM_CORRECTION_PROBABILITY
//constexpr double MINIMUM_CORRECTION_PROBABILITY = 0.0;

constexpr bool CORRECT_CANDIDATE_READS_TOO = false;

ErrorCorrector::ErrorCorrector(const Args& args,
		int nInserterThreads_, int nCorrectorThreads_) :
		args(args), nInserterThreads(nInserterThreads_), nCorrectorThreads(
				nCorrectorThreads_){

    if(!args.isValid()){
        throw std::runtime_error("Invalid arguments!");
    }

    minhashparams = args.getMinhashOptions();

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

void ErrorCorrector::correct_impl(CorrectionOptions& opts, const std::string& filename, FileFormat format, const std::string& outputfilename){
    auto nReads = getNumberOfReads(filename, format);

    Minhasher minhasher(minhashparams);
    ReadStorage readStorage;
    readStorage.setUseQualityScores(useQualityScores);

    int minlen, maxlen;
    std::cout << "begin build" << std::endl;

	TIMERSTARTCPU(BUILD);
    build(filename, format, readStorage, minhasher, nInserterThreads, minlen, maxlen);
	TIMERSTOPCPU(BUILD);

    std::cout << "min sequence length " << minlen << ", max sequence length " << maxlen << '\n';

    opts.maximum_sequence_length = maxlen;

    TIMERSTARTCPU(MAP_TRANSFORM);
	minhasher.transform();
	TIMERSTOPCPU(MAP_TRANSFORM);

	TIMERSTARTCPU(readstorage_transform);
	readStorage.noMoreInserts();
	TIMERSTOPCPU(readstorage_transform);

    std::vector<std::string> tmpfiles;
    for(int i = 0; i < nCorrectorThreads; i++){
        tmpfiles.emplace_back(outputfilename + "_tmp_" + std::to_string(i));
        std::cout << tmpfiles.back();
    }

    std::cout << "begin correct" << std::endl;

	TIMERSTARTCPU(CORRECT);

    std::vector<BatchGenerator> generators(nCorrectorThreads);
    std::vector<ErrorCorrectionThread> ecthreads(nCorrectorThreads);

    std::mutex writelock;

    for(int threadId = 0; threadId < nCorrectorThreads; threadId++){

        generators[threadId] = BatchGenerator(nReads, batchsize, threadId, nCorrectorThreads);
        CorrectionThreadOptions threadOpts;
        threadOpts.threadId = threadId;
        threadOpts.deviceId = deviceIds.size() == 0 ? -1 : deviceIds[threadId % deviceIds.size()];
        threadOpts.outputfile = tmpfiles[threadId];
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

    std::uint64_t maxprogress = nReads;
    std::uint64_t progress = 0;
    while(progress < maxprogress){
        progress = 0;
        for(int threadId = 0; threadId < nCorrectorThreads; threadId++){
            progress += ecthreads[threadId].nProcessedReads;
        }
        printf("Progress: %3.2f %%\r",
    			((progress * 1.0 / maxprogress) * 100.0));
    	std::cout << std::flush;
        if(progress < maxprogress)
		std::this_thread::sleep_for(std::chrono::seconds(5));
    }


	for (auto& thread : ecthreads)
		thread.join();

    printf("Progress: %3.2f %%\n", 100.00);


	TIMERSTOPCPU(CORRECT);

	std::cout << "end correct" << std::endl;

    if (CORRECT_CANDIDATE_READS_TOO) {
		int asd = std::count_if(readIsProcessedVector.begin(),
				readIsProcessedVector.end(), [](auto b) {return b;});
		std::cout << "total corrected reads: " << asd << std::endl;
	}

	minhasher.init(0);
	readStorage.destroy();


	std::cout << "begin merge" << std::endl;
    mergeResultFiles(nReads, filename, format, tmpfiles, outputfilename);
    deleteFiles(tmpfiles);

	std::cout << "end merge" << std::endl;

}

void ErrorCorrector::correct(const std::string& filename, const std::string& format, const std::string& outputfilename) {
    FileFormat inputfileformat = FileFormat::FASTQ;

    if (format == "fastq")
		inputfileformat = FileFormat::FASTQ;
	else
		throw std::runtime_error("Set invalid file format : " + format);

	std::cout << "Set file format to " << format << std::endl;
	//if (format == FileFormat::FASTA)
	//	setUseQualityScores(false);

    if (CORRECT_CANDIDATE_READS_TOO) {
        auto nReads = getNumberOfReads(filename, inputfileformat);

		readIsProcessedVector.resize(nReads, 0);
		nLocksForProcessedFlags = batchsize * nCorrectorThreads * 1000;
		locksForProcessedFlags.reset(new std::mutex[nLocksForProcessedFlags]);
	}

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

    correct_impl(opts, filename, inputfileformat, outputfilename);

	readIsProcessedVector.clear();
    locksForProcessedFlags.reset();
}

void ErrorCorrector::setGraphSettings(double alpha, double x) {
	graphalpha = alpha;
	graphx = x;
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
