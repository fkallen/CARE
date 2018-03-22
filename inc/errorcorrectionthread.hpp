#ifndef CARE_ERROR_CORRECTION_THREAD_HPP
#define CARE_ERROR_CORRECTION_THREAD_HPP

#include "batchelem.hpp"
#include "minhasher.hpp"
#include "readstorage.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace care{

enum class CorrectionMode {Hamming, Graph};

struct BatchGenerator{
    BatchGenerator(){}
    BatchGenerator(std::uint32_t firstBatch, std::uint32_t lastBatch, std::uint32_t batchsize);
    BatchGenerator(std::uint32_t totalNumberOfReads, std::uint32_t batchsize, int threadId, int nThreads);
    BatchGenerator(const BatchGenerator& rhs);
    BatchGenerator(BatchGenerator&& rhs);
    BatchGenerator& operator=(const BatchGenerator& rhs);
    BatchGenerator& operator=(BatchGenerator&& rhs);

    std::vector<std::uint32_t> getNextReadIds();
private:
    std::uint32_t batchsize;
    std::uint32_t firstBatch;
    std::uint32_t lastBatch;
    std::uint32_t currentBatch;
};

struct CorrectionOptions{
    CorrectionMode correctionMode = CorrectionMode::Hamming;
    bool correctCandidates = false;
    bool useQualityScores = true;
    int alignmentscore_match = 1;
    int alignmentscore_sub = -1;
    int alignmentscore_ins = -100;
    int alignmentscore_del = -100;
    int min_overlap = 35;
    int kmerlength = 16;
    double max_mismatch_ratio = 0.2;
    double min_overlap_ratio = 0.35;
    double estimatedCoverage = 1.0;
    double errorrate = 0.01;
    double m_coverage = 0.6;
    double graphalpha = 1.0;
    double graphx = 1.5;
    int maximum_sequence_length = 0;
};

struct CorrectionThreadOptions{
    int threadId;
    int deviceId;

    std::string outputfile;
    BatchGenerator* batchGen;
    const Minhasher* minhasher;
    const ReadStorage* readStorage;
    std::mutex* coutLock;
    std::vector<char>* readIsProcessedVector;
    std::mutex* locksForProcessedFlags;
    std::size_t nLocksForProcessedFlags;
};

struct ErrorCorrectionThread{
    CorrectionOptions opts;
    CorrectionThreadOptions threadOpts;

    std::uint32_t nProcessedReads = 0;

    DetermineGoodAlignmentStats goodAlignmentStats;
    int duplicates = 0;
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

    void run();
    void join();
private:
    void execute();
};

}


#endif
