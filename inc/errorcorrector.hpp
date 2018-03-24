#ifndef ERRORCORRECTOR_HPP
#define ERRORCORRECTOR_HPP

#include "errorcorrectionthread.hpp"
#include "minhasher.hpp"
#include "read.hpp"
#include "readstorage.hpp"
#include "sequencefileio.hpp"
#include "args.hpp"

#include <cstdint>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <tuple>

namespace care{

class Barrier
{

 public:
    Barrier(int count): counter(0), thread_count(count){}

    void wait()
    {
        //fence mechanism
        std::unique_lock<std::mutex> lk(m);
        ++counter;
	if(counter < thread_count)
	        cv.wait(lk);
	else{
		counter = 0;
		cv.notify_all();
	}
    }

 private:
      std::mutex m;
      std::condition_variable cv;
      int counter;
      int thread_count;
};


/*
    Corrects a single fastq file
*/
struct ErrorCorrector {

	ErrorCorrector();
	ErrorCorrector(const Args& args, int nInserterThreads, int nCorrectorThreads);

	void correct(const std::string& filename, const std::string& format, const std::string& outputfilename);
	void setGraphSettings(double alpha, double x);
	void setBatchsize(int n);
	void setAlignmentScores(int matchscore, int subscore, int insertscore, int delscore);
	void setMaxMismatchRatio(double ratio);
	void setMinimumAlignmentOverlap(int overlap);
	void setMinimumAlignmentOverlapRatio(double ratio);
	void setUseQualityScores(bool val);
	void setEstimatedCoverage(int cov);
	void setEstimatedErrorRate(double rate);
	void setM(double m);



private:
    void correct_impl(const std::string& filename, FileFormat format, const std::string& outputfilename);

    Args args;
	MinhashOptions minhashparams;

	std::vector<int> deviceIds;

	bool useMultithreading;

	int nInserterThreads;
	int nCorrectorThreads;

	std::uint32_t batchsize = 20;

	std::vector<char> readIsProcessedVector;
	std::unique_ptr<std::mutex[]> locksForProcessedFlags;
	size_t nLocksForProcessedFlags = 0;
};

}

#endif
