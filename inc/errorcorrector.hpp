#ifndef ERRORCORRECTOR_HPP
#define ERRORCORRECTOR_HPP

#include "minhasher.hpp"
#include "read.hpp"
#include "readstorage.hpp"
#include "threadsafe_buffer.hpp"

#include <cstdint>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <tuple>

//#define USE_QUALITY_SCORES


enum class CorrectionMode {Hamming, Graph};

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

	ErrorCorrector(const MinhashParameters& minhashparameters, int nInserterThreads, int nCorrectorThreads);

	void correct(const std::string& filename);

	void setOutputPath(const std::string& path);

	void setGraphSettings(double alpha, double x);

	void setOutputFilename(const std::string& filename);

	void setBatchsize(int n);

	void setAlignmentScores(int matchscore, int subscore, int insertscore, int delscore);

	void setMaxMismatchRatio(double ratio);
	void setMinimumAlignmentOverlap(int overlap);
	void setMinimumAlignmentOverlapRatio(double ratio);
	void setFileFormat(const std::string& format);
	void setUseQualityScores(bool val);
	void setEstimatedCoverage(int cov);
	void setEstimatedErrorRate(double rate);
	void setM(double m);

	

private:



	enum class Fileformat {FASTA, FASTQ};

	void insertFile(const std::string& filename, bool buildHashmap);

	void errorcorrectFile(const std::string& filename);

	void errorcorrectWork(int threadId, int nThreads, const std::string& fileToCorrect);

	void mergeThreadResults(const std::string& filename) const;

	void mergeUnorderedThreadResults(const std::string& filename) const;

	std::uint64_t getReadPos(const std::string& readheader) const;

	void updateGlobalProgress(std::uint64_t increment, std::uint64_t maxglobalprogress);

	Minhasher minhasher;

	MinhashParameters minhashparams;

	mutable ReadStorage readStorage;

	std::vector<ThreadsafeBuffer<
			    std::pair<Read, std::uint32_t>,
			    30000> > buffers;

	std::vector<int> deviceIds;

	bool useMultithreading;

	int nInserterThreads;
	int nCorrectorThreads;

	std::string outputPath;
	std::string outputFilename = "";

	std::uint32_t batchsize = 20;

	int ALIGNMENTSCORE_MATCH = 1;
	int ALIGNMENTSCORE_SUB = -1;
	int ALIGNMENTSCORE_INS = -100;
	int ALIGNMENTSCORE_DEL = -100;

	double MAX_MISMATCH_RATIO = 0.2;
	int MIN_OVERLAP = 35;
	double MIN_OVERLAP_RATIO = 0.35;

	bool useQualityScores = false;

	Fileformat inputfileformat = Fileformat::FASTQ;

	std::map<std::string, std::uint64_t> readsPerFile;

	std::mutex writelock;

	// settings for error graph
	double graphx = 2.0;
	double graphalpha = 1.0;

	std::mutex progresslock;
	std::uint64_t progress = 0;

	std::vector<char> readIsProcessedVector;
	std::unique_ptr<std::mutex[]> locksForProcessedFlags;
	size_t nLocksForProcessedFlags = 0;

	int estimatedCoverage;
	double errorrate;
	double m_coverage;

	CorrectionMode correctionmode = CorrectionMode::Hamming;
	int maximum_sequence_length = 0;

};

#endif
