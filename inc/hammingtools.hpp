#ifndef HAMMINGTOOLS_HPP
#define HAMMINGTOOLS_HPP

#include "../inc/alignment.hpp"
#include "../inc/read.hpp"
#include "../inc/batchelem.hpp"

#include <vector>
#include <chrono>
#include <tuple>
#include <climits>

#ifdef __NVCC__
#include <cublas_v2.h>
#endif

namespace hammingtools{

struct SHDdata{

	AlignResultCompact* d_results = nullptr;
	char* d_subjectsdata = nullptr;
	char* d_queriesdata = nullptr;
	int* d_queriesPerSubject = nullptr;
	int* d_subjectlengths = nullptr;
	int* d_querylengths = nullptr;

	AlignResultCompact* h_results = nullptr;
	char* h_subjectsdata = nullptr;
	char* h_queriesdata = nullptr;
	int* h_queriesPerSubject = nullptr;
	int* h_subjectlengths = nullptr;
	int* h_querylengths = nullptr;

		int* h_lengths = nullptr;
		int* d_lengths = nullptr;

#ifdef __NVCC__
	cudaStream_t streams[8];
#endif

	SHDdata* d_this;

	int deviceId = -1;
	size_t sequencepitch = 0;
	int max_sequence_length = 0;
	int max_sequence_bytes = 0;
	int n_subjects = 0;
	int n_queries = 0;
	int max_n_subjects = 0;
	int max_n_queries = 0;

	int shd_max_blocks = 1;
	int max_batch_size = 1;

	std::chrono::duration<double> resizetime{0};
	std::chrono::duration<double> preprocessingtime{0};
	std::chrono::duration<double> h2dtime{0};
	std::chrono::duration<double> alignmenttime{0};
	std::chrono::duration<double> d2htime{0};
	std::chrono::duration<double> postprocessingtime{0};

	SHDdata(int deviceId_, int cpuThreadsOnDevice, int maxseqlength);

	void resize(int n_sub, int n_quer);
};

struct CorrectionBuffers{
	std::unique_ptr<int[]> h_As;
	std::unique_ptr<int[]> h_Cs;
	std::unique_ptr<int[]> h_Gs;
	std::unique_ptr<int[]> h_Ts;
	std::unique_ptr<double[]> h_Aweights;
	std::unique_ptr<double[]> h_Cweights;
	std::unique_ptr<double[]> h_Gweights;
	std::unique_ptr<double[]> h_Tweights;
	std::unique_ptr<char[]> h_consensus;
	std::unique_ptr<double[]> h_support;
	std::unique_ptr<int[]> h_coverage;
	std::unique_ptr<double[]> h_origWeights;
	std::unique_ptr<int[]> h_origCoverage;

	int max_n_columns = 0;
	int n_columns = 0;

	double avg_support = 0;
	double min_support = 0;
	int max_coverage = 0;
	int min_coverage = 0;

	std::chrono::duration<double> resizetime{0};
	std::chrono::duration<double> preprocessingtime{0};
	std::chrono::duration<double> h2dtime{0};
	std::chrono::duration<double> correctiontime{0};
	std::chrono::duration<double> d2htime{0};
	std::chrono::duration<double> postprocessingtime{0};

	void resize(int cols);
	void reset();
};

void print_SHDdata(const SHDdata& data);

void cuda_cleanup_SHDdata(SHDdata& data);

void init_once();

void getMultipleAlignments(SHDdata& mybuffers, std::vector<BatchElem>& batch, bool useGpu);

std::tuple<int,std::chrono::duration<double>,std::chrono::duration<double>>
performCorrection(CorrectionBuffers& buffers, BatchElem& batchElem,
            double maxErrorRate,
            bool useQScores,
            bool correctQueries_,
            int estimatedCoverage,
            double errorrate,
            double m,
            int kmerlength,
            bool useGpu);



} //end namespace hammingtools



#endif
