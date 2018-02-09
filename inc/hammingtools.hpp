#ifndef HAMMINGTOOLS_HPP
#define HAMMINGTOOLS_HPP

#include "../inc/alignment.hpp"
#include "../inc/read.hpp"

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
	int* d_lengths = nullptr; //first n_subjects entries are lengths of subjects

	AlignResultCompact* h_results = nullptr;
	char* h_subjectsdata = nullptr;
	char* h_queriesdata = nullptr;
	int* h_queriesPerSubject = nullptr;
	int* h_lengths = nullptr; //first n_subjects entries are lengths of subjects

#ifdef __NVCC__
	cudaStream_t stream = nullptr;
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
//work memory
	char* d_consensus = nullptr;
	double* d_support = nullptr;
	int* d_coverage = nullptr;
	double* d_origWeights = nullptr;
	int* d_origCoverage = nullptr;
	int* d_As = nullptr;
	int* d_Cs = nullptr;
	int* d_Gs = nullptr;
	int* d_Ts = nullptr;
	double* d_Aweights = nullptr;
	double* d_Cweights = nullptr;
	double* d_Gweights = nullptr;
	double* d_Tweights = nullptr;

	int* h_As = nullptr;
	int* h_Cs = nullptr;
	int* h_Gs = nullptr;
	int* h_Ts = nullptr;
	double* h_Aweights = nullptr;
	double* h_Cweights = nullptr;
	double* h_Gweights = nullptr;
	double* h_Tweights = nullptr;

//transfer memory
	int* d_lengths = nullptr;
	AlignResultCompact* d_alignments = nullptr;
	int* d_frequencies_prefix_sum = nullptr;

	int* h_lengths = nullptr;
	AlignResultCompact* h_alignments = nullptr;
	int* h_frequencies_prefix_sum = nullptr;

	char* h_consensus = nullptr;
	double* h_support = nullptr;
	int* h_coverage = nullptr;
	double* h_origWeights = nullptr;
	int* h_origCoverage = nullptr;

	char* h_pileup = nullptr;
	char* d_pileup = nullptr;
	char* d_pileup_transposed = nullptr;

	char* h_qual_pileup = nullptr;
	char* d_qual_pileup = nullptr;
	char* d_qual_pileup_transposed = nullptr;

	CorrectionBuffers* d_this = nullptr;
	

	int deviceId = -1;
#ifdef __NVCC__
	cudaStream_t stream = nullptr;
	cublasHandle_t handle;
#endif

	int max_n_columns = 0;
	int n_columns = 0;
	int max_n_sequences = 0;
	int n_sequences = 0;
	int max_n_qualityscores = 0;
	int n_qualityscores = 0;

	int max_seq_length = -1;

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

	CorrectionBuffers(int id, int maxseqlength);
	void resize(int cols, int nsequences, int nqualityscores);
	void resize_host_cols(int cols);
};

void print_SHDdata(const SHDdata& data);

void cuda_cleanup_SHDdata(SHDdata& data);
void cuda_cleanup_CorrectionBuffers(CorrectionBuffers& buffers);

void init_once();

//we assume that each sequence has the same length and same number of bytes which is specified in SHDdata buffer if useGPU = true
std::vector<std::vector<AlignResultCompact>> getMultipleAlignments(SHDdata& buffer, const std::vector<const Sequence*>& subjects,
			   const std::vector<std::vector<const Sequence*>>& queries,
			   std::vector<bool> activeBatches, bool useGpu);

std::tuple<int,std::chrono::duration<double>,std::chrono::duration<double>>
performCorrection(CorrectionBuffers& buffers, std::string& subject,
				int nQueries, 
				std::vector<std::string>& queries,
				const std::vector<AlignResultCompact>& alignments,
				const std::string& subjectqualityScores, 
				const std::vector<const std::string*>& queryqualityScores,
				const std::vector<int>& frequenciesPrefixSum,
				double maxErrorRate,
				bool useQScores,
				std::vector<bool>& correctedQueries,
				bool correctQueries_,
				int estimatedCoverage,
				double errorrate,
				double m,
				int kmerlength,
				bool useGpu);







} //end namespace hammingtools



#endif
