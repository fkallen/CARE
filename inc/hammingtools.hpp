#ifndef HAMMINGTOOLS_HPP
#define HAMMINGTOOLS_HPP

#include "../inc/alignment.hpp"
#include "../inc/read.hpp"

#include <vector>

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
	int deviceId = -1;
	size_t sequencepitch = 0;
	int max_sequence_length = 0;
	int max_sequence_bytes = 0;
	int n_subjects = 0;
	int n_queries = 0;
	int max_n_subjects = 0;
	int max_n_queries = 0;

	int shd_max_blocks = 1;

	SHDdata(int deviceId_, int cpuThreadsOnDevice, int maxseqlength);

	void resize(int n_sub, int n_quer);
};

void print_SHDdata(const SHDdata& data);

void cuda_cleanup_SHDdata(SHDdata& data);

void init_once();

//we assume that each sequence has the same length and same number of bytes which is specified in SHDdata buffer if useGPU = true
std::vector<std::vector<AlignResultCompact>> getMultipleAlignments(SHDdata& buffer, const std::vector<const Sequence*>& subjects,
			   const std::vector<std::vector<const Sequence*>>& queries,
			   std::vector<bool> activeBatches, bool useGpu);

int performCorrection(std::string& subject,
				int nQueries, 
				std::vector<std::string>& queries,
				const std::vector<AlignResultCompact>& alignments,
				const std::string& subjectqualityScores, 
				const std::vector<std::string>& queryqualityScores,
				const std::vector<int>& frequenciesPrefixSum,
				double maxErrorRate,
				bool useQScores,
				std::vector<bool>& correctedQueries,
				bool correctQueries_,
				int estimatedCoverage,
				double errorrate,
				double m,
				int kmerlength);







} //end namespace hammingtools



#endif
