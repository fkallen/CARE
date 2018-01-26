#ifndef HAMMINGTOOLS_HPP
#define HAMMINGTOOLS_HPP

#include "../inc/alignment.hpp"
#include "../inc/read.hpp"

#include <vector>

namespace hammingtools{

//assumes subject and queries to have same length and same number of bytes

struct SHDdata{

	AlignResultCompact* d_results;
	char* d_subjectsdata;
	char* d_queriesdata;
	int* d_queriesPerSubject;

	AlignResultCompact* h_results;
	char* h_subjectsdata;
	char* h_queriesdata;
	int* h_queriesPerSubject;

#ifdef __CUDACC__
	cudaStream_t stream = nullptr;
#endif
	int deviceId;
	size_t sequencepitch;
	int sequencelength;
	int sequencebytes;
	int n_subjects;
	int n_queries;
	int max_n_subjects;
	int max_n_queries;

	SHDdata(int deviceId_, int seqlength, int seqbytes);

	void resize(int n_sub, int n_quer);
};

struct SHDAligner{

	SHDAligner();

	//we assume that each sequence has the same length and same number of bytes which is specified in SHDdata buffer if useGPU = true
	std::vector<std::vector<AlignResult>> getMultipleAlignments(SHDdata& buffer, const std::vector<const Sequence*>& subjects,
				   const std::vector<std::vector<const Sequence*>>& queries,
				   std::vector<bool> activeBatches, bool useGpu) const;
};

void cuda_cleanup_SHDdata(SHDdata& data);





} //end namespace hammingtools



#endif
