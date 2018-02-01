#ifndef GRAPHTOOLS_HPP
#define GRAPHTOOLS_HPP

#include "../inc/alignment.hpp"
#include "../inc/read.hpp"

#include <vector>
#include <chrono>
#include <tuple>

namespace graphtools{

	struct AlignerDataArrays{

		AlignResultCompact* d_results = nullptr;
		AlignOp* d_ops = nullptr;
		char* d_subjectsdata = nullptr;
		char* d_queriesdata = nullptr;
		int* d_rBytesPrefixSum = nullptr;
		int* d_rLengths = nullptr;
		int* d_rIsEncoded = nullptr;
		int* d_cBytesPrefixSum = nullptr;
		int* d_cLengths = nullptr;
		int* d_cIsEncoded = nullptr;
		int* d_r2PerR1 = nullptr;

		AlignResultCompact* h_results = nullptr;
		AlignOp* h_ops = nullptr;
		char* h_subjectsdata = nullptr;
		char* h_queriesdata = nullptr;
		int* h_rBytesPrefixSum = nullptr;
		int* h_rLengths = nullptr;
		int* h_rIsEncoded = nullptr;
		int* h_cBytesPrefixSum = nullptr;
		int* h_cLengths = nullptr;
		int* h_cIsEncoded = nullptr;
		int* h_r2PerR1 = nullptr;

		size_t results_size = 0;
		size_t ops_size = 0;
		size_t r_size = 0;
		size_t c_size = 0;


	#ifdef __NVCC__
		cudaStream_t stream = nullptr;
	#endif
		int deviceId;

		int ALIGNMENTSCORE_MATCH = 1;
		int ALIGNMENTSCORE_SUB = -1;
		int ALIGNMENTSCORE_INS = -1;
		int ALIGNMENTSCORE_DEL = -1;

		int max_ops_per_alignment = 0;
		int n_subjects = 0;
		int n_candidates = 0;
		int maximumQueryLength = 0;
		int maximumCandidateLength = 0;

		AlignerDataArrays(int deviceId, int scorematch, int scoresub, int scoreins, int scoredel);

		/*
			res = number of alignments
			ops = maximum total number of alignment operations for res alignments
			r = number of bytes to store all subjects
			c = number of bytes to store all candidates
		*/
		void resize(size_t res, size_t ops, size_t r, size_t c);
	};

	void cuda_cleanup_AlignerDataArrays(AlignerDataArrays& data);

	void init_once();

	std::vector<std::vector<AlignResult>> 
	getMultipleAlignments(AlignerDataArrays& buffer, const std::vector<const Sequence*>& subjects,
			   const std::vector<std::vector<const Sequence*>>& queries,
			   std::vector<bool> activeBatches, bool useGpu);

	void performCorrection(std::string& subject,
				std::vector<AlignResult>& alignments,
				const std::string& subjectqualityScores, 
				const std::vector<const std::string*>& queryqualityScores,
				const std::vector<int>& frequenciesPrefixSum,
				bool useQScores,
				double MAX_MISMATCH_RATIO,
				double graphalpha,
				double graphx);

} //namespace end

#endif
