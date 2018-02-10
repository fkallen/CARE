#ifndef GRAPHTOOLS_HPP
#define GRAPHTOOLS_HPP

#include "../inc/alignment.hpp"
#include "../inc/read.hpp"

#include <vector>
#include <chrono>
#include <tuple>

namespace graphtools{

	struct AlignerDataArrays{

		AlignOp* d_ops = nullptr;
		AlignOp* h_ops = nullptr;

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

		AlignerDataArrays* d_this;
	#ifdef __NVCC__
		cudaStream_t stream = nullptr;
	#endif
		int deviceId;

		int ALIGNMENTSCORE_MATCH = 1;
		int ALIGNMENTSCORE_SUB = -1;
		int ALIGNMENTSCORE_INS = -1;
		int ALIGNMENTSCORE_DEL = -1;

		int max_ops_per_alignment = 0;
		int max_sequence_length = 0;
		int max_sequence_bytes = 0;
		int n_subjects = 0;
		int n_queries = 0;
		int max_n_subjects = 0;
		int max_n_queries = 0;

		size_t sequencepitch = 0;
		int max_blocks = 1;

		std::chrono::duration<double> resizetime{0};
		std::chrono::duration<double> preprocessingtime{0};
		std::chrono::duration<double> h2dtime{0};
		std::chrono::duration<double> alignmenttime{0};
		std::chrono::duration<double> d2htime{0};
		std::chrono::duration<double> postprocessingtime{0};

		void resize(int n_sub, int n_quer);

		AlignerDataArrays(int deviceId, int maxseqlength, int scorematch, int scoresub, int scoreins, int scoredel);

	};

	void cuda_cleanup_AlignerDataArrays(AlignerDataArrays& data);

	void init_once();

	std::vector<std::vector<AlignResult>> 
	getMultipleAlignments(AlignerDataArrays& buffer, const std::vector<const Sequence*>& subjects,
			   const std::vector<std::vector<const Sequence*>>& queries,
			   std::vector<bool> activeBatches, bool useGpu);

	std::tuple<std::chrono::duration<double>,std::chrono::duration<double>> performCorrection(std::string& subject,
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
