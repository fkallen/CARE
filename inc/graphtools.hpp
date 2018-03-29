#ifndef GRAPHTOOLS_HPP
#define GRAPHTOOLS_HPP

#include "../inc/sga.hpp"
#include "../inc/alignment.hpp"
#include "../inc/read.hpp"
#include "../inc/batchelem.hpp"


#include <vector>
#include <chrono>
#include <tuple>

namespace care{

namespace graphtools{



	struct AlignerDataArrays{
		AlignOp* d_ops = nullptr;
		AlignOp* h_ops = nullptr;

		AlignResultCompact* d_results = nullptr;
		char* d_subjectsdata = nullptr;
		char* d_queriesdata = nullptr;
		int* d_subjectlengths = nullptr;
		int* d_querylengths = nullptr;

		AlignResultCompact* h_results = nullptr;
		char* h_subjectsdata = nullptr;
		char* h_queriesdata = nullptr;
		int* h_subjectlengths = nullptr;
		int* h_querylengths = nullptr;

	#ifdef __NVCC__
		cudaStream_t streams[8];
		cudaStream_t stream;
	#endif
		int deviceId;

		int ALIGNMENTSCORE_MATCH = 1;
		int ALIGNMENTSCORE_SUB = -1;
		int ALIGNMENTSCORE_INS = -1;
		int ALIGNMENTSCORE_DEL = -1;

		int max_sequence_length = 0;
		int max_sequence_bytes = 0;
		int max_ops_per_alignment = 0;
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

	void getMultipleAlignments(AlignerDataArrays& mybuffers, std::vector<BatchElem>& batch, bool useGpu);

} //namespace end
}

#endif
