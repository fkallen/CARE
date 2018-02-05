#ifndef PILEUP_HPP
#define PILEUP_HPP

#include "../inc/hammingtools.hpp"
#include "../inc/alignment.hpp"

#include <vector>
#include <string>

namespace hammingtools{

	namespace correction{

		void init_once();

		std::tuple<int,std::chrono::duration<double>,std::chrono::duration<double>>
		correct_cpu(std::string& subject,
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
				int k);

		int cpu_pileup_correct(CorrectionBuffers* buffers, int nQueries, int columnsToCheck, 
						int subjectColumnsBegin_incl, int subjectColumnsEnd_excl,
						int startindex, int endindex,
						double errorrate, int estimatedCoverage, double m, 
						bool correctQueries, int k, std::vector<bool>& correctedQueries);

#ifdef __NVCC__

		void call_cuda_pileup_vote_kernel_async(CorrectionBuffers* buffers, const int nQueries, const int columnsToCheck, 
						const int subjectColumnsBegin_incl, const int subjectColumnsEnd_excl,
						const double maxErrorRate, 
						const bool useQScores);

		void call_cuda_pileup_vote_kernel(CorrectionBuffers* buffers, const int nQueries, const int columnsToCheck, 
								const int subjectColumnsBegin_incl, const int subjectColumnsEnd_excl,
								const double maxErrorRate, 
								const bool useQScores);

#endif

	}

}

#endif
