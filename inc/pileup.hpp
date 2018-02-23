#ifndef PILEUP_HPP
#define PILEUP_HPP

#include "../inc/hammingtools.hpp"
#include "../inc/alignment.hpp"
#include "../inc/batchelem.hpp"

#include <vector>
#include <string>

namespace hammingtools{

	namespace correction{

		void init_once();

		std::tuple<int,std::chrono::duration<double>,std::chrono::duration<double>>
		cpu_pileup_all_in_one(const CorrectionBuffers* buffers, std::string& subject,
				int nQueries,
				std::vector<std::string>& queries,
				const std::vector<AlignResultCompact>& alignments,
				const std::string& subjectqualityScores,
				const std::vector<const std::string*>& queryqualityScores,
				const std::vector<int>& frequenciesPrefixSum,
				const int startindex, const int endindex,
				const int columnsToCheck, const int subjectColumnsBegin_incl, const int subjectColumnsEnd_excl,
				double maxErrorRate,
				bool useQScores,
				std::vector<bool>& correctedQueries,
				bool correctQueries_,
				int estimatedCoverage,
				double errorrate,
				double m,
				int k);

		void cpu_pileup_create(const CorrectionBuffers* buffers, std::string& subject,
						std::vector<std::string>& queries,
						const std::string& subjectqualityScores,
						const std::vector<const std::string*>& queryqualityScores,
						const std::vector<AlignResultCompact>& alignments,
						const std::vector<int>& frequenciesPrefixSum,
						const int subjectColumnsBegin_incl);

		void cpu_pileup_vote(CorrectionBuffers* buffers, const std::vector<AlignResultCompact>& alignments,
						const std::vector<int>& frequenciesPrefixSum,
						const double maxErrorRate,
						const bool useQScores,
						const int subjectColumnsBegin_incl, const int subjectColumnsEnd_excl);

		int cpu_pileup_correct(CorrectionBuffers* buffers, const std::vector<AlignResultCompact>& alignments, const std::vector<int>& frequenciesPrefixSum,
						const int columnsToCheck,
						const int subjectColumnsBegin_incl, const int subjectColumnsEnd_excl,
						const int startindex, const int endindex,
						const double errorrate, const int estimatedCoverage, const double m,
						const bool correctQueries, int k, std::vector<bool>& correctedQueries);

#ifdef __NVCC__



		void gpu_pileup_transpose(const CorrectionBuffers* buffers);

		void gpu_qual_pileup_transpose(const CorrectionBuffers* buffers);

		void call_cuda_pileup_vote_transposed_kernel(CorrectionBuffers* buffers, const int nSequences, const int nQualityScores, const int columnsToCheck,
						const int subjectColumnsBegin_incl, const int subjectColumnsEnd_excl,
						const double maxErrorRate,
						const bool useQScores);

#endif









std::tuple<int,std::chrono::duration<double>,std::chrono::duration<double>>
cpu_pileup_all_in_one(const CorrectionBuffers* buffers, BatchElem& batchElem,
                const int startindex, const int endindex,
                const int columnsToCheck, const int subjectColumnsBegin_incl, const int subjectColumnsEnd_excl,
                double maxErrorRate,
                bool useQScores,
                const bool correctQueries,
                int estimatedCoverage,
                double errorrate,
                double m,
                int k);


	}

}

#endif
