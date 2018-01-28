#ifndef PILEUP_HPP
#define PILEUP_HPP

#include "../inc/alignment.hpp"

#include <vector>
#include <string>

namespace hammingtools{

	namespace correction{

		void init_once();

		int correct_cpu(std::string& subject,
				int nQueries, 
				std::vector<std::string>& queries,
				const std::vector<AlignResult>& alignments,
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
				int k);

	}

}

#endif
