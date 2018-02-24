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

		std::chrono::duration<double>
		cpu_add_weights(const CorrectionBuffers* buffers, const BatchElem& batchElem,
						const int startindex, const int endindex,
						const int columnsToCheck, const int subjectColumnsBegin_incl, const int subjectColumnsEnd_excl,
						const double maxErrorRate,
						const bool useQScores);	
			
		void cpu_find_consensus(const CorrectionBuffers* buffers, const BatchElem& batchElem,
						const int columnsToCheck, const int subjectColumnsBegin_incl);
			
		std::tuple<int,std::chrono::duration<double>>
		cpu_correct(const CorrectionBuffers* buffers, BatchElem& batchElem,
						const int startindex, const int endindex,
						const int columnsToCheck, const int subjectColumnsBegin_incl, const int subjectColumnsEnd_excl,
						const double maxErrorRate,
						const bool correctQueries,
						const int estimatedCoverage,
						const double errorrate,
						const double m,
						const int k);	
	
	
	}
	
}

#endif
