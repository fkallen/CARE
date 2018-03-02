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

        struct PileupProperties{
            double avg_support;
            double min_support;
            int max_coverage;
            int min_coverage;
            bool isHQ;
            bool failedAvgSupport;
            bool failedMinSupport;
            bool failedMinCoverage;
        };

        struct PileupColumnProperties{
            int startindex;
            int endindex;
            int columnsToCheck;
            int subjectColumnsBegin_incl;
            int subjectColumnsEnd_excl;
        };

        struct PileupCorrectionSettings{
            bool useQScores;
            bool correctQueries;
            int estimatedCoverage;
            double maxErrorRate;
            double errorrate;
            double m;
            double k;
        };

        struct TaskTimings{
        	std::chrono::duration<double> preprocessingtime{0};
        	std::chrono::duration<double> h2dtime{0};
        	std::chrono::duration<double> executiontime{0};
        	std::chrono::duration<double> d2htime{0};
        	std::chrono::duration<double> postprocessingtime{0};
        };

        struct PileupTimings{
            std::chrono::duration<double> findconsensustime{0};
            std::chrono::duration<double> correctiontime{0};
        };

        struct PileupImage{
            //buffers
            std::unique_ptr<int[]> h_As;
            std::unique_ptr<int[]> h_Cs;
            std::unique_ptr<int[]> h_Gs;
            std::unique_ptr<int[]> h_Ts;
            std::unique_ptr<double[]> h_Aweights;
            std::unique_ptr<double[]> h_Cweights;
            std::unique_ptr<double[]> h_Gweights;
            std::unique_ptr<double[]> h_Tweights;
            std::unique_ptr<char[]> h_consensus;
            std::unique_ptr<double[]> h_support;
            std::unique_ptr<int[]> h_coverage;
            std::unique_ptr<double[]> h_origWeights;
            std::unique_ptr<int[]> h_origCoverage;

            int max_n_columns = 0; //number of elements per buffer
            int n_columns = 0; //number of used elements per buffer

            PileupProperties properties;
            PileupColumnProperties columnProperties;
            PileupCorrectionSettings correctionSettings;
            PileupTimings timings;
            TaskTimings taskTimings;

            PileupImage(bool useQScores, bool correctQueries, int estimatedCoverage,
                        double maxErrorRate, double errorrate, double m, double k);

            PileupImage(const PileupImage& other);
            PileupImage(PileupImage&& other);
            PileupImage& operator=(const PileupImage& other);
            PileupImage& operator=(PileupImage&& other);

            void resize(int cols);
            void clear();

            TaskTimings correct_batch_elem(BatchElem& batchElem);
            void init_from_batch_elem(const BatchElem& batchElem);
            void cpu_add_weights(const BatchElem& batchElem);
            void cpu_find_consensus(const BatchElem& batchElem);
            void cpu_correct(BatchElem& batchElem);
        };


	}

}

#endif
