#ifndef CARE_OPTIONS_HPP
#define CARE_OPTIONS_HPP

#include "sequencefileio.hpp"

#include <string>
#include <vector>

namespace care{
    enum class CorrectionMode {Hamming, Graph};

	//Options which can be parsed from command-line arguments

    struct MinhashOptions {
        int maps = 2;
        int k = 16;
        double min_hits_per_candidate = 0.0;
    };

    struct AlignmentOptions{
        int alignmentscore_match = 1;
        int alignmentscore_sub = -1;
        int alignmentscore_ins = -100;
        int alignmentscore_del = -100;
    };

    struct GoodAlignmentProperties{
        int min_overlap = 35;
        double maxErrorRate = 0.2;
        double min_overlap_ratio = 0.35;
    };

    struct CorrectionOptions{
        CorrectionMode correctionMode = CorrectionMode::Hamming;
        bool correctCandidates = false;
        bool useQualityScores = true;
        double estimatedCoverage = 1.0;
        double estimatedErrorrate = 0.01;
        double m_coverage = 0.6;
        double graphalpha = 1.0;
        double graphx = 1.5;
        int kmerlength = 16;
		int batchsize = 5;
        int new_columns_to_correct = 0;
        bool extractFeatures = false;
        bool classicMode = false;
    };

	struct RuntimeOptions{
		int threads = 1;
        int threadsForGPUs = 0;
		int nInserterThreads = 1;
		int nCorrectorThreads = 1;
        bool showProgress = true;
        bool canUseGpu = false;
        int max_candidates = 0;
        std::vector<int> deviceIds;
	};

	struct FileOptions{
		FileFormat format;
		std::string fileformatstring;
		std::string inputfile;
		std::string outputdirectory;
        std::string outputfilename;
		std::string outputfile;
		std::uint64_t nReads;
        std::string save_binary_reads_to;
        std::string load_binary_reads_from;
	};
}



#endif
