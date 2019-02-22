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
        float min_hits_per_candidate = 0.0f;
        float padding = 0;
        bool operator==(const MinhashOptions& other) const{
            return maps == other.maps && k == other.k && min_hits_per_candidate == other.min_hits_per_candidate;
        };
        bool operator!=(const MinhashOptions& other) const{
            return !(*this == other);
        };
    };

    struct AlignmentOptions{
        int alignmentscore_match = 1;
        int alignmentscore_sub = -1;
        int alignmentscore_ins = -100;
        int alignmentscore_del = -100;
    };

    struct GoodAlignmentProperties{
        int min_overlap = 35;
        float maxErrorRate = 0.2f;
        float min_overlap_ratio = 0.35f;
    };

    struct CorrectionOptions{
        CorrectionMode correctionMode = CorrectionMode::Hamming;
        bool correctCandidates = false;
        bool useQualityScores = true;
        float estimatedCoverage = 1.0f;
        float estimatedErrorrate = 0.01f;
        float m_coverage = 0.6f;
        float graphalpha = 1.0f;
        float graphx = 1.5f;
        int kmerlength = 16;
		int batchsize = 5;
        int new_columns_to_correct = 0;
        bool extractFeatures = false;
        bool classicMode = false;
        int hits_per_candidate = 1;
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
        int maximum_sequence_length;
        std::string save_binary_reads_to;
        std::string load_binary_reads_from;
        std::string save_hashtables_to;
        std::string load_hashtables_from;
        std::string forestfilename;
	};
}



#endif
