#ifndef CARE_OPTIONS_HPP
#define CARE_OPTIONS_HPP

#include <config.hpp>

#include <readlibraryio.hpp>

#include <string>
#include <vector>

namespace care{

	//Options which can be parsed from command-line arguments

    struct MinhashOptions {
        int maps = 32;
        int k = 20;
        int numResultsPerMapQueryThreshold = 100;
        bool operator==(const MinhashOptions& other) const{
            return maps == other.maps && k == other.k 
                    && numResultsPerMapQueryThreshold == other.numResultsPerMapQueryThreshold;
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
        int min_overlap = 20;
        float maxErrorRate = 0.2f;
        float min_overlap_ratio = 0.20f;
    };

    struct CorrectionOptions{
        bool correctCandidates = true;
        bool useQualityScores = true;
        float estimatedCoverage = 1.0f;
        float estimatedErrorrate = 0.06f; //this is not the error rate of the dataset
        float m_coverage = 0.6f;
        int kmerlength = 20;
		int batchsize = 1000;
        int new_columns_to_correct = 15;
    };

	struct RuntimeOptions{
		int threads = 1;
		int nInserterThreads = 1;
		int nCorrectorThreads = 1;
        bool showProgress = true;
        bool canUseGpu = false;
        std::vector<int> deviceIds;
	};

    struct MemoryOptions{
        std::size_t memoryForHashtables = 0;
        std::size_t memoryTotalLimit = 0;
    };

	struct FileOptions{
		FileFormat format = FileFormat::NONE;
		std::string inputfile;
		std::string outputdirectory;
        std::string outputfilename;
		std::string outputfile = "";
		std::uint64_t nReads = 0;
        int minimum_sequence_length = -1;
        int maximum_sequence_length = 0;
        std::string save_binary_reads_to = "";
        std::string load_binary_reads_from = "";
        std::string save_hashtables_to = "";
        std::string load_hashtables_from = "";
        std::string tempdirectory;
	};

    struct AllOptions{
        MinhashOptions minhashOptions;
        AlignmentOptions alignmentOptions;
        GoodAlignmentProperties goodAlignmentProperties;
        CorrectionOptions correctionOptions;
        RuntimeOptions runtimeOptions;
        FileOptions fileOptions;
    };
}



#endif
