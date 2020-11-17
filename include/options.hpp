#ifndef CARE_OPTIONS_HPP
#define CARE_OPTIONS_HPP

#include <config.hpp>

#include <readlibraryio.hpp>

#include <string>
#include <vector>

namespace care{

    enum class SequencePairType{
        Invalid,
        SingleEnd,
        PairedEnd,
    };

    __inline__
    std::string to_string(SequencePairType s){
        switch(s){
            case SequencePairType::Invalid: return "Invalid";
            case SequencePairType::SingleEnd: return "SingleEnd";
            case SequencePairType::PairedEnd: return "PairedEnd";
            default: return "Error";
        }
    }

	//Options which can be parsed from command-line arguments

    struct GoodAlignmentProperties{
        int min_overlap = 30;
        float maxErrorRate = 0.2f;
        float min_overlap_ratio = 0.30f;
    };

    struct CorrectionOptions{
        bool excludeAmbiguousReads = false;
        bool correctCandidates = false;
        bool useQualityScores = false;
        bool autodetectKmerlength = false;
        bool mustUseAllHashfunctions = false;
        float estimatedCoverage = 1.0f;
        float estimatedErrorrate = 0.06f; //this is not the error rate of the dataset
        float m_coverage = 0.6f;
		int batchsize = 1000;
        int new_columns_to_correct = 15;
        int kmerlength = 20;
        int numHashFunctions = 48;
    };

    struct ExtensionOptions{
        int insertSize;
        int insertSizeStddev;
    };

	struct RuntimeOptions{
		int threads = 1;
        bool showProgress = false;
        bool canUseGpu = false;
        std::vector<int> deviceIds;
	};

    struct MemoryOptions{
        std::size_t memoryForHashtables = 0;
        std::size_t memoryTotalLimit = 0;
    };

	struct FileOptions{
        bool mergedoutput = false;
        SequencePairType pairType = SequencePairType::Invalid;
		std::string outputdirectory = "";
		std::uint64_t nReads = 0;
        int minimum_sequence_length = 0;
        int maximum_sequence_length = 0;
        std::string save_binary_reads_to = "";
        std::string load_binary_reads_from = "";
        std::string save_hashtables_to = "";
        std::string load_hashtables_from = "";
        std::string tempdirectory = "";
        std::string extendedReadsOutputfilename = "UNSET_";
        std::vector<std::string> inputfiles;
        std::vector<std::string> outputfilenames;
	};

    struct AllOptions{
        GoodAlignmentProperties goodAlignmentProperties;
        CorrectionOptions correctionOptions;
        RuntimeOptions runtimeOptions;
        FileOptions fileOptions;
    };
}



#endif
