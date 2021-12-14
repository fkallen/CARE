#ifndef CARE_OPTIONS_HPP
#define CARE_OPTIONS_HPP

#include <config.hpp>
#include <readlibraryio.hpp>

#include "cxxopts/cxxopts.hpp"

#include <string>
#include <vector>

namespace care
{

    enum class SequencePairType
    {
        Invalid,
        SingleEnd,
        PairedEnd,
    };

    enum class CorrectionType : int
    {
        Classic,
        Forest,
        Print
    };

    std::string to_string(SequencePairType s);
    std::string to_string(CorrectionType t);


    //Options which can be parsed from command-line arguments

    struct ProgramOptions{
        int min_overlap = 30;
        float maxErrorRate = 0.2f;
        float min_overlap_ratio = 0.30f;
        bool excludeAmbiguousReads = false;
        bool correctCandidates = false;
        bool useQualityScores = false;
        bool autodetectKmerlength = false;
        bool mustUseAllHashfunctions = false;
        bool singlehash = false;
        float estimatedCoverage = 1.0f;
        float estimatedErrorrate = 0.06f; //this is not the error rate of the dataset
        float m_coverage = 0.6f;
        int batchsize = 1000;
        int new_columns_to_correct = 15;
        int kmerlength = 20;
        int numHashFunctions = 48;
        CorrectionType correctionType = CorrectionType::Classic;
        CorrectionType correctionTypeCands = CorrectionType::Classic;
        float thresholdAnchor = .5f; // threshold for anchor classifier
        float thresholdCands = .5f;  // threshold for cands classifier
        float sampleRateAnchor = 1.f;
        float sampleRateCands = 0.01f;
        float pairedthreshold1 = 0.06f;
        bool allowOutwardExtension{};
        bool sortedOutput = false;
        bool outputRemainingReads = false;
        int insertSize{};
        int insertSizeStddev{};
        int fixedStddev{};
        int fixedStepsize{};
        bool showProgress = false;
        bool canUseGpu = false;
        bool replicateGpuData = false;
        int warpcore = 0;
        int threads = 1;
        std::size_t fixedNumberOfReads = 0;
        std::vector<int> deviceIds;
        int qualityScoreBits = 8;
        float hashtableLoadfactor = 0.8f;
        std::size_t memoryForHashtables = 0;
        std::size_t memoryTotalLimit = 0;
        SequencePairType pairType = SequencePairType::SingleEnd;
        int minimum_sequence_length = 0;
        int maximum_sequence_length = 0;
        std::uint64_t nReads = 0;
        std::string outputdirectory = "";
        std::string save_binary_reads_to = "";
        std::string load_binary_reads_from = "";
        std::string save_hashtables_to = "";
        std::string load_hashtables_from = "";
        std::string tempdirectory = "";
        std::string extendedReadsOutputfilename = "UNSET_";
        std::string mlForestfileAnchor = "";
        std::string mlForestfileCands = "";
        std::vector<std::string> inputfiles;
        std::vector<std::string> outputfilenames;

        ProgramOptions() = default;
        ProgramOptions(const ProgramOptions&) = default;
        ProgramOptions(ProgramOptions&&) = default;

        ProgramOptions(const cxxopts::ParseResult& pr);

        bool isValid() const noexcept;
    };


    template<class ReadStorage>
    std::size_t getNumReadsToProcess(const ReadStorage* readStorage, const ProgramOptions& options){
        if(options.fixedNumberOfReads == 0){ 
            return readStorage->getNumberOfReads();
        }else{
            return options.fixedNumberOfReads;
        }
    }
}

#endif
