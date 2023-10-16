#ifndef CARE_OPTIONS_HPP
#define CARE_OPTIONS_HPP

#include <config.hpp>
#include <readlibraryio.hpp>

#include "cxxopts/cxxopts.hpp"

#include <string>
#include <vector>
#include <ostream>

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

    enum class GpuDataLayout{
        FirstFit,
        EvenShare
    };

    std::string to_string(SequencePairType s);
    std::string to_string(CorrectionType t);
    std::string to_string(GpuDataLayout t);

    struct GpuCorrectorThreadConfig{
        int numCorrectors = 0;
        int numHashers = 0;

        bool isAutomatic() const noexcept{
            return numCorrectors == 0 && numHashers == 0;
        }
    };

    //Options which can be parsed from command-line arguments

    struct ProgramOptions{
        bool directPeerAccess = true;
        int min_overlap = 30;
        float maxErrorRate = 0.2f;
        float min_overlap_ratio = 0.30f;
        bool excludeAmbiguousReads = false;
        bool correctCandidates = false;
        bool useQualityScores = false;
        bool autodetectKmerlength = false;
        bool mustUseAllHashfunctions = false;
        bool singlehash = false;
        bool outputCorrectionQualityLabels = false;
        bool gzoutput = false;
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
        float pairedFilterThreshold = 0.06f;
        bool showProgress = false;
        bool canUseGpu = false;
        bool replicateGpuReadData = false;
        bool replicateGpuHashtables = false;
        GpuDataLayout gpuReadDataLayout = GpuDataLayout::EvenShare;
        GpuDataLayout gpuHashtableLayout = GpuDataLayout::EvenShare;

        GpuCorrectorThreadConfig gpuCorrectorThreadConfig;

        int warpcore = 0;
        int threads = 1;
        std::size_t fixedNumberOfReads = 0;
        std::vector<int> deviceIds;
        int qualityScoreBits = 8;
        float hashtableLoadfactor = 0.8f;
        std::size_t memoryForHashtables = 0;
        std::size_t memoryTotalLimit = 0;
        std::size_t gpuMemoryLimitReads = std::numeric_limits<std::size_t>::max();
        SequencePairType pairType = SequencePairType::SingleEnd;
        uint32_t maxForestTreesAnchor = std::numeric_limits<uint32_t>::max();
        uint32_t maxForestTreesCands = std::numeric_limits<uint32_t>::max();
        std::uint64_t nReads = 0;
        std::string outputdirectory = "";
        std::string save_binary_reads_to = "";
        std::string load_binary_reads_from = "";
        std::string save_hashtables_to = "";
        std::string load_hashtables_from = "";
        std::string tempdirectory = "";
        std::string mlForestfileAnchor = "";
        std::string mlForestfileCands = "";
        std::string mlForestfilePrintAnchor = "anchorfeatures.txt";
        std::string mlForestfilePrintCands = "candidatefeatures.txt";
        std::vector<std::string> inputfiles;
        std::vector<std::string> outputfilenames;

        ProgramOptions() = default;
        ProgramOptions(const ProgramOptions&) = default;
        ProgramOptions(ProgramOptions&&) = default;

        ProgramOptions(const cxxopts::ParseResult& pr);

        bool isValid() const noexcept;

        void printMandatoryOptions(std::ostream& stream) const;
        void printMandatoryOptionsCorrectCpu(std::ostream& stream) const;
        void printMandatoryOptionsCorrectGpu(std::ostream& stream) const;

        void printAdditionalOptions(std::ostream& stream) const;
        void printAdditionalOptionsCorrectCpu(std::ostream& stream) const;
        void printAdditionalOptionsCorrectGpu(std::ostream& stream) const;
    };


    template<class ReadStorage>
    std::size_t getNumReadsToProcess(const ReadStorage* readStorage, const ProgramOptions& options){
        if(options.fixedNumberOfReads == 0){ 
            return readStorage->getNumberOfReads();
        }else{
            return options.fixedNumberOfReads;
        }
    }


    void addMandatoryOptions(cxxopts::Options& commandLineOptions);
    void addMandatoryOptionsCorrectCpu(cxxopts::Options& commandLineOptions);
    void addMandatoryOptionsCorrectGpu(cxxopts::Options& commandLineOptions);

    void addAdditionalOptions(cxxopts::Options& commandLineOptions);
    void addAdditionalOptionsCorrectCpu(cxxopts::Options& commandLineOptions);
    void addAdditionalOptionsCorrectGpu(cxxopts::Options& commandLineOptions);


}

#endif
