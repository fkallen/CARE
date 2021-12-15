#include <options.hpp>
#include <hpc_helpers.cuh>
#include <util.hpp>
#include <config.hpp>
#include <readlibraryio.hpp>
#include <memorymanagement.hpp>
#include <filehelpers.hpp>

#include <iostream>
#include <thread>
#include <string>
#include <stdexcept>

#include <experimental/filesystem>

namespace filesys = std::experimental::filesystem;

namespace care{


    std::string to_string(SequencePairType s){
        switch (s)
        {
        case SequencePairType::Invalid:
            return "Invalid";
        case SequencePairType::SingleEnd:
            return "SingleEnd";
        case SequencePairType::PairedEnd:
            return "PairedEnd";
        default:
            return "Error";
        }
    }

    std::string to_string(CorrectionType t)
    {
        switch (t)
        {
        case CorrectionType::Classic:
            return "Classic";
            break;
        case CorrectionType::Forest:
            return "Forest";
            break;
        case CorrectionType::Print:
            return "Print";
            break;
        default:
            return "Forgot to name correction type";
            break;
        }
    }

    ProgramOptions::ProgramOptions(const cxxopts::ParseResult& pr){
        ProgramOptions& result = *this;

        if(pr.count("minalignmentoverlap")){
            result.min_overlap = pr["minalignmentoverlap"].as<int>();
        }
        if(pr.count("maxmismatchratio")){
            result.maxErrorRate = pr["maxmismatchratio"].as<float>();
        }
        if(pr.count("minalignmentoverlapratio")){
            result.min_overlap_ratio = pr["minalignmentoverlapratio"].as<float>();
        }

        if(pr.count("excludeAmbiguous")){
            result.excludeAmbiguousReads = pr["excludeAmbiguous"].as<bool>();
        }

        if(pr.count("candidateCorrection")){
            result.correctCandidates = pr["candidateCorrection"].as<bool>();
        }

        if(pr.count("correctionTypeCands")){
            const int val = pr["correctionTypeCands"].as<int>();

            switch(val){
                case 1: result.correctionTypeCands = CorrectionType::Forest; break;
                case 2: result.correctionTypeCands = CorrectionType::Print; break;
                default: result.correctionTypeCands = CorrectionType::Classic; break;
            }
        }

        if(pr.count("useQualityScores")){
            result.useQualityScores = pr["useQualityScores"].as<bool>();
        }

        if(pr.count("enforceHashmapCount")){
            result.mustUseAllHashfunctions = pr["enforceHashmapCount"].as<bool>();
        }

        if(pr.count("singlehash")){
            result.singlehash = pr["singlehash"].as<bool>();
        }

        if(pr.count("coverage")){
            result.estimatedCoverage = pr["coverage"].as<float>();
        }

        if(pr.count("errorfactortuning")){
            result.estimatedErrorrate = pr["errorfactortuning"].as<float>();
        }

        if(pr.count("coveragefactortuning")){
            result.m_coverage = pr["coveragefactortuning"].as<float>();
        }

        if(pr.count("kmerlength")){
            result.kmerlength = pr["kmerlength"].as<int>();
            if(result.kmerlength == 0){
                result.autodetectKmerlength = true;
            }else{
                result.autodetectKmerlength = false;
            }
        }else{
            result.autodetectKmerlength = true;
        }

        if(pr.count("hashmaps")){
            result.numHashFunctions = pr["hashmaps"].as<int>();
        }        

        if(pr.count("batchsize")){
            result.batchsize = pr["batchsize"].as<int>();
        }

        if(pr.count("candidateCorrectionNewColumns")){
            result.new_columns_to_correct = pr["candidateCorrectionNewColumns"].as<int>();
        }

        if(pr.count("correctionType")){
            const int val = pr["correctionType"].as<int>();

            switch(val){
                case 1: result.correctionType = CorrectionType::Forest; break;
                case 2: result.correctionType = CorrectionType::Print; break;
                default: result.correctionType = CorrectionType::Classic; break;
            }
        }

        if(pr.count("thresholdAnchor")){
            float t = pr["thresholdAnchor"].as<float>();
            result.thresholdAnchor = t>1.0?t/100:t;
        }

        if(pr.count("thresholdCands")){
            float t = pr["thresholdCands"].as<float>();
            result.thresholdCands = t>1.0?t/100:t;
        }

        if(pr.count("samplingRateAnchor")){
            float t = pr["samplingRateAnchor"].as<float>();
            result.sampleRateAnchor = t>1.0?t/100:t;
        }

        if(pr.count("samplingRateCands")){
            float t = pr["samplingRateCands"].as<float>();
            result.sampleRateCands = t>1.0?t/100:t;
        }

        if(pr.count("pairedthreshold1")){
            result.pairedthreshold1 = pr["pairedthreshold1"].as<float>();
        }

        if(pr.count("insertsize")){
            result.insertSize = pr["insertsize"].as<int>();
        }

        if(pr.count("insertsizedev")){
            result.insertSizeStddev = pr["insertsizedev"].as<int>();
        }

        if(pr.count("fixedStddev")){
            result.fixedStddev = pr["fixedStddev"].as<int>();
        }

        if(pr.count("fixedStepsize")){
            result.fixedStepsize = pr["fixedStepsize"].as<int>();
        }

        if(pr.count("allowOutwardExtension")){
            result.allowOutwardExtension = pr["allowOutwardExtension"].as<bool>();
        }

        if(pr.count("sortedOutput")){
            result.sortedOutput = pr["sortedOutput"].as<bool>();
        }

        if(pr.count("outputRemaining")){
            result.outputRemainingReads = pr["outputRemaining"].as<bool>();
        }

        if(pr.count("threads")){
            result.threads = pr["threads"].as<int>();
        }
        result.threads = std::min(result.threads, (int)std::thread::hardware_concurrency());
      
        if(pr.count("showProgress")){
            result.showProgress = pr["showProgress"].as<bool>();
        }

        if(pr.count("gpu")){
            result.deviceIds = pr["gpu"].as<std::vector<int>>();
        }

        result.canUseGpu = result.deviceIds.size() > 0;

        if(pr.count("warpcore")){
            result.warpcore = pr["warpcore"].as<int>();
        }

        if(pr.count("replicateGpuData")){
            result.replicateGpuData = pr["replicateGpuData"].as<bool>();
        }

        if(pr.count("fixedNumberOfReads")){
            result.fixedNumberOfReads = pr["fixedNumberOfReads"].as<std::size_t>();
        }

        auto parseMemoryString = [](const auto& string) -> std::size_t{
            if(string.length() > 0){
                std::size_t factor = 1;
                bool foundSuffix = false;
                switch(string.back()){
                    case 'K':{
                        factor = std::size_t(1) << 10; 
                        foundSuffix = true;
                    }break;
                    case 'M':{
                        factor = std::size_t(1) << 20;
                        foundSuffix = true;
                    }break;
                    case 'G':{
                        factor = std::size_t(1) << 30;
                        foundSuffix = true;
                    }break;
                }
                if(foundSuffix){
                    const auto numberString = string.substr(0, string.size()-1);
                    return factor * std::stoull(numberString);
                }else{
                    return std::stoull(string);
                }
            }else{
                return 0;
            }
        };

        if(pr.count("memTotal")){
            const auto memoryTotalLimitString = pr["memTotal"].as<std::string>();
            const std::size_t parsedMemory = parseMemoryString(memoryTotalLimitString);
            const std::size_t availableMemory = getAvailableMemoryInKB() * 1024;

            // user-provided memory limit could be greater than currently available memory.
            result.memoryTotalLimit = std::min(parsedMemory, availableMemory);
        }else{
            std::size_t availableMemoryInBytes = getAvailableMemoryInKB() * 1024;
            if(availableMemoryInBytes > 2*(std::size_t(1) << 30)){
                availableMemoryInBytes = availableMemoryInBytes - 2*(std::size_t(1) << 30);
            }

            result.memoryTotalLimit = availableMemoryInBytes;
        }

        if(pr.count("memHashtables")){
            const auto memoryForHashtablesString = pr["memHashtables"].as<std::string>();
            result.memoryForHashtables = parseMemoryString(memoryForHashtablesString);
        }else{
            std::size_t availableMemoryInBytes = result.memoryTotalLimit;
            if(availableMemoryInBytes > 1*(std::size_t(1) << 30)){
                availableMemoryInBytes = availableMemoryInBytes - 1*(std::size_t(1) << 30);
            }

            result.memoryForHashtables = availableMemoryInBytes;
        }

        result.memoryForHashtables = std::min(result.memoryForHashtables, result.memoryTotalLimit);

        if(pr.count("hashloadfactor")){
            result.hashtableLoadfactor = pr["hashloadfactor"].as<float>();
        }

        if(pr.count("qualityScoreBits")){
            result.qualityScoreBits = pr["qualityScoreBits"].as<int>();
        }

        if(pr.count("outdir")){
		    result.outputdirectory = pr["outdir"].as<std::string>();
        }

        if(pr.count("pairmode")){
            const std::string arg = pr["pairmode"].as<std::string>();

            if(arg == "se" || arg == "SE"){
                result.pairType = SequencePairType::SingleEnd;
            }else if(arg == "pe" || arg == "PE"){
                result.pairType = SequencePairType::PairedEnd;
            }else{
                result.pairType = SequencePairType::Invalid;
            }
        }  

        if(pr.count("eo")){
            result.extendedReadsOutputfilename = pr["eo"].as<std::string>();
        }

        if(pr.count("nReads")){
		    result.nReads = pr["nReads"].as<std::uint64_t>();
        }

        if(pr.count("min_length")){
            result.minimum_sequence_length = pr["min_length"].as<int>();
        }

        if(pr.count("max_length")){
            result.maximum_sequence_length = pr["max_length"].as<int>();
        }

        if(pr.count("save-preprocessedreads-to")){
            result.save_binary_reads_to = pr["save-preprocessedreads-to"].as<std::string>();
        }

        if(pr.count("load-preprocessedreads-from")){
            result.load_binary_reads_from = pr["load-preprocessedreads-from"].as<std::string>();
        }

        if(pr.count("save-hashtables-to")){
            result.save_hashtables_to = pr["save-hashtables-to"].as<std::string>();
        }

        if(pr.count("load-hashtables-from")){
            result.load_hashtables_from = pr["load-hashtables-from"].as<std::string>();
        }

        if(pr.count("tempdir")){
            result.tempdirectory = pr["tempdir"].as<std::string>();
        }else{
            result.tempdirectory = result.outputdirectory;
        }

        if(pr.count("ml-forestfile")){
            result.mlForestfileAnchor = pr["ml-forestfile"].as<std::string>();
        }

        if(pr.count("ml-cands-forestfile")){
            result.mlForestfileCands = pr["ml-cands-forestfile"].as<std::string>();
        }

        if(pr.count("inputfiles")){
            result.inputfiles = pr["inputfiles"].as<std::vector<std::string>>();
        }

        if(pr.count("outputfilenames")){
            result.outputfilenames = pr["outputfilenames"].as<std::vector<std::string>>();
        }

    }

    bool ProgramOptions::isValid() const noexcept{
        const ProgramOptions& opt = *this;
        bool valid = true;

        if(opt.maxErrorRate < 0.0f || opt.maxErrorRate > 1.0f){
            valid = false;
            std::cout << "Error: maxmismatchratio must be in range [0.0, 1.0], is " + std::to_string(opt.maxErrorRate) << std::endl;
        }

        if(opt.min_overlap < 1){
            valid = false;
            std::cout << "Error: min_overlap must be > 0, is " + std::to_string(opt.min_overlap) << std::endl;
        }

        if(opt.min_overlap_ratio < 0.0f || opt.min_overlap_ratio > 1.0f){
            valid = false;
            std::cout << "Error: min_overlap_ratio must be in range [0.0, 1.0], is "
                        + std::to_string(opt.min_overlap_ratio) << std::endl;
        }

        if(opt.estimatedCoverage <= 0.0f){
            valid = false;
            std::cout << "Error: estimatedCoverage must be > 0.0, is " + std::to_string(opt.estimatedCoverage) << std::endl;
        }

        if(opt.estimatedErrorrate <= 0.0f){
            valid = false;
            std::cout << "Error: estimatedErrorrate must be > 0.0, is " + std::to_string(opt.estimatedErrorrate) << std::endl;
        }

        if(opt.batchsize < 1 /*|| corOpts.batchsize > 16*/){
            valid = false;
            std::cout << "Error: batchsize must be in range [1, ], is " + std::to_string(opt.batchsize) << std::endl;
        }

        if(opt.numHashFunctions < 1){
            valid = false;
            std::cout << "Error: Number of hashmaps must be >= 1, is " + std::to_string(opt.numHashFunctions) << std::endl;
        }

        if(opt.kmerlength < 0 || opt.kmerlength > max_k<kmer_type>::value){
            valid = false;
            std::cout << "Error: kmer length must be in range [0, " << max_k<kmer_type>::value 
                << "], is " + std::to_string(opt.kmerlength) << std::endl;
        }

        if(opt.insertSize < 0){
            valid = false;
            std::cout << "Error: insert size must be >= 0, is " 
                << opt.insertSize << std::endl;
        }

        if(opt.insertSizeStddev < 0){
            valid = false;
            std::cout << "Error: insert size deviation must be >= 0, is " 
                << opt.insertSizeStddev << std::endl;
        }

        if(opt.fixedStddev < 0){
            valid = false;
            std::cout << "Error: fixedStddev must be >= 0, is " 
                << opt.fixedStddev << std::endl;
        }

        if(opt.fixedStepsize < 0){
            valid = false;
            std::cout << "Error: fixedStepsize must be >= 0, is " 
                << opt.fixedStepsize << std::endl;
        }

        if(opt.threads < 1){
            valid = false;
            std::cout << "Error: threads must be > 0, is " + std::to_string(opt.threads) << std::endl;
        }

        if(opt.qualityScoreBits != 1 && opt.qualityScoreBits != 2 && opt.qualityScoreBits != 8){
            valid = false;
            std::cout << "Error: qualityScoreBits must be 1,2,or 8, is " + std::to_string(opt.qualityScoreBits) << std::endl;
        }

        if(!filesys::exists(opt.tempdirectory)){
            bool created = filesys::create_directories(opt.tempdirectory);
            if(!created){
                valid = false;
                std::cout << "Error: Could not create temp directory" << opt.tempdirectory << std::endl;
            }
        }

        if(!filesys::exists(opt.outputdirectory)){
            bool created = filesys::create_directories(opt.outputdirectory);
            if(!created){
                valid = false;
                std::cout << "Error: Could not create output directory" << opt.outputdirectory << std::endl;
            }
        }

        {
            for(const auto& inputfile : opt.inputfiles){
                std::ifstream is(inputfile);
                if(!(bool)is){
                    valid = false;
                    std::cout << "Error: cannot find input file " << inputfile << std::endl;
                }
            }            
        }

        {
            std::vector<FileFormat> formats;
            for(const auto& inputfile : opt.inputfiles){
                FileFormat f = getFileFormat(inputfile);
                if(f == FileFormat::FASTQGZ)
                    f = FileFormat::FASTQ;
                if(f == FileFormat::FASTAGZ)
                    f = FileFormat::FASTA;
                formats.emplace_back(f);
            }
            bool sameFormats = std::all_of(
                formats.begin()+1, 
                formats.end(), [&](const auto f){
                    return f == formats[0];
                }
            );
            if(!sameFormats){
                valid = false;
                std::cout << "Error: Must not specify both fasta and fastq files!" << std::endl;
            }
        }

        {
            if(opt.outputfilenames.size() > 1 && opt.inputfiles.size() != opt.outputfilenames.size()){
                valid = false;
                std::cout << "Error: An output file name must be specified for each input file. Number of input files : " << opt.inputfiles.size() << ", number of output file names: " << opt.outputfilenames.size() << "\n";
            }
        }

        {
            //Disallow invalid type
            if(opt.pairType == SequencePairType::Invalid){
                valid = false;
                std::cout << "Error: pairmode is invalid." << std::endl;
            }

            //In paired end mode, there must be a single input file with interleaved reads, or exactly two input files, one per direction.
            if(opt.pairType == SequencePairType::PairedEnd){
                const int countOk = opt.inputfiles.size() == 1 || opt.inputfiles.size() == 2;
                if(!countOk){
                    valid = false;
                    std::cout << "Error: Invalid number of input files for selected pairmode 'PairedEnd'." << std::endl;
                }
            }

            //In single end mode, a single file allowed
            if(opt.pairType == SequencePairType::SingleEnd){
                const int countOk = opt.inputfiles.size() == 1;
                if(!countOk){
                    valid = false;
                    std::cout << "Error: Invalid number of input files for selected pairmode 'SingleEnd'." << std::endl;
                }
            }
        }

        return valid;
    }


    void addMandatoryOptions(cxxopts::Options& commandLineOptions){
        commandLineOptions.add_options("Mandatory")
            ("d,outdir", "The output directory. Will be created if it does not exist yet.", 
            cxxopts::value<std::string>())
            ("c,coverage", "Estimated coverage of input file. (i.e. number_of_reads * read_length / genome_size)", 
            cxxopts::value<float>())
            ("i,inputfiles", 
                "The file(s) to correct. "
                "Fasta or Fastq format. May be gzip'ed. "
                "Repeat this option for each input file (e.g. -i file1.fastq -i file2.fastq). "
                "Must not mix fasta and fastq files. "
                "The collection of input files is treated as a single read library",
                cxxopts::value<std::vector<std::string>>())
            ("o,outputfilenames", 
                "The names of outputfiles. "
                "Repeat this option for each output file (e.g. -o file1_corrected.fastq -o file2_corrected.fastq). "
                "If a single output file is specified, it will contain the concatenated results of all input files. "
                "If multiple output files are specified, the number of output files must be equal to the number of input files. "
                "In this case, output file i will contain the results of input file i. "
                "Output files are uncompressed.", 
                cxxopts::value<std::vector<std::string>>())
            ("pairmode", 
                "Type of input reads."
                "SE / se : Single-end reads"
                "PE / pe : Paired-end reads",
                cxxopts::value<std::string>());
    }

    void addMandatoryOptionsExtend(cxxopts::Options& commandLineOptions){
        commandLineOptions.add_options("Mandatory")
            ("insertsize", 
                "Insert size for paired reads. -- explanation how insert size is interpreted ---", 
                cxxopts::value<int>())
            ("insertsizedev", 
                "Insert size deviation for paired reads.", 
                cxxopts::value<int>())
            ("eo", 
                "The name of the output file containing extended reads",
                cxxopts::value<std::string>());
    }

    void addMandatoryOptionsCorrectCpu(cxxopts::Options&){
        //nothing
    }

    void addMandatoryOptionsCorrectGpu(cxxopts::Options& commandLineOptions){
        commandLineOptions.add_options("Mandatory")
		    ("g,gpu", "Comma-separated list of GPU device ids to be used. (Example: --gpu 0,1 to use GPU 0 and GPU 1)", cxxopts::value<std::vector<int>>());
    }

    void addMandatoryOptionsExtendCpu(cxxopts::Options&){
        //nothing
    }

    void addMandatoryOptionsExtendGpu(cxxopts::Options& commandLineOptions){
        commandLineOptions.add_options("Mandatory")        
		    ("g,gpu", "Comma-separated list of GPU device ids to be used. (Example: --gpu 0,1 to use GPU 0 and GPU 1)", cxxopts::value<std::vector<int>>());
    }


} //namespace care