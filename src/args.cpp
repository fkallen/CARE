#include <args.hpp>
#include <hpc_helpers.cuh>
#include <util.hpp>
#include <config.hpp>
#include <readlibraryio.hpp>
#include <minhasher.hpp>
#include <filehelpers.hpp>

#include <iostream>
#include <thread>
#include <string>
#include <stdexcept>

#include <experimental/filesystem>

namespace filesys = std::experimental::filesystem;

namespace care{
namespace args{

    std::vector<std::string> split(const std::string& str, char c){
    	std::vector<std::string> result;

    	std::stringstream ss(str);
    	std::string s;

    	while (std::getline(ss, s, c)) {
    		result.emplace_back(s);
    	}

    	return result;
    }

	template<>
	GoodAlignmentProperties to<GoodAlignmentProperties>(const cxxopts::ParseResult& pr){

        GoodAlignmentProperties result{};

        if(pr.count("minalignmentoverlap")){
            result.min_overlap = pr["minalignmentoverlap"].as<int>();
        }
        if(pr.count("maxmismatchratio")){
            result.maxErrorRate = pr["maxmismatchratio"].as<float>();
        }
        if(pr.count("minalignmentoverlapratio")){
            result.min_overlap_ratio = pr["minalignmentoverlapratio"].as<float>();
        }

        return result;
	}

	template<>
	CorrectionOptions to<CorrectionOptions>(const cxxopts::ParseResult& pr){
        CorrectionOptions result{};

        if(pr.count("excludeAmbiguous")){
            result.excludeAmbiguousReads = pr["excludeAmbiguous"].as<bool>();
        }

        if(pr.count("candidateCorrection")){
            result.correctCandidates = pr["candidateCorrection"].as<bool>();
        }

        if(pr.count("useQualityScores")){
            result.useQualityScores = pr["useQualityScores"].as<bool>();
        }

        if(pr.count("enforceHashmapCount")){
            result.mustUseAllHashfunctions = pr["enforceHashmapCount"].as<bool>();
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

        return result;
	}

	template<>
	RuntimeOptions to<RuntimeOptions>(const cxxopts::ParseResult& pr){
        RuntimeOptions result{};

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

        return result;
	}

    template<>
	MemoryOptions to<MemoryOptions>(const cxxopts::ParseResult& pr){
        MemoryOptions result{};

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
            result.memoryTotalLimit = parseMemoryString(memoryTotalLimitString);
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
        

        

        return result;
	}

	template<>
	FileOptions to<FileOptions>(const cxxopts::ParseResult& pr){
        FileOptions result{};

        // result.format = FileFormat::NONE;
        // if(pr.count("inputfile")){
		//     result.inputfile = pr["inputfile"].as<std::string>();
        //     result.format = getFileFormat(result.inputfile);
        // }
        if(pr.count("outdir")){
		    result.outputdirectory = pr["outdir"].as<std::string>();
        }
        // if(pr.count("outfile")){
        //     result.outputfilename = pr["outfile"].as<std::string>();
        // }

        // if(result.outputfilename == "")
        //     result.outputfilename = "care_corrected_" + filehelpers::getFileName(result.inputfile);

		// result.outputfile = result.outputdirectory + "/" + result.outputfilename;

        
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

        if(pr.count("inputfiles")){
            result.inputfiles = pr["inputfiles"].as<std::vector<std::string>>();
        }
        if(pr.count("outputfilenames")){
            result.outputfilenames = pr["outputfilenames"].as<std::vector<std::string>>();
        }

        return result;
	}


    template<>
    bool isValid<GoodAlignmentProperties>(const GoodAlignmentProperties& opt){
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

        return valid;
    }

    template<>
    bool isValid<CorrectionOptions>(const CorrectionOptions& opt){
        bool valid = true;

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

        return valid;
    }

    template<>
    bool isValid<RuntimeOptions>(const RuntimeOptions& opt){
        bool valid = true;

        if(opt.threads < 1){
            valid = false;
            std::cout << "Error: threads must be > 0, is " + std::to_string(opt.threads) << std::endl;
        }

        // if(opt.threadsForGPUs < 0){
        //     valid = false;
        //     std::cout << "Error: threadsForGPUs must be >= 0, is " + std::to_string(opt.threadsForGPUs) << std::endl;
        // }
        //
        // if(opt.threadsForGPUs > opt.threads){
        //     valid = false;
        //     std::cout << "Error: threadsForGPUs must be <= threads, is " + std::to_string(opt.threadsForGPUs) << std::endl;
        // }

        return valid;
    }

    template<>
    bool isValid<MemoryOptions>(const MemoryOptions& opt){
        bool valid = true;

        return valid;
    }

    template<>
    bool isValid<FileOptions>(const FileOptions& opt){
        bool valid = true;

        // {
        //     std::ifstream is(opt.inputfile);
        //     if(!(bool)is){
        //         valid = false;
        //         std::cout << "Error: cannot find input file " << opt.inputfile << std::endl;
        //     }
        // }

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
            for(const auto& outputfilename : opt.outputfilenames){
                const std::string outputfile = opt.outputdirectory + "/" + outputfilename;
                std::ofstream os(outputfile);
                if(!(bool)os){
                    valid = false;
                    std::cout << "Error: cannot open output file " << outputfile << std::endl;
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
            std::ofstream os(opt.tempdirectory+"/tmptest");
            if(!(bool)os){
                valid = false;
                std::cout << "Error: cannot open temporary test file " << opt.tempdirectory+"/tmptest" << std::endl;
            }else{
                filehelpers::removeFile(opt.tempdirectory+"/tmptest");
            }
        }

        {
            if(opt.outputfilenames.size() > 1 && opt.inputfiles.size() != opt.outputfilenames.size()){
                valid = false;
                std::cout << "Error: An output file name must be specified for each input file. Number of input files : " << opt.inputfiles.size() << ", number of output file names: " << opt.outputfilenames.size() << "\n";
            }
        }
        
        return valid;
    }

}
}
