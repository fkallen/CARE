#include "../inc/args.hpp"

#include <iostream>
#include <thread>
#include <string>
#include <stdexcept>

#include <experimental/filesystem>

namespace filesys = std::experimental::filesystem;

namespace care{
namespace args{
	
	std::string getFileName(std::string filePath)
	{
		filesys::path path(filePath);
		return path.filename().string();
	}	
	
	template<>
	MinhashOptions to<MinhashOptions>(const cxxopts::ParseResult& pr){
        MinhashOptions result{pr["hashmaps"].as<int>(),
    					      pr["kmerlength"].as<int>()};

        return result;
	}

	
	template<>
	AlignmentOptions to<AlignmentOptions>(const cxxopts::ParseResult& pr){
        AlignmentOptions result{
            pr["matchscore"].as<int>(),
            pr["subscore"].as<int>(),
            pr["insertscore"].as<int>(),
            pr["deletionscore"].as<int>()
        };

        return result;
	}
	
	template<>
	GoodAlignmentProperties to<GoodAlignmentProperties>(const cxxopts::ParseResult& pr){
        GoodAlignmentProperties result{
            pr["minalignmentoverlap"].as<int>(),
            pr["maxmismatchratio"].as<double>(),
            pr["minalignmentoverlapratio"].as<double>(),
        };

        return result;
	}
	
	template<>
	CorrectionOptions to<CorrectionOptions>(const cxxopts::ParseResult& pr){
        CorrectionOptions result{
            CorrectionMode::Hamming,
            false, //correct candidates
			pr["useQualityScores"].as<bool>(),
            pr["coverage"].as<double>(),
            pr["errorrate"].as<double>(),
            pr["m_coverage"].as<double>(),
            pr["alpha"].as<double>(),
            pr["base"].as<double>(),
            pr["kmerlength"].as<int>(),
            pr["batchsize"].as<int>()
        };

        return result;
	}
	
	template<>
	RuntimeOptions to<RuntimeOptions>(const cxxopts::ParseResult& pr){
        RuntimeOptions result;
		
		result.threads = pr["threads"].as<int>();
		result.nInserterThreads = std::min(result.threads, (int)std::min(4u, std::thread::hardware_concurrency()));
		result.nCorrectorThreads = std::min(result.threads, (int)std::thread::hardware_concurrency());

        return result;
	}
	
	template<>
	FileOptions to<FileOptions>(const cxxopts::ParseResult& pr){
        FileOptions result;
		
		result.inputfile = pr["inputfile"].as<std::string>();
		result.outputdirectory = pr["outdir"].as<std::string>();
	
		if(pr["outfile"].as<std::string>() == ""){
			result.outputfile = result.outputdirectory + "/corrected_" + getFileName(result.inputfile);
		}else{
			result.outputfile = result.outputdirectory + "/" + pr["outfile"].as<std::string>();
		}
		
		result.fileformatstring = pr["fileformat"].as<std::string>();
		
		if (result.fileformatstring == "fastq" || result.fileformatstring == "FASTQ" || result.fileformatstring == "fq" || result.fileformatstring == "FQ")
			result.format = FileFormat::FASTQ;
		else
			throw std::runtime_error("Set invalid file format : " + result.fileformatstring);		

        return result;
	}

    bool areValid(const cxxopts::ParseResult& args){
        bool valid = true;

		//check minhash options
        auto minhashOptions = args::to<MinhashOptions>(args);
        if(minhashOptions.maps < 1){
            valid = false;
            std::cout << "Error: Number of hashmaps must be >= 1, is " + std::to_string(minhashOptions.maps) << std::endl;
        }

        if(minhashOptions.k < 1 || minhashOptions.k > 16){
            valid = false;
            std::cout << "Error: kmer length must be in range [1, 16], is " + std::to_string(minhashOptions.k) << std::endl;
        }

        //check good alignment properties
        auto goodAlignmentProperties = args::to<GoodAlignmentProperties>(args);

        if(goodAlignmentProperties.max_mismatch_ratio < 0.0 || goodAlignmentProperties.max_mismatch_ratio > 1.0){
            valid = false;
            std::cout << "Error: maxmismatchratio must be in range [0.0, 1.0], is " + std::to_string(goodAlignmentProperties.max_mismatch_ratio) << std::endl;
        }

        if(goodAlignmentProperties.min_overlap < 1){
            valid = false;
            std::cout << "Error: min_overlap must be > 0, is " + std::to_string(goodAlignmentProperties.min_overlap) << std::endl;
        }

        if(goodAlignmentProperties.min_overlap_ratio < 0.0 || goodAlignmentProperties.min_overlap_ratio > 1.0){
            valid = false;
            std::cout << "Error: min_overlap_ratio must be in range [0.0, 1.0], is "
                        + std::to_string(goodAlignmentProperties.min_overlap_ratio) << std::endl;
        }
        
        //check correction options        
        auto corOpts = args::to<CorrectionOptions>(args);
		
        if(corOpts.estimatedCoverage <= 0.0){
            valid = false;
            std::cout << "Error: estimatedCoverage must be > 0.0, is " + std::to_string(corOpts.estimatedCoverage) << std::endl;
        }
        
        if(corOpts.estimatedErrorrate <= 0.0){
            valid = false;
            std::cout << "Error: estimatedErrorrate must be > 0.0, is " + std::to_string(corOpts.estimatedErrorrate) << std::endl;
        }	 
        
        if(corOpts.batchsize < 1 || corOpts.batchsize > 5){
            valid = false;
            std::cout << "Error: batchsize must be in range [1, 5], is " + std::to_string(corOpts.batchsize) << std::endl;
        }	
        
        //check runtime options
        auto runtimeOpts = args::to<RuntimeOptions>(args);
		
		if(runtimeOpts.threads < 1){
            valid = false;
            std::cout << "Error: threads must be > 0, is " + std::to_string(runtimeOpts.threads) << std::endl;
        }
        
        auto fileOpts = args::to<FileOptions>(args);
		
		std::ifstream is(fileOpts.inputfile);
		if(!(bool)is){
			std::cout << "Error: cannot find input file " << fileOpts.inputfile << std::endl;
		}

        return valid;
    }



}
}
