#include "../inc/args.hpp"
#include "../inc/hpc_helpers.cuh"
#include "../inc/util.hpp"

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

    std::string getFileName(std::string filePath){
        filesys::path path(filePath);
        return path.filename().string();
    }

	template<>
	MinhashOptions to<MinhashOptions>(const cxxopts::ParseResult& pr){
        MinhashOptions result{pr["hashmaps"].as<int>(),
    					      pr["kmerlength"].as<int>(),
                              0.0};

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
        CorrectionMode correctionMode = CorrectionMode::Hamming;
        if(pr["indels"].as<bool>()){
            correctionMode = CorrectionMode::Graph;
        }
        CorrectionOptions result{
            correctionMode,
            pr["candidateCorrection"].as<bool>(),
			pr["useQualityScores"].as<bool>(),
            pr["coverage"].as<double>(),
            pr["errorrate"].as<double>(),
            pr["m_coverage"].as<double>(),
            pr["alpha"].as<double>(),
            pr["base"].as<double>(),
            pr["kmerlength"].as<int>(),
            pr["batchsize"].as<int>(),
            3, //new_columns_to_correct
            pr["extractFeatures"].as<bool>(),
            pr["classicMode"].as<bool>(),
        };

        return result;
	}

	template<>
	RuntimeOptions to<RuntimeOptions>(const cxxopts::ParseResult& pr){
        RuntimeOptions result;

		result.threads = pr["threads"].as<int>();
		result.nInserterThreads = std::min(result.threads, (int)std::min(4u, std::thread::hardware_concurrency()));
		result.nCorrectorThreads = std::min(result.threads, (int)std::thread::hardware_concurrency());
        result.showProgress = pr["progress"].as<bool>();
        result.max_candidates = pr["maxCandidates"].as<int>();

        auto deviceIdsStrings = pr["deviceIds"].as<std::vector<std::string>>();

        for(const auto& s : deviceIdsStrings){
            result.deviceIds.emplace_back(std::stoi(s));
        }

#ifdef __NVCC__
        int nDevices;

        cudaGetDeviceCount(&nDevices); CUERR;

        std::vector<int> invalidIds;

        for(int id : result.deviceIds){
            if(id >= nDevices){
                invalidIds.emplace_back(id);
                std::cout << "Found invalid device Id: " << id << std::endl;
            }
        }

        if(invalidIds.size() > 0){
            std::cout << "Available GPUs on your machine:" << std::endl;
            for(int j = 0; j < nDevices; j++){
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, j); CUERR;
                std::cout << "Id " << j << " : " << prop.name << std::endl;
            }

            for(int invalidid : invalidIds){
                result.deviceIds.erase(std::find(result.deviceIds.begin(), result.deviceIds.end(), invalidid));
            }
        }

        result.canUseGpu = result.deviceIds.size() > 0;

#endif
        return result;
	}

	template<>
	FileOptions to<FileOptions>(const cxxopts::ParseResult& pr){
        FileOptions result;

		result.inputfile = pr["inputfile"].as<std::string>();
		result.outputdirectory = pr["outdir"].as<std::string>();
        result.outputfilename = pr["outfile"].as<std::string>();

        if(result.outputfilename == "")
            result.outputfilename = "corrected_" + getFileName(result.inputfile);

		result.outputfile = result.outputdirectory + "/" + result.outputfilename;

		result.fileformatstring = pr["fileformat"].as<std::string>();

		if (result.fileformatstring == "fastq" || result.fileformatstring == "FASTQ" || result.fileformatstring == "fq" || result.fileformatstring == "FQ")
			result.format = FileFormat::FASTQ;
		else
			throw std::runtime_error("Set invalid file format : " + result.fileformatstring);
		
		result.nReads = pr["nReads"].as<std::uint64_t>();

        return result;
	}



    template<>
    bool isValid<MinhashOptions>(const MinhashOptions& opt){
        bool valid = true;

        if(opt.maps < 1){
            valid = false;
            std::cout << "Error: Number of hashmaps must be >= 1, is " + std::to_string(opt.maps) << std::endl;
        }

        if(opt.k < 1 || opt.k > 32){
            valid = false;
            std::cout << "Error: kmer length must be in range [1, 16], is " + std::to_string(opt.k) << std::endl;
        }

        return valid;
    }

    template<>
    bool isValid<AlignmentOptions>(const AlignmentOptions& opt){
        bool valid = true;

        return valid;
    }

    template<>
    bool isValid<GoodAlignmentProperties>(const GoodAlignmentProperties& opt){
        bool valid = true;

        if(opt.maxErrorRate < 0.0 || opt.maxErrorRate > 1.0){
            valid = false;
            std::cout << "Error: maxmismatchratio must be in range [0.0, 1.0], is " + std::to_string(opt.maxErrorRate) << std::endl;
        }

        if(opt.min_overlap < 1){
            valid = false;
            std::cout << "Error: min_overlap must be > 0, is " + std::to_string(opt.min_overlap) << std::endl;
        }

        if(opt.min_overlap_ratio < 0.0 || opt.min_overlap_ratio > 1.0){
            valid = false;
            std::cout << "Error: min_overlap_ratio must be in range [0.0, 1.0], is "
                        + std::to_string(opt.min_overlap_ratio) << std::endl;
        }

        return valid;
    }

    template<>
    bool isValid<CorrectionOptions>(const CorrectionOptions& opt){
        bool valid = true;

        if(opt.estimatedCoverage <= 0.0){
            valid = false;
            std::cout << "Error: estimatedCoverage must be > 0.0, is " + std::to_string(opt.estimatedCoverage) << std::endl;
        }

        if(opt.estimatedErrorrate <= 0.0){
            valid = false;
            std::cout << "Error: estimatedErrorrate must be > 0.0, is " + std::to_string(opt.estimatedErrorrate) << std::endl;
        }

        if(opt.batchsize < 1 /*|| corOpts.batchsize > 16*/){
            valid = false;
            std::cout << "Error: batchsize must be in range [1, ], is " + std::to_string(opt.batchsize) << std::endl;
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

        return valid;
    }

    template<>
    bool isValid<FileOptions>(const FileOptions& opt){
        bool valid = true;

        std::ifstream is(opt.inputfile);
        if(!(bool)is){
            valid = false;
            std::cout << "Error: cannot find input file " << opt.inputfile << std::endl;
        }

        return valid;
    }

}
}
