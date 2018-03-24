#include "../inc/args.hpp"

#include <iostream>

namespace care{
namespace args{
	
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
            pr["kmerlength"].as<int>()
        };

        return result;
	}	

    bool areValid(const cxxopts::ParseResult& args){
        bool valid = true;

        auto minhashOptions = args::to<MinhashOptions>(args);
        if(minhashOptions.maps < 1){
            valid = false;
            std::cout << "Error: Number of hashmaps must be >= 1, is " + std::to_string(minhashOptions.maps) << std::endl;
        }

        if(minhashOptions.k < 1 || minhashOptions.k > 16){
            valid = false;
            std::cout << "Error: kmer length must be in range [1, 16], is " + std::to_string(minhashOptions.k) << std::endl;
        }

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

        return valid;
    }



}
}
