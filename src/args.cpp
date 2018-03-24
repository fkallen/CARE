#include "../inc/args.hpp"

#include <iostream>

namespace care{

    Args::Args(int argc, char** argv)
        : options(cxxopts::Options("CARE", "Perform error correction on a fastq file.")){

    	options.add_options("Group")
    		("h", "Show this help message", cxxopts::value<bool>(help))
    		("i,inputfile", "The fastq file to correct", cxxopts::value<std::string>())
    		("o,outdir", "The output directory", cxxopts::value<std::string>()->default_value("")->implicit_value(""))
    		("outfile", "The output file", cxxopts::value<std::string>()->default_value("")->implicit_value(""))
    		("m,hashmaps", "The number of hash maps. Must be greater than 0.", cxxopts::value<int>()->default_value("2")->implicit_value("2"))
    		("k,kmerlength", "The kmer length for minhashing. Must be greater than 0.", cxxopts::value<int>()->default_value("16")->implicit_value("16"))
    		("insertthreads", "Number of threads to build database. Must be greater than 0.", cxxopts::value<int>()->default_value("1")->implicit_value("1"))
    		("correctorthreads", "Number of threads to correct reads. Must be greater than 0.", cxxopts::value<int>()->default_value("1")->implicit_value("1"))
    		("x,base", "Graph parameter for cutoff (alpha*pow(base,edge))", cxxopts::value<double>()->default_value("1.1")->implicit_value("1.1"))
    		("a,alpha", "Graph parameter for cutoff (alpha*pow(base,edge))", cxxopts::value<double>()->default_value("1.0")->implicit_value("1.0"))
    		("b,batchsize", "This mainly affects the GPU alignment since the alignments of batchsize reads to their candidates is done in parallel.Must be greater than 0.",
    				 cxxopts::value<int>()->default_value("10")->implicit_value("10"))
    		("useQualityScores", "If set, quality scores (if any) are considered during read correction",
    				 cxxopts::value<bool>(useQScores))

    		("matchscore", "Score for match during alignment.", cxxopts::value<int>()->default_value("1")->implicit_value("1"))
    		("subscore", "Score for substitution during alignment.", cxxopts::value<int>()->default_value("-1")->implicit_value("-1"))
    		("insertscore", "Score for insertion during alignment.", cxxopts::value<int>()->default_value("-100")->implicit_value("-100"))
    		("deletionscore", "Score for deletion during alignment.", cxxopts::value<int>()->default_value("-100")->implicit_value("-100"))

    		("maxmismatchratio", "Overlap between query and candidate must contain at most maxmismatchratio * overlapsize mismatches",
    					cxxopts::value<double>()->default_value("0.2")->implicit_value("0.2"))
    		("minalignmentoverlap", "Overlap between query and candidate must be at least this long", cxxopts::value<int>()->default_value("35")->implicit_value("35"))
    		("minalignmentoverlapratio", "Overlap between query and candidate must be at least as long as minalignmentoverlapratio * querylength",
    					cxxopts::value<double>()->default_value("0.35")->implicit_value("0.35"))
    		("f,fileformat", "Format of input file. Allowed values: {fastq}",
    					cxxopts::value<std::string>()->default_value("fastq")->implicit_value("fastq"))

    		("c,coverage", "estimated coverage of input file",
    					cxxopts::value<double>()->default_value("20.0")->implicit_value("20.0"))
    		("r,errorrate", "estimated error rate of input file",
    					cxxopts::value<double>()->default_value("0.03")->implicit_value("0.03"))
    		("m_coverage", "m",
    					cxxopts::value<double>()->default_value("0.6")->implicit_value("0.6"))
    	;

        options.parse(argc, argv);
    }

    Args::Args(const cxxopts::Options& options) : options(options){}

    MinhashOptions Args::getMinhashOptions() const{
        MinhashOptions result{options["hashmaps"].as<int>(),
    					      options["kmerlength"].as<int>()};

        return result;
    }

    AlignmentOptions Args::getAlignmentOptions() const{
        AlignmentOptions result{
            options["matchscore"].as<int>(),
            options["subscore"].as<int>(),
            options["insertscore"].as<int>(),
            options["deletionscore"].as<int>()
        };

        return result;
    }

    GoodAlignmentProperties Args::getGoodAlignmentProperties() const{
        GoodAlignmentProperties result{
            options["minalignmentoverlap"].as<int>(),
            options["maxmismatchratio"].as<double>(),
            options["minalignmentoverlapratio"].as<double>(),
        };

        return result;
    }

    CorrectionOptions Args::getCorrectionOptions() const{
        CorrectionOptions result{
            CorrectionMode::Hamming,
            false, //correct candidates
            useQScores,
            options["coverage"].as<double>(),
            options["errorrate"].as<double>(),
            options["m_coverage"].as<double>(),
            options["alpha"].as<double>(),
            options["base"].as<double>(),
            options["kmerlength"].as<int>()
        };

        return result;
    }

    bool Args::isValid() const{
        bool valid = true;

        auto minhashOptions = getMinhashOptions();
        if(minhashOptions.maps < 1){
            valid = false;
            std::cout << "Error: Number of hashmaps must be >= 1, is " + std::to_string(minhashOptions.maps) << std::endl;
        }

        if(minhashOptions.k < 1 || minhashOptions.k > 16){
            valid = false;
            std::cout << "Error: kmer length must be in range [1, 16], is " + std::to_string(minhashOptions.k) << std::endl;
        }

        auto goodAlignmentProperties = getGoodAlignmentProperties();

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
