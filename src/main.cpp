#include "../inc/errorcorrector.hpp"

#include "../inc/ganja/hpc_helpers.cuh"

#include "../inc/cxxopts/cxxopts.hpp"


#include <iostream>
#include <string>
#include <chrono>
#include <cstdlib>
#include <experimental/filesystem>

namespace filesys = std::experimental::filesystem;

std::string getFileName(std::string filePath)
{
	filesys::path path(filePath);
	return path.filename().string();
}

int main(int argc, char** argv){

	//for(int i = 0; i < argc; i++)
	//	std::cout << argv[i] << std::endl;

	bool useQScores = false;
	bool help = false;


	cxxopts::Options options(argv[0], "Perform error correction on a fastq file or fasta file");

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
		("f,fileformat", "Format of input file. Allowed values: {fasta, fastq}",
					cxxopts::value<std::string>()->default_value("fastq")->implicit_value("fastq"))

		("c,coverage", "estimated coverage of input file",
					cxxopts::value<int>()->default_value("20")->implicit_value("20"))
		("r,errorrate", "estimated error rate of input file",
					cxxopts::value<double>()->default_value("0.03")->implicit_value("0.03"))
		("m_coverage", "m",
					cxxopts::value<double>()->default_value("0.6")->implicit_value("0.6"))

	;

	options.parse(argc, argv);

	if(help){
	      	std::cout << options.help({"", "Group"}) << std::endl;
		exit(0);
	}


	if(options["hashmaps"].as<int>() < 1
		|| options["kmerlength"].as<int>() < 1
		|| options["insertthreads"].as<int>() < 1
		|| options["correctorthreads"].as<int>() < 1 ){

	      	std::cout << options.help({"", "Group"}) << std::endl;
		exit(0);
	}


TIMERSTARTCPU(INIT)
    care::Args args(options);

	care::ErrorCorrector corrector(args, options["insertthreads"].as<int>(), options["correctorthreads"].as<int>());
TIMERSTOPCPU(INIT)

    std::string inputfile = options["inputfile"].as<std::string>();
    std::string fileformat = options["fileformat"].as<std::string>();
    std::string outputdirectory = options["outdir"].as<std::string>();
    std::string outputfile;
    if(options["outfile"].as<std::string>() == ""){
        outputfile = outputdirectory + "/corrected_" + getFileName(inputfile);
    }else{
        outputfile = outputdirectory + "/" + options["outfile"].as<std::string>();
    }
    filesys::create_directories(outputdirectory);

	corrector.setGraphSettings(options["alpha"].as<double>(), options["base"].as<double>());
	corrector.setBatchsize(options["batchsize"].as<int>());
	corrector.setAlignmentScores(options["matchscore"].as<int>(),
					options["subscore"].as<int>(),
					options["insertscore"].as<int>(),
					options["deletionscore"].as<int>());

	corrector.setMaxMismatchRatio(options["maxmismatchratio"].as<double>());
	corrector.setMinimumAlignmentOverlap(options["minalignmentoverlap"].as<int>());
	corrector.setMinimumAlignmentOverlapRatio(options["minalignmentoverlapratio"].as<double>());
	corrector.setUseQualityScores(useQScores);

	corrector.setEstimatedCoverage(options["coverage"].as<int>());
	corrector.setEstimatedErrorRate(options["errorrate"].as<double>());
	corrector.setM(options["m_coverage"].as<double>());

	corrector.correct(inputfile, fileformat, outputfile);


	return 0;
}
