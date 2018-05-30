#include "../inc/care.hpp"

#include "../inc/cxxopts/cxxopts.hpp"


#include <iostream>
#include <string>

int main(int argc, const char** argv){

	bool help = false;

	cxxopts::Options options(argv[0], "Perform error correction on a fastq file");

	options.add_options("Group")
		("h", "Show this help message", cxxopts::value<bool>(help))
		("inputfile", "The fastq file to correct", cxxopts::value<std::string>())
		("outdir", "The output directory", cxxopts::value<std::string>()->default_value("."))
		("outfile", "The output file", cxxopts::value<std::string>()->default_value("")->implicit_value(""))
		("hashmaps", "The number of hash maps. Must be greater than 0.", cxxopts::value<int>()->default_value("2")->implicit_value("2"))
		("kmerlength", "The kmer length for minhashing. Must be greater than 0.", cxxopts::value<int>()->default_value("16")->implicit_value("16"))
		("threads", "Maximum number of thread to use. Must be greater than 0", cxxopts::value<int>()->default_value("1"))
		("base", "Graph parameter for cutoff (alpha*pow(base,edge))", cxxopts::value<double>()->default_value("1.1")->implicit_value("1.1"))
		("alpha", "Graph parameter for cutoff (alpha*pow(base,edge))", cxxopts::value<double>()->default_value("1.0")->implicit_value("1.0"))
		("batchsize", "This mainly affects the GPU alignment since the alignments of batchsize reads to their candidates is done in parallel.Must be greater than 0.",
				 cxxopts::value<int>()->default_value("5"))
		("useQualityScores", "If set, quality scores (if any) are considered during read correction",
				 cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("candidateCorrection", "If set, candidate reads will be corrected,too.",
 				 cxxopts::value<bool>()->default_value("false")->implicit_value("true"))

		("matchscore", "Score for match during alignment.", cxxopts::value<int>()->default_value("1")->implicit_value("1"))
		("subscore", "Score for substitution during alignment.", cxxopts::value<int>()->default_value("-1")->implicit_value("-1"))
		("insertscore", "Score for insertion during alignment.", cxxopts::value<int>()->default_value("-100")->implicit_value("-100"))
		("deletionscore", "Score for deletion during alignment.", cxxopts::value<int>()->default_value("-100")->implicit_value("-100"))

		("maxmismatchratio", "Overlap between query and candidate must contain at most maxmismatchratio * overlapsize mismatches",
					cxxopts::value<double>()->default_value("0.2")->implicit_value("0.2"))
		("minalignmentoverlap", "Overlap between query and candidate must be at least this long", cxxopts::value<int>()->default_value("35")->implicit_value("35"))
		("minalignmentoverlapratio", "Overlap between query and candidate must be at least as long as minalignmentoverlapratio * querylength",
					cxxopts::value<double>()->default_value("0.35")->implicit_value("0.35"))
		("fileformat", "Format of input file. Allowed values: {fastq}",
					cxxopts::value<std::string>()->default_value("fastq")->implicit_value("fastq"))

		("coverage", "estimated coverage of input file",
					cxxopts::value<double>()->default_value("20.0")->implicit_value("20.0"))
		("errorrate", "estimated error rate of input file",
					cxxopts::value<double>()->default_value("0.03")->implicit_value("0.03"))
		("m_coverage", "m",
					cxxopts::value<double>()->default_value("0.6")->implicit_value("0.6"))
        ("indels", "If set, a semi-global alignment is performed which allows for the correction of both substitutions and indels. If not set, the shifted hamming distance is calculated which allows for the correction of substitutions.",
                 cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("extractFeatures", "If set, extract MSA features",
              cxxopts::value<bool>()->default_value("false")->implicit_value("true"))

	;

	auto parseresults = options.parse(argc, argv);

	if(help){
	      	std::cout << options.help({"", "Group"}) << std::endl;
		exit(0);
	}

	care::performCorrection(parseresults);


	return 0;
}
