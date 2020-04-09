
#include <config.hpp>

#include <cxxopts/cxxopts.hpp>
#include <args.hpp>
#include <options.hpp>
#include <dispatch_care.hpp>

#include <cpugpuproxy.hpp>

#include <threadpool.hpp>

#include <readlibraryio.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>

#include <experimental/filesystem>

namespace filesys = std::experimental::filesystem;

using namespace care;

void printCommandlineArguments(std::ostream& out, const cxxopts::ParseResult& parseresults){

	const auto args = parseresults.arguments();
	for(const auto& opt : args){
		out << opt.key() << '=' << opt.value() << '\n';
	}
}

bool checkMandatoryArguments(const cxxopts::ParseResult& parseresults){

	const std::vector<std::string> mandatory = {
		"inputfile", "outdir", "outfile", "coverage"
	};

	bool success = true;
	for(const auto& opt : mandatory){
		if(parseresults.count(opt) == 0){
			success = false;
			std::cerr << "Mandatory argument " << opt << " is missing.\n";
		}
	}

	return success;
}

int main(int argc, char** argv){

	bool help = false;

	cxxopts::Options options(argv[0], "Perform error correction on a fastq file");

	options.add_options("Mandatory")
		("inputfile", "The file to correct. May be a gzip file. Reads are treated as single end.", cxxopts::value<std::string>())
		("outdir", "The output directory", cxxopts::value<std::string>())
		("outfile", "The output file", cxxopts::value<std::string>())
		("coverage", "Estimated coverage of input file", cxxopts::value<float>());

	options.add_options("Optional")
		("h", "Show this help message", cxxopts::value<bool>(help))
		("tempdir", "Directory to store temporary files. Default is output directory", cxxopts::value<std::string>())
		("hashmaps", "The number of hash maps. Must be greater than 0.", cxxopts::value<int>()->default_value("2"))
		("kmerlength", "The kmer length for minhashing. Must be greater than 0.", cxxopts::value<int>()
			->default_value(std::to_string(max_k<kmer_type>::value)))
		("threads", "Maximum number of thread to use. Must be greater than 0", cxxopts::value<int>()->default_value("1"))
		("batchsize", "Number of reads to correct in a single batch. Must be greater than 0.",
		cxxopts::value<int>()->default_value("1000"))

		("useQualityScores", "If set, quality scores (if any) are considered during read correction",
		cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
		("candidateCorrection", "If set, candidate reads will be corrected,too.",
		cxxopts::value<bool>()->default_value("false")->implicit_value("true"))

        ("candidateCorrectionNewColumns", "If candidateCorrection is set, a candidates with an absolute shift of candidateCorrectionNewColumns compared to anchor are corrected",
		cxxopts::value<int>()->default_value("5"))


		("maxmismatchratio", "Overlap between query and candidate must contain at most maxmismatchratio * overlapsize mismatches",
		cxxopts::value<float>()->default_value("0.2"))
		("minalignmentoverlap", "Overlap between query and candidate must be at least this long", 
		cxxopts::value<int>()->default_value("35"))
		("minalignmentoverlapratio", "Overlap between query and candidate must be at least as long as minalignmentoverlapratio * querylength",
		cxxopts::value<float>()->default_value("0.35"))
		("fileformat", "Format of input file. Overrides automatic detection. Allowed values: {fasta, fastq, fastagz, fastqgz}",
		cxxopts::value<std::string>()->default_value(""))

		("errorrate", "estimated error rate of input file",
		cxxopts::value<float>()->default_value("0.03"))
		("m_coverage", "m",
		cxxopts::value<float>()->default_value("0.6"))

		("deviceIds", "Space separated GPU device ids to be used for correction", cxxopts::value<std::vector<std::string> >()->default_value({}))
		("nReads", "Upper bound for number of reads in the inputfile. If set 0, the input file is parsed to find the exact number of reads before any work is done.",
		cxxopts::value<std::uint64_t>()->default_value("0"))
		("min_length", "Lower bound for read length in file. If set negative, the input file is parsed to find the exact minimum length before any work is done.",
		cxxopts::value<int>()->default_value("-1"))
		("max_length", "Upper bound for read length in file. If set 0, the input file is parsed to find the exact maximum length before any work is done.",
		cxxopts::value<int>()->default_value("0"))

		("progress", "If set, progress bar is shown during correction",
		cxxopts::value<bool>()->default_value("false")->implicit_value("true"))

		("save-binary-reads-to", "Save binary dump of loaded reads from inputfile to disk",
		cxxopts::value<std::string>()->default_value(""))
		("load-binary-reads-from", "Load binary dump of reads from disk",
		cxxopts::value<std::string>()->default_value(""))
		("save-hashtables-to", "Save binary dump of hash tables to disk",
		cxxopts::value<std::string>()->default_value(""))
		("load-hashtables-from", "Load binary dump of hash tables from disk",
		cxxopts::value<std::string>()->default_value(""))

		("memHashtables", "Memory limit for hash tables and hash table construction",
		cxxopts::value<std::string>()->default_value("0"))

		("memTotal", "Total memory limit (This is not a hard limit)",
		cxxopts::value<std::string>()->default_value("0"))
		
	;

	options.parse_positional({"deviceIds"});

	auto parseresults = options.parse(argc, argv);

	if(help) {
		std::cout << options.help({"", "Mandatory", "Optional"}) << std::endl;
		exit(0);
	}

	//printCommandlineArguments(std::cerr, parseresults);

	const bool mandatoryPresent = checkMandatoryArguments(parseresults);
	if(!mandatoryPresent){
		std::cout << options.help({"Mandatory"}) << std::endl;
		std::exit(0);
	}

	MinhashOptions minhashOptions = args::to<care::MinhashOptions>(parseresults);
	AlignmentOptions alignmentOptions = args::to<AlignmentOptions>(parseresults);
	GoodAlignmentProperties goodAlignmentProperties = args::to<GoodAlignmentProperties>(parseresults);
	CorrectionOptions correctionOptions = args::to<CorrectionOptions>(parseresults);
	RuntimeOptions runtimeOptions = args::to<RuntimeOptions>(parseresults);
	MemoryOptions memoryOptions = args::to<MemoryOptions>(parseresults);
	FileOptions fileOptions = args::to<FileOptions>(parseresults);

	if(!args::isValid(minhashOptions)) throw std::runtime_error("Invalid minhashOptions!");
	if(!args::isValid(alignmentOptions)) throw std::runtime_error("Invalid alignmentOptions!");
	if(!args::isValid(goodAlignmentProperties)) throw std::runtime_error("Invalid goodAlignmentProperties!");
	if(!args::isValid(correctionOptions)) throw std::runtime_error("Invalid correctionOptions!");
	if(!args::isValid(runtimeOptions)) throw std::runtime_error("Invalid runtimeOptions!");
	if(!args::isValid(memoryOptions)) throw std::runtime_error("Invalid memoryOptions!");
	if(!args::isValid(fileOptions)) throw std::runtime_error("Invalid fileOptions!");


	std::cout << "Tempdir = " << fileOptions.tempdirectory << std::endl;

    runtimeOptions.deviceIds = getUsableDeviceIds(runtimeOptions.deviceIds);
    runtimeOptions.canUseGpu = runtimeOptions.deviceIds.size() > 0;

	if(correctionOptions.useQualityScores){
		const bool fileHasQscores = hasQualityScores(fileOptions.inputfile);

		if(!fileHasQscores){
			std::cerr << "Quality scores have been disabled because no quality scores were found in the input file.\n";
			
			correctionOptions.useQualityScores = false;
		}
		
	}
		

    const int numThreads = parseresults["threads"].as<int>();

	omp_set_num_threads(numThreads);

    care::performCorrection(minhashOptions,
				alignmentOptions,
				correctionOptions,
				runtimeOptions,
				memoryOptions,
				fileOptions,
				goodAlignmentProperties);

	return 0;
}
