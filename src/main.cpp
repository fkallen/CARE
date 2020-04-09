
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
#include <ios>
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
		("inputfile", "The file to correct. May be a gzip file. Always treated as single end.", cxxopts::value<std::string>())
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
		("maxmismatchratio", "Overlap between anchor and candidate must contain at most maxmismatchratio * overlapsize mismatches",
		cxxopts::value<float>()->default_value("0.2"))
		("minalignmentoverlap", "Overlap between anchor and candidate must be at least this long", 
		cxxopts::value<int>()->default_value("35"))
		("minalignmentoverlapratio", "Overlap between anchor and candidate must be at least as long as minalignmentoverlapratio * querylength",
		cxxopts::value<float>()->default_value("0.35"))

		("errorfactortuning", "errorfactortuning",
		cxxopts::value<float>()->default_value("0.06"))
		("coveragefactortuning", "coveragefactortuning",
		cxxopts::value<float>()->default_value("0.6"))
		("deviceIds", "Space separated GPU device ids to be used for correction", cxxopts::value<std::vector<std::string> >()->default_value({}))
		("nReads", "Upper bound for number of reads in the inputfile. If missing or set 0, the input file is parsed to find the exact number of reads before any work is done.",
		cxxopts::value<std::uint64_t>()->default_value("0"))
		("min_length", "Lower bound for read length in file. If missing or set negative, the input file is parsed to find the exact minimum length before any work is done.",
		cxxopts::value<int>()->default_value("-1"))
		("max_length", "Upper bound for read length in file. If missing or set 0, the input file is parsed to find the exact maximum length before any work is done.",
		cxxopts::value<int>()->default_value("0"))

		("progress", "If set, progress bar is shown during correction",
		cxxopts::value<bool>()->default_value("false")->implicit_value("true"))

		("save-preprocessedreads-to", "Save binary dump of data structure which stores input reads to disk",
		cxxopts::value<std::string>()->default_value(""))
		("load-preprocessedreads-from", "Load binary dump of read data structure from disk",
		cxxopts::value<std::string>()->default_value(""))
		("save-hashtables-to", "Save binary dump of hash tables to disk",
		cxxopts::value<std::string>()->default_value(""))
		("load-hashtables-from", "Load binary dump of hash tables from disk",
		cxxopts::value<std::string>()->default_value(""))

		("memHashtables", "Memory limit for hash tables and hash table construction (This is not a hard limit)",
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

    runtimeOptions.deviceIds = getUsableDeviceIds(runtimeOptions.deviceIds);
    runtimeOptions.canUseGpu = runtimeOptions.deviceIds.size() > 0;

	if(correctionOptions.useQualityScores){
		const bool fileHasQscores = hasQualityScores(fileOptions.inputfile);

		if(!fileHasQscores){
			std::cerr << "Quality scores have been disabled because no quality scores were found in the input file.\n";
			
			correctionOptions.useQualityScores = false;
		}
		
	}

	//print all options that will be used
	std::cout << "CARE will be started with the following parameters:\n";

	std::cout << "--------------------------------\n";

	std::cout << "Number of hash tables / hash functions: " << minhashOptions.maps << "\n";
	std::cout << "K-mer size of hashing: " << minhashOptions.k << "\n";

	std::cout << "Alignment absolute required overlap: " << goodAlignmentProperties.min_overlap << "\n";
	std::cout << "Alignment relative required overlap: " << goodAlignmentProperties.min_overlap_ratio << "\n";
	std::cout << "Alignment max relative number of mismatches in overlap: " << goodAlignmentProperties.maxErrorRate << "\n";

	std::cout << "Correct candidate reads: " << std::boolalpha << correctionOptions.correctCandidates << "\n";
	std::cout << "Max shift for candidate correction: " << correctionOptions.new_columns_to_correct << "\n";
	std::cout << "Use quality scores: " << std::boolalpha << correctionOptions.useQualityScores << "\n";
	std::cout << "Estimated dataset coverage: " << correctionOptions.estimatedCoverage << "\n";
	std::cout << "errorfactortuning: " << correctionOptions.estimatedErrorrate << "\n";
	std::cout << "coveragefactortuning: " << correctionOptions.m_coverage << "\n";
	std::cout << "Batch size: " << correctionOptions.batchsize << "\n";

	std::cout << "Threads: " << runtimeOptions.threads << "\n";
	std::cout << "Show progress bar: " << std::boolalpha << runtimeOptions.showProgress << "\n";
	std::cout << "Can use GPU(s): " << std::boolalpha << runtimeOptions.canUseGpu << "\n";
	if(runtimeOptions.canUseGpu){
		std::cout << "GPU device ids: ";
		for(int id : runtimeOptions.deviceIds){
			std::cout << id << " ";
		}
		std::cout << "\n";
	}

	std::cout << "Maximum memory for hash tables: " << memoryOptions.memoryForHashtables << "\n";
	std::cout << "Maximum memory total: " << memoryOptions.memoryTotalLimit << "\n";

	std::cout << "Input file: " << fileOptions.inputfile << "\n";
	std::cout << "Input file type: ";
	switch(fileOptions.format){
		case FileFormat::FASTA: std::cout << "fasta \n"; break;
		case FileFormat::FASTQ: std::cout << "fastq \n"; break;
		case FileFormat::FASTAGZ: std::cout << "fasta.gz \n"; break;
		case FileFormat::FASTQGZ: std::cout << "fastq.gz \n"; break;
		default: std::cout << "unknown! \n"; break;
	}
	std::cout << "Minimum read length: " << fileOptions.minimum_sequence_length << "\n";
	std::cout << "Maximum read length: " << fileOptions.maximum_sequence_length << "\n";
	std::cout << "Maximum number of reads: " << fileOptions.nReads << "\n";
	std::cout << "Output directory: " << fileOptions.outputdirectory << "\n";
	std::cout << "Output filename: " << fileOptions.outputfilename << "\n";
	std::cout << "Output file: " << fileOptions.outputfile << "\n";
	std::cout << "Temporary directory: " << fileOptions.tempdirectory << "\n";
	std::cout << "Save preprocessed reads to file: " << fileOptions.save_binary_reads_to << "\n";
	std::cout << "Load preprocessed reads from file: " << fileOptions.load_binary_reads_from << "\n";
	std::cout << "Save hash tables to file: " << fileOptions.save_hashtables_to << "\n";
	std::cout << "Load hash tables from file: " << fileOptions.load_hashtables_from << "\n";

	std::cout << "--------------------------------\n";


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
