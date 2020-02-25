
#include <config.hpp>

#include <cxxopts/cxxopts.hpp>
#include <args.hpp>
#include <options.hpp>
#include <dispatch_care.hpp>

#include <cpugpuproxy.hpp>

#include <threadpool.hpp>

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
		//("threadsForGPUs", "Number of thread to use for GPU work. Must be not be greater than threads and not negative", cxxopts::value<int>()->default_value("0"))
		("base", "Graph parameter for cutoff (alpha*pow(base,edge))", cxxopts::value<float>()->default_value("1.1"))
		("alpha", "Graph parameter for cutoff (alpha*pow(base,edge))", cxxopts::value<float>()->default_value("1.0"))
		("batchsize", "Number of reads to correct in a single batch. Must be greater than 0.",
		cxxopts::value<int>()->default_value("1000"))
		("gpuParallelBatches", "Number of batches to process in parallel in the GPU version. Must be greater than 0.",
		cxxopts::value<int>()->default_value("4"))
		("useQualityScores", "If set, quality scores (if any) are considered during read correction",
		cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
		("candidateCorrection", "If set, candidate reads will be corrected,too.",
		cxxopts::value<bool>()->default_value("false")->implicit_value("true"))

        ("candidateCorrectionNewColumns", "If candidateCorrection is set, a candidates with an absolute shift of candidateCorrectionNewColumns compared to anchor are corrected",
		cxxopts::value<int>()->default_value("5"))


		("matchscore", "Score for match during alignment.", cxxopts::value<int>()->default_value("1"))
		("subscore", "Score for substitution during alignment.", cxxopts::value<int>()->default_value("-1"))
		("insertscore", "Score for insertion during alignment.", cxxopts::value<int>()->default_value("-100"))
		("deletionscore", "Score for deletion during alignment.", cxxopts::value<int>()->default_value("-100"))

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
		("indels", "If set, a semi-global alignment is performed which allows for the correction of both substitutions and indels. If not set, the shifted hamming distance is calculated which allows for the correction of substitutions.",
		cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
		("extractFeatures", "If set, extract MSA features",
		cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
		("deviceIds", "Space separated GPU device ids to be used for correction", cxxopts::value<std::vector<std::string> >()->default_value({}))
		//("classicMode", "If set, MSA correction does not use decision trees",
		//cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("correctionType", "0: Classic, 1: Forest, 2: Convnet",
		cxxopts::value<int>()->default_value("0"))
		("maxCandidates", "Upper bound for number of candidates per read. Reads with more than max_candidates candidates will not be corrected. The program will guess the limit if max_candidates == 0",
		cxxopts::value<int>()->default_value("0"))
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
		("hits_per_candidate", "A read must be hit in at least hits_per_candidate maps to be considered a candidate",
		cxxopts::value<int>()->default_value("1"))
        ("forest", "Forest model",
		cxxopts::value<std::string>()->default_value(""))
        ("nnmodel", "DL model",
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

	if(correctionOptions.correctCandidates && correctionOptions.extractFeatures) {
		std::cout << "Warning! correctCandidates=true cannot be used with extractFeatures=true. Using correctCandidates=false." << std::endl;
		correctionOptions.correctCandidates = false;
	}

    if(correctionOptions.correctionType == CorrectionType::Forest){
        if(fileOptions.forestfilename == ""){
            throw std::runtime_error("Must specify shared object file for forest if forest correction is selected");
        }else{
            std::ifstream is(fileOptions.forestfilename);
            if(!is){
                throw std::runtime_error("Cannot open shared object file for forest.");
            }
        }
    }

    if(correctionOptions.correctionType == CorrectionType::Convnet){
        if(fileOptions.nnmodelfilename == ""){
            throw std::runtime_error("Must specify convnet model if convnet correction is selected.");
        }
    }

	std::cout << "Tempdir = " << fileOptions.tempdirectory << std::endl;

    runtimeOptions.deviceIds = getUsableDeviceIds(runtimeOptions.deviceIds);
    runtimeOptions.canUseGpu = runtimeOptions.deviceIds.size() > 0;

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
