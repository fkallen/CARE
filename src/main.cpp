
#include <config.hpp>

#include <cxxopts/cxxopts.hpp>
#include <args.hpp>
#include <options.hpp>
#include <dispatch_care.hpp>

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
		"inputfiles", "outdir", "outputfilenames", "coverage"
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

template<class T>
std::string tostring(const T& t){
	return std::to_string(t);
}

template<>
std::string tostring(const bool& b){
	return b ? "true" : "false";
}

int main(int argc, char** argv){

	bool help = false;

	cxxopts::Options options(argv[0], "CARE: Context-Aware Read Error Correction for Illumina reads");

	options.add_options("Mandatory")
		("d,outdir", "The output directory. Will be created if it does not exist yet.", 
		cxxopts::value<std::string>())
		("c,coverage", "Estimated coverage of input file. (i.e. number_of_reads * read_length / genome_size)", 
		cxxopts::value<float>())
		("i,inputfiles", 
			"The file(s) to correct. "
			"Fasta or Fastq format. May be gzip'ed. "
			"Repeat this option for each input file (e.g. -i file1.fastq -i file2.fastq). "
			"Must not mix fasta and fastq files. Input files are treated as unpaired. "
			"The collection of input files is treated as a single read library",
			cxxopts::value<std::vector<std::string>>())
		("o,outputfilenames", 
			"The names of outputfiles. "
			"Repeat this option for each output file (e.g. -o file1_corrected.fastq -o file2_corrected.fastq). "
			"If a single output file is specified, it will contain the concatenated results of all input files. "
			"If multiple output files are specified, the number of output files must be equal to the number of input files. "
			"In this case, output file i will contain the results of input file i. "
			"Output files are uncompressed.", 
			cxxopts::value<std::vector<std::string>>());

	options.add_options("Mandatory GPU")
		("g,gpu", "One or more GPU device ids to be used for correction. ", cxxopts::value<std::vector<int>>());

	options.add_options("Additional")
			
		("help", "Show this help message", cxxopts::value<bool>(help))
		("tempdir", "Directory to store temporary files. Default: output directory", cxxopts::value<std::string>())
		("h,hashmaps", "The requested number of hash maps. Must be greater than 0. "
			"The actual number of used hash maps may be lower to respect the set memory limit. "
			"Default: " + tostring(CorrectionOptions{}.numHashFunctions), 
			cxxopts::value<int>())
		("k,kmerlength", "The kmer length for minhashing. If 0 or missing, it is automatically determined.", cxxopts::value<int>())
		("enforceHashmapCount",
			"If the requested number of hash maps cannot be fullfilled, the program terminates without error correction. "
			"Default: " + tostring(CorrectionOptions{}.mustUseAllHashfunctions),
			cxxopts::value<bool>()->implicit_value("true")
		)
		("t,threads", "Maximum number of thread to use. Must be greater than 0", cxxopts::value<int>())
		("batchsize", "Number of reads to correct in a single batch. Must be greater than 0. "
			"In CARE CPU, one batch per thread is used. In CARE GPU, two batches per GPU are used. "
			"Default: " + tostring(CorrectionOptions{}.batchsize),
		cxxopts::value<int>())
		("q,useQualityScores", "If set, quality scores (if any) are considered during read correction. "
			"Default: " + tostring(CorrectionOptions{}.useQualityScores),
		cxxopts::value<bool>()->implicit_value("true"))
		("excludeAmbiguous", 
			"If set, reads which contain at least one ambiguous nucleotide will not be corrected. "
			"Default: " + tostring(CorrectionOptions{}.excludeAmbiguousReads),
		cxxopts::value<bool>()->implicit_value("true"))
		("candidateCorrection", "If set, candidate reads will be corrected,too. "
			"Default: " + tostring(CorrectionOptions{}.correctCandidates),
		cxxopts::value<bool>()->implicit_value("true"))
        ("candidateCorrectionNewColumns", "If candidateCorrection is set, a candidates with an absolute shift of candidateCorrectionNewColumns compared to anchor are corrected. "
			"Default: " + tostring(CorrectionOptions{}.new_columns_to_correct),
		cxxopts::value<int>())
		("maxmismatchratio", "Overlap between anchor and candidate must contain at "
			"most (maxmismatchratio * overlapsize) mismatches. "
			"Default: " + tostring(GoodAlignmentProperties{}.maxErrorRate),
		cxxopts::value<float>())
		("minalignmentoverlap", "Overlap between anchor and candidate must be at least this long. "
			"Default: " + tostring(GoodAlignmentProperties{}.min_overlap),
		cxxopts::value<int>())
		("minalignmentoverlapratio", "Overlap between anchor and candidate must be at least as "
			"long as (minalignmentoverlapratio * candidatelength). "
			"Default: " + tostring(GoodAlignmentProperties{}.min_overlap_ratio),
		cxxopts::value<float>())
		("errorfactortuning", "errorfactortuning. "
			"Default: " + tostring(CorrectionOptions{}.estimatedErrorrate),
		cxxopts::value<float>())
		("coveragefactortuning", "coveragefactortuning. "
			"Default: " + tostring(CorrectionOptions{}.m_coverage),
		cxxopts::value<float>())
		("nReads", "Upper bound for number of reads in the inputfile. If missing or set 0, the input file is parsed to find the exact number of reads before any work is done.",
		cxxopts::value<std::uint64_t>())
		("min_length", "Lower bound for read length in file. If missing or set 0, the input file is parsed to find the exact minimum length before any work is done.",
		cxxopts::value<int>())
		("max_length", "Upper bound for read length in file. If missing or set 0, the input file is parsed to find the exact maximum length before any work is done.",
		cxxopts::value<int>())
		("p,showProgress", "If set, progress bar is shown during correction",
		cxxopts::value<bool>()->implicit_value("true"))
		("save-preprocessedreads-to", "Save binary dump of data structure which stores input reads to disk",
		cxxopts::value<std::string>())
		("load-preprocessedreads-from", "Load binary dump of read data structure from disk",
		cxxopts::value<std::string>())
		("save-hashtables-to", "Save binary dump of hash tables to disk",
		cxxopts::value<std::string>())
		("load-hashtables-from", "Load binary dump of hash tables from disk",
		cxxopts::value<std::string>())
		("memHashtables", "Memory limit in bytes for hash tables and hash table construction. Can use suffix K,M,G , e.g. 20G means 20 gigabyte. This option is not a hard limit. Default: A bit less than memTotal.",
		cxxopts::value<std::string>())
		("m,memTotal", "Total memory limit in bytes. Can use suffix K,M,G , e.g. 20G means 20 gigabyte. This option is not a hard limit. Default: All free memory.",
		cxxopts::value<std::string>())
		("correctionType", "0: Classic, 1: Forest, 2: Print",
			cxxopts::value<int>()->default_value("0"))
		("correctionTypeCands", "0: Classic, 1: Forest, 2: Print",
			cxxopts::value<int>()->default_value("0"))
		("ml-forestfile", "The file for interfaceing with the scikit-learn classifier (Anchor correction)",
			cxxopts::value<std::string>())
		("ml-cands-forestfile", "The file for interfaceing with the scikit-learn classifier (Candidate correction)",
			cxxopts::value<std::string>())

		
	;

	//options.parse_positional({"deviceIds"});

	auto parseresults = options.parse(argc, argv);

	if(help) {
		std::cout << options.help({"", "Mandatory", "Mandatory GPU", "Additional"}) << std::endl;
		std::exit(0);
	}

	//printCommandlineArguments(std::cerr, parseresults);

	const bool mandatoryPresent = checkMandatoryArguments(parseresults);
	if(!mandatoryPresent){
		std::cout << options.help({"Mandatory"}) << std::endl;
		std::exit(0);
	}

	GoodAlignmentProperties goodAlignmentProperties = args::to<GoodAlignmentProperties>(parseresults);
	CorrectionOptions correctionOptions = args::to<CorrectionOptions>(parseresults);
	RuntimeOptions runtimeOptions = args::to<RuntimeOptions>(parseresults);
	MemoryOptions memoryOptions = args::to<MemoryOptions>(parseresults);
	FileOptions fileOptions = args::to<FileOptions>(parseresults);

	if(!args::isValid(goodAlignmentProperties)) throw std::runtime_error("Invalid goodAlignmentProperties!");
	if(!args::isValid(correctionOptions)) throw std::runtime_error("Invalid correctionOptions!");
	if(!args::isValid(runtimeOptions)) throw std::runtime_error("Invalid runtimeOptions!");
	if(!args::isValid(memoryOptions)) throw std::runtime_error("Invalid memoryOptions!");
	if(!args::isValid(fileOptions)) throw std::runtime_error("Invalid fileOptions!");

    runtimeOptions.deviceIds = getUsableDeviceIds(runtimeOptions.deviceIds);
    runtimeOptions.canUseGpu = runtimeOptions.deviceIds.size() > 0;

	if(correctionOptions.useQualityScores){
		// const bool fileHasQscores = hasQualityScores(fileOptions.inputfile);

		// if(!fileHasQscores){
		// 	std::cerr << "Quality scores have been disabled because no quality scores were found in the input file.\n";
			
		// 	correctionOptions.useQualityScores = false;
		// }
		
		const bool hasQ = std::all_of(
			fileOptions.inputfiles.begin(),
			fileOptions.inputfiles.end(),
			[](const auto& s){
				return hasQualityScores(s);
			}
		);

		if(!hasQ){
			std::cerr << "Quality scores have been disabled because there exist reads in an input file without quality scores.\n";
			
			correctionOptions.useQualityScores = false;
		}

	}

	if(correctionOptions.correctionType != CorrectionType::Classic){
		if(fileOptions.mlForestfileAnchor == ""){
			std::cerr << "CorrectionType is not set to Classic, but no valid classifier file is provided. Abort!\n";
			return 0;
		}

		if(fileOptions.mlForestfileCands == ""){
			fileOptions.mlForestfileCands = fileOptions.mlForestfileAnchor;
		}
	}


	//print all options that will be used
	std::cout << std::boolalpha;
	std::cout << "CARE will be started with the following parameters:\n";

	std::cout << "----------------------------------------\n";


	std::cout << "Alignment absolute required overlap: " << goodAlignmentProperties.min_overlap << "\n";
	std::cout << "Alignment relative required overlap: " << goodAlignmentProperties.min_overlap_ratio << "\n";
	std::cout << "Alignment max relative number of mismatches in overlap: " << goodAlignmentProperties.maxErrorRate << "\n";

	std::cout << "Number of hash tables / hash functions: " << correctionOptions.numHashFunctions << "\n";
	if(correctionOptions.autodetectKmerlength){
		std::cout << "K-mer size for hashing: auto\n";
	}else{
		std::cout << "K-mer size for hashing: " << correctionOptions.kmerlength << "\n";
	}
	
	std::cout << "Exclude ambigious reads from correction: " << correctionOptions.excludeAmbiguousReads << "\n";
	std::cout << "Correct candidate reads: " << correctionOptions.correctCandidates << "\n";
	std::cout << "Max shift for candidate correction: " << correctionOptions.new_columns_to_correct << "\n";
	std::cout << "Use quality scores: " << correctionOptions.useQualityScores << "\n";
	std::cout << "Estimated dataset coverage: " << correctionOptions.estimatedCoverage << "\n";
	std::cout << "errorfactortuning: " << correctionOptions.estimatedErrorrate << "\n";
	std::cout << "coveragefactortuning: " << correctionOptions.m_coverage << "\n";
	std::cout << "Batch size: " << correctionOptions.batchsize << "\n";
	std::cout << "Correction type (anchor): " << int(correctionOptions.correctionType) 
		<< " (" << nameOfCorrectionType(correctionOptions.correctionType) << ")\n";
	std::cout << "Correction type (cands): " << int(correctionOptions.correctionTypeCands) 
		<< " (" << nameOfCorrectionType(correctionOptions.correctionTypeCands) << ")\n";

	std::cout << "Threads: " << runtimeOptions.threads << "\n";
	std::cout << "Show progress bar: " << runtimeOptions.showProgress << "\n";
	std::cout << "Can use GPU(s): " << runtimeOptions.canUseGpu << "\n";
	if(runtimeOptions.canUseGpu){
		std::cout << "GPU device ids: ";
		for(int id : runtimeOptions.deviceIds){
			std::cout << id << " ";
		}
		std::cout << "\n";
	}

	std::cout << "Maximum memory for hash tables: " << memoryOptions.memoryForHashtables << "\n";
	std::cout << "Maximum memory total: " << memoryOptions.memoryTotalLimit << "\n";

	std::cout << "Minimum read length: " << fileOptions.minimum_sequence_length << "\n";
	std::cout << "Maximum read length: " << fileOptions.maximum_sequence_length << "\n";
	std::cout << "Maximum number of reads: " << fileOptions.nReads << "\n";
	std::cout << "Output directory: " << fileOptions.outputdirectory << "\n";
	std::cout << "Temporary directory: " << fileOptions.tempdirectory << "\n";
	std::cout << "Save preprocessed reads to file: " << fileOptions.save_binary_reads_to << "\n";
	std::cout << "Load preprocessed reads from file: " << fileOptions.load_binary_reads_from << "\n";
	std::cout << "Save hash tables to file: " << fileOptions.save_hashtables_to << "\n";
	std::cout << "Load hash tables from file: " << fileOptions.load_hashtables_from << "\n";
	std::cout << "Input files: ";
	for(auto& s : fileOptions.inputfiles){
		std::cout << s << ' ';
	}
	std::cout << "\n";
	std::cout << "Output file names: ";
	for(auto& s : fileOptions.outputfilenames){
		std::cout << s << ' ';
	}
	std::cout << "\n";
	std::cout << "ml-forestfile: " << fileOptions.mlForestfileAnchor << "\n";
	std::cout << "ml-cands-forestfile: " << fileOptions.mlForestfileCands << "\n";
	std::cout << "----------------------------------------\n";
	std::cout << std::noboolalpha;


    const int numThreads = runtimeOptions.threads;

	omp_set_num_threads(numThreads);

    care::performCorrection(
		correctionOptions,
		runtimeOptions,
		memoryOptions,
		fileOptions,
		goodAlignmentProperties
	);

	return 0;
}
