#include <version.hpp>
#include <config.hpp>

#include <cxxopts/cxxopts.hpp>
#include <options.hpp>
#include <dispatch_care_correct_cpu.hpp>

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
		"inputfiles", "outdir", "outputfilenames", "coverage",
		"pairmode"
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
	bool showVersion = false;

	cxxopts::Options commandLineOptions(argv[0], "CARE: Context-Aware Read Error Correction for Illumina reads");

	addMandatoryOptions(commandLineOptions);
	addMandatoryOptionsCorrectCpu(commandLineOptions);

	commandLineOptions.add_options("Additional")			
		("help", "Show this help message", cxxopts::value<bool>(help))
		("version", "Print version", cxxopts::value<bool>(showVersion))
		("tempdir", "Directory to store temporary files. Default: output directory", cxxopts::value<std::string>())
		("h,hashmaps", "The requested number of hash maps. Must be greater than 0. "
			"The actual number of used hash maps may be lower to respect the set memory limit. "
			"Default: " + tostring(ProgramOptions{}.numHashFunctions), 
			cxxopts::value<int>())
		("k,kmerlength", "The kmer length for minhashing. If 0 or missing, it is automatically determined.", cxxopts::value<int>())
		("enforceHashmapCount",
			"If the requested number of hash maps cannot be fullfilled, the program terminates without error correction. "
			"Default: " + tostring(ProgramOptions{}.mustUseAllHashfunctions),
			cxxopts::value<bool>()->implicit_value("true")
		)
		("t,threads", "Maximum number of thread to use. Must be greater than 0", cxxopts::value<int>())
		("q,useQualityScores", "If set, quality scores (if any) are considered during read correction. "
			"Default: " + tostring(ProgramOptions{}.useQualityScores),
		cxxopts::value<bool>()->implicit_value("true"))
		("excludeAmbiguous", 
			"If set, reads which contain at least one ambiguous nucleotide will not be corrected. "
			"Default: " + tostring(ProgramOptions{}.excludeAmbiguousReads),
		cxxopts::value<bool>()->implicit_value("true"))
		("candidateCorrection", "If set, candidate reads will be corrected,too. "
			"Default: " + tostring(ProgramOptions{}.correctCandidates),
		cxxopts::value<bool>()->implicit_value("true"))
        ("candidateCorrectionNewColumns", "If candidateCorrection is set, a candidates with an absolute shift of candidateCorrectionNewColumns compared to anchor are corrected. "
			"Default: " + tostring(ProgramOptions{}.new_columns_to_correct),
		cxxopts::value<int>())
		("maxmismatchratio", "Overlap between anchor and candidate must contain at "
			"most (maxmismatchratio * overlapsize) mismatches. "
			"Default: " + tostring(ProgramOptions{}.maxErrorRate),
		cxxopts::value<float>())
		("minalignmentoverlap", "Overlap between anchor and candidate must be at least this long. "
			"Default: " + tostring(ProgramOptions{}.min_overlap),
		cxxopts::value<int>())
		("minalignmentoverlapratio", "Overlap between anchor and candidate must be at least as "
			"long as (minalignmentoverlapratio * candidatelength). "
			"Default: " + tostring(ProgramOptions{}.min_overlap_ratio),
		cxxopts::value<float>())
		("errorfactortuning", "errorfactortuning. "
			"Default: " + tostring(ProgramOptions{}.estimatedErrorrate),
		cxxopts::value<float>())
		("coveragefactortuning", "coveragefactortuning. "
			"Default: " + tostring(ProgramOptions{}.m_coverage),
		cxxopts::value<float>())
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
		("thresholdAnchor", "Classification threshold for anchor classifier (\"Forest\") mode",
			cxxopts::value<float>())
		("thresholdCands", "Classification threshold for candidates classifier (\"Forest\") mode",
			cxxopts::value<float>())
		("samplingRateAnchor", "sampling rate for anchor features (print mode)",
			cxxopts::value<float>())
		("samplingRateCands", "sampling rate for candidates features (print mode)",
			cxxopts::value<float>())
		("pairedthreshold1", "pairedthreshold1", cxxopts::value<float>())
		("hashloadfactor", "Load factor of hashtables. 0.0 < hashloadfactor < 1.0. Smaller values can improve the runtime at the expense of greater memory usage."
			"Default: " + std::to_string(ProgramOptions{}.hashtableLoadfactor), cxxopts::value<float>())
		("qualityScoreBits", "How many bits should be used to store a single quality score. Allowed values: 1,2,8. If not 8, a lossy compression via binning is used."
			"Default: " + tostring(ProgramOptions{}.qualityScoreBits), cxxopts::value<int>())

		("fixedNumberOfReads", "Process only the first n reads. Default: " + tostring(ProgramOptions{}.fixedNumberOfReads), cxxopts::value<std::size_t>())
		("singlehash", "Use 1 hashtables with h smallest unique hashes. Default: " + tostring(ProgramOptions{}.singlehash), cxxopts::value<bool>())
	;

	//options.parse_positional({"deviceIds"});

	auto parseresults = commandLineOptions.parse(argc, argv);

	if(showVersion){
		std::cout << "CARE version " << CARE_VERSION_STRING << std::endl;
		std::exit(0);
	}

	if(help) {
		std::cout << commandLineOptions.help({"", "Mandatory", "Additional"}) << std::endl;
		std::exit(0);
	}

	//printCommandlineArguments(std::cerr, parseresults);

	const bool mandatoryPresent = checkMandatoryArguments(parseresults);
	if(!mandatoryPresent){
		std::cout << commandLineOptions.help({"Mandatory"}) << std::endl;
		std::exit(0);
	}

	ProgramOptions programOptions(parseresults);

	programOptions.batchsize = 16;

	if(!programOptions.isValid()) throw std::runtime_error("Invalid program options!");

	if(programOptions.useQualityScores){
		// const bool fileHasQscores = hasQualityScores(programOptions.inputfile);

		// if(!fileHasQscores){
		// 	std::cerr << "Quality scores have been disabled because no quality scores were found in the input file.\n";
			
		// 	programOptions.useQualityScores = false;
		// }
		
		const bool hasQ = std::all_of(
			programOptions.inputfiles.begin(),
			programOptions.inputfiles.end(),
			[](const auto& s){
				return hasQualityScores(s);
			}
		);

		if(!hasQ){
			std::cerr << "Quality scores have been disabled because there exist reads in an input file without quality scores.\n";
			
			programOptions.useQualityScores = false;
		}

	}

	if(programOptions.correctionType != CorrectionType::Classic){
		if(programOptions.mlForestfileAnchor == ""){
			std::cerr << "CorrectionType is not set to Classic, but no valid classifier file is provided. Abort!\n";
			return 0;
		}

		if(programOptions.mlForestfileCands == ""){
			programOptions.mlForestfileCands = programOptions.mlForestfileAnchor;
		}
	}


	//print all options that will be used
	std::cout << std::boolalpha;
	std::cout << "CARE CORRECT CPU will be started with the following parameters:\n";

	std::cout << "----------------------------------------\n";


	std::cout << "Alignment absolute required overlap: " << programOptions.min_overlap << "\n";
	std::cout << "Alignment relative required overlap: " << programOptions.min_overlap_ratio << "\n";
	std::cout << "Alignment max relative number of mismatches in overlap: " << programOptions.maxErrorRate << "\n";

	std::cout << "Number of hash tables / hash functions: " << programOptions.numHashFunctions << "\n";
	if(programOptions.autodetectKmerlength){
		std::cout << "K-mer size for hashing: auto\n";
	}else{
		std::cout << "K-mer size for hashing: " << programOptions.kmerlength << "\n";
	}
	
	std::cout << "Exclude ambigious reads from correction: " << programOptions.excludeAmbiguousReads << "\n";
	std::cout << "Correct candidate reads: " << programOptions.correctCandidates << "\n";
	std::cout << "Max shift for candidate correction: " << programOptions.new_columns_to_correct << "\n";
	std::cout << "Use quality scores: " << programOptions.useQualityScores << "\n";
	std::cout << "Estimated dataset coverage: " << programOptions.estimatedCoverage << "\n";
	std::cout << "errorfactortuning: " << programOptions.estimatedErrorrate << "\n";
	std::cout << "coveragefactortuning: " << programOptions.m_coverage << "\n";
	std::cout << "Correction type (anchor): " << int(programOptions.correctionType) 
		<< " (" << to_string(programOptions.correctionType) << ")\n";
	std::cout << "Correction type (cands): " << int(programOptions.correctionTypeCands) 
		<< " (" << to_string(programOptions.correctionTypeCands) << ")\n";

	std::cout << "pairedthreshold1 " << programOptions.pairedthreshold1 << "\n";
	std::cout << "Threads: " << programOptions.threads << "\n";
	std::cout << "Show progress bar: " << programOptions.showProgress << "\n";

	std::cout << "Maximum memory for hash tables: " << programOptions.memoryForHashtables << "\n";
	std::cout << "Maximum memory total: " << programOptions.memoryTotalLimit << "\n";
	std::cout << "Hashtable load factor: " << programOptions.hashtableLoadfactor << "\n";
	std::cout << "Bits per quality score: " << programOptions.qualityScoreBits << "\n";

	std::cout << "Paired mode: " << to_string(programOptions.pairType) << "\n";
	std::cout << "Output directory: " << programOptions.outputdirectory << "\n";
	std::cout << "Temporary directory: " << programOptions.tempdirectory << "\n";
	std::cout << "Save preprocessed reads to file: " << programOptions.save_binary_reads_to << "\n";
	std::cout << "Load preprocessed reads from file: " << programOptions.load_binary_reads_from << "\n";
	std::cout << "Save hash tables to file: " << programOptions.save_hashtables_to << "\n";
	std::cout << "Load hash tables from file: " << programOptions.load_hashtables_from << "\n";
	std::cout << "Input files: ";
	for(auto& s : programOptions.inputfiles){
		std::cout << s << ' ';
	}
	std::cout << "\n";
	std::cout << "Output file names: ";
	for(auto& s : programOptions.outputfilenames){
		std::cout << s << ' ';
	}
	std::cout << "\n";
	std::cout << "ml-forestfile: " << programOptions.mlForestfileAnchor << "\n";
	std::cout << "ml-cands-forestfile: " << programOptions.mlForestfileCands << "\n";
	std::cout << "anchor sampling rate: " << programOptions.sampleRateAnchor << "\n";
	std::cout << "cands sampling rate: " << programOptions.sampleRateCands << "\n";
	std::cout << "classification thresholds: " << programOptions.thresholdAnchor << " | " << programOptions.thresholdCands << "\n";
	std::cout << "----------------------------------------\n";
	std::cout << std::noboolalpha;


    const int numThreads = programOptions.threads;

	omp_set_num_threads(numThreads);

    care::performCorrection(programOptions);

	return 0;
}
