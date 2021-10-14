#include <version.hpp>
#include <config.hpp>

#include <cxxopts/cxxopts.hpp>
#include <args.hpp>
#include <options.hpp>
#include <dispatch_care_extend_cpu.hpp>

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
		"insertsize", "insertsizedev", "pairmode"
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
			"Must not mix fasta and fastq files. "
			"The collection of input files is treated as a single read library",
			cxxopts::value<std::vector<std::string>>())
		("o,outputfilenames", 
			"The names of outputfiles. "
			"Repeat this option for each output file (e.g. -o file1_corrected.fastq -o file2_corrected.fastq). "
			"If a single output file is specified, it will contain the concatenated results of all input files. "
			"If multiple output files are specified, the number of output files must be equal to the number of input files. "
			"In this case, output file i will contain the results of input file i. "
			"Output files are uncompressed.", 
			cxxopts::value<std::vector<std::string>>())
		("insertsize", 
			"Insert size for paired reads. -- explanation how insert size is interpreted ---", 
			cxxopts::value<int>())
		("insertsizedev", 
			"Insert size deviation for paired reads.", 
			cxxopts::value<int>())
		("pairmode", 
			"Type of input reads."
			"SE / se : Single-end reads"
			"PE / pe : Paired-end reads",
			cxxopts::value<std::string>())
		("eo", 
			"The name of the output file containing extended reads",
			cxxopts::value<std::string>());

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
		("q,useQualityScores", "If set, quality scores (if any) are considered during read correction. "
			"Default: " + tostring(CorrectionOptions{}.useQualityScores),
		cxxopts::value<bool>()->implicit_value("true"))
		("excludeAmbiguous", 
			"If set, reads which contain at least one ambiguous nucleotide will not be corrected. "
			"Default: " + tostring(CorrectionOptions{}.excludeAmbiguousReads),
		cxxopts::value<bool>()->implicit_value("true"))
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
		("mergedoutput", "extension results will not be split into _extended and _remaining", cxxopts::value<bool>()->implicit_value("true"))
		("allowOutwardExtension", "Will try to fill the gap and extend to the outside"
			"Default: " + tostring(ExtensionOptions{}.allowOutwardExtension), cxxopts::value<bool>()->implicit_value("true"))
		("sortedOutput", "Extended reads in output file will be sorted by read id."
			"Default: " + tostring(ExtensionOptions{}.sortedOutput), cxxopts::value<bool>()->implicit_value("true"))	
		("outputRemaining", "Output remaining reads which could not be extended. Will be sorted by read id."
			"Default: " + tostring(ExtensionOptions{}.outputRemainingReads), cxxopts::value<bool>()->implicit_value("true"))

		("hashloadfactor", "Load factor of hashtables. 0.0 < hashloadfactor < 1.0. Smaller values can improve the runtime at the expense of greater memory usage."
			"Default: " + std::to_string(MemoryOptions{}.hashtableLoadfactor), cxxopts::value<float>())

		("fixedStepsize", "fixedStepsize "
			"Default: " + tostring(ExtensionOptions{}.fixedStepsize),
		cxxopts::value<int>())
		("fixedStddev", "fixedStddev "
			"Default: " + tostring(ExtensionOptions{}.fixedStddev),
		cxxopts::value<int>())
		("qualityScoreBits", "How many bits should be used to store a single quality score. Allowed values: 1,2,8. If not 8, a lossy compression via binning is used."
			"Default: " + tostring(MemoryOptions{}.qualityScoreBits), cxxopts::value<int>())

		("fixedNumberOfReads", "Process only the first n reads. Default: " + tostring(RuntimeOptions{}.fixedNumberOfReads), cxxopts::value<std::size_t>())
	;

	//options.parse_positional({"deviceIds"});

	auto parseresults = options.parse(argc, argv);

	if(help) {
		std::cout << options.help({"", "Mandatory", "Additional"}) << std::endl;
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
	ExtensionOptions extensionOptions = args::to<ExtensionOptions>(parseresults);
	RuntimeOptions runtimeOptions = args::to<RuntimeOptions>(parseresults);
	MemoryOptions memoryOptions = args::to<MemoryOptions>(parseresults);
	FileOptions fileOptions = args::to<FileOptions>(parseresults);

	correctionOptions.batchsize = 16;

	if(!args::isValid(goodAlignmentProperties)) throw std::runtime_error("Invalid goodAlignmentProperties!");
	if(!args::isValid(correctionOptions)) throw std::runtime_error("Invalid correctionOptions!");
	if(!args::isValid(extensionOptions)) throw std::runtime_error("Invalid extensionOptions!");
	if(!args::isValid(runtimeOptions)) throw std::runtime_error("Invalid runtimeOptions!");
	if(!args::isValid(memoryOptions)) throw std::runtime_error("Invalid memoryOptions!");
	if(!args::isValid(fileOptions)) throw std::runtime_error("Invalid fileOptions!");

    runtimeOptions.canUseGpu = false;

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

	//print all options that will be used
	std::cout << std::boolalpha;
	std::cout << "CARE EXTEND CPU  will be started with the following parameters:\n";

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
	
	std::cout << "Exclude ambigious reads: " << correctionOptions.excludeAmbiguousReads << "\n";
	std::cout << "Use quality scores: " << correctionOptions.useQualityScores << "\n";
	std::cout << "Estimated dataset coverage: " << correctionOptions.estimatedCoverage << "\n";
	std::cout << "errorfactortuning: " << correctionOptions.estimatedErrorrate << "\n";
	std::cout << "coveragefactortuning: " << correctionOptions.m_coverage << "\n";

	std::cout << "Insert size: " << extensionOptions.insertSize << "\n";
	std::cout << "Insert size deviation: " << extensionOptions.insertSizeStddev << "\n";
	std::cout << "Allow extension outside of gap: " << extensionOptions.allowOutwardExtension << "\n";
	std::cout << "Sort extended reads: " << extensionOptions.sortedOutput << "\n";
	std::cout << "Output remaining reads: " << extensionOptions.outputRemainingReads << "\n";

	std::cout << "Threads: " << runtimeOptions.threads << "\n";
	std::cout << "Show progress bar: " << runtimeOptions.showProgress << "\n";

	std::cout << "Maximum memory for hash tables: " << memoryOptions.memoryForHashtables << "\n";
	std::cout << "Maximum memory total: " << memoryOptions.memoryTotalLimit << "\n";
	std::cout << "Hashtable load factor: " << memoryOptions.hashtableLoadfactor << "\n";
	std::cout << "Bits per quality score: " << memoryOptions.qualityScoreBits << "\n";

	std::cout << "Paired mode: " << to_string(fileOptions.pairType) << "\n";
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
	std::cout << "Extended reads output file: " << fileOptions.extendedReadsOutputfilename << "\n";
	std::cout << "Output file names: ";
	for(auto& s : fileOptions.outputfilenames){
		std::cout << s << ' ';
	}
	std::cout << "\n";
	std::cout << "Merged output: " << fileOptions.mergedoutput << "\n";
	std::cout << "fixedStddev: " << extensionOptions.fixedStddev << "\n";
	std::cout << "fixedStepsize: " << extensionOptions.fixedStepsize << "\n";
	std::cout << "----------------------------------------\n";
	std::cout << std::noboolalpha;

	if(fileOptions.pairType == SequencePairType::SingleEnd || fileOptions.pairType == SequencePairType::Invalid){
		std::cout << "Only paired-end extension is supported. Abort.\n";
		return 0;
	}


    const int numThreads = runtimeOptions.threads;

	omp_set_num_threads(numThreads);

	care::performExtension(
		correctionOptions,
		extensionOptions,
		runtimeOptions,
		memoryOptions,
		fileOptions,
		goodAlignmentProperties
	);

	return 0;
}
