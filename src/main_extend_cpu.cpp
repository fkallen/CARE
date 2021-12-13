#include <version.hpp>
#include <config.hpp>

#include <cxxopts/cxxopts.hpp>
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
		("allowOutwardExtension", "Will try to fill the gap and extend to the outside"
			"Default: " + tostring(ProgramOptions{}.allowOutwardExtension), cxxopts::value<bool>()->implicit_value("true"))
		("sortedOutput", "Extended reads in output file will be sorted by read id."
			"Default: " + tostring(ProgramOptions{}.sortedOutput), cxxopts::value<bool>()->implicit_value("true"))	
		("outputRemaining", "Output remaining reads which could not be extended. Will be sorted by read id."
			"Default: " + tostring(ProgramOptions{}.outputRemainingReads), cxxopts::value<bool>()->implicit_value("true"))

		("hashloadfactor", "Load factor of hashtables. 0.0 < hashloadfactor < 1.0. Smaller values can improve the runtime at the expense of greater memory usage."
			"Default: " + std::to_string(ProgramOptions{}.hashtableLoadfactor), cxxopts::value<float>())

		("fixedStepsize", "fixedStepsize "
			"Default: " + tostring(ProgramOptions{}.fixedStepsize),
		cxxopts::value<int>())
		("fixedStddev", "fixedStddev "
			"Default: " + tostring(ProgramOptions{}.fixedStddev),
		cxxopts::value<int>())
		("qualityScoreBits", "How many bits should be used to store a single quality score. Allowed values: 1,2,8. If not 8, a lossy compression via binning is used."
			"Default: " + tostring(ProgramOptions{}.qualityScoreBits), cxxopts::value<int>())

		("fixedNumberOfReads", "Process only the first n reads. Default: " + tostring(ProgramOptions{}.fixedNumberOfReads), cxxopts::value<std::size_t>())
		("singlehash", "Use 1 hashtables with h smallest unique hashes. Default: " + tostring(ProgramOptions{}.singlehash), cxxopts::value<bool>())
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

	ProgramOptions programOptions = makeProgramOptions(parseresults);

	programOptions.batchsize = 16;

	if(!isValid(programOptions)) throw std::runtime_error("Invalid program options!");

    programOptions.canUseGpu = false;

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

	if(programOptions.pairType == SequencePairType::SingleEnd){
        throw std::runtime_error("single end extension not possible");
    }

	//print all options that will be used
	std::cout << std::boolalpha;
	std::cout << "CARE EXTEND CPU  will be started with the following parameters:\n";

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
	
	std::cout << "Exclude ambigious reads: " << programOptions.excludeAmbiguousReads << "\n";
	std::cout << "Use quality scores: " << programOptions.useQualityScores << "\n";
	std::cout << "Estimated dataset coverage: " << programOptions.estimatedCoverage << "\n";
	std::cout << "errorfactortuning: " << programOptions.estimatedErrorrate << "\n";
	std::cout << "coveragefactortuning: " << programOptions.m_coverage << "\n";

	std::cout << "Insert size: " << programOptions.insertSize << "\n";
	std::cout << "Insert size deviation: " << programOptions.insertSizeStddev << "\n";
	std::cout << "Allow extension outside of gap: " << programOptions.allowOutwardExtension << "\n";
	std::cout << "Sort extended reads: " << programOptions.sortedOutput << "\n";
	std::cout << "Output remaining reads: " << programOptions.outputRemainingReads << "\n";

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
	std::cout << "Extended reads output file: " << programOptions.extendedReadsOutputfilename << "\n";
	std::cout << "Output file names: ";
	for(auto& s : programOptions.outputfilenames){
		std::cout << s << ' ';
	}
	std::cout << "\n";
	std::cout << "fixedStddev: " << programOptions.fixedStddev << "\n";
	std::cout << "fixedStepsize: " << programOptions.fixedStepsize << "\n";
	std::cout << "Allow outward extension: " << programOptions.allowOutwardExtension << "\n";
	std::cout << "Sorted output: " << programOptions.sortedOutput << "\n";
	std::cout << "Output remaining reads: " << programOptions.outputRemainingReads << "\n";
	std::cout << "----------------------------------------\n";
	std::cout << std::noboolalpha;

	if(programOptions.pairType == SequencePairType::SingleEnd || programOptions.pairType == SequencePairType::Invalid){
		std::cout << "Only paired-end extension is supported. Abort.\n";
		return 0;
	}


    const int numThreads = programOptions.threads;

	omp_set_num_threads(numThreads);

	care::performExtension(
		programOptions
	);

	return 0;
}
