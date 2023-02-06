#include <version.hpp>
#include <config.hpp>

#include <cxxopts/cxxopts.hpp>
#include <options.hpp>
#include <gpu/dispatch_care_correct_gpu.cuh>

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
		"gpu", "pairmode"
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


void listAvailableGpus(std::ostream& os){
	int numGpus = 0;
    cudaError_t status = cudaGetDeviceCount(&numGpus);
	if(status == cudaSuccess){
		os << "Found " << numGpus << " CUDA-capable GPUs in your system\n";
		for(int deviceId = 0; deviceId < numGpus; deviceId++){
			os << "Device ID " << deviceId << ": ";

			cudaDeviceProp prop;
			status = cudaGetDeviceProperties(&prop, deviceId);
			if(status == cudaSuccess){
				os << prop.name << ", compute capability " << prop.major << "." << prop.minor << "\n";
			}else{
				os << "Error: Could not retrieve device properties for device ID " << deviceId << "\n";
			}
		}
	}else{
		os << "Error: Could not determine number of GPUs\n";
	}
}

int main(int argc, char** argv){

	bool help = false;
	bool showVersion = false;
	bool printAvailableGpus = false;

	cxxopts::Options commandLineOptions(argv[0], "CARE: Context-Aware Read Error Correction for Illumina reads");

	addMandatoryOptions(commandLineOptions);
	addMandatoryOptionsCorrectGpu(commandLineOptions);

	addAdditionalOptions(commandLineOptions);
	addAdditionalOptionsCorrectGpu(commandLineOptions);

	commandLineOptions.add_options("Additional")
		("help", "Show this help message", cxxopts::value<bool>(help))
		("version", "Print version", cxxopts::value<bool>(showVersion))
		("listGpus", "Print a list of available GPUs in the system and their corresponding device ID", cxxopts::value<bool>(printAvailableGpus));

	auto parseresults = commandLineOptions.parse(argc, argv);

	if(showVersion){
		std::cout << "CARE version " << CARE_VERSION_STRING << std::endl;
		std::exit(0);
	}

	if(help) {
		std::cout << commandLineOptions.help({"", "Mandatory", "Additional"}) << std::endl;
		std::exit(0);
	}

	if(printAvailableGpus){
		listAvailableGpus(std::cout);
		std::exit(0);
	}

	const bool mandatoryPresent = checkMandatoryArguments(parseresults);
	if(!mandatoryPresent){
		std::cout << commandLineOptions.help({"Mandatory"}) << std::endl;
		std::exit(0);
	}

	ProgramOptions programOptions(parseresults);

	if(!programOptions.isValid()) throw std::runtime_error("Invalid program options!");

    programOptions.deviceIds = correction::getUsableDeviceIds(programOptions.deviceIds);
    programOptions.canUseGpu = programOptions.deviceIds.size() > 0;

	if(programOptions.useQualityScores){
		const bool hasQ = std::all_of(
			programOptions.inputfiles.begin(),
			programOptions.inputfiles.end(),
			[](const auto& s){
				return hasQualityScores(s);
			}
		);

		if(!hasQ){
			std::cout << "Quality scores have been disabled because there exist reads in an input file without quality scores.\n";
			
			programOptions.useQualityScores = false;
		}
	}

	if(programOptions.correctionType != CorrectionType::Classic){
		//disable print mode in gpu version
		if(programOptions.correctionType != CorrectionType::Forest){
			std::cout << "CorrectionType is invalid. Abort!\n";
			std::exit(0);
		}
		if(programOptions.mlForestfileAnchor == ""){
			std::cout << "CorrectionType is not set to Classic, but no valid classifier file is provided. Abort!\n";
			std::exit(0);
		}

		if(programOptions.mlForestfileCands == ""){
			programOptions.mlForestfileCands = programOptions.mlForestfileAnchor;
		}
	}

	if(programOptions.correctionTypeCands != CorrectionType::Classic){
		//disable print mode in gpu version
		if(programOptions.correctionTypeCands != CorrectionType::Forest){
			std::cout << "CorrectionTypeCands is invalid. Abort!\n";
			std::exit(0);
		}
	}

	std::cout << std::boolalpha;
	std::cout << "CARE CORRECT GPU  will be started with the following parameters:\n";
	std::cout << "----------------------------------------\n";

	programOptions.printMandatoryOptions(std::cout);
	programOptions.printMandatoryOptionsCorrectGpu(std::cout);

	programOptions.printAdditionalOptions(std::cout);
	programOptions.printAdditionalOptionsCorrectGpu(std::cout);

	std::cout << "----------------------------------------\n";
	std::cout << std::noboolalpha;


	if(!programOptions.canUseGpu){
		std::cout << "No valid GPUs selected. Abort\n";
		std::exit(0);
	}

    const int numThreads = programOptions.threads;

	omp_set_num_threads(numThreads);

    care::performCorrection(
		programOptions
	);

	return 0;
}
