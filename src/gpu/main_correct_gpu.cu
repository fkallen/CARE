#include <config.hpp>
#include <options.hpp>
#include <readlibraryio.hpp>
#include <version.hpp>
#include <gpu/cudaerrorcheck.cuh>
#include <gpu/dispatch_care_correct_gpu.cuh>

#include <cxxopts/cxxopts.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <omp.h>


using namespace care;

bool checkMandatoryArguments(const cxxopts::ParseResult& parseresults){

	const std::vector<std::string> mandatory = {
		"inputfiles", "outdir", "outputfilenames", "coverage", "pairmode"
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


std::vector<int> getAllDeviceIds(){
	int numGpus = 0;
	CUDACHECK(cudaGetDeviceCount(&numGpus));
	std::vector<int> ids(numGpus);
	std::iota(ids.begin(), ids.end(), 0);
	return ids;
}

bool deviceIsSuitable(int deviceId){
	cudaDeviceProp prop;
	CUDACHECK(cudaGetDeviceProperties(&prop, deviceId));

	if(prop.major < 6) return false;
	if(prop.managedMemory != 1) return false;
	if(prop.unifiedAddressing != 1) return false;
	return true;
}

bool devicesHavePairwisePeerAccess(const std::vector<int>& deviceIds){
	const int numGpus = deviceIds.size();
	for(int i = 0; i < numGpus; i++){
		for(int k = 0; k < numGpus; k++){
			if(i != k){
				const int device = deviceIds[i];
				const int peerDevice = deviceIds[k];
				int canAccessPeer = 0;
				CUDACHECK(cudaDeviceCanAccessPeer(&canAccessPeer,  device, peerDevice));
				if(canAccessPeer == 0) return false;
			}
		}
	}
	return true;
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

    programOptions.deviceIds = getAllDeviceIds();
	programOptions.deviceIds.erase(
		std::remove_if(programOptions.deviceIds.begin(), programOptions.deviceIds.end(), [](int id){ return !deviceIsSuitable(id); }), 
		programOptions.deviceIds.end()
	);
    

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


	if(programOptions.deviceIds.empty()){
		std::cout << "All available GPUs are not suitable. Abort\n";
		std::exit(0);
	}else{
		std::cout << "List of detected suitable device(s): [";
		for(auto x : programOptions.deviceIds){
			std::cout << " " << x;
		}
		std::cout << " ]\n";

		const bool peerOk = devicesHavePairwisePeerAccess(programOptions.deviceIds);
		if(!peerOk){
			std::cout << "The suitable available GPUs do not allow pair-wise peer access. Abort\n";
			std::exit(0);
		}
		
	}

    const int numThreads = programOptions.threads;

	omp_set_num_threads(numThreads);

    care::performCorrection(
		programOptions
	);

	return 0;
}
