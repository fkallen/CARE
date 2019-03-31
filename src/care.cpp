#include "../include/care.hpp"

//#include "../include/args.hpp"
#include "../include/build.hpp"
#include "../include/correct.hpp"
#include "../include/minhasher.hpp"
#include "../include/options.hpp"
#include "../include/readstorage.hpp"
#include "../include/sequence.hpp"

#include "../include/config.hpp"

#include <vector>
#include <iostream>
#include <mutex>
#include <thread>

#include <experimental/filesystem>

namespace filesys = std::experimental::filesystem;


#ifdef __NVCC__
#include "../include/gpu/correct.hpp"
#include "../include/gpu/readstorage.hpp"
#endif

namespace care {



template<class minhasher_t,
         class readStorage_t>
void printDataStructureMemoryUsage(const minhasher_t& minhasher, const readStorage_t& readStorage){
	auto toGB = [](std::size_t bytes){
			    double gb = bytes / 1024. / 1024. / 1024.0;
			    return gb;
		    };

	std::cout << "reads take up " << toGB(readStorage.size()) << " GB." << std::endl;
	std::cout << "hash maps take up " << toGB(minhasher.numBytes()) << " GB." << std::endl;
}

void printFileProperties(const std::string& filename, const SequenceFileProperties& props){
	std::cout << "----------------------------------------" << std::endl;
	std::cout << "File: " << filename << std::endl;
	std::cout << "Reads: " << props.nReads << std::endl;
	std::cout << "Minimum sequence length: " << props.minSequenceLength << std::endl;
	std::cout << "Maximum sequence length: " << props.maxSequenceLength << std::endl;
	std::cout << "----------------------------------------" << std::endl;
}

template<class readStorage_t>
void saveReadStorageToFile(const readStorage_t& readStorage, const FileOptions& fileOptions){
	if(fileOptions.save_binary_reads_to != "") {
		readStorage.saveToFile(fileOptions.save_binary_reads_to);
		std::cout << "Saved binary reads to file " << fileOptions.save_binary_reads_to << std::endl;
	}
}

template<class minhasher_t>
void saveMinhasherToFile(const minhasher_t& minhasher, const FileOptions& fileOptions){
	if(fileOptions.save_hashtables_to != "") {
		minhasher.saveToFile(fileOptions.save_hashtables_to);
		std::cout << "Saved hash tables to file " << fileOptions.save_hashtables_to << std::endl;
	}
}

template<class minhasher_t,
         class readStorage_t>
void saveDataStructuresToFile(const minhasher_t& minhasher, const readStorage_t& readStorage, const FileOptions& fileOptions){
	saveReadStorageToFile(readStorage, fileOptions);
	saveMinhasherToFile(minhasher, fileOptions);
}



template<class minhasher_t,
         class readStorage_t,
         bool indels,
         class StartCorrectionFunction>
void buildAndCorrect_cpu(const MinhashOptions& minhashOptions,
			const AlignmentOptions& alignmentOptions,
			const GoodAlignmentProperties& goodAlignmentProperties,
			const CorrectionOptions& correctionOptions,
			const RuntimeOptions& runtimeOptions,
			const FileOptions& fileOptions,
			SequenceFileProperties& sequenceFileProperties,
			std::vector<char>& readIsCorrectedVector,
			std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
			std::size_t nLocksForProcessedFlags,
			StartCorrectionFunction startCorrection){

	//constexpr bool indelAlignment = indels;

	using Minhasher_t = minhasher_t;
	using ReadStorage_t = readStorage_t;
	using Sequence_t = typename ReadStorage_t::Sequence_t;

	std::cout << "Sequence type: " << getSequenceType<Sequence_t>() << std::endl;

	Minhasher_t minhasher(minhashOptions);
	ReadStorage_t readStorage(sequenceFileProperties.nReads, correctionOptions.useQualityScores, sequenceFileProperties.maxSequenceLength);

	std::cout << "loading file and building data structures..." << std::endl;

	TIMERSTARTCPU(load_and_build);
	sequenceFileProperties = build_readstorage(fileOptions, runtimeOptions, readStorage);
	saveReadStorageToFile(readStorage, fileOptions);
	build_minhasher(fileOptions, runtimeOptions, sequenceFileProperties.nReads, readStorage, minhasher);
	saveMinhasherToFile(minhasher, fileOptions);
	TIMERSTOPCPU(load_and_build);

	printFileProperties(fileOptions.inputfile, sequenceFileProperties);

//		saveDataStructuresToFile(minhasher, readStorage, fileOptions);

	readIsCorrectedVector.resize(sequenceFileProperties.nReads, 0);

	printDataStructureMemoryUsage(minhasher, readStorage);

	startCorrection(minhasher, readStorage, sequenceFileProperties);

	/*correct<Minhasher_t,
	                ReadStorage_t,
	                indelAlignment>(minhashOptions, alignmentOptions,
	                                                goodAlignmentProperties, correctionOptions,
	                                                runtimeOptions, fileOptions, props,
	                                                minhasher, readStorage,
	                                                readIsCorrectedVector, locksForProcessedFlags,
	                                                nLocksForProcessedFlags, runtimeOptions.deviceIds);*/

}

void selectCpuCorrection(
			const MinhashOptions& minhashOptions,
			const AlignmentOptions& alignmentOptions,
			const GoodAlignmentProperties& goodAlignmentProperties,
			const CorrectionOptions& correctionOptions,
			const RuntimeOptions& runtimeOptions,
			const FileOptions& fileOptions,
			SequenceFileProperties& sequenceFileProperties,
			std::vector<char>& readIsCorrectedVector,
			std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
			std::size_t nLocksForProcessedFlags){


    using Minhasher_t = Minhasher<kmer_type, read_number>;

	using NoIndelSequence_t = Sequence2BitHiLo;
	//using IndelSequence_t = Sequence2BitHiLo;
    using NoIndelReadStorage_t = cpu::ContiguousReadStorage<NoIndelSequence_t, read_number>;
    //using IndelReadStorage_t = ReadStorageMinMemory<IndelSequence_t, read_number>;

	if(correctionOptions.correctionMode == CorrectionMode::Hamming) {
		constexpr bool indels = false;

		auto func = [&](Minhasher_t& minhasher, NoIndelReadStorage_t& readStorage, SequenceFileProperties props){
				    cpu::correct_cpu<Minhasher_t,
				                     NoIndelReadStorage_t,
				                     indels>(minhashOptions, alignmentOptions,
							    goodAlignmentProperties, correctionOptions,
							    runtimeOptions, fileOptions, props,
							    minhasher, readStorage,
							    readIsCorrectedVector, locksForProcessedFlags,
							    nLocksForProcessedFlags);
			    };

		buildAndCorrect_cpu<Minhasher_t,
		                    NoIndelReadStorage_t,
		                    indels>
		(
					minhashOptions,
					alignmentOptions,
					goodAlignmentProperties,
					correctionOptions,
					runtimeOptions,
					fileOptions,
					sequenceFileProperties,
					readIsCorrectedVector,
					locksForProcessedFlags,
					nLocksForProcessedFlags,
					func
		);
	}else{
		//constexpr bool indels = true;

		std::cout << "Cannot correct indels with CPU version" << std::endl;
		return;
	}

}


#ifdef __NVCC__

template<class minhasher_t,
         class readStorage_t,
         bool indels,
         class StartCorrectionFunction>
void buildAndCorrect_gpu(const MinhashOptions& minhashOptions,
			const AlignmentOptions& alignmentOptions,
			const GoodAlignmentProperties& goodAlignmentProperties,
			const CorrectionOptions& correctionOptions,
			const RuntimeOptions& runtimeOptions,
			const FileOptions& fileOptions,
			SequenceFileProperties& sequenceFileProperties,
			std::vector<char>& readIsCorrectedVector,
			std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
			std::size_t nLocksForProcessedFlags,
			StartCorrectionFunction startCorrection){

	constexpr bool indelAlignment = indels;

	using Minhasher_t = minhasher_t;
	using ReadStorage_t = readStorage_t;
	using Sequence_t = typename ReadStorage_t::Sequence_t;

	std::cout << "Sequence type: " << getSequenceType<Sequence_t>() << std::endl;

	Minhasher_t minhasher(minhashOptions, runtimeOptions.deviceIds);
	ReadStorage_t readStorage(sequenceFileProperties.nReads,
	                          correctionOptions.useQualityScores,
	                          sequenceFileProperties.maxSequenceLength,
	                          runtimeOptions.deviceIds);

	std::cout << "loading file and building data structures..." << std::endl;

	TIMERSTARTCPU(load_and_build);
	sequenceFileProperties = build_readstorage(fileOptions, runtimeOptions, readStorage);
	saveReadStorageToFile(readStorage, fileOptions);
	build_minhasher(fileOptions, runtimeOptions, sequenceFileProperties.nReads, readStorage, minhasher);
	saveMinhasherToFile(minhasher, fileOptions);
	TIMERSTOPCPU(load_and_build);

	printFileProperties(fileOptions.inputfile, sequenceFileProperties);

//		saveDataStructuresToFile(minhasher, readStorage, fileOptions);

	readIsCorrectedVector.resize(sequenceFileProperties.nReads, 0);

	printDataStructureMemoryUsage(minhasher, readStorage);

	startCorrection(minhasher, readStorage, sequenceFileProperties);

	/*correct<Minhasher_t,
	                ReadStorage_t,
	                indelAlignment>(minhashOptions, alignmentOptions,
	                                                goodAlignmentProperties, correctionOptions,
	                                                runtimeOptions, fileOptions, props,
	                                                minhasher, readStorage,
	                                                readIsCorrectedVector, locksForProcessedFlags,
	                                                nLocksForProcessedFlags, runtimeOptions.deviceIds);*/

}

void selectGpuCorrection(
			const MinhashOptions& minhashOptions,
			const AlignmentOptions& alignmentOptions,
			const GoodAlignmentProperties& goodAlignmentProperties,
			const CorrectionOptions& correctionOptions,
			const RuntimeOptions& runtimeOptions,
			const FileOptions& fileOptions,
			SequenceFileProperties& sequenceFileProperties,
			std::vector<char>& readIsCorrectedVector,
			std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
			std::size_t nLocksForProcessedFlags){

    using Minhasher_t = Minhasher<kmer_type, read_number>;

	using NoIndelSequence_t = Sequence2BitHiLo;
	//using IndelSequence_t = Sequence2BitHiLo;
    using NoIndelReadStorage_t = gpu::ContiguousReadStorage<NoIndelSequence_t, read_number>;
    //using IndelReadStorage_t = ReadStorageMinMemory<IndelSequence_t, read_number>;

	if(correctionOptions.correctionMode == CorrectionMode::Hamming) {
		constexpr bool indels = false;

		auto func = [&](Minhasher_t& minhasher, NoIndelReadStorage_t& readStorage, SequenceFileProperties props){
				    //using Minhasher_t = decltype(minhasher);
				    //using ReadStorage_t = decltype(readStorage);
				    gpu::correct_gpu<Minhasher_t,
				                     NoIndelReadStorage_t,
				                     indels>(minhashOptions, alignmentOptions,
							    goodAlignmentProperties, correctionOptions,
							    runtimeOptions, fileOptions, props,
							    minhasher, readStorage,
							    readIsCorrectedVector, locksForProcessedFlags,
							    nLocksForProcessedFlags);
			    };

		buildAndCorrect_gpu<Minhasher_t, NoIndelReadStorage_t, indels>
		(
					minhashOptions,
					alignmentOptions,
					goodAlignmentProperties,
					correctionOptions,
					runtimeOptions,
					fileOptions,
					sequenceFileProperties,
					readIsCorrectedVector,
					locksForProcessedFlags,
					nLocksForProcessedFlags,
					func
		);
	}else{
		//constexpr bool indels = true;

		std::cout << "Cannot correct indels with GPU version" << std::endl;
		return;
	}

}
#endif




void performCorrection(MinhashOptions minhashOptions,
			AlignmentOptions alignmentOptions,
			CorrectionOptions correctionOptions,
			RuntimeOptions runtimeOptions,
			FileOptions fileOptions,
			GoodAlignmentProperties goodAlignmentProperties){
#if 0
	void performCorrection(const cxxopts::ParseResult& args) {
		//check arguments
		/*if(!args::areValid(args)){
		        throw std::runtime_error("care::performCorrection: Invalid arguments!");
		   }*/

		//parse options from arguments
		MinhashOptions minhashOptions = args::to<MinhashOptions>(args);
		AlignmentOptions alignmentOptions = args::to<AlignmentOptions>(args);
		GoodAlignmentProperties goodAlignmentProperties = args::to<GoodAlignmentProperties>(args);
		CorrectionOptions correctionOptions = args::to<CorrectionOptions>(args);
		RuntimeOptions runtimeOptions = args::to<RuntimeOptions>(args);
		FileOptions fileOptions = args::to<FileOptions>(args);

		{
			if(!args::isValid(minhashOptions)) throw std::runtime_error("care::performCorrection: Invalid minhashOptions!");
			if(!args::isValid(alignmentOptions)) throw std::runtime_error("care::performCorrection: Invalid alignmentOptions!");
			if(!args::isValid(goodAlignmentProperties)) throw std::runtime_error("care::performCorrection: Invalid goodAlignmentProperties!");
			if(!args::isValid(correctionOptions)) throw std::runtime_error("care::performCorrection: Invalid correctionOptions!");
			if(!args::isValid(runtimeOptions)) throw std::runtime_error("care::performCorrection: Invalid runtimeOptions!");
			if(!args::isValid(fileOptions)) throw std::runtime_error("care::performCorrection: Invalid fileOptions!");
		}

		if(correctionOptions.correctCandidates && correctionOptions.extractFeatures) {
			std::cout << "Warning! correctCandidates=true cannot be used with extractFeatures=true. Using correctCandidates=false" << std::endl;
			correctionOptions.correctCandidates = false;
		}
#endif
	//create output directory
	filesys::create_directories(fileOptions.outputdirectory);

	SequenceFileProperties sequenceFileProperties;
	sequenceFileProperties.maxSequenceLength = 0;
	sequenceFileProperties.minSequenceLength = 0;
	sequenceFileProperties.nReads = 0;

	if(fileOptions.load_binary_reads_from == "") {
		if(fileOptions.nReads == 0 || fileOptions.maximum_sequence_length == 0) {
			std::cout << "Scanning file to get number of reads and maximum sequence length." << std::endl;
			sequenceFileProperties = getSequenceFileProperties(fileOptions.inputfile, fileOptions.format);
		}else{
			sequenceFileProperties.maxSequenceLength = fileOptions.maximum_sequence_length;
			sequenceFileProperties.minSequenceLength = 0;
			sequenceFileProperties.nReads = fileOptions.nReads;
		}
	}

	std::cerr << "sequenceFileProperties.maxSequenceLength = " << sequenceFileProperties.maxSequenceLength << '\n';
	std::cerr << "sequenceFileProperties.minSequenceLength = " << sequenceFileProperties.minSequenceLength << '\n';
	std::cerr << "sequenceFileProperties.nReads = " << sequenceFileProperties.nReads << '\n';

	std::vector<char> readIsCorrectedVector;
	std::size_t nLocksForProcessedFlags = runtimeOptions.nCorrectorThreads * 1000;
	std::unique_ptr<std::mutex[]> locksForProcessedFlags(new std::mutex[nLocksForProcessedFlags]);

	const int iters = 1;
	int iter = 0;

	auto thread_id = std::this_thread::get_id();
	std::string thread_id_string;
	{
		std::stringstream ss;
		ss << thread_id;
		thread_id_string = ss.str();
	}

	#define DO_ALTERNATE

	// correct file in multiple passes
	do {
		FileOptions iterFileOptions = fileOptions;

	#ifdef DO_ALTERNATE
		//alternate between two output files
		// on even iteration, correct file _iter_odd and save to _iter_even
		// on odd iteration, correct file _iter_even and save to _iter_odd
		if(iter == 0) {
			//inputfile remains original input file
			iterFileOptions.outputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even";
		}else{
			if(iter % 2 == 0) {
				iterFileOptions.inputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_odd";
				iterFileOptions.outputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even";
			}else{
				iterFileOptions.inputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even";
				iterFileOptions.outputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_odd";
			}
		}

	#else
		if(iter == 0) {
			//inputfile remains original input file
			iterFileOptions.outputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_0";
		}else{
			iterFileOptions.inputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_" + std::to_string(iter-1);
			iterFileOptions.outputfile = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_" + std::to_string(iter);
		}
	#endif

		#ifndef __NVCC__
		std::cout << "Running CARE CPU" << std::endl;

		selectCpuCorrection(minhashOptions, alignmentOptions,
					goodAlignmentProperties, correctionOptions,
					runtimeOptions, iterFileOptions,
					sequenceFileProperties,
					readIsCorrectedVector, locksForProcessedFlags,
					nLocksForProcessedFlags);
		#else
		int nDevices;

		cudaGetDeviceCount(&nDevices); CUERR;

		std::vector<int> invalidIds;

		for(int id : runtimeOptions.deviceIds) {
			if(id >= nDevices) {
				invalidIds.emplace_back(id);
				std::cout << "Found invalid device Id: " << id << std::endl;
			}
		}

		if(invalidIds.size() > 0) {
			std::cout << "Available GPUs on your machine:" << std::endl;
			for(int j = 0; j < nDevices; j++) {
				cudaDeviceProp prop;
				cudaGetDeviceProperties(&prop, j); CUERR;
				std::cout << "Id " << j << " : " << prop.name << std::endl;
			}

			for(int invalidid : invalidIds) {
				runtimeOptions.deviceIds.erase(std::find(runtimeOptions.deviceIds.begin(), runtimeOptions.deviceIds.end(), invalidid));
			}
		}

		runtimeOptions.canUseGpu = runtimeOptions.deviceIds.size() > 0;

		if(runtimeOptions.canUseGpu && runtimeOptions.deviceIds.size() > 0 && runtimeOptions.threadsForGPUs > 0) {
			std::cout << "Running CARE GPU" << std::endl;
			std::cout << "Can use the following GPU device Ids: ";

			for(int i : runtimeOptions.deviceIds)
				std::cout << i << " ";

			std::cout << std::endl;

			selectGpuCorrection(minhashOptions, alignmentOptions,
						goodAlignmentProperties, correctionOptions,
						runtimeOptions, iterFileOptions,
						sequenceFileProperties,
						readIsCorrectedVector, locksForProcessedFlags,
						nLocksForProcessedFlags);
		}else{
			std::cout << "Running CARE CPU" << std::endl;

			selectCpuCorrection(minhashOptions, alignmentOptions,
						goodAlignmentProperties, correctionOptions,
						runtimeOptions, iterFileOptions,
						sequenceFileProperties,
						readIsCorrectedVector, locksForProcessedFlags,
						nLocksForProcessedFlags);
		}
		#endif

		iter++;

	} while(iter < iters);


	//rename final result to requested output file name and delete intermediate files
	bool keepIntermediateResults = false;

    TIMERSTARTCPU(finalizing_files);

	#ifdef DO_ALTERNATE
	if(iters % 2 == 0) {
		std::string toRename = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_odd";
		std::rename(toRename.c_str(), fileOptions.outputfile.c_str());

        //rename feature file
        if(correctionOptions.extractFeatures){
            std::string tmpfeaturename = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_odd_features";
            std::string outputfeaturename = fileOptions.outputfile + "_features";

    		std::rename(tmpfeaturename.c_str(), outputfeaturename.c_str());
        }

		if(!keepIntermediateResults && iters > 1)
			deleteFiles({fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even"});
	}else{
		std::string toRename = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even";
		std::rename(toRename.c_str(), fileOptions.outputfile.c_str());

        if(correctionOptions.extractFeatures){
            std::string tmpfeaturename = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_even_features";
            std::string outputfeaturename = fileOptions.outputfile + "_features";

    		std::rename(tmpfeaturename.c_str(), outputfeaturename.c_str());
        }

		if(!keepIntermediateResults && iters > 1)
			deleteFiles({fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_odd"});
	}
	#else
	std::string toRename = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_" + std::to_string(iters-1);
	std::rename(toRename.c_str(), fileOptions.outputfile.c_str());

    if(correctionOptions.extractFeatures){
        std::string tmpfeaturename = fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_" + std::to_string(iters-1) + "_features";
        std::string outputfeaturename = fileOptions.outputfile + "_features";

        std::rename(tmpfeaturename.c_str(), outputfeaturename.c_str());
    }

	if(!keepIntermediateResults) {
		std::vector<std::string> filestodelete;
		for(int i = 0; i < iters-1; i++)
			filestodelete.push_back(fileOptions.outputdirectory + "/" + thread_id_string + "_" + fileOptions.outputfilename + "_iter_" + std::to_string(i));
		deleteFiles(filestodelete);
	}
	#endif

    TIMERSTOPCPU(finalizing_files);
}


}
