#include "../include/care.hpp"

//#include "../include/args.hpp"
#include "../include/build.hpp"
#include "../include/correct.hpp"
#include "../include/minhasher.hpp"
#include <minhasher_transform.hpp>
#include "../include/options.hpp"

#include "../include/sequence.hpp"
#include <readstorage.hpp>
#include <config.hpp>

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



template<class StartCorrectionFunction>
void buildAndCorrect_cpu(const MinhashOptions& minhashOptions,
			const AlignmentOptions& alignmentOptions,
			const GoodAlignmentProperties& goodAlignmentProperties,
			const CorrectionOptions& correctionOptions,
			const RuntimeOptions& runtimeOptions,
			const FileOptions& fileOptions,
			std::vector<char>& readIsCorrectedVector,
			std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
			std::size_t nLocksForProcessedFlags,
			StartCorrectionFunction startCorrection){

    std::cout << "loading file and building data structures..." << std::endl;

	TIMERSTARTCPU(load_and_build);

    SequenceFileProperties sequenceFileProperties;

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

    cpu::ContiguousReadStorage readStorage = build_readstorage(fileOptions,
                                                              runtimeOptions,
                                                              correctionOptions.useQualityScores,
                                                              sequenceFileProperties.nReads,
                                                              sequenceFileProperties.maxSequenceLength);

	saveReadStorageToFile(readStorage, fileOptions);

    if(fileOptions.load_binary_reads_from != "") {
        auto stats = readStorage.getSequenceStatistics(runtimeOptions.threads);
        sequenceFileProperties.nReads = readStorage.getNumberOfSequences();
        sequenceFileProperties.maxSequenceLength = stats.maxSequenceLength;
        sequenceFileProperties.minSequenceLength = stats.minSequenceLength;
    }

    Minhasher minhasher = build_minhasher(fileOptions, runtimeOptions, sequenceFileProperties.nReads, minhashOptions, readStorage);//, minhasher);
    TIMERSTARTCPU(finalize_hashtables);
    transform_minhasher(minhasher, runtimeOptions.deviceIds);
    TIMERSTOPCPU(finalize_hashtables);
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
			std::vector<char>& readIsCorrectedVector,
			std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
			std::size_t nLocksForProcessedFlags){

	if(correctionOptions.correctionMode == CorrectionMode::Hamming) {
        auto func = [&](Minhasher& minhasher, cpu::ContiguousReadStorage& readStorage, SequenceFileProperties props){
				    cpu::correct_cpu(minhashOptions, alignmentOptions,
							    goodAlignmentProperties, correctionOptions,
							    runtimeOptions, fileOptions, props,
							    minhasher, readStorage,
							    readIsCorrectedVector, locksForProcessedFlags,
							    nLocksForProcessedFlags);
			    };

		buildAndCorrect_cpu(
					minhashOptions,
					alignmentOptions,
					goodAlignmentProperties,
					correctionOptions,
					runtimeOptions,
					fileOptions,
					readIsCorrectedVector,
					locksForProcessedFlags,
					nLocksForProcessedFlags,
					func);
	}else{
		//constexpr bool indels = true;

		std::cout << "Cannot correct indels with CPU version" << std::endl;
		return;
	}

}





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

		std::cout << "Running CARE CPU" << std::endl;

		selectCpuCorrection(minhashOptions, alignmentOptions,
					goodAlignmentProperties, correctionOptions,
					runtimeOptions, iterFileOptions,
					readIsCorrectedVector, locksForProcessedFlags,
					nLocksForProcessedFlags);

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
