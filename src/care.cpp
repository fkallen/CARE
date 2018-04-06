#include "../inc/care.hpp"

#include "../inc/args.hpp"
#include "../inc/build.hpp"
#include "../inc/correct.hpp"
#include "../inc/minhasher.hpp"
#include "../inc/options.hpp"
#include "../inc/readstorage.hpp"
#include "../inc/types.hpp"

#include <vector>
#include <iostream>
#include <mutex>

#include <experimental/filesystem>

namespace filesys = std::experimental::filesystem;

namespace care{

/*
    Correct fileOptions.inputfile and save result to fileOptions.outputfile
*/
void correctFile(const MinhashOptions& minhashOptions,
				  const AlignmentOptions& alignmentOptions,
				  const GoodAlignmentProperties& goodAlignmentProperties,
				  const CorrectionOptions& correctionOptions,
				  const RuntimeOptions& runtimeOptions,
				  const FileOptions& fileOptions,
				  std::vector<char>& readIsCorrectedVector,
				  std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
				  size_t nLocksForProcessedFlags,
				  const std::vector<int>& deviceIds){


    Minhasher minhasher(minhashOptions);
    ReadStorage readStorage;
    readStorage.setUseQualityScores(correctionOptions.useQualityScores);

    std::cout << "begin build" << std::endl;

	TIMERSTARTCPU(LOAD_FILE);
    build(fileOptions, readStorage, minhasher, runtimeOptions.nInserterThreads);
	TIMERSTOPCPU(LOAD_FILE);

    TIMERSTARTCPU(PREPROCESSING);
	minhasher.transform();
	readStorage.noMoreInserts();
	TIMERSTOPCPU(PREPROCESSING);

    std::cout << "begin correct" << std::endl;

	TIMERSTARTCPU(CORRECT);

    correct(minhashOptions, alignmentOptions,
        goodAlignmentProperties, correctionOptions,
        runtimeOptions, fileOptions,
        minhasher, readStorage,
        readIsCorrectedVector, locksForProcessedFlags,
        nLocksForProcessedFlags, deviceIds);


	TIMERSTOPCPU(CORRECT);

    if (correctionOptions.correctCandidates) {
		int asd = std::count_if(readIsCorrectedVector.begin(),
				readIsCorrectedVector.end(), [](auto b) {return b;});
		std::cout << "total corrected reads: " << asd << std::endl;
	}
}

void performCorrection(const cxxopts::ParseResult& args) {
	//check arguments
    if(!args::areValid(args)){
        throw std::runtime_error("care::performCorrection: Invalid arguments!");
    }

	//parse options from arguments
	MinhashOptions minhashOptions = args::to<MinhashOptions>(args);
	AlignmentOptions alignmentOptions = args::to<AlignmentOptions>(args);
	GoodAlignmentProperties goodAlignmentProperties = args::to<GoodAlignmentProperties>(args);
    CorrectionOptions correctionOptions = args::to<CorrectionOptions>(args);
	RuntimeOptions runtimeOptions = args::to<RuntimeOptions>(args);
	FileOptions fileOptions = args::to<FileOptions>(args);

	//create output directory
	filesys::create_directories(fileOptions.outputdirectory);

    SequenceFileProperties props = getSequenceFileProperties(fileOptions.inputfile, fileOptions.format);

    if(props.nReads > std::numeric_limits<ReadId_t>::max()){
        throw std::runtime_error("Found " + std::to_string(props.nReads)
                                + " reads, but CARE is compiled for only "
                                + std::to_string(std::numeric_limits<ReadId_t>::max())
                                + " reads.");
    }

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "File: " << fileOptions.inputfile << std::endl;
    std::cout << "Reads: " << props.nReads << std::endl;
    std::cout << "Minimum sequence length: " << props.minSequenceLength << std::endl;
    std::cout << "Maximum sequence length: " << props.maxSequenceLength << std::endl;
    std::cout << "----------------------------------------" << std::endl;

	std::vector<char> readIsCorrectedVector(props.nReads, 0);
	size_t nLocksForProcessedFlags = correctionOptions.batchsize * runtimeOptions.nCorrectorThreads * 1000;
	std::unique_ptr<std::mutex[]> locksForProcessedFlags(new std::mutex[nLocksForProcessedFlags]);

	std::vector<int> deviceIds;

#ifdef __CUDACC__

	int nGpus;
	cudaGetDeviceCount(&nGpus); CUERR;
    //TODO instead of failing, fall back to CPU mode by introducing variable canUseGpu and setting it to false
	if(nGpus == 0)
        throw std::runtime_error("No CUDA capable device found!");
	for(int i = 0; i < nGpus; i++)
	   deviceIds.push_back(i);

#endif

	const int iters = 1;
	int iter = 0;

#define DO_ALTERNATE

	// correct file in multiple passes
	do{
		FileOptions iterFileOptions = fileOptions;

#ifdef DO_ALTERNATE
		//alternate between two output files
		// on even iteration, correct file _iter_odd and save to _iter_even
		// on odd iteration, correct file _iter_even and save to _iter_odd
		if(iter == 0){
			//inputfile remains original input file
			iterFileOptions.outputfile = iterFileOptions.outputfile + "_iter_even";
		}else{
			if(iter % 2 == 0){
				iterFileOptions.inputfile = iterFileOptions.outputfile + "_iter_odd";
				iterFileOptions.outputfile = iterFileOptions.outputfile + "_iter_even";
			}else{
				iterFileOptions.inputfile = iterFileOptions.outputfile + "_iter_even";
				iterFileOptions.outputfile = iterFileOptions.outputfile + "_iter_odd";
			}
		}
#else
		if(iter == 0){
			//inputfile remains original input file
			iterFileOptions.outputfile = iterFileOptions.outputfile + "_iter_0";
		}else{
			iterFileOptions.inputfile = iterFileOptions.outputfile + "_iter_" + std::to_string(iter-1);
			iterFileOptions.outputfile = iterFileOptions.outputfile + "_iter_" + std::to_string(iter);
		}
#endif
		correctFile(minhashOptions, alignmentOptions,
            goodAlignmentProperties, correctionOptions,
            runtimeOptions, iterFileOptions,
            readIsCorrectedVector, locksForProcessedFlags,
            nLocksForProcessedFlags, deviceIds);

		iter++;

	}while(iter < iters);


	//rename final result to requested output file name and delete intermediate files
	bool keepIntermediateResults = false;

#ifdef DO_ALTERNATE
	if(iters % 2 == 0){
		std::string toRename = fileOptions.outputfile + "_iter_odd";
		std::rename(toRename.c_str(), fileOptions.outputfile.c_str());

		if(!keepIntermediateResults && iters > 1)
			deleteFiles({fileOptions.outputfile + "_iter_even"});
	}else{
		std::string toRename = fileOptions.outputfile + "_iter_even";
		std::rename(toRename.c_str(), fileOptions.outputfile.c_str());

		if(!keepIntermediateResults && iters > 1)
			deleteFiles({fileOptions.outputfile + "_iter_odd"});
	}
#else
	std::string toRename = fileOptions.outputfile + "_iter_" + std::to_string(iters-1);
	std::rename(toRename.c_str(), fileOptions.outputfile.c_str());

	if(!keepIntermediateResults){
		std::vector<std::string> filestodelete;
		for(int i = 0; i < iters-1; i++)
			filestodelete.push_back(iterFileOptions.outputfile + "_iter_" + std::to_string(i));
		deleteFiles(filestodelete);
	}
#endif


}


}
