#include "../inc/care.hpp"

#include "../inc/read.hpp"
#include "../inc/sequencefileio.hpp"
#include "../inc/binarysequencehelpers.hpp"
#include "../inc/build.hpp"

#include "../inc/ganja/hpc_helpers.cuh"

#include "../inc/hammingtools.hpp"
#include "../inc/graphtools.hpp"

#include "../inc/errorcorrectionthread.hpp"
#include "../inc/batchelem.hpp"
#include "../inc/pileup.hpp"
#include "../inc/graph.hpp"

#include "../inc/args.hpp"

#include <cstdint>
#include <thread>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <map>
#include <set>
#include <functional>
#include <mutex>
#include <chrono>
#include <iterator>
#include <future>

#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __NVCC__
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/iterator/constant_iterator.h>

#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/sort.h>

#include <cuda_profiler_api.h>
#endif

#include <experimental/filesystem>

namespace filesys = std::experimental::filesystem;

namespace care{
	
void correct_impl(const MinhashOptions& minhashOptions, 
				  const AlignmentOptions& alignmentOptions, 
				  const GoodAlignmentProperties& goodAlignmentProperties, 
				  const CorrectionOptions& correctionOptions, 
				  const RuntimeOptions& runtimeOptions, 
				  const FileOptions& fileOptions,
				  std::vector<char>& readIsProcessedVector, 
				  std::unique_ptr<std::mutex[]>& locksForProcessedFlags, 
				  size_t nLocksForProcessedFlags,
				  const std::vector<int>& deviceIds){
	
    SequenceFileProperties props = getSequenceFileProperties(fileOptions.inputfile, fileOptions.format);
	

    Minhasher minhasher(minhashOptions);
    ReadStorage readStorage;
    readStorage.setUseQualityScores(correctionOptions.useQualityScores);

    std::cout << "begin build" << std::endl;

	TIMERSTARTCPU(BUILD);
    build(fileOptions.inputfile, fileOptions.format, readStorage, minhasher, runtimeOptions.nInserterThreads);
	TIMERSTOPCPU(BUILD);

    std::cout << "min sequence length " << props.minSequenceLength << ", max sequence length " << props.maxSequenceLength << '\n';

    TIMERSTARTCPU(MAP_TRANSFORM);
	minhasher.transform();
	TIMERSTOPCPU(MAP_TRANSFORM);

	TIMERSTARTCPU(readstorage_transform);
	readStorage.noMoreInserts();
	TIMERSTOPCPU(readstorage_transform);

    std::vector<std::string> tmpfiles;
    for(int i = 0; i < runtimeOptions.nCorrectorThreads; i++){
        tmpfiles.emplace_back(fileOptions.outputfile + "_tmp_" + std::to_string(i));
    }

    std::cout << "begin correct" << std::endl;

	TIMERSTARTCPU(CORRECT);

    std::vector<BatchGenerator> generators(runtimeOptions.nCorrectorThreads);
    std::vector<ErrorCorrectionThread> ecthreads(runtimeOptions.nCorrectorThreads);

    std::mutex writelock;

    for(int threadId = 0; threadId < runtimeOptions.nCorrectorThreads; threadId++){

        generators[threadId] = BatchGenerator(props.nReads, correctionOptions.batchsize, threadId, runtimeOptions.nCorrectorThreads);
        CorrectionThreadOptions threadOpts;
        threadOpts.threadId = threadId;
        threadOpts.deviceId = deviceIds.size() == 0 ? -1 : deviceIds[threadId % deviceIds.size()];
        threadOpts.outputfile = tmpfiles[threadId];
        threadOpts.batchGen = &generators[threadId];
        threadOpts.minhasher = &minhasher;
        threadOpts.readStorage = &readStorage;
        threadOpts.coutLock = &writelock;
        threadOpts.readIsProcessedVector = &readIsProcessedVector;
        threadOpts.locksForProcessedFlags = locksForProcessedFlags.get();
        threadOpts.nLocksForProcessedFlags = nLocksForProcessedFlags;

        ecthreads[threadId].alignmentOptions = alignmentOptions;
        ecthreads[threadId].goodAlignmentProperties = goodAlignmentProperties;
        ecthreads[threadId].correctionOptions = correctionOptions;
        ecthreads[threadId].threadOpts = threadOpts;
        ecthreads[threadId].fileProperties = props;

        ecthreads[threadId].run();
    }

    std::uint64_t progress = 0;
    while(progress < props.nReads){
        progress = 0;
        for(int threadId = 0; threadId < runtimeOptions.nCorrectorThreads; threadId++){
            progress += ecthreads[threadId].nProcessedReads;
        }
        printf("Progress: %3.2f %%\r",
    			((progress * 1.0 / props.nReads) * 100.0));
    	std::cout << std::flush;
        if(progress < props.nReads)
		      std::this_thread::sleep_for(std::chrono::seconds(5));
    }


	for (auto& thread : ecthreads)
		thread.join();

    printf("Progress: %3.2f %%\n", 100.00);


	TIMERSTOPCPU(CORRECT);

	std::cout << "end correct" << std::endl;

    if (correctionOptions.correctCandidates) {
		int asd = std::count_if(readIsProcessedVector.begin(),
				readIsProcessedVector.end(), [](auto b) {return b;});
		std::cout << "total corrected reads: " << asd << std::endl;
	}

	minhasher.init(0);
	readStorage.destroy();


	std::cout << "begin merge" << std::endl;
    mergeResultFiles(props.nReads, fileOptions.inputfile, fileOptions.format, tmpfiles, fileOptions.outputfile);
    deleteFiles(tmpfiles);

	std::cout << "end merge" << std::endl;

}

void performCorrection(const cxxopts::ParseResult& args) {
	//check arguments
    if(!args::areValid(args)){
        throw std::runtime_error("care::performCorrection: Invalid arguments!");
    }

    // initialize global correction data structures
	hammingtools::init_once();
	graphtools::init_once();
	
	//parse options from arguments
	MinhashOptions minhashOptions = args::to<MinhashOptions>(args);
	AlignmentOptions alignmentOptions = args::to<AlignmentOptions>(args);
	GoodAlignmentProperties goodAlignmentProperties = args::to<GoodAlignmentProperties>(args);
    CorrectionOptions correctionOptions = args::to<CorrectionOptions>(args);
	RuntimeOptions runtimeOptions = args::to<RuntimeOptions>(args);
	FileOptions fileOptions = args::to<FileOptions>(args);
	
	//create output directory
	filesys::create_directories(fileOptions.outputdirectory);
	
	//data which is used for multiple correction passes
	std::vector<char> readIsProcessedVector;
	std::unique_ptr<std::mutex[]> locksForProcessedFlags;
	size_t nLocksForProcessedFlags = 0;	

    if (correctionOptions.correctCandidates) {
        SequenceFileProperties props = getSequenceFileProperties(fileOptions.inputfile, fileOptions.format);

		readIsProcessedVector.resize(props.nReads, 0);
		nLocksForProcessedFlags = correctionOptions.batchsize * runtimeOptions.nCorrectorThreads * 1000;
		locksForProcessedFlags.reset(new std::mutex[nLocksForProcessedFlags]);
	}
	
	std::vector<int> deviceIds;
	
#ifdef __CUDACC__

	int nGpus;
	cudaGetDeviceCount(&nGpus); CUERR;
	if(nGpus == 0) throw std::runtime_error("No CUDA capable device found!");
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
		correct_impl(minhashOptions, alignmentOptions, goodAlignmentProperties, correctionOptions, runtimeOptions, iterFileOptions, readIsProcessedVector, locksForProcessedFlags, nLocksForProcessedFlags, deviceIds);
		
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
