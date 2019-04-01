#ifndef CARE_CORRECT_GPU_HPP
#define CARE_CORRECT_GPU_HPP

#include <config.hpp>

#include "../options.hpp"
#include "../rangegenerator.hpp"

#include "../cpu_correction_thread.hpp"
#include "../candidatedistribution.hpp"

#include "gpu_correction_thread.hpp"
#include <gpu/qualityscoreweights.hpp>
#include <qualityscoreweights.hpp>

#include <minhasher.hpp>
#include <gpu/readstorage.hpp>

#include <cuda_profiler_api.h>

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>
#include <sstream>
#include <cstdlib>

namespace care {
namespace gpu {



void correct_gpu(const MinhashOptions& minhashOptions,
			const AlignmentOptions& alignmentOptions,
			const GoodAlignmentProperties& goodAlignmentProperties,
			const CorrectionOptions& correctionOptions,
			const RuntimeOptions& runtimeOptions,
			const FileOptions& fileOptions,
			const SequenceFileProperties& sequenceFileProperties,
            Minhasher& minhasher,
            gpu::ContiguousReadStorage& readStorage,
			std::vector<char>& readIsCorrectedVector,
			std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
			std::size_t nLocksForProcessedFlags){

    using Minhasher_t = Minhasher;
    using ReadStorage_t = gpu::ContiguousReadStorage;
    
	using CPUErrorCorrectionThread_t = cpu::CPUCorrectionThread<Minhasher_t, ReadStorage_t, false>;
	using GPUErrorCorrectionThread_t = gpu::ErrorCorrectionThreadOnlyGPU;

	constexpr int maxCPUThreadsPerGPU = 64;

	const auto& deviceIds = runtimeOptions.deviceIds;

	//#define DO_PROFILE

    #if 1
	const int nCorrectorThreads = deviceIds.size() == 0 ? runtimeOptions.nCorrectorThreads
	                              : std::min(runtimeOptions.nCorrectorThreads, maxCPUThreadsPerGPU * int(deviceIds.size()));
    #else
	const int nCorrectorThreads = 1;
    #endif

	std::cout << "Using " << nCorrectorThreads << " corrector threads" << std::endl;

	// initialize qscore-to-weight lookup table
	gpu::init_weights(deviceIds);

	//SequenceFileProperties sequenceFileProperties = getSequenceFileProperties(fileOptions.inputfile, fileOptions.format);

	/*
	    Make candidate statistics
	 */

	std::uint64_t max_candidates = runtimeOptions.max_candidates;
	//std::uint64_t max_candidates = std::numeric_limits<std::uint64_t>::max();

	if(max_candidates == 0) {
		std::cout << "estimating candidate cutoff" << std::endl;

		cpu::Dist<std::int64_t, std::int64_t> candidateDistribution;
		cpu::Dist2<std::int64_t, std::int64_t> candidateDistribution2;

		{
			TIMERSTARTCPU(candidateestimation);
			std::map<std::int64_t, std::int64_t> candidateHistogram
			        = cpu::getCandidateCountHistogram(minhasher,
						readStorage,
						sequenceFileProperties.nReads / 10,
						correctionOptions.hits_per_candidate,
						runtimeOptions.threads);

			TIMERSTOPCPU(candidateestimation);

			candidateDistribution = cpu::estimateDist(candidateHistogram);

			//candidateDistribution2 = cpu::estimateDist2(candidateHistogram);

			std::vector<std::pair<std::int64_t, std::int64_t> > vec(candidateHistogram.begin(), candidateHistogram.end());
			std::sort(vec.begin(), vec.end(), [](auto p1, auto p2){
						return p1.second < p2.second;
					});

			std::ofstream of("ncandidates.txt");
			for(const auto& p : vec)
				of << p.first << " " << p.second << '\n';
			of.flush();
		}

		std::cout << "candidates.max " << candidateDistribution.max << std::endl;
		std::cout << "candidates.average " << candidateDistribution.average << std::endl;
		std::cout << "candidates.stddev " << candidateDistribution.stddev << std::endl;

		const std::uint64_t estimatedMeanAlignedCandidates = candidateDistribution.max;
		const std::uint64_t estimatedDeviationAlignedCandidates = candidateDistribution.stddev;
		const std::uint64_t estimatedAlignmentCountThreshold = estimatedMeanAlignedCandidates
		                                                       + 2.5 * estimatedDeviationAlignedCandidates;

		max_candidates = estimatedAlignmentCountThreshold;
		//max_candidates = candidateDistribution2.percentRanges[90].first;
	}

	std::cout << "Using candidate cutoff: " << max_candidates << std::endl;

	/*
	    Spawn correction threads
	 */

	std::vector<std::string> tmpfiles;
	for(int i = 0; i < nCorrectorThreads; i++) {
		tmpfiles.emplace_back(fileOptions.outputfile + "_tmp_" + std::to_string(1000 + i));
	}

	int nGpuThreads = std::min(nCorrectorThreads, runtimeOptions.threadsForGPUs);
	int nCpuThreads = nCorrectorThreads - nGpuThreads;

	cpu::RangeGenerator<read_number> readIdGenerator(sequenceFileProperties.nReads);

	std::vector<CPUErrorCorrectionThread_t> cpucorrectorThreads(nCpuThreads);
	std::vector<GPUErrorCorrectionThread_t> gpucorrectorThreads(nGpuThreads);
	std::vector<char> readIsProcessedVector(readIsCorrectedVector);
	std::mutex writelock;

	for(int threadId = 0; threadId < nCpuThreads; threadId++) {

		//cpubatchgenerators[threadId] = BatchGenerator<read_number>(ncpuReads, 1, threadId, nCpuThreads);
		typename CPUErrorCorrectionThread_t::CorrectionThreadOptions threadOpts;
		threadOpts.threadId = threadId;

		threadOpts.outputfile = tmpfiles[threadId];
		threadOpts.readIdGenerator = &readIdGenerator;
		threadOpts.minhasher = &minhasher;
		threadOpts.readStorage = &readStorage;
		threadOpts.coutLock = &writelock;
		threadOpts.readIsProcessedVector = &readIsProcessedVector;
		threadOpts.readIsCorrectedVector = &readIsCorrectedVector;
		threadOpts.locksForProcessedFlags = locksForProcessedFlags.get();
		threadOpts.nLocksForProcessedFlags = nLocksForProcessedFlags;

		cpucorrectorThreads[threadId].alignmentOptions = alignmentOptions;
		cpucorrectorThreads[threadId].goodAlignmentProperties = goodAlignmentProperties;
		cpucorrectorThreads[threadId].correctionOptions = correctionOptions;
        cpucorrectorThreads[threadId].fileOptions = fileOptions;
		cpucorrectorThreads[threadId].threadOpts = threadOpts;
		cpucorrectorThreads[threadId].fileProperties = sequenceFileProperties;
		cpucorrectorThreads[threadId].max_candidates = max_candidates;

		cpucorrectorThreads[threadId].run();
	}

	readStorage.initGPUData();

	std::cout << "Sequence Type: " << readStorage.getNameOfSequenceType() << std::endl;
	std::cout << "Quality Type: " << readStorage.getNameOfQualityType() << std::endl;

	assert(!(deviceIds.size() == 0 && nGpuThreads > 0));

	for(int threadId = 0; threadId < nGpuThreads; threadId++) {

		typename GPUErrorCorrectionThread_t::CorrectionThreadOptions threadOpts;
		threadOpts.threadId = threadId;
		threadOpts.deviceId = deviceIds.size() == 0 ? -1 : deviceIds[threadId % deviceIds.size()];
		threadOpts.canUseGpu = runtimeOptions.canUseGpu;
		threadOpts.outputfile = tmpfiles[nCpuThreads + threadId];
		threadOpts.readIdGenerator = &readIdGenerator;
		threadOpts.minhasher = &minhasher;
		threadOpts.gpuReadStorage = &readStorage;
		threadOpts.coutLock = &writelock;
		threadOpts.readIsProcessedVector = &readIsProcessedVector;
		threadOpts.readIsCorrectedVector = &readIsCorrectedVector;
		threadOpts.locksForProcessedFlags = locksForProcessedFlags.get();
		threadOpts.nLocksForProcessedFlags = nLocksForProcessedFlags;

		gpucorrectorThreads[threadId].alignmentOptions = alignmentOptions;
		gpucorrectorThreads[threadId].goodAlignmentProperties = goodAlignmentProperties;
		gpucorrectorThreads[threadId].correctionOptions = correctionOptions;
        gpucorrectorThreads[threadId].fileOptions = fileOptions;
		gpucorrectorThreads[threadId].threadOpts = threadOpts;
		gpucorrectorThreads[threadId].fileProperties = sequenceFileProperties;
		gpucorrectorThreads[threadId].max_candidates = max_candidates;

		gpucorrectorThreads[threadId].run();
	}

	std::cout << "Correcting..." << std::endl;


    #ifndef DO_PROFILE

	bool showProgress = runtimeOptions.showProgress;

	std::thread progressThread = std::thread([&]() -> void {
				if(!showProgress)
					return;

				std::chrono::time_point<std::chrono::system_clock> timepoint_begin = std::chrono::system_clock::now();
				std::chrono::duration<double> runtime = std::chrono::seconds(0);
				std::chrono::duration<int> sleepinterval = std::chrono::seconds(1);

				while(showProgress) {
				        read_number progress = 0;
				        /*read_number correctorProgress = 0;

				           for(int i = 0; i < nCpuThreads; i++){
				            correctorProgress += cpucorrectorThreads[i].nProcessedReads;
				           }

				           for(int i = 0; i < nGpuThreads; i++){
				            correctorProgress += gpucorrectorThreads[i].nProcessedReads;
				           }

				           progress = correctorProgress;*/
				        progress = readIdGenerator.getCurrentUnsafe() - readIdGenerator.getBegin();

				        printf("Progress: %3.2f %% %10u %10lu (Runtime: %03d:%02d:%02d)\r",
								((progress * 1.0 / sequenceFileProperties.nReads) * 100.0),
								progress, sequenceFileProperties.nReads,
								int(std::chrono::duration_cast<std::chrono::hours>(runtime).count()),
								int(std::chrono::duration_cast<std::chrono::minutes>(runtime).count()) % 60,
								int(runtime.count()) % 60);
				        std::cout << std::flush;

				        if(progress < sequenceFileProperties.nReads) {
				                std::this_thread::sleep_for(sleepinterval);
				                runtime = std::chrono::system_clock::now() - timepoint_begin;
					}
				}
			});

    #else

	constexpr int sleepiterbegin = 1;
	constexpr int sleepiterend = 2;

	int sleepiter = 0;

	std::chrono::duration<double> runtime = std::chrono::seconds(0);
	std::chrono::duration<int> sleepinterval = std::chrono::seconds(1);

	while(true) {

		std::this_thread::sleep_for(sleepinterval);

		sleepiter++;

	  #ifdef __NVCC__
		if(sleepiter == sleepiterbegin)
			cudaProfilerStart(); CUERR;
	  #endif


	  #ifdef __NVCC__
		if(sleepiter == sleepiterend) {
			cudaProfilerStop(); CUERR;

			for(int i = 0; i < nCpuThreads; i++) {
				cpucorrectorThreads[i].stopAndAbort = true;
				cpucorrectorThreads[i].join();
			}

			for(int i = 0; i < nGpuThreads; i++) {
				gpucorrectorThreads[i].stopAndAbort = true;
				gpucorrectorThreads[i].join();
			}

			std::exit(0);
		}
	  #endif

	}
    #endif

	TIMERSTARTCPU(correction);

	for (auto& thread : cpucorrectorThreads)
		thread.join();

	for (auto& thread : gpucorrectorThreads)
		thread.join();

    #ifndef DO_PROFILE
	showProgress = false;
	progressThread.join();
	if(runtimeOptions.showProgress)
		printf("Progress: %3.2f %%\n", 100.00);
    #endif

	TIMERSTOPCPU(correction);

	//std::cout << "threads done" << std::endl;


	minhasher.destroy();
	readStorage.destroy();

	// generators.clear();
	// ecthreads.clear();
	readIsProcessedVector.clear();
	readIsProcessedVector.shrink_to_fit();

	std::cout << "begin merge" << std::endl;
	TIMERSTARTCPU(merge);

	mergeResultFiles(sequenceFileProperties.nReads, fileOptions.inputfile, fileOptions.format, tmpfiles, fileOptions.outputfile);

	TIMERSTOPCPU(merge);

	deleteFiles(tmpfiles);

    std::vector<std::string> featureFiles(tmpfiles);
    for(auto& s : featureFiles)
        s = s + "_features";

    //concatenate feature files of each thread into one file

    if(correctionOptions.extractFeatures){
        std::stringstream commandbuilder;

        commandbuilder << "cat";

        for(const auto& featureFile : featureFiles){
            commandbuilder << " \"" << featureFile << "\"";
        }

        commandbuilder << " > \"" << fileOptions.outputfile << "_features\"";

        const std::string command = commandbuilder.str();
        TIMERSTARTCPU(concat_feature_files);
        int r1 = std::system(command.c_str());
        TIMERSTOPCPU(concat_feature_files);

        if(r1 != 0){
            std::cerr << "Warning. Feature files could not be concatenated!\n";
            std::cerr << "This command returned a non-zero error value: \n";
            std::cerr << command +  '\n';
            std::cerr << "Please concatenate the following files manually\n";
            for(const auto& s : featureFiles)
                std::cerr << s << '\n';
        }else{
            deleteFiles(featureFiles);
        }
    }else{
        deleteFiles(featureFiles);
    }




	std::cout << "end merge" << std::endl;
}

}
}

#endif
