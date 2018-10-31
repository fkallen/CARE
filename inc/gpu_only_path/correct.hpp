#ifndef CARE_CORRECT_GPU_HPP
#define CARE_CORRECT_GPU_HPP

#include "../options.hpp"
#include "../rangegenerator.hpp"

#include "../cpu_correction_thread.hpp"
#include "gpu_correction_thread.hpp"


#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

namespace care{
namespace gpu{


template<class minhasher_t,
		 class readStorage_t,
		 bool indels>
void correct_gpu(const MinhashOptions& minhashOptions,
				  const AlignmentOptions& alignmentOptions,
				  const GoodAlignmentProperties& goodAlignmentProperties,
				  const CorrectionOptions& correctionOptions,
				  const RuntimeOptions& runtimeOptions,
				  const FileOptions& fileOptions,
                  const SequenceFileProperties& sequenceFileProperties,
                  minhasher_t& minhasher,
                  readStorage_t& readStorage,
				  std::vector<char>& readIsCorrectedVector,
				  std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
				  std::size_t nLocksForProcessedFlags,
				  const std::vector<int>& deviceIds){

	assert(indels == false);

	using Minhasher_t = minhasher_t;
	using ReadStorage_t = readStorage_t;
	using Sequence_t = typename ReadStorage_t::Sequence_t;
	using ReadId_t = typename ReadStorage_t::ReadId_t;
    using GPUReadStorage_t = GPUReadStorage<ReadStorage_t>;

	using CPUErrorCorrectionThread_t = ErrorCorrectionThreadCombined<Minhasher_t, ReadStorage_t, indels>;

	using GPUErrorCorrectionThread_t = gpu::ErrorCorrectionThreadOnlyGPU<Minhasher_t, ReadStorage_t, GPUReadStorage_t, care::gpu::RangeGenerator<ReadId_t>>;


//#define DO_PROFILE

#if 1
    const int nCorrectorThreads = deviceIds.size() == 0 ? runtimeOptions.nCorrectorThreads
                        : std::min(runtimeOptions.nCorrectorThreads, maxCPUThreadsPerGPU * int(deviceIds.size()));
#else
	const int nCorrectorThreads = 1;
#endif

	std::cout << "Using " << nCorrectorThreads << " corrector threads" << std::endl;



      // initialize qscore-to-weight lookup table
  	init_weights();

    //SequenceFileProperties sequenceFileProperties = getSequenceFileProperties(fileOptions.inputfile, fileOptions.format);

    /*
        Make candidate statistics
    */

    std::uint64_t max_candidates = runtimeOptions.max_candidates;
    //std::uint64_t max_candidates = std::numeric_limits<std::uint64_t>::max();

    if(max_candidates == 0){
        std::cout << "estimating candidate cutoff" << std::endl;

        correctiondetail::Dist<std::int64_t, std::int64_t> candidateDistribution;

        {
            TIMERSTARTCPU(candidateestimation);
            std::map<std::int64_t, std::int64_t> candidateHistogram
                    = correctiondetail::getCandidateCountHistogram(minhasher,
                                                readStorage,
                                                sequenceFileProperties.nReads / 10,
                                                runtimeOptions.threads);

            TIMERSTOPCPU(candidateestimation);

            candidateDistribution = correctiondetail::estimateDist(candidateHistogram);

            std::vector<std::pair<std::int64_t, std::int64_t>> vec(candidateHistogram.begin(), candidateHistogram.end());
            std::sort(vec.begin(), vec.end(), [](auto p1, auto p2){ return p1.second < p2.second;});

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
    }

    std::cout << "Using candidate cutoff: " << max_candidates << std::endl;

    /*
        Spawn correction threads
    */

    std::vector<std::string> tmpfiles;
    for(int i = 0; i < nCorrectorThreads; i++){
        tmpfiles.emplace_back(fileOptions.outputfile + "_tmp_" + std::to_string(1000 + i));
    }

    int nGpuThreads = std::min(nCorrectorThreads, runtimeOptions.threadsForGPUs);
	int nCpuThreads = nCorrectorThreads - nGpuThreads;

    std::vector<BatchGenerator<ReadId_t>> cpubatchgenerators(nCpuThreads);
	std::vector<care::gpu::BatchGenerator<ReadId_t>> gpubatchgenerators(nGpuThreads);

    std::vector<CPUErrorCorrectionThread_t> cpucorrectorThreads(nCpuThreads);
	std::vector<GPUErrorCorrectionThread_t> gpucorrectorThreads(nGpuThreads);
    std::vector<char> readIsProcessedVector(readIsCorrectedVector);
    std::mutex writelock;

	std::uint64_t ncpuReads = nCpuThreads > 0 ? std::uint64_t(sequenceFileProperties.nReads / 7.0) : 0;
	std::uint64_t ngpuReads = sequenceFileProperties.nReads - ncpuReads;
	std::uint64_t nReadsPerGPUThread = nGpuThreads > 0 ? SDIV(ngpuReads, nGpuThreads) : 0;
    if(nGpuThreads == 0){
        ncpuReads += ngpuReads;
        ngpuReads = 0;
    }

	std::cout << "nCpuThreads: " << nCpuThreads << ", nGpuThreads: " << nGpuThreads << std::endl;
	std::cout << "ncpuReads: " << ncpuReads << ", ngpuReads: " << ngpuReads << std::endl;

	for(int threadId = 0; threadId < nCpuThreads; threadId++){

        cpubatchgenerators[threadId] = BatchGenerator<ReadId_t>(ncpuReads, 1, threadId, nCpuThreads);
        typename CPUErrorCorrectionThread_t::CorrectionThreadOptions threadOpts;
        threadOpts.threadId = threadId;
        threadOpts.deviceId = 0;
        threadOpts.canUseGpu = false;
        threadOpts.outputfile = tmpfiles[threadId];
        threadOpts.batchGen = &cpubatchgenerators[threadId];
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
        cpucorrectorThreads[threadId].threadOpts = threadOpts;
        cpucorrectorThreads[threadId].fileProperties = sequenceFileProperties;
        cpucorrectorThreads[threadId].max_candidates = max_candidates;

        cpucorrectorThreads[threadId].run();
    }

    GPUReadStorage_t gpuReadStorage;
    bool canUseGPUReadStorage = true;
    /*GPUReadStorageType bestGPUReadStorageType = GPUReadStorage_t::getBestPossibleType(readStorage,
                                                                            Sequence_t::getNumBytes(fileProperties.maxSequenceLength),
                                                                            fileProperties.maxSequenceLength,
                                                                            0.8f,
                                                                            threadOpts.deviceId);


    if(bestGPUReadStorageType != GPUReadStorageType::None){
        //bestGPUReadStorageType = GPUReadStorageType::Sequences;

        gpuReadStorage = GPUReadStorage_t::createFrom(readStorage,
                                                        bestGPUReadStorageType,
                                                        Sequence_t::getNumBytes(fileProperties.maxSequenceLength),
                                                        fileProperties.maxSequenceLength,
                                                        threadOpts.deviceId);

        canUseGPUReadStorage = true;
        std::cout << "Using gpu read storage, type " << GPUReadStorage_t::nameOf(bestGPUReadStorageType) << std::endl;
    }*/
    std::cout << "External gpu read storage" << std::endl;
    gpuReadStorage = GPUReadStorage_t::createFrom(readStorage,
                                                Sequence_t::getNumBytes(sequenceFileProperties.maxSequenceLength),
                                                sequenceFileProperties.maxSequenceLength,
                                                0.8f,
                                                true,
                                                deviceIds.size() == 0 ? -1 : deviceIds[0]);

    std::cout << "Sequence Type: " << gpuReadStorage.getNameOfSequenceType() << std::endl;
    //std::cout << "Sequence length Type: " << gpuReadStorage.getNameOfSequenceLengthType() << std::endl;
    std::cout << "Quality Type: " << gpuReadStorage.getNameOfQualityType() << std::endl;


    for(int threadId = 0; threadId < nGpuThreads; threadId++){

        gpubatchgenerators[threadId] = care::gpu::BatchGenerator<ReadId_t>(ncpuReads + threadId * nReadsPerGPUThread,
                                                                            std::min(sequenceFileProperties.nReads,
                                                                            ncpuReads + (threadId+1) * nReadsPerGPUThread));
        typename GPUErrorCorrectionThread_t::CorrectionThreadOptions threadOpts;
        threadOpts.threadId = threadId;
        threadOpts.deviceId = deviceIds.size() == 0 ? -1 : deviceIds[threadId % deviceIds.size()];
        threadOpts.canUseGpu = runtimeOptions.canUseGpu;
        threadOpts.outputfile = tmpfiles[nCpuThreads + threadId];
        threadOpts.batchGen = &gpubatchgenerators[threadId];
        threadOpts.minhasher = &minhasher;
        threadOpts.readStorage = &readStorage;
        threadOpts.gpuReadStorage = &gpuReadStorage;
        threadOpts.canUseGPUReadStorage = canUseGPUReadStorage;
        threadOpts.coutLock = &writelock;
        threadOpts.readIsProcessedVector = &readIsProcessedVector;
        threadOpts.readIsCorrectedVector = &readIsCorrectedVector;
        threadOpts.locksForProcessedFlags = locksForProcessedFlags.get();
        threadOpts.nLocksForProcessedFlags = nLocksForProcessedFlags;

        gpucorrectorThreads[threadId].alignmentOptions = alignmentOptions;
        gpucorrectorThreads[threadId].goodAlignmentProperties = goodAlignmentProperties;
        gpucorrectorThreads[threadId].correctionOptions = correctionOptions;
        gpucorrectorThreads[threadId].threadOpts = threadOpts;
        gpucorrectorThreads[threadId].fileProperties = sequenceFileProperties;
        gpucorrectorThreads[threadId].max_candidates = max_candidates;

        gpucorrectorThreads[threadId].run();
    }

    std::cout << "Correcting..." << std::endl;


#ifndef DO_PROFILE

    bool showProgress = runtimeOptions.showProgress;

    std::thread progressThread = std::thread([&]() -> void{
        if(!showProgress)
            return;

        std::chrono::time_point<std::chrono::system_clock> timepoint_begin = std::chrono::system_clock::now();
        std::chrono::duration<double> runtime = std::chrono::seconds(0);
        std::chrono::duration<int> sleepinterval = std::chrono::seconds(1);

        while(showProgress){
            ReadId_t progress = 0;
            ReadId_t correctorProgress = 0;

            for(int i = 0; i < nCpuThreads; i++){
                correctorProgress += cpucorrectorThreads[i].nProcessedReads;
            }

            for(int i = 0; i < nGpuThreads; i++){
                correctorProgress += gpucorrectorThreads[i].nProcessedReads;
            }

            progress = correctorProgress;

            printf("Progress: %3.2f %% %10u %10lu (Runtime: %03d:%02d:%02d)\r",
                    ((progress * 1.0 / sequenceFileProperties.nReads) * 100.0),
                    correctorProgress, sequenceFileProperties.nReads,
                    int(std::chrono::duration_cast<std::chrono::hours>(runtime).count()),
                    int(std::chrono::duration_cast<std::chrono::minutes>(runtime).count()) % 60,
                    int(runtime.count()) % 60);
            std::cout << std::flush;

            if(progress < sequenceFileProperties.nReads){
                  std::this_thread::sleep_for(sleepinterval);
                  runtime = std::chrono::system_clock::now() - timepoint_begin;
            }
        }
    });

#else

    constexpr int sleepiterbegin = 1;
    constexpr int sleepiterend = 10;

    int sleepiter = 0;

    std::chrono::duration<double> runtime = std::chrono::seconds(0);
    std::chrono::duration<int> sleepinterval = std::chrono::seconds(1);

    while(true){

        std::this_thread::sleep_for(sleepinterval);

        sleepiter++;

        #ifdef __NVCC__
            if(sleepiter == sleepiterbegin)
                cudaProfilerStart(); CUERR;
        #endif


        #ifdef __NVCC__
        if(sleepiter == sleepiterend){
            cudaProfilerStop(); CUERR;

			for(int i = 0; i < nCpuThreads; i++){
                cpucorrectorThreads[i].stopAndAbort = true;
				cpucorrectorThreads[i].join();
            }

            for(int i = 0; i < nGpuThreads; i++){
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

    if(canUseGPUReadStorage){
        GPUReadStorage_t::destroy(gpuReadStorage);
    }

   // generators.clear();
   // ecthreads.clear();
    readIsProcessedVector.clear();
    readIsProcessedVector.shrink_to_fit();

    std::cout << "begin merge" << std::endl;
    TIMERSTARTCPU(merge);

    mergeResultFiles(sequenceFileProperties.nReads, fileOptions.inputfile, fileOptions.format, tmpfiles, fileOptions.outputfile);

    TIMERSTOPCPU(merge);

    deleteFiles(tmpfiles);

    if(!correctionOptions.extractFeatures){
        std::vector<std::string> featureFiles(tmpfiles);
        for(auto& s : featureFiles)
            s = s + "_features";
        deleteFiles(featureFiles);
    }

    std::cout << "end merge" << std::endl;
}

}
}

#endif
