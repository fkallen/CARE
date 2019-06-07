#include <gpu/correct_gpu.hpp>

#include <config.hpp>

#include <options.hpp>
#include <rangegenerator.hpp>

#include <cpu_correction_thread.hpp>
#include <candidatedistribution.hpp>

#include <gpu/readstorage.hpp>
#include <gpu/gpu_correction_thread.hpp>
#include <gpu/qualityscoreweights.hpp>
#include <qualityscoreweights.hpp>

#include <minhasher.hpp>


#include <cuda_profiler_api.h>

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>
#include <sstream>
#include <cstdlib>

//#define DO_PROFILE

namespace care{
namespace gpu{

    void correct_gpu(const MinhashOptions& minhashOptions,
    				  const AlignmentOptions& alignmentOptions,
    				  const GoodAlignmentProperties& goodAlignmentProperties,
    				  const CorrectionOptions& correctionOptions,
    				  const RuntimeOptions& runtimeOptions,
    				  const FileOptions& fileOptions,
                      const SequenceFileProperties& sequenceFileProperties,
                      Minhasher& minhasher,
                      cpu::ContiguousReadStorage& cpuReadStorage,
                      std::uint64_t maxCandidatesPerRead,
    				  std::vector<char>& readIsCorrectedVector,
    				  std::unique_ptr<std::mutex[]>& locksForProcessedFlags,
    				  std::size_t nLocksForProcessedFlags){

    using CPUErrorCorrectionThread_t = cpu::CPUCorrectionThread;
    using GPUErrorCorrectionThread_t = gpu::ErrorCorrectionThreadOnlyGPU;

    constexpr int maxCPUThreadsPerGPU = 64;

    const auto& deviceIds = runtimeOptions.deviceIds;


    #if 1
    const int nCorrectorThreads = deviceIds.size() == 0 ? runtimeOptions.nCorrectorThreads
                          : std::min(runtimeOptions.nCorrectorThreads, maxCPUThreadsPerGPU * int(deviceIds.size()));
    #else
    const int nCorrectorThreads = 1;
    #endif

    std::cout << "Using " << nCorrectorThreads << " corrector threads" << std::endl;

    // initialize qscore-to-weight lookup table
    gpu::init_weights(deviceIds);


    /*
    Spawn correction threads
    */

    std::vector<std::string> tmpfiles;
    for(int i = 0; i < nCorrectorThreads; i++) {
    tmpfiles.emplace_back(fileOptions.outputfile + "_tmp_" + std::to_string(1000 + i));
    }

    int nGpuThreads = std::min(nCorrectorThreads, runtimeOptions.threadsForGPUs);
    int nCpuThreads = nCorrectorThreads - nGpuThreads;

#ifdef DO_PROFILE
    cpu::RangeGenerator<read_number> readIdGenerator(100000);
#else
    cpu::RangeGenerator<read_number> readIdGenerator(sequenceFileProperties.nReads);
    //cpu::RangeGenerator<read_number> readIdGenerator(50000);
#endif

    NN_Correction_Classifier_Base nnClassifierBase;
    if(correctionOptions.correctionType == CorrectionType::Convnet){
        nnClassifierBase = std::move(NN_Correction_Classifier_Base{"./nn_sources", fileOptions.nnmodelfilename});
    }

    std::vector<CPUErrorCorrectionThread_t> cpucorrectorThreads(nCpuThreads);
    std::vector<GPUErrorCorrectionThread_t> gpucorrectorThreads(nGpuThreads);
    std::mutex writelock;

    for(int threadId = 0; threadId < nCpuThreads; threadId++) {

    //cpubatchgenerators[threadId] = BatchGenerator<read_number>(ncpuReads, 1, threadId, nCpuThreads);
    typename CPUErrorCorrectionThread_t::CorrectionThreadOptions threadOpts;
    threadOpts.threadId = threadId;

    threadOpts.outputfile = tmpfiles[threadId];
    threadOpts.readIdGenerator = &readIdGenerator;
    threadOpts.minhasher = &minhasher;
    threadOpts.readStorage = &cpuReadStorage;
    threadOpts.coutLock = &writelock;
    threadOpts.readIsCorrectedVector = &readIsCorrectedVector;
    threadOpts.locksForProcessedFlags = locksForProcessedFlags.get();
    threadOpts.nLocksForProcessedFlags = nLocksForProcessedFlags;

    cpucorrectorThreads[threadId].alignmentOptions = alignmentOptions;
    cpucorrectorThreads[threadId].goodAlignmentProperties = goodAlignmentProperties;
    cpucorrectorThreads[threadId].correctionOptions = correctionOptions;
    cpucorrectorThreads[threadId].fileOptions = fileOptions;
    cpucorrectorThreads[threadId].threadOpts = threadOpts;
    cpucorrectorThreads[threadId].fileProperties = sequenceFileProperties;
    cpucorrectorThreads[threadId].max_candidates = maxCandidatesPerRead;

    cpucorrectorThreads[threadId].run();
    }

    gpu::ContiguousReadStorage gpuReadStorage(&cpuReadStorage, deviceIds);

    gpuReadStorage.initGPUData();

    std::cout << "Sequence Type: " << gpuReadStorage.getNameOfSequenceType() << std::endl;
    std::cout << "Quality Type: " << gpuReadStorage.getNameOfQualityType() << std::endl;

    assert(!(deviceIds.size() == 0 && nGpuThreads > 0));

#ifdef DO_PROFILE
    cudaProfilerStart(); CUERR;
#endif

    for(int threadId = 0; threadId < nGpuThreads; threadId++) {

    typename GPUErrorCorrectionThread_t::CorrectionThreadOptions threadOpts;
    threadOpts.threadId = threadId;
    threadOpts.deviceId = deviceIds.size() == 0 ? -1 : deviceIds[threadId % deviceIds.size()];
    threadOpts.canUseGpu = runtimeOptions.canUseGpu;
    threadOpts.outputfile = tmpfiles[nCpuThreads + threadId];
    threadOpts.readIdGenerator = &readIdGenerator;
    threadOpts.minhasher = &minhasher;
    threadOpts.gpuReadStorage = &gpuReadStorage;
    threadOpts.coutLock = &writelock;
    threadOpts.readIsCorrectedVector = &readIsCorrectedVector;
    threadOpts.locksForProcessedFlags = locksForProcessedFlags.get();
    threadOpts.nLocksForProcessedFlags = nLocksForProcessedFlags;

    gpucorrectorThreads[threadId].alignmentOptions = alignmentOptions;
    gpucorrectorThreads[threadId].goodAlignmentProperties = goodAlignmentProperties;
    gpucorrectorThreads[threadId].correctionOptions = correctionOptions;
    gpucorrectorThreads[threadId].fileOptions = fileOptions;
    gpucorrectorThreads[threadId].threadOpts = threadOpts;
    gpucorrectorThreads[threadId].fileProperties = sequenceFileProperties;
    gpucorrectorThreads[threadId].max_candidates = maxCandidatesPerRead;
    gpucorrectorThreads[threadId].classifierBase = &nnClassifierBase;

    gpucorrectorThreads[threadId].run();
    }



#ifndef DO_PROFILE
    std::cout << "Correcting..." << std::endl;

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
    std::cout << "Profiling..." << std::endl;
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

#ifdef DO_PROFILE
    cudaProfilerStop(); CUERR;

    std::exit(0);
#endif

    //std::cout << "threads done" << std::endl;

    size_t occupiedMemory = minhasher.numBytes() + cpuReadStorage.size();

    minhasher.destroy();
    cpuReadStorage.destroy();
    gpuReadStorage.destroy();

    std::cout << "begin merge" << std::endl;

    if(!correctionOptions.extractFeatures){

        std::cout << "begin merging reads" << std::endl;

        TIMERSTARTCPU(merge);

        mergeResultFiles(sequenceFileProperties.nReads, fileOptions.inputfile, fileOptions.format, tmpfiles, fileOptions.outputfile);
        //mergeResultFiles2(sequenceFileProperties.nReads, fileOptions.inputfile, fileOptions.format, tmpfiles, fileOptions.outputfile, occupiedMemory);

        TIMERSTOPCPU(merge);

        std::cout << "end merging reads" << std::endl;

    }

    deleteFiles(tmpfiles);

    std::vector<std::string> featureFiles(tmpfiles);
    for(auto& s : featureFiles)
        s = s + "_features";

    //concatenate feature files of each thread into one file

    if(correctionOptions.extractFeatures){
        std::cout << "begin merging features" << std::endl;

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

        std::cout << "end merging features" << std::endl;
    }else{
        deleteFiles(featureFiles);
    }

    std::cout << "end merge" << std::endl;
    }


}
}
