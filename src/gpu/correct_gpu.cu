

#include <gpu/gpucorrector.cuh>
#include <gpu/multigpucorrector.cuh>
#include <gpu/gpureadstorage.cuh>

#include <gpu/gpuminhasher.cuh>
#include <gpu/cudaerrorcheck.cuh>

#include <options.hpp>
#include <readlibraryio.hpp>
#include <memorymanagement.hpp>
#include <serializedobjectstorage.hpp>
#include <threadpool.hpp>
#include <rangegenerator.hpp>
#include <concurrencyhelpers.hpp>
#include <corrector.hpp>
#include <corrector_common.hpp>

#include <classification.hpp>
#include <forest.hpp>
#include <gpu/forest_gpu.cuh>


#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <future>

#include <thrust/iterator/counting_iterator.h>

namespace care{
namespace gpu{


template<class Minhasher>
class SimpleGpuCorrectionPipeline{    
    /*
        SimpleGpuCorrectionPipeline uses
        thread which is responsible for everything.
        Threadpool may be used for internal parallelization.
    */

    using AnchorHasher = GpuAnchorHasher;
public:
    struct RunStatistics{
        double hasherTimeAverage{};
        double correctorTimeAverage{};
        double outputconstructorTimeAverage{};
        MemoryUsage memoryInputData{};
        MemoryUsage memoryRawOutputData{};
        MemoryUsage memoryHasher{};
        MemoryUsage memoryCorrector{};
        MemoryUsage memoryOutputConstructor{};
    };

    SimpleGpuCorrectionPipeline(
        const GpuReadStorage& readStorage_,
        const Minhasher& minhasher_,
        ThreadPool* threadPool_,
        const GpuForest* gpuForestAnchor_, 
        const GpuForest* gpuForestCandidate_
    ) :
        readStorage(&readStorage_),
        minhasher(&minhasher_),
        threadPool(threadPool_),
        gpuForestAnchor(gpuForestAnchor_),
        gpuForestCandidate(gpuForestCandidate_)
    {

    }

    template<class IdGenerator, class BatchCompletion, class ResultProcessorSerialized>
    RunStatistics runToCompletion(
        int deviceId,
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions,
        GpuReadCorrectionFlags& correctionFlags,
        BatchCompletion batchCompleted,
        ResultProcessorSerialized processSerializedResults
    ) const {

        auto continueCondition = [&](){ return !readIdGenerator.empty(); };

        return run_impl(
            deviceId,
            readIdGenerator,
            programOptions,
            correctionFlags,
            batchCompleted,
            continueCondition,
            processSerializedResults
        );
    }

    template<class IdGenerator, class BatchCompletion, class ResultProcessorSerialized>
    RunStatistics runSomeBatches(
        int deviceId,
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions,
        GpuReadCorrectionFlags& correctionFlags,
        BatchCompletion batchCompleted,
        int numBatches,
        ResultProcessorSerialized processSerializedResults
    ) const {

        auto continueCondition = [&](){ bool success = !readIdGenerator.empty() && numBatches > 0; numBatches--; return success;};

        return run_impl(
            deviceId,
            readIdGenerator,
            programOptions,
            correctionFlags,
            batchCompleted,
            continueCondition,
            processSerializedResults
        );
    }

    template<class IdGenerator, class BatchCompletion, class ResultProcessorSerialized>
    RunStatistics runToCompletionDoubleBuffered(
        int deviceId,
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions,
        GpuReadCorrectionFlags& correctionFlags,
        BatchCompletion batchCompleted,
        ResultProcessorSerialized processSerializedResults
    ) const {

        auto continueCondition = [&](){ return !readIdGenerator.empty(); };

        return runDoubleBuffered_impl(
            deviceId,
            readIdGenerator,
            programOptions,
            correctionFlags,
            batchCompleted,
            continueCondition,
            processSerializedResults
        );
    }

    template<class IdGenerator, class BatchCompletion, class ResultProcessorSerialized>
    RunStatistics runSomeBatchesDoubleBuffered(
        int deviceId,
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions,
        GpuReadCorrectionFlags& correctionFlags,
        BatchCompletion batchCompleted,
        int numBatches,
        ResultProcessorSerialized processSerializedResults
    ) const {

        auto continueCondition = [&](){ bool success = !readIdGenerator.empty() && numBatches > 0; numBatches--; return success;};

        return runDoubleBuffered_impl(
            deviceId,
            readIdGenerator,
            programOptions,
            correctionFlags,
            batchCompleted,
            continueCondition,
            processSerializedResults
        );
    }

    template<class IdGenerator, class BatchCompletion, class ContinueCondition, class ResultProcessorSerialized>
    RunStatistics runDoubleBuffered_impl(
        int deviceId,
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions,
        GpuReadCorrectionFlags& correctionFlags,
        BatchCompletion batchCompleted,
        ContinueCondition continueCondition,
        ResultProcessorSerialized processSerializedResults
    ) const {
        cub::SwitchDevice sd{deviceId};

        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();

        try{

            constexpr int numextra = 1;
            constexpr int numStreams = 1 + numextra;

            std::array<CudaStream, numStreams> streams{}; //this will cause issues when rmm tries to synchronize the streams to free cached memory
            int streamIndex = 0;

            std::array<GpuErrorCorrectorInput, 1 + numextra> inputArray;
            SingleProducerSingleConsumerQueue<GpuErrorCorrectorInput*> freeInputsQueue;

            for(auto& a : inputArray){
                freeInputsQueue.push(&a);
            }


            std::array<GpuErrorCorrectorRawOutput, 2 + numextra> rawOutputArray;
            //SingleProducerSingleConsumerQueue<GpuErrorCorrectorRawOutput*> freeRawOutputQueue;
            MultiProducerMultiConsumerQueue<GpuErrorCorrectorRawOutput*> freeRawOutputQueue;

            for(auto& a : rawOutputArray){
                freeRawOutputQueue.push(&a);
            }

            SingleProducerSingleConsumerQueue<std::pair<GpuErrorCorrectorInput*,
                GpuErrorCorrectorRawOutput*>> dataInFlight;

            AnchorHasher gpuAnchorHasher(
                *readStorage,
                *minhasher,
                threadPool,
                mr
            );

            GpuErrorCorrector gpuErrorCorrector{
                *readStorage,
                correctionFlags,
                programOptions,
                programOptions.batchsize,
                mr,
                threadPool,
                gpuForestAnchor,
                gpuForestCandidate
            };

            int iterations = 0;
            std::vector<double> elapsedHashingTimes;
            std::vector<double> elapsedCorrectionTimes;
            std::vector<double> elapsedOutputTimes;

            double elapsedHashingTime = 0.0;
            double elapsedCorrectionTime = 0.0;

            std::atomic_int outstandingProcessing{0};

            auto processDataInFlight = [&](std::pair<GpuErrorCorrectorInput*, GpuErrorCorrectorRawOutput*> pointers){
                auto inputPtr = pointers.first;
                auto rawOutputPtr = pointers.second;

                CUDACHECK(inputPtr->event.synchronize());
                freeInputsQueue.push(inputPtr);

                CUDACHECK(rawOutputPtr->event.synchronize());

                outstandingProcessing++;
                processSerializedResults(
                    rawOutputPtr,
                    [&](GpuErrorCorrectorRawOutput* ptr){
                        freeRawOutputQueue.push(ptr);
                        outstandingProcessing--;
                    }
                );
            };

            RunStatistics runStatistics;

            std::vector<read_number> anchorIds(programOptions.batchsize);

            int globalcounter = 0;

            for(int i = 0; i < numextra; i++){
                if(continueCondition()){
                    cudaStream_t stream = streams[streamIndex];
                    streamIndex = (streamIndex + 1) % numStreams;

                    helpers::CpuTimer hashingTimer;
                
                    anchorIds.resize(programOptions.batchsize);
                    readIdGenerator.process_next_n(
                        programOptions.batchsize, 
                        [&](auto begin, auto end){
                            auto readIdsEnd = std::copy(begin, end, anchorIds.begin());
                            anchorIds.erase(readIdsEnd, anchorIds.end());
                        }
                    ); 

                    if(anchorIds.size() == 0){
                        continue;
                    }

                    GpuErrorCorrectorInput* inputPtr = freeInputsQueue.pop();
                    assert(inputPtr != nullptr);

                    GpuErrorCorrectorRawOutput* rawOutputPtr = freeRawOutputQueue.pop();
            
                    nvtx::push_range("makeErrorCorrectorInput", 0);
                    gpuAnchorHasher.makeErrorCorrectorInput(
                        anchorIds.data(),
                        anchorIds.size(),
                        programOptions.useQualityScores,
                        *inputPtr,
                        stream
                    );
                    nvtx::pop_range();

                    CUDACHECK(inputPtr->event.synchronize());

                    globalcounter++;

                    hashingTimer.stop();
                    elapsedHashingTime += hashingTimer.elapsed();

                    nvtx::push_range("correct", 1);
                    gpuErrorCorrector.correct(*inputPtr, *rawOutputPtr, stream);
                    nvtx::pop_range();

                    gpuErrorCorrector.releaseCandidateMemory(stream);

                    dataInFlight.push(std::make_pair(inputPtr, rawOutputPtr));

                    batchCompleted(anchorIds.size());
                    iterations++;
                }
            }


            while(continueCondition()){            
                
                anchorIds.resize(programOptions.batchsize);
                readIdGenerator.process_next_n(
                    programOptions.batchsize, 
                    [&](auto begin, auto end){
                        auto readIdsEnd = std::copy(begin, end, anchorIds.begin());
                        anchorIds.erase(readIdsEnd, anchorIds.end());
                    }
                ); 

                if(anchorIds.size() > 0){
                    cudaStream_t stream = streams[streamIndex];
                    streamIndex = (streamIndex + 1) % numStreams;

                    helpers::CpuTimer hashingTimer;

                    GpuErrorCorrectorInput* const inputPtr = freeInputsQueue.pop();

                    assert(cudaSuccess == inputPtr->event.query());

                    nvtx::push_range("makeErrorCorrectorInput", 0);
                    gpuAnchorHasher.makeErrorCorrectorInput(
                        anchorIds.data(),
                        anchorIds.size(),
                        programOptions.useQualityScores,
                        *inputPtr,
                        stream
                    );
                    nvtx::pop_range();

                    CUDACHECK(inputPtr->event.synchronize());

                    hashingTimer.stop();
                    elapsedHashingTime += hashingTimer.elapsed();

                    GpuErrorCorrectorRawOutput* rawOutputPtr = freeRawOutputQueue.pop();

                    nvtx::push_range("correct", 1);
                    gpuErrorCorrector.correct(*inputPtr, *rawOutputPtr, stream);
                    nvtx::pop_range();

                    gpuErrorCorrector.releaseCandidateMemory(stream);

                    dataInFlight.push(std::make_pair(inputPtr, rawOutputPtr));

                    auto pointers = dataInFlight.pop();
                    processDataInFlight(pointers);
                }

                batchCompleted(anchorIds.size());

                iterations++;
            }

            //signal output constructor thread
            dataInFlight.push(std::make_pair(nullptr, nullptr));

            //process remaining cached results
            auto pointers = dataInFlight.pop();
            while(pointers.first != nullptr){
                processDataInFlight(pointers);
                pointers = dataInFlight.pop();
            }

            while(outstandingProcessing != 0){};

            
            runStatistics.hasherTimeAverage = elapsedHashingTime / iterations;
            runStatistics.correctorTimeAverage = elapsedCorrectionTime / iterations;
            runStatistics.memoryHasher = gpuAnchorHasher.getMemoryInfo();
            runStatistics.memoryCorrector = gpuErrorCorrector.getMemoryInfo();

            MemoryUsage memoryInputData{};        
            for(int i = 0; i < 1 + numextra; i++){
                auto ptr = freeInputsQueue.pop();
                memoryInputData += ptr->getMemoryInfo();
            }
            runStatistics.memoryInputData = memoryInputData;

            return runStatistics;
        }catch (const rmm::bad_alloc& e){
            std::cerr << e.what() << "\n";
            std::exit(EXIT_FAILURE);
        }catch( ... ){
            std::cerr << "Caught exception\n"; std::exit(EXIT_FAILURE);
        }

        // std::cerr << "hashing times: ";
        // for(auto d : elapsedHashingTimes) std::cerr << d << ", ";
        // std::cerr << "\n";
        // //std::cerr << "Average: " << std::accumulate(elapsedHashingTimes.begin(), elapsedHashingTimes.end(), 0.0) / iterations << "\n";
        // std::cerr << "Average: " << elapsedHashingTime / iterations << "\n";

        // std::cerr << "correction times: ";
        // for(auto d : elapsedCorrectionTimes) std::cerr << d << ", ";
        // std::cerr << "\n";
        // //std::cerr << "Average: " << std::accumulate(elapsedCorrectionTimes.begin(), elapsedCorrectionTimes.end(), 0.0) / iterations << "\n";
        // std::cerr << "Average: " << elapsedCorrectionTime / iterations << "\n";

        // std::cerr << "output times: ";
        // for(auto d : elapsedOutputTimes) std::cerr << d << ", ";
        // std::cerr << "\n";
        // //std::cerr << "Average: " << std::accumulate(elapsedOutputTimes.begin(), elapsedOutputTimes.end(), 0.0) / iterations << "\n";
        // std::cerr << "Average: " << elapsedOutputTime / iterations << "\n";
    }

    template<class IdGenerator, class BatchCompletion, class ContinueCondition, class ResultProcessorSerialized>
    RunStatistics run_impl(
        int deviceId,
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions,
        GpuReadCorrectionFlags& correctionFlags,
        BatchCompletion batchCompleted,
        ContinueCondition continueCondition,
        ResultProcessorSerialized processSerializedResults
    ) const {
        cub::SwitchDevice sd{deviceId};

        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();
        cudaMemPool_t memPool;
        CUDACHECK(cudaDeviceGetMemPool(&memPool, deviceId));

        struct UsageStatistics {
            std::uint64_t reserved;
            std::uint64_t reservedHigh;
            std::uint64_t used;
            std::uint64_t usedHigh;
        };

        try{
            //constexpr int numextra = 1;

            cudaStream_t stream = cudaStreamPerThread;
            GpuErrorCorrectorInput input;

            //std::unique_ptr<GpuErrorCorrectorRawOutput> rawOutputPtr = std::make_unique<GpuErrorCorrectorRawOutput>();

            AnchorHasher gpuAnchorHasher(
                *readStorage,
                *minhasher,
                threadPool,
                mr
            );

            GpuErrorCorrector gpuErrorCorrector{
                *readStorage,
                correctionFlags,
                programOptions,
                programOptions.batchsize,
                mr,
                threadPool,
                gpuForestAnchor,
                gpuForestCandidate
            };

            std::array<GpuErrorCorrectorRawOutput, 3> rawOutputArray;
            MultiProducerMultiConsumerQueue<GpuErrorCorrectorRawOutput*> freeRawOutputQueue;

            for(auto& a : rawOutputArray){
                freeRawOutputQueue.push(&a);
            }

            // auto printUsageStatistics = [&](){
            //     UsageStatistics statistics;
            //     cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent, &statistics.reserved);
            //     cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, &statistics.reservedHigh);
            //     cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemCurrent, &statistics.used);
            //     cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, &statistics.usedHigh);
            //     MemoryUsage memhasher = gpuAnchorHasher.getMemoryInfo();
            //     MemoryUsage memcorrector = gpuErrorCorrector.getMemoryInfo();
            //     std::cerr << "reserved: " << statistics.reserved << ", reservedHigh: " << statistics.reservedHigh << ", used: " << statistics.used << ", usedHigh: " << statistics.usedHigh << "\n";
            //     std::cerr << "memcorrector: " << memcorrector.host << " - " << memcorrector.device[0] << "\n";

            //     CUDACHECK(cudaStreamSynchronize(stream));
                
            //     cudaMemPool_t mempool;
            //     CUDACHECK(cudaDeviceGetMemPool(&mempool, deviceId));
            //     CUDACHECK(cudaMemPoolTrimTo(mempool, 0));
            // };

            RunStatistics runStatistics;

            std::vector<read_number> anchorIds(programOptions.batchsize);

            int iterations = 0;
            std::vector<double> elapsedHashingTimes;
            std::vector<double> elapsedCorrectionTimes;
            std::vector<double> elapsedOutputTimes;

            double elapsedHashingTime = 0.0;
            double elapsedCorrectionTime = 0.0;

            std::atomic_int outstandingProcessing{0};

            //int globalcounter = 0;

            while(continueCondition()){


                //printUsageStatistics();

                helpers::CpuTimer hashingTimer;
                
                anchorIds.resize(programOptions.batchsize);
                readIdGenerator.process_next_n(
                    programOptions.batchsize, 
                    [&](auto begin, auto end){
                        auto readIdsEnd = std::copy(begin, end, anchorIds.begin());
                        anchorIds.erase(readIdsEnd, anchorIds.end());
                    }
                );            

                if(anchorIds.size() > 0){

                    CUDACHECK(input.event.synchronize());
                    GpuErrorCorrectorRawOutput* rawOutputPtr = freeRawOutputQueue.pop();
            
                    nvtx::push_range("makeErrorCorrectorInput", 0);
                    gpuAnchorHasher.makeErrorCorrectorInput(
                        anchorIds.data(),
                        anchorIds.size(),
                        programOptions.useQualityScores,
                        input,
                        stream
                    );
                    nvtx::pop_range();

                    CUDACHECK(input.event.synchronize());

                    hashingTimer.stop();
                    if(iterations >= 10){
                        elapsedHashingTime += hashingTimer.elapsed();
                    }

                    helpers::CpuTimer correctionTimer;

                    nvtx::push_range("correct", 1);
                    gpuErrorCorrector.correct(input, *rawOutputPtr, stream);
                    nvtx::pop_range();

                    gpuErrorCorrector.releaseCandidateMemory(stream);

                    CUDACHECK(rawOutputPtr->event.synchronize());

                    correctionTimer.stop();
                    if(iterations >= 10){
                        elapsedCorrectionTime += correctionTimer.elapsed();
                    }

                    outstandingProcessing++;
                    processSerializedResults(
                        rawOutputPtr,
                        [&](GpuErrorCorrectorRawOutput* ptr){
                            freeRawOutputQueue.push(ptr);
                            outstandingProcessing--;
                        }
                    );
                }

                batchCompleted(anchorIds.size());

                iterations++;
            }

            const int timediterations = std::max(1, iterations - 10);

            runStatistics.hasherTimeAverage = elapsedHashingTime / timediterations;
            runStatistics.correctorTimeAverage = elapsedCorrectionTime / timediterations;
            runStatistics.memoryHasher = gpuAnchorHasher.getMemoryInfo();
            runStatistics.memoryCorrector = gpuErrorCorrector.getMemoryInfo();
            runStatistics.memoryInputData = input.getMemoryInfo();

            while(outstandingProcessing != 0){};

            return runStatistics;
        }catch (const rmm::bad_alloc& e){
            std::cerr << e.what() << "\n";
            std::exit(EXIT_FAILURE);
        }catch( ... ){
            std::cerr << "Caught exception\n"; std::exit(EXIT_FAILURE);
        }

        // std::cerr << "hashing times: ";
        // for(auto d : elapsedHashingTimes) std::cerr << d << ", ";
        // std::cerr << "\n";
        // //std::cerr << "Average: " << std::accumulate(elapsedHashingTimes.begin(), elapsedHashingTimes.end(), 0.0) / iterations << "\n";
        // std::cerr << "Average: " << elapsedHashingTime / iterations << "\n";

        // std::cerr << "correction times: ";
        // for(auto d : elapsedCorrectionTimes) std::cerr << d << ", ";
        // std::cerr << "\n";
        // //std::cerr << "Average: " << std::accumulate(elapsedCorrectionTimes.begin(), elapsedCorrectionTimes.end(), 0.0) / iterations << "\n";
        // std::cerr << "Average: " << elapsedCorrectionTime / iterations << "\n";

        // std::cerr << "output times: ";
        // for(auto d : elapsedOutputTimes) std::cerr << d << ", ";
        // std::cerr << "\n";
        // //std::cerr << "Average: " << std::accumulate(elapsedOutputTimes.begin(), elapsedOutputTimes.end(), 0.0) / iterations << "\n";
        // std::cerr << "Average: " << elapsedOutputTime / iterations << "\n";
    }

private:
    const GpuReadStorage* readStorage;
    const Minhasher* minhasher;
    ThreadPool* threadPool;
    const GpuForest* gpuForestAnchor;
    const GpuForest* gpuForestCandidate;
};

template<class Minhasher>
class SimpleMultiGpuCorrectionPipeline{    

    using AnchorHasher = GpuAnchorHasher;
public:
    struct RunStatistics{
        double hasherTimeAverage{};
        double correctorTimeAverage{};
        double outputconstructorTimeAverage{};
        MemoryUsage memoryInputData{};
        MemoryUsage memoryRawOutputData{};
        MemoryUsage memoryHasher{};
        MemoryUsage memoryCorrector{};
        MemoryUsage memoryOutputConstructor{};
    };

    SimpleMultiGpuCorrectionPipeline(
        const GpuReadStorage& readStorage_,
        const Minhasher& minhasher_,
        std::vector<int> deviceIds_,
        std::vector<const GpuForest*> vec_gpuForestAnchor_, 
        std::vector<const GpuForest*> vec_gpuForestCandidate_
    ) :
        readStorage(&readStorage_),
        minhasher(&minhasher_),
        vec_gpuForestAnchor(std::move(vec_gpuForestAnchor_)),
        vec_gpuForestCandidate(std::move(vec_gpuForestCandidate_)),
        deviceIds(std::move(deviceIds_))
    {

    }

    template<class IdGenerator, class BatchCompletion, class ResultProcessorSerialized>
    RunStatistics runToCompletion(
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions,
        GpuReadCorrectionFlags& correctionFlags,
        BatchCompletion batchCompleted,
        ResultProcessorSerialized processSerializedResults
    ) const {

        auto continueCondition = [&](){ return !readIdGenerator.empty(); };

        return run_impl(
            readIdGenerator,
            programOptions,
            correctionFlags,
            batchCompleted,
            continueCondition,
            processSerializedResults
        );
    }

    template<class IdGenerator, class BatchCompletion, class ResultProcessorSerialized>
    RunStatistics runSomeBatches(
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions,
        GpuReadCorrectionFlags& correctionFlags,
        BatchCompletion batchCompleted,
        int numBatches,
        ResultProcessorSerialized processSerializedResults
    ) const {

        auto continueCondition = [&](){ bool success = !readIdGenerator.empty() && numBatches > 0; numBatches--; return success;};

        return run_impl(
            readIdGenerator,
            programOptions,
            correctionFlags,
            batchCompleted,
            continueCondition,
            processSerializedResults
        );
    }


    template<class IdGenerator, class BatchCompletion, class ContinueCondition, class ResultProcessorSerialized>
    RunStatistics run_impl(
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions,
        GpuReadCorrectionFlags& correctionFlags,
        BatchCompletion batchCompleted,
        ContinueCondition continueCondition,
        ResultProcessorSerialized processSerializedResults
    ) const {
        const int numGpus = deviceIds.size();

        //std::vector<CudaStream> streamsObjects;
        std::vector<cudaStream_t> streams;
        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd{deviceIds[g]};
            //streamsObjects.emplace_back();
            //streams.push_back(streamsObjects.back().getStream());
            streams.push_back(cudaStreamPerThread);
        }


        try{

            MultiGpuErrorCorrectorInput input(deviceIds, streams);

            MultiGpuAnchorHasher gpuAnchorHasher(
                *readStorage,
                *minhasher,
                deviceIds
            );

            assert(programOptions.batchsize % 2 == 0);

            //distribute anchorIds amongst the gpus
            const int numIdPairs = programOptions.batchsize / 2;
            int maxNumIdPairsPerGpu = numIdPairs / numGpus;
            const int numLeftoverIdPairs = numIdPairs % numGpus;
            if(numLeftoverIdPairs > 0){
                maxNumIdPairsPerGpu++;
            }


            MultiGpuErrorCorrector gpuErrorCorrector(
                *readStorage,
                correctionFlags,
                programOptions,
                maxNumIdPairsPerGpu*2,
                deviceIds,
                vec_gpuForestAnchor,
                vec_gpuForestCandidate,
                streams
            );

            std::vector<std::vector<GpuErrorCorrectorRawOutput>> rawOutputArray;

            for(int i = 0; i < 3; i++){
                std::vector<GpuErrorCorrectorRawOutput> vec;
                for(int g = 0; g < numGpus; g++){
                    cub::SwitchDevice sd{deviceIds[g]};
                    vec.emplace_back();
                }
                rawOutputArray.push_back(std::move(vec));
            }

            MultiProducerMultiConsumerQueue<std::vector<GpuErrorCorrectorRawOutput>*> freeRawOutputQueue;

            for(auto& a : rawOutputArray){
                freeRawOutputQueue.push(&a);
            }

            RunStatistics runStatistics;

            std::vector<read_number> anchorIds(programOptions.batchsize);

            int iterations = 0;
            std::vector<double> elapsedHashingTimes;
            std::vector<double> elapsedCorrectionTimes;
            std::vector<double> elapsedOutputTimes;

            double elapsedHashingTime = 0.0;
            double elapsedCorrectionTime = 0.0;

            std::atomic_int outstandingProcessing{0};

            //int globalcounter = 0;

            while(continueCondition()){


                //printUsageStatistics();

                helpers::CpuTimer hashingTimer;
                
                anchorIds.resize(programOptions.batchsize);
                readIdGenerator.process_next_n(
                    programOptions.batchsize, 
                    [&](auto begin, auto end){
                        auto readIdsEnd = std::copy(begin, end, anchorIds.begin());
                        anchorIds.erase(readIdsEnd, anchorIds.end());
                    }
                );            

                if(anchorIds.size() > 0){
                    for(int g = 0; g < numGpus; g++){
                        cub::SwitchDevice sd{deviceIds[g]};
                        CUDACHECK(input.vec_event[g].synchronize());
                    }
                    
                    std::vector<GpuErrorCorrectorRawOutput>* rawOutputPtr = freeRawOutputQueue.pop();
            
                    nvtx::push_range("makeErrorCorrectorInput", 0);
                    gpuAnchorHasher.makeErrorCorrectorInput(
                        anchorIds.data(),
                        anchorIds.size(),
                        programOptions.useQualityScores,
                        input,
                        streams
                    );
                    nvtx::pop_range();

                    for(int g = 0; g < numGpus; g++){
                        cub::SwitchDevice sd{deviceIds[g]};
                        CUDACHECK(input.vec_event[g].synchronize());
                    }

                    hashingTimer.stop();
                    if(iterations >= 10){
                        elapsedHashingTime += hashingTimer.elapsed();
                    }

                    helpers::CpuTimer correctionTimer;

                    nvtx::push_range("correct", 1);
                    gpuErrorCorrector.correct(input, *rawOutputPtr, streams);
                    nvtx::pop_range();

                    //gpuErrorCorrector.releaseCandidateMemory(streams);

                    for(int g = 0; g < numGpus; g++){
                        cub::SwitchDevice sd{deviceIds[g]};
                        CUDACHECK((*rawOutputPtr)[g].event.synchronize());
                    }

                    correctionTimer.stop();
                    if(iterations >= 10){
                        elapsedCorrectionTime += correctionTimer.elapsed();
                    }

                    outstandingProcessing++;
                    processSerializedResults(
                        rawOutputPtr,
                        [&](std::vector<GpuErrorCorrectorRawOutput>* ptr){
                            freeRawOutputQueue.push(ptr);
                            outstandingProcessing--;
                        }
                    );
                }

                batchCompleted(anchorIds.size());

                iterations++;
            }

            const int timediterations = std::max(1, iterations - 10);

            runStatistics.hasherTimeAverage = elapsedHashingTime / timediterations;
            runStatistics.correctorTimeAverage = elapsedCorrectionTime / timediterations;
            runStatistics.memoryHasher = gpuAnchorHasher.getMemoryInfo();
            runStatistics.memoryCorrector = gpuErrorCorrector.getMemoryInfo();
            runStatistics.memoryInputData = input.getMemoryInfo();

            while(outstandingProcessing != 0){};

            return runStatistics;
        }catch (const rmm::bad_alloc& e){
            std::cerr << e.what() << "\n";
            std::exit(EXIT_FAILURE);
        }catch( ... ){
            std::cerr << "Caught exception\n"; std::exit(EXIT_FAILURE);
        }

        // std::cerr << "hashing times: ";
        // for(auto d : elapsedHashingTimes) std::cerr << d << ", ";
        // std::cerr << "\n";
        // //std::cerr << "Average: " << std::accumulate(elapsedHashingTimes.begin(), elapsedHashingTimes.end(), 0.0) / iterations << "\n";
        // std::cerr << "Average: " << elapsedHashingTime / iterations << "\n";

        // std::cerr << "correction times: ";
        // for(auto d : elapsedCorrectionTimes) std::cerr << d << ", ";
        // std::cerr << "\n";
        // //std::cerr << "Average: " << std::accumulate(elapsedCorrectionTimes.begin(), elapsedCorrectionTimes.end(), 0.0) / iterations << "\n";
        // std::cerr << "Average: " << elapsedCorrectionTime / iterations << "\n";

        // std::cerr << "output times: ";
        // for(auto d : elapsedOutputTimes) std::cerr << d << ", ";
        // std::cerr << "\n";
        // //std::cerr << "Average: " << std::accumulate(elapsedOutputTimes.begin(), elapsedOutputTimes.end(), 0.0) / iterations << "\n";
        // std::cerr << "Average: " << elapsedOutputTime / iterations << "\n";
    }

private:
    const GpuReadStorage* readStorage;
    const Minhasher* minhasher;
    std::vector<const GpuForest*> vec_gpuForestAnchor;
    std::vector<const GpuForest*> vec_gpuForestCandidate;
    std::vector<int> deviceIds;
};


template<class Minhasher>
class ComplexGpuCorrectionPipeline{
    using AnchorHasher = GpuAnchorHasher;
public:
    struct Config{
        int numHashers;
        int numCorrectors;
    };

    ComplexGpuCorrectionPipeline(
        const GpuReadStorage& readStorage_,
        const Minhasher& minhasher_,
        ThreadPool* threadPool_,
        const GpuForest* gpuForestAnchor_, 
        const GpuForest* gpuForestCandidate_
    ) :
        readStorage(&readStorage_),
        minhasher(&minhasher_),
        threadPool(threadPool_),
        gpuForestAnchor(gpuForestAnchor_),
        gpuForestCandidate(gpuForestCandidate_)
    {

    }

    template<class IdGenerator, class BatchCompletion, class ResultProcessorSerialized>
    void run(
        int deviceId,
        const Config& config,
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions,
        GpuReadCorrectionFlags& correctionFlags,
        BatchCompletion batchCompleted,
        ResultProcessorSerialized processSerializedResults
    ){
        cub::SwitchDevice sd{deviceId};

        noMoreInputs = false;
        activeHasherThreads = config.numHashers;
        activeCorrectorThreads = config.numCorrectors;
        currentConfig = config;

        int numBatches = config.numHashers + config.numCorrectors; // such that all hashers and all correctors could be busy simultaneously
        numBatches += config.numCorrectors; //double buffer in correctors

        int numInputBatches = config.numHashers + config.numCorrectors;
        numInputBatches += config.numCorrectors * getNumExtraBuffers();

        std::vector<GpuErrorCorrectorInput> inputs(numInputBatches);
        for(auto& i : inputs){
            freeInputs.push(&i);
        }

        std::vector<std::future<void>> futures;

        for(int i = 0; i < config.numHashers; i++){
            futures.emplace_back(
                std::async(
                    std::launch::async,
                    [&](){ 
                        hasherThreadFunction(deviceId, readIdGenerator, 
                            programOptions); 
                    }
                )
            );
        }

        for(int i = 0; i < config.numCorrectors; i++){
            futures.emplace_back(
                std::async(
                    std::launch::async,
                    [&](){ 
                        correctorThreadFunctionMultiBufferWithOutput(deviceId, 
                            programOptions, 
                            correctionFlags,
                            batchCompleted, 
                            processSerializedResults
                        );                          
                    }
                )
            );
        }            

        for(auto& future : futures){
            future.wait();
        }
    }
    
    template<class IdGenerator>
    void hasherThreadFunction(
        int deviceId,
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions
    ){
        cudaSetDevice(deviceId);

        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();

        //catch out-of-memory exceptions
        try{

            AnchorHasher gpuAnchorHasher(
                *readStorage,
                *minhasher,
                nullptr, //threadPool,
                mr
            );

            cudaStream_t hasherStream = cudaStreamPerThread;
            ThreadPool::ParallelForHandle pforHandle;


            while(!readIdGenerator.empty()){
                CUDACHECK(cudaStreamSynchronize(hasherStream));

                std::vector<read_number> anchorIds(programOptions.batchsize);

                readIdGenerator.process_next_n(
                    programOptions.batchsize, 
                    [&](auto begin, auto end){
                        auto readIdsEnd = std::copy(begin, end, anchorIds.begin());
                        anchorIds.erase(readIdsEnd, anchorIds.end());
                    }
                ); 

                nvtx::push_range("getFreeInput",1);
                GpuErrorCorrectorInput* const inputPtr = freeInputs.pop();
                nvtx::pop_range();

                assert(cudaSuccess == inputPtr->event.query());

                nvtx::push_range("makeErrorCorrectorInput", 0);
                gpuAnchorHasher.makeErrorCorrectorInput(
                    anchorIds.data(),
                    anchorIds.size(),
                    programOptions.useQualityScores,
                    *inputPtr,
                    hasherStream
                );
                nvtx::pop_range();

                CUDACHECK(inputPtr->event.synchronize());

                unprocessedInputs.push(inputPtr);
                
            }

            activeHasherThreads--;

            if(activeHasherThreads == 0){
                noMoreInputs = true;
                for(int i = 0; i < 2 * currentConfig.numCorrectors; i++){
                    unprocessedInputs.push(nullptr);
                }
            }

            CUDACHECK(cudaStreamSynchronize(hasherStream));
        }catch (const rmm::bad_alloc& e){
            std::cerr << e.what() << "\n";
            std::exit(EXIT_FAILURE);
        }catch( ... ){
            std::cerr << "Caught exception\n"; std::exit(EXIT_FAILURE);
        }

        // std::cerr << "Hasher memory usage\n";
        // {
        //     auto meminfo = gpuAnchorHasher.getMemoryInfo();
        //     std::cerr << "host: " << meminfo.host << ", ";
        //     for(auto d : meminfo.device){
        //         std::cerr << "device " << d.first << ": " << d.second << " ";
        //     }
        //     std::cerr << "\n";
        // }
    };

    static constexpr int getNumExtraBuffers() noexcept{
        return 1;
    }

    template<class BatchCompletion, class ResultProcessorSerialized>
    void correctorThreadFunctionMultiBufferWithOutput(
        int deviceId,
        const ProgramOptions& programOptions,
        GpuReadCorrectionFlags& correctionFlags,
        BatchCompletion batchCompleted,
        ResultProcessorSerialized processSerializedResults
    ){
        cudaSetDevice(deviceId);

        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();

        try{

            GpuErrorCorrector gpuErrorCorrector{
                *readStorage,
                correctionFlags,
                programOptions,
                programOptions.batchsize,
                mr,
                threadPool,
                gpuForestAnchor,
                gpuForestCandidate
            };

            ThreadPool::ParallelForHandle pforHandle;

            std::array<GpuErrorCorrectorRawOutput, 2 + getNumExtraBuffers()> rawOutputs{};
            MultiProducerMultiConsumerQueue<GpuErrorCorrectorRawOutput*> myFreeOutputsQueue;

            for(auto& i : rawOutputs){
                myFreeOutputsQueue.push(&i);
            }

            std::atomic_int outstandingProcessing{0};

            auto constructOutput = [&](GpuErrorCorrectorRawOutput* rawOutputPtr){

                outstandingProcessing++;
                processSerializedResults(
                    rawOutputPtr,
                    [&](GpuErrorCorrectorRawOutput* ptr){
                        myFreeOutputsQueue.push(ptr);
                        outstandingProcessing--;
                    }
                );

                batchCompleted(rawOutputPtr->numAnchors);
            };

            cudaStream_t stream = cudaStreamPerThread;

            std::queue<std::pair<GpuErrorCorrectorInput*,
                GpuErrorCorrectorRawOutput*>> dataInFlight;

            GpuErrorCorrectorInput* inputPtr = unprocessedInputs.pop(); 

            for(int preIters = 0; preIters < getNumExtraBuffers(); preIters++){

                if(inputPtr != nullptr){
                    nvtx::push_range("getFreeRawOutput",1);
                    GpuErrorCorrectorRawOutput* rawOutputPtr = myFreeOutputsQueue.pop();
                    nvtx::pop_range();

                    nvtx::push_range("correct", 0);
                    gpuErrorCorrector.correct(*inputPtr, *rawOutputPtr, stream);
                    nvtx::pop_range();

                    gpuErrorCorrector.releaseCandidateMemory(stream);

                    dataInFlight.emplace(inputPtr, rawOutputPtr);

                    inputPtr = unprocessedInputs.pop();
                }
            }

            while(inputPtr != nullptr){
                nvtx::push_range("getFreeRawOutput",1);
                GpuErrorCorrectorRawOutput* rawOutputPtr = myFreeOutputsQueue.pop();
                nvtx::pop_range();

                cudaError_t cstatus = cudaSuccess;
                cstatus = inputPtr->event.query();
                if(cstatus != cudaSuccess){
                    std::cerr << cudaGetErrorString(cstatus) << "\n";
                    assert(false);
                }
                cstatus = rawOutputPtr->event.query();
                if(cstatus != cudaSuccess){
                    std::cerr << cudaGetErrorString(cstatus) << "\n";
                    assert(false);
                }

                CUDACHECK(cudaStreamSynchronize(stream));

                // assert(cudaSuccess == inputPtr->event.query());
                // assert(cudaSuccess == rawOutputPtr->event.query());

                nvtx::push_range("correct", 0);
                gpuErrorCorrector.correct(*inputPtr, *rawOutputPtr, stream);
                nvtx::pop_range();

                gpuErrorCorrector.releaseCandidateMemory(stream);

                dataInFlight.emplace(inputPtr, rawOutputPtr);

                if(!dataInFlight.empty()){
                    auto pointers = dataInFlight.front();
                    dataInFlight.pop();

                    CUDACHECK(pointers.first->event.synchronize());
                    freeInputs.push(pointers.first);

                    CUDACHECK(pointers.second->event.synchronize());
                    constructOutput(pointers.second);
                }            

                nvtx::push_range("getUnprocessedInput",2);
                inputPtr = unprocessedInputs.pop();
                nvtx::pop_range();

            };

            //process outstanding buffered work
            while(!dataInFlight.empty()){
                auto pointers = dataInFlight.front();
                dataInFlight.pop();

                CUDACHECK(pointers.first->event.synchronize());
                freeInputs.push(pointers.first);

                CUDACHECK(pointers.second->event.synchronize());
                constructOutput(pointers.second);
            }

            while(outstandingProcessing != 0){};

            activeCorrectorThreads--;

            CUDACHECK(cudaStreamSynchronize(stream));
        }catch (const rmm::bad_alloc& e){
            std::cerr << e.what() << "\n";
            std::exit(EXIT_FAILURE);
        }catch( ... ){
            std::cerr << "Caught exception\n"; std::exit(EXIT_FAILURE);
        }
    };

private:
    const GpuReadStorage* readStorage;
    const Minhasher* minhasher;
    ThreadPool* threadPool;
    const GpuForest* gpuForestAnchor;
    const GpuForest* gpuForestCandidate;

    SimpleConcurrentQueue<GpuErrorCorrectorInput*> freeInputs;
    SimpleConcurrentQueue<GpuErrorCorrectorInput*> unprocessedInputs;

    std::atomic<bool> noMoreInputs{false};
    std::atomic<int> activeHasherThreads{0};
    std::atomic<int> activeCorrectorThreads{0};

    Config currentConfig;
};








template<class Minhasher>
class ComplexMultiGpuCorrectionPipeline{
    using AnchorHasher = GpuAnchorHasher;
public:
    struct Config{
        int numHashers;
        int numCorrectors;
    };

    ComplexMultiGpuCorrectionPipeline(
        const GpuReadStorage& readStorage_,
        const Minhasher& minhasher_,
        std::vector<int> deviceIds_,
        std::vector<const GpuForest*> vec_gpuForestAnchor_, 
        std::vector<const GpuForest*> vec_gpuForestCandidate_
    ) :
        readStorage(&readStorage_),
        minhasher(&minhasher_),
        vec_gpuForestAnchor(std::move(vec_gpuForestAnchor_)),
        vec_gpuForestCandidate(std::move(vec_gpuForestCandidate_)),
        deviceIds(std::move(deviceIds_))
    {

    }

    template<class IdGenerator, class BatchCompletion, class ResultProcessorSerialized>
    void run(
        const Config& config,
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions,
        GpuReadCorrectionFlags& correctionFlags,
        BatchCompletion batchCompleted,
        ResultProcessorSerialized processSerializedResults
    ){
        const int numGpus = deviceIds.size();

        noMoreInputs = false;
        activeHasherThreads = config.numHashers;
        activeCorrectorThreads = config.numCorrectors;
        currentConfig = config;

        int numInputBatches = config.numHashers + config.numCorrectors;  // such that all hashers and all correctors could be busy simultaneously

        std::vector<cudaStream_t> streams;
        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd{deviceIds[g]};
            streams.push_back(cudaStreamPerThread);
        }

        std::vector<std::unique_ptr<MultiGpuErrorCorrectorInput>> inputs(numInputBatches);
        for(int i = 0; i < numInputBatches; i++){
            inputs[i] = std::make_unique<MultiGpuErrorCorrectorInput>(deviceIds, streams);
        }
        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd{deviceIds[g]};
            CUDACHECK(cudaStreamSynchronize(streams[g]));
        }
        for(auto& i : inputs){
            freeInputs.push(i.get());
        }

        std::vector<std::future<void>> futures;

        for(int i = 0; i < config.numHashers; i++){
            futures.emplace_back(
                std::async(
                    std::launch::async,
                    [&](){ 
                        hasherThreadFunction(
                            readIdGenerator, 
                            programOptions
                        ); 
                    }
                )
            );
        }

        for(int i = 0; i < config.numCorrectors; i++){
            futures.emplace_back(
                std::async(
                    std::launch::async,
                    [&](){ 
                        correctorThreadFunctionMultiBufferWithOutput(
                            programOptions, 
                            correctionFlags,
                            batchCompleted, 
                            processSerializedResults
                        );                          
                    }
                )
            );
        }            

        for(auto& future : futures){
            future.wait();
        }
    }
    
    template<class IdGenerator>
    void hasherThreadFunction(
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions
    ){
        const int numGpus = deviceIds.size();

        std::vector<cudaStream_t> streams;
        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd{deviceIds[g]};
            streams.push_back(cudaStreamPerThread);
        }

        //catch out-of-memory exceptions
        try{

            MultiGpuAnchorHasher gpuAnchorHasher(
                *readStorage,
                *minhasher,
                deviceIds
            );

            std::vector<read_number> anchorIds(programOptions.batchsize);

            while(!readIdGenerator.empty()){
                anchorIds.resize(programOptions.batchsize);
                readIdGenerator.process_next_n(
                    programOptions.batchsize, 
                    [&](auto begin, auto end){
                        auto readIdsEnd = std::copy(begin, end, anchorIds.begin());
                        anchorIds.erase(readIdsEnd, anchorIds.end());
                    }
                );

                nvtx::push_range("getFreeInput",1);
                MultiGpuErrorCorrectorInput* const inputPtr = freeInputs.pop();
                nvtx::pop_range();

                nvtx::push_range("makeErrorCorrectorInput", 0);
                gpuAnchorHasher.makeErrorCorrectorInput(
                    anchorIds.data(),
                    anchorIds.size(),
                    programOptions.useQualityScores,
                    *inputPtr,
                    streams
                );
                nvtx::pop_range();

                for(int g = 0; g < numGpus; g++){
                    cub::SwitchDevice sd{deviceIds[g]};
                    CUDACHECK(cudaStreamSynchronize(streams[g]));
                }
                //std::cout << "hash push\n";

                unprocessedInputs.push(inputPtr);
                
            }

            activeHasherThreads--;

            if(activeHasherThreads == 0){
                noMoreInputs = true;
                for(int i = 0; i < 2 * currentConfig.numCorrectors; i++){
                    unprocessedInputs.push(nullptr);
                }
            }

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                CUDACHECK(cudaStreamSynchronize(streams[g]));
            }
        }catch (const rmm::bad_alloc& e){
            std::cerr << e.what() << "\n";
            std::exit(EXIT_FAILURE);
        }catch( ... ){
            std::cerr << "Caught exception\n"; std::exit(EXIT_FAILURE);
        }

        // std::cerr << "Hasher memory usage\n";
        // {
        //     auto meminfo = gpuAnchorHasher.getMemoryInfo();
        //     std::cerr << "host: " << meminfo.host << ", ";
        //     for(auto d : meminfo.device){
        //         std::cerr << "device " << d.first << ": " << d.second << " ";
        //     }
        //     std::cerr << "\n";
        // }
    };

    template<class BatchCompletion, class ResultProcessorSerialized>
    void correctorThreadFunctionMultiBufferWithOutput(
        const ProgramOptions& programOptions,
        GpuReadCorrectionFlags& correctionFlags,
        BatchCompletion batchCompleted,
        ResultProcessorSerialized processSerializedResults
    ){
        const int numGpus = deviceIds.size();

        std::vector<cudaStream_t> streams;
        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd{deviceIds[g]};
            //streamsObjects.emplace_back();
            //streams.push_back(streamsObjects.back().getStream());
            streams.push_back(cudaStreamPerThread);
        }

        try{

            assert(programOptions.batchsize % 2 == 0);

            //distribute anchorIds amongst the gpus
            const int numIdPairs = programOptions.batchsize / 2;
            int maxNumIdPairsPerGpu = numIdPairs / numGpus;
            const int numLeftoverIdPairs = numIdPairs % numGpus;
            if(numLeftoverIdPairs > 0){
                maxNumIdPairsPerGpu++;
            }

            MultiGpuErrorCorrector gpuErrorCorrector(
                *readStorage,
                correctionFlags,
                programOptions,
                maxNumIdPairsPerGpu*2,
                deviceIds,
                vec_gpuForestAnchor,
                vec_gpuForestCandidate,
                streams
            );

            std::vector<std::vector<GpuErrorCorrectorRawOutput>> rawOutputArray;

            for(int i = 0; i < 3; i++){
                std::vector<GpuErrorCorrectorRawOutput> vec;
                for(int g = 0; g < numGpus; g++){
                    cub::SwitchDevice sd{deviceIds[g]};
                    vec.emplace_back();
                }
                rawOutputArray.push_back(std::move(vec));
            }

            MultiProducerMultiConsumerQueue<std::vector<GpuErrorCorrectorRawOutput>*> freeRawOutputQueue;

            for(auto& a : rawOutputArray){
                freeRawOutputQueue.push(&a);
            }

            std::atomic_int outstandingProcessing{0};

            auto constructOutput = [&](std::vector<GpuErrorCorrectorRawOutput>* rawOutputPtr){

                outstandingProcessing++;
                processSerializedResults(
                    rawOutputPtr,
                    [&](std::vector<GpuErrorCorrectorRawOutput>* ptr){
                        freeRawOutputQueue.push(ptr);
                        outstandingProcessing--;
                    }
                );

                int n = 0;
                for(const auto& output : *rawOutputPtr){
                    n += output.numAnchors;
                }
                batchCompleted(n);
            };

            // for(int g = 0; g < numGpus; g++){
            //     cub::SwitchDevice sd{deviceIds[g]};
            //     CUDACHECK(cudaDeviceSynchronize());
            // }

            MultiGpuErrorCorrectorInput* inputPtr = unprocessedInputs.pop(); 
            //int foo1= 0;
            while(inputPtr != nullptr){
                //std::cout << "iter " << foo1++ << "\n";
                nvtx::push_range("getFreeRawOutput",1);
                std::vector<GpuErrorCorrectorRawOutput>* rawOutputPtr = freeRawOutputQueue.pop();
                nvtx::pop_range();

                nvtx::push_range("correct", 0);
                gpuErrorCorrector.correct(*inputPtr, *rawOutputPtr, streams);
                nvtx::pop_range();
                // for(int g = 0; g < numGpus; g++){
                //     cub::SwitchDevice sd{deviceIds[g]};
                //     CUDACHECK(cudaDeviceSynchronize());
                // }

                //wait for output
                for(int g = 0; g < numGpus; g++){
                    cub::SwitchDevice sd{deviceIds[g]};
                    CUDACHECK(cudaStreamSynchronize(streams[g]));
                }

                //gpuErrorCorrector.releaseCandidateMemory(streams);

                freeInputs.push(inputPtr);
                constructOutput(rawOutputPtr); 

                nvtx::push_range("getUnprocessedInput",2);
                inputPtr = unprocessedInputs.pop();
                nvtx::pop_range();

            };

            while(outstandingProcessing != 0){};

            activeCorrectorThreads--;

            for(int g = 0; g < numGpus; g++){
                cub::SwitchDevice sd{deviceIds[g]};
                CUDACHECK(cudaStreamSynchronize(streams[g]));
            }
        }catch (const rmm::bad_alloc& e){
            std::cerr << e.what() << "\n";
            std::exit(EXIT_FAILURE);
        }catch( ... ){
            std::cerr << "Caught exception\n"; std::exit(EXIT_FAILURE);
        }
    };

private:
    const GpuReadStorage* readStorage;
    const Minhasher* minhasher;
    std::vector<const GpuForest*> vec_gpuForestAnchor;
    std::vector<const GpuForest*> vec_gpuForestCandidate;
    std::vector<int> deviceIds;

    SimpleConcurrentQueue<MultiGpuErrorCorrectorInput*> freeInputs;
    SimpleConcurrentQueue<MultiGpuErrorCorrectorInput*> unprocessedInputs;

    std::atomic<bool> noMoreInputs{false};
    std::atomic<int> activeHasherThreads{0};
    std::atomic<int> activeCorrectorThreads{0};

    Config currentConfig;
};



template<class Minhasher>
SerializedObjectStorage correct_gpu_impl(
    const ProgramOptions& programOptions,
    Minhasher& minhasher,
    GpuReadStorage& readStorage,
    const std::vector<GpuForest>& anchorForests,
    const std::vector<GpuForest>& candidateForests
){

    assert(programOptions.canUseGpu);
    //assert(programOptions.max_candidates > 0);
    assert(programOptions.deviceIds.size() > 0);

    const auto& deviceIds = programOptions.deviceIds;

    const auto rsMemInfo = readStorage.getMemoryInfo();
    const auto mhMemInfo = minhasher.getMemoryInfo();

    std::size_t memoryAvailableBytesHost = programOptions.memoryTotalLimit;

    if(memoryAvailableBytesHost > rsMemInfo.host){
        memoryAvailableBytesHost -= rsMemInfo.host;
    }else{
        memoryAvailableBytesHost = 0;
    }

    if(memoryAvailableBytesHost > mhMemInfo.host){
        memoryAvailableBytesHost -= mhMemInfo.host;
    }else{
        memoryAvailableBytesHost = 0;
    }

    //ReadCorrectionFlags correctionFlags(readStorage.getNumberOfReads());
    GpuReadCorrectionFlags correctionFlags(programOptions.deviceIds, readStorage.getNumberOfReads(), programOptions.directPeerAccess);

    // std::cerr << "Status flags per reads require " << correctionFlags.sizeInBytes() / 1024. / 1024. << " MB\n";

    // if(memoryAvailableBytesHost > correctionFlags.sizeInBytes()){
    //     memoryAvailableBytesHost -= correctionFlags.sizeInBytes();
    // }else{
    //     memoryAvailableBytesHost = 0;
    // }

    const std::size_t availableMemoryInBytes = memoryAvailableBytesHost; //getAvailableMemoryInKB() * 1024;
    std::size_t memoryForPartialResultsInBytes = 0;

    if(availableMemoryInBytes > 2*(std::size_t(1) << 30)){
        memoryForPartialResultsInBytes = availableMemoryInBytes - 2*(std::size_t(1) << 30);
    }

    std::cerr << "Partial results may occupy " << (memoryForPartialResultsInBytes /1024. / 1024. / 1024.) 
        << " GB in memory. Remaining partial results will be stored in temp directory. \n";

    const std::size_t memoryLimitData = memoryForPartialResultsInBytes * 0.75;
    const std::size_t memoryLimitOffsets = memoryForPartialResultsInBytes * 0.25;

    SerializedObjectStorage partialResults(memoryLimitData, memoryLimitOffsets, programOptions.tempdirectory + "/");

    //std::mutex outputstreamlock;

    BackgroundThread outputThread;

    auto processSerializedResultsThenCallback = [&](
        GpuErrorCorrectorRawOutput* results,
        auto whenDone
    ){
        const std::size_t numA = *results->h_numCorrectedAnchors;
        const std::size_t numC = results->numCorrectedCandidates;

        if(numA > 0 || numC > 0){
            auto outputFunction = [
                &,
                results,
                numA,
                numC,
                whenDone = std::move(whenDone)
            ](){

                nvtx::ScopedRange sr("outputFunction", 7);

                auto saveWithFlagCheck = [&](const std::uint8_t* encodedBegin, const std::uint8_t* encodedEnd){
                    ParsedEncodedFlags flags = EncodedTempCorrectedSequence::parseEncodedFlags(encodedBegin);
                    if(!(flags.hq && flags.useEdits && flags.numEdits == 0)){
                        partialResults.insert(encodedBegin, encodedEnd);
                    }
                };

                for(std::size_t i = 0; i < numA; i++){
                    const std::uint8_t* encodedBegin = results->serializedAnchorResults.data()
                        + results->serializedAnchorOffsets[i];
                    const std::uint8_t* encodedEnd = results->serializedAnchorResults.data()
                        + results->serializedAnchorOffsets[i+1];

                    saveWithFlagCheck(
                        encodedBegin,
                        encodedEnd
                    );
                }

                if(numC > 0){

                    // don't need to check flags for candidates. insert blob.
                    partialResults.bulkInsert(
                        results->serializedCandidateResults.data(),
                        results->serializedCandidateResults.data() + results->serializedCandidateOffsets[numC],
                        results->serializedCandidateOffsets.data(),
                        results->serializedCandidateOffsets.data() + numC
                    );

                }

                whenDone(results);
            };


            outputThread.enqueue(std::move(outputFunction));
            //outputFunction();
        }else{
            whenDone(results);
        }
    };

    auto processSerializedResultsThenCallback_vec = [&](
        std::vector<GpuErrorCorrectorRawOutput>* vec_results,
        auto whenDone
    ){
        size_t numA_total = 0;
        size_t numC_total = 0;
        for(int g = 0; g < int(vec_results->size()); g++){
            GpuErrorCorrectorRawOutput* results = &(*vec_results)[g];
            numA_total += *results->h_numCorrectedAnchors;
            numC_total += results->numCorrectedCandidates;
        }

        if(numA_total > 0 || numC_total > 0){
            auto outputFunction = [
                &,
                vec_results,
                whenDone = std::move(whenDone)
            ](){

                nvtx::ScopedRange sr("outputFunction", 7);

                for(int g = 0; g < int(vec_results->size()); g++){
                    GpuErrorCorrectorRawOutput* results = &(*vec_results)[g];
                    const std::size_t numA = *results->h_numCorrectedAnchors;
                    const std::size_t numC = results->numCorrectedCandidates;

                    auto saveWithFlagCheck = [&](const std::uint8_t* encodedBegin, const std::uint8_t* encodedEnd){
                        ParsedEncodedFlags flags = EncodedTempCorrectedSequence::parseEncodedFlags(encodedBegin);
                        if(!(flags.hq && flags.useEdits && flags.numEdits == 0)){
                            partialResults.insert(encodedBegin, encodedEnd);
                        }
                    };

                    for(std::size_t i = 0; i < numA; i++){
                        const std::uint8_t* encodedBegin = results->serializedAnchorResults.data()
                            + results->serializedAnchorOffsets[i];
                        const std::uint8_t* encodedEnd = results->serializedAnchorResults.data()
                            + results->serializedAnchorOffsets[i+1];

                        saveWithFlagCheck(
                            encodedBegin,
                            encodedEnd
                        );
                    }

                    if(numC > 0){

                        // don't need to check flags for candidates. insert blob.
                        partialResults.bulkInsert(
                            results->serializedCandidateResults.data(),
                            results->serializedCandidateResults.data() + results->serializedCandidateOffsets[numC],
                            results->serializedCandidateOffsets.data(),
                            results->serializedCandidateOffsets.data() + numC
                        );

                    }
                }

                whenDone(vec_results);
            };

            outputThread.enqueue(std::move(outputFunction));
        }else{
            whenDone(vec_results);
        }
    };


    outputThread.setMaximumQueueSize(std::max(16, programOptions.threads));

    outputThread.start();

    const std::size_t numReadsToProcess = getNumReadsToProcess(&readStorage, programOptions);

    IteratorRangeTraversal<thrust::counting_iterator<read_number>> readIdGenerator(
        thrust::make_counting_iterator<read_number>(0),
        thrust::make_counting_iterator<read_number>(0) + numReadsToProcess
    );

    auto showProgress = [&](std::int64_t totalCount, int seconds){
        if(programOptions.showProgress){

            int hours = seconds / 3600;
            seconds = seconds % 3600;
            int minutes = seconds / 60;
            seconds = seconds % 60;

            std::size_t numreads = numReadsToProcess;
            
            printf("Processed %10lu of %10lu reads (Runtime: %03d:%02d:%02d)\r",
            totalCount, numreads,
            hours, minutes, seconds);

            std::cout.flush();
        }
    };

    auto updateShowProgressInterval = [](auto duration){
        return duration;
    };

    ProgressThread<std::int64_t> progressThread(numReadsToProcess, showProgress, updateShowProgressInterval);

    auto batchCompleted = [&](int size){
        //std::cerr << "Add progress " << size << "\n";
        progressThread.addProgress(size);
    };

    auto runSimpleGpuPipeline = [&](int deviceId,
        const GpuForest* gpuForestAnchor, 
        const GpuForest* gpuForestCandidate
    ){
        SimpleGpuCorrectionPipeline<Minhasher> pipeline(
            readStorage,
            minhasher,
            nullptr, //&threadPool
            gpuForestAnchor,
            gpuForestCandidate
        );

        //pipeline.runToCompletionDoubleBufferedWithExtraThread(
        //pipeline.runToCompletionDoubleBuffered(
        pipeline.runToCompletion(
            deviceId,
            readIdGenerator,
            programOptions,
            correctionFlags,
            batchCompleted,
            processSerializedResultsThenCallback
        );
    };

    auto runSimpleMultiGpuPipeline = [&](
        const std::vector<int>& deviceIds,
        const std::vector<const GpuForest*>& anchorForestsPtrs,
        const std::vector<const GpuForest*>& candidateForestsPtrs
    ){

        SimpleMultiGpuCorrectionPipeline<Minhasher> pipeline(
            readStorage,
            minhasher,
            deviceIds,
            anchorForestsPtrs,
            candidateForestsPtrs
        );

        pipeline.runToCompletion(
            readIdGenerator,
            programOptions,
            correctionFlags,
            batchCompleted,
            processSerializedResultsThenCallback_vec
        );
    };

    auto runComplexGpuPipeline = [&](int deviceId, typename ComplexGpuCorrectionPipeline<Minhasher>::Config config,
        const GpuForest* gpuForestAnchor, 
        const GpuForest* gpuForestCandidate
    ){
        
        ComplexGpuCorrectionPipeline<Minhasher> pipeline(
            readStorage, 
            minhasher, 
            nullptr,
            gpuForestAnchor,
            gpuForestCandidate
        );

        pipeline.run(
            deviceId,
            config,
            readIdGenerator,
            programOptions,
            correctionFlags,
            batchCompleted,
            //processSerializedResults
            processSerializedResultsThenCallback
        );
    };

    auto runComplexMultiGpuPipeline = [&](
        typename ComplexMultiGpuCorrectionPipeline<Minhasher>::Config config,
        const std::vector<int>& deviceIds,
        const std::vector<const GpuForest*>& anchorForestsPtrs,
        const std::vector<const GpuForest*>& candidateForestsPtrs
    ){
        ComplexMultiGpuCorrectionPipeline<Minhasher> pipeline(
            readStorage, 
            minhasher, 
            deviceIds,
            anchorForestsPtrs,
            candidateForestsPtrs
        );

        pipeline.run(
            config,
            readIdGenerator,
            programOptions,
            correctionFlags,
            batchCompleted,
            processSerializedResultsThenCallback_vec
        );
    };

    if(programOptions.pureGpu){
        std::vector<const GpuForest*> anchorForestsPtrs;
        std::vector<const GpuForest*> candidateForestsPtrs;

        std::vector<int> deviceIds = programOptions.deviceIds;
        const int numGpus = deviceIds.size();

        for(int g = 0; g < numGpus; g++){
            cub::SwitchDevice sd{deviceIds[g]};

            anchorForestsPtrs.push_back(&anchorForests[g]);
            candidateForestsPtrs.push_back(&candidateForests[g]);
        }

        // const int numPipelines = std::max(1, programOptions.gpuCorrectorThreadConfig.numCorrectors);
        // std::vector<std::future<void>> futures;
        // for(int i = 0; i < numPipelines; i++){
        //     futures.emplace_back(
        //         std::async(std::launch::async, runSimpleMultiGpuPipeline, anchorForestsPtrs, candidateForestsPtrs)
        //     );
        // }
        // for(auto& f : futures){
        //     f.wait();
        // } 


        if(programOptions.gpuCorrectorThreadConfig.isAutomatic()){
            //Process a few batches on the first gpu to estimate runtime per step
            //These estimates will be used to spawn an appropriate number of threads for each gpu (assuming all gpus are similar)


            typename SimpleMultiGpuCorrectionPipeline<Minhasher>::RunStatistics runStatistics;

            {
                SimpleMultiGpuCorrectionPipeline<Minhasher> pipeline(
                    readStorage,
                    minhasher,
                    deviceIds,
                    anchorForestsPtrs,
                    candidateForestsPtrs
                );

                constexpr int numBatches = 50;

                runStatistics = pipeline.runSomeBatches(
                    readIdGenerator,
                    programOptions,
                    correctionFlags,
                    batchCompleted,
                    numBatches,
                    processSerializedResultsThenCallback_vec
                );                    
            }

            const int numHashersPerCorrectorByTime = std::ceil(runStatistics.hasherTimeAverage / runStatistics.correctorTimeAverage);
            std::cerr << runStatistics.hasherTimeAverage << " " << runStatistics.correctorTimeAverage << "\n";

            std::vector<std::future<void>> futures;

            const int numDevices = deviceIds.size();
            const int requiredNumThreadsForComplex = numHashersPerCorrectorByTime + (2 + 1 + 1);
            int availableThreads = programOptions.threads;

            if(minhasher.hasGpuTables()){
                constexpr int maxThreads = 4;
                const int use = std::min(maxThreads, availableThreads);
                std::cerr << "\nUsing " << use << " gpu hashtable pipelines\n";

                for(int i = 0; i < use; i++){
                    futures.emplace_back(std::async(
                        std::launch::async,
                        runSimpleMultiGpuPipeline,
                        deviceIds,
                        anchorForestsPtrs,
                        candidateForestsPtrs
                    ));
                }
            }else{

                int threadsForComplex = std::min(availableThreads, requiredNumThreadsForComplex);
                if(threadsForComplex > 3){
                    typename ComplexMultiGpuCorrectionPipeline<Minhasher>::Config pipelineConfig;

                    pipelineConfig.numCorrectors = 1;
                    threadsForComplex -= pipelineConfig.numCorrectors;                        
                    pipelineConfig.numHashers = std::max(1, std::min(threadsForComplex, numHashersPerCorrectorByTime));
                    threadsForComplex -= pipelineConfig.numHashers;
                    if(threadsForComplex > 0){
                        pipelineConfig.numCorrectors++;
                        threadsForComplex--;
                    }
                    pipelineConfig.numHashers += threadsForComplex;

                    std::cerr << "\nWill use " << pipelineConfig.numHashers << " hasher(s), "
                    << pipelineConfig.numCorrectors << " corrector(s) "
                    << "\n";
                    futures.emplace_back(
                        std::async(
                            std::launch::async,
                            runComplexMultiGpuPipeline,
                            pipelineConfig,
                            deviceIds,
                            anchorForestsPtrs,
                            candidateForestsPtrs
                        )
                    );
                    availableThreads -= pipelineConfig.numCorrectors;
                    availableThreads -= pipelineConfig.numHashers;
                }else{
                    std::cerr << "\nWill use " << threadsForComplex << " simple pipelines\n";

                    while(threadsForComplex > 0){
                        futures.emplace_back(std::async(
                            std::launch::async,
                            runSimpleMultiGpuPipeline,
                            deviceIds,
                            anchorForestsPtrs,
                            candidateForestsPtrs
                        ));

                        threadsForComplex--;
                    }
                }
            }

            for(auto& f : futures){
                f.wait();
            }        
        }else{
            std::vector<std::future<void>> futures;

            const int numDevices = deviceIds.size();
            const int numHashers = programOptions.gpuCorrectorThreadConfig.numHashers;
            const int numCorrectors = programOptions.gpuCorrectorThreadConfig.numCorrectors;

            if(numHashers == 0){

                std::cerr << "Use " << numCorrectors << " simple pipelines\n";
                for(int c = 0; c < numCorrectors; c++){
                    futures.emplace_back(std::async(
                        std::launch::async,
                        runSimpleMultiGpuPipeline,
                        deviceIds,
                        anchorForestsPtrs,
                        candidateForestsPtrs
                    ));
                }

            }else{
                typename ComplexMultiGpuCorrectionPipeline<Minhasher>::Config pipelineConfig;
                pipelineConfig.numCorrectors = numCorrectors;
                pipelineConfig.numHashers = numHashers;

                std::cerr << "Use complex pipeline with " << numCorrectors << ":" << numHashers << "\n";
                futures.emplace_back(std::async(
                    std::launch::async,
                    runComplexMultiGpuPipeline,
                    pipelineConfig,
                    deviceIds,
                    anchorForestsPtrs,
                    candidateForestsPtrs
                ));
            }

            for(auto& f : futures){
                f.wait();
            }
        }

    }else{


        if(programOptions.gpuCorrectorThreadConfig.isAutomatic()){
            //Process a few batches on the first gpu to estimate runtime per step
            //These estimates will be used to spawn an appropriate number of threads for each gpu (assuming all gpus are similar)


            typename SimpleGpuCorrectionPipeline<Minhasher>::RunStatistics runStatistics;

            {
                SimpleGpuCorrectionPipeline<Minhasher> pipeline(
                    readStorage,
                    minhasher,
                    nullptr, //&threadPool
                    &anchorForests[0],
                    &candidateForests[0]
                );

                constexpr int numBatches = 50;

                runStatistics = pipeline.runSomeBatches(
                    deviceIds[0],
                    readIdGenerator,
                    programOptions,
                    correctionFlags,
                    batchCompleted,
                    numBatches,
                    //processSerializedResults
                    processSerializedResultsThenCallback
                );   
                    
            }

            const int numHashersPerCorrectorByTime = std::ceil(runStatistics.hasherTimeAverage / runStatistics.correctorTimeAverage);
            std::cerr << runStatistics.hasherTimeAverage << " " << runStatistics.correctorTimeAverage << "\n";

            std::vector<std::future<void>> futures;

            const int numDevices = deviceIds.size();
            const int requiredNumThreadsForComplex = numHashersPerCorrectorByTime + (2 + 1 + 1);
            int availableThreads = programOptions.threads;

            if(minhasher.hasGpuTables()){
                constexpr int maxThreadsPerGpu = 4;
                std::vector<int> usedPerGpu(numDevices, 0);
                int usedTotal = 0;
                int current = 0;

                while(availableThreads > 0 && usedTotal < numDevices * maxThreadsPerGpu){
                    if(usedPerGpu[current] < maxThreadsPerGpu){    
                        futures.emplace_back(std::async(
                            std::launch::async,
                            runSimpleGpuPipeline,
                            deviceIds[current],
                            &anchorForests[current],
                            &candidateForests[current]
                        ));
                        availableThreads--;
                        usedTotal++;
                        usedPerGpu[current]++;
                    }
                    current = (current + 1) % numDevices;
                }

                for(int i = 0; i < numDevices; i++){
                    if(usedPerGpu[i] > 0){
                        std::cerr << "\nUsing " << usedPerGpu[i] << " gpu hashtable pipelines on device " << deviceIds[i] << "\n";
                    }
                }
            }else{

                const int maxThreadsPerDevice = SDIV(availableThreads, numDevices);
                for(int i = 0; i < numDevices; i++){
                    if(availableThreads > 0){
                        const int deviceId = deviceIds[i];
                        int threadsForDevice = std::min(maxThreadsPerDevice, std::min(availableThreads, requiredNumThreadsForComplex));
                        if(threadsForDevice > 3){
                            typename ComplexGpuCorrectionPipeline<Minhasher>::Config pipelineConfig;

                            pipelineConfig.numCorrectors = 1;
                            threadsForDevice -= pipelineConfig.numCorrectors;                        
                            pipelineConfig.numHashers = std::max(1, std::min(threadsForDevice, numHashersPerCorrectorByTime));
                            threadsForDevice -= pipelineConfig.numHashers;
                            if(threadsForDevice > 0){
                                pipelineConfig.numCorrectors++;
                                threadsForDevice--;
                            }
                            pipelineConfig.numHashers += threadsForDevice;

                            std::cerr << "\nWill use " << pipelineConfig.numHashers << " hasher(s), "
                            << pipelineConfig.numCorrectors << " corrector(s) "
                            << "on device " << deviceId << "\n";
                            futures.emplace_back(
                                std::async(
                                    std::launch::async,
                                    runComplexGpuPipeline,
                                    deviceId, pipelineConfig,
                                    &anchorForests[i],
                                    &candidateForests[i]
                                )
                            );
                            availableThreads -= pipelineConfig.numCorrectors;
                            availableThreads -= pipelineConfig.numHashers;
                        }else{
                            std::cerr << "\nWill use " << threadsForDevice << " simple pipelines on device " << deviceId << "\n";

                            while(threadsForDevice > 0){
                                futures.emplace_back(std::async(
                                    std::launch::async,
                                    runSimpleGpuPipeline,
                                    deviceId,
                                    &anchorForests[i],
                                    &candidateForests[i]
                                ));

                                threadsForDevice--;
                            }
                        }
                    }
                }

            }

            for(auto& f : futures){
                f.wait();
            }        
        }else{
            std::vector<std::future<void>> futures;

            const int numDevices = deviceIds.size();
            const int numHashers = programOptions.gpuCorrectorThreadConfig.numHashers;
            const int numCorrectors = programOptions.gpuCorrectorThreadConfig.numCorrectors;

            if(numHashers == 0){
                for(int d = 0; d < numDevices; d++){
                    std::cerr << "Use " << numCorrectors << " simple pipelines on device " << deviceIds[d] << "\n";
                    for(int c = 0; c < numCorrectors; c++){
                        futures.emplace_back(std::async(
                            std::launch::async,
                            runSimpleGpuPipeline,
                            deviceIds[d],
                            &anchorForests[d],
                            &candidateForests[d]
                        ));
                    }
                }
            }else{
                typename ComplexGpuCorrectionPipeline<Minhasher>::Config pipelineConfig;
                pipelineConfig.numCorrectors = numCorrectors;
                pipelineConfig.numHashers = numHashers;

                for(int d = 0; d < numDevices; d++){
                    std::cerr << "Use complex pipeline with " << numCorrectors << ":" << numHashers 
                        << " on device " << deviceIds[d] << "\n";
                    futures.emplace_back(std::async(
                        std::launch::async,
                        runComplexGpuPipeline,
                        deviceIds[d],
                        pipelineConfig,
                        &anchorForests[d],
                        &candidateForests[d]
                    ));
                }
            }

            for(auto& f : futures){
                f.wait();
            }
        }
    }

    progressThread.finished(); 
        
    std::cout << std::endl;

    outputThread.stopThread(BackgroundThread::StopType::FinishAndStop);

    return partialResults;
}

SerializedObjectStorage correct_gpu(
    const ProgramOptions& programOptions,
    GpuMinhasher& minhasher,
    GpuReadStorage& readStorage,
    const std::vector<GpuForest>& anchorForests,
    const std::vector<GpuForest>& candidateForests
){

    return correct_gpu_impl(
        programOptions,
        minhasher,
        readStorage,
        anchorForests,
        candidateForests
    );
}


}
}

