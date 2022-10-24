



#include <gpu/gpucorrector.cuh>
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

//#define NORMAL_RESULTS

namespace care{
namespace gpu{


#if 0

//TODO classifier?
class SimpleCpuCorrectionPipeline{
    template<class T>
    using HostContainer = helpers::SimpleAllocationPinnedHost<T, 0>;

public:
    template<class ResultProcessor, class BatchCompletion>
    void runToCompletion(
        cpu::RangeGenerator<read_number>& readIdGenerator,
        const CorrectionOptions& programOptions,
        const GoodAlignmentProperties& programOptions,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted
    ) const {
        assert(false);
#if 0                
        //const int threadId = omp_get_thread_num();

        const std::size_t encodedSequencePitchInInts2Bit = SequenceHelpers::getEncodedNumInts2Bit(gpuReadStorage->getSequenceLengthUpperBound());
        const std::size_t decodedSequencePitchInBytes = gpuReadStorage->getSequenceLengthUpperBound();
        const std::size_t qualityPitchInBytes = gpuReadStorage->getSequenceLengthUpperBound();

        CpuErrorCorrector errorCorrector(
            encodedSequencePitchInInts2Bit,
            decodedSequencePitchInBytes,
            qualityPitchInBytes,
            programOptions,
            programOptions,
            *candidateIdsProvider,
            *readProvider,
            correctionFlags
        );

        HostContainer<read_number> batchReadIds(programOptions.batchsize);
        HostContainer<unsigned int> batchEncodedData(programOptions.batchsize * encodedSequencePitchInInts2Bit);
        HostContainer<char> batchQualities(programOptions.batchsize * qualityPitchInBytes);
        HostContainer<int> batchReadLengths(programOptions.batchsize);

        std::vector<read_number> tmpids(programOptions.batchsize);

        while(!(readIdGenerator.empty())){
            tmpids.resize(programOptions.batchsize);            

            auto readIdsEnd = readIdGenerator.next_n_into_buffer(
                programOptions.batchsize, 
                tmpids.begin()
            );
            
            tmpids.erase(readIdsEnd, tmpids.end());

            if(tmpids.empty()){
                continue;
            }

            const int numAnchors = tmpids.size();

            batchReadIds.resize(numAnchors);
            std::copy(tmpids.begin(), tmpids.end(), batchReadIds.begin());

            //collect input data of all reads in batch
            readProvider->setReadIds(batchReadIds.data(), batchReadIds.size());

            readProvider->gatherSequenceLengths(
                batchReadLengths.data()
            );

            readProvider->gatherSequenceData(
                batchEncodedData.data(),
                encodedSequencePitchInInts2Bit
            );

            if(programOptions.useQualityScores){
                readProvider->gatherSequenceQualities(
                    batchQualities.data(),
                    qualityPitchInBytes
                );
            }

            CpuErrorCorrector::MultiCorrectionInput input;
            input.anchorLengths.insert(input.anchorLengths.end(), batchReadLengths.begin(), batchReadLengths.end());
            input.anchorReadIds.insert(input.anchorReadIds.end(), batchReadIds.begin(), batchReadIds.end());

            input.encodedAnchors.resize(numAnchors);            
            for(int i = 0; i < numAnchors; i++){
                input.encodedAnchors[i] = batchEncodedData.data() + encodedSequencePitchInInts2Bit * i;
            }

            if(programOptions.useQualityScores){
                input.anchorQualityscores.resize(numAnchors);
                for(int i = 0; i < numAnchors; i++){
                    input.anchorQualityscores[i] = batchQualities.data() + qualityPitchInBytes * i;
                }
            }
            
            auto errorCorrectorOutputVector = errorCorrector.processMulti(input);
            
            CorrectionOutput correctionOutput;

            for(auto& output : errorCorrectorOutputVector){
                if(output.hasAnchorCorrection){
                    correctionOutput.encodedAnchorCorrections.emplace_back(output.anchorCorrection.encode());
                    correctionOutput.anchorCorrections.emplace_back(std::move(output.anchorCorrection));
                }

                for(auto& tmp : output.candidateCorrections){
                    correctionOutput.encodedCandidateCorrections.emplace_back(tmp.encode());
                    correctionOutput.candidateCorrections.emplace_back(std::move(tmp));
                }
            }

            processResults(std::move(correctionOutput));

            batchCompleted(batchReadIds.size()); 
            
        } //while unprocessed reads exist loop end   
#endif
    }
};
#endif

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

    template<class IdGenerator, class ResultProcessor, class BatchCompletion, class ResultProcessorSerialized>
    RunStatistics runToCompletion(
        int deviceId,
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted,
        ResultProcessorSerialized processSerializedResults
    ) const {

        auto continueCondition = [&](){ return !readIdGenerator.empty(); };

        return run_impl(
            deviceId,
            readIdGenerator,
            programOptions,
            correctionFlags,
            processResults,
            batchCompleted,
            continueCondition,
            processSerializedResults
        );
    }

    template<class IdGenerator, class ResultProcessor, class BatchCompletion, class ResultProcessorSerialized>
    RunStatistics runSomeBatches(
        int deviceId,
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
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
            processResults,
            batchCompleted,
            continueCondition,
            processSerializedResults
        );
    }

    template<class IdGenerator, class ResultProcessor, class BatchCompletion, class ResultProcessorSerialized>
    RunStatistics runToCompletionDoubleBuffered(
        int deviceId,
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted,
        ResultProcessorSerialized processSerializedResults
    ) const {

        constexpr bool useThreadForOutputConstruction = false;

        auto continueCondition = [&](){ return !readIdGenerator.empty(); };

        return runDoubleBuffered_impl(
            deviceId,
            readIdGenerator,
            programOptions,
            correctionFlags,
            useThreadForOutputConstruction,
            processResults,
            batchCompleted,
            continueCondition,
            processSerializedResults
        );
    }

    template<class IdGenerator, class ResultProcessor, class BatchCompletion, class ResultProcessorSerialized>
    RunStatistics runToCompletionDoubleBufferedWithExtraThread(
        int deviceId,
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted,
        ResultProcessorSerialized processSerializedResults
    ) const {

        constexpr bool useThreadForOutputConstruction = true;

        auto continueCondition = [&](){ return !readIdGenerator.empty(); };

        return runDoubleBuffered_impl(
            deviceId,
            readIdGenerator,
            programOptions,
            correctionFlags,
            useThreadForOutputConstruction,
            processResults,
            batchCompleted,
            continueCondition,
            processSerializedResults
        );
    }

    template<class IdGenerator, class ResultProcessor, class BatchCompletion, class ResultProcessorSerialized>
    RunStatistics runSomeBatchesDoubleBuffered(
        int deviceId,
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted,
        int numBatches,
        ResultProcessorSerialized processSerializedResults
    ) const {

        constexpr bool useThreadForOutputConstruction = false;

        auto continueCondition = [&](){ bool success = !readIdGenerator.empty() && numBatches > 0; numBatches--; return success;};

        return runDoubleBuffered_impl(
            deviceId,
            readIdGenerator,
            programOptions,
            correctionFlags,
            useThreadForOutputConstruction,
            processResults,
            batchCompleted,
            continueCondition,
            processSerializedResults
        );
    }

    template<class IdGenerator, class ResultProcessor, class BatchCompletion, class ResultProcessorSerialized>
    RunStatistics runSomeBatchesDoubleBufferedWithExtraThread(
        int deviceId,
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted,
        int numBatches,
        ResultProcessorSerialized processSerializedResults
    ) const {

        constexpr bool useThreadForOutputConstruction = true;

        auto continueCondition = [&](){ bool success = !readIdGenerator.empty() && numBatches > 0; numBatches--; return success;};

        return runDoubleBuffered_impl(
            deviceId,
            readIdGenerator,
            programOptions,
            correctionFlags,
            useThreadForOutputConstruction,
            processResults,
            batchCompleted,
            continueCondition,
            processSerializedResults
        );
    }

    template<class IdGenerator, class ResultProcessor, class BatchCompletion, class ContinueCondition, class ResultProcessorSerialized>
    RunStatistics runDoubleBuffered_impl(
        int deviceId,
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions,
        ReadCorrectionFlags& correctionFlags,
        bool useThreadForOutputConstruction,
        ResultProcessor processResults,
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


            std::array<GpuErrorCorrectorRawOutput, 1 + numextra> rawOutputArray;
            SingleProducerSingleConsumerQueue<GpuErrorCorrectorRawOutput*> freeRawOutputQueue;

            for(auto& a : rawOutputArray){
                freeRawOutputQueue.push(&a);
            }

            SingleProducerSingleConsumerQueue<std::pair<GpuErrorCorrectorInput*,
                GpuErrorCorrectorRawOutput*>> dataInFlight;

            SequentialForLoopExecutor forLoopExecutor;

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

            OutputConstructor outputConstructor(            
                correctionFlags,
                programOptions
            );

            int iterations = 0;
            std::vector<double> elapsedHashingTimes;
            std::vector<double> elapsedCorrectionTimes;
            std::vector<double> elapsedOutputTimes;

            double elapsedHashingTime = 0.0;
            double elapsedCorrectionTime = 0.0;
            double elapsedOutputTime = 0.0;

            auto processDataInFlight = [&](std::pair<GpuErrorCorrectorInput*, GpuErrorCorrectorRawOutput*> pointers){
                auto inputPtr = pointers.first;
                auto rawOutputPtr = pointers.second;

                //std::cerr << "processDataInFlight (" << inputPtr << ", " << rawOutputPtr << ")\n";

                CUDACHECK(inputPtr->event.synchronize());
                freeInputsQueue.push(inputPtr);

                CUDACHECK(rawOutputPtr->event.synchronize());
                //std::cerr << "Synchronized output " << pointers.second << "\n";

                helpers::CpuTimer outputTimer;

                #ifdef NORMAL_RESULTS

                nvtx::push_range("constructEncodedResults", 2);
                //helpers::CpuTimer constructResultsTimer("constructEncodedResults");
                EncodedCorrectionOutput encodedCorrectionOutput = outputConstructor.constructEncodedResults(*rawOutputPtr, forLoopExecutor);
                // constructResultsTimer.stop();
                // constructResultsTimer.print();
                nvtx::pop_range();

                freeRawOutputQueue.push(rawOutputPtr);
                
                outputTimer.stop();
                //elapsedOutputTimes.emplace_back(outputTimer.elapsed());
                elapsedOutputTime += outputTimer.elapsed();

                processResults(
                    std::move(encodedCorrectionOutput)
                );

                #else

                nvtx::push_range("constructSerializedEncodedResults", 2);
                //helpers::CpuTimer constructResultsTimer("constructEncodedResults");

                //SerializedEncodedCorrectionOutput serializedEncodedCorrectionOutput = outputConstructor.constructSerializedEncodedResults(*rawOutputPtr);
                SerializedEncodedCorrectionOutput serializedEncodedCorrectionOutput = outputConstructor.constructSerializedEncodedResultsFaster(*rawOutputPtr);
                

                // constructResultsTimer.stop();
                // constructResultsTimer.print();
                nvtx::pop_range();

                freeRawOutputQueue.push(rawOutputPtr);
                
                outputTimer.stop();
                //elapsedOutputTimes.emplace_back(outputTimer.elapsed());
                elapsedOutputTime += outputTimer.elapsed();

                processSerializedResults(
                    std::move(serializedEncodedCorrectionOutput)
                );



                #endif
            };

            std::future<void> outputConstructorFuture{};

            if(useThreadForOutputConstruction){
                auto outputConstructorThreadFunc = [&](){
                    std::pair<GpuErrorCorrectorInput*, GpuErrorCorrectorRawOutput*> pointers;
                    
                    pointers = dataInFlight.pop();

                    while(pointers.first != nullptr){
                        processDataInFlight(pointers);

                        pointers = dataInFlight.pop();
                    }
                };

                outputConstructorFuture = std::async(std::launch::async, outputConstructorThreadFunc);
            }

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

                    //std::cerr << "globalcounter " << globalcounter << "\n";
            
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
                    //elapsedHashingTimes.emplace_back(hashingTimer.elapsed());
                    elapsedHashingTime += hashingTimer.elapsed();

                    nvtx::push_range("correct", 1);
                    gpuErrorCorrector.correct(*inputPtr, *rawOutputPtr, stream);
                    nvtx::pop_range();

                    gpuErrorCorrector.releaseCandidateMemory(stream);

                    dataInFlight.push(std::make_pair(inputPtr, rawOutputPtr));

                    batchCompleted(anchorIds.size());
                    iterations++;
                    //std::cerr << "Added extra (" << inputPtr << ", " << rawOutputPtr << ")\n";
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

                    //std::cerr << "globalcounter " << globalcounter << "\n";

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

                    //globalcounter++;

                    hashingTimer.stop();
                    //elapsedHashingTimes.emplace_back(hashingTimer.elapsed());
                    elapsedHashingTime += hashingTimer.elapsed();

                    GpuErrorCorrectorRawOutput* rawOutputPtr = freeRawOutputQueue.pop();

                    nvtx::push_range("correct", 1);
                    gpuErrorCorrector.correct(*inputPtr, *rawOutputPtr, stream);
                    nvtx::pop_range();

                    gpuErrorCorrector.releaseCandidateMemory(stream);

                    dataInFlight.push(std::make_pair(inputPtr, rawOutputPtr));
                    //std::cerr << "Submitted (" << inputPtr << ", " << rawOutputPtr << ")\n";

                    if(!useThreadForOutputConstruction){
                        auto pointers = dataInFlight.pop();
                        processDataInFlight(pointers);
                    }
                }

                // if(!dataInFlight.empty()){
                //     processDataInFlight();
                // }

                batchCompleted(anchorIds.size());

                iterations++;
            }

            //signal output constructor thread
            dataInFlight.push(std::make_pair(nullptr, nullptr));

            if(!useThreadForOutputConstruction){
                //process remaining cached results
                auto pointers = dataInFlight.pop();
                while(pointers.first != nullptr){
                    processDataInFlight(pointers);
                    pointers = dataInFlight.pop();
                }

            }else{
                outputConstructorFuture.wait();
            }

            
            runStatistics.hasherTimeAverage = elapsedHashingTime / iterations;
            runStatistics.correctorTimeAverage = elapsedCorrectionTime / iterations;
            runStatistics.outputconstructorTimeAverage = elapsedOutputTime / iterations;
            runStatistics.memoryHasher = gpuAnchorHasher.getMemoryInfo();
            runStatistics.memoryCorrector = gpuErrorCorrector.getMemoryInfo();
            runStatistics.memoryOutputConstructor = outputConstructor.getMemoryInfo();

            MemoryUsage memoryInputData{};        
            for(int i = 0; i < 1 + numextra; i++){
                auto ptr = freeInputsQueue.pop();
                memoryInputData += ptr->getMemoryInfo();
            }
            runStatistics.memoryInputData = memoryInputData;
            //runStatistics.memoryRawOutputData = rawOutput.getMemoryInfo();

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

    template<class IdGenerator, class ResultProcessor, class BatchCompletion, class ContinueCondition, class ResultProcessorSerialized>
    RunStatistics run_impl(
        int deviceId,
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
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

            GpuErrorCorrectorRawOutput rawOutput;

            //ThreadPool::ParallelForHandle pforHandle;
            //ForLoopExecutor forLoopExecutor(threadPool, &pforHandle);
            SequentialForLoopExecutor forLoopExecutor;

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

            OutputConstructor outputConstructor(            
                correctionFlags,
                programOptions
            );

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
            double elapsedOutputTime = 0.0;

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

                    //std::cerr << "globalcounter " << globalcounter << "\n";
            
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

                    //globalcounter++;

                    hashingTimer.stop();
                    //elapsedHashingTimes.emplace_back(hashingTimer.elapsed());
                    if(iterations >= 10){
                        elapsedHashingTime += hashingTimer.elapsed();
                    }

                    helpers::CpuTimer correctionTimer;

                    nvtx::push_range("correct", 1);
                    gpuErrorCorrector.correct(input, rawOutput, stream);
                    nvtx::pop_range();

                    gpuErrorCorrector.releaseCandidateMemory(stream);

                    CUDACHECK(rawOutput.event.synchronize());

                    correctionTimer.stop();
                    //elapsedCorrectionTimes.emplace_back(correctionTimer.elapsed());
                    if(iterations >= 10){
                        elapsedCorrectionTime += correctionTimer.elapsed();
                    }

                    helpers::CpuTimer outputTimer;

                    #ifdef NORMAL_RESULTS
                    nvtx::push_range("constructEncodedResults", 2);
                    EncodedCorrectionOutput encodedCorrectionOutput = outputConstructor.constructEncodedResults(rawOutput, forLoopExecutor);
                    nvtx::pop_range();

                    outputTimer.stop();
                    //elapsedOutputTimes.emplace_back(outputTimer.elapsed());
                    if(iterations >= 10){
                        elapsedOutputTime += outputTimer.elapsed();
                    }

                    processResults(
                        std::move(encodedCorrectionOutput)
                    );

                    #else

                    nvtx::push_range("constructSerializedEncodedResults", 2);
                    //helpers::CpuTimer constructResultsTimer("constructSerializedEncodedResults");

                    //SerializedEncodedCorrectionOutput serializedEncodedCorrectionOutput = outputConstructor.constructSerializedEncodedResults(rawOutput);
                    SerializedEncodedCorrectionOutput serializedEncodedCorrectionOutput = outputConstructor.constructSerializedEncodedResultsFaster(rawOutput);

                    // constructResultsTimer.stop();
                    // constructResultsTimer.print();
                    nvtx::pop_range();

                    outputTimer.stop();
                    //elapsedOutputTimes.emplace_back(outputTimer.elapsed());
                    if(iterations >= 10){
                        elapsedOutputTime += outputTimer.elapsed();
                    }

                    processSerializedResults(
                        std::move(serializedEncodedCorrectionOutput)
                    );



                    #endif
                }

                batchCompleted(anchorIds.size());

                iterations++;
            }

            const int timediterations = std::max(1, iterations - 10);

            runStatistics.hasherTimeAverage = elapsedHashingTime / timediterations;
            runStatistics.correctorTimeAverage = elapsedCorrectionTime / timediterations;
            runStatistics.outputconstructorTimeAverage = elapsedOutputTime / timediterations;
            runStatistics.memoryHasher = gpuAnchorHasher.getMemoryInfo();
            runStatistics.memoryCorrector = gpuErrorCorrector.getMemoryInfo();
            runStatistics.memoryOutputConstructor = outputConstructor.getMemoryInfo();
            runStatistics.memoryInputData = input.getMemoryInfo();
            //runStatistics.memoryRawOutputData = rawOutput.getMemoryInfo();

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
class ComplexGpuCorrectionPipeline{
    using AnchorHasher = GpuAnchorHasher;
public:
    struct Config{
        int numHashers;
        int numCorrectors;
        int numOutputConstructors;
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

    template<class IdGenerator, class ResultProcessor, class BatchCompletion>
    void run(
        int deviceId,
        const Config& config,
        IdGenerator& readIdGenerator,
        const ProgramOptions& programOptions,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted
    ){
        cub::SwitchDevice sd{deviceId};

        noMoreInputs = false;
        activeHasherThreads = config.numHashers;
        activeCorrectorThreads = config.numCorrectors;
        currentConfig = config;

        bool combinedCorrectionAndOutputconstruction = config.numOutputConstructors == 0;

        if(combinedCorrectionAndOutputconstruction){

            int numBatches = config.numHashers + config.numCorrectors; // such that all hashers and all correctors could be busy simultaneously
            numBatches += config.numCorrectors; //double buffer in correctors

            int numInputBatches = config.numHashers + config.numCorrectors;
            numInputBatches += config.numCorrectors * getNumExtraBuffers();

            std::vector<GpuErrorCorrectorInput> inputs(numInputBatches);
            for(auto& i : inputs){
                freeInputs.push(&i);
            }

            //int numOutputBatches = config.numHashers + config.numCorrectors;
            //numOutputBatches += config.numCorrectors;

            // std::vector<GpuErrorCorrectorRawOutput> rawOutputs(numOutputBatches);
            // for(auto& i : rawOutputs){
            //     freeRawOutputs.push(&i);
            // }

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
                                processResults, batchCompleted);                          
                        }
                    )
                );
            }            

            for(auto& future : futures){
                future.wait();
            }

        }else{
            int numInputBatches = config.numHashers + config.numCorrectors; // such that all hashers and all correctors could be busy simultaneously

            std::vector<GpuErrorCorrectorInput> inputs(numInputBatches);
            for(auto& i : inputs){
                freeInputs.push(&i);
            }

            int numOutputBatches = config.numHashers + config.numCorrectors;

            std::vector<GpuErrorCorrectorRawOutput> rawOutputs(numOutputBatches);
            for(auto& i : rawOutputs){
                freeRawOutputs.push(&i);
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
                            correctorThreadFunction(
                                deviceId, 
                                programOptions, 
                                correctionFlags
                            );                          
                        }
                    )
                );
            }

            for(int i = 0; i < config.numOutputConstructors; i++){
                futures.emplace_back(
                    std::async(
                        std::launch::async,
                        [&](){ 
                            outputConstructorThreadFunction(programOptions, correctionFlags,
                                processResults, batchCompleted); 
                        }
                    )
                );
            }

            for(auto& future : futures){
                future.wait();
            }

        }

        // std::cerr << "input data sizes\n";
        // for(const auto& i : inputs){
        //     auto meminfo = i.getMemoryInfo();
        //     std::cerr << "host: " << meminfo.host << ", ";
        //     for(auto d : meminfo.device){
        //         std::cerr << "device " << d.first << ": " << d.second << " ";
        //     }
        //     std::cerr << "\n";
        // }

        // std::cerr << "output data sizes\n";
        // for(const auto& o : rawOutputs){
        //     auto meminfo = o.getMemoryInfo();
        //     std::cerr << "host: " << meminfo.host << ", ";
        //     for(auto d : meminfo.device){
        //         std::cerr << "device " << d.first << ": " << d.second << " ";
        //     }
        //     std::cerr << "\n";
        // }

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

    void correctorThreadFunction(
        int deviceId,
        const ProgramOptions& programOptions,
        const ReadCorrectionFlags& correctionFlags
    ){
        cudaSetDevice(deviceId);

        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();

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

        cudaStream_t stream = cudaStreamPerThread;

        // GpuErrorCorrectorInput* inputPtr = unprocessedInputs.popOrDefault(
        //     [&](){
        //         return !noMoreInputs;  //if noMoreInputs, return nullptr
        //     },
        //     nullptr
        // ); 

        GpuErrorCorrectorInput* inputPtr = unprocessedInputs.pop(); 

        while(inputPtr != nullptr){
            nvtx::push_range("getFreeRawOutput",1);
            GpuErrorCorrectorRawOutput* rawOutputPtr = freeRawOutputs.pop();
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

            CUDACHECK(inputPtr->event.synchronize());
            freeInputs.push(inputPtr);

            
            CUDACHECK(rawOutputPtr->event.synchronize());
            //std::cerr << "Synchronized output " << rawOutputPtr << "\n";
            unprocessedRawOutputs.push(rawOutputPtr);
        
            nvtx::push_range("getUnprocessedInput",2);
            // inputPtr = unprocessedInputs.popOrDefault(
            //     [&](){
            //         return !noMoreInputs;  //if noMoreInputs, return nullptr
            //     },
            //     nullptr
            // ); 
            inputPtr = unprocessedInputs.pop(); 
            nvtx::pop_range();

        };

        activeCorrectorThreads--;

        if(activeCorrectorThreads == 0){
            for(int i = 0; i < 2*currentConfig.numOutputConstructors; i++){
                unprocessedRawOutputs.push(nullptr);
            }
        }

        CUDACHECK(cudaStreamSynchronize(stream));
    };

    static constexpr int getNumExtraBuffers() noexcept{
        return 1;
    }

    template<class ResultProcessor, class BatchCompletion>
    void correctorThreadFunctionMultiBufferWithOutput(
        int deviceId,
        const ProgramOptions& programOptions,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted
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

            OutputConstructor outputConstructor(            
                correctionFlags,
                programOptions
            );

            ThreadPool::ParallelForHandle pforHandle;
            //ForLoopExecutor forLoopExecutor(&threadPool, &pforHandle);
            SequentialForLoopExecutor forLoopExecutor;

            std::array<GpuErrorCorrectorRawOutput, 1 + getNumExtraBuffers()> rawOutputs{};
            std::queue<GpuErrorCorrectorRawOutput*> myFreeOutputsQueue;

            for(auto& i : rawOutputs){
                myFreeOutputsQueue.push(&i);
            }

            auto constructOutput = [&](GpuErrorCorrectorRawOutput* rawOutputPtr){
                nvtx::push_range("constructEncodedResults", 2);
                EncodedCorrectionOutput encodedCorrectionOutput = outputConstructor.constructEncodedResults(*rawOutputPtr, forLoopExecutor);
                nvtx::pop_range();

                processResults(
                    std::move(encodedCorrectionOutput)
                );

                batchCompleted(rawOutputPtr->numAnchors); 

                myFreeOutputsQueue.push(rawOutputPtr);
            };

            cudaStream_t stream = cudaStreamPerThread;

            std::queue<std::pair<GpuErrorCorrectorInput*,
                GpuErrorCorrectorRawOutput*>> dataInFlight;

            // GpuErrorCorrectorInput* inputPtr = unprocessedInputs.popOrDefault(
            //     [&](){
            //         return !noMoreInputs;  //if noMoreInputs, return nullptr
            //     },
            //     nullptr
            // ); 

            GpuErrorCorrectorInput* inputPtr = unprocessedInputs.pop(); 

            for(int preIters = 0; preIters < getNumExtraBuffers(); preIters++){

                if(inputPtr != nullptr){
                    nvtx::push_range("getFreeRawOutput",1);
                    //GpuErrorCorrectorRawOutput* rawOutputPtr = freeRawOutputs.pop();
                    GpuErrorCorrectorRawOutput* rawOutputPtr = myFreeOutputsQueue.front();
                    myFreeOutputsQueue.pop();
                    nvtx::pop_range();

                    nvtx::push_range("correct", 0);
                    gpuErrorCorrector.correct(*inputPtr, *rawOutputPtr, stream);
                    nvtx::pop_range();

                    gpuErrorCorrector.releaseCandidateMemory(stream);

                    dataInFlight.emplace(inputPtr, rawOutputPtr);

                    // inputPtr = unprocessedInputs.popOrDefault(
                    //     [&](){
                    //         return !noMoreInputs;  //if noMoreInputs, return nullptr
                    //     },
                    //     nullptr
                    // ); 
                    inputPtr = unprocessedInputs.pop();
                }
            }

            while(inputPtr != nullptr){
                nvtx::push_range("getFreeRawOutput",1);
                //GpuErrorCorrectorRawOutput* rawOutputPtr = freeRawOutputs.pop();
                GpuErrorCorrectorRawOutput* rawOutputPtr = myFreeOutputsQueue.front();
                myFreeOutputsQueue.pop();
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
                    //std::cerr << "Synchronized output " << pointers.second << "\n";
                    constructOutput(pointers.second);
                }            

                nvtx::push_range("getUnprocessedInput",2);
                // inputPtr = unprocessedInputs.popOrDefault(
                //     [&](){
                //         return !noMoreInputs;  //if noMoreInputs, return nullptr
                //     },
                //     nullptr
                // ); 
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

            activeCorrectorThreads--;

            if(activeCorrectorThreads == 0){            
                for(int i = 0; i < 2*currentConfig.numOutputConstructors; i++){
                    unprocessedRawOutputs.push(nullptr);
                }
            }

            CUDACHECK(cudaStreamSynchronize(stream));
        }catch (const rmm::bad_alloc& e){
            std::cerr << e.what() << "\n";
            std::exit(EXIT_FAILURE);
        }catch( ... ){
            std::cerr << "Caught exception\n"; std::exit(EXIT_FAILURE);
        }
    };


    template<class ResultProcessor, class BatchCompletion>
    void outputConstructorThreadFunction(
        const ProgramOptions& programOptions,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted
    ){

        OutputConstructor outputConstructor(            
            correctionFlags,
            programOptions
        );

        ThreadPool::ParallelForHandle pforHandle;
        //ForLoopExecutor forLoopExecutor(&threadPool, &pforHandle);
        SequentialForLoopExecutor forLoopExecutor;

        GpuErrorCorrectorRawOutput* rawOutputPtr = unprocessedRawOutputs.pop();

        while(rawOutputPtr != nullptr){
            nvtx::push_range("constructEncodedResults", 2);
            EncodedCorrectionOutput encodedCorrectionOutput = outputConstructor.constructEncodedResults(*rawOutputPtr, forLoopExecutor);
            nvtx::pop_range();

            processResults(
                std::move(encodedCorrectionOutput)
            );

            batchCompleted(rawOutputPtr->numAnchors); 


            freeRawOutputs.push(rawOutputPtr);

            nvtx::push_range("getUnprocessedRawOutput", 2);

            rawOutputPtr = unprocessedRawOutputs.pop();

            nvtx::pop_range();
        }
    };

private:
    const GpuReadStorage* readStorage;
    const Minhasher* minhasher;
    ThreadPool* threadPool;
    const GpuForest* gpuForestAnchor;
    const GpuForest* gpuForestCandidate;

    SimpleSingleProducerSingleConsumerQueue<GpuErrorCorrectorInput*> freeInputs;
    SimpleSingleProducerSingleConsumerQueue<GpuErrorCorrectorInput*> unprocessedInputs;
    SimpleSingleProducerSingleConsumerQueue<GpuErrorCorrectorRawOutput*> freeRawOutputs;
    SimpleSingleProducerSingleConsumerQueue<GpuErrorCorrectorRawOutput*> unprocessedRawOutputs;

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

    ReadCorrectionFlags correctionFlags(readStorage.getNumberOfReads());

    std::cerr << "Status flags per reads require " << correctionFlags.sizeInBytes() / 1024. / 1024. << " MB\n";

    if(memoryAvailableBytesHost > correctionFlags.sizeInBytes()){
        memoryAvailableBytesHost -= correctionFlags.sizeInBytes();
    }else{
        memoryAvailableBytesHost = 0;
    }

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



    auto processResults = [&](
        EncodedCorrectionOutput&& encodedCorrectionOutput
    ){

        const std::size_t numA = encodedCorrectionOutput.encodedAnchorCorrections.size();
        const std::size_t numC = encodedCorrectionOutput.encodedCandidateCorrections.size();

        auto outputFunction = [
            &,
            encodedCorrectionOutput = std::move(encodedCorrectionOutput),
            numA,
            numC
        ](){
            std::vector<std::uint8_t> tempbuffer(256);

            //std::ofstream file("oldflags.txt", std::ios::app);
            //file << numA << "\n";

            auto saveEncodedCorrectedSequence = [&](const EncodedTempCorrectedSequence* encoded){
                //file << encoded->isHQ() << " " << encoded->useEdits() << " " << encoded->getNumEdits() << "\n";
                if(!(encoded->isHQ() && encoded->useEdits() && encoded->getNumEdits() == 0)){
                    const std::size_t serializedSize = encoded->getSerializedNumBytes();
                    tempbuffer.resize(serializedSize);

                    auto end = encoded->copyToContiguousMemory(tempbuffer.data(), tempbuffer.data() + tempbuffer.size());
                    assert(end != nullptr);

                    //file << std::distance(tempbuffer.data(), end) << "\n";

                    partialResults.insert(tempbuffer.data(), end);
                }
            };

            for(std::size_t i = 0; i < numA; i++){
                saveEncodedCorrectedSequence(
                    &encodedCorrectionOutput.encodedAnchorCorrections[i]
                );
            }

            for(std::size_t i = 0; i < numC; i++){
                saveEncodedCorrectedSequence(
                    &encodedCorrectionOutput.encodedCandidateCorrections[i]
                );
            }
        };

        if(numA > 0 || numC > 0){
            outputThread.enqueue(std::move(outputFunction));
            //outputFunction();
        }
    };


    auto processSerializedResults = [&](
        SerializedEncodedCorrectionOutput&& serializedEncodedCorrectionOutput
    ){

        const std::size_t numA = serializedEncodedCorrectionOutput.numAnchors;
        const std::size_t numC = serializedEncodedCorrectionOutput.numCandidates;

        auto outputFunction = [
            &,
            serializedEncodedCorrectionOutput = std::move(serializedEncodedCorrectionOutput),
            numA,
            numC
        ](){
            //std::ofstream file("newflags.txt", std::ios::app);
            //file << numA << "\n";

            auto saveEncodedCorrectedSequence = [&](const std::uint8_t* encodedBegin, const std::uint8_t* encodedEnd){
                ParsedEncodedFlags flags = EncodedTempCorrectedSequence::parseEncodedFlags(encodedBegin);

                //file << flags.hq << " " << flags.useEdits << " " << flags.numEdits << "\n";

                if(!(flags.hq && flags.useEdits && flags.numEdits == 0)){
                    //file << std::distance(encodedBegin, encodedEnd) << "\n";

                    partialResults.insert(encodedBegin, encodedEnd);
                }
            };

            for(std::size_t i = 0; i < numA; i++){
                const std::uint8_t* encodedBegin = serializedEncodedCorrectionOutput.serializedEncodedAnchorCorrections.data()
                    + serializedEncodedCorrectionOutput.beginOffsetsAnchors[i];
                const std::uint8_t* encodedEnd = serializedEncodedCorrectionOutput.serializedEncodedAnchorCorrections.data()
                    + serializedEncodedCorrectionOutput.beginOffsetsAnchors[i+1];

                saveEncodedCorrectedSequence(
                    encodedBegin,
                    encodedEnd
                );
            }

            for(std::size_t i = 0; i < numC; i++){
                const std::uint8_t* encodedBegin = serializedEncodedCorrectionOutput.serializedEncodedCandidateCorrections.data()
                    + serializedEncodedCorrectionOutput.beginOffsetsCandidates[i];
                const std::uint8_t* encodedEnd = serializedEncodedCorrectionOutput.serializedEncodedCandidateCorrections.data()
                    + serializedEncodedCorrectionOutput.beginOffsetsCandidates[i+1];

                saveEncodedCorrectedSequence(
                    encodedBegin,
                    encodedEnd
                );
            }
        };

        if(numA > 0 || numC > 0){
            outputThread.enqueue(std::move(outputFunction));
            //outputFunction();
        }
    };

    outputThread.setMaximumQueueSize(programOptions.threads);

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

    if(false /* && programOptions.threads <= 6*/){
        //execute a single thread pipeline with each available thread

        auto runPipeline = [&](int deviceId, 
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
    
            pipeline.runToCompletion(
                deviceId,
                readIdGenerator,
                programOptions,
                correctionFlags,
                processResults,
                batchCompleted,
                processSerializedResults
            );
        };
    
        std::vector<std::future<void>> futures;
    
        for(int i = 0; i < programOptions.threads; i++){
            const int position = i % deviceIds.size();
            const int deviceId = deviceIds[position];

            futures.emplace_back(std::async(
                std::launch::async,
                runPipeline,
                deviceId,
                &anchorForests[position],
                &candidateForests[position]
            ));
        }
    
        for(auto& f : futures){
            f.wait();
        }
    }else{

     

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
                processResults,
                batchCompleted,
                numBatches,
                processSerializedResults
            );   
                
        }

        const int numHashersPerCorrectorByTime = std::ceil(runStatistics.hasherTimeAverage / runStatistics.correctorTimeAverage);
        std::cerr << runStatistics.hasherTimeAverage << " " << runStatistics.correctorTimeAverage << "\n";

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
            pipeline.runToCompletionDoubleBuffered(
            //pipeline.runToCompletion(
                deviceId,
                readIdGenerator,
                programOptions,
                correctionFlags,
                processResults,
                batchCompleted,
                processSerializedResults
            );  

            // pipeline.runToCompletion(
            //     deviceId,
            //     readIdGenerator,
            //     programOptions,
            //     correctionFlags,
            //     processResults,
            //     batchCompleted
            // );  
        };

        // auto runSimpleGpuPipelineWithExtraThread = [&](int deviceId,
        //     const GpuForest* gpuForestAnchor, 
        //     const GpuForest* gpuForestCandidate
        // ){
        //     SimpleGpuCorrectionPipeline<Minhasher> pipeline(
        //         readStorage,
        //         minhasher,
        //         nullptr, //&threadPool
        //         gpuForestAnchor,
        //         gpuForestCandidate
        //     );

        //     pipeline.runToCompletionDoubleBufferedWithExtraThread(
        //         deviceId,
        //         readIdGenerator,
        //         programOptions,
        //         programOptions,
        //         correctionFlags,
        //         processResults,
        //         batchCompleted
        //     );  
        // };

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
                processResults,
                batchCompleted
            );
        };

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
                        #if 1
                        pipelineConfig.numOutputConstructors = 0; //always 0
                        pipelineConfig.numCorrectors = 1;
                        threadsForDevice -= pipelineConfig.numCorrectors;                        
                        pipelineConfig.numHashers = std::max(1, std::min(threadsForDevice, numHashersPerCorrectorByTime));
                        threadsForDevice -= pipelineConfig.numHashers;
                        if(threadsForDevice > 0){
                            pipelineConfig.numCorrectors++;
                            threadsForDevice--;
                        }
                        pipelineConfig.numHashers += threadsForDevice;
                        #else
                        pipelineConfig.numOutputConstructors = 0; //always 0
                        pipelineConfig.numCorrectors = 2;
                        pipelineConfig.numHashers = 14;
                        #endif

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
                        availableThreads -= pipelineConfig.numOutputConstructors;
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
    }

#if 0

    auto printRunStats = [](const auto& runStatistics){
        std::cerr << "hashing time average: " << runStatistics.hasherTimeAverage << "\n";
        std::cerr << "corrector time average: " << runStatistics.correctorTimeAverage << "\n";
        std::cerr << "output constructor time average: " << runStatistics.outputconstructorTimeAverage << "\n";

        std::cerr << "input size: ";
        std::cerr << "host: " << runStatistics.memoryInputData.host << ", ";
        for(const auto& d : runStatistics.memoryInputData.device){
            std::cerr << "device " << d.first << ": " << d.second << " ";
        }
        std::cerr << "\n";

        std::cerr << "raw output size ";
        std::cerr << "host: " << runStatistics.memoryRawOutputData.host << ", ";
        for(const auto& d : runStatistics.memoryRawOutputData.device){
            std::cerr << "device " << d.first << ": " << d.second << " ";
        }
        std::cerr << "\n";

        std::cerr << "hasher size ";
        std::cerr << "host: " << runStatistics.memoryHasher.host << ", ";
        for(const auto& d : runStatistics.memoryHasher.device){
            std::cerr << "device " << d.first << ": " << d.second << " ";
        }
        std::cerr << "\n";

        std::cerr << "corrector size ";
        std::cerr << "host: " << runStatistics.memoryCorrector.host << ", ";
        for(const auto& d : runStatistics.memoryCorrector.device){
            std::cerr << "device " << d.first << ": " << d.second << " ";
        }
        std::cerr << "\n";

        std::cerr << "output constructor size ";
        std::cerr << "host: " << runStatistics.memoryOutputConstructor.host << ", ";
        for(const auto& d : runStatistics.memoryOutputConstructor.device){
            std::cerr << "device " << d.first << ": " << d.second << " ";
        }
        std::cerr << "\n";
    };
#endif   

    progressThread.finished(); 
        
    std::cout << std::endl;

    //threadPool.wait();
    outputThread.stopThread(BackgroundThread::StopType::FinishAndStop);

    //assert(threadPool.empty());

    // std::ofstream flagsstream(programOptions.outputfilenames[0] + "_flags");

    // for(std::uint64_t i = 0; i < gpuReadStorage->getNumberOfReads(); i++){
    //     flagsstream << correctionFlags.isCorrectedAsHQAnchor(i) << " " 
    //         << correctionFlags.isNotCorrectedAsAnchor(i) << "\n";
    // }

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

