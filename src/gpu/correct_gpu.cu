



#include <gpu/gpucorrector.cuh>
#include <gpu/gpureadstorage.cuh>

#include <gpu/gpuminhasher.cuh>
#include <gpu/cudaerrorcheck.cuh>

#include <options.hpp>
#include <readlibraryio.hpp>
#include <memorymanagement.hpp>
#include <memoryfile.hpp>
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


#if 0

//TODO classifier?
class SimpleCpuCorrectionPipeline{
    template<class T>
    using HostContainer = helpers::SimpleAllocationPinnedHost<T, 0>;

public:
    template<class ResultProcessor, class BatchCompletion>
    void runToCompletion(
        cpu::RangeGenerator<read_number>& readIdGenerator,
        const CorrectionOptions& correctionOptions,
        const GoodAlignmentProperties& goodAlignmentProperties,
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
            correctionOptions,
            goodAlignmentProperties,
            *candidateIdsProvider,
            *readProvider,
            correctionFlags
        );

        HostContainer<read_number> batchReadIds(correctionOptions.batchsize);
        HostContainer<unsigned int> batchEncodedData(correctionOptions.batchsize * encodedSequencePitchInInts2Bit);
        HostContainer<char> batchQualities(correctionOptions.batchsize * qualityPitchInBytes);
        HostContainer<int> batchReadLengths(correctionOptions.batchsize);

        std::vector<read_number> tmpids(correctionOptions.batchsize);

        while(!(readIdGenerator.empty())){
            tmpids.resize(correctionOptions.batchsize);            

            auto readIdsEnd = readIdGenerator.next_n_into_buffer(
                correctionOptions.batchsize, 
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

            if(correctionOptions.useQualityScores){
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

            if(correctionOptions.useQualityScores){
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

    template<class IdGenerator, class ResultProcessor, class BatchCompletion>
    RunStatistics runToCompletion(
        int deviceId,
        IdGenerator& readIdGenerator,
        const CorrectionOptions& correctionOptions,
        const GoodAlignmentProperties& goodAlignmentProperties,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted
    ) const {

        auto continueCondition = [&](){ return !readIdGenerator.empty(); };

        return run_impl(
            deviceId,
            readIdGenerator,
            correctionOptions,
            goodAlignmentProperties,
            correctionFlags,
            processResults,
            batchCompleted,
            continueCondition
        );
    }

    template<class IdGenerator, class ResultProcessor, class BatchCompletion>
    RunStatistics runSomeBatches(
        int deviceId,
        IdGenerator& readIdGenerator,
        const CorrectionOptions& correctionOptions,
        const GoodAlignmentProperties& goodAlignmentProperties,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted,
        int numBatches
    ) const {

        auto continueCondition = [&](){ bool success = !readIdGenerator.empty() && numBatches > 0; numBatches--; return success;};

        return run_impl(
            deviceId,
            readIdGenerator,
            correctionOptions,
            goodAlignmentProperties,
            correctionFlags,
            processResults,
            batchCompleted,
            continueCondition
        );
    }

    template<class IdGenerator, class ResultProcessor, class BatchCompletion>
    RunStatistics runToCompletionDoubleBuffered(
        int deviceId,
        IdGenerator& readIdGenerator,
        const CorrectionOptions& correctionOptions,
        const GoodAlignmentProperties& goodAlignmentProperties,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted
    ) const {

        constexpr bool useThreadForOutputConstruction = false;

        auto continueCondition = [&](){ return !readIdGenerator.empty(); };

        return runDoubleBuffered_impl(
            deviceId,
            readIdGenerator,
            correctionOptions,
            goodAlignmentProperties,
            correctionFlags,
            useThreadForOutputConstruction,
            processResults,
            batchCompleted,
            continueCondition
        );
    }

    template<class IdGenerator, class ResultProcessor, class BatchCompletion>
    RunStatistics runToCompletionDoubleBufferedWithExtraThread(
        int deviceId,
        IdGenerator& readIdGenerator,
        const CorrectionOptions& correctionOptions,
        const GoodAlignmentProperties& goodAlignmentProperties,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted
    ) const {

        constexpr bool useThreadForOutputConstruction = true;

        auto continueCondition = [&](){ return !readIdGenerator.empty(); };

        return runDoubleBuffered_impl(
            deviceId,
            readIdGenerator,
            correctionOptions,
            goodAlignmentProperties,
            correctionFlags,
            useThreadForOutputConstruction,
            processResults,
            batchCompleted,
            continueCondition
        );
    }

    template<class IdGenerator, class ResultProcessor, class BatchCompletion>
    RunStatistics runSomeBatchesDoubleBuffered(
        int deviceId,
        IdGenerator& readIdGenerator,
        const CorrectionOptions& correctionOptions,
        const GoodAlignmentProperties& goodAlignmentProperties,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted,
        int numBatches
    ) const {

        constexpr bool useThreadForOutputConstruction = false;

        auto continueCondition = [&](){ bool success = !readIdGenerator.empty() && numBatches > 0; numBatches--; return success;};

        return runDoubleBuffered_impl(
            deviceId,
            readIdGenerator,
            correctionOptions,
            goodAlignmentProperties,
            correctionFlags,
            useThreadForOutputConstruction,
            processResults,
            batchCompleted,
            continueCondition
        );
    }

    template<class IdGenerator, class ResultProcessor, class BatchCompletion>
    RunStatistics runSomeBatchesDoubleBufferedWithExtraThread(
        int deviceId,
        IdGenerator& readIdGenerator,
        const CorrectionOptions& correctionOptions,
        const GoodAlignmentProperties& goodAlignmentProperties,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted,
        int numBatches
    ) const {

        constexpr bool useThreadForOutputConstruction = true;

        auto continueCondition = [&](){ bool success = !readIdGenerator.empty() && numBatches > 0; numBatches--; return success;};

        return runDoubleBuffered_impl(
            deviceId,
            readIdGenerator,
            correctionOptions,
            goodAlignmentProperties,
            correctionFlags,
            useThreadForOutputConstruction,
            processResults,
            batchCompleted,
            continueCondition
        );
    }

    template<class IdGenerator, class ResultProcessor, class BatchCompletion, class ContinueCondition>
    RunStatistics runDoubleBuffered_impl(
        int deviceId,
        IdGenerator& readIdGenerator,
        const CorrectionOptions& correctionOptions,
        const GoodAlignmentProperties& goodAlignmentProperties,
        ReadCorrectionFlags& correctionFlags,
        bool useThreadForOutputConstruction,
        ResultProcessor processResults,
        BatchCompletion batchCompleted,
        ContinueCondition continueCondition
    ) const {
        cub::SwitchDevice sd{deviceId};

        constexpr int numextra = 1;
        constexpr int numStreams = 1 + numextra;

        std::array<CudaStream, numStreams> streams{};
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
            threadPool
        );

        GpuErrorCorrector gpuErrorCorrector{
            *readStorage,
            correctionFlags,
            correctionOptions,
            goodAlignmentProperties,
            correctionOptions.batchsize,
            threadPool,
            gpuForestAnchor,
            gpuForestCandidate
        };

        OutputConstructor outputConstructor(            
            correctionFlags,
            correctionOptions
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

            nvtx::push_range("constructEncodedResults", 2);
            EncodedCorrectionOutput encodedCorrectionOutput = outputConstructor.constructEncodedResults(*rawOutputPtr, forLoopExecutor);
            nvtx::pop_range();

            freeRawOutputQueue.push(rawOutputPtr);
            
            outputTimer.stop();
            //elapsedOutputTimes.emplace_back(outputTimer.elapsed());
            elapsedOutputTime += outputTimer.elapsed();

            processResults(
                std::move(encodedCorrectionOutput)
            );
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

        std::vector<read_number> anchorIds(correctionOptions.batchsize);

        int globalcounter = 0;

        for(int i = 0; i < numextra; i++){
            if(continueCondition()){
                cudaStream_t stream = streams[streamIndex];
                streamIndex = (streamIndex + 1) % numStreams;

                helpers::CpuTimer hashingTimer;
            
                anchorIds.resize(correctionOptions.batchsize);
                readIdGenerator.process_next_n(
                    correctionOptions.batchsize, 
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

                dataInFlight.push(std::make_pair(inputPtr, rawOutputPtr));

                batchCompleted(anchorIds.size());
                iterations++;
                //std::cerr << "Added extra (" << inputPtr << ", " << rawOutputPtr << ")\n";
            }
        }


        while(continueCondition()){            
            
            anchorIds.resize(correctionOptions.batchsize);
            readIdGenerator.process_next_n(
                correctionOptions.batchsize, 
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

    template<class IdGenerator, class ResultProcessor, class BatchCompletion, class ContinueCondition>
    RunStatistics run_impl(
        int deviceId,
        IdGenerator& readIdGenerator,
        const CorrectionOptions& correctionOptions,
        const GoodAlignmentProperties& goodAlignmentProperties,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted,
        ContinueCondition continueCondition
    ) const {
        cub::SwitchDevice sd{deviceId};

        //constexpr int numextra = 1;

        CudaStream stream;
        GpuErrorCorrectorInput input;

        GpuErrorCorrectorRawOutput rawOutput;

        //ThreadPool::ParallelForHandle pforHandle;
        //ForLoopExecutor forLoopExecutor(threadPool, &pforHandle);
        SequentialForLoopExecutor forLoopExecutor;

        AnchorHasher gpuAnchorHasher(
            *readStorage,
            *minhasher,
            threadPool
        );

        GpuErrorCorrector gpuErrorCorrector{
            *readStorage,
            correctionFlags,
            correctionOptions,
            goodAlignmentProperties,
            correctionOptions.batchsize,
            threadPool,
            gpuForestAnchor,
            gpuForestCandidate
        };

        OutputConstructor outputConstructor(            
            correctionFlags,
            correctionOptions
        );

        RunStatistics runStatistics;

        std::vector<read_number> anchorIds(correctionOptions.batchsize);

        int iterations = 0;
        std::vector<double> elapsedHashingTimes;
        std::vector<double> elapsedCorrectionTimes;
        std::vector<double> elapsedOutputTimes;

        double elapsedHashingTime = 0.0;
        double elapsedCorrectionTime = 0.0;
        double elapsedOutputTime = 0.0;

        //int globalcounter = 0;

        while(continueCondition()){

            helpers::CpuTimer hashingTimer;
            
            anchorIds.resize(correctionOptions.batchsize);
            readIdGenerator.process_next_n(
                correctionOptions.batchsize, 
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

                CUDACHECK(rawOutput.event.synchronize());

                correctionTimer.stop();
                //elapsedCorrectionTimes.emplace_back(correctionTimer.elapsed());
                if(iterations >= 10){
                    elapsedCorrectionTime += correctionTimer.elapsed();
                }

                helpers::CpuTimer outputTimer;

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
        const CorrectionOptions& correctionOptions,
        const GoodAlignmentProperties& goodAlignmentProperties,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted
    ){
        cub::SwitchDevice sd{deviceId};

        noMoreInputs = false;
        activeHasherThreads = config.numHashers;
        noMoreRawOutputs = false;
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
                                correctionOptions); 
                        }
                    )
                );
            }

            for(int i = 0; i < config.numCorrectors; i++){
                futures.emplace_back(
                    std::async(
                        std::launch::async,
                        [&](){ 
                            correctorThreadFunctionMultiBufferWithOutput(deviceId, correctionOptions, 
                                goodAlignmentProperties, 
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
                                correctionOptions); 
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
                                correctionOptions, 
                                goodAlignmentProperties,
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
                            outputConstructorThreadFunction(correctionOptions, correctionFlags,
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
        const CorrectionOptions& correctionOptions
    ){
        cudaSetDevice(deviceId);

        AnchorHasher gpuAnchorHasher(
            *readStorage,
            *minhasher,
            nullptr//threadPool
        );

        CudaStream hasherStream;
        ThreadPool::ParallelForHandle pforHandle;


        while(!readIdGenerator.empty()){
            CUDACHECK(cudaStreamSynchronize(hasherStream));

            std::vector<read_number> anchorIds(correctionOptions.batchsize);

            readIdGenerator.process_next_n(
                correctionOptions.batchsize, 
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
        const CorrectionOptions& correctionOptions,
        const GoodAlignmentProperties& goodAlignmentProperties,
        const ReadCorrectionFlags& correctionFlags
    ){
        cudaSetDevice(deviceId);

        GpuErrorCorrector gpuErrorCorrector{
            *readStorage,
            correctionFlags,
            correctionOptions,
            goodAlignmentProperties,
            correctionOptions.batchsize,
            threadPool,
            gpuForestAnchor,
            gpuForestCandidate
        };

        CudaStream stream;

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
            noMoreRawOutputs = true;
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
        const CorrectionOptions& correctionOptions,
        const GoodAlignmentProperties& goodAlignmentProperties,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted
    ){
        cudaSetDevice(deviceId);

        GpuErrorCorrector gpuErrorCorrector{
            *readStorage,
            correctionFlags,
            correctionOptions,
            goodAlignmentProperties,
            correctionOptions.batchsize,
            threadPool,
            gpuForestAnchor,
            gpuForestCandidate
        };

        OutputConstructor outputConstructor(            
            correctionFlags,
            correctionOptions
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

        CudaStream stream;

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
            noMoreRawOutputs = true;
            for(int i = 0; i < 2*currentConfig.numOutputConstructors; i++){
                unprocessedRawOutputs.push(nullptr);
            }
        }

        CUDACHECK(cudaStreamSynchronize(stream));
    };


    template<class ResultProcessor, class BatchCompletion>
    void outputConstructorThreadFunction(
        const CorrectionOptions& correctionOptions,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted
    ){

        OutputConstructor outputConstructor(            
            correctionFlags,
            correctionOptions
        );

        ThreadPool::ParallelForHandle pforHandle;
        //ForLoopExecutor forLoopExecutor(&threadPool, &pforHandle);
        SequentialForLoopExecutor forLoopExecutor;

        // GpuErrorCorrectorRawOutput* rawOutputPtr = unprocessedRawOutputs.popOrDefault(
        //     [&](){
        //         return !noMoreRawOutputs;  //if noMoreRawOutputs, return nullptr
        //     },
        //     nullptr
        // );

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
            // rawOutputPtr = unprocessedRawOutputs.popOrDefault(
            //     [&](){
            //         return !noMoreRawOutputs;  //if noMoreRawOutputs, return nullptr
            //     },
            //     nullptr
            // );  

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
    std::atomic<bool> noMoreRawOutputs{false};
    std::atomic<int> activeCorrectorThreads{0};

    Config currentConfig;
};


template<class Minhasher>
MemoryFileFixedSize<EncodedTempCorrectedSequence> 
correct_gpu_impl(
    const GoodAlignmentProperties& goodAlignmentProperties,
    const CorrectionOptions& correctionOptions,
    const RuntimeOptions& runtimeOptions,
    const FileOptions& fileOptions,
    const MemoryOptions& memoryOptions,
    Minhasher& minhasher,
    GpuReadStorage& readStorage,
    const std::vector<GpuForest>& anchorForests,
    const std::vector<GpuForest>& candidateForests
){

    assert(runtimeOptions.canUseGpu);
    //assert(runtimeOptions.max_candidates > 0);
    assert(runtimeOptions.deviceIds.size() > 0);

    const auto& deviceIds = runtimeOptions.deviceIds;

    const auto rsMemInfo = readStorage.getMemoryInfo();
    const auto mhMemInfo = minhasher.getMemoryInfo();

    std::size_t memoryAvailableBytesHost = memoryOptions.memoryTotalLimit;

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

    const std::string tmpfilename{fileOptions.tempdirectory + "/" + "MemoryFileFixedSizetmp"};
    MemoryFileFixedSize<EncodedTempCorrectedSequence> partialResults(memoryForPartialResultsInBytes, tmpfilename);

    //std::mutex outputstreamlock;

    BackgroundThread outputThread;

    auto saveEncodedCorrectedSequence = [&](const EncodedTempCorrectedSequence* encoded){
        if(!(encoded->isHQ() && encoded->useEdits() && encoded->getNumEdits() == 0)){
            partialResults.storeElement(encoded);
        }
    };

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

    outputThread.setMaximumQueueSize(runtimeOptions.threads);

    outputThread.start();

    IteratorRangeTraversal<thrust::counting_iterator<read_number>> readIdGenerator(
        thrust::make_counting_iterator<read_number>(0),
        thrust::make_counting_iterator<read_number>(0) + readStorage.getNumberOfReads()
    );

    auto showProgress = [&](std::int64_t totalCount, int seconds){
        if(runtimeOptions.showProgress){

            int hours = seconds / 3600;
            seconds = seconds % 3600;
            int minutes = seconds / 60;
            seconds = seconds % 60;

            std::size_t numreads = readStorage.getNumberOfReads();
            
            printf("Processed %10lu of %10lu reads (Runtime: %03d:%02d:%02d)\r",
            totalCount, numreads,
            hours, minutes, seconds);

            std::fflush(stdout);
        }
    };

    auto updateShowProgressInterval = [](auto duration){
        return duration;
    };

    ProgressThread<std::int64_t> progressThread(readStorage.getNumberOfReads(), showProgress, updateShowProgressInterval);

    auto batchCompleted = [&](int size){
        //std::cerr << "Add progress " << size << "\n";
        progressThread.addProgress(size);
    };

    if(false /* && runtimeOptions.threads <= 6*/){
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
                correctionOptions,
                goodAlignmentProperties,
                correctionFlags,
                processResults,
                batchCompleted
            );
        };
    
        std::vector<std::future<void>> futures;
    
        for(int i = 0; i < runtimeOptions.threads; i++){
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
                correctionOptions,
                goodAlignmentProperties,
                correctionFlags,
                processResults,
                batchCompleted,
                numBatches
            );   
                
        }

        const int numHashersPerCorrectorByTime = std::ceil(runStatistics.hasherTimeAverage / runStatistics.correctorTimeAverage);
        //std::cerr << runStatistics.hasherTimeAverage << " " << runStatistics.correctorTimeAverage << "\n";

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

            pipeline.runToCompletionDoubleBuffered(
                deviceId,
                readIdGenerator,
                correctionOptions,
                goodAlignmentProperties,
                correctionFlags,
                processResults,
                batchCompleted
            );  
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
        //         correctionOptions,
        //         goodAlignmentProperties,
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
                correctionOptions,
                goodAlignmentProperties,
                correctionFlags,
                processResults,
                batchCompleted
            );
        };

        std::vector<std::future<void>> futures;

        const int numDevices = deviceIds.size();
        const int requiredNumThreadsForComplex = numHashersPerCorrectorByTime + (2 + 1 + 1);
        int availableThreads = runtimeOptions.threads;

        for(int i = 0; i < numDevices; i++){ 
            const int deviceId = deviceIds[i];

            int threadsForDevice = std::max(1,std::min(availableThreads, requiredNumThreadsForComplex));

            if(minhasher.hasGpuTables()){
                int threadsForDeviceWithGpuTables = std::min(4, threadsForDevice);
                std::cerr << "\nWill use " << threadsForDeviceWithGpuTables << " gpu hashtable pipelines on device " << deviceId << "\n";

                while(threadsForDeviceWithGpuTables > 0){
                    futures.emplace_back(std::async(
                        std::launch::async,
                        runSimpleGpuPipeline,
                        deviceId,
                        &anchorForests[i],
                        &candidateForests[i]
                    ));

                    threadsForDeviceWithGpuTables--;
                    availableThreads--;
                }
            }else{

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
                    threadsForDevice = 0;
                    #else
                    pipelineConfig.numOutputConstructors = 0; //always 0
                    pipelineConfig.numCorrectors = 13;
                    pipelineConfig.numHashers = 3;
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

    partialResults.flush();

    // std::ofstream flagsstream(fileOptions.outputfilenames[0] + "_flags");

    // for(std::uint64_t i = 0; i < gpuReadStorage->getNumberOfReads(); i++){
    //     flagsstream << correctionFlags.isCorrectedAsHQAnchor(i) << " " 
    //         << correctionFlags.isNotCorrectedAsAnchor(i) << "\n";
    // }

    return partialResults;
}


MemoryFileFixedSize<EncodedTempCorrectedSequence> 
correct_gpu(
    const GoodAlignmentProperties& goodAlignmentProperties,
    const CorrectionOptions& correctionOptions,
    const RuntimeOptions& runtimeOptions,
    const FileOptions& fileOptions,
    const MemoryOptions& memoryOptions,
    GpuMinhasher& minhasher,
    GpuReadStorage& readStorage,
    const std::vector<GpuForest>& anchorForests,
    const std::vector<GpuForest>& candidateForests
){

    return correct_gpu_impl(
        goodAlignmentProperties,
        correctionOptions,
        runtimeOptions,
        fileOptions,
        memoryOptions,
        minhasher,
        readStorage,
        anchorForests,
        candidateForests
    );
}


}
}

