



#include <gpu/gpucorrector.cuh>
#include <gpu/gpureadstorage.cuh>

#include <gpu/gpuminhasher.cuh>

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
#include <forest_gpu.cuh>


#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <future>

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
        ClfAgent* clfAgent_
    ) :
        readStorage(&readStorage_),
        minhasher(&minhasher_),
        threadPool(threadPool_),
        clfAgent(clfAgent_)
    {

    }

    template<class ResultProcessor, class BatchCompletion>
    RunStatistics runToCompletion(
        int deviceId,
        cpu::RangeGenerator<read_number>& readIdGenerator,
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

    template<class ResultProcessor, class BatchCompletion>
    RunStatistics runSomeBatches(
        int deviceId,
        cpu::RangeGenerator<read_number>& readIdGenerator,
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

    template<class ResultProcessor, class BatchCompletion>
    RunStatistics runToCompletionDoubleBuffered(
        int deviceId,
        cpu::RangeGenerator<read_number>& readIdGenerator,
        const CorrectionOptions& correctionOptions,
        const GoodAlignmentProperties& goodAlignmentProperties,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted
    ) const {

        auto continueCondition = [&](){ return !readIdGenerator.empty(); };

        return runDoubleBuffered_impl(
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

    template<class ResultProcessor, class BatchCompletion>
    RunStatistics runSomeBatchesDoubleBuffered(
        int deviceId,
        cpu::RangeGenerator<read_number>& readIdGenerator,
        const CorrectionOptions& correctionOptions,
        const GoodAlignmentProperties& goodAlignmentProperties,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted,
        int numBatches
    ) const {

        auto continueCondition = [&](){ bool success = !readIdGenerator.empty() && numBatches > 0; numBatches--; return success;};

        return runDoubleBuffered_impl(
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

    template<class ResultProcessor, class BatchCompletion, class ContinueCondition>
    RunStatistics runDoubleBuffered_impl(
        int deviceId,
        cpu::RangeGenerator<read_number>& readIdGenerator,
        const CorrectionOptions& correctionOptions,
        const GoodAlignmentProperties& goodAlignmentProperties,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted,
        ContinueCondition continueCondition
    ) const {
        int cur = 0;
        cudaGetDevice(&cur); CUERR;
        cudaSetDevice(deviceId);

        constexpr int numextra = 1;

        CudaStream stream;
        GpuErrorCorrectorInput input;
        std::array<GpuErrorCorrectorRawOutput, 1 + numextra> rawOutputArray;
        std::queue<GpuErrorCorrectorRawOutput*> freeRawOutputQueue;
        std::queue<GpuErrorCorrectorRawOutput*> unprocessedRawOutputQueue;
        for(auto& a : rawOutputArray){
            freeRawOutputQueue.push(&a);
        }
        //GpuErrorCorrectorRawOutput rawOutput;

        cudaError_t querystatus = input.event.query();
        if(querystatus != cudaSuccess){
            std::cout << "CUDA error: " << cudaGetErrorString(querystatus) << " : "
                << __FILE__ << ", line " << __LINE__ << std::endl;
        }
        assert(cudaSuccess == querystatus);

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
            clfAgent
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

        int globalcounter = 0;

        for(int i = 0; i < numextra; i++){
            if(continueCondition()){
                helpers::CpuTimer hashingTimer;
            
                anchorIds.resize(correctionOptions.batchsize);
                auto readIdsEnd = readIdGenerator.next_n_into_buffer(correctionOptions.batchsize, anchorIds.begin());
                anchorIds.erase(readIdsEnd, anchorIds.end());

                if(anchorIds.size() == 0){
                    continue;
                }

                //std::cerr << "globalcounter " << globalcounter << "\n";
        
                nvtx::push_range("makeErrorCorrectorInput", 0);
                gpuAnchorHasher.makeErrorCorrectorInput(
                    anchorIds.data(),
                    anchorIds.size(),
                    input,
                    stream
                );
                nvtx::pop_range();

                input.event.synchronize();

                globalcounter++;

                GpuErrorCorrectorRawOutput* rawOutputPtr = freeRawOutputQueue.front();
                freeRawOutputQueue.pop();

                hashingTimer.stop();
                //elapsedHashingTimes.emplace_back(hashingTimer.elapsed());
                elapsedHashingTime += hashingTimer.elapsed();

                nvtx::push_range("correct", 1);
                gpuErrorCorrector.correct(input, *rawOutputPtr, stream);
                nvtx::pop_range();

                unprocessedRawOutputQueue.push(rawOutputPtr);
            }
        }


        while(continueCondition()){

            
            
            anchorIds.resize(correctionOptions.batchsize);
            auto readIdsEnd = readIdGenerator.next_n_into_buffer(correctionOptions.batchsize, anchorIds.begin());
            anchorIds.erase(readIdsEnd, anchorIds.end());

            helpers::CpuTimer correctionTimer;

            if(anchorIds.size() > 0){
                helpers::CpuTimer hashingTimer;

                input.event.synchronize();

                //std::cerr << "globalcounter " << globalcounter << "\n";
        
                nvtx::push_range("makeErrorCorrectorInput", 0);
                gpuAnchorHasher.makeErrorCorrectorInput(
                    anchorIds.data(),
                    anchorIds.size(),
                    input,
                    stream
                );
                nvtx::pop_range();

                input.event.synchronize();

                //globalcounter++;

                hashingTimer.stop();
                //elapsedHashingTimes.emplace_back(hashingTimer.elapsed());
                elapsedHashingTime += hashingTimer.elapsed();

                GpuErrorCorrectorRawOutput* rawOutputPtr = freeRawOutputQueue.front();
                freeRawOutputQueue.pop();

                //helpers::CpuTimer correctionTimer;
                correctionTimer.reset();
                correctionTimer.start();

                nvtx::push_range("correct", 1);
                gpuErrorCorrector.correct(input, *rawOutputPtr, stream);
                nvtx::pop_range();

                unprocessedRawOutputQueue.push(rawOutputPtr);
            }

            if(unprocessedRawOutputQueue.size() > 0){

                GpuErrorCorrectorRawOutput* rawOutputPtr = unprocessedRawOutputQueue.front();
                unprocessedRawOutputQueue.pop();

                rawOutputPtr->event.synchronize();

                if(anchorIds.size() > 0){
                    correctionTimer.stop();
                    //elapsedCorrectionTimes.emplace_back(correctionTimer.elapsed());
                    elapsedCorrectionTime += correctionTimer.elapsed();
                }


                helpers::CpuTimer outputTimer;

                nvtx::push_range("constructResults", 2);
                auto correctionOutput = outputConstructor.constructResults(*rawOutputPtr, forLoopExecutor);
                nvtx::pop_range();

                freeRawOutputQueue.push(rawOutputPtr);


                nvtx::push_range("encodeResults", 3);

                correctionOutput.encode();

                nvtx::pop_range();

                outputTimer.stop();
                //elapsedOutputTimes.emplace_back(outputTimer.elapsed());
                elapsedOutputTime += outputTimer.elapsed();

                processResults(
                    std::move(correctionOutput)
                );

            }

            batchCompleted(anchorIds.size());

            iterations++;
        }

        //process remaining cached results
        while(unprocessedRawOutputQueue.size() > 0){
            GpuErrorCorrectorRawOutput* rawOutputPtr = unprocessedRawOutputQueue.front();
            unprocessedRawOutputQueue.pop();

            rawOutputPtr->event.synchronize();

            //correctionTimer.stop();
            //elapsedCorrectionTimes.emplace_back(correctionTimer.elapsed());
            //elapsedCorrectionTime += correctionTimer.elapsed();


            helpers::CpuTimer outputTimer;

            nvtx::push_range("constructResults", 2);
            auto correctionOutput = outputConstructor.constructResults(*rawOutputPtr, forLoopExecutor);
            nvtx::pop_range();

            freeRawOutputQueue.push(rawOutputPtr);

            nvtx::push_range("encodeResults", 3);

            correctionOutput.encode();

            nvtx::pop_range();

            outputTimer.stop();
            //elapsedOutputTimes.emplace_back(outputTimer.elapsed());
            elapsedOutputTime += outputTimer.elapsed();

            processResults(
                std::move(correctionOutput)
            );

            batchCompleted(*input.h_numAnchors.get());

            iterations++;
        }

        cudaSetDevice(cur); CUERR;

        runStatistics.hasherTimeAverage = elapsedHashingTime / iterations;
        runStatistics.correctorTimeAverage = elapsedCorrectionTime / iterations;
        runStatistics.outputconstructorTimeAverage = elapsedOutputTime / iterations;
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

    template<class ResultProcessor, class BatchCompletion, class ContinueCondition>
    RunStatistics run_impl(
        int deviceId,
        cpu::RangeGenerator<read_number>& readIdGenerator,
        const CorrectionOptions& correctionOptions,
        const GoodAlignmentProperties& goodAlignmentProperties,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted,
        ContinueCondition continueCondition
    ) const {
        int cur = 0;
        cudaGetDevice(&cur); CUERR;
        cudaSetDevice(deviceId);

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
            clfAgent
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
            auto readIdsEnd = readIdGenerator.next_n_into_buffer(correctionOptions.batchsize, anchorIds.begin());
            anchorIds.erase(readIdsEnd, anchorIds.end());

            if(anchorIds.size() > 0){

                input.event.synchronize();

                //std::cerr << "globalcounter " << globalcounter << "\n";
        
                nvtx::push_range("makeErrorCorrectorInput", 0);
                gpuAnchorHasher.makeErrorCorrectorInput(
                    anchorIds.data(),
                    anchorIds.size(),
                    input,
                    stream
                );
                nvtx::pop_range();

                input.event.synchronize();

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

                rawOutput.event.synchronize();

                correctionTimer.stop();
                //elapsedCorrectionTimes.emplace_back(correctionTimer.elapsed());
                if(iterations >= 10){
                    elapsedCorrectionTime += correctionTimer.elapsed();
                }

                helpers::CpuTimer outputTimer;

                nvtx::push_range("constructResults", 2);
                auto correctionOutput = outputConstructor.constructResults(rawOutput, forLoopExecutor);
                nvtx::pop_range();

                nvtx::push_range("encodeResults", 3);

                correctionOutput.encode();

                nvtx::pop_range();

                outputTimer.stop();
                //elapsedOutputTimes.emplace_back(outputTimer.elapsed());
                if(iterations >= 10){
                    elapsedOutputTime += outputTimer.elapsed();
                }

                processResults(
                    std::move(correctionOutput)
                );
            }

            batchCompleted(anchorIds.size());

            iterations++;
        }

        cudaSetDevice(cur); CUERR;

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
    ClfAgent* clfAgent;
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
        ClfAgent* clfAgent_ 
    ) :
        readStorage(&readStorage_),
        minhasher(&minhasher_),
        threadPool(threadPool_),
        clfAgent(clfAgent_)
    {

    }

    template<class ResultProcessor, class BatchCompletion>
    void run(
        int deviceId,
        const Config& config,
        cpu::RangeGenerator<read_number>& readIdGenerator,
        const CorrectionOptions& correctionOptions,
        const GoodAlignmentProperties& goodAlignmentProperties,
        ReadCorrectionFlags& correctionFlags,
        ResultProcessor processResults,
        BatchCompletion batchCompleted
    ){
        int curDevice = 0;
        cudaGetDevice(&curDevice); CUERR;
        cudaSetDevice(deviceId); CUERR;

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

        cudaSetDevice(curDevice); CUERR;
    }
    

    void hasherThreadFunction(
        int deviceId,
        cpu::RangeGenerator<read_number>& readIdGenerator,
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
            cudaStreamSynchronize(hasherStream);

            std::vector<read_number> anchorIds(correctionOptions.batchsize);

            auto readIdsEnd = readIdGenerator.next_n_into_buffer(correctionOptions.batchsize, anchorIds.begin());
            anchorIds.erase(readIdsEnd, anchorIds.end());

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

            inputPtr->event.synchronize();

            unprocessedInputs.push(inputPtr);
            
        }

        activeHasherThreads--;

        if(activeHasherThreads == 0){
            noMoreInputs = true;
        }

        cudaStreamSynchronize(hasherStream);

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
            clfAgent
        };

        CudaStream stream;

        GpuErrorCorrectorInput* inputPtr = unprocessedInputs.popOrDefault(
            [&](){
                return !noMoreInputs;  //if noMoreInputs, return nullptr
            },
            nullptr
        ); 

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

            cudaStreamSynchronize(stream); CUERR;

            // assert(cudaSuccess == inputPtr->event.query());
            // assert(cudaSuccess == rawOutputPtr->event.query());

            nvtx::push_range("correct", 0);
            gpuErrorCorrector.correct(*inputPtr, *rawOutputPtr, stream);
            nvtx::pop_range();

            inputPtr->event.synchronize();
            freeInputs.push(inputPtr);

            //cudaStreamSynchronize(stream);
            
            rawOutputPtr->event.synchronize();
            //std::cerr << "Synchronized output " << rawOutputPtr << "\n";
            unprocessedRawOutputs.push(rawOutputPtr);
        
            nvtx::push_range("getUnprocessedInput",2);
            inputPtr = unprocessedInputs.popOrDefault(
                [&](){
                    return !noMoreInputs;  //if noMoreInputs, return nullptr
                },
                nullptr
            ); 
            nvtx::pop_range();

        };

        activeCorrectorThreads--;

        if(activeCorrectorThreads == 0){
            noMoreRawOutputs = true;
        }

        cudaStreamSynchronize(stream); CUERR;
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
            clfAgent
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
            nvtx::push_range("constructResults", 0);
            auto correctionOutput = outputConstructor.constructResults(*rawOutputPtr, forLoopExecutor);
            nvtx::pop_range();

            nvtx::push_range("encodeResults", 1);
            correctionOutput.encode();

            nvtx::pop_range();

            processResults(
                std::move(correctionOutput)
            );

            batchCompleted(rawOutputPtr->numAnchors); 

            myFreeOutputsQueue.push(rawOutputPtr);
        };

        CudaStream stream;

        std::queue<std::pair<GpuErrorCorrectorInput*,
            GpuErrorCorrectorRawOutput*>> dataInFlight;

        GpuErrorCorrectorInput* inputPtr = unprocessedInputs.popOrDefault(
            [&](){
                return !noMoreInputs;  //if noMoreInputs, return nullptr
            },
            nullptr
        ); 

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

                inputPtr = unprocessedInputs.popOrDefault(
                    [&](){
                        return !noMoreInputs;  //if noMoreInputs, return nullptr
                    },
                    nullptr
                ); 
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

            cudaStreamSynchronize(stream); CUERR;

            // assert(cudaSuccess == inputPtr->event.query());
            // assert(cudaSuccess == rawOutputPtr->event.query());

            nvtx::push_range("correct", 0);
            gpuErrorCorrector.correct(*inputPtr, *rawOutputPtr, stream);
            nvtx::pop_range();

            dataInFlight.emplace(inputPtr, rawOutputPtr);

            if(!dataInFlight.empty()){
                auto pointers = dataInFlight.front();
                dataInFlight.pop();

                pointers.first->event.synchronize();
                freeInputs.push(pointers.first);

                pointers.second->event.synchronize();
                //std::cerr << "Synchronized output " << pointers.second << "\n";
                constructOutput(pointers.second);
            }            

            nvtx::push_range("getUnprocessedInput",2);
            inputPtr = unprocessedInputs.popOrDefault(
                [&](){
                    return !noMoreInputs;  //if noMoreInputs, return nullptr
                },
                nullptr
            ); 
            nvtx::pop_range();

        };

        //process outstanding buffered work
        while(!dataInFlight.empty()){
            auto pointers = dataInFlight.front();
            dataInFlight.pop();

            pointers.first->event.synchronize();
            freeInputs.push(pointers.first);

            pointers.second->event.synchronize();
            constructOutput(pointers.second);
        }

        activeCorrectorThreads--;

        if(activeCorrectorThreads == 0){
            noMoreRawOutputs = true;
        }

        cudaStreamSynchronize(stream); CUERR;
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

        GpuErrorCorrectorRawOutput* rawOutputPtr = unprocessedRawOutputs.popOrDefault(
            [&](){
                return !noMoreRawOutputs;  //if noMoreRawOutputs, return nullptr
            },
            nullptr
        );

        while(rawOutputPtr != nullptr){
            nvtx::push_range("constructResults", 0);
            auto correctionOutput = outputConstructor.constructResults(*rawOutputPtr, forLoopExecutor);
            nvtx::pop_range();

            nvtx::push_range("encodeResults", 1);
            correctionOutput.encode();

            // std::vector<EncodedTempCorrectedSequence> encodedAnchorCorrections;
            // std::vector<EncodedTempCorrectedSequence> encodedCandidateCorrections;

            // if(correctionOutput.anchorCorrections.size() > 0){
            //     encodedAnchorCorrections.resize(correctionOutput.anchorCorrections.size());

            //     forLoopExecutor(std::size_t(0), correctionOutput.anchorCorrections.size(), 
            //         [&](auto begin, auto end, auto /*threadId*/){
            //             for(auto i = begin; i < end; i++){
            //                 correctionOutput.anchorCorrections[i].encodeInto(encodedAnchorCorrections[i]);
            //             }
            //         }
            //     );
            // }

            // if(correctionOutput.candidateCorrections.size() > 0){
            //     encodedCandidateCorrections.resize(correctionOutput.candidateCorrections.size());

            //     forLoopExecutor(std::size_t(0), correctionOutput.candidateCorrections.size(), 
            //         [&](auto begin, auto end, auto /*threadId*/){
            //             for(auto i = begin; i < end; i++){
            //                 correctionOutput.candidateCorrections[i].encodeInto(encodedCandidateCorrections[i]);
            //             }
            //         }
            //     );
            // }

            nvtx::pop_range();

            processResults(
                std::move(correctionOutput)
                // std::move(correctionOutput.anchorCorrections),
                // std::move(correctionOutput.candidateCorrections),
                // std::move(encodedAnchorCorrections),
                // std::move(encodedCandidateCorrections)
            );

            batchCompleted(rawOutputPtr->numAnchors); 


            freeRawOutputs.push(rawOutputPtr);

            nvtx::push_range("getUnprocessedRawOutput", 2);
            rawOutputPtr = unprocessedRawOutputs.popOrDefault(
                [&](){
                    return !noMoreRawOutputs;  //if noMoreRawOutputs, return nullptr
                },
                nullptr
            );  

            nvtx::pop_range();
        }
    };

private:
    const GpuReadStorage* readStorage;
    const Minhasher* minhasher;
    ThreadPool* threadPool;
    ClfAgent* clfAgent;

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
        GpuReadStorage& readStorage){

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

    auto saveCorrectedSequence = [&](const TempCorrectedSequence* tmp, const EncodedTempCorrectedSequence* encoded){
        //std::cerr << *tmp << "\n";
        //useEditsCountMap[tmp.useEdits]++;
        //std::unique_lock<std::mutex> l(outputstreammutex);
        if(!(tmp->hq && tmp->useEdits && tmp->edits.empty())){
            //outputstream << tmp << '\n';
            partialResults.storeElement(encoded);
            //useEditsSavedCountMap[tmp.useEdits]++;
            //numEditsHistogram[tmp.edits.size()]++;

            // std::cerr << tmp.edits.size() << " " << encoded.data.capacity() << "\n";
        }
    };

    // auto processResults = [&](
    //     std::vector<TempCorrectedSequence>&& anchorCorrections,
    //     std::vector<TempCorrectedSequence>&& candidateCorrections,
    //     std::vector<EncodedTempCorrectedSequence>&& encodedAnchorCorrections,
    //     std::vector<EncodedTempCorrectedSequence>&& encodedCandidateCorrections
    // ){
    //     assert(anchorCorrections.size() == encodedAnchorCorrections.size());
    //     assert(candidateCorrections.size() == encodedCandidateCorrections.size());

    //     const int numA = encodedAnchorCorrections.size();
    //     const int numC = encodedCandidateCorrections.size();

    //     auto outputFunction = [
    //         &,
    //         anchorCorrections = std::move(anchorCorrections),
    //         candidateCorrections = std::move(candidateCorrections),
    //         encodedAnchorCorrections = std::move(encodedAnchorCorrections),
    //         encodedCandidateCorrections = std::move(encodedCandidateCorrections)
    //     ](){

    //         const int numA = encodedAnchorCorrections.size();
    //         const int numC = encodedCandidateCorrections.size();

    //         for(int i = 0; i < numA; i++){
    //             saveCorrectedSequence(
    //                 &anchorCorrections[i], 
    //                 &encodedAnchorCorrections[i]
    //             );
    //         }

    //         for(int i = 0; i < numC; i++){
    //             saveCorrectedSequence(
    //                 &candidateCorrections[i], 
    //                 &encodedCandidateCorrections[i]
    //             );
    //         }
    //     };

    //     if(numA > 0 || numC > 0){
    //         outputThread.enqueue(std::move(outputFunction));
    //         //outputFunction();
    //     }
    // };

    auto processResults = [&](
        CorrectionOutput&& correctionOutput
    ){
        assert(correctionOutput.anchorCorrections.size() == correctionOutput.encodedAnchorCorrections.size());
        assert(correctionOutput.candidateCorrections.size() == correctionOutput.encodedCandidateCorrections.size());

        const int numA = correctionOutput.encodedAnchorCorrections.size();
        const int numC = correctionOutput.encodedCandidateCorrections.size();

        auto outputFunction = [
            &,
            correctionOutput = std::move(correctionOutput)
        ](){

            const int numA = correctionOutput.encodedAnchorCorrections.size();
            const int numC = correctionOutput.encodedCandidateCorrections.size();

            for(int i = 0; i < numA; i++){
                saveCorrectedSequence(
                    &correctionOutput.anchorCorrections[i], 
                    &correctionOutput.encodedAnchorCorrections[i]
                );
            }

            for(int i = 0; i < numC; i++){
                saveCorrectedSequence(
                    &correctionOutput.candidateCorrections[i], 
                    &correctionOutput.encodedCandidateCorrections[i]
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

    //const int threadPoolSize = std::max(1, runtimeOptions.threads - 2*int(deviceIds.size()));
    //std::cerr << "threadpool size for correction = " << threadPoolSize << "\n";
    //ThreadPool threadPool(threadPoolSize);

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

    ClfAgent clfAgent_(correctionOptions, fileOptions);



    cpu::RangeGenerator<read_number> readIdGenerator(readStorage.getNumberOfReads());
    //cpu::RangeGenerator<read_number> readIdGenerator(std::min(1500000u, readStorage.getNumberOfReads()));

    if(false /* && runtimeOptions.threads <= 6*/){
        //execute a single thread pipeline with each available thread

        auto runPipeline = [&](int deviceId){    
            SimpleGpuCorrectionPipeline<Minhasher> pipeline(
                readStorage,
                minhasher,
                nullptr, //&threadPool
                &clfAgent_
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
            const int deviceId = deviceIds[i % deviceIds.size()];

            futures.emplace_back(std::async(
                std::launch::async,
                runPipeline,
                deviceId
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
                &clfAgent_
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

        // auto runSimpleCpuPipeline = [&](int deviceId){
        //     // cudaSetDevice(deviceId); CUERR;

        //     // SimpleCpuCorrectionPipeline pipeline;

        //     // std::unique_ptr<ReadProvider> readProvider = std::make_unique<GpuReadStorageReadProvider>(readStorage);
        //     // std::unique_ptr<CandidateIdsProvider> candidateIdsProvider = std::make_unique<GpuMinhasherCandidateIdsProvider>(minhasher);

        //     // pipeline.runToCompletion(
        //     //     readIdGenerator,
        //     //     correctionOptions,
        //     //     goodAlignmentProperties,
        //     //     correctionFlags,
        //     //     readProvider.get(),
        //     //     candidateIdsProvider.get(),
        //     //     processResults,
        //     //     batchCompleted
        //     // ); 
        // };

        auto runSimpleGpuPipeline = [&](int deviceId){
            SimpleGpuCorrectionPipeline<Minhasher> pipeline(
                readStorage,
                minhasher,
                nullptr, //&threadPool        
                &clfAgent_
            );

            pipeline.runToCompletionDoubleBuffered(
                deviceIds[0],
                readIdGenerator,
                correctionOptions,
                goodAlignmentProperties,
                correctionFlags,
                processResults,
                batchCompleted
            );  
        };

        auto runComplexGpuPipeline = [&](int deviceId, typename ComplexGpuCorrectionPipeline<Minhasher>::Config config){
            
            ComplexGpuCorrectionPipeline<Minhasher> pipeline(readStorage, minhasher, nullptr, &clfAgent_);

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

        //std::cerr << "numDevice " << numDevices << ", requiredNumThreadsForComplex " << requiredNumThreadsForComplex << ", availableThreads " << availableThreads << "\n";

        // auto launchSimplePipelines = [&](int firstIdIndex, int lastIdIndex){
        //     constexpr int maxNumThreadsPerDevice = 3;
        //     assert(lastIdIndex <= numDevices);

        //     std::vector<int> numThreadsPerDevice(numDevices, 0);

        //     for(int i = 0; i < maxNumThreadsPerDevice; i++){
        //         for(int d = firstIdIndex; d < lastIdIndex; d++){
        //             if(availableThreads > 0){
        //                 futures.emplace_back(std::async(
        //                     std::launch::async,
        //                     runSimpleGpuPipeline,
        //                     deviceIds[d]
        //                 ));

        //                 availableThreads--;

        //                 numThreadsPerDevice[d]++;
        //             }
        //         }
        //     }

        //     for(int d = firstIdIndex; d < lastIdIndex; d++){
        //         if(numThreadsPerDevice[d] > 0){
        //             std::cerr << "Use " << numThreadsPerDevice[d] << " simple threads on device " << deviceIds[d] << "\n";
        //         }else{
        //             std::cerr << "Device " << deviceIds[d] << " will be unused. (Not enough threads available.)\n";
        //         }
        //     }
        // };

        for(int i = 0; i < numDevices; i++){ 
            const int deviceId = deviceIds[i];

            int threadsForDevice = std::max(1,std::min(availableThreads, requiredNumThreadsForComplex));

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
                        deviceId, pipelineConfig
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
                        deviceId
                    ));

                    threadsForDevice--;
                }
            }
        }

        //if there are not enough threads to run one complex pipeline on any device, only use simple pipelines
        // if(requiredNumThreadsForComplex > availableThreads){
        //     launchSimplePipelines(0, numDevices);
        // }else{

            // std::vector<bool> useComplexPipeline(numDevices, false);
            // int numSimple = 0;
            // int firstSimpleDevice = numDevices;

            // for(int i = 0; i < numDevices; i++){            

            //     if(availableThreads >= requiredNumThreadsForComplex){  
            //         availableThreads -= requiredNumThreadsForComplex;
            //     }else{
            //         numSimple++;

            //         if(firstSimpleDevice == numDevices){
            //             firstSimpleDevice = i;
            //         }
            //     }
            // }

            // for(int i = 0; i < firstSimpleDevice; i++){
            //     const int deviceId = deviceIds[i];

            //     typename ComplexGpuCorrectionPipeline<Minhasher>::Config pipelineConfig;
            //     pipelineConfig.numHashers = numHashersPerCorrectorByTime + 2;
            //     pipelineConfig.numCorrectors = 1 + 1;
            //     pipelineConfig.numOutputConstructors = 0;

            //     // std::cerr << "\nWill use " << pipelineConfig.numHashers << " hasher(s), " 
            //     // << pipelineConfig.numCorrectors << " corrector(s), " 
            //     // << pipelineConfig.numOutputConstructors << " output constructor(s) "
            //     // << "on device " << deviceId << "\n";                

            //     futures.emplace_back(
            //         std::async(
            //             std::launch::async,
            //             runComplexGpuPipeline,
            //             deviceId, pipelineConfig
            //         )
            //     );
            // }

            // launchSimplePipelines(firstSimpleDevice, numDevices);
        //}

        //std::cerr << "Remaing threads after launching gpu pipelines: " << availableThreads << "\n";

        //use remaining threads to correct on the host
        // for(int i = 0; i < availableThreads; i++){
        //     futures.emplace_back(
        //         std::async(
        //             std::launch::async,
        //             runSimpleCpuPipeline,
        //             deviceIds[i % deviceIds.size()]
        //         )
        //     );                
        // }

        for(auto& f : futures){
            f.wait();
        }


        
    }

#if 0

auto runPipeline = [&](int deviceId){
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
        GpuReadStorage& readStorage){

    return correct_gpu_impl(
        goodAlignmentProperties,
        correctionOptions,
        runtimeOptions,
        fileOptions,
        memoryOptions,
        minhasher,
        readStorage
    );
}


}
}

