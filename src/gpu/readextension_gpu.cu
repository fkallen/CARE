
#include <gpu/gpuminhasher.cuh>
#include <gpu/gpureadstorage.cuh>
#include <gpu/readextender_gpu.hpp>

#include <config.hpp>
#include <sequencehelpers.hpp>
#include <options.hpp>
#include <cpu_alignment.hpp>
#include <bestalignment.hpp>
#include <msa.hpp>
#include <concurrencyhelpers.hpp>

#include <hpc_helpers.cuh>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>
#include <iostream>
#include <string>
#include <memory>
#include <mutex>
#include <numeric>


#include <extensionresultprocessing.hpp>
#include <rangegenerator.hpp>
#include <threadpool.hpp>
#include <memoryfile.hpp>
#include <util.hpp>
#include <filehelpers.hpp>

#include <omp.h>
#include <cub/cub.cuh>


namespace care{
namespace gpu{



void initializePairedEndExtensionBatchData(
    BatchData& batchData,
    const std::vector<ExtendInput>& inputs,
    std::size_t encodedSequencePitchInInts, 
    std::size_t decodedSequencePitchInBytes, 
    std::size_t msaColumnPitchInElements
){
    std::vector<ReadExtenderBase::Task> tasks(inputs.size() * 2);

    //std::cerr << "Transform LR " << batchId << "\n";
    auto itertmp = std::transform(inputs.begin(), inputs.end(), tasks.begin(), 
        [](auto&& i){return ReadExtenderBase::makePairedEndTask(std::move(i), ExtensionDirection::LR);});

    std::transform(inputs.begin(), inputs.end(), itertmp, 
        [](auto&& i){return ReadExtenderBase::makePairedEndTask(std::move(i), ExtensionDirection::RL);});

    batchData.init(
        std::move(tasks), 
        encodedSequencePitchInInts, 
        decodedSequencePitchInBytes, 
        msaColumnPitchInElements
    );
}

void initializePairedEndExtensionBatchData2(
    BatchData& batchData,
    const std::vector<ExtendInput>& inputs,
    std::size_t encodedSequencePitchInInts, 
    std::size_t decodedSequencePitchInBytes, 
    std::size_t msaColumnPitchInElements
){
    const int batchsizePairs = inputs.size();
    if(batchsizePairs == 0){
        batchData.tasks.clear();
        return;
    }

    batchData.tasks.resize(batchsizePairs * 2);

    for(int i = 0; i < batchsizePairs; i++){
        const auto& input = inputs[i];

        auto& taskl = batchData.tasks[2*i];
        taskl.reset();

        taskl.pairedEnd = true;
        taskl.direction = ExtensionDirection::LR;      
        taskl.currentAnchor = std::move(input.encodedRead1);
        taskl.encodedMate = std::move(input.encodedRead2);
        taskl.currentAnchorLength = input.readLength1;
        taskl.currentAnchorReadId = input.readId1;
        taskl.accumExtensionLengths = 0;
        taskl.iteration = 0;
        taskl.myLength = input.readLength1;
        taskl.myReadId = input.readId1;
        taskl.mateLength = input.readLength2;
        taskl.mateReadId = input.readId2;
        taskl.decodedMate.resize(taskl.mateLength);
        SequenceHelpers::decode2BitSequence(
            taskl.decodedMate.data(),
            taskl.encodedMate.data(),
            taskl.mateLength
        );
        taskl.decodedMateRevC = SequenceHelpers::reverseComplementSequenceDecoded(taskl.decodedMate.data(), taskl.mateLength);
        taskl.resultsequence.resize(input.readLength1);
        SequenceHelpers::decode2BitSequence(
            taskl.resultsequence.data(),
            taskl.currentAnchor.data(),
            taskl.myLength
        );

        taskl.totalDecodedAnchors.emplace_back(taskl.resultsequence);
        taskl.totalAnchorBeginInExtendedRead.emplace_back(0);

        auto& taskr = batchData.tasks[2*i + 1];
        taskr.reset();

        taskr.pairedEnd = true;
        taskr.direction = ExtensionDirection::RL;
        taskr.currentAnchor = taskl.encodedMate;
        taskr.encodedMate = taskl.currentAnchor;

        taskr.currentAnchorLength = input.readLength2;
        taskr.currentAnchorReadId = input.readId2;
        taskr.accumExtensionLengths = 0;
        taskr.iteration = 0;

        taskr.myLength = input.readLength2;
        taskr.myReadId = input.readId2;

        taskr.mateLength = input.readLength1;
        taskr.mateReadId = input.readId1;

        taskr.decodedMate = taskl.resultsequence;        
        taskr.decodedMateRevC = SequenceHelpers::reverseComplementSequenceDecoded(taskr.decodedMate.data(), taskr.mateLength);

        taskr.resultsequence = taskl.decodedMate;
        taskr.totalDecodedAnchors.emplace_back(taskr.resultsequence);
        taskr.totalAnchorBeginInExtendedRead.emplace_back(0);
    }

    batchData.encodedSequencePitchInInts = encodedSequencePitchInInts;
    batchData.decodedSequencePitchInBytes = decodedSequencePitchInBytes;
    batchData.msaColumnPitchInElements = msaColumnPitchInElements;

    batchData.indicesOfActiveTasks.resize(batchData.tasks.size());
    std::iota(batchData.indicesOfActiveTasks.begin(), batchData.indicesOfActiveTasks.end(), 0);

    batchData.splitTracker.clear();
    for(const auto& t : batchData.tasks){
        batchData.splitTracker[t.myReadId] = 1;
    }

    batchData.pairedEnd = batchData.tasks[0].pairedEnd;
}


MemoryFileFixedSize<ExtendedRead> 
//std::vector<ExtendedRead>
extend_gpu_pairedend(
    const GoodAlignmentProperties& goodAlignmentProperties,
    const CorrectionOptions& correctionOptions,
    const ExtensionOptions& extensionOptions,
    const RuntimeOptions& runtimeOptions,
    const FileOptions& fileOptions,
    const MemoryOptions& memoryOptions,
    const GpuMinhasher& minhasher,
    const GpuReadStorage& gpuReadStorage
){
    const auto rsMemInfo = gpuReadStorage.getMemoryInfo();
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

    std::unique_ptr<std::uint8_t[]> correctionStatusFlagsPerRead = std::make_unique<std::uint8_t[]>(gpuReadStorage.getNumberOfReads());

    #pragma omp parallel for
    for(read_number i = 0; i < gpuReadStorage.getNumberOfReads(); i++){
        correctionStatusFlagsPerRead[i] = 0;
    }

    std::cerr << "correctionStatusFlagsPerRead bytes: " << sizeof(std::uint8_t) * gpuReadStorage.getNumberOfReads() / 1024. / 1024. << " MB\n";

    if(memoryAvailableBytesHost > sizeof(std::uint8_t) * gpuReadStorage.getNumberOfReads()){
        memoryAvailableBytesHost -= sizeof(std::uint8_t) * gpuReadStorage.getNumberOfReads();
    }else{
        memoryAvailableBytesHost = 0;
    }

    const std::size_t availableMemoryInBytes = memoryAvailableBytesHost; //getAvailableMemoryInKB() * 1024;
    std::size_t memoryForPartialResultsInBytes = 0;

    if(availableMemoryInBytes > 3*(std::size_t(1) << 30)){
        memoryForPartialResultsInBytes = availableMemoryInBytes - 3*(std::size_t(1) << 30);
    }

    const std::string tmpfilename{fileOptions.tempdirectory + "/" + "MemoryFileFixedSizetmp"};
    MemoryFileFixedSize<ExtendedRead> partialResults(memoryForPartialResultsInBytes, tmpfilename);

    std::vector<ExtendedRead> resultExtendedReads;

    //cpu::RangeGenerator<read_number> readIdGenerator(gpuReadStorage.getNumberOfReads());
    cpu::RangeGenerator<read_number> readIdGenerator(1000000);
    //readIdGenerator.skip(4200000);
 
    BackgroundThread outputThread(true);

    const std::uint64_t totalNumReadPairs = gpuReadStorage.getNumberOfReads() / 2;

    auto showProgress = [&](auto totalCount, auto seconds){
        if(runtimeOptions.showProgress){

            printf("Processed %10u of %10lu read pairs (Runtime: %03d:%02d:%02d)\r",
                    totalCount, totalNumReadPairs,
                    int(seconds / 3600),
                    int(seconds / 60) % 60,
                    int(seconds) % 60);
            std::cout.flush();
        }

        if(totalCount == totalNumReadPairs){
            std::cerr << '\n';
        }
    };

    auto updateShowProgressInterval = [](auto duration){
        return duration;
    };

    ProgressThread<read_number> progressThread(totalNumReadPairs, showProgress, updateShowProgressInterval);

    
    const int insertSize = extensionOptions.insertSize;
    const int insertSizeStddev = extensionOptions.insertSizeStddev;
    const int maximumSequenceLength = gpuReadStorage.getSequenceLengthUpperBound();
    const std::size_t encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength);
    const std::size_t decodedSequencePitchInBytes = maximumSequenceLength;
    const std::size_t qualityPitchInBytes = maximumSequenceLength;

    const std::size_t min_overlap = std::max(
        1, 
        std::max(
            goodAlignmentProperties.min_overlap, 
            int(maximumSequenceLength * goodAlignmentProperties.min_overlap_ratio)
        )
    );
    const std::size_t msa_max_column_count = (3*gpuReadStorage.getSequenceLengthUpperBound() - 2*min_overlap);
    //round up to 32 elements
    const std::size_t msaColumnPitchInElements = SDIV(msa_max_column_count, 32) * 32;

    constexpr int maxextensionPerStep = 20;

    std::mutex ompCriticalMutex;

    std::int64_t totalNumSuccess0 = 0;
    std::int64_t totalNumSuccess1 = 0;
    std::int64_t totalNumSuccess01 = 0;
    std::int64_t totalNumSuccessRead = 0;

    std::map<int, int> totalExtensionLengthsMap;

    std::map<int, int> totalMismatchesBetweenMateExtensions;

    //omp_set_num_threads(1);

    cudaSetDevice(runtimeOptions.deviceIds[0]); CUERR;

    const int batchsizePairs = correctionOptions.batchsize;
    const int numBatchesToProcess = SDIV(readIdGenerator.getEnd(), batchsizePairs * 2);

    #if 1

    #if 1

    constexpr int numparallelbatches = 16;

    int numInitializerThreads = 1;
    int numCpuWorkerThreads = 14;
    int numGpuWorkerThreads = 2;

    std::vector<BatchData> batches(numparallelbatches);
    SimpleConcurrentQueue<BatchData*> freeBatchesQueue;
    SimpleConcurrentQueue<BatchData*> cpuWorkBatchesQueue;
    SimpleConcurrentQueue<BatchData*> gpuWorkBatchesQueue;

    for(int i = 0; i < numparallelbatches; i++){
        batches[i].someId = i;
    }

    for(auto& batch : batches){
        freeBatchesQueue.push(&batch);
    }

    std::mutex processedMutex{};
    std::atomic<int> numProcessedBatchesByCpuWorkerThreads = 0;

    auto initializerThreadFunc = [&](){

        cub::CachingDeviceAllocator cubAllocator;

        ReadStorageHandle readStorageHandle = gpuReadStorage.makeHandle();

        helpers::SimpleAllocationPinnedHost<read_number> currentIds(2 * batchsizePairs);
        helpers::SimpleAllocationPinnedHost<unsigned int> currentEncodedReads(2 * encodedSequencePitchInInts * batchsizePairs);
        helpers::SimpleAllocationPinnedHost<int> currentReadLengths(2 * batchsizePairs);

        CudaStream stream;

        while(!(readIdGenerator.empty())){

            auto readIdsEnd = readIdGenerator.next_n_into_buffer(
                batchsizePairs * 2, 
                currentIds.get()
            );

            const int numReadsInBatch = std::distance(currentIds.get(), readIdsEnd);

            if(numReadsInBatch % 2 == 1){
                throw std::runtime_error("Input files not properly paired. Aborting read extension.");
            }
            
            if(numReadsInBatch == 0){
                continue; //this should only happen if all reads have been processed
            }

            nvtx::push_range("init", 2);

            gpuReadStorage.gatherSequences(
                readStorageHandle,
                currentEncodedReads.get(),
                encodedSequencePitchInInts,
                currentIds.get(),
                currentIds.get(), //device accessible
                numReadsInBatch,
                stream
            );

            gpuReadStorage.gatherSequenceLengths(
                readStorageHandle,
                currentReadLengths.get(),
                currentIds.get(),
                numReadsInBatch,
                stream
            );

            cudaStreamSynchronize(stream); CUERR;

            const int numReadPairsInBatch = numReadsInBatch / 2;

            std::vector<ExtendInput> inputs(numReadPairsInBatch); 

            for(int i = 0; i < numReadPairsInBatch; i++){
                auto& input = inputs[i];

                input.readLength1 = currentReadLengths[2*i];
                input.readLength2 = currentReadLengths[2*i+1];
                input.readId1 = currentIds[2*i];
                input.readId2 = currentIds[2*i+1];
                input.encodedRead1.resize(encodedSequencePitchInInts);
                input.encodedRead2.resize(encodedSequencePitchInInts);
                std::copy_n(currentEncodedReads.get() + (2*i) * encodedSequencePitchInInts, encodedSequencePitchInInts, input.encodedRead1.begin());
                std::copy_n(currentEncodedReads.get() + (2*i + 1) * encodedSequencePitchInInts, encodedSequencePitchInInts, input.encodedRead2.begin());
            }

            //std::cerr << "initializer freeBatchesQueue.pop()\n";
            BatchData* batchData = freeBatchesQueue.pop();
            assert(batchData != nullptr);

            
            initializePairedEndExtensionBatchData2(
                *batchData,
                inputs,
                encodedSequencePitchInInts, 
                decodedSequencePitchInBytes, 
                msaColumnPitchInElements
            );

            batchData->setState(BatchData::State::BeforePrepare);

            nvtx::pop_range();

            //std::cerr << "initializer firstIterationBatchesQueue.push()\n";
            cpuWorkBatchesQueue.push(batchData);
        }

        // std::lock_guard<std::mutex> lg(flushMutex);
        // numRemainingFirstIterationBatchesQueueProducers--;
        // if(numRemainingFirstIterationBatchesQueueProducers == 0){
        //     //std::cerr << "initializer thread flushes firstIterationBatchesQueue by " << numHasherThreads << "\n";
        //     // for(int i = 0; i < numHasherThreads; i++){
        //     //     firstIterationBatchesQueue.push(nullptr);
        //     // }
        //     //firstIterationBatchesQueue.enableDefaultElement(nullptr);
        // }

        gpuReadStorage.destroyHandle(readStorageHandle);

        std::cerr << "initializerThreadFunc finished\n";
    };

    auto cpuWorkerThreadFunc = [&](){
        std::int64_t numSuccessRead = 0;

        cub::CachingDeviceAllocator cubAllocator;

        GpuReadHasher gpuReadHasher(minhasher);

        GpuExtensionStepper gpuExtensionStepper(
            gpuReadStorage, 
            correctionOptions,
            goodAlignmentProperties,
            insertSize,
            insertSizeStddev,
            maxextensionPerStep,
            cubAllocator
        );

        BatchData* batchData = cpuWorkBatchesQueue.pop();

        auto prepare = [&](){
            nvtx::push_range("prepare", 0);
            gpuExtensionStepper.prepareStep(*batchData);
            batchData->setState(BatchData::State::BeforeHash);
            nvtx::pop_range();
        };

        auto hash = [&](){
            nvtx::push_range("hash", 1);
            gpuReadHasher.getCandidateReadIds(*batchData);
            batchData->setState(BatchData::State::BeforeStep);
            nvtx::pop_range();
        };

        auto step = [&](){
            throw std::runtime_error("Error. Observed state BeforeStep in cpuWorkerThread");
        };

        auto extend = [&](){
            nvtx::push_range("extend", 3);
            gpuExtensionStepper.extendAfterStep(*batchData);
            if(!batchData->isEmpty()){
                batchData->setState(BatchData::State::BeforePrepare);
            }else{
                batchData->setState(BatchData::State::BeforeOutput);
            }
            nvtx::pop_range();
        };

        auto output = [&](){
            nvtx::push_range("output", 5);
            std::vector<ExtendResult> extendResults;

            for(const auto& task : batchData->tasks){

                ExtendResult extendResult;
                extendResult.direction = task.direction;
                extendResult.numIterations = task.iteration;
                extendResult.aborted = task.abort;
                extendResult.abortReason = task.abortReason;
                extendResult.readId1 = task.myReadId;
                extendResult.readId2 = task.mateReadId;
                extendResult.originalLength = task.myLength;
                //construct extended read
                //build msa of all saved totalDecodedAnchors[0]

                const int numsteps = task.totalDecodedAnchors.size();

                // if(task.myReadId == 90 || task.mateReadId == 90){
                //     std::cerr << "task.totalDecodedAnchors\n";
                // }

                int maxlen = 0;
                for(const auto& s: task.totalDecodedAnchors){
                    const int len = s.length();
                    if(len > maxlen){
                        maxlen = len;
                    }

                    // if(task.myReadId == 90 || task.mateReadId == 90){
                    //     std::cerr << s << "\n";
                    // }
                }

                // if(task.myReadId == 90 || task.mateReadId == 90){
                //     std::cerr << "\n";
                // }

                const std::string& decodedAnchor = task.totalDecodedAnchors[0];

                const std::vector<int> shifts(task.totalAnchorBeginInExtendedRead.begin() + 1, task.totalAnchorBeginInExtendedRead.end());
                std::vector<float> initialWeights(numsteps-1, 1.0f);


                std::vector<char> stepstrings(maxlen * (numsteps-1), '\0');
                std::vector<int> stepstringlengths(numsteps-1);
                for(int c = 1; c < numsteps; c++){
                    std::copy(
                        task.totalDecodedAnchors[c].begin(),
                        task.totalDecodedAnchors[c].end(),
                        stepstrings.begin() + (c-1) * maxlen
                    );
                    stepstringlengths[c-1] = task.totalDecodedAnchors[c].size();
                }

                MultipleSequenceAlignment::InputData msaInput;
                msaInput.useQualityScores = false;
                msaInput.subjectLength = decodedAnchor.length();
                msaInput.nCandidates = numsteps-1;
                msaInput.candidatesPitch = maxlen;
                msaInput.candidateQualitiesPitch = 0;
                msaInput.subject = decodedAnchor.c_str();
                msaInput.candidates = stepstrings.data();
                msaInput.subjectQualities = nullptr;
                msaInput.candidateQualities = nullptr;
                msaInput.candidateLengths = stepstringlengths.data();
                msaInput.candidateShifts = shifts.data();
                msaInput.candidateDefaultWeightFactors = initialWeights.data();

                MultipleSequenceAlignment msa;

                msa.build(msaInput);

                extendResult.success = true;

                std::string extendedRead(msa.consensus.begin(), msa.consensus.end());

                std::copy(decodedAnchor.begin(), decodedAnchor.end(), extendedRead.begin());
                if(task.mateHasBeenFound){
                    std::copy(
                        task.decodedMateRevC.begin(),
                        task.decodedMateRevC.end(),
                        extendedRead.begin() + extendedRead.length() - task.decodedMateRevC.length()
                    );
                }

                extendResult.extendedRead = std::move(extendedRead);

                extendResult.mateHasBeenFound = task.mateHasBeenFound;

                extendResults.emplace_back(std::move(extendResult));
            }

            std::vector<ExtendResult> extendResultsCombined = ReadExtenderBase::combinePairedEndDirectionResults(
                extendResults,
                insertSize,
                insertSizeStddev
            );

            std::vector<ExtendedRead> resultvector(extendResultsCombined.size());

            const int numReadPairsInBatch = resultvector.size();
            
            for(int i = 0; i < numReadPairsInBatch; i++){
                auto& extensionOutput = extendResultsCombined[i];
                ExtendedRead& er = resultvector[i];

                er.readId = extensionOutput.readId1;
                er.extendedSequence = std::move(extensionOutput.extendedRead);
                assert(er.extendedSequence.size() < 500);

                if(extensionOutput.mateHasBeenFound){
                    er.status = ExtendedReadStatus::FoundMate;
                }else{
                    if(extensionOutput.aborted){
                        if(extensionOutput.abortReason == AbortReason::NoPairedCandidates
                                || extensionOutput.abortReason == AbortReason::NoPairedCandidatesAfterAlignment){

                            er.status = ExtendedReadStatus::CandidateAbort;
                        }else if(extensionOutput.abortReason == AbortReason::MsaNotExtended){
                            er.status = ExtendedReadStatus::MSANoExtension;
                        }
                    }else{
                        er.status = ExtendedReadStatus::LengthAbort;
                    }
                }  
                
                if(extensionOutput.success){
                    numSuccessRead++;
                }                
            }

            auto outputfunc = [&, vec = std::move(resultvector)](){
                for(const auto& er : vec){
                    partialResults.storeElement(&er);
                }
            };

            outputThread.enqueue(
                std::move(outputfunc)
            );

            batchData->setState(BatchData::State::None);

            nvtx::pop_range();

            progressThread.addProgress(numReadPairsInBatch);
        };
        
        while(batchData != nullptr){

            switch(batchData->state){
            case BatchData::State::BeforePrepare:
                    prepare();
                    hash();
                    gpuWorkBatchesQueue.push(batchData);
                    break;
            case BatchData::State::BeforeHash:
                    hash();
                    gpuWorkBatchesQueue.push(batchData);
                    break;
            case BatchData::State::BeforeStep:
                    step();
                    break;
            case BatchData::State::BeforeExtend:
                    extend();
                    cpuWorkBatchesQueue.push(batchData);
                    break;
            case BatchData::State::BeforeOutput:
                    output();
                    freeBatchesQueue.push(batchData);

                    numProcessedBatchesByCpuWorkerThreads++;

                    if(numProcessedBatchesByCpuWorkerThreads == numBatchesToProcess){
                        freeBatchesQueue.enableDefaultElement(nullptr);
                        cpuWorkBatchesQueue.enableDefaultElement(nullptr);
                        gpuWorkBatchesQueue.enableDefaultElement(nullptr);
                    }
                    break;
            default:
                    throw std::runtime_error("Error: Did not implement case");
                    break;
            };

            batchData = cpuWorkBatchesQueue.pop();
        }

        std::cerr << "cpuWorkerThreadFunc finished\n";
    };

    auto gpuWorkerThreadFunc = [&](){
        cub::CachingDeviceAllocator cubAllocator;

        GpuExtensionStepper gpuExtensionStepper(
            gpuReadStorage, 
            correctionOptions,
            goodAlignmentProperties,
            insertSize,
            insertSizeStddev,
            maxextensionPerStep,
            cubAllocator
        );        

        BatchData* batchData = gpuWorkBatchesQueue.pop();

        while(batchData != nullptr){

            assert(batchData->state == BatchData::State::BeforeStep);

            nvtx::push_range("step", 4);

            gpuExtensionStepper.step(*batchData);

            batchData->setState(BatchData::State::BeforeExtend);

            nvtx::pop_range();

            cpuWorkBatchesQueue.push(batchData);

            batchData = gpuWorkBatchesQueue.pop();
        }

        std::cerr << "gpuWorkerThreadFunc finished\n";
    };

    std::vector<std::future<void>> futures;

    for(int i = 0; i < numInitializerThreads; i++){
        futures.emplace_back(std::async(std::launch::async, initializerThreadFunc));
    }

    for(int i = 0; i < numCpuWorkerThreads; i++){
        futures.emplace_back(std::async(std::launch::async, cpuWorkerThreadFunc));
    }

    for(int i = 0; i < numGpuWorkerThreads; i++){
        futures.emplace_back(std::async(std::launch::async, gpuWorkerThreadFunc));
    }

    for(auto& f : futures){
        f.wait();
    }

    #else

    constexpr int numparallelbatches = 16;

    int numInitializerThreads = 1;
    int numHasherThreads = 12;
    int numStepperThreads = 2;
    int numFinalizerThreads = 1;

    std::vector<BatchData> batches(numparallelbatches);
    SimpleConcurrentQueue<BatchData*> freeBatchesQueue;
    SimpleConcurrentQueue<BatchData*> firstIterationBatchesQueue;
    SimpleConcurrentQueue<BatchData*> advancedIterationBatchesQueue;
    SimpleConcurrentQueue<BatchData*> ongoingBatchesQueue;
    SimpleConcurrentQueue<BatchData*> finishedBatchesQueue;

    for(auto& batch : batches){
        freeBatchesQueue.push(&batch);
    }

    std::mutex processedMutex{};
    std::atomic<int> numProcessedBatchesByInitializerThreads = 0;
    std::atomic<int> numProcessedBatchesByStepperThreads = 0;

    std::mutex flushMutex{};
    int numRemainingFirstIterationBatchesQueueProducers = numInitializerThreads;
    int numRemainingAdvancedIterationBatchesQueueProducers = numStepperThreads;
    int numRemainingOngoingBatchesQueueProducers = numHasherThreads;
    int numRemainingFinishedBatchesQueueProducers = numStepperThreads;


    cub::CachingDeviceAllocator pipelineCubAllocator{};

    auto initializerThreadFunc = [&](){

        cub::CachingDeviceAllocator cubAllocator;

        ReadStorageHandle readStorageHandle = gpuReadStorage.makeHandle();

        GpuExtensionStepper gpuExtensionStepper(
            gpuReadStorage, 
            correctionOptions,
            goodAlignmentProperties,
            insertSize,
            insertSizeStddev,
            maxextensionPerStep,
            cubAllocator
        );

        helpers::SimpleAllocationPinnedHost<read_number> currentIds(2 * batchsizePairs);
        helpers::SimpleAllocationPinnedHost<unsigned int> currentEncodedReads(2 * encodedSequencePitchInInts * batchsizePairs);
        helpers::SimpleAllocationPinnedHost<int> currentReadLengths(2 * batchsizePairs);

        CudaStream stream;

        int myNumProcessed = 0;

        while(!(readIdGenerator.empty())){

            auto readIdsEnd = readIdGenerator.next_n_into_buffer(
                batchsizePairs * 2, 
                currentIds.get()
            );

            const int numReadsInBatch = std::distance(currentIds.get(), readIdsEnd);

            if(numReadsInBatch % 2 == 1){
                throw std::runtime_error("Input files not properly paired. Aborting read extension.");
            }
            
            if(numReadsInBatch == 0){
                continue; //this should only happen if all reads have been processed
            }

            gpuReadStorage.gatherSequences(
                readStorageHandle,
                currentEncodedReads.get(),
                encodedSequencePitchInInts,
                currentIds.get(),
                currentIds.get(), //device accessible
                numReadsInBatch,
                stream
            );

            gpuReadStorage.gatherSequenceLengths(
                readStorageHandle,
                currentReadLengths.get(),
                currentIds.get(),
                numReadsInBatch,
                stream
            );

            cudaStreamSynchronize(stream); CUERR;

            const int numReadPairsInBatch = numReadsInBatch / 2;

            std::vector<ExtendInput> inputs(numReadPairsInBatch); 

            for(int i = 0; i < numReadPairsInBatch; i++){
                auto& input = inputs[i];

                input.readLength1 = currentReadLengths[2*i];
                input.readLength2 = currentReadLengths[2*i+1];
                input.readId1 = currentIds[2*i];
                input.readId2 = currentIds[2*i+1];
                input.encodedRead1.resize(encodedSequencePitchInInts);
                input.encodedRead2.resize(encodedSequencePitchInInts);
                std::copy_n(currentEncodedReads.get() + (2*i) * encodedSequencePitchInInts, encodedSequencePitchInInts, input.encodedRead1.begin());
                std::copy_n(currentEncodedReads.get() + (2*i + 1) * encodedSequencePitchInInts, encodedSequencePitchInInts, input.encodedRead2.begin());
            }

            //std::cerr << "initializer freeBatchesQueue.pop()\n";
            BatchData* batchData = freeBatchesQueue.pop();
            assert(batchData != nullptr);

            nvtx::push_range("initAndPrepare", 2);
            initializePairedEndExtensionBatchData(
                *batchData,
                inputs,
                encodedSequencePitchInInts, 
                decodedSequencePitchInBytes, 
                msaColumnPitchInElements
            );

            gpuExtensionStepper.prepareStep(*batchData);
            batchData->needPrepareStep = false;
            nvtx::pop_range();

            //std::cerr << "initializer firstIterationBatchesQueue.push()\n";
            firstIterationBatchesQueue.push(batchData);

            myNumProcessed = 0;
        }

        numProcessedBatchesByInitializerThreads += myNumProcessed;

        std::lock_guard<std::mutex> lg(flushMutex);
        numRemainingFirstIterationBatchesQueueProducers--;
        if(numRemainingFirstIterationBatchesQueueProducers == 0){
            //std::cerr << "initializer thread flushes firstIterationBatchesQueue by " << numHasherThreads << "\n";
            // for(int i = 0; i < numHasherThreads; i++){
            //     firstIterationBatchesQueue.push(nullptr);
            // }
            //firstIterationBatchesQueue.enableDefaultElement(nullptr);
        }

        gpuReadStorage.destroyHandle(readStorageHandle);

        std::cerr << "initializerThreadFunc finished\n";
    };

    auto hasherThreadFunc = [&](){
        cub::CachingDeviceAllocator cubAllocator;

        GpuReadHasher gpuReadHasher(minhasher);

        GpuExtensionStepper gpuExtensionStepper(
            gpuReadStorage, 
            correctionOptions,
            goodAlignmentProperties,
            insertSize,
            insertSizeStddev,
            maxextensionPerStep,
            cubAllocator
        );

        bool firstIterationQueueClosed = false;
        bool advancedIterationQueueClosed = false;

        BatchData* batchData = firstIterationBatchesQueue.pop();
        
        // if(!firstIterationQueueClosed){
        //     batchData = firstIterationBatchesQueue.pop();

        //     if(batchData == nullptr){
        //         firstIterationQueueClosed = true;
        //     }
        // }
        // if(batchData == nullptr){
        //     if(!advancedIterationQueueClosed){
        //         batchData = advancedIterationBatchesQueue.pop();
    
        //         if(batchData == nullptr){
        //             advancedIterationQueueClosed = true;
        //         }
        //     }
        // }

        while(batchData != nullptr){

            nvtx::push_range("hash", 3);

            if(batchData->needPrepareStep){
                gpuExtensionStepper.prepareStep(*batchData);
                batchData->needPrepareStep = false;
            }

            gpuReadHasher.getCandidateReadIds(*batchData);

            nvtx::pop_range();

            //std::cerr << "hasher ongoingBatchesQueue.push()\n";
            ongoingBatchesQueue.push(batchData);

            batchData = firstIterationBatchesQueue.pop();  
               
            // if(!advancedIterationQueueClosed){
            //     batchData = advancedIterationBatchesQueue.pop();
    
            //     if(batchData == nullptr){
            //         advancedIterationQueueClosed = true;
            //     }
            // }            
            // if(batchData == nullptr){
            //     if(!firstIterationQueueClosed){
            //         batchData = firstIterationBatchesQueue.pop();
    
            //         if(batchData == nullptr){
            //             firstIterationQueueClosed = true;
            //         }
            //     }
            // }
        }

        std::lock_guard<std::mutex> lg(flushMutex);
        numRemainingOngoingBatchesQueueProducers--;
        if(numRemainingOngoingBatchesQueueProducers == 0){
            //std::cerr << "hasherThreadFunc thread flushes ongoingBatchesQueue by " << numStepperThreads << "\n";
            // for(int i = 0; i < numStepperThreads; i++){
            //     ongoingBatchesQueue.push(nullptr);
            // }
            //ongoingBatchesQueue.enableDefaultElement(nullptr);
        }

        std::cerr << "hasherThreadFunc finished\n";
    };

    auto stepperThreadFunc = [&](){
        cub::CachingDeviceAllocator cubAllocator;

        GpuExtensionStepper gpuExtensionStepper(
            gpuReadStorage, 
            correctionOptions,
            goodAlignmentProperties,
            insertSize,
            insertSizeStddev,
            maxextensionPerStep,
            cubAllocator
        );

        //std::cerr << "stepper ongoingBatchesQueue.pop()\n";

        int myNumProcessed = 0;

        auto popNext = [&](){
            return ongoingBatchesQueue.pop();
        };
        

        BatchData* batchData = popNext();

        while(batchData != nullptr){

            nvtx::push_range("step", 4);

            gpuExtensionStepper.step(*batchData);
            gpuExtensionStepper.extendAfterStep(*batchData);

            nvtx::pop_range();
            
            if(batchData->isEmpty()){
                //std::cerr << "stepper finishedBatchesQueue.push()\n";
                finishedBatchesQueue.push(batchData);
                numProcessedBatchesByStepperThreads++;

                if(numProcessedBatchesByStepperThreads == numBatchesToProcess){
                    firstIterationBatchesQueue.enableDefaultElement(nullptr);
                    finishedBatchesQueue.enableDefaultElement(nullptr);
                    ongoingBatchesQueue.enableDefaultElement(nullptr);
                }
            }else{
                batchData->needPrepareStep = true;
                
                //std::cerr << "stepper firstIterationBatchesQueue.push()\n";
                firstIterationBatchesQueue.push(batchData);
            }

            //std::cerr << "stepper ongoingBatchesQueue.pop()\n";
            batchData = popNext();
        }

        //std::lock_guard<std::mutex> lg(processedMutex);
        //numProcessedBatchesByStepperThreads += myNumProcessed;
        // if(numProcessedBatchesByStepperThreads == numBatchesToProcess){
        //     //std::cerr << "stepperThreadFunc thread flushes finishedBatchesQueue by " << numFinalizerThreads << "\n";
        //     for(int i = 0; i < numFinalizerThreads; i++){
        //         finishedBatchesQueue.push(nullptr);
        //     }
        //     for(int i = 0; i < numHasherThreads; i++){
        //         firstIterationBatchesQueue.push(nullptr);
        //     }
        //     //finishedBatchesQueue.enableDefaultElement(nullptr);
        // }else if(numProcessedBatchesByStepperThreads > numBatchesToProcess){
        //     std::cerr << "Error. Processed " << numProcessedBatchesByStepperThreads << "batches, expected " << numBatchesToProcess << " batches\n";
        //     assert(false);
        // }       

        std::cerr << "stepperThreadFunc finished\n";
    };

    auto finalizerThreadFunc = [&](){
        std::int64_t numSuccess0 = 0;
        std::int64_t numSuccess1 = 0;
        std::int64_t numSuccess01 = 0;
        std::int64_t numSuccessRead = 0;

        //std::cerr << "finalizer finishedBatchesQueue.pop()\n";
        BatchData* batchData = finishedBatchesQueue.pop();

        while(batchData != nullptr){
            nvtx::push_range("finalize", 5);
            std::vector<ExtendResult> extendResults;

            for(const auto& task : batchData->tasks){

                ExtendResult extendResult;
                extendResult.direction = task.direction;
                extendResult.numIterations = task.iteration;
                extendResult.aborted = task.abort;
                extendResult.abortReason = task.abortReason;
                extendResult.readId1 = task.myReadId;
                extendResult.readId2 = task.mateReadId;
                extendResult.originalLength = task.myLength;
                //construct extended read
                //build msa of all saved totalDecodedAnchors[0]

                const int numsteps = task.totalDecodedAnchors.size();

                // if(task.myReadId == 90 || task.mateReadId == 90){
                //     std::cerr << "task.totalDecodedAnchors\n";
                // }

                int maxlen = 0;
                for(const auto& s: task.totalDecodedAnchors){
                    const int len = s.length();
                    if(len > maxlen){
                        maxlen = len;
                    }

                    // if(task.myReadId == 90 || task.mateReadId == 90){
                    //     std::cerr << s << "\n";
                    // }
                }

                // if(task.myReadId == 90 || task.mateReadId == 90){
                //     std::cerr << "\n";
                // }

                const std::string& decodedAnchor = task.totalDecodedAnchors[0];

                const std::vector<int> shifts(task.totalAnchorBeginInExtendedRead.begin() + 1, task.totalAnchorBeginInExtendedRead.end());
                std::vector<float> initialWeights(numsteps-1, 1.0f);


                std::vector<char> stepstrings(maxlen * (numsteps-1), '\0');
                std::vector<int> stepstringlengths(numsteps-1);
                for(int c = 1; c < numsteps; c++){
                    std::copy(
                        task.totalDecodedAnchors[c].begin(),
                        task.totalDecodedAnchors[c].end(),
                        stepstrings.begin() + (c-1) * maxlen
                    );
                    stepstringlengths[c-1] = task.totalDecodedAnchors[c].size();
                }

                MultipleSequenceAlignment::InputData msaInput;
                msaInput.useQualityScores = false;
                msaInput.subjectLength = decodedAnchor.length();
                msaInput.nCandidates = numsteps-1;
                msaInput.candidatesPitch = maxlen;
                msaInput.candidateQualitiesPitch = 0;
                msaInput.subject = decodedAnchor.c_str();
                msaInput.candidates = stepstrings.data();
                msaInput.subjectQualities = nullptr;
                msaInput.candidateQualities = nullptr;
                msaInput.candidateLengths = stepstringlengths.data();
                msaInput.candidateShifts = shifts.data();
                msaInput.candidateDefaultWeightFactors = initialWeights.data();

                MultipleSequenceAlignment msa;

                msa.build(msaInput);

                extendResult.success = true;

                std::string extendedRead(msa.consensus.begin(), msa.consensus.end());

                std::copy(decodedAnchor.begin(), decodedAnchor.end(), extendedRead.begin());
                if(task.mateHasBeenFound){
                    std::copy(
                        task.decodedMateRevC.begin(),
                        task.decodedMateRevC.end(),
                        extendedRead.begin() + extendedRead.length() - task.decodedMateRevC.length()
                    );
                }

                extendResult.extendedRead = std::move(extendedRead);

                extendResult.mateHasBeenFound = task.mateHasBeenFound;

                extendResults.emplace_back(std::move(extendResult));
            }

            std::vector<ExtendResult> extendResultsCombined = ReadExtenderBase::combinePairedEndDirectionResults(
                extendResults,
                insertSize,
                insertSizeStddev
            );

            std::vector<ExtendedRead> resultvector(extendResultsCombined.size());

            const int numReadPairsInBatch = resultvector.size();

            for(int i = 0; i < numReadPairsInBatch; i++){
                auto& extensionOutput = extendResultsCombined[i];
                ExtendedRead& er = resultvector[i];

                er.readId = extensionOutput.readId1;
                er.extendedSequence = std::move(extensionOutput.extendedRead);

                if(extensionOutput.mateHasBeenFound){
                    er.status = ExtendedReadStatus::FoundMate;
                }else{
                    if(extensionOutput.aborted){
                        if(extensionOutput.abortReason == AbortReason::NoPairedCandidates
                                || extensionOutput.abortReason == AbortReason::NoPairedCandidatesAfterAlignment){

                            er.status = ExtendedReadStatus::CandidateAbort;
                        }else if(extensionOutput.abortReason == AbortReason::MsaNotExtended){
                            er.status = ExtendedReadStatus::MSANoExtension;
                        }
                    }else{
                        er.status = ExtendedReadStatus::LengthAbort;
                    }
                }  
                
                if(extensionOutput.success){
                    numSuccessRead++;
                }                
            }

            auto outputfunc = [&, vec = std::move(resultvector)](){
                for(const auto& er : vec){
                    partialResults.storeElement(&er);
                }
            };

            // outputThread.enqueue(
            //     std::move(outputfunc)
            // );

            outputfunc();

            nvtx::pop_range();

            progressThread.addProgress(numReadPairsInBatch);  

            //std::cerr << "finalizer freeBatchesQueue.push()\n";
            freeBatchesQueue.push(batchData);
            //std::cerr << "finalizer finishedBatchesQueue.pop()\n";
            batchData = finishedBatchesQueue.pop();
        }

        //finalizer thread is sink. don't need to flush queues

        std::cerr << "finalizerThreadFunc finished\n";
    };

    std::vector<std::future<void>> futures;

    for(int i = 0; i < numInitializerThreads; i++){
        futures.emplace_back(std::async(std::launch::async, initializerThreadFunc));
    }

    for(int i = 0; i < numHasherThreads; i++){
        futures.emplace_back(std::async(std::launch::async, hasherThreadFunc));
    }

    for(int i = 0; i < numStepperThreads; i++){
        futures.emplace_back(std::async(std::launch::async, stepperThreadFunc));
    }

    for(int i = 0; i < numFinalizerThreads; i++){
        futures.emplace_back(std::async(std::launch::async, finalizerThreadFunc));
    }

    std::cerr << numRemainingFirstIterationBatchesQueueProducers << ", " 
        << numRemainingAdvancedIterationBatchesQueueProducers << ", " 
        << numRemainingOngoingBatchesQueueProducers << ", " 
        << numRemainingFinishedBatchesQueueProducers << "\n";

    for(auto& f : futures){
        f.wait();

        std::cerr << numRemainingFirstIterationBatchesQueueProducers << ", " 
        << numRemainingAdvancedIterationBatchesQueueProducers << ", " 
        << numRemainingOngoingBatchesQueueProducers << ", " 
        << numRemainingFinishedBatchesQueueProducers << "\n";
    }

    #endif

    #else

    std::atomic<int> numProcessedBatches{0};

    #pragma omp parallel
    {
        const int numDeviceIds = runtimeOptions.deviceIds.size();

        cub::CachingDeviceAllocator cubAllocator{};

        assert(numDeviceIds > 0);

        const int ompThreadId = omp_get_thread_num();
        const int deviceId = runtimeOptions.deviceIds.at(ompThreadId % numDeviceIds);
        cudaSetDevice(deviceId); CUERR;

        GoodAlignmentProperties goodAlignmentProperties2 = goodAlignmentProperties;
        //goodAlignmentProperties2.maxErrorRate = 0.05;

        

        ReadExtenderGpu readExtenderGpu{
            insertSize,
            insertSizeStddev,
            maxextensionPerStep,
            maximumSequenceLength,
            correctionOptions.kmerlength,
            gpuReadStorage, 
            minhasher,
            correctionOptions,
            goodAlignmentProperties2,
            cubAllocator
        };

        std::int64_t numSuccess0 = 0;
        std::int64_t numSuccess1 = 0;
        std::int64_t numSuccess01 = 0;
        std::int64_t numSuccessRead = 0;

        std::map<int, int> extensionLengthsMap;
        std::map<int, int> mismatchesBetweenMateExtensions;

        ReadStorageHandle readStorageHandle = gpuReadStorage.makeHandle();

        helpers::SimpleAllocationPinnedHost<read_number> currentIds(2 * batchsizePairs);
        helpers::SimpleAllocationPinnedHost<unsigned int> currentEncodedReads(2 * encodedSequencePitchInInts * batchsizePairs);
        helpers::SimpleAllocationPinnedHost<int> currentReadLengths(2 * batchsizePairs);

        cudaStream_t stream;
        cudaStreamCreate(&stream); CUERR;
        

        while(!(readIdGenerator.empty())){

            auto readIdsEnd = readIdGenerator.next_n_into_buffer(
                batchsizePairs * 2, 
                currentIds.get()
            );

            const int numReadsInBatch = std::distance(currentIds.get(), readIdsEnd);

            if(numReadsInBatch % 2 == 1){
                throw std::runtime_error("Input files not properly paired. Aborting read extension.");
            }
            
            if(numReadsInBatch == 0){
                continue; //this should only happen if all reads have been processed
            }

            gpuReadStorage.gatherSequences(
                readStorageHandle,
                currentEncodedReads.get(),
                encodedSequencePitchInInts,
                currentIds.get(),
                currentIds.get(), //device accessible
                numReadsInBatch,
                stream
            );

            gpuReadStorage.gatherSequenceLengths(
                readStorageHandle,
                currentReadLengths.get(),
                currentIds.get(),
                numReadsInBatch,
                stream
            );

            cudaStreamSynchronize(stream); CUERR;

            const int numReadPairsInBatch = numReadsInBatch / 2;

            std::vector<ExtendInput> inputs(numReadPairsInBatch); 

            for(int i = 0; i < numReadPairsInBatch; i++){
                auto& input = inputs[i];

                input.readLength1 = currentReadLengths[2*i];
                input.readLength2 = currentReadLengths[2*i+1];
                input.readId1 = currentIds[2*i];
                input.readId2 = currentIds[2*i+1];
                input.encodedRead1.resize(encodedSequencePitchInInts);
                input.encodedRead2.resize(encodedSequencePitchInInts);
                std::copy_n(currentEncodedReads.get() + (2*i) * encodedSequencePitchInInts, encodedSequencePitchInInts, input.encodedRead1.begin());
                std::copy_n(currentEncodedReads.get() + (2*i + 1) * encodedSequencePitchInInts, encodedSequencePitchInInts, input.encodedRead2.begin());
            }

            auto extensionResultsBatch = readExtenderGpu.extendPairedReadBatch(inputs);

            //convert results of ReadExtender
            std::vector<ExtendedRead> resultvector(extensionResultsBatch.size());

            assert(resultvector.size() == std::size_t(numReadPairsInBatch));

            for(int i = 0; i < numReadPairsInBatch; i++){
                auto& extensionOutput = extensionResultsBatch[i];
                ExtendedRead& er = resultvector[i];

                er.readId = extensionOutput.readId1;
                er.extendedSequence = std::move(extensionOutput.extendedRead);

                if(extensionOutput.mateHasBeenFound){
                    er.status = ExtendedReadStatus::FoundMate;
                }else{
                    if(extensionOutput.aborted){
                        if(extensionOutput.abortReason == AbortReason::NoPairedCandidates
                                || extensionOutput.abortReason == AbortReason::NoPairedCandidatesAfterAlignment){

                            er.status = ExtendedReadStatus::CandidateAbort;
                        }else if(extensionOutput.abortReason == AbortReason::MsaNotExtended){
                            er.status = ExtendedReadStatus::MSANoExtension;
                        }
                    }else{
                        er.status = ExtendedReadStatus::LengthAbort;
                    }
                }  
                
                if(extensionOutput.success){
                    numSuccessRead++;
                }                
            }

            auto outputfunc = [&, vec = std::move(resultvector)](){
                for(const auto& er : vec){
                    partialResults.storeElement(&er);
                }
            };

            outputThread.enqueue(
                std::move(outputfunc)
            );

            progressThread.addProgress(numReadPairsInBatch);    
            
            numProcessedBatches++;
        }


        cudaStreamDestroy(stream); CUERR;

        //#pragma omp critical
        {
            std::lock_guard<std::mutex> lg(ompCriticalMutex);

            totalNumSuccess0 += numSuccess0;
            totalNumSuccess1 += numSuccess1;
            totalNumSuccess01 += numSuccess01;
            totalNumSuccessRead += numSuccessRead;

            for(const auto& pair : extensionLengthsMap){
                totalExtensionLengthsMap[pair.first] += pair.second;
            }

            for(const auto& pair : mismatchesBetweenMateExtensions){
                totalMismatchesBetweenMateExtensions[pair.first] += pair.second;
            }

            if(0 == ompThreadId){
                readExtenderGpu.printTimers();
            }      
        }

        
        gpuReadStorage.destroyHandle(readStorageHandle);

    } //end omp parallel

    std::cerr << "numBatchesToProcess: " << numBatchesToProcess << ", numProcessedBatches: " << numProcessedBatches << "\n";

    #endif

    progressThread.finished();

    outputThread.stopThread(BackgroundThread::StopType::FinishAndStop);



    //outputstream.flush();
    partialResults.flush();

    std::cout << "totalNumSuccess0: " << totalNumSuccess0 << std::endl;
    std::cout << "totalNumSuccess1: " << totalNumSuccess1 << std::endl;
    std::cout << "totalNumSuccess01: " << totalNumSuccess01 << std::endl;
    std::cout << "totalNumSuccessRead: " << totalNumSuccessRead << std::endl;

    // std::cout << "Extension lengths:\n";

    // for(const auto& pair : totalExtensionLengthsMap){
    //     std::cout << pair.first << ": " << pair.second << "\n";
    // }

    // std::cout << "mismatches between mate extensions:\n";

    // for(const auto& pair : totalMismatchesBetweenMateExtensions){
    //     std::cout << pair.first << ": " << pair.second << "\n";
    // }



    return partialResults;
    //return resultExtendedReads;
}




MemoryFileFixedSize<ExtendedRead> 
//std::vector<ExtendedRead>
extend_gpu_singleend(
    const GoodAlignmentProperties& goodAlignmentProperties,
    const CorrectionOptions& correctionOptions,
    const ExtensionOptions& extensionOptions,
    const RuntimeOptions& runtimeOptions,
    const FileOptions& fileOptions,
    const MemoryOptions& memoryOptions,
    const GpuMinhasher& minhasher,
    const GpuReadStorage& gpuReadStorage
){
    std::cerr << "extend_gpu_singleend\n";

    const auto rsMemInfo = gpuReadStorage.getMemoryInfo();
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

    std::unique_ptr<std::uint8_t[]> correctionStatusFlagsPerRead = std::make_unique<std::uint8_t[]>(gpuReadStorage.getNumberOfReads());

    #pragma omp parallel for
    for(read_number i = 0; i < gpuReadStorage.getNumberOfReads(); i++){
        correctionStatusFlagsPerRead[i] = 0;
    }

    std::cerr << "correctionStatusFlagsPerRead bytes: " << sizeof(std::uint8_t) * gpuReadStorage.getNumberOfReads() / 1024. / 1024. << " MB\n";

    if(memoryAvailableBytesHost > sizeof(std::uint8_t) * gpuReadStorage.getNumberOfReads()){
        memoryAvailableBytesHost -= sizeof(std::uint8_t) * gpuReadStorage.getNumberOfReads();
    }else{
        memoryAvailableBytesHost = 0;
    }

    const std::size_t availableMemoryInBytes = memoryAvailableBytesHost; //getAvailableMemoryInKB() * 1024;
    std::size_t memoryForPartialResultsInBytes = 0;

    if(availableMemoryInBytes > 3*(std::size_t(1) << 30)){
        memoryForPartialResultsInBytes = availableMemoryInBytes - 3*(std::size_t(1) << 30);
    }

    const std::string tmpfilename{fileOptions.tempdirectory + "/" + "MemoryFileFixedSizetmp"};
    MemoryFileFixedSize<ExtendedRead> partialResults(memoryForPartialResultsInBytes, tmpfilename);

    std::vector<ExtendedRead> resultExtendedReads;

    cpu::RangeGenerator<read_number> readIdGenerator(gpuReadStorage.getNumberOfReads());
    //cpu::RangeGenerator<read_number> readIdGenerator(100000);

    BackgroundThread outputThread(true);

    auto showProgress = [&](auto totalCount, auto seconds){
        if(runtimeOptions.showProgress){

            std::size_t numreads = gpuReadStorage.getNumberOfReads();

            printf("Processed %10u of %10lu read pairs (Runtime: %03d:%02d:%02d)\r",
                    totalCount, numreads,
                    int(seconds / 3600),
                    int(seconds / 60) % 60,
                    int(seconds) % 60);
            std::cout.flush();
        }

        if(totalCount == gpuReadStorage.getNumberOfReads()){
            std::cerr << '\n';
        }
    };

    auto updateShowProgressInterval = [](auto duration){
        return duration;
    };

    ProgressThread<read_number> progressThread(gpuReadStorage.getNumberOfReads(), showProgress, updateShowProgressInterval);

    
    const int insertSize = extensionOptions.insertSize;
    const int insertSizeStddev = extensionOptions.insertSizeStddev;
    const int maximumSequenceLength = gpuReadStorage.getSequenceLengthUpperBound();
    const std::size_t encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength);

    std::mutex ompCriticalMutex;

    std::int64_t totalNumSuccess0 = 0;
    std::int64_t totalNumSuccess1 = 0;
    std::int64_t totalNumSuccess01 = 0;
    std::int64_t totalNumSuccessRead = 0;

    std::map<int, int> totalExtensionLengthsMap;

    std::map<int, int> totalMismatchesBetweenMateExtensions;

    //omp_set_num_threads(1);

    #pragma omp parallel
    {
        const int numDeviceIds = runtimeOptions.deviceIds.size();

        assert(numDeviceIds > 0);

        cub::CachingDeviceAllocator cubAllocator{};

        const int ompThreadId = omp_get_thread_num();
        const int deviceId = runtimeOptions.deviceIds.at(ompThreadId % numDeviceIds);
        cudaSetDevice(deviceId); CUERR;

        GoodAlignmentProperties goodAlignmentProperties2 = goodAlignmentProperties;
        //goodAlignmentProperties2.maxErrorRate = 0.05;

        constexpr int maxextensionPerStep = 20;

        ReadExtenderGpu readExtenderGpu{
            insertSize,
            insertSizeStddev,
            maxextensionPerStep,
            maximumSequenceLength,
            correctionOptions.kmerlength,
            gpuReadStorage, 
            minhasher,
            correctionOptions,
            goodAlignmentProperties2,
            cubAllocator
        };

        std::int64_t numSuccess0 = 0;
        std::int64_t numSuccess1 = 0;
        std::int64_t numSuccess01 = 0;
        std::int64_t numSuccessRead = 0;

        std::map<int, int> extensionLengthsMap;
        std::map<int, int> mismatchesBetweenMateExtensions;

        ReadStorageHandle readStorageHandle = gpuReadStorage.makeHandle();

        const int batchsize = correctionOptions.batchsize;

        helpers::SimpleAllocationPinnedHost<read_number> currentIds(batchsize);
        helpers::SimpleAllocationPinnedHost<unsigned int> currentEncodedReads(encodedSequencePitchInInts * batchsize);
        helpers::SimpleAllocationPinnedHost<int> currentReadLengths(batchsize);

        cudaStream_t stream;
        cudaStreamCreate(&stream); CUERR;
        

        while(!(readIdGenerator.empty())){

            auto readIdsEnd = readIdGenerator.next_n_into_buffer(
                batchsize, 
                currentIds.get()
            );

            const int numReadsInBatch = std::distance(currentIds.get(), readIdsEnd);
            
            if(numReadsInBatch == 0){
                continue; //this should only happen if all reads have been processed
            }

            gpuReadStorage.gatherSequences(
                readStorageHandle,
                currentEncodedReads.get(),
                encodedSequencePitchInInts,
                currentIds.get(),
                currentIds.get(), //device accessible
                numReadsInBatch,
                stream
            );

            gpuReadStorage.gatherSequenceLengths(
                readStorageHandle,
                currentReadLengths.get(),
                currentIds.get(),
                numReadsInBatch,
                stream
            );
    
            cudaStreamSynchronize(stream); CUERR;

            std::vector<ExtendInput> inputs(numReadsInBatch); 

            for(int i = 0; i < numReadsInBatch; i++){
                auto& input = inputs[i];

                input.readLength1 = currentReadLengths[i];
                input.readLength2 = 0;
                input.readId1 = currentIds[i];
                input.readId2 = std::numeric_limits<read_number>::max();
                input.encodedRead1.resize(encodedSequencePitchInInts);
                std::copy_n(currentEncodedReads.get() + (2*i) * encodedSequencePitchInInts, encodedSequencePitchInInts, input.encodedRead1.begin());
            }

            auto extensionResultsBatch = readExtenderGpu.extendSingleEndReadBatch(inputs);

            //convert results of ReadExtender
            std::vector<ExtendedRead> resultvector(extensionResultsBatch.size());

            for(int i = 0; i < numReadsInBatch; i++){
                auto& extensionOutput = extensionResultsBatch[i];
                ExtendedRead& er = resultvector[i];

                er.readId = extensionOutput.readId1;
                er.extendedSequence = std::move(extensionOutput.extendedRead);

                if(extensionOutput.mateHasBeenFound){
                    er.status = ExtendedReadStatus::FoundMate;
                }else{
                    if(extensionOutput.aborted){
                        if(extensionOutput.abortReason == AbortReason::NoPairedCandidates
                                || extensionOutput.abortReason == AbortReason::NoPairedCandidatesAfterAlignment){

                            er.status = ExtendedReadStatus::CandidateAbort;
                        }else if(extensionOutput.abortReason == AbortReason::MsaNotExtended){
                            er.status = ExtendedReadStatus::MSANoExtension;
                        }
                    }else{
                        er.status = ExtendedReadStatus::LengthAbort;
                    }
                }  
                
                if(extensionOutput.success){
                    numSuccessRead++;
                }                
            }

            auto outputfunc = [&, vec = std::move(resultvector)](){
                for(const auto& er : vec){
                    partialResults.storeElement(&er);
                }
            };

            outputThread.enqueue(
                std::move(outputfunc)
            );

            progressThread.addProgress(numReadsInBatch);            
        }


        cudaStreamDestroy(stream); CUERR;

        //#pragma omp critical
        {
            std::lock_guard<std::mutex> lg(ompCriticalMutex);

            totalNumSuccess0 += numSuccess0;
            totalNumSuccess1 += numSuccess1;
            totalNumSuccess01 += numSuccess01;
            totalNumSuccessRead += numSuccessRead;

            for(const auto& pair : extensionLengthsMap){
                totalExtensionLengthsMap[pair.first] += pair.second;
            }

            for(const auto& pair : mismatchesBetweenMateExtensions){
                totalMismatchesBetweenMateExtensions[pair.first] += pair.second;
            }

            if(0 == ompThreadId){
                readExtenderGpu.printTimers();
            }      
        }

        gpuReadStorage.destroyHandle(readStorageHandle);
        
    } //end omp parallel

    progressThread.finished();

    outputThread.stopThread(BackgroundThread::StopType::FinishAndStop);

    //outputstream.flush();
    partialResults.flush();

    std::cout << "totalNumSuccess0: " << totalNumSuccess0 << std::endl;
    std::cout << "totalNumSuccess1: " << totalNumSuccess1 << std::endl;
    std::cout << "totalNumSuccess01: " << totalNumSuccess01 << std::endl;
    std::cout << "totalNumSuccessRead: " << totalNumSuccessRead << std::endl;

    // std::cout << "Extension lengths:\n";

    // for(const auto& pair : totalExtensionLengthsMap){
    //     std::cout << pair.first << ": " << pair.second << "\n";
    // }

    // std::cout << "mismatches between mate extensions:\n";

    // for(const auto& pair : totalMismatchesBetweenMateExtensions){
    //     std::cout << pair.first << ": " << pair.second << "\n";
    // }



    return partialResults;
    //return resultExtendedReads;
}


MemoryFileFixedSize<ExtendedRead> 
//std::vector<ExtendedRead>
extend_gpu(
    const GoodAlignmentProperties& goodAlignmentProperties,
    const CorrectionOptions& correctionOptions,
    const ExtensionOptions& extensionOptions,
    const RuntimeOptions& runtimeOptions,
    const FileOptions& fileOptions,
    const MemoryOptions& memoryOptions,
    const GpuMinhasher& gpumMinhasher,
    const GpuReadStorage& gpuReadStorage
){
    if(fileOptions.pairType == SequencePairType::SingleEnd){
        return extend_gpu_singleend(
            goodAlignmentProperties,
            correctionOptions,
            extensionOptions,
            runtimeOptions,
            fileOptions,
            memoryOptions,
            gpumMinhasher,
            gpuReadStorage
        );
    }else{
        return extend_gpu_pairedend(
            goodAlignmentProperties,
            correctionOptions,
            extensionOptions,
            runtimeOptions,
            fileOptions,
            memoryOptions,
            gpumMinhasher,
            gpuReadStorage
        );
    }
}





} // namespace gpu

} // namespace care