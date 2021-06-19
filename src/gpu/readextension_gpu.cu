
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
#include <readextender_common.hpp>

#include <omp.h>
#include <cub/cub.cuh>


namespace care{
namespace gpu{


#if 0
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
        [](auto&& i){return ReadExtenderBase::makePairedEndTask(std::move(i), extension::ExtensionDirection::LR);});

    std::transform(inputs.begin(), inputs.end(), itertmp, 
        [](auto&& i){return ReadExtenderBase::makePairedEndTask(std::move(i), extension::ExtensionDirection::RL);});

    batchData.init(
        std::move(tasks), 
        encodedSequencePitchInInts, 
        decodedSequencePitchInBytes, 
        msaColumnPitchInElements
    );
}
#endif

void initializePairedEndExtensionBatchData2(
    BatchData& batchData,
    const std::vector<extension::ExtendInput>& inputs,
    std::size_t encodedSequencePitchInInts, 
    std::size_t decodedSequencePitchInBytes, 
    std::size_t msaColumnPitchInElements,
    std::size_t qualityPitchInBytes
){
    const int batchsizePairs = inputs.size();
    batchData.numReadPairs = batchsizePairs;
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
        taskl.direction = extension::ExtensionDirection::LR;      
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

        taskl.currentQualityScores.insert(taskl.currentQualityScores.begin(), input.qualityScores1.begin(), input.qualityScores1.end());
        taskl.resultQualityScores = taskl.currentQualityScores;
        taskl.totalAnchorQualityScores.emplace_back(taskl.currentQualityScores);

        auto& taskr = batchData.tasks[2*i + 1];
        taskr.reset();

        taskr.pairedEnd = true;
        taskr.direction = extension::ExtensionDirection::RL;
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

        taskr.currentQualityScores.insert(taskr.currentQualityScores.begin(), input.qualityScores2.begin(), input.qualityScores2.end());
        taskr.resultQualityScores = taskr.currentQualityScores;
        taskr.totalAnchorQualityScores.emplace_back(taskr.currentQualityScores);
    }

    batchData.encodedSequencePitchInInts = encodedSequencePitchInInts;
    batchData.decodedSequencePitchInBytes = decodedSequencePitchInBytes;
    batchData.msaColumnPitchInElements = msaColumnPitchInElements;
    batchData.qualityPitchInBytes = qualityPitchInBytes;

    batchData.indicesOfActiveTasks.resize(batchData.tasks.size());
    std::iota(batchData.indicesOfActiveTasks.begin(), batchData.indicesOfActiveTasks.end(), 0);

    for(int i = 0; i < batchsizePairs * 2; i++){
        batchData.tasks[i].id = i;
    }

    batchData.pairedEnd = true;
}


void initializePairedEndExtensionBatchData4(
    BatchData& batchData,
    const std::vector<extension::ExtendInput>& inputs,
    std::size_t encodedSequencePitchInInts, 
    std::size_t decodedSequencePitchInBytes, 
    std::size_t msaColumnPitchInElements,
    std::size_t qualityPitchInBytes
){


#if 1
    const int batchsizePairs = inputs.size();

    if(batchsizePairs == 0) return;

    //batchData.pairedEnd = true;
    batchData.encodedSequencePitchInInts = encodedSequencePitchInInts;
    batchData.decodedSequencePitchInBytes = decodedSequencePitchInBytes;
    batchData.msaColumnPitchInElements = msaColumnPitchInElements;
    batchData.qualityPitchInBytes = qualityPitchInBytes;

    std::vector<extension::Task> tasks(batchsizePairs * 4);
    auto endIter = makePairedEndTasksFromInput4(inputs.begin(), inputs.end(), tasks.begin());
    assert(endIter == tasks.end());

    batchData.addTasks(std::make_move_iterator(tasks.begin()), std::make_move_iterator(tasks.end()));

#else
    const int batchsizePairs = inputs.size();
    batchData.numReadPairs = batchsizePairs ;
    if(batchsizePairs == 0){
        batchData.tasks.clear();
        return;
    }

    batchData.tasks.resize(batchsizePairs * 4);

    auto endIter = makePairedEndTasksFromInput4(inputs.begin(), inputs.end(), batchData.tasks.begin());
    assert(endIter == batchData.tasks.end());

    for(int i = 0; i < batchsizePairs * 4; i++){
        batchData.tasks[i].id = i;
    }


    batchData.encodedSequencePitchInInts = encodedSequencePitchInInts;
    batchData.decodedSequencePitchInBytes = decodedSequencePitchInBytes;
    batchData.msaColumnPitchInElements = msaColumnPitchInElements;
    batchData.qualityPitchInBytes = qualityPitchInBytes;

    #if 1
    batchData.indicesOfActiveTasks.resize(batchData.tasks.size());
    std::iota(batchData.indicesOfActiveTasks.begin(), batchData.indicesOfActiveTasks.end(), 0);
    #else
    assert(batchData.indicesOfActiveTasks.size() % 4 == 0);

    //first, only the LR direction is active. If mate is not found on LR direction, RL direction will be enabled
    batchData.indicesOfActiveTasks.resize(batchsizePairs * 2);

    for(int i = 0, k = 0; i < batchsizePairs; i++){
        batchData.indicesOfActiveTasks[k++] = 4 * i + 0;
        batchData.indicesOfActiveTasks[k++] = 4 * i + 1;
    }
    #endif



    batchData.pairedEnd = true;

    const int numAnchors = batchsizePairs * 4;

    //initialize buffers
    batchData.d_numUsedReadIdsPerAnchor.resize(numAnchors);
    batchData.d_numUsedReadIdsPerAnchorPrefixSum.resize(numAnchors);
    batchData.h_numUsedReadIds.resize(1); // <---
    batchData.d_numFullyUsedReadIdsPerAnchor.resize(numAnchors);
    batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum.resize(numAnchors);
    batchData.h_numFullyUsedReadIds.resize(1); // <---

    batchData.numTasks = numAnchors;

    batchData.h_numAnchors.resize(1);// <---
    batchData.d_numAnchors.resize(1);// <---
    batchData.h_numCandidates.resize(1);// <---
    batchData.d_numCandidates.resize(1);// <---
    batchData.d_numCandidates2.resize(1);// <---
    batchData.h_numAnchorsWithRemovedMates.resize(1);// <---

    batchData.h_numUsedReadIds.resize(1);// <---

    batchData.h_anchorReadIds.resize(numAnchors);
    batchData.d_anchorReadIds.resize(numAnchors);
    batchData.h_mateReadIds.resize(numAnchors);
    batchData.d_mateReadIds.resize(numAnchors);
    
    batchData.d_subjectSequencesData.resize(batchData.encodedSequencePitchInInts * numAnchors);
    batchData.h_subjectSequencesDataDecoded.resize(batchData.decodedSequencePitchInBytes * numAnchors);   
    batchData.d_subjectSequencesDataDecoded.resize(batchData.decodedSequencePitchInBytes * numAnchors);

    batchData.h_anchorSequencesLength.resize(numAnchors);
    batchData.d_anchorSequencesLength.resize(numAnchors);

    batchData.h_anchorQualityScores.resize(batchData.qualityPitchInBytes * numAnchors);
    batchData.d_anchorQualityScores.resize(batchData.qualityPitchInBytes * numAnchors);

    batchData.d_anchormatedata.resize(numAnchors * batchData.encodedSequencePitchInInts);

    batchData.h_inputanchormatedata.resize(numAnchors * batchData.encodedSequencePitchInInts);
    batchData.d_inputanchormatedata.resize(numAnchors * batchData.encodedSequencePitchInInts);

    // batchData.h_numCandidatesPerAnchor.resize(numAnchors);
    // batchData.d_numCandidatesPerAnchor.resize(numAnchors);
    // batchData.d_numCandidatesPerAnchor2.resize(numAnchors);
    // batchData.h_numCandidatesPerAnchorPrefixSum.resize(numAnchors+1);
    // batchData.d_numCandidatesPerAnchorPrefixSum.resize(numAnchors+1);
    // batchData.d_numCandidatesPerAnchorPrefixSum2.resize(numAnchors+1);

    // batchData.d_anchorIndicesWithRemovedMates.resize(numAnchors);

    *batchData.h_numUsedReadIds = 0;
    *batchData.h_numFullyUsedReadIds = 0;

    cudaMemsetAsync(batchData.d_numUsedReadIdsPerAnchor.data(), 0, sizeof(int) * numAnchors, batchData.streams[0]);
    cudaMemsetAsync(batchData.d_numUsedReadIdsPerAnchorPrefixSum.data(), 0, sizeof(int) * numAnchors, batchData.streams[0]);
    cudaMemsetAsync(batchData.d_numFullyUsedReadIdsPerAnchor.data(), 0, sizeof(int) * numAnchors, batchData.streams[0]);
    cudaMemsetAsync(batchData.d_numFullyUsedReadIdsPerAnchorPrefixSum.data(), 0, sizeof(int) * numAnchors, batchData.streams[0]);

    for(int t = 0; t < numAnchors; t++){
        auto& task = batchData.tasks[t];
        task.dataIsAvailable = false;

        batchData.h_anchorReadIds[t] = task.myReadId;
        batchData.h_mateReadIds[t] = task.mateReadId;

        std::copy(
            task.encodedMate.begin(),
            task.encodedMate.end(),
            batchData.h_inputanchormatedata.begin() + t * batchData.encodedSequencePitchInInts
        );

        batchData.h_anchorSequencesLength[t] = task.currentAnchorLength;

        std::copy(
            task.totalDecodedAnchors.back().begin(),
            task.totalDecodedAnchors.back().end(),
            batchData.h_subjectSequencesDataDecoded.begin() + t * batchData.decodedSequencePitchInBytes
        );

        assert(batchData.h_anchorQualityScores.size() >= (t+1) * batchData.qualityPitchInBytes);

        std::copy(
            task.currentQualityScores.begin(),
            task.currentQualityScores.end(),
            batchData.h_anchorQualityScores.begin() + t * batchData.qualityPitchInBytes
        );
    }

    cudaMemcpyAsync(
        batchData.d_inputanchormatedata.data(),
        batchData.h_inputanchormatedata.data(),
        sizeof(unsigned int) * batchData.encodedSequencePitchInInts * batchData.numTasks,
        H2D,
        batchData.streams[0]
    ); CUERR;

    cudaMemcpyAsync(
        batchData.d_anchorSequencesLength.data(),
        batchData.h_anchorSequencesLength.data(),
        sizeof(int) * batchData.numTasks,
        H2D,
        batchData.streams[0]
    ); CUERR;

    cudaMemcpyAsync(
        batchData.d_anchorReadIds.data(),
        batchData.h_anchorReadIds.data(),
        sizeof(read_number) * batchData.numTasks,
        H2D,
        batchData.streams[0]
    ); CUERR;

    cudaMemcpyAsync(
        batchData.d_mateReadIds.data(),
        batchData.h_mateReadIds.data(),
        sizeof(read_number) * batchData.numTasks,
        H2D,
        batchData.streams[0]
    ); CUERR;

    cudaMemcpyAsync(
        batchData.d_subjectSequencesDataDecoded.data(),
        batchData.h_subjectSequencesDataDecoded.data(),
        batchData.numTasks * batchData.decodedSequencePitchInBytes,
        H2D,
        batchData.streams[0]
    ); CUERR;

    cudaMemcpyAsync(
        batchData.d_anchorQualityScores.data(),
        batchData.h_anchorQualityScores.data(),
        batchData.numTasks * batchData.qualityPitchInBytes,
        H2D,
        batchData.streams[0]
    ); CUERR;

    readextendergpukernels::encodeSequencesTo2BitKernel<8>
    <<<SDIV(batchData.numTasks, (128 / 8)), 128, 0, batchData.streams[0]>>>(
        batchData.d_subjectSequencesData.data(),
        batchData.d_subjectSequencesDataDecoded.data(),
        batchData.d_anchorSequencesLength.data(),
        batchData.decodedSequencePitchInBytes,
        batchData.encodedSequencePitchInInts,
        numAnchors
    ); CUERR;

    cudaStreamSynchronize(batchData.streams[0]); CUERR;
#endif    
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
    constexpr unsigned int cub_CachingDeviceAllocator_INVALID_BIN = (unsigned int) -1;
    constexpr size_t cub_CachingDeviceAllocator_INVALID_SIZE = (size_t) -1;

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
    cpu::RangeGenerator<read_number> readIdGenerator(500000);
    //readIdGenerator.skip(2);

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

    cpu::QualityScoreConversion qualityConversion{};

    
    const int insertSize = extensionOptions.insertSize;
    const int insertSizeStddev = extensionOptions.insertSizeStddev;
    const int maximumSequenceLength = gpuReadStorage.getSequenceLengthUpperBound();
    const std::size_t encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength);
    const std::size_t decodedSequencePitchInBytes = SDIV(maximumSequenceLength, 128) * 128;
    const std::size_t qualityPitchInBytes = SDIV(maximumSequenceLength, 128) * 128;

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

    #if 0

    #if 0

    int numparallelbatches = runtimeOptions.threads;

    int numInitializerThreads = 0;
    int numCpuWorkerThreads = std::max(1, runtimeOptions.threads - 2);
    int numGpuWorkerThreads = 2;

    std::vector<BatchData> batches(numparallelbatches);
    MultiProducerMultiConsumerQueue<BatchData*> freeBatchesQueue;
    MultiProducerMultiConsumerQueue<BatchData*> cpuWorkBatchesQueue;
    MultiProducerMultiConsumerQueue<BatchData*> gpuWorkBatchesQueue;

    for(int i = 0; i < numparallelbatches; i++){
        batches[i].someId = i;
        batches[i].setState(BatchData::State::None);
    }

    for(auto& batch : batches){
        //freeBatchesQueue.push(&batch);
        cpuWorkBatchesQueue.push(&batch);
    }

    std::mutex processedMutex{};
    std::atomic<int> numProcessedBatchesByCpuWorkerThreads = 0;

    auto initializerThreadFunc = [&](){

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
                msaColumnPitchInElements,
                qualityPitchInBytes
            );

            batchData->setState(BatchData::State::BeforeHash);

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

    auto chooseWorkerQueue = [&](BatchData* batchData){
        auto type = GpuExtensionStepper::typeOfNextStep(*batchData);

        if(type == GpuExtensionStepper::ComputeType::CPU){
            cpuWorkBatchesQueue.push(batchData);
        }else{
            gpuWorkBatchesQueue.push(batchData);
        }
    };

    auto cpuWorkerThreadFunc = [&](){
        std::int64_t numSuccessRead = 0;

        cub::CachingDeviceAllocator cubAllocator;

        GpuExtensionStepper gpuExtensionStepper(
            gpuReadStorage,
            minhasher, 
            correctionOptions,
            goodAlignmentProperties,
            qualityConversion,
            insertSize,
            insertSizeStddev,
            maxextensionPerStep,
            cubAllocator
        );

        ReadStorageHandle readStorageHandle = gpuReadStorage.makeHandle();

        helpers::SimpleAllocationPinnedHost<read_number> currentIds(2 * batchsizePairs);
        helpers::SimpleAllocationPinnedHost<unsigned int> currentEncodedReads(2 * encodedSequencePitchInInts * batchsizePairs);
        helpers::SimpleAllocationPinnedHost<int> currentReadLengths(2 * batchsizePairs);
        helpers::SimpleAllocationPinnedHost<char> currentQualityScores(2 * qualityPitchInBytes * batchsizePairs);

        if(!correctionOptions.useQualityScores){
            std::fill(currentQualityScores.begin(), currentQualityScores.end(), 'I');
        }

        CudaStream stream;

        BatchData* batchData = cpuWorkBatchesQueue.pop();

        auto init = [&](){
            nvtx::push_range("init", 2);

            auto readIdsEnd = readIdGenerator.next_n_into_buffer(
                batchsizePairs * 2, 
                currentIds.get()
            );

            const int numReadsInBatch = std::distance(currentIds.get(), readIdsEnd);

            if(numReadsInBatch % 2 == 1){
                throw std::runtime_error("Input files not properly paired. Aborting read extension.");
            }
            
            if(numReadsInBatch > 0){
                
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

                assert(currentQualityScores.size() >= numReadsInBatch * qualityPitchInBytes);

                if(correctionOptions.useQualityScores){
                    gpuReadStorage.gatherQualities(
                        readStorageHandle,
                        currentQualityScores.get(),
                        qualityPitchInBytes,
                        currentIds.get(),
                        currentIds.get(), //device accessible
                        numReadsInBatch,
                        stream
                    );
                }

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
                    
                    input.qualityScores1.resize(input.readLength1);
                    input.qualityScores2.resize(input.readLength2);
                    std::copy_n(currentQualityScores.get() + (2*i) * qualityPitchInBytes, input.readLength1, input.qualityScores1.begin());
                    std::copy_n(currentQualityScores.get() + (2*i + 1) * qualityPitchInBytes, input.readLength2, input.qualityScores2.begin());
                }
            
                #if 1
                
                initializePairedEndExtensionBatchData4(
                    *batchData,
                    inputs,
                    encodedSequencePitchInInts, 
                    decodedSequencePitchInBytes, 
                    msaColumnPitchInElements,
                    qualityPitchInBytes
                );

                #else

                initializePairedEndExtensionBatchData2(
                    *batchData,
                    inputs,
                    encodedSequencePitchInInts, 
                    decodedSequencePitchInBytes, 
                    msaColumnPitchInElements,
                    qualityPitchInBytes
                );
                #endif

                batchData->setState(BatchData::State::BeforeHash);
            }else{
                batchData->setState(BatchData::State::None); //this should only happen if all reads have been processed
            }

            nvtx::pop_range();
        };

        auto output = [&](){
            nvtx::push_range("output", 5);
            std::vector<ExtendResult> extensionResults = gpuExtensionStepper.constructResults(*batchData);

            const int numresults = extensionResults.size();

            std::vector<ExtendedRead> extendedReads(numresults);
            
            for(int i = 0; i < numresults; i++){
                auto& extensionOutput = extensionResults[i];
                ExtendedRead& er = extendedReads[i];

                er.readId = extensionOutput.readId1;
                er.mergedFromReadsWithoutMate = extensionOutput.mergedFromReadsWithoutMate;
                er.extendedSequence = std::move(extensionOutput.extendedRead);
                //er.qualityScores = std::move(extensionOutput.qualityScores);
                er.read1begin = extensionOutput.read1begin;
                er.read1end = extensionOutput.read1begin + extensionOutput.originalLength;
                er.read2begin = extensionOutput.read2begin;
                if(er.read2begin != -1){
                    er.read2end = extensionOutput.read2begin + extensionOutput.originalMateLength;
                }else{
                    er.read2end = -1;
                }

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
            }

            auto outputfunc = [&, vec = std::move(extendedReads)](){
                for(const auto& er : vec){
                    partialResults.storeElement(&er);
                }
            };

            outputThread.enqueue(
                std::move(outputfunc)
            );

            batchData->setState(BatchData::State::None);

            nvtx::pop_range();

            progressThread.addProgress(batchData->numReadPairs);
        };

        while(batchData != nullptr){

            if(batchData->state == BatchData::State::None){
                init();   
                if(batchData->state == BatchData::State::BeforeHash){
                    chooseWorkerQueue(batchData);
                }
            }else if(batchData->state == BatchData::State::Finished){
                output();

                chooseWorkerQueue(batchData);

                numProcessedBatchesByCpuWorkerThreads++;
                
                if(numProcessedBatchesByCpuWorkerThreads == numBatchesToProcess){
                    for(int i = 0; i < runtimeOptions.threads; i++){
                        freeBatchesQueue.push(nullptr);
                        cpuWorkBatchesQueue.push(nullptr);
                        gpuWorkBatchesQueue.push(nullptr);
                    }
                }
            }else{
                while(GpuExtensionStepper::typeOfNextStep(*batchData) == GpuExtensionStepper::ComputeType::CPU && batchData->state != BatchData::State::Finished){

                    gpuExtensionStepper.performNextStep(*batchData);
    
                }
                //gpuExtensionStepper.performNextStep(*batchData);

                chooseWorkerQueue(batchData);
            }            

            batchData = cpuWorkBatchesQueue.pop();
        }

        std::cerr << "cpuWorkerThreadFunc finished\n";
    };

    auto gpuWorkerThreadFunc = [&](){
        cub::CachingDeviceAllocator myCubAllocator(
            8, //bin_growth
            1, //min_bin
            cub_CachingDeviceAllocator_INVALID_BIN, //max_bin
            cub_CachingDeviceAllocator_INVALID_SIZE, //max_cached_bytes
            false, //skip_cleanup 
            false //debug
        );

        GpuExtensionStepper gpuExtensionStepper(
            gpuReadStorage,
            minhasher, 
            correctionOptions,
            goodAlignmentProperties,
            qualityConversion,
            insertSize,
            insertSizeStddev,
            maxextensionPerStep,
            myCubAllocator
        );       

        BatchData* batchData = gpuWorkBatchesQueue.pop();

        while(batchData != nullptr){

            while(GpuExtensionStepper::typeOfNextStep(*batchData) == GpuExtensionStepper::ComputeType::GPU){

                gpuExtensionStepper.performNextStep(*batchData);

            }

            chooseWorkerQueue(batchData);

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

    int numparallelbatches = runtimeOptions.threads + 2;

    const int numWorkerThreads = runtimeOptions.threads;

    std::vector<BatchData> batches(numparallelbatches);
    MultiProducerMultiConsumerQueue<BatchData*> workBatchesQueue;

    for(int i = 0; i < numparallelbatches; i++){
        batches[i].someId = i;
        batches[i].setState(BatchData::State::None);
    }

    for(auto& batch : batches){
        workBatchesQueue.push(&batch);
    }

    std::mutex processedMutex{};
    std::atomic<int> numProcessedBatchesByCpuWorkerThreads = 0;

    auto workerThreadFunc = [&](){
        std::int64_t numSuccessRead = 0;

        cub::CachingDeviceAllocator myCubAllocator(
            8, //bin_growth
            1, //min_bin
            cub_CachingDeviceAllocator_INVALID_BIN, //max_bin
            cub_CachingDeviceAllocator_INVALID_SIZE, //max_cached_bytes
            false, //skip_cleanup 
            false //debug
        );

        GpuExtensionStepper gpuExtensionStepper(
            gpuReadStorage,
            minhasher, 
            correctionOptions,
            goodAlignmentProperties,
            qualityConversion,
            insertSize,
            insertSizeStddev,
            maxextensionPerStep,
            myCubAllocator
        );

        ReadStorageHandle readStorageHandle = gpuReadStorage.makeHandle();

        helpers::SimpleAllocationPinnedHost<read_number> currentIds(2 * batchsizePairs);
        helpers::SimpleAllocationPinnedHost<unsigned int> currentEncodedReads(2 * encodedSequencePitchInInts * batchsizePairs);
        helpers::SimpleAllocationPinnedHost<int> currentReadLengths(2 * batchsizePairs);
        helpers::SimpleAllocationPinnedHost<char> currentQualityScores(2 * qualityPitchInBytes * batchsizePairs);

        if(!correctionOptions.useQualityScores){
            std::fill(currentQualityScores.begin(), currentQualityScores.end(), 'I');
        }

        CudaStream stream;

        BatchData* batchData = workBatchesQueue.pop();

        auto init = [&](){
            nvtx::push_range("init", 2);

            auto readIdsEnd = readIdGenerator.next_n_into_buffer(
                batchsizePairs * 2, 
                currentIds.get()
            );

            const int numReadsInBatch = std::distance(currentIds.get(), readIdsEnd);

            if(numReadsInBatch % 2 == 1){
                throw std::runtime_error("Input files not properly paired. Aborting read extension.");
            }
            
            if(numReadsInBatch > 0){
                
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

                assert(currentQualityScores.size() >= numReadsInBatch * qualityPitchInBytes);

                if(correctionOptions.useQualityScores){
                    gpuReadStorage.gatherQualities(
                        readStorageHandle,
                        currentQualityScores.get(),
                        qualityPitchInBytes,
                        currentIds.get(),
                        currentIds.get(), //device accessible
                        numReadsInBatch,
                        stream
                    );
                }

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
                    
                    input.qualityScores1.resize(input.readLength1);
                    input.qualityScores2.resize(input.readLength2);
                    std::copy_n(currentQualityScores.get() + (2*i) * qualityPitchInBytes, input.readLength1, input.qualityScores1.begin());
                    std::copy_n(currentQualityScores.get() + (2*i + 1) * qualityPitchInBytes, input.readLength2, input.qualityScores2.begin());
                }
            
                #if 1
                
                initializePairedEndExtensionBatchData4(
                    *batchData,
                    inputs,
                    encodedSequencePitchInInts, 
                    decodedSequencePitchInBytes, 
                    msaColumnPitchInElements,
                    qualityPitchInBytes
                );

                #else

                initializePairedEndExtensionBatchData2(
                    *batchData,
                    inputs,
                    encodedSequencePitchInInts, 
                    decodedSequencePitchInBytes, 
                    msaColumnPitchInElements,
                    qualityPitchInBytes
                );
                #endif

                batchData->setState(BatchData::State::BeforeHash);
            }else{
                batchData->setState(BatchData::State::None); //this should only happen if all reads have been processed
            }

            nvtx::pop_range();
        };

        auto output = [&](){
            nvtx::push_range("output", 5);
            std::vector<ExtendResult> extensionResults = gpuExtensionStepper.constructResults(*batchData);

            const int numresults = extensionResults.size();

            std::vector<ExtendedRead> extendedReads(numresults);
            
            for(int i = 0; i < numresults; i++){
                auto& extensionOutput = extensionResults[i];
                ExtendedRead& er = extendedReads[i];

                er.readId = extensionOutput.readId1;
                er.mergedFromReadsWithoutMate = extensionOutput.mergedFromReadsWithoutMate;
                er.extendedSequence = std::move(extensionOutput.extendedRead);
                //er.qualityScores = std::move(extensionOutput.qualityScores);
                er.read1begin = extensionOutput.read1begin;
                er.read1end = extensionOutput.read1begin + extensionOutput.originalLength;
                er.read2begin = extensionOutput.read2begin;
                if(er.read2begin != -1){
                    er.read2end = extensionOutput.read2begin + extensionOutput.originalMateLength;
                }else{
                    er.read2end = -1;
                }

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

            auto outputfunc = [&, vec = std::move(extendedReads)](){
                for(const auto& er : vec){
                    partialResults.storeElement(&er);
                }
            };

            outputThread.enqueue(
                std::move(outputfunc)
            );

            batchData->setState(BatchData::State::None);

            nvtx::pop_range();

            progressThread.addProgress(batchData->numReadPairs);
        };

        

        while(batchData != nullptr){

            if(batchData->state == BatchData::State::None){
                init();   
                if(batchData->state == BatchData::State::BeforeHash){
                    workBatchesQueue.push(batchData);
                }
            }else if(batchData->state == BatchData::State::Finished){
                output();

                workBatchesQueue.push(batchData);

                numProcessedBatchesByCpuWorkerThreads++;
                
                if(numProcessedBatchesByCpuWorkerThreads == numBatchesToProcess){
                    for(int i = 0; i < runtimeOptions.threads; i++){
                        workBatchesQueue.push(nullptr);
                    }
                }
            }else{
                auto previousType = GpuExtensionStepper::typeOfNextStep(*batchData);
                auto nextType = GpuExtensionStepper::ComputeType::CPU;

                do{
                    gpuExtensionStepper.performNextStep(*batchData);
                    nextType = GpuExtensionStepper::typeOfNextStep(*batchData);
                }while(previousType == nextType && batchData->state != BatchData::State::Finished);

                workBatchesQueue.push(batchData);
            }            

            batchData = workBatchesQueue.pop();
        }

        std::cerr << "cpuWorkerThreadFunc finished\n";
    };


    std::vector<std::future<void>> futures;

    for(int i = 0; i < numWorkerThreads; i++){
        futures.emplace_back(std::async(std::launch::async, workerThreadFunc));
    }


    for(auto& f : futures){
        f.wait();
    }

    #endif

    #else

    std::atomic<int> numProcessedBatches{0};

    std::vector<std::unique_ptr<cub::CachingDeviceAllocator>> cubAllocators; 

    

    for(auto d : runtimeOptions.deviceIds){
        cub::SwitchDevice sd{d};

        cubAllocators.emplace_back(
            std::make_unique<cub::CachingDeviceAllocator>(
                8, //bin_growth
                1, //min_bin
                cub_CachingDeviceAllocator_INVALID_BIN, //max_bin
                cub_CachingDeviceAllocator_INVALID_SIZE, //max_cached_bytes
                false, //skip_cleanup 
                false //debug
            )
        );
    }

    #pragma omp parallel
    {
        const int numDeviceIds = runtimeOptions.deviceIds.size();

        assert(numDeviceIds > 0);

        const int ompThreadId = omp_get_thread_num();
        const int deviceIdIndex = ompThreadId % numDeviceIds;
        const int deviceId = runtimeOptions.deviceIds.at(deviceIdIndex);
        cudaSetDevice(deviceId); CUERR;

        cub::CachingDeviceAllocator myCubAllocator(
            8, //bin_growth
            1, //min_bin
            cub_CachingDeviceAllocator_INVALID_BIN, //max_bin
            cub_CachingDeviceAllocator_INVALID_SIZE, //max_cached_bytes
            false, //skip_cleanup 
            false //debug
        );

        //cub::CachingDeviceAllocator* myCubAllocator = cubAllocators[deviceIdIndex].get();

        std::int64_t numSuccess0 = 0;
        std::int64_t numSuccess1 = 0;
        std::int64_t numSuccess01 = 0;
        std::int64_t numSuccessRead = 0;

        std::map<int, int> extensionLengthsMap;
        std::map<int, int> mismatchesBetweenMateExtensions;

        ReadStorageHandle readStorageHandle = gpuReadStorage.makeHandle();

        // GpuExtensionStepper gpuExtensionStepper(
        //     gpuReadStorage, 
        //     minhasher,
        //     correctionOptions,
        //     goodAlignmentProperties,
        //     qualityConversion,
        //     insertSize,
        //     insertSizeStddev,
        //     maxextensionPerStep,
        //     myCubAllocator
        // );

        helpers::SimpleAllocationPinnedHost<read_number> currentIds(2 * batchsizePairs);
        helpers::SimpleAllocationPinnedHost<unsigned int> currentEncodedReads(2 * encodedSequencePitchInInts * batchsizePairs);
        helpers::SimpleAllocationPinnedHost<int> currentReadLengths(2 * batchsizePairs);
        helpers::SimpleAllocationPinnedHost<char> currentQualityScores(2 * qualityPitchInBytes * batchsizePairs);

        if(!correctionOptions.useQualityScores){
            std::fill(currentQualityScores.begin(), currentQualityScores.end(), 'I');
        }

        CudaStream stream;

        const bool isPairedEnd = true;

        auto batchData = std::make_unique<BatchData>(
            isPairedEnd,
            gpuReadStorage, 
            minhasher,
            correctionOptions,
            goodAlignmentProperties,
            qualityConversion,
            insertSize,
            insertSizeStddev,
            maxextensionPerStep,
            myCubAllocator
        );
        batchData->someId = ompThreadId;

        int minCoverageForExtension = 3;
        int fixedStepsize = 20;

        //gpuExtensionStepper.setMaxExtensionPerStep(fixedStepsize);
        //gpuExtensionStepper.setMinCoverageForExtension(minCoverageForExtension);

        batchData->setMaxExtensionPerStep(fixedStepsize);
        batchData->setMinCoverageForExtension(minCoverageForExtension);

        std::vector<std::pair<read_number, read_number>> pairsWhichShouldBeRepeated;
        std::vector<std::pair<read_number, read_number>> pairsWhichShouldBeRepeatedTemp;
        bool isLastIteration = false;

        std::vector<extension::ExtendInput> inputs;

        auto init = [&](){
            nvtx::push_range("init", 2);

            auto readIdsEnd = readIdGenerator.next_n_into_buffer(
                batchsizePairs * 2, 
                currentIds.get()
            );

            int numReadsInBatch = std::distance(currentIds.get(), readIdsEnd);

            if(numReadsInBatch % 2 == 1){
                throw std::runtime_error("Input files not properly paired. Aborting read extension.");
            }

            if(numReadsInBatch == 0 && pairsWhichShouldBeRepeated.size() > 0){

                const int numPairsToCopy = std::min(batchsizePairs, int(pairsWhichShouldBeRepeated.size()));

                for(int i = 0; i < numPairsToCopy; i++){
                    currentIds[2*i + 0] = pairsWhichShouldBeRepeated[i].first;
                    currentIds[2*i + 1] = pairsWhichShouldBeRepeated[i].second;
                }

                for(int i = 0; i < numPairsToCopy; i++){
                    if(currentIds[2*i + 0] > currentIds[2*i + 1]){
                        std::swap(currentIds[2*i + 0], currentIds[2*i + 1]);
                    }
                    assert(currentIds[2*i + 1] == currentIds[2*i + 0] + 1);
                }

                pairsWhichShouldBeRepeated.erase(pairsWhichShouldBeRepeated.begin(), pairsWhichShouldBeRepeated.begin() + numPairsToCopy);

                numReadsInBatch = 2 * numPairsToCopy;
            }
            
            if(numReadsInBatch > 0){
                
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

                assert(currentQualityScores.size() >= numReadsInBatch * qualityPitchInBytes);

                if(correctionOptions.useQualityScores){
                    gpuReadStorage.gatherQualities(
                        readStorageHandle,
                        currentQualityScores.get(),
                        qualityPitchInBytes,
                        currentIds.get(),
                        currentIds.get(), //device accessible
                        numReadsInBatch,
                        stream
                    );
                }

                cudaStreamSynchronize(stream); CUERR;

                const int numReadPairsInBatch = numReadsInBatch / 2;

                inputs.resize(numReadPairsInBatch); 

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

                    input.qualityScores1.resize(input.readLength1);
                    input.qualityScores2.resize(input.readLength2);
                    std::copy_n(currentQualityScores.get() + (2*i) * qualityPitchInBytes, input.readLength1, input.qualityScores1.begin());
                    std::copy_n(currentQualityScores.get() + (2*i + 1) * qualityPitchInBytes, input.readLength2, input.qualityScores2.begin());
                }
            
                #if 1
                
                initializePairedEndExtensionBatchData4(
                    *batchData,
                    inputs,
                    encodedSequencePitchInInts, 
                    decodedSequencePitchInBytes, 
                    msaColumnPitchInElements,
                    qualityPitchInBytes
                );

                #else

                initializePairedEndExtensionBatchData2(
                    *batchData,
                    inputs,
                    encodedSequencePitchInInts, 
                    decodedSequencePitchInBytes, 
                    msaColumnPitchInElements,
                    qualityPitchInBytes
                );
                #endif

                batchData->setState(BatchData::State::BeforeHash);
            }else{
                batchData->setState(BatchData::State::None); //this should only happen if all reads have been processed
            }

            nvtx::pop_range();
        };


        auto output = [&](){
            nvtx::push_range("output", 5);
            //std::vector<extension::ExtendResult> extensionResults = gpuExtensionStepper.constructResults(*batchData);
            std::vector<extension::ExtendResult> extensionResults = batchData->constructResults();

            const int numresults = extensionResults.size();

            std::vector<ExtendedRead> extendedReads;
            extendedReads.reserve(numresults);

            int repeated = 0;
            
            for(int i = 0; i < numresults; i++){
                auto& extensionOutput = extensionResults[i];
                const int extendedReadLength = extensionOutput.extendedRead.size();
                //if(extendedReadLength == extensionOutput.originalLength){
                //if(!extensionOutput.mateHasBeenFound){
                if(extendedReadLength > extensionOutput.originalLength && !extensionOutput.mateHasBeenFound && !isLastIteration){
                    //do not insert directly into pairsWhichShouldBeRepeated. it causes an infinite loop
                    pairsWhichShouldBeRepeatedTemp.emplace_back(std::make_pair(extensionOutput.readId1, extensionOutput.readId2));
                    repeated++;
                }else{
                    //assert(extensionOutput.extendedRead.size() > extensionOutput.originalLength);

                    ExtendedRead er;

                    er.readId = extensionOutput.readId1;
                    er.mergedFromReadsWithoutMate = extensionOutput.mergedFromReadsWithoutMate;
                    er.extendedSequence = std::move(extensionOutput.extendedRead);
                    //er.qualityScores = std::move(extensionOutput.qualityScores);
                    er.read1begin = extensionOutput.read1begin;
                    er.read1end = extensionOutput.read1begin + extensionOutput.originalLength;
                    er.read2begin = extensionOutput.read2begin;
                    if(er.read2begin != -1){
                        er.read2end = extensionOutput.read2begin + extensionOutput.originalMateLength;
                    }else{
                        er.read2end = -1;
                    }

                    if(extensionOutput.mateHasBeenFound){
                        er.status = ExtendedReadStatus::FoundMate;
                    }else{
                        if(extensionOutput.aborted){
                            if(extensionOutput.abortReason == extension::AbortReason::NoPairedCandidates
                                    || extensionOutput.abortReason == extension::AbortReason::NoPairedCandidatesAfterAlignment){

                                er.status = ExtendedReadStatus::CandidateAbort;
                            }else if(extensionOutput.abortReason == extension::AbortReason::MsaNotExtended){
                                er.status = ExtendedReadStatus::MSANoExtension;
                            }
                        }else{
                            er.status = ExtendedReadStatus::LengthAbort;
                        }
                    }  
                    
                    extendedReads.emplace_back(std::move(er));

                }
                              
            }

            auto outputfunc = [&, vec = std::move(extendedReads)](){
                for(const auto& er : vec){
                    partialResults.storeElement(&er);
                }
            };

            outputThread.enqueue(
                std::move(outputfunc)
            );

            batchData->setState(BatchData::State::None);

            nvtx::pop_range();

            progressThread.addProgress(batchData->numReadPairs - repeated);

            batchData->resetTasks();
        };

        //std::cerr << "thread " << ompThreadId << " begins main loop\n";

        isLastIteration = false;
        while(!(readIdGenerator.empty())){
            init();
            if(batchData->state != BatchData::State::None){
                //gpuExtensionStepper.process(*batchData);
                batchData->process();
                output();
            }
        }

        //batchData->printTransitions = true;
        //std::cerr << "thread " << ompThreadId << " finished main loop\n";
 
        // constexpr int increment = 1;
        // constexpr int limit = 10;

        fixedStepsize -= 4;
        //minCoverageForExtension += increment;
        std::swap(pairsWhichShouldBeRepeatedTemp, pairsWhichShouldBeRepeated);

        while(pairsWhichShouldBeRepeated.size() > 0 && (fixedStepsize > 0)){
            batchData->setMaxExtensionPerStep(fixedStepsize);
            //batchData->setMinCoverageForExtension(minCoverageForExtension);

            //gpuExtensionStepper.setMaxExtensionPerStep(fixedStepsize);
            //std::cerr << "fixedStepsize = " << fixedStepsize << "\n"; 
            //gpuExtensionStepper.setMinCoverageForExtension(minCoverageForExtension);

            std::cerr << "thread " << ompThreadId << " will repeat extension of " << pairsWhichShouldBeRepeated.size() << " read pairs with fixedStepsize = " << fixedStepsize << "\n";
            isLastIteration = (fixedStepsize <= 4);

            while(pairsWhichShouldBeRepeated.size() > 0){
                init();
                if(batchData->state != BatchData::State::None){
                    //gpuExtensionStepper.process(*batchData);
                    batchData->process();
                    output();
                }
            }

            //std::cerr << "thread " << ompThreadId << " finished extra loop with fixedStepsize = " << fixedStepsize << "\n";

            fixedStepsize -= 4;
            std::swap(pairsWhichShouldBeRepeatedTemp, pairsWhichShouldBeRepeated);
        }

        //std::cerr << "thread " << ompThreadId << " finished all extensions\n";

        // while(pairsWhichShouldBeRepeated.size() > 0 && ((minCoverageForExtension < limit))){

        //     //std::cerr << "Will repeat extension of " << pairsWhichShouldBeRepeated.size() << " read pairs with minCoverageForExtension = " << minCoverageForExtension << ", fixedStepsize = " << fixedStepsize << "\n";
        //     isLastIteration = (minCoverageForExtension + increment >= limit);

        //     while(pairsWhichShouldBeRepeated.size() > 0){
        //         init();
        //         if(batchData->state != BatchData::State::None){
        //             gpuExtensionStepper.process(*batchData);
        //             output();
        //         }
        //     }

        //     minCoverageForExtension += increment;
        //     std::swap(pairsWhichShouldBeRepeatedTemp, pairsWhichShouldBeRepeated);
        // }

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
    throw std::runtime_error("extend_gpu_singleend NOT IMPLEMENTED");
#if 0
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
    #endif
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