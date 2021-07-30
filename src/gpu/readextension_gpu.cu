
#include <gpu/gpuminhasher.cuh>
#include <gpu/gpureadstorage.cuh>
#include <gpu/readextender_gpu.hpp>
#include <gpu/cudaerrorcheck.cuh>

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

template<class IdGenerator>
void initializeExtenderInput(
    IdGenerator& readIdGenerator,
    int requestedSizeOfTasks,
    const GpuReadStorage& gpuReadStorage,
    ReadStorageHandle& readStorageHandle,
    read_number* currentIds, // pinned memory
    int* currentReadLengths, //device accessible
    unsigned int* currentEncodedReads, //device accessible
    bool useQualityScores,
    char* currentQualityScores, //device accessible
    std::size_t encodedSequencePitchInInts,
    std::size_t qualityPitchInBytes,
    GpuReadExtender::TaskData& tasks,
    cudaStream_t stream
){
    nvtx::push_range("init", 2);

    const int maxNumPairs = (requestedSizeOfTasks - tasks.size()) / 4;

    int numNewReadsInBatch = 0;

    readIdGenerator.process_next_n(
        maxNumPairs * 2, 
        [&](auto begin, auto end){
            auto readIdsEnd = std::copy(begin, end, currentIds);
            numNewReadsInBatch = std::distance(currentIds, readIdsEnd);
        }
    );

    if(numNewReadsInBatch % 2 == 1){
        throw std::runtime_error("Input files not properly paired. Aborting read extension.");
    }
   
    if(numNewReadsInBatch > 0){
        
        gpuReadStorage.gatherSequences(
            readStorageHandle,
            currentEncodedReads,
            encodedSequencePitchInInts,
            makeAsyncConstBufferWrapper(currentIds),
            currentIds, //device accessible
            numNewReadsInBatch,
            stream
        );

        gpuReadStorage.gatherSequenceLengths(
            readStorageHandle,
            currentReadLengths,
            currentIds,
            numNewReadsInBatch,
            stream
        );

        if(useQualityScores){
            gpuReadStorage.gatherQualities(
                readStorageHandle,
                currentQualityScores,
                qualityPitchInBytes,
                makeAsyncConstBufferWrapper(currentIds),
                currentIds, //device accessible
                numNewReadsInBatch,
                stream
            );
        }
        
        const int numReadPairsInBatch = numNewReadsInBatch / 2; 

        //std::cerr << "thread " << std::this_thread::get_id() << "add tasks\n";
        tasks.addTasks(numReadPairsInBatch, currentIds, currentReadLengths, currentEncodedReads, currentQualityScores, stream);

        //gpuReadExtender->setState(GpuReadExtender::State::UpdateWorkingSet);

        //std::cerr << "Added " << (numReadPairsInBatch * 4) << " new tasks to batch\n";
    }

    nvtx::pop_range();
};


std::pair< std::vector<ExtendedRead>, std::vector<read_number> >
makeAndSplitExtensionOutput(GpuReadExtender::TaskData& finishedTasks, GpuReadExtender::RawExtendResult& rawExtendResult, const GpuReadExtender* gpuReadExtender, bool collectPairsToRepeat, cudaStream_t stream){

    nvtx::push_range("constructRawResults", 4);
    gpuReadExtender->constructRawResults(finishedTasks, rawExtendResult, stream);
    nvtx::pop_range();

    CUDACHECK(cudaStreamSynchronizeWrapper(stream));

    std::vector<extension::ExtendResult> extensionResults = gpuReadExtender->convertRawExtendResults(rawExtendResult);

    const int numresults = extensionResults.size();

    std::vector<read_number> pairsWhichShouldBeRepeatedTemp;

    std::vector<ExtendedRead> extendedReads;
    extendedReads.reserve(numresults);

    int repeated = 0;

    nvtx::push_range("convert extension results", 7);

    for(int i = 0; i < numresults; i++){
        auto& extensionOutput = extensionResults[i];
        const int extendedReadLength = extensionOutput.extendedRead.size();
        //if(extendedReadLength == extensionOutput.originalLength){
        //if(!extensionOutput.mateHasBeenFound){
        if(extendedReadLength > extensionOutput.originalLength && !extensionOutput.mateHasBeenFound && collectPairsToRepeat){
            //do not insert directly into pairsWhichShouldBeRepeated. it causes an infinite loop
            pairsWhichShouldBeRepeatedTemp.push_back(extensionOutput.readId1);
            pairsWhichShouldBeRepeatedTemp.push_back(extensionOutput.readId2);
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
    
    nvtx::pop_range();

    return std::make_pair(std::move(extendedReads), std::move(pairsWhichShouldBeRepeatedTemp));
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

    constexpr bool isPairedEnd = true;

    auto submitReadyResults = [&](std::vector<ExtendedRead> extendedReads){
        outputThread.enqueue(
            [&, vec = std::move(extendedReads)](){
                for(const auto& er : vec){
                    partialResults.storeElement(&er);
                }
            }
        );
    };
           

    assert(runtimeOptions.deviceIds.size() > 0);

    //will need at least one thread per gpu
    const int numDeviceIds = std::min(runtimeOptions.threads, int(runtimeOptions.deviceIds.size()));

    struct GpuData{
        int deviceId;
        std::unique_ptr<cub::CachingDeviceAllocator> extenderAllocator;
        std::unique_ptr<cub::CachingDeviceAllocator> dataAllocator;
        std::unique_ptr<GpuReadExtender> gpuReadExtender;

        GpuData() = default;

        GpuData(const GpuData&) = delete;
        GpuData& operator=(const GpuData&) = delete;

        GpuData(GpuData&&) = default;
        GpuData& operator=(GpuData&&) = default;

        ~GpuData(){
            cub::SwitchDevice ds(deviceId);
            gpuReadExtender = nullptr;
            dataAllocator = nullptr;
            extenderAllocator = nullptr;
        }
    };

    std::vector<std::array<CudaStream, 4>> gpustreams;

    std::vector<GpuData> gpuDataVector;

    for(int d = 0; d < numDeviceIds; d++){
        const int deviceId = runtimeOptions.deviceIds[d];
        CUDACHECK(cudaSetDevice(deviceId));

        gpustreams.emplace_back();

        GpuData gpudata;
        gpudata.deviceId = deviceId;

        gpudata.extenderAllocator = std::make_unique<cub::CachingDeviceAllocator>(
            2, //bin_growth
            1, //min_bin
            cub_CachingDeviceAllocator_INVALID_BIN, //max_bin
            cub_CachingDeviceAllocator_INVALID_SIZE, //max_cached_bytes
            false, //skip_cleanup 
            false //debug
        );

        gpudata.dataAllocator = std::make_unique<cub::CachingDeviceAllocator>(
            2, //bin_growth
            1, //min_bin
            cub_CachingDeviceAllocator_INVALID_BIN, //max_bin
            cub_CachingDeviceAllocator_INVALID_SIZE, //max_cached_bytes
            false, //skip_cleanup 
            false //debug
        );

        gpudata.gpuReadExtender = std::make_unique<GpuReadExtender>(
            encodedSequencePitchInInts,
            decodedSequencePitchInBytes,
            qualityPitchInBytes,
            msaColumnPitchInElements,
            isPairedEnd,
            gpuReadStorage, 
            correctionOptions,
            goodAlignmentProperties,
            qualityConversion,
            insertSize,
            insertSizeStddev,
            maxextensionPerStep,
            *gpudata.extenderAllocator
        );

        gpuDataVector.push_back(std::move(gpudata));
    }

    auto extenderThreadFunc = [&](int gpuIndex, int threadId, auto* idGenerator, bool isLastIteration, GpuReadExtender::IterationConfig iterationConfig){
        //std::cerr << "extenderThreadFunc( " << gpuIndex << ", " << threadId << ")\n";
        auto& gpudata = gpuDataVector[gpuIndex];
        //auto stream = gpustreams[gpuIndex].back().getStream();
        cudaStream_t stream = cudaStreamPerThread;

        CUDACHECK(cudaSetDevice(gpudata.deviceId));

        std::int64_t numSuccess0 = 0;
        std::int64_t numSuccess1 = 0;
        std::int64_t numSuccess01 = 0;
        std::int64_t numSuccessRead = 0;

        std::map<int, int> extensionLengthsMap;
        std::map<int, int> mismatchesBetweenMateExtensions;

        ReadStorageHandle readStorageHandle = gpuReadStorage.makeHandle();


        helpers::SimpleAllocationPinnedHost<read_number> currentIds(2 * batchsizePairs);
        helpers::SimpleAllocationDevice<unsigned int> currentEncodedReads(2 * encodedSequencePitchInInts * batchsizePairs);
        helpers::SimpleAllocationDevice<int> currentReadLengths(2 * batchsizePairs);
        helpers::SimpleAllocationDevice<char> currentQualityScores(2 * qualityPitchInBytes * batchsizePairs);

        if(!correctionOptions.useQualityScores){
            helpers::call_fill_kernel_async(currentQualityScores.data(), currentQualityScores.size(), 'I', stream);
        }


        GpuReadExtender::Hasher anchorHasher(minhasher);

        GpuReadExtender::TaskData tasks(*gpudata.dataAllocator, 0, encodedSequencePitchInInts, decodedSequencePitchInBytes, qualityPitchInBytes, stream);
        GpuReadExtender::TaskData finishedTasks(*gpudata.dataAllocator, 0, encodedSequencePitchInInts, decodedSequencePitchInBytes, qualityPitchInBytes, stream);

        GpuReadExtender::AnchorData anchorData(*gpudata.dataAllocator);
        GpuReadExtender::AnchorHashResult anchorHashResult(*gpudata.dataAllocator);

        GpuReadExtender::RawExtendResult rawExtendResult{};

        std::vector<read_number> pairsWhichShouldBeRepeated;

        auto output = [&](){

            nvtx::push_range("output", 5);

            auto [extendedReads, pairsTmp] = makeAndSplitExtensionOutput(finishedTasks, rawExtendResult, gpudata.gpuReadExtender.get(), !isLastIteration, stream);

            pairsWhichShouldBeRepeated.insert(pairsWhichShouldBeRepeated.end(), pairsTmp.begin(), pairsTmp.end());

            const std::size_t numExtended = extendedReads.size();

            submitReadyResults(std::move(extendedReads));

            progressThread.addProgress(numExtended);

            nvtx::pop_range();
        };

        while(!(idGenerator->empty() && tasks.size() == 0)){
            if(int(tasks.size()) < (batchsizePairs * 4) / 2){
                initializeExtenderInput(
                    *idGenerator,
                    batchsizePairs * 4,
                    gpuReadStorage,
                    readStorageHandle,
                    currentIds.data(), 
                    currentReadLengths.data(), 
                    currentEncodedReads.data(),
                    correctionOptions.useQualityScores,
                    currentQualityScores.data(), 
                    encodedSequencePitchInInts,
                    qualityPitchInBytes,
                    tasks,
                    stream
                );
            }

            tasks.aggregateAnchorData(anchorData, stream);
            
            nvtx::push_range("getCandidateReadIds", 4);
            anchorHasher.getCandidateReadIds(anchorData, anchorHashResult, stream);
            nvtx::pop_range();

            gpudata.gpuReadExtender->processOneIteration(
                tasks, 
                anchorData, 
                anchorHashResult, 
                finishedTasks, 
                iterationConfig,
                stream
            );

            CUDACHECK(cudaStreamSynchronizeWrapper(stream));
            
            if(finishedTasks.size() > std::size_t((batchsizePairs * 4) / 2)){
                output();
            }

            //std::cerr << "Remaining: tasks " << tasks.size() << ", finishedtasks " << gpuReadExtender->finishedTasks->size() << "\n";
        }

        output();
        assert(finishedTasks.size() == 0);

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

        return pairsWhichShouldBeRepeated;
    };

    bool isLastIteration = false;
    GpuReadExtender::IterationConfig iterationConfig;
    iterationConfig.maxextensionPerStep = 20;
    iterationConfig.minCoverageForExtension = 3;

    std::vector<read_number> pairsWhichShouldBeRepeated;
    std::vector<read_number> pairsWhichShouldBeRepeatedTmp;


    {
        std::vector<std::future<std::vector<read_number>>> futures;

        const std::size_t numReadsToProcess = 500000;
        //const std::size_t numReadsToProcess = gpuReadStorage.getNumberOfReads();

        auto idGenerator = cpu::makeIteratorRangeTraversal(
            thrust::make_counting_iterator<read_number>(0),
            thrust::make_counting_iterator<read_number>(0) + numReadsToProcess
        );

        const int maxNumThreads = runtimeOptions.threads;

        std::cerr << "use " << maxNumThreads << " threads\n";

        for(int t = 0; t < maxNumThreads; t++){
            futures.emplace_back(
                std::async(
                    std::launch::async,
                    extenderThreadFunc,
                    t % numDeviceIds,
                    t,
                    &idGenerator,
                    isLastIteration,
                    iterationConfig
                )
            );
        }

        for(auto& f : futures){
            auto vec = f.get();
            pairsWhichShouldBeRepeatedTmp.insert(pairsWhichShouldBeRepeatedTmp.end(), vec.begin(), vec.end());
        }

        std::swap(pairsWhichShouldBeRepeated, pairsWhichShouldBeRepeatedTmp);
        pairsWhichShouldBeRepeatedTmp.clear();
        std::sort(pairsWhichShouldBeRepeated.begin(), pairsWhichShouldBeRepeated.end());

        iterationConfig.maxextensionPerStep -= 4;
    }

    while(pairsWhichShouldBeRepeated.size() > 0 && (iterationConfig.maxextensionPerStep > 0)){
        const int numPairsToRepeat = pairsWhichShouldBeRepeated.size() / 2;
        std::cerr << "Will repeat extension of " << numPairsToRepeat << " read pairs with fixedStepsize = " << iterationConfig.maxextensionPerStep << "\n";

        isLastIteration = (iterationConfig.maxextensionPerStep <= 4);

        auto idGenerator = cpu::makeIteratorRangeTraversal(
            pairsWhichShouldBeRepeated.data(), 
            pairsWhichShouldBeRepeated.data() + pairsWhichShouldBeRepeated.size()
        );

        const int threadsForPairs = SDIV(numPairsToRepeat, batchsizePairs);
        const int maxNumThreads = std::min(threadsForPairs, runtimeOptions.threads);
        std::cerr << "use " << maxNumThreads << " threads\n";

        std::vector<std::future<std::vector<read_number>>> futures;

        for(int t = 0; t < maxNumThreads; t++){
            futures.emplace_back(
                std::async(
                    std::launch::async,
                    extenderThreadFunc,
                    t % numDeviceIds,
                    t,
                    &idGenerator,
                    isLastIteration,
                    iterationConfig
                )
            );
        }

        for(auto& f : futures){
            auto vec = f.get();
            pairsWhichShouldBeRepeatedTmp.insert(pairsWhichShouldBeRepeatedTmp.end(), vec.begin(), vec.end());
        }

        std::swap(pairsWhichShouldBeRepeated, pairsWhichShouldBeRepeatedTmp);
        pairsWhichShouldBeRepeatedTmp.clear();
        std::sort(pairsWhichShouldBeRepeated.begin(), pairsWhichShouldBeRepeated.end());

        iterationConfig.maxextensionPerStep -= 4;
    }
   

    progressThread.finished();

    outputThread.stopThread(BackgroundThread::StopType::FinishAndStop);



    //outputstream.flush();
    partialResults.flush();

    // std::cout << "totalNumSuccess0: " << totalNumSuccess0 << std::endl;
    // std::cout << "totalNumSuccess1: " << totalNumSuccess1 << std::endl;
    // std::cout << "totalNumSuccess01: " << totalNumSuccess01 << std::endl;
    // std::cout << "totalNumSuccessRead: " << totalNumSuccessRead << std::endl;

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
    
            cudaStreamSynchronizeWrapper(stream); CUERR;

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