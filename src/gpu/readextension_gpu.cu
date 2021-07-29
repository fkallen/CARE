
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

#if 0
struct Pipeline1{
    int deviceId = 0;
    std::size_t batchsizePairs;
    std::size_t encodedSequencePitchInInts;
    std::size_t decodedSequencePitchInBytes;
    std::size_t qualityPitchInBytes;
    const GpuReadStorage* gpuReadStorage;
    const GpuMinhasher* gpuMinhasher;
    const RuntimeOptions* runtimeOptions;
    const CorrectionOptions* correctionOptions;
    cub::CachingDeviceAllocator* myCubAllocator;

    std::array<cudaStream_t, 4> streamsraw{};

    Pipeline1(
        int deviceId_,
        std::size_t encodedSequencePitchInInts_,
        std::size_t decodedSequencePitchInBytes_,
        std::size_t qualityPitchInBytes_,
        const GpuReadStorage& gpuReadStorage_,
        const GpuMinhasher& gpuMinhasher_,
        const RuntimeOptions& runtimeOptions_,
        const CorrectionOptions& correctionOptions_,
        cub::CachingDeviceAllocator& cubAllocator_,
        std::array<cudaStream_t, 4> streamsraw_
    ) :
        deviceId(deviceId_),
        encodedSequencePitchInInts(encodedSequencePitchInInts_),
        decodedSequencePitchInBytes(decodedSequencePitchInBytes_),
        qualityPitchInBytes(qualityPitchInBytes_),
        gpuReadStorage(&gpuReadStorage_),
        gpuMinhasher(&gpuMinhasher_),
        runtimeOptions(&runtimeOptions_),
        correctionOptions(&correctionOptions_),
        myCubAllocator(&cubAllocator_),
        streamsraw(streamsraw_)
    {

    }

    void run(){
        cudaSetDevice(deviceId); CUERR;

        std::int64_t numSuccess0 = 0;
        std::int64_t numSuccess1 = 0;
        std::int64_t numSuccess01 = 0;
        std::int64_t numSuccessRead = 0;

        std::map<int, int> extensionLengthsMap;
        std::map<int, int> mismatchesBetweenMateExtensions;

        ReadStorageHandle readStorageHandle = gpuReadStorage->makeHandle();


        helpers::SimpleAllocationPinnedHost<read_number> currentIds(2 * batchsizePairs);
        helpers::SimpleAllocationDevice<unsigned int> currentEncodedReads(2 * encodedSequencePitchInInts * batchsizePairs);
        helpers::SimpleAllocationDevice<int> currentReadLengths(2 * batchsizePairs);
        helpers::SimpleAllocationDevice<char> currentQualityScores(2 * qualityPitchInBytes * batchsizePairs);

        CudaStream stream;

        if(!correctionOptions->useQualityScores){
            //std::fill(currentQualityScores.begin(), currentQualityScores.end(), 'I');
            helpers::call_fill_kernel_async(currentQualityScores.data(), currentQualityScores.size(), 'I', stream);
        }


        const bool isPairedEnd = true;

        GpuReadExtender::Hasher anchorHasher(*gpuMinhasher);

        GpuReadExtender::TaskData tasks(*myCubAllocator, 0, encodedSequencePitchInInts, decodedSequencePitchInBytes, qualityPitchInBytes, (cudaStream_t)0);
        GpuReadExtender::TaskData finishedTasks(*myCubAllocator, 0, encodedSequencePitchInInts, decodedSequencePitchInBytes, qualityPitchInBytes, (cudaStream_t)0);

        GpuReadExtender::AnchorData anchorData(*myCubAllocator);
        GpuReadExtender::AnchorHashResult anchorHashResult(*myCubAllocator);

        auto gpuReadExtender = std::make_unique<GpuReadExtender>(
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
            streamsraw,
            myCubAllocator
        );
        gpuReadExtender->someId = ompThreadId;

        int minCoverageForExtension = 3;
        int fixedStepsize = 20;

        //gpuExtensionStepper.setMaxExtensionPerStep(fixedStepsize);
        //gpuExtensionStepper.setMinCoverageForExtension(minCoverageForExtension);

        gpuReadExtender->setMaxExtensionPerStep(fixedStepsize);
        gpuReadExtender->setMinCoverageForExtension(minCoverageForExtension);

        GpuReadExtender::RawExtendResult rawExtendResult{};

        std::vector<std::pair<read_number, read_number>> pairsWhichShouldBeRepeated;
        std::vector<std::pair<read_number, read_number>> pairsWhichShouldBeRepeatedTemp;
        bool isLastIteration = false;

        std::vector<extension::ExtendInput> inputs;

        auto init = [&](){
            nvtx::push_range("init", 2);

            const int maxNumPairs = (batchsizePairs * 4 - tasks.size()) / 4;
            assert(maxNumPairs <= batchsizePairs);

            auto readIdsEnd = readIdGenerator.next_n_into_buffer(
                maxNumPairs * 2, 
                currentIds.get()
            );

            int numNewReadsInBatch = std::distance(currentIds.get(), readIdsEnd);

            if(numNewReadsInBatch % 2 == 1){
                throw std::runtime_error("Input files not properly paired. Aborting read extension.");
            }

            if(numNewReadsInBatch == 0 && pairsWhichShouldBeRepeated.size() > 0){

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

                numNewReadsInBatch = 2 * numPairsToCopy;
            }
            
            if(numNewReadsInBatch > 0){
                
                gpuReadStorage->gatherSequences(
                    readStorageHandle,
                    currentEncodedReads.get(),
                    encodedSequencePitchInInts,
                    makeAsyncConstBufferWrapper(currentIds.get()),
                    currentIds.get(), //device accessible
                    numNewReadsInBatch,
                    stream
                );

                gpuReadStorage->gatherSequenceLengths(
                    readStorageHandle,
                    currentReadLengths.get(),
                    currentIds.get(),
                    numNewReadsInBatch,
                    stream
                );

                assert(currentQualityScores.size() >= numNewReadsInBatch * qualityPitchInBytes);

                if(correctionOptions->useQualityScores){
                    gpuReadStorage->gatherQualities(
                        readStorageHandle,
                        currentQualityScores.get(),
                        qualityPitchInBytes,
                        makeAsyncConstBufferWrapper(currentIds.get()),
                        currentIds.get(), //device accessible
                        numNewReadsInBatch,
                        stream
                    );
                }
                
                const int numReadPairsInBatch = numNewReadsInBatch / 2;

                tasks.addTasks(numReadPairsInBatch, currentIds.data(), currentReadLengths.data(), currentEncodedReads.data(), currentQualityScores.data(), stream);

                //gpuReadExtender->setState(GpuReadExtender::State::UpdateWorkingSet);

                //std::cerr << "Added " << (numReadPairsInBatch * 4) << " new tasks to batch\n";
            }

            nvtx::pop_range();
        };


        auto output = [&](){
            nvtx::push_range("output", 5);

            gpuReadExtender->constructRawResults(finishedTasks, rawExtendResult, stream);
            cudaStreamSynchronizeWrapper(stream); CUERR;

            std::vector<extension::ExtendResult> extensionResults = gpuReadExtender->convertRawExtendResults(rawExtendResult);

            //std::vector<extension::ExtendResult> extensionResults = gpuReadExtender->constructResults4();
            const int numresults = extensionResults.size();

            //std::cerr << "Got " << (numresults) << " extended reads. Remaining unprocessed finished tasks: " << gpuReadExtender->finishedTasks->size() << "\n";


            std::vector<ExtendedRead> extendedReads;
            extendedReads.reserve(numresults);

            int repeated = 0;

            nvtx::push_range("convert extension results", 7);

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
            nvtx::pop_range();

            //gpuReadExtender->setState(GpuReadExtender::State::None);

            nvtx::pop_range();

            progressThread.addProgress(numresults - repeated);
        };

        //std::cerr << "thread " << ompThreadId << " begins main loop\n";

        isLastIteration = false;
        while(!(readIdGenerator.empty() && tasks.size() == 0)){
            if(int(tasks.size()) < (batchsizePairs * 4) / 2){
                init();
            }

            tasks.aggregateAnchorData(anchorData, stream);
            anchorHasher.getCandidateReadIds(anchorData, anchorHashResult, stream);

            gpuReadExtender->processOneIteration(tasks, anchorData, anchorHashResult, finishedTasks, stream);

            cudaStreamSynchronizeWrapper(stream); CUERR;
            
            if(finishedTasks.size() > std::size_t((batchsizePairs * 4) / 2)){
                output();
            }

            //std::cerr << "Remaining: tasks " << tasks.size() << ", finishedtasks " << gpuReadExtender->finishedTasks->size() << "\n";
        }

        output();
        assert(finishedTasks.size() == 0);

        std::cerr << "\nalltimetotalTaskBytes = " << gpuReadExtender->alltimetotalTaskBytes << "\n";

        //gpuReadExtender->printTransitions = true;
        //std::cerr << "thread " << ompThreadId << " finished main loop\n";

        // constexpr int increment = 1;
        // constexpr int limit = 10;

        fixedStepsize -= 4;
        //minCoverageForExtension += increment;
        std::swap(pairsWhichShouldBeRepeatedTemp, pairsWhichShouldBeRepeated);

        std::sort(pairsWhichShouldBeRepeated.begin(), pairsWhichShouldBeRepeated.end(), [](const auto& p1, const auto& p2){
            return p1.first < p2.first;
        });

        while(pairsWhichShouldBeRepeated.size() > 0 && (fixedStepsize > 0)){
            gpuReadExtender->setMaxExtensionPerStep(fixedStepsize);
            //gpuReadExtender->setMinCoverageForExtension(minCoverageForExtension);

            //gpuExtensionStepper.setMaxExtensionPerStep(fixedStepsize);
            //std::cerr << "fixedStepsize = " << fixedStepsize << "\n"; 
            //gpuExtensionStepper.setMinCoverageForExtension(minCoverageForExtension);

            std::cerr << "thread " << ompThreadId << " will repeat extension of " << pairsWhichShouldBeRepeated.size() << " read pairs with fixedStepsize = " << fixedStepsize << "\n";
            isLastIteration = (fixedStepsize <= 4);

            while(!(pairsWhichShouldBeRepeated.size() == 0 && tasks.size() == 0)){
                if(int(tasks.size()) < (batchsizePairs * 4) / 2){
                    init();
                }
    
                tasks.aggregateAnchorData(anchorData, stream);
                anchorHasher.getCandidateReadIds(anchorData, anchorHashResult, stream);

                gpuReadExtender->processOneIteration(tasks, anchorData, anchorHashResult, finishedTasks, stream);

                cudaStreamSynchronizeWrapper(stream); CUERR;
                
                if(finishedTasks.size() > std::size_t((batchsizePairs * 4))){
                    output();
                }
            }
    
            output();
            assert(finishedTasks.size() == 0);

            //std::cerr << "thread " << ompThreadId << " finished extra loop with fixedStepsize = " << fixedStepsize << "\n";

            fixedStepsize -= 4;
            std::swap(pairsWhichShouldBeRepeatedTemp, pairsWhichShouldBeRepeated);

            std::sort(pairsWhichShouldBeRepeated.begin(), pairsWhichShouldBeRepeated.end(), [](const auto& p1, const auto& p2){
                return p1.first < p2.first;
            });
        }

        
        gpuReadStorage.destroyHandle(readStorageHandle);

    }
};
#endif

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

    auto readIdsEnd = readIdGenerator.next_n_into_buffer(
        maxNumPairs * 2, 
        currentIds
    );

    int numNewReadsInBatch = std::distance(currentIds, readIdsEnd);

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

    std::vector<GpuReadExtender> batches(numparallelbatches);
    MultiProducerMultiConsumerQueue<GpuReadExtender*> freeBatchesQueue;
    MultiProducerMultiConsumerQueue<GpuReadExtender*> cpuWorkBatchesQueue;
    MultiProducerMultiConsumerQueue<GpuReadExtender*> gpuWorkBatchesQueue;

    for(int i = 0; i < numparallelbatches; i++){
        batches[i].someId = i;
        batches[i].setState(GpuReadExtender::State::None);
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

            cudaStreamSynchronizeWrapper(stream); CUERR;

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
            GpuReadExtender* gpuReadExtender = freeBatchesQueue.pop();
            assert(gpuReadExtender != nullptr);

            
            initializePairedEndExtensionBatchData2(
                *gpuReadExtender,
                inputs,
                encodedSequencePitchInInts, 
                decodedSequencePitchInBytes, 
                msaColumnPitchInElements,
                qualityPitchInBytes
            );

            gpuReadExtender->setState(GpuReadExtender::State::UpdateWorkingSet);

            nvtx::pop_range();

            //std::cerr << "initializer firstIterationBatchesQueue.push()\n";
            cpuWorkBatchesQueue.push(gpuReadExtender);
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

    auto chooseWorkerQueue = [&](GpuReadExtender* gpuReadExtender){
        auto type = GpuExtensionStepper::typeOfNextStep(*gpuReadExtender);

        if(type == GpuExtensionStepper::ComputeType::CPU){
            cpuWorkBatchesQueue.push(gpuReadExtender);
        }else{
            gpuWorkBatchesQueue.push(gpuReadExtender);
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

        GpuReadExtender* gpuReadExtender = cpuWorkBatchesQueue.pop();

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

                cudaStreamSynchronizeWrapper(stream); CUERR;

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
                    *gpuReadExtender,
                    inputs,
                    encodedSequencePitchInInts, 
                    decodedSequencePitchInBytes, 
                    msaColumnPitchInElements,
                    qualityPitchInBytes
                );

                #else

                initializePairedEndExtensionBatchData2(
                    *gpuReadExtender,
                    inputs,
                    encodedSequencePitchInInts, 
                    decodedSequencePitchInBytes,
                    msaColumnPitchInElements,
                    qualityPitchInBytes
                );
                #endif

                gpuReadExtender->setState(GpuReadExtender::State::UpdateWorkingSet);
            }else{
                gpuReadExtender->setState(GpuReadExtender::State::None); //this should only happen if all reads have been processed
            }

            nvtx::pop_range();
        };

        auto output = [&](){
            nvtx::push_range("output", 5);
            std::vector<ExtendResult> extensionResults = gpuExtensionStepper.constructResults(*gpuReadExtender);

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

            gpuReadExtender->setState(GpuReadExtender::State::None);

            nvtx::pop_range();

            progressThread.addProgress(numresults);
        };

        while(gpuReadExtender != nullptr){

            if(gpuReadExtender->state == GpuReadExtender::State::None){
                init();   
                if(gpuReadExtender->state == GpuReadExtender::State::UpdateWorkingSet){
                    chooseWorkerQueue(gpuReadExtender);
                }
            }else if(gpuReadExtender->state == GpuReadExtender::State::Finished){
                output();

                chooseWorkerQueue(gpuReadExtender);

                numProcessedBatchesByCpuWorkerThreads++;
                
                if(numProcessedBatchesByCpuWorkerThreads == numBatchesToProcess){
                    for(int i = 0; i < runtimeOptions.threads; i++){
                        freeBatchesQueue.push(nullptr);
                        cpuWorkBatchesQueue.push(nullptr);
                        gpuWorkBatchesQueue.push(nullptr);
                    }
                }
            }else{
                while(GpuExtensionStepper::typeOfNextStep(*gpuReadExtender) == GpuExtensionStepper::ComputeType::CPU && gpuReadExtender->state != GpuReadExtender::State::Finished){

                    gpuExtensionStepper.performNextStep(*gpuReadExtender);
    
                }
                //gpuExtensionStepper.performNextStep(*gpuReadExtender);

                chooseWorkerQueue(gpuReadExtender);
            }            

            gpuReadExtender = cpuWorkBatchesQueue.pop();
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

        GpuReadExtender* gpuReadExtender = gpuWorkBatchesQueue.pop();

        while(gpuReadExtender != nullptr){

            while(GpuExtensionStepper::typeOfNextStep(*gpuReadExtender) == GpuExtensionStepper::ComputeType::GPU){

                gpuExtensionStepper.performNextStep(*gpuReadExtender);

            }

            chooseWorkerQueue(gpuReadExtender);

            gpuReadExtender = gpuWorkBatchesQueue.pop();
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

    std::vector<GpuReadExtender> batches(numparallelbatches);
    MultiProducerMultiConsumerQueue<GpuReadExtender*> workBatchesQueue;

    for(int i = 0; i < numparallelbatches; i++){
        batches[i].someId = i;
        batches[i].setState(GpuReadExtender::State::None);
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

        GpuReadExtender* gpuReadExtender = workBatchesQueue.pop();

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

                cudaStreamSynchronizeWrapper(stream); CUERR;

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
                    *gpuReadExtender,
                    inputs,
                    encodedSequencePitchInInts, 
                    decodedSequencePitchInBytes, 
                    msaColumnPitchInElements,
                    qualityPitchInBytes
                );

                #else

                initializePairedEndExtensionBatchData2(
                    *gpuReadExtender,
                    inputs,
                    encodedSequencePitchInInts, 
                    decodedSequencePitchInBytes, 
                    msaColumnPitchInElements,
                    qualityPitchInBytes
                );
                #endif

                gpuReadExtender->setState(GpuReadExtender::State::UpdateWorkingSet);
            }else{
                gpuReadExtender->setState(GpuReadExtender::State::None); //this should only happen if all reads have been processed
            }

            nvtx::pop_range();
        };

        auto output = [&](){
            nvtx::push_range("output", 5);
            std::vector<ExtendResult> extensionResults = gpuExtensionStepper.constructResults(*gpuReadExtender);

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

            gpuReadExtender->setState(GpuReadExtender::State::None);

            nvtx::pop_range();

            progressThread.addProgress(numresults);
        };

        

        while(gpuReadExtender != nullptr){

            if(gpuReadExtender->state == GpuReadExtender::State::None){
                init();   
                if(gpuReadExtender->state == GpuReadExtender::State::UpdateWorkingSet){
                    workBatchesQueue.push(gpuReadExtender);
                }
            }else if(gpuReadExtender->state == GpuReadExtender::State::Finished){
                output();

                workBatchesQueue.push(gpuReadExtender);

                numProcessedBatchesByCpuWorkerThreads++;
                
                if(numProcessedBatchesByCpuWorkerThreads == numBatchesToProcess){
                    for(int i = 0; i < runtimeOptions.threads; i++){
                        workBatchesQueue.push(nullptr);
                    }
                }
            }else{
                auto previousType = GpuExtensionStepper::typeOfNextStep(*gpuReadExtender);
                auto nextType = GpuExtensionStepper::ComputeType::CPU;

                do{
                    gpuExtensionStepper.performNextStep(*gpuReadExtender);
                    nextType = GpuExtensionStepper::typeOfNextStep(*gpuReadExtender);
                }while(previousType == nextType && gpuReadExtender->state != GpuReadExtender::State::Finished);

                workBatchesQueue.push(gpuReadExtender);
            }            

            gpuReadExtender = workBatchesQueue.pop();
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


#if 1

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

    std::cerr << "newmode\n";


    auto extenderThreadFunc = [&](int gpuIndex, int threadId){
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

        GpuReadExtender::TaskData tasks(*gpudata.dataAllocator, 0, encodedSequencePitchInInts, decodedSequencePitchInBytes, qualityPitchInBytes, (cudaStream_t)0);
        GpuReadExtender::TaskData finishedTasks(*gpudata.dataAllocator, 0, encodedSequencePitchInInts, decodedSequencePitchInBytes, qualityPitchInBytes, (cudaStream_t)0);

        GpuReadExtender::AnchorData anchorData(*gpudata.dataAllocator);
        GpuReadExtender::AnchorHashResult anchorHashResult(*gpudata.dataAllocator);

        GpuReadExtender::IterationConfig iterationConfig{};
        iterationConfig.maxextensionPerStep = 20;
        iterationConfig.minCoverageForExtension = 3;

        GpuReadExtender::RawExtendResult rawExtendResult{};

        std::vector<read_number> pairsWhichShouldBeRepeated;
        std::vector<read_number> pairsWhichShouldBeRepeatedTemp;

        cpu::RangeGeneratorWrapper<read_number> generatorWrapper(
            pairsWhichShouldBeRepeated.data(), 
            pairsWhichShouldBeRepeated.data() + pairsWhichShouldBeRepeated.size()
        );

        bool isLastIteration = false;

        auto init1 = [&](){
            initializeExtenderInput(
                readIdGenerator,
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
        };

        auto init2 = [&](){
            initializeExtenderInput(
                generatorWrapper,
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
        };

        // auto init = [&](){
        //     nvtx::push_range("init", 2);

        //     const int maxNumPairs = (batchsizePairs * 4 - tasks.size()) / 4;
        //     assert(maxNumPairs <= batchsizePairs);

        //     auto readIdsEnd = readIdGenerator.next_n_into_buffer(
        //         maxNumPairs * 2, 
        //         currentIds.get()
        //     );

        //     int numNewReadsInBatch = std::distance(currentIds.get(), readIdsEnd);

        //     if(numNewReadsInBatch % 2 == 1){
        //         throw std::runtime_error("Input files not properly paired. Aborting read extension.");
        //     }

        //     if(numNewReadsInBatch == 0 && pairsWhichShouldBeRepeated.size() > 0){

        //         const int numPairsToCopy = std::min(batchsizePairs, int(pairsWhichShouldBeRepeated.size()));

        //         for(int i = 0; i < numPairsToCopy; i++){
        //             currentIds[2*i + 0] = pairsWhichShouldBeRepeated[i].first;
        //             currentIds[2*i + 1] = pairsWhichShouldBeRepeated[i].second;
        //         }

        //         for(int i = 0; i < numPairsToCopy; i++){
        //             if(currentIds[2*i + 0] > currentIds[2*i + 1]){
        //                 std::swap(currentIds[2*i + 0], currentIds[2*i + 1]);
        //             }
        //             assert(currentIds[2*i + 1] == currentIds[2*i + 0] + 1);
        //         }

        //         pairsWhichShouldBeRepeated.erase(pairsWhichShouldBeRepeated.begin(), pairsWhichShouldBeRepeated.begin() + numPairsToCopy);

        //         numNewReadsInBatch = 2 * numPairsToCopy;
        //     }
            
        //     if(numNewReadsInBatch > 0){
                
        //         gpuReadStorage.gatherSequences(
        //             readStorageHandle,
        //             currentEncodedReads.get(),
        //             encodedSequencePitchInInts,
        //             makeAsyncConstBufferWrapper(currentIds.get()),
        //             currentIds.get(), //device accessible
        //             numNewReadsInBatch,
        //             stream
        //         );

        //         gpuReadStorage.gatherSequenceLengths(
        //             readStorageHandle,
        //             currentReadLengths.get(),
        //             currentIds.get(),
        //             numNewReadsInBatch,
        //             stream
        //         );

        //         assert(currentQualityScores.size() >= numNewReadsInBatch * qualityPitchInBytes);

        //         if(correctionOptions.useQualityScores){
        //             gpuReadStorage.gatherQualities(
        //                 readStorageHandle,
        //                 currentQualityScores.get(),
        //                 qualityPitchInBytes,
        //                 makeAsyncConstBufferWrapper(currentIds.get()),
        //                 currentIds.get(), //device accessible
        //                 numNewReadsInBatch,
        //                 stream
        //             );
        //         }
                
        //         const int numReadPairsInBatch = numNewReadsInBatch / 2; 

        //         //std::cerr << "thread " << std::this_thread::get_id() << "add tasks\n";
        //         tasks.addTasks(numReadPairsInBatch, currentIds.data(), currentReadLengths.data(), currentEncodedReads.data(), currentQualityScores.data(), stream);

        //         //gpuReadExtender->setState(GpuReadExtender::State::UpdateWorkingSet);

        //         //std::cerr << "Added " << (numReadPairsInBatch * 4) << " new tasks to batch\n";
        //     }

        //     nvtx::pop_range();
        // };


        auto output = [&](){

            nvtx::push_range("output", 5);

            auto [extendedReads, pairsTmp] = makeAndSplitExtensionOutput(finishedTasks, rawExtendResult, gpudata.gpuReadExtender.get(), !isLastIteration, stream);

            pairsWhichShouldBeRepeatedTemp.insert(pairsWhichShouldBeRepeatedTemp.end(), pairsTmp.begin(), pairsTmp.end());

            const std::size_t numExtended = extendedReads.size();

            submitReadyResults(std::move(extendedReads));

            progressThread.addProgress(numExtended);

            nvtx::pop_range();
        };

        isLastIteration = false;
        while(!(readIdGenerator.empty() && tasks.size() == 0)){
            if(int(tasks.size()) < (batchsizePairs * 4) / 2){
                init1();
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

        iterationConfig.maxextensionPerStep -= 4;

        std::swap(pairsWhichShouldBeRepeatedTemp, pairsWhichShouldBeRepeated);
        pairsWhichShouldBeRepeatedTemp.clear();

        std::sort(pairsWhichShouldBeRepeated.begin(), pairsWhichShouldBeRepeated.end());

        generatorWrapper.reset(
            pairsWhichShouldBeRepeated.data(), 
            pairsWhichShouldBeRepeated.data() + pairsWhichShouldBeRepeated.size()
        );

        while(pairsWhichShouldBeRepeated.size() > 0 && (iterationConfig.maxextensionPerStep > 0)){

            std::cerr << "thread " << threadId << " will repeat extension of " << pairsWhichShouldBeRepeated.size() / 2 << " read pairs with fixedStepsize = " << iterationConfig.maxextensionPerStep << "\n";
            isLastIteration = (iterationConfig.maxextensionPerStep <= 4);

            //while(!(pairsWhichShouldBeRepeated.size() == 0 && tasks.size() == 0)){
            while(!(generatorWrapper.empty() && tasks.size() == 0)){
                if(int(tasks.size()) < (batchsizePairs * 4) / 2){
                    init2();
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
                
                if(finishedTasks.size() > std::size_t((batchsizePairs * 4))){
                    output();
                }
            }
    
            output();
            assert(finishedTasks.size() == 0);

            //std::cerr << "thread " << ompThreadId << " finished extra loop with fixedStepsize = " << fixedStepsize << "\n";

            iterationConfig.maxextensionPerStep -= 4;
            std::swap(pairsWhichShouldBeRepeatedTemp, pairsWhichShouldBeRepeated);
            pairsWhichShouldBeRepeatedTemp.clear();
            
            std::sort(pairsWhichShouldBeRepeated.begin(), pairsWhichShouldBeRepeated.end());

            generatorWrapper.reset(
                pairsWhichShouldBeRepeated.data(), 
                pairsWhichShouldBeRepeated.data() + pairsWhichShouldBeRepeated.size()
            );
        }

        //std::cerr << "thread " << ompThreadId << " finished all extensions\n";

        // while(pairsWhichShouldBeRepeated.size() > 0 && ((minCoverageForExtension < limit))){

        //     //std::cerr << "Will repeat extension of " << pairsWhichShouldBeRepeated.size() << " read pairs with minCoverageForExtension = " << minCoverageForExtension << ", fixedStepsize = " << fixedStepsize << "\n";
        //     isLastIteration = (minCoverageForExtension + increment >= limit);

        //     while(pairsWhichShouldBeRepeated.size() > 0){
        //         init();
        //         if(gpuReadExtender->state != GpuReadExtender::State::None){
        //             gpuExtensionStepper.process(*gpuReadExtender);
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

    };


    std::vector<std::future<void>> futures;

    const int maxNumThreads = runtimeOptions.threads;

    for(int t = 0; t < maxNumThreads; t++){
        futures.emplace_back(
            std::async(
                std::launch::async,
                extenderThreadFunc,
                t % numDeviceIds,
                t
            )
        );
    }

    for(auto& f : futures){
        f.wait();
    }


#else

    #pragma omp parallel
    {
        const int numDeviceIds = runtimeOptions.deviceIds.size();

        assert(numDeviceIds > 0);

        const int ompThreadId = omp_get_thread_num();
        const int deviceIdIndex = ompThreadId % numDeviceIds;
        const int deviceId = runtimeOptions.deviceIds.at(deviceIdIndex);
        cudaSetDevice(deviceId); CUERR;

        std::array<CudaStream, 4> streams{};
        std::array<cudaStream_t, 4> streamsraw{};
        for(int i = 0; i < 4; i++){
            streamsraw[i] = streams[i].getStream();
        }

        cub::CachingDeviceAllocator myCubAllocator(
            2, //bin_growth
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


        helpers::SimpleAllocationPinnedHost<read_number> currentIds(2 * batchsizePairs);
        helpers::SimpleAllocationDevice<unsigned int> currentEncodedReads(2 * encodedSequencePitchInInts * batchsizePairs);
        helpers::SimpleAllocationDevice<int> currentReadLengths(2 * batchsizePairs);
        helpers::SimpleAllocationDevice<char> currentQualityScores(2 * qualityPitchInBytes * batchsizePairs);

        CudaStream stream;

        if(!correctionOptions.useQualityScores){
            helpers::call_fill_kernel_async(currentQualityScores.data(), currentQualityScores.size(), 'I', stream);
        }


        const bool isPairedEnd = true;

        GpuReadExtender::Hasher anchorHasher(minhasher);

        GpuReadExtender::TaskData tasks(myCubAllocator, 0, encodedSequencePitchInInts, decodedSequencePitchInBytes, qualityPitchInBytes, (cudaStream_t)0);
        GpuReadExtender::TaskData finishedTasks(myCubAllocator, 0, encodedSequencePitchInInts, decodedSequencePitchInBytes, qualityPitchInBytes, (cudaStream_t)0);

        GpuReadExtender::AnchorData anchorData(myCubAllocator);
        GpuReadExtender::AnchorHashResult anchorHashResult(myCubAllocator);

        GpuReadExtender::IterationConfig iterationConfig{};

        auto gpuReadExtender = std::make_unique<GpuReadExtender>(
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
            streamsraw,
            myCubAllocator
        );
        gpuReadExtender->someId = ompThreadId;

        iterationConfig.maxextensionPerStep = 20;
        iterationConfig.minCoverageForExtension = 3;

        // int minCoverageForExtension = 3;
        // int fixedStepsize = 20;

        //gpuExtensionStepper.setMaxExtensionPerStep(fixedStepsize);
        //gpuExtensionStepper.setMinCoverageForExtension(minCoverageForExtension);

        // gpuReadExtender->setMaxExtensionPerStep(fixedStepsize);
        // gpuReadExtender->setMinCoverageForExtension(minCoverageForExtension);

        GpuReadExtender::RawExtendResult rawExtendResult{};

        std::vector<std::pair<read_number, read_number>> pairsWhichShouldBeRepeated;
        std::vector<std::pair<read_number, read_number>> pairsWhichShouldBeRepeatedTemp;
        bool isLastIteration = false;

        auto init = [&](){
            nvtx::push_range("init", 2);

            const int maxNumPairs = (batchsizePairs * 4 - tasks.size()) / 4;
            assert(maxNumPairs <= batchsizePairs);

            auto readIdsEnd = readIdGenerator.next_n_into_buffer(
                maxNumPairs * 2, 
                currentIds.get()
            );

            int numNewReadsInBatch = std::distance(currentIds.get(), readIdsEnd);

            if(numNewReadsInBatch % 2 == 1){
                throw std::runtime_error("Input files not properly paired. Aborting read extension.");
            }

            if(numNewReadsInBatch == 0 && pairsWhichShouldBeRepeated.size() > 0){

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

                numNewReadsInBatch = 2 * numPairsToCopy;
            }
            
            if(numNewReadsInBatch > 0){
                
                gpuReadStorage.gatherSequences(
                    readStorageHandle,
                    currentEncodedReads.get(),
                    encodedSequencePitchInInts,
                    makeAsyncConstBufferWrapper(currentIds.get()),
                    currentIds.get(), //device accessible
                    numNewReadsInBatch,
                    stream
                );

                gpuReadStorage.gatherSequenceLengths(
                    readStorageHandle,
                    currentReadLengths.get(),
                    currentIds.get(),
                    numNewReadsInBatch,
                    stream
                );

                assert(currentQualityScores.size() >= numNewReadsInBatch * qualityPitchInBytes);

                if(correctionOptions.useQualityScores){
                    gpuReadStorage.gatherQualities(
                        readStorageHandle,
                        currentQualityScores.get(),
                        qualityPitchInBytes,
                        makeAsyncConstBufferWrapper(currentIds.get()),
                        currentIds.get(), //device accessible
                        numNewReadsInBatch,
                        stream
                    );
                }
                
                const int numReadPairsInBatch = numNewReadsInBatch / 2;

                tasks.addTasks(numReadPairsInBatch, currentIds.data(), currentReadLengths.data(), currentEncodedReads.data(), currentQualityScores.data(), stream);

                //gpuReadExtender->setState(GpuReadExtender::State::UpdateWorkingSet);

                //std::cerr << "Added " << (numReadPairsInBatch * 4) << " new tasks to batch\n";
            }

            nvtx::pop_range();
        };


        auto output = [&](){
            nvtx::push_range("output", 5);

            gpuReadExtender->constructRawResults(finishedTasks, rawExtendResult, stream);
            cudaStreamSynchronizeWrapper(stream); CUERR;

            std::vector<extension::ExtendResult> extensionResults = gpuReadExtender->convertRawExtendResults(rawExtendResult);

            //std::vector<extension::ExtendResult> extensionResults = gpuReadExtender->constructResults4();
            const int numresults = extensionResults.size();

            //std::cerr << "Got " << (numresults) << " extended reads. Remaining unprocessed finished tasks: " << gpuReadExtender->finishedTasks->size() << "\n";


            std::vector<ExtendedRead> extendedReads;
            extendedReads.reserve(numresults);

            int repeated = 0;

            nvtx::push_range("convert extension results", 7);

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
            nvtx::pop_range();

            //gpuReadExtender->setState(GpuReadExtender::State::None);

            nvtx::pop_range();

            progressThread.addProgress(numresults - repeated);
        };

        //std::cerr << "thread " << ompThreadId << " begins main loop\n";

        isLastIteration = false;
        while(!(readIdGenerator.empty() && tasks.size() == 0)){
            if(int(tasks.size()) < (batchsizePairs * 4) / 2){
                init();
            }

            tasks.aggregateAnchorData(anchorData, stream);
            anchorHasher.getCandidateReadIds(anchorData, anchorHashResult, stream);

            gpuReadExtender->processOneIteration(tasks, anchorData, anchorHashResult, finishedTasks, iterationConfig, stream);

            cudaStreamSynchronizeWrapper(stream); CUERR;
            
            if(finishedTasks.size() > std::size_t((batchsizePairs * 4) / 2)){
                output();
            }

            //std::cerr << "Remaining: tasks " << tasks.size() << ", finishedtasks " << gpuReadExtender->finishedTasks->size() << "\n";
        }

        output();
        assert(finishedTasks.size() == 0);

        std::cerr << "\nalltimetotalTaskBytes = " << gpuReadExtender->alltimetotalTaskBytes << "\n";

        //gpuReadExtender->printTransitions = true;
        //std::cerr << "thread " << ompThreadId << " finished main loop\n";

        // constexpr int increment = 1;
        // constexpr int limit = 10;

        iterationConfig.maxextensionPerStep -= 4;

        //minCoverageForExtension += increment;
        std::swap(pairsWhichShouldBeRepeatedTemp, pairsWhichShouldBeRepeated);

        std::sort(pairsWhichShouldBeRepeated.begin(), pairsWhichShouldBeRepeated.end(), [](const auto& p1, const auto& p2){
            return p1.first < p2.first;
        });

        while(pairsWhichShouldBeRepeated.size() > 0 && (iterationConfig.maxextensionPerStep > 0)){

            //gpuReadExtender->setMinCoverageForExtension(minCoverageForExtension);

            //gpuExtensionStepper.setMaxExtensionPerStep(fixedStepsize);
            //std::cerr << "fixedStepsize = " << fixedStepsize << "\n"; 
            //gpuExtensionStepper.setMinCoverageForExtension(minCoverageForExtension);

            std::cerr << "thread " << ompThreadId << " will repeat extension of " << pairsWhichShouldBeRepeated.size() << " read pairs with fixedStepsize = " << iterationConfig.maxextensionPerStep << "\n";
            isLastIteration = (iterationConfig.maxextensionPerStep <= 4);

            while(!(pairsWhichShouldBeRepeated.size() == 0 && tasks.size() == 0)){
                if(int(tasks.size()) < (batchsizePairs * 4) / 2){
                    init();
                }
    
                tasks.aggregateAnchorData(anchorData, stream);
                anchorHasher.getCandidateReadIds(anchorData, anchorHashResult, stream);

                gpuReadExtender->processOneIteration(tasks, anchorData, anchorHashResult, finishedTasks, iterationConfig, stream);

                cudaStreamSynchronizeWrapper(stream); CUERR;
                
                if(finishedTasks.size() > std::size_t((batchsizePairs * 4))){
                    output();
                }
            }
    
            output();
            assert(finishedTasks.size() == 0);

            //std::cerr << "thread " << ompThreadId << " finished extra loop with fixedStepsize = " << fixedStepsize << "\n";

            iterationConfig.maxextensionPerStep -= 4;
            std::swap(pairsWhichShouldBeRepeatedTemp, pairsWhichShouldBeRepeated);

            std::sort(pairsWhichShouldBeRepeated.begin(), pairsWhichShouldBeRepeated.end(), [](const auto& p1, const auto& p2){
                return p1.first < p2.first;
            });
        }

        //std::cerr << "thread " << ompThreadId << " finished all extensions\n";

        // while(pairsWhichShouldBeRepeated.size() > 0 && ((minCoverageForExtension < limit))){

        //     //std::cerr << "Will repeat extension of " << pairsWhichShouldBeRepeated.size() << " read pairs with minCoverageForExtension = " << minCoverageForExtension << ", fixedStepsize = " << fixedStepsize << "\n";
        //     isLastIteration = (minCoverageForExtension + increment >= limit);

        //     while(pairsWhichShouldBeRepeated.size() > 0){
        //         init();
        //         if(gpuReadExtender->state != GpuReadExtender::State::None){
        //             gpuExtensionStepper.process(*gpuReadExtender);
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

#endif



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