
#include <gpu/cudaerrorcheck.cuh>
#include <gpu/gpuminhasher.cuh>
#include <gpu/gpureadstorage.cuh>
#include <gpu/readextender_gpu.hpp>
#include <gpu/readextension_gpu.hpp>

#include <alignmentorientation.hpp>
#include <concurrencyhelpers.hpp>
#include <config.hpp>
#include <cpu_alignment.hpp>
#include <extendedread.hpp>
#include <filehelpers.hpp>
#include <hpc_helpers.cuh>
#include <msa.hpp>
#include <options.hpp>
#include <rangegenerator.hpp>
#include <readextender_common.hpp>
#include <sequencehelpers.hpp>
#include <serializedobjectstorage.hpp>
#include <threadpool.hpp>
#include <util.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>




#include <omp.h>
#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>




namespace care{
namespace gpu{

template<class T>
struct IsGreaterThan{
    T value;

    __host__ __device__
    IsGreaterThan(T t) : value(t){}

    template<class V>
    __host__ __device__
    bool operator()(V item) const noexcept{
        return item > value;
    }
};

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
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr
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
            stream,
            mr
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
                stream,
                mr
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


extension::SplittedExtensionOutput makeAndSplitExtensionOutput(
    GpuReadExtender::TaskData& finishedTasks, 
    GpuReadExtender::RawExtendResult& rawExtendResult, 
    const GpuReadExtender* gpuReadExtender, 
    bool isRepeatedIteration, 
    cudaStream_t stream
){

    nvtx::push_range("constructRawResults", 4);
    gpuReadExtender->constructRawResults(finishedTasks, rawExtendResult, stream);
    nvtx::pop_range();

    CUDACHECK(cudaStreamSynchronizeWrapper(stream));

    std::vector<extension::ExtendResult> extensionResults = gpuReadExtender->convertRawExtendResults(rawExtendResult);


    return splitExtensionOutput(extensionResults, isRepeatedIteration);
}




struct ExtensionPipeline{
    using ReadyResultsCallback = std::function<void(std::vector<ExtendedRead>&&, std::vector<EncodedExtendedRead>&&, std::vector<read_number>&&)>;
    static constexpr bool isPairedEnd = true; 

    const ProgramOptions& programOptions;
    const GpuMinhasher& minhasher;
    const GpuReadStorage& gpuReadStorage;
    std::size_t encodedSequencePitchInInts;
    std::size_t decodedSequencePitchInBytes;
    std::size_t qualityPitchInBytes;
    std::size_t msaColumnPitchInElements;
    ReadyResultsCallback submitReadyResults;
    ProgressThread<read_number>* progressThread;

    std::vector<read_number> pairsWhichShouldBeRepeated{};
    std::vector<read_number> pairsWhichShouldBeRepeatedTmp{};

    ExtensionPipeline(
        const ProgramOptions& programOptions_,
        const GpuMinhasher& minhasher_,
        const GpuReadStorage& gpuReadStorage_,
        std::size_t encodedSequencePitchInInts_,
        std::size_t decodedSequencePitchInBytes_,
        std::size_t qualityPitchInBytes_,
        std::size_t msaColumnPitchInElements_,
        ReadyResultsCallback submitReadyResults_,
        ProgressThread<read_number>* progressThread_
    )
    : programOptions(programOptions_),
        minhasher(minhasher_),
        gpuReadStorage(gpuReadStorage_),
        encodedSequencePitchInInts(encodedSequencePitchInInts_),
        decodedSequencePitchInBytes(decodedSequencePitchInBytes_),
        qualityPitchInBytes(qualityPitchInBytes_),
        msaColumnPitchInElements(msaColumnPitchInElements_),
        submitReadyResults(submitReadyResults_),
        progressThread(progressThread_)
    {

    }

    void executeFirstPass(){
        constexpr bool extraHashing = false;
        constexpr bool isRepeatedIteration = false;

        GpuReadExtender::IterationConfig iterationConfig;
        iterationConfig.maxextensionPerStep = programOptions.fixedStepsize == 0 ? 20 : programOptions.fixedStepsize;
        iterationConfig.minCoverageForExtension = 3;

        const std::size_t numReadsToProcess = getNumReadsToProcess(&gpuReadStorage, programOptions);

        IteratorRangeTraversal<thrust::counting_iterator<read_number>> readIdGenerator(
            thrust::make_counting_iterator<read_number>(0),
            thrust::make_counting_iterator<read_number>(0) + numReadsToProcess
        );

        const int maxNumThreads = programOptions.threads;
        const int numDevices = programOptions.deviceIds.size();

        std::cerr << "First Pass\n";
        std::cerr << "use " << maxNumThreads << " threads\n";

        std::vector<std::future<std::vector<read_number>>> futures;
        for(int t = 0; t < maxNumThreads; t++){
            futures.emplace_back(
                std::async(
                    std::launch::async,
                    [&](){
                        const int deviceId = programOptions.deviceIds[t % numDevices];
                        return extenderThreadFunc(
                            deviceId,
                            &readIdGenerator,
                            isRepeatedIteration,
                            extraHashing,
                            iterationConfig
                        );
                    }
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
    }


    void runExtraHashingPass(){
        constexpr bool extraHashing = true;
        constexpr bool isRepeatedIteration = true;

        GpuReadExtender::IterationConfig iterationConfig;
        iterationConfig.maxextensionPerStep = programOptions.fixedStepsize == 0 ? 20 : programOptions.fixedStepsize;
        iterationConfig.minCoverageForExtension = 3;


        const int numPairsToRepeat = pairsWhichShouldBeRepeated.size() / 2;
        std::cerr << "Extra hashing pass\n";
        std::cerr << "Will repeat extension of " << numPairsToRepeat << " read pairs\n";

        //isLastIteration = (iterationConfig.maxextensionPerStep <= 4);

        auto readIdGenerator = makeIteratorRangeTraversal(
            pairsWhichShouldBeRepeated.data(), 
            pairsWhichShouldBeRepeated.data() + pairsWhichShouldBeRepeated.size()
        );

        const int threadsForPairs = SDIV(numPairsToRepeat, programOptions.batchsize);
        const int maxNumThreads = std::min(threadsForPairs, programOptions.threads);
        std::cerr << "use " << maxNumThreads << " threads\n";
        const int numDevices = programOptions.deviceIds.size();

        std::vector<std::future<std::vector<read_number>>> futures;
        for(int t = 0; t < maxNumThreads; t++){
            futures.emplace_back(
                std::async(
                    std::launch::async,
                    [&](){
                        const int deviceId = programOptions.deviceIds[t % numDevices];
                        return extenderThreadFunc(
                            deviceId,
                            &readIdGenerator,
                            isRepeatedIteration,
                            extraHashing,
                            iterationConfig
                        );
                    }
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

        submitReadyResults(
            {}, 
            {},
            std::move(pairsWhichShouldBeRepeated) //pairs which did not find mate after repetition will remain unextended
        );
    }

    template<class ReadIdGenerator>
    std::vector<read_number> extenderThreadFunc(
        int deviceId,
        ReadIdGenerator* readIdGenerator,
        bool isRepeatedIteration,
        bool extraHashing,
        GpuReadExtender::IterationConfig iterationConfig
    ){
        CUDACHECK(cudaSetDevice(deviceId));
        cudaStream_t stream = cudaStreamPerThread;
        rmm::mr::device_memory_resource* rmmDeviceResource = rmm::mr::get_current_device_resource();

        cpu::QualityScoreConversion qualityConversion{};

        auto gpuReadExtender = std::make_unique<GpuReadExtender>(
            encodedSequencePitchInInts,
            decodedSequencePitchInBytes,
            qualityPitchInBytes,
            msaColumnPitchInElements,
            isPairedEnd,
            gpuReadStorage, 
            programOptions,
            qualityConversion,
            programOptions.insertSize,
            (programOptions.fixedStddev == 0 ? programOptions.insertSizeStddev : programOptions.fixedStddev),
            stream,
            rmmDeviceResource
        );

        std::map<int, int> extensionLengthsMap;
        std::map<int, int> mismatchesBetweenMateExtensions;

        ReadStorageHandle readStorageHandle = gpuReadStorage.makeHandle();


        helpers::SimpleAllocationPinnedHost<read_number> currentIds(2 * programOptions.batchsize);
        helpers::SimpleAllocationDevice<unsigned int> currentEncodedReads(2 * encodedSequencePitchInInts * programOptions.batchsize);
        helpers::SimpleAllocationDevice<int> currentReadLengths(2 * programOptions.batchsize);
        helpers::SimpleAllocationDevice<char> currentQualityScores(2 * qualityPitchInBytes * programOptions.batchsize);

        if(!programOptions.useQualityScores){
            helpers::call_fill_kernel_async(currentQualityScores.data(), currentQualityScores.size(), 'I', stream);
        }


        GpuReadExtender::Hasher anchorHasher(minhasher, rmmDeviceResource);

        GpuReadExtender::TaskData tasks(rmmDeviceResource, 0, encodedSequencePitchInInts, decodedSequencePitchInBytes, qualityPitchInBytes, stream);
        GpuReadExtender::TaskData finishedTasks(rmmDeviceResource, 0, encodedSequencePitchInInts, decodedSequencePitchInBytes, qualityPitchInBytes, stream);

        GpuReadExtender::AnchorData anchorData(rmmDeviceResource);
        GpuReadExtender::AnchorHashResult anchorHashResult(rmmDeviceResource);

        GpuReadExtender::RawExtendResult rawExtendResult{};

        std::vector<read_number> pairsWhichShouldBeRepeated;

        auto output = [&](){

            nvtx::push_range("output", 5);

            nvtx::push_range("convert extension results", 7);

            auto splittedExtOutput = makeAndSplitExtensionOutput(finishedTasks, rawExtendResult, gpuReadExtender.get(), isRepeatedIteration, stream);

            nvtx::pop_range();

            pairsWhichShouldBeRepeated.insert(
                pairsWhichShouldBeRepeated.end(), 
                splittedExtOutput.idsOfPartiallyExtendedReads.begin(), 
                splittedExtOutput.idsOfPartiallyExtendedReads.end()
            );

            const std::size_t numExtended = splittedExtOutput.extendedReads.size();

            if(!programOptions.allowOutwardExtension){
                for(auto& er : splittedExtOutput.extendedReads){
                    er.removeOutwardExtension();
                }
            }

            std::vector<EncodedExtendedRead> encvec(numExtended);
            for(std::size_t i = 0; i < numExtended; i++){
                splittedExtOutput.extendedReads[i].encodeInto(encvec[i]);
            }

            submitReadyResults(
                std::move(splittedExtOutput.extendedReads), 
                std::move(encvec),
                std::move(splittedExtOutput.idsOfNotExtendedReads)
            );

            progressThread->addProgress(numExtended);

            nvtx::pop_range();
        };

        while(!(readIdGenerator->empty() && tasks.size() == 0)){
            if(int(tasks.size()) < (programOptions.batchsize * 4) / 2){
                initializeExtenderInput(
                    *readIdGenerator,
                    programOptions.batchsize * 4,
                    gpuReadStorage,
                    readStorageHandle,
                    currentIds.data(), 
                    currentReadLengths.data(), 
                    currentEncodedReads.data(),
                    programOptions.useQualityScores,
                    currentQualityScores.data(), 
                    encodedSequencePitchInInts,
                    qualityPitchInBytes,
                    tasks,
                    stream,
                    rmmDeviceResource
                );
            }

            tasks.aggregateAnchorData(anchorData, stream);
            
            nvtx::push_range("getCandidateReadIds", 4);
            if(extraHashing){
            //if(false){
                anchorHasher.getCandidateReadIdsWithExtraExtensionHash(
                    anchorData, 
                    anchorHashResult,
                    iterationConfig, 
                    thrust::make_transform_iterator(
                        tasks.iteration.data(),
                        IsGreaterThan<int>{0}
                    ),
                    stream
                );
            }else{
                anchorHasher.getCandidateReadIds(anchorData, anchorHashResult, stream);
            }
            // #if 0
            // anchorHasher.getCandidateReadIds(anchorData, anchorHashResult, stream);
            // #else
            // anchorHasher.getCandidateReadIdsWithExtraExtensionHash(
            //     *gpudata.dataAllocator,
            //     anchorData, 
            //     anchorHashResult,
            //     iterationConfig, 
            //     thrust::make_transform_iterator(
            //         tasks.iteration.data(),
            //         IsGreaterThan<int>{0}
            //     ),
            //     stream
            // );
            // #endif
            nvtx::pop_range();

            gpuReadExtender->processOneIteration(
                tasks,
                anchorData, 
                anchorHashResult, 
                finishedTasks, 
                iterationConfig,
                stream
            );

            CUDACHECK(cudaStreamSynchronizeWrapper(stream));
            
            if(finishedTasks.size() > std::size_t((programOptions.batchsize * 4) / 2)){
                output();
            }

            //std::cerr << "Remaining: tasks " << tasks.size() << ", finishedtasks " << gpuReadExtender->finishedTasks->size() << "\n";
        }

        output();
        assert(finishedTasks.size() == 0);
        
        gpuReadStorage.destroyHandle(readStorageHandle);

        return pairsWhichShouldBeRepeated;
    };
};
















template<class Callback>
void extend_gpu_pairedend(
    const ProgramOptions& programOptions,
    const GpuMinhasher& minhasher,
    const GpuReadStorage& gpuReadStorage,
    Callback submitReadyResults
){
 

    const std::uint64_t totalNumReadPairs = getNumReadsToProcess(&gpuReadStorage, programOptions) / 2;

    auto showProgress = [&](auto totalCount, auto seconds){
        if(programOptions.showProgress){

            printf("Processed %10u of %10lu read pairs (Runtime: %03d:%02d:%02d)\r",
                    totalCount, totalNumReadPairs,
                    int(seconds / 3600),
                    int(seconds / 60) % 60,
                    int(seconds) % 60);
            std::cout.flush();
        }

        if(totalCount == totalNumReadPairs){
            std::cout << '\n';
        }
    };

    auto updateShowProgressInterval = [](auto duration){
        return duration;
    };

    ProgressThread<read_number> progressThread(totalNumReadPairs, showProgress, updateShowProgressInterval);

    const int maximumSequenceLength = gpuReadStorage.getSequenceLengthUpperBound();
    const std::size_t encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength);
    const std::size_t decodedSequencePitchInBytes = SDIV(maximumSequenceLength, 128) * 128;
    const std::size_t qualityPitchInBytes = SDIV(maximumSequenceLength, 128) * 128;

    const std::size_t min_overlap = std::max(
        1, 
        std::max(
            programOptions.min_overlap, 
            int(maximumSequenceLength * programOptions.min_overlap_ratio)
        )
    );
    const std::size_t msa_max_column_count = (3*gpuReadStorage.getSequenceLengthUpperBound() - 2*min_overlap);
    //round up to 32 elements
    const std::size_t msaColumnPitchInElements = SDIV(msa_max_column_count, 32) * 32;

    //omp_set_num_threads(1);

    assert(programOptions.deviceIds.size() > 0);
    CUDACHECK(cudaSetDevice(programOptions.deviceIds[0]));


    ExtensionPipeline extensionPipeline(
        programOptions,
        minhasher,
        gpuReadStorage,
        encodedSequencePitchInInts,
        decodedSequencePitchInBytes,
        qualityPitchInBytes,
        msaColumnPitchInElements,
        submitReadyResults,
        &progressThread
    );

    extensionPipeline.executeFirstPass();
    extensionPipeline.runExtraHashingPass();

   

    progressThread.finished();

}


#if 0

SerializedObjectStorage extend_gpu_singleend(
    const ProgramOptions& programOptions,
    const GpuMinhasher& minhasher,
    const GpuReadStorage& gpuReadStorage
){
    std::cerr << "extend_gpu_singleend\n";
    throw std::runtime_error("extend_gpu_singleend NOT IMPLEMENTED");
    
}
#endif

void extend_gpu(
    const ProgramOptions& programOptions,
    const GpuMinhasher& gpumMinhasher,
    const GpuReadStorage& gpuReadStorage,
    SubmitReadyExtensionResultsCallback submitReadyResults
){
    // if(programOptions.pairType == SequencePairType::SingleEnd){
    //     return extend_gpu_singleend(
    //         programOptions,
    //         gpumMinhasher,
    //         gpuReadStorage
    //     );
    // }else{
        extend_gpu_pairedend(
            programOptions,
            gpumMinhasher,
            gpuReadStorage,
            submitReadyResults
        );
    //}
}





} // namespace gpu

} // namespace care