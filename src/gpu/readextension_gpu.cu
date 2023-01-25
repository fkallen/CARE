
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

#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <gpu/rmm_utilities.cuh>

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

//get read ids from generator, 
//populate currentIds, currentReadLengths, currentEncodedReads, currentQualityScores
//return number of new reads
template<class IdGenerator>
int getNewReadsForExtender(
    IdGenerator& readIdGenerator,
    int maxNumNewPairs,
    const GpuReadStorage& gpuReadStorage,
    ReadStorageHandle& readStorageHandle,
    read_number* currentIds, // pinned memory
    int* currentReadLengths, //device accessible
    unsigned int* currentEncodedReads, //device accessible
    bool useQualityScores,
    char* currentQualityScores, //device accessible
    std::size_t encodedSequencePitchInInts,
    std::size_t qualityPitchInBytes,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr
){
    nvtx::ScopedRange sr("getNewReadsForExtender", 2);

    int numNewReadsInBatch = 0;

    readIdGenerator.process_next_n(
        maxNumNewPairs * 2, 
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
    }

    return numNewReadsInBatch;
};

void addNewReadsToTasks(
    int numNewReads,
    read_number* currentIds, // pinned memory
    int* currentReadLengths, //device accessible
    unsigned int* currentEncodedReads, //device accessible
    bool /*useQualityScores*/,
    char* currentQualityScores, //device accessible
    std::size_t /*encodedSequencePitchInInts*/,
    std::size_t /*qualityPitchInBytes*/,
    GpuReadExtender::TaskData& tasks,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* /*mr*/
){
    nvtx::ScopedRange sr("addNewReadsToTasks", 2);

   
    const int numReadPairsInBatch = numNewReads / 2; 
    tasks.addTasks(
        numReadPairsInBatch, 
        currentIds, 
        currentReadLengths, 
        currentEncodedReads, 
        currentQualityScores, 
        stream
    );

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


    return splitExtensionOutput(std::move(extensionResults), isRepeatedIteration);
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

    int deviceId;

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
        CUDACHECK(cudaGetDevice(&deviceId));
    }


    std::vector<read_number> executeFirstPass(){
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

        const int maxNumThreads = std::min(programOptions.threads, programOptions.gpuExtenderThreadConfig.numExtenders);
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
                        return combinedHasherAndExtenderThreadFunc(
                            &readIdGenerator,
                            isRepeatedIteration,
                            extraHashing,
                            iterationConfig
                        );
                    }
                )
            );
        }

        std::vector<read_number> pairsWhichShouldBeRepeated{};

        for(auto& f : futures){
            auto vec = f.get();
            pairsWhichShouldBeRepeated.insert(pairsWhichShouldBeRepeated.end(), vec.begin(), vec.end());
        }

        std::sort(pairsWhichShouldBeRepeated.begin(), pairsWhichShouldBeRepeated.end());

        return pairsWhichShouldBeRepeated;
    }


    std::vector<read_number> executeExtraHashingPass(const std::vector<read_number>& pairsWhichShouldBeRepeated){
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
        const int maxNumThreads = std::min(std::min(threadsForPairs, programOptions.threads), programOptions.gpuExtenderThreadConfig.numExtenders);
        std::cerr << "use " << maxNumThreads << " threads\n";
        const int numDevices = programOptions.deviceIds.size();

        std::vector<std::future<std::vector<read_number>>> futures;
        for(int t = 0; t < maxNumThreads; t++){
            futures.emplace_back(
                std::async(
                    std::launch::async,
                    [&](){
                        return combinedHasherAndExtenderThreadFunc(
                            &readIdGenerator,
                            isRepeatedIteration,
                            extraHashing,
                            iterationConfig
                        );
                    }
                )
            );
        }

        std::vector<read_number> pairsWhichShouldBeRepeatedTmp;

        for(auto& f : futures){
            auto vec = f.get();
            pairsWhichShouldBeRepeatedTmp.insert(pairsWhichShouldBeRepeatedTmp.end(), vec.begin(), vec.end());
        }

        std::sort(pairsWhichShouldBeRepeatedTmp.begin(), pairsWhichShouldBeRepeatedTmp.end());

        return pairsWhichShouldBeRepeatedTmp;
    }

    template<class ReadIdGenerator>
    std::vector<read_number> combinedHasherAndExtenderThreadFunc(
        ReadIdGenerator* readIdGenerator,
        bool isRepeatedIteration,
        bool extraHashing,
        GpuReadExtender::IterationConfig iterationConfig
    ){
        CUDACHECK(cudaSetDevice(deviceId));
        cudaStream_t stream = cudaStreamPerThread;
        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();

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
            stream,
            mr
        );

        ReadStorageHandle readStorageHandle = gpuReadStorage.makeHandle();

        helpers::SimpleAllocationPinnedHost<read_number> currentIds(2 * programOptions.batchsize);
        rmm::device_uvector<unsigned int> currentEncodedReads(2 * encodedSequencePitchInInts * programOptions.batchsize, stream, mr);
        rmm::device_uvector<int> currentReadLengths(2 * programOptions.batchsize, stream, mr);
        rmm::device_uvector<char> currentQualityScores(2 * qualityPitchInBytes * programOptions.batchsize, stream, mr);

        if(!programOptions.useQualityScores){
            thrust::fill(
                rmm::exec_policy_nosync(stream, mr),
                currentQualityScores.begin(),
                currentQualityScores.end(),
                'I'
            );
        }

        GpuReadExtender::Hasher anchorHasher(minhasher, mr);

        GpuReadExtender::TaskData tasks(mr, 0, encodedSequencePitchInInts, decodedSequencePitchInBytes, qualityPitchInBytes, stream);
        GpuReadExtender::TaskData finishedTasks(mr, 0, encodedSequencePitchInInts, decodedSequencePitchInBytes, qualityPitchInBytes, stream);

        GpuReadExtender::AnchorData anchorData(stream, mr);
        GpuReadExtender::AnchorHashResult anchorHashResult(stream, mr);

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
                const int maxNumNewPairs = (programOptions.batchsize * 4 - tasks.size()) / 4;
                const int numNewReads = getNewReadsForExtender(
                    *readIdGenerator,
                    maxNumNewPairs,
                    gpuReadStorage,
                    readStorageHandle,
                    currentIds.data(), 
                    currentReadLengths.data(), 
                    currentEncodedReads.data(),
                    programOptions.useQualityScores,
                    currentQualityScores.data(), 
                    encodedSequencePitchInInts,
                    qualityPitchInBytes,
                    stream,
                    mr
                );

                addNewReadsToTasks(
                    numNewReads,
                    currentIds.data(), 
                    currentReadLengths.data(), 
                    currentEncodedReads.data(),
                    programOptions.useQualityScores,
                    currentQualityScores.data(), 
                    encodedSequencePitchInInts,
                    qualityPitchInBytes,
                    tasks,
                    stream,
                    mr
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



struct ExtensionPipelineProducerConsumer{
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

    int deviceId;

    struct TaskBatch{
        static int counter;

        int extenderId;
        GpuReadExtender::TaskData* taskData{};
        GpuReadExtender::AnchorData* anchorData{};
        GpuReadExtender::AnchorHashResult* anchorHashResult{};
        CudaEvent event{cudaEventDisableTiming};
        cudaStream_t stream;

        TaskBatch(
            cudaStream_t stream_,
            int extenderId_,
            GpuReadExtender::TaskData* taskData_,
            GpuReadExtender::AnchorData* anchorData_,
            GpuReadExtender::AnchorHashResult* anchorHashResult_
        ):
            extenderId(extenderId_++),
            taskData(taskData_),
            anchorData(anchorData_),
            anchorHashResult(anchorHashResult_),
            stream(stream_)
        {

        }
    };

    ExtensionPipelineProducerConsumer(
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
        CUDACHECK(cudaGetDevice(&deviceId));
    }

    template<class ReadIdGenerator>
    std::vector<read_number> producer_consumer_impl(
        bool extraHashing,
        bool isRepeatedIteration,
        GpuReadExtender::IterationConfig iterationConfig,
        ReadIdGenerator& readIdGenerator,
        int numReadsToProcess
    ){

        cudaStream_t stream = cudaStreamPerThread;
        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();

        const int numDevices = programOptions.deviceIds.size();

        auto threadsPerGpu = programOptions.gpuExtenderThreadConfig;
        if(threadsPerGpu.isAutomatic()){
            threadsPerGpu.numHashers = 1;
            threadsPerGpu.numExtenders = 1;

            int numThreadsRemaining = programOptions.threads / numDevices;
            while(numThreadsRemaining > 0 && threadsPerGpu.numHashers < 4){
                threadsPerGpu.numHashers++;
                numThreadsRemaining--;
            }
            if(numThreadsRemaining >= 2){
                threadsPerGpu.numHashers++;
                threadsPerGpu.numExtenders++;
            }
            while(numThreadsRemaining > 0 && threadsPerGpu.numHashers < 10){
                threadsPerGpu.numHashers++;
                numThreadsRemaining--;
            }
        }

        const int numHashers = threadsPerGpu.numHashers;
        const int numExtenders = threadsPerGpu.numExtenders;
        const int numTaskBatches = SDIV(numHashers + numExtenders, numExtenders) * numExtenders;

        std::cerr << "Per GPU: " << "numHashers = " << numHashers << ", numExtenders = " << numExtenders << "\n";

        std::vector<CudaStream> taskStreams;
        std::vector<GpuReadExtender::TaskData> taskDataVec;
        std::vector<GpuReadExtender::AnchorData> anchorDataVec;
        std::vector<GpuReadExtender::AnchorHashResult> anchorHashResultVec;
        std::vector<TaskBatch> taskBatchVec;

        MultiProducerMultiConsumerQueue<TaskBatch*> freeTaskBatchQueue;
        std::vector<MultiProducerMultiConsumerQueue<TaskBatch*>> hashedTaskBatchQueuePerExtender(numExtenders);

        for(int i = 0; i < numTaskBatches; i++){
            taskStreams.emplace_back();
            taskDataVec.emplace_back(mr, 0, encodedSequencePitchInInts, decodedSequencePitchInBytes, qualityPitchInBytes, stream);
            anchorDataVec.emplace_back(stream, mr);
            anchorHashResultVec.emplace_back(stream, mr);            
        }

        taskBatchVec.reserve(numTaskBatches);
        for(int i = 0; i < numTaskBatches; i++){
            taskBatchVec.emplace_back(
                taskStreams[i], 
                i % numExtenders, 
                &taskDataVec[i],
                &anchorDataVec[i],
                &anchorHashResultVec[i]
            );
            freeTaskBatchQueue.push(&taskBatchVec[i]);
        }
        CUDACHECK(cudaDeviceSynchronize());


        std::atomic<std::int64_t> totalNumFinishedTasks = 0;
        const std::int64_t numTasksToProcess = (std::int64_t)numReadsToProcess * 2;

        std::vector<std::future<void>> hasherFutures;
        std::vector<std::future<std::vector<read_number>>> extenderfutures;

        for(int t = 0; t < numHashers; t++){
            hasherFutures.emplace_back(
                std::async(
                    std::launch::async,
                    [&](){
                        hasherThreadFunc(
                            &readIdGenerator,
                            freeTaskBatchQueue,
                            hashedTaskBatchQueuePerExtender,
                            isRepeatedIteration,
                            extraHashing,
                            iterationConfig,
                            totalNumFinishedTasks,
                            numTasksToProcess
                        );
                    }
                )
            );
        }

        for(int t = 0; t < numExtenders; t++){
            extenderfutures.emplace_back(
                std::async(
                    std::launch::async,
                    [&](int extenderId){
                        return extenderThreadFunc(
                            &readIdGenerator,
                            freeTaskBatchQueue,
                            hashedTaskBatchQueuePerExtender[extenderId],
                            isRepeatedIteration,
                            extraHashing,
                            iterationConfig,
                            totalNumFinishedTasks,
                            numTasksToProcess,
                            extenderId
                        );
                    },
                    t
                )
            );
        }


        std::vector<read_number> pairsWhichShouldBeRepeatedTmp{};
        //wait for hashers
        for(auto& f : hasherFutures){
            f.wait();
        }

        //hashers are done. notify extenders
        for(int i = 0; i < numExtenders; i++){
            hashedTaskBatchQueuePerExtender[i].push(nullptr);
        }

        //wait for extenders
        for(auto& f : extenderfutures){
            auto vec = f.get();
            pairsWhichShouldBeRepeatedTmp.insert(pairsWhichShouldBeRepeatedTmp.end(), vec.begin(), vec.end());
        }

        
        CUDACHECK(cudaDeviceSynchronize());

        std::sort(pairsWhichShouldBeRepeatedTmp.begin(), pairsWhichShouldBeRepeatedTmp.end());
        return pairsWhichShouldBeRepeatedTmp;
    }

    //return list of read ids of pairs that should be processed again
    std::vector<read_number> executeFirstPass(){
        constexpr bool extraHashing = false;
        constexpr bool isRepeatedIteration = false;

        GpuReadExtender::IterationConfig iterationConfig;
        iterationConfig.maxextensionPerStep = programOptions.fixedStepsize == 0 ? 20 : programOptions.fixedStepsize;
        iterationConfig.minCoverageForExtension = 3;

        const int numReadsToProcess = getNumReadsToProcess(&gpuReadStorage, programOptions);

        IteratorRangeTraversal<thrust::counting_iterator<read_number>> readIdGenerator(
            thrust::make_counting_iterator<read_number>(0),
            thrust::make_counting_iterator<read_number>(0) + numReadsToProcess
        );

        return producer_consumer_impl(
            extraHashing,
            isRepeatedIteration,
            iterationConfig,
            readIdGenerator,
            numReadsToProcess
        );
    }

    std::vector<read_number> executeExtraHashingPass(const std::vector<read_number>& pairsWhichShouldBeRepeated){
        constexpr bool extraHashing = true;
        constexpr bool isRepeatedIteration = true;

        GpuReadExtender::IterationConfig iterationConfig;
        iterationConfig.maxextensionPerStep = programOptions.fixedStepsize == 0 ? 20 : programOptions.fixedStepsize;
        iterationConfig.minCoverageForExtension = 3;

        const int numPairsToRepeat = pairsWhichShouldBeRepeated.size() / 2;

        //isLastIteration = (iterationConfig.maxextensionPerStep <= 4);

        auto readIdGenerator = makeIteratorRangeTraversal(
            pairsWhichShouldBeRepeated.data(), 
            pairsWhichShouldBeRepeated.data() + pairsWhichShouldBeRepeated.size()
        );

        return producer_consumer_impl(
            extraHashing,
            isRepeatedIteration,
            iterationConfig,
            readIdGenerator,
            numPairsToRepeat * 2
        );
    }


    template<class ReadIdGenerator, class UnhashedQueue, class HashedQueue>
    void hasherThreadFunc(
        ReadIdGenerator* readIdGenerator,
        UnhashedQueue& freeTaskBatchQueue,
        std::vector<HashedQueue>& hashedTaskBatchQueues,
        bool /*isRepeatedIteration*/,
        bool extraHashing,
        GpuReadExtender::IterationConfig iterationConfig,
        std::atomic<std::int64_t>& totalNumFinishedTasks,
        std::int64_t numTasksToProcess
    ){
        CUDACHECK(cudaSetDevice(deviceId));
        cudaStream_t stream = cudaStreamPerThread;
        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();

        ReadStorageHandle readStorageHandle = gpuReadStorage.makeHandle();

        helpers::SimpleAllocationPinnedHost<read_number> currentIds(2 * programOptions.batchsize);
        rmm::device_uvector<unsigned int> currentEncodedReads(2 * encodedSequencePitchInInts * programOptions.batchsize, stream, mr);
        rmm::device_uvector<int> currentReadLengths(2 * programOptions.batchsize, stream, mr);
        rmm::device_uvector<char> currentQualityScores(2 * qualityPitchInBytes * programOptions.batchsize, stream, mr);

        if(!programOptions.useQualityScores){
            thrust::fill(
                rmm::exec_policy_nosync(stream, mr),
                currentQualityScores.begin(),
                currentQualityScores.end(),
                'I'
            );
        }


        GpuReadExtender::Hasher anchorHasher(minhasher, mr);
        CUDACHECK(cudaDeviceSynchronize());

        std::set<int> seenEmptyTaskBatchesAfterInit;

        while(totalNumFinishedTasks != numTasksToProcess){
            TaskBatch* const taskBatchPtr = freeTaskBatchQueue.pop();
            assert(taskBatchPtr != nullptr);

            auto& tasks = *taskBatchPtr->taskData;
            auto& anchorData = *taskBatchPtr->anchorData;
            auto& anchorHashResult = *taskBatchPtr->anchorHashResult;

            //CUDACHECK(cudaStreamWaitEvent(stream, taskBatchPtr->event, 0));
            //std::cerr << "num tasks before init " << tasks.size() << "\n";
            if(int(tasks.size()) < (programOptions.batchsize * 4) / 2){
                const int maxNumNewPairs = (programOptions.batchsize * 4 - tasks.size()) / 4;
                const int numNewReads = getNewReadsForExtender(
                    *readIdGenerator,
                    maxNumNewPairs,
                    gpuReadStorage,
                    readStorageHandle,
                    currentIds.data(), 
                    currentReadLengths.data(), 
                    currentEncodedReads.data(),
                    programOptions.useQualityScores,
                    currentQualityScores.data(), 
                    encodedSequencePitchInInts,
                    qualityPitchInBytes,
                    taskBatchPtr->stream,
                    mr
                );

                addNewReadsToTasks(
                    numNewReads,
                    currentIds.data(), 
                    currentReadLengths.data(), 
                    currentEncodedReads.data(),
                    programOptions.useQualityScores,
                    currentQualityScores.data(), 
                    encodedSequencePitchInInts,
                    qualityPitchInBytes,
                    tasks,
                    taskBatchPtr->stream,
                    mr
                );
            }
            //std::cerr << "num tasks after init " << tasks.size() << "\n";
            if(int(tasks.size()) > 0){
                tasks.aggregateAnchorData(anchorData, taskBatchPtr->stream);
                
                nvtx::push_range("getCandidateReadIds", 4);
                if(extraHashing){
                    anchorHasher.getCandidateReadIdsWithExtraExtensionHash(
                        anchorData, 
                        anchorHashResult,
                        iterationConfig, 
                        thrust::make_transform_iterator(
                            tasks.iteration.data(),
                            IsGreaterThan<int>{0}
                        ),
                        taskBatchPtr->stream
                    );
                }else{
                    anchorHasher.getCandidateReadIds(anchorData, anchorHashResult, taskBatchPtr->stream);
                }

                CUDACHECK(cudaEventRecord(taskBatchPtr->event, taskBatchPtr->stream));
                nvtx::pop_range();

                hashedTaskBatchQueues[taskBatchPtr->extenderId].push(taskBatchPtr);
            }else{
                //seenEmptyTaskBatchesAfterInit.insert(taskBatchPtr->id);
                freeTaskBatchQueue.push(taskBatchPtr);
            }
        }

        gpuReadStorage.destroyHandle(readStorageHandle);
    };

    template<class ReadIdGenerator, class UnhashedQueue, class HashedQueue>
    std::vector<read_number> extenderThreadFunc(
        ReadIdGenerator* /*readIdGenerator*/,
        UnhashedQueue& freeTaskBatchQueue,
        HashedQueue& hashedTaskBatchQueue,
        bool isRepeatedIteration,
        bool /*extraHashing*/,
        GpuReadExtender::IterationConfig iterationConfig,
        std::atomic<std::int64_t>& totalNumFinishedTasks,
        std::int64_t numTasksToProcess,
        int extenderId
    ){
        CUDACHECK(cudaSetDevice(deviceId));
        cudaStream_t stream = cudaStreamPerThread;
        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();

        cpu::QualityScoreConversion qualityConversion{};

        rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource> extenderpool(mr);
        rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource> finishedtaskspool(mr);

        auto gpuReadExtender = std::make_unique<GpuReadExtender>(
            encodedSequencePitchInInts,
            decodedSequencePitchInBytes,
            qualityPitchInBytes,
            msaColumnPitchInElements,
            isPairedEnd,
            gpuReadStorage, 
            programOptions,
            qualityConversion,
            stream,
            &extenderpool
        );

        GpuReadExtender::TaskData finishedTasks(&finishedtaskspool, 0, encodedSequencePitchInInts, decodedSequencePitchInBytes, qualityPitchInBytes, stream);

        GpuReadExtender::RawExtendResult rawExtendResult{};

        CUDACHECK(cudaDeviceSynchronize());

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


            // numFinishedReads += splittedExtOutput.extendedReads.size() + splittedExtOutput.idsOfPartiallyExtendedReads.size() + splittedExtOutput.idsOfNotExtendedReads.size();
            // std::cerr << "update numFinishedReads to " << numFinishedReads << "\n";

            if(!programOptions.allowOutwardExtension){
                for(auto& er : splittedExtOutput.extendedReads){
                    er.removeOutwardExtension();
                }
            }

            const std::size_t numExtended = splittedExtOutput.extendedReads.size();
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

        TaskBatch* taskBatchPtr = hashedTaskBatchQueue.pop();
        while(taskBatchPtr != nullptr){
            assert(taskBatchPtr->extenderId == extenderId);

            auto& tasks = *taskBatchPtr->taskData;
            auto& anchorData = *taskBatchPtr->anchorData;
            auto& anchorHashResult = *taskBatchPtr->anchorHashResult;

            //CUDACHECK(cudaStreamWaitEvent(stream, taskBatchPtr->event, 0));

            //std::cerr << "num tasks before extend " << tasks.size() << "\n";

            const std::int64_t finishedBefore = finishedTasks.size();

            gpuReadExtender->processOneIteration(
                tasks,
                anchorData, 
                anchorHashResult, 
                finishedTasks, 
                iterationConfig,
                taskBatchPtr->stream
            );


            CUDACHECK(cudaStreamSynchronizeWrapper(taskBatchPtr->stream));

            const std::int64_t finishedAfter = finishedTasks.size();

            //std::cerr << "num tasks after extend " << tasks.size() << "\n";
            //std::cerr << "num finished tasks after extend " << finishedTasks.size() << "\n";

            totalNumFinishedTasks += (finishedAfter - finishedBefore);
            //std::cerr << "update totalNumFinishedTasks to " << totalNumFinishedTasks << "\n";


            const std::size_t limit = (programOptions.batchsize * 4);
            if(finishedTasks.size() > limit){
                output();
                //std::cerr << "num finished tasks after output " << finishedTasks.size() << "\n";
            }

            // {
            //     CUDACHECK(cudaDeviceSynchronize());
            //     rmm::mr::cuda_async_memory_resource* asyncmr = dynamic_cast<rmm::mr::cuda_async_memory_resource*>(mr);
            //     assert(asyncmr != nullptr);
            //     CUDACHECK(cudaMemPoolTrimTo(asyncmr->pool_handle(), 0));
            // }

            CUDACHECK(cudaEventRecord(taskBatchPtr->event, taskBatchPtr->stream));

            freeTaskBatchQueue.push(taskBatchPtr);

            taskBatchPtr = hashedTaskBatchQueue.pop();
        }
        auto extenderbytes = extenderpool.get_bytes_counter();
        std::cerr << "extenderbytes: peak " << extenderbytes.peak << ", total " << extenderbytes.total << ", current " << extenderbytes.value << "\n";

        auto finishedtasksbytes = finishedtaskspool.get_bytes_counter();
        std::cerr << "finishedtasksbytes: peak " << finishedtasksbytes.peak << ", total " << finishedtasksbytes.total << ", current " << finishedtasksbytes.value << "\n";

        std::cerr << "num finished tasks after extend loop " << finishedTasks.size() << "\n";
        output();
        std::cerr << "num finished tasks after output " << finishedTasks.size() << "\n";

        // for(int i = 0; i < finishedTasks.size(); i++){
        //     std::cerr << "(" << finishedTasks.pairId.element(i,cudaStreamPerThread) << "," << finishedTasks.id.element(i,cudaStreamPerThread) << "), ";
        // }
        assert(finishedTasks.size() == 0);

        
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

    if(programOptions.gpuExtenderThreadConfig.isAutomatic() || programOptions.gpuExtenderThreadConfig.numHashers > 0){
        ExtensionPipelineProducerConsumer extensionPipelineProducerConsumer(
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
    
        std::vector<read_number> pairsWhichShouldBeRepeated = extensionPipelineProducerConsumer.executeFirstPass();    
        pairsWhichShouldBeRepeated = extensionPipelineProducerConsumer.executeExtraHashingPass(pairsWhichShouldBeRepeated);
    
        submitReadyResults(
            {}, 
            {},
            std::move(pairsWhichShouldBeRepeated) //pairs which did not find mate after repetition will remain unextended
        );
    }else{
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
    
        std::vector<read_number> pairsWhichShouldBeRepeated = extensionPipeline.executeFirstPass();    
        pairsWhichShouldBeRepeated = extensionPipeline.executeExtraHashingPass(pairsWhichShouldBeRepeated);
    
        submitReadyResults(
            {}, 
            {},
            std::move(pairsWhichShouldBeRepeated) //pairs which did not find mate after repetition will remain unextended
        );
    }

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