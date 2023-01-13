
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

    struct TaskBatch{
        static int counter;

        int id;
        GpuReadExtender::TaskData* taskData{};
        GpuReadExtender::AnchorData* anchorData{};
        GpuReadExtender::AnchorHashResult* anchorHashResult{};
        CudaEvent event{cudaEventDisableTiming};
        cudaStream_t stream;

        TaskBatch(
            cudaStream_t stream_,
            int id_,
            GpuReadExtender::TaskData* taskData_,
            GpuReadExtender::AnchorData* anchorData_,
            GpuReadExtender::AnchorHashResult* anchorHashResult_
        ):
            id(id_++),
            taskData(taskData_),
            anchorData(anchorData_),
            anchorHashResult(anchorHashResult_),
            stream(stream_)
        {

        }
    };

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

    template<class ReadIdGenerator>
    std::vector<read_number> producer_consumer_impl(
        const std::vector<read_number>& pairsWhichShouldBeRepeated,
        bool extraHashing,
        bool isRepeatedIteration,
        GpuReadExtender::IterationConfig iterationConfig,
        ReadIdGenerator& readIdGenerator,
        int numReadsToProcess
    ){

        cudaStream_t stream = cudaStreamPerThread;
        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();

        const int maxNumThreads = programOptions.threads;
        const int numDevices = programOptions.deviceIds.size();

        const int numHashersPerGroup = 6;
        const int numExtendersPerGroup = 1;
        const int numTaskBatchesPerGroup = numHashersPerGroup + numExtendersPerGroup;
        const int numGroups = 2;
        std::cerr << "numGroups = " << numGroups << ", numHashersPerGroup = " << numHashersPerGroup << ", numExtendersPerGroup = " << numExtendersPerGroup << "\n";

        std::vector<std::vector<CudaStream>> taskStreamsPerGroup(numGroups);
        std::vector<std::vector<GpuReadExtender::TaskData>> taskDataVecPerGroup(numGroups);
        std::vector<std::vector<GpuReadExtender::AnchorData>> anchorDataVecPerGroup(numGroups);
        std::vector<std::vector<GpuReadExtender::AnchorHashResult>> anchorHashResultVecPerGroup(numGroups);
        std::vector<std::vector<TaskBatch>> taskBatchVecPerGroup(numGroups);

        std::vector<MultiProducerMultiConsumerQueue<TaskBatch*>> freeTaskBatchQueuePerGroup(numGroups);
        std::vector<MultiProducerMultiConsumerQueue<TaskBatch*>> hashedTaskBatchQueuePerGroup(numGroups);

        std::cerr << "queues at \n";
        for(auto& x : freeTaskBatchQueuePerGroup){
            std::cerr << &x << "\n";
        }
        for(auto& x : hashedTaskBatchQueuePerGroup){
            std::cerr << &x << "\n";
        }

        for(int groupId = 0; groupId < numGroups; groupId++){
            for(int i = 0; i < numTaskBatchesPerGroup; i++){
                taskStreamsPerGroup[groupId].emplace_back();
                taskDataVecPerGroup[groupId].emplace_back(mr, 0, encodedSequencePitchInInts, decodedSequencePitchInBytes, qualityPitchInBytes, stream);
                anchorDataVecPerGroup[groupId].emplace_back(stream, mr);
                anchorHashResultVecPerGroup[groupId].emplace_back(stream, mr);            
            }

            taskBatchVecPerGroup[groupId].reserve(numTaskBatchesPerGroup);
            for(int i = 0; i < numTaskBatchesPerGroup; i++){
                taskBatchVecPerGroup[groupId].emplace_back(
                    taskStreamsPerGroup[groupId][i], 
                    i, 
                    &taskDataVecPerGroup[groupId][i],
                    &anchorDataVecPerGroup[groupId][i],
                    &anchorHashResultVecPerGroup[groupId][i]
                );
                freeTaskBatchQueuePerGroup[groupId].push(&taskBatchVecPerGroup[groupId][i]);
            }
        }
        CUDACHECK(cudaDeviceSynchronize());


        std::atomic<std::int64_t> totalNumFinishedTasks = 0;
        const std::int64_t numTasksToProcess = (std::int64_t)numReadsToProcess * 2;

        std::cerr << "taskStreamsPerGroup.size() = " << taskStreamsPerGroup.size() << "\n";
        std::cerr << "taskDataVecPerGroup.size() = " << taskDataVecPerGroup.size() << "\n";
        std::cerr << "anchorDataVecPerGroup.size() = " << anchorDataVecPerGroup.size() << "\n";
        std::cerr << "anchorHashResultVecPerGroup.size() = " << anchorHashResultVecPerGroup.size() << "\n";
        std::cerr << "taskBatchVecPerGroup.size() = " << taskBatchVecPerGroup.size() << "\n";
        std::cerr << "freeTaskBatchQueuePerGroup.size() = " << freeTaskBatchQueuePerGroup.size() << "\n";
        std::cerr << "hashedTaskBatchQueuePerGroup.size() = " << hashedTaskBatchQueuePerGroup.size() << "\n";

        for(const auto& x : taskStreamsPerGroup){
            std::cerr << "taskStreamsPerGroup[x].size() = " << x.size() << "\n";
        }
        for(const auto& x : taskDataVecPerGroup){
            std::cerr << "taskDataVecPerGroup[x].size() = " << x.size() << "\n";
        }
        for(const auto& x : anchorDataVecPerGroup){
            std::cerr << "anchorDataVecPerGroup[x].size() = " << x.size() << "\n";
        }
        for(const auto& x : anchorHashResultVecPerGroup){
            std::cerr << "anchorHashResultVecPerGroup[x].size() = " << x.size() << "\n";
        }
        for(const auto& x : taskBatchVecPerGroup){
            std::cerr << "taskBatchVecPerGroup[x].size() = " << x.size() << "\n";
        }


        std::vector<std::vector<std::future<void>>> hasherFuturesPerGroup(numGroups);
        std::vector<std::vector<std::future<std::vector<read_number>>>> extenderfuturesPerGroup(numGroups);

        for(int groupId = 0; groupId < numGroups; groupId++){
            for(int t = 0; t < numHashersPerGroup; t++){
                //std::cerr << "launch hasher " << &freeTaskBatchQueuePerGroup[groupId] << " " << &hashedTaskBatchQueuePerGroup[groupId] << "\n";
                hasherFuturesPerGroup[groupId].emplace_back(
                    std::async(
                        std::launch::async,
                        [&](int groupId){
                            hasherThreadFunc(
                                &readIdGenerator,
                                &freeTaskBatchQueuePerGroup[groupId],
                                &hashedTaskBatchQueuePerGroup[groupId],
                                isRepeatedIteration,
                                extraHashing,
                                iterationConfig,
                                totalNumFinishedTasks,
                                numTasksToProcess
                            );
                        },
                        groupId
                    )
                );
            }

            for(int t = 0; t < numExtendersPerGroup; t++){
                std::cerr << "freeTaskBatchQueuePerGroup.data() = " << freeTaskBatchQueuePerGroup.data() << "\n";
                std::cerr << "launch extender " << &freeTaskBatchQueuePerGroup[groupId] << " " << &hashedTaskBatchQueuePerGroup[groupId] << "\n";
                extenderfuturesPerGroup[groupId].emplace_back(
                    std::async(
                        std::launch::async,
                        [&](int groupId){
                            std::cerr << "freeTaskBatchQueuePerGroup.data() = " << freeTaskBatchQueuePerGroup.data() << "\n";
                            std::cerr << "launch extender lambda " << &freeTaskBatchQueuePerGroup[groupId] << " " << &hashedTaskBatchQueuePerGroup[groupId] << "\n";

                            return extenderThreadFunc(
                                &readIdGenerator,
                                &freeTaskBatchQueuePerGroup[groupId],
                                &hashedTaskBatchQueuePerGroup[groupId],
                                isRepeatedIteration,
                                extraHashing,
                                iterationConfig,
                                totalNumFinishedTasks,
                                numTasksToProcess
                            );
                        },
                        groupId
                    )
                );
            }
        }


        std::vector<read_number> pairsWhichShouldBeRepeatedTmp{};

        for(int groupId = 0; groupId < numGroups; groupId++){
            //wait for hashers
            for(auto& f : hasherFuturesPerGroup[groupId]){
                f.wait();
            }

            //hashers are done. notify extenders
            for(int i = 0; i < numExtendersPerGroup; i++){
                hashedTaskBatchQueuePerGroup[groupId].push(nullptr);
            }

            //wait for extenders
            for(auto& f : extenderfuturesPerGroup[groupId]){
                auto vec = f.get();
                pairsWhichShouldBeRepeatedTmp.insert(pairsWhichShouldBeRepeatedTmp.end(), vec.begin(), vec.end());
            }
        }

        
        CUDACHECK(cudaDeviceSynchronize());

        std::sort(pairsWhichShouldBeRepeatedTmp.begin(), pairsWhichShouldBeRepeatedTmp.end());
        return pairsWhichShouldBeRepeatedTmp;
    }

    //return list of read ids of pairs that should be processed again
    std::vector<read_number> executeFirstPassProducerConsumer(){
        constexpr bool extraHashing = false;
        constexpr bool isRepeatedIteration = false;

        cudaStream_t stream = cudaStreamPerThread;
        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();

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



        const int numHashersPerGroup = 6;
        const int numExtendersPerGroup = 1;
        const int numTaskBatchesPerGroup = numHashersPerGroup + numExtendersPerGroup;
        const int numGroups = 2;
        std::cerr << "numGroups = " << numGroups << ", numHashersPerGroup = " << numHashersPerGroup << ", numExtendersPerGroup = " << numExtendersPerGroup << "\n";

        std::vector<std::vector<CudaStream>> taskStreamsPerGroup(numGroups);
        std::vector<std::vector<GpuReadExtender::TaskData>> taskDataVecPerGroup(numGroups);
        std::vector<std::vector<GpuReadExtender::AnchorData>> anchorDataVecPerGroup(numGroups);
        std::vector<std::vector<GpuReadExtender::AnchorHashResult>> anchorHashResultVecPerGroup(numGroups);
        std::vector<std::vector<TaskBatch>> taskBatchVecPerGroup(numGroups);

        std::vector<MultiProducerMultiConsumerQueue<TaskBatch*>> freeTaskBatchQueuePerGroup(numGroups);
        std::vector<MultiProducerMultiConsumerQueue<TaskBatch*>> hashedTaskBatchQueuePerGroup(numGroups);

        std::cerr << "queues at \n";
        for(auto& x : freeTaskBatchQueuePerGroup){
            std::cerr << &x << "\n";
        }
        for(auto& x : hashedTaskBatchQueuePerGroup){
            std::cerr << &x << "\n";
        }

        for(int groupId = 0; groupId < numGroups; groupId++){
            for(int i = 0; i < numTaskBatchesPerGroup; i++){
                taskStreamsPerGroup[groupId].emplace_back();
                taskDataVecPerGroup[groupId].emplace_back(mr, 0, encodedSequencePitchInInts, decodedSequencePitchInBytes, qualityPitchInBytes, stream);
                anchorDataVecPerGroup[groupId].emplace_back(stream, mr);
                anchorHashResultVecPerGroup[groupId].emplace_back(stream, mr);            
            }

            taskBatchVecPerGroup[groupId].reserve(numTaskBatchesPerGroup);
            for(int i = 0; i < numTaskBatchesPerGroup; i++){
                taskBatchVecPerGroup[groupId].emplace_back(
                    taskStreamsPerGroup[groupId][i], 
                    i, 
                    &taskDataVecPerGroup[groupId][i],
                    &anchorDataVecPerGroup[groupId][i],
                    &anchorHashResultVecPerGroup[groupId][i]
                );
                freeTaskBatchQueuePerGroup[groupId].push(&taskBatchVecPerGroup[groupId][i]);
            }
        }
        CUDACHECK(cudaDeviceSynchronize());


        std::atomic<std::int64_t> totalNumFinishedTasks = 0;
        const std::int64_t numTasksToProcess = (std::int64_t)numReadsToProcess * 2;

        std::cerr << "taskStreamsPerGroup.size() = " << taskStreamsPerGroup.size() << "\n";
        std::cerr << "taskDataVecPerGroup.size() = " << taskDataVecPerGroup.size() << "\n";
        std::cerr << "anchorDataVecPerGroup.size() = " << anchorDataVecPerGroup.size() << "\n";
        std::cerr << "anchorHashResultVecPerGroup.size() = " << anchorHashResultVecPerGroup.size() << "\n";
        std::cerr << "taskBatchVecPerGroup.size() = " << taskBatchVecPerGroup.size() << "\n";
        std::cerr << "freeTaskBatchQueuePerGroup.size() = " << freeTaskBatchQueuePerGroup.size() << "\n";
        std::cerr << "hashedTaskBatchQueuePerGroup.size() = " << hashedTaskBatchQueuePerGroup.size() << "\n";

        for(const auto& x : taskStreamsPerGroup){
            std::cerr << "taskStreamsPerGroup[x].size() = " << x.size() << "\n";
        }
        for(const auto& x : taskDataVecPerGroup){
            std::cerr << "taskDataVecPerGroup[x].size() = " << x.size() << "\n";
        }
        for(const auto& x : anchorDataVecPerGroup){
            std::cerr << "anchorDataVecPerGroup[x].size() = " << x.size() << "\n";
        }
        for(const auto& x : anchorHashResultVecPerGroup){
            std::cerr << "anchorHashResultVecPerGroup[x].size() = " << x.size() << "\n";
        }
        for(const auto& x : taskBatchVecPerGroup){
            std::cerr << "taskBatchVecPerGroup[x].size() = " << x.size() << "\n";
        }


        std::vector<std::vector<std::future<void>>> hasherFuturesPerGroup(numGroups);
        std::vector<std::vector<std::future<std::vector<read_number>>>> extenderfuturesPerGroup(numGroups);

        for(int groupId = 0; groupId < numGroups; groupId++){
            for(int t = 0; t < numHashersPerGroup; t++){
                //std::cerr << "launch hasher " << &freeTaskBatchQueuePerGroup[groupId] << " " << &hashedTaskBatchQueuePerGroup[groupId] << "\n";
                hasherFuturesPerGroup[groupId].emplace_back(
                    std::async(
                        std::launch::async,
                        [&](int groupId){
                            hasherThreadFunc(
                                &readIdGenerator,
                                &freeTaskBatchQueuePerGroup[groupId],
                                &hashedTaskBatchQueuePerGroup[groupId],
                                isRepeatedIteration,
                                extraHashing,
                                iterationConfig,
                                totalNumFinishedTasks,
                                numTasksToProcess
                            );
                        },
                        groupId
                    )
                );
            }

            for(int t = 0; t < numExtendersPerGroup; t++){
                std::cerr << "freeTaskBatchQueuePerGroup.data() = " << freeTaskBatchQueuePerGroup.data() << "\n";
                std::cerr << "launch extender " << &freeTaskBatchQueuePerGroup[groupId] << " " << &hashedTaskBatchQueuePerGroup[groupId] << "\n";
                extenderfuturesPerGroup[groupId].emplace_back(
                    std::async(
                        std::launch::async,
                        [&](int groupId){
                            std::cerr << "freeTaskBatchQueuePerGroup.data() = " << freeTaskBatchQueuePerGroup.data() << "\n";
                            std::cerr << "launch extender lambda " << &freeTaskBatchQueuePerGroup[groupId] << " " << &hashedTaskBatchQueuePerGroup[groupId] << "\n";

                            return extenderThreadFunc(
                                &readIdGenerator,
                                &freeTaskBatchQueuePerGroup[groupId],
                                &hashedTaskBatchQueuePerGroup[groupId],
                                isRepeatedIteration,
                                extraHashing,
                                iterationConfig,
                                totalNumFinishedTasks,
                                numTasksToProcess
                            );
                        },
                        groupId
                    )
                );
            }
        }


        std::vector<read_number> pairsWhichShouldBeRepeated{};

        for(int groupId = 0; groupId < numGroups; groupId++){
            //wait for hashers
            for(auto& f : hasherFuturesPerGroup[groupId]){
                f.wait();
            }

            //hashers are done. notify extenders
            for(int i = 0; i < numExtendersPerGroup; i++){
                hashedTaskBatchQueuePerGroup[groupId].push(nullptr);
            }

            //wait for extenders
            for(auto& f : extenderfuturesPerGroup[groupId]){
                auto vec = f.get();
                pairsWhichShouldBeRepeated.insert(pairsWhichShouldBeRepeated.end(), vec.begin(), vec.end());
            }
        }

        
        CUDACHECK(cudaDeviceSynchronize());

        std::sort(pairsWhichShouldBeRepeated.begin(), pairsWhichShouldBeRepeated.end());
        return pairsWhichShouldBeRepeated;
    }


    template<class ReadIdGenerator, class UnhashedQueue, class HashedQueue>
    void hasherThreadFunc(
        ReadIdGenerator* readIdGenerator,
        UnhashedQueue* freeTaskBatchQueue,
        HashedQueue* hashedTaskBatchQueue,
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

        //std::cerr << "hasherThread queues: " << freeTaskBatchQueue << " " << hashedTaskBatchQueue << "\n";


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
            //std::cerr << totalNumFinishedTasks << " " << numTasksToProcess << "\n";
            TaskBatch* const taskBatchPtr = freeTaskBatchQueue->pop();
            assert(taskBatchPtr != nullptr);

            auto& tasks = *taskBatchPtr->taskData;
            auto& anchorData = *taskBatchPtr->anchorData;
            auto& anchorHashResult = *taskBatchPtr->anchorHashResult;

            //CUDACHECK(cudaStreamWaitEvent(stream, taskBatchPtr->event, 0));
            //std::cerr << "num tasks before init " << tasks.size() << "\n";
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

                hashedTaskBatchQueue->push(taskBatchPtr);
            }else{
                //seenEmptyTaskBatchesAfterInit.insert(taskBatchPtr->id);
                freeTaskBatchQueue->push(taskBatchPtr);
            }
        }

        gpuReadStorage.destroyHandle(readStorageHandle);
    };

    template<class ReadIdGenerator, class UnhashedQueue, class HashedQueue>
    std::vector<read_number> extenderThreadFunc(
        ReadIdGenerator* /*readIdGenerator*/,
        UnhashedQueue* freeTaskBatchQueue,
        HashedQueue* hashedTaskBatchQueue,
        bool isRepeatedIteration,
        bool /*extraHashing*/,
        GpuReadExtender::IterationConfig iterationConfig,
        std::atomic<std::int64_t>& totalNumFinishedTasks,
        std::int64_t numTasksToProcess
    ){
        CUDACHECK(cudaSetDevice(deviceId));
        cudaStream_t stream = cudaStreamPerThread;
        rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();

        cpu::QualityScoreConversion qualityConversion{};

        std::cerr << "extenderThreadFunc queues: " << freeTaskBatchQueue << " " << hashedTaskBatchQueue << "\n";

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
            programOptions.insertSize,
            (programOptions.fixedStddev == 0 ? programOptions.insertSizeStddev : programOptions.fixedStddev),
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

        TaskBatch* taskBatchPtr = hashedTaskBatchQueue->pop();
        while(taskBatchPtr != nullptr){

            auto& tasks = *taskBatchPtr->taskData;
            auto& anchorData = *taskBatchPtr->anchorData;
            auto& anchorHashResult = *taskBatchPtr->anchorHashResult;
            //auto& finishedTasks = *taskBatchPtr->finishedTaskData;

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


            const std::size_t limit = (programOptions.batchsize * 4) / 2;
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

            freeTaskBatchQueue->push(taskBatchPtr);

            taskBatchPtr = hashedTaskBatchQueue->pop();
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


    std::vector<read_number> runExtraHashingPass(const std::vector<read_number>& pairsWhichShouldBeRepeated){
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
            programOptions.insertSize,
            (programOptions.fixedStddev == 0 ? programOptions.insertSizeStddev : programOptions.fixedStddev),
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

    //std::vector<read_number> pairsWhichShouldBeRepeated = extensionPipeline.executeFirstPass();
    std::vector<read_number> pairsWhichShouldBeRepeated = extensionPipeline.executeFirstPassProducerConsumer();

    pairsWhichShouldBeRepeated = extensionPipeline.runExtraHashingPass(pairsWhichShouldBeRepeated);

    submitReadyResults(
        {}, 
        {},
        std::move(pairsWhichShouldBeRepeated) //pairs which did not find mate after repetition will remain unextended
    );

   

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