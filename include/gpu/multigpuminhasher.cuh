#ifdef CARE_HAS_WARPCORE

#ifndef CARE_MULTI_GPU_MINHASHER_CUH
#define CARE_MULTI_GPU_MINHASHER_CUH

#include <config.hpp>


#include <gpu/gpureadstorage.cuh>
#include <gpu/cuda_unique.cuh>
#include <gpu/singlegpuminhasher.cuh>
#include <gpu/kernels.hpp>
#include <gpu/gpuminhasher.cuh>
#include <gpu/cudaerrorcheck.cuh>
#include <gpu/cubwrappers.cuh>

#include <options.hpp>
#include <util.hpp>
#include <hpc_helpers.cuh>
#include <filehelpers.hpp>

#include <sequencehelpers.hpp>
#include <memorymanagement.hpp>
#include <threadpool.hpp>
#include <sharedmutex.hpp>

#include <cub/cub.cuh>

#include <vector>
#include <memory>
#include <limits>
#include <string>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cassert>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/device_uvector.hpp>
#include <gpu/rmm_utilities.cuh>
namespace care{
namespace gpu{

    namespace multigpuminhasherkernels{
        template<int blocksize, int itemsPerThread>
        __global__
        void aggregatePartitionResultsSingleBlockKernel(
            const int* __restrict__ numResultsPerSequencePerPartition,
            int* __restrict__ numResultsPerSequence,
            int numSequences,
            int numPartitions,
            int* __restrict__ maxNumResultsPerSequence,
            int* __restrict__ offsets
        ){
            assert(gridDim.x * gridDim.y * gridDim.z == 1);

            struct BlockPrefixCallbackOp{
                // Running prefix
                int running_total;
                // Constructor
                __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
                // Callback operator to be entered by the first warp of threads in the block.
                // Thread-0 is responsible for returning a value for seeding the block-wide scan.
                __device__ int operator()(int block_aggregate)
                {
                    int old_prefix = running_total;
                    running_total += block_aggregate;
                    return old_prefix;
                }
            };

            using BlockReduce = cub::BlockReduce<int, blocksize>;
            using BlockScan = cub::BlockScan<int, blocksize>;

            __shared__ typename BlockReduce::TempStorage smem_reduce;
            __shared__ typename BlockScan::TempStorage smem_scan;

            constexpr int itemsPerIteration = blocksize * itemsPerThread;

            const int numIterations = SDIV(numSequences, itemsPerIteration);
            int myMax = 0;
            BlockPrefixCallbackOp prefix_op(0);

            //full iterations
            for(int iteration = 0; iteration < numIterations; iteration++){
                const int s = iteration * itemsPerIteration + threadIdx.x;
                int sum = 0;

                if(s < numSequences){
                    for(int r = 0; r < numPartitions; r++){
                        sum += numResultsPerSequencePerPartition[r * numSequences + s];
                    }

                    numResultsPerSequence[s] = sum;
                    myMax = max(myMax, sum);
                }

                BlockScan(smem_scan).ExclusiveSum(sum, sum, prefix_op);

                if(s < numSequences){
                    offsets[s] = sum;
                }
            }

            myMax = BlockReduce(smem_reduce).Reduce(myMax, cub::Max{});

            if(threadIdx.x == 0){
                *maxNumResultsPerSequence = myMax;
                offsets[numSequences] = prefix_op.running_total;
            }
        };

        template<class Dummy = void>
        __global__
        void aggregateNumValuesPartitionResultsKernel(
            const int* __restrict__ numResultsPerSequencePerPartition,
            int* __restrict__ numResultsPerSequence,
            int numSequences,
            int numPartitions
        ){
            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
            const int stride = blockDim.x * gridDim.x;

            for(int s = tid; s < numSequences; s += stride){
                int sum = 0;
                
                for(int r = 0; r < numPartitions; r++){
                    sum += numResultsPerSequencePerPartition[r * numSequences + s];
                }

                numResultsPerSequence[s] = sum;
            }

        };
    
        template<class Dummy = void>
        __global__
        void copyToInterleavedKernel(
            const read_number* __restrict__ inputdata,
            const int* __restrict__ inputoffsets,
            const int* __restrict__ inputsegmentsizes,
            const int* __restrict__ inputnumpergpuPS,
            const int* __restrict__ outputbeginoffsets,
            read_number* __restrict__ outputdata,
            int numSequences,
            int partitions
        ){                
            for(int i = blockIdx.x; i < numSequences; i += gridDim.x){
                const int beginoffset = outputbeginoffsets[i];

                int runningOffset = 0;

                for(int r = 0; r < partitions; r++){
                    const int segmentsize = inputsegmentsizes[r * numSequences + i];
                    const int inputOffset = inputoffsets[r * (numSequences+1) + i];
                    const int gpuOffset = inputnumpergpuPS[r];

                    const read_number* myinput = inputdata + gpuOffset + inputOffset;

                    for(int k = threadIdx.x; k < segmentsize; k += blockDim.x){
                        outputdata[beginoffset + runningOffset + k] = myinput[k];
                    }

                    runningOffset += segmentsize;
                }
            }
        }

        template<class Dummy = void>
        __global__ 
        void copyResultsToDestinationKernel(
            const read_number* input,
            int* inputoffsets,
            read_number* output,
            const int* outputoffsets,
            const int* d_numValuesPerSequence,
            int numSequences
        ){
            for(int s = blockIdx.x; s < numSequences; s += gridDim.x){
                const int inputoffset = inputoffsets[s];
                const int outputoffset = outputoffsets[s];
                const int segmentsize = d_numValuesPerSequence[s];

                for(int i = threadIdx.x; i < segmentsize; i += blockDim.x){
                    output[outputoffset + i] = input[inputoffset + i];
                }

                __syncthreads();
                //update d_offsets within this kernel to avoid additional api call
                if(threadIdx.x == 0){
                    inputoffsets[s] = outputoffset;
                }
            }

            //last entry of offsets (total number) is not used for copying. no need for sync
            if(blockIdx.x == 0 && threadIdx.x == 0){
                inputoffsets[numSequences] = outputoffsets[numSequences];
            }

        }

    }


    class MultiGpuMinhasher : public GpuMinhasher{
    public:
        using Key = GpuMinhasher::Key;
    private:
        using DeviceSwitcher = cub::SwitchDevice;

        template<class T>
        using HostBuffer = helpers::SimpleAllocationPinnedHost<T, 5>;
        template<class T>
        using DeviceBuffer = helpers::SimpleAllocationDevice<T, 5>;
        //using DeviceBuffer = helpers::SimpleAllocationPinnedHost<T, 5>;

        struct QueryData{
            enum class Stage{
                None,
                NumValues,
                Retrieve
            };

            struct RemotePersistentData{
                RemotePersistentData(cudaStream_t stream, rmm::mr::device_memory_resource* mr)
                : d_persistent(0, stream, mr),
                    d_numResultsPerSequence(0, stream, mr)
                {}
                
                rmm::device_uvector<read_number> d_persistent;
                rmm::device_uvector<int> d_numResultsPerSequence;
            };

            struct DetermineNumValuesTempStorage{
                DetermineNumValuesTempStorage(
                    cudaStream_t stream, //stream of caller
                    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource() //mr of caller
                ) : caller_d_numResultsPerSequence(0, stream, mr){}

                std::vector<rmm::device_uvector<int>> remote_d_input_lengths;
                std::vector<rmm::device_uvector<unsigned int>> remote_d_input_sequences;
                std::vector<rmm::device_uvector<int>> remote_d_numResultsPerSequence;
                std::vector<rmm::device_uvector<int>> remote_d_offsets;

                rmm::device_uvector<int> caller_d_numResultsPerSequence;
            };

            struct RetrieveValuesTempStorage{
                RetrieveValuesTempStorage(
                    cudaStream_t stream, //stream of caller
                    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource() //mr of caller
                ) : caller_d_results(0, stream, mr),
                    caller_d_numResultsPerSequence(0, stream, mr),
                    caller_d_offsets(0, stream, mr),
                    caller_d_offsets_tmp(0, stream, mr)
                {}

                std::vector<rmm::device_uvector<read_number>> remote_d_readIds;
                std::vector<rmm::device_uvector<read_number>> remote_d_results;
                std::vector<rmm::device_uvector<int>> remote_d_offsets;

                rmm::device_uvector<read_number> caller_d_results;
                rmm::device_uvector<int> caller_d_numResultsPerSequence;
                rmm::device_uvector<int> caller_d_offsets;
                rmm::device_uvector<int> caller_d_offsets_tmp;
            };

            void allocate(
                DetermineNumValuesTempStorage& tempstorage,
                int numSequences,
                std::size_t encodedSequencePitchInInts,
                cudaStream_t stream //stream of caller
            ) const {
                const int numIds = deviceIds.size();

                tempstorage.caller_d_numResultsPerSequence.resize(numSequences * numIds, stream);

                for(int d = 0; d < numIds; d++){
                    cub::SwitchDevice sd(deviceIds[d]);
                    cudaStream_t remoteStream = streams[d];

                    tempstorage.remote_d_input_lengths[d].resize(numSequences, remoteStream);
                    tempstorage.remote_d_input_sequences[d].resize(encodedSequencePitchInInts * numSequences, remoteStream);
                    tempstorage.remote_d_numResultsPerSequence[d].resize(numSequences, remoteStream);
                    tempstorage.remote_d_offsets[d].resize(numSequences + 1, remoteStream);
                }
            }

            void deallocate(
                DetermineNumValuesTempStorage& tempstorage,
                cudaStream_t stream //stream of caller
            ) const {
                const int numIds = deviceIds.size();

                ::destroy(tempstorage.caller_d_numResultsPerSequence, stream);

                for(int d = 0; d < numIds; d++){
                    cub::SwitchDevice sd(deviceIds[d]);
                    cudaStream_t remoteStream = streams[d];

                    ::destroy(tempstorage.remote_d_input_lengths[d], remoteStream);
                    ::destroy(tempstorage.remote_d_input_sequences[d], remoteStream);
                    ::destroy(tempstorage.remote_d_numResultsPerSequence[d], remoteStream);
                    ::destroy(tempstorage.remote_d_offsets[d], remoteStream);
                }
            }

            void allocate(
                RetrieveValuesTempStorage& tempstorage,
                int numSequences,
                int totalNumValues,
                cudaStream_t stream //stream of caller
            ) const {
                const int numIds = deviceIds.size();

                tempstorage.caller_d_results.resize(totalNumValues, stream);
                tempstorage.caller_d_numResultsPerSequence.resize(numSequences * numIds, stream);
                tempstorage.caller_d_offsets.resize((numSequences + 1) * numIds, stream);
                tempstorage.caller_d_offsets_tmp.resize(numSequences * numIds, stream);

                for(int d = 0; d < numIds; d++){
                    cub::SwitchDevice sd(deviceIds[d]);
                    cudaStream_t remoteStream = streams[d];

                    tempstorage.remote_d_readIds[d].resize(numSequences, remoteStream);
                    //will allocate exact size in retrieveValuesOnEachGpu
                    tempstorage.remote_d_results[d].resize(0, remoteStream);
                    tempstorage.remote_d_offsets[d].resize(numSequences + 1, remoteStream);
                }
            }

            void deallocate(
                RetrieveValuesTempStorage& tempstorage,
                cudaStream_t stream //stream of caller
            ) const {
                const int numIds = deviceIds.size();

                ::destroy(tempstorage.caller_d_results, stream);
                ::destroy(tempstorage.caller_d_numResultsPerSequence, stream);
                ::destroy(tempstorage.caller_d_offsets, stream);
                ::destroy(tempstorage.caller_d_offsets_tmp, stream);

                for(int d = 0; d < numIds; d++){
                    cub::SwitchDevice sd(deviceIds[d]);
                    cudaStream_t remoteStream = streams[d];

                    ::destroy(tempstorage.remote_d_readIds[d], remoteStream);
                    ::destroy(tempstorage.remote_d_results[d], remoteStream);
                    ::destroy(tempstorage.remote_d_offsets[d], remoteStream);
                }
            }

            std::unique_ptr<DetermineNumValuesTempStorage> createTempStorageDetermineNumValues(
                int numSequences,
                std::size_t encodedSequencePitchInInts,
                cudaStream_t stream, //stream of caller
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource() //mr of caller
            ) const {
                auto result = std::make_unique<DetermineNumValuesTempStorage>(stream, mr);

                const int numIds = deviceIds.size();

                for(int d = 0; d < numIds; d++){
                    cub::SwitchDevice sd(deviceIds[d]);
                    cudaStream_t remoteStream = streams[d];
                    auto* remoteMr = rmm::mr::get_current_device_resource();

                    result->remote_d_input_lengths.emplace_back(0, remoteStream, remoteMr);
                    result->remote_d_input_sequences.emplace_back(0, remoteStream, remoteMr);
                    result->remote_d_numResultsPerSequence.emplace_back(0, remoteStream, remoteMr);
                    result->remote_d_offsets.emplace_back(0, remoteStream, remoteMr);
                }

                allocate(
                    *result,
                    numSequences, 
                    encodedSequencePitchInInts,
                    stream
                );

                return result;
            }

            std::unique_ptr<RetrieveValuesTempStorage> createTempStorageRetrieveValues(
                int numSequences,
                int totalNumValues,
                cudaStream_t stream, //stream of caller
                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource() //mr of caller
            ) const {
                auto result = std::make_unique<RetrieveValuesTempStorage>(stream, mr);

                const int numIds = deviceIds.size();

                for(int d = 0; d < numIds; d++){
                    cub::SwitchDevice sd(deviceIds[d]);
                    cudaStream_t remoteStream = streams[d];
                    auto* remoteMr = rmm::mr::get_current_device_resource();

                    result->remote_d_readIds.emplace_back(0, remoteStream, remoteMr);
                    result->remote_d_results.emplace_back(0, remoteStream, remoteMr);
                    result->remote_d_offsets.emplace_back(0, remoteStream, remoteMr);
                }

                allocate(
                    *result,
                    numSequences, 
                    totalNumValues,
                    stream
                );

                return result;
            }

            ~QueryData(){
                if(remotePersistentDataPerGpu.size() > 0){
                    assert(remotePersistentDataPerGpu.size() == deviceIds.size());

                    const int numIds = deviceIds.size();

                    for(int d = 0; d < numIds; d++){
                        cub::SwitchDevice sd(deviceIds[d]);

                        ::destroy(remotePersistentDataPerGpu[d].d_persistent, (cudaStream_t)0);
                        ::destroy(remotePersistentDataPerGpu[d].d_numResultsPerSequence, (cudaStream_t)0);
                    }
                }

                if(numValuesTempStorage){
                    deallocate(*numValuesTempStorage, (cudaStream_t)0);
                }
                if(retrieveTempStorage){
                    deallocate(*retrieveTempStorage, (cudaStream_t)0);
                }
            }

            int numSequences{};
            int callerDeviceId{};
            std::size_t encodedSequencePitchInInts{};
            const read_number* d_readIds{};
            const unsigned int* d_sequenceData2Bit{};
            const int* d_sequenceLengths{};

            int totalNumValues{};
            Stage previousStage = Stage::None;
            rmm::mr::device_memory_resource* mr;

            CudaEvent callerEvent{cudaEventDisableTiming};

            std::vector<RemotePersistentData> remotePersistentDataPerGpu{};
            std::unique_ptr<DetermineNumValuesTempStorage> numValuesTempStorage;
            std::unique_ptr<RetrieveValuesTempStorage> retrieveTempStorage;

            HostBuffer<int> pinnedData{};

            std::vector<CudaStream> streams{};
            std::vector<CudaEvent> events{};

            std::vector<int> deviceIds{};

            std::vector<std::unique_ptr<MinhasherHandle>> singlegpuMinhasherHandles;

            MemoryUsage getMemoryInfo() const{
                MemoryUsage mem{};

                return mem;
            }
        };

    public: 

        MultiGpuMinhasher(int maxNumKeys_, int maxValuesPerKey, int k, std::vector<int> deviceIds_)
            : maxNumKeys(maxNumKeys_), kmerSize(k), resultsPerMapThreshold(maxValuesPerKey), deviceIds(deviceIds_)
        {
            
        }

        int constructFromReadStorage(
            const RuntimeOptions &runtimeOptions,
            std::uint64_t nReads,
            const GpuReadStorage& gpuReadStorage,
            int upperBoundSequenceLength,
            int maxNumHashfunctions,
            int hashFunctionOffset = 0
        ) {
            
            const int numDevices = deviceIds.size();
            sgpuMinhashers.clear();
            usableDeviceIds.clear();

            helpers::CpuTimer cpuTimer{"MultiGpuMinhasher construction"};
            cpuTimer.start();

            int currentHashFunctionOffset = hashFunctionOffset;

            vec_h_currentHashFunctionNumbers.resize(numDevices);

            int remainingNumHashfunctions = maxNumHashfunctions;

            for(int d = 0; d < numDevices; d++){
                DeviceSwitcher ds(deviceIds[d]);

                if(remainingNumHashfunctions > 0){

                    nvtx::push_range("Construct SingleGpuMinhasher", 4);

                    auto mh = std::make_unique<SingleGpuMinhasher>(nReads, resultsPerMapThreshold, kmerSize);
                    const int createdTables =  mh->constructFromReadStorage(
                        runtimeOptions, 
                        nReads, 
                        gpuReadStorage, 
                        upperBoundSequenceLength, 
                        remainingNumHashfunctions, 
                        //std::min(remainingNumHashfunctions, 24), //debugging
                        currentHashFunctionOffset
                    );

                    if(createdTables > 0){
                        vec_h_currentHashFunctionNumbers.push_back({});
                        sgpuMinhashers.emplace_back(std::move(mh));

                        auto& buffer = vec_h_currentHashFunctionNumbers[d];
                        buffer.resize(createdTables);
                        std::iota(buffer.begin(), buffer.end(), currentHashFunctionOffset);


                        currentHashFunctionOffset += createdTables;
                        remainingNumHashfunctions -= createdTables;

                        usableDeviceIds.emplace_back(deviceIds[d]);

                        std::cerr << "Placed " << createdTables << " tables on gpu with id " << deviceIds[d] << ". (id at position " << d << " in list)\n";
                    }

                    nvtx::pop_range();
                }                
            }

            for(int d = 0; d < numDevices; d++){
                DeviceSwitcher ds(deviceIds[d]);
                CUDACHECK(cudaDeviceSynchronize());
            }

            cpuTimer.stop();
            cpuTimer.print();

            const int numberOfAvailableHashFunctions = maxNumHashfunctions - remainingNumHashfunctions;

            return numberOfAvailableHashFunctions; 
        }

        bool tryReplication(){
            if(sgpuMinhashers.size() == 1 && usableDeviceIds.size() < deviceIds.size()){
                //all hashtables fit into one gpu. try to replace the hash tables on all gpus

                std::vector<std::unique_ptr<SingleGpuMinhasher>> replicas{};
                bool ok = false;
                try{
                    nvtx::push_range("replicate single gpu minhasher", 0);

                    for(std::size_t i = 1; i < deviceIds.size(); i++){
                        const int targetDeviceId = deviceIds[i];
                        helpers::CpuTimer rtimer("make singlegpu minhasher replica");
                        replicas.emplace_back(sgpuMinhashers[0]->makeCopy(targetDeviceId));
                        rtimer.print();
                    }
                    ok = std::all_of(replicas.begin(), replicas.end(), [](const auto& uniqueptr){ return bool(uniqueptr); });

                    nvtx::pop_range();
                }catch(...){
                    cudaGetLastError();
                    std::cerr << "error replicating single gpu minhasher. Skipping.\n";
                }
                if(ok){                    
                    sgpuMinhashers.insert(sgpuMinhashers.end(), std::make_move_iterator(replicas.begin()), std::make_move_iterator(replicas.end()));

                    HostBuffer<int> h_currentHashFunctionNumbers(vec_h_currentHashFunctionNumbers[0].size());
                    std::copy(vec_h_currentHashFunctionNumbers[0].begin(), vec_h_currentHashFunctionNumbers[0].end(), h_currentHashFunctionNumbers.begin());
                    vec_h_currentHashFunctionNumbers.push_back(std::move(h_currentHashFunctionNumbers));

                    usableDeviceIds = deviceIds;

                    isReplicatedSingleGpu = true;
                }

                return ok;
            }else{
                return false;
            }
        }

        MinhasherHandle makeMinhasherHandle() const override{
            auto ptr = std::make_unique<QueryData>();

            const int numMinhashers = sgpuMinhashers.size();

            ptr->streams.resize(numMinhashers);
            ptr->events.resize(numMinhashers);
            ptr->deviceIds = usableDeviceIds;

            for(int i = 0; i < numMinhashers; i++){
                DeviceSwitcher ds(sgpuMinhashers[i]->getDeviceId());

                ptr->streams[i] = std::move(CudaStream{});
                ptr->events[i] = std::move(CudaEvent{cudaEventDisableTiming});

                ptr->singlegpuMinhasherHandles.emplace_back(std::make_unique<MinhasherHandle>(sgpuMinhashers[i]->makeMinhasherHandle()));
            }

            ptr->numValuesTempStorage = ptr->createTempStorageDetermineNumValues(
                0,
                0,
                cudaStreamPerThread
            );

            ptr->retrieveTempStorage = ptr->createTempStorageRetrieveValues(
                0,
                0,
                cudaStreamPerThread
            );

            CUDACHECK(cudaStreamSynchronize(cudaStreamPerThread));

            std::unique_lock<SharedMutex> lock(sharedmutex);
            const int handleid = counter++;
            MinhasherHandle h = constructHandle(handleid);

            tempdataVector.emplace_back(std::move(ptr));

            return h;
        }

        void destroyHandle(MinhasherHandle& handle) const override{            

            std::unique_lock<SharedMutex> lock(sharedmutex);

            const int id = handle.getId();
            assert(id < int(tempdataVector.size()));

            const int numMinhashers = sgpuMinhashers.size();
            for(int i = 0; i < numMinhashers; i++){
                sgpuMinhashers[i]->destroyHandle(*tempdataVector[id]->singlegpuMinhasherHandles[i]);
            }
            
            {
                cub::SwitchDevice sd(tempdataVector[id]->callerDeviceId);
                tempdataVector[id] = nullptr;
            }
            handle = constructHandle(std::numeric_limits<int>::max());
        }

        void determineNumValues(
            MinhasherHandle& queryHandle,
            const unsigned int* d_sequenceData2Bit,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int* d_numValuesPerSequence,
            int& totalNumValues,
            cudaStream_t stream,
            rmm::mr::device_memory_resource* mr //resource for current device
        ) const override {
            
            if(numSequences == 0){
                return;
            }

            int currentDeviceId = 0;
            CUDACHECK(cudaGetDevice(&currentDeviceId));

            auto it = usableDeviceIds.end();
            if(isReplicatedSingleGpu || (usableDeviceIds.size() == 1)){
                it = std::find(usableDeviceIds.begin(), usableDeviceIds.end(), currentDeviceId);
            }

            if(it != usableDeviceIds.end()){
                int index = std::distance(usableDeviceIds.begin(), it);

                determineNumValues_singlegpuimpl(
                    index,
                    queryHandle,
                    d_sequenceData2Bit,
                    encodedSequencePitchInInts,
                    d_sequenceLengths,
                    numSequences,
                    d_numValuesPerSequence,
                    totalNumValues,
                    stream,
                    mr
                );
                
            }else{

                int currentDeviceId = 0;
                CUDACHECK(cudaGetDevice(&currentDeviceId));

                QueryData* const queryData = getQueryDataFromHandle(queryHandle);

                auto& callerEvent = queryData->callerEvent;
                callerEvent.synchronize(); //Ensure that handle is not in use by a previous call (may need to reallocate memory)

                const int numUsable = usableDeviceIds.size();

                //set up data which needs to remain valid after this call
                queryData->remotePersistentDataPerGpu.clear();
                queryData->remotePersistentDataPerGpu.reserve(numUsable);
                for(int d = 0; d < numUsable; d++){                
                    DeviceSwitcher ds(usableDeviceIds[d]);
                    queryData->remotePersistentDataPerGpu.emplace_back(queryData->streams[d], rmm::mr::get_current_device_resource());
                }              

                queryData->numSequences = numSequences;
                queryData->callerDeviceId = currentDeviceId;
                queryData->encodedSequencePitchInInts = encodedSequencePitchInInts;
                queryData->d_readIds = nullptr;
                queryData->d_sequenceData2Bit = d_sequenceData2Bit;
                queryData->d_sequenceLengths = d_sequenceLengths;
                queryData->mr = mr;

                queryData->pinnedData.resize(2*numUsable + numUsable + 1);

                //Create dependency for internal streams
                forkInternalStreams(queryData, stream);

                queryData->allocate(
                    *queryData->numValuesTempStorage,
                    numSequences,
                    encodedSequencePitchInInts,
                    stream
                );

                broadcastNumValuesInputToRemote(queryData);
                
                determineNumValuesOnEachGpu(queryData);

                copyRemoteNumValuesResultsToCallerData(queryData);

                joinInternalStreams(queryData, stream);

                combineNumValuesResults(queryData, d_numValuesPerSequence, stream);
                
                CUDACHECK(cudaStreamSynchronize(stream)); //wait for pinned memory

                queryData->totalNumValues = 0;
                for(int d = 0; d < numUsable; d++){
                    const int* const myPinnedData = queryData->pinnedData + 2*d;
                    queryData->totalNumValues += myPinnedData[0];
                }
                
                totalNumValues = queryData->totalNumValues;

                for(int d = 0; d < numUsable; d++){                
                    DeviceSwitcher ds(usableDeviceIds[d]);
                    queryData->remotePersistentDataPerGpu[d].d_numResultsPerSequence = std::move(queryData->numValuesTempStorage->remote_d_numResultsPerSequence[d]);
                }

                queryData->deallocate(
                    *queryData->numValuesTempStorage,
                    stream
                );

                CUDACHECK(callerEvent.record(stream));
                queryData->previousStage = QueryData::Stage::NumValues; 

            }            
        }

        void retrieveValues(
            MinhasherHandle& queryHandle,
            const read_number* d_readIds,
            int numSequences,
            int totalNumValues,
            read_number* d_values,
            int* d_numValuesPerSequence,
            int* d_offsets, //numSequences + 1
            cudaStream_t stream,
            rmm::mr::device_memory_resource* mr
        ) const override {
            if(numSequences == 0){
                return;
            }

            if(totalNumValues == 0){
                CUDACHECK(cudaMemsetAsync(d_numValuesPerSequence, 0, sizeof(int) * numSequences, stream));
                CUDACHECK(cudaMemsetAsync(d_offsets, 0, sizeof(int) * (numSequences + 1), stream));
                return;
            }

            int currentDeviceId = 0;
            CUDACHECK(cudaGetDevice(&currentDeviceId));

            auto it = usableDeviceIds.end();
            if(isReplicatedSingleGpu || (usableDeviceIds.size() == 1)){
                it = std::find(usableDeviceIds.begin(), usableDeviceIds.end(), currentDeviceId);
            }

            if(it != usableDeviceIds.end()){
                int index = std::distance(usableDeviceIds.begin(), it);
                retrieveValues_singlegpuimpl(
                    index, //index for usableDeviceIds
                    queryHandle,
                    d_readIds,
                    numSequences,
                    totalNumValues,
                    d_values,
                    d_numValuesPerSequence,
                    d_offsets,
                    stream,
                    mr
                );
            }else{

                QueryData* const queryData = getQueryDataFromHandle(queryHandle);

                assert(queryData->previousStage == QueryData::Stage::NumValues);
                assert(queryData->callerDeviceId == currentDeviceId);
                assert(queryData->numSequences == numSequences);

                auto& callerEvent = queryData->callerEvent;
                CUDACHECK(callerEvent.synchronize()); //Ensure that handle is not in use by a previous call (may need to reallocate memory)

                queryData->d_readIds = d_readIds;
                queryData->mr = mr;


                forkInternalStreams(queryData, stream);

                queryData->allocate(
                    *queryData->retrieveTempStorage,
                    numSequences,
                    queryData->totalNumValues,
                    stream
                );

                broadcastRetrieveInputToRemote(queryData);

                retrieveValuesOnEachGpu(queryData);

                const int numUsable = usableDeviceIds.size();
                int* const h_numPerGpu_ps = &queryData->pinnedData[2*numUsable];
                h_numPerGpu_ps[0] = 0;
                for(int d = 0; d < numUsable-1; d++){
                    const int numValues = queryData->pinnedData[2*d];
                    h_numPerGpu_ps[d+1] = h_numPerGpu_ps[d] + numValues;
                }
                CUDACHECK(cudaMemcpyAsync(queryData->retrieveTempStorage->caller_d_offsets_tmp.data(), h_numPerGpu_ps, sizeof(int) * numUsable, H2D, stream));

                copyRemoteResultsToCallerData(queryData);

                joinInternalStreams(queryData, stream);

                combineResults(
                    queryData, 
                    d_values, 
                    d_numValuesPerSequence, 
                    d_offsets, 
                    stream
                );

                queryData->remotePersistentDataPerGpu.clear();

                queryData->deallocate(
                    *queryData->retrieveTempStorage,
                    stream
                );

                CUDACHECK(callerEvent.record(stream));
                queryData->previousStage = QueryData::Stage::Retrieve;
            }
        }

        void compact(cudaStream_t stream = 0) {
            for(auto& minhasher : sgpuMinhashers){
                DeviceSwitcher ds(minhasher->getDeviceId());

                minhasher->compact(stream);
            }            
        }

        MemoryUsage getMemoryInfo() const noexcept override{
            MemoryUsage mem{};

            for(const auto& minhasher : sgpuMinhashers){
                mem += minhasher->getMemoryInfo();
            }

            return mem;
        }

        MemoryUsage getMemoryInfo(const MinhasherHandle& handle) const noexcept override{
            return tempdataVector[handle.getId()]->getMemoryInfo();
        }

        int getNumResultsPerMapThreshold() const noexcept override{
            return resultsPerMapThreshold;
        }
        
        int getNumberOfMaps() const noexcept override{
            
            if(isReplicatedSingleGpu){
                if(sgpuMinhashers.empty()){
                    return 0;
                }else{
                    return sgpuMinhashers[0]->getNumberOfMaps();
                }
            }else{
                int num = 0;
                for(const auto& minhasher : sgpuMinhashers){
                    num += minhasher->getNumberOfMaps();
                }
                return num;
            }
        }

        int getKmerSize() const noexcept override{
            return kmerSize;
        }

        void destroy(){
            for(auto& minhasher : sgpuMinhashers){
                DeviceSwitcher sd(minhasher->getDeviceId());
                minhasher->destroy();
            }
        }

        bool hasGpuTables() const noexcept override {
            return true;
        }



private:        

        void determineNumValues_singlegpuimpl(
            int whichSinglegpu,
            MinhasherHandle& queryHandle,
            const unsigned int* d_sequenceData2Bit,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int* d_numValuesPerSequence,
            int& totalNumValues,
            cudaStream_t stream,
            rmm::mr::device_memory_resource* mr
        ) const {
            assert(int(usableDeviceIds.size()) > whichSinglegpu);

            if(numSequences == 0){
                return;
            }

            int currentDeviceId = 0;
            CUDACHECK(cudaGetDevice(&currentDeviceId));

            QueryData* const queryData = getQueryDataFromHandle(queryHandle);

            auto& callerEvent = queryData->callerEvent;
            callerEvent.synchronize(); //Ensure that handle is not in use by a previous call (may need to reallocate memory)

            queryData->callerDeviceId = currentDeviceId;
            queryData->numSequences = numSequences;
            queryData->pinnedData.resize(2*1 + 1 + 1);
            int* const myPinnedData = queryData->pinnedData;
            int& pinnedtotalNumValues = myPinnedData[0];

            forkInternalStreams(queryData, stream);
            
            {
                const int d = whichSinglegpu;
                DeviceSwitcher ds(usableDeviceIds[d]);

                CudaStream& myStream = queryData->streams[whichSinglegpu];

                const auto& minhasher = *sgpuMinhashers[d];
                minhasher.determineNumValues(
                    *queryData->singlegpuMinhasherHandles[d],
                    d_sequenceData2Bit,
                    encodedSequencePitchInInts,
                    d_sequenceLengths,
                    numSequences,
                    d_numValuesPerSequence,
                    pinnedtotalNumValues,
                    myStream,
                    mr
                );

                CUDACHECK(cudaStreamSynchronize(myStream));
                totalNumValues = pinnedtotalNumValues;

                CUDACHECK(queryData->events[d].record(queryData->streams[d]));

            }

            joinInternalStreams(queryData, stream);

            CUDACHECK(callerEvent.record(stream));
            queryData->previousStage = QueryData::Stage::NumValues;   
        }

        void retrieveValues_singlegpuimpl(
            int whichSinglegpu,
            MinhasherHandle& queryHandle,
            const read_number* d_readIds,
            int numSequences,
            int totalNumValues,
            read_number* d_values,
            int* d_numValuesPerSequence,
            int* d_offsets, //numSequences + 1
            cudaStream_t stream,
            rmm::mr::device_memory_resource* mr
        ) const {
            assert(int(usableDeviceIds.size()) > whichSinglegpu);

            if(numSequences == 0){
                return;
            }

            int currentDeviceId = 0;
            CUDACHECK(cudaGetDevice(&currentDeviceId));

            QueryData* const queryData = getQueryDataFromHandle(queryHandle);

            assert(queryData->previousStage == QueryData::Stage::NumValues);
            assert(queryData->callerDeviceId == currentDeviceId);
            assert(queryData->numSequences == numSequences);

            auto& callerEvent = queryData->callerEvent;
            CUDACHECK(callerEvent.synchronize()); //Ensure that handle is not in use by a previous call (may need to reallocate memory)

            forkInternalStreams(queryData, stream);

            {
                const int d = whichSinglegpu;
                DeviceSwitcher ds(usableDeviceIds[d]);
                CudaStream& myStream = queryData->streams[d];

                const auto& minhasher = *sgpuMinhashers[d];
                
                minhasher.retrieveValues(
                    *queryData->singlegpuMinhasherHandles[whichSinglegpu],
                    d_readIds,
                    numSequences,              
                    totalNumValues,
                    d_values,
                    d_numValuesPerSequence,
                    d_offsets,
                    myStream,
                    mr
                );

                CUDACHECK(queryData->events[d].record(queryData->streams[d]));
            }

            joinInternalStreams(queryData, stream);

            CUDACHECK(callerEvent.record(stream));
            queryData->previousStage = QueryData::Stage::Retrieve;
        }



        void forkInternalStreams(QueryData* queryData, cudaStream_t stream) const{
            auto& callerEvent = queryData->callerEvent;

            callerEvent.record(stream);
            const int numUsable = usableDeviceIds.size();
            for(int d = 0; d < numUsable; d++){
                DeviceSwitcher ds(usableDeviceIds[d]);                
                CUDACHECK(queryData->streams[d].waitEvent(callerEvent, 0));
            }
        }

        void broadcastNumValuesInputToRemote(QueryData* queryHandle) const {
            assert(!isReplicatedSingleGpu);

            const int numUsable = usableDeviceIds.size();
            const int numSequences = queryHandle->numSequences;
            const std::size_t encodedSequencePitchInInts = queryHandle->encodedSequencePitchInInts;

            for(int d = 0; d < numUsable; d++){                
                DeviceSwitcher ds(usableDeviceIds[d]);

                CUDACHECK(cudaMemcpyPeerAsync(
                    queryHandle->numValuesTempStorage->remote_d_input_lengths[d].data(),
                    usableDeviceIds[d],
                    queryHandle->d_sequenceLengths,
                    queryHandle->callerDeviceId,
                    sizeof(int) * numSequences,
                    queryHandle->streams[d]
                ));

                CUDACHECK(cudaMemcpyPeerAsync(
                    queryHandle->numValuesTempStorage->remote_d_input_sequences[d].data(),
                    usableDeviceIds[d],
                    queryHandle->d_sequenceData2Bit,
                    queryHandle->callerDeviceId,
                    sizeof(unsigned int) * encodedSequencePitchInInts * numSequences,
                    queryHandle->streams[d]
                ));
            }

        }

        void broadcastRetrieveInputToRemote(QueryData* queryHandle) const {
            assert(!isReplicatedSingleGpu);

            const int numUsable = usableDeviceIds.size();
            const int numSequences = queryHandle->numSequences;

            for(int d = 0; d < numUsable; d++){                
                if(queryHandle->d_readIds != nullptr){
                    DeviceSwitcher ds(usableDeviceIds[d]);

                    CUDACHECK(cudaMemcpyPeerAsync(
                        queryHandle->retrieveTempStorage->remote_d_readIds[d].data(),
                        usableDeviceIds[d],
                        queryHandle->d_readIds,
                        queryHandle->callerDeviceId,
                        sizeof(read_number) * numSequences,
                        queryHandle->streams[d]
                    ));
                }
            }
        }

        void determineNumValuesOnEachGpu(QueryData* queryHandle) const{
            assert(!isReplicatedSingleGpu);

            const int numUsable = usableDeviceIds.size();
            const int numSequences = queryHandle->numSequences;
            const std::size_t encodedSequencePitchInInts = queryHandle->encodedSequencePitchInInts;
            
            for(int d = 0; d < numUsable; d++){
                DeviceSwitcher ds(usableDeviceIds[d]);

                const auto& minhasher = *sgpuMinhashers[d];

                std::size_t persistent_storage_bytes = 0;
                std::size_t temp_storage_bytes = 0;

                const int* const myInputLengths = queryHandle->numValuesTempStorage->remote_d_input_lengths[d].data();
                const unsigned int* const myInputSequences = queryHandle->numValuesTempStorage->remote_d_input_sequences[d].data();
                int* const myNumValuesPerSequence = queryHandle->numValuesTempStorage->remote_d_numResultsPerSequence[d].data();
                int* const myPinnedData = queryHandle->pinnedData + 2*d;

                int& totalNumValues = myPinnedData[0];
                int* const d_largestSegment = &myPinnedData[1];
                CudaStream& myStream = queryHandle->streams[d];

                minhasher.determineNumValues(
                    nullptr,
                    persistent_storage_bytes,
                    nullptr,
                    temp_storage_bytes,
                    myInputSequences,
                    encodedSequencePitchInInts,
                    myInputLengths,
                    numSequences,
                    myNumValuesPerSequence,
                    totalNumValues,
                    myStream
                );

                queryHandle->remotePersistentDataPerGpu[d].d_persistent.resize(persistent_storage_bytes, myStream.getStream());

                auto* myMr = rmm::mr::get_current_device_resource();

                rmm::device_uvector<char> d_temp(temp_storage_bytes, myStream.getStream(), myMr);

                minhasher.determineNumValues(
                    queryHandle->remotePersistentDataPerGpu[d].d_persistent.data(),
                    persistent_storage_bytes,
                    d_temp.data(),
                    temp_storage_bytes,
                    myInputSequences,
                    encodedSequencePitchInInts,
                    myInputLengths,
                    numSequences,
                    myNumValuesPerSequence,
                    totalNumValues,
                    myStream
                );

                CubCallWrapper(myMr).cubReduceMax(
                    myNumValuesPerSequence, 
                    d_largestSegment, 
                    numSequences, 
                    myStream
                );
            }
        }

        void retrieveValuesOnEachGpu(QueryData* queryHandle) const{
            assert(!isReplicatedSingleGpu);

            const int numUsable = usableDeviceIds.size();
            const int numSequences = queryHandle->numSequences;

            for(int d = 0; d < numUsable; d++){
                DeviceSwitcher ds(usableDeviceIds[d]);

                nvtx::push_range("retrieveValuesOnEachGpu from device " + std::to_string(usableDeviceIds[d]), 3);

                auto* myMr = rmm::mr::get_current_device_resource();

                const auto& minhasher = *sgpuMinhashers[d];

                std::size_t persistent_storage_bytes = queryHandle->remotePersistentDataPerGpu[d].d_persistent.size();

                int* const myNumValuesPerSequence = queryHandle->remotePersistentDataPerGpu[d].d_numResultsPerSequence.data();                
                int* const myOffsets = queryHandle->retrieveTempStorage->remote_d_offsets[d].data();
                const read_number* const myReadIds = (queryHandle->d_readIds == nullptr) ? 
                    nullptr: queryHandle->retrieveTempStorage->remote_d_readIds[d].data();

                int* const myPinnedData = queryHandle->pinnedData + 2*d;
                
                CudaStream& myStream = queryHandle->streams[d];

                CUDACHECK(cudaStreamSynchronize(myStream)); //Wait for number of values and max segment from async memcpy call.

                const int totalNumValues = myPinnedData[0];
                const int largestSegment = myPinnedData[1];

                queryHandle->retrieveTempStorage->remote_d_results[d].resize(totalNumValues, myStream.getStream());
                read_number* const myValues = queryHandle->retrieveTempStorage->remote_d_results[d].data();

                std::size_t temp_storage_bytes = 0;

                minhasher.retrieveValues(
                    queryHandle->remotePersistentDataPerGpu[d].d_persistent.data(),
                    persistent_storage_bytes,
                    nullptr,
                    temp_storage_bytes,
                    myReadIds,
                    numSequences,                    
                    totalNumValues,
                    123456, //largest segment unused for dry-run
                    myValues,
                    myNumValuesPerSequence,
                    myOffsets, //numSequences + 1
                    myStream
                );

                rmm::device_uvector<char> d_temp(temp_storage_bytes, myStream.getStream(), myMr);

                minhasher.retrieveValues(
                    queryHandle->remotePersistentDataPerGpu[d].d_persistent.data(),
                    persistent_storage_bytes,
                    d_temp.data(),
                    temp_storage_bytes,
                    myReadIds,
                    numSequences,
                    totalNumValues,
                    largestSegment,
                    myValues,
                    myNumValuesPerSequence,
                    myOffsets, //numSequences + 1
                    myStream
                );

                nvtx::pop_range();
            }
        }

        void copyRemoteNumValuesResultsToCallerData(QueryData* queryHandle) const{
            assert(!isReplicatedSingleGpu);

            const int numUsable = usableDeviceIds.size();
            const int numSequences = queryHandle->numSequences;            

            for(int d = 0; d < numUsable; d++){                
                DeviceSwitcher ds(usableDeviceIds[d]);
                // CUDACHECK(cudaStreamSynchronize(queryHandle->streams[d])); //wait for pinned memory

                // const int numValues = queryHandle->pinnedData[2*d];

                CUDACHECK(cudaMemcpyPeerAsync(
                    queryHandle->numValuesTempStorage->caller_d_numResultsPerSequence.data() + d * numSequences,
                    queryHandle->callerDeviceId,
                    queryHandle->numValuesTempStorage->remote_d_numResultsPerSequence[d].data(),
                    usableDeviceIds[d],
                    sizeof(int) * numSequences,
                    queryHandle->streams[d]
                ));

                CUDACHECK(queryHandle->events[d].record(queryHandle->streams[d]));
            }

        }

        void copyRemoteResultsToCallerData(QueryData* queryHandle) const{
            assert(!isReplicatedSingleGpu);

            const int numUsable = usableDeviceIds.size();
            const int numSequences = queryHandle->numSequences;
            
            int* const h_numPerGpu_ps = &queryHandle->pinnedData[2*numUsable];

            for(int d = 0; d < numUsable; d++){                
                DeviceSwitcher ds(usableDeviceIds[d]);
                const int numValues = queryHandle->pinnedData[2*d];

                nvtx::push_range("copyRemoteResultsToCallerData from device " + std::to_string(usableDeviceIds[d]), 5);

                CUDACHECK(cudaMemcpyPeerAsync(
                    queryHandle->retrieveTempStorage->caller_d_results.data() + h_numPerGpu_ps[d],
                    queryHandle->callerDeviceId,
                    queryHandle->retrieveTempStorage->remote_d_results[d].data(),
                    usableDeviceIds[d],
                    sizeof(read_number) * numValues,
                    queryHandle->streams[d]
                ));

                CUDACHECK(cudaMemcpyPeerAsync(
                    queryHandle->retrieveTempStorage->caller_d_numResultsPerSequence.data() + d * numSequences,
                    queryHandle->callerDeviceId,
                    queryHandle->remotePersistentDataPerGpu[d].d_numResultsPerSequence.data(),
                    usableDeviceIds[d],
                    sizeof(int) * numSequences,
                    queryHandle->streams[d]
                ));

                CUDACHECK(cudaMemcpyPeerAsync(
                    queryHandle->retrieveTempStorage->caller_d_offsets.data() + d * (numSequences + 1),
                    queryHandle->callerDeviceId,
                    queryHandle->retrieveTempStorage->remote_d_offsets[d].data(),
                    usableDeviceIds[d],
                    sizeof(int) * (numSequences + 1),
                    queryHandle->streams[d]
                ));

                CUDACHECK(queryHandle->events[d].record(queryHandle->streams[d]));

                nvtx::pop_range();
            }
        }

        void joinInternalStreams(QueryData* queryHandle, cudaStream_t stream) const{
            const int numUsable = usableDeviceIds.size();

            for(int d = 0; d < numUsable; d++){
                CUDACHECK(cudaStreamWaitEvent(stream, queryHandle->events[d], 0));
            }
        }

        void combineNumValuesResults(
            QueryData* queryHandle, 
            int* d_numValuesPerSequence,
            cudaStream_t stream
        ) const{
            assert(!isReplicatedSingleGpu);

            const int numUsable = usableDeviceIds.size();
            const int numSequences = queryHandle->numSequences;
            
                        
            dim3 block(128, 1, 1);
            dim3 grid(SDIV(numSequences, block.x), 1, 1);

            multigpuminhasherkernels::aggregateNumValuesPartitionResultsKernel
                    <<<grid, block, 0, stream>>>(
                queryHandle->numValuesTempStorage->caller_d_numResultsPerSequence.data(),
                d_numValuesPerSequence,
                numSequences,
                numUsable
            ); CUDACHECKASYNC;

        }

        void combineResults(
            QueryData* queryHandle, 
            read_number* d_values, 
            int* d_numValuesPerSequence, 
            int* d_offsets, 
            cudaStream_t stream
        ) const{
            assert(!isReplicatedSingleGpu);

            const int numUsable = usableDeviceIds.size();
            const int numSequences = queryHandle->numSequences;
            
            auto& callerEvent = queryHandle->callerEvent;
            
            int* h_maxSegmentSize = queryHandle->pinnedData + 2 * numUsable + numUsable;

            multigpuminhasherkernels::aggregatePartitionResultsSingleBlockKernel<1024, 1><<<1, 1024, 0, stream>>>(
                queryHandle->retrieveTempStorage->caller_d_numResultsPerSequence.data(),
                d_numValuesPerSequence,
                numSequences,
                numUsable,
                h_maxSegmentSize,
                d_offsets
            ); CUDACHECKASYNC;

            CUDACHECK(callerEvent.record(stream));

            /*
                Copy gpu results into contiguous range. Interleave results for same sequences
                i.e. for input GPU 0: AABBBBC, GPU 1: AAAABBC result will be AAAAAABBBBBCC
            */
            multigpuminhasherkernels::copyToInterleavedKernel<<<numSequences, 128, 0, stream>>>(
                queryHandle->retrieveTempStorage->caller_d_results.data(),
                queryHandle->retrieveTempStorage->caller_d_offsets.data(),
                queryHandle->retrieveTempStorage->caller_d_numResultsPerSequence.data(),
                queryHandle->retrieveTempStorage->caller_d_offsets_tmp.data(),
                d_offsets,
                d_values,
                numSequences,
                numUsable
            ); CUDACHECKASYNC;


            CUDACHECK(callerEvent.synchronize()); //wait for h_maxSegmentSize

            //values of same sequence are now stored in contiguous locations. make unique
            GpuSegmentedUnique::unique(
                d_values, //input values
                queryHandle->totalNumValues,
                queryHandle->retrieveTempStorage->caller_d_results.data(), //output values
                d_numValuesPerSequence, //output segment sizes
                numSequences,
                *h_maxSegmentSize,
                d_offsets, //segment begin offsets
                d_offsets + 1, //segment end offsets
                0,
                sizeof(read_number) * 8,
                stream,
                queryHandle->mr
            );

            //compute final offsets.
            //queryHandle->callerBuffers->d_offsets[0] == 0 is set to 0 in aggregatePartitionResultsSingleBlockKernel

            CubCallWrapper(queryHandle->mr).cubInclusiveSum(
                d_numValuesPerSequence,
                queryHandle->retrieveTempStorage->caller_d_offsets.data() + 1,
                numSequences,
                stream
            );

            //copy results to destination
            multigpuminhasherkernels::copyResultsToDestinationKernel<<<numSequences, 128, 0, stream>>>(
                queryHandle->retrieveTempStorage->caller_d_results.data(),
                d_offsets,
                d_values,
                queryHandle->retrieveTempStorage->caller_d_offsets.data(),
                d_numValuesPerSequence,
                numSequences
            ); CUDACHECKASYNC;

        }

        void finalize(){
            compact();
        }


        std::uint64_t getKmerMask() const{
            constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;

            return std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - getKmerSize()) * 2);
        }

        constexpr float getLoad() const noexcept{
            return 0.8f;
        }

        QueryData* getQueryDataFromHandle(const MinhasherHandle& queryHandle) const{
            std::shared_lock<SharedMutex> lock(sharedmutex);

            return tempdataVector[queryHandle.getId()].get();
        }
        

        mutable int counter = 0;
        mutable SharedMutex sharedmutex{};

        bool isReplicatedSingleGpu = false;

        int maxNumKeys{};
        int kmerSize{};
        int resultsPerMapThreshold{};
        std::vector<int> deviceIds;
        std::vector<int> usableDeviceIds;
        std::vector<HostBuffer<int>> vec_h_currentHashFunctionNumbers{};
        std::vector<std::unique_ptr<SingleGpuMinhasher>> sgpuMinhashers{};
        mutable std::vector<std::unique_ptr<QueryData>> tempdataVector{};
    };


}
}




#endif

#endif //#ifdef CARE_HAS_WARPCORE