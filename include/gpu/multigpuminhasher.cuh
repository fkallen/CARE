#ifndef CARE_MULTI_GPU_MINHASHER_CUH
#define CARE_MULTI_GPU_MINHASHER_CUH

#include <config.hpp>


#include <gpu/distributedreadstorage.hpp>
#include <gpu/cuda_unique.cuh>
#include <gpu/singlegpuminhasher.cuh>
#include <gpu/kernels.hpp>
#include <gpu/gpuminhasher.cuh>

#include <options.hpp>
#include <util.hpp>
#include <hpc_helpers.cuh>
#include <filehelpers.hpp>

#include <sequencehelpers.hpp>
#include <memorymanagement.hpp>
#include <threadpool.hpp>

#include <cub/cub.cuh>

#include <vector>
#include <memory>
#include <limits>
#include <string>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cassert>

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
            struct RemoteData{
                DeviceBuffer<unsigned int> d_input_sequences{};
                DeviceBuffer<int> d_input_lengths{};
                DeviceBuffer<read_number> d_results{};
                DeviceBuffer<int> d_numResultsPerSequence{};
                DeviceBuffer<int> d_offsets{};
                DeviceBuffer<char> d_persistent{};
                DeviceBuffer<char> d_temp{};
                DeviceBuffer<read_number> d_readIds{};
            };

            struct CallerData{
                DeviceBuffer<read_number> d_results{};
                DeviceBuffer<int> d_numResultsPerSequence{};
                DeviceBuffer<int> d_offsets{};
                DeviceBuffer<int> d_offsets_tmp{};
                DeviceBuffer<char> d_temp{};
                CudaEvent event{cudaEventDisableTiming};
            };

            int numSequences{};
            int callerDeviceId{};
            std::size_t encodedSequencePitchInInts{};
            const read_number* d_readIds{};
            const unsigned int* d_sequenceData2Bit{};
            const int* d_sequenceLengths{};

            int totalNumValues{};

            std::vector<RemoteData> remotedataPerGpu{};
            std::map<int, CallerData> callerDataMap{};

            HostBuffer<int> pinnedData{};

            std::vector<CudaStream> streams{};
            std::vector<CudaEvent> events{};

            std::vector<int> deviceIds{};

            MemoryUsage getMemoryInfo() const{
                MemoryUsage mem{};

                const int numIds = deviceIds.size();

                for(int i = 0; i < numIds; i++){
                    //mem.device[deviceIds[i]] = remoteResultsPerGpu[i].capacityInBytes();
                }

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
            const DistributedReadStorage& gpuReadStorage,
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
                        std::min(remainingNumHashfunctions, 4), //debugging. allow only 4 tables per gpu
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
                cudaDeviceSynchronize(); CUERR;
            }

            cpuTimer.stop();
            cpuTimer.print();

            const int numberOfAvailableHashFunctions = maxNumHashfunctions - remainingNumHashfunctions;

            return numberOfAvailableHashFunctions; 
        }

        QueryHandle makeQueryHandle() const override{
            auto ptr = std::make_unique<QueryData>();

            const int numMinhashers = sgpuMinhashers.size();

            ptr->remotedataPerGpu.resize(numMinhashers);

            ptr->streams.resize(numMinhashers);
            ptr->events.resize(numMinhashers);
            ptr->deviceIds = usableDeviceIds;

            for(int i = 0; i < numMinhashers; i++){
                DeviceSwitcher ds(sgpuMinhashers[i]->getDeviceId());
                ptr->streams[i] = std::move(CudaStream{});
                ptr->events[i] = std::move(CudaEvent{cudaEventDisableTiming});

                ptr->remotedataPerGpu[i] = std::move(QueryData::RemoteData{});
                ptr->callerDataMap.emplace(sgpuMinhashers[i]->getDeviceId(), std::move(QueryData::CallerData{}));
            }

            CUERR;

            std::lock_guard<std::mutex> lg(m);
            const int handleid = counter++;
            QueryHandle h = constructHandle(handleid);

            tempdataVector.emplace_back(std::move(ptr));

            return h;
        }

        void query(
            QueryHandle& handle,
            const unsigned int* d_encodedSequences,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int deviceId, 
            cudaStream_t stream,
            read_number* d_similarReadIds,
            int* d_similarReadsPerSequence,
            int* d_similarReadsPerSequencePrefixSum
        ) const override {
            query_impl(
                handle,
                nullptr,
                d_encodedSequences,
                encodedSequencePitchInInts,
                d_sequenceLengths,
                numSequences,
                deviceId,
                stream, 
                d_similarReadIds,
                d_similarReadsPerSequence,
                d_similarReadsPerSequencePrefixSum
            );
        }

        void queryExcludingSelf(
            QueryHandle& handle,
            const read_number* d_readIds,
            const unsigned int* d_encodedSequences,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int deviceId, 
            cudaStream_t stream,
            read_number* d_similarReadIds,
            int* d_similarReadsPerSequence,
            int* d_similarReadsPerSequencePrefixSum
        ) const override{
            query_impl(
                handle,
                d_readIds,
                d_encodedSequences,
                encodedSequencePitchInInts,
                d_sequenceLengths,
                numSequences,
                deviceId,
                stream, 
                d_similarReadIds,
                d_similarReadsPerSequence,
                d_similarReadsPerSequencePrefixSum
            );
        }

        void compact(cudaStream_t stream = 0) override{
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

        MemoryUsage getMemoryInfo(const QueryHandle& handle) const noexcept override{
            return tempdataVector[handle.getId()]->getMemoryInfo();
        }

        int getNumResultsPerMapThreshold() const noexcept override{
            return resultsPerMapThreshold;
        }
        
        int getNumberOfMaps() const noexcept override{
            int num = 0;
            for(const auto& minhasher : sgpuMinhashers){
                num += minhasher->getNumberOfMaps();
            }
            return num;
        }

        void destroy() override{
            for(auto& minhasher : sgpuMinhashers){
                DeviceSwitcher sd(minhasher->getDeviceId());
                minhasher->destroy();
            }
        }



private:        


        void query_impl(
            QueryHandle& queryHandle,
            const read_number* d_readIds,
            const unsigned int* d_sequenceData2Bit,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int deviceId, 
            cudaStream_t stream,
            read_number* d_values,
            int* d_numValuesPerSequence,
            int* d_offsets //numSequences + 1
        ) const {
            if(numSequences == 0){
                return;
            }

            DeviceSwitcher globalds(deviceId);

            QueryData* const queryData = tempdataVector[queryHandle.getId()].get();

            auto& callerData = queryData->callerDataMap[deviceId]; //Implicit creation of node is safe because deviceId is currently set
            auto& callerEvent = callerData.event;
            callerEvent.synchronize(); //Ensure that handle is not in use by a previous call (may need to reallocate memory)

            queryData->numSequences = numSequences;
            queryData->callerDeviceId = deviceId;
            queryData->encodedSequencePitchInInts = encodedSequencePitchInInts;
            queryData->d_readIds = d_readIds;
            queryData->d_sequenceData2Bit = d_sequenceData2Bit;
            queryData->d_sequenceLengths = d_sequenceLengths;

            const int numUsable = usableDeviceIds.size();
            queryData->pinnedData.resize(2*numUsable + numUsable + 1);

            std::size_t cubsize = 0;
            cub::DeviceReduce::Max(
                nullptr, 
                cubsize, 
                (int*)nullptr, 
                (int*)nullptr, 
                numSequences, 
                stream
            );
            cubsize = SDIV(cubsize, 4) * 4;

            //Create dependency for internal streams
            callerEvent.record(stream);
            for(int d = 0; d < numUsable; d++){
                DeviceSwitcher ds(usableDeviceIds[d]);                
                queryData->streams[d].waitEvent(callerEvent, 0);
            }

            resizeRemoteInputBuffers(queryData);          

            broadcastInputToRemote(queryData);
            
            resizeRemoteNumValuesBuffers(queryData);

            determineNumValuesOnEachGpu(queryData);
            
            retrieveValuesOnEachGpu(queryData);
                        
            //early exit if there is only one minhasher. Simply copy remote results to destination buffers and return
            if(numUsable == 1){        
                copyRemoteResultsToCaller(queryData, 0, d_values, d_numValuesPerSequence, d_offsets);

                cudaStreamWaitEvent(stream, queryData->events[0], 0); CUERR; //join the internal stream

                return;
            }

            queryData->totalNumValues = 0;
            for(int d = 0; d < numUsable; d++){
                const int* const myPinnedData = queryData->pinnedData + 2*d;
                queryData->totalNumValues += myPinnedData[0];
            }

            resizeCallerData(queryData);            

            int* const h_numPerGpu_ps = &queryData->pinnedData[2*numUsable];
            h_numPerGpu_ps[0] = 0;
            for(int d = 0; d < numUsable-1; d++){
                const int numValues = queryData->pinnedData[2*d];
                h_numPerGpu_ps[d+1] = h_numPerGpu_ps[d] + numValues;
            }
            cudaMemcpyAsync(callerData.d_offsets_tmp.data(), h_numPerGpu_ps, sizeof(int) * numUsable, H2D, stream); CUERR;

            copyRemoteResultsToCallerData(queryData);

            joinInternalStreams(queryData, stream);

            combineResults(
                queryData, 
                d_values, 
                d_numValuesPerSequence, 
                d_offsets, 
                stream
            );

            callerEvent.record(stream); CUERR;
        }

        std::size_t getRequiredCubTempSizeForResultMerging(int numSequences) const noexcept{
            std::size_t bytes = 0;

            cub::DeviceScan::InclusiveSum(
                nullptr,
                bytes,
                (int*)nullptr,
                (int*)nullptr,
                numSequences
            );

            std::size_t bytes2 = 0;

            cub::DeviceReduce::Max(
                nullptr, 
                bytes2, 
                (int*)nullptr,
                (int*)nullptr,
                numSequences
            );

            bytes = std::max(bytes, bytes2);

            cub::DeviceReduce::Sum(
                nullptr, 
                bytes2, 
                (int*)nullptr,
                (int*)nullptr,
                numSequences
            );

            bytes = std::max(bytes, bytes2);

            return bytes;
        }

        void resizeRemoteInputBuffers(QueryData* queryHandle) const{
            const int numUsable = usableDeviceIds.size();
            const int numSequences = queryHandle->numSequences;
            const std::size_t encodedSequencePitchInInts = queryHandle->encodedSequencePitchInInts;

            for(int d = 0; d < numUsable; d++){
                DeviceSwitcher ds(usableDeviceIds[d]);
                queryHandle->remotedataPerGpu[d].d_input_lengths.resize(numSequences);
                queryHandle->remotedataPerGpu[d].d_input_sequences.resize(encodedSequencePitchInInts * numSequences);
                
                if(queryHandle->d_readIds != nullptr){
                    queryHandle->remotedataPerGpu[d].d_readIds.resize(numSequences);
                }
            }
        }

        void broadcastInputToRemote(QueryData* queryHandle) const {
            const int numUsable = usableDeviceIds.size();
            const int numSequences = queryHandle->numSequences;
            const std::size_t encodedSequencePitchInInts = queryHandle->encodedSequencePitchInInts;

            for(int d = 0; d < numUsable; d++){                
                DeviceSwitcher ds(usableDeviceIds[d]);

                cudaMemcpyAsync(
                    queryHandle->remotedataPerGpu[d].d_input_lengths.data(),
                    queryHandle->d_sequenceLengths,
                    sizeof(int) * numSequences,
                    D2D,
                    queryHandle->streams[d]
                ); CUERR;

                cudaMemcpyAsync(
                    queryHandle->remotedataPerGpu[d].d_input_sequences.data(),
                    queryHandle->d_sequenceData2Bit,
                    sizeof(unsigned int) * encodedSequencePitchInInts * numSequences,
                    D2D,
                    queryHandle->streams[d]
                ); CUERR;

                if(queryHandle->d_readIds != nullptr){
                    cudaMemcpyAsync(
                        queryHandle->remotedataPerGpu[d].d_readIds.data(),
                        queryHandle->d_readIds,
                        sizeof(read_number) * numSequences,
                        D2D,
                        queryHandle->streams[d]
                    ); CUERR;
                }
            }
        }

        void resizeRemoteNumValuesBuffers(QueryData* queryHandle) const{
            const int numUsable = usableDeviceIds.size();
            const int numSequences = queryHandle->numSequences;

            //resize remote buffers numValuesPerSequence, offsets
            for(int d = 0; d < numUsable; d++){
                DeviceSwitcher ds(usableDeviceIds[d]);
                queryHandle->remotedataPerGpu[d].d_numResultsPerSequence.resize(numSequences);
                queryHandle->remotedataPerGpu[d].d_offsets.resize(numSequences+1);
            }
        }

        void determineNumValuesOnEachGpu(QueryData* queryHandle) const{
            const int numUsable = usableDeviceIds.size();
            const int numSequences = queryHandle->numSequences;
            const std::size_t encodedSequencePitchInInts = queryHandle->encodedSequencePitchInInts;

            std::size_t cubsize = 0;
            cub::DeviceReduce::Max(
                nullptr, 
                cubsize, 
                (int*)nullptr, 
                (int*)nullptr, 
                numSequences
            );
            cubsize = SDIV(cubsize, 4) * 4;
            
            for(int d = 0; d < numUsable; d++){
                DeviceSwitcher ds(usableDeviceIds[d]);

                const auto& minhasher = *sgpuMinhashers[d];

                std::size_t persistent_storage_bytes = 0;
                std::size_t temp_storage_bytes = 0;

                const int* const myInputLengths = queryHandle->remotedataPerGpu[d].d_input_lengths.data();
                const unsigned int* const myInputSequences = queryHandle->remotedataPerGpu[d].d_input_sequences.data();
                int* const myNumValuesPerSequence = queryHandle->remotedataPerGpu[d].d_numResultsPerSequence.data();
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

                temp_storage_bytes = std::max(temp_storage_bytes, cubsize);

                queryHandle->remotedataPerGpu[d].d_persistent.resize(persistent_storage_bytes);
                queryHandle->remotedataPerGpu[d].d_temp.resize(temp_storage_bytes);

                minhasher.determineNumValues(
                    queryHandle->remotedataPerGpu[d].d_persistent.data(),
                    persistent_storage_bytes,
                    queryHandle->remotedataPerGpu[d].d_temp.data(),
                    temp_storage_bytes,
                    myInputSequences,
                    encodedSequencePitchInInts,
                    myInputLengths,
                    numSequences,
                    myNumValuesPerSequence,
                    totalNumValues,
                    myStream
                );

                cub::DeviceReduce::Max(
                    queryHandle->remotedataPerGpu[d].d_temp.data(), 
                    temp_storage_bytes, 
                    myNumValuesPerSequence, 
                    d_largestSegment, 
                    numSequences, 
                    myStream
                );
            }
        }

        void retrieveValuesOnEachGpu(QueryData* queryHandle) const{
            const int numUsable = usableDeviceIds.size();
            const int numSequences = queryHandle->numSequences;

            for(int d = 0; d < numUsable; d++){
                DeviceSwitcher ds(usableDeviceIds[d]);

                const auto& minhasher = *sgpuMinhashers[d];

                std::size_t persistent_storage_bytes = queryHandle->remotedataPerGpu[d].d_persistent.sizeInBytes();
                std::size_t temp_storage_bytes = queryHandle->remotedataPerGpu[d].d_temp.sizeInBytes();

                int* const myNumValuesPerSequence = queryHandle->remotedataPerGpu[d].d_numResultsPerSequence.data();                
                int* const myOffsets = queryHandle->remotedataPerGpu[d].d_offsets.data();
                const read_number* const myReadIds = (queryHandle->d_readIds == nullptr) ? nullptr: queryHandle->remotedataPerGpu[d].d_readIds.data();

                int* const myPinnedData = queryHandle->pinnedData + 2*d;
                
                CudaStream& myStream = queryHandle->streams[d];

                cudaStreamSynchronize(myStream); CUERR; //Wait for number of values and max segment from async memcpy call.

                const int totalNumValues = myPinnedData[0];
                const int largestSegment = myPinnedData[1];

                queryHandle->remotedataPerGpu[d].d_results.resize(totalNumValues);
                read_number* const myValues = queryHandle->remotedataPerGpu[d].d_results.data();

                minhasher.retrieveValues(
                    queryHandle->remotedataPerGpu[d].d_persistent.data(),
                    persistent_storage_bytes,
                    nullptr,
                    temp_storage_bytes,
                    myReadIds,
                    numSequences,
                    123456, //unused
                    myStream,
                    totalNumValues,
                    123456, //largest segment unused for dry-run
                    myValues,
                    myNumValuesPerSequence,
                    myOffsets //numSequences + 1
                );

                queryHandle->remotedataPerGpu[d].d_temp.resize(temp_storage_bytes);

                minhasher.retrieveValues(
                    queryHandle->remotedataPerGpu[d].d_persistent.data(),
                    persistent_storage_bytes,
                    queryHandle->remotedataPerGpu[d].d_temp.data(),
                    temp_storage_bytes,
                    myReadIds,
                    numSequences,
                    123456, //unused
                    myStream,
                    totalNumValues,
                    largestSegment,
                    myValues,
                    myNumValuesPerSequence,
                    myOffsets //numSequences + 1
                );
            }
        }

        void resizeCallerData(QueryData* queryHandle) const{
            const int numUsable = usableDeviceIds.size();
            const int numSequences = queryHandle->numSequences;

            auto& callerData = queryHandle->callerDataMap[queryHandle->callerDeviceId];

            callerData.d_results.resize(queryHandle->totalNumValues);
            callerData.d_numResultsPerSequence.resize(numSequences * numUsable);
            callerData.d_offsets.resize((numSequences + 1) * numUsable);
            callerData.d_offsets_tmp.resize(numSequences * numUsable);

            //Allocate temp storage for the remaining steps
            std::size_t temp_storage_bytes = 0;

            GpuSegmentedUnique::unique(
                nullptr, //tempstorage
                temp_storage_bytes,
                (read_number*)nullptr, //input values
                queryHandle->totalNumValues,
                (read_number*)nullptr, //output values
                (int*)nullptr, //output segment sizes
                numSequences,
                123456, //sizeOfLargestSegment, unused for dry-run
                (int*)nullptr, //segment begin offsets
                (int*)nullptr, //segment end offsets
                0,
                sizeof(read_number) * 8,
                (cudaStream_t)0
            );

            temp_storage_bytes = std::max(temp_storage_bytes, getRequiredCubTempSizeForResultMerging(numSequences));
            callerData.d_temp.resize(temp_storage_bytes);
        }

        void copyRemoteResultsToCaller(
            QueryData* queryHandle, 
            int deviceIndex,
            read_number* d_values,
            int* d_numValuesPerSequence,
            int* d_offsets
        ) const {
            const int numUsable = usableDeviceIds.size();
            const int numSequences = queryHandle->numSequences;

            const int d = deviceIndex;

            DeviceSwitcher ds(usableDeviceIds[d]);
            const int numValues = queryHandle->pinnedData[2*d];

            cudaMemcpyAsync(
                d_values,
                queryHandle->remotedataPerGpu[d].d_results.data(),
                sizeof(read_number) * numValues,
                D2D,
                queryHandle->streams[d]
            ); CUERR;

            cudaMemcpyAsync(
                d_numValuesPerSequence,
                queryHandle->remotedataPerGpu[d].d_numResultsPerSequence.data(),
                sizeof(int) * numSequences,
                D2D,
                queryHandle->streams[d]
            ); CUERR;

            cudaMemcpyAsync(
                d_offsets,
                queryHandle->remotedataPerGpu[d].d_offsets.data(),
                sizeof(int) * (numSequences + 1),
                D2D,
                queryHandle->streams[d]
            ); CUERR;

            queryHandle->events[d].record(queryHandle->streams[d]);
        }

        void copyRemoteResultsToCallerData(QueryData* queryHandle) const{
            const int numUsable = usableDeviceIds.size();
            const int numSequences = queryHandle->numSequences;
            auto& callerData = queryHandle->callerDataMap[queryHandle->callerDeviceId];
            int* const h_numPerGpu_ps = &queryHandle->pinnedData[2*numUsable];

            for(int d = 0; d < numUsable; d++){                
                DeviceSwitcher ds(usableDeviceIds[d]);
                const int numValues = queryHandle->pinnedData[2*d];

                cudaMemcpyAsync(
                    callerData.d_results.data() + h_numPerGpu_ps[d],
                    queryHandle->remotedataPerGpu[d].d_results.data(),
                    sizeof(read_number) * numValues,
                    D2D,
                    queryHandle->streams[d]
                ); CUERR;

                cudaMemcpyAsync(
                    callerData.d_numResultsPerSequence.data() + d * numSequences,
                    queryHandle->remotedataPerGpu[d].d_numResultsPerSequence.data(),
                    sizeof(int) * numSequences,
                    D2D,
                    queryHandle->streams[d]
                ); CUERR;

                cudaMemcpyAsync(
                    callerData.d_offsets.data() + d * (numSequences + 1),
                    queryHandle->remotedataPerGpu[d].d_offsets.data(),
                    sizeof(int) * (numSequences + 1),
                    D2D,
                    queryHandle->streams[d]
                ); CUERR;

                queryHandle->events[d].record(queryHandle->streams[d]);
            }
        }

        void joinInternalStreams(QueryData* queryHandle, cudaStream_t stream) const{
            const int numUsable = usableDeviceIds.size();

            for(int d = 0; d < numUsable; d++){
                cudaStreamWaitEvent(stream, queryHandle->events[d], 0); CUERR;
            }
        }

        void combineResults(
            QueryData* queryHandle, 
            read_number* d_values, 
            int* d_numValuesPerSequence, 
            int* d_offsets, 
            cudaStream_t stream
        ) const{
            const int numUsable = usableDeviceIds.size();
            const int numSequences = queryHandle->numSequences;
            auto& callerData = queryHandle->callerDataMap[queryHandle->callerDeviceId];
            auto& callerEvent = callerData.event;
            
            int* h_maxSegmentSize = queryHandle->pinnedData + 2 * numUsable + numUsable;
            std::size_t temp_storage_bytes = callerData.d_temp.sizeInBytes();

            multigpuminhasherkernels::aggregatePartitionResultsSingleBlockKernel<1024, 1><<<1, 1024, 0, stream>>>(
                callerData.d_numResultsPerSequence.data(),
                d_numValuesPerSequence,
                numSequences,
                numUsable,
                h_maxSegmentSize,
                d_offsets
            ); CUERR;

            callerEvent.record(stream);

            /*
                Copy gpu results into contiguous range. Interleave results for same sequences
                i.e. for input GPU 0: AABBBBC, GPU 1: AAAABBC result will be AAAAAABBBBBCC
            */
            multigpuminhasherkernels::copyToInterleavedKernel<<<numSequences, 128, 0, stream>>>(
                callerData.d_results.data(),
                callerData.d_offsets.data(),
                callerData.d_numResultsPerSequence.data(),
                callerData.d_offsets_tmp.data(),
                d_offsets,
                d_values,
                numSequences,
                numUsable
            ); CUERR;


            callerEvent.synchronize(); CUERR; //wait for h_maxSegmentSize

            //values of same sequence are now stored in contiguous locations. make unique
            GpuSegmentedUnique::unique(
                callerData.d_temp.data(),
                temp_storage_bytes,
                d_values, //input values
                queryHandle->totalNumValues,
                callerData.d_results.data(), //output values
                d_numValuesPerSequence, //output segment sizes
                numSequences,
                *h_maxSegmentSize,
                d_offsets, //segment begin offsets
                d_offsets + 1, //segment end offsets
                0,
                sizeof(read_number) * 8,
                stream
            );

            //compute final offsets.
            //callerData.d_offsets[0] == 0 is set to 0 in aggregatePartitionResultsSingleBlockKernel

            cub::DeviceScan::InclusiveSum(
                callerData.d_temp.data(),
                temp_storage_bytes,
                d_numValuesPerSequence,
                callerData.d_offsets.data() + 1,
                numSequences,
                stream
            );

            //copy results to destination
            multigpuminhasherkernels::copyResultsToDestinationKernel<<<numSequences, 128, 0, stream>>>(
                callerData.d_results.data(),
                d_offsets,
                d_values,
                callerData.d_offsets.data(),
                d_numValuesPerSequence,
                numSequences
            ); CUERR;

        }

        void finalize(){
            compact();
        }

        constexpr int getKmerSize() const noexcept{
            return kmerSize;
        }

        std::uint64_t getKmerMask() const{
            constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;

            return std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - getKmerSize()) * 2);
        }

        constexpr float getLoad() const noexcept{
            return 0.8f;
        }
        

        mutable int counter = 0;
        mutable std::mutex m{};

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