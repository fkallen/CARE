#ifndef CARE_MULTI_GPU_MINHASHER_CUH
#define CARE_MULTI_GPU_MINHASHER_CUH

#include <config.hpp>


#include <gpu/distributedreadstorage.hpp>
#include <gpu/cuda_unique.cuh>
#include <gpu/singlegpuminhasher.cuh>
#include <gpu/kernels.hpp>


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

namespace care{
namespace gpu{


    class MultiGpuMinhasher{
    public:
        using Key = kmer_type;
        using Value = read_number;

        using DeviceSwitcher = cub::SwitchDevice;

        template<class T>
        using HostBuffer = helpers::SimpleAllocationPinnedHost<T, 5>;
        template<class T>
        using DeviceBuffer = helpers::SimpleAllocationDevice<T, 5>;


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
        ){
            
            const int numDevices = deviceIds.size();
            sgpuMinhashers.clear();
            usableDeviceIds.clear();

            int currentHashFunctionOffset = hashFunctionOffset;

            vec_h_currentHashFunctionNumbers.resize(numDevices);

            int remainingNumHashfunctions = maxNumHashfunctions;

            for(int d = 0; d < numDevices; d++){
                DeviceSwitcher ds(deviceIds[d]);

                if(remainingNumHashfunctions > 0){

                    SingleGpuMinhasher mh(nReads, resultsPerMapThreshold, kmerSize);
                    const int createdTables =  mh.constructFromReadStorage(
                        runtimeOptions, 
                        nReads, 
                        gpuReadStorage, 
                        upperBoundSequenceLength, 
                        remainingNumHashfunctions, 
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
                    }
                }                
            }

            const int numberOfAvailableHashFunctions = maxNumHashfunctions - remainingNumHashfunctions;

            return numberOfAvailableHashFunctions; 
        }

        struct QueryHandleStruct{
            struct RemoteData{
                DeviceBuffer<unsigned int> d_input_sequences;
                DeviceBuffer<int> d_input_lengths;
                DeviceBuffer<Value> d_results;
                DeviceBuffer<int> d_numResultsPerSequence;
                DeviceBuffer<int> d_offsets;
                DeviceBuffer<char> d_persistent;
                DeviceBuffer<char> d_temp;
                DeviceBuffer<read_number> d_readIds;
            };

            struct CallerData{
                DeviceBuffer<Value> d_results;
                DeviceBuffer<int> d_numResultsPerSequence;
                DeviceBuffer<int> d_offsets;
                DeviceBuffer<int> d_offsets_tmp;
                DeviceBuffer<char> d_temp;
                CudaEvent event;
            };

            std::vector<RemoteData> remotedataPerGpu;
            std::map<int, CallerData> callerDataMap;

            HostBuffer<int> pinnedData;

            std::vector<CudaStream> streams;
            std::vector<CudaEvent> events;

            std::vector<int> deviceIds;

            MemoryUsage getMemoryInfo() const{
                MemoryUsage mem{};

                const int numIds = deviceIds.size();

                for(int i = 0; i < numIds; i++){
                    //mem.device[deviceIds[i]] = remoteResultsPerGpu[i].capacityInBytes();
                }

                return mem;
            }
        };

        using QueryHandle = std::shared_ptr<QueryHandleStruct>;

        QueryHandle makeQueryHandle(){
            auto ptr = std::make_shared<QueryHandleStruct>();

            const int numMinhashers = sgpuMinhashers.size();

            ptr->remotedataPerGpu.resize(numMinhashers);

            ptr->streams.resize(numMinhashers);
            ptr->events.resize(numMinhashers);
            ptr->deviceIds = usableDeviceIds;

            for(int i = 0; i < numMinhashers; i++){
                DeviceSwitcher ds(sgpuMinhashers[i].getDeviceId());
                ptr->streams[i] = std::move(CudaStream{});
                ptr->events[i] = std::move(CudaEvent{cudaEventDisableTiming});

                ptr->remotedataPerGpu[i] = std::move(QueryHandleStruct::RemoteData{});
                ptr->callerDataMap.emplace(sgpuMinhashers[i].getDeviceId(), std::move(QueryHandleStruct::CallerData{}));
            }

            return ptr;
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
        ) const{
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
        ) const{
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

            DeviceSwitcher globalds(deviceId);

            queryHandle->callerDataMap[deviceId].event.synchronize(); //Ensure that handle is not in use by a previous call (may need to reallocate memory)


            const int numUsable = usableDeviceIds.size();
            queryHandle->pinnedData.resize(2*numUsable + numUsable + 1);

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
            for(int d = 0; d < numUsable; d++){
                DeviceSwitcher ds(usableDeviceIds[d]);
                queryHandle->events[d].record(stream);
                queryHandle->streams[d].waitEvent(queryHandle->events[d], 0);
            }

            //resize remote buffers for input data
            for(int d = 0; d < numUsable; d++){
                DeviceSwitcher ds(usableDeviceIds[d]);
                queryHandle->remotedataPerGpu[d].d_input_lengths.resize(numSequences);
                queryHandle->remotedataPerGpu[d].d_input_sequences.resize(encodedSequencePitchInInts * numSequences);
                
                if(d_readIds != nullptr){
                    queryHandle->remotedataPerGpu[d].d_readIds.resize(numSequences);
                }
            }

            //broadcast input to remote buffers
            for(int d = 0; d < numUsable; d++){                
                DeviceSwitcher ds(usableDeviceIds[d]);

                cudaMemcpyAsync(
                    queryHandle->remotedataPerGpu[d].d_input_lengths.data(),
                    d_sequenceLengths,
                    sizeof(int) * numSequences,
                    D2D,
                    queryHandle->streams[d]
                ); CUERR;

                cudaMemcpyAsync(
                    queryHandle->remotedataPerGpu[d].d_input_sequences.data(),
                    d_sequenceData2Bit,
                    sizeof(unsigned int) * encodedSequencePitchInInts * numSequences,
                    D2D,
                    queryHandle->streams[d]
                ); CUERR;

                if(d_readIds != nullptr){
                    cudaMemcpyAsync(
                        queryHandle->remotedataPerGpu[d].d_readIds.data(),
                        d_readIds,
                        sizeof(read_number) * numSequences,
                        D2D,
                        queryHandle->streams[d]
                    ); CUERR;
                }
            }

            //resize remote buffers numValuesPerSequence, offsets
            for(int d = 0; d < numUsable; d++){
                DeviceSwitcher ds(usableDeviceIds[d]);
                queryHandle->remotedataPerGpu[d].d_numResultsPerSequence.resize(numSequences);
                queryHandle->remotedataPerGpu[d].d_offsets.resize(numSequences+1);
            }

            //Determine total number of values, number of values per key, max(number of values per key)
            for(int d = 0; d < numUsable; d++){
                DeviceSwitcher ds(usableDeviceIds[d]);

                const auto& minhasher = sgpuMinhashers[d];

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

            //Retrieve the values
            for(int d = 0; d < numUsable; d++){
                DeviceSwitcher ds(usableDeviceIds[d]);

                const auto& minhasher = sgpuMinhashers[d];

                std::size_t persistent_storage_bytes = queryHandle->remotedataPerGpu[d].d_persistent.sizeInBytes();
                std::size_t temp_storage_bytes = queryHandle->remotedataPerGpu[d].d_temp.sizeInBytes();

                int* const myNumValuesPerSequence = queryHandle->remotedataPerGpu[d].d_numResultsPerSequence.data();                
                int* const myOffsets = queryHandle->remotedataPerGpu[d].d_offsets.data();
                const read_number* const myReadIds = (d_readIds == nullptr) ? nullptr: queryHandle->remotedataPerGpu[d].d_readIds.data();

                int* const myPinnedData = queryHandle->pinnedData + 2*d;
                
                CudaStream& myStream = queryHandle->streams[d];

                cudaStreamSynchronize(myStream); CUERR; //Wait for number of values and max segment

                const int totalNumValues = myPinnedData[0];
                const int largestSegment = myPinnedData[1];

                queryHandle->remotedataPerGpu[d].d_results.resize(totalNumValues);
                read_number* const myValues = queryHandle->remotedataPerGpu[d].d_results.data();

                minhasher.retrieveValues(
                    queryHandle->remotedataPerGpu[d].d_persistent.data(),
                    persistent_storage_bytes,
                    nullptr,
                    temp_storage_bytes,
                    d_readIds,
                    numSequences,
                    123456, //unused
                    myStream,
                    totalNumValues,
                    123456, //unused largest segment
                    myValues,
                    myNumValuesPerSequence,
                    myOffsets //numSequences + 1
                );

                queryHandle->remotedataPerGpu[d].d_temp.resize(temp_storage_bytes);

                minhasher.retrieveValues(
                    queryHandle->remotedataPerGpu[d].d_persistent.data(),
                    persistent_storage_bytes,
                    nullptr,
                    temp_storage_bytes,
                    d_readIds,
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
        
            //early exit if there is only one minhasher. Simply copy remote results to destination buffers and return
            if(numUsable == 1){        
                const int d = 0;        
                DeviceSwitcher ds(usableDeviceIds[d]);
                const int numValues = queryHandle->pinnedData[2*d];

                cudaMemcpyAsync(
                    d_values,
                    queryHandle->remotedataPerGpu[d].d_results.data(),
                    sizeof(Value) * numValues,
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
                cudaStreamWaitEvent(stream, queryHandle->events[d], 0); //join the internal stream
                return;
            }

            auto& callerData = queryHandle->callerDataMap[deviceId]; //Implicit creation of node is safe because deviceId is currently set

            int totalNumValues = 0;
            for(int d = 0; d < numUsable; d++){
                const int* const myPinnedData = queryHandle->pinnedData + 2*d;
                totalNumValues += myPinnedData[0];
            }

            callerData.d_results.resize(totalNumValues);
            callerData.d_numResultsPerSequence.resize(numSequences * numUsable);
            callerData.d_offsets.resize((numSequences + 1) * numUsable);
            callerData.d_offsets_tmp.resize(numSequences * numUsable);

            int* const h_numPerGpu_ps = &queryHandle->pinnedData[2*numUsable];

            int runningSum = 0;
            h_numPerGpu_ps[0] = 0;
            //Copy remote results to caller
            for(int d = 0; d < numUsable; d++){                
                DeviceSwitcher ds(usableDeviceIds[d]);
                const int numValues = queryHandle->pinnedData[2*d];

                cudaMemcpyAsync(
                    callerData.d_results.data() + runningSum,
                    queryHandle->remotedataPerGpu[d].d_results.data(),
                    sizeof(Value) * numValues,
                    D2D,
                    queryHandle->streams[d]
                ); CUERR;

                runningSum += numValues;
                if(d < numUsable - 1){
                    h_numPerGpu_ps[d+1] = runningSum;
                }

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
                cudaStreamWaitEvent(stream, queryHandle->events[d], 0); //join the internal stream
            }

            cudaMemcpyAsync(callerData.d_offsets_tmp.data(), h_numPerGpu_ps, sizeof(int) * numUsable, H2D, stream);

            //reduce number of results per sequence of different gpus
            helpers::lambda_kernel<<<SDIV(numSequences, 128), 128, 0, stream>>>(
                [
                    inputsegmentsizes = callerData.d_numResultsPerSequence.data(),
                    outputsegmentsizes = d_numValuesPerSequence,
                    numSequences,
                    numUsable
                ] __device__ (){

                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    const int stride = blockDim.x * gridDim.x;

                    for(int i = tid; i < numSequences; i += stride){
                        int sum = 0;

                        for(int r = 0; r < numUsable; r++){
                            sum += inputsegmentsizes[r * numSequences + i];
                        }

                        outputsegmentsizes[i] = sum;
                    }
                }
            );

            //Allocate temp storage for the remaining steps
            std::size_t temp_storage_bytes = 0;

            GpuSegmentedUnique::unique(
                nullptr, //tempstorage
                temp_storage_bytes,
                (Value*)nullptr, //input values
                totalNumValues,
                (Value*)nullptr, //output values
                (int*)nullptr, //output segment sizes
                numSequences,
                123456, //sizeOfLargestSegment, unused for memory calculation
                (int*)nullptr, //segment begin offsets
                (int*)nullptr, //segment end offsets
                0,
                sizeof(Value) * 8,
                stream
            );

            temp_storage_bytes = std::max(temp_storage_bytes, getRequiredCubTempSizeForResultMerging(numSequences));
            callerData.d_temp.resize(temp_storage_bytes);

            //compute begin offsets of contiguous segments per sequence   
            cudaMemsetAsync(d_offsets, 0, sizeof(int), stream); CUERR;

            cub::DeviceScan::InclusiveSum(
                callerData.d_temp.data(),
                temp_storage_bytes,
                d_numValuesPerSequence,
                d_offsets + 1,
                numSequences,
                stream
            );

            int* h_maxSegmentSize = queryHandle->pinnedData + 2 * numUsable + numUsable;

            cudaMemcpyAsync(
                h_maxSegmentSize,
                d_offsets + numSequences,
                sizeof(int),
                D2H,
                stream
            );

            helpers::lambda_kernel<<<numSequences, 128, 0, stream>>>(
                [
                    inputdata = callerData.d_results.data(),
                    inputoffsets = callerData.d_offsets.data(),
                    inputsegmentsizes = callerData.d_numResultsPerSequence.data(),
                    inputnumpergpuPS = callerData.d_offsets_tmp.data(),
                    outputdata = d_values,
                    outputbeginoffsets = d_offsets,
                    numSequences,
                    numUsable
                ] __device__ (){
                    
                    for(int i = blockIdx.x; i < numSequences; i += gridDim.x){
                        const int beginoffset = outputbeginoffsets[i];

                        int runningOffset = 0;

                        for(int r = 0; r < numUsable; r++){
                            const int segmentsize = inputsegmentsizes[r * numSequences + i];
                            const int inputOffset = inputoffsets[r * (numSequences+1) + i];
                            const int gpuOffset = inputnumpergpuPS[r];

                            const Value* myinput = inputdata + gpuOffset + inputOffset;

                            for(int k = threadIdx.x; k < segmentsize; k += blockDim.x){
                                outputdata[beginoffset + runningOffset + k] = myinput[k];
                            }

                            runningOffset += segmentsize;
                        }
                    }
                }
            ); CUERR;

            cudaStreamSynchronize(stream); CUERR; //wait for h_maxSegmentSize

            //values of same sequence are now stored in contiguous locations. make unique
            GpuSegmentedUnique::unique(
                callerData.d_temp.data(),
                temp_storage_bytes,
                d_values, //input values
                totalNumValues,
                callerData.d_results.data(), //output values
                d_numValuesPerSequence, //output segment sizes
                numSequences,
                *h_maxSegmentSize,
                d_offsets, //segment begin offsets
                d_offsets + 1, //segment end offsets
                0,
                sizeof(Value) * 8,
                stream
            );

            //compute final offsets. d_offsets[0] == 0 from previous memset call
            cub::DeviceScan::InclusiveSum(
                callerData.d_temp.data(),
                temp_storage_bytes,
                d_numValuesPerSequence,
                d_offsets + 1,
                numSequences,
                stream
            );

            //copy results to destintation
            helpers::lambda_kernel<<<4096, 128, 0, stream>>>(
                [
                    numElementsPtr = d_offsets + numSequences,
                    input = callerData.d_results.data(),
                    output = d_values
                ] __device__ (){
                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    const int stride = blockDim.x * gridDim.x;

                    const int numElements = *numElementsPtr;
                    for(int i = tid; i < numElements; i += stride){
                        output[i] = input[i];
                    }
                }
            ); CUERR;
        
            callerData.event.record(stream);
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

        void compact(){
            for(auto& minhasher : sgpuMinhashers){
                DeviceSwitcher ds(minhasher.getDeviceId());

                minhasher.compact();
            }            
        }

        void finalize(){
            compact();
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage mem{};

            for(const auto& minhasher : sgpuMinhashers){
                mem += minhasher.getMemoryInfo();
            }

            return mem;
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

        constexpr int getNumResultsPerMapThreshold() const noexcept{
            return resultsPerMapThreshold;
        }
        
        int getNumberOfMaps() const noexcept{
            int num = 0;
            for(const auto& minhasher : sgpuMinhashers){
                num += minhasher.getNumberOfMaps();
            }
            return num;
        }

        void destroy(){
            for(auto& minhasher : sgpuMinhashers){
                DeviceSwitcher sd(minhasher.getDeviceId());
                minhasher.destroy();
            }
        }

        int maxNumKeys{};
        int kmerSize{};
        int resultsPerMapThreshold{};
        std::vector<int> deviceIds;
        std::vector<int> usableDeviceIds;
        std::vector<HostBuffer<int>> vec_h_currentHashFunctionNumbers{};
        std::vector<SingleGpuMinhasher> sgpuMinhashers{};
    };


}
}




#endif