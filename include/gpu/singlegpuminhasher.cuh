#ifndef CARE_SINGLE_GPU_MINHASHER_CUH
#define CARE_SINGLE_GPU_MINHASHER_CUH

#include <config.hpp>

#include <warpcore.cuh>

#include <gpu/distributedreadstorage.hpp>
#include <gpu/cuda_unique.cuh>
#include <cpuhashtable.hpp>
#include <gpu/gpuhashtable.cuh>
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


    class SingleGpuMinhasher{
    public:
        using Key = kmer_type;
        using Value = read_number;

        using GpuTable = GpuHashtable<Key, Value>;

        using DeviceSwitcher = cub::SwitchDevice;

        template<class T>
        using HostBuffer = helpers::SimpleAllocationPinnedHost<T, 5>;
        template<class T>
        using DeviceBuffer = helpers::SimpleAllocationDevice<T, 5>;


        SingleGpuMinhasher(int maxNumKeys_, int maxValuesPerKey, int k)
            : maxNumKeys(maxNumKeys_), kmerSize(k), resultsPerMapThreshold(maxValuesPerKey)
        {
            cudaGetDevice(&deviceId); CUERR;
        }

        int constructFromReadStorage(
            const RuntimeOptions &runtimeOptions,
            std::uint64_t nReads,
            const DistributedReadStorage& gpuReadStorage,
            int upperBoundSequenceLength,
            int maxNumHashfunctions,
            int hashFunctionOffset = 0
        ){
            
            DeviceSwitcher ds(deviceId);

            gpuHashTables.clear();


            constexpr read_number parallelReads = 1000000;
            const read_number numReads = nReads;
            const int numIters = SDIV(numReads, parallelReads);
            const std::size_t encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(upperBoundSequenceLength);

            const int numThreads = runtimeOptions.threads;
            ThreadPool::ParallelForHandle pforHandle;
            ThreadPool threadPool(numThreads);

            helpers::SimpleAllocationDevice<unsigned int, 1> d_sequenceData(encodedSequencePitchInInts * parallelReads);
            helpers::SimpleAllocationDevice<int, 0> d_lengths(parallelReads);

            helpers::SimpleAllocationPinnedHost<read_number, 0> h_indices(parallelReads);
            helpers::SimpleAllocationDevice<read_number, 0> d_indices(parallelReads);

            std::size_t insert_temp_size = 0;
            insert(
                nullptr,
                insert_temp_size,
                (const unsigned int*)nullptr,
                int(parallelReads),
                (const int*)nullptr,
                encodedSequencePitchInInts,
                (const read_number*)nullptr,
                0,
                maxNumHashfunctions,
                (const int*)nullptr,
                (cudaStream_t)0
            );

            helpers::SimpleAllocationDevice<char, 0> d_temp(insert_temp_size);

            std::vector<int> usedHashFunctionNumbers;

            CudaStream stream{};

            auto sequencehandle = gpuReadStorage.makeGatherHandleSequences();

            // auto showProgress = [&](auto totalCount, auto seconds){
            //     if(runtimeOptions.showProgress){
            //         std::cout << "Hashed " << totalCount << " / " << numReads << " reads. Elapsed time: " 
            //                 << seconds << " seconds.\n";
            //     }
            // };

            // auto updateShowProgressInterval = [](auto duration){
            //     return duration * 2;
            // };

            int remainingHashFunctions = maxNumHashfunctions;
            bool keepGoing = true;

            while(remainingHashFunctions > 0 && keepGoing){
                const int alreadyExistingHashFunctions = maxNumHashfunctions - remainingHashFunctions;
                int addedHashFunctions = addHashfunctions(remainingHashFunctions + 1);

                if(addedHashFunctions == 0){
                    keepGoing = false;
                    break;
                    //throw std::runtime_error("Unable to construct a single gpu hashtable. Abort.");
                }else{
                    // safety for memory
                    addedHashFunctions -= 1;
                    gpuHashTables.pop_back();

                    if(addedHashFunctions == 0){
                        keepGoing = false;
                        break;
                    }
                }

                //ProgressThread<read_number> progressThread(numReads, showProgress, updateShowProgressInterval);

                std::cout << "Constructing maps: ";
                for(int i = 0; i < addedHashFunctions; i++){
                    std::cout << (alreadyExistingHashFunctions + i) << "(" << (hashFunctionOffset + alreadyExistingHashFunctions + i) << ") ";
                }
                std::cout << '\n';

                std::vector<int> h_hashfunctionNumbers(addedHashFunctions);
                std::iota(
                    h_hashfunctionNumbers.begin(),
                    h_hashfunctionNumbers.end(),
                    alreadyExistingHashFunctions + hashFunctionOffset
                );

                usedHashFunctionNumbers.insert(usedHashFunctionNumbers.end(), h_hashfunctionNumbers.begin(), h_hashfunctionNumbers.end());

                for (int iter = 0; iter < numIters; iter++){
                    read_number readIdBegin = iter * parallelReads;
                    read_number readIdEnd = std::min((iter + 1) * parallelReads, numReads);

                    const std::size_t curBatchsize = readIdEnd - readIdBegin;

                    std::iota(h_indices.get(), h_indices.get() + curBatchsize, readIdBegin);

                    cudaMemcpyAsync(d_indices, h_indices, sizeof(read_number) * curBatchsize, H2D, stream); CUERR;

                    gpuReadStorage.gatherSequenceDataToGpuBufferAsync(
                        &threadPool,
                        sequencehandle,
                        d_sequenceData,
                        encodedSequencePitchInInts,
                        h_indices,
                        d_indices,
                        curBatchsize,
                        deviceId,
                        stream
                    );
                
                    gpuReadStorage.gatherSequenceLengthsToGpuBufferAsync(
                        d_lengths,
                        deviceId,
                        d_indices,
                        curBatchsize,
                        stream
                    );

                    insert(
                        d_temp.data(),
                        insert_temp_size,
                        d_sequenceData,
                        curBatchsize,
                        d_lengths,
                        encodedSequencePitchInInts,
                        d_indices,
                        alreadyExistingHashFunctions,
                        addedHashFunctions,
                        h_hashfunctionNumbers.data(),
                        stream
                    );

                    cudaStreamSynchronize(stream); CUERR;

                    //progressThread.addProgress(curBatchsize);
                }

                std::cerr << "Compacting\n";
                finalize();

                //progressThread.finished();

                remainingHashFunctions -= addedHashFunctions;
            }

            const int numberOfAvailableHashFunctions = maxNumHashfunctions - remainingHashFunctions;

            h_currentHashFunctionNumbers.resize(numberOfAvailableHashFunctions);
            std::copy(usedHashFunctionNumbers.begin(), usedHashFunctionNumbers.end(), h_currentHashFunctionNumbers.begin());

            return numberOfAvailableHashFunctions; 
        }

#if 0
        void constructFromReadStorage(
            const RuntimeOptions &runtimeOptions,
            std::uint64_t nReads,
            const DistributedReadStorage& gpuReadStorage,
            int upperBoundSequenceLength,
            int firstHashFunc,
            int numHashFuncs
        ){
            assert(firstHashFunc + numHashFuncs <= int(gpuHashTables.size()));

            DeviceSwitcher ds(deviceId);

            constexpr read_number parallelReads = 1000000;
            const read_number numReads = nReads;
            const int numIters = SDIV(numReads, parallelReads);
            const std::size_t encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(upperBoundSequenceLength);

            const int numThreads = runtimeOptions.threads;

            ThreadPool::ParallelForHandle pforHandle;

            std::cout << "Constructing maps: ";
            for(int i = 0; i < numHashFuncs; i++){
                std::cout << (firstHashFunc + i) << ' ';
            }
            std::cout << '\n';

            auto showProgress = [&](auto totalCount, auto seconds){
                if(runtimeOptions.showProgress){
                    std::cout << "Hashed " << totalCount << " / " << numReads << " reads. Elapsed time: " 
                            << seconds << " seconds.\n";
                }
            };

            auto updateShowProgressInterval = [](auto duration){
                return duration * 2;
            };

            ProgressThread<read_number> progressThread(numReads, showProgress, updateShowProgressInterval);

            ThreadPool threadPool(numThreads);

            helpers::SimpleAllocationDevice<unsigned int, 1> d_sequenceData(encodedSequencePitchInInts * parallelReads);
            helpers::SimpleAllocationDevice<int, 0> d_lengths(parallelReads);

            helpers::SimpleAllocationPinnedHost<read_number, 0> h_indices(parallelReads);
            helpers::SimpleAllocationDevice<read_number, 0> d_indices(parallelReads);

            CudaStream stream{};

            auto sequencehandle = gpuReadStorage.makeGatherHandleSequences();

            for (int iter = 0; iter < numIters; iter++){
                read_number readIdBegin = iter * parallelReads;
                read_number readIdEnd = std::min((iter + 1) * parallelReads, numReads);

                const std::size_t curBatchsize = readIdEnd - readIdBegin;

                std::iota(h_indices.get(), h_indices.get() + curBatchsize, readIdBegin);

                cudaMemcpyAsync(d_indices, h_indices, sizeof(read_number) * curBatchsize, H2D, stream); CUERR;

                gpuReadStorage.gatherSequenceDataToGpuBufferAsync(
                    &threadPool,
                    sequencehandle,
                    d_sequenceData,
                    encodedSequencePitchInInts,
                    h_indices,
                    d_indices,
                    curBatchsize,
                    deviceId,
                    stream
                );
            
                gpuReadStorage.gatherSequenceLengthsToGpuBufferAsync(
                    d_lengths,
                    deviceId,
                    d_indices,
                    curBatchsize,
                    stream
                );

                insert(
                    d_sequenceData,
                    curBatchsize,
                    d_lengths,
                    encodedSequencePitchInInts,
                    d_indices,
                    firstHashFunc,
                    numHashFuncs,
                    stream
                );

                cudaStreamSynchronize(stream); CUERR;

                progressThread.addProgress(curBatchsize);
            }

            progressThread.finished();

            std::cerr << "Compacting\n";
            finalize();
        }

#endif 
        int addHashfunctions(int numExtraFunctions){
            
            DeviceSwitcher ds(deviceId);

            int added = 0;
            int cur = gpuHashTables.size();

            assert(!(numExtraFunctions + cur > 64));

            for(int i = 0; i < numExtraFunctions; i++){
                auto ptr = std::make_unique<GpuTable>(std::size_t(maxNumKeys / getLoad()),
                    getLoad(),
                    resultsPerMapThreshold);

                auto status = ptr->pop_status((cudaStream_t)0);
                cudaDeviceSynchronize(); CUERR;
                if(status.has_any_errors()){
                    std::cerr << "observed error when initialiting hash function " << (gpuHashTables.size() + 1) << " : " << i << ", " << status << "\n";
                    break;
                }else{

                    assert(!status.has_any_errors()); 
                    //TODO errorhandling

                    gpuHashTables.emplace_back(std::move(ptr));

                    added++;
                }
            }

            return added;
        }

        void insert(
            void* d_temp,
            std::size_t& temp_storage_bytes,
            const unsigned int* d_sequenceData2Bit,
            int numSequences,
            const int* d_sequenceLengths,
            std::size_t encodedSequencePitchInInts,
            const read_number* d_readIds,
            int firstHashfunction,
            int numHashfunctions,
            const int* h_hashFunctionNumbers,
            cudaStream_t stream
        ){

            const std::size_t signaturesRowPitchElements = numHashfunctions;

            void* temp_allocations[3];
            std::size_t temp_allocation_sizes[3];
            
            temp_allocation_sizes[0] = sizeof(std::uint64_t) * numHashfunctions * numSequences; // d_sig
            temp_allocation_sizes[1] = sizeof(std::uint64_t) * numHashfunctions * numSequences; // d_sig_trans
            temp_allocation_sizes[2] = sizeof(int) * numHashfunctions; // d_hashFunctionNumbers
            
            cudaError_t cubstatus = cub::AliasTemporaries(
                d_temp,
                temp_storage_bytes,
                temp_allocations,
                temp_allocation_sizes
            );
            assert(cubstatus == cudaSuccess);

            if(d_temp == nullptr){
                return;
            }

            assert(firstHashfunction + numHashfunctions <= int(gpuHashTables.size()));

            DeviceSwitcher ds(deviceId);

            std::uint64_t* const d_signatures = static_cast<std::uint64_t*>(temp_allocations[0]);
            std::uint64_t* const d_signatures_transposed = static_cast<std::uint64_t*>(temp_allocations[1]);
            int* const d_hashFunctionNumbers = static_cast<int*>(temp_allocations[2]);
            
            cudaMemcpyAsync(
                d_hashFunctionNumbers, 
                h_hashFunctionNumbers, 
                sizeof(int) * numHashfunctions, 
                H2D, 
                stream
            ); CUERR;

            callMinhashSignaturesKernel(
                d_signatures,
                signaturesRowPitchElements,
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                getKmerSize(),
                numHashfunctions,
                d_hashFunctionNumbers,
                stream
            ); CUERR;

            helpers::call_transpose_kernel(
                d_signatures_transposed, 
                d_signatures, 
                numSequences, 
                signaturesRowPitchElements, 
                signaturesRowPitchElements,
                stream
            );

            fixKeysForGpuHashTable(d_signatures_transposed, numSequences * numHashfunctions, stream);

            for(int i = 0; i < numHashfunctions; i++){
                gpuHashTables[firstHashfunction + i]->insert(
                    d_signatures_transposed + i * numSequences,
                    d_readIds,
                    numSequences,
                    stream
                );
            }

            cudaStreamSynchronize(stream);

            for(int i = 0; i < numHashfunctions; i++){
                auto status = gpuHashTables[firstHashfunction + i]->pop_status(stream);
                cudaStreamSynchronize(stream);

                if(status.has_any_errors()){
                    std::cerr << "Error table " << (firstHashfunction + i) << " after insertion: " << status << "\n";
                }
            }
        }

        struct QueryHandleStruct{
            int deviceId;
            DeviceBuffer<char> d_singletempbuffer;
            DeviceBuffer<char> d_singlepersistentbuffer;

            MemoryUsage getMemoryInfo() const{
                MemoryUsage mem{};

                mem.device[deviceId] += d_singletempbuffer.capacityInBytes();
                mem.device[deviceId] += d_singlepersistentbuffer.capacityInBytes();

                return mem;
            }
        };

        using QueryHandle = std::shared_ptr<QueryHandleStruct>;

        static QueryHandle makeQueryHandle(){
            auto ptr = std::make_shared<QueryHandleStruct>();
            cudaGetDevice(&ptr->deviceId); CUERR;
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
            int /*deviceId*/, 
            cudaStream_t stream,
            read_number* d_values,
            int* d_numValuesPerSequence,
            int* d_offsets //numSequences + 1
        ) const {
           
            int totalNumValues = 0;

            std::size_t persistent_storage_bytes = 0;
            std::size_t temp_storage_bytes = 0;

            determineNumValues(
                nullptr,
                persistent_storage_bytes,
                nullptr,
                temp_storage_bytes,
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                d_sequenceLengths,
                numSequences,
                d_numValuesPerSequence,
                totalNumValues,
                stream
            );

            queryHandle->d_singlepersistentbuffer.resize(persistent_storage_bytes);
            queryHandle->d_singletempbuffer.resize(temp_storage_bytes);

            determineNumValues(
                queryHandle->d_singlepersistentbuffer.data(),
                persistent_storage_bytes,
                queryHandle->d_singletempbuffer.data(),
                temp_storage_bytes,
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                d_sequenceLengths,
                numSequences,
                d_numValuesPerSequence,
                totalNumValues,
                stream
            );

            cudaStreamSynchronize(stream); CUERR;

            if(totalNumValues == 0){
                return;
            }

            retrieveValues(
                queryHandle->d_singlepersistentbuffer.data(),
                persistent_storage_bytes,
                nullptr,
                temp_storage_bytes,
                d_readIds,
                numSequences,
                123456, //unused
                stream,
                totalNumValues,
                123456, //unused
                d_values,
                d_numValuesPerSequence,
                d_offsets //numSequences + 1
            );

            std::size_t cubsize = 0;
            cub::DeviceReduce::Max(
                nullptr, 
                cubsize, 
                d_numValuesPerSequence, 
                (int*)nullptr, 
                numSequences, 
                stream
            );
            cubsize = SDIV(cubsize, 4) * 4;

            temp_storage_bytes = std::max(temp_storage_bytes, cubsize + sizeof(int));

            queryHandle->d_singletempbuffer.resize(temp_storage_bytes);

            int* d_maxvalue = (int*)(queryHandle->d_singletempbuffer.data() + cubsize);

            cub::DeviceReduce::Max(
                queryHandle->d_singletempbuffer.data(), 
                temp_storage_bytes, 
                d_numValuesPerSequence, 
                d_maxvalue, 
                numSequences, 
                stream
            );
            
            int sizeOfLargestSegment = 0;
            cudaMemcpyAsync(&sizeOfLargestSegment, d_maxvalue, sizeof(int), D2H, stream); CUERR;
            cudaStreamSynchronize(stream);

            retrieveValues(
                queryHandle->d_singlepersistentbuffer.data(),
                persistent_storage_bytes,
                queryHandle->d_singletempbuffer.data(),
                temp_storage_bytes,
                d_readIds,
                numSequences,
                123456, //unused
                stream,
                totalNumValues,
                sizeOfLargestSegment,
                d_values,
                d_numValuesPerSequence,
                d_offsets //numSequences + 1
            );
        }


        void determineNumValues(
            void* persistent_storage,            
            std::size_t& persistent_storage_bytes,
            void* temp_storage,
            std::size_t& temp_storage_bytes,
            const unsigned int* d_sequenceData2Bit,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int* d_numValuesPerSequence,
            int& totalNumValues,
            cudaStream_t stream
        ) const {

            const int numHashfunctions = gpuHashTables.size();
            const std::size_t signaturesRowPitchElements = numHashfunctions;

            void* persistent_allocations[3]{};
            std::size_t persistent_allocation_sizes[3]{};

            persistent_allocation_sizes[0] = sizeof(std::uint64_t) * numHashfunctions * numSequences; // d_sig_trans
            persistent_allocation_sizes[1] = sizeof(int) * numSequences * numHashfunctions; // d_numValuesPerSequencePerHash
            persistent_allocation_sizes[2] = sizeof(int) * numSequences * numHashfunctions; // d_numValuesPerSequencePerHashExclPSVert

            cudaError_t cubstatus = cub::AliasTemporaries(
                persistent_storage,
                persistent_storage_bytes,
                persistent_allocations,
                persistent_allocation_sizes
            );
            assert(cubstatus == cudaSuccess);

            std::size_t cubtempbytes = 0;
            cub::DeviceReduce::Sum(
                nullptr, 
                cubtempbytes, 
                (int*)nullptr, 
                (int*)nullptr, 
                numSequences, 
                stream
            );

            void* temp_allocations[4];
            std::size_t temp_allocation_sizes[4];
            
            temp_allocation_sizes[0] = sizeof(int) * numHashfunctions; // d_hashFunctionNumbers
            temp_allocation_sizes[1] = cubtempbytes; // d_cub_temp
            temp_allocation_sizes[2] = sizeof(std::uint64_t) * numHashfunctions * numSequences; // d_sig
            temp_allocation_sizes[3] = sizeof(int); // d_cub_sum
            
            cubstatus = cub::AliasTemporaries(
                temp_storage,
                temp_storage_bytes,
                temp_allocations,
                temp_allocation_sizes
            );
            assert(cubstatus == cudaSuccess);

            if(persistent_storage == nullptr || temp_storage == nullptr){
                return;
            }

            std::uint64_t* const d_signatures_transposed = static_cast<std::uint64_t*>(persistent_allocations[0]);
            int* const d_numValuesPerSequencePerHash = static_cast<int*>(persistent_allocations[1]);
            int* const d_numValuesPerSequencePerHashExclPSVert = static_cast<int*>(persistent_allocations[2]);

            int* const d_hashFunctionNumbers = static_cast<int*>(temp_allocations[0]);
            void* const d_cubTemp = temp_allocations[1];
            std::uint64_t* const d_signatures = static_cast<std::uint64_t*>(temp_allocations[2]);
            int* const d_cub_sum = static_cast<int*>(temp_allocations[3]);

            DeviceSwitcher ds(deviceId);

            cudaMemcpyAsync(
                d_hashFunctionNumbers,
                h_currentHashFunctionNumbers.data(), 
                sizeof(int) * numHashfunctions, 
                H2D, 
                stream
            ); CUERR;           

            dim3 block(128,1,1);
            dim3 grid(SDIV(numHashfunctions * numSequences, block.x),1,1);

            callMinhashSignaturesKernel(
                d_signatures,
                signaturesRowPitchElements,
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                getKmerSize(),
                numHashfunctions,
                d_hashFunctionNumbers,
                stream
            ); CUERR;

            helpers::call_transpose_kernel(
                d_signatures_transposed, 
                d_signatures, 
                numSequences, 
                signaturesRowPitchElements, 
                signaturesRowPitchElements,
                stream
            );

            fixKeysForGpuHashTable(d_signatures_transposed, numSequences * numHashfunctions, stream);

            //determine number of values per hashfunction per sequence
            for(int i = 0; i < numHashfunctions; i++){
                gpuHashTables[i]->numValuesPerKeyCompact(
                    d_signatures_transposed + i * numSequences,
                    numSequences,
                    d_numValuesPerSequencePerHash + i * numSequences,
                    stream
                );
            }

            // accumulate number of values per sequence in d_numValuesPerSequence
            // calculate vertical exclusive prefix sum
            helpers::lambda_kernel<<<1024, 256, 0, stream>>>(
                [=] __device__ (){
                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    const int stride = blockDim.x * gridDim.x;

                    for(int i = tid; i < numSequences; i += stride){
                        d_numValuesPerSequencePerHashExclPSVert[0 * numSequences + i] = 0;
                    }

                    for(int i = tid; i < numSequences; i += stride){
                        int vertPS = 0;
                        for(int k = 0; k < numHashfunctions; k++){
                            const int num = d_numValuesPerSequencePerHash[k * numSequences + i];

                            vertPS += num;
                            if(k < numHashfunctions - 1){
                                d_numValuesPerSequencePerHashExclPSVert[(k+1) * numSequences + i] = vertPS;
                            }else{
                                d_numValuesPerSequence[i] = vertPS;
                            }
                        }
                    }
                }
            );

            cub::DeviceReduce::Sum(
                d_cubTemp, 
                cubtempbytes, 
                d_numValuesPerSequence, 
                d_cub_sum, 
                numSequences, 
                stream
            );

            cudaMemcpyAsync(&totalNumValues, d_cub_sum, sizeof(int), D2H, stream); CUERR;
        }

        void retrieveValues(
            void* persistentbufferFromNumValues,            
            std::size_t persistent_storage_bytes,
            void* temp_storage,
            std::size_t& temp_storage_bytes,
            const read_number* d_readIds,
            int numSequences,
            int /*deviceId*/, 
            cudaStream_t stream,
            int totalNumValues,
            int sizeOfLargestSegment,
            read_number* d_values,
            int* d_numValuesPerSequence,
            int* d_offsets //numSequences + 1
        ) const {
            assert(persistentbufferFromNumValues != nullptr);

            const int numHashfunctions = gpuHashTables.size();

            std::size_t cubtempbytes = 0;

            cub::DeviceScan::InclusiveSum(
                nullptr,
                cubtempbytes,
                (int*)nullptr, 
                (int*)nullptr, 
                numSequences,
                stream
            );

            std::size_t cubtempbytes2 = 0;
            cub::DeviceReduce::Max(
                nullptr, 
                cubtempbytes2, 
                (int*)nullptr, 
                (int*)nullptr, 
                numSequences, 
                stream
            );

            cubtempbytes = std::max(cubtempbytes, cubtempbytes2);

            void* temp_allocations[5]{};
            std::size_t temp_allocation_sizes[5]{};
            
            temp_allocation_sizes[0] = cubtempbytes; // d_cub_temp
            temp_allocation_sizes[1] = sizeof(int) * (numSequences + 1); // d_cub_sum

            GpuSegmentedUnique::unique(
                nullptr,
                temp_allocation_sizes[2],
                (read_number*)nullptr,
                totalNumValues,
                (read_number*)nullptr,
                d_numValuesPerSequence,
                numSequences,
                0,
                (int*)nullptr,
                (int*)nullptr,
                0,
                sizeof(read_number) * 8,
                stream
            );

            temp_allocation_sizes[2] = std::max(temp_allocation_sizes[2], sizeof(int) * numSequences * numHashfunctions); // d_queryOffsetsPerSequencePerHash, d_uniquetemp

            temp_allocation_sizes[3] = sizeof(read_number) * totalNumValues; // d_values_tmp
            temp_allocation_sizes[4] = sizeof(int) * numSequences; // d_end_offsets
            
            cudaError_t cubstatus = cub::AliasTemporaries(
                temp_storage,
                temp_storage_bytes,
                temp_allocations,
                temp_allocation_sizes
            );
            assert(cubstatus == cudaSuccess);

            if(temp_storage == nullptr) return;

            void* persistent_allocations[3]{};
            std::size_t persistent_allocation_sizes[3]{};

            persistent_allocation_sizes[0] = sizeof(std::uint64_t) * numHashfunctions * numSequences; // d_sig_trans
            persistent_allocation_sizes[1] = sizeof(int) * numSequences * numHashfunctions; // d_numValuesPerSequencePerHash
            persistent_allocation_sizes[2] = sizeof(int) * numSequences * numHashfunctions; // d_numValuesPerSequencePerHashExclPSVert

            cubstatus = cub::AliasTemporaries(
                persistentbufferFromNumValues,
                persistent_storage_bytes,
                persistent_allocations,
                persistent_allocation_sizes
            );
            assert(cubstatus == cudaSuccess);



            DeviceSwitcher ds(deviceId);

            std::uint64_t* const d_signatures_transposed = static_cast<std::uint64_t*>(persistent_allocations[0]);
            int* const d_numValuesPerSequencePerHash = static_cast<int*>(persistent_allocations[1]);
            int* const d_numValuesPerSequencePerHashExclPSVert = static_cast<int*>(persistent_allocations[2]);

            void* const d_cubTemp = temp_allocations[0];
            int* const d_cub_sum = static_cast<int*>(temp_allocations[1]);
            int* const d_queryOffsetsPerSequencePerHash = static_cast<int*>(temp_allocations[2]);
            void* const d_uniquetemp = temp_allocations[2];
            read_number* const d_values_tmp = static_cast<read_number*>(temp_allocations[3]);
            int* const d_end_offsets = static_cast<int*>(temp_allocations[4]);
     


            //calculate global offsets for each sequence in output array
            cudaMemsetAsync(d_offsets, 0, sizeof(int), stream); CUERR;

            cub::DeviceScan::InclusiveSum(
                d_cubTemp,
                cubtempbytes,
                d_numValuesPerSequence,
                d_offsets + 1,
                numSequences,
                stream
            );

            // compute destination offsets for each hashtable such that values of different tables 
            // for the same sequence are stored contiguous in the result array

            helpers::lambda_kernel<<<1024, 256, 0, stream>>>(
                [=] __device__ (){
                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
                    const int stride = blockDim.x * gridDim.x;

                    for(int i = tid; i < numSequences; i += stride){
                        
                        const int base = d_offsets[i];

                        //k == 0 is a copy from d_offsets
                        d_queryOffsetsPerSequencePerHash[0 * numSequences + i] = base;

                        for(int k = 1; k < numHashfunctions; k++){
                            d_queryOffsetsPerSequencePerHash[k * numSequences + i] = base + d_numValuesPerSequencePerHashExclPSVert[k * numSequences + i];
                        }
                    }
                }
            );

            cub::DeviceReduce::Max(
                d_cubTemp, 
                cubtempbytes, 
                d_numValuesPerSequence, 
                d_cub_sum, 
                numSequences, 
                stream
            );

            //results will be in Current() buffer
            cub::DoubleBuffer<read_number> d_values_dblbuf(d_values, d_values_tmp);

            cudaMemcpyAsync(d_end_offsets, d_offsets + 1, sizeof(int) * numSequences, D2D, stream); CUERR;

            //retrieve values

            for(int i = 0; i < numHashfunctions; i++){
                gpuHashTables[i]->retrieveCompact(
                    d_signatures_transposed + i * numSequences,
                    d_queryOffsetsPerSequencePerHash  + i * numSequences,
                    d_numValuesPerSequencePerHash + i * numSequences,
                    numSequences,
                    d_values_dblbuf.Current(),
                    stream
                );
            }

            // all values for the same key are stored in consecutive locations in d_values_tmp.
            // now, make value ranges unique

            GpuSegmentedUnique::unique(
                d_uniquetemp,
                temp_allocation_sizes[2],
                d_values_dblbuf.Current(), //input values
                totalNumValues,
                d_values_dblbuf.Alternate(), //output values
                d_numValuesPerSequence, //output segment sizes
                numSequences,
                sizeOfLargestSegment,
                d_offsets, //device accessible
                d_end_offsets, //device accessible
                0,
                sizeof(read_number) * 8,
                stream
            );

            if(d_readIds != nullptr){

                // State: d_values contains unique values per sequence from all tables. num unique values per sequence are computed in d_numValuesPerSequence
                // Segment of values for sequence i begins at d_offsets[i]
                // Remove d_readIds[i] from segment i, if present. Operation is performed inplace

                callFindAndRemoveFromSegmentKernel<read_number,128,4>(
                    d_readIds,
                    d_values_dblbuf.Alternate(),
                    numSequences,
                    d_numValuesPerSequence,
                    d_offsets,
                    stream
                );

            }

            //copy values to compact array

            //repurpose
            int* d_newOffsets = d_cub_sum;

            cudaMemsetAsync(d_newOffsets, 0, sizeof(int), stream); CUERR;

            cub::DeviceScan::InclusiveSum(
                d_cubTemp,
                cubtempbytes,
                d_numValuesPerSequence,
                d_newOffsets + 1,
                numSequences,
                stream
            );

            helpers::lambda_kernel<<<numSequences, 128, 0, stream>>>(
                [
                    d_values_in = d_values_dblbuf.Alternate(),
                    d_values_out = d_values_dblbuf.Current(),
                    numSequences,
                    d_numValuesPerSequence,
                    d_offsets,
                    d_newOffsets
                ] __device__ (){

                    for(int s = blockIdx.x; s < numSequences; s += gridDim.x){
                        const int numValues = d_numValuesPerSequence[s];
                        const int inOffset = d_offsets[s];
                        const int outOffset = d_newOffsets[s];

                        for(int c = threadIdx.x; c < numValues; c += blockDim.x){
                            d_values_out[outOffset + c] = d_values_in[inOffset + c];    
                        }
                    }
                }
            ); CUERR;

            cudaMemcpyAsync(d_offsets, d_newOffsets, sizeof(int) * (numSequences+1), D2D, stream); CUERR;
        }


        void compact(cudaStream_t stream = 0){
            DeviceSwitcher ds(deviceId);

            std::size_t required_temp_bytes = 0;

            for(auto& table : gpuHashTables){
                std::size_t temp_bytes2 = 0;
                table->compact(nullptr, temp_bytes2, stream);
                required_temp_bytes = std::max(required_temp_bytes, temp_bytes2);
            }

            std::size_t freeMem, totalMem; 
            cudaMemGetInfo(&freeMem, &totalMem); CUERR;

            void* temp = nullptr;
            if(required_temp_bytes < freeMem){
                cudaMalloc(&temp, required_temp_bytes); CUERR;
            }else{
                cudaMallocManaged(&temp, required_temp_bytes); CUERR;
            }

            for(auto& table : gpuHashTables){
                table->compact(temp, required_temp_bytes, stream);
            }

            cudaFree(temp); CUERR;
        }

        void finalize(cudaStream_t stream = 0){
            compact(stream);
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage mem{};

            for(const auto& table : gpuHashTables){
                mem += table->getMemoryInfo();
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
            return gpuHashTables.size();
        }

        void destroy(){
            DeviceSwitcher sd(getDeviceId());
            gpuHashTables.clear();
        }

        constexpr int getDeviceId() const noexcept{
            return deviceId;
        }

        int deviceId{};
        int maxNumKeys{};
        int kmerSize{};
        int resultsPerMapThreshold{};
        HostBuffer<int> h_currentHashFunctionNumbers{};
        std::vector<std::unique_ptr<GpuTable>> gpuHashTables{};
    };


}
}




#endif