#ifdef CARE_HAS_WARPCORE

#ifndef CARE_SINGLE_GPU_MINHASHER_CUH
#define CARE_SINGLE_GPU_MINHASHER_CUH

#include <config.hpp>

#include <gpu/gpureadstorage.cuh>
#include <gpu/cuda_unique.cuh>
#include <cpuhashtable.hpp>
#include <gpu/gpuhashtable.cuh>
#include <gpu/kernels.hpp>
#include <gpu/gpuminhasher.cuh>
#include <gpu/cudaerrorcheck.cuh>


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
#include <mutex>

namespace care{
namespace gpu{

    class MultiGpuMinhasher; //forward declaration

    class SingleGpuMinhasher : public GpuMinhasher{
        friend class MultiGpuMinhasher;
    private:
        using GpuTable = GpuHashtable<Key, read_number>;

        using DeviceSwitcher = cub::SwitchDevice;

        template<class T>
        using HostBuffer = helpers::SimpleAllocationPinnedHost<T, 5>;
        template<class T>
        using DeviceBuffer = helpers::SimpleAllocationDevice<T, 5>;

        struct QueryData{
            enum class Stage{
                None,
                NumValues,
                Retrieve
            };

            int deviceId{};
            Stage previousStage = Stage::None;
            int* d_numValuesPerSequence{};
            DeviceBuffer<char> d_singletempbuffer{};
            DeviceBuffer<char> d_singlepersistentbuffer{};
            DeviceBuffer<char> d_cubTemp{};


            MemoryUsage getMemoryInfo() const{
                MemoryUsage mem{};

                mem.device[deviceId] += d_singletempbuffer.capacityInBytes();
                mem.device[deviceId] += d_singlepersistentbuffer.capacityInBytes();

                return mem;
            }
        };

    public:
        using Key = kmer_type;


        SingleGpuMinhasher(int maxNumKeys_, int maxValuesPerKey, int k)
            : maxNumKeys(maxNumKeys_), kmerSize(k), resultsPerMapThreshold(maxValuesPerKey)
        {
            CUDACHECK(cudaGetDevice(&deviceId));
        }

        int constructFromReadStorage(
            const RuntimeOptions &runtimeOptions,
            std::uint64_t nReads,
            const GpuReadStorage& gpuReadStorage,
            int upperBoundSequenceLength,
            int maxNumHashfunctions,
            int hashFunctionOffset = 0
        ) {
            
            DeviceSwitcher ds(deviceId);

            gpuHashTables.clear();

            //helpers::SimpleAllocationDevice<char,0> dummy(20ull * 1024ull * 1024ull * 1024ull);


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

            auto sequencehandle = gpuReadStorage.makeHandle();

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

                    CUDACHECK(cudaMemcpyAsync(d_indices, h_indices, sizeof(read_number) * curBatchsize, H2D, stream));

                    gpuReadStorage.gatherSequences(
                        sequencehandle,
                        d_sequenceData,
                        encodedSequencePitchInInts,
                        makeAsyncConstBufferWrapper(h_indices.data()),
                        d_indices,
                        curBatchsize,
                        stream
                    );
                
                    gpuReadStorage.gatherSequenceLengths(
                        sequencehandle,
                        d_lengths,
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

                    CUDACHECK(cudaStreamSynchronize(stream));

                    //progressThread.addProgress(curBatchsize);
                }

                std::cerr << "Compacting\n";
                finalize();

                //progressThread.finished();

                remainingHashFunctions -= addedHashFunctions;
            }

            d_temp.destroy();

            const int numberOfAvailableHashFunctions = maxNumHashfunctions - remainingHashFunctions;

            h_currentHashFunctionNumbers.resize(numberOfAvailableHashFunctions);
            std::copy(usedHashFunctionNumbers.begin(), usedHashFunctionNumbers.end(), h_currentHashFunctionNumbers.begin());

            gpuReadStorage.destroyHandle(sequencehandle);

            std::vector<GpuTable::DeviceTableView> views;
            for(const auto& ptr : gpuHashTables){
                views.emplace_back(ptr->makeDeviceView());
            }

            d_deviceAccessibleTableViews.resize(numberOfAvailableHashFunctions);
            CUDACHECK(cudaMemcpyAsync(
                d_deviceAccessibleTableViews.data(),
                views.data(),
                sizeof(GpuTable::DeviceTableView) * numberOfAvailableHashFunctions,
                H2D,
                stream
            ));

            CUDACHECK(cudaStreamSynchronize(stream));

            return numberOfAvailableHashFunctions; 
        }

        std::unique_ptr<SingleGpuMinhasher> makeCopy(int targetDeviceId) const{
            DeviceSwitcher ds(targetDeviceId);

            auto result = std::make_unique<SingleGpuMinhasher>(0,0,0);
            if(!result) return nullptr;
            
            result->maxNumKeys = maxNumKeys;
            result->kmerSize = kmerSize;
            result->resultsPerMapThreshold = resultsPerMapThreshold;
            result->h_currentHashFunctionNumbers.resize(h_currentHashFunctionNumbers.size());
            std::copy(h_currentHashFunctionNumbers.begin(), h_currentHashFunctionNumbers.end(), result->h_currentHashFunctionNumbers.begin());

            std::size_t requiredTempBytes = 0;
            for(const auto& ptr : gpuHashTables){
                std::size_t bytes = ptr->getMakeCopyTempBytes();
                requiredTempBytes = std::max(requiredTempBytes, bytes);
            }

            helpers::SimpleAllocationDevice<char, 0> d_copytemp(requiredTempBytes);

            for(const auto& ptr : gpuHashTables){
                auto newtableptr = ptr->makeCopy(d_copytemp.data(), targetDeviceId);
                if(newtableptr){
                    result->gpuHashTables.push_back(std::move(newtableptr));
                }else{
                    cudaGetLastError();
                    return nullptr;
                }
            }

            std::vector<GpuTable::DeviceTableView> views;
            for(const auto& ptr : result->gpuHashTables){
                views.emplace_back(ptr->makeDeviceView());
            }

            result->d_deviceAccessibleTableViews.resize(views.size());
            CUDACHECK(cudaMemcpyAsync(
                result->d_deviceAccessibleTableViews.data(),
                views.data(),
                sizeof(GpuTable::DeviceTableView) * views.size(),
                H2D,
                cudaStreamPerThread
            ));

            CUDACHECK(cudaStreamSynchronize(cudaStreamPerThread));

            return result;
        }


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
                CUDACHECK(cudaDeviceSynchronize());
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
            
            temp_allocation_sizes[0] = sizeof(kmer_type) * signaturesRowPitchElements * numSequences; // d_sig
            temp_allocation_sizes[1] = sizeof(kmer_type) * signaturesRowPitchElements * numSequences; // d_sig_trans
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

            kmer_type* const d_signatures = static_cast<kmer_type*>(temp_allocations[0]);
            kmer_type* const d_signatures_transposed = static_cast<kmer_type*>(temp_allocations[1]);
            int* const d_hashFunctionNumbers = static_cast<int*>(temp_allocations[2]);
            
            CUDACHECK(cudaMemcpyAsync(
                d_hashFunctionNumbers, 
                h_hashFunctionNumbers, 
                sizeof(int) * numHashfunctions, 
                H2D, 
                stream
            ));

            callMinhashSignatures3264Kernel(
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
            ); CUDACHECKASYNC;

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

            CUDACHECK(cudaStreamSynchronize(stream));

            for(int i = 0; i < numHashfunctions; i++){
                auto status = gpuHashTables[firstHashfunction + i]->pop_status(stream);
                CUDACHECK(cudaStreamSynchronize(stream));

                if(status.has_any_errors()){
                    std::cerr << "Error table " << (firstHashfunction + i) << " after insertion: " << status << "\n";
                }
            }
        }

        MinhasherHandle makeMinhasherHandle() const override {
            auto data = std::make_unique<QueryData>();
            CUDACHECK(cudaGetDevice(&data->deviceId));

            std::unique_lock<SharedMutex> lock(sharedmutex);
            const int handleid = counter++;
            MinhasherHandle h = constructHandle(handleid);

            tempdataVector.emplace_back(std::move(data));
            return h;
        }

        void destroyHandle(MinhasherHandle& handle) const override{

            std::unique_lock<SharedMutex> lock(sharedmutex);

            const int id = handle.getId();
            assert(id < int(tempdataVector.size()));
            
            tempdataVector[id] = nullptr;
            handle = constructHandle(std::numeric_limits<int>::max());
        }

        void compact(cudaStream_t stream = 0) {
            DeviceSwitcher ds(deviceId);

            std::size_t required_temp_bytes = 0;

            for(auto& table : gpuHashTables){
                std::size_t temp_bytes2 = 0;
                table->compact(nullptr, temp_bytes2, stream);
                required_temp_bytes = std::max(required_temp_bytes, temp_bytes2);
            }

            std::size_t freeMem, totalMem; 
            CUDACHECK(cudaMemGetInfo(&freeMem, &totalMem));

            void* temp = nullptr;
            if(required_temp_bytes < freeMem){
                CUDACHECK(cudaMalloc(&temp, required_temp_bytes));
            }else{
                CUDACHECK(cudaMallocManaged(&temp, required_temp_bytes));
                int deviceId = 0;
                CUDACHECK(cudaGetDevice(&deviceId));
                CUDACHECK(cudaMemAdvise(temp, required_temp_bytes, cudaMemAdviseSetAccessedBy, deviceId));
            }

            for(auto& table : gpuHashTables){
                table->compact(temp, required_temp_bytes, stream);
            }

            CUDACHECK(cudaFree(temp));
        }

        MemoryUsage getMemoryInfo() const noexcept override{
            MemoryUsage mem{};

            for(const auto& table : gpuHashTables){
                mem += table->getMemoryInfo();
            }

            return mem;
        }

        MemoryUsage getMemoryInfo(const MinhasherHandle& handle) const noexcept override{
            return getQueryDataFromHandle(handle)->getMemoryInfo();
        }

        int getNumResultsPerMapThreshold() const noexcept override{
            return resultsPerMapThreshold;
        }
        
        int getNumberOfMaps() const noexcept override{
            return gpuHashTables.size();
        }

        void destroy(){
            DeviceSwitcher sd(getDeviceId());
            gpuHashTables.clear();
        }

        bool hasGpuTables() const noexcept override {
            return true;
        }

        void determineNumValues(
            MinhasherHandle& queryHandle,
            const unsigned int* d_sequenceData2Bit,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int* d_numValuesPerSequence,
            int& totalNumValues,
            cudaStream_t stream
        ) const override {

            QueryData* const queryData = getQueryDataFromHandle(queryHandle);

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

            queryData->d_singlepersistentbuffer.resize(persistent_storage_bytes);
            queryData->d_singletempbuffer.resize(temp_storage_bytes);

            determineNumValues(
                queryData->d_singlepersistentbuffer.data(),
                persistent_storage_bytes,
                queryData->d_singletempbuffer.data(),
                temp_storage_bytes,
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                d_sequenceLengths,
                numSequences,
                d_numValuesPerSequence,
                totalNumValues,
                stream
            );

            queryData->previousStage = QueryData::Stage::NumValues;
            queryData->d_numValuesPerSequence = d_numValuesPerSequence;
        }

        void retrieveValues(
            MinhasherHandle& queryHandle,
            const read_number* d_readIds,
            int numSequences,
            int totalNumValues,
            read_number* d_values,
            int* d_numValuesPerSequence,
            int* d_offsets, //numSequences + 1
            cudaStream_t stream
        ) const override {
            QueryData* const queryData = getQueryDataFromHandle(queryHandle);

            assert(queryData->previousStage == QueryData::Stage::NumValues);
            //STUPID INTERFACE EXPECTS d_numValuesPerSequence TO CONTAIN THE SAME VALUES AS RETURNED BY determineNumValues. This needs to be refactored. 
            //assert(queryData->d_numValuesPerSequence = d_numValuesPerSequence);

            std::size_t persistent_storage_bytes = queryData->d_singlepersistentbuffer.size();
            std::size_t temp_storage_bytes = 0;

            retrieveValues(
                queryData->d_singlepersistentbuffer.data(),
                persistent_storage_bytes,
                nullptr,
                temp_storage_bytes,
                d_readIds,
                numSequences,
                totalNumValues,
                123456, //unused
                d_values,
                d_numValuesPerSequence,
                d_offsets, //numSequences + 1
                stream
            );

            std::size_t cubsize = 0;
            CUDACHECK(cub::DeviceReduce::Max(
                nullptr, 
                cubsize, 
                d_numValuesPerSequence, 
                (int*)nullptr, 
                numSequences, 
                stream
            ));
            cubsize = SDIV(cubsize, 4) * 4;
            temp_storage_bytes = std::max(temp_storage_bytes, cubsize + sizeof(int));

            queryData->d_singletempbuffer.resize(temp_storage_bytes);

            int* d_maxvalue = (int*)(queryData->d_singletempbuffer.data() + cubsize);

            CUDACHECK(cub::DeviceReduce::Max(
                queryData->d_singletempbuffer.data(), 
                cubsize, 
                d_numValuesPerSequence, 
                d_maxvalue, 
                numSequences, 
                stream
            ));
            
            int sizeOfLargestSegment = 0;
            CUDACHECK(cudaMemcpyAsync(&sizeOfLargestSegment, d_maxvalue, sizeof(int), D2H, stream));
            CUDACHECK(cudaStreamSynchronize(stream));

            retrieveValues(
                queryData->d_singlepersistentbuffer.data(),
                persistent_storage_bytes,
                queryData->d_singletempbuffer.data(),
                temp_storage_bytes,
                d_readIds,
                numSequences,
                totalNumValues,
                sizeOfLargestSegment,
                d_values,
                d_numValuesPerSequence,
                d_offsets, //numSequences + 1
                stream
            );

        }


        //to make the following to functions private, their lambda kernels have to be replaced by normal kernels

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

            persistent_allocation_sizes[0] = sizeof(kmer_type) * numHashfunctions * numSequences; // d_sig_trans
            persistent_allocation_sizes[1] = sizeof(int) * numSequences * numHashfunctions; // d_numValuesPerSequencePerHash
            persistent_allocation_sizes[2] = sizeof(int) * numSequences * numHashfunctions; // d_numValuesPerSequencePerHashExclPSVert

            CUDACHECK(cub::AliasTemporaries(
                persistent_storage,
                persistent_storage_bytes,
                persistent_allocations,
                persistent_allocation_sizes
            ));

            std::size_t cubtempbytes = 0;
            CUDACHECK(cub::DeviceReduce::Sum(
                nullptr, 
                cubtempbytes, 
                (int*)nullptr, 
                (int*)nullptr, 
                numSequences, 
                stream
            ));

            void* temp_allocations[4];
            std::size_t temp_allocation_sizes[4];
            
            temp_allocation_sizes[0] = sizeof(int) * numHashfunctions; // d_hashFunctionNumbers
            temp_allocation_sizes[1] = cubtempbytes; // d_cub_temp
            temp_allocation_sizes[2] = sizeof(kmer_type) * numHashfunctions * numSequences; // d_sig
            temp_allocation_sizes[3] = sizeof(int); // d_cub_sum
            
            CUDACHECK(cub::AliasTemporaries(
                temp_storage,
                temp_storage_bytes,
                temp_allocations,
                temp_allocation_sizes
            ));

            if(persistent_storage == nullptr || temp_storage == nullptr){
                return;
            }

            kmer_type* const d_signatures_transposed = static_cast<kmer_type*>(persistent_allocations[0]);
            int* const d_numValuesPerSequencePerHash = static_cast<int*>(persistent_allocations[1]);
            int* const d_numValuesPerSequencePerHashExclPSVert = static_cast<int*>(persistent_allocations[2]);

            int* const d_hashFunctionNumbers = static_cast<int*>(temp_allocations[0]);
            void* const d_cubTemp = temp_allocations[1];
            kmer_type* const d_signatures = static_cast<kmer_type*>(temp_allocations[2]);
            int* const d_cub_sum = static_cast<int*>(temp_allocations[3]);

            DeviceSwitcher ds(deviceId);

            CUDACHECK(cudaMemcpyAsync(
                d_hashFunctionNumbers,
                h_currentHashFunctionNumbers.data(), 
                sizeof(int) * numHashfunctions, 
                H2D, 
                stream
            ));           

            dim3 block(128,1,1);
            dim3 grid(SDIV(numHashfunctions * numSequences, block.x),1,1);

            callMinhashSignatures3264Kernel(
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
            ); CUDACHECKASYNC;

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
            #if 1

            {

            const int signaturesPitchInElements = numSequences;
            const int numValuesPerKeyPitchInElements = numSequences;
            constexpr int cgsize = GpuTable::DeviceTableView::cg_size();

            dim3 block(256, 1, 1);
            const int numBlocksPerTable = SDIV(numSequences, (block.x / cgsize));
            dim3 grid(numBlocksPerTable, std::min(65535, numHashfunctions), 1);

            gpuhashtablekernels::numValuesPerKeyCompactMultiTableKernel<<<grid, block, 0, stream>>>(
                d_deviceAccessibleTableViews.data(),
                numHashfunctions,
                resultsPerMapThreshold,
                d_signatures_transposed,
                signaturesPitchInElements,
                numSequences,
                d_numValuesPerSequencePerHash,
                numValuesPerKeyPitchInElements
            );

            }
            #else

            
            for(int i = 0; i < numHashfunctions; i++){
                gpuHashTables[i]->numValuesPerKeyCompact(
                    d_signatures_transposed + i * numSequences,
                    numSequences,
                    d_numValuesPerSequencePerHash + i * numSequences,
                    stream
                );
            }

            #endif

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

            CUDACHECK(cub::DeviceReduce::Sum(
                d_cubTemp, 
                cubtempbytes, 
                d_numValuesPerSequence, 
                d_cub_sum, 
                numSequences, 
                stream
            ));

            CUDACHECK(cudaMemcpyAsync(&totalNumValues, d_cub_sum, sizeof(int), D2H, stream));
        }

        void retrieveValues(
            void* persistentbufferFromNumValues,            
            std::size_t persistent_storage_bytes,
            void* temp_storage,
            std::size_t& temp_storage_bytes,
            const read_number* d_readIds,
            int numSequences,
            int totalNumValues,
            int sizeOfLargestSegment,
            read_number* d_values,
            int* d_numValuesPerSequence,
            int* d_offsets, //numSequences + 1
            cudaStream_t stream
        ) const {
            assert(persistentbufferFromNumValues != nullptr);

            const int numHashfunctions = gpuHashTables.size();

            std::size_t cubtempbytes = 0;

            CUDACHECK(cub::DeviceScan::InclusiveSum(
                nullptr,
                cubtempbytes,
                (int*)nullptr, 
                (int*)nullptr, 
                numSequences,
                stream
            ));

            std::size_t cubtempbytes2 = 0;
            CUDACHECK(cub::DeviceReduce::Max(
                nullptr, 
                cubtempbytes2, 
                (int*)nullptr, 
                (int*)nullptr, 
                numSequences, 
                stream
            ));

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
            
            CUDACHECK(cub::AliasTemporaries(
                temp_storage,
                temp_storage_bytes,
                temp_allocations,
                temp_allocation_sizes
            ));

            if(temp_storage == nullptr) return;

            void* persistent_allocations[3]{};
            std::size_t persistent_allocation_sizes[3]{};

            persistent_allocation_sizes[0] = sizeof(kmer_type) * numHashfunctions * numSequences; // d_sig_trans
            persistent_allocation_sizes[1] = sizeof(int) * numSequences * numHashfunctions; // d_numValuesPerSequencePerHash
            persistent_allocation_sizes[2] = sizeof(int) * numSequences * numHashfunctions; // d_numValuesPerSequencePerHashExclPSVert

            CUDACHECK(cub::AliasTemporaries(
                persistentbufferFromNumValues,
                persistent_storage_bytes,
                persistent_allocations,
                persistent_allocation_sizes
            ));

            DeviceSwitcher ds(deviceId);

            kmer_type* const d_signatures_transposed = static_cast<kmer_type*>(persistent_allocations[0]);
            int* const d_numValuesPerSequencePerHash = static_cast<int*>(persistent_allocations[1]);
            int* const d_numValuesPerSequencePerHashExclPSVert = static_cast<int*>(persistent_allocations[2]);

            void* const d_cubTemp = temp_allocations[0];
            int* const d_cub_sum = static_cast<int*>(temp_allocations[1]);
            int* const d_queryOffsetsPerSequencePerHash = static_cast<int*>(temp_allocations[2]);
            void* const d_uniquetemp = temp_allocations[2];
            read_number* const d_values_tmp = static_cast<read_number*>(temp_allocations[3]);
            int* const d_end_offsets = static_cast<int*>(temp_allocations[4]);
     


            //calculate global offsets for each sequence in output array
            CUDACHECK(cudaMemsetAsync(d_offsets, 0, sizeof(int), stream));

            CUDACHECK(cub::DeviceScan::InclusiveSum(
                d_cubTemp,
                cubtempbytes,
                d_numValuesPerSequence,
                d_offsets + 1,
                numSequences,
                stream
            ));

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

            CUDACHECK(cub::DeviceReduce::Max(
                d_cubTemp, 
                cubtempbytes, 
                d_numValuesPerSequence, 
                d_cub_sum, 
                numSequences, 
                stream
            ));

            //results will be in Current() buffer
            cub::DoubleBuffer<read_number> d_values_dblbuf(d_values, d_values_tmp);

            CUDACHECK(cudaMemcpyAsync(d_end_offsets, d_offsets + 1, sizeof(int) * numSequences, D2D, stream));

            //retrieve values

            #if 1
            {
            const int signaturesPitchInElements = numSequences;
            const int numValuesPerKeyPitchInElements = numSequences;
            const int beginOffsetsPitchInElements = numSequences;
            constexpr int cgsize = GpuTable::DeviceTableView::cg_size();

            dim3 block(256, 1, 1);
            const int numBlocksPerTable = SDIV(numSequences, (block.x / cgsize));
            dim3 grid(numBlocksPerTable, std::min(65535, numHashfunctions), 1);

            gpuhashtablekernels::retrieveCompactKernel<<<grid, block, 0, stream>>>(
                d_deviceAccessibleTableViews.data(),
                numHashfunctions,
                d_signatures_transposed,
                signaturesPitchInElements,
                d_queryOffsetsPerSequencePerHash,
                beginOffsetsPitchInElements,
                d_numValuesPerSequencePerHash,
                numValuesPerKeyPitchInElements,
                resultsPerMapThreshold,
                numSequences,
                d_values_dblbuf.Current()
            );
            }
            #else

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

            #endif

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

            CUDACHECK(cudaMemsetAsync(d_newOffsets, 0, sizeof(int), stream));

            CUDACHECK(cub::DeviceScan::InclusiveSum(
                d_cubTemp,
                cubtempbytes,
                d_numValuesPerSequence,
                d_newOffsets + 1,
                numSequences,
                stream
            ));

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
            ); CUDACHECKASYNC;

            CUDACHECK(cudaMemcpyAsync(d_offsets, d_newOffsets, sizeof(int) * (numSequences+1), D2D, stream));
        }

        int getKmerSize() const noexcept override{
            return kmerSize;
        }


        constexpr int getDeviceId() const noexcept{
            return deviceId;
        }

private:

        void finalize(cudaStream_t stream = 0){
            compact(stream);
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
        //mutable std::shared_mutex sharedmutex{};

        int deviceId{};
        int maxNumKeys{};
        int kmerSize{};
        int resultsPerMapThreshold{};
        HostBuffer<int> h_currentHashFunctionNumbers{};
        std::vector<std::unique_ptr<GpuTable>> gpuHashTables{};
        DeviceBuffer<GpuTable::DeviceTableView> d_deviceAccessibleTableViews{};
        mutable std::vector<std::unique_ptr<QueryData>> tempdataVector{};
    };


}
}




#endif

#endif //#ifdef CARE_HAS_WARPCORE