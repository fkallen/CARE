#ifndef CARE_FAKEGPUMINHASHER_CUH
#define CARE_FAKEGPUMINHASHER_CUH

#include <config.hpp>

#include <gpu/gpureadstorage.cuh>
#include <gpu/cuda_unique.cuh>
#include <gpu/minhashingkernels.cuh>
#include <cpuhashtable.hpp>
#include <gpu/gpuminhasher.cuh>
#include <groupbykey.hpp>
#include <gpu/cudaerrorcheck.cuh>

#include "minhashing.hpp"

#include <options.hpp>
#include <util.hpp>
#include <hpc_helpers.cuh>
#include <filehelpers.hpp>

#include <sequencehelpers.hpp>
#include <memorymanagement.hpp>
#include <threadpool.hpp>

#include <sharedmutex.hpp>


#include <vector>
#include <memory>
#include <limits>
#include <string>
#include <fstream>
#include <algorithm>

#include <cub/cub.cuh>


namespace care{
namespace gpu{



    /*
        Minhasher which can store query results in gpu memory and uses the gpu to parallelize some portions of the code
        However, hash tables reside on the host
    */
    class FakeGpuMinhasher : public GpuMinhasher{
    public:
        using Key_t = GpuMinhasher::Key;
        using Value_t = read_number;
    private:
        using HashTable = CpuReadOnlyMultiValueHashTable<kmer_type, read_number>;
        using Range_t = std::pair<const Value_t*, const Value_t*>;

        struct QueryData{
            static constexpr int overprovisioningPercent = 0;

            template<class T>
            using DeviceBuffer = helpers::SimpleAllocationDevice<T, overprovisioningPercent>;
            
            template<class T>
            using PinnedBuffer = helpers::SimpleAllocationPinnedHost<T, overprovisioningPercent>;

            enum class Stage{
                None,
                NumValues,
                Retrieve
            };


            bool isInitialized = false;
            int deviceId;
            Stage previousStage = Stage::None;

            std::vector<Range_t> ranges{};
            SetUnionHandle suHandle{};

            DeviceBuffer<std::uint64_t> d_minhashSignatures{};
            DeviceBuffer<std::uint64_t> d_minhashSignatures_transposed{};
            PinnedBuffer<std::uint64_t> h_minhashSignatures{};            

            PinnedBuffer<read_number> h_candidate_read_ids_tmp{};
            DeviceBuffer<read_number> d_candidate_read_ids_tmp{};

            PinnedBuffer<int> h_begin_offsets{};
            DeviceBuffer<int> d_begin_offsets{};
            PinnedBuffer<int> h_end_offsets{};
            DeviceBuffer<int> d_end_offsets{};
            PinnedBuffer<int> h_numValuesPerSequence{};
            DeviceBuffer<int> d_global_begin_offsets{};

            DeviceBuffer<char> d_cub_temp{};

            std::vector<Range_t> allRanges{};
            std::vector<int> idsPerChunk{};   
            std::vector<int> numAnchorsPerChunk{};
            std::vector<int> idsPerChunkPrefixSum{};
            std::vector<int> numAnchorsPerChunkPrefixSum{};

            DeviceBuffer<std::uint64_t> d_temp{};
            DeviceBuffer<int> d_signatureSizePerSequence{};
            PinnedBuffer<int> h_signatureSizePerSequence{};
            DeviceBuffer<int> d_hashFuncIds{};
            PinnedBuffer<int> h_hashFuncIds{};

            GpuSegmentedUnique::Handle segmentedUniqueHandle;
            std::vector<GpuSegmentedUnique::Handle> segmentedUniqueHandles;


            // void resize(const FakeGpuMinhasher& minhasher, std::size_t numSequences, int numThreads = 1){
            //     const std::size_t maximumResultSize 
            //         = minhasher.getNumResultsPerMapThreshold() * minhasher.getNumberOfMaps() * numSequences;

            //     d_minhashSignatures.resize(minhasher.getNumberOfMaps() * numSequences);
            //     h_minhashSignatures.resize(minhasher.getNumberOfMaps() * numSequences);
            //     h_candidate_read_ids_tmp.resize(maximumResultSize);
            //     d_candidate_read_ids_tmp.resize(maximumResultSize);

            //     h_begin_offsets.resize(numSequences+1);
            //     d_begin_offsets.resize(numSequences+1);
            //     h_end_offsets.resize(numSequences+1);
            //     d_end_offsets.resize(numSequences+1);
            //     h_numValuesPerSequence.resize(numSequences);
            //     d_global_begin_offsets.resize(numSequences);
            
            //     allRanges.resize(minhasher.getNumberOfMaps() * numSequences);
            //     idsPerChunk.resize(numThreads, 0);   
            //     numAnchorsPerChunk.resize(numThreads, 0);
            //     idsPerChunkPrefixSum.resize(numThreads, 0);
            //     numAnchorsPerChunkPrefixSum.resize(numThreads, 0);

            //     d_temp.resize(minhasher.getNumberOfMaps() * numSequences);
            //     d_signatureSizePerSequence.resize(numSequences);
            //     h_signatureSizePerSequence.resize(numSequences);

            //     d_hashFuncIds.resize(minhasher.getNumberOfMaps() * numSequences);
            //     h_hashFuncIds.resize(minhasher.getNumberOfMaps() * numSequences);
            // }

            MemoryUsage getMemoryInfo() const{
                MemoryUsage info;
                info.host = 0;
                info.device[deviceId] = 0;
    
                auto handlehost = [&](const auto& buff){
                    info.host += buff.capacityInBytes();
                };
    
                auto handledevice = [&](const auto& buff){
                    info.device[deviceId] += buff.capacityInBytes();
                };

                auto handlevector = [&](const auto& buff){
                    info.host += 
                        sizeof(typename std::remove_reference<decltype(buff)>::type::value_type) * buff.capacity();
                };
    
                handlehost(h_minhashSignatures);
                handlehost(h_candidate_read_ids_tmp);
                handlehost(h_begin_offsets);
                handlehost(h_end_offsets);
                handlehost(h_numValuesPerSequence);
    
                handledevice(d_minhashSignatures);
                handledevice(d_minhashSignatures_transposed);
                handledevice(d_candidate_read_ids_tmp);
                handledevice(d_begin_offsets);
                handledevice(d_end_offsets);
                handledevice(d_global_begin_offsets);

                handledevice(d_cub_temp);

                handlevector(allRanges);
                handlevector(idsPerChunk);
                handlevector(numAnchorsPerChunk);
                handlevector(idsPerChunkPrefixSum);
                handlevector(numAnchorsPerChunkPrefixSum);

                handledevice(d_temp);
                handledevice(d_signatureSizePerSequence);
                handledevice(d_hashFuncIds);
                handlehost(h_signatureSizePerSequence);
                handlehost(h_hashFuncIds);

                info += segmentedUniqueHandle->getMemoryInfo();

                for(const auto& h : segmentedUniqueHandles){
                    info += h->getMemoryInfo();
                }
    
                return info;
            }

            void destroy(){
                cub::SwitchDevice sd{deviceId};

                d_minhashSignatures.destroy();
                d_minhashSignatures_transposed.destroy();
                h_minhashSignatures.destroy();
                h_candidate_read_ids_tmp.destroy();
                d_candidate_read_ids_tmp.destroy();
                h_begin_offsets.destroy();
                d_begin_offsets.destroy();
                h_end_offsets.destroy();
                d_end_offsets.destroy();
                h_numValuesPerSequence.destroy();
                d_global_begin_offsets.destroy();

                d_cub_temp.destroy();

                allRanges.clear();
                allRanges.shrink_to_fit();

                d_temp.destroy();
                d_signatureSizePerSequence.destroy();
                h_signatureSizePerSequence.destroy();

                d_hashFuncIds.destroy();
                h_hashFuncIds.destroy();

                segmentedUniqueHandle = nullptr;
                for(auto& h : segmentedUniqueHandles){
                    h = nullptr;
                }

                isInitialized = false;
            }
        };

        
    public:

        FakeGpuMinhasher() : FakeGpuMinhasher(0, 50, 16, 0.8f){

        }

        FakeGpuMinhasher(int maxNumKeys_, int maxValuesPerKey, int k, float loadfactor_)
            : loadfactor(loadfactor_), maxNumKeys(maxNumKeys_), kmerSize(k), resultsPerMapThreshold(maxValuesPerKey){

        }

        FakeGpuMinhasher(const FakeGpuMinhasher&) = delete;
        FakeGpuMinhasher(FakeGpuMinhasher&&) = default;
        FakeGpuMinhasher& operator=(const FakeGpuMinhasher&) = delete;
        FakeGpuMinhasher& operator=(FakeGpuMinhasher&&) = default;


    void constructFromReadStorage(
        const FileOptions &fileOptions,
        const RuntimeOptions &runtimeOptions,
        const MemoryOptions& memoryOptions,
        std::uint64_t nReads,
        const CorrectionOptions& correctionOptions,
        const GpuReadStorage& gpuReadStorage
    ){
        
        auto& readStorage = gpuReadStorage;
        const auto& deviceIds = runtimeOptions.deviceIds;

        int deviceId = deviceIds[0];

        cub::SwitchDevice sd{deviceId};

        const int requestedNumberOfMaps = correctionOptions.numHashFunctions;

        const read_number numReads = readStorage.getNumberOfReads();
        const int maximumSequenceLength = readStorage.getSequenceLengthUpperBound();

        auto sequencehandle = gpuReadStorage.makeHandle();
        const std::size_t encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength);

        constexpr read_number parallelReads = 1000000;
        const int numIters = SDIV(numReads, parallelReads);

        const MemoryUsage memoryUsageOfReadStorage = readStorage.getMemoryInfo();
        std::size_t totalLimit = memoryOptions.memoryTotalLimit;
        if(totalLimit > memoryUsageOfReadStorage.host){
            totalLimit -= memoryUsageOfReadStorage.host;
        }else{
            totalLimit = 0;
        }
        if(totalLimit == 0){
            throw std::runtime_error("Not enough memory available for hash tables. Abort!");
        }
        std::size_t maxMemoryForTables = getAvailableMemoryInKB() * 1024;
        // std::cerr << "available: " << maxMemoryForTables 
        //         << ",memoryForHashtables: " << memoryOptions.memoryForHashtables
        //         << ", memoryTotalLimit: " << memoryOptions.memoryTotalLimit
        //         << ", rsHostUsage: " << memoryUsageOfReadStorage.host << "\n";

        maxMemoryForTables = std::min(maxMemoryForTables, 
                                std::min(memoryOptions.memoryForHashtables, totalLimit));

        std::cerr << "maxMemoryForTables = " << maxMemoryForTables << " bytes\n";

        const int hashFunctionOffset = 0;

        
        std::vector<int> usedHashFunctionNumbers;
        
        helpers::SimpleAllocationDevice<unsigned int, 1> d_sequenceData(encodedSequencePitchInInts * parallelReads);
        helpers::SimpleAllocationDevice<int, 0> d_lengths(parallelReads);
        
        helpers::SimpleAllocationPinnedHost<read_number, 0> h_indices(parallelReads);
        helpers::SimpleAllocationDevice<read_number, 0> d_indices(parallelReads);
        
        std::size_t d_insert_temp_size = 0;
        std::size_t h_insert_temp_size = 0;
        insert(
            nullptr,
            d_insert_temp_size,
            nullptr,
            h_insert_temp_size,
            (const unsigned int*)nullptr,
            int(parallelReads),
            (const int*)nullptr,
            encodedSequencePitchInInts,
            (const read_number*)nullptr,
            (const read_number*)nullptr,
            0,
            requestedNumberOfMaps,
            (const int*)nullptr,
            (cudaStream_t)0
        );
        
        
        CudaStream stream{};
        ThreadPool tpForHashing(runtimeOptions.threads);
        ThreadPool tpForCompacting(std::min(2,runtimeOptions.threads));

        
        setMemoryLimitForConstruction(maxMemoryForTables);
        
        //std::size_t bytesOfCachedConstructedTables = 0;
        int remainingHashFunctions = requestedNumberOfMaps;
        bool keepGoing = true;

        while(remainingHashFunctions > 0 && keepGoing){

            setThreadPool(&tpForHashing);

            const int alreadyExistingHashFunctions = requestedNumberOfMaps - remainingHashFunctions;
            int addedHashFunctions = addHashfunctions(remainingHashFunctions);

            if(addedHashFunctions == 0){
                keepGoing = false;
                break;
            }

            helpers::SimpleAllocationDevice<char, 0> d_temp(d_insert_temp_size);
            helpers::SimpleAllocationPinnedHost<char> h_temp(h_insert_temp_size);

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

                std::size_t s1 = d_insert_temp_size;
                std::size_t s2 = h_insert_temp_size;

                insert(
                    d_temp.data(),
                    s1,
                    h_temp.data(),
                    s2,
                    d_sequenceData,
                    curBatchsize,
                    d_lengths,
                    encodedSequencePitchInInts,
                    d_indices,
                    h_indices,
                    alreadyExistingHashFunctions,
                    addedHashFunctions,
                    h_hashfunctionNumbers.data(),
                    stream
                );

                CUDACHECK(cudaStreamSynchronize(stream));
            }

            d_temp.destroy();
            h_temp.destroy();

            std::cerr << "Compacting\n";
            if(tpForCompacting.getConcurrency() > 1){
                setThreadPool(&tpForCompacting);
            }else{
                setThreadPool(nullptr);
            }
            
            finalize();

            remainingHashFunctions -= addedHashFunctions;
        }

        setThreadPool(nullptr); 
        
        gpuReadStorage.destroyHandle(sequencehandle);
    }
 

        MinhasherHandle makeMinhasherHandle() const override {
            auto data = std::make_unique<QueryData>();
            data->segmentedUniqueHandle = GpuSegmentedUnique::makeHandle();
            CUDACHECK(cudaGetDevice(&data->deviceId));
            data->isInitialized = true;

            //std::unique_lock<std::shared_mutex> lock(sharedmutex);
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

        #define FAKEGPUMINHASHER_RUN_ON_GPU

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
            #ifdef FAKEGPUMINHASHER_RUN_ON_GPU
            determineNumValuesOnGpu(
                queryHandle,
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                d_sequenceLengths,
                numSequences,
                d_numValuesPerSequence,
                totalNumValues,
                stream
            );
            #else
            determineNumValuesOnCpu(
                queryHandle,
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                d_sequenceLengths,
                numSequences,
                d_numValuesPerSequence,
                totalNumValues,
                stream
            );
            #endif
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
            #ifdef FAKEGPUMINHASHER_RUN_ON_GPU
                retrieveValuesOnGpu(
                    queryHandle,
                    d_readIds,
                    numSequences,
                    totalNumValues,
                    d_values,
                    d_numValuesPerSequence,
                    d_offsets,
                    stream
                );
            #else
                retrieveValuesOnCpu(
                    queryHandle,
                    d_readIds,
                    numSequences,
                    totalNumValues,
                    d_values,
                    d_numValuesPerSequence,
                    d_offsets,
                    stream
                );
            #endif
        }

        #ifdef FAKEGPUMINHASHER_RUN_ON_GPU
        #undef FAKEGPUMINHASHER_RUN_ON_GPU
        #endif

        void compact(cudaStream_t stream) {
            int id;
            CUDACHECK(cudaGetDevice(&id));

            auto groupByKey = [&](auto& keys, auto& values, auto& countsPrefixSum){
                constexpr bool valuesOfSameKeyMustBeSorted = false;
                const int maxValuesPerKey = getNumResultsPerMapThreshold();

                bool success = false;

                using GroupByKeyCpuOp = GroupByKeyCpu<Key_t, Value_t, read_number>;
                using GroupByKeyGpuOp = GroupByKeyGpu<Key_t, Value_t, read_number>;
                
                GroupByKeyGpuOp groupByKeyGpu(valuesOfSameKeyMustBeSorted, maxValuesPerKey);
                success = groupByKeyGpu.execute(keys, values, countsPrefixSum);         

                if(!success){
                    GroupByKeyCpuOp groupByKeyCpu(valuesOfSameKeyMustBeSorted, maxValuesPerKey);
                    groupByKeyCpu.execute(keys, values, countsPrefixSum);
                }
            };

            const int num = minhashTables.size();
            for(int i = 0, l = 0; i < num; i++){
                auto& ptr = minhashTables[i];
            
                if(!ptr->isInitialized()){
                    //after processing 3 tables, available memory should be sufficient for multithreading
                    if(l >= 3){
                        ptr->finalize(groupByKey, threadPool);
                    }else{
                        ptr->finalize(groupByKey, nullptr);
                    }
                    l++;
                }                
            }

            if(threadPool != nullptr){
                threadPool->wait();
            }
        }

        MemoryUsage getMemoryInfo() const noexcept override{
            MemoryUsage result;

            result.host = sizeof(HashTable) * minhashTables.size();
            
            for(const auto& tableptr : minhashTables){
                auto m = tableptr->getMemoryInfo();
                result.host += m.host;

                for(auto pair : m.device){
                    result.device[pair.first] += pair.second;
                }
            }

            return result;
        }

        MemoryUsage getMemoryInfo(const MinhasherHandle& handle) const noexcept override{
            return getQueryDataFromHandle(handle)->getMemoryInfo();
        }

        int getNumResultsPerMapThreshold() const noexcept override{
            return resultsPerMapThreshold;
        }
        
        int getNumberOfMaps() const noexcept override{
            return minhashTables.size();
        }

        void destroy() {
            minhashTables.clear();
        }

        bool hasGpuTables() const noexcept override {
            return false;
        }

        void finalize(cudaStream_t stream = 0){
            compact(stream);
        }

        int getKmerSize() const{
            return kmerSize;
        }

        std::uint64_t getKmerMask() const{
            constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;

            return std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - getKmerSize()) * 2);
        }


        void writeToStream(std::ostream& os) const{

            os.write(reinterpret_cast<const char*>(&kmerSize), sizeof(int));
            os.write(reinterpret_cast<const char*>(&resultsPerMapThreshold), sizeof(int));
            os.write(reinterpret_cast<const char*>(&loadfactor), sizeof(float));

            const int numTables = getNumberOfMaps();
            os.write(reinterpret_cast<const char*>(&numTables), sizeof(int));

            for(const auto& tableptr : minhashTables){
                tableptr->writeToStream(os);
            }
        }

        int loadFromStream(std::ifstream& is, int numMapsUpperLimit = std::numeric_limits<int>::max()){
            destroy();

            is.read(reinterpret_cast<char*>(&kmerSize), sizeof(int));
            is.read(reinterpret_cast<char*>(&resultsPerMapThreshold), sizeof(int));
            is.read(reinterpret_cast<char*>(&loadfactor), sizeof(float));

            int numMaps = 0;

            is.read(reinterpret_cast<char*>(&numMaps), sizeof(int));

            const int mapsToLoad = std::min(numMapsUpperLimit, numMaps);

            for(int i = 0; i < mapsToLoad; i++){
                auto ptr = std::make_unique<HashTable>();
                ptr->loadFromStream(is);
                minhashTables.emplace_back(std::move(ptr));
            }

            return mapsToLoad;
        } 

        int addHashfunctions(int numExtraFunctions){
            int added = 0;
            const int cur = minhashTables.size();

            assert(!(numExtraFunctions + cur > 64));

            std::size_t bytesOfCachedConstructedTables = 0;
            for(const auto& ptr : minhashTables){
                auto memusage = ptr->getMemoryInfo();
                bytesOfCachedConstructedTables += memusage.host;
            }

            std::size_t requiredMemPerTable = (sizeof(kmer_type) + sizeof(read_number)) * maxNumKeys;
            int numTablesToConstruct = (memoryLimit - bytesOfCachedConstructedTables) / requiredMemPerTable;
            numTablesToConstruct -= 2; // keep free memory of 2 tables to perform transformation 
            numTablesToConstruct = std::min(numTablesToConstruct, numExtraFunctions);

            for(int i = 0; i < numTablesToConstruct; i++){
                try{
                    auto ptr = std::make_unique<HashTable>(maxNumKeys, loadfactor);

                    minhashTables.emplace_back(std::move(ptr));
                    added++;
                }catch(...){

                }
            }

            return added;
        } 

        void insert(
            void* d_temp,
            std::size_t& d_temp_storage_bytes,
            void* h_temp,
            std::size_t& h_temp_storage_bytes,
            const unsigned int* d_sequenceData2Bit,
            int numSequences,
            const int* d_sequenceLengths,
            std::size_t encodedSequencePitchInInts,
            const read_number* d_readIds,
            const read_number* h_readIds,
            int firstHashfunction,
            int numHashfunctions,
            const int* h_hashFunctionNumbers,
            cudaStream_t stream
        ){
            ThreadPool::ParallelForHandle pforHandle{};

            ForLoopExecutor forLoopExecutor(threadPool, &pforHandle);

            const std::size_t signaturesRowPitchElements = numHashfunctions;

            void* d_temp_allocations[3]{};
            std::size_t d_temp_allocation_sizes[3]{};            
            d_temp_allocation_sizes[0] = sizeof(std::uint64_t) * signaturesRowPitchElements * numSequences; // d_sig
            d_temp_allocation_sizes[1] = sizeof(std::uint64_t) * signaturesRowPitchElements * numSequences; // d_sig_trans
            d_temp_allocation_sizes[2] = sizeof(int) * numHashfunctions; // d_hashFunctionNumbers
            
            cudaError_t cubstatus = cub::AliasTemporaries(
                d_temp,
                d_temp_storage_bytes,
                d_temp_allocations,
                d_temp_allocation_sizes
            );
            assert(cubstatus == cudaSuccess);

            void* h_temp_allocations[1]{};
            std::size_t h_temp_allocation_sizes[1]{};            
            h_temp_allocation_sizes[0] = sizeof(std::uint64_t) * signaturesRowPitchElements * numSequences; // h_signatures_transposed
    
            cubstatus = cub::AliasTemporaries(
                h_temp,
                h_temp_storage_bytes,
                h_temp_allocations,
                h_temp_allocation_sizes
            );
            assert(cubstatus == cudaSuccess);

            if(d_temp == nullptr || h_temp == nullptr){
                return;
            }

            assert(firstHashfunction + numHashfunctions <= int(minhashTables.size()));

            std::uint64_t* const d_signatures = static_cast<std::uint64_t*>(d_temp_allocations[0]);
            std::uint64_t* const d_signatures_transposed = static_cast<std::uint64_t*>(d_temp_allocations[1]);
            int* const d_hashFunctionNumbers = static_cast<int*>(d_temp_allocations[2]);

            CUDACHECK(cudaMemcpyAsync(
                d_hashFunctionNumbers, 
                h_hashFunctionNumbers, 
                sizeof(int) * numHashfunctions, 
                H2D, 
                stream
            ));

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
            );

            helpers::call_transpose_kernel(
                d_signatures_transposed, 
                d_signatures, 
                numSequences, 
                signaturesRowPitchElements, 
                signaturesRowPitchElements,
                stream
            );

            std::uint64_t* const h_signatures_transposed = static_cast<std::uint64_t*>(h_temp_allocations[0]);

            CUDACHECK(cudaMemcpyAsync(
                h_signatures_transposed, 
                d_signatures_transposed, 
                sizeof(std::uint64_t) * signaturesRowPitchElements * numSequences, 
                D2H, 
                stream
            ));

            CUDACHECK(cudaStreamSynchronize(stream));

            auto loopbody = [&](auto begin, auto end, int /*threadid*/){
                for(int h = begin; h < end; h++){

                    std::uint64_t* const hashesBegin = &h_signatures_transposed[h * numSequences];

                    std::for_each(
                        hashesBegin, hashesBegin + numSequences,
                        [kmermask = getKmerMask()](auto& hash){
                            hash &= kmermask;
                        }
                    );

                    minhashTables[firstHashfunction + h]->insert(
                        hashesBegin, h_readIds, numSequences
                    );
                }
            };

            forLoopExecutor(0, numHashfunctions, loopbody);
        }   

        void setThreadPool(ThreadPool* tp){
            threadPool = tp;
        }

        void setMemoryLimitForConstruction(std::size_t limit){
            memoryLimit = limit;
        }

    public: //should be private, but lambda kernel....

        void retrieveValuesOnGpu(
            MinhasherHandle& queryHandle,
            const read_number* d_readIds,
            int numSequences,
            int totalNumValues,
            read_number* d_values,
            int* d_numValuesPerSequence,
            int* d_offsets, //numSequences + 1
            cudaStream_t stream
        ) const {
            QueryData* const queryData = getQueryDataFromHandle(queryHandle);

            assert(queryData->isInitialized);
            if(numSequences == 0) return;

            assert(queryData->previousStage == QueryData::Stage::NumValues);

            std::size_t cubtempbytes = 0;
            cub::DeviceScan::InclusiveSum(
                nullptr,
                cubtempbytes,
                (int*) nullptr,
                (int*) nullptr,
                numSequences,
                stream
            );

            std::size_t segmentedUniqueTempBytes = 0;
            GpuSegmentedUnique::unique(
                nullptr,
                segmentedUniqueTempBytes,
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

            queryData->d_minhashSignatures.resize(
                std::max(
                    int(SDIV(segmentedUniqueTempBytes, sizeof(std::uint64_t))),
                    int(SDIV(cubtempbytes, sizeof(std::uint64_t)))
                )
            );

            void* d_temp = queryData->d_minhashSignatures.data();

            //std::cerr << "totalNumValues: " << totalNumValues << "\n";

            if(totalNumValues == 0){
                cudaMemsetAsync(d_numValuesPerSequence, 0, sizeof(int) * numSequences, stream);
                cudaMemsetAsync(d_offsets, 0, sizeof(int) * (numSequences + 1), stream);
                return;
            }

            constexpr int roundUpTo = 10000;
            const int roundedTotalNum = SDIV(totalNumValues, roundUpTo) * roundUpTo;
            queryData->h_candidate_read_ids_tmp.resize(roundedTotalNum);
            queryData->d_candidate_read_ids_tmp.resize(roundedTotalNum);


            //results will be in Current() buffer
            cub::DoubleBuffer<read_number> d_values_dblbuf(d_values, queryData->d_candidate_read_ids_tmp.data());

            auto copyHitsToPinnedMemory = [queryData, numSequences, minhasher = this](){
                int* h_numValuesPerSequence = queryData->h_numValuesPerSequence.data();

                queryData->h_begin_offsets[0] = 0;
                std::partial_sum(
                    h_numValuesPerSequence, 
                    h_numValuesPerSequence + numSequences - 1, 
                    queryData->h_begin_offsets.begin() + 1
                );
                std::partial_sum(
                    h_numValuesPerSequence, 
                    h_numValuesPerSequence + numSequences, 
                    queryData->h_end_offsets.begin()
                );

                std::vector<int> processedPerSequence(numSequences, 0);

                for(int map = 0; map < minhasher->getNumberOfMaps(); ++map){
                    for(int i = 0; i < numSequences; i++){

                        

                        //prefetch first element of next range if the next range is not empty
                        // {
                        //     constexpr int nextprefetch = 2;
                        //     int prefetchSequence = 0;
                        //     int prefetchMap = 0;
                        //     if(i + nextprefetch < numSequences){
                        //         prefetchMap = map;
                        //         prefetchSequence = i + nextprefetch;
                        //     }else{
                        //         prefetchMap = map + 1;
                        //         prefetchSequence = i + nextprefetch - numSequences;
                        //     }

                        //     if(prefetchMap < minhasher->getNumberOfMaps()){
                        //         const auto& range = queryData->allRanges[prefetchMap * numSequences + prefetchSequence];
                        //         if(range.first != range.second){
                        //             __builtin_prefetch(range.first, 0, 0);
                        //         }
                        //     }
                        // }

                        const auto& range = queryData->allRanges[map * numSequences + i];

                        std::copy(
                            range.first, 
                            range.second, 
                            queryData->h_candidate_read_ids_tmp.data() 
                                + queryData->h_begin_offsets[i] + processedPerSequence[i]
                        );

                        processedPerSequence[i] += std::distance(range.first, range.second);
                    }
                }
            };

            

            copyHitsToPinnedMemory();

            CUDACHECK(cudaMemcpyAsync(
                d_values_dblbuf.Current(),
                queryData->h_candidate_read_ids_tmp.data(),
                sizeof(read_number) * totalNumValues,
                H2D,
                stream
            ));

            //copy h_endoffsets to d_endoffsets.
            //Then copy d_endoffsets to d_begin_offsets shifted to the right by 1.
            //Then copy d_begin_offsets to d_global_begin_offsets
            helpers::lambda_kernel<<<SDIV(numSequences, 1024), 1024, 0, stream>>>(
                [
                    numSequences,
                    h_end_offsets = queryData->h_end_offsets.data(),
                    d_end_offsets = queryData->d_end_offsets.data(),
                    d_begin_offsets = queryData->d_begin_offsets.data(),
                    d_global_begin_offsets = queryData->d_global_begin_offsets.data()
                ] __device__ (){
                    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

                    if(tid < numSequences){
                        const int data = h_end_offsets[tid];
                        d_end_offsets[tid] = data;

                        if(tid < numSequences - 1){
                            d_begin_offsets[tid + 1] = data;
                            d_global_begin_offsets[tid + 1] = data;
                        }
                        
                        if(tid == 0){
                            d_begin_offsets[0] = 0;
                            d_global_begin_offsets[0] = 0;
                        }
                    }
                }
            );

            GpuSegmentedUnique::unique(
                queryData->segmentedUniqueHandle,
                d_values_dblbuf.Current(), //input
                totalNumValues,
                d_values_dblbuf.Alternate(), //output
                d_numValuesPerSequence,
                numSequences,
                queryData->d_begin_offsets, //device accessible
                queryData->d_end_offsets, //device accessible
                queryData->h_begin_offsets,
                queryData->h_end_offsets,
                0,
                sizeof(read_number) * 8,
                stream
            );

            if(d_readIds != nullptr){

                //remove self read ids (inplace)
                //--------------------------------------------------------------------
                callFindAndRemoveFromSegmentKernel<read_number,128,4>(
                    d_readIds,
                    d_values_dblbuf.Alternate(),
                    numSequences,
                    d_numValuesPerSequence,
                    queryData->d_global_begin_offsets.data(),
                    stream
                );

            }

            int* d_newOffsets = d_offsets;
            void* d_cubTemp = queryData->d_minhashSignatures.data();

            CUDACHECK(cudaMemsetAsync(d_newOffsets, 0, sizeof(int), stream));

            CUDACHECK(cub::DeviceScan::InclusiveSum(
                d_cubTemp,
                cubtempbytes,
                d_numValuesPerSequence,
                d_newOffsets + 1,
                numSequences,
                stream
            ));

            //copy final remaining values into contiguous range
            helpers::lambda_kernel<<<numSequences, 128, 0, stream>>>(
                [
                    d_values_in = d_values_dblbuf.Alternate(),
                    d_values_out = d_values_dblbuf.Current(),
                    numSequences,
                    d_numValuesPerSequence,
                    d_offsets = queryData->d_global_begin_offsets.data(),
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

            queryData->previousStage = QueryData::Stage::Retrieve;
        }

        void retrieveValuesOnCpu(
            MinhasherHandle& queryHandle,
            const read_number* d_readIds,
            int numSequences,
            int totalNumValues,
            read_number* d_values,
            int* d_numValuesPerSequence,
            int* d_offsets, //numSequences + 1
            cudaStream_t stream
        ) const {
            if(numSequences == 0) return;

            QueryData* const queryData = getQueryDataFromHandle(queryHandle);

            assert(queryData->previousStage == QueryData::Stage::NumValues);

            std::vector<read_number> h_readIds{};
            if(d_readIds != nullptr){
                h_readIds.resize(numSequences);

                CUDACHECK(cudaMemcpyAsync(
                    h_readIds.data(),
                    d_readIds,
                    sizeof(read_number) * numSequences,
                    D2H,
                    stream
                ));

                CUDACHECK(cudaStreamSynchronize(stream));
            }

            std::vector<int> h_numValuesPerSequence(numSequences);
            std::vector<int> h_offsets(numSequences+1);
            std::vector<read_number> h_values(totalNumValues);


            h_offsets[0] = 0;
            auto first = h_values.data();

            for(int s = 0; s < numSequences; s++){
                auto rangesbegin = queryData->ranges.data() + s * getNumberOfMaps();
                auto end = k_way_set_union(queryData->suHandle, first, rangesbegin, getNumberOfMaps());
                if(d_readIds != nullptr){
                    auto readIdPos = std::lower_bound(
                        first,
                        end,
                        h_readIds[s]
                    );

                    if(readIdPos != end && *readIdPos == h_readIds[s]){
                        end = std::copy(readIdPos + 1, end, readIdPos);
                    }
                }
                h_numValuesPerSequence[s] = std::distance(first, end);
                h_offsets[s+1] = h_offsets[s] + std::distance(first, end);
                first = end;
            }

            CUDACHECK(cudaMemcpyAsync(
                d_values,
                h_values.data(),
                sizeof(read_number) * std::distance(h_values.data(), first),
                H2D,
                stream
            ));

            CUDACHECK(cudaMemcpyAsync(
                d_numValuesPerSequence,
                h_numValuesPerSequence.data(),
                sizeof(int) * numSequences,
                H2D,
                stream
            ));

            CUDACHECK(cudaMemcpyAsync(
                d_offsets,
                h_offsets.data(),
                sizeof(int) * (numSequences + 1),
                H2D,
                stream
            ));

            CUDACHECK(cudaStreamSynchronize(stream));

            queryData->previousStage = QueryData::Stage::Retrieve;
        }

    private:

        void determineNumValuesOnGpu(
            MinhasherHandle& queryHandle,
            const unsigned int* d_sequenceData2Bit,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int* d_numValuesPerSequence,
            int& totalNumValues,
            cudaStream_t stream
        ) const {
            QueryData* const queryData = getQueryDataFromHandle(queryHandle);

            assert(queryData->isInitialized);
            if(numSequences == 0) return;

            std::size_t cubtempbytes = 0;

            cub::DeviceScan::InclusiveSum(
                nullptr,
                cubtempbytes,
                (int*) nullptr,
                (int*) nullptr,
                numSequences,
                stream
            );

            queryData->d_minhashSignatures.resize(
                std::max(
                    getNumberOfMaps() * numSequences, 
                    int(SDIV(cubtempbytes, sizeof(std::uint64_t)))
                )
            );
            queryData->d_minhashSignatures_transposed.resize(getNumberOfMaps() * numSequences);
            queryData->h_minhashSignatures.resize(getNumberOfMaps() * numSequences);
            queryData->h_begin_offsets.resize(numSequences+1);
            queryData->d_begin_offsets.resize(numSequences+1);
            queryData->h_end_offsets.resize(numSequences+1);
            queryData->d_end_offsets.resize(numSequences+1);
            queryData->h_numValuesPerSequence.resize(numSequences);
            queryData->d_global_begin_offsets.resize(numSequences);

            std::vector<Range_t>& allRanges = queryData->allRanges;

            allRanges.resize(getNumberOfMaps() * numSequences);

            const std::size_t hashValuesPitchInElements = getNumberOfMaps();
            const int firstHashFunc = 0;

            callMinhashSignaturesKernel(
                queryData->d_minhashSignatures.get(),
                hashValuesPitchInElements,
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                getKmerSize(),
                getNumberOfMaps(),
                firstHashFunc,
                stream
            );

            helpers::call_transpose_kernel(
                queryData->d_minhashSignatures_transposed.data(), 
                queryData->d_minhashSignatures.data(), 
                numSequences, 
                getNumberOfMaps(), 
                hashValuesPitchInElements,
                stream
            );

            CUDACHECK(cudaMemcpyAsync(
                queryData->h_minhashSignatures.get(),
                queryData->d_minhashSignatures_transposed.get(),
                queryData->h_minhashSignatures.sizeInBytes(),
                H2D,
                stream
            ));
    
            CUDACHECK(cudaStreamSynchronize(stream)); //wait for D2H transfers of signatures

            nvtx::push_range("queryPrecalculatedSignatures", 6);
            totalNumValues = 0;

            const std::uint64_t kmer_mask = getKmerMask();

            int* h_numValuesPerSequence = queryData->h_numValuesPerSequence.data();
            std::fill(h_numValuesPerSequence, h_numValuesPerSequence + numSequences, 0);

            for(int map = 0; map < getNumberOfMaps(); ++map){
                for(int i = 0; i < numSequences; i++){
                    FakeGpuMinhasher::Range_t* const range = &allRanges[i * getNumberOfMaps()];
                
                    kmer_type key = queryData->h_minhashSignatures[map * numSequences + i] & kmer_mask;
                    auto entries_range = queryMap(map, key);
                    const int num = std::distance(entries_range.first, entries_range.second);
                    if(num <= getNumResultsPerMapThreshold()){
                        totalNumValues += num;
                        h_numValuesPerSequence[i] += num;
                    }else{
                        //set range to empty range
                        entries_range.second = entries_range.first;
                    }

                    //allRanges[i * getNumberOfMaps() + map] = entries_range;
                    allRanges[map * numSequences + i] = entries_range;
                }
            }

            // for(int i = 0; i < numSequences; i++){
            //     const std::uint64_t* const signature = &queryData->h_minhashSignatures[i * getNumberOfMaps()];
            //     FakeGpuMinhasher::Range_t* const range = &allRanges[i * getNumberOfMaps()];            

            //     for(int map = 0; map < getNumberOfMaps(); ++map){
            //         kmer_type key = signature[map] & kmer_mask;
            //         auto entries_range = queryMap(map, key);
            //         totalNumValues += std::distance(entries_range.first, entries_range.second);
            //         range[map] = entries_range;
            //     }
            // }
            nvtx::pop_range();

            CUDACHECK(cudaMemcpyAsync(d_numValuesPerSequence, h_numValuesPerSequence, sizeof(int) * numSequences, H2D, stream));

            // std::vector<int> numValuesPerSequence(numSequences);

            // for(int sequenceIndex = 0; sequenceIndex < numSequences; sequenceIndex++){

            //     int num = 0;

            //     for(int mapIndex = 0; mapIndex < getNumberOfMaps(); mapIndex++){
            //         const int k = sequenceIndex * getNumberOfMaps() + mapIndex;
                    
            //         const auto& range = allRanges[k];
            //         if(std::distance(range.first, range.second) <= getNumResultsPerMapThreshold())
            //             num += std::distance(range.first, range.second);
            //     }

            //     numValuesPerSequence[sequenceIndex] = num;
            // }

            //CUDACHECK(cudaMemcpyAsync(d_numValuesPerSequence, numValuesPerSequence.data(), sizeof(int) * numSequences, H2D, stream));

            queryData->previousStage = QueryData::Stage::NumValues;
        }

        void determineNumValuesOnCpu(
            MinhasherHandle& queryHandle,
            const unsigned int* d_sequenceData2Bit,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int* d_numValuesPerSequence,
            int& totalNumValues,
            cudaStream_t stream
        ) const {
            if(numSequences == 0) return;

            QueryData* const queryData = getQueryDataFromHandle(queryHandle);

            std::vector<unsigned int> h_sequenceData2Bit(encodedSequencePitchInInts * numSequences);
            std::vector<int> h_sequenceLengths(numSequences);
            std::vector<int> h_numValuesPerSequence(numSequences);

            CUDACHECK(cudaMemcpyAsync(
                h_sequenceData2Bit.data(),
                d_sequenceData2Bit,
                sizeof(unsigned int) * encodedSequencePitchInInts * numSequences,
                D2H,
                stream
            ));

            CUDACHECK(cudaMemcpyAsync(
                h_sequenceLengths.data(),
                d_sequenceLengths,
                sizeof(int) * numSequences,
                D2H,
                stream
            ));

            queryData->ranges.clear();

            CUDACHECK(cudaStreamSynchronize(stream));

            totalNumValues = 0;

            for(int s = 0; s < numSequences; s++){
                const int length = h_sequenceLengths[s];
                const unsigned int* sequence = h_sequenceData2Bit.data() + encodedSequencePitchInInts * s;                

                if(length < getKmerSize()){
                    h_numValuesPerSequence[s] = 0;
                    for(int map = 0; map < getNumberOfMaps(); ++map){
                        queryData->ranges.emplace_back(nullptr, nullptr);
                    }
                }else{                    

                    auto hashValues = calculateMinhashSignature(
                        sequence, 
                        length, 
                        getKmerSize(), 
                        getNumberOfMaps(),
                        0
                    );

                    std::for_each(
                        hashValues.begin(), hashValues.end(),
                        [kmermask = getKmerMask()](auto& hash){
                            hash &= kmermask;
                        }
                    );

                    for(int map = 0; map < getNumberOfMaps(); ++map){
                        const kmer_type key = hashValues[map];
                        auto entries_range = queryMap(map, key);
                        const int n_entries = std::distance(entries_range.first, entries_range.second);
                        if(n_entries > 0){
                            totalNumValues += n_entries;
                        }
                        queryData->ranges.emplace_back(entries_range);
                    }
                }
            }

            CUDACHECK(cudaMemcpyAsync(
                d_numValuesPerSequence,
                h_numValuesPerSequence.data(),
                sizeof(int) * numSequences,
                H2D,
                stream
            ));

            CUDACHECK(cudaStreamSynchronize(stream));

            queryData->previousStage = QueryData::Stage::NumValues;
        }

        QueryData* getQueryDataFromHandle(const MinhasherHandle& queryHandle) const{
            std::shared_lock<SharedMutex> lock(sharedmutex);

            return tempdataVector[queryHandle.getId()].get();
        }
        

        Range_t queryMap(int id, const Key_t& key) const{
            HashTable::QueryResult qr = minhashTables[id]->query(key);

            return std::make_pair(qr.valuesBegin, qr.valuesBegin + qr.numValues);
        }


        mutable int counter = 0;
        mutable SharedMutex sharedmutex{};

        float loadfactor = 0.8f;
        int maxNumKeys{};
        int kmerSize{};
        int resultsPerMapThreshold{};
        ThreadPool* threadPool;
        std::size_t memoryLimit;
        std::vector<std::unique_ptr<HashTable>> minhashTables{};
        mutable std::vector<std::unique_ptr<QueryData>> tempdataVector{};
    };






    
}
}



#endif
