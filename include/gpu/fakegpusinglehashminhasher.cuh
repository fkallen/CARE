#ifndef CARE_FAKEGPU_SINGLE_HASH_MINHASHER_CUH
#define CARE_FAKEGPU_SINGLE_HASH_MINHASHER_CUH

#include <config.hpp>

#include <gpu/gpureadstorage.cuh>
#include <gpu/cuda_unique.cuh>
#include <gpu/minhashingkernels.cuh>
#include <cpuhashtable.hpp>
#include <gpu/gpuminhasher.cuh>
#include <groupbykey.hpp>
#include <gpu/cudaerrorcheck.cuh>
#include <gpu/gpusequencehasher.cuh>
#include <gpu/cubwrappers.cuh>

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
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/device_uvector.hpp>
#include <gpu/rmm_utilities.cuh>

namespace care{
namespace gpu{



    /*
        Minhasher which can store query results in gpu memory and uses the gpu to parallelize some portions of the code
        However, hash tables reside on the host
    */
    class FakeGpuSingleHashMinhasher : public GpuMinhasher{
    public:
        using Key_t = GpuMinhasher::Key;
        using Value_t = read_number;
    private:
        using HashTable = CpuReadOnlyMultiValueHashTable<kmer_type, read_number>;

        //using Range_t = std::pair<const Value_t*, const Value_t*>;
        using SingleHashTableConstIter = std::unordered_multimap<kmer_type, read_number>::const_iterator;
        using Range_t = std::pair<SingleHashTableConstIter, SingleHashTableConstIter>;

        struct QueryData{
            static constexpr int overprovisioningPercent = 0;
            
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

            SetUnionHandle suHandle{};

            PinnedBuffer<kmer_type> h_minhashSignatures{};
            PinnedBuffer<read_number> h_candidate_read_ids_tmp{};
            PinnedBuffer<int> h_begin_offsets{};
            PinnedBuffer<int> h_end_offsets{};
            PinnedBuffer<int> h_numValuesPerSequence{};

            std::vector<Range_t> allRanges{};
            thrust::device_vector<int> d_hashFunctionNumbers{};

            MemoryUsage getMemoryInfo() const{
                MemoryUsage info;
                info.host = 0;
                info.device[deviceId] = 0;
    
                auto handlehost = [&](const auto& buff){
                    info.host += buff.capacityInBytes();
                };
    
                auto handledevice = [&](const auto& buff){
                    using ElementType = typename std::remove_reference<decltype(buff)>::type::value_type;
                    info.device[deviceId] += buff.size() * sizeof(ElementType);
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
    
                handlevector(allRanges);
                handledevice(d_hashFunctionNumbers);
    
                return info;
            }

            void destroy(){
                cub::SwitchDevice sd{deviceId};

                h_minhashSignatures.destroy();
                h_candidate_read_ids_tmp.destroy();
                h_begin_offsets.destroy();
                h_end_offsets.destroy();
                h_numValuesPerSequence.destroy();

                allRanges.clear();
                allRanges.shrink_to_fit();

                d_hashFunctionNumbers.clear();
                d_hashFunctionNumbers.shrink_to_fit();

                isInitialized = false;
            }
        };

        
    public:

        FakeGpuSingleHashMinhasher() : FakeGpuSingleHashMinhasher(0, 50, 16, 0.8f){

        }

        FakeGpuSingleHashMinhasher(int maxNumKeys_, int maxValuesPerKey, int k, float loadfactor_)
            : loadfactor(loadfactor_), maxNumKeys(maxNumKeys_), kmerSize(k), resultsPerMapThreshold(maxValuesPerKey){

        }

        FakeGpuSingleHashMinhasher(const FakeGpuSingleHashMinhasher&) = delete;
        FakeGpuSingleHashMinhasher(FakeGpuSingleHashMinhasher&&) = default;
        FakeGpuSingleHashMinhasher& operator=(const FakeGpuSingleHashMinhasher&) = delete;
        FakeGpuSingleHashMinhasher& operator=(FakeGpuSingleHashMinhasher&&) = default;


        void constructFromReadStorage(
            const FileOptions &/*fileOptions*/,
            const RuntimeOptions &runtimeOptions,
            const MemoryOptions& memoryOptions,
            std::uint64_t /*nReads*/,
            const CorrectionOptions& correctionOptions,
            const GpuReadStorage& gpuReadStorage
        ){
            
            std::size_t memoryBeforeSinglehashtable = getAvailableMemoryInKB() * 1024;

            allKeys.reserve(
                (memoryBeforeSinglehashtable - 6ull * (1 << 30)) / sizeof(kmer_type)
            );

            {
                auto& readStorage = gpuReadStorage;
                const auto& deviceIds = runtimeOptions.deviceIds;

                int deviceId = deviceIds[0];

                cub::SwitchDevice sd{deviceId};

                const int requestedNumberOfMaps = correctionOptions.numHashFunctions;
                numSmallest = requestedNumberOfMaps;

                const read_number numReads = readStorage.getNumberOfReads();
                const int maximumSequenceLength = readStorage.getSequenceLengthUpperBound();

                auto sequencehandle = gpuReadStorage.makeHandle();
                const std::size_t encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength);

                rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();

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

                cudaStream_t stream = cudaStreamPerThread;
                
                rmm::device_uvector<unsigned int> d_sequenceData(encodedSequencePitchInInts * parallelReads, stream, mr);
                rmm::device_uvector<int> d_lengths(parallelReads, stream, mr);
                rmm::device_uvector<read_number> d_indices(parallelReads, stream, mr);
                
                helpers::SimpleAllocationPinnedHost<read_number, 0> h_indices(parallelReads);
                            
                ThreadPool tpForHashing(runtimeOptions.threads);
                ThreadPool tpForCompacting(std::min(2,runtimeOptions.threads));

                
                setMemoryLimitForConstruction(maxMemoryForTables);
                
                //std::size_t bytesOfCachedConstructedTables = 0;
                int remainingHashFunctions = requestedNumberOfMaps;
                bool keepGoing = true;

                //while(remainingHashFunctions > 0 && keepGoing)
                {

                    setThreadPool(&tpForHashing);

                    singlehashtable = std::unordered_multimap<kmer_type, read_number>(0); //numReads

                    std::cout << "Constructing maps\n";

                    std::size_t numKeys = 0;

                    for (int iter = 0; iter < numIters; iter++){
                        read_number readIdBegin = iter * parallelReads;
                        read_number readIdEnd = std::min((iter + 1) * parallelReads, numReads);

                        const std::size_t curBatchsize = readIdEnd - readIdBegin;

                        std::iota(h_indices.get(), h_indices.get() + curBatchsize, readIdBegin);

                        CUDACHECK(cudaMemcpyAsync(d_indices.data(), h_indices, sizeof(read_number) * curBatchsize, H2D, stream));

                        gpuReadStorage.gatherSequences(
                            sequencehandle,
                            d_sequenceData.data(),
                            encodedSequencePitchInInts,
                            makeAsyncConstBufferWrapper(h_indices.data()),
                            d_indices.data(),
                            curBatchsize,
                            stream,
                            mr
                        );
                    
                        gpuReadStorage.gatherSequenceLengths(
                            sequencehandle,
                            d_lengths.data(),
                            d_indices.data(),
                            curBatchsize,
                            stream
                        );

                        constexpr bool dryrun = false;

                        auto numNewKeys = insert(
                            dryrun,
                            d_sequenceData.data(),
                            curBatchsize,
                            d_lengths.data(),
                            encodedSequencePitchInInts,
                            d_indices.data(),
                            h_indices,
                            stream
                        );

                        numKeys += numNewKeys;

                        CUDACHECK(cudaStreamSynchronize(stream));
                    }

                    CUDACHECK(cudaStreamSynchronize(stream));

                    std::cerr << "numKeys = " << numKeys << "\n";

                    std::cerr << "Skip compacting unordered multimap\n";
                    // if(tpForCompacting.getConcurrency() > 1){
                    //     setThreadPool(&tpForCompacting);
                    // }else{
                    //     setThreadPool(nullptr);
                    // }
                    
                    // finalize();

                    // remainingHashFunctions -= addedHashFunctions;
                }

                setThreadPool(nullptr); 
                
                gpuReadStorage.destroyHandle(sequencehandle);
            }

            std::cerr << "allKeys.size() " << allKeys.size() << "\n";
            std::cerr << "allKeys.capacity() " << allKeys.capacity() << "\n";

            std::size_t memoryAfterSinglehashtable = getAvailableMemoryInKB() * 1024;

            singlehashtablememoryusage = memoryAfterSinglehashtable > memoryBeforeSinglehashtable ? memoryAfterSinglehashtable - memoryBeforeSinglehashtable : 0;

            std::cerr << "singlehashtablememoryusage: " << singlehashtablememoryusage << "\n";

            std::sort(allKeys.begin(), allKeys.end());
            auto it = std::unique(allKeys.begin(), allKeys.end());
            allKeys.erase(it, allKeys.end());

            std::cerr << "allKeys.size() unique " << allKeys.size() << "\n";
        }
    

        MinhasherHandle makeMinhasherHandle() const override {
            auto data = std::make_unique<QueryData>();
            data->d_hashFunctionNumbers.resize(getNumberOfMaps());

            thrust::sequence(thrust::device, data->d_hashFunctionNumbers.begin(), data->d_hashFunctionNumbers.end(), 0);

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

        void determineNumValues(
            MinhasherHandle& queryHandle,
            const unsigned int* d_sequenceData2Bit,
            std::size_t encodedSequencePitchInInts,
            const int* d_sequenceLengths,
            int numSequences,
            int* d_numValuesPerSequence,
            int& totalNumValues,
            cudaStream_t stream,
            rmm::mr::device_memory_resource* mr
        ) const override {
            QueryData* const queryData = getQueryDataFromHandle(queryHandle);

            DEBUGSTREAMSYNC(stream);

            assert(queryData->isInitialized);
            if(numSequences == 0) return;

            // queryData->h_minhashSignatures.resize(numSmallest * numSequences);
            queryData->h_numValuesPerSequence.resize(numSequences);
            std::vector<Range_t>& allRanges = queryData->allRanges;

            allRanges.resize(numSmallest * numSequences);

            GPUSequenceHasher<kmer_type> hasher;

            auto hashResult = hasher.getTopSmallestKmerHashes(
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                getKmerSize(),
                numSmallest,
                stream,
                mr
            );

            rmm::device_uvector<int> d_numHashesPerSequencePrefixSum(numSequences, stream, mr);
            CubCallWrapper(mr).cubExclusiveSum(hashResult.d_numPerSequences.data(), d_numHashesPerSequencePrefixSum.data(), numSequences, stream);

            std::vector<kmer_type> h_hashes(numSequences * numSmallest);
            std::vector<int> h_numHashesPerSequence(numSequences);
            std::vector<int> h_numHashesPerSequencePrefixSum(numSequences);

            CUDACHECK(cudaMemcpyAsync(
                h_hashes.data(), 
                hashResult.d_hashvalues.data(), 
                sizeof(kmer_type) * numSmallest * numSequences, 
                D2H, 
                stream
            ));

            CUDACHECK(cudaMemcpyAsync(
                h_numHashesPerSequence.data(), 
                hashResult.d_numPerSequences.data(), 
                sizeof(int) * numSequences, 
                D2H, 
                stream
            ));

            CUDACHECK(cudaMemcpyAsync(
                h_numHashesPerSequencePrefixSum.data(), 
                d_numHashesPerSequencePrefixSum.data(), 
                sizeof(int) * numSequences, 
                D2H, 
                stream
            ));

            CUDACHECK(cudaStreamSynchronize(stream));

            for(int s = 0; s < numSequences; s++){
                const int num = h_numHashesPerSequence[s];

                int numValues = 0;

                for(int i = 0; i < num; i++){
                    auto [b,e] = singlehashtable.equal_range(h_hashes[h_numHashesPerSequencePrefixSum[s] + i]);
                    numValues += std::distance(b,e);

                    allRanges[s * numSmallest + i] = {b, e};
                }

                for(int i = num; i < numSmallest; i++){
                    allRanges[s * numSmallest + i] = {singlehashtable.end(), singlehashtable.end()};
                }

                queryData->h_numValuesPerSequence[s] = numValues;

                totalNumValues += numValues;
            }

            DEBUGSTREAMSYNC(stream);

            CUDACHECK(cudaMemcpyAsync(d_numValuesPerSequence, queryData->h_numValuesPerSequence.data(), sizeof(int) * numSequences, H2D, stream));

            DEBUGSTREAMSYNC(stream);

            queryData->previousStage = QueryData::Stage::NumValues;
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
            QueryData* const queryData = getQueryDataFromHandle(queryHandle);

            DEBUGSTREAMSYNC(stream);

            assert(queryData->isInitialized);
            if(numSequences == 0) return;

            assert(queryData->previousStage == QueryData::Stage::NumValues);

            //std::cerr << "totalNumValues: " << totalNumValues << "\n";

            if(totalNumValues == 0){
                cudaMemsetAsync(d_numValuesPerSequence, 0, sizeof(int) * numSequences, stream);
                cudaMemsetAsync(d_offsets, 0, sizeof(int) * (numSequences + 1), stream);

                queryData->previousStage = QueryData::Stage::Retrieve;

                DEBUGSTREAMSYNC(stream);

                return;
            }

            constexpr int roundUpTo = 10000;
            const int roundedTotalNum = SDIV(totalNumValues, roundUpTo) * roundUpTo;
            queryData->h_candidate_read_ids_tmp.resize(roundedTotalNum);

            rmm::device_uvector<read_number> d_candidate_read_ids_tmp(roundedTotalNum, stream, mr);

            queryData->h_begin_offsets.resize(numSequences+1);
            queryData->h_end_offsets.resize(numSequences+1);


            //results will be in Current() buffer
            cub::DoubleBuffer<read_number> d_values_dblbuf(d_values, d_candidate_read_ids_tmp.data());

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

                        std::transform(
                            range.first, 
                            range.second, 
                            queryData->h_candidate_read_ids_tmp.data() 
                                + queryData->h_begin_offsets[i] + processedPerSequence[i],
                            [](const auto mapIterator){
                                return mapIterator.second;
                            }
                        );

                        processedPerSequence[i] += std::distance(range.first, range.second);
                    }
                }
            };

            

            copyHitsToPinnedMemory();

            DEBUGSTREAMSYNC(stream);

            CUDACHECK(cudaMemcpyAsync(
                d_values_dblbuf.Current(),
                queryData->h_candidate_read_ids_tmp.data(),
                sizeof(read_number) * totalNumValues,
                H2D,
                stream
            ));

            DEBUGSTREAMSYNC(stream);

            rmm::device_uvector<int> d_begin_offsets((numSequences + 1), stream, mr);
            rmm::device_uvector<int> d_end_offsets((numSequences + 1), stream, mr);
            rmm::device_uvector<int> d_global_begin_offsets((numSequences), stream, mr);

            DEBUGSTREAMSYNC(stream);

            //copy h_endoffsets to d_endoffsets.
            //Then copy d_endoffsets to d_begin_offsets shifted to the right by 1.
            //Then copy d_begin_offsets to d_global_begin_offsets
            helpers::lambda_kernel<<<SDIV(numSequences, 1024), 1024, 0, stream>>>(
                [
                    numSequences,
                    h_end_offsets = queryData->h_end_offsets.data(),
                    d_end_offsets = d_end_offsets.data(),
                    d_begin_offsets = d_begin_offsets.data(),
                    d_global_begin_offsets = d_global_begin_offsets.data()
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

            DEBUGSTREAMSYNC(stream);

            GpuSegmentedUnique::unique(
                d_values_dblbuf.Current(), //input
                totalNumValues,
                d_values_dblbuf.Alternate(), //output
                d_numValuesPerSequence,
                numSequences,
                d_begin_offsets.data(), //device accessible
                d_end_offsets.data(), //device accessible
                queryData->h_begin_offsets,
                queryData->h_end_offsets,
                0,
                sizeof(read_number) * 8,
                stream,
                mr
            );

            DEBUGSTREAMSYNC(stream);

            ::destroy(d_begin_offsets, stream);
            ::destroy(d_end_offsets, stream);

            if(d_readIds != nullptr){

                //remove self read ids (inplace)
                //--------------------------------------------------------------------
                callFindAndRemoveFromSegmentKernel<read_number,128,4>(
                    d_readIds,
                    d_values_dblbuf.Alternate(),
                    numSequences,
                    d_numValuesPerSequence,
                    d_global_begin_offsets.data(),
                    stream
                );

                DEBUGSTREAMSYNC(stream);

            }

            int* d_newOffsets = d_offsets;

            CUDACHECK(cudaMemsetAsync(d_newOffsets, 0, sizeof(int), stream));

            DEBUGSTREAMSYNC(stream);

            std::size_t cubtempbytes = 0;
            CUDACHECK(cub::DeviceScan::InclusiveSum(
                nullptr,
                cubtempbytes,
                d_numValuesPerSequence,
                d_newOffsets + 1,
                numSequences,
                stream
            ));

            rmm::device_uvector<char> d_cubTemp(cubtempbytes, stream);

            CUDACHECK(cub::DeviceScan::InclusiveSum(
                d_cubTemp.data(),
                cubtempbytes,
                d_numValuesPerSequence,
                d_newOffsets + 1,
                numSequences,
                stream
            ));

            DEBUGSTREAMSYNC(stream);

            //copy final remaining values into contiguous range
            helpers::lambda_kernel<<<numSequences, 128, 0, stream>>>(
                [
                    d_values_in = d_values_dblbuf.Alternate(),
                    d_values_out = d_values_dblbuf.Current(),
                    numSequences,
                    d_numValuesPerSequence,
                    d_offsets = d_global_begin_offsets.data(),
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

            DEBUGSTREAMSYNC(stream);

            queryData->previousStage = QueryData::Stage::Retrieve;
        }

        void compact(cudaStream_t /*stream*/) {
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

            // const int num = minhashTables.size();
            // for(int i = 0, l = 0; i < num; i++){
            //     auto& ptr = minhashTables[i];
            
            //     if(!ptr->isInitialized()){
            //         //after processing 3 tables, available memory should be sufficient for multithreading
            //         if(l >= 3){
            //             ptr->finalize(groupByKey, threadPool);
            //         }else{
            //             ptr->finalize(groupByKey, nullptr);
            //         }
            //         l++;
            //     }                
            // }

            if(threadPool != nullptr){
                threadPool->wait();
            }
        }

        MemoryUsage getMemoryInfo() const noexcept override{
            MemoryUsage result;

            result.host = singlehashtablememoryusage;

            // result.host = sizeof(HashTable) * minhashTables.size();
            
            // for(const auto& tableptr : minhashTables){
            //     auto m = tableptr->getMemoryInfo();
            //     result.host += m.host;

            //     std::cerr << std::distance(minhashTables.data(), &tableptr) << ": " << m.host << "\n";

            //     for(auto pair : m.device){
            //         result.device[pair.first] += pair.second;
            //     }
            // }

            return result;
        }

        MemoryUsage getMemoryInfo(const MinhasherHandle& handle) const noexcept override{
            return getQueryDataFromHandle(handle)->getMemoryInfo();
        }

        int getNumResultsPerMapThreshold() const noexcept override{
            return resultsPerMapThreshold;
        }
        
        int getNumberOfMaps() const noexcept override{
            return 1;
        }

        void destroy() {
            //minhashTables.clear();
            singlehashtable = std::unordered_multimap<kmer_type, read_number>();
        }

        int getKmerSize() const noexcept override{
            return kmerSize;
        }

        bool hasGpuTables() const noexcept override {
            return false;
        }

        void finalize(cudaStream_t stream = 0){
            compact(stream);
        }

        std::uint64_t getKmerMask() const{
            constexpr int maximum_kmer_length = max_k<std::uint64_t>::value;

            return std::numeric_limits<std::uint64_t>::max() >> ((maximum_kmer_length - getKmerSize()) * 2);
        }


        // void writeToStream(std::ostream& os) const{

        //     os.write(reinterpret_cast<const char*>(&kmerSize), sizeof(int));
        //     os.write(reinterpret_cast<const char*>(&resultsPerMapThreshold), sizeof(int));
        //     os.write(reinterpret_cast<const char*>(&loadfactor), sizeof(float));

        //     const int numTables = getNumberOfMaps();
        //     os.write(reinterpret_cast<const char*>(&numTables), sizeof(int));

        //     for(const auto& tableptr : minhashTables){
        //         tableptr->writeToStream(os);
        //     }
        // }

        // int loadFromStream(std::ifstream& is, int numMapsUpperLimit = std::numeric_limits<int>::max()){
        //     destroy();

        //     is.read(reinterpret_cast<char*>(&kmerSize), sizeof(int));
        //     is.read(reinterpret_cast<char*>(&resultsPerMapThreshold), sizeof(int));
        //     is.read(reinterpret_cast<char*>(&loadfactor), sizeof(float));

        //     int numMaps = 0;

        //     is.read(reinterpret_cast<char*>(&numMaps), sizeof(int));

        //     const int mapsToLoad = std::min(numMapsUpperLimit, numMaps);

        //     for(int i = 0; i < mapsToLoad; i++){
        //         auto ptr = std::make_unique<HashTable>();
        //         ptr->loadFromStream(is);
        //         minhashTables.emplace_back(std::move(ptr));
        //     }

        //     return mapsToLoad;
        // } 

        // int addHashfunctions(int numExtraFunctions){
        //     int added = 0;
        //     const int cur = minhashTables.size();

        //     assert(!(numExtraFunctions + cur > 64));

        //     std::size_t bytesOfCachedConstructedTables = 0;
        //     for(const auto& ptr : minhashTables){
        //         auto memusage = ptr->getMemoryInfo();
        //         bytesOfCachedConstructedTables += memusage.host;
        //     }

        //     std::size_t requiredMemPerTable = (sizeof(kmer_type) + sizeof(read_number)) * maxNumKeys;
        //     int numTablesToConstruct = (memoryLimit - bytesOfCachedConstructedTables) / requiredMemPerTable;
        //     numTablesToConstruct -= 2; // keep free memory of 2 tables to perform transformation 
        //     numTablesToConstruct = std::min(numTablesToConstruct, numExtraFunctions);

        //     for(int i = 0; i < numTablesToConstruct; i++){
        //         try{
        //             auto ptr = std::make_unique<HashTable>(maxNumKeys, loadfactor);

        //             minhashTables.emplace_back(std::move(ptr));
        //             added++;
        //         }catch(...){

        //         }
        //     }

        //     return added;
        // } 

        std::size_t insert(
            bool dryrun,
            const unsigned int* d_sequenceData2Bit,
            int numSequences,
            const int* d_sequenceLengths,
            std::size_t encodedSequencePitchInInts,
            const read_number* /*d_readIds*/,
            const read_number* h_readIds,
            cudaStream_t stream,
            rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource()
        ){

            GPUSequenceHasher<kmer_type> hasher;

            auto hashResult = hasher.getTopSmallestKmerHashes(
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                getKmerSize(),
                numSmallest,
                stream,
                mr
            );

            rmm::device_uvector<int> d_numHashesPerSequencePrefixSum(numSequences, stream, mr);
            CubCallWrapper(mr).cubExclusiveSum(hashResult.d_numPerSequences.data(), d_numHashesPerSequencePrefixSum.data(), numSequences, stream);

            std::vector<kmer_type> h_hashes(numSequences * numSmallest);
            std::vector<int> h_numHashesPerSequence(numSequences);
            std::vector<int> h_numHashesPerSequencePrefixSum(numSequences);

            CUDACHECK(cudaMemcpyAsync(
                h_hashes.data(), 
                hashResult.d_hashvalues.data(), 
                sizeof(kmer_type) * numSmallest * numSequences, 
                D2H, 
                stream
            ));

            CUDACHECK(cudaMemcpyAsync(
                h_numHashesPerSequence.data(), 
                hashResult.d_numPerSequences.data(), 
                sizeof(int) * numSequences, 
                D2H, 
                stream
            ));

            CUDACHECK(cudaMemcpyAsync(
                h_numHashesPerSequencePrefixSum.data(), 
                d_numHashesPerSequencePrefixSum.data(), 
                sizeof(int) * numSequences, 
                D2H, 
                stream
            ));

            CUDACHECK(cudaStreamSynchronize(stream));

            std::size_t numKeys = 0;

            for(int s = 0; s < numSequences; s++){
                const int num = h_numHashesPerSequence[s];
                if(num > 0){
                    numKeys += num;
                    if(!dryrun){
                        std::vector<std::pair<kmer_type,read_number>> entries(num);
                        for(int i = 0; i < num; i++){
                            entries[i] = std::make_pair(h_hashes[h_numHashesPerSequencePrefixSum[s] + i], h_readIds[s]);
                        }
                        //singlehashtable.insert(entries.begin(), entries.end());
                        allKeys.insert(allKeys.end(), &h_hashes[h_numHashesPerSequencePrefixSum[s]], &h_hashes[h_numHashesPerSequencePrefixSum[s] + num]);
                    }
                }
            }

            return numKeys;
        }   

        void setThreadPool(ThreadPool* tp){
            threadPool = tp;
        }

        void setMemoryLimitForConstruction(std::size_t limit){
            memoryLimit = limit;
        }

    private:

        
        QueryData* getQueryDataFromHandle(const MinhasherHandle& queryHandle) const{
            std::shared_lock<SharedMutex> lock(sharedmutex);

            return tempdataVector[queryHandle.getId()].get();
        }
        

        mutable int counter = 0;
        mutable SharedMutex sharedmutex{};

        float loadfactor = 0.8f;
        int numSmallest{};
        int maxNumKeys{};
        int kmerSize{};
        int resultsPerMapThreshold{};
        ThreadPool* threadPool;
        std::size_t memoryLimit;
        std::size_t singlehashtablememoryusage{};
        //std::vector<std::unique_ptr<HashTable>> minhashTables{};
        std::unordered_multimap<kmer_type, read_number> singlehashtable{};
        std::vector<kmer_type> allKeys{};
        std::unique_ptr<HashTable> singlecustomhashtable{};
        mutable std::vector<std::unique_ptr<QueryData>> tempdataVector{};
    };






    
}
}



#endif
