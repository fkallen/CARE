#ifndef CARE_FAKEGPUMINHASHER_CUH
#define CARE_FAKEGPUMINHASHER_CUH

#include <config.hpp>

#include <gpu/gpureadstorage.cuh>
#include <gpu/cuda_unique.cuh>
#include <cpuhashtable.hpp>
#include <gpu/gpuminhasher.cuh>
#include <groupbykey.hpp>
#include <gpu/cudaerrorcheck.cuh>
#include <gpu/gpusequencehasher.cuh>
#include <gpu/kernels.hpp>
#include <gpu/cubwrappers.cuh>
#include <options.hpp>
#include <util.hpp>
#include <hpc_helpers.cuh>
#include <filehelpers.hpp>

#include <sequencehelpers.hpp>
#include <memorymanagement.hpp>
#include <threadpool.hpp>

#include <sharedmutex.hpp>

#include <cuda.h> //CUDA_VERSION

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
#include <thrust/iterator/zip_iterator.h>

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
    class FakeGpuMinhasher : public GpuMinhasher{
    public:
        using Key_t = GpuMinhasher::Key;
        using Value_t = read_number;
    private:
        using HashTable = CpuReadOnlyMultiValueHashTable<kmer_type, read_number>;
        using Range_t = std::pair<const Value_t*, const Value_t*>;

    public: //should be private, but nvcc bug can cause compilation failure in some cuda versions
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
            PinnedBuffer<read_number> h_anchorReadIds{};
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

        FakeGpuMinhasher() : FakeGpuMinhasher(0, 50, 16, 0.8f){

        }

        FakeGpuMinhasher(int maxNumKeys_, int maxValuesPerKey, int k, float loadfactor_)
            : loadfactor(loadfactor_), maxNumKeys(maxNumKeys_), kmerSize(k), resultsPerMapThreshold(maxValuesPerKey){

        }

        FakeGpuMinhasher(const FakeGpuMinhasher&) = delete;
        FakeGpuMinhasher(FakeGpuMinhasher&&) = default;
        FakeGpuMinhasher& operator=(const FakeGpuMinhasher&) = delete;
        FakeGpuMinhasher& operator=(FakeGpuMinhasher&&) = default;
   

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

            queryData->h_minhashSignatures.resize(getNumberOfMaps() * numSequences);
            queryData->h_numValuesPerSequence.resize(numSequences);

            std::vector<Range_t>& allRanges = queryData->allRanges;

            allRanges.resize(getNumberOfMaps() * numSequences);

            const std::size_t hashValuesPitchInElements = getNumberOfMaps();
            //const int firstHashFunc = 0;

            GPUSequenceHasher<kmer_type> hasher;
            auto hashResult = hasher.hash(
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                getKmerSize(),
                getNumberOfMaps(),
                thrust::raw_pointer_cast(queryData->d_hashFunctionNumbers.data()),
                stream,
                mr
            );

            DEBUGSTREAMSYNC(stream);

            rmm::device_uvector<kmer_type> d_minhashSignatures_transposed(
                getNumberOfMaps() * numSequences,
                stream,
                mr
            );

            helpers::call_transpose_kernel(
                d_minhashSignatures_transposed.data(), 
                //d_minhashSignatures.data(), 
                hashResult.d_hashvalues.data(),
                numSequences, 
                getNumberOfMaps(), 
                hashValuesPitchInElements,
                stream
            );

            DEBUGSTREAMSYNC(stream);

            CUDACHECK(cudaMemcpyAsync(
                queryData->h_minhashSignatures.get(),
                d_minhashSignatures_transposed.data(),
                queryData->h_minhashSignatures.sizeInBytes(),
                D2H,
                stream
            ));

            DEBUGSTREAMSYNC(stream);

            ::destroy(d_minhashSignatures_transposed, stream);

            rmm::device_uvector<bool> d_isValid_transposed(
                getNumberOfMaps() * numSequences,
                stream,
                mr
            );

            helpers::call_transpose_kernel(
                d_isValid_transposed.data(), 
                hashResult.d_isValid.data(),
                numSequences, 
                getNumberOfMaps(), 
                getNumberOfMaps(),
                stream
            );

            DEBUGSTREAMSYNC(stream);

            auto h_isValid = std::make_unique<bool[]>(numSequences * getNumberOfMaps());

            CUDACHECK(cudaMemcpyAsync(
                h_isValid.get(),
                d_isValid_transposed.data(),
                sizeof(bool) * d_isValid_transposed.size(),
                D2H,
                stream
            ));

            DEBUGSTREAMSYNC(stream);
    
            CUDACHECK(cudaStreamSynchronize(stream)); //wait for D2H transfers of signatures

            nvtx::push_range("queryPrecalculatedSignatures", 6);
            totalNumValues = 0;

            const std::uint64_t kmer_mask = getKmerMask();

            int* h_numValuesPerSequence = queryData->h_numValuesPerSequence.data();
            std::fill(h_numValuesPerSequence, h_numValuesPerSequence + numSequences, 0);

            for(int map = 0; map < getNumberOfMaps(); ++map){
                for(int i = 0; i < numSequences; i++){
                    FakeGpuMinhasher::Range_t* const range = &allRanges[i * getNumberOfMaps()];

                    const bool valid = h_isValid[map * numSequences + i];

                    if(valid){
                
                        kmer_type key = queryData->h_minhashSignatures[map * numSequences + i];/* & kmer_mask*/;
                        auto entries_range = queryMap(map, key);
                        const int num = std::distance(entries_range.first, entries_range.second);
                        if(num <= getNumResultsPerMapThreshold()){
                            totalNumValues += num;
                            h_numValuesPerSequence[i] += num;
                        }else{
                            //set range to empty range
                            entries_range.second = entries_range.first;
                        }
                        allRanges[map * numSequences + i] = entries_range;
                    }else{
                        allRanges[map * numSequences + i].first = nullptr;
                        allRanges[map * numSequences + i].second = nullptr;
                    }

                    //allRanges[i * getNumberOfMaps() + map] = entries_range;
                }
            }

            nvtx::pop_range();

            DEBUGSTREAMSYNC(stream);

            CUDACHECK(cudaMemcpyAsync(d_numValuesPerSequence, h_numValuesPerSequence, sizeof(int) * numSequences, H2D, stream));

            DEBUGSTREAMSYNC(stream);

            queryData->previousStage = QueryData::Stage::NumValues;
        }

        void retrieveValues(
            MinhasherHandle& queryHandle,
            int numSequences,
            int totalNumValues,
            read_number* d_values,
            const int* d_numValuesPerSequence,
            int* d_offsets, //numSequences + 1
            cudaStream_t stream,
            rmm::mr::device_memory_resource* mr
        ) const override {
            QueryData* const queryData = getQueryDataFromHandle(queryHandle);

            DEBUGSTREAMSYNC(stream);

            assert(queryData->isInitialized);
            assert(queryData->previousStage == QueryData::Stage::NumValues);
            queryData->previousStage = QueryData::Stage::Retrieve;

            if(numSequences == 0) return;

            if(totalNumValues == 0){
                CUDACHECK(cudaMemsetAsync(d_offsets, 0, sizeof(int) * (numSequences + 1), stream));
                return;
            }

            CubCallWrapper(mr).cubInclusiveSum(
                d_numValuesPerSequence,
                d_offsets + 1,
                numSequences,
                stream
            );

            CUDACHECK(cudaMemsetAsync(d_offsets, 0, sizeof(int), stream));

            constexpr int roundUpTo = 10000;
            const int roundedTotalNum = SDIV(totalNumValues, roundUpTo) * roundUpTo;
            queryData->h_candidate_read_ids_tmp.resize(roundedTotalNum);

            queryData->h_begin_offsets.resize(numSequences+1);
            queryData->h_end_offsets.resize(numSequences+1);

            auto copyHitsToPinnedMemory = [queryData, numSequences, minhasher = this](){
                int* h_numValuesPerSequence = queryData->h_numValuesPerSequence.data();

                std::exclusive_scan(
                    h_numValuesPerSequence, 
                    h_numValuesPerSequence + numSequences, 
                    queryData->h_begin_offsets.begin(),
                    0
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
                d_values,
                queryData->h_candidate_read_ids_tmp.data(),
                sizeof(read_number) * totalNumValues,
                H2D,
                stream
            ));
            
        }

        void compact(cudaStream_t /*stream*/) override{
            int id;
            CUDACHECK(cudaGetDevice(&id));

            auto groupByKey = [&](auto& keys, auto& values, auto& countsPrefixSum){
                constexpr bool valuesOfSameKeyMustBeSorted = false;
                const int maxValuesPerKey = getNumResultsPerMapThreshold();

                //if only 1 value exists, it belongs to the anchor read itself and does not need to be stored.
                const int minValuesPerKey = 2;

                bool success = false;

                using GroupByKeyCpuOp = GroupByKeyCpu<Key_t, Value_t, read_number>;
                using GroupByKeyGpuOp = GroupByKeyGpu<Key_t, Value_t, read_number>;
                
                GroupByKeyGpuOp groupByKeyGpu(valuesOfSameKeyMustBeSorted, maxValuesPerKey, minValuesPerKey);
                success = groupByKeyGpu.execute(keys, values, countsPrefixSum);         

                if(!success){
                    GroupByKeyCpuOp groupByKeyCpu(valuesOfSameKeyMustBeSorted, maxValuesPerKey, minValuesPerKey);
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

                //std::cerr << std::distance(minhashTables.data(), &tableptr) << ": " << m.host << "\n";

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


        void writeToStream(std::ostream& os) const override{

            os.write(reinterpret_cast<const char*>(&kmerSize), sizeof(int));
            os.write(reinterpret_cast<const char*>(&resultsPerMapThreshold), sizeof(int));
            os.write(reinterpret_cast<const char*>(&loadfactor), sizeof(float));

            const int numTables = getNumberOfMaps();
            os.write(reinterpret_cast<const char*>(&numTables), sizeof(int));

            for(const auto& tableptr : minhashTables){
                tableptr->writeToStream(os);
            }
        }

        int loadFromStream(std::ifstream& is, int numMapsUpperLimit) override{
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

        bool canWriteToStream() const noexcept override { return true; };
        bool canLoadFromStream() const noexcept override { return true; };

        int addHashTables(int numAdditionalTables, const int* /*hashFunctionIds*/, cudaStream_t /*stream*/) override{
            int added = 0;
            const int cur = minhashTables.size();

            assert(!(numAdditionalTables + cur > 64));

            std::size_t bytesOfCachedConstructedTables = 0;
            for(const auto& ptr : minhashTables){
                auto memusage = ptr->getMemoryInfo();
                bytesOfCachedConstructedTables += memusage.host;
            }

            std::size_t requiredMemPerTable = (sizeof(kmer_type) + sizeof(read_number)) * maxNumKeys;
            int numTablesToConstruct = (memoryLimit - bytesOfCachedConstructedTables) / requiredMemPerTable;
            numTablesToConstruct -= 2; // keep free memory of 2 tables to perform transformation 
            numTablesToConstruct = std::min(numTablesToConstruct, numAdditionalTables);

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
            const unsigned int* d_sequenceData2Bit,
            int numSequences,
            const int* d_sequenceLengths,
            std::size_t encodedSequencePitchInInts,
            const read_number* /*d_readIds*/,
            const read_number* h_readIds,
            int firstHashfunction,
            int numHashfunctions,
            const int* h_hashFunctionNumbers,
            cudaStream_t stream,
            rmm::mr::device_memory_resource* mr
        ) override {
            ThreadPool::ParallelForHandle pforHandle{};

            ForLoopExecutor forLoopExecutor(threadPool, &pforHandle);

            const std::size_t signaturesRowPitchElements = numHashfunctions;

            assert(firstHashfunction + numHashfunctions <= int(minhashTables.size()));

            rmm::device_uvector<int> d_hashFunctionNumbers(numHashfunctions, stream, mr);

            CUDACHECK(cudaMemcpyAsync(
                d_hashFunctionNumbers.data(), 
                h_hashFunctionNumbers, 
                sizeof(int) * numHashfunctions, 
                H2D, 
                stream
            ));

            GPUSequenceHasher<kmer_type> hasher;

            auto hashResult = hasher.hash(
                d_sequenceData2Bit,
                encodedSequencePitchInInts,
                numSequences,
                d_sequenceLengths,
                getKmerSize(),
                numHashfunctions,
                d_hashFunctionNumbers.data(),
                stream,
                mr
            );

            rmm::device_uvector<kmer_type> d_signatures_transposed(signaturesRowPitchElements * numSequences, stream, mr);
            helpers::call_transpose_kernel(
                d_signatures_transposed.data(), 
                hashResult.d_hashvalues.data(),
                numSequences, 
                signaturesRowPitchElements, 
                signaturesRowPitchElements,
                stream
            );

            rmm::device_uvector<bool> d_isValid_transposed(
                numHashfunctions * numSequences,
                stream,
                mr
            );

            helpers::call_transpose_kernel(
                d_isValid_transposed.data(), 
                hashResult.d_isValid.data(),
                numSequences, 
                numHashfunctions, 
                numHashfunctions,
                stream
            );

            auto h_isValid = std::make_unique<bool[]>(numSequences * numHashfunctions);
            auto h_signatures_transposed = std::make_unique<kmer_type[]>(signaturesRowPitchElements * numSequences);

            CUDACHECK(cudaMemcpyAsync(
                h_isValid.get(),
                d_isValid_transposed.data(),
                sizeof(bool) * d_isValid_transposed.size(),
                D2H,
                stream
            ));

            CUDACHECK(cudaMemcpyAsync(
                h_signatures_transposed.get(), 
                d_signatures_transposed.data(), 
                sizeof(kmer_type) * signaturesRowPitchElements * numSequences, 
                D2H, 
                stream
            ));

            CUDACHECK(cudaStreamSynchronize(stream));

            auto loopbody = [&](auto begin, auto end, int /*threadid*/){
                for(int h = begin; h < end; h++){

                    const kmer_type* const orighashesBegin = &h_signatures_transposed[h * numSequences];
                    const bool* const validflagsBegin = &h_isValid[h * numSequences];

                    std::vector<kmer_type> hashes(numSequences);

                    auto hashesEnd = select_if(
                        orighashesBegin,
                        orighashesBegin + numSequences,
                        validflagsBegin,
                        hashes.begin()
                    );

                    std::vector<read_number> validIds(numSequences);

                    auto validIdsEnd = select_if(
                        h_readIds,
                        h_readIds + numSequences,
                        validflagsBegin,
                        validIds.begin()
                    );
                    const auto numvalid = std::distance(hashes.begin(), hashesEnd);
                    assert(numvalid == std::distance(validIds.begin(), validIdsEnd));

                    // std::for_each(
                    //     hashesBegin, hashesBegin + numSequences,
                    //     [kmermask = getKmerMask()](auto& hash){
                    //         hash &= kmermask;
                    //     }
                    // );

                    minhashTables[firstHashfunction + h]->insert(
                        hashes.data(), validIds.data(), numvalid
                    );
                }
            };

            forLoopExecutor(0, numHashfunctions, loopbody);
        }

        int checkInsertionErrors(
            int /*firstHashfunction*/,
            int /*numHashfunctions*/,
            cudaStream_t /*stream*/
        ) override{
            return 0;
        }

        void setThreadPool(ThreadPool* tp) override{
            threadPool = tp;
        }

        void setHostMemoryLimitForConstruction(std::size_t limit) override{
            memoryLimit = limit;
        }

        void setDeviceMemoryLimitsForConstruction(const std::vector<std::size_t>&) override{

        }

        void constructionIsFinished(cudaStream_t /*stream*/) override{

        }

    private:        
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
