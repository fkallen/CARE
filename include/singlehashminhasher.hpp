#ifndef CARE_SINGLEHASHCPUMINHASHER_HPP
#define CARE_SINGLEHASHCPUMINHASHER_HPP


#include <cpuminhasher.hpp>
#include <cpureadstorage.hpp>
#include <groupbykey.hpp>
#include <cpusequencehasher.hpp>

#include <config.hpp>

#include <cpuhashtable.hpp>

#include <options.hpp>
#include <util.hpp>
#include <hpc_helpers.cuh>
#include <filehelpers.hpp>
#include <minhashing.hpp>
#include <sequencehelpers.hpp>
#include <memorymanagement.hpp>
#include <threadpool.hpp>
#include <sharedmutex.hpp>


#include <cassert>
#include <array>
#include <vector>
#include <memory>
#include <limits>
#include <string>
#include <fstream>
#include <algorithm>

namespace care{


    class SingleHashCpuMinhasher : public CpuMinhasher{
    public:
        using Key_t = CpuMinhasher::Key;
        using Value_t = read_number;
    private:
        using HashTable = CpuReadOnlyMultiValueHashTable<kmer_type, read_number>;
        using Range_t = std::pair<const Value_t*, const Value_t*>;

        struct QueryData{

            enum class Stage{
                None,
                NumValues,
                Retrieve
            };

            Stage previousStage = Stage::None;
            std::vector<Range_t> ranges{};
            SetUnionHandle suHandle{};

            MemoryUsage getMemoryInfo() const{
                MemoryUsage info{};
                info.host += sizeof(Range_t) * ranges.capacity();
    
                return info;
            }

            void destroy(){
            }
        };

        
    public:

        SingleHashCpuMinhasher() : SingleHashCpuMinhasher(0, 50, 16, 0.8f){

        }

        SingleHashCpuMinhasher(int maxNumKeys_, int maxValuesPerKey, int k, float loadfactor_)
            : loadfactor(loadfactor_), maxNumKeys(maxNumKeys_), kmerSize(k), resultsPerMapThreshold(maxValuesPerKey){

        }

        SingleHashCpuMinhasher(const SingleHashCpuMinhasher&) = delete;
        SingleHashCpuMinhasher(SingleHashCpuMinhasher&&) = default;
        SingleHashCpuMinhasher& operator=(const SingleHashCpuMinhasher&) = delete;
        SingleHashCpuMinhasher& operator=(SingleHashCpuMinhasher&&) = default;        


        void constructFromReadStorage(
            const FileOptions &/*fileOptions*/,
            const RuntimeOptions &runtimeOptions,
            const MemoryOptions& memoryOptions,
            std::uint64_t /*nReads*/,
            const CorrectionOptions& correctionOptions,
            const CpuReadStorage& cpuReadStorage
        ){
            auto& readStorage = cpuReadStorage;

            const int requestedNumberOfMaps = correctionOptions.numHashFunctions;
            numSmallest = requestedNumberOfMaps;

            const std::uint64_t numReads = readStorage.getNumberOfReads();
            const int maximumSequenceLength = readStorage.getSequenceLengthUpperBound();
            const std::size_t encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength);

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

            std::size_t maxNumPairs = std::size_t(numReads) * std::size_t(numSmallest);

            singlehashtable = std::make_unique<HashTable>(maxNumPairs, loadfactor);



            ThreadPool tpForHashing(runtimeOptions.threads);
            ThreadPool tpForCompacting(std::min(2,runtimeOptions.threads));

            setMemoryLimitForConstruction(maxMemoryForTables);

            //while(remainingHashFunctions > 0 && keepGoing){
            {
                setThreadPool(&tpForHashing);

                std::size_t numKeys = 0;

                constexpr int batchsize = 1000000;
                const int numIterations = SDIV(numReads, batchsize);

                std::vector<read_number> currentReadIds(batchsize);
                std::vector<unsigned int> sequencedata(batchsize * encodedSequencePitchInInts);
                std::vector<int> sequencelengths(batchsize);

                for(int iteration = 0; iteration < numIterations; iteration++){
                    const read_number beginid = iteration * batchsize;
                    const read_number endid = std::min((iteration + 1) * batchsize, int(numReads));
                    const read_number currentbatchsize = endid - beginid;

                    std::iota(currentReadIds.begin(), currentReadIds.end(), beginid);

                    readStorage.gatherSequences(
                        sequencedata.data(),
                        encodedSequencePitchInInts,
                        currentReadIds.data(),
                        currentbatchsize
                    );

                    readStorage.gatherSequenceLengths(
                        sequencelengths.data(),
                        currentReadIds.data(),
                        currentbatchsize
                    );

                    constexpr bool dryrun = false;

                    auto numNewKeys = insert(
                        dryrun,
                        sequencedata.data(),
                        currentbatchsize,
                        sequencelengths.data(),
                        encodedSequencePitchInInts,
                        currentReadIds.data()
                    );

                    numKeys += numNewKeys;
                
                }

                std::cerr << "Compacting custom table\n";

                if(tpForCompacting.getConcurrency() > 1){
                    setThreadPool(&tpForCompacting);
                }else{
                    setThreadPool(nullptr);
                }
                
                finalize();
            }

            setThreadPool(nullptr); 
        }
 

        MinhasherHandle makeMinhasherHandle() const override {
            auto data = std::make_unique<QueryData>();

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
            const unsigned int* h_sequenceData2Bit,
            std::size_t encodedSequencePitchInInts,
            const int* h_sequenceLengths,
            int numSequences,
            int* h_numValuesPerSequence,
            int& totalNumValues
        ) const override {

            if(numSequences == 0) return;

            QueryData* const queryData = getQueryDataFromHandle(queryHandle);

            queryData->ranges.clear();

            totalNumValues = 0;

            queryData->ranges.resize(numSequences * numSmallest);
            CPUSequenceHasher<kmer_type> sequenceHasher;

            for(int s = 0; s < numSequences; s++){
                const int length = h_sequenceLengths[s];
                const unsigned int* sequence = h_sequenceData2Bit + encodedSequencePitchInInts * s;

                auto hashValues = sequenceHasher.getTopSmallestKmerHashes(
                    sequence, 
                    length, 
                    getKmerSize(), 
                    numSmallest
                );

                int numValues = 0;

                const auto numHashValues = hashValues.size();
                for(std::size_t i = 0; i < numHashValues; i++){
                    const auto mapQueryResult = singlehashtable->query(hashValues[i]);

                    queryData->ranges[s * numSmallest + i] 
                        = std::make_pair(mapQueryResult.valuesBegin, mapQueryResult.valuesBegin + mapQueryResult.numValues);

                    numValues += mapQueryResult.numValues;
                }

                for(int i = numHashValues; i < numSmallest; i++){
                    queryData->ranges[s * numSmallest + i] = std::make_pair(nullptr, nullptr);
                }

                h_numValuesPerSequence[s] = numValues;
                totalNumValues += numValues;
            }
           
            queryData->previousStage = QueryData::Stage::NumValues;
        }

        void retrieveValues(
            MinhasherHandle& queryHandle,
            const read_number* h_readIds,
            int numSequences,
            int /*totalNumValues*/,
            read_number* h_values,
            int* h_numValuesPerSequence,
            int* h_offsets //numSequences + 1
        ) const override {
            if(numSequences == 0) return;

            QueryData* const queryData = getQueryDataFromHandle(queryHandle);

            assert(queryData->previousStage == QueryData::Stage::NumValues);

            h_offsets[0] = 0;

            for(int s = 0; s < numSequences; s++){
                int numValues = 0;
                for(int i = 0; i < numSmallest; i++){
                    numValues += std::distance(
                        queryData->ranges[s * numSequences + i].first,
                        queryData->ranges[s * numSequences + i].second
                    );
                }

                std::vector<Value_t> valuestmp(numValues);
                auto valueIter = valuestmp.begin();
                for(int i = 0; i < numSmallest; i++){
                    valueIter = std::copy(
                        queryData->ranges[s * numSequences + i].first,
                        queryData->ranges[s * numSequences + i].second,
                        valueIter
                    );
                }

                std::sort(valuestmp.begin(), valuestmp.end());
                auto uniqueEnd = std::unique(valuestmp.begin(), valuestmp.end());

                if(h_readIds != nullptr){
                    auto readIdPos = std::lower_bound(
                        valuestmp.begin(),
                        uniqueEnd,
                        h_readIds[s]
                    );

                    if(readIdPos != uniqueEnd && *readIdPos == h_readIds[s]){
                        //TODO optimization: avoid this copy, instead skip the element when copying to h_values
                        uniqueEnd = std::copy(readIdPos + 1, uniqueEnd, readIdPos);
                    }
                }

                std::copy(valuestmp.begin(), uniqueEnd, h_values + h_offsets[s]);

                h_numValuesPerSequence[s] = std::distance(valuestmp.begin(), uniqueEnd);
                h_offsets[s+1] = h_offsets[s] + std::distance(valuestmp.begin(), uniqueEnd);
            }

            queryData->previousStage = QueryData::Stage::Retrieve;
        }

        void compact() {

            auto groupByKey = [&](auto& keys, auto& values, auto& countsPrefixSum){
                constexpr bool valuesOfSameKeyMustBeSorted = false;
                const int maxValuesPerKey = getNumResultsPerMapThreshold();

                care::GroupByKeyCpu<Key_t, Value_t, read_number> groupByKey(valuesOfSameKeyMustBeSorted, maxValuesPerKey);
                groupByKey.execute(keys, values, countsPrefixSum);
            };
        
            if(!singlehashtable->isInitialized()){
                singlehashtable->finalize(groupByKey, nullptr);
            }                

            if(threadPool != nullptr){
                threadPool->wait();
            }
        }

        MemoryUsage getMemoryInfo() const noexcept override{
            MemoryUsage result;

            result += singlehashtable->getMemoryInfo();

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

        int getKmerSize() const noexcept override{
            return kmerSize;
        }

        void destroy() {
            singlehashtable = nullptr;
        }

        void finalize(){
            compact();
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
        //     //maxNumTablesInIteration = std::min(numTablesToConstruct, 4);

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
            const unsigned int* h_sequenceData2Bit,
            int numSequences,
            const int* h_sequenceLengths,
            std::size_t encodedSequencePitchInInts,
            const read_number* h_readIds
        ){
            if(numSequences == 0) return 0;

            ThreadPool::ParallelForHandle pforHandle{};

            ForLoopExecutor forLoopExecutor(threadPool, &pforHandle);
            const int numThreads = forLoopExecutor.getNumThreads();

            struct ThreadData{
                std::vector<kmer_type> hashes{};
                std::vector<read_number> ids{};
            };

            std::vector<ThreadData> threadData(numThreads);

            auto hashloopbody = [&](auto begin, auto end, int threadid){
                CPUSequenceHasher<kmer_type> sequenceHasher;

                threadData[threadid].hashes.resize((end - begin) * numSmallest);
                threadData[threadid].ids.resize((end - begin) * numSmallest);

                auto hashIter = threadData[threadid].hashes.begin();
                auto idIter = threadData[threadid].ids.begin();

                for(int s = begin; s < end; s++){
                    const int length = h_sequenceLengths[s];
                    const unsigned int* sequence = h_sequenceData2Bit + encodedSequencePitchInInts * s;

                    auto hashValues = sequenceHasher.getTopSmallestKmerHashes(
                        sequence, 
                        length, 
                        getKmerSize(), 
                        numSmallest
                    );

                    hashIter = std::copy(hashValues.begin(), hashValues.end(), hashIter);
                    std::fill(idIter, idIter + hashValues.size(), h_readIds[s]);
                    idIter = idIter + hashValues.size();
                }

                assert(std::distance(threadData[threadid].hashes.begin(), hashIter) == std::distance(threadData[threadid].ids.begin(), idIter));
            };

            forLoopExecutor(0, numSequences, hashloopbody);

            std::size_t numKeys = 0;

            for(const auto& data : threadData){
                numKeys += data.ids.size();
                if(!dryrun){
                    if(data.hashes.size() > 0){
                        singlehashtable->insert(data.hashes.data(), data.ids.data(), data.ids.size());
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

        // Range_t queryMap(int mapid, const Key_t& key) const{
        //     assert(mapid < getNumberOfMaps());

        //     const int numResultsPerMapQueryThreshold = getNumResultsPerMapThreshold();

        //     const auto mapQueryResult = minhashTables[mapid]->query(key);

        //     if(mapQueryResult.numValues > numResultsPerMapQueryThreshold){
        //         return std::make_pair(nullptr, nullptr); //return empty range
        //     }

        //     return std::make_pair(mapQueryResult.valuesBegin, mapQueryResult.valuesBegin + mapQueryResult.numValues);
        // }


        mutable int counter = 0;
        mutable SharedMutex sharedmutex{};

        float loadfactor = 0.8f;
        int numSmallest = 1;
        int maxNumKeys{};
        int kmerSize{};
        int resultsPerMapThreshold{};
        ThreadPool* threadPool;
        std::size_t memoryLimit;
        std::unique_ptr<HashTable> singlehashtable;
        mutable std::vector<std::unique_ptr<QueryData>> tempdataVector{};
    };


}

#endif