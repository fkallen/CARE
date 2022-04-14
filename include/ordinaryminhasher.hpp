#ifndef CARE_OrdinaryCpuMinhasher_HPP
#define CARE_OrdinaryCpuMinhasher_HPP


#include <cpuminhasher.hpp>
#include <cpureadstorage.hpp>
#include <groupbykey.hpp>


#include <config.hpp>

#include <cpuhashtable.hpp>

#include <options.hpp>
#include <util.hpp>
#include <hpc_helpers.cuh>
#include <filehelpers.hpp>
#include <sequencehelpers.hpp>
#include <memorymanagement.hpp>
#include <threadpool.hpp>
#include <sharedmutex.hpp>

#include <cpusequencehasher.hpp>


#include <cassert>
#include <array>
#include <vector>
#include <memory>
#include <limits>
#include <string>
#include <fstream>
#include <algorithm>

namespace care{


    class OrdinaryCpuMinhasher : public CpuMinhasher{
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

        OrdinaryCpuMinhasher() : OrdinaryCpuMinhasher(0, 50, 16, 0.8f){

        }

        OrdinaryCpuMinhasher(int maxNumKeys_, int maxValuesPerKey, int k, float loadfactor_)
            : loadfactor(loadfactor_), maxNumKeys(maxNumKeys_), kmerSize(k), resultsPerMapThreshold(maxValuesPerKey){

        }

        OrdinaryCpuMinhasher(const OrdinaryCpuMinhasher&) = delete;
        OrdinaryCpuMinhasher(OrdinaryCpuMinhasher&&) = default;
        OrdinaryCpuMinhasher& operator=(const OrdinaryCpuMinhasher&) = delete;
        OrdinaryCpuMinhasher& operator=(OrdinaryCpuMinhasher&&) = default;        


        void constructFromReadStorage(
            const ProgramOptions& programOptions,
            std::uint64_t /*nReads*/,
            const CpuReadStorage& cpuReadStorage
        ){
            auto& readStorage = cpuReadStorage;

            const int requestedNumberOfMaps = programOptions.numHashFunctions;

            const std::uint64_t numReads = readStorage.getNumberOfReads();
            const int maximumSequenceLength = readStorage.getSequenceLengthUpperBound();
            const std::size_t encodedSequencePitchInInts = SequenceHelpers::getEncodedNumInts2Bit(maximumSequenceLength);

            const MemoryUsage memoryUsageOfReadStorage = readStorage.getMemoryInfo();
            std::size_t totalLimit = programOptions.memoryTotalLimit;
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
            //         << ",memoryForHashtables: " << programOptions.memoryForHashtables
            //         << ", memoryTotalLimit: " << programOptions.memoryTotalLimit
            //         << ", rsHostUsage: " << memoryUsageOfReadStorage.host << "\n";

            maxMemoryForTables = std::min(maxMemoryForTables, 
                                    std::min(programOptions.memoryForHashtables, totalLimit));

            std::cerr << "maxMemoryForTables = " << maxMemoryForTables << " bytes\n";

            const int hashFunctionOffset = 0;

            
            std::vector<int> usedHashFunctionNumbers;

            ThreadPool tpForHashing(programOptions.threads);
            ThreadPool tpForCompacting(std::min(2,programOptions.threads));

            setMemoryLimitForConstruction(maxMemoryForTables);

            int remainingHashFunctions = requestedNumberOfMaps;
            bool keepGoing = true;

            std::vector<kmer_type> tempvector{};

            while(remainingHashFunctions > 0 && keepGoing){

                setThreadPool(&tpForHashing);

                const int alreadyExistingHashFunctions = requestedNumberOfMaps - remainingHashFunctions;
                int addedHashFunctions = addHashfunctions(remainingHashFunctions);

                if(addedHashFunctions == 0){
                    keepGoing = false;
                    break;
                }

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

                    insert(
                        tempvector,
                        sequencedata.data(),
                        currentbatchsize,
                        sequencelengths.data(),
                        encodedSequencePitchInInts,
                        currentReadIds.data(),
                        alreadyExistingHashFunctions,
                        addedHashFunctions,
                        h_hashfunctionNumbers.data()
                    );
                
                }

                std::cerr << "Compacting\n";
                //setThreadPool(nullptr);
                if(tpForCompacting.getConcurrency() > 1){
                    setThreadPool(&tpForCompacting);
                }else{
                    setThreadPool(nullptr);
                }
                
                finalize();

                remainingHashFunctions -= addedHashFunctions;
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

            std::vector<kmer_type> allHashValues(numSequences * getNumberOfMaps());

            CPUSequenceHasher<kmer_type> hasher;

            for(int s = 0; s < numSequences; s++){
                const int length = h_sequenceLengths[s];
                const unsigned int* const sequence = h_sequenceData2Bit + encodedSequencePitchInInts * s;

                hasher.hashInto(
                    allHashValues.begin() + getNumberOfMaps() * s,
                    sequence, 
                    length, 
                    getKmerSize(), 
                    getNumberOfMaps(),
                    0
                );
            }

            queryData->ranges.resize(numSequences * getNumberOfMaps());

            std::fill(h_numValuesPerSequence, h_numValuesPerSequence + numSequences, 0);

            for(int map = 0; map < getNumberOfMaps(); ++map){
                for(int s = 0; s < numSequences; s++){
                    const int length = h_sequenceLengths[s];

                    if(length >= getKmerSize()){
                        const kmer_type key = allHashValues[s * getNumberOfMaps() + map];
                        auto entries_range = queryMap(map, key);
                        const int n_entries = std::distance(entries_range.first, entries_range.second);
                        totalNumValues += n_entries;
                        h_numValuesPerSequence[s] += n_entries;
                        //queryData->ranges[s * getNumberOfMaps() + map] = entries_range;
                        queryData->ranges[map * numSequences + s] = entries_range;
                    }
                }
            }

            queryData->previousStage = QueryData::Stage::NumValues;
        }


        #if 0

        //retrieve values with sort + unique instead of set_union
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

            read_number* h_valuesEnd = h_values;

            for(int s = 0; s < numSequences; s++){
                read_number* myValues = h_valuesEnd;

                for(int map = 0; map < getNumberOfMaps(); ++map){
                    const auto& range = queryData->ranges[map * numSequences + s]; //todo transpose
                    h_valuesEnd = std::copy(range.first, range.second, h_valuesEnd);
                }

                std::sort(myValues, h_valuesEnd);
                if(h_readIds != nullptr){
                    const auto anchorReadId = h_readIds[s];

                    auto isEqual = [toremove = anchorReadId](const auto& l, const auto& r){
                        if(l == toremove) return true;
                        if(r == toremove) return true;
                        return l == r;
                    };

                    //make unique range and remove anchorReadId
                    h_valuesEnd = std::unique(myValues, h_valuesEnd, isEqual);
                }else{
                    //make unique range
                    h_valuesEnd = std::unique(myValues, h_valuesEnd);
                }

                h_numValuesPerSequence[s] = std::distance(myValues, h_valuesEnd);
                h_offsets[s+1] = h_numValuesPerSequence[s] + h_offsets[s];
            }

            queryData->previousStage = QueryData::Stage::Retrieve;
        }

        #else

        void retrieveValues(
            MinhasherHandle& queryHandle,
            const read_number* h_readIds,
            int numSequences,
            int totalNumValues,
            read_number* h_values,
            int* h_numValuesPerSequence,
            int* h_offsets //numSequences + 1
        ) const override {
            if(numSequences == 0) return;

            QueryData* const queryData = getQueryDataFromHandle(queryHandle);

            assert(queryData->previousStage == QueryData::Stage::NumValues);

            #if 0
            h_offsets[0] = 0;
            auto first = h_values;

            for(int s = 0; s < numSequences; s++){
                auto rangesbegin = queryData->ranges.data() + s * getNumberOfMaps();

                auto end = k_way_set_union(queryData->suHandle, first, rangesbegin, getNumberOfMaps());
                if(h_readIds != nullptr){
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
            #else

            h_offsets[0] = 0;

            std::vector<Value_t> valuestmp(totalNumValues);
            std::vector<Range_t> rangestmp(numSequences * getNumberOfMaps());

            //std::cerr << numSequences << "\n";

            auto iter = valuestmp.begin();
            int counter = 0;
            for(int map = 0; map < getNumberOfMaps(); ++map){
                for(int s = 0; s < numSequences; s++){
                    //const auto& range = queryData->ranges[s * getNumberOfMaps() + map];
                    const auto& range = queryData->ranges[map * numSequences + s];
                    auto& newrange = rangestmp[s * getNumberOfMaps() + map];

                    newrange.first = valuestmp.data() + counter;

                    iter = std::copy(range.first, range.second, iter);
                    counter += std::distance(range.first, range.second);

                    newrange.second = valuestmp.data() + counter;
                }
            }

            
            auto first = h_values;


            for(int s = 0; s < numSequences; s++){
                auto rangesbegin = rangestmp.data() + s * getNumberOfMaps();

                auto end = k_way_set_union(queryData->suHandle, first, rangesbegin, getNumberOfMaps());
                if(h_readIds != nullptr){
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
            #endif

            queryData->previousStage = QueryData::Stage::Retrieve;
        }

        #endif

        void compact() {
            const int num = minhashTables.size();

            auto groupByKey = [&](auto& keys, auto& values, auto& countsPrefixSum){
                constexpr bool valuesOfSameKeyMustBeSorted = true;
                const int maxValuesPerKey = getNumResultsPerMapThreshold();

                //if only 1 value exists, it belongs to the anchor read itself and does not need to be stored.
                const int minValuesPerKey = 2;

                care::GroupByKeyCpu<Key_t, Value_t, read_number> groupByKey(valuesOfSameKeyMustBeSorted, maxValuesPerKey, minValuesPerKey);
                groupByKey.execute(keys, values, countsPrefixSum);
            };

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

        int getKmerSize() const noexcept override{
            return kmerSize;
        }

        void destroy() {
            minhashTables.clear();
        }

        void finalize(){
            compact();
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

            #ifndef NDEBUG
            const int cur = minhashTables.size();
            assert(!(numExtraFunctions + cur > 64));
            #endif

            std::size_t bytesOfCachedConstructedTables = 0;
            for(const auto& ptr : minhashTables){
                auto memusage = ptr->getMemoryInfo();
                bytesOfCachedConstructedTables += memusage.host;
            }

            std::size_t requiredMemPerTable = (sizeof(kmer_type) + sizeof(read_number)) * maxNumKeys;
            int numTablesToConstruct = (memoryLimit - bytesOfCachedConstructedTables) / requiredMemPerTable;
            numTablesToConstruct -= 2; // keep free memory of 2 tables to perform transformation 
            numTablesToConstruct = std::min(numTablesToConstruct, numExtraFunctions);
            //maxNumTablesInIteration = std::min(numTablesToConstruct, 4);

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
            std::vector<kmer_type>& tempvector,
            const unsigned int* h_sequenceData2Bit,
            int numSequences,
            const int* h_sequenceLengths,
            std::size_t encodedSequencePitchInInts,
            const read_number* h_readIds,
            int firstHashfunction,
            int numHashfunctions,
            const int* /*h_hashFunctionNumbers*/
        ){
            if(numSequences == 0) return;

            ThreadPool::ParallelForHandle pforHandle{};

            ForLoopExecutor forLoopExecutor(threadPool, &pforHandle);

            auto& allHashValues = tempvector;
            allHashValues.resize(numSequences * getNumberOfMaps());            

            auto hashloopbody = [&](auto begin, auto end, int /*threadid*/){
                CPUSequenceHasher<kmer_type> hasher;

                for(int s = begin; s < end; s++){
                    const int length = h_sequenceLengths[s];
                    const unsigned int* sequence = h_sequenceData2Bit + encodedSequencePitchInInts * s;

                    auto hashValues = hasher.hash(
                        sequence, 
                        length, 
                        getKmerSize(), 
                        getNumberOfMaps(),
                        0
                    );

                    for(int h = 0; h < getNumberOfMaps(); h++){
                        allHashValues[h * numSequences + s] = hashValues[h];
                    }
                }
            };

            forLoopExecutor(0, numSequences, hashloopbody);

            auto insertloopbody = [&](auto begin, auto end, int /*threadid*/){
                for(int h = begin; h < end; h++){
                    minhashTables[h]->insert(
                        &allHashValues[h * numSequences], h_readIds, numSequences
                    );
                }
            };

            forLoopExecutor(firstHashfunction, firstHashfunction + numHashfunctions, insertloopbody);
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

        Range_t queryMap(int mapid, const Key_t& key) const{
            assert(mapid < getNumberOfMaps());

            const int numResultsPerMapQueryThreshold = getNumResultsPerMapThreshold();

            const auto mapQueryResult = minhashTables[mapid]->query(key);

            if(mapQueryResult.numValues > numResultsPerMapQueryThreshold){
                return std::make_pair(nullptr, nullptr); //return empty range
            }

            return std::make_pair(mapQueryResult.valuesBegin, mapQueryResult.valuesBegin + mapQueryResult.numValues);
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

#endif