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
#include <numeric>

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

        struct KVMetadata{
            read_number offset;
            BucketSize numValues;

            bool operator==(const KVMetadata& rhs) const noexcept{
                if(offset != rhs.offset) return false;
                if(numValues != rhs.numValues) return false;
                return true;
            }

            bool operator!=(const KVMetadata& rhs) const noexcept{
                return !operator==(rhs);
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

            #if 0

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

            #else 

            singlehashtable = std::make_unique<HashTable>(0, loadfactor);



            ThreadPool tpForHashing(runtimeOptions.threads);
            ThreadPool tpForCompacting(std::min(2,runtimeOptions.threads));

            setMemoryLimitForConstruction(maxMemoryForTables);

            //AoSCpuSingleValueHashTable<kmer_type, KVMetadata> metadataTable(536870912, 0.8f);

            kvtable = std::make_unique<DoublePassMultiValueHashTable<kmer_type, read_number>>((numReads * numSmallest) / 8, loadfactor);

            //while(remainingHashFunctions > 0 && keepGoing){
            {
                setThreadPool(&tpForHashing);

                std::size_t numKeys = 0;

                constexpr int batchsize = 1000000;
                //constexpr int batchsize = 20;
                const int numIterations = SDIV(numReads, batchsize);

                std::vector<read_number> currentReadIds(batchsize);
                std::vector<unsigned int> sequencedata(batchsize * encodedSequencePitchInInts);
                std::vector<int> sequencelengths(batchsize);

                //std::vector<kmer_type> keys1(maxNumPairs);

                helpers::CpuTimer firstpasstimer("firstpass");

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

                    std::vector<kmer_type> tmpkeys(currentbatchsize * numSmallest);
                    std::vector<read_number> tmpids(currentbatchsize * numSmallest);

                    auto numNewKeys = insertPart1(
                        tmpkeys.data(),
                        //keys1.data() + numKeys,
                        tmpids.data(),
                        sequencedata.data(),
                        currentbatchsize,
                        sequencelengths.data(),
                        encodedSequencePitchInInts,
                        currentReadIds.data()
                    );

                    numKeys += numNewKeys;
                    tmpkeys.erase(tmpkeys.begin() + numNewKeys, tmpkeys.end());
                    tmpids.erase(tmpids.begin() + numNewKeys, tmpids.end());

                    // for(std::size_t i = 0; i < tmpkeys.size(); i++){
                    //     KVMetadata* valueptr = metadataTable.queryPointer(tmpkeys[i]);
                    //     KVMetadata initvalue;
                    //     initvalue.offset = 1;
                    //     initvalue.numValues = 0;

                    //     if(valueptr == nullptr){
                    //         metadataTable.insert(tmpkeys[i], initvalue);
                    //     }else{
                    //         (*valueptr).offset++;
                    //     }
                    // }

                    kvtable->firstPassInsert(tmpkeys.data(), tmpids.data(), tmpkeys.size());

                    // std::cerr << "keys: \n";
                    // for(std::size_t i = 0; i < tmpkeys.size(); i++){
                    //     std::cerr << tmpkeys[i] << " ";
                    // }
                    // std::cerr << "\n";

                    // std::cerr << "ids: \n";
                    // for(std::size_t i = 0; i < tmpids.size(); i++){
                    //     std::cerr << tmpids[i] << " ";
                    // }
                    // std::cerr << "\n";

                    // std::exit(0);
                }

                //keys1.erase(keys1.begin() + numKeys, keys1.end());

                firstpasstimer.print();

                // {
                // std::ofstream outputstream("cputablestemp1.bin", std::ios::binary);
                // kvtable->writeToStream(outputstream);
                // }

                //kvtable->firstPassDone(2, 75);
                kvtable->firstPassDone(2, 255);

                // {
                // std::ofstream outputstream("cputablestemp2.bin", std::ios::binary);
                // kvtable->writeToStream(outputstream);
                // }

                helpers::CpuTimer secondPassTimer("secondpass");

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

                    std::vector<kmer_type> tmpkeys(currentbatchsize * numSmallest);
                    std::vector<read_number> tmpids(currentbatchsize * numSmallest);

                    auto numNewKeys = insertPart1(
                        tmpkeys.data(),
                        //keys1.data() + numKeys,
                        tmpids.data(),
                        sequencedata.data(),
                        currentbatchsize,
                        sequencelengths.data(),
                        encodedSequencePitchInInts,
                        currentReadIds.data()
                    );

                    numKeys += numNewKeys;
                    tmpkeys.erase(tmpkeys.begin() + numNewKeys, tmpkeys.end());
                    tmpids.erase(tmpids.begin() + numNewKeys, tmpids.end());

                    kvtable->secondPassInsert(tmpkeys.data(), tmpids.data(), tmpkeys.size());
                }

                //keys1.erase(keys1.begin() + numKeys, keys1.end());

                secondPassTimer.print();

                kvtable->secondPassDone();

                // {
                // std::ofstream outputstream("cputablestemp3.bin", std::ios::binary);
                // kvtable->writeToStream(outputstream);
                // }

                // {

                // std::cerr << "check key counts\n";
                // helpers::CpuTimer timer5("check key counts");

                // std::size_t tablecountedUniqueKeys = 0;
                // std::size_t tablecountedValues = 0;

                // std::size_t tablecountednumKeysGreater1 = 0;
                // std::size_t tablecountednumValuesWithKeysGreater1 = 0;

                // metadataTable.forEachKeyValuePair(
                //     [&](const auto& /*key*/, const auto& value){
                //         tablecountedUniqueKeys++;
                //         tablecountedValues += value.offset;

                //         if(value.offset > 1){
                //             tablecountednumKeysGreater1++;
                //             tablecountednumValuesWithKeysGreater1 += value.offset;
                //         }
                //     }
                // );
                // timer5.print();

                // std::cerr << "tablecountedUniqueKeys:"  << tablecountedUniqueKeys << "\n";
                // std::cerr << "tablecountedValues:"  << tablecountedValues << "\n";
                // std::cerr << "tablecountednumKeysGreater1:"  << tablecountednumKeysGreater1 << "\n";
                // std::cerr << "tablecountednumValuesWithKeysGreater1:"  << tablecountednumValuesWithKeysGreater1 << "\n";

                

                // }

                // {
                //     //AoSCpuSingleValueHashTable<kmer_type, int> keyCountTable(70'000'000, 0.8f);
                //     AoSCpuSingleValueHashTable<kmer_type, int> keyCountTable(1, 0.8f);

                //     std::cerr << "insert key counts\n";
                //     helpers::CpuTimer timer4("insert key counts");

                //     for(std::size_t i = 0; i < keys1.size(); i++){
                //         int* valueptr = keyCountTable.queryPointer(keys1[i]);
                //         if(valueptr == nullptr){
                //             keyCountTable.insert(keys1[i], 1);
                //         }else{
                //             (*valueptr)++;
                //         }
                //     }
                //     timer4.print();

                //     std::cerr << "check key counts\n";
                //     helpers::CpuTimer timer5("check key counts");

                //     std::size_t tablecountedUniqueKeys = 0;
                //     std::size_t tablecountedValues = 0;

                //     keyCountTable.forEachKeyValuePair(
                //         [&](const auto& /*key*/, const auto& value){
                //             tablecountedUniqueKeys++;
                //             tablecountedValues += value;
                //         }
                //     );
                //     timer5.print();

                //     std::cerr << "tablecountedUniqueKeys:"  << tablecountedUniqueKeys << "\n";
                //     std::cerr << "tablecountedValues:"  << tablecountedValues << "\n";


                // }

                // std::cerr << "sort keys\n";
                // helpers::CpuTimer timer1("sort keys");
                // std::sort(keys1.begin(), keys1.end());
                // timer1.print();

                // std::cerr << "count unique keys\n";
                // helpers::CpuTimer timer2("count unique keys");
                // std::size_t numUniqueKeys = (keys1.size() > 0) ? 1 : 0;
                // for(std::size_t i = 1; i < keys1.size(); i++){
                //     if(keys1[i-1] != keys1[i]){
                //         numUniqueKeys++;
                //     }
                // }
                // timer2.print();
                // std::cerr << "numUniqueKeys: " << numUniqueKeys << "\n";

                // std::vector<int> countsPerKey(numUniqueKeys, 0);
                // if(keys1.size() > 0){
                //     countsPerKey[0] = 1;
                // }
                // std::cerr << "count values per key\n";
                // helpers::CpuTimer timer3("count values per key");
                // for(std::size_t i = 1, uniqueIndex = 0; i < keys1.size(); i++){
                //     if(keys1[i-1] != keys1[i]){
                //         uniqueIndex++;
                //     }
                //     countsPerKey[uniqueIndex]++;
                // }
                // timer3.print();

                // std::size_t numValues = std::reduce(countsPerKey.begin(), countsPerKey.end(), std::size_t(0));
                // std::cerr << "numValues: " << numValues << "\n";

                // std::size_t numKeysGreater1 = std::transform_reduce(countsPerKey.begin(), countsPerKey.end(), std::size_t(0), std::plus{}, [](auto x){return x > 1 ? 1 : 0;});
                // std::size_t numValuesWithKeysGreater1 = std::transform_reduce(countsPerKey.begin(), countsPerKey.end(), std::size_t(0), std::plus{}, [](auto x){return x > 1 ? x : 0;});

                // std::cerr << "numKeysGreater1: " << numKeysGreater1 << "\n";
                // std::cerr << "numValuesWithKeysGreater1: " << numValuesWithKeysGreater1 << "\n";

                
                


                std::cerr << "Compacting custom table\n";

                if(tpForCompacting.getConcurrency() > 1){
                    setThreadPool(&tpForCompacting);
                }else{
                    setThreadPool(nullptr);
                }
                
                finalize();
            }

            #endif

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
                    //const auto mapQueryResult = singlehashtable->query(hashValues[i]);
                    const auto mapQueryResult = kvtable->query(hashValues[i]);


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

            //result += singlehashtable->getMemoryInfo();
            result += kvtable->getMemoryInfo();

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

                threadData[threadid].hashes.erase(hashIter, threadData[threadid].hashes.end());
                threadData[threadid].ids.erase(idIter, threadData[threadid].ids.end());

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
        
        std::size_t insertPart1(
            kmer_type* keyoutput,
            read_number* readIdsOutput,
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

                threadData[threadid].hashes.erase(hashIter, threadData[threadid].hashes.end());
                threadData[threadid].ids.erase(idIter, threadData[threadid].ids.end());

                assert(std::distance(threadData[threadid].hashes.begin(), hashIter) == std::distance(threadData[threadid].ids.begin(), idIter));
            };

            forLoopExecutor(0, numSequences, hashloopbody);

            std::size_t numKeys = 0;

            for(const auto& data : threadData){
                std::copy(data.hashes.begin(), data.hashes.end(), keyoutput + numKeys);
                std::copy(data.ids.begin(), data.ids.end(), readIdsOutput + numKeys);
                numKeys += data.hashes.size();
            }

            return numKeys;
        }

        void setThreadPool(ThreadPool* tp){
            threadPool = tp;
        }

        void setMemoryLimitForConstruction(std::size_t limit){
            memoryLimit = limit;
        }

        void writeToStream(std::ostream& os) const{

            os.write(reinterpret_cast<const char*>(&kmerSize), sizeof(int));
            os.write(reinterpret_cast<const char*>(&numSmallest), sizeof(int));
            os.write(reinterpret_cast<const char*>(&resultsPerMapThreshold), sizeof(int));

            os.write(reinterpret_cast<const char*>(&loadfactor), sizeof(float));

            kvtable->writeToStream(os);
        }

        int loadFromStream(std::ifstream& is, int numMapsUpperLimit = std::numeric_limits<int>::max()){
            destroy();

            is.read(reinterpret_cast<char*>(&kmerSize), sizeof(int));
            is.read(reinterpret_cast<char*>(&numSmallest), sizeof(int));
            is.read(reinterpret_cast<char*>(&resultsPerMapThreshold), sizeof(int));

            is.read(reinterpret_cast<char*>(&loadfactor), sizeof(float));

            kvtable = std::make_unique<DoublePassMultiValueHashTable<kmer_type, read_number>>(1, loadfactor);
            kvtable->loadFromStream(is);

            return 0;
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
        std::unique_ptr<HashTable> singlehashtable{};
        mutable std::vector<std::unique_ptr<QueryData>> tempdataVector{};
        std::unique_ptr<DoublePassMultiValueHashTable<kmer_type, read_number>> kvtable{};
    };


}

#endif