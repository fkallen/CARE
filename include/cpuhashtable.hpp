#ifndef CARE_CPUHASHTABLE_HPP
#define CARE_CPUHASHTABLE_HPP 

#include <config.hpp>
#include <memorymanagement.hpp>
#include <threadpool.hpp>
#include <hostdevicefunctions.cuh>

#include <map>
#include <vector>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <type_traits>
#include <limits>
#include <iostream>

#include <hpc_helpers.cuh>


namespace care{

    template<class Key, class Value>
    class AoSCpuSingleValueHashTable{
        static_assert(std::is_integral<Key>::value, "Key must be integral!");

    public:
        class QueryResult{
        public:
            QueryResult() = default;
            QueryResult(const QueryResult&) = default;
            QueryResult(QueryResult&&) = default;
            QueryResult& operator=(const QueryResult&) = default;
            QueryResult& operator=(QueryResult&&) = default;

            QueryResult(bool b, Value v) : keyFound(b), val(std::move(v)){}

            bool valid() const{
                return keyFound;
            }
            const Value& value() const{
                return val;
            }
            Value& value(){
                return val;
            }
        private:
            bool keyFound;
            Value val;
        };

        AoSCpuSingleValueHashTable() = default;
        AoSCpuSingleValueHashTable(const AoSCpuSingleValueHashTable&) = default;
        AoSCpuSingleValueHashTable(AoSCpuSingleValueHashTable&&) = default;
        AoSCpuSingleValueHashTable& operator=(const AoSCpuSingleValueHashTable&) = default;
        AoSCpuSingleValueHashTable& operator=(AoSCpuSingleValueHashTable&&) = default;

        AoSCpuSingleValueHashTable(std::size_t size, float load)
            : load(load), size(size), capacity(size/load)
        {
            storage.resize(capacity, emptySlot);
        }

        AoSCpuSingleValueHashTable(std::size_t size) 
            : AoSCpuSingleValueHashTable(size, 0.8f){            
        }

        bool operator==(const AoSCpuSingleValueHashTable& rhs) const{
            return emptySlot == rhs.emptySlot 
                && feq(load, rhs.load)
                && maxProbes == rhs.maxProbes
                && size == rhs.size 
                && capacity == rhs.capacity
                && storage == rhs.storage;
        }

        bool operator!=(const AoSCpuSingleValueHashTable& rhs) const{
            return !(operator==(rhs));
        }

        void insert(const Key& key, const Value& value){
            using hasher = hashers::MurmurHash<std::uint64_t>;

            const std::uint64_t key64 = std::uint64_t(key);
            std::size_t probes = 0;
            std::size_t pos = hasher::hash(key64) % capacity;
            while(storage[pos] != emptySlot){
                pos++;
                //wrap-around
                if(pos == capacity){
                    pos = 0;
                }
                probes++;
            }
            storage[pos].first = key;
            storage[pos].second = value;

            maxProbes = std::max(maxProbes, probes);
        }

        QueryResult query(const Key& key) const{
            using hasher = hashers::MurmurHash<std::uint64_t>;
            
            const std::uint64_t key64 = std::uint64_t(key);
            std::size_t probes = 0;
            std::size_t pos = hasher::hash(key64) % capacity;
            while(storage[pos].first != key){
                if(storage[pos] == emptySlot){
                    return {false, Value{}};
                }
                pos++;
                //wrap-around
                if(pos == capacity){
                    pos = 0;
                }
                probes++;
                if(maxProbes < probes){
                    return {false, Value{}};
                }
            }
            return {true, storage[pos].second};
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage result;
            result.host = sizeof(Data) * capacity;

            return result;
        }

        void writeToStream(std::ostream& os) const{
            os.write(reinterpret_cast<const char*>(&load), sizeof(float));
            os.write(reinterpret_cast<const char*>(&maxProbes), sizeof(std::size_t));
            os.write(reinterpret_cast<const char*>(&size), sizeof(std::size_t));
            os.write(reinterpret_cast<const char*>(&capacity), sizeof(std::size_t));

            const std::size_t elements = storage.size();
            const std::size_t bytes = sizeof(Data) * elements;
            os.write(reinterpret_cast<const char*>(&elements), sizeof(std::size_t));
            os.write(reinterpret_cast<const char*>(storage.data()), bytes);
        }

        void loadFromStream(std::ifstream& is){
            destroy();

            is.read(reinterpret_cast<char*>(&load), sizeof(float));
            is.read(reinterpret_cast<char*>(&maxProbes), sizeof(std::size_t));
            is.read(reinterpret_cast<char*>(&size), sizeof(std::size_t));
            is.read(reinterpret_cast<char*>(&capacity), sizeof(std::size_t));

            std::size_t elements;
            is.read(reinterpret_cast<char*>(&elements), sizeof(std::size_t));
            storage.resize(elements);
            const std::size_t bytes = sizeof(Data) * elements;
            is.read(reinterpret_cast<char*>(storage.data()), bytes);
        }

        void destroy(){
            std::vector<Data> tmp;
            std::swap(storage, tmp);

            maxProbes = 0;
            size = 0;
            capacity = 0;
        }

        std::size_t getCapacity() const{
            return capacity;
        }

        std::map<int, int> getCountDistribution() const{
            std::map<int, int> map;

            for(std::size_t i = 0; i < capacity; i++){
                if(storage[i] != emptySlot){
                    map[storage[i].second.second]++;
                }
            }
            return map;
        }

    private:

        using Data = std::pair<Key,Value>;

        Data emptySlot 
            = std::pair<Key,Value>{std::numeric_limits<Key>::max(), std::numeric_limits<Value>::max()};

        float load{};
        std::size_t maxProbes{};
        std::size_t size{};
        std::size_t capacity{};
        std::vector<Data> storage{};        
    };


    template<class Key, class Value>
    class SoACpuSingleValueHashTable{
        static_assert(std::is_integral<Key>::value, "Key must be integral!");

    public:
        class QueryResult{
        public:
            QueryResult() = default;
            QueryResult(const QueryResult&) = default;
            QueryResult(QueryResult&&) = default;
            QueryResult& operator=(const QueryResult&) = default;
            QueryResult& operator=(QueryResult&&) = default;

            QueryResult(bool b, Value v) : keyFound(b), val(std::move(v)){}

            bool valid() const{
                return keyFound;
            }
            const Value& value() const{
                return val;
            }
            Value& value(){
                return val;
            }
        private:
            bool keyFound;
            Value val;
        };

        SoACpuSingleValueHashTable() = default;
        SoACpuSingleValueHashTable(const SoACpuSingleValueHashTable&) = default;
        SoACpuSingleValueHashTable(SoACpuSingleValueHashTable&&) = default;
        SoACpuSingleValueHashTable& operator=(const SoACpuSingleValueHashTable&) = default;
        SoACpuSingleValueHashTable& operator=(SoACpuSingleValueHashTable&&) = default;

        SoACpuSingleValueHashTable(std::size_t size, float load)
            : load(load), size(size), capacity(size/load), 
              keys(capacity, emptyKey), values(capacity){
        }

        SoACpuSingleValueHashTable(std::size_t size) 
            : SoACpuSingleValueHashTable(size, 0.8f){            
        }

        bool operator==(const SoACpuSingleValueHashTable& rhs) const{
            return emptyKey == rhs.emptyKey 
                && feq(load, rhs.load)
                && maxProbes == rhs.maxProbes
                && size == rhs.size 
                && capacity == rhs.capacity
                && keys == rhs.keys
                && values = rhs.values;
        }

        bool operator!=(const SoACpuSingleValueHashTable& rhs) const{
            return !(operator==(rhs));
        }

        void insert(const Key& key, const Value& value){
            if(key == emptyKey){
                std::cerr << "1\n";
            }
            using hasher = hashers::MurmurHash<std::uint64_t>;

            std::size_t probes = 0;
            std::size_t pos = hasher::hash(key) % capacity;
            while(keys[pos] != emptyKey){
                pos++;
                //wrap-around
                if(pos == capacity){
                    pos = 0;
                }
                probes++;
            }
            keys[pos] = key;
            values[pos] = value;

            maxProbes = std::max(maxProbes, probes);
        }

        QueryResult query(const Key& key) const{
            using hasher = hashers::MurmurHash<std::uint64_t>;
            
            std::size_t probes = 0;
            std::size_t pos = hasher::hash(key) % capacity;
            while(keys[pos] != key){
                if(keys[pos] == emptyKey){
                    return {false, Value{}};
                }
                pos++;
                //wrap-around
                if(pos == capacity){
                    pos = 0;
                }
                probes++;
                if(maxProbes < probes){
                    return {false, Value{}};
                }
            }
            return {true, values[pos]};
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage result;
            result.host = sizeof(Key) * capacity;
            result.host += sizeof(Value) * capacity;

            return result;
        }

        void writeToStream(std::ostream& os) const{
            os.write(reinterpret_cast<const char*>(&load), sizeof(float));
            os.write(reinterpret_cast<const char*>(&maxProbes), sizeof(std::size_t));
            os.write(reinterpret_cast<const char*>(&size), sizeof(std::size_t));
            os.write(reinterpret_cast<const char*>(&capacity), sizeof(std::size_t));

            const std::size_t elements = keys.size();
            const std::size_t keysbytes = sizeof(Key) * elements;
            const std::size_t valuesbytes = sizeof(Value) * elements;
            os.write(reinterpret_cast<const char*>(&elements), sizeof(std::size_t));
            os.write(reinterpret_cast<const char*>(keys.data()), keysbytes);
            os.write(reinterpret_cast<const char*>(values.data()), valuesbytes);
        }

        void loadFromStream(std::ifstream& is){
            destroy();

            is.read(reinterpret_cast<char*>(&load), sizeof(float));
            is.read(reinterpret_cast<char*>(&maxProbes), sizeof(std::size_t));
            is.read(reinterpret_cast<char*>(&size), sizeof(std::size_t));
            is.read(reinterpret_cast<char*>(&capacity), sizeof(std::size_t));

            std::size_t elements;
            is.read(reinterpret_cast<char*>(&elements), sizeof(std::size_t));
            keys.resize(elements);
            const std::size_t keysbytes = sizeof(Key) * elements;
            is.read(reinterpret_cast<char*>(keys.data()), keysbytes);

            values.resize(elements);
            const std::size_t valuesbytes = sizeof(Value) * elements;
            is.read(reinterpret_cast<char*>(values.data()), valuesbytes);
        }

        void destroy(){
            std::vector<Key> ktmp;
            std::swap(keys, ktmp);

            std::vector<Value> vtmp;
            std::swap(values, vtmp);

            maxProbes = 0;
            size = 0;
            capacity = 0;
        }

    private:

        Key emptyKey = std::numeric_limits<Key>::max();

        float load{};
        std::size_t maxProbes{};
        std::size_t size{};
        std::size_t capacity{};
        std::vector<Key> keys{};
        std::vector<Value> values{};
    };






    template<class Key, class Value>
    class CpuReadOnlyMultiValueHashTable{
        static_assert(std::is_integral<Key>::value, "Key must be integral!");
    public:

        struct QueryResult{
            int numValues;
            const Value* valuesBegin;
        };

        CpuReadOnlyMultiValueHashTable() = default;
        CpuReadOnlyMultiValueHashTable(const CpuReadOnlyMultiValueHashTable&) = default;
        CpuReadOnlyMultiValueHashTable(CpuReadOnlyMultiValueHashTable&&) = default;
        CpuReadOnlyMultiValueHashTable& operator=(const CpuReadOnlyMultiValueHashTable&) = default;
        CpuReadOnlyMultiValueHashTable& operator=(CpuReadOnlyMultiValueHashTable&&) = default;

        CpuReadOnlyMultiValueHashTable(
            std::uint64_t maxNumValues_,
            float loadfactor_
        ) : loadfactor(loadfactor_), buildMaxNumValues{maxNumValues_}{
            buildkeys.reserve(buildMaxNumValues);
            buildvalues.reserve(buildMaxNumValues);
        }

        bool operator==(const CpuReadOnlyMultiValueHashTable& rhs) const{
            return values == rhs.values && lookup == rhs.lookup;
        }

        bool operator!=(const CpuReadOnlyMultiValueHashTable& rhs) const{
            return !(operator==(rhs));
        }

        void init(
            std::vector<Key> keys, 
            std::vector<Value> vals, 
            int maxValuesPerKey,
            ThreadPool* threadPool,
            bool valuesOfSameKeyMustBeSorted = true
        ){
            init(std::move(keys), std::move(vals), maxValuesPerKey, threadPool, {}, valuesOfSameKeyMustBeSorted);
        }

        template<class GroupByKeyOp>
        void init(
            GroupByKeyOp groupByKey,
            std::vector<Key> keys, 
            std::vector<Value> vals,
            ThreadPool* threadPool
        ){
            assert(keys.size() == vals.size());

            //std::cerr << "init valuesOfSameKeyMustBeSorted = " << valuesOfSameKeyMustBeSorted << "\n";

            if(isInit) return;

            std::vector<read_number> countsPrefixSum;
            values = std::move(vals);

            groupByKey(keys, values, countsPrefixSum);

            std::size_t nonEmtpyKeysCount = 0;
            for(std::size_t i = 0; i < keys.size(); i++){
                const auto count = countsPrefixSum[i+1] - countsPrefixSum[i];
                if(count > 0){
                    nonEmtpyKeysCount++;
                }
            }

            //lookup = std::move(AoSCpuSingleValueHashTable<Key, ValueIndex>(keys.size(), loadfactor));
            lookup = std::move(AoSCpuSingleValueHashTable<Key, ValueIndex>(nonEmtpyKeysCount, loadfactor));

            auto buildKeyLookup = [me=this, keys = std::move(keys), countsPrefixSum = std::move(countsPrefixSum)](){
                for(std::size_t i = 0; i < keys.size(); i++){
                    const auto count = countsPrefixSum[i+1] - countsPrefixSum[i];
                    if(count > 0){
                        me->lookup.insert(
                            keys[i], 
                            ValueIndex{countsPrefixSum[i], count}
                        );
                    }
                }
                me->isInit = true;
            };

            if(threadPool != nullptr){
                threadPool->enqueue(std::move(buildKeyLookup));
            }else{
                buildKeyLookup();
            }
        }

        void insert(const Key* keys, const Value* values, int N){
            assert(keys != nullptr);
            assert(values != nullptr);
            assert(buildMaxNumValues >= buildkeys.size() + N);

            buildkeys.insert(buildkeys.end(), keys, keys + N);
            buildvalues.insert(buildvalues.end(), values, values + N);
        }

        template<class GroupByKeyOp>
        void finalize(GroupByKeyOp groupByKey, ThreadPool* threadPool){
            init(groupByKey, std::move(buildkeys), std::move(buildvalues), threadPool);            
        }

        bool isInitialized() const noexcept{
            return isInit;
        }

        QueryResult query(const Key& key) const{
            assert(isInit);

            auto lookupQueryResult = lookup.query(key);

            if(lookupQueryResult.valid()){
                QueryResult result;

                result.numValues = lookupQueryResult.value().second;
                const auto valuepos = lookupQueryResult.value().first;
                result.valuesBegin = &values[valuepos];

                return result;
            }else{
                QueryResult result;

                result.numValues = 0;
                result.valuesBegin = nullptr;

                return result;
            }
        }

        void query(const Key* keys, std::size_t numKeys, QueryResult* resultsOutput) const{
            assert(isInit);
            for(std::size_t i = 0; i < numKeys; i++){
                resultsOutput[i] = query(keys[i]);
            }
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage result;
            result.host = sizeof(Value) * values.capacity();
            result.host += lookup.getMemoryInfo().host;
            result.host += sizeof(Key) * buildkeys.capacity();
            result.host += sizeof(Value) * buildvalues.capacity();

            result.device = lookup.getMemoryInfo().device;

            //std::cerr << lookup.getMemoryInfo().host << " " << result.host << " bytes\n";

            // std::cerr << "readonlytable. keys capacity: " << lookup.getCapacity() << ", values.size() " << values.size() << "\n";

            // auto map = lookup.getCountDistribution();
            // for(auto pair : map){
            //     std::cerr << pair.first << ": " << pair.second << "\n";
            // }

            return result;
        }

        void writeToStream(std::ostream& os) const{
            assert(isInit);

            const std::size_t elements = values.size();
            const std::size_t bytes = sizeof(Value) * elements;
            os.write(reinterpret_cast<const char*>(&elements), sizeof(std::size_t));
            os.write(reinterpret_cast<const char*>(values.data()), bytes);

            lookup.writeToStream(os);
        }

        void loadFromStream(std::ifstream& is){
            destroy();

            std::size_t elements;
            is.read(reinterpret_cast<char*>(&elements), sizeof(std::size_t));
            values.resize(elements);
            const std::size_t bytes = sizeof(Value) * elements;
            is.read(reinterpret_cast<char*>(values.data()), bytes);

            lookup.loadFromStream(is);
            isInit = true;
        }

        void destroy(){
            std::vector<Value> tmp;
            std::swap(values, tmp);

            lookup.destroy();
            isInit = false;
        }

        static std::size_t estimateGpuMemoryRequiredForInit(std::size_t numElements){

            std::size_t mem = 0;
            mem += sizeof(Key) * numElements; //d_keys
            mem += sizeof(Value) * numElements; //d_values
            mem += sizeof(read_number) * numElements; //d_indices
            mem += std::max(sizeof(read_number), sizeof(Value)) * numElements; //d_indices_tmp for sorting d_indices or d_values_tmp for sorted values

            return mem;
        }

    private:

        using ValueIndex = std::pair<read_number, BucketSize>;
        bool isInit = false;
        float loadfactor = 0.8f;
        std::uint64_t buildMaxNumValues = 0;
        std::vector<Key> buildkeys;
        std::vector<Value> buildvalues;
        // values with the same key are stored in contiguous memory locations
        // a single-value hashmap maps keys to the range of the corresponding values
        std::vector<Value> values; 
        AoSCpuSingleValueHashTable<Key, ValueIndex> lookup;
    };




}

#endif // CARE_CPUHASHTABLE_HPP
