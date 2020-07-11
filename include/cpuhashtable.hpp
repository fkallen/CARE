#ifndef CARE_CPUHASHTABLE_CUH
#define CARE_CPUHASHTABLE_CUH 

#include <config.hpp>
#include <minhasher_transform.hpp>
#include <memorymanagement.hpp>



#include <vector>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <array>
#include <type_traits>
#include <limits>

namespace care{
namespace gpu{

    template<class Key, class Value>
    class NaiveCpuSingleValueHashTable{
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

        NaiveCpuSingleValueHashTable() = default;
        NaiveCpuSingleValueHashTable(const NaiveCpuSingleValueHashTable&) = default;
        NaiveCpuSingleValueHashTable(NaiveCpuSingleValueHashTable&&) = default;
        NaiveCpuSingleValueHashTable& operator=(const NaiveCpuSingleValueHashTable&) = default;
        NaiveCpuSingleValueHashTable& operator=(NaiveCpuSingleValueHashTable&&) = default;

        NaiveCpuSingleValueHashTable(std::size_t size, float load)
            : load(load), size(size), capacity(size/load)
        {
            storage.resize(capacity, emptySlot);
        }

        NaiveCpuSingleValueHashTable(std::size_t size) 
            : NaiveCpuSingleValueHashTable(size, 0.8f){            
        }

        bool operator==(const NaiveCpuSingleValueHashTable& rhs) const{
            return emptySlot == rhs.emptySlot 
                && load == rhs.load
                && maxProbes == rhs.maxProbes
                && size == rhs.size 
                && capacity == rhs.capacity
                && storage == rhs.storage;
        }

        bool operator!=(const NaiveCpuSingleValueHashTable& rhs) const{
            return !(operator==(rhs));
        }

        void insert(const Key& key, const Value& value){
            std::size_t probes = 0;
            std::size_t pos = hashfunc(key) % capacity;
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
            std::size_t probes = 0;
            std::size_t pos = hashfunc(key) % capacity;
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

    private:
        std::uint64_t hashfunc(std::uint64_t x) const{
            //murmur64
            x ^= x >> 33;
            x *= 0xff51afd7ed558ccd;
            x ^= x >> 33;
            x *= 0xc4ceb9fe1a85ec53;
            x ^= x >> 33;
            return x;
        }

        using Data = std::pair<Key,Value>;

        Data emptySlot 
            = std::pair<Key,Value>{std::numeric_limits<Key>::max(), std::numeric_limits<Value>::max()};

        float load;
        std::size_t maxProbes;
        std::size_t size;
        std::size_t capacity;
        std::vector<Data> storage;        
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
            std::vector<Key> keys, 
            std::vector<Value> vals, 
            int maxValuesPerKey,
            const std::vector<int>& gpuIds
        ){
            init(std::move(keys), std::move(vals), maxValuesPerKey, gpuIds);
        }

        CpuReadOnlyMultiValueHashTable(
            std::vector<Key> keys, 
            std::vector<Value> vals, 
            int maxValuesPerKey
        ){
            init(std::move(keys), std::move(vals), maxValuesPerKey);
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
            int maxValuesPerKey
        ){
            init(std::move(keys), std::move(vals), maxValuesPerKey, {});
        }

        void init(
            std::vector<Key> keys, 
            std::vector<Value> vals, 
            int maxValuesPerKey,
            const std::vector<int>& gpuIds
        ){
            assert(keys.size() == vals.size());

            std::vector<Key> tmpunused;
            std::vector<read_number> counts;
            std::vector<read_number> countsPrefixSum;
            values = std::move(vals);

            MinhashTransformResult result;

            if(keys.size() == 0) return;

            #ifdef __NVCC__            
            if(gpuIds.size() == 0){
            #endif
                result = cpu_transformation(
                    keys, 
                    values, 
                    counts, 
                    countsPrefixSum, 
                    tmpunused, 
                    maxValuesPerKey
                );
            #ifdef __NVCC__
            }else{
                
                std::pair<bool, MinhashTransformResult> pair = GPUTransformation<false>::execute(
                    keys, 
                    values, 
                    counts, 
                    countsPrefixSum, 
                    tmpunused, 
                    gpuIds, 
                    maxValuesPerKey
                );

                bool success = pair.first;
                result = pair.second;

                if(!success){
                    std::cerr << "Fallback to managed memory transformation.\n";

                    pair = GPUTransformation<true>::execute(
                        keys, 
                        values, 
                        counts, 
                        countsPrefixSum, 
                        tmpunused, 
                        gpuIds, 
                        maxValuesPerKey
                    );

                    success = pair.first;
                    result = pair.second;
                }

                if(!success){
                    std::cerr << "\nFallback to cpu transformation.\n";
                    
                    result = cpu_transformation(
                        keys, 
                        values, 
                        counts, 
                        countsPrefixSum, 
                        tmpunused, 
                        maxValuesPerKey
                    );
                }
            }
            #endif

            lookup = NaiveCpuSingleValueHashTable<Key, ValueIndex>(keys.size(), 0.8f);

            for(std::size_t i = 0; i < keys.size(); i++){
                // if(i < 10){
                //     std::cerr << keys[i] << " " << countsPrefixSum[i] << " " << (countsPrefixSum[i+1] - countsPrefixSum[i]) << "\n";
                // }
                
                lookup.insert(
                    keys[i], 
                    ValueIndex{countsPrefixSum[i], countsPrefixSum[i+1] - countsPrefixSum[i]}
                );
                
            }

            (void)result;
        }

        QueryResult query(const Key& key) const{
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
            for(std::size_t i = 0; i < numKeys; i++){
                resultsOutput[i] = query(keys[i]);
            }
        }

        MemoryUsage getMemoryInfo() const{
            MemoryUsage result;
            result.host = sizeof(Value) * values.capacity();
            result.host += lookup.getMemoryInfo().host;

            result.device = lookup.getMemoryInfo().device;

            return result;
        }

        void writeToStream(std::ostream& os) const{

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
        }

        void destroy(){
            std::vector<Value> tmp;
            std::swap(values, tmp);

            lookup.destroy();
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

        std::vector<Value> values;
        NaiveCpuSingleValueHashTable<Key, ValueIndex> lookup;
    };



}
}

#endif // CARE_GPUHASHTABLE_CUH
