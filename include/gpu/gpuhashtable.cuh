#ifndef CARE_GPUHASHTABLE_CUH
#define CARE_GPUHASHTABLE_CUH 

#include <config.hpp>
#include <kvtable.hpp>
#include <minhasher_transform.hpp>

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
        static_assert(std::is_assignable<std::uint64_t, Key>::value, "Key must be assignable to uint64_t!");
        static_assert(std::is_integral<Key>::value, "Key must be integral!");
        static_assert(std::is_integral<Value>::value, "Value must be integral!");

    public:
        class QueryResult{
        public:
            bool valid() const{
                return keyFound;
            }
            const Value& value() const{
                return v;
            }
            Value& value(){
                return v;
            }
        private:
            bool keyFound;
            Value v;
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

        void insert(const Key& key, const Value& value){
            std::size_t probes = 0;
            std::size_t pos = hashfunc(key) % size;
            while(storage[pos] != emptySlot){
                pos++;
                //wrap-around
                if(pos == size){
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
            std::size_t pos = murmur_hash_3_uint64_t(key) % size;
            while(storage[pos].first != key){
                if(storage[pos] == emptySlot){
                    return {false, Value{}};
                }
                pos++;
                //wrap-around
                if(pos == size){
                    pos = 0;
                }
                probes++;
                if(maxProbes < probes){
                    return {false, Value{}};
                }
            }
            return {true, storage[pos].second};
        }

    private:
        std::uint64_t hashfunc(const Key& key){
            //murmur64

            std::uint64_t x = key;

            x ^= x >> 33;
            x *= 0xff51afd7ed558ccd;
            x ^= x >> 33;
            x *= 0xc4ceb9fe1a85ec53;
            x ^= x >> 33;
            return x;
        }

        using Data = std::pair<Key,Value>;

        static constexpr Data emptySlot 
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
        static_assert(std::is_integral<Value>::value, "Value must be integral!");
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
            construct(std::move(keys), std::move(vals), maxValuesPerKey, gpuIds);
        }

        CpuReadOnlyMultiValueHashTable(
            std::vector<Key> keys, 
            std::vector<Value> vals, 
            int maxValuesPerKey
        ){
            construct(std::move(keys), std::move(vals), maxValuesPerKey);
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

    private:

        void construct(
            std::vector<Key> keys, 
            std::vector<Value> vals, 
            int maxValuesPerKey
        ){
            construct(std::move(keys), std::move(vals), maxValuesPerKey, {});
        }

        void construct(
            std::vector<Key> keys, 
            std::vector<Value> vals, 
            int maxValuesPerKey
            const std::vector<int>& gpuIds,
        ){
            assert(keys.size() == values.size());

            std::vector<Key> tmpunused;
            std::vector<read_number> counts;
            std::vector<read_number> countsPrefixSum;
            values = std::move(vals);

            MinhashTransformResult result;

            if(keys.size() == 0) return;

            #ifdef __NVCC__            
            if(deviceIds.size() == 0){
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
                    deviceIds, 
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
                        deviceIds, 
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
                lookup.insert(
                    keys[i], 
                    ValueIndex{countsPrefixSum[i], countsPrefixSum[i+1] - countsPrefixSum[i]};
                );
            }
        }

        using ValueIndex = std::pair<read_number, BucketSize>;

        std::vector<Value> values;
        NaiveCpuSingleValueHashTable<Key, ValueIndex> lookup;
    };

 

    








}
}

#endif CARE_GPUHASHTABLE_CUH
