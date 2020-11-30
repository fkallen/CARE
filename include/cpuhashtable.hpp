#ifndef CARE_CPUHASHTABLE_HPP
#define CARE_CPUHASHTABLE_HPP 

#include <config.hpp>
#include <memorymanagement.hpp>

#include <vector>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <array>
#include <type_traits>
#include <limits>
#include <algorithm>
#include <iostream>

#include <thrust/system/omp/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/inner_product.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#ifdef __NVCC__
#include <thrust/device_vector.h>
#include <hpc_helpers.cuh>
#endif

namespace care{

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

        float load{};
        std::size_t maxProbes{};
        std::size_t size{};
        std::size_t capacity{};
        std::vector<Data> storage{};        
    };





namespace cpuhashtabledetail{

        struct AllocCounter{
            std::size_t maxbytes = 0;
            std::size_t bytes = 0;

            template<class T>
            void alloc(std::size_t elements){
                bytes += sizeof(T) * elements;
                maxbytes = std::max(maxbytes, bytes);
            }

            template<class T>
            void dealloc(std::size_t elements){
                auto size = sizeof(T) * elements;
                assert(size <= bytes);
                bytes -= size;
            }
        };

        struct TransformResult{
            std::uint64_t numberOfUniqueKeys = 0;
            std::uint64_t numberOfRemovedKeys = 0;
            std::uint64_t numberOfRemovedValues = 0;
        };

        struct TransformRemoveKeysResult{
            std::uint64_t numberOfRemovedKeys = 0;
            std::uint64_t numberOfRemovedValues = 0;
        };


        template<class Key_t, class Value_t, class Index_t>
        std::uint64_t transformCPUCompactKeys(std::vector<Key_t>& keys,
                                            std::vector<Value_t>& values,
                                            std::vector<Index_t>& countsPrefixSum,
                                            bool valuesOfSameKeyMustBeSorted = true){

            auto deallocVector = [](auto& vec){
                using T = typename std::remove_reference<decltype(vec)>::type;
                T tmp{};
                vec.swap(tmp);
            };

            assert(keys.size() == values.size());
            assert(std::numeric_limits<Index_t>::max() >= keys.size());

            const std::size_t size = keys.size();
            auto policy = thrust::omp::par;

            std::vector<Index_t> indices(size);
            auto* indices_begin = indices.data();
            auto* indices_end = indices.data() + indices.size();

            //TIMERSTARTCPU(iota);
            thrust::sequence(policy, indices_begin, indices_end, Index_t(0));
            //TIMERSTOPCPU(iota);

            //TIMERSTARTCPU(sortindices);
            if(valuesOfSameKeyMustBeSorted){
            //sort indices by key. if keys are equal, sort by value
                thrust::sort(
                    policy,
                    indices_begin,
                    indices_end,
                    [&] (const auto &lhs, const auto &rhs) {
                        if(keys[lhs] == keys[rhs]){
                            return values[lhs] < values[rhs];
                        }
                        return keys[lhs] < keys[rhs];
                    }
                );
            }else{
                thrust::sort(
                    policy,
                    indices_begin,
                    indices_end,
                    [&] (const auto &lhs, const auto &rhs) {
                        return keys[lhs] < keys[rhs];
                    }
                );
            }
            //TIMERSTOPCPU(sortindices);


            //TIMERSTARTCPU(sortvalues);
            std::vector<Value_t> sortedValues(size);
            thrust::copy(policy,
                        thrust::make_permutation_iterator(values.begin(), indices_begin),
                        thrust::make_permutation_iterator(values.begin(), indices_end),
                        sortedValues.begin());

            //TIMERSTOPCPU(sortvalues);

            std::swap(sortedValues, values);

            deallocVector(sortedValues);

            //TIMERSTARTCPU(sortkeys);
            std::vector<Key_t> sortedKeys(size);
            thrust::copy(policy,
                        thrust::make_permutation_iterator(keys.begin(), indices_begin),
                        thrust::make_permutation_iterator(keys.begin(), indices_end),
                        sortedKeys.begin());

            //TIMERSTOPCPU(sortkeys);

            std::swap(sortedKeys, keys);

            deallocVector(sortedKeys);
            deallocVector(indices);

            const Index_t nUniqueKeys = thrust::inner_product(policy,
                                                keys.begin(),
                                                keys.end() - 1,
                                                keys.begin() + 1,
                                                Index_t(1),
                                                thrust::plus<Key_t>(),
                                                thrust::not_equal_to<Key_t>());

            //std::cout << "unique keys " << nUniqueKeys << ". ";

            std::vector<Key_t> histogram_keys(nUniqueKeys);
            std::vector<Index_t> histogram_counts(nUniqueKeys);

            auto* keys_begin = keys.data();
            auto* keys_end = keys.data() + keys.size();
            auto* histogram_keys_begin = histogram_keys.data();
            auto* histogram_counts_begin = histogram_counts.data();

            //make key - frequency histogram
            auto histogramEndIterators = thrust::reduce_by_key(policy,
                                    keys_begin,
                                    keys_end,
                                    thrust::constant_iterator<Index_t>(1),
                                    histogram_keys_begin,
                                    histogram_counts_begin);

            assert(histogramEndIterators.first == histogram_keys.data() + nUniqueKeys);
            assert(histogramEndIterators.second == histogram_counts.data() + nUniqueKeys);

            countsPrefixSum.resize(nUniqueKeys+1, Index_t(0));

            thrust::inclusive_scan(policy,
                                    histogram_counts_begin,
                                    histogramEndIterators.second,
                                    countsPrefixSum.data() + 1);

            keys.swap(histogram_keys);

            return nUniqueKeys;
        }

        template<class Key_t, class Value_t, class Index_t>
        TransformRemoveKeysResult transformCPURemoveKeysWithToManyValues(std::vector<Key_t>& keys, 
                                                            std::vector<Value_t>& values, 
                                                            std::vector<Index_t>& countsPrefixSum,
                                                            int maxValuesPerKey_){
            auto deallocVector = [](auto& vec){
                using T = typename std::remove_reference<decltype(vec)>::type;
                T tmp{};
                vec.swap(tmp);
            };

            TransformRemoveKeysResult result;

            auto policy = thrust::omp::par;

            const Index_t maxValuesPerKey = maxValuesPerKey_;

            std::size_t oldSizeKeys = keys.size();
            std::size_t oldSizeValues = values.size();

            std::vector<Index_t> counts(countsPrefixSum.size()-1);
            {
                auto psPtr = thrust::raw_pointer_cast(countsPrefixSum.data());
                auto countsPtr = thrust::raw_pointer_cast(counts.data());
            
                thrust::adjacent_difference(policy,
                                            psPtr+1,
                                            psPtr + countsPrefixSum.size(),
                                            countsPtr);

            }
            
            //handle keys
            int numKeysToRemove = 0;
            {
                std::vector<char> removeflags(countsPrefixSum.size()-1);

                thrust::transform(policy,
                                    counts.begin(),
                                    counts.end(),
                                    removeflags.begin(),
                                    [=](auto i){
                                        return i > maxValuesPerKey ? 1 : 0;
                                    });

                numKeysToRemove = thrust::count_if(policy,
                                                    removeflags.begin(),
                                                    removeflags.end(),
                                                    [] (auto flag){
                                                        return flag == 1;
                                                    });         
            }

            result.numberOfRemovedKeys = numKeysToRemove;

            //std::cerr << "Can remove values of " << numKeysToRemove << " high frequency keys. "; 

            //handle values
            int numValuesToRemove = 0;
            {
                std::vector<char> removeflags(values.size(), false);

                thrust::for_each(policy,
                                thrust::counting_iterator<Index_t>(0),
                                thrust::counting_iterator<Index_t>(0) + oldSizeKeys,
                                [&](int index){
                                    if(counts[index] > maxValuesPerKey){
                                        auto begin = countsPrefixSum[index];
                                        auto end = countsPrefixSum[index+1];
                                        for(Index_t k = begin; k < end; k++){
                                            removeflags[k] = true;
                                        }
                                    }
                                });          

                numValuesToRemove = thrust::count_if(policy,
                                                        removeflags.begin(),
                                                        removeflags.end(),
                                                        [](auto flag){
                                                            return flag == 1;
                                                        });

                std::vector<Value_t> values_tmp(oldSizeValues - numValuesToRemove);

                thrust::copy_if(values.begin(),
                                values.end(),
                                removeflags.begin(),
                                values_tmp.begin(),
                                [](auto flag){
                                    return flag == 0;
                                });

                values.swap(values_tmp);
            }

            result.numberOfRemovedValues = numValuesToRemove;

            //std::cerr << "Removed corresponding values: " << numValuesToRemove << ". "; 

            //handle counts prefix sum
            {
                thrust::for_each(policy,
                                thrust::counting_iterator<Index_t>(0),
                                thrust::counting_iterator<Index_t>(0) + counts.size(),
                                [&] (auto i){
                                    if(counts[i] > maxValuesPerKey){
                                        counts[i] = 0;
                                    }
                                });
                                
                deallocVector(countsPrefixSum);

                countsPrefixSum.resize(keys.size() + 1);
                countsPrefixSum[0] = 0;

                thrust::inclusive_scan(policy,
                                        counts.begin(),
                                        counts.end(),
                                        countsPrefixSum.begin() + 1);              
            }

            return result;
        }

        template<class Key_t, class Value_t, class Index_t>
        TransformResult cpu_transformation(std::vector<Key_t>& keys,
                                std::vector<Value_t>& values,
                                std::vector<Index_t>& countsPrefixSum,
                                int maxValuesPerKey,
                                bool valuesOfSameKeyMustBeSorted = true){

            std::uint64_t uniqueKeys = transformCPUCompactKeys(
                keys, 
                values, 
                countsPrefixSum,
                valuesOfSameKeyMustBeSorted
            );

            TransformRemoveKeysResult removeKeysResult = transformCPURemoveKeysWithToManyValues(
                keys, 
                values, 
                countsPrefixSum, 
                maxValuesPerKey
            );

            TransformResult result;
            result.numberOfUniqueKeys = uniqueKeys;
            result.numberOfRemovedKeys = removeKeysResult.numberOfRemovedKeys;
            result.numberOfRemovedValues = removeKeysResult.numberOfRemovedValues;

            return result;
        };


    #ifdef __NVCC__

        template<bool allowFallback>
        struct TransformGPUCompactKeys{
            template<class T>
            using ThrustAlloc = helpers::ThrustFallbackDeviceAllocator<T, allowFallback>;

            template<class Key_t, class Value_t, class Index_t>
            static std::size_t estimateRequiredGpuMem(
                                std::vector<Key_t>& keys, 
                                std::vector<Value_t>& values, 
                                std::vector<Index_t>& countsPrefixSum,
                                bool valuesOfSameKeyMustBeSorted = true){
                
                return estimateRequiredGpuMem<Key_t, Value_t, Index_t>(values.size()), valuesOfSameKeyMustBeSorted;
            }

            template<class Key_t, class Value_t, class Index_t>
            static std::size_t estimateRequiredGpuMem(Index_t numEntries, bool valuesOfSameKeyMustBeSorted = true){

                std::size_t mem = 0;
                mem += sizeof(Key_t) * numEntries; //d_keys
                mem += sizeof(Value_t) * numEntries; //d_values
                mem += sizeof(Index_t) * numEntries; //d_indices
                mem += std::max(sizeof(Index_t), sizeof(Value_t)) * numEntries; //d_indices_tmp for sorting d_indices or d_values_tmp for sorted values

                return mem;
            }

            template<class Key_t, class Value_t, class Index_t>
            static std::uint64_t execute(std::vector<Key_t>& keys, 
                                std::vector<Value_t>& values, 
                                std::vector<Index_t>& countsPrefixSum,
                                const std::vector<int>& /*deviceIds*/,
                                bool valuesOfSameKeyMustBeSorted = true){

                auto deallocVector = [](auto& vec){
                    using T = typename std::remove_reference<decltype(vec)>::type;
                    T tmp{};
                    vec.swap(tmp);
                };

                std::size_t size = values.size();

                ThrustAlloc<char> allocator;
                auto allocatorPolicy = thrust::cuda::par(allocator);

                thrust::device_vector<Key_t, ThrustAlloc<Key_t>> d_keys(size);                
                thrust::device_vector<Index_t, ThrustAlloc<Index_t>> d_indices(size);

                thrust::copy(keys.begin(), keys.end(), d_keys.begin());                
                thrust::sequence(allocatorPolicy, d_indices.begin(), d_indices.end(), Index_t(0));

                thrust::device_ptr<Key_t> d_keys_ptr = d_keys.data();                

                thrust::device_vector<Value_t, ThrustAlloc<Value_t>> d_values;

                //std::cerr << "before sort\n";



                //sort indices
                if(valuesOfSameKeyMustBeSorted){
                    d_values.resize(size);
                    thrust::copy(values.begin(), values.end(), d_values.begin());
                    thrust::device_ptr<Value_t> d_values_ptr = d_values.data();

                    thrust::sort(
                        allocatorPolicy,
                        d_indices.begin(),
                        d_indices.end(),
                        [=] __device__ (const auto& lhs, const auto& rhs) {
                            if(d_keys_ptr[lhs] == d_keys_ptr[rhs]){
                                return d_values_ptr[lhs] < d_values_ptr[rhs];
                            }
                            return d_keys_ptr[lhs] < d_keys_ptr[rhs];
                        }
                    );
                }else{
                    thrust::sort(allocatorPolicy,
                        d_indices.begin(),
                        d_indices.end(),
                        [=] __device__ (const auto& lhs, const auto& rhs) {
                            return d_keys_ptr[lhs] < d_keys_ptr[rhs];
                        }
                    );

                    d_values.resize(size);
                    thrust::copy(values.begin(), values.end(), d_values.begin());
                }
                
                deallocVector(d_keys);

                //std::cerr << "after sort\n";

                // std::cerr << "before sort by key\n";

                // //sort indices
                // thrust::sort_by_key(allocatorPolicy, d_keys.begin(), d_keys.end(), d_values.begin());

                // std::cerr << "after sort by key\n";

                //sort values by order defined by indices and copy sorted values to host.
                thrust::device_vector<Value_t, ThrustAlloc<Value_t>> d_values_tmp(size);

                thrust::copy(thrust::make_permutation_iterator(d_values.begin(), d_indices.begin()),
                            thrust::make_permutation_iterator(d_values.begin(), d_indices.end()),
                            d_values_tmp.begin());

                thrust::copy(d_values_tmp.begin(),
                            d_values_tmp.end(),
                            values.begin());

                deallocVector(d_values_tmp);
                deallocVector(d_values);

                d_keys.resize(size);
                thrust::copy(keys.begin(), keys.end(), d_keys.begin());

                //sort keys by order defined by indices
                thrust::device_vector<Key_t, ThrustAlloc<Key_t>> d_keys_tmp(size);

                thrust::copy(thrust::make_permutation_iterator(d_keys.begin(), d_indices.begin()),
                            thrust::make_permutation_iterator(d_keys.begin(), d_indices.end()),
                            d_keys_tmp.begin());

                std::swap(d_keys, d_keys_tmp);

                deallocVector(d_keys_tmp);
                deallocVector(d_indices);

                const Index_t nUniqueKeys = thrust::inner_product(allocatorPolicy,
                    d_keys.begin(),
                    d_keys.end() - 1,
                    d_keys.begin() + 1,
                    Index_t(1),
                    thrust::plus<Key_t>(),
                    thrust::not_equal_to<Key_t>());

                //std::cerr << "unique keys " << nUniqueKeys << ". ";

                //histogram storage
                thrust::device_vector<Key_t, ThrustAlloc<Key_t>> d_histogram_keys(nUniqueKeys);
                thrust::device_vector<Index_t, ThrustAlloc<Index_t>> d_histogram_counts(nUniqueKeys);                

                //make key multiplicity histogram
                auto histogramEndIterators = thrust::reduce_by_key(allocatorPolicy,
                    d_keys.begin(),
                    d_keys.end(),
                    thrust::constant_iterator<Index_t>(1),
                    d_histogram_keys.begin(),
                    d_histogram_counts.begin());

                assert(histogramEndIterators.first == d_histogram_keys.end());
                assert(histogramEndIterators.second == d_histogram_counts.end());

                deallocVector(keys);

                keys.resize(nUniqueKeys);

                thrust::copy(d_histogram_keys.begin(),
                            d_histogram_keys.end(),
                            keys.begin());

                deallocVector(d_histogram_keys);

                thrust::device_vector<Index_t, ThrustAlloc<Index_t>> d_histogram_counts_prefixsum(nUniqueKeys+1, Index_t(0));

                thrust::inclusive_scan(allocatorPolicy,
                    d_histogram_counts.begin(),
                    histogramEndIterators.second,
                    d_histogram_counts_prefixsum.begin() + 1);

                deallocVector(countsPrefixSum);

                countsPrefixSum.resize(nUniqueKeys+1);
                thrust::copy(d_histogram_counts_prefixsum.begin(),
                            d_histogram_counts_prefixsum.end(),
                            countsPrefixSum.begin());                

                return nUniqueKeys;
            }
        };


        template<bool allowFallback>
        struct TransformGPURemoveKeysWithToManyValues{
            template<class T>
            using ThrustAlloc = helpers::ThrustFallbackDeviceAllocator<T, allowFallback>;

            template<class Key_t, class Value_t, class Index_t>
            static TransformRemoveKeysResult execute(std::vector<Key_t>& keys, 
                                std::vector<Value_t>& values, 
                                std::vector<Index_t>& countsPrefixSum,
                                int maxValuesPerKey_,
                                const std::vector<int>& /*deviceIds*/){

                auto deallocVector = [](auto& vec){
                    using T = typename std::remove_reference<decltype(vec)>::type;
                    T tmp{};
                    vec.swap(tmp);
                };

                TransformRemoveKeysResult result;

                const Index_t maxValuesPerKey = maxValuesPerKey_;

                ThrustAlloc<char> allocator;
                auto allocatorPolicy = thrust::cuda::par(allocator);

                const std::size_t oldSizeKeys = keys.size();
                const std::size_t oldSizeValues = values.size();

                thrust::device_vector<Index_t, ThrustAlloc<Index_t>> d_countsPrefixSum(countsPrefixSum.size());

                thrust::copy(countsPrefixSum.begin(), countsPrefixSum.end(), d_countsPrefixSum.begin());

                thrust::device_vector<Index_t, ThrustAlloc<Index_t>> d_counts(countsPrefixSum.size()-1);
                //make counts array from prefixsum
                thrust::adjacent_difference(allocatorPolicy,
                                            d_countsPrefixSum.begin()+1,
                                            d_countsPrefixSum.end(),
                                            d_counts.begin());
                
                //handle keys
                int numKeysToRemove = 0;
                {
                    thrust::device_vector<bool, ThrustAlloc<bool>> d_removeflags(countsPrefixSum.size()-1);

                    thrust::transform(allocatorPolicy,
                                        d_counts.begin(),
                                        d_counts.end(),
                                        d_removeflags.begin(),
                                        [=] __device__ (auto i){
                                            return i > maxValuesPerKey;
                                        });

                    numKeysToRemove = thrust::count_if(allocatorPolicy,
                                                            d_removeflags.begin(),
                                                            d_removeflags.end(),
                                                            [] __device__ (auto flag){
                                                                return flag;
                                                            });
                }

                result.numberOfRemovedKeys = numKeysToRemove;
                
                //std::cerr << "Can remove values of " << numKeysToRemove << " high frequency keys. "; 

                //handle values
                int numValuesToRemove = 0;
                {
                    thrust::device_vector<bool, ThrustAlloc<bool>> d_removeflags(values.size(), false);

                    auto countsPtr = d_counts.data();
                    auto countPrefixSumPtr = d_countsPrefixSum.data();
                    auto flagsPtr = d_removeflags.data();

                    //if counts[i] > maxValuesPerKey, set removeflag for all corresponding values of this key                
                    thrust::for_each(allocatorPolicy,
                        thrust::counting_iterator<Index_t>(0),
                        thrust::counting_iterator<Index_t>(0) + oldSizeKeys,
                        [=] __device__ (int index){
                            if(countsPtr[index] > maxValuesPerKey){
                                auto begin = countPrefixSumPtr[index];
                                auto end = countPrefixSumPtr[index+1];
                                for(Index_t k = begin; k < end; k++){
                                    flagsPtr[k] = true;
                                }
                            }
                        });

                    //copy values to device
                    thrust::device_vector<Value_t, ThrustAlloc<Value_t>> d_values(values.size());
                    thrust::copy(values.begin(), values.end(), d_values.begin());                

                    //determine number of set remove flags
                    numValuesToRemove = thrust::count_if(allocatorPolicy,
                                                            d_removeflags.begin(),
                                                            d_removeflags.end(),
                                                            [] __device__ (auto flag){
                                                                return flag;
                                                            });

                    //select the remaining values, then copy them back to host
                    thrust::device_vector<Value_t, ThrustAlloc<Value_t>> d_values_tmp(oldSizeValues - numValuesToRemove);

                    thrust::copy_if(d_values.begin(),
                                    d_values.end(),
                                    d_removeflags.begin(),
                                    d_values_tmp.begin(),
                                    [] __device__ (auto flag){
                                        return !flag;
                                    });

                    deallocVector(values);
                    values.resize(d_values_tmp.size());

                    thrust::copy(d_values_tmp.begin(), d_values_tmp.end(), values.begin());
                }

                result.numberOfRemovedValues = numValuesToRemove;

                // std::cerr << "Removed corresponding values: " << numValuesToRemove << ". "; 

                //handle counts prefix sum
                {
                    auto countsPtr = d_counts.data();

                    //set counts of removed keys to 0
                    thrust::for_each(allocatorPolicy,
                                    thrust::counting_iterator<Index_t>(0),
                                    thrust::counting_iterator<Index_t>(0) + d_counts.size(),
                                    [=] __device__ (auto i){
                                        if(countsPtr[i] > maxValuesPerKey){
                                            countsPtr[i] = 0;
                                        }
                                    });

                    //make new prefix_sum
                    auto psend = thrust::inclusive_scan(allocatorPolicy,
                                            d_counts.begin(),
                                            d_counts.end(),
                                            d_countsPrefixSum.begin());

                    assert(keys.size() == thrust::distance(d_countsPrefixSum.begin(), psend));
                    
                    deallocVector(countsPrefixSum);
                    countsPrefixSum.resize(keys.size() + 1);
                    countsPrefixSum[0] = 0;
                    thrust::copy(d_countsPrefixSum.begin(), psend, countsPrefixSum.begin() + 1);                
                }

                return result;
            }
        };

        template<bool allowFallback>
        struct GPUTransformation{

            template<class Key_t, class Value_t, class Index_t>
            static std::pair<bool, TransformResult> execute(std::vector<Key_t>& keys, 
                                std::vector<Value_t>& values, 
                                std::vector<Index_t>& countsPrefixSum, 
                                const std::vector<int>& deviceIds,
                                int maxValuesPerKey,
                                bool valuesOfSameKeyMustBeSorted = true){

                assert(keys.size() == values.size());
                assert(std::numeric_limits<Index_t>::max() >= keys.size());
                assert(deviceIds.size() > 0);

                std::pair<bool, TransformResult> result;

                if(keys.empty()){
                    std::cerr << "Want to transform empty map!\n";
                    assert(false);
                    //return true;
                }

                int oldDeviceId;
                cudaError_t setupStatus = cudaGetDevice(&oldDeviceId);
                if(setupStatus != cudaSuccess) //we cannot recover from this.
                    throw std::runtime_error("Could not query current device id!");

                setupStatus = cudaSetDevice(deviceIds.front()); //TODO choose appropriate id from deviceIds
                if(setupStatus != cudaSuccess) //we cannot recover from this.
                    throw std::runtime_error("Could not set device id!");

                bool& success = result.first;
                TransformResult& transformresult = result.second;

                success = false;

                try{           
                    transformresult.numberOfUniqueKeys = TransformGPUCompactKeys<allowFallback>
                            ::execute(keys, values, countsPrefixSum, deviceIds, valuesOfSameKeyMustBeSorted);

                    auto removeresult = TransformGPURemoveKeysWithToManyValues<allowFallback>
                            ::execute(keys, values, countsPrefixSum, maxValuesPerKey, deviceIds);  

                    transformresult.numberOfRemovedKeys = removeresult.numberOfRemovedKeys;
                    transformresult.numberOfRemovedValues = removeresult.numberOfRemovedValues;

                    success = true;

                }catch(const thrust::system_error& ex){
                    std::cerr << ex.what() << '\n';
                    cudaGetLastError();
                    success = false;
                }catch(const std::exception& ex){
                    std::cerr << ex.what() << '\n';
                    cudaGetLastError();
                    success = false;
                }catch(...){
                    cudaGetLastError();
                    success = false;
                }

                setupStatus = cudaSetDevice(oldDeviceId);
                if(setupStatus != cudaSuccess) //we cannot recover from this.
                    throw std::runtime_error("Could not revert device id!");

                return result;
            }

        };
    #endif
}















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
            const std::vector<int>& gpuIds,
            bool valuesOfSameKeyMustBeSorted = true
        ){
            init(std::move(keys), std::move(vals), maxValuesPerKey, gpuIds, valuesOfSameKeyMustBeSorted);
        }

        CpuReadOnlyMultiValueHashTable(
            std::vector<Key> keys, 
            std::vector<Value> vals, 
            int maxValuesPerKey,
            bool valuesOfSameKeyMustBeSorted = true
        ){
            init(std::move(keys), std::move(vals), maxValuesPerKey, valuesOfSameKeyMustBeSorted);
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
            bool valuesOfSameKeyMustBeSorted = true
        ){
            init(std::move(keys), std::move(vals), maxValuesPerKey, {}, valuesOfSameKeyMustBeSorted);
        }

        void init(
            std::vector<Key> keys, 
            std::vector<Value> vals, 
            int maxValuesPerKey,
            const std::vector<int>& gpuIds,
            bool valuesOfSameKeyMustBeSorted = true
        ){
            assert(keys.size() == vals.size());

            std::vector<read_number> countsPrefixSum;
            values = std::move(vals);

            if(keys.size() == 0) return;

            #ifdef __NVCC__            
            if(gpuIds.size() == 0){
            #endif
                cpuhashtabledetail::cpu_transformation(
                    keys, 
                    values, 
                    countsPrefixSum, 
                    maxValuesPerKey,
                    valuesOfSameKeyMustBeSorted
                );
            #ifdef __NVCC__
            }else{
                
                auto pair = cpuhashtabledetail::GPUTransformation<false>::execute(
                    keys, 
                    values, 
                    countsPrefixSum, 
                    gpuIds, 
                    maxValuesPerKey,
                    valuesOfSameKeyMustBeSorted
                );

                bool success = pair.first;

                if(!success){
                    std::cerr << "Fallback to managed memory transformation.\n";

                    pair = cpuhashtabledetail::GPUTransformation<true>::execute(
                        keys, 
                        values, 
                        countsPrefixSum, 
                        gpuIds, 
                        maxValuesPerKey,
                        valuesOfSameKeyMustBeSorted
                    );

                    success = pair.first;
                }

                if(!success){
                    std::cerr << "\nFallback to cpu transformation.\n";
                    
                    cpuhashtabledetail::cpu_transformation(
                        keys, 
                        values, 
                        countsPrefixSum, 
                        maxValuesPerKey,
                        valuesOfSameKeyMustBeSorted
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

        // values with the same key are stored in contiguous memory locations
        // a single-value hashmap maps keys to the range of the corresponding values
        std::vector<Value> values; 
        NaiveCpuSingleValueHashTable<Key, ValueIndex> lookup;
    };



#if 0

    template<class Key, class Value, class Index>
    struct CpuMultiValueHashTableInConstruction{
    public:

        CpuMultiValueHashTableInConstruction(std::size_t pairs_, float load_, std::size_t maxValuesPerKey_)
            : maxPairs(pairs_), load(load_), maxValuesPerKey(maxValuesPerKey_){

            // if(maxPairs > std::size_t(std::numeric_limits<int>::max())){
            //     assert(maxPairs <= std::size_t(std::numeric_limits<int>::max())); 
            // }

            myKeys.resize(maxPairs);
            myValues.resize(maxPairs);
        }

        void insert(
            const Key* keys, 
            const Value* values, 
            Index N
        ){
            if(N == 0) return;

            assert(keys != nullptr);
            assert(values != nullptr);
            assert(numKeys + N <= maxPairs);    
            assert(numValues + N <= maxPairs);

            myKeys.insert(myKeys.end(), keys, keys + N);
            myValues.insert(myValues.end(), values, values + N);

            numKeys += N;
            numValues += N;
        }

        MemoryUsage getMemoryInfo() const{

            MemoryUsage result{};
            result.host += myKeys.size() * sizeof(Key);
            result.host += myValues.size() * sizeof(Value);

            return result;
        }

        float load{};
        std::size_t numKeys{};
        std::size_t numValues{};
        std::size_t maxPairs{};
        std::size_t maxValuesPerKey{};
        std::vector<Key> myKeys;
        std::vector<Value> myValues;
    };

    template<class Key, class Value, class Index>
    class CpuMultiValueHashTable{
        static_assert(std::is_integral<Key>::value, "Key must be integral!");
    public:

        struct QueryResult{
            int numValues;
            const Value* valuesBegin;
        };


        bool operator==(const CpuMultiValueHashTable& rhs) const{
            return values == rhs.values && lookup == rhs.lookup;
        }

        bool operator!=(const CpuMultiValueHashTable& rhs) const{
            return !(operator==(rhs));
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

            result.device = lookup.getMemoryInfo().device;

            //std::cerr << lookup.getMemoryInfo().host << " " << result.host << " bytes\n";

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
        private:

        using ValueIndex = std::pair<read_number, BucketSize>;
        bool isInit = false;

        // values with the same key are stored in contiguous memory locations
        // a single-value hashmap maps keys to the range of the corresponding values
        std::vector<Value> values; 
        NaiveCpuSingleValueHashTable<Key, ValueIndex> lookup;
    };

#endif

}

#endif // CARE_CPUHASHTABLE_HPP
