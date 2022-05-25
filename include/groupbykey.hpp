#ifndef CARE_GROUP_BY_KEY_HPP
#define CARE_GROUP_BY_KEY_HPP

#include <hpc_helpers.cuh>
#include <gpu/cudaerrorcheck.cuh>

#include <thrust/system/omp/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/inner_product.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/equal.h>

#ifdef __NVCC__
#include <thrust/device_vector.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#endif

#include <vector>
#include <cassert>
#include <cstddef>
#include <iostream>



namespace care{

    template<class Key_t, class Value_t, class Offset_t>
    struct GroupByKeyCpu{
        bool valuesOfSameKeyMustBeSorted = false;
        int maxValuesPerKey = 0;
        int minValuesPerKey = 0;

        GroupByKeyCpu(bool sortValues, int maxValuesPerKey_, int minValuesPerKey_) 
            : valuesOfSameKeyMustBeSorted(sortValues), 
                maxValuesPerKey(maxValuesPerKey_),
                minValuesPerKey(minValuesPerKey_){}

        /*
            Input: keys and values. keys[i] and values[i] form a key-value pair
            Output: unique keys. values with the same key are stored consecutive. values of unique_keys[i] are
            stored at values[offsets[i]] to values[offsets[i+1]] (exclusive)
            If valuesOfSameKeyMustBeSorted == true, values with the same key are sorted in ascending order.
            If there are more than maxValuesPerKey values with the same key, all of those values are removed,
            i.e. the key ends up with 0 values
        */
        void execute(std::vector<Key_t>& keys, std::vector<Value_t>& values, std::vector<Offset_t>& offsets){
            if(keys.size() == 0){
                //deallocate unused memory if capacity > 0
                keys = std::vector<Key_t>{};
                values = std::vector<Value_t>{};
                return;
            }

            if(valuesOfSameKeyMustBeSorted){
                const bool isIotaValues = checkIotaValues(values);
                if(!isIotaValues)
                    throw std::runtime_error("Error hashtable compaction");
            }

            executeWithIotaValues(keys, values, offsets);
        }

        bool checkIotaValues(const std::vector<Value_t>& values){
            auto policy = thrust::host;

            bool isIotaValues = thrust::equal(
                policy,
                thrust::counting_iterator<Value_t>{0},
                thrust::counting_iterator<Value_t>{0} + values.size(),
                values.data()
            ); 

            return isIotaValues;
        }

        void executeWithIotaValues(std::vector<Key_t>& keys, std::vector<Value_t>& values, std::vector<Offset_t>& offsets){
            assert(keys.size() == values.size()); //key value pairs
            assert(std::numeric_limits<Offset_t>::max() >= keys.size()); //total number of keys must fit into Offset_t

            auto deallocVector = [](auto& vec){
                using T = typename std::remove_reference<decltype(vec)>::type;
                T tmp{};
                vec.swap(tmp);
            };

            deallocVector(offsets); //don't need offsets at the moment

            const std::size_t size = keys.size();
            auto policy = thrust::omp::par;

            if(valuesOfSameKeyMustBeSorted){
                auto kb = keys.data();
                auto ke = keys.data() + size;
                auto vb = values.data();

                thrust::stable_sort_by_key(policy, kb, ke, vb);
            }else{
                auto kb = keys.data();
                auto ke = keys.data() + size;
                auto vb = values.data();

                thrust::sort_by_key(policy, kb, ke, vb);
            }

            const Offset_t nUniqueKeys = thrust::inner_product(policy,
                                            keys.begin(),
                                            keys.end() - 1,
                                            keys.begin() + 1,
                                            Offset_t(1),
                                            thrust::plus<Key_t>(),
                                            thrust::not_equal_to<Key_t>());

            //std::cout << "unique keys " << nUniqueKeys << ". ";

            std::vector<Key_t> uniqueKeys(nUniqueKeys);
            std::vector<Offset_t> valuesPerKey(nUniqueKeys);

            auto* keys_begin = keys.data();
            auto* keys_end = keys.data() + size;
            auto* uniqueKeys_begin = uniqueKeys.data();
            auto* valuesPerKey_begin = valuesPerKey.data();

            //make histogram
            auto histogramEndIterators = thrust::reduce_by_key(
                policy,
                keys_begin,
                keys_end,
                thrust::constant_iterator<Offset_t>(1),
                uniqueKeys_begin,
                valuesPerKey_begin
            );

            if(histogramEndIterators.first != uniqueKeys.data() + nUniqueKeys) 
                throw std::runtime_error("Error hashtable compaction");
            if(histogramEndIterators.second != valuesPerKey.data() + nUniqueKeys) 
                throw std::runtime_error("Error hashtable compaction");

            keys.swap(uniqueKeys);
            deallocVector(uniqueKeys);

            offsets.resize(nUniqueKeys+1);
            offsets[0] = 0;

            thrust::inclusive_scan(
                policy,
                valuesPerKey_begin,
                valuesPerKey_begin + nUniqueKeys,
                offsets.data() + 1
            );

            std::vector<char> removeflags(values.size(), false);

            thrust::for_each(
                policy,
                thrust::counting_iterator<Offset_t>(0),
                thrust::counting_iterator<Offset_t>(0) + nUniqueKeys,
                [&](Offset_t index){
                    const auto begin = offsets[index];
                    const auto end = offsets[index+1];
                    const auto num = end - begin;

                    if(num > Offset_t(maxValuesPerKey) || num < Offset_t(minValuesPerKey)){
                        valuesPerKey_begin[index] = 0;

                        for(Offset_t k = begin; k < end; k++){
                            removeflags[k] = true;
                        }
                    }
                }
            );

            deallocVector(offsets);    

            Offset_t numValuesToRemove = thrust::reduce(
                policy,
                removeflags.begin(),
                removeflags.end(),
                Offset_t(0)
            );

            std::vector<Value_t> values_tmp(size - numValuesToRemove);

            thrust::copy_if(
                values.begin(),
                values.end(),
                removeflags.begin(),
                values_tmp.begin(),
                [](auto flag){
                    return flag == 0;
                }
            );

            values.swap(values_tmp);
            deallocVector(values_tmp);

            offsets.resize(nUniqueKeys+1);
            offsets[0] = 0;

            thrust::inclusive_scan(
                policy,
                valuesPerKey_begin,
                valuesPerKey_begin + nUniqueKeys,
                offsets.data() + 1
            );
        }
    };

    #ifdef __NVCC__

    template<class Key_t, class Value_t, class Offset_t>
    struct GroupByKeyGpu{
        //gpu allocator which uses cudaMallocManaged if not enough gpu memory is available for cudaMalloc
        template<class T>
        using ThrustAlloc = helpers::ThrustFallbackDeviceAllocator<T, true>;


        bool valuesOfSameKeyMustBeSorted = false;
        int maxValuesPerKey = 0;
        int minValuesPerKey = 0;

        GroupByKeyGpu(bool sortValues, int maxValuesPerKey_, int minValuesPerKey_) 
            : valuesOfSameKeyMustBeSorted(sortValues), 
                maxValuesPerKey(maxValuesPerKey_),
                minValuesPerKey(minValuesPerKey_){}

        /*
            Input: keys and values. keys[i] and values[i] form a key-value pair
            Output: unique keys. values with the same key are stored consecutive. values of unique_keys[i] are
            stored at values[offsets[i]] to values[offsets[i+1]] (exclusive)
            If valuesOfSameKeyMustBeSorted == true, values with the same key are sorted in ascending order.
            If there are more than maxValuesPerKey values with the same key, all of those values are removed,
            i.e. the key ends up with 0 values
        */
        bool execute(std::vector<Key_t>& keys, std::vector<Value_t>& values, std::vector<Offset_t>& offsets){
            if(keys.size() == 0){
                //deallocate unused memory if capacity > 0
                keys = std::vector<Key_t>{};
                values = std::vector<Value_t>{};
                return true;
            }

            bool success = false;

            if(valuesOfSameKeyMustBeSorted){
                bool isIotaValues = checkIotaValues(values);
                assert(isIotaValues);
            }

            //if(isIotaValues){                   
                try{           
                    executeWithIotaValues(keys, values, offsets);
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
            //}else{
            //    assert(false && "not implemented");
            //}

            return success;
        }

        bool checkIotaValues(const std::vector<Value_t>& values){
            auto policy = thrust::host;

            nvtx::push_range("checkIotaValues", 6);

            bool isIotaValues = thrust::equal(
                policy,
                thrust::counting_iterator<Value_t>{0},
                thrust::counting_iterator<Value_t>{0} + values.size(),
                values.data()
            ); 

            nvtx::pop_range();

            return isIotaValues;
        }

        void executeWithIotaValues(std::vector<Key_t>& keys, std::vector<Value_t>& values, std::vector<Offset_t>& offsets){
            assert(keys.size() == values.size()); //key value pairs
            assert(std::numeric_limits<Offset_t>::max() >= keys.size()); //total number of keys must fit into Offset_t

            auto deallocVector = [](auto& vec){
                using T = typename std::remove_reference<decltype(vec)>::type;
                T tmp{};
                vec.swap(tmp);
            };

            deallocVector(offsets); //don't need offsets at the moment

            const std::size_t size = keys.size();

            ThrustAlloc<char> allocator;
            auto allocatorPolicy = thrust::cuda::par(allocator);

            thrust::device_vector<Key_t, ThrustAlloc<Key_t>> d_keys(size);
            thrust::device_vector<Value_t, ThrustAlloc<Value_t>> d_values(size);

            thrust::copy(keys.begin(), keys.end(), d_keys.begin());
            thrust::copy(values.begin(), values.end(), d_values.begin());

            if(valuesOfSameKeyMustBeSorted){
                thrust::stable_sort_by_key(allocatorPolicy, d_keys.begin(), d_keys.end(), d_values.begin());
            }else{
                auto kb = keys.data();
                auto ke = keys.data() + size;
                auto vb = values.data();

                thrust::sort_by_key(allocatorPolicy, d_keys.begin(), d_keys.end(), d_values.begin());
            }

            thrust::copy(d_values.begin(), d_values.end(), values.begin());
            deallocVector(d_values);

            const Offset_t nUniqueKeys = thrust::inner_product(
                allocatorPolicy,
                d_keys.begin(),
                d_keys.end() - 1,
                d_keys.begin() + 1,
                Offset_t(1),
                thrust::plus<Key_t>(),
                thrust::not_equal_to<Key_t>()
            );

            //std::cout << "unique keys " << nUniqueKeys << ". ";

            thrust::device_vector<Key_t, ThrustAlloc<Key_t>> d_uniqueKeys(nUniqueKeys);
            thrust::device_vector<Offset_t, ThrustAlloc<Offset_t>> d_valuesPerKey(nUniqueKeys);

            //make histogram
            auto histogramEndIterators = thrust::reduce_by_key(
                allocatorPolicy,
                d_keys.begin(),
                d_keys.end(),
                thrust::constant_iterator<Offset_t>(1),
                d_uniqueKeys.begin(),
                d_valuesPerKey.begin()
            );

            if(histogramEndIterators.first != d_uniqueKeys.begin() + nUniqueKeys) 
                throw std::runtime_error("Error hashtable compaction");
            if(histogramEndIterators.second != d_valuesPerKey.begin() + nUniqueKeys) 
                throw std::runtime_error("Error hashtable compaction");

            deallocVector(keys);
            keys.resize(nUniqueKeys);
            thrust::copy(d_uniqueKeys.begin(), d_uniqueKeys.end(), keys.begin());
            deallocVector(d_keys);
            deallocVector(d_uniqueKeys);

            thrust::device_vector<Offset_t, ThrustAlloc<Offset_t>> d_offsets(nUniqueKeys + 1, 0);

            thrust::inclusive_scan(
                allocatorPolicy,
                d_valuesPerKey.begin(),
                d_valuesPerKey.end(),
                d_offsets.begin() + 1
            );

            thrust::device_vector<char, ThrustAlloc<char>> d_removeflags(size, false);
            auto d_removeflags_begin = d_removeflags.data();
            auto d_offsets_begin = d_offsets.data();
            auto d_valuesPerKey_begin = d_valuesPerKey.data();
            auto maxValuesPerKey_copy = maxValuesPerKey;
            auto minValuesPerKey_copy = minValuesPerKey;
            thrust::for_each(
                allocatorPolicy,
                thrust::counting_iterator<Offset_t>(0),
                thrust::counting_iterator<Offset_t>(0) + nUniqueKeys,
                [=] __device__ (Offset_t index){
                    const auto begin = d_offsets_begin[index];
                    const auto end = d_offsets_begin[index+1];
                    const auto num = end - begin;

                    if(num > Offset_t(maxValuesPerKey_copy) || num < Offset_t(minValuesPerKey_copy)){
                        d_valuesPerKey_begin[index] = 0;

                        for(Offset_t k = begin; k < end; k++){
                            d_removeflags_begin[k] = true;
                        }
                    }
                }
            );

            deallocVector(d_offsets);    

            Offset_t numValuesToRemove = thrust::reduce(
                allocatorPolicy,
                d_removeflags.begin(),
                d_removeflags.end(),
                Offset_t(0)
            );

            thrust::device_vector<Value_t, ThrustAlloc<Value_t>> d_values_tmp(size - numValuesToRemove);
            d_values.resize(values.size());
            thrust::copy(values.begin(), values.end(), d_values.begin());

            thrust::copy_if(
                d_values.begin(),
                d_values.end(),
                d_removeflags.begin(),
                d_values_tmp.begin(),
                [] __device__ (auto flag){
                    return flag == 0;
                }
            );

            deallocVector(d_removeflags);
            deallocVector(values);
            deallocVector(d_values);
            values.resize(size - numValuesToRemove);
            thrust::copy(d_values_tmp.begin(), d_values_tmp.end(), values.begin());

            offsets.resize(nUniqueKeys+1);               
            d_offsets.resize(nUniqueKeys);

            thrust::inclusive_scan(
                allocatorPolicy,
                d_valuesPerKey.begin(),
                d_valuesPerKey.end(),
                d_offsets.begin()
            );

            thrust::copy(d_offsets.begin(), d_offsets.end(), offsets.begin() + 1);
            offsets[0] = 0;
        }
    };

    #endif //#ifdef __NVCC__
}


#endif