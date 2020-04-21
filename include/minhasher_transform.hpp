#ifndef CARE_MINHASHER_TRANSFORM_HPP
#define CARE_MINHASHER_TRANSFORM_HPP

#include <minhasher.hpp>
#include <config.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <omp.h>

#include <thrust/system/omp/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/inner_product.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#ifdef __NVCC__
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <gpu/thrust_custom_allocators.hpp>
#endif

namespace care{

    void transform_minhasher(Minhasher& minhasher);
    void transform_minhasher(Minhasher& minhasher, int map);

#ifdef __NVCC__
    void transform_minhasher_gpu(Minhasher& minhasher, const std::vector<int>& deviceIds);
    void transform_minhasher_gpu(Minhasher& minhasher, int map, const std::vector<int>& deviceIds);
#endif

    template<class Key_t, class Value_t, class Index_t>
    void minhasherTransformCPUCompactKeys(std::vector<Key_t>& keys,
                                        std::vector<Value_t>& values,
                                        std::vector<Index_t>& countsPrefixSum){

        assert(keys.size() == values.size());
        assert(std::numeric_limits<Index_t>::max() >= keys.size());

        const std::size_t size = keys.size();
        auto policy = thrust::omp::par;

        std::vector<Index_t> indices(size);
        //TIMERSTARTCPU(iota);
        thrust::sequence(policy, indices.begin(), indices.end(), Index_t(0));
        //TIMERSTOPCPU(iota);

        //TIMERSTARTCPU(sortindices);
        //sort indices by key. if keys are equal, sort by value
        thrust::sort(policy,
                    indices.begin(),
                    indices.end(),
                    [&] (const auto &lhs, const auto &rhs) {
                        if(keys[lhs] == keys[rhs]){
                            return values[lhs] < values[rhs];
                        }
                        return keys[lhs] < keys[rhs];
                    });

        //TIMERSTOPCPU(sortindices);

        //TIMERSTARTCPU(sortvalues);
        std::vector<Value_t> sortedValues(size);
        thrust::copy(policy,
                    thrust::make_permutation_iterator(values.begin(), indices.begin()),
                    thrust::make_permutation_iterator(values.begin(), indices.end()),
                    sortedValues.begin());

        //TIMERSTOPCPU(sortvalues);

        std::swap(sortedValues, values);

        { // deallocate sortedValues
            std::vector<Value_t> tmp{};
            std::swap(sortedValues, tmp);
        }

        //TIMERSTARTCPU(sortkeys);
        std::vector<Key_t> sortedKeys(size);
        thrust::copy(policy,
                    thrust::make_permutation_iterator(keys.begin(), indices.begin()),
                    thrust::make_permutation_iterator(keys.begin(), indices.end()),
                    sortedKeys.begin());

        //TIMERSTOPCPU(sortkeys);

        std::swap(sortedKeys, keys);

        { // deallocate sortedKeys
            std::vector<Key_t> tmp{};
            std::swap(sortedKeys, tmp);
        }

        const Index_t nUniqueKeys = thrust::inner_product(policy,
                                            keys.begin(),
                                            keys.end() - 1,
                                            keys.begin() + 1,
                                            Index_t(1),
                                            thrust::plus<Key_t>(),
                                            thrust::not_equal_to<Key_t>());

        std::cout << "unique keys " << nUniqueKeys << ". ";

        std::vector<Key_t> histogram_keys(nUniqueKeys);
        std::vector<Index_t> histogram_counts(nUniqueKeys);

        //make key - frequency histogram
        auto histogramEndIterators = thrust::reduce_by_key(policy,
                                keys.begin(),
                                keys.end(),
                                thrust::constant_iterator<Index_t>(1),
                                histogram_keys.begin(),
                                histogram_counts.begin());

        assert(histogramEndIterators.first == histogram_keys.end());
        assert(histogramEndIterators.second == histogram_counts.end());

        countsPrefixSum.resize(nUniqueKeys+1, Index_t(0));

        thrust::inclusive_scan(policy,
                                histogram_counts.begin(),
                                histogramEndIterators.second,
                                countsPrefixSum.begin() + 1);

        keys.swap(histogram_keys);

    }

    template<class Key_t, class Value_t, class Index_t>
    void minhasherTransformCPURemoveKeysWithToManyValues(std::vector<Key_t>& keys, 
                                                        std::vector<Value_t>& values, 
                                                        std::vector<Index_t>& countsPrefixSum,
                                                        std::vector<Key_t>& keysWithoutValues,
                                                        int maxValuesPerKey_){
        auto deallocVector = [](auto& vec){
            using T = typename std::remove_reference<decltype(vec)>::type;
            T tmp{};
            vec.swap(tmp);
        };

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

        std::cout << "Can remove values of " << numKeysToRemove << " high frequency keys. "; 

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

        std::cout << "Removed corresponding values: " << numValuesToRemove << ". "; 

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
    }

    template<class Key_t, class Value_t, class Index_t, class Count_t>
    void cpu_transformation(std::vector<Key_t>& keys,
                            std::vector<Value_t>& values,
                            std::vector<Count_t>& counts,
                            std::vector<Index_t>& countsPrefixSum,
                            std::vector<Key_t>& keysWithoutValues,
                            int maxValuesPerKey){

        // int oldNumOmpThreads = 0;
        // #pragma omp parallel
        // {
        //     #pragma omp master
        //     oldNumOmpThreads = omp_get_num_threads();
        // }

        //omp_set_num_threads(numThreads);

        minhasherTransformCPUCompactKeys(keys, values, countsPrefixSum);

        minhasherTransformCPURemoveKeysWithToManyValues(keys, values, countsPrefixSum, keysWithoutValues, maxValuesPerKey);

        std::cout << "Transformation done." << std::endl;

        //omp_set_num_threads(oldNumOmpThreads);
    };

    template<class KeyValueMap>
    void transform_keyvaluemap(KeyValueMap& map, int maxValuesPerKey){
        if(map.noMoreWrites) return;

        if(map.size == 0) return;

        cpu_transformation(map.keys, map.values, 
                            map.counts, map.countsPrefixSum, 
                            map.keysWithoutValues, maxValuesPerKey);

        map.nKeys = map.keys.size();
        map.nValues = map.values.size();
        map.noMoreWrites = true;

        using Key_t = typename KeyValueMap::Key_t;
        using Index_t = typename KeyValueMap::Index_t;

        map.keyIndexMap = minhasherdetail::KeyIndexMap<Key_t, Index_t>(map.nKeys / map.load);
        for(Index_t i = 0; i < map.nKeys; i++){
            map.keyIndexMap.insert(map.keys[i], i);
        }
        map.keyToIndexLengthPairMap = minhasherdetail::KeyToIndexLengthPairMap<Key_t, Index_t>(map.nKeys / map.load);
        for(Index_t i = 0; i < map.nKeys; i++){
            map.keyToIndexLengthPairMap.insert(map.keys[i], map.countsPrefixSum[i], map.countsPrefixSum[i+1] - map.countsPrefixSum[i]);
        }

        {
            std::vector<Key_t> tmp;
            map.keys.swap(tmp);
        }

        // {
        //     std::vector<Index_t> tmp;
        //     map.countsPrefixSum.swap(tmp);
        // }

        /*keyIndexMap = KeyIndexMap(nKeys / load);
        for(Index_t i = 0; i < nKeys; i++){
            keyIndexMap.insert(keys[i], i);
        }
        for(Index_t i = 0; i < nKeys; i++){
            assert(keyIndexMap.get(keys[i]) == i);
        }*/
    }

#ifdef __NVCC__

    template<bool allowFallback>
    struct MinhasherTransformGPUCompactKeys{
        template<class T>
        using ThrustAlloc = ThrustFallbackDeviceAllocator<T, allowFallback>;

        template<class Key_t, class Value_t, class Index_t>
        static std::size_t estimateRequiredGpuMem(
                            std::vector<Key_t>& keys, 
                            std::vector<Value_t>& values, 
                            std::vector<Index_t>& countsPrefixSum){
            
            return estimateRequiredGpuMem<Key_t, Value_t, Index_t>(values.size());
        }

        template<class Key_t, class Value_t, class Index_t>
        static std::size_t estimateRequiredGpuMem(Index_t numEntries){

            std::size_t mem = 0;
            mem += sizeof(Key_t) * numEntries; //d_keys
            mem += sizeof(Value_t) * numEntries; //d_values
            mem += sizeof(Index_t) * numEntries; //d_indices
            mem += std::max(sizeof(Index_t), sizeof(Value_t)) * numEntries; //d_indices_tmp for sorting d_indices or d_values_tmp for sorted values

            return mem;
        }

        template<class Key_t, class Value_t, class Index_t>
        static void execute(std::vector<Key_t>& keys, 
                            std::vector<Value_t>& values, 
                            std::vector<Index_t>& countsPrefixSum,
                            const std::vector<int>& /*deviceIds*/){

            auto deallocVector = [](auto& vec){
                using T = typename std::remove_reference<decltype(vec)>::type;
                T tmp{};
                vec.swap(tmp);
            };

            std::size_t size = values.size();

            ThrustAlloc<char> allocator;
            auto allocatorPolicy = thrust::cuda::par(allocator);

            thrust::device_vector<Key_t, ThrustAlloc<Key_t>> d_keys(size);
            thrust::device_vector<Value_t, ThrustAlloc<Value_t>> d_values(size);
            thrust::device_vector<Index_t, ThrustAlloc<Index_t>> d_indices(size);

            thrust::copy(keys.begin(), keys.end(), d_keys.begin());
            thrust::copy(values.begin(), values.end(), d_values.begin());
            thrust::sequence(allocatorPolicy, d_indices.begin(), d_indices.end(), Index_t(0));

            thrust::device_ptr<Key_t> d_keys_ptr = d_keys.data();
            thrust::device_ptr<Value_t> d_values_ptr = d_values.data();

            //sort indices
            thrust::sort(allocatorPolicy,
                        d_indices.begin(),
                        d_indices.end(),
                        [=] __device__ (const auto &lhs, const auto &rhs) {
                            if(d_keys_ptr[lhs] == d_keys_ptr[rhs]){
                                return d_values_ptr[lhs] < d_values_ptr[rhs];
                            }
                            return d_keys_ptr[lhs] < d_keys_ptr[rhs];
                        });

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

            std::cout << "unique keys " << nUniqueKeys << ". ";

            //histogram storage
            thrust::device_vector<Key_t, ThrustAlloc<Key_t>> d_histogram_keys(nUniqueKeys);
            thrust::device_vector<Index_t, ThrustAlloc<Index_t>> d_histogram_counts(nUniqueKeys);
            thrust::device_vector<Index_t, ThrustAlloc<Index_t>> d_histogram_counts_prefixsum(nUniqueKeys+1, Index_t(0));

            //make key multiplicity histogram
            auto histogramEndIterators = thrust::reduce_by_key(allocatorPolicy,
                d_keys.begin(),
                d_keys.end(),
                thrust::constant_iterator<Index_t>(1),
                d_histogram_keys.begin(),
                d_histogram_counts.begin());

            assert(histogramEndIterators.first == d_histogram_keys.end());
            assert(histogramEndIterators.second == d_histogram_counts.end());

            thrust::inclusive_scan(allocatorPolicy,
                d_histogram_counts.begin(),
                histogramEndIterators.second,
                d_histogram_counts_prefixsum.begin() + 1);

            countsPrefixSum.resize(nUniqueKeys+1);
            thrust::copy(d_histogram_counts_prefixsum.begin(),
                        d_histogram_counts_prefixsum.end(),
                        countsPrefixSum.begin());

            deallocVector(keys);

            keys.resize(nUniqueKeys);

            thrust::copy(d_histogram_keys.begin(),
                        d_histogram_keys.end(),
                        keys.begin());
        }
    };


    template<bool allowFallback>
    struct MinhasherTransformGPURemoveKeysWithToManyValues{
        template<class T>
        using ThrustAlloc = ThrustFallbackDeviceAllocator<T, allowFallback>;

        template<class Key_t, class Value_t, class Index_t>
        static void execute(std::vector<Key_t>& keys, 
                            std::vector<Value_t>& values, 
                            std::vector<Index_t>& countsPrefixSum,
                            std::vector<Key_t>& keysWithoutValues,
                            int maxValuesPerKey_,
                            const std::vector<int>& /*deviceIds*/){

            auto deallocVector = [](auto& vec){
                using T = typename std::remove_reference<decltype(vec)>::type;
                T tmp{};
                vec.swap(tmp);
            };

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

            std::cout << "Can remove values of " << numKeysToRemove << " high frequency keys. "; 

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

            std::cout << "Removed corresponding values: " << numValuesToRemove << ". "; 

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
        }
    };

    template<bool allowFallback>
    struct GPUTransformation{
        template<class T>
        using ThrustAlloc = ThrustFallbackDeviceAllocator<T, allowFallback>;

        template<class Key_t, class Value_t, class Index_t, class Count_t>
        static bool execute(std::vector<Key_t>& keys, 
                            std::vector<Value_t>& values, 
                            std::vector<Count_t>& counts,
                            std::vector<Index_t>& countsPrefixSum, 
                            std::vector<Key_t>& keysWithoutValues,
                            const std::vector<int>& deviceIds,
                            int maxValuesPerKey){

            assert(keys.size() == values.size());
            assert(std::numeric_limits<Index_t>::max() >= keys.size());
            assert(deviceIds.size() > 0);

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

            bool success = false;

            try{           
                MinhasherTransformGPUCompactKeys<allowFallback>
                        ::execute(keys, values, countsPrefixSum, deviceIds);

                MinhasherTransformGPURemoveKeysWithToManyValues<allowFallback>
                        ::execute(keys, values, countsPrefixSum, keysWithoutValues, maxValuesPerKey, deviceIds);                

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

            if(success)
                std::cout << "Transformation done." << std::endl;
            return success;
        }

    };

    template<class KeyValueMap>
    std::size_t estimateGpuMemoryForTransformKeyValueMap(KeyValueMap& map){
        return MinhasherTransformGPUCompactKeys<true>::estimateRequiredGpuMem(map.keys, map.values, map.countsPrefixSum);
    }

    template<class KeyValueMap>
    void transform_keyvaluemap_gpu(KeyValueMap& map, const std::vector<int>& deviceIds, int maxValuesPerKey){
        if(map.noMoreWrites) return;

        if(map.size == 0) return;

        if(deviceIds.size() == 0){

            cpu_transformation(map.keys, map.values, 
                                map.counts, map.countsPrefixSum, 
                                map.keysWithoutValues, maxValuesPerKey);

        }else{
            bool success = GPUTransformation<false>::execute(map.keys, map.values, 
                                                            map.counts, map.countsPrefixSum, 
                                                            map.keysWithoutValues, deviceIds, maxValuesPerKey);

            if(!success){
                std::cout << "\nFallback to managed memory transformation. ";
                success = GPUTransformation<true>::execute(map.keys, map.values, 
                                                            map.counts, map.countsPrefixSum, 
                                                            map.keysWithoutValues, deviceIds, maxValuesPerKey);
            }

            if(!success){
                std::cout << "\nFallback to cpu transformation. ";
                std::cout.flush();
                cpu_transformation(map.keys, map.values, 
                                map.counts, map.countsPrefixSum, 
                                map.keysWithoutValues, maxValuesPerKey);
            }
        }

        map.nKeys = map.keys.size();
        map.nValues = map.values.size();
        map.noMoreWrites = true;

        using Key_t = typename KeyValueMap::Key_t;
        using Index_t = typename KeyValueMap::Index_t;


        map.keyIndexMap = minhasherdetail::KeyIndexMap<Key_t, Index_t>(map.nKeys / map.load);
        for(Index_t i = 0; i < map.nKeys; i++){
            map.keyIndexMap.insert(map.keys[i], i);
        }
        map.keyToIndexLengthPairMap = minhasherdetail::KeyToIndexLengthPairMap<Key_t, Index_t>(map.nKeys / map.load);
        for(Index_t i = 0; i < map.nKeys; i++){
            map.keyToIndexLengthPairMap.insert(map.keys[i], map.countsPrefixSum[i], map.countsPrefixSum[i+1] - map.countsPrefixSum[i]);
        }

        {
            std::vector<Key_t> tmp;
            map.keys.swap(tmp);
        }

        /*for(Index_t i = 0; i < nKeys; i++){
            assert(keyIndexMap.get(keys[i]) == i);
        }*/
    }

#endif
}

#endif
