#ifndef CARE_MINHASHER_TRANSFORM_HPP
#define CARE_MINHASHER_TRANSFORM_HPP

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#ifdef __NVCC__
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/iterator/constant_iterator.h>

#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/sort.h>

#include <gpu/thrust_custom_allocators.hpp>
#endif




namespace care{

    template<class Key_t, class Value_t, class Index_t>
    void cpu_transformation(std::vector<Key_t>& keys,
                            std::vector<Value_t>& values,
                            std::vector<Index_t>& countsPrefixSum){

        assert(keys.size() == values.size());
        assert(std::numeric_limits<Index_t>::max() >= keys.size());

        const std::size_t size = keys.size();

        std::vector<Index_t> indices(size);
        //TIMERSTARTCPU(iota);
        std::iota(indices.begin(), indices.end(), Index_t(0));
        //TIMERSTOPCPU(iota);

        //TIMERSTARTCPU(sortindices);
        //sort indices by key. if keys are equal, sort by value
        std::sort(indices.begin(), indices.end(), [&](auto a, auto b)->bool{
            if(keys[a] == keys[b]){
                return values[a] < values[b];
            }
            return keys[a] < keys[b];
        });

        //TIMERSTOPCPU(sortindices);

        //TIMERSTARTCPU(sortvalues);
        std::vector<Value_t> sortedValues(size);
        for(Index_t i = 0; i < size; i++){
            sortedValues[i] = values[indices[i]];
        }
        //TIMERSTOPCPU(sortvalues);

        std::swap(sortedValues, values);

        { // deallocate sortedValues
            std::vector<Value_t> tmp{};
            std::swap(sortedValues, tmp);
        }

        //TIMERSTARTCPU(sortkeys);
        std::vector<Key_t> sortedKeys(size);
        for(Index_t i = 0; i < size; i++){
            sortedKeys[i] = keys[indices[i]];
        }
        //TIMERSTOPCPU(sortkeys);

        std::swap(sortedKeys, keys);

        { // deallocate sortedKeys
            std::vector<Key_t> tmp{};
            std::swap(sortedKeys, tmp);
        }

        std::vector<Index_t> counts(size, 0);

        //TIMERSTARTCPU(unique);
        //make keys unique and count frequency of each key
        Index_t unique_end = 1;
        counts[0]++;
        Key_t prev = keys[0];
        for(Index_t i = 1; i < size; i++){
            Key_t cur = keys[i];
            if(cur == prev){
                counts[unique_end-1]++;
            }else{
                keys[unique_end] = std::move(cur);
                counts[unique_end]++;
                unique_end++;
            }
            prev = cur;
        }
        //TIMERSTOPCPU(unique);
        keys.resize(unique_end);

        //TIMERSTARTCPU(prefixsum);
        //make prefix sum of counts
        countsPrefixSum.resize(unique_end+1);
        countsPrefixSum[0] = 0;
        for(Index_t i = 0; i < unique_end; i++)
            countsPrefixSum[i+1] = countsPrefixSum[i] + counts[i];
        //TIMERSTOPCPU(prefixsum);
    };

    #ifdef __NVCC__


    template<bool allowFallback>
    struct GPUTransformation{
        template<class T>
        using ThrustAlloc = ThrustFallbackDeviceAllocator<T, allowFallback>;

        template<class Key_t, class Value_t, class Index_t>
        static bool execute(std::vector<Key_t>& keys, std::vector<Value_t>& values, std::vector<Index_t>& countsPrefixSum, const std::vector<int>& deviceIds){
            assert(keys.size() == values.size());
            assert(std::numeric_limits<Index_t>::max() >= keys.size());
            assert(deviceIds.size() > 0);

            const std::size_t size = keys.size();

            int oldDeviceId;
            cudaError_t setupStatus = cudaGetDevice(&oldDeviceId);
            if(setupStatus != cudaSuccess) //we cannot recover from this.
                throw std::runtime_error("Could not query current device id!");

            setupStatus = cudaSetDevice(deviceIds.front()); //TODO choose appropriate id from deviceIds
            if(setupStatus != cudaSuccess) //we cannot recover from this.
                throw std::runtime_error("Could not set device id!");

            bool success = false;

            try{
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

                {   //deallocate d_values_tmp and d_values
                    thrust::device_vector<Value_t, ThrustAlloc<Value_t>> tmp{};
                    std::swap(d_values_tmp, tmp);
                    thrust::device_vector<Value_t, ThrustAlloc<Value_t>> tmp2{};
                    std::swap(d_values, tmp2);
                }

                //sort keys by order defined by indices
                thrust::device_vector<Key_t, ThrustAlloc<Key_t>> d_keys_tmp(size);

                thrust::copy(thrust::make_permutation_iterator(d_keys.begin(), d_indices.begin()),
                            thrust::make_permutation_iterator(d_keys.begin(), d_indices.end()),
                            d_keys_tmp.begin());

                std::swap(d_keys, d_keys_tmp);

                {   //deallocate d_keys_tmp
                    thrust::device_vector<Key_t, ThrustAlloc<Key_t>> tmp{};
                    std::swap(d_keys_tmp, tmp);
                }

                {   //deallocate d_indices
                    thrust::device_vector<Index_t, ThrustAlloc<Index_t>> tmp{};
                    std::swap(d_indices, tmp);
                }

                const Index_t nUniqueKeys = thrust::inner_product(allocatorPolicy,
                                                    d_keys.begin(),
                                                    d_keys.end() - 1,
                                                    d_keys.begin() + 1,
                                                    Index_t(1),
                                                    thrust::plus<Key_t>(),
                                                    thrust::not_equal_to<Key_t>());

                std::cout << "nkeys = " << keys.size() << ", unique keys = " << nUniqueKeys << std::endl;

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

                auto newKeysEnd = thrust::copy(d_histogram_keys.begin(),
                            d_histogram_keys.end(),
                            keys.begin());

                keys.erase(newKeysEnd, keys.end());

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
                std::cout << "Transformation done" << std::endl;
            return success;
        }

    };

    #endif

    template<class KeyValueMap>
    void transform_keyvaluemap(KeyValueMap& map, const std::vector<int>& deviceIds){
        if(map.noMoreWrites) return;

        if(map.size == 0) return;

    #ifdef __NVCC__

        if(deviceIds.size() == 0){

    #endif

            cpu_transformation(map.keys, map.values, map.countsPrefixSum);

    #ifdef __NVCC__

        }else{
            bool success = GPUTransformation<false>::execute(map.keys, map.values, map.countsPrefixSum, deviceIds);

            if(!success){
                std::cerr << "Fallback to managed memory transformation\n";
                success = GPUTransformation<true>::execute(map.keys, map.values, map.countsPrefixSum, deviceIds);
            }

            if(!success){
                std::cerr << "Fallback to cpu transformation\n";
                cpu_transformation(map.keys, map.values, map.countsPrefixSum);
            }

            map.nKeys = map.keys.size();
        }

    #endif
        map.noMoreWrites = true;


        /*keyIndexMap = KeyIndexMap(nKeys / load);
        for(Index_t i = 0; i < nKeys; i++){
            keyIndexMap.insert(keys[i], i);
        }
        for(Index_t i = 0; i < nKeys; i++){
            assert(keyIndexMap.get(keys[i]) == i);
        }*/
    }
}

#endif
