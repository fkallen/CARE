#ifndef CARE_MINHASHER_TRANSFORM_HPP
#define CARE_MINHASHER_TRANSFORM_HPP

#include <minhasher.hpp>
#include <config.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include <thrust/system/omp/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/inner_product.h>
#include <thrust/sequence.h>

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

            deallocVector(keysWithoutValues);
            keysWithoutValues.resize(numKeysToRemove);

            thrust::copy_if(keys.begin(),
                            keys.end(),
                            removeflags.begin(),
                            keysWithoutValues.begin(),
                            [](auto flag){
                                return flag == 1;
                            });

            std::vector<Key_t> keys_tmp(keys.size() - numKeysToRemove);
            thrust::copy_if(keys.begin(),
                            keys.end(),
                            removeflags.begin(),
                            keys_tmp.begin(),
                            [](auto flag){
                                return flag == 0;
                            });
            keys.swap(keys_tmp);            
        }

        std::cout << "Removed high frequency keys: " << numKeysToRemove << ". "; 

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
            std::vector<Index_t> counts_tmp(keys.size());
            thrust::copy_if(
                            counts.begin(),
                            counts.end(),
                            counts_tmp.begin(),
                            [=](auto i){
                                return i <= maxValuesPerKey;
                            });

            deallocVector(counts);
            deallocVector(countsPrefixSum);

            countsPrefixSum.resize(keys.size() + 1);
            countsPrefixSum[0] = 0;

            thrust::inclusive_scan(policy,
                                    counts_tmp.begin(),
                                    counts_tmp.end(),
                                    countsPrefixSum.begin() + 1);              
        }
    }

    template<class Key_t, class Value_t, class Index_t>
    void cpu_transformation(std::vector<Key_t>& keys,
                            std::vector<Value_t>& values,
                            std::vector<Index_t>& countsPrefixSum,
                            std::vector<Key_t>& keysWithoutValues,
                            int maxValuesPerKey){

        minhasherTransformCPUCompactKeys(keys, values, countsPrefixSum);

        minhasherTransformCPURemoveKeysWithToManyValues(keys, values, countsPrefixSum, keysWithoutValues, maxValuesPerKey);

        std::cout << "Transformation done." << std::endl;
    };
}

#endif
