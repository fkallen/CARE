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
    void cpu_transformation(std::vector<Key_t>& keys,
                            std::vector<Value_t>& values,
                            std::vector<Index_t>& countsPrefixSum,
                            int /*maxValuesPerKey*/){

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

        std::cout << "Transformation done." << std::endl;
    };

}

#endif
