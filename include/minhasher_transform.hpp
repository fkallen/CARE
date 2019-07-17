#ifndef CARE_MINHASHER_TRANSFORM_HPP
#define CARE_MINHASHER_TRANSFORM_HPP

#include <minhasher.hpp>
#include <config.hpp>

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

    void transform_minhasher(Minhasher& minhasher, const std::vector<int>& deviceIds);
    void transform_minhasher(Minhasher& minhasher, int map, const std::vector<int>& deviceIds);


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

}

#endif
