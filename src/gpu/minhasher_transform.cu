#include <minhasher.hpp>
#include <minhasher_transform.hpp>

#include <config.hpp>

#include <iostream>
#include <vector>


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/iterator/constant_iterator.h>

#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/sort.h>

#include <gpu/thrust_custom_allocators.hpp>

namespace care{

    template<bool allowFallback>
    struct MinhasherTransformGPUCompactKeys{
        template<class T>
        using ThrustAlloc = ThrustFallbackDeviceAllocator<T, allowFallback>;

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

                thrust::device_vector<Key_t, ThrustAlloc<Key_t>> d_keys(keys.size());
                thrust::copy(keys.begin(), keys.end(), d_keys.begin());

                //thrust::device_vector<Key_t, ThrustAlloc<Key_t>> d_keys_tmp(keys.size() - numKeysToRemove);
                thrust::device_vector<Key_t, ThrustAlloc<Key_t>> d_keys_tmp(numKeysToRemove);
                thrust::copy_if(d_keys.begin(),
                                d_keys.end(),
                                d_removeflags.begin(),
                                d_keys_tmp.begin(),
                                [] __device__ (auto flag){
                                    return flag;
                                });

                deallocVector(keysWithoutValues);
                keysWithoutValues.resize(numKeysToRemove);
                thrust::copy(d_keys_tmp.begin(), d_keys_tmp.end(), keysWithoutValues.begin());


                d_keys_tmp.resize(keys.size() - numKeysToRemove);
                thrust::copy_if(d_keys.begin(),
                                d_keys.end(),
                                d_removeflags.begin(),
                                d_keys_tmp.begin(),
                                [] __device__ (auto flag){
                                    return !flag;
                                });

                deallocVector(keys);
                keys.resize(d_keys_tmp.size());
                thrust::copy(d_keys_tmp.begin(), d_keys_tmp.end(), keys.begin()); 
            }

            std::cout << "Removed high frequency keys: " << numKeysToRemove << ". "; 

            //handle values
            int numValuesToRemove = 0;
            {
                thrust::device_vector<bool, ThrustAlloc<bool>> d_removeflags(values.size(), false);

                auto countsPtr = d_counts.data();
                auto countPrefixSumPtr = d_countsPrefixSum.data();
                auto flagsPtr = d_removeflags.data();

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

                thrust::device_vector<Value_t, ThrustAlloc<Value_t>> d_values(values.size());
                thrust::copy(values.begin(), values.end(), d_values.begin());                

                numValuesToRemove = thrust::count_if(allocatorPolicy,
                                                        d_removeflags.begin(),
                                                        d_removeflags.end(),
                                                        [] __device__ (auto flag){
                                                            return flag;
                                                        });

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
                thrust::device_vector<Index_t, ThrustAlloc<Index_t>> d_counts_tmp(keys.size());
                auto tmpend = thrust::copy_if(allocatorPolicy,
                                            d_counts.begin(),
                                            d_counts.end(),
                                            d_counts_tmp.begin(),
                                            [=] __device__ (auto i){
                                                return i <= maxValuesPerKey;
                                            });

                assert(d_counts_tmp.size() == thrust::distance(d_counts_tmp.begin(), tmpend));

                auto psend = thrust::inclusive_scan(allocatorPolicy,
                                        d_counts_tmp.begin(),
                                        d_counts_tmp.end(),
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

        template<class Key_t, class Value_t, class Index_t>
        static bool execute(std::vector<Key_t>& keys, 
                            std::vector<Value_t>& values, 
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
    void transform_keyvaluemap(KeyValueMap& map, const std::vector<int>& deviceIds, int maxValuesPerKey){
        if(map.noMoreWrites) return;

        if(map.size == 0) return;

        if(deviceIds.size() == 0){

            cpu_transformation(map.keys, map.values, map.countsPrefixSum, 
                                map.keysWithoutValues, maxValuesPerKey);

        }else{
            bool success = GPUTransformation<false>::execute(map.keys, map.values, map.countsPrefixSum, 
                                                            map.keysWithoutValues, deviceIds, maxValuesPerKey);

            //if(!success){
            //    std::cout << "\nFallback to managed memory transformation. ";
            //    success = GPUTransformation<true>::execute(map.keys, map.values, map.countsPrefixSum, 
            //                                                map.keysWithoutValues, deviceIds, maxValuesPerKey);
            //}

            if(!success){
                std::cout << "\nFallback to cpu transformation. ";
                cpu_transformation(map.keys, map.values, map.countsPrefixSum, 
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

        //std::cerr << "maxProbes = " << map.keyIndexMap.maxProbes << "\n";

        {
            std::vector<Key_t> tmp;
            map.keys.swap(tmp);
        }
        /*for(Index_t i = 0; i < nKeys; i++){
            assert(keyIndexMap.get(keys[i]) == i);
        }*/
    }

    void transform_minhasher_gpu(Minhasher& minhasher, int map, const std::vector<int>& deviceIds){
        assert(map < int(minhasher.minhashTables.size()));

        auto& tableptr = minhasher.minhashTables[map];
        int maxValuesPerKey = minhasher.getResultsPerMapThreshold();
        if(!tableptr->noMoreWrites){
            std::cerr << "Transforming table " << map << ". ";
            transform_keyvaluemap(*tableptr, deviceIds, maxValuesPerKey);
        }
    }

    void transform_minhasher_gpu(Minhasher& minhasher, const std::vector<int>& deviceIds){
        for (std::size_t i = 0; i < minhasher.minhashTables.size(); ++i){
            transform_minhasher_gpu(minhasher, i, deviceIds);
        }
    }

}
