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

                // auto largercount = thrust::count_if(d_histogram_counts.begin(), d_histogram_counts.end(), []__device__(auto i){
                //     return i > 100;
                // });
                //
                // auto affectedlargercountvalues = thrust::transform_reduce(d_histogram_counts.begin(),
                //                                                             d_histogram_counts.end(),
                //                                                             []__device__(auto i){
                //                                                                 return i <= 100 ? 0 : i;
                //                                                             },
                //                                                             0,
                //                                                             thrust::plus<int>{});
                //
                // std::cerr << "largercount " << largercount << std::endl;
                // std::cerr << "affectedlargercountvalues " << affectedlargercountvalues << std::endl;

                countsPrefixSum.resize(nUniqueKeys+1);
                thrust::copy(d_histogram_counts_prefixsum.begin(),
                            d_histogram_counts_prefixsum.end(),
                            countsPrefixSum.begin());

                {
                    std::vector<Key_t> tmp;
                    keys.swap(tmp);
                }

                keys.resize(nUniqueKeys);

                //auto newKeysEnd =
                thrust::copy(d_histogram_keys.begin(),
                            d_histogram_keys.end(),
                            keys.begin());

                //keys.erase(newKeysEnd, keys.end());

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

    template<class KeyValueMap>
    void transform_keyvaluemap(KeyValueMap& map, const std::vector<int>& deviceIds){
        if(map.noMoreWrites) return;

        if(map.size == 0) return;

        if(deviceIds.size() == 0){

            cpu_transformation(map.keys, map.values, map.countsPrefixSum);

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
        }

        map.nKeys = map.keys.size();
        map.noMoreWrites = true;

        using Key_t = typename KeyValueMap::Key_t;
        using Index_t = typename KeyValueMap::Index_t;


        map.keyIndexMap = minhasherdetail::KeyIndexMap<Key_t, Index_t>(map.nKeys / map.load);
        for(Index_t i = 0; i < map.nKeys; i++){
            map.keyIndexMap.insert(map.keys[i], i);
        }
        /*for(Index_t i = 0; i < nKeys; i++){
            assert(keyIndexMap.get(keys[i]) == i);
        }*/
    }

    void transform_minhasher_gpu(Minhasher& minhasher, int map, const std::vector<int>& deviceIds){
        assert(map < int(minhasher.minhashTables.size()));

        auto& tableptr = minhasher.minhashTables[map];
        if(!tableptr->noMoreWrites){
            std::cerr << "Transforming table " << map << std::endl;
            transform_keyvaluemap(*tableptr, deviceIds);
        }
    }

    void transform_minhasher_gpu(Minhasher& minhasher, const std::vector<int>& deviceIds){
        for (std::size_t i = 0; i < minhasher.minhashTables.size(); ++i){
            transform_minhasher_gpu(minhasher, i, deviceIds);
        }
    }

}
