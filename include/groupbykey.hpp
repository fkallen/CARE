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
#include <gpu/gpuhashtable.cuh>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#endif


#ifdef CARE_HAS_WARPCORE

#include <warpcore/multi_value_hash_table.cuh>

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
                bool isIotaValues = checkIotaValues(values);
                assert(isIotaValues);
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

            assert(histogramEndIterators.first == uniqueKeys.data() + nUniqueKeys);
            assert(histogramEndIterators.second == valuesPerKey.data() + nUniqueKeys);

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

            assert(histogramEndIterators.first == d_uniqueKeys.begin() + nUniqueKeys);
            assert(histogramEndIterators.second == d_valuesPerKey.begin() + nUniqueKeys);

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
     
    #ifdef CARE_HAS_WARPCORE


        template<class Key_t, class Value_t, class Offset_t>
        struct GroupByKeyGpuWarpcore{

            int maxValuesPerKey = 0;
            int minValuesPerKey = 0;

            GroupByKeyGpuWarpcore(int maxValuesPerKey_, int minValuesPerKey_)
                : maxValuesPerKey(maxValuesPerKey_),
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
                if(keys.size() == 0) return true;

                bool success = false;

                // if(valuesOfSameKeyMustBeSorted){
                //     bool isIotaValues = checkIotaValues(values);
                //     assert(isIotaValues);
                // }

                //if(isIotaValues){                   
                    try{           
                        success = executeWithIotaValues(keys, values, offsets);
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
                // }else{
                //     assert(false && "not implemented");
                // }

                std::cerr << "GroupByKeyGpuWarpcore success = " << success << "\n";

                return success;
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

            bool executeWithIotaValues(std::vector<Key_t>& keys, std::vector<Value_t>& values, std::vector<Offset_t>& offsets){
                assert(keys.size() == values.size()); //key value pairs
                assert(std::numeric_limits<Offset_t>::max() >= keys.size()); //total number of keys must fit into Offset_t

                int deviceId = 0;
                CUDACHECK(cudaGetDevice(&deviceId));
                int canUseHostPointerForRegister = 0;
                CUDACHECK(cudaDeviceGetAttribute(&canUseHostPointerForRegister, cudaDevAttrCanUseHostPointerForRegisteredMem, deviceId));
                if(0 == canUseHostPointerForRegister){
                    return false;
                }

                auto deallocVector = [](auto& vec){
                    using T = typename std::remove_reference<decltype(vec)>::type;
                    T tmp{};
                    vec.swap(tmp);
                };

                constexpr int cggroupsize = 8;
                using MultiValueHashTable2 = warpcore::MultiValueHashTable<
                    Key_t,
                    Value_t,
                    warpcore::defaults::empty_key<Key_t>(),
                    warpcore::defaults::tombstone_key<Key_t>(),
                    warpcore::defaults::probing_scheme_t<Key_t, cggroupsize>,
                    warpcore::defaults::table_storage_t<Key_t, Value_t>,
                    warpcore::defaults::temp_memory_bytes()>;

                const std::size_t size = keys.size();
                const float load = 0.9;
                const std::size_t capacity = size / load;
                assert(capacity > size);

                auto gpuTable = MultiValueHashTable2(
                    capacity, warpcore::defaults::seed<Key_t>(), (maxValuesPerKey + 1)
                );

                warpcore::Status tablestatus = gpuTable.pop_status((cudaStream_t)0);
                CUDACHECK(cudaStreamSynchronize((cudaStream_t)0));

                if(tablestatus.has_any_errors()){
                    std::cerr << "groupByKeyWarpcore init status" << tablestatus << "\n";
                    return false;
                }

                constexpr int numbuf = 2;
                constexpr std::size_t buffersize = 100000;
                std::array<helpers::SimpleAllocationPinnedHost<Key_t>, numbuf> h_keysarray{};
                std::array<helpers::SimpleAllocationDevice<Key_t>, numbuf> d_keysarray{};
                std::array<helpers::SimpleAllocationPinnedHost<Value_t>, numbuf> h_valuesarray{};
                std::array<helpers::SimpleAllocationDevice<Value_t>, numbuf> d_valuesarray{};
                std::array<CudaStream, numbuf> streams{};

                for(int i = 0; i < numbuf; i++){
                    h_keysarray[i].resize(buffersize);
                    d_keysarray[i].resize(buffersize);
                    h_valuesarray[i].resize(buffersize);
                    d_valuesarray[i].resize(buffersize);
                }

                //build hashtable

                int bufferindex = 0;
                for(std::size_t i = 0; i < size; i += buffersize){
                    std::size_t currentbatchsize = std::min(buffersize, size - i);
                    
                    CUDACHECK(cudaStreamSynchronize(streams[bufferindex])); //protect pinned buffer

                    std::copy_n(keys.begin() + i, currentbatchsize, h_keysarray[bufferindex].data());

                    CUDACHECK(cudaMemcpyAsync(
                        d_keysarray[bufferindex].data(),
                        h_keysarray[bufferindex].data(),
                        sizeof(Key_t) * currentbatchsize,
                        H2D,
                        streams[bufferindex]
                    ));

                    //iota values can be generated on the device instead of transfer -> iota kernel
                    helpers::lambda_kernel<<<SDIV(currentbatchsize, 128), 128, 0, streams[bufferindex]>>>(
                        [
                            values = d_valuesarray[bufferindex].data(), offset = i, num = currentbatchsize
                        ] __device__ (){
                            const int tid = threadIdx.x + blockIdx.x * blockDim.x;

                            if(tid < num){
                                values[tid] = offset + tid;
                            }
                        }
                    ); CUDACHECKASYNC;

                    gpuTable.insert(
                        d_keysarray[bufferindex],
                        d_valuesarray[bufferindex],
                        currentbatchsize,
                        streams[bufferindex],
                        warpcore::defaults::probing_length()
                    );

                    bufferindex = (bufferindex + 1) % numbuf;
                }

                for(int i = 0; i < numbuf; i++){
                    CUDACHECK(cudaStreamSynchronize(streams[i]));
                }

                tablestatus = gpuTable.pop_status((cudaStream_t)0);
                CUDACHECK(cudaStreamSynchronize((cudaStream_t)0));

                if(tablestatus.has_any_errors()){
                    std::cerr << "groupByKeyWarpcore insert status" << tablestatus << "\n";
                    return false;
                }

                for(int i = 0; i < numbuf; i++){
                    h_keysarray[i].destroy();
                    d_keysarray[i].destroy();
                    h_valuesarray[i].destroy();
                    d_valuesarray[i].destroy();
                }

                std::size_t numUniqueKeys = gpuTable.num_keys((cudaStream_t)0);
                CUDACHECK(cudaStreamSynchronize((cudaStream_t)0));

                if(numUniqueKeys > std::size_t(std::numeric_limits<int>::max())){
                    return false;
                }

                Key_t* d_unique_keys{};
                CUDACHECK(cudaMalloc(&d_unique_keys, sizeof(Key_t) * numUniqueKeys));
                std::size_t* d_numbers{};
                CUDACHECK(cudaMalloc(&d_numbers, sizeof(std::size_t) * (numUniqueKeys + 1)));

                // keys.resize(numUniqueKeys);
                // //treat vector memory as pinned memory to allow direct retrieval from hashtable
                // CUDACHECK(cudaHostRegister(keys.data(), numUniqueKeys * sizeof(Key_t), cudaHostRegisterMapped));



                gpuTable.retrieve_all_keys(
                    d_unique_keys,
                    numUniqueKeys,
                    (cudaStream_t)0
                ); CUDACHECKASYNC;

                dim3 block(512, 1, 1);
                dim3 grid(SDIV(numUniqueKeys, block.x / cggroupsize), 1, 1);
                auto maxValuesPerKeytmp = maxValuesPerKey;
                auto minValuesPerKeytmp = minValuesPerKey;

                helpers::lambda_kernel<<<grid, block, 0, (cudaStream_t)0>>>(
                    [
                        gpuTable,
                        numUniqueKeys,
                        d_unique_keys,
                        d_numbers,
                        maxValuesPerKey = maxValuesPerKeytmp,
                        minValuesPerKey = minValuesPerKeytmp
                    ] __device__ (){
                        using Core = MultiValueHashTable2;

                        const std::size_t tid = helpers::global_thread_id();
                        const std::size_t gid = tid / Core::cg_size();
                        const auto group = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

                        if(gid == 0 && group.thread_rank() == 0){
                            d_numbers[0] = 0;
                        }

                        if(gid < numUniqueKeys){

                            const Key_t key = d_unique_keys[gid];

                            std::size_t numValuesForKey = 0;

                            gpuTable.retrieve(
                                key,
                                nullptr,
                                numValuesForKey,
                                group
                            );

                            if(numValuesForKey > maxValuesPerKey || numValuesForKey < minValuesPerKey){
                                numValuesForKey = 0;
                            }
                            if(group.thread_rank() == 0){
                                d_numbers[gid + 1] = numValuesForKey;
                            }

                        }
                    }
                ); CUDACHECKASYNC;

                std::size_t cubbytes = 0;
                CUDACHECK(cub::DeviceScan::InclusiveSum(
                    nullptr,
                    cubbytes,
                    d_numbers,
                    d_numbers,
                    numUniqueKeys + 1,
                    (cudaStream_t)0	
                ));

                void* cubtemp{};
                CUDACHECK(cudaMalloc(&cubtemp, cubbytes));

                CUDACHECK(cub::DeviceScan::InclusiveSum(
                    cubtemp,
                    cubbytes,
                    d_numbers,
                    d_numbers,
                    numUniqueKeys + 1,
                    (cudaStream_t)0	
                ));

                std::size_t numRemainingValues = 0;
                CUDACHECK(cudaMemcpyAsync(&numRemainingValues, d_numbers + numUniqueKeys, sizeof(std::size_t), D2H, (cudaStream_t)0));
                CUDACHECK(cudaStreamSynchronize((cudaStream_t)0));

                cudaFree(cubtemp);

                Value_t* d_values;
                CUDACHECK(cudaMalloc(&d_values, sizeof(Value_t) * numRemainingValues));

                helpers::lambda_kernel<<<grid, block, 0, (cudaStream_t)0>>>(
                    [
                        gpuTable,
                        numUniqueKeys,
                        d_unique_keys,
                        d_numbers,
                        maxValuesPerKey = maxValuesPerKeytmp,
                        minValuesPerKey = minValuesPerKeytmp,
                        d_values
                    ] __device__ (){
                        using Core = MultiValueHashTable2;

                        const std::size_t tid = helpers::global_thread_id();
                        const std::size_t gid = tid / Core::cg_size();
                        const auto group = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

                        if(gid < numUniqueKeys){

                            const Key_t key = d_unique_keys[gid];

                            std::size_t numValuesForKey = 0;

                            gpuTable.retrieve(
                                key,
                                nullptr,
                                numValuesForKey,
                                group
                            );

                            if(numValuesForKey <= maxValuesPerKey && numValuesForKey >= minValuesPerKey){
                                //real run to obtain values
                                gpuTable.retrieve(
                                    key,
                                    d_values + d_numbers[gid],
                                    numValuesForKey,
                                    group
                                );
                            }
                        }
                    }
                ); CUDACHECKASYNC;

                deallocVector(values);
                values.resize(numRemainingValues);
                CUDACHECK(cudaMemcpyAsync(values.data(), d_values, sizeof(Value_t) * numRemainingValues, D2H, (cudaStream_t)0));

                deallocVector(keys);
                keys.resize(numUniqueKeys);
                CUDACHECK(cudaMemcpyAsync(keys.data(), d_unique_keys, sizeof(Key_t) * numUniqueKeys, D2H, (cudaStream_t)0));

                deallocVector(offsets);
                offsets.resize(numUniqueKeys+1);

                std::array<helpers::SimpleAllocationPinnedHost<std::size_t>, numbuf> h_offsetsarray{};
                for(int i = 0; i < numbuf; i++){
                    h_offsetsarray[i].resize(buffersize);
                }

                bufferindex = 0;
                for(std::size_t i = 0; i < numUniqueKeys+1; i += buffersize){
                    std::size_t currentbatchsize = std::min(buffersize, numUniqueKeys+1 - i);                    

                    CUDACHECK(cudaMemcpyAsync(h_offsetsarray[bufferindex].data(), d_numbers + i, sizeof(Key_t) * currentbatchsize, D2H, streams[bufferindex]));

                    CUDACHECK(cudaStreamSynchronize(streams[bufferindex]));

                    std::copy_n(h_offsetsarray[bufferindex].data(), currentbatchsize, offsets.begin() + i);
                }

                CUDACHECK(cudaFree(d_numbers));
                CUDACHECK(cudaFree(d_values));
                CUDACHECK(cudaFree(d_unique_keys));            
                
                return true;
            }
        };



        template<class Key, class Value, class PsInt>
        struct WarpcoreTransformer{
            using GpuMultiValueHashTable = care::gpu::GpuHashtable<Key, Value>;
            
            struct Result{
                bool success;
            };

            static Result execute(
                std::vector<Key>& keys, 
                std::vector<Value>& values, 
                std::vector<PsInt>& countsPrefixSum, 
                const std::vector<int>& deviceIds,
                int maxValuesPerKey,
                bool valuesOfSameKeyMustBeSorted = true
            ){

                assert(keys.size() == values.size());
                assert(std::numeric_limits<PsInt>::max() >= keys.size());
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

                CudaStream stream;

                constexpr float load = 0.8f;
                std::size_t tablecapacity = keys.size() / load;
                
                auto gputable = std::make_unique<GpuMultiValueHashTable>(
                    keys.size(), load, (maxValuesPerKey + 1)
                );

                warpcore::Status tablestatus = gputable->pop_status(stream);
                CUDACHECK(cudaStreamSynchronize(stream));
                if(tablestatus.has_any_errors()){
                    Result res;
                    res.success = false;
                    return res;
                }

                auto deallocVector = [](auto& vec){
                    using T = typename std::remove_reference<decltype(vec)>::type;
                    T tmp{};
                    vec.swap(tmp);
                };

                constexpr int numstreams = 3;
                const std::size_t batchsize = 1000000;
                const std::size_t iters =  SDIV(keys.size(), batchsize);
                std::array<helpers::SimpleAllocationPinnedHost<Key>, numstreams> h_keys_array{};
                std::array<helpers::SimpleAllocationDevice<Key>, numstreams> d_keys_array{};
                std::array<helpers::SimpleAllocationPinnedHost<Value>, numstreams> h_values_array{};
                std::array<helpers::SimpleAllocationDevice<Value>, numstreams> d_values_array{};
                std::array<CudaStream, numstreams> stream_array{};

                for(int i = 0; i < numstreams; i++){
                    h_keys_array[i].resize(std::min(batchsize, keys.size()));
                    d_keys_array[i].resize(std::min(batchsize, keys.size()));
                    h_values_array[i].resize(std::min(batchsize, keys.size()));
                    d_values_array[i].resize(std::min(batchsize, keys.size()));
                }

                //std::ofstream cpukeysoutput("cpukeysoutput.txt");
            #if 1
                // for(std::size_t i = 0; i < numstreams - 1; i++){
                //     const std::size_t begin = i * batchsize;
                //     const std::size_t end = std::min((i+1) * batchsize, keys.size());
                //     const std::size_t num = end - begin;

                //     const int which = i % numstreams;

                //     std::copy_n(keys.begin() + begin, num, h_keys_array[which].begin());
                //     CUDACHECK(cudaMemcpyAsync(d_keys_array[which].data(), h_keys_array[which].data(), sizeof(Key) * num, H2D, stream_array[which]));
                //     care::gpu::fixKeysForGpuHashTable(d_keys_array[which].data(), num, stream_array[which]);
                //     std::copy_n(values.begin() + begin, num, h_values_array[which].begin());
                //     CUDACHECK(cudaMemcpyAsync(d_values_array[which].data(), h_values_array[which].data(), sizeof(Value) * num, H2D, stream_array[which]));
                    
                //     gputable->insert(
                //         d_keys_array[which].data(),
                //         d_values_array[which].data(),
                //         num,
                //         stream_array[which],
                //         nullptr
                //     );
                // }                

                for(std::size_t i = 0; i < iters; i++){
                    const std::size_t begin = i * batchsize;
                    const std::size_t end = std::min((i+1) * batchsize, keys.size());
                    const std::size_t num = end - begin;

                    const int which = i % numstreams;
                    CUDACHECK(cudaStreamSynchronize(stream_array[which]));

                    std::copy_n(keys.begin() + begin, num, h_keys_array[which].begin());
                    CUDACHECK(cudaMemcpyAsync(d_keys_array[which].data(), h_keys_array[which].data(), sizeof(Key) * num, H2D, stream_array[which]));
                    care::gpu::fixKeysForGpuHashTable(d_keys_array[which].data(), num, stream_array[which]);
                    std::copy_n(values.begin() + begin, num, h_values_array[which].begin());
                    CUDACHECK(cudaMemcpyAsync(d_values_array[which].data(), h_values_array[which].data(), sizeof(Value) * num, H2D, stream_array[which]));
                    
                    gputable->insert(
                        d_keys_array[which].data(),
                        d_values_array[which].data(),
                        num,
                        stream_array[which],
                        nullptr
                    );
                }

                for(int i = 0; i < numstreams; i++){
                    CUDACHECK(cudaStreamSynchronize(stream_array[i]));
                }

            #else
                for(std::size_t i = 0; i < 1; i++){
                    const std::size_t begin = i * batchsize;
                    const std::size_t end = std::min((i+1) * batchsize, keys.size());
                    const std::size_t num = end - begin;

                    const int which = i % 2;
                    CUDACHECK(cudaStreamSynchronize(stream_array[which]));

                    std::copy_n(keys.begin() + begin, num, h_keys_array[which].begin());
                    CUDACHECK(cudaMemcpyAsync(d_keys_array[which].data(), h_keys_array[which].data(), sizeof(Key) * num, H2D, stream_array[which]));
                    care::gpu::fixKeysForGpuHashTable(d_keys_array[which].data(), num, stream_array[which]);
                    std::copy_n(values.begin() + begin, num, h_values_array[which].begin());
                    CUDACHECK(cudaMemcpyAsync(d_values_array[which].data(), h_values_array[which].data(), sizeof(Value) * num, H2D, stream_array[which]));
                    
                    gputable->insert(
                        d_keys_array[which].data(),
                        d_values_array[which].data(),
                        num,
                        stream_array[which],
                        nullptr
                    );
                }                

                for(std::size_t i = 1; i < iters; i++){
                    const std::size_t begin = i * batchsize;
                    const std::size_t end = std::min((i+1) * batchsize, keys.size());
                    const std::size_t num = end - begin;

                    const int which = i % 2;
                    CUDACHECK(cudaStreamSynchronize(stream_array[which]));

                    std::copy_n(keys.begin() + begin, num, h_keys_array[which].begin());
                    CUDACHECK(cudaMemcpyAsync(d_keys_array[which].data(), h_keys_array[which].data(), sizeof(Key) * num, H2D, stream_array[which]));
                    care::gpu::fixKeysForGpuHashTable(d_keys_array[which].data(), num, stream_array[which]);
                    std::copy_n(values.begin() + begin, num, h_values_array[which].begin());
                    CUDACHECK(cudaMemcpyAsync(d_values_array[which].data(), h_values_array[which].data(), sizeof(Value) * num, H2D, stream_array[which]));
                    
                    gputable->insert(
                        d_keys_array[which].data(),
                        d_values_array[which].data(),
                        num,
                        stream_array[which],
                        nullptr
                    );
                }

                CUDACHECK(cudaStreamSynchronize(stream_array[0]));
                CUDACHECK(cudaStreamSynchronize(stream_array[1]));
            #endif                

                tablestatus = gputable->pop_status(stream);
                CUDACHECK(cudaStreamSynchronize(stream));
                if(tablestatus.has_any_errors()){
                    Result res;
                    res.success = false;
                    return res;
                }

                deallocVector(keys);
                deallocVector(countsPrefixSum);

                std::size_t numUniqueKeys = gputable->getNumUniqueKeys();
                keys.resize(numUniqueKeys);
                countsPrefixSum.resize(numUniqueKeys+1);

                std::cerr << "start compaction\n";
                gputable->compactIntoHostBuffers(keys.data(), values.data(), countsPrefixSum.data());
                std::cerr << "end compaction\n";

                setupStatus = cudaSetDevice(oldDeviceId);
                if(setupStatus != cudaSuccess) //we cannot recover from this.
                    throw std::runtime_error("Could not revert device id!");

                Result res;
                res.success = true;
                return res;
            }


        };
    #endif // #ifdef CARE_HAS_WARPCORE
    
    #endif //#ifdef __NVCC__


}


#endif