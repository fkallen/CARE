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

#include <unordered_map>

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
#endif

#include <hpc_helpers.cuh>

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
            using hasher = hashers::MurmurHash<std::uint64_t>;

            std::size_t probes = 0;
            std::size_t pos = hasher::hash(key) % capacity;
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
            using hasher = hashers::MurmurHash<std::uint64_t>;
            
            std::size_t probes = 0;
            std::size_t pos = hasher::hash(key) % capacity;
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

        template<class Key_t, class Value_t, class Offset_t>
        struct GroupByKeyCpu{
            bool valuesOfSameKeyMustBeSorted = false;
            int maxValuesPerKey = 0;

            GroupByKeyCpu(bool sortValues, int maxValuesPerKey_) 
                : valuesOfSameKeyMustBeSorted(sortValues), maxValuesPerKey(maxValuesPerKey_){}

            /*
                Input: keys and values. keys[i] and values[i] form a key-value pair
                Output: unique keys. values with the same key are stored consecutive. values of unique_keys[i] are
                stored at values[offsets[i]] to values[offsets[i+1]] (exclusive)
                If valuesOfSameKeyMustBeSorted == true, values with the same key are sorted in ascending order.
                If there are more than maxValuesPerKey values with the same key, all of those values are removed,
                i.e. the key ends up with 0 values
            */
            void execute(std::vector<Key_t>& keys, std::vector<Value_t>& values, std::vector<Offset_t>& offsets){
                if(keys.size() == 0) return;

                bool isIotaValues = checkIotaValues(values);

                if(isIotaValues){
                    executeWithIotaValues(keys, values, offsets);
                }else{
                    assert(false && "not implemented");
                }
            }

            bool checkIotaValues(const std::vector<Value_t>& values){
                auto policy = thrust::omp::par;

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
                        auto begin = offsets[index];
                        auto end = offsets[index+1];
                        if(end - begin > Offset_t(maxValuesPerKey)){
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

            GroupByKeyGpu(bool sortValues, int maxValuesPerKey_) 
                : valuesOfSameKeyMustBeSorted(sortValues), maxValuesPerKey(maxValuesPerKey_){}

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

                bool isIotaValues = checkIotaValues(values);

                if(isIotaValues){                   
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
                }else{
                    assert(false && "not implemented");
                }

                return success;
            }

            bool checkIotaValues(const std::vector<Value_t>& values){
                auto policy = thrust::omp::par;

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
                thrust::for_each(
                    allocatorPolicy,
                    thrust::counting_iterator<Offset_t>(0),
                    thrust::counting_iterator<Offset_t>(0) + nUniqueKeys,
                    [=] __device__ (Offset_t index){
                        auto begin = d_offsets_begin[index];
                        auto end = d_offsets_begin[index+1];
                        if(end - begin > Offset_t(maxValuesPerKey_copy)){
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
                cudaStreamSynchronize(stream); CUERR;
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
                //     cudaMemcpyAsync(d_keys_array[which].data(), h_keys_array[which].data(), sizeof(Key) * num, H2D, stream_array[which]); CUERR;
                //     care::gpu::fixKeysForGpuHashTable(d_keys_array[which].data(), num, stream_array[which]);
                //     std::copy_n(values.begin() + begin, num, h_values_array[which].begin());
                //     cudaMemcpyAsync(d_values_array[which].data(), h_values_array[which].data(), sizeof(Value) * num, H2D, stream_array[which]); CUERR;
                    
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
                    cudaStreamSynchronize(stream_array[which]); CUERR;

                    std::copy_n(keys.begin() + begin, num, h_keys_array[which].begin());
                    cudaMemcpyAsync(d_keys_array[which].data(), h_keys_array[which].data(), sizeof(Key) * num, H2D, stream_array[which]); CUERR;
                    care::gpu::fixKeysForGpuHashTable(d_keys_array[which].data(), num, stream_array[which]);
                    std::copy_n(values.begin() + begin, num, h_values_array[which].begin());
                    cudaMemcpyAsync(d_values_array[which].data(), h_values_array[which].data(), sizeof(Value) * num, H2D, stream_array[which]); CUERR;
                    
                    gputable->insert(
                        d_keys_array[which].data(),
                        d_values_array[which].data(),
                        num,
                        stream_array[which],
                        nullptr
                    );
                }

                for(int i = 0; i < numstreams; i++){
                    cudaStreamSynchronize(stream_array[i]); CUERR;
                }

#else
                for(std::size_t i = 0; i < 1; i++){
                    const std::size_t begin = i * batchsize;
                    const std::size_t end = std::min((i+1) * batchsize, keys.size());
                    const std::size_t num = end - begin;

                    const int which = i % 2;
                    cudaStreamSynchronize(stream_array[which]); CUERR;

                    std::copy_n(keys.begin() + begin, num, h_keys_array[which].begin());
                    cudaMemcpyAsync(d_keys_array[which].data(), h_keys_array[which].data(), sizeof(Key) * num, H2D, stream_array[which]); CUERR;
                    care::gpu::fixKeysForGpuHashTable(d_keys_array[which].data(), num, stream_array[which]);
                    std::copy_n(values.begin() + begin, num, h_values_array[which].begin());
                    cudaMemcpyAsync(d_values_array[which].data(), h_values_array[which].data(), sizeof(Value) * num, H2D, stream_array[which]); CUERR;
                    
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
                    cudaStreamSynchronize(stream_array[which]); CUERR;

                    std::copy_n(keys.begin() + begin, num, h_keys_array[which].begin());
                    cudaMemcpyAsync(d_keys_array[which].data(), h_keys_array[which].data(), sizeof(Key) * num, H2D, stream_array[which]); CUERR;
                    care::gpu::fixKeysForGpuHashTable(d_keys_array[which].data(), num, stream_array[which]);
                    std::copy_n(values.begin() + begin, num, h_values_array[which].begin());
                    cudaMemcpyAsync(d_values_array[which].data(), h_values_array[which].data(), sizeof(Value) * num, H2D, stream_array[which]); CUERR;
                    
                    gputable->insert(
                        d_keys_array[which].data(),
                        d_values_array[which].data(),
                        num,
                        stream_array[which],
                        nullptr
                    );
                }

                cudaStreamSynchronize(stream_array[0]); CUERR;
                cudaStreamSynchronize(stream_array[1]); CUERR;
#endif                

                tablestatus = gputable->pop_status(stream);
                cudaStreamSynchronize(stream); CUERR;
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
                            //Try to build a warpcore hashtable and copy it back to cpu

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

        CpuReadOnlyMultiValueHashTable(
            std::uint64_t maxNumValues_
        ) : buildMaxNumValues{maxNumValues_}{
            buildkeys.reserve(buildMaxNumValues);
            buildvalues.reserve(buildMaxNumValues);
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
                using GroupByKeyCpu = cpuhashtabledetail::GroupByKeyCpu<Key, Value, read_number>;

                GroupByKeyCpu groupByKey(valuesOfSameKeyMustBeSorted, maxValuesPerKey);
                groupByKey.execute(keys, values, countsPrefixSum);
            #ifdef __NVCC__
            }else{

                using GroupByKeyCpu = cpuhashtabledetail::GroupByKeyCpu<Key, Value, read_number>;
                using GroupByKeyGpu = cpuhashtabledetail::GroupByKeyGpu<Key, Value, read_number>;

                GroupByKeyGpu groupByKeyGpu(valuesOfSameKeyMustBeSorted, maxValuesPerKey);
                bool success = groupByKeyGpu.execute(keys, values, countsPrefixSum);

                if(!success){
                    GroupByKeyCpu groupByKeyCpu(valuesOfSameKeyMustBeSorted, maxValuesPerKey);
                    groupByKeyCpu.execute(keys, values, countsPrefixSum);
                }
            }
            #endif

            lookup = std::move(NaiveCpuSingleValueHashTable<Key, ValueIndex>(keys.size(), 0.8f));
            //std::cerr << "keys.size(): " << keys.size() << "\n";
            //nvtx::push_range("build_key_to_index_table", 0);
            for(std::size_t i = 0; i < keys.size(); i++){
                // if(keys[i] == 390602873081ull){
                //     std::cerr << keys[i] << " " << countsPrefixSum[i] << " " << (countsPrefixSum[i+1] - countsPrefixSum[i]) << "\n";
                // }
                
                lookup.insert(
                    keys[i], 
                    ValueIndex{countsPrefixSum[i], countsPrefixSum[i+1] - countsPrefixSum[i]}
                );
                
            }
            //nvtx::pop_range();

            isInit = true;
        }

        void insert(Key* keys, Value* values, int N){
            assert(keys != nullptr);
            assert(values != nullptr);
            assert(buildMaxNumValues >= buildkeys.size() + N);

            buildkeys.insert(buildkeys.end(), keys, keys + N);
            buildvalues.insert(buildvalues.end(), values, values + N);
        }

        void finalize(int maxValuesPerKey, const std::vector<int>& gpuIds = {}){
            init(std::move(buildkeys), std::move(buildvalues), maxValuesPerKey, gpuIds);            
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
            result.host += sizeof(Key) * buildkeys.capacity();
            result.host += sizeof(Value) * buildvalues.capacity();

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
        bool isInit = false;
        std::uint64_t buildMaxNumValues = 0;
        std::vector<Key> buildkeys;
        std::vector<Value> buildvalues;
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
